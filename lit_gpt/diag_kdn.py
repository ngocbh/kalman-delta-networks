"""Production Diagonal Kalman Delta Network layer."""

from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn as nn
from einops import rearrange
from fla.modules import FusedRMSNormGated

_SUPPORTED_HEAD_DIMS = (64, 128)
_OMEGA_MIN = 0.0
_R_MIN = 0.01
_INITIAL_PRECISION = 1.0
_INITIAL_PRECISION_FLOOR = 0.1


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _positive_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return result


def _nonnegative_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a nonnegative finite real")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be nonnegative and finite, got {value!r}")
    return result


def _validate_training_dtype(*, training: bool, hidden_dtype: torch.dtype) -> None:
    if not training:
        return
    if (
        not torch.is_autocast_enabled("cuda")
        or torch.get_autocast_dtype("cuda") != torch.bfloat16
    ):
        raise TypeError("Diagonal KDN training requires CUDA BF16 autocast")
    if hidden_dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(
            "Diagonal KDN training requires FP32 or BF16 hidden states under CUDA BF16 autocast"
        )


def _fixed_short_conv_source(hidden_size: int, kernel_size: int) -> nn.Conv1d:
    """Allocate the frozen depthwise-convolution parameters without an env selector."""

    convolution = nn.Conv1d(
        in_channels=hidden_size,
        out_channels=hidden_size,
        kernel_size=kernel_size,
        groups=hidden_size,
        bias=False,
        padding=kernel_size - 1,
    )
    convolution.activation = "silu"
    convolution.backend = "triton"
    return convolution


def _fixed_short_conv(
    convolution: nn.Conv1d, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Run the sole fixed-length Triton short-convolution path."""

    from fla.modules.conv.causal_conv1d import causal_conv1d

    output, final_state = causal_conv1d(
        x=hidden_states,
        weight=rearrange(convolution.weight, "d 1 w -> d w"),
        bias=None,
        residual=None,
        initial_state=None,
        output_final_state=False,
        activation="silu",
        backend="triton",
        cu_seqlens=None,
        chunk_indices=None,
    )
    if final_state is not None:
        raise RuntimeError("fixed-length Diagonal KDN convolution returned state")
    return output


class DiagonalKalmanDeltaNetwork(nn.Module):
    """Fixed-length CUDA implementation of the Diagonal KDN mixer."""

    def __init__(
        self,
        hidden_size: int = 2304,
        head_dim: int = 128,
        num_heads: int = 18,
        expand_v: float = 1.0,
        num_v_heads: int | None = None,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        omega_min: float = _OMEGA_MIN,
        r_min: float = _R_MIN,
        info_scale: float | None = None,
        norm_eps: float = 1e-5,
        layer_idx: int | None = None,
    ) -> None:
        for name, value in (
            ("hidden_size", hidden_size),
            ("head_dim", head_dim),
            ("num_heads", num_heads),
        ):
            _validate_positive_int(name, value)
        if head_dim not in _SUPPORTED_HEAD_DIMS:
            raise ValueError(
                f"unsupported Diagonal KDN head_dim {head_dim}; "
                f"expected one of {_SUPPORTED_HEAD_DIMS}"
            )
        if hidden_size != head_dim * num_heads:
            raise ValueError(
                "Diagonal KDN hidden_size must equal head_dim * num_heads; "
                f"got hidden_size={hidden_size}, head_dim={head_dim}, "
                f"num_heads={num_heads}"
            )
        resolved_num_v_heads = num_heads if num_v_heads is None else num_v_heads
        _validate_positive_int("num_v_heads", resolved_num_v_heads)
        if resolved_num_v_heads != num_heads:
            raise ValueError("num_v_heads must equal num_heads for Diagonal KDN")
        if (
            isinstance(expand_v, bool)
            or not isinstance(expand_v, Real)
            or float(expand_v) != 1.0
        ):
            raise ValueError(f"expand_v must be 1.0 for Diagonal KDN, got {expand_v!r}")
        if not use_short_conv or conv_size != 4 or conv_bias:
            raise ValueError(
                "Diagonal KDN requires the biasless width-4 short-convolution path"
            )

        resolved_omega_min = _nonnegative_real("omega_min", omega_min)
        resolved_r_min = _positive_real("r_min", r_min)
        resolved_info_scale = (
            float(head_dim)
            if info_scale is None
            else _positive_real("info_scale", info_scale)
        )
        if resolved_omega_min != _OMEGA_MIN:
            raise ValueError(f"Diagonal KDN fixes omega_min={_OMEGA_MIN}")
        if resolved_r_min != _R_MIN:
            raise ValueError(f"Diagonal KDN fixes r_min={_R_MIN}")
        if resolved_info_scale != float(head_dim):
            raise ValueError("Diagonal KDN fixes info_scale to the head dimension")

        super().__init__()

        self.hidden_size = hidden_size
        self.expand_v = 1.0
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = resolved_num_v_heads
        self.head_k_dim = head_dim
        self.head_v_dim = head_dim
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.layer_idx = layer_idx
        self.gate_dim = self.num_v_heads * self.head_k_dim
        self.omega_min = resolved_omega_min
        self.r_min = resolved_r_min
        self.info_scale = resolved_info_scale

        # Preserve the frozen KimiDeltaAttention allocation and RNG order while
        # exposing none of its mode/backend/safety controls.
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.q_conv1d = _fixed_short_conv_source(self.key_dim, 4)
        self.k_conv1d = _fixed_short_conv_source(self.key_dim, 4)
        self.v_conv1d = _fixed_short_conv_source(self.value_dim, 4)

        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.gate_dim, bias=False),
        )
        # Allocate and discard the frozen scalar-write projection so all later
        # parameters consume the same random-number stream.
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        self.A_log = nn.Parameter(
            torch.log(
                torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(1, 16)
            )
        )
        self.A_log._no_weight_decay = True
        dt = torch.exp(
            torch.rand(self.gate_dim, dtype=torch.float32)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = FusedRMSNormGated(
            self.head_v_dim,
            activation="sigmoid",
            eps=norm_eps,
        )
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        del self.b_proj
        self.omega_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.gate_dim, bias=True),
        )
        nn.init.zeros_(self.omega_proj[1].weight)
        nn.init.zeros_(self.omega_proj[1].bias)
        self.omega_proj[1]._kdn_omega_zero = True
        self.r_proj = nn.Linear(hidden_size, self.num_v_heads, bias=True)

        inverse_initial_precision = math.log(
            math.expm1(_INITIAL_PRECISION - _INITIAL_PRECISION_FLOOR)
        )
        self.initial_precision_param = nn.Parameter(
            torch.full(
                (self.num_v_heads,),
                inverse_initial_precision,
                dtype=torch.float32,
            )
        )
        self.initial_precision_param._no_weight_decay = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        unsupported = {
            "attention_mask": attention_mask is not None,
            "past_key_values": past_key_values is not None,
            "use_cache": use_cache is not False,
            "output_attentions": output_attentions is not False,
            "kwargs": bool(kwargs),
        }
        requested = tuple(name for name, enabled in unsupported.items() if enabled)
        if requested:
            raise ValueError(
                "DiagonalKalmanDeltaNetwork supports only fixed-length sequences; "
                f"unsupported arguments: {requested}"
            )
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must have shape [B,T,{self.hidden_size}], "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[0] == 0 or hidden_states.shape[1] == 0:
            raise ValueError(
                "Diagonal KDN requires non-empty batch and sequence dimensions"
            )
        if torch.is_grad_enabled() and hidden_states.shape[1] < 2:
            raise ValueError(
                "differentiable Diagonal KDN training requires sequence length >= 2; "
                "the exact one-token convolution update is inference-only"
            )
        if not hidden_states.is_cuda:
            raise ValueError("DiagonalKalmanDeltaNetwork requires CUDA hidden_states")
        _validate_training_dtype(
            training=self.training, hidden_dtype=hidden_states.dtype
        )

        from lit_gpt.kdn_ops.diag_kdn_chunk import (
            _diag_kdn_frontend_precumsum_paired,
            _diag_kdn_gain,
            _diag_kdn_normalize,
            chunk_kalman,
        )

        q = _fixed_short_conv(self.q_conv1d, self.q_proj(hidden_states))
        k = _fixed_short_conv(self.k_conv1d, self.k_proj(hidden_states))
        v = _fixed_short_conv(self.v_conv1d, self.v_proj(hidden_states))

        retention_raw = rearrange(
            self.f_proj(hidden_states),
            "... (h d) -> ... h d",
            d=self.head_k_dim,
        )
        omega_raw = rearrange(
            self.omega_proj(hidden_states),
            "... (h d) -> ... h d",
            d=self.head_k_dim,
        )
        r_raw = self.r_proj(hidden_states)
        q, k = (
            rearrange(value, "... (h d) -> ... h d", d=self.head_k_dim)
            for value in (q, k)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        q_memory, k_gain, k_memory = _diag_kdn_normalize(q, k)
        g_cumsum, alpha, omega, r, initial_precision = (
            _diag_kdn_frontend_precumsum_paired(
                retention_raw,
                omega_raw,
                r_raw,
                self.A_log.float(),
                self.dt_bias.float().view(self.num_heads, self.head_k_dim),
                self.initial_precision_param.float(),
            )
        )
        kappa = _diag_kdn_gain(
            k_gain,
            alpha,
            omega,
            r,
            initial_precision,
            omega_raw,
            r_raw,
        )
        output = chunk_kalman(
            q_memory,
            k_memory,
            kappa,
            v,
            g_cumsum,
            float(self.head_k_dim**-0.5),
        )
        # Match the production S3 path: its FP32 recurrence is materialized at
        # the public value dtype before the fused output normalization.
        output = output.to(v.dtype)
        output = self.o_norm(
            output,
            rearrange(
                self.g_proj(hidden_states),
                "... (h d) -> ... h d",
                d=self.head_v_dim,
            ),
        )
        output = rearrange(output, "b t h d -> b t (h d)")
        return self.o_proj(output), None, None


__all__ = ("DiagonalKalmanDeltaNetwork",)
