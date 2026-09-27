"""Production Isotropic Kalman Delta Network layer."""

from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn as nn
from einops import rearrange
from fla.modules import FusedRMSNormGated

from lit_gpt.kdn_ops.iso_kdn_chunk import (
    _chunk_kda_precumsum_paired,
    _iso_kdn_gate_precumsum_paired,
    iso_kdn_gain,
    pack_f_omega_r_projection,
    pack_qkv_projection,
)

_SUPPORTED_HEAD_DIMS = (64, 128)
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
        raise TypeError("Isotropic KDN training requires CUDA BF16 autocast")
    if hidden_dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(
            "Isotropic KDN training requires FP32 or BF16 hidden states under CUDA BF16 autocast"
        )


def _fixed_short_conv_source(hidden_size: int, kernel_size: int) -> nn.Conv1d:
    """Build a constructor-only convolution without consulting FLA environment knobs."""

    convolution = nn.Conv1d(
        in_channels=hidden_size,
        out_channels=hidden_size,
        kernel_size=kernel_size,
        groups=hidden_size,
        bias=False,
        padding=kernel_size - 1,
    )
    # PackedQKV validates these attributes before copying the initialized
    # weight. The temporary modules are deleted immediately after packing.
    convolution.activation = "silu"
    convolution.backend = "triton"
    return convolution


class IsotropicKalmanDeltaNetwork(nn.Module):
    """Fixed-length CUDA implementation of the Isotropic KDN mixer."""

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
        omega_min: float = 0.0,
        r_min: float = 0.01,
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
                f"unsupported Isotropic KDN head_dim {head_dim}; "
                f"expected one of {_SUPPORTED_HEAD_DIMS}"
            )
        if hidden_size != head_dim * num_heads:
            raise ValueError(
                "Isotropic KDN hidden_size must equal head_dim * num_heads; "
                f"got hidden_size={hidden_size}, head_dim={head_dim}, "
                f"num_heads={num_heads}"
            )
        resolved_num_v_heads = num_heads if num_v_heads is None else num_v_heads
        _validate_positive_int("num_v_heads", resolved_num_v_heads)
        if resolved_num_v_heads != num_heads:
            raise ValueError("num_v_heads must equal num_heads for Isotropic KDN")
        if (
            isinstance(expand_v, bool)
            or not isinstance(expand_v, Real)
            or float(expand_v) != 1.0
        ):
            raise ValueError(
                f"expand_v must be 1.0 for Isotropic KDN, got {expand_v!r}"
            )
        if not use_short_conv or conv_size != 4 or conv_bias:
            raise ValueError(
                "Isotropic KDN requires the biasless width-4 short-convolution path"
            )

        self.omega_min = _nonnegative_real("omega_min", omega_min)
        self.r_min = _positive_real("r_min", r_min)
        self.info_scale = (
            float(head_dim)
            if info_scale is None
            else _positive_real("info_scale", info_scale)
        )

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

        # Preserve the pinned FLA KDA constructor's allocation and RNG order.
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
        # This discarded projection is still allocated to preserve the frozen
        # constructor's random-number stream.
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

        # Preserve the frozen constructor RNG order for checkpoint/state parity:
        # replace b_proj, initialize r first, then initialize process noise.
        del self.b_proj
        self.r_proj = nn.Linear(hidden_size, self.num_v_heads, bias=True)
        self.omega_proj = nn.Linear(hidden_size, self.num_v_heads, bias=True)
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

        pack_f_omega_r_projection(self)
        pack_qkv_projection(self)

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
                "IsotropicKalmanDeltaNetwork supports only fixed-length sequences; "
                f"unsupported arguments: {requested}"
            )
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must have shape [B,T,{self.hidden_size}], "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[0] == 0 or hidden_states.shape[1] == 0:
            raise ValueError(
                "Isotropic KDN requires non-empty batch and sequence dimensions"
            )
        if torch.is_grad_enabled() and hidden_states.shape[1] < 2:
            raise ValueError(
                "differentiable Isotropic KDN training requires sequence length >= 2; "
                "the exact one-token convolution update is inference-only"
            )
        if not hidden_states.is_cuda:
            raise ValueError("IsotropicKalmanDeltaNetwork requires CUDA hidden_states")
        _validate_training_dtype(
            training=self.training,
            hidden_dtype=hidden_states.dtype,
        )

        q, k, v = self.qkv_pack(hidden_states)

        packed_f_omega_r = self.f_omega_r_proj.forward_packed(hidden_states)
        f_hidden, omega_r_raw = packed_f_omega_r.split(
            (self.head_v_dim, 2 * self.num_v_heads), dim=-1
        )
        g_raw = self.f_proj[1](f_hidden)

        q, k = (
            rearrange(value, "... (h d) -> ... h d", d=self.head_k_dim)
            for value in (q, k)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        g_raw = rearrange(g_raw, "... (h d) -> ... h d", d=self.head_k_dim)

        g_cumsum, a, omega, r, initial_precision = _iso_kdn_gate_precumsum_paired(
            g_raw,
            omega_r_raw,
            self.omega_proj.bias.float(),
            self.r_proj.bias.float(),
            self.A_log.float(),
            self.dt_bias.float(),
            self.initial_precision_param.float(),
            self.omega_min,
            self.r_min,
        )
        gain = iso_kdn_gain(
            a,
            omega,
            r,
            initial_precision,
            head_dim=self.head_k_dim,
            out_dtype=torch.float32,
            info_scale=self.info_scale,
        )

        output = _chunk_kda_precumsum_paired(
            q=q,
            k=k,
            v=v,
            g_cumsum=g_cumsum,
            beta=gain,
        )
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


__all__ = ("IsotropicKalmanDeltaNetwork",)
