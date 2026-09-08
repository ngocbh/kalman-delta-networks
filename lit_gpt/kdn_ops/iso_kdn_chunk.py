"""Consolidated frozen production operations for Isotropic KDN.

This file mechanically incorporates the reachable production closure from the
following files at frozen reference commit
e7925b900b42e3aac36ac1309a74d8abfdc612d8:

- iso_qkv_conv.py
- iso_projection.py
- iso_gate.py
- iso_chunk.py
- kda_precumsum.py

The KDA orchestration and reused FLA kernels retain their upstream MIT license
and Copyright 2023-2026 FLA contributors.  The scalar covariance scan is the
normalized-key Isotropic KDN specialization; its information scale changes the
posterior covariance only, never the current-token gain.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.modules.conv.causal_conv1d import causal_conv1d as _installed_causal_conv1d
from fla.modules.conv.triton.ops import causal_conv1d_update as _installed_causal_conv1d_update
from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu,
    chunk_gated_delta_rule_fwd_h,
)
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv, chunk_kda_bwd_wy_dqkg_fused
from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra, chunk_kda_fwd_intra
from fla.ops.kda.wy_fast import recompute_w_u_fwd
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


# ---- Frozen iso_qkv_conv.py -------------------------------------------------

GROUPS = 3
KERNEL_WIDTH = 4
CHUNK_SIZE = 64
RAW_FORWARD_BLOCK_D = 64
RAW_FORWARD_NUM_WARPS = 8
BACKWARD_BLOCK_D = 32
BACKWARD_NUM_WARPS = 4
NUM_STAGES = 3


@triton.jit
def _packed_raw_forward_kernel(
    x,
    raw_y,
    weight,
    B,
    T,
    stride_x_b,
    stride_x_t,
    stride_x_d,
    stride_y_g,
    stride_y_b,
    stride_y_t,
    stride_y_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
):
    i_d = tl.program_id(0)
    i_t = tl.program_id(1)
    i_gb = tl.program_id(2)
    i_g = i_gb // B
    i_b = i_gb - i_g * B

    b_offset = tl.cast(i_b, tl.int64)
    p_x = x + b_offset * stride_x_b + i_g * D * stride_x_d
    p_y_base = raw_y + i_g * stride_y_g + b_offset * stride_y_b
    p_weight = weight + i_g * D * W

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    b_w = tl.load(
        p_weight + o_d[:, None] * W + o_w,
        mask=m_d[:, None] & m_w,
        other=0,
    ).to(tl.float32)

    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    for i_w in tl.static_range(-W + 1, 1):
        p_x_block = tl.make_block_ptr(
            p_x,
            (T, D),
            (stride_x_t, stride_x_d),
            (i_t * BT + i_w, i_d * BD),
            (BT, BD),
            (1, 0),
        )
        b_yi = tl.load(p_x_block, boundary_check=(0, 1)).to(tl.float32)
        b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
        b_y += b_yi

    p_y = tl.make_block_ptr(
        p_y_base,
        (T, D),
        (stride_y_t, stride_y_d),
        (i_t * BT, i_d * BD),
        (BT, BD),
        (1, 0),
    )
    tl.store(
        p_y,
        tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@triton.jit
def _grouped_backward_kernel(
    x,
    raw_y,
    weight,
    dq,
    dk,
    dv,
    dx,
    dw_partials,
    B,
    T,
    stride_x_b,
    stride_x_t,
    stride_x_d,
    stride_y_g,
    stride_y_b,
    stride_y_t,
    stride_y_d,
    stride_dy_b,
    stride_dy_t,
    stride_dy_d,
    stride_dx_b,
    stride_dx_t,
    stride_dx_d,
    D: tl.constexpr,
    PACKED_D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
):
    i_d = tl.program_id(0)
    i_t = tl.program_id(1)
    i_gb = tl.program_id(2)
    i_g = i_gb // B
    i_b = i_gb - i_g * B
    i_tg = i_b * tl.num_programs(1) + i_t

    b_offset = tl.cast(i_b, tl.int64)
    p_x_base = x + b_offset * stride_x_b + i_g * D * stride_x_d
    p_raw_y = raw_y + i_g * stride_y_g + b_offset * stride_y_b
    p_weight = weight + i_g * D * W
    p_dx_base = dx + b_offset * stride_dx_b + i_g * D * stride_dx_d

    if i_g == 0:
        p_dy = dq + b_offset * stride_dy_b
    elif i_g == 1:
        p_dy = dk + b_offset * stride_dy_b
    else:
        p_dy = dv + b_offset * stride_dy_b

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    p_x = tl.make_block_ptr(
        p_x_base,
        (T, D),
        (stride_x_t, stride_x_d),
        (i_t * BT, i_d * BD),
        (BT, BD),
        (1, 0),
    )
    b_x = tl.load(p_x, boundary_check=(0, 1))
    b_w = tl.load(
        p_weight + o_d[:, None] * W + o_w,
        mask=m_d[:, None] & m_w,
        other=0,
    )

    b_dx = tl.zeros((BT, BD), dtype=tl.float32)
    for i_w in tl.static_range(0, W):
        p_dy_block = tl.make_block_ptr(
            p_dy,
            (T, D),
            (stride_dy_t, stride_dy_d),
            (i_t * BT + i_w, i_d * BD),
            (BT, BD),
            (1, 0),
        )
        b_dy = tl.load(p_dy_block, boundary_check=(0, 1)).to(tl.float32)

        p_raw_y_block = tl.make_block_ptr(
            p_raw_y,
            (T, D),
            (stride_y_t, stride_y_d),
            (i_t * BT + i_w, i_d * BD),
            (BT, BD),
            (1, 0),
        )
        b_raw_y = tl.load(p_raw_y_block, boundary_check=(0, 1)).to(tl.float32)
        b_sigmoid = tl.sigmoid(b_raw_y)
        b_dy = b_dy * b_sigmoid * (1 + b_raw_y * (1 - b_sigmoid))

        b_wdy = b_dy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)
        b_dw = tl.sum(b_dy * b_x, 0)
        tl.store(
            dw_partials
            + i_tg * PACKED_D * W
            + (i_g * D + o_d) * W
            + W
            - i_w
            - 1,
            b_dw.to(dw_partials.dtype.element_ty),
            mask=m_d,
        )
        b_dx += b_wdy

    p_dx = tl.make_block_ptr(
        p_dx_base,
        (T, D),
        (stride_dx_t, stride_dx_d),
        (i_t * BT, i_d * BD),
        (BT, BD),
        (1, 0),
    )
    tl.store(
        p_dx,
        tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


def _validate_qkv_conv_inputs(x: torch.Tensor, weight: torch.Tensor) -> int:
    if x.ndim != 3:
        raise ValueError(f"packed input must be rank-3 [B,T,3D], got {tuple(x.shape)}")
    if x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError(f"batch and sequence dimensions must be positive, got {tuple(x.shape)}")
    packed_width = x.shape[-1]
    if packed_width < GROUPS or packed_width % GROUPS != 0:
        raise ValueError(
            f"packed input width must be a positive multiple of {GROUPS}, got {packed_width}"
        )
    if weight.shape != (packed_width, KERNEL_WIDTH):
        raise ValueError(
            f"packed weight must be [{packed_width},{KERNEL_WIDTH}], got {tuple(weight.shape)}"
        )
    if x.device.type != "cuda" or weight.device != x.device:
        raise ValueError("QKV causal convolution requires colocated CUDA tensors")
    allowed_dtypes = (torch.bfloat16, torch.float32)
    if x.dtype not in allowed_dtypes:
        raise TypeError(f"packed input must be bfloat16 or float32, got {x.dtype}")
    if weight.dtype not in allowed_dtypes:
        raise TypeError(f"packed weight must be bfloat16 or float32, got {weight.dtype}")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("packed input and weight must already be contiguous")
    return packed_width // GROUPS


def _prepare_gradients(
    dq: torch.Tensor | None,
    dk: torch.Tensor | None,
    dv: torch.Tensor | None,
    *,
    B: int,
    T: int,
    D: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expected_shape = (B, T, D)
    expected_stride = (T * D, D, 1)
    prepared = []
    for name, gradient in (("dq", dq), ("dk", dk), ("dv", dv)):
        if gradient is None:
            prepared.append(torch.zeros(expected_shape, device=device, dtype=dtype))
            continue
        if gradient.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(gradient.shape)}")
        if gradient.device != device or gradient.dtype != dtype:
            raise TypeError(f"{name} must be a {dtype} tensor on {device}")
        if not gradient.is_contiguous() or gradient.stride() != expected_stride:
            raise ValueError(f"{name} must already have the standard contiguous layout")
        prepared.append(gradient)
    return tuple(prepared)


def _call_installed_forward(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    output, final_state = _installed_causal_conv1d(
        x=x,
        weight=weight,
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
        raise RuntimeError("fixed-length convolution unexpectedly returned a final state")
    return output


def _call_installed_step(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    cache = x.new_zeros(x.shape[0], x.shape[2], weight.shape[1])
    output, _ = _installed_causal_conv1d_update(
        x=x,
        cache=cache,
        residual=None,
        weight=weight,
        bias=None,
        activation="silu",
    )
    return output


def _recompute_raw_preactivation(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    D = _validate_qkv_conv_inputs(x, weight)
    B, T, _ = x.shape
    raw_y = torch.empty((GROUPS, B, T, D), dtype=x.dtype, device=x.device)
    stride_x_b, stride_x_t, stride_x_d = x.stride()
    stride_y_g, stride_y_b, stride_y_t, stride_y_d = raw_y.stride()
    num_time_blocks = triton.cdiv(T, CHUNK_SIZE)
    grid = (triton.cdiv(D, RAW_FORWARD_BLOCK_D), num_time_blocks, GROUPS * B)
    _packed_raw_forward_kernel[grid](
        x=x,
        raw_y=raw_y,
        weight=weight,
        B=B,
        T=T,
        stride_x_b=stride_x_b,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        stride_y_g=stride_y_g,
        stride_y_b=stride_y_b,
        stride_y_t=stride_y_t,
        stride_y_d=stride_y_d,
        D=D,
        W=KERNEL_WIDTH,
        BT=CHUNK_SIZE,
        BW=triton.next_power_of_2(KERNEL_WIDTH),
        BD=RAW_FORWARD_BLOCK_D,
        num_warps=RAW_FORWARD_NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return raw_y


def _grouped_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    dq: torch.Tensor | None,
    dk: torch.Tensor | None,
    dv: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    D = _validate_qkv_conv_inputs(x, weight)
    B, T, _ = x.shape
    dq, dk, dv = _prepare_gradients(
        dq, dk, dv, B=B, T=T, D=D, device=x.device, dtype=x.dtype
    )

    raw_y = _recompute_raw_preactivation(x, weight)
    dx = torch.empty_like(x)
    num_time_blocks = triton.cdiv(T, CHUNK_SIZE)
    dw_partials = weight.new_empty(B * num_time_blocks, *weight.shape, dtype=torch.float32)

    stride_x_b, stride_x_t, stride_x_d = x.stride()
    stride_y_g, stride_y_b, stride_y_t, stride_y_d = raw_y.stride()
    stride_dy_b, stride_dy_t, stride_dy_d = dq.stride()
    stride_dx_b, stride_dx_t, stride_dx_d = dx.stride()
    grid = (triton.cdiv(D, BACKWARD_BLOCK_D), num_time_blocks, GROUPS * B)
    _grouped_backward_kernel[grid](
        x=x,
        raw_y=raw_y,
        weight=weight,
        dq=dq,
        dk=dk,
        dv=dv,
        dx=dx,
        dw_partials=dw_partials,
        B=B,
        T=T,
        stride_x_b=stride_x_b,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        stride_y_g=stride_y_g,
        stride_y_b=stride_y_b,
        stride_y_t=stride_y_t,
        stride_y_d=stride_y_d,
        stride_dy_b=stride_dy_b,
        stride_dy_t=stride_dy_t,
        stride_dy_d=stride_dy_d,
        stride_dx_b=stride_dx_b,
        stride_dx_t=stride_dx_t,
        stride_dx_d=stride_dx_d,
        D=D,
        PACKED_D=GROUPS * D,
        W=KERNEL_WIDTH,
        BT=CHUNK_SIZE,
        BW=triton.next_power_of_2(KERNEL_WIDTH),
        BD=BACKWARD_BLOCK_D,
        num_warps=BACKWARD_NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return dx, dw_partials.sum(0).to(weight)


class _QKVCausalConv1d(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        D = _validate_qkv_conv_inputs(x, weight)
        ctx.save_for_backward(x, weight)
        ctx.set_materialize_grads(False)
        q = _call_installed_forward(x[..., :D], weight[:D])
        k = _call_installed_forward(x[..., D : 2 * D], weight[D : 2 * D])
        v = _call_installed_forward(x[..., 2 * D :], weight[2 * D :])
        return q, k, v

    @staticmethod
    def backward(
        ctx: Any,
        dq: torch.Tensor | None,
        dk: torch.Tensor | None,
        dv: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if dq is None and dk is None and dv is None:
            return None, None
        x, weight = ctx.saved_tensors
        return _grouped_backward(x, weight, dq, dk, dv)


def qkv_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply fixed-length SiLU causal convolution to packed Q/K/V channels."""

    d = _validate_qkv_conv_inputs(x, weight)
    if x.shape[1] == 1:
        return (
            _call_installed_step(x[..., :d], weight[:d]),
            _call_installed_step(x[..., d : 2 * d], weight[d : 2 * d]),
            _call_installed_step(x[..., 2 * d :], weight[2 * d :]),
        )
    if not torch.is_grad_enabled() or not (x.requires_grad or weight.requires_grad):
        return (
            _call_installed_forward(x[..., :d], weight[:d]),
            _call_installed_forward(x[..., d : 2 * d], weight[d : 2 * d]),
            _call_installed_forward(x[..., 2 * d :], weight[2 * d :]),
        )
    return _QKVCausalConv1d.apply(x, weight)

# ---- Frozen iso_projection.py -----------------------------------------------
class PackedLinear(nn.Module):
    """One canonical biasless weight split into output segments."""

    def __init__(
        self,
        in_features: int,
        segment_sizes: Sequence[int],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        sizes = tuple(int(size) for size in segment_sizes)
        if in_features <= 0 or not sizes or any(size <= 0 for size in sizes):
            raise ValueError(
                f"packed linear dimensions must be positive: {in_features=}, {sizes=}"
            )
        self.in_features = int(in_features)
        self.out_features = sum(sizes)
        self.segment_sizes = sizes
        # Construction is RNG-neutral. Initializer-compatible slice modules at
        # the original positions initialize this tensor in the original order.
        self.weight = nn.Parameter(
            torch.empty(
                self.out_features,
                self.in_features,
                device=device,
                dtype=dtype,
            )
        )

    @classmethod
    def from_linears(cls, linears: Sequence[nn.Linear]) -> "PackedLinear":
        if not linears:
            raise ValueError("at least one source linear is required")
        in_features = linears[0].in_features
        if any(
            not isinstance(layer, nn.Linear) or layer.in_features != in_features
            for layer in linears
        ):
            raise ValueError("all packed source linears must share their input width")
        packed = cls(
            in_features,
            tuple(layer.out_features for layer in linears),
            device=linears[0].weight.device,
            dtype=linears[0].weight.dtype,
        )
        with torch.no_grad():
            packed.weight.copy_(torch.cat(tuple(layer.weight for layer in linears), dim=0))
        return packed

    def forward_packed(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, None)

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.forward_packed(input).split(self.segment_sizes, dim=-1)


class PackedLinearSlice(nn.Linear):
    """Non-owning linear view that preserves initialization traversal order."""

    def __init__(
        self,
        owner: PackedLinear,
        segment: int,
        *,
        bias: torch.Tensor | None,
    ) -> None:
        if segment < 0 or segment >= len(owner.segment_sizes):
            raise ValueError(f"segment {segment} is out of range for {owner.segment_sizes}")
        nn.Module.__init__(self)
        self.in_features = owner.in_features
        self.out_features = owner.segment_sizes[segment]
        self._segment = int(segment)
        object.__setattr__(self, "_packed_owner", owner)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            if tuple(bias.shape) != (self.out_features,):
                raise ValueError(
                    f"bias shape {tuple(bias.shape)} does not match {self.out_features}"
                )
            self.bias = nn.Parameter(bias.detach().clone())

    @property
    def weight(self) -> torch.Tensor:
        owner = object.__getattribute__(self, "_packed_owner")
        offset = sum(owner.segment_sizes[: self._segment])
        return owner.weight[offset : offset + self.out_features]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


class PackedQKV(nn.Module):
    """Canonical QKV projection and depthwise-convolution weights."""

    def __init__(
        self,
        hidden_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.hidden_size = int(hidden_size)
        self.out_features = 3 * self.hidden_size
        self.kernel_size = KERNEL_WIDTH
        self.projection_weight = nn.Parameter(
            torch.empty(
                self.out_features,
                self.hidden_size,
                device=device,
                dtype=dtype,
            )
        )
        self.convolution_weight = nn.Parameter(
            torch.empty(
                self.out_features,
                self.kernel_size,
                device=device,
                dtype=dtype,
            )
        )

    @classmethod
    def from_modules(
        cls,
        projections: Sequence[nn.Linear],
        convolutions: Sequence[nn.Module],
    ) -> "PackedQKV":
        if len(projections) != 3 or len(convolutions) != 3:
            raise ValueError("packed QKV requires three projections and three convolutions")
        hidden_size = projections[0].in_features
        if any(
            not isinstance(projection, nn.Linear)
            or projection.in_features != hidden_size
            or projection.out_features != hidden_size
            or projection.bias is not None
            for projection in projections
        ):
            raise ValueError("Q/K/V projections must be biasless square linear modules")

        shapes = tuple(tuple(convolution.weight.shape) for convolution in convolutions)
        expected_shape = (hidden_size, 1, KERNEL_WIDTH)
        if any(shape != expected_shape for shape in shapes):
            raise ValueError(
                f"Q/K/V convolutions must be depthwise {expected_shape}, got {shapes}"
            )
        if any(getattr(convolution, "bias", None) is not None for convolution in convolutions):
            raise ValueError("Q/K/V convolutions must be biasless")
        if any(
            getattr(convolution, "activation", None) not in ("silu", "swish")
            for convolution in convolutions
        ):
            raise ValueError("Q/K/V convolutions must use SiLU activation")
        if any(getattr(convolution, "backend", None) != "triton" for convolution in convolutions):
            raise ValueError("Q/K/V convolutions must use the Triton backend")

        packed = cls(
            hidden_size,
            device=projections[0].weight.device,
            dtype=projections[0].weight.dtype,
        )
        with torch.no_grad():
            packed.projection_weight.copy_(
                torch.cat(tuple(projection.weight for projection in projections), dim=0)
            )
            packed.convolution_weight.copy_(
                torch.cat(tuple(convolution.weight.squeeze(1) for convolution in convolutions), dim=0)
            )
        return packed

    def project_packed(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.projection_weight, None)

    def forward(
        self,
        input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input.ndim != 3 or input.shape[-1] != self.hidden_size:
            raise ValueError(
                f"packed QKV input must be [B,T,{self.hidden_size}], got {tuple(input.shape)}"
            )
        if input.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(f"packed QKV input must be bfloat16 or float32, got {input.dtype}")
        packed = self.project_packed(input)
        return qkv_causal_conv1d(packed, self.convolution_weight)


class PackedQKVLinearSlice(nn.Linear):
    """Non-owning QKV projection view used for initialization and inspection."""

    def __init__(self, owner: PackedQKV, segment: int) -> None:
        if segment not in (0, 1, 2):
            raise ValueError(f"QKV segment must be 0, 1, or 2, got {segment}")
        nn.Module.__init__(self)
        self.in_features = owner.hidden_size
        self.out_features = owner.hidden_size
        self._segment = int(segment)
        object.__setattr__(self, "_packed_owner", owner)
        self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        owner = object.__getattribute__(self, "_packed_owner")
        start = self._segment * owner.hidden_size
        return owner.projection_weight[start : start + owner.hidden_size]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, None)


def pack_f_omega_r_projection(module: nn.Module) -> PackedLinear:
    """Install the canonical ``[f0, omega, r]`` weight and non-owning views."""

    f_proj = getattr(module, "f_proj")
    f0 = f_proj[0]
    omega_proj = getattr(module, "omega_proj")
    r_proj = getattr(module, "r_proj")
    if not all(isinstance(layer, nn.Linear) for layer in (f0, omega_proj, r_proj)):
        raise TypeError("Isotropic KDN f0/omega/r projections must be linear modules")
    if f0.bias is not None or omega_proj.bias is None or r_proj.bias is None:
        raise ValueError("Isotropic KDN requires biasless f0 and biased omega/r projections")

    owner = PackedLinear.from_linears((f0, omega_proj, r_proj))
    module.add_module("f_omega_r_proj", owner)
    f_proj[0] = PackedLinearSlice(owner, 0, bias=None)
    module.omega_proj = PackedLinearSlice(owner, 1, bias=omega_proj.bias)
    module.r_proj = PackedLinearSlice(owner, 2, bias=r_proj.bias)
    return owner


def pack_qkv_projection(module: nn.Module) -> PackedQKV:
    """Install canonical QKV weights and non-owning views."""

    projection_names = ("q_proj", "k_proj", "v_proj")
    convolution_names = ("q_conv1d", "k_conv1d", "v_conv1d")
    projections = tuple(getattr(module, name) for name in projection_names)
    convolutions = tuple(getattr(module, name) for name in convolution_names)
    owner = PackedQKV.from_modules(projections, convolutions)
    module.add_module("qkv_pack", owner)
    for segment, name in enumerate(projection_names):
        setattr(module, name, PackedQKVLinearSlice(owner, segment))
    for name in convolution_names:
        delattr(module, name)
    return owner

# ---- Frozen iso_gate.py -----------------------------------------------------
ISO_GATE_CHUNK_SIZE = 64
ISO_GATE_NUM_WARPS = 4
ISO_GATE_NUM_STAGES = 2
ISO_GATE_REDUCE_NUM_WARPS = 4
ISO_GATE_REDUCE_NUM_STAGES = 1


@triton.jit
def _iso_kdn_gate_precumsum_fwd_kernel(
    g_raw_ptr,
    omega_r_raw_ptr,
    omega_bias_ptr,
    r_bias_ptr,
    stride_omega_r_b,
    stride_omega_r_t,
    A_log_ptr,
    dt_bias_ptr,
    initial_precision_param_ptr,
    g_cumsum_ptr,
    a_ptr,
    omega_ptr,
    r_ptr,
    initial_precision_ptr,
    T,
    NT,
    omega_min,
    r_min,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    LOG2E: tl.constexpr,
):
    i_bc = tl.program_id(0)
    i_h = tl.program_id(1)
    i_b = i_bc // NT
    i_c = i_bc % NT

    offs_t_1d = i_c * BT + tl.arange(0, BT)
    offs_k_1d = tl.arange(0, BK)
    offs_t = offs_t_1d[:, None]
    offs_k = offs_k_1d[None, :]
    mask_t_1d = offs_t_1d < T
    mask_k_1d = offs_k_1d < K
    mask = mask_t_1d[:, None] & mask_k_1d[None, :]

    offs_g = ((i_b * T + offs_t) * H + i_h) * K + offs_k
    b_g_raw = tl.load(g_raw_ptr + offs_g, mask=mask, other=0.0).to(tl.float32)
    b_bias = tl.load(
        dt_bias_ptr + i_h * K + offs_k_1d,
        mask=mask_k_1d,
        other=0.0,
    ).to(tl.float32)
    b_A = tl.load(A_log_ptr + i_h).to(tl.float32)

    b_g_log = -exp(b_A) * softplus(b_g_raw + b_bias[None, :])
    b_g_log = tl.where(mask, b_g_log, 0.0)
    # FLA's KDA kernels use exp2, so their cumulative gate is base-2.  The
    # inclusive prefix restarts at every 64-token KDA chunk.
    b_g_cumsum = tl.cumsum(b_g_log, axis=0) * LOG2E
    tl.store(g_cumsum_ptr + offs_g, b_g_cumsum, mask=mask)

    b_alpha_sq = tl.where(mask, exp(2.0 * b_g_log), 0.0)
    b_a = tl.sum(b_alpha_sq, axis=1) / K
    offs_a = (i_b * T + offs_t_1d) * H + i_h
    tl.store(a_ptr + offs_a, b_a, mask=mask_t_1d)

    offs_omega_r = i_b * stride_omega_r_b + offs_t_1d * stride_omega_r_t + i_h
    b_omega_bias = tl.load(omega_bias_ptr + i_h).to(tl.float32)
    b_r_bias = tl.load(r_bias_ptr + i_h).to(tl.float32)
    b_omega_raw = tl.load(omega_r_raw_ptr + offs_omega_r, mask=mask_t_1d, other=0.0).to(tl.float32) + b_omega_bias
    b_r_raw = tl.load(omega_r_raw_ptr + offs_omega_r + H, mask=mask_t_1d, other=0.0).to(tl.float32) + b_r_bias
    b_omega = omega_min + softplus(b_omega_raw)
    b_r = r_min + softplus(b_r_raw)
    tl.store(omega_ptr + offs_a, b_omega, mask=mask_t_1d)
    tl.store(r_ptr + offs_a, b_r, mask=mask_t_1d)

    # Exactly one batch-chunk program owns and computes each head's prior.
    if i_bc == 0:
        b_initial_precision_param = tl.load(initial_precision_param_ptr + i_h).to(tl.float32)
        b_initial_precision = 0.1 + softplus(b_initial_precision_param)
        tl.store(initial_precision_ptr + i_h, b_initial_precision)


@triton.jit
def _iso_kdn_gate_precumsum_bwd_kernel(
    g_raw_ptr,
    omega_r_raw_ptr,
    omega_bias_ptr,
    r_bias_ptr,
    stride_omega_r_b,
    stride_omega_r_t,
    A_log_ptr,
    dt_bias_ptr,
    initial_precision_param_ptr,
    a_ptr,
    dg_pre_reverse_ptr,
    da_ptr,
    domega_ptr,
    dr_ptr,
    d_initial_precision_ptr,
    dg_raw_ptr,
    domega_r_raw_ptr,
    domega_r_bias_partial_ptr,
    dA_partial_ptr,
    ddt_partial_ptr,
    d_initial_precision_param_ptr,
    T,
    NT,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    HAS_DG_PRE_REVERSE: tl.constexpr,
    HAS_DA: tl.constexpr,
    HAS_DOMEGA: tl.constexpr,
    HAS_DR: tl.constexpr,
    HAS_DINITIAL_PRECISION: tl.constexpr,
    NEED_DG_RAW: tl.constexpr,
    NEED_DOMEGA_RAW: tl.constexpr,
    NEED_DR_RAW: tl.constexpr,
    NEED_DOMEGA_BIAS: tl.constexpr,
    NEED_DR_BIAS: tl.constexpr,
    NEED_DA_LOG: tl.constexpr,
    NEED_DDT_BIAS: tl.constexpr,
    NEED_DINITIAL_PRECISION_PARAM: tl.constexpr,
):
    i_bc = tl.program_id(0)
    i_h = tl.program_id(1)
    i_b = i_bc // NT
    i_c = i_bc % NT

    offs_t_1d = i_c * BT + tl.arange(0, BT)
    offs_k_1d = tl.arange(0, BK)
    offs_t = offs_t_1d[:, None]
    offs_k = offs_k_1d[None, :]
    mask_t_1d = offs_t_1d < T
    mask_k_1d = offs_k_1d < K
    mask = mask_t_1d[:, None] & mask_k_1d[None, :]
    offs_g = ((i_b * T + offs_t) * H + i_h) * K + offs_k

    b_g_raw = tl.load(g_raw_ptr + offs_g, mask=mask, other=0.0).to(tl.float32)
    b_bias = tl.load(
        dt_bias_ptr + i_h * K + offs_k_1d,
        mask=mask_k_1d,
        other=0.0,
    ).to(tl.float32)
    b_A = tl.load(A_log_ptr + i_h).to(tl.float32)
    b_neg_exp_A = -exp(b_A)
    b_x = b_g_raw + b_bias[None, :]
    b_g_log = b_neg_exp_A * softplus(b_x)

    offs_a = (i_b * T + offs_t_1d) * H + i_h
    offs_omega_r = i_b * stride_omega_r_b + offs_t_1d * stride_omega_r_t + i_h
    offs_domega_r = (i_b * T + offs_t_1d) * (2 * H) + i_h
    b_da_total = tl.zeros((BT,), dtype=tl.float32)
    if HAS_DA:
        b_da_total += tl.load(da_ptr + offs_a, mask=mask_t_1d, other=0.0).to(tl.float32)

    b_domega_raw = tl.zeros((BT,), dtype=tl.float32)
    if HAS_DOMEGA:
        b_domega = tl.load(domega_ptr + offs_a, mask=mask_t_1d, other=0.0).to(tl.float32)
        b_omega_bias = tl.load(omega_bias_ptr + i_h).to(tl.float32)
        b_omega_raw = (
            tl.load(omega_r_raw_ptr + offs_omega_r, mask=mask_t_1d, other=0.0).to(tl.float32)
            + b_omega_bias
        )
        b_domega_raw = b_domega * tl.sigmoid(b_omega_raw)
    if NEED_DOMEGA_RAW:
        tl.store(domega_r_raw_ptr + offs_domega_r, b_domega_raw, mask=mask_t_1d)

    b_dr_raw = tl.zeros((BT,), dtype=tl.float32)
    if HAS_DR:
        b_dr = tl.load(dr_ptr + offs_a, mask=mask_t_1d, other=0.0).to(tl.float32)
        b_r_bias = tl.load(r_bias_ptr + i_h).to(tl.float32)
        b_r_raw = (
            tl.load(omega_r_raw_ptr + offs_omega_r + H, mask=mask_t_1d, other=0.0).to(tl.float32)
            + b_r_bias
        )
        b_dr_raw = b_dr * tl.sigmoid(b_r_raw)
    if NEED_DR_RAW:
        tl.store(domega_r_raw_ptr + offs_domega_r + H, b_dr_raw, mask=mask_t_1d)

    if NEED_DOMEGA_BIAS:
        tl.store(
            domega_r_bias_partial_ptr + i_bc * (2 * H) + i_h,
            tl.sum(tl.where(mask_t_1d, b_domega_raw, 0.0), axis=0),
        )
    if NEED_DR_BIAS:
        tl.store(
            domega_r_bias_partial_ptr + i_bc * (2 * H) + H + i_h,
            tl.sum(tl.where(mask_t_1d, b_dr_raw, 0.0), axis=0),
        )

    b_dg_log = tl.zeros((BT, BK), dtype=tl.float32)
    if HAS_DG_PRE_REVERSE:
        b_dg_pre = tl.load(
            dg_pre_reverse_ptr + offs_g,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        # Deliberately unscaled.  FLA's exp2-based KDA kernels return their
        # pre-reverse cotangent in natural-log coordinates, so ln(2) from
        # exp2 cancels the LOG2E used by the forward prefix.
        b_dg_log += tl.cumsum(b_dg_pre, axis=0, reverse=True)
    if HAS_DA:
        b_dg_log += b_da_total[:, None] * (2.0 / K) * exp(2.0 * b_g_log)
    b_dg_log = tl.where(mask, b_dg_log, 0.0)

    # FLA's approximate softplus is intentionally paired with sigmoid here.
    b_dg_raw = b_dg_log * b_neg_exp_A * tl.sigmoid(b_x)
    if NEED_DG_RAW:
        tl.store(dg_raw_ptr + offs_g, b_dg_raw, mask=mask)

    if NEED_DA_LOG:
        b_dA_term = tl.where(mask, b_dg_log * b_g_log, 0.0)
        b_dA = tl.sum(tl.sum(b_dA_term, axis=1), axis=0)
        tl.store(dA_partial_ptr + i_bc * H + i_h, b_dA)

    if NEED_DDT_BIAS:
        b_ddt = tl.sum(tl.where(mask, b_dg_raw, 0.0), axis=0)
        offs_partial = (i_bc * H + i_h) * K + offs_k_1d
        tl.store(ddt_partial_ptr + offs_partial, b_ddt, mask=mask_k_1d)

    if NEED_DINITIAL_PRECISION_PARAM:
        if i_bc == 0:
            if HAS_DINITIAL_PRECISION:
                b_d_initial_precision = tl.load(d_initial_precision_ptr + i_h).to(tl.float32)
                b_initial_precision_param = tl.load(initial_precision_param_ptr + i_h).to(tl.float32)
                b_d_initial_precision_param = b_d_initial_precision * tl.sigmoid(b_initial_precision_param)
            else:
                b_d_initial_precision_param = 0.0
            tl.store(d_initial_precision_param_ptr + i_h, b_d_initial_precision_param)


@triton.jit
def _reduce_dA_kernel(partial_ptr, out_ptr, NP, H: tl.constexpr, BNP: tl.constexpr):
    i_h = tl.program_id(0)
    offs_n = tl.arange(0, BNP)
    b_partial = tl.load(
        partial_ptr + offs_n * H + i_h,
        mask=offs_n < NP,
        other=0.0,
    ).to(tl.float32)
    tl.store(out_ptr + i_h, tl.sum(b_partial, axis=0))


@triton.jit
def _reduce_ddt_kernel(
    partial_ptr,
    out_ptr,
    NP,
    H: tl.constexpr,
    K: tl.constexpr,
    BNP: tl.constexpr,
):
    i_hk = tl.program_id(0)
    i_h = i_hk // K
    i_k = i_hk % K
    offs_n = tl.arange(0, BNP)
    b_partial = tl.load(
        partial_ptr + (offs_n * H + i_h) * K + i_k,
        mask=offs_n < NP,
        other=0.0,
    ).to(tl.float32)
    tl.store(out_ptr + i_hk, tl.sum(b_partial, axis=0))


@triton.jit
def _reduce_domega_r_bias_kernel(partial_ptr, out_ptr, NP, H: tl.constexpr, BNP: tl.constexpr):
    i_omega_r = tl.program_id(0)
    offs_n = tl.arange(0, BNP)
    b_partial = tl.load(
        partial_ptr + offs_n * (2 * H) + i_omega_r,
        mask=offs_n < NP,
        other=0.0,
    ).to(tl.float32)
    tl.store(out_ptr + i_omega_r, tl.sum(b_partial, axis=0))


def _validate_iso_gate_inputs(
    g_raw: torch.Tensor,
    omega_r_raw: torch.Tensor,
    omega_bias: torch.Tensor,
    r_bias: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_precision_param: torch.Tensor,
    omega_min: float,
    r_min: float,
) -> tuple[int, int, int, int]:
    if g_raw.ndim != 4:
        raise ValueError(f"g_raw must have shape [B,T,H,K], got {tuple(g_raw.shape)}")
    B, T, H, K = g_raw.shape
    if K == 0:
        raise ValueError("g_raw's K dimension must be nonzero")
    if omega_r_raw.shape != (B, T, 2 * H):
        raise ValueError(f"omega_r_raw must have shape {(B, T, 2 * H)}, got {tuple(omega_r_raw.shape)}")
    if omega_r_raw.stride(-1) != 1:
        raise ValueError(f"omega_r_raw's head dimension must be contiguous, got strides={omega_r_raw.stride()}")
    for name, value in (("omega_bias", omega_bias), ("r_bias", r_bias)):
        if value.shape != (H,):
            raise ValueError(f"{name} must have shape [{H}], got {tuple(value.shape)}")
    if A_log.shape != (H,):
        raise ValueError(f"A_log must have shape [{H}], got {tuple(A_log.shape)}")
    if dt_bias.shape != (H * K,):
        raise ValueError(f"dt_bias must have shape [{H * K}], got {tuple(dt_bias.shape)}")
    if initial_precision_param.shape != (H,):
        raise ValueError(f"initial_precision_param must have shape [{H}], got {tuple(initial_precision_param.shape)}")
    inputs = (g_raw, omega_r_raw, omega_bias, r_bias, A_log, dt_bias, initial_precision_param)
    if not all(value.is_cuda for value in inputs):
        raise ValueError("_iso_kdn_gate_precumsum_paired is a CUDA-only Triton op")
    if any(value.device != g_raw.device for value in inputs[1:]):
        raise ValueError("all scalar-preparation inputs must be on the same CUDA device")
    for name, value in (("g_raw", g_raw), ("omega_r_raw", omega_r_raw)):
        if value.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(f"{name} must be bf16 or fp32, got {value.dtype}")
    if any(value.dtype != torch.float32 for value in (omega_bias, r_bias, A_log, dt_bias, initial_precision_param)):
        raise TypeError("omega_bias, r_bias, A_log, dt_bias, and initial_precision_param must be FP32")
    for name, value in (("omega_min", omega_min), ("r_min", r_min)):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a Python scalar, got {type(value).__name__}")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")
    return B, T, H, K


def _iso_kdn_gate_precumsum_fwd(
    g_raw: torch.Tensor,
    omega_r_raw: torch.Tensor,
    omega_bias: torch.Tensor,
    r_bias: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_precision_param: torch.Tensor,
    omega_min: float,
    r_min: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = g_raw.shape
    g_cumsum = torch.empty_like(g_raw, dtype=torch.float32)
    a = torch.empty((B, T, H), device=g_raw.device, dtype=torch.float32)
    omega = torch.empty_like(a)
    r = torch.empty_like(a)
    initial_precision = torch.empty_like(initial_precision_param, dtype=torch.float32)
    if B == 0 or T == 0:
        # Empty token streams still expose the learned prior.  This exceptional
        # path does not affect production timing and the custom backward below
        # supplies its VJP explicitly.
        initial_precision.copy_(torch.nn.functional.softplus(initial_precision_param.float()) + 0.1)
        return g_cumsum, a, omega, r, initial_precision

    NT = triton.cdiv(T, ISO_GATE_CHUNK_SIZE)
    grid = (B * NT, H)
    with torch.cuda.device(g_raw.device):
        _iso_kdn_gate_precumsum_fwd_kernel[grid](
            g_raw,
            omega_r_raw,
            omega_bias,
            r_bias,
            omega_r_raw.stride(0),
            omega_r_raw.stride(1),
            A_log,
            dt_bias,
            initial_precision_param,
            g_cumsum,
            a,
            omega,
            r,
            initial_precision,
            T,
            NT,
            float(omega_min),
            float(r_min),
            H=H,
            K=K,
            BT=ISO_GATE_CHUNK_SIZE,
            BK=triton.next_power_of_2(K),
            LOG2E=RCP_LN2,
            num_warps=ISO_GATE_NUM_WARPS,
            num_stages=ISO_GATE_NUM_STAGES,
        )
    return g_cumsum, a, omega, r, initial_precision


def _iso_kdn_gate_precumsum_bwd(
    g_raw: torch.Tensor,
    omega_r_raw: torch.Tensor,
    omega_bias: torch.Tensor,
    r_bias: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_precision_param: torch.Tensor,
    a: torch.Tensor,
    dg_pre_reverse: torch.Tensor | None,
    da: torch.Tensor | None,
    domega: torch.Tensor | None,
    dr: torch.Tensor | None,
    d_initial_precision: torch.Tensor | None,
    needs_input_grad: tuple[bool, bool, bool, bool, bool, bool, bool],
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    B, T, H, K = g_raw.shape
    (
        need_dg_raw,
        need_domega_r_raw,
        need_domega_bias,
        need_dr_bias,
        need_dA_log,
        need_ddt_bias,
        need_d_initial_precision_param,
    ) = needs_input_grad

    if B == 0 or T == 0:
        d_initial_precision_param = None
        if need_d_initial_precision_param:
            d_initial_precision_param = (
                torch.zeros_like(initial_precision_param)
                if d_initial_precision is None
                else d_initial_precision.float() * torch.sigmoid(initial_precision_param)
            )
        return (
            torch.zeros_like(g_raw) if need_dg_raw else None,
            torch.zeros_like(omega_r_raw) if need_domega_r_raw else None,
            torch.zeros_like(omega_bias) if need_domega_bias else None,
            torch.zeros_like(r_bias) if need_dr_bias else None,
            torch.zeros_like(A_log, dtype=torch.float32) if need_dA_log else None,
            torch.zeros_like(dt_bias, dtype=torch.float32) if need_ddt_bias else None,
            d_initial_precision_param,
        )

    if dg_pre_reverse is not None:
        if dg_pre_reverse.shape != g_raw.shape:
            raise ValueError(
                f"dg_pre_reverse must have shape {tuple(g_raw.shape)}, "
                f"got {tuple(dg_pre_reverse.shape)}"
            )
        dg_pre_reverse = dg_pre_reverse.contiguous()
    if da is not None:
        if da.shape != (B, T, H):
            raise ValueError(f"da must have shape {(B, T, H)}, got {tuple(da.shape)}")
        da = da.contiguous()
    for name, value in (("domega", domega), ("dr", dr)):
        if value is not None:
            if value.shape != (B, T, H):
                raise ValueError(f"{name} must have shape {(B, T, H)}, got {tuple(value.shape)}")
            if value.device != g_raw.device:
                raise ValueError(f"{name} must be on {g_raw.device}, got {value.device}")
    domega = domega.contiguous() if domega is not None else None
    dr = dr.contiguous() if dr is not None else None
    if d_initial_precision is not None:
        if d_initial_precision.shape != (H,):
            raise ValueError(f"d_initial_precision must have shape {(H,)}, got {tuple(d_initial_precision.shape)}")
        if d_initial_precision.device != g_raw.device:
            raise ValueError(f"d_initial_precision must be on {g_raw.device}, got {d_initial_precision.device}")
        d_initial_precision = d_initial_precision.contiguous()

    NT = triton.cdiv(T, ISO_GATE_CHUNK_SIZE)
    NP = B * NT
    dg_raw = torch.empty_like(g_raw) if need_dg_raw else g_raw.new_empty((0,))
    domega_r_raw = (
        torch.empty(omega_r_raw.shape, device=omega_r_raw.device, dtype=omega_r_raw.dtype)
        if need_domega_r_raw
        else omega_r_raw.new_empty((0,))
    )
    need_domega_r_bias = need_domega_bias or need_dr_bias
    domega_r_bias_partial = (
        torch.empty((NP, 2 * H), device=g_raw.device, dtype=torch.float32)
        if need_domega_r_bias
        else g_raw.new_empty((0,), dtype=torch.float32)
    )
    dA_partial = (
        torch.empty((NP, H), device=g_raw.device, dtype=torch.float32)
        if need_dA_log
        else g_raw.new_empty((0,), dtype=torch.float32)
    )
    ddt_partial = (
        torch.empty((NP, H, K), device=g_raw.device, dtype=torch.float32)
        if need_ddt_bias
        else g_raw.new_empty((0,), dtype=torch.float32)
    )
    dA_log = torch.empty_like(A_log, dtype=torch.float32) if need_dA_log else None
    ddt_bias = torch.empty_like(dt_bias, dtype=torch.float32) if need_ddt_bias else None
    d_initial_precision_param = torch.empty_like(initial_precision_param, dtype=torch.float32) if need_d_initial_precision_param else None
    domega_r_bias = (
        torch.empty((2 * H,), device=g_raw.device, dtype=torch.float32)
        if need_domega_r_bias
        else None
    )

    grid = (NP, H)
    BNP = triton.next_power_of_2(NP)
    with torch.cuda.device(g_raw.device):
        _iso_kdn_gate_precumsum_bwd_kernel[grid](
            g_raw,
            omega_r_raw,
            omega_bias,
            r_bias,
            omega_r_raw.stride(0),
            omega_r_raw.stride(1),
            A_log,
            dt_bias,
            initial_precision_param,
            a,
            dg_pre_reverse,
            da,
            domega,
            dr,
            d_initial_precision,
            dg_raw,
            domega_r_raw,
            domega_r_bias_partial,
            dA_partial,
            ddt_partial,
            d_initial_precision_param,
            T,
            NT,
            H=H,
            K=K,
            BT=ISO_GATE_CHUNK_SIZE,
            BK=triton.next_power_of_2(K),
            HAS_DG_PRE_REVERSE=dg_pre_reverse is not None,
            HAS_DA=da is not None,
            HAS_DOMEGA=domega is not None,
            HAS_DR=dr is not None,
            HAS_DINITIAL_PRECISION=d_initial_precision is not None,
            NEED_DG_RAW=need_dg_raw,
            NEED_DOMEGA_RAW=need_domega_r_raw,
            NEED_DR_RAW=need_domega_r_raw,
            NEED_DOMEGA_BIAS=need_domega_bias,
            NEED_DR_BIAS=need_dr_bias,
            NEED_DA_LOG=need_dA_log,
            NEED_DDT_BIAS=need_ddt_bias,
            NEED_DINITIAL_PRECISION_PARAM=need_d_initial_precision_param,
            num_warps=ISO_GATE_NUM_WARPS,
            num_stages=ISO_GATE_NUM_STAGES,
        )
        if need_dA_log:
            _reduce_dA_kernel[(H,)](
                dA_partial,
                dA_log,
                NP,
                H=H,
                BNP=BNP,
                num_warps=ISO_GATE_REDUCE_NUM_WARPS,
                num_stages=ISO_GATE_REDUCE_NUM_STAGES,
            )
        if need_ddt_bias:
            _reduce_ddt_kernel[(H * K,)](
                ddt_partial,
                ddt_bias,
                NP,
                H=H,
                K=K,
                BNP=BNP,
                num_warps=ISO_GATE_REDUCE_NUM_WARPS,
                num_stages=ISO_GATE_REDUCE_NUM_STAGES,
            )
        if need_domega_r_bias:
            _reduce_domega_r_bias_kernel[(2 * H,)](
                domega_r_bias_partial,
                domega_r_bias,
                NP,
                H=H,
                BNP=BNP,
                num_warps=ISO_GATE_REDUCE_NUM_WARPS,
                num_stages=ISO_GATE_REDUCE_NUM_STAGES,
            )
    return (
        dg_raw if need_dg_raw else None,
        domega_r_raw if need_domega_r_raw else None,
        domega_r_bias[:H] if need_domega_bias else None,
        domega_r_bias[H:] if need_dr_bias else None,
        dA_log,
        ddt_bias,
        d_initial_precision_param,
    )


class _IsoKDNGatePrecumsumFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        g_raw: torch.Tensor,
        omega_r_raw: torch.Tensor,
        omega_bias: torch.Tensor,
        r_bias: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        initial_precision_param: torch.Tensor,
        omega_min: float,
        r_min: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        g_cumsum, a, omega, r, initial_precision = _iso_kdn_gate_precumsum_fwd(
            g_raw,
            omega_r_raw,
            omega_bias,
            r_bias,
            A_log,
            dt_bias,
            initial_precision_param,
            omega_min,
            r_min,
        )
        ctx.save_for_backward(g_raw, omega_r_raw, omega_bias, r_bias, A_log, dt_bias, initial_precision_param, a)
        ctx.set_materialize_grads(False)
        return g_cumsum, a, omega, r, initial_precision

    @staticmethod
    def backward(
        ctx,
        dg_pre_reverse: torch.Tensor | None,
        da: torch.Tensor | None,
        domega: torch.Tensor | None,
        dr: torch.Tensor | None,
        d_initial_precision: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
    ]:
        g_raw, omega_r_raw, omega_bias, r_bias, A_log, dt_bias, initial_precision_param, a = ctx.saved_tensors
        dg_raw, domega_r_raw, domega_bias, dr_bias, dA_log, ddt_bias, d_initial_precision_param = _iso_kdn_gate_precumsum_bwd(
            g_raw,
            omega_r_raw,
            omega_bias,
            r_bias,
            A_log,
            dt_bias,
            initial_precision_param,
            a,
            dg_pre_reverse,
            da,
            domega,
            dr,
            d_initial_precision,
            ctx.needs_input_grad[:7],
        )
        return (
            dg_raw,
            domega_r_raw,
            domega_bias,
            dr_bias,
            dA_log,
            ddt_bias,
            d_initial_precision_param,
            None,
            None,
        )


@torch.compiler.disable
def _iso_kdn_gate_precumsum_paired(
    g_raw: torch.Tensor,
    omega_r_raw: torch.Tensor,
    omega_bias: torch.Tensor,
    r_bias: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_precision_param: torch.Tensor,
    omega_min: float,
    r_min: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return private paired ``(base2_prefix, a, omega, r, initial_precision)`` tensors.

    ``base2_prefix`` must have exactly one consumer:
    ``_chunk_kda_precumsum_paired`` in this module. Branching
    it or consuming it independently gives invalid gradients because the two
    custom Functions intentionally exchange a natural-log-coordinate
    cotangent rather than a generic tensor VJP.
    """
    _validate_iso_gate_inputs(
        g_raw,
        omega_r_raw,
        omega_bias,
        r_bias,
        A_log,
        dt_bias,
        initial_precision_param,
        omega_min,
        r_min,
    )
    return _IsoKDNGatePrecumsumFunction.apply(
        g_raw.contiguous(),
        omega_r_raw,
        omega_bias.contiguous(),
        r_bias.contiguous(),
        A_log.contiguous(),
        dt_bias.contiguous(),
        initial_precision_param.contiguous(),
        float(omega_min),
        float(r_min),
    )

# ---- Frozen iso_chunk.py ----------------------------------------------------
_CHUNK_SIZE = 64
_FORWARD_NUM_WARPS = 1
_FORWARD_NUM_STAGES = 1
_BACKWARD_NUM_WARPS = 1
_BACKWARD_NUM_STAGES = 2


def _require_same_cuda_device(
    a: torch.Tensor,
    *named_tensors: tuple[str, torch.Tensor],
) -> torch.device:
    if not a.is_cuda:
        raise ValueError(f"a must be a CUDA tensor; got device={a.device}")
    device = a.device
    for name, tensor in named_tensors:
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor; got device={tensor.device}")
        if tensor.device != device:
            raise ValueError(
                f"all tensors must share a CUDA device; a is on {device}, "
                f"but {name} is on {tensor.device}"
            )
    return device


@triton.jit(do_not_specialize=["T", "posterior_scale"])
def _iso_kdn_gain_forward_map_kernel(
    a_ptr,
    omega_ptr,
    r_ptr,
    pm_ptr,         # [B, NT, H, 4] fp32 chunk maps
    T,
    posterior_scale,
    NT: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    UNIT_POSTERIOR_SCALE: tl.constexpr,
):
    i_nt = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H
    i_h = pid % H

    base = i_b * T * H + i_h
    t0 = i_nt * BT
    t1 = tl.minimum(t0 + BT, T)

    mA = 1.0
    mB = 0.0
    mC = 0.0
    mD = 1.0
    for t in range(t0, t1):
        off = base + t * H
        a = tl.load(a_ptr + off).to(tl.float32)
        omega = tl.load(omega_ptr + off).to(tl.float32)
        r = tl.load(r_ptr + off).to(tl.float32)

        # M_t maps p_{t-1} to p_t. Left multiplication preserves time order.
        tA = r * a
        tB = r * omega
        if UNIT_POSTERIOR_SCALE:
            tC = a
            tD = r + omega
        else:
            tC = posterior_scale * a
            tD = r + posterior_scale * omega
        nA = tA * mA + tB * mC
        nB = tA * mB + tB * mD
        nC = tC * mA + tD * mC
        nD = tC * mB + tD * mD

        scale = tl.maximum(
            tl.maximum(tl.abs(nA), tl.abs(nB)),
            tl.maximum(tl.abs(nC), tl.abs(nD)),
        )
        inv_scale = 1.0 / tl.maximum(scale, 1e-30)
        mA = nA * inv_scale
        mB = nB * inv_scale
        mC = nC * inv_scale
        mD = nD * inv_scale

    pm_base = (i_b * NT * H + i_nt * H + i_h) * 4
    tl.store(pm_ptr + pm_base + 0, mA)
    tl.store(pm_ptr + pm_base + 1, mB)
    tl.store(pm_ptr + pm_base + 2, mC)
    tl.store(pm_ptr + pm_base + 3, mD)


@triton.jit
def _iso_kdn_gain_forward_carry_kernel(
    pm_ptr,
    initial_precision_ptr,
    carry_ptr,      # [B, NT, H] fp32 exclusive incoming p
    NT: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(0)
    i_b = pid // H
    i_h = pid % H

    initial_precision = tl.load(initial_precision_ptr + i_h).to(tl.float32)
    p = 1.0 / initial_precision
    for nt in range(0, NT):
        carry_off = i_b * NT * H + nt * H + i_h
        tl.store(carry_ptr + carry_off, p)
        pm_base = (i_b * NT * H + nt * H + i_h) * 4
        mA = tl.load(pm_ptr + pm_base + 0).to(tl.float32)
        mB = tl.load(pm_ptr + pm_base + 1).to(tl.float32)
        mC = tl.load(pm_ptr + pm_base + 2).to(tl.float32)
        mD = tl.load(pm_ptr + pm_base + 3).to(tl.float32)
        p = (mA * p + mB) / (mC * p + mD)


@triton.jit(do_not_specialize=["T", "posterior_scale"])
def _iso_kdn_gain_forward_emit_kernel(
    a_ptr,
    omega_ptr,
    r_ptr,
    carry_ptr,
    gain_ptr,
    previous_covariance_ptr,      # optional [B,T,H] fp32 output; ignored unless SAVE_P
    T,
    posterior_scale,
    NT: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    SAVE_P: tl.constexpr,
    UNIT_POSTERIOR_SCALE: tl.constexpr,
):
    i_nt = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H
    i_h = pid % H

    base = i_b * T * H + i_h
    t0 = i_nt * BT
    t1 = tl.minimum(t0 + BT, T)
    carry_off = i_b * NT * H + i_nt * H + i_h
    p = tl.load(carry_ptr + carry_off).to(tl.float32)

    for t in range(t0, t1):
        off = base + t * H
        a = tl.load(a_ptr + off).to(tl.float32)
        omega = tl.load(omega_ptr + off).to(tl.float32)
        r = tl.load(r_ptr + off).to(tl.float32)
        if SAVE_P:
            tl.store(previous_covariance_ptr + off, p)
        z = a * p + omega
        gain_d = r + z
        gain = z / gain_d
        if UNIT_POSTERIOR_SCALE:
            # Preserve the frozen scale-one arithmetic order.
            p = r * gain
        else:
            posterior_d = r + posterior_scale * z
            p = r * z / posterior_d
        tl.store(gain_ptr + off, gain.to(gain_ptr.dtype.element_ty))


def _iso_kdn_gain_forward(
    a: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    initial_precision: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    save_state: bool,
    posterior_scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H = a.shape
    device = _require_same_cuda_device(a, ("omega", omega), ("r", r), ("initial_precision", initial_precision))
    gain = torch.empty((B, T, H), dtype=out_dtype, device=device)
    previous_covariance = (
        torch.empty((B, T, H), dtype=torch.float32, device=device)
        if save_state
        else None
    )
    if T == 0:
        return gain, previous_covariance

    NT = triton.cdiv(T, _CHUNK_SIZE)
    maps = torch.empty((B, NT, H, 4), dtype=torch.float32, device=device)
    carry = torch.empty((B, NT, H), dtype=torch.float32, device=device)
    grid = (NT, B * H)
    with torch.cuda.device(device):
        _iso_kdn_gain_forward_map_kernel[grid](
            a,
            omega,
            r,
            maps,
            T,
            posterior_scale,
            NT=NT,
            H=H,
            BT=_CHUNK_SIZE,
            UNIT_POSTERIOR_SCALE=posterior_scale == 1.0,
            num_warps=_FORWARD_NUM_WARPS,
            num_stages=_FORWARD_NUM_STAGES,
        )
        _iso_kdn_gain_forward_carry_kernel[(B * H,)](
            maps,
            initial_precision,
            carry,
            NT=NT,
            H=H,
            num_warps=_FORWARD_NUM_WARPS,
            num_stages=_FORWARD_NUM_STAGES,
        )
        _iso_kdn_gain_forward_emit_kernel[grid](
            a,
            omega,
            r,
            carry,
            gain,
            previous_covariance if previous_covariance is not None else carry,
            T,
            posterior_scale,
            NT=NT,
            H=H,
            BT=_CHUNK_SIZE,
            SAVE_P=save_state,
            UNIT_POSTERIOR_SCALE=posterior_scale == 1.0,
            num_warps=_FORWARD_NUM_WARPS,
            num_stages=_FORWARD_NUM_STAGES,
        )
    return gain, previous_covariance


@triton.jit(do_not_specialize=["T", "posterior_scale"])
def _iso_kdn_gain_backward_map_kernel(
    a_ptr,
    omega_ptr,
    r_ptr,
    dgain_ptr,
    previous_covariance_ptr,
    ab_ptr,         # [B,NT,H,2] fp32 reverse affine chunk maps
    T,
    posterior_scale,
    NT: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    UNIT_POSTERIOR_SCALE: tl.constexpr,
):
    i_nt = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H
    i_h = pid % H

    base = i_b * T * H + i_h
    t0 = i_nt * BT
    t1 = tl.minimum(t0 + BT, T)
    A_net = 1.0
    B_net = 0.0
    for t in range(t1 - 1, t0 - 1, -1):
        off = base + t * H
        a = tl.load(a_ptr + off).to(tl.float32)
        omega = tl.load(omega_ptr + off).to(tl.float32)
        r = tl.load(r_ptr + off).to(tl.float32)
        delta = tl.load(dgain_ptr + off).to(tl.float32)
        previous_covariance = tl.load(previous_covariance_ptr + off).to(tl.float32)
        z = a * previous_covariance + omega
        if UNIT_POSTERIOR_SCALE:
            d = r + z
            inv_d = 1.0 / d
            inv_d2 = inv_d * inv_d
            ratio = r * inv_d
            A_t = a * ratio * ratio
            B_t = a * delta * r * inv_d2
        else:
            gain_d = r + z
            gain_inv = 1.0 / gain_d
            gain_inv2 = gain_inv * gain_inv
            posterior_d = r + posterior_scale * z
            posterior_ratio = r / posterior_d
            A_t = a * posterior_ratio * posterior_ratio
            B_t = a * delta * r * gain_inv2
        A_net = A_t * A_net
        B_net = A_t * B_net + B_t

    ab_base = (i_b * NT * H + i_nt * H + i_h) * 2
    tl.store(ab_ptr + ab_base + 0, A_net)
    tl.store(ab_ptr + ab_base + 1, B_net)


@triton.jit
def _iso_kdn_gain_backward_carry_kernel(
    ab_ptr,
    carry_ptr,      # [B,NT,H] exclusive lambda entering each chunk high end
    NT: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(0)
    i_b = pid // H
    i_h = pid % H

    lam = 0.0
    for nt in range(NT - 1, -1, -1):
        carry_off = i_b * NT * H + nt * H + i_h
        tl.store(carry_ptr + carry_off, lam)
        ab_base = (i_b * NT * H + nt * H + i_h) * 2
        A_chunk = tl.load(ab_ptr + ab_base + 0).to(tl.float32)
        B_chunk = tl.load(ab_ptr + ab_base + 1).to(tl.float32)
        lam = A_chunk * lam + B_chunk


@triton.jit(do_not_specialize=["T", "posterior_scale"])
def _iso_kdn_gain_backward_emit_kernel(
    a_ptr,
    omega_ptr,
    r_ptr,
    initial_precision_ptr,
    dgain_ptr,
    previous_covariance_ptr,
    carry_ptr,
    da_ptr,
    domega_ptr,
    dr_ptr,
    d_initial_precision_ptr,        # [H] fp32 raw accumulator
    T,
    posterior_scale,
    NT: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    UNIT_POSTERIOR_SCALE: tl.constexpr,
):
    i_nt = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H
    i_h = pid % H

    base = i_b * T * H + i_h
    t0 = i_nt * BT
    t1 = tl.minimum(t0 + BT, T)
    carry_off = i_b * NT * H + i_nt * H + i_h
    lam = tl.load(carry_ptr + carry_off).to(tl.float32)

    for t in range(t1 - 1, t0 - 1, -1):
        off = base + t * H
        a = tl.load(a_ptr + off).to(tl.float32)
        omega = tl.load(omega_ptr + off).to(tl.float32)
        r = tl.load(r_ptr + off).to(tl.float32)
        delta = tl.load(dgain_ptr + off).to(tl.float32)
        previous_covariance = tl.load(previous_covariance_ptr + off).to(tl.float32)

        z = a * previous_covariance + omega
        if UNIT_POSTERIOR_SCALE:
            d = r + z
            inv_d = 1.0 / d
            inv_d2 = inv_d * inv_d
            betabar = delta + r * lam
            zbar = betabar * r * inv_d2
            # Cancellation-resistant frozen scale-one form.
            dr = z * (lam * z - delta) * inv_d2
        else:
            gain_d = r + z
            gain_inv = 1.0 / gain_d
            gain_inv2 = gain_inv * gain_inv
            posterior_d = r + posterior_scale * z
            posterior_inv = 1.0 / posterior_d
            posterior_inv2 = posterior_inv * posterior_inv
            zbar = delta * r * gain_inv2 + lam * r * r * posterior_inv2
            dr = -delta * z * gain_inv2 + lam * posterior_scale * z * z * posterior_inv2
        da = zbar * previous_covariance
        domega = zbar
        lam = a * zbar

        tl.store(da_ptr + off, da.to(da_ptr.dtype.element_ty))
        tl.store(domega_ptr + off, domega.to(domega_ptr.dtype.element_ty))
        tl.store(dr_ptr + off, dr.to(dr_ptr.dtype.element_ty))

    if i_nt == 0:
        initial_precision = tl.load(initial_precision_ptr + i_h).to(tl.float32)
        # p_-1=1/initial_precision, so its VJP is
        # -lambda_-1/initial_precision^2. The prior is shared across batch,
        # hence one atomic contribution per (b,h) stream.
        tl.atomic_add(d_initial_precision_ptr + i_h, -lam / (initial_precision * initial_precision))


def _iso_kdn_gain_backward(
    a: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    initial_precision: torch.Tensor,
    dgain: torch.Tensor,
    previous_covariance: torch.Tensor,
    *,
    posterior_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H = a.shape
    device = _require_same_cuda_device(
        a,
        ("omega", omega),
        ("r", r),
        ("initial_precision", initial_precision),
        ("dgain", dgain),
        ("previous_covariance", previous_covariance),
    )
    da = torch.empty_like(a)
    domega = torch.empty_like(omega)
    dr = torch.empty_like(r)
    d_initial_precision_accum = torch.zeros((H,), dtype=torch.float32, device=device)
    if T == 0:
        return da, domega, dr, d_initial_precision_accum.to(initial_precision.dtype)

    NT = triton.cdiv(T, _CHUNK_SIZE)
    maps = torch.empty((B, NT, H, 2), dtype=torch.float32, device=device)
    carry = torch.empty((B, NT, H), dtype=torch.float32, device=device)
    grid = (NT, B * H)
    with torch.cuda.device(device):
        _iso_kdn_gain_backward_map_kernel[grid](
            a,
            omega,
            r,
            dgain,
            previous_covariance,
            maps,
            T,
            posterior_scale,
            NT=NT,
            H=H,
            BT=_CHUNK_SIZE,
            UNIT_POSTERIOR_SCALE=posterior_scale == 1.0,
            num_warps=_BACKWARD_NUM_WARPS,
            num_stages=_BACKWARD_NUM_STAGES,
        )
        _iso_kdn_gain_backward_carry_kernel[(B * H,)](
            maps,
            carry,
            NT=NT,
            H=H,
            num_warps=_BACKWARD_NUM_WARPS,
            num_stages=_BACKWARD_NUM_STAGES,
        )
        _iso_kdn_gain_backward_emit_kernel[grid](
            a,
            omega,
            r,
            initial_precision,
            dgain,
            previous_covariance,
            carry,
            da,
            domega,
            dr,
            d_initial_precision_accum,
            T,
            posterior_scale,
            NT=NT,
            H=H,
            BT=_CHUNK_SIZE,
            UNIT_POSTERIOR_SCALE=posterior_scale == 1.0,
            num_warps=_BACKWARD_NUM_WARPS,
            num_stages=_BACKWARD_NUM_STAGES,
        )
    return da, domega, dr, d_initial_precision_accum.to(initial_precision.dtype)


class _IsoKDNGainFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        omega: torch.Tensor,
        r: torch.Tensor,
        initial_precision: torch.Tensor,
        out_dtype: torch.dtype,
        posterior_scale: float,
        needs_backward: bool,
    ) -> torch.Tensor:
        gain, previous_covariance = _iso_kdn_gain_forward(
            a,
            omega,
            r,
            initial_precision,
            out_dtype=out_dtype,
            save_state=needs_backward,
            posterior_scale=posterior_scale,
        )
        ctx.posterior_scale = posterior_scale
        ctx.has_saved_state = needs_backward
        if needs_backward:
            if previous_covariance is None:
                raise RuntimeError("Isotropic KDN gain forward did not save covariance state")
            ctx.save_for_backward(a, omega, r, initial_precision, previous_covariance)
        return gain

    @staticmethod
    def backward(
        ctx,
        dgain: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
    ]:
        if dgain is None:
            return None, None, None, None, None, None, None
        if not ctx.has_saved_state:
            raise RuntimeError("Isotropic KDN gain backward called after a no-save forward")

        a, omega, r, initial_precision, previous_covariance = ctx.saved_tensors
        da, domega, dr, d_initial_precision = _iso_kdn_gain_backward(
            a,
            omega,
            r,
            initial_precision,
            dgain.contiguous(),
            previous_covariance,
            posterior_scale=ctx.posterior_scale,
        )
        needs = ctx.needs_input_grad
        return (
            da if needs[0] else None,
            domega if needs[1] else None,
            dr if needs[2] else None,
            d_initial_precision if needs[3] else None,
            None,
            None,
            None,
        )


@torch.compiler.disable
def iso_kdn_gain(
    a: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    initial_precision: torch.Tensor,
    *,
    head_dim: int,
    out_dtype: torch.dtype | None = None,
    info_scale: Real | None = None,
) -> torch.Tensor:
    """Compute the differentiable Isotropic KDN scalar write gate.

    Args:
        a: Per-token covariance contraction with shape [B, T, H].
        omega: Additive process noise with shape [B, T, H].
        r: Observation noise with shape [B, T, H].
        initial_precision: Positive information prior with shape [H].
        head_dim: Normalized key width used to convert information scale into
            the posterior covariance multiplier.
        out_dtype: Output dtype. Compute and saved covariance state remain FP32.
        info_scale: Positive finite measurement-information scale. ``None``
            selects ``head_dim`` and exactly recovers the frozen scale-one
            recurrence for unit-normalized production keys.
    """
    if a.ndim != 3:
        raise ValueError(f"a must have shape [B,T,H], got {tuple(a.shape)}")
    if omega.shape != a.shape or r.shape != a.shape:
        raise ValueError(
            f"a, omega, and r must share shape [B,T,H]; got "
            f"a={tuple(a.shape)}, omega={tuple(omega.shape)}, r={tuple(r.shape)}"
        )
    H = a.shape[-1]
    if initial_precision.ndim != 1 or initial_precision.shape[0] != H:
        raise ValueError(f"initial_precision must have shape [{H}], got {tuple(initial_precision.shape)}")
    _require_same_cuda_device(a, ("omega", omega), ("r", r), ("initial_precision", initial_precision))

    if isinstance(head_dim, bool) or not isinstance(head_dim, int):
        raise TypeError(f"head_dim must be a positive int, got {type(head_dim).__name__}")
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    if out_dtype is None:
        out_dtype = torch.float32
    if info_scale is None:
        resolved_info_scale = float(head_dim)
    else:
        if isinstance(info_scale, bool) or not isinstance(info_scale, Real):
            raise TypeError(
                f"info_scale must be a positive finite real, got {type(info_scale).__name__}"
            )
        resolved_info_scale = float(info_scale)
        if not math.isfinite(resolved_info_scale) or resolved_info_scale <= 0:
            raise ValueError(f"info_scale must be positive and finite, got {info_scale!r}")
    posterior_scale = resolved_info_scale / float(head_dim)

    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (a, omega, r, initial_precision)
    )
    return _IsoKDNGainFunction.apply(
        a.contiguous(),
        omega.contiguous(),
        r.contiguous(),
        initial_precision.contiguous(),
        out_dtype,
        posterior_scale,
        needs_backward,
    )

# ---- Frozen kda_precumsum.py -----------------------------------------------
KDA_CHUNK_SIZE = 64


def _chunk_kda_fwd_precum(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run FLA KDA forward after its gate-cumsum boundary."""
    w, u, _qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g_cumsum,
        beta=beta,
        scale=scale,
        cu_seqlens=None,
        chunk_size=KDA_CHUNK_SIZE,
        chunk_indices=None,
        safe_gate=False,
        disable_recompute=False,
    )

    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g_cumsum,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        chunk_indices=None,
        use_exp2=True,
        transpose_state_layout=False,
    )
    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g_cumsum,
        A=Aqk,
        h=h,
        scale=scale,
        cu_seqlens=None,
        chunk_size=KDA_CHUNK_SIZE,
        chunk_indices=None,
        use_exp2=True,
        transpose_state_layout=False,
    )
    return o, Aqk, Akk


def _chunk_kda_bwd_precum(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    do: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run FLA KDA backward and stop before its reverse gate cumsum."""
    w, u, qg, kg = recompute_w_u_fwd(
        q=q,
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        gk=g_cumsum,
        cu_seqlens=None,
        chunk_indices=None,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g_cumsum,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        chunk_indices=None,
        use_exp2=True,
        transpose_state_layout=False,
    )

    dAqk, dv = chunk_kda_bwd_dAv(
        q=q,
        k=k,
        v=v_new,
        do=do,
        A=Aqk,
        scale=scale,
        cu_seqlens=None,
        chunk_size=KDA_CHUNK_SIZE,
        chunk_indices=None,
    )
    dh, _dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=qg,
        k=kg,
        w=w,
        gk=g_cumsum,
        h0=None,
        dht=None,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=None,
        chunk_indices=None,
        use_exp2=True,
        transpose_state_layout=False,
    )
    dq, dk, dv, db, dg_pre_reverse, dAkk = chunk_kda_bwd_wy_dqkg_fused(
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g_cumsum,
        beta=beta,
        A=Akk,
        h=h,
        do=do,
        dh=dh,
        dv=dv,
        scale=scale,
        cu_seqlens=None,
        chunk_size=KDA_CHUNK_SIZE,
        chunk_indices=None,
        transpose_state_layout=False,
    )
    dq, dk, db, dg_pre_reverse = chunk_kda_bwd_intra(
        q=q,
        k=k,
        g=g_cumsum,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        db=db,
        dg=dg_pre_reverse,
        cu_seqlens=None,
        chunk_size=KDA_CHUNK_SIZE,
        chunk_indices=None,
        safe_gate=False,
    )
    # Do not call chunk_local_cumsum(reverse=True) here.  FLA's lower KDA
    # kernels already express this cotangent in the natural-log coordinate;
    # iso_gate's fused backward performs the unscaled reverse chunk sum.
    return dq, dk, dv, db, dg_pre_reverse


class _ChunkKDAPrecumsumFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g_cumsum: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        q, q_rstd = l2norm_fwd(q)
        k, k_rstd = l2norm_fwd(k)
        o, Aqk, Akk = _chunk_kda_fwd_precum(q, k, v, g_cumsum, beta, scale)
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g_cumsum, beta, Aqk, Akk)
        ctx.scale = scale
        return o.type_as(q)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, q_rstd, k, k_rstd, v, g_cumsum, beta, Aqk, Akk = ctx.saved_tensors
        dq, dk, dv, db, dg_pre_reverse = _chunk_kda_bwd_precum(
            q=q,
            k=k,
            v=v,
            g_cumsum=g_cumsum,
            beta=beta,
            Aqk=Aqk,
            Akk=Akk,
            scale=ctx.scale,
            do=do,
        )
        dq = l2norm_bwd(q, q_rstd, dq)
        dk = l2norm_bwd(k, k_rstd, dk)
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg_pre_reverse.to(g_cumsum),
            db.to(beta),
            None,
        )


@torch.compiler.disable
def _chunk_kda_precumsum_paired(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Consume the private paired cumulative gate on the fixed-length path.

    ``g_cumsum`` must be the unbranched output of
    :func:`_iso_kdn_gate_precumsum_paired`.  This Function returns the
    natural-log-coordinate cotangent expected by that exact producer, not the
    generic derivative of an independently constructed base-2 tensor.
    """
    if q.ndim != 4 or q.shape != k.shape:
        raise ValueError(f"q and k must share [B,T,H,K], got {tuple(q.shape)} and {tuple(k.shape)}")
    B, T, H, K = q.shape
    if min(B, T, H, K) == 0:
        raise ValueError("_chunk_kda_precumsum_paired requires non-empty B,T,H,K dimensions")
    if v.ndim != 4 or v.shape[:3] != (B, T, H):
        raise NotImplementedError(
            "_chunk_kda_precumsum_paired does not support GVA; v must have q's B,T,H"
        )
    if v.shape[-1] == 0:
        raise ValueError("_chunk_kda_precumsum_paired requires a non-empty value dimension")
    if g_cumsum.shape != (B, T, H, K):
        raise ValueError(f"g_cumsum must have shape {(B, T, H, K)}, got {tuple(g_cumsum.shape)}")
    if beta.shape != (B, T, H):
        raise ValueError(f"beta must have shape {(B, T, H)}, got {tuple(beta.shape)}")
    if K > 256:
        raise ValueError(f"KDA supports key head dimension <=256, got {K}")
    if g_cumsum.dtype != torch.float32:
        raise TypeError(f"g_cumsum must be FP32 base-2 cumulative gate, got {g_cumsum.dtype}")
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError(f"q, k, and v must share a dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    tensors = (q, k, v, g_cumsum, beta)
    if not all(x.is_cuda for x in tensors):
        raise ValueError("_chunk_kda_precumsum_paired is CUDA-only")
    if len({x.device for x in tensors}) != 1:
        raise ValueError("all _chunk_kda_precumsum_paired tensors must be on the same CUDA device")
    if scale is None:
        scale = K ** -0.5
    return _ChunkKDAPrecumsumFunction.apply(q, k, v, g_cumsum, beta, scale)

__all__ = (
    "_chunk_kda_precumsum_paired",
    "_iso_kdn_gate_precumsum_paired",
    "iso_kdn_gain",
    "pack_f_omega_r_projection",
    "pack_qkv_projection",
    "qkv_causal_conv1d",
)
