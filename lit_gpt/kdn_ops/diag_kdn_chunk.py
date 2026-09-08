"""Fixed production kernels for Diagonal Kalman Delta Networks.

Generated mechanically from the reachable qualified covariance-state S3
implementation at frozen reference commit
95de7b2fc8b2ebec478761b1f32b32b7afa32474.  The
FLA-derived kernels retain the upstream MIT license and Copyright 2023-2026
FLA contributors.  The mechanical generator remains available in repository
history but is not shipped in the minimal public tree.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp, exp2, gather
from fla.ops.utils.softplus import softplus
from fla.utils import (
    IS_GATHER_SUPPORTED,
    IS_NVIDIA_HOPPER,
    USE_CUDA_GRAPH,
    autotune_cache_kwargs,
    check_shared_mem,
)


_DIAG_KDN_PRECISION_ARM = "P1CBWU32_INTRADQFP32_WYDKDM_TF32_BWDOUTBF16"
_DIAG_KDN_MEMORY_CHUNK_SIZE = 32
_DIAG_KDN_GAIN_CHUNK_SIZE = 256
_DIAG_KDN_FRONTEND_LOCAL_CHUNK_SIZE = 64
_DIAG_KDN_READOUT_FORWARD = "FBT"
_DIAG_KDN_READOUT_BACKWARD = "BQ"
_DIAG_KDN_VECTOR_DTYPE = torch.bfloat16
_DIAG_KDN_STATE_DTYPE = torch.float32
_DIAG_KDN_SCORE_DOT_PRECISION = "tf32x3"
_DIAG_KDN_SOLVE_DOT_PRECISION = "tf32x3"
_DIAG_KDN_STATE_DOT_PRECISION = "tf32x3"
_DIAG_KDN_STATE_RESIDUAL_DOT_PRECISION = "tf32"
_DIAG_KDN_BACKWARD_DOT_PRECISION = "tf32x3"
_DIAG_KDN_INTRA_QUERY_DTYPE = torch.float32
_DIAG_KDN_FINAL_GRADIENT_DTYPE = torch.bfloat16

# The frozen file defines this from hardware support.  The release is H200-only.
_frozen_intra_SOLVE_TRIL_DOT_PRECISION = tl.constexpr("tf32")

# ---- frozen source: lit_gpt/kla_ops/diag_norm.py ----------------

_ROW_BLOCK = 4

_frozen_norm_NUM_WARPS = 4

_frozen_norm_NUM_STAGES = 1

@triton.jit
def _frozen_norm_div_rn_fp32(numerator, denominator):
    """IEEE round-to-nearest FP32 division, never reciprocal multiplication."""
    return tl.inline_asm_elementwise('div.rn.f32 $0, $1, $2;', '=f,f,f', [numerator, denominator], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def _frozen_norm_sqrt_rn_fp32(value):
    """Non-FTZ IEEE round-to-nearest FP32 square root."""
    return tl.inline_asm_elementwise('sqrt.rn.f32 $0, $1;', '=f,f', [value], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def _frozen_norm_mul_rn_fp32(left, right):
    """FP32 multiply barrier used by each measured autograd edge."""
    return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', '=f,f,f', [left, right], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def _frozen_norm_add_rn_fp32(left, right):
    """FP32 add barrier preventing edge multiplication/addition contraction."""
    return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;', '=f,f,f', [left, right], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def _frozen_norm_fma_rn_fp32(left, right, accumulator):
    """Non-FTZ FP32 FMA used by PyTorch's NormTwo reduction."""
    return tl.inline_asm_elementwise('fma.rn.f32 $0, $1, $2, $3;', '=f,f,f,f', [left, right, accumulator], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def _frozen_norm_shfl_down_fp32(value, OFFSET: tl.constexpr):
    value_bits = tl.cast(value, tl.int32, bitcast=True)
    offset = tl.full(value.shape, OFFSET, tl.int32)
    shuffled_bits = tl.inline_asm_elementwise('shfl.sync.down.b32 $0, $1, $2, 0x1f, 0xffffffff;', '=r,r,r', [value_bits, offset], dtype=tl.int32, is_pure=True, pack=1)
    return tl.cast(shuffled_bits, tl.float32, bitcast=True)

@triton.jit
def _frozen_norm_broadcast_lane0_fp32(value):
    value_bits = tl.cast(value, tl.int32, bitcast=True)
    source_lane = tl.zeros(value.shape, tl.int32)
    broadcast_bits = tl.inline_asm_elementwise('shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;', '=r,r,r', [value_bits, source_lane], dtype=tl.int32, is_pure=True, pack=1)
    return tl.cast(broadcast_bits, tl.float32, bitcast=True)

@triton.jit
def _frozen_norm_combine_vt0(left0, left1, left2, left3):
    value = _frozen_norm_add_rn_fp32(left0, left1)
    value = _frozen_norm_add_rn_fp32(value, left2)
    return _frozen_norm_add_rn_fp32(value, left3)

@triton.jit
def _frozen_norm_torch_reduce_cuh_accumulators(accumulator0, accumulator1, accumulator2, accumulator3, K: tl.constexpr, SMALL_ROWS: tl.constexpr, REDUCTION_WIDTH: tl.constexpr):
    """PyTorch 2.8 Reduce.cuh topology, returned as a lane-0 broadcast."""
    positive_zero = tl.zeros(accumulator0.shape, tl.float32)
    if REDUCTION_WIDTH == 128:
        tl.static_assert(K == 128 and SMALL_ROWS)
        p0 = _frozen_norm_combine_vt0(accumulator0, positive_zero, positive_zero, positive_zero)
        p64 = _frozen_norm_combine_vt0(accumulator2, positive_zero, positive_zero, positive_zero)
        p32 = _frozen_norm_combine_vt0(accumulator1, positive_zero, positive_zero, positive_zero)
        p96 = _frozen_norm_combine_vt0(accumulator3, positive_zero, positive_zero, positive_zero)
        left = _frozen_norm_add_rn_fp32(p0, p64)
        right = _frozen_norm_add_rn_fp32(p32, p96)
        value = _frozen_norm_add_rn_fp32(left, right)
    elif REDUCTION_WIDTH == 64:
        tl.static_assert(SMALL_ROWS)
        p0 = _frozen_norm_combine_vt0(accumulator0, accumulator2, positive_zero, positive_zero)
        p32 = _frozen_norm_combine_vt0(accumulator1, accumulator3, positive_zero, positive_zero)
        value = _frozen_norm_add_rn_fp32(p0, p32)
    else:
        tl.static_assert(REDUCTION_WIDTH == 32 and (not SMALL_ROWS))
        value = _frozen_norm_combine_vt0(accumulator0, accumulator1, accumulator2, accumulator3)
    value = _frozen_norm_add_rn_fp32(value, _frozen_norm_shfl_down_fp32(value, OFFSET=1))
    value = _frozen_norm_add_rn_fp32(value, _frozen_norm_shfl_down_fp32(value, OFFSET=2))
    value = _frozen_norm_add_rn_fp32(value, _frozen_norm_shfl_down_fp32(value, OFFSET=4))
    value = _frozen_norm_add_rn_fp32(value, _frozen_norm_shfl_down_fp32(value, OFFSET=8))
    value = _frozen_norm_add_rn_fp32(value, _frozen_norm_shfl_down_fp32(value, OFFSET=16))
    return _frozen_norm_broadcast_lane0_fp32(value)

@triton.jit
def _frozen_norm_normalize_row_forward(x0, x1, x2, x3, K: tl.constexpr, SMALL_ROWS: tl.constexpr, REDUCTION_WIDTH: tl.constexpr):
    """Exact PyTorch NormTwo row reduction on a [ROW_BLOCK,32] tile."""
    positive_zero = tl.zeros(x0.shape, tl.float32)
    accumulator0 = _frozen_norm_fma_rn_fp32(x0, x0, positive_zero)
    accumulator1 = _frozen_norm_fma_rn_fp32(x1, x1, positive_zero)
    if K == 128:
        accumulator2 = _frozen_norm_fma_rn_fp32(x2, x2, positive_zero)
        accumulator3 = _frozen_norm_fma_rn_fp32(x3, x3, positive_zero)
    else:
        accumulator2 = positive_zero
        accumulator3 = positive_zero
    square_sum = _frozen_norm_torch_reduce_cuh_accumulators(accumulator0, accumulator1, accumulator2, accumulator3, K=K, SMALL_ROWS=SMALL_ROWS, REDUCTION_WIDTH=REDUCTION_WIDTH)
    return _frozen_norm_sqrt_rn_fp32(square_sum)

@triton.jit
def _frozen_norm_normalize_row_backward(denominator_lane0, denominator_lane1, denominator_lane2, denominator_lane3, K: tl.constexpr, SMALL_ROWS: tl.constexpr, REDUCTION_WIDTH: tl.constexpr):
    """Exact PyTorch sum row reduction for the denominator VJP."""
    positive_zero = tl.zeros(denominator_lane0.shape, tl.float32)
    accumulator0 = _frozen_norm_add_rn_fp32(positive_zero, denominator_lane0)
    accumulator1 = _frozen_norm_add_rn_fp32(positive_zero, denominator_lane1)
    if K == 128:
        accumulator2 = _frozen_norm_add_rn_fp32(positive_zero, denominator_lane2)
        accumulator3 = _frozen_norm_add_rn_fp32(positive_zero, denominator_lane3)
    else:
        accumulator2 = positive_zero
        accumulator3 = positive_zero
    return _frozen_norm_torch_reduce_cuh_accumulators(accumulator0, accumulator1, accumulator2, accumulator3, K=K, SMALL_ROWS=SMALL_ROWS, REDUCTION_WIDTH=REDUCTION_WIDTH)

@triton.jit
def _frozen_norm_store_forward_component(x, denominator, output_ptr, offsets, mask):
    normalized = _frozen_norm_div_rn_fp32(x, denominator)
    memory = tl.cast(normalized, tl.bfloat16, fp_downcast_rounding='rtne')
    tl.store(output_ptr + offsets, memory, mask=mask)

@triton.jit
def _frozen_norm_store_key_forward_component(x, denominator, gain_output_ptr, memory_output_ptr, offsets, mask):
    normalized = _frozen_norm_div_rn_fp32(x, denominator)
    memory = tl.cast(normalized, tl.bfloat16, fp_downcast_rounding='rtne')
    tl.store(gain_output_ptr + offsets, normalized, mask=mask)
    tl.store(memory_output_ptr + offsets, memory, mask=mask)

@triton.jit
def _frozen_norm_store_backward_component(x, cotangent, denominator, norm, denominator_grad, output_ptr, offsets, mask, INPUT_IS_BF16: tl.constexpr, FINAL_ROUND: tl.constexpr):
    direct_fp32 = _frozen_norm_div_rn_fp32(cotangent, denominator)
    safe_norm = tl.where(norm == 0.0, 1.0, norm)
    unit = _frozen_norm_div_rn_fp32(x, safe_norm)
    unit = tl.where(norm == 0.0, 0.0, unit)
    norm_fp32 = tl.where(norm >= 1e-12, _frozen_norm_mul_rn_fp32(denominator_grad, unit), 0.0)
    if INPUT_IS_BF16:
        if FINAL_ROUND:
            combined_fp32 = _frozen_norm_add_rn_fp32(direct_fp32, norm_fp32)
            combined_raw = tl.cast(combined_fp32, tl.bfloat16, fp_downcast_rounding='rtne')
            tl.store(output_ptr + offsets, combined_raw, mask=mask)
        else:
            direct_raw = tl.cast(direct_fp32, tl.bfloat16, fp_downcast_rounding='rtne')
            norm_raw = tl.cast(norm_fp32, tl.bfloat16, fp_downcast_rounding='rtne')
            tl.store(output_ptr + offsets, direct_raw + norm_raw, mask=mask)
    else:
        tl.store(output_ptr + offsets, _frozen_norm_add_rn_fp32(direct_fp32, norm_fp32), mask=mask)

@triton.jit
def _frozen_norm_diag_exact_norm_fwd_kernel(q_ptr, k_ptr, q_memory_ptr, k_gain_ptr, k_memory_ptr, q_norm_ptr, k_norm_ptr, n_rows, K: tl.constexpr, ROW_BLOCK: tl.constexpr, PROCESS_K: tl.constexpr, SMALL_ROWS: tl.constexpr, REDUCTION_WIDTH: tl.constexpr):
    """One launch for q rows; the reviewed helper is reused by the future k arm."""
    row = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    lane = tl.arange(0, 32)
    offsets0 = row[:, None] * K + lane[None, :]
    offsets1 = offsets0 + 32
    offsets2 = offsets0 + 64
    offsets3 = offsets0 + 96
    row_mask = row < n_rows
    mask0 = row_mask[:, None]
    mask1 = row_mask[:, None] & (lane[None, :] + 32 < K)
    mask2 = row_mask[:, None] & (lane[None, :] + 64 < K)
    mask3 = row_mask[:, None] & (lane[None, :] + 96 < K)
    q0 = tl.load(q_ptr + offsets0, mask=mask0, other=0.0).to(tl.float32)
    q1 = tl.load(q_ptr + offsets1, mask=mask1, other=0.0).to(tl.float32)
    q2 = tl.load(q_ptr + offsets2, mask=mask2, other=0.0).to(tl.float32)
    q3 = tl.load(q_ptr + offsets3, mask=mask3, other=0.0).to(tl.float32)
    q_norm = _frozen_norm_normalize_row_forward(q0, q1, q2, q3, K=K, SMALL_ROWS=SMALL_ROWS, REDUCTION_WIDTH=REDUCTION_WIDTH)
    denominator = tl.where(q_norm != q_norm, q_norm, tl.maximum(q_norm, 1e-12))
    _frozen_norm_store_forward_component(q0, denominator, q_memory_ptr, offsets0, mask0)
    _frozen_norm_store_forward_component(q1, denominator, q_memory_ptr, offsets1, mask1)
    if K == 128:
        _frozen_norm_store_forward_component(q2, denominator, q_memory_ptr, offsets2, mask2)
        _frozen_norm_store_forward_component(q3, denominator, q_memory_ptr, offsets3, mask3)
    lane0_mask = row_mask[:, None] & (lane[None, :] == 0)
    tl.store(q_norm_ptr + row[:, None] + lane[None, :], q_norm, mask=lane0_mask)
    if PROCESS_K:
        k0 = tl.load(k_ptr + offsets0, mask=mask0, other=0.0).to(tl.float32)
        k1 = tl.load(k_ptr + offsets1, mask=mask1, other=0.0).to(tl.float32)
        k2 = tl.load(k_ptr + offsets2, mask=mask2, other=0.0).to(tl.float32)
        k3 = tl.load(k_ptr + offsets3, mask=mask3, other=0.0).to(tl.float32)
        k_norm = _frozen_norm_normalize_row_forward(k0, k1, k2, k3, K=K, SMALL_ROWS=SMALL_ROWS, REDUCTION_WIDTH=REDUCTION_WIDTH)
        k_denominator = tl.where(k_norm != k_norm, k_norm, tl.maximum(k_norm, 1e-12))
        _frozen_norm_store_key_forward_component(k0, k_denominator, k_gain_ptr, k_memory_ptr, offsets0, mask0)
        _frozen_norm_store_key_forward_component(k1, k_denominator, k_gain_ptr, k_memory_ptr, offsets1, mask1)
        if K == 128:
            _frozen_norm_store_key_forward_component(k2, k_denominator, k_gain_ptr, k_memory_ptr, offsets2, mask2)
            _frozen_norm_store_key_forward_component(k3, k_denominator, k_gain_ptr, k_memory_ptr, offsets3, mask3)
        tl.store(k_norm_ptr + row[:, None] + lane[None, :], k_norm, mask=lane0_mask)

@triton.jit
def _frozen_norm_diag_exact_norm_bwd_kernel(q_ptr, k_ptr, dq_memory_ptr, dk_gain_ptr, dk_memory_ptr, q_norm_ptr, k_norm_ptr, dq_ptr, dk_ptr, B, T, H, n_rows, dq_stride_b, dq_stride_t, dq_stride_h, dq_stride_k, dkg_stride_b, dkg_stride_t, dkg_stride_h, dkg_stride_k, dkm_stride_b, dkm_stride_t, dkm_stride_h, dkm_stride_k, K: tl.constexpr, ROW_BLOCK: tl.constexpr, PROCESS_Q: tl.constexpr, PROCESS_K: tl.constexpr, HAS_GAIN: tl.constexpr, HAS_MEMORY: tl.constexpr, INPUT_IS_BF16: tl.constexpr, FINAL_ROUND: tl.constexpr, SMALL_ROWS: tl.constexpr, REDUCTION_WIDTH: tl.constexpr):
    """One launch with direct strided cotangent loads for the shared row VJP."""
    row = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    lane = tl.arange(0, 32)
    offsets0 = row[:, None] * K + lane[None, :]
    offsets1 = offsets0 + 32
    offsets2 = offsets0 + 64
    offsets3 = offsets0 + 96
    row_mask = row < n_rows
    mask0 = row_mask[:, None]
    mask1 = row_mask[:, None] & (lane[None, :] + 32 < K)
    mask2 = row_mask[:, None] & (lane[None, :] + 64 < K)
    mask3 = row_mask[:, None] & (lane[None, :] + 96 < K)
    if PROCESS_Q:
        h_index = row % H
        bt_index = row // H
        t_index = bt_index % T
        b_index = bt_index // T
        dq_base = b_index.to(tl.int64)[:, None] * dq_stride_b + t_index.to(tl.int64)[:, None] * dq_stride_t + h_index.to(tl.int64)[:, None] * dq_stride_h
        k0 = lane.to(tl.int64)[None, :]
        dq_offsets0 = dq_base + k0 * dq_stride_k
        dq_offsets1 = dq_base + (k0 + 32) * dq_stride_k
        dq_offsets2 = dq_base + (k0 + 64) * dq_stride_k
        dq_offsets3 = dq_base + (k0 + 96) * dq_stride_k
        q0 = tl.load(q_ptr + offsets0, mask=mask0, other=0.0).to(tl.float32)
        q1 = tl.load(q_ptr + offsets1, mask=mask1, other=0.0).to(tl.float32)
        q2 = tl.load(q_ptr + offsets2, mask=mask2, other=0.0).to(tl.float32)
        q3 = tl.load(q_ptr + offsets3, mask=mask3, other=0.0).to(tl.float32)
        u0 = tl.load(dq_memory_ptr + dq_offsets0, mask=mask0, other=0.0).to(tl.float32)
        u1 = tl.load(dq_memory_ptr + dq_offsets1, mask=mask1, other=0.0).to(tl.float32)
        u2 = tl.load(dq_memory_ptr + dq_offsets2, mask=mask2, other=0.0).to(tl.float32)
        u3 = tl.load(dq_memory_ptr + dq_offsets3, mask=mask3, other=0.0).to(tl.float32)
        lane0_mask = row_mask[:, None] & (lane[None, :] == 0)
        q_norm_seed = tl.load(q_norm_ptr + row[:, None] + lane[None, :], mask=lane0_mask, other=0.0).to(tl.float32)
        q_norm = _frozen_norm_broadcast_lane0_fp32(q_norm_seed)
        denominator = tl.where(q_norm != q_norm, q_norm, tl.maximum(q_norm, 1e-12))
        q0_over_d = _frozen_norm_div_rn_fp32(q0, denominator)
        q1_over_d = _frozen_norm_div_rn_fp32(q1, denominator)
        z0 = _frozen_norm_mul_rn_fp32(-u0, _frozen_norm_div_rn_fp32(q0_over_d, denominator))
        z1 = _frozen_norm_mul_rn_fp32(-u1, _frozen_norm_div_rn_fp32(q1_over_d, denominator))
        if K == 128:
            q2_over_d = _frozen_norm_div_rn_fp32(q2, denominator)
            q3_over_d = _frozen_norm_div_rn_fp32(q3, denominator)
            z2 = _frozen_norm_mul_rn_fp32(-u2, _frozen_norm_div_rn_fp32(q2_over_d, denominator))
            z3 = _frozen_norm_mul_rn_fp32(-u3, _frozen_norm_div_rn_fp32(q3_over_d, denominator))
        else:
            z2 = tl.zeros(z0.shape, tl.float32)
            z3 = tl.zeros(z0.shape, tl.float32)
        denominator_grad = _frozen_norm_normalize_row_backward(z0, z1, z2, z3, K=K, SMALL_ROWS=SMALL_ROWS, REDUCTION_WIDTH=REDUCTION_WIDTH)
        _frozen_norm_store_backward_component(q0, u0, denominator, q_norm, denominator_grad, dq_ptr, offsets0, mask0, INPUT_IS_BF16=INPUT_IS_BF16, FINAL_ROUND=FINAL_ROUND)
        _frozen_norm_store_backward_component(q1, u1, denominator, q_norm, denominator_grad, dq_ptr, offsets1, mask1, INPUT_IS_BF16=INPUT_IS_BF16, FINAL_ROUND=FINAL_ROUND)
        if K == 128:
            _frozen_norm_store_backward_component(q2, u2, denominator, q_norm, denominator_grad, dq_ptr, offsets2, mask2, INPUT_IS_BF16=INPUT_IS_BF16, FINAL_ROUND=FINAL_ROUND)
            _frozen_norm_store_backward_component(q3, u3, denominator, q_norm, denominator_grad, dq_ptr, offsets3, mask3, INPUT_IS_BF16=INPUT_IS_BF16, FINAL_ROUND=FINAL_ROUND)
    if PROCESS_K:
        h_index = row % H
        bt_index = row // H
        t_index = bt_index % T
        b_index = bt_index // T
        dkg_base = b_index.to(tl.int64)[:, None] * dkg_stride_b + t_index.to(tl.int64)[:, None] * dkg_stride_t + h_index.to(tl.int64)[:, None] * dkg_stride_h
        dkm_base = b_index.to(tl.int64)[:, None] * dkm_stride_b + t_index.to(tl.int64)[:, None] * dkm_stride_t + h_index.to(tl.int64)[:, None] * dkm_stride_h
        key_index = lane.to(tl.int64)[None, :]
        dkg_offsets0 = dkg_base + key_index * dkg_stride_k
        dkg_offsets1 = dkg_base + (key_index + 32) * dkg_stride_k
        dkg_offsets2 = dkg_base + (key_index + 64) * dkg_stride_k
        dkg_offsets3 = dkg_base + (key_index + 96) * dkg_stride_k
        dkm_offsets0 = dkm_base + key_index * dkm_stride_k
        dkm_offsets1 = dkm_base + (key_index + 32) * dkm_stride_k
        dkm_offsets2 = dkm_base + (key_index + 64) * dkm_stride_k
        dkm_offsets3 = dkm_base + (key_index + 96) * dkm_stride_k
        k0 = tl.load(k_ptr + offsets0, mask=mask0, other=0.0).to(tl.float32)
        k1 = tl.load(k_ptr + offsets1, mask=mask1, other=0.0).to(tl.float32)
        k2 = tl.load(k_ptr + offsets2, mask=mask2, other=0.0).to(tl.float32)
        k3 = tl.load(k_ptr + offsets3, mask=mask3, other=0.0).to(tl.float32)
        lane0_mask = row_mask[:, None] & (lane[None, :] == 0)
        k_norm_seed = tl.load(k_norm_ptr + row[:, None] + lane[None, :], mask=lane0_mask, other=0.0).to(tl.float32)
        k_norm = _frozen_norm_broadcast_lane0_fp32(k_norm_seed)
        k_denominator = tl.where(k_norm != k_norm, k_norm, tl.maximum(k_norm, 1e-12))
        if HAS_GAIN:
            kg0 = tl.load(dk_gain_ptr + dkg_offsets0, mask=mask0, other=0.0).to(tl.float32)
            kg1 = tl.load(dk_gain_ptr + dkg_offsets1, mask=mask1, other=0.0).to(tl.float32)
            kg2 = tl.load(dk_gain_ptr + dkg_offsets2, mask=mask2, other=0.0).to(tl.float32)
            kg3 = tl.load(dk_gain_ptr + dkg_offsets3, mask=mask3, other=0.0).to(tl.float32)
            ku0 = kg0
            ku1 = kg1
            ku2 = kg2
            ku3 = kg3
            if HAS_MEMORY:
                km0 = tl.load(dk_memory_ptr + dkm_offsets0, mask=mask0, other=0.0).to(tl.float32)
                km1 = tl.load(dk_memory_ptr + dkm_offsets1, mask=mask1, other=0.0).to(tl.float32)
                km2 = tl.load(dk_memory_ptr + dkm_offsets2, mask=mask2, other=0.0).to(tl.float32)
                km3 = tl.load(dk_memory_ptr + dkm_offsets3, mask=mask3, other=0.0).to(tl.float32)
                ku0 = _frozen_norm_add_rn_fp32(kg0, km0)
                ku1 = _frozen_norm_add_rn_fp32(kg1, km1)
                ku2 = _frozen_norm_add_rn_fp32(kg2, km2)
                ku3 = _frozen_norm_add_rn_fp32(kg3, km3)
        else:
            km0 = tl.load(dk_memory_ptr + dkm_offsets0, mask=mask0, other=0.0).to(tl.float32)
            km1 = tl.load(dk_memory_ptr + dkm_offsets1, mask=mask1, other=0.0).to(tl.float32)
            km2 = tl.load(dk_memory_ptr + dkm_offsets2, mask=mask2, other=0.0).to(tl.float32)
            km3 = tl.load(dk_memory_ptr + dkm_offsets3, mask=mask3, other=0.0).to(tl.float32)
            ku0 = km0
            ku1 = km1
            ku2 = km2
            ku3 = km3
        kz0 = _frozen_norm_mul_rn_fp32(-ku0, _frozen_norm_div_rn_fp32(_frozen_norm_div_rn_fp32(k0, k_denominator), k_denominator))
        kz1 = _frozen_norm_mul_rn_fp32(-ku1, _frozen_norm_div_rn_fp32(_frozen_norm_div_rn_fp32(k1, k_denominator), k_denominator))
        if K == 128:
            kz2 = _frozen_norm_mul_rn_fp32(-ku2, _frozen_norm_div_rn_fp32(_frozen_norm_div_rn_fp32(k2, k_denominator), k_denominator))
            kz3 = _frozen_norm_mul_rn_fp32(-ku3, _frozen_norm_div_rn_fp32(_frozen_norm_div_rn_fp32(k3, k_denominator), k_denominator))
        else:
            kz2 = tl.zeros(kz0.shape, tl.float32)
            kz3 = tl.zeros(kz0.shape, tl.float32)
        k_denominator_grad = _frozen_norm_normalize_row_backward(kz0, kz1, kz2, kz3, K=K, SMALL_ROWS=SMALL_ROWS, REDUCTION_WIDTH=REDUCTION_WIDTH)
        _frozen_norm_store_backward_component(k0, ku0, k_denominator, k_norm, k_denominator_grad, dk_ptr, offsets0, mask0, INPUT_IS_BF16=INPUT_IS_BF16, FINAL_ROUND=FINAL_ROUND)
        _frozen_norm_store_backward_component(k1, ku1, k_denominator, k_norm, k_denominator_grad, dk_ptr, offsets1, mask1, INPUT_IS_BF16=INPUT_IS_BF16, FINAL_ROUND=FINAL_ROUND)
        if K == 128:
            _frozen_norm_store_backward_component(k2, ku2, k_denominator, k_norm, k_denominator_grad, dk_ptr, offsets2, mask2, INPUT_IS_BF16=INPUT_IS_BF16, FINAL_ROUND=FINAL_ROUND)
            _frozen_norm_store_backward_component(k3, ku3, k_denominator, k_norm, k_denominator_grad, dk_ptr, offsets3, mask3, INPUT_IS_BF16=INPUT_IS_BF16, FINAL_ROUND=FINAL_ROUND)

def _frozen_norm_reduction_width(n_rows: int, k: int) -> int:
    return 32 if n_rows >= 16 else 64 if n_rows >= 8 else k

def _frozen_norm_launch_diag_norm_boundary_forward(q_raw: torch.Tensor, k_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_memory = torch.empty_like(q_raw, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
    k_gain = torch.empty_like(k_raw, dtype=torch.float32, memory_format=torch.contiguous_format)
    k_memory = torch.empty_like(k_raw, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
    q_norm = torch.empty(q_raw.shape[:-1], device=q_raw.device, dtype=torch.float32)
    k_norm = torch.empty(k_raw.shape[:-1], device=k_raw.device, dtype=torch.float32)
    n_rows = q_raw.numel() // q_raw.shape[-1]
    if n_rows != 0:
        grid = (triton.cdiv(n_rows, _ROW_BLOCK),)
        _frozen_norm_diag_exact_norm_fwd_kernel[grid](q_raw, k_raw, q_memory, k_gain, k_memory, q_norm, k_norm, n_rows, K=q_raw.shape[-1], ROW_BLOCK=_ROW_BLOCK, PROCESS_K=True, SMALL_ROWS=n_rows < 16, REDUCTION_WIDTH=_frozen_norm_reduction_width(n_rows, q_raw.shape[-1]), num_warps=_frozen_norm_NUM_WARPS, num_stages=_frozen_norm_NUM_STAGES)
    return (q_memory, k_gain, k_memory, q_norm, k_norm)


# ---- frozen source: lit_gpt/kla_ops/diag_chunk.py ----------------

@triton.jit(do_not_specialize=['T', 'info_scale', 'info_scale_base'])
def _frozen_gain_kla_kappa_passA_kernel(k_ptr, a_ptr, omega_ptr, r_ptr, cm_ptr, T, info_scale, log_info_scale_ptr, info_scale_base, NT: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr, USE_LOG_SCALE: tl.constexpr):
    i_nt = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H
    i_h = pid % H
    if USE_LOG_SCALE:
        log_info_scale = tl.load(log_info_scale_ptr + i_h).to(tl.float32)
        head_info_scale = info_scale_base * tl.exp(log_info_scale)
        tl.device_assert((head_info_scale > 0.0) & (head_info_scale <= 3.402823466e+38), 'effective info scale must be positive and finite')
    i_k = tl.arange(0, BK)
    mask = i_k < K
    base = i_b * T * H * K + i_h * K
    base_r = i_b * T * H + i_h
    t0 = i_nt * BT
    t1 = tl.minimum(t0 + BT, T)
    mA = tl.full((BK,), 1.0, tl.float32)
    mB = tl.full((BK,), 0.0, tl.float32)
    mC = tl.full((BK,), 0.0, tl.float32)
    mD = tl.full((BK,), 1.0, tl.float32)
    for t in range(t0, t1):
        off = base + t * H * K + i_k
        k = tl.load(k_ptr + off, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
        omega = tl.load(omega_ptr + off, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + base_r + t * H).to(tl.float32)
        a2 = a * a
        if USE_LOG_SCALE:
            u = head_info_scale * k * k / r
        else:
            u = info_scale * k * k / r
        tA = a2
        tB = omega
        tC = u * a2
        tD = 1.0 + u * omega
        nA = tB * mC + tA * mA
        nB = tB * mD + tA * mB
        nC = tD * mC + tC * mA
        nD = tD * mD + tC * mB
        m = tl.maximum(tl.maximum(tl.abs(nD), tl.abs(nC)), tl.maximum(tl.abs(nB), tl.abs(nA)))
        inv = 1.0 / tl.maximum(m, 1e-30)
        mA = nA * inv
        mB = nB * inv
        mC = nC * inv
        mD = nD * inv
    cm_base = ((i_b * NT + i_nt) * H + i_h) * K * 4
    tl.store(cm_ptr + cm_base + i_k * 4 + 0, mA, mask=mask)
    tl.store(cm_ptr + cm_base + i_k * 4 + 1, mB, mask=mask)
    tl.store(cm_ptr + cm_base + i_k * 4 + 2, mC, mask=mask)
    tl.store(cm_ptr + cm_base + i_k * 4 + 3, mD, mask=mask)

@triton.jit
def _frozen_gain_kla_kappa_passB_kernel(cm_ptr, mu_ptr, carry_ptr, NT: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    i_b = pid // H
    i_h = pid % H
    i_k = tl.arange(0, BK)
    mask = i_k < K
    mu = tl.load(mu_ptr + i_h).to(tl.float32)
    p_num = tl.full((BK,), 1.0, tl.float32)
    p_den = tl.full((BK,), 0.0, tl.float32) + mu
    for nt in range(0, NT):
        carry_base = ((i_b * NT + nt) * H + i_h) * K * 2
        tl.store(carry_ptr + carry_base + i_k * 2 + 0, p_num, mask=mask)
        tl.store(carry_ptr + carry_base + i_k * 2 + 1, p_den, mask=mask)
        cm_base = ((i_b * NT + nt) * H + i_h) * K * 4
        mA = tl.load(cm_ptr + cm_base + i_k * 4 + 0, mask=mask, other=0.0)
        mB = tl.load(cm_ptr + cm_base + i_k * 4 + 1, mask=mask, other=0.0)
        mC = tl.load(cm_ptr + cm_base + i_k * 4 + 2, mask=mask, other=0.0)
        mD = tl.load(cm_ptr + cm_base + i_k * 4 + 3, mask=mask, other=0.0)
        p_num_new = mB * p_den + mA * p_num
        p_den_new = mD * p_den + mC * p_num
        m = tl.maximum(tl.abs(p_den_new), tl.abs(p_num_new))
        inv = 1.0 / tl.maximum(m, 1e-30)
        p_den = p_den_new * inv
        p_num = p_num_new * inv

@triton.jit(do_not_specialize=['T', 'info_scale', 'info_scale_base'])
def _frozen_gain_kla_kappa_passC_kappa_only_kernel(k_ptr, a_ptr, omega_ptr, r_ptr, carry_ptr, kappa_ptr, n_ptr, d_ptr, T, info_scale, log_info_scale_ptr, info_scale_base, NT: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr, SAVE_STATE: tl.constexpr, USE_LOG_SCALE: tl.constexpr):
    """Pass C specialization which neither accepts nor writes beta_ch."""
    i_nt = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H
    i_h = pid % H
    if USE_LOG_SCALE:
        log_info_scale = tl.load(log_info_scale_ptr + i_h).to(tl.float32)
        head_info_scale = info_scale_base * tl.exp(log_info_scale)
        tl.device_assert((head_info_scale > 0.0) & (head_info_scale <= 3.402823466e+38), 'effective info scale must be positive and finite')
    i_k = tl.arange(0, BK)
    mask = i_k < K
    base = i_b * T * H * K + i_h * K
    base_r = i_b * T * H + i_h
    t0 = i_nt * BT
    t1 = tl.minimum(t0 + BT, T)
    carry_base = ((i_b * NT + i_nt) * H + i_h) * K * 2
    p_num = tl.load(carry_ptr + carry_base + i_k * 2, mask=mask, other=1.0).to(tl.float32)
    p_den = tl.load(carry_ptr + carry_base + i_k * 2 + 1, mask=mask, other=0.0).to(tl.float32)
    for t in range(t0, t1):
        off = base + t * H * K + i_k
        k = tl.load(k_ptr + off, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
        omega = tl.load(omega_ptr + off, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + base_r + t * H).to(tl.float32)
        if SAVE_STATE:
            tl.store(n_ptr + off, p_den, mask=mask)
            tl.store(d_ptr + off, p_num, mask=mask)
        a2 = a * a
        p_hat = a2 * p_num / p_den + omega
        ksq = k * k
        denom = r + tl.sum(tl.where(mask, p_hat * ksq, 0.0), axis=0)
        kappa = p_hat * k / denom
        tl.store(kappa_ptr + off, kappa.to(kappa_ptr.dtype.element_ty), mask=mask)
        if USE_LOG_SCALE:
            u = head_info_scale * ksq / r
        else:
            u = info_scale * ksq / r
        p_num_new = omega * p_den + a2 * p_num
        p_den_new = (1.0 + u * omega) * p_den + u * a2 * p_num
        m = tl.maximum(tl.abs(p_den_new), tl.abs(p_num_new))
        inv = 1.0 / tl.maximum(m, 1e-30)
        p_den = p_den_new * inv
        p_num = p_num_new * inv

@triton.jit(do_not_specialize=['T', 'info_scale', 'info_scale_base'])
def _frozen_gain_kla_kappa_bwd_fill_kernel(k_ptr, a_ptr, omega_ptr, r_ptr, carry_ptr, p_excl_ptr, T, info_scale, log_info_scale_ptr, info_scale_base, NT: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr, USE_LOG_SCALE: tl.constexpr):
    i_nt = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H
    i_h = pid % H
    if USE_LOG_SCALE:
        log_info_scale = tl.load(log_info_scale_ptr + i_h).to(tl.float32)
        head_info_scale = info_scale_base * tl.exp(log_info_scale)
        tl.device_assert((head_info_scale > 0.0) & (head_info_scale <= 3.402823466e+38), 'effective info scale must be positive and finite')
    i_k = tl.arange(0, BK)
    mask = i_k < K
    base = i_b * T * H * K + i_h * K
    base_r = i_b * T * H + i_h
    t0 = i_nt * BT
    t1 = tl.minimum(t0 + BT, T)
    carry_base = ((i_b * NT + i_nt) * H + i_h) * K * 2
    if USE_LOG_SCALE:
        if log_info_scale != 0.0:
            p_num64 = tl.load(carry_ptr + carry_base + i_k * 2 + 0, mask=mask, other=1.0).to(tl.float64)
            p_den64 = tl.load(carry_ptr + carry_base + i_k * 2 + 1, mask=mask, other=0.0).to(tl.float64)
            Kf64 = head_info_scale.to(tl.float64)
            for t in range(t0, t1):
                off = base + t * H * K + i_k
                tl.store(p_excl_ptr + off, p_num64 / p_den64, mask=mask)
                k64 = tl.load(k_ptr + off, mask=mask, other=0.0).to(tl.float64)
                a64 = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float64)
                omega64 = tl.load(omega_ptr + off, mask=mask, other=0.0).to(tl.float64)
                r64 = tl.load(r_ptr + base_r + t * H).to(tl.float64)
                a264 = a64 * a64
                u64 = Kf64 * k64 * k64 / r64
                p_num_new64 = omega64 * p_den64 + a264 * p_num64
                p_den_new64 = (1.0 + u64 * omega64) * p_den64 + u64 * a264 * p_num64
                magnitude64 = tl.maximum(tl.abs(p_den_new64), tl.abs(p_num_new64))
                inv64 = 1.0 / tl.maximum(magnitude64, 1e-30)
                p_den64 = tl.where(mask, p_den_new64 * inv64, p_den64)
                p_num64 = tl.where(mask, p_num_new64 * inv64, p_num64)
        else:
            p_num = tl.load(carry_ptr + carry_base + i_k * 2 + 0, mask=mask, other=1.0).to(tl.float32)
            p_den = tl.load(carry_ptr + carry_base + i_k * 2 + 1, mask=mask, other=0.0).to(tl.float32)
            for t in range(t0, t1):
                off = base + t * H * K + i_k
                tl.store(p_excl_ptr + off, p_num / p_den, mask=mask)
                k = tl.load(k_ptr + off, mask=mask, other=0.0).to(tl.float32)
                a = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
                omega = tl.load(omega_ptr + off, mask=mask, other=0.0).to(tl.float32)
                r = tl.load(r_ptr + base_r + t * H).to(tl.float32)
                a2 = a * a
                u = head_info_scale * k * k / r
                p_num_new = omega * p_den + a2 * p_num
                p_den_new = (1.0 + u * omega) * p_den + u * a2 * p_num
                magnitude = tl.maximum(tl.abs(p_den_new), tl.abs(p_num_new))
                inv = 1.0 / tl.maximum(magnitude, 1e-30)
                p_den = p_den_new * inv
                p_num = p_num_new * inv
    else:
        p_num = tl.load(carry_ptr + carry_base + i_k * 2 + 0, mask=mask, other=1.0).to(tl.float32)
        p_den = tl.load(carry_ptr + carry_base + i_k * 2 + 1, mask=mask, other=0.0).to(tl.float32)
        for t in range(t0, t1):
            off = base + t * H * K + i_k
            tl.store(p_excl_ptr + off, p_num / p_den, mask=mask)
            k = tl.load(k_ptr + off, mask=mask, other=0.0).to(tl.float32)
            a = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
            omega = tl.load(omega_ptr + off, mask=mask, other=0.0).to(tl.float32)
            r = tl.load(r_ptr + base_r + t * H).to(tl.float32)
            a2 = a * a
            u = info_scale * k * k / r
            p_num_new = omega * p_den + a2 * p_num
            p_den_new = (1.0 + u * omega) * p_den + u * a2 * p_num
            magnitude = tl.maximum(tl.abs(p_den_new), tl.abs(p_num_new))
            inv = 1.0 / tl.maximum(magnitude, 1e-30)
            p_den = p_den_new * inv
            p_num = p_num_new * inv

@triton.jit(do_not_specialize=['T', 'info_scale', 'info_scale_base'])
def _frozen_gain_kla_kappa_bwd_passA_kappa_only_kernel(k_ptr, a_ptr, omega_ptr, r_ptr, dkappa_ptr, p_excl_ptr, ab_ptr, T, info_scale, log_info_scale_ptr, info_scale_base, NT: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr, USE_LOG_SCALE: tl.constexpr):
    """Parallel reverse-map construction without a dbeta pointer."""
    i_nt = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H
    i_h = pid % H
    if USE_LOG_SCALE:
        log_info_scale = tl.load(log_info_scale_ptr + i_h).to(tl.float32)
        Kf = info_scale_base * tl.exp(log_info_scale)
        tl.device_assert((Kf > 0.0) & (Kf <= 3.402823466e+38), 'effective info scale must be positive and finite')
    else:
        Kf = info_scale
    i_k = tl.arange(0, BK)
    mask = i_k < K
    base = i_b * T * H * K + i_h * K
    base_r = i_b * T * H + i_h
    t0 = i_nt * BT
    t1 = tl.minimum(t0 + BT, T)
    A_net = tl.full((BK,), 1.0, tl.float32)
    B_net = tl.full((BK,), 0.0, tl.float32)
    for t in range(t1 - 1, t0 - 1, -1):
        off = base + t * H * K + i_k
        pprev = tl.load(p_excl_ptr + off, mask=mask, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + off, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
        omega = tl.load(omega_ptr + off, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + base_r + t * H).to(tl.float32)
        dki = tl.load(dkappa_ptr + off, mask=mask, other=0.0).to(tl.float32)
        a2 = a * a
        ksq = k * k
        p_hat = a2 * pprev + omega
        u = Kf * ksq / r
        h = 1.0 + u * p_hat
        D = r + tl.sum(tl.where(mask, p_hat * ksq, 0.0), axis=0)
        kap = p_hat * k / D
        G = tl.sum(tl.where(mask, dki * kap, 0.0), axis=0)
        Pbar_gain = dki * k / D - ksq * G / D
        A_t = a2 / (h * h)
        B_t = a2 * Pbar_gain
        A_net_new = A_t * A_net
        B_net = A_t * B_net + B_t
        A_net = A_net_new
    ab_base = ((i_b * NT + i_nt) * H + i_h) * K * 2
    tl.store(ab_ptr + ab_base + i_k * 2, A_net, mask=mask)
    tl.store(ab_ptr + ab_base + i_k * 2 + 1, B_net, mask=mask)

@triton.jit
def _frozen_gain_kla_kappa_bwd_passB_kernel(ab_ptr, carry_ptr, NT: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    i_b = pid // H
    i_h = pid % H
    i_k = tl.arange(0, BK)
    mask = i_k < K
    lam = tl.zeros((BK,), tl.float32)
    for nt in range(NT - 1, -1, -1):
        carry_base = ((i_b * NT + nt) * H + i_h) * K
        tl.store(carry_ptr + carry_base + i_k, lam, mask=mask)
        ab_base = ((i_b * NT + nt) * H + i_h) * K * 2
        A_chunk = tl.load(ab_ptr + ab_base + i_k * 2 + 0, mask=mask, other=0.0)
        B_chunk = tl.load(ab_ptr + ab_base + i_k * 2 + 1, mask=mask, other=0.0)
        lam = A_chunk * lam + B_chunk

@triton.jit(do_not_specialize=['T', 'info_scale', 'info_scale_base'])
def _frozen_gain_kla_kappa_bwd_passC_kappa_only_kernel(k_ptr, a_ptr, omega_ptr, r_ptr, dkappa_ptr, p_excl_ptr, carry_ptr, omega_raw_ptr, r_raw_ptr, dk_ptr, dalpha_ptr, domega_raw_ptr, dr_raw_ptr, dmu_ptr, dlog_ptr, T, info_scale, log_info_scale_ptr, info_scale_base, NT: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr, RAW_ACTIVATION_VJP: tl.constexpr, USE_LOG_SCALE: tl.constexpr, NEED_DMU: tl.constexpr, NEED_DLOG: tl.constexpr):
    """Parallel gradient emission without a dbeta pointer."""
    i_nt = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H
    i_h = pid % H
    if USE_LOG_SCALE:
        log_info_scale = tl.load(log_info_scale_ptr + i_h).to(tl.float32)
        Kf = info_scale_base * tl.exp(log_info_scale)
        tl.device_assert((Kf > 0.0) & (Kf <= 3.402823466e+38), 'effective info scale must be positive and finite')
    else:
        Kf = info_scale
    i_k = tl.arange(0, BK)
    mask = i_k < K
    base = i_b * T * H * K + i_h * K
    base_r = i_b * T * H + i_h
    t0 = i_nt * BT
    t1 = tl.minimum(t0 + BT, T)
    carry_base = ((i_b * NT + i_nt) * H + i_h) * K
    lam = tl.load(carry_ptr + carry_base + i_k, mask=mask, other=0.0).to(tl.float32)
    dlog = 0.0
    for t in range(t1 - 1, t0 - 1, -1):
        off = base + t * H * K + i_k
        pprev = tl.load(p_excl_ptr + off, mask=mask, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + off, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
        omega = tl.load(omega_ptr + off, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + base_r + t * H).to(tl.float32)
        dki = tl.load(dkappa_ptr + off, mask=mask, other=0.0).to(tl.float32)
        a2 = a * a
        ksq = k * k
        p_hat = a2 * pprev + omega
        u = Kf * ksq / r
        h = 1.0 + u * p_hat
        pnext = p_hat / h
        D = r + tl.sum(tl.where(mask, p_hat * ksq, 0.0), axis=0)
        kap = p_hat * k / D
        beta = p_hat / D
        G = tl.sum(tl.where(mask, dki * kap, 0.0), axis=0)
        Pbar_gain = dki * k / D - ksq * G / D
        kbar_gain = dki * beta - 2.0 * p_hat * k * G / D
        Pbar_tot = Pbar_gain + lam / (h * h)
        ubar = -lam * pnext * pnext
        dk = kbar_gain + ubar * (2.0 * Kf * k / r)
        dalpha = 2.0 * a * pprev * Pbar_tot
        dr = -G / D + tl.sum(tl.where(mask, ubar * (-Kf * ksq / (r * r)), 0.0), axis=0)
        tl.store(dk_ptr + off, dk.to(dk_ptr.dtype.element_ty), mask=mask)
        tl.store(dalpha_ptr + off, dalpha.to(dalpha_ptr.dtype.element_ty), mask=mask)
        if RAW_ACTIVATION_VJP:
            omega_raw = tl.load(omega_raw_ptr + off, mask=mask, other=0.0).to(tl.float32)
            r_raw = tl.load(r_raw_ptr + base_r + t * H).to(tl.float32)
            domega_raw = Pbar_tot * tl.sigmoid(omega_raw)
            dr_raw = dr * tl.sigmoid(r_raw)
            tl.store(domega_raw_ptr + off, domega_raw.to(domega_raw_ptr.dtype.element_ty), mask=mask)
            tl.store(dr_raw_ptr + base_r + t * H, dr_raw.to(dr_raw_ptr.dtype.element_ty))
        else:
            tl.store(domega_raw_ptr + off, Pbar_tot.to(domega_raw_ptr.dtype.element_ty), mask=mask)
            tl.store(dr_raw_ptr + base_r + t * H, dr.to(dr_raw_ptr.dtype.element_ty))
        if NEED_DLOG:
            dlog += tl.sum(tl.where(mask, ubar * u, 0.0), axis=0)
        lam = tl.where(mask, a2 * Pbar_tot, 0.0)
    if NEED_DMU:
        if i_nt == 0:
            p_initial = tl.load(p_excl_ptr + base + i_k, mask=mask, other=0.0).to(tl.float32)
            tl.atomic_add(dmu_ptr + i_h, tl.sum(tl.where(mask, -lam * p_initial * p_initial, 0.0), axis=0))
    if NEED_DLOG:
        tl.atomic_add(dlog_ptr + i_h, dlog)


# ---- frozen source: lit_gpt/kla_ops/kalman_s3/mixed_precision_dot.py ----------------

@triton.jit
def _frozen_dot_whitelisted_dot(lhs, rhs, BF16_DOT_OPERANDS: tl.constexpr, DOT_PRECISION: tl.constexpr):
    if BF16_DOT_OPERANDS:
        lhs = lhs.to(tl.bfloat16)
        rhs = rhs.to(tl.bfloat16)
    return tl.dot(lhs, rhs, input_precision=DOT_PRECISION, out_dtype=tl.float32)


# ---- frozen source: lit_gpt/kla_ops/kalman_s3/precision_policy.py ----------------

# The selected precision-policy record is materialized as the fixed constants above; no runtime policy registry is retained.


# ---- frozen source: lit_gpt/kla_ops/kalman_s3/chunk_kda.py ----------------

_frozen_kda_NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]

@triton.heuristics({'USE_G': lambda args: args['g'] is not None, 'USE_GK': lambda args: args['gk'] is not None, 'USE_INITIAL_STATE': lambda args: args['h0'] is not None, 'STORE_FINAL_STATE': lambda args: args['ht'] is not None, 'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None, 'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages) for num_warps in [2, 4] for num_stages in ([2, 3, 4] if check_shared_mem('ampere') else [2, 1]) for BV in ([32, 64] if check_shared_mem('ada') else [32])], key=['H', 'K', 'V', 'BT', 'USE_EXP2', 'TRANSPOSE_STATE', 'DOT_PRECISION', 'BF16_DOT_OPERANDS', 'RESIDUAL_DOT_PRECISION', 'RESIDUAL_BF16_DOT_OPERANDS'], use_cuda_graph=USE_CUDA_GRAPH, **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def _frozen_kda_chunk_gated_delta_rule_fwd_kernel_h_blockdim64_diag(k, v, w, v_new, g, gk, h, h0, ht, cu_seqlens, chunk_offsets, T, H: tl.constexpr, Hq: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, USE_G: tl.constexpr, USE_GK: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr, SAVE_NEW_VALUE: tl.constexpr, USE_EXP2: tl.constexpr, TRANSPOSE_STATE: tl.constexpr, IS_VARLEN: tl.constexpr, DOT_PRECISION: tl.constexpr, BF16_DOT_OPERANDS: tl.constexpr, RESIDUAL_DOT_PRECISION: tl.constexpr, RESIDUAL_BF16_DOT_OPERANDS: tl.constexpr):
    i_v, i_nh = (tl.program_id(0), tl.program_id(1))
    i_n, i_h = (i_nh // H, i_nh % H)
    if IS_VARLEN:
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = (i_n * T, i_n * T + T)
        NT = tl.cdiv(T, BT)
        boh = i_n * NT
    if TRANSPOSE_STATE:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)
    h += (boh * H + i_h).to(tl.int64) * K * V
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * Hq + i_h // (H // Hq)).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h).to(tl.int64) * V
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            if TRANSPOSE_STATE:
                p_h0_2 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            if TRANSPOSE_STATE:
                p_h0_3 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            if TRANSPOSE_STATE:
                p_h0_4 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        if TRANSPOSE_STATE:
            p_h1 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_h1 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_h2 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_h2 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_h3 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_h3 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_h4 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_h4 = tl.make_block_ptr(h + i_t_int64 * H * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))
        p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1)).to(tl.float32)
        if TRANSPOSE_STATE:
            b_v = _frozen_dot_whitelisted_dot(b_w, tl.trans(b_h1).to(b_w.dtype), RESIDUAL_BF16_DOT_OPERANDS, RESIDUAL_DOT_PRECISION)
        else:
            b_v = _frozen_dot_whitelisted_dot(b_w, b_h1.to(b_w.dtype), RESIDUAL_BF16_DOT_OPERANDS, RESIDUAL_DOT_PRECISION)
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1)).to(tl.float32)
            if TRANSPOSE_STATE:
                b_v += _frozen_dot_whitelisted_dot(b_w, tl.trans(b_h2).to(b_w.dtype), RESIDUAL_BF16_DOT_OPERANDS, RESIDUAL_DOT_PRECISION)
            else:
                b_v += _frozen_dot_whitelisted_dot(b_w, b_h2.to(b_w.dtype), RESIDUAL_BF16_DOT_OPERANDS, RESIDUAL_DOT_PRECISION)
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1)).to(tl.float32)
            if TRANSPOSE_STATE:
                b_v += _frozen_dot_whitelisted_dot(b_w, tl.trans(b_h3).to(b_w.dtype), RESIDUAL_BF16_DOT_OPERANDS, RESIDUAL_DOT_PRECISION)
            else:
                b_v += _frozen_dot_whitelisted_dot(b_w, b_h3.to(b_w.dtype), RESIDUAL_BF16_DOT_OPERANDS, RESIDUAL_DOT_PRECISION)
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1)).to(tl.float32)
            if TRANSPOSE_STATE:
                b_v += _frozen_dot_whitelisted_dot(b_w, tl.trans(b_h4).to(b_w.dtype), RESIDUAL_BF16_DOT_OPERANDS, RESIDUAL_DOT_PRECISION)
            else:
                b_v += _frozen_dot_whitelisted_dot(b_w, b_h4.to(b_w.dtype), RESIDUAL_BF16_DOT_OPERANDS, RESIDUAL_DOT_PRECISION)
        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32) - b_v
        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(v_new, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = i_t * BT + tl.arange(0, BT) < T
            b_g_last = tl.load(g + (bos * H + last_idx * H + i_h).to(tl.int64)).to(tl.float32)
            p_g = tl.make_block_ptr(g + (bos * H + i_h).to(tl.int64), (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last
        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k1, mask=o_k1 < K, other=0.0).to(tl.float32)
            if TRANSPOSE_STATE:
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[None, :]
                else:
                    b_h1 *= exp(b_gk_last1)[None, :]
            elif USE_EXP2:
                b_h1 *= exp2(b_gk_last1)[:, None]
            else:
                b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k2, mask=o_k2 < K, other=0.0).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h2 *= exp2(b_gk_last2)[None, :]
                    else:
                        b_h2 *= exp(b_gk_last2)[None, :]
                elif USE_EXP2:
                    b_h2 *= exp2(b_gk_last2)[:, None]
                else:
                    b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k3, mask=o_k3 < K, other=0.0).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h3 *= exp2(b_gk_last3)[None, :]
                    else:
                        b_h3 *= exp(b_gk_last3)[None, :]
                elif USE_EXP2:
                    b_h3 *= exp2(b_gk_last3)[:, None]
                else:
                    b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k4, mask=o_k4 < K, other=0.0).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h4 *= exp2(b_gk_last4)[None, :]
                    else:
                        b_h4 *= exp(b_gk_last4)[None, :]
                elif USE_EXP2:
                    b_h4 *= exp2(b_gk_last4)[:, None]
                else:
                    b_h4 *= exp(b_gk_last4)[:, None]
        b_v = b_v.to(tl.float32)
        p_k = tl.make_block_ptr(k, (K, T), (1, Hq * K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        if TRANSPOSE_STATE:
            b_h1 += tl.trans(tl.dot(b_k, b_v, input_precision=DOT_PRECISION, out_dtype=tl.float32))
        else:
            b_h1 += tl.dot(b_k, b_v, input_precision=DOT_PRECISION, out_dtype=tl.float32)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, Hq * K), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            if TRANSPOSE_STATE:
                b_h2 += tl.trans(tl.dot(b_k, b_v, input_precision=DOT_PRECISION, out_dtype=tl.float32))
            else:
                b_h2 += tl.dot(b_k, b_v, input_precision=DOT_PRECISION, out_dtype=tl.float32)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, Hq * K), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            if TRANSPOSE_STATE:
                b_h3 += tl.trans(tl.dot(b_k, b_v, input_precision=DOT_PRECISION, out_dtype=tl.float32))
            else:
                b_h3 += tl.dot(b_k, b_v, input_precision=DOT_PRECISION, out_dtype=tl.float32)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, Hq * K), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            if TRANSPOSE_STATE:
                b_h4 += tl.trans(tl.dot(b_k, b_v, input_precision=DOT_PRECISION, out_dtype=tl.float32))
            else:
                b_h4 += tl.dot(b_k, b_v, input_precision=DOT_PRECISION, out_dtype=tl.float32)
    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

@triton.heuristics({'USE_G': lambda args: args['g'] is not None, 'USE_GK': lambda args: args['gk'] is not None, 'USE_INITIAL_STATE': lambda args: args['dh0'] is not None, 'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None, 'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages) for num_warps in [2, 4] for num_stages in ([2, 3, 4] if check_shared_mem('ampere') else [1]) for BV in ([32, 64] if check_shared_mem('ada') else [32])], key=['H', 'K', 'V', 'BT', 'BV', 'USE_G', 'USE_EXP2', 'TRANSPOSE_STATE', 'DOT_PRECISION', 'BF16_QG_DO_OPERANDS', 'BF16_W_DV_OPERANDS'], use_cuda_graph=USE_CUDA_GRAPH, **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def _frozen_kda_chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64_diag(q, k, w, g, gk, dht, dh0, do, dh, dv, dv2, cu_seqlens, chunk_offsets, scale, T, H: tl.constexpr, Hq: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, USE_G: tl.constexpr, USE_GK: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, USE_FINAL_STATE_GRADIENT: tl.constexpr, USE_EXP2: tl.constexpr, TRANSPOSE_STATE: tl.constexpr, IS_VARLEN: tl.constexpr, DOT_PRECISION: tl.constexpr, BF16_QG_DO_OPERANDS: tl.constexpr, BF16_W_DV_OPERANDS: tl.constexpr):
    i_v, i_nh = (tl.program_id(0), tl.program_id(1))
    i_n, i_h = (i_nh // H, i_nh % H)
    if IS_VARLEN:
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = (i_n * T, i_n * T + T)
        NT = tl.cdiv(T, BT)
        boh = i_n * NT
    if TRANSPOSE_STATE:
        b_dh1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([64, BV], dtype=tl.float32)
    q += (bos * Hq + i_h // (H // Hq)).to(tl.int64) * K
    k += (bos * Hq + i_h // (H // Hq)).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    do += (bos * H + i_h).to(tl.int64) * V
    dv += (bos * H + i_h).to(tl.int64) * V
    dv2 += (bos * H + i_h).to(tl.int64) * V
    dh += (boh * H + i_h).to(tl.int64) * K * V
    if USE_GK:
        gk += (bos * H + i_h).to(tl.int64) * K
    if USE_INITIAL_STATE:
        dh0 += i_nh * K * V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K * V
    if USE_FINAL_STATE_GRADIENT:
        if TRANSPOSE_STATE:
            p_dht1 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_dht2 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_dht3 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_dht4 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))
    for i_t in range(NT - 1, -1, -1):
        i_t_int64 = i_t.to(tl.int64)
        if TRANSPOSE_STATE:
            p_dh1 = tl.make_block_ptr(dh + i_t_int64 * H * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_dh1 = tl.make_block_ptr(dh + i_t_int64 * H * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_dh2 = tl.make_block_ptr(dh + i_t_int64 * H * K * V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_dh2 = tl.make_block_ptr(dh + i_t_int64 * H * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_dh3 = tl.make_block_ptr(dh + i_t_int64 * H * K * V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_dh3 = tl.make_block_ptr(dh + i_t_int64 * H * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_dh4 = tl.make_block_ptr(dh + i_t_int64 * H * K * V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_dh4 = tl.make_block_ptr(dh + i_t_int64 * H * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            bg_last = tl.load(g + (bos + last_idx) * H + i_h).to(tl.float32)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                bg_last_exp = exp2(bg_last)
                b_g_exp = exp2(b_g)
            else:
                bg_last_exp = exp(bg_last)
                b_g_exp = exp(b_g)
        p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
        p_k = tl.make_block_ptr(k, (T, K), (Hq * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + last_idx * H * K + o_k1, mask=o_k1 < K, other=0.0).to(tl.float32)
        if TRANSPOSE_STATE:
            b_dv = tl.dot(b_k, tl.trans(b_dh1).to(b_k.dtype), input_precision=DOT_PRECISION, out_dtype=tl.float32)
        else:
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype), input_precision=DOT_PRECISION, out_dtype=tl.float32)
        if K > 64:
            p_k = tl.make_block_ptr(k, (T, K), (Hq * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            if USE_GK:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + last_idx * H * K + o_k2, mask=o_k2 < K, other=0.0).to(tl.float32)
            if TRANSPOSE_STATE:
                b_dv += tl.dot(b_k, tl.trans(b_dh2).to(b_k.dtype), input_precision=DOT_PRECISION, out_dtype=tl.float32)
            else:
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype), input_precision=DOT_PRECISION, out_dtype=tl.float32)
        if K > 128:
            p_k = tl.make_block_ptr(k, (T, K), (Hq * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            if USE_GK:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + last_idx * H * K + o_k3, mask=o_k3 < K, other=0.0).to(tl.float32)
            if TRANSPOSE_STATE:
                b_dv += tl.dot(b_k, tl.trans(b_dh3).to(b_k.dtype), input_precision=DOT_PRECISION, out_dtype=tl.float32)
            else:
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype), input_precision=DOT_PRECISION, out_dtype=tl.float32)
        if K > 192:
            p_k = tl.make_block_ptr(k, (T, K), (Hq * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            if USE_GK:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + last_idx * H * K + o_k4, mask=o_k4 < K, other=0.0).to(tl.float32)
            if TRANSPOSE_STATE:
                b_dv += tl.dot(b_k, tl.trans(b_dh4).to(b_k.dtype), input_precision=DOT_PRECISION, out_dtype=tl.float32)
            else:
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype), input_precision=DOT_PRECISION, out_dtype=tl.float32)
        if USE_G:
            m_t = i_t * BT + tl.arange(0, BT) < T
            if USE_EXP2:
                b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
            else:
                b_dv *= tl.where(m_t, exp(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, boundary_check=(0, 1))
        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        p_w = tl.make_block_ptr(w, (K, T), (1, H * K), (0, i_t * BT), (64, BT), (0, 1))
        p_q = tl.make_block_ptr(q, (K, T), (1, Hq * K), (0, i_t * BT), (64, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1)).to(tl.float32)
        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        if USE_GK:
            if TRANSPOSE_STATE:
                if USE_EXP2:
                    b_dh1 *= exp2(b_gk_last1)[None, :]
                else:
                    b_dh1 *= exp(b_gk_last1)[None, :]
            elif USE_EXP2:
                b_dh1 *= exp2(b_gk_last1[:, None])
            else:
                b_dh1 *= exp(b_gk_last1[:, None])
        if TRANSPOSE_STATE:
            b_dh1 += tl.trans(_frozen_dot_whitelisted_dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), BF16_QG_DO_OPERANDS, DOT_PRECISION) * scale - _frozen_dot_whitelisted_dot(b_w, b_dv.to(b_w.dtype), BF16_W_DV_OPERANDS, DOT_PRECISION))
        else:
            b_dh1 += _frozen_dot_whitelisted_dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), BF16_QG_DO_OPERANDS, DOT_PRECISION) * scale - _frozen_dot_whitelisted_dot(b_w, b_dv.to(b_w.dtype), BF16_W_DV_OPERANDS, DOT_PRECISION)
        if K > 64:
            p_q = tl.make_block_ptr(q, (K, T), (1, Hq * K), (64, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, H * K), (64, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
            b_w = tl.load(p_w, boundary_check=(0, 1)).to(tl.float32)
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_dh2 *= exp2(b_gk_last2)[None, :]
                    else:
                        b_dh2 *= exp(b_gk_last2)[None, :]
                elif USE_EXP2:
                    b_dh2 *= exp2(b_gk_last2[:, None])
                else:
                    b_dh2 *= exp(b_gk_last2[:, None])
            if TRANSPOSE_STATE:
                b_dh2 += tl.trans(_frozen_dot_whitelisted_dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), BF16_QG_DO_OPERANDS, DOT_PRECISION) * scale - _frozen_dot_whitelisted_dot(b_w, b_dv.to(b_w.dtype), BF16_W_DV_OPERANDS, DOT_PRECISION))
            else:
                b_dh2 += _frozen_dot_whitelisted_dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), BF16_QG_DO_OPERANDS, DOT_PRECISION) * scale - _frozen_dot_whitelisted_dot(b_w, b_dv.to(b_w.dtype), BF16_W_DV_OPERANDS, DOT_PRECISION)
        if K > 128:
            p_q = tl.make_block_ptr(q, (K, T), (1, Hq * K), (128, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, H * K), (128, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
            b_w = tl.load(p_w, boundary_check=(0, 1)).to(tl.float32)
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_dh3 *= exp2(b_gk_last3)[None, :]
                    else:
                        b_dh3 *= exp(b_gk_last3)[None, :]
                elif USE_EXP2:
                    b_dh3 *= exp2(b_gk_last3[:, None])
                else:
                    b_dh3 *= exp(b_gk_last3[:, None])
            if TRANSPOSE_STATE:
                b_dh3 += tl.trans(_frozen_dot_whitelisted_dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), BF16_QG_DO_OPERANDS, DOT_PRECISION) * scale - _frozen_dot_whitelisted_dot(b_w, b_dv.to(b_w.dtype), BF16_W_DV_OPERANDS, DOT_PRECISION))
            else:
                b_dh3 += _frozen_dot_whitelisted_dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), BF16_QG_DO_OPERANDS, DOT_PRECISION) * scale - _frozen_dot_whitelisted_dot(b_w, b_dv.to(b_w.dtype), BF16_W_DV_OPERANDS, DOT_PRECISION)
        if K > 192:
            p_q = tl.make_block_ptr(q, (K, T), (1, Hq * K), (192, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, H * K), (192, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
            b_w = tl.load(p_w, boundary_check=(0, 1)).to(tl.float32)
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_dh4 *= exp2(b_gk_last4)[None, :]
                    else:
                        b_dh4 *= exp(b_gk_last4)[None, :]
                elif USE_EXP2:
                    b_dh4 *= exp2(b_gk_last4[:, None])
                else:
                    b_dh4 *= exp(b_gk_last4[:, None])
            if TRANSPOSE_STATE:
                b_dh4 += tl.trans(_frozen_dot_whitelisted_dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), BF16_QG_DO_OPERANDS, DOT_PRECISION) * scale - _frozen_dot_whitelisted_dot(b_w, b_dv.to(b_w.dtype), BF16_W_DV_OPERANDS, DOT_PRECISION))
            else:
                b_dh4 += _frozen_dot_whitelisted_dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), BF16_QG_DO_OPERANDS, DOT_PRECISION) * scale - _frozen_dot_whitelisted_dot(b_w, b_dv.to(b_w.dtype), BF16_W_DV_OPERANDS, DOT_PRECISION)
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_dh0 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_dh1 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_dh1 = tl.make_block_ptr(dh0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_dh2 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_dh2 = tl.make_block_ptr(dh0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_dh3 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_dh3 = tl.make_block_ptr(dh0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages=num_stages) for num_warps in _frozen_kda_NUM_WARPS for num_stages in [2, 3, 4]], key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'DOT_PRECISION', 'BF16_DOT_OPERANDS', 'BF16_DAQK_DO_VNEW_OPERANDS', 'DVNEW_AQK_DO_DOT_PRECISION', 'BF16_DVNEW_AQK_DO_OPERANDS'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def _frozen_kda_chunk_kda_bwd_kernel_dAv_diag(q, k, v, A, do, dv, dA, cu_seqlens, chunk_indices, scale, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr, DOT_PRECISION: tl.constexpr, BF16_DOT_OPERANDS: tl.constexpr, BF16_DAQK_DO_VNEW_OPERANDS: tl.constexpr, DVNEW_AQK_DO_DOT_PRECISION: tl.constexpr, BF16_DVNEW_AQK_DO_OPERANDS: tl.constexpr):
    i_t, i_bh = (tl.program_id(0), tl.program_id(1))
    i_b, i_h = (i_bh // H, i_bh % H)
    if IS_VARLEN:
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
    else:
        bos, eos = (i_b * T, i_b * T + T)
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    dA += (bos * H + i_h) * BT
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (BT, T), (1, H * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0).to(tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (V, T), (1, H * V), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
        b_dA += _frozen_dot_whitelisted_dot(b_do, b_v, BF16_DAQK_DO_VNEW_OPERANDS, DOT_PRECISION)
        b_dv = _frozen_dot_whitelisted_dot(b_A, b_do, BF16_DVNEW_AQK_DO_OPERANDS, DVNEW_AQK_DO_DOT_PRECISION)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    p_dA = tl.make_block_ptr(dA, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_dA = tl.where(o_t[:, None] >= o_t, b_dA * scale, 0.0)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


# ---- frozen source: lit_gpt/kla_ops/kalman_s3/fla_compat.py ----------------

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@fla_cache_autotune(configs=[triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages) for BK in [32, 64] for BV in [64, 128] for num_warps in [2, 4, 8] for num_stages in [2, 3, 4]], key=['BT', 'HV', 'TRANSPOSE_STATE', 'DOT_PRECISION', 'BF16_INTER_DOT_OPERANDS', 'BF16_INTRA_DOT_OPERANDS'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def _frozen_readout_chunk_gla_fwd_kernel_o_diag(q, v, g, h, o, A, cu_seqlens, chunk_indices, scale, T, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_EXP2: tl.constexpr, TRANSPOSE_STATE: tl.constexpr, IS_VARLEN: tl.constexpr, DOT_PRECISION: tl.constexpr, BF16_INTER_DOT_OPERANDS: tl.constexpr, BF16_INTRA_DOT_OPERANDS: tl.constexpr):
    i_v, i_t, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_b, i_hv = (i_bh // HV, i_bh % HV)
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64))
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = ((i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64))
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    q += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    o += (bos * HV + i_hv) * V
    h += (i_tg * HV + i_hv).to(tl.int64) * K * V
    A += (bos * HV + i_hv) * BT
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        if TRANSPOSE_STATE:
            p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        else:
            p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        if USE_EXP2:
            b_qg = (b_q * exp2(b_g)).to(tl.float32)
        else:
            b_qg = (b_q * exp(b_g)).to(tl.float32)
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
        if i_k >= 0:
            if TRANSPOSE_STATE:
                b_o += _frozen_dot_whitelisted_dot(b_qg, tl.trans(b_h).to(b_qg.dtype), BF16_INTER_DOT_OPERANDS, DOT_PRECISION)
            else:
                b_o += _frozen_dot_whitelisted_dot(b_qg, b_h.to(b_qg.dtype), BF16_INTER_DOT_OPERANDS, DOT_PRECISION)
    b_o *= scale
    p_v = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    b_A = tl.where(m_s, b_A, 0.0).to(tl.float32)
    b_o += _frozen_dot_whitelisted_dot(b_A, b_v, BF16_INTRA_DOT_OPERANDS, DOT_PRECISION)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# ---- frozen source: lit_gpt/kla_ops/kalman_s3/kalman_intra_triton.py ----------------

NUM_WARPS_INTRA = [1, 2, 4] if IS_NVIDIA_HOPPER else [1, 2, 4, 8]

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BH': BH}, num_warps=num_warps) for BH in [1, 2, 4, 8] for num_warps in [1, 2, 4, 8]], key=['K', 'H'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T', 'N'])
def _frozen_intra_chunk_kalman_fwd_kernel_intra_token_parallel(q, k, kappa, g, Aqk, Akk, scale, cu_seqlens, N, T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BH: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_tg, i_hg = (tl.program_id(0), tl.program_id(1))
    if IS_VARLEN:
        i_n = 0
        left, right = (0, N)
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1
        i_n = left
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
        i_t = i_tg - bos
    else:
        bos = i_tg // T * T
        i_t = i_tg % T
    if i_t >= T:
        return
    i_c = i_t // BT
    i_s = i_t % BT // BC
    i_tc = i_c * BT
    i_ts = i_tc + i_s * BC
    q += bos * H * K
    k += bos * H * K
    kappa += bos * H * K
    g += bos * H * K
    Aqk += bos * H * BT
    Akk += bos * H * BC
    BK: tl.constexpr = triton.next_power_of_2(K)
    o_h = tl.arange(0, BH)
    o_k = tl.arange(0, BK)
    m_h = i_hg * BH + o_h < H
    m_k = o_k < K
    p_q = tl.make_block_ptr(q + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
        p_kappaj = tl.make_block_ptr(kappa + j * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
        p_gj = tl.make_block_ptr(g + j * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
        b_kappaj = tl.load(p_kappaj, boundary_check=(0, 1)).to(tl.float32)
        b_gj = tl.load(p_gj, boundary_check=(0, 1)).to(tl.float32)
        b_kappagj = b_kappaj * exp2(b_g - b_gj)
        b_kappagj = tl.where(m_k[None, :], b_kappagj, 0.0)
        b_Aqk = tl.sum(b_q * b_kappagj, axis=1) * scale
        b_Akk = tl.sum(b_k * b_kappagj, axis=1) * tl.where(j < i_t, 1.0, 0.0)
        tl.store(Aqk + i_t * H * BT + (i_hg * BH + o_h) * BT + j % BT, b_Aqk.to(Aqk.dtype.element_ty), mask=m_h)
        tl.store(Akk + i_t * H * BC + (i_hg * BH + o_h) * BC + j - i_ts, b_Akk.to(Akk.dtype.element_ty), mask=m_h)

@triton.heuristics({'STORE_M': lambda args: args['Mraw'] is not None})
@triton.autotune(configs=[triton.Config({'BK': BK}, num_warps=num_warps) for BK in [32, 64] for num_warps in [1, 2, 4]], key=['H', 'K', 'BC', 'SCORE_DOT_PRECISION', 'SOLVE_DOT_PRECISION'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def _frozen_intra_chunk_kalman_fwd_kernel_inter_solve_fused_bt32_explicit(q, k, kappa, g, Aqk, Akkd, Akk, Mraw, scale, T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr, STORE_M: tl.constexpr, SCORE_DOT_PRECISION: tl.constexpr, SOLVE_DOT_PRECISION: tl.constexpr):
    """Two-subchunk BT32 solve with the same FP32/TF32x3 contract as BT64."""
    tl.static_assert(BT == 32)
    tl.static_assert(BC == 16)
    tl.static_assert(NC == 2)
    tl.static_assert(BT == BC * NC)
    i_t, i_bh = (tl.program_id(0), tl.program_id(1))
    i_b, i_h = (i_bh // H, i_bh % H)
    bos = i_b * T
    if i_t * BT >= T:
        return
    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    kappa += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT
    Akkd += (bos * H + i_h) * BC
    if STORE_M:
        Mraw += (bos * H + i_h) * BT
    o_i = tl.arange(0, BC)
    m_tc1 = i_tc1 + o_i < T
    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_M10 = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_kappa0 = tl.make_block_ptr(kappa, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        p_g0 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_kappa0 = tl.load(p_kappa0, boundary_check=(0, 1), padding_option='zero').to(tl.float32)
        b_g0 = tl.load(p_g0, boundary_check=(0, 1), padding_option='zero').to(tl.float32)
        if i_tc1 < T:
            p_q1 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_g1 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            b_q1 = tl.load(p_q1, boundary_check=(0, 1), padding_option='zero').to(tl.float32)
            b_k1 = tl.load(p_k1, boundary_check=(0, 1), padding_option='zero').to(tl.float32)
            b_g1 = tl.load(p_g1, boundary_check=(0, 1), padding_option='zero').to(tl.float32)
            b_gn1 = tl.load(g + i_tc1 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            b_kgt = tl.trans(b_kappa0 * exp2(b_gn1[None, :] - b_g0))
            b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt, input_precision=SCORE_DOT_PRECISION, out_dtype=tl.float32)
            b_M10 += tl.dot(b_k1 * b_gqn, b_kgt, input_precision=SCORE_DOT_PRECISION, out_dtype=tl.float32)
    b_Aqk10 = tl.where(m_tc1[:, None], b_Aqk10, 0.0)
    b_M10 = tl.where(m_tc1[:, None], b_M10, 0.0)
    if i_tc1 < T:
        p_Aqk10 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
        tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    p_Akkd00 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akkd11 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0))
    b_Ai00 = tl.load(p_Akkd00, boundary_check=(0, 1), padding_option='zero').to(tl.float32)
    b_Ai11 = tl.load(p_Akkd11, boundary_check=(0, 1), padding_option='zero').to(tl.float32)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    b_Ai00 = -tl.where(m_A, b_Ai00, 0)
    b_Ai11 = -tl.where(m_A, b_Ai11, 0)
    if STORE_M:
        p_M00 = tl.make_block_ptr(Mraw, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
        p_M11 = tl.make_block_ptr(Mraw, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
        p_M10 = tl.make_block_ptr(Mraw, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
        tl.store(p_M00, (-b_Ai00).to(Mraw.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_M11, (-b_Ai11).to(Mraw.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_M10, b_M10.to(Mraw.dtype.element_ty), boundary_check=(0, 1))
    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
        b_a00 = tl.where(o_i < i, b_a00, 0.0)
        b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(BC + 2, min(2 * BC, T - i_tc0)):
        b_a11 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
        b_a11 = tl.where(o_i < i - BC, b_a11, 0.0)
        b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai10 = -tl.dot(tl.dot(b_Ai11, b_M10, input_precision=SOLVE_DOT_PRECISION, out_dtype=tl.float32), b_Ai00, input_precision=SOLVE_DOT_PRECISION, out_dtype=tl.float32)
    b_Ai10 = tl.where(m_tc1[:, None], b_Ai10, 0.0)
    p_Akk00 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk10 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))

@triton.heuristics({'STORE_QG': lambda args: args['qg'] is not None, 'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages=num_stages) for num_warps in [2, 4, 8] for num_stages in [2, 3, 4]], key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN', 'DOT_PRECISION', 'BF16_DOT_OPERANDS'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def _frozen_intra_recompute_w_u_fwd_kalman_kernel_explicit(q, k, kappa, qg, kg, v, w, u, A, gk, cu_seqlens, chunk_indices, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, STORE_QG: tl.constexpr, IS_VARLEN: tl.constexpr, DOT_PRECISION: tl.constexpr, BF16_DOT_OPERANDS: tl.constexpr):
    i_t, i_bh = (tl.program_id(0), tl.program_id(1))
    i_b, i_h = (i_bh // H, i_bh % H)
    if IS_VARLEN:
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
    else:
        bos, eos = (i_b * T, i_b * T + T)
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_u = _frozen_dot_whitelisted_dot(b_A, b_v, BF16_DOT_OPERANDS, DOT_PRECISION)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))
    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        p_gk = tl.make_block_ptr(gk + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        b_kb = b_k * exp2(b_gk)
        if STORE_QG:
            p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_qg = tl.make_block_ptr(qg + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))
        last_idx = min(i_t * BT + BT, T) - 1
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_gn = tl.load(gk + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.0).to(tl.float32)
        p_kappa = tl.make_block_ptr(kappa + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_kappa = tl.load(p_kappa, boundary_check=(0, 1)).to(tl.float32)
        b_kg = b_kappa * tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], exp2(b_gn[None, :] - b_gk), 0)
        p_kg = tl.make_block_ptr(kg + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))
        b_w = _frozen_dot_whitelisted_dot(b_A, b_kb.to(tl.float32), BF16_DOT_OPERANDS, DOT_PRECISION)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages=num_stages) for num_warps in NUM_WARPS_INTRA for num_stages in [2, 3, 4]], key=['BK', 'NC', 'BT', 'DOT_PRECISION', 'BF16_DOT_OPERANDS', 'BF16_DK_ROW_OPERANDS', 'BF16_DQ_ROW_OPERANDS', 'DQ_ROW_DOT_PRECISION', 'K_TILES_PER_CTA'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['B', 'T'])
def _frozen_intra_chunk_kalman_bwd_kernel_intra_explicit(q, k, kappa, g, dAqk, dAkk, dq, dq2, dk, dk2, dkappa, dkappa2, dg, dg2, cu_seqlens, chunk_indices, B, T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr, IS_VARLEN: tl.constexpr, SAFE_GATE: tl.constexpr, USE_GATHER: tl.constexpr, DOT_PRECISION: tl.constexpr, BF16_DOT_OPERANDS: tl.constexpr, BF16_DK_ROW_OPERANDS: tl.constexpr, BF16_DQ_ROW_OPERANDS: tl.constexpr, DQ_ROW_DOT_PRECISION: tl.constexpr, K_TILES_PER_CTA: tl.constexpr):
    i_kc, i_t, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_b, i_h = (i_bh // H, i_bh % H)
    i_k_group, i_i = (i_kc // NC, i_kc % NC)
    all = B * T
    if IS_VARLEN:
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
    else:
        bos, eos = (i_b * T, i_b * T + T)
    T = eos - bos
    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    kappa += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    dAqk += (bos * H + i_h) * BT
    dAkk += (bos * H + i_h) * BT
    dq += (bos * H + i_h) * K
    dq2 += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dk2 += (bos * H + i_h) * K
    dkappa += (bos * H + i_h) * K
    dkappa2 += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K
    dg2 += (bos * H + i_h) * K
    for i_k_offset in range(0, K_TILES_PER_CTA):
        i_k = i_k_group * K_TILES_PER_CTA + i_k_offset
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
        b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)
        if i_i > 0:
            p_gn = g + i_ti * H * K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
            for i_j in range(0, i_i):
                p_kappa = tl.make_block_ptr(kappa, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
                p_gk = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
                p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (H * BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
                p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (H * BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
                b_kappa = tl.load(p_kappa, boundary_check=(0, 1)).to(tl.float32)
                b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
                b_kappag = b_kappa * exp2(b_gn - b_gk)
                b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
                b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))
                b_dq2 += _frozen_dot_whitelisted_dot(b_dAqk, b_kappag, BF16_DQ_ROW_OPERANDS, DQ_ROW_DOT_PRECISION)
                b_dk2 += _frozen_dot_whitelisted_dot(b_dAkk, b_kappag, BF16_DK_ROW_OPERANDS, DOT_PRECISION)
            b_gqn = exp2(b_g - b_gn)
            b_dq2 *= b_gqn
            b_dk2 *= b_gqn
        o_i = tl.arange(0, BC)
        m_dA = i_ti + o_i < T
        o_dA = (i_ti + o_i) * H * BT + i_i * BC
        p_kappaj = kappa + i_ti * H * K + o_k
        p_gkj = g + i_ti * H * K + o_k
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        if SAFE_GATE:
            p_kappa_diag = tl.make_block_ptr(kappa, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            b_kappa_diag = tl.load(p_kappa_diag, boundary_check=(0, 1)).to(tl.float32)
            if USE_GATHER:
                b_gn = gather(b_g, tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16), axis=0)
            else:
                p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H * K + o_k
                b_gn = tl.load(p_gn, mask=m_k, other=0)[None, :]
            p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (H * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
            p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (H * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
            b_dAqk_diag_qk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
            b_dAkk_diag_qk = tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32)
            m_i_diag_qk = (o_i[:, None] >= o_i[None, :]) & (i_ti + o_i[:, None] < T) & (i_ti + o_i[None, :] < T)
            m_j_diag_qk = i_ti + o_i[:, None] < T
            b_dAqk_diag_qk = tl.where(m_i_diag_qk, b_dAqk_diag_qk, 0.0)
            b_dAkk_diag_qk = tl.where(m_i_diag_qk, b_dAkk_diag_qk, 0.0)
            b_g_diag_qk = tl.where(m_j_diag_qk, b_g - b_gn, 0.0)
            exp_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(b_g_diag_qk), 0.0)
            exp_neg_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(-b_g_diag_qk), 0.0)
            b_kappa_exp_diag_qk = b_kappa_diag * exp_neg_b_g_diag_qk
            b_dq2 += _frozen_dot_whitelisted_dot(b_dAqk_diag_qk, b_kappa_exp_diag_qk, BF16_DQ_ROW_OPERANDS, DQ_ROW_DOT_PRECISION) * exp_b_g_diag_qk
            b_dk2 += _frozen_dot_whitelisted_dot(b_dAkk_diag_qk, b_kappa_exp_diag_qk, BF16_DK_ROW_OPERANDS, DOT_PRECISION) * exp_b_g_diag_qk
        else:
            for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
                b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)
                b_dAkk = tl.load(dAkk + o_dA + j, mask=m_dA, other=0)
                b_kappaj = tl.load(p_kappaj, mask=m_k, other=0).to(tl.float32)
                b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
                m_i = o_i[:, None] >= j
                b_gqk = exp2(b_g - b_gkj[None, :])
                b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kappaj[None, :] * b_gqk, 0.0)
                b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kappaj[None, :] * b_gqk, 0.0)
                p_kappaj += H * K
                p_gkj += H * K
        p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        p_dq2 = tl.make_block_ptr(dq2, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        b_dg2 = b_q * b_dq2
        b_dq2 = b_dq2 + tl.load(p_dq, boundary_check=(0, 1))
        tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=(0, 1))
        tl.debug_barrier()
        b_dkt = tl.zeros([BC, BK], dtype=tl.float32)
        active_nc = min(NC, tl.cdiv(T - i_t * BT, BC))
        if i_i < active_nc - 1:
            p_gn = g + (min(i_ti + BC, T) - 1) * H * K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
            for i_j in range(i_i + 1, active_nc):
                p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
                p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
                p_gk = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
                p_dAqk = tl.make_block_ptr(dAqk, (BT, T), (1, H * BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
                p_dAkk = tl.make_block_ptr(dAkk, (BT, T), (1, H * BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
                b_k_row = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
                b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
                b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
                b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))
                o_j = i_t * BT + i_j * BC + o_i
                m_j = o_j < T
                b_gkn = exp2(b_gk - b_gn)
                b_qg = b_q * tl.where(m_j[:, None], b_gkn, 0)
                b_kg_row = b_k_row * tl.where(m_j[:, None], b_gkn, 0)
                b_dkt += tl.dot(b_dAqk, b_qg, input_precision=DOT_PRECISION, out_dtype=tl.float32)
                b_dkt += tl.dot(b_dAkk, b_kg_row, input_precision=DOT_PRECISION, out_dtype=tl.float32)
            b_dkt *= exp2(b_gn - b_g)
        o_dA = i_ti * H * BT + i_i * BC + o_i
        p_qj = q + i_ti * H * K + o_k
        p_kj = k + i_ti * H * K + o_k
        p_gkj = g + i_ti * H * K + o_k
        if SAFE_GATE:
            if USE_GATHER:
                b_gn = gather(b_g, tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16), axis=0)
            else:
                p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H * K + o_k
                b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
            p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
            p_dAqk = tl.make_block_ptr(dAqk, (BT, T), (1, H * BT), (i_i * BC, i_ti), (BC, BC), (0, 1))
            p_dAkk = tl.make_block_ptr(dAkk, (BT, T), (1, H * BT), (i_i * BC, i_ti), (BC, BC), (0, 1))
            b_dAqk_diag_kk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
            b_dAkk_diag_kk = tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32)
            m_i_diag_kk = (o_i[:, None] <= o_i[None, :]) & (i_ti + o_i[:, None] < T) & (i_ti + o_i[None, :] < T)
            m_j_diag_kk = i_ti + o_i[:, None] < T
            b_dAqk_diag_kk = tl.where(m_i_diag_kk, b_dAqk_diag_kk, 0.0)
            b_dAkk_diag_kk = tl.where(m_i_diag_kk, b_dAkk_diag_kk, 0.0)
            b_g_diag_kk = tl.where(m_j_diag_kk, b_g - b_gn, 0.0)
            exp_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(b_g_diag_kk), 0.0)
            exp_neg_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(-b_g_diag_kk), 0.0)
            b_q_exp = b_q * exp_b_g_diag_kk
            b_k_exp = b_k * exp_b_g_diag_kk
            b_dkt += tl.dot(b_dAqk_diag_kk, b_q_exp, input_precision=DOT_PRECISION, out_dtype=tl.float32) * exp_neg_b_g_diag_kk
            b_dkt += tl.dot(b_dAkk_diag_kk, b_k_exp, input_precision=DOT_PRECISION, out_dtype=tl.float32) * exp_neg_b_g_diag_kk
        else:
            for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
                b_dAqk = tl.load(dAqk + o_dA + j * H * BT)
                b_dAkk = tl.load(dAkk + o_dA + j * H * BT)
                b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
                b_kj_row = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
                b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
                m_i = o_i[:, None] <= j
                b_gkq = exp2(b_gkj[None, :] - b_g)
                b_dkt += tl.where(m_i, b_dAqk[:, None] * b_qj[None, :] * b_gkq, 0.0)
                b_dkt += tl.where(m_i, b_dAkk[:, None] * b_kj_row[None, :] * b_gkq, 0.0)
                p_qj += H * K
                p_kj += H * K
                p_gkj += H * K
        p_kappa_row = tl.make_block_ptr(kappa, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        b_kappa_row = tl.load(p_kappa_row, boundary_check=(0, 1))
        p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        p_dk2 = tl.make_block_ptr(dk2, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        p_dkappa = tl.make_block_ptr(dkappa, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        p_dkappa2 = tl.make_block_ptr(dkappa2, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        p_dg2 = tl.make_block_ptr(dg2, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        b_dg2 += b_dk2 * b_k - b_dkt * b_kappa_row + tl.load(p_dg, boundary_check=(0, 1))
        b_dk2 += tl.load(p_dk, boundary_check=(0, 1))
        b_dkt += tl.load(p_dkappa, boundary_check=(0, 1))
        tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dkappa2, b_dkt.to(p_dkappa2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), boundary_check=(0, 1))


# ---- frozen source: lit_gpt/kla_ops/kalman_s3/wy_buffered_dw.py ----------------

BT = 32

BK = 64

BV = 64

@triton.jit(do_not_specialize=['T'])
def _frozen_wy_chunk_kalman_bwd_kernel_wy_buffered_dw_producer(q, k, kappa, v_new, g, A, h, do, dh, dv, dq, dk, dkappa, dv2, dg, dw_scratch, scale, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, DOT_PRECISION: tl.constexpr, DKAPPA_DM_DOT_PRECISION: tl.constexpr, DQ_DO_H_DOT_PRECISION: tl.constexpr, BF16_DOT_OPERANDS: tl.constexpr, BF16_DW_DU_H_OPERANDS: tl.constexpr, BF16_DK_ABAR_DW_OPERANDS: tl.constexpr, BF16_DQ_DO_H_OPERANDS: tl.constexpr):
    """Produce the five non-dM outputs and post-negation FP32 dw."""
    tl.static_assert(BT == 32)
    tl.static_assert(BK == 64)
    tl.static_assert(BV == 64)
    tl.static_assert(K == V)
    tl.static_assert(K == 64 or K == 128)
    tl.static_assert(DOT_PRECISION == 'tf32x3')
    tl.static_assert(DKAPPA_DM_DOT_PRECISION == 'tf32')
    tl.static_assert(DQ_DO_H_DOT_PRECISION == 'tf32x3' or DQ_DO_H_DOT_PRECISION == 'tf32')
    tl.static_assert(BF16_DOT_OPERANDS)
    tl.static_assert(BF16_DW_DU_H_OPERANDS)
    tl.static_assert(BF16_DK_ABAR_DW_OPERANDS)
    i_t, i_bh = (tl.program_id(0), tl.program_id(1))
    i_b, i_h = (i_bh // H, i_bh % H)
    NT = tl.cdiv(T, BT)
    i_tg = (i_b * NT + i_t).to(tl.int64)
    bos = (i_b * T).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_last = o_t == min(T, i_t * BT + BT) - 1
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    kappa += (bos * H + i_h) * K
    v_new += (bos * H + i_h) * V
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT
    h += (i_tg * H + i_h) * K * V
    do += (bos * H + i_h) * V
    dh += (i_tg * H + i_h) * K * V
    dv += (bos * H + i_h) * V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dkappa += (bos * H + i_h) * K
    dv2 += (bos * H + i_h) * V
    dg += (bos * H + i_h) * K
    dw_scratch += (bos * H + i_h) * K
    p_A_p = tl.make_block_ptr(A, (BT, T), (1, H * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A_p = tl.load(p_A_p, boundary_check=(0, 1)).to(tl.float32)
    for i_k_p in range(tl.cdiv(K, BK)):
        o_k = i_k_p * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_dq_p = tl.zeros([BT, BK], dtype=tl.float32)
        b_dkappa_p = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw_p = tl.zeros([BT, BK], dtype=tl.float32)
        b_dgk_p = tl.zeros([BK], dtype=tl.float32)
        for i_v_p in range(tl.cdiv(V, BV)):
            p_v_new_p = tl.make_block_ptr(v_new, (T, V), (H * V, 1), (i_t * BT, i_v_p * BV), (BT, BV), (1, 0))
            p_do_p = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v_p * BV), (BT, BV), (1, 0))
            p_h_p = tl.make_block_ptr(h, (V, K), (1, V), (i_v_p * BV, i_k_p * BK), (BV, BK), (0, 1))
            p_dh_p = tl.make_block_ptr(dh, (V, K), (1, V), (i_v_p * BV, i_k_p * BK), (BV, BK), (0, 1))
            p_dv_p = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v_p * BV), (BT, BV), (1, 0))
            b_v_new_p = tl.load(p_v_new_p, boundary_check=(0, 1)).to(tl.float32)
            b_do_p = tl.load(p_do_p, boundary_check=(0, 1)).to(tl.float32)
            b_h_p = tl.load(p_h_p, boundary_check=(0, 1)).to(tl.float32)
            b_dh_p = tl.load(p_dh_p, boundary_check=(0, 1)).to(tl.float32)
            b_dv_p = tl.load(p_dv_p, boundary_check=(0, 1)).to(tl.float32)
            b_dgk_p += tl.sum(b_h_p * b_dh_p, axis=0)
            b_dq_p += _frozen_dot_whitelisted_dot(b_do_p, b_h_p, BF16_DQ_DO_H_OPERANDS, DQ_DO_H_DOT_PRECISION)
            b_dkappa_p += tl.dot(b_v_new_p, b_dh_p, input_precision=DKAPPA_DM_DOT_PRECISION, out_dtype=tl.float32)
            b_dw_p += _frozen_dot_whitelisted_dot(b_dv_p, b_h_p, BF16_DW_DU_H_OPERANDS, DOT_PRECISION)
            tl.debug_barrier()
            if i_k_p == 0:
                p_dv2_p = tl.make_block_ptr(dv2, (T, V), (H * V, 1), (i_t * BT, i_v_p * BV), (BT, BV), (1, 0))
                b_dv2_p = _frozen_dot_whitelisted_dot(b_A_p, b_dv_p, BF16_DOT_OPERANDS, DOT_PRECISION)
                tl.store(p_dv2_p, b_dv2_p.to(p_dv2_p.dtype.element_ty), boundary_check=(0, 1))
        p_k_p = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k_p * BK), (BT, BK), (1, 0))
        p_g_p = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT, i_k_p * BK), (BT, BK), (1, 0))
        b_k_p = tl.load(p_k_p, boundary_check=(0, 1)).to(tl.float32)
        b_g_p = tl.load(p_g_p, boundary_check=(0, 1)).to(tl.float32)
        p_gn_p = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * H * K + o_k
        b_gn_p = tl.load(p_gn_p, mask=m_k, other=0.0).to(tl.float32)
        b_g_exp_p = exp2(b_g_p)
        b_dgk_p *= exp2(b_gn_p)
        b_dq_p = b_dq_p * b_g_exp_p * scale
        b_dkappa_p *= tl.where(m_t[:, None], exp2(b_gn_p[None, :] - b_g_p), 0.0)
        b_kg_p = b_k_p * b_g_exp_p
        b_dw_p = -b_dw_p.to(tl.float32)
        b_dkgb_p = _frozen_dot_whitelisted_dot(b_A_p, b_dw_p, BF16_DK_ABAR_DW_OPERANDS, DOT_PRECISION)
        p_dw_scratch_p = tl.make_block_ptr(dw_scratch, (T, K), (H * K, 1), (i_t * BT, i_k_p * BK), (BT, BK), (1, 0))
        tl.store(p_dw_scratch_p, b_dw_p.to(p_dw_scratch_p.dtype.element_ty), boundary_check=(0, 1))
        p_q_p = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k_p * BK), (BT, BK), (1, 0))
        p_kappa_p = tl.make_block_ptr(kappa, (T, K), (H * K, 1), (i_t * BT, i_k_p * BK), (BT, BK), (1, 0))
        b_q_p = tl.load(p_q_p, boundary_check=(0, 1)).to(tl.float32)
        b_kappa_p = tl.load(p_kappa_p, boundary_check=(0, 1)).to(tl.float32)
        b_kdk_p = b_kappa_p * b_dkappa_p
        b_dgk_p += tl.sum(b_kdk_p, axis=0)
        b_dg_p = b_q_p * b_dq_p - b_kdk_p + m_last[:, None] * b_dgk_p + b_kg_p * b_dkgb_p
        b_dk_p = b_dkgb_p * b_g_exp_p
        p_dq_p = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_t * BT, i_k_p * BK), (BT, BK), (1, 0))
        p_dk_p = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_t * BT, i_k_p * BK), (BT, BK), (1, 0))
        p_dkappa_p = tl.make_block_ptr(dkappa, (T, K), (H * K, 1), (i_t * BT, i_k_p * BK), (BT, BK), (1, 0))
        p_dg_p = tl.make_block_ptr(dg, (T, K), (H * K, 1), (i_t * BT, i_k_p * BK), (BT, BK), (1, 0))
        tl.store(p_dq_p, b_dq_p.to(p_dq_p.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk_p, b_dk_p.to(p_dk_p.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dkappa_p, b_dkappa_p.to(p_dkappa_p.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg_p, b_dg_p.to(p_dg_p.dtype.element_ty), boundary_check=(0, 1))

@triton.jit(do_not_specialize=['T'])
def _frozen_wy_chunk_kalman_bwd_kernel_wy_buffered_dw_consumer(k, v, g, A, dv, dw_scratch, dM, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, DKAPPA_DM_DOT_PRECISION: tl.constexpr):
    """Consume buffered post-negation dw to form the exact Phase-D dM."""
    tl.static_assert(BT == 32)
    tl.static_assert(BK == 64)
    tl.static_assert(BV == 64)
    tl.static_assert(K == V)
    tl.static_assert(K == 64 or K == 128)
    tl.static_assert(DKAPPA_DM_DOT_PRECISION == 'tf32')
    i_t, i_bh = (tl.program_id(0), tl.program_id(1))
    i_b, i_h = (i_bh // H, i_bh % H)
    bos = (i_b * T).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT
    dv += (bos * H + i_h) * V
    dw_scratch += (bos * H + i_h) * K
    dM += (bos * H + i_h) * BT
    b_dM_c = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v_c in range(tl.cdiv(V, BV)):
        p_v_c = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v_c * BV), (BT, BV), (1, 0))
        p_dv_c = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v_c * BV), (BT, BV), (1, 0))
        b_v_c = tl.load(p_v_c, boundary_check=(0, 1)).to(tl.float32)
        b_dv_c = tl.load(p_dv_c, boundary_check=(0, 1)).to(tl.float32)
        b_dM_c += tl.dot(b_dv_c, tl.trans(b_v_c), input_precision=DKAPPA_DM_DOT_PRECISION, out_dtype=tl.float32)
    for i_k_c in range(tl.cdiv(K, BK)):
        p_dw_scratch_c = tl.make_block_ptr(dw_scratch, (T, K), (H * K, 1), (i_t * BT, i_k_c * BK), (BT, BK), (1, 0))
        b_dw_c = tl.load(p_dw_scratch_c, boundary_check=(0, 1)).to(tl.float32)
        p_k_c = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k_c * BK), (BT, BK), (1, 0))
        p_g_c = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT, i_k_c * BK), (BT, BK), (1, 0))
        b_k_c = tl.load(p_k_c, boundary_check=(0, 1)).to(tl.float32)
        b_g_c = tl.load(p_g_c, boundary_check=(0, 1)).to(tl.float32)
        b_kg_c = b_k_c * exp2(b_g_c)
        b_dM_c += tl.dot(b_dw_c, tl.trans(b_kg_c.to(tl.float32)), input_precision=DKAPPA_DM_DOT_PRECISION, out_dtype=tl.float32)
    m_M = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dM_c = tl.where(m_M, b_dM_c, 0.0)
    p_A_c = tl.make_block_ptr(A, (BT, T), (1, H * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A_c = tl.load(p_A_c, boundary_check=(0, 1)).to(tl.float32)
    b_dM_c = tl.dot(b_dM_c.to(tl.float32), b_A_c, input_precision=DKAPPA_DM_DOT_PRECISION, out_dtype=tl.float32)
    b_dM_c = tl.dot(b_A_c, b_dM_c.to(tl.float32), input_precision=DKAPPA_DM_DOT_PRECISION, out_dtype=tl.float32)
    b_dM_c = tl.where(m_M, -b_dM_c, 0.0)
    p_dM_c = tl.make_block_ptr(dM, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_dM_c, b_dM_c.to(p_dM_c.dtype.element_ty), boundary_check=(0, 1))


# ---- frozen source: lit_gpt/kla_ops/kalman_s3/diag_frontend.py ----------------

@triton.jit
def _frozen_frontend_diag_frontend_fwd_kernel(retention_raw_ptr, omega_raw_ptr, r_raw_ptr, A_log_ptr, dt_bias_ptr, initial_precision_param_ptr, first_ptr, alpha_ptr, omega_ptr, r_ptr, initial_precision_ptr, omega_min, r_min, T, NT, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, OMEGA_COUPLING: tl.constexpr, EMIT_RAW_G: tl.constexpr):
    i_bc = tl.program_id(0)
    i_h = tl.program_id(1)
    i_b = i_bc // NT
    i_c = i_bc % NT
    offs_t_1d = i_c * BT + tl.arange(0, BT)
    offs_k_1d = tl.arange(0, BK)
    offs_t = offs_t_1d[:, None]
    offs_k = offs_k_1d[None, :]
    mask_bthk = (offs_t_1d[:, None] < T) & (offs_k_1d[None, :] < K)
    mask_bth = offs_t_1d < T
    offsets_bthk = ((i_b * T + offs_t) * H + i_h) * K + offs_k
    offsets_bth = (i_b * T + offs_t_1d) * H + i_h
    f_value = tl.load(retention_raw_ptr + offsets_bthk, mask=mask_bthk, other=0.0).to(tl.float32)
    z_value = tl.load(omega_raw_ptr + offsets_bthk, mask=mask_bthk, other=0.0).to(tl.float32)
    y_value = tl.load(r_raw_ptr + offsets_bth, mask=mask_bth, other=0.0).to(tl.float32)
    A_log_value = tl.load(A_log_ptr + i_h).to(tl.float32)
    dt_bias_value = tl.load(dt_bias_ptr + i_h * K + offs_k_1d, mask=offs_k_1d < K, other=0.0).to(tl.float32)
    rate_value = tl.exp(A_log_value)
    x_value = f_value + dt_bias_value[None, :]
    softplus_value = softplus(x_value)
    ell_value = -rate_value * softplus_value
    alpha_value = tl.exp(ell_value)
    clamped_alpha_value = tl.maximum(alpha_value, 1e-06)
    memory_alpha_value = tl.where(alpha_value != alpha_value, alpha_value, clamped_alpha_value)
    ell_mem_value = tl.log(memory_alpha_value)
    masked_ell_mem_value = tl.where(mask_bthk, ell_mem_value, 0.0)
    prefix_value = tl.cumsum(masked_ell_mem_value, axis=0) * 1.4426950408889634
    omega_base_value = softplus(z_value)
    omega_value = omega_min + omega_base_value
    r_value = r_min + softplus(y_value)
    tl.store(first_ptr + offsets_bthk, ell_mem_value if EMIT_RAW_G else prefix_value, mask=mask_bthk)
    tl.store(alpha_ptr + offsets_bthk, alpha_value, mask=mask_bthk)
    tl.store(omega_ptr + offsets_bthk, omega_value, mask=mask_bthk)
    tl.store(r_ptr + offsets_bth, r_value, mask=mask_bth)
    if i_bc == 0:
        mu_value = tl.load(initial_precision_param_ptr + i_h).to(tl.float32)
        mu_output_value = 0.1 + softplus(mu_value)
        tl.store(initial_precision_ptr + i_h, mu_output_value)

@triton.jit
def _frozen_frontend_diag_frontend_uncoupled_omega_local_vjp_kernel(omega_raw_ptr, d_omega_ptr, domega_local_ptr, T, NT, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
    i_bc = tl.program_id(0)
    i_h = tl.program_id(1)
    i_b = i_bc // NT
    i_c = i_bc % NT
    offs_t_1d = i_c * BT + tl.arange(0, BT)
    offs_k_1d = tl.arange(0, BK)
    offs_t = offs_t_1d[:, None]
    offs_k = offs_k_1d[None, :]
    offsets = ((i_b * T + offs_t) * H + i_h) * K + offs_k
    mask = (offs_t_1d[:, None] < T) & (offs_k_1d[None, :] < K)
    z_value = tl.load(omega_raw_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    d_omega_value = tl.load(d_omega_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid_value = tl.sigmoid(z_value)
    dz_local_value = d_omega_value * sigmoid_value
    tl.store(domega_local_ptr + offsets, dz_local_value, mask=mask)

@triton.jit
def _frozen_frontend_diag_frontend_r_local_vjp_kernel(r_raw_ptr, d_r_ptr, dr_local_ptr, T, NT, H: tl.constexpr, BT: tl.constexpr):
    i_bc = tl.program_id(0)
    i_h = tl.program_id(1)
    i_b = i_bc // NT
    i_c = i_bc % NT
    offs_t = i_c * BT + tl.arange(0, BT)
    offsets = (i_b * T + offs_t) * H + i_h
    mask = offs_t < T
    y_value = tl.load(r_raw_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    d_r_value = tl.load(d_r_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid_value = tl.sigmoid(y_value)
    dy_local_value = d_r_value * sigmoid_value
    tl.store(dr_local_ptr + offsets, dy_local_value, mask=mask)

@triton.jit
def _frozen_frontend_diag_frontend_mu_local_vjp_kernel(initial_precision_param_ptr, d_initial_precision_ptr, dinitial_precision_local_ptr, H, BH: tl.constexpr):
    offs_h = tl.program_id(0) * BH + tl.arange(0, BH)
    mask = offs_h < H
    mu_value = tl.load(initial_precision_param_ptr + offs_h, mask=mask, other=0.0).to(tl.float32)
    d_mu_value = tl.load(d_initial_precision_ptr + offs_h, mask=mask, other=0.0).to(tl.float32)
    sigmoid_value = tl.sigmoid(mu_value)
    dmu_local_value = d_mu_value * sigmoid_value
    tl.store(dinitial_precision_local_ptr + offs_h, dmu_local_value, mask=mask)

@triton.jit
def _frozen_frontend_diag_frontend_dA_reduction_vjp_kernel(dA_partial_ptr, dA_log_ptr, NP, H: tl.constexpr, BNP: tl.constexpr):
    i_h = tl.program_id(0)
    offs_n = tl.arange(0, BNP)
    dA_partial_value = tl.load(dA_partial_ptr + offs_n * H + i_h, mask=offs_n < NP, other=0.0).to(tl.float32)
    dA_log_value = tl.sum(dA_partial_value, axis=0)
    tl.store(dA_log_ptr + i_h, dA_log_value)

@triton.jit
def _frozen_frontend_diag_frontend_ddt_reduction_vjp_kernel(ddt_partial_ptr, ddt_bias_ptr, NP, H: tl.constexpr, K: tl.constexpr, BNP: tl.constexpr):
    i_hk = tl.program_id(0)
    i_h = i_hk // K
    i_k = i_hk % K
    offs_n = tl.arange(0, BNP)
    ddt_partial_value = tl.load(ddt_partial_ptr + (offs_n * H + i_h) * K + i_k, mask=offs_n < NP, other=0.0).to(tl.float32)
    ddt_bias_value = tl.sum(ddt_partial_value, axis=0)
    tl.store(ddt_bias_ptr + i_hk, ddt_bias_value)

@triton.jit
def _frozen_frontend_diag_frontend_alpha_presence_vjp_kernel(retention_raw_ptr, A_log_ptr, dt_bias_ptr, alpha_ptr, d_first_ptr, d_alpha_gain_ptr, dretention_raw_ptr, dA_partial_ptr, ddt_partial_ptr, stride_d_first_b, stride_d_first_t, stride_d_first_h, stride_d_first_k, T, NT, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, ALPHA_FLOOR: tl.constexpr, HAS_DFIRST: tl.constexpr, HAS_DALPHA: tl.constexpr, NEED_DF_RAW: tl.constexpr, NEED_DA_PARTIAL: tl.constexpr, NEED_DDT_PARTIAL: tl.constexpr, ROUND_MEMORY_COTANGENT: tl.constexpr):
    i_bc = tl.program_id(0)
    i_h = tl.program_id(1)
    i_b = i_bc // NT
    i_c = i_bc % NT
    offs_t_1d = i_c * BT + tl.arange(0, BT)
    offs_k_1d = tl.arange(0, BK)
    offs_t = offs_t_1d[:, None]
    offs_k = offs_k_1d[None, :]
    offsets = ((i_b * T + offs_t) * H + i_h) * K + offs_k
    mask_t = offs_t_1d < T
    mask_k = offs_k_1d < K
    mask = mask_t[:, None] & mask_k[None, :]
    f_value = tl.load(retention_raw_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    A_log_value = tl.load(A_log_ptr + i_h).to(tl.float32)
    dt_bias_value = tl.load(dt_bias_ptr + i_h * K + offs_k_1d, mask=mask_k, other=0.0).to(tl.float32)
    alpha_value = tl.load(alpha_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    dalpha_value = tl.zeros((BT, BK), tl.float32)
    if HAS_DFIRST:
        d_first_offsets = i_b * stride_d_first_b + offs_t * stride_d_first_t + i_h * stride_d_first_h + offs_k * stride_d_first_k
        loaded_d_first_value = tl.load(d_first_ptr + d_first_offsets, mask=mask, other=0.0).to(tl.float32)
        d_first_value = tl.where(mask, loaded_d_first_value, 0.0)
        reverse_value = tl.cumsum(d_first_value, axis=0, reverse=True)
        if ROUND_MEMORY_COTANGENT:
            memory_cotangent = tl.cast(reverse_value, tl.bfloat16, fp_downcast_rounding='rtne').to(tl.float32)
        else:
            memory_cotangent = reverse_value
        dalpha_value += tl.where(alpha_value >= ALPHA_FLOOR, memory_cotangent / alpha_value, 0.0)
    if HAS_DALPHA:
        dalpha_value += tl.load(d_alpha_gain_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rate_value = tl.exp(A_log_value)
    x_value = f_value + dt_bias_value[None, :]
    ell_value = -rate_value * softplus(x_value)
    dell_value = dalpha_value * alpha_value
    sigmoid_value = tl.sigmoid(x_value)
    df_value = dell_value * -rate_value * sigmoid_value
    dA_value = dell_value * ell_value
    if NEED_DA_PARTIAL:
        valid_dA_value = tl.where(mask, dA_value, 0.0)
        dA_partial_value = tl.sum(tl.sum(valid_dA_value, axis=1), axis=0)
        tl.store(dA_partial_ptr + i_bc * H + i_h, dA_partial_value)
    if NEED_DDT_PARTIAL:
        valid_df_value = tl.where(mask, df_value, 0.0)
        ddt_partial_value = tl.sum(valid_df_value, axis=0)
        partial_k_offsets = (i_bc * H + i_h) * K + offs_k_1d
        tl.store(ddt_partial_ptr + partial_k_offsets, ddt_partial_value, mask=mask_k)
    if NEED_DF_RAW:
        tl.store(dretention_raw_ptr + offsets, df_value, mask=mask)


# ---- frozen source: lit_gpt/kla_ops/kalman_s3/kalman_chunk.py ----------------

# The selected orchestration is specialized below; legacy selectors, fallbacks, profiling, state-cache, and variable-length routes are omitted.

# ---- fixed DiagKDN integration ------------------------------------------------

def _validate_diag_kdn_norm_inputs(q_raw, k_raw):
    for name, value in (("q_raw", q_raw), ("k_raw", k_raw)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 4:
            raise ValueError(f"{name} must have rank 4 [B,T,H,K]")
        if not value.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if value.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(f"{name} must be BF16 or FP32")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if q_raw.shape != k_raw.shape or q_raw.device != k_raw.device:
        raise ValueError("q_raw and k_raw must have matching shape and device")
    if q_raw.dtype != k_raw.dtype or q_raw.requires_grad != k_raw.requires_grad:
        raise ValueError("q_raw and k_raw must have matching dtype and grad mode")
    if q_raw.shape[-1] not in (64, 128):
        raise ValueError("q_raw and k_raw require K in {64,128}")


class _DiagKdnNormFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=None)
    def forward(ctx, q_raw, k_raw):
        q_memory, k_gain, k_memory, q_norm, k_norm = (
            _frozen_norm_launch_diag_norm_boundary_forward(q_raw, k_raw)
        )
        ctx.save_for_backward(q_raw, k_raw, q_norm, k_norm)
        ctx.set_materialize_grads(False)
        return q_memory, k_gain, k_memory

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    @once_differentiable
    def backward(ctx, d_q_memory, d_k_gain, d_k_memory):
        if not ctx.needs_input_grad[0]:
            d_q_memory = None
        if not ctx.needs_input_grad[1]:
            d_k_gain = None
            d_k_memory = None
        if d_q_memory is None and d_k_gain is None and d_k_memory is None:
            return None, None
        q_raw, k_raw, q_norm, k_norm = ctx.saved_tensors
        return _diag_kdn_norm_backward(
            q_raw,
            k_raw,
            d_q_memory,
            d_k_gain,
            d_k_memory,
            q_norm,
            k_norm,
        )


def _diag_kdn_norm_backward(
    q_raw,
    k_raw,
    d_q_memory,
    d_k_gain,
    d_k_memory,
    q_norm,
    k_norm,
):
    process_q = d_q_memory is not None
    has_gain = d_k_gain is not None
    has_memory = d_k_memory is not None
    process_k = has_gain or has_memory
    d_q_raw = (
        torch.empty_like(q_raw, memory_format=torch.contiguous_format)
        if process_q
        else None
    )
    d_k_raw = (
        torch.empty_like(k_raw, memory_format=torch.contiguous_format)
        if process_k
        else None
    )
    use_single_row = (
        (d_q_memory is not None and not d_q_memory.is_contiguous())
        or (d_k_gain is not None and not d_k_gain.is_contiguous())
        or (d_k_memory is not None and not d_k_memory.is_contiguous())
    )
    row_block = 1 if use_single_row else _ROW_BLOCK
    num_warps = 1 if use_single_row else _frozen_norm_NUM_WARPS
    n_rows = q_raw.numel() // q_raw.shape[-1]
    if n_rows and (process_q or process_k):
        q_source = d_q_memory if d_q_memory is not None else q_raw
        k_gain_source = d_k_gain if d_k_gain is not None else q_raw
        k_memory_source = d_k_memory if d_k_memory is not None else q_raw
        q_output = d_q_raw if d_q_raw is not None else q_raw
        k_output = d_k_raw if d_k_raw is not None else k_raw
        _frozen_norm_diag_exact_norm_bwd_kernel[
            (triton.cdiv(n_rows, row_block),)
        ](
            q_raw,
            k_raw,
            q_source,
            k_gain_source,
            k_memory_source,
            q_norm,
            k_norm,
            q_output,
            k_output,
            q_raw.shape[0],
            q_raw.shape[1],
            q_raw.shape[2],
            n_rows,
            q_source.stride(0),
            q_source.stride(1),
            q_source.stride(2),
            q_source.stride(3),
            k_gain_source.stride(0),
            k_gain_source.stride(1),
            k_gain_source.stride(2),
            k_gain_source.stride(3),
            k_memory_source.stride(0),
            k_memory_source.stride(1),
            k_memory_source.stride(2),
            k_memory_source.stride(3),
            K=q_raw.shape[-1],
            ROW_BLOCK=row_block,
            PROCESS_Q=process_q,
            PROCESS_K=process_k,
            HAS_GAIN=has_gain,
            HAS_MEMORY=has_memory,
            INPUT_IS_BF16=q_raw.dtype == torch.bfloat16,
            FINAL_ROUND=True,
            SMALL_ROWS=n_rows < 16,
            REDUCTION_WIDTH=_frozen_norm_reduction_width(
                n_rows, q_raw.shape[-1]
            ),
            num_warps=num_warps,
            num_stages=_frozen_norm_NUM_STAGES,
        )
    return d_q_raw, d_k_raw


def _diag_kdn_normalize(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize q/k at the certified BF16 public boundary."""
    _validate_diag_kdn_norm_inputs(q_raw, k_raw)
    return _DiagKdnNormFunction.apply(q_raw, k_raw)


def _validate_diag_kdn_frontend_inputs(
    retention_raw,
    omega_raw,
    r_raw,
    A_log,
    dt_bias,
    initial_precision_param,
):
    tensors = {
        "retention_raw": retention_raw,
        "omega_raw": omega_raw,
        "r_raw": r_raw,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "initial_precision_param": initial_precision_param,
    }
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not value.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if value.device != retention_raw.device:
            raise ValueError("all DiagKDN frontend inputs must share one CUDA device")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if retention_raw.ndim != 4:
        raise ValueError("retention_raw must have rank 4 [B,T,H,K]")
    B, T, H, K = retention_raw.shape
    if min(B, T, H) <= 0 or K not in (64, 128):
        raise ValueError("DiagKDN frontend requires non-empty B/T/H and K in {64,128}")
    if omega_raw.shape != (B, T, H, K):
        raise ValueError("omega_raw must match retention_raw")
    if r_raw.shape != (B, T, H):
        raise ValueError("r_raw must have shape [B,T,H]")
    if A_log.shape != (H,) or initial_precision_param.shape != (H,):
        raise ValueError("A_log and initial_precision_param must have shape [H]")
    if dt_bias.shape != (H, K):
        raise ValueError("dt_bias must have shape [H,K]")
    for name in ("retention_raw", "omega_raw", "r_raw"):
        if tensors[name].dtype != _DIAG_KDN_VECTOR_DTYPE:
            raise TypeError(f"{name} must be torch.bfloat16")
    for name in ("A_log", "dt_bias", "initial_precision_param"):
        if tensors[name].dtype != _DIAG_KDN_STATE_DTYPE:
            raise TypeError(f"{name} must be torch.float32")
    return B, T, H, K


def _diag_kdn_frontend_forward(
    retention_raw,
    omega_raw,
    r_raw,
    A_log,
    dt_bias,
    initial_precision_param,
):
    B, T, H, K = retention_raw.shape
    BT = _DIAG_KDN_MEMORY_CHUNK_SIZE
    NT = triton.cdiv(T, BT)
    g_cumsum = torch.empty_like(retention_raw, dtype=_DIAG_KDN_STATE_DTYPE)
    alpha = torch.empty_like(retention_raw, dtype=_DIAG_KDN_STATE_DTYPE)
    omega = torch.empty_like(omega_raw, dtype=_DIAG_KDN_STATE_DTYPE)
    r = torch.empty_like(r_raw, dtype=_DIAG_KDN_STATE_DTYPE)
    initial_precision = torch.empty_like(
        initial_precision_param, dtype=_DIAG_KDN_STATE_DTYPE
    )
    with torch.cuda.device(retention_raw.device):
        _frozen_frontend_diag_frontend_fwd_kernel[(B * NT, H)](
            retention_raw,
            omega_raw,
            r_raw,
            A_log,
            dt_bias,
            initial_precision_param,
            g_cumsum,
            alpha,
            omega,
            r,
            initial_precision,
            0.0,
            0.01,
            T=T,
            NT=NT,
            H=H,
            K=K,
            BT=BT,
            BK=K,
            OMEGA_COUPLING=False,
            EMIT_RAW_G=False,
            num_warps=4,
            num_stages=2,
        )
    return g_cumsum, alpha, omega, r, initial_precision


def _diag_kdn_frontend_backward(
    retention_raw,
    omega_raw,
    r_raw,
    A_log,
    dt_bias,
    initial_precision_param,
    alpha,
    d_g_cumsum,
    d_alpha,
    d_omega,
    d_r,
    d_initial_precision,
    needs,
):
    B, T, H, K = retention_raw.shape
    BT = _DIAG_KDN_MEMORY_CHUNK_SIZE
    NT, NP = triton.cdiv(T, BT), B * triton.cdiv(T, BT)
    local_BT = _DIAG_KDN_FRONTEND_LOCAL_CHUNK_SIZE
    local_NT = triton.cdiv(T, local_BT)
    need_retention, need_omega, need_r, need_A, need_dt, need_initial = needs
    has_decay = d_g_cumsum is not None or d_alpha is not None
    need_decay = need_retention or need_A or need_dt
    dretention_raw = None
    dA_partial = None
    ddt_partial = None
    if has_decay and need_decay:
        dretention_raw = (
            torch.empty_like(retention_raw)
            if need_retention
            else None
        )
        dA_partial = (
            torch.empty((NP, H), device=retention_raw.device, dtype=torch.float32)
            if need_A
            else None
        )
        ddt_partial = (
            torch.empty(
                (NP, H, K), device=retention_raw.device, dtype=torch.float32
            )
            if need_dt
            else None
        )
        with torch.cuda.device(retention_raw.device):
            _frozen_frontend_diag_frontend_alpha_presence_vjp_kernel[(NP, H)](
                retention_raw,
                A_log,
                dt_bias,
                alpha,
                d_g_cumsum if d_g_cumsum is not None else alpha,
                d_alpha if d_alpha is not None else alpha,
                dretention_raw if dretention_raw is not None else retention_raw,
                dA_partial if dA_partial is not None else A_log,
                ddt_partial if ddt_partial is not None else dt_bias,
                d_g_cumsum.stride(0) if d_g_cumsum is not None else alpha.stride(0),
                d_g_cumsum.stride(1) if d_g_cumsum is not None else alpha.stride(1),
                d_g_cumsum.stride(2) if d_g_cumsum is not None else alpha.stride(2),
                d_g_cumsum.stride(3) if d_g_cumsum is not None else alpha.stride(3),
                T,
                NT,
                H=H,
                K=K,
                BT=BT,
                BK=K,
                ALPHA_FLOOR=1.0e-6,
                HAS_DFIRST=d_g_cumsum is not None,
                HAS_DALPHA=d_alpha is not None,
                NEED_DF_RAW=need_retention,
                NEED_DA_PARTIAL=need_A,
                NEED_DDT_PARTIAL=need_dt,
                ROUND_MEMORY_COTANGENT=False,
                num_warps=4,
                num_stages=2,
            )
    dA_log = None
    if dA_partial is not None:
        dA_log = torch.empty_like(A_log)
        with torch.cuda.device(retention_raw.device):
            _frozen_frontend_diag_frontend_dA_reduction_vjp_kernel[(H,)](
                dA_partial,
                dA_log,
                NP,
                H=H,
                BNP=triton.next_power_of_2(NP),
                num_warps=4,
                num_stages=1,
            )
    ddt_bias = None
    if ddt_partial is not None:
        ddt_bias = torch.empty_like(dt_bias)
        with torch.cuda.device(retention_raw.device):
            _frozen_frontend_diag_frontend_ddt_reduction_vjp_kernel[(H * K,)](
                ddt_partial,
                ddt_bias,
                NP,
                H=H,
                K=K,
                BNP=triton.next_power_of_2(NP),
                num_warps=4,
                num_stages=1,
            )
    domega_raw = None
    if need_omega and d_omega is not None:
        domega_raw = torch.empty_like(omega_raw)
        with torch.cuda.device(retention_raw.device):
            _frozen_frontend_diag_frontend_uncoupled_omega_local_vjp_kernel[
                (B * local_NT, H)
            ](
                omega_raw,
                d_omega,
                domega_raw,
                T,
                local_NT,
                H=H,
                K=K,
                BT=local_BT,
                BK=K,
                num_warps=4,
                num_stages=2,
            )
    dr_raw = None
    if need_r and d_r is not None:
        dr_raw = torch.empty_like(r_raw)
        with torch.cuda.device(retention_raw.device):
            _frozen_frontend_diag_frontend_r_local_vjp_kernel[(B * local_NT, H)](
                r_raw,
                d_r,
                dr_raw,
                T,
                local_NT,
                H=H,
                BT=local_BT,
                num_warps=4,
                num_stages=2,
            )
    dinitial_precision_param = None
    if need_initial and d_initial_precision is not None:
        dinitial_precision_param = torch.empty_like(initial_precision_param)
        with torch.cuda.device(retention_raw.device):
            _frozen_frontend_diag_frontend_mu_local_vjp_kernel[
                (triton.cdiv(H, local_BT),)
            ](
                initial_precision_param,
                d_initial_precision,
                dinitial_precision_param,
                H,
                BH=local_BT,
                num_warps=4,
                num_stages=2,
            )
    return (
        dretention_raw,
        domega_raw,
        dr_raw,
        dA_log,
        ddt_bias,
        dinitial_precision_param,
    )


class _DiagKdnFrontendFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        retention_raw,
        omega_raw,
        r_raw,
        A_log,
        dt_bias,
        initial_precision_param,
    ):
        outputs = _diag_kdn_frontend_forward(
            retention_raw,
            omega_raw,
            r_raw,
            A_log,
            dt_bias,
            initial_precision_param,
        )
        ctx.save_for_backward(
            retention_raw,
            omega_raw,
            r_raw,
            A_log,
            dt_bias,
            initial_precision_param,
            outputs[1],
        )
        ctx.set_materialize_grads(False)
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, d_g_cumsum, d_alpha, d_omega, d_r, d_initial_precision):
        return _diag_kdn_frontend_backward(
            *ctx.saved_tensors,
            d_g_cumsum,
            d_alpha,
            d_omega,
            d_r,
            d_initial_precision,
            ctx.needs_input_grad,
        )


def _diag_kdn_frontend_precumsum_paired(
    retention_raw: torch.Tensor,
    omega_raw: torch.Tensor,
    r_raw: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_precision_param: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the fixed uncoupled omega=0/r=0.01 BT32 frontend."""
    _validate_diag_kdn_frontend_inputs(
        retention_raw,
        omega_raw,
        r_raw,
        A_log,
        dt_bias,
        initial_precision_param,
    )
    return _DiagKdnFrontendFunction.apply(
        retention_raw,
        omega_raw,
        r_raw,
        A_log,
        dt_bias,
        initial_precision_param,
    )


@torch.compiler.disable
def _diag_kdn_gain_forward(
    k: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    initial_precision: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    BT = _DIAG_KDN_GAIN_CHUNK_SIZE
    NT = (T + BT - 1) // BT
    BK = triton.next_power_of_2(K)
    kappa = torch.empty_like(k, dtype=_DIAG_KDN_STATE_DTYPE)
    chunk_maps = torch.empty(
        (B, NT, H, K, 4), dtype=_DIAG_KDN_STATE_DTYPE, device=k.device
    )
    carry = torch.empty(
        (B, NT, H, K, 2), dtype=_DIAG_KDN_STATE_DTYPE, device=k.device
    )
    grid = (NT, B * H)
    _frozen_gain_kla_kappa_passA_kernel[grid](
        k,
        alpha,
        omega,
        r,
        chunk_maps,
        T,
        float(K),
        k,
        1.0,
        NT=NT,
        H=H,
        K=K,
        BK=BK,
        BT=BT,
        USE_LOG_SCALE=False,
    )
    _frozen_gain_kla_kappa_passB_kernel[(B * H,)](
        chunk_maps,
        initial_precision,
        carry,
        NT=NT,
        H=H,
        K=K,
        BK=BK,
    )
    _frozen_gain_kla_kappa_passC_kappa_only_kernel[grid](
        k,
        alpha,
        omega,
        r,
        carry,
        kappa,
        kappa,
        kappa,
        T,
        float(K),
        k,
        1.0,
        NT=NT,
        H=H,
        K=K,
        BK=BK,
        BT=BT,
        SAVE_STATE=False,
        USE_LOG_SCALE=False,
    )
    return kappa, carry


@torch.compiler.disable
def _diag_kdn_gain_backward(
    k: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    initial_precision: torch.Tensor,
    omega_raw: torch.Tensor,
    r_raw: torch.Tensor,
    dkappa: torch.Tensor | None,
    carry: torch.Tensor,
    *,
    need_initial_precision: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K = k.shape
    BT = _DIAG_KDN_GAIN_CHUNK_SIZE
    NT = (T + BT - 1) // BT
    BK = triton.next_power_of_2(K)
    if carry.shape != (B, NT, H, K, 2):
        raise RuntimeError("saved DiagKDN gain carry shape changed")
    if carry.dtype != _DIAG_KDN_STATE_DTYPE or not carry.is_contiguous():
        raise TypeError("saved DiagKDN gain carry must be contiguous FP32")
    dkappa = torch.zeros_like(k) if dkappa is None else dkappa.contiguous()
    dk = torch.empty_like(k, dtype=_DIAG_KDN_STATE_DTYPE)
    dalpha = torch.empty_like(alpha, dtype=_DIAG_KDN_STATE_DTYPE)
    domega_raw = torch.empty_like(omega_raw)
    dr_raw = torch.empty_like(r_raw)
    dinitial_precision = (
        torch.zeros((H,), dtype=_DIAG_KDN_STATE_DTYPE, device=k.device)
        if need_initial_precision
        else None
    )
    p_exclusive = torch.empty_like(k, dtype=_DIAG_KDN_STATE_DTYPE)
    reverse_maps = torch.empty(
        (B, NT, H, K, 2), dtype=_DIAG_KDN_STATE_DTYPE, device=k.device
    )
    reverse_carry = torch.empty(
        (B, NT, H, K), dtype=_DIAG_KDN_STATE_DTYPE, device=k.device
    )
    grid = (NT, B * H)
    _frozen_gain_kla_kappa_bwd_fill_kernel[grid](
        k,
        alpha,
        omega,
        r,
        carry,
        p_exclusive,
        T,
        float(K),
        k,
        1.0,
        NT=NT,
        H=H,
        K=K,
        BK=BK,
        BT=BT,
        USE_LOG_SCALE=False,
    )
    _frozen_gain_kla_kappa_bwd_passA_kappa_only_kernel[grid](
        k,
        alpha,
        omega,
        r,
        dkappa,
        p_exclusive,
        reverse_maps,
        T,
        float(K),
        k,
        1.0,
        NT=NT,
        H=H,
        K=K,
        BK=BK,
        BT=BT,
        USE_LOG_SCALE=False,
    )
    _frozen_gain_kla_kappa_bwd_passB_kernel[(B * H,)](
        reverse_maps,
        reverse_carry,
        NT=NT,
        H=H,
        K=K,
        BK=BK,
    )
    _frozen_gain_kla_kappa_bwd_passC_kappa_only_kernel[grid](
        k,
        alpha,
        omega,
        r,
        dkappa,
        p_exclusive,
        reverse_carry,
        omega_raw,
        r_raw,
        dk,
        dalpha,
        domega_raw,
        dr_raw,
        dinitial_precision if dinitial_precision is not None else initial_precision,
        initial_precision,
        T,
        float(K),
        k,
        1.0,
        NT=NT,
        H=H,
        K=K,
        BK=BK,
        BT=BT,
        RAW_ACTIVATION_VJP=True,
        USE_LOG_SCALE=False,
        NEED_DMU=need_initial_precision,
        NEED_DLOG=False,
    )
    return dk, dalpha, domega_raw, dr_raw, dinitial_precision


class _DiagKdnGainFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, alpha, omega, r, initial_precision, omega_raw, r_raw):
        kappa, carry = _diag_kdn_gain_forward(
            k, alpha, omega, r, initial_precision
        )
        ctx.save_for_backward(
            k, alpha, omega, r, initial_precision, omega_raw, r_raw, carry
        )
        ctx.set_materialize_grads(False)
        return kappa

    @staticmethod
    @once_differentiable
    def backward(ctx, dkappa):
        (
            k,
            alpha,
            omega,
            r,
            initial_precision,
            omega_raw,
            r_raw,
            carry,
        ) = ctx.saved_tensors
        needs = ctx.needs_input_grad
        dk, dalpha, domega_raw, dr_raw, dinitial_precision = (
            _diag_kdn_gain_backward(
                k,
                alpha,
                omega,
                r,
                initial_precision,
                omega_raw,
                r_raw,
                dkappa,
                carry,
                need_initial_precision=needs[4],
            )
        )
        return (
            dk if needs[0] else None,
            dalpha if needs[1] else None,
            None,
            None,
            dinitial_precision if needs[4] else None,
            domega_raw if needs[5] else None,
            dr_raw if needs[6] else None,
        )


def _diag_kdn_gain(
    k_gain: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    initial_precision: torch.Tensor,
    omega_raw: torch.Tensor,
    r_raw: torch.Tensor,
) -> torch.Tensor:
    """Return the fixed-scale FP32 diagonal gain with fused raw-logit VJPs."""
    tensors = {
        "k_gain": k_gain,
        "alpha": alpha,
        "omega": omega,
        "r": r,
        "initial_precision": initial_precision,
        "omega_raw": omega_raw,
        "r_raw": r_raw,
    }
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not value.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if value.device != k_gain.device:
            raise ValueError("all DiagKDN gain inputs must share one CUDA device")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if k_gain.ndim != 4:
        raise ValueError("k_gain must have rank 4 [B,T,H,K]")
    B, T, H, K = k_gain.shape
    if min(B, T, H) <= 0 or K not in (64, 128):
        raise ValueError("DiagKDN gain requires non-empty B/T/H and K in {64,128}")
    expected = (B, T, H, K)
    for name in ("alpha", "omega", "omega_raw"):
        if tensors[name].shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if r.shape != (B, T, H) or r_raw.shape != (B, T, H):
        raise ValueError("r and r_raw must have shape [B,T,H]")
    if initial_precision.shape != (H,):
        raise ValueError(f"initial_precision must have shape [{H}]")
    for name in ("k_gain", "alpha", "omega", "r", "initial_precision"):
        if tensors[name].dtype != torch.float32:
            raise TypeError(f"{name} must be torch.float32")
    if omega_raw.dtype != _DIAG_KDN_VECTOR_DTYPE or r_raw.dtype != _DIAG_KDN_VECTOR_DTYPE:
        raise TypeError("omega_raw and r_raw must be torch.bfloat16")
    return _DiagKdnGainFunction.apply(
        k_gain,
        alpha,
        omega.detach(),
        r.detach(),
        initial_precision,
        omega_raw,
        r_raw,
    )


def _validate_diag_kdn_memory_inputs(q, k, kappa, v, g_cumsum, scale):
    tensors = {"q": q, "k": k, "kappa": kappa, "v": v, "g_cumsum": g_cumsum}
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 4:
            raise ValueError(f"{name} must have rank 4 [B,T,H,D]")
        if not value.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if value.device != q.device:
            raise ValueError("all DiagKDN memory inputs must share one CUDA device")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    B, T, H, K = q.shape
    if min(B, T, H) <= 0 or K not in (64, 128):
        raise ValueError("DiagKDN memory requires non-empty B/T/H and K in {64,128}")
    expected = (B, T, H, K)
    for name in ("k", "kappa", "v", "g_cumsum"):
        if tensors[name].shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    for name in ("q", "k", "v"):
        if tensors[name].dtype != _DIAG_KDN_VECTOR_DTYPE:
            raise TypeError(f"{name} must be torch.bfloat16")
    if kappa.dtype != _DIAG_KDN_STATE_DTYPE or g_cumsum.dtype != _DIAG_KDN_STATE_DTYPE:
        raise TypeError("kappa and g_cumsum must be torch.float32")
    if type(scale) is not float or not math.isfinite(scale):
        raise TypeError("scale must be an exact finite float")
    return B, T, H, K


def _diag_kdn_recompute_forward(k, kappa, v, Abar, g_cumsum):
    """Fixed no-varlen recompute without the optional query output."""
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT, BK, BV = _DIAG_KDN_MEMORY_CHUNK_SIZE, 64, 64
    NT = triton.cdiv(T, BT)
    w = torch.empty_like(k, dtype=_DIAG_KDN_STATE_DTYPE)
    u = torch.empty_like(v, dtype=_DIAG_KDN_STATE_DTYPE)
    kappa_fed = torch.empty_like(kappa, dtype=_DIAG_KDN_STATE_DTYPE)
    _frozen_intra_recompute_w_u_fwd_kalman_kernel_explicit[(NT, B * H)](
        q=None,
        k=k,
        kappa=kappa,
        qg=None,
        kg=kappa_fed,
        v=v,
        w=w,
        u=u,
        A=Abar,
        gk=g_cumsum,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        DOT_PRECISION=_DIAG_KDN_SCORE_DOT_PRECISION,
        BF16_DOT_OPERANDS=False,
    )
    return w, u, kappa_fed


def _diag_kdn_recompute_backward(q, k, kappa, v, Abar, g_cumsum):
    """Fixed no-varlen recompute with the FP32 gated-query history."""
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT, BK, BV = _DIAG_KDN_MEMORY_CHUNK_SIZE, 64, 64
    NT = triton.cdiv(T, BT)
    w = torch.empty_like(k, dtype=_DIAG_KDN_STATE_DTYPE)
    u = torch.empty_like(v, dtype=_DIAG_KDN_STATE_DTYPE)
    kappa_fed = torch.empty_like(kappa, dtype=_DIAG_KDN_STATE_DTYPE)
    qg = torch.empty_like(q, dtype=_DIAG_KDN_INTRA_QUERY_DTYPE)
    _frozen_intra_recompute_w_u_fwd_kalman_kernel_explicit[(NT, B * H)](
        q=q,
        k=k,
        kappa=kappa,
        qg=qg,
        kg=kappa_fed,
        v=v,
        w=w,
        u=u,
        A=Abar,
        gk=g_cumsum,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        DOT_PRECISION=_DIAG_KDN_SCORE_DOT_PRECISION,
        BF16_DOT_OPERANDS=False,
    )
    return w, u, kappa_fed, qg


def _diag_kdn_intra_forward(q, k, kappa, v, g_cumsum, scale):
    """Build the fixed BT32 FP32 Aqk/Abar caches and WY auxiliaries."""
    B, T, H, K = k.shape
    BT, BC = _DIAG_KDN_MEMORY_CHUNK_SIZE, 16
    NT, NC = triton.cdiv(T, BT), 2
    Aqk = torch.zeros(B, T, H, BT, device=k.device, dtype=_DIAG_KDN_STATE_DTYPE)
    Abar = torch.zeros(B, T, H, BT, device=k.device, dtype=_DIAG_KDN_STATE_DTYPE)
    diagonal = torch.empty(B, T, H, BC, device=k.device, dtype=_DIAG_KDN_STATE_DTYPE)

    def diagonal_grid(meta):
        return B * T, triton.cdiv(H, meta["BH"])

    _frozen_intra_chunk_kalman_fwd_kernel_intra_token_parallel[diagonal_grid](
        q=q,
        k=k,
        kappa=kappa,
        g=g_cumsum,
        Aqk=Aqk,
        Akk=diagonal,
        scale=scale,
        cu_seqlens=None,
        N=B,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
    )
    _frozen_intra_chunk_kalman_fwd_kernel_inter_solve_fused_bt32_explicit[
        (NT, B * H)
    ](
        q=q,
        k=k,
        kappa=kappa,
        g=g_cumsum,
        Aqk=Aqk,
        Akkd=diagonal,
        Akk=Abar,
        Mraw=None,
        scale=scale,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
        SCORE_DOT_PRECISION=_DIAG_KDN_SCORE_DOT_PRECISION,
        SOLVE_DOT_PRECISION=_DIAG_KDN_SOLVE_DOT_PRECISION,
    )
    w, u, kappa_fed = _diag_kdn_recompute_forward(
        k, kappa, v, Abar, g_cumsum
    )
    return w, u, kappa_fed, Aqk, Abar


def _diag_kdn_state_forward(kappa_fed, w, u, g_cumsum):
    """Run only the fixed no-state-input/no-final-state BT32 recurrence."""
    B, T, Hq, K = kappa_fed.shape
    H, V = u.shape[2], u.shape[-1]
    BT = _DIAG_KDN_MEMORY_CHUNK_SIZE
    NT = triton.cdiv(T, BT)
    h = torch.empty((B, NT, H, K, V), device=u.device, dtype=_DIAG_KDN_STATE_DTYPE)
    v_new = torch.empty_like(u, dtype=_DIAG_KDN_STATE_DTYPE)

    def grid(meta):
        return triton.cdiv(V, meta["BV"]), B * H

    _frozen_kda_chunk_gated_delta_rule_fwd_kernel_h_blockdim64_diag[grid](
        k=kappa_fed,
        v=u,
        w=w,
        v_new=v_new,
        g=None,
        gk=g_cumsum,
        h=h,
        h0=None,
        ht=None,
        cu_seqlens=None,
        chunk_offsets=None,
        T=T,
        H=H,
        Hq=Hq,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=True,
        TRANSPOSE_STATE=False,
        DOT_PRECISION=_DIAG_KDN_STATE_DOT_PRECISION,
        BF16_DOT_OPERANDS=True,
        RESIDUAL_DOT_PRECISION=_DIAG_KDN_STATE_RESIDUAL_DOT_PRECISION,
        RESIDUAL_BF16_DOT_OPERANDS=False,
    )
    return h, v_new


def _diag_kdn_readout(q, v_new, g_cumsum, Aqk, h, scale):
    """Run only the fixed FBT no-varlen readout."""
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    BT = _DIAG_KDN_MEMORY_CHUNK_SIZE
    NT = triton.cdiv(T, BT)
    output = torch.zeros(v_new.shape, device=v_new.device, dtype=_DIAG_KDN_STATE_DTYPE)

    def grid(meta):
        return triton.cdiv(V, meta["BV"]), NT, B * H

    _frozen_readout_chunk_gla_fwd_kernel_o_diag[grid](
        q=q,
        v=v_new,
        g=g_cumsum,
        h=h,
        o=output,
        A=Aqk,
        cu_seqlens=None,
        chunk_indices=None,
        scale=scale,
        T=T,
        H=H,
        HV=H,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=True,
        TRANSPOSE_STATE=False,
        DOT_PRECISION="tf32",
        BF16_INTER_DOT_OPERANDS=False,
        BF16_INTRA_DOT_OPERANDS=False,
    )
    return output


def _diag_kdn_output_backward(q, kappa_fed, v_new, do, Aqk, scale):
    """Run the fixed BQ dAqk/dv_new readout VJP."""
    B, T, H, K = kappa_fed.shape
    V = do.shape[-1]
    BT = _DIAG_KDN_MEMORY_CHUNK_SIZE
    BK = min(max(triton.next_power_of_2(K), 16), 128)
    BV = min(max(triton.next_power_of_2(V), 16), 128)
    NT = triton.cdiv(T, BT)
    dAqk = torch.empty((B, T, H, BT), device=q.device, dtype=_DIAG_KDN_STATE_DTYPE)
    dv_new = torch.empty_like(do, dtype=_DIAG_KDN_STATE_DTYPE)
    _frozen_kda_chunk_kda_bwd_kernel_dAv_diag[(NT, B * H)](
        q=q,
        k=kappa_fed,
        v=v_new,
        A=Aqk,
        do=do,
        dv=dv_new,
        dA=dAqk,
        cu_seqlens=None,
        chunk_indices=None,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        DOT_PRECISION=_DIAG_KDN_BACKWARD_DOT_PRECISION,
        BF16_DOT_OPERANDS=True,
        BF16_DAQK_DO_VNEW_OPERANDS=False,
        DVNEW_AQK_DO_DOT_PRECISION="tf32x3",
        BF16_DVNEW_AQK_DO_OPERANDS=True,
    )
    return dAqk, dv_new


def _diag_kdn_state_backward(qg, kappa_fed, w, g_cumsum, do, dv_new, scale):
    """Run only the fixed no-state-input BT32 state VJP."""
    B, T, Hq, K = qg.shape
    H, V = do.shape[2], do.shape[-1]
    BT = _DIAG_KDN_MEMORY_CHUNK_SIZE
    NT = triton.cdiv(T, BT)
    dh = torch.empty((B, NT, H, K, V), device=qg.device, dtype=_DIAG_KDN_STATE_DTYPE)
    du = torch.empty_like(dv_new, dtype=_DIAG_KDN_STATE_DTYPE)

    def grid(meta):
        return triton.cdiv(V, meta["BV"]), B * H

    _frozen_kda_chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64_diag[grid](
        q=qg,
        k=kappa_fed,
        w=w,
        g=None,
        gk=g_cumsum,
        dht=None,
        dh0=None,
        do=do,
        dh=dh,
        dv=dv_new,
        dv2=du,
        cu_seqlens=None,
        chunk_offsets=None,
        scale=scale,
        T=T,
        H=H,
        Hq=Hq,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=True,
        TRANSPOSE_STATE=False,
        DOT_PRECISION=_DIAG_KDN_STATE_DOT_PRECISION,
        BF16_QG_DO_OPERANDS=False,
        BF16_W_DV_OPERANDS=True,
    )
    return dh, du


def _diag_kdn_wy_backward(q, k, kappa, v, v_new, g, Abar, h, do, dh, du, scale):
    """Run the sole fixed buffered-WY BQ backward."""
    B, T, H, K = k.shape
    V = v.shape[-1]
    dq = torch.empty_like(q, dtype=_DIAG_KDN_STATE_DTYPE)
    dk = torch.empty_like(k, dtype=_DIAG_KDN_STATE_DTYPE)
    dkappa = torch.empty_like(kappa, dtype=_DIAG_KDN_STATE_DTYPE)
    dv = torch.empty_like(v, dtype=_DIAG_KDN_FINAL_GRADIENT_DTYPE)
    dg = torch.empty_like(g, dtype=_DIAG_KDN_STATE_DTYPE)
    dM = torch.empty_like(Abar, dtype=_DIAG_KDN_STATE_DTYPE)
    dw = torch.empty_like(k, dtype=_DIAG_KDN_STATE_DTYPE)
    grid = (triton.cdiv(T, _DIAG_KDN_MEMORY_CHUNK_SIZE), B * H)
    _frozen_wy_chunk_kalman_bwd_kernel_wy_buffered_dw_producer[grid](
        q,
        k,
        kappa,
        v_new,
        g,
        Abar,
        h,
        do,
        dh,
        du,
        dq,
        dk,
        dkappa,
        dv,
        dg,
        dw,
        scale,
        T,
        H,
        K,
        V,
        _DIAG_KDN_MEMORY_CHUNK_SIZE,
        64,
        64,
        _DIAG_KDN_BACKWARD_DOT_PRECISION,
        "tf32",
        "tf32",
        True,
        True,
        True,
        False,
        num_warps=4,
        num_stages=4,
        num_ctas=1,
    )
    _frozen_wy_chunk_kalman_bwd_kernel_wy_buffered_dw_consumer[grid](
        k,
        v,
        g,
        Abar,
        du,
        dw,
        dM,
        T,
        H,
        K,
        V,
        _DIAG_KDN_MEMORY_CHUNK_SIZE,
        64,
        64,
        "tf32",
        num_warps=4,
        num_stages=4,
        num_ctas=1,
    )
    return dq, dk, dkappa, dv, dg, dM


def _diag_kdn_intra_backward(q, k, kappa, g, dAqk, dM, dq, dk, dkappa, dg):
    """Run only the fixed no-varlen Triton intra VJP."""
    B, T, H, K = k.shape
    BT, BC, BK = _DIAG_KDN_MEMORY_CHUNK_SIZE, 16, 32
    NT, NC, NK = triton.cdiv(T, BT), 2, triton.cdiv(K, BK)
    dq_out = torch.empty_like(q, dtype=_DIAG_KDN_FINAL_GRADIENT_DTYPE)
    dk_out = torch.empty_like(k, dtype=_DIAG_KDN_FINAL_GRADIENT_DTYPE)
    dkappa_out = torch.empty_like(kappa, dtype=_DIAG_KDN_STATE_DTYPE)
    dg_out = torch.empty_like(dg, dtype=_DIAG_KDN_STATE_DTYPE)
    _frozen_intra_chunk_kalman_bwd_kernel_intra_explicit[
        (NK * NC, NT, B * H)
    ](
        q=q,
        k=k,
        kappa=kappa,
        g=g,
        dAqk=dAqk,
        dAkk=dM,
        dq=dq,
        dq2=dq_out,
        dk=dk,
        dk2=dk_out,
        dkappa=dkappa,
        dkappa2=dkappa_out,
        dg=dg,
        dg2=dg_out,
        cu_seqlens=None,
        chunk_indices=None,
        B=B,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
        SAFE_GATE=False,
        USE_GATHER=IS_GATHER_SUPPORTED,
        DOT_PRECISION=_DIAG_KDN_BACKWARD_DOT_PRECISION,
        BF16_DOT_OPERANDS=True,
        BF16_DK_ROW_OPERANDS=True,
        BF16_DQ_ROW_OPERANDS=False,
        DQ_ROW_DOT_PRECISION="tf32x3",
        K_TILES_PER_CTA=1,
    )
    return dq_out, dk_out, dkappa_out, dg_out


def _diag_kdn_memory_forward(q, k, kappa, v, g_cumsum, scale):
    B, T, H, K = _validate_diag_kdn_memory_inputs(
        q, k, kappa, v, g_cumsum, scale
    )
    with torch.cuda.device(q.device):
        w, u, kappa_fed, Aqk, Abar = _diag_kdn_intra_forward(
            q, k, kappa, v, g_cumsum, scale
        )
        BT = _DIAG_KDN_MEMORY_CHUNK_SIZE
        for name, value in (("Aqk", Aqk), ("Abar", Abar)):
            if value.shape != (B, T, H, BT):
                raise RuntimeError(f"saved {name} shape changed")
            if value.dtype != torch.float32 or not value.is_contiguous():
                raise TypeError(f"saved {name} must be contiguous FP32")
        h, v_new = _diag_kdn_state_forward(kappa_fed, w, u, g_cumsum)
        output = _diag_kdn_readout(
            q, v_new, g_cumsum, Aqk, h, scale
        )
    if output.shape != (B, T, H, K):
        raise RuntimeError("fixed DiagKDN output shape changed")
    return output, Aqk, Abar


class _DiagKdnMemoryFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, kappa, v, g_cumsum, scale):
        output, Aqk, Abar = _diag_kdn_memory_forward(
            q, k, kappa, v, g_cumsum, scale
        )
        ctx.save_for_backward(q, k, kappa, v, g_cumsum, Aqk, Abar)
        ctx.scale = scale
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, do):
        q, k, kappa, v, g_cumsum, Aqk, Abar = ctx.saved_tensors
        if do is None:
            return None, None, None, None, None, None
        do = do.contiguous()
        B, T, H, K = _validate_diag_kdn_memory_inputs(
            q, k, kappa, v, g_cumsum, ctx.scale
        )
        if do.shape != (B, T, H, K) or do.dtype != torch.float32:
            raise TypeError("DiagKDN output cotangent must be FP32 BTHK")
        with torch.cuda.device(q.device):
            w, u, kappa_fed, qg = _diag_kdn_recompute_backward(
                q, k, kappa, v, Abar, g_cumsum
            )
            h, v_new = _diag_kdn_state_forward(
                kappa_fed, w, u, g_cumsum
            )
            do_work = do.to(torch.bfloat16)
            dAqk, dv_new = _diag_kdn_output_backward(
                q, kappa_fed, v_new, do_work, Aqk, ctx.scale
            )
            dAqk = dAqk.to(torch.float32).contiguous()
            dv_new = dv_new.to(torch.float32)
            dh, du = _diag_kdn_state_backward(
                qg, kappa_fed, w, g_cumsum, do_work, dv_new, ctx.scale
            )
            dq_w, dk_read, dkappa, dv, dg_w, dM = _diag_kdn_wy_backward(
                q,
                k,
                kappa,
                v,
                v_new,
                g_cumsum,
                Abar,
                h,
                do_work,
                dh,
                du,
                ctx.scale,
            )
            dq, dk_read, dkappa, dG = _diag_kdn_intra_backward(
                q,
                k,
                kappa,
                g_cumsum,
                dAqk,
                dM,
                dq_w,
                dk_read,
                dkappa,
                dg_w,
            )
        return dq, dk_read, dkappa, dv, dG.contiguous(), None


def chunk_kalman(
    q_memory: torch.Tensor,
    k_memory: torch.Tensor,
    kappa: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run the sole fixed BT32 production DiagKDN memory route."""
    _validate_diag_kdn_memory_inputs(
        q_memory, k_memory, kappa, v, g_cumsum, scale
    )
    return _DiagKdnMemoryFunction.apply(
        q_memory, k_memory, kappa, v, g_cumsum, scale
    )


__all__ = ("chunk_kalman",)
