"""Raw checkpointed recurrent-memory Triton kernels for value-wise KDN.

This module stays below the operator's single autograd boundary.  A
program owns one ``(batch, head, eight-value tile)`` and advances a ``K x 8``
physical-memory tile in FP32.  The forward emits post-write reads and sparse
64-token checkpoints.  Reverse support is added below the same raw ABI; no
class in this module subclasses :class:`torch.autograd.Function`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch

from ._value_iso_gain import (
    GAIN_CHUNK_SIZE,
    GAIN_COPY_BLOCK_SIZE,
    GAIN_NUM_STAGES,
    GAIN_NUM_WARPS,
    GainForwardPlan,
    GainReversePlan,
    RawGainBackwardResult,
    RawGainForwardResult,
    gain_forward_plan,
    gain_reverse_plan,
    raw_value_iso_gain_backward,
    raw_value_iso_gain_forward,
)

try:  # Host metadata/tests must remain importable without Triton or CUDA.
    import triton
    import triton.language as tl
except (ImportError, OSError) as error:  # pragma: no cover - environment specific
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR: BaseException | None = error
else:
    _TRITON_IMPORT_ERROR = None


MEMORY_CHUNK_SIZE = 64
MEMORY_BLOCK_VALUE = 8
MEMORY_NUM_WARPS = 4
MEMORY_NUM_STAGES = 1
SUPPORTED_KEY_DIMS = frozenset({64, 128})
MAX_VALUE_DIM = 256


@dataclass(frozen=True)
class MemoryForwardPlan:
    """Pure host metadata for one raw recurrent-memory forward."""

    batch_size: int
    total_tokens: int
    heads: int
    key_dim: int
    value_dim: int
    chunk_size: int
    block_value: int
    value_tiles: int
    chunk_count: int
    boundary_indices: tuple[int, ...]
    segment_lengths: tuple[int, ...]
    grid: tuple[int, int, int]
    output_shape: tuple[int, int, int, int]
    checkpoints_shape: tuple[int, int, int, int, int]
    checkpoints_nbytes: int
    final_memory_shape: tuple[int, int, int, int]
    forward_launch_count: int
    num_warps: int
    num_stages: int
    token_loop_is_runtime: bool


@dataclass(frozen=True)
class RawMemoryForwardResult:
    """Outputs and saved state produced by the raw forward helper."""

    output: torch.Tensor
    checkpoints: torch.Tensor
    final_memory: torch.Tensor | None
    plan: MemoryForwardPlan


@dataclass(frozen=True)
class MemoryBackwardPlan:
    """Pure host metadata for exact checkpoint replay and reverse."""

    batch_size: int
    total_tokens: int
    heads: int
    key_dim: int
    value_dim: int
    chunk_size: int
    block_value: int
    value_tiles: int
    segment_starts: tuple[int, ...]
    segment_lengths: tuple[int, ...]
    checkpoint_indices: tuple[int, ...]
    grid: tuple[int, int, int]
    replay_shape: tuple[int, int, int, int, int]
    replay_nbytes: int
    carry_shape: tuple[int, int, int, int]
    carry_nbytes: int
    segment_launch_count: int
    initialization_launch_count: int
    total_launch_count: int
    programs_per_segment_launch: int
    total_segment_programs: int
    logical_atomic_count: int
    initialize_outputs: bool
    write_dq: bool
    write_dk: bool
    write_dalpha: bool
    write_dv: bool
    write_dbeta: bool


@dataclass(frozen=True)
class RawMemoryBackwardResult:
    """Caller-owned destinations updated by the raw memory reverse."""

    dq_acc: torch.Tensor | None
    dk_acc: torch.Tensor | None
    dalpha: torch.Tensor | None
    dv: torch.Tensor | None
    dbeta_domega: torch.Tensor | None
    dstate_carry: torch.Tensor
    dinitial_memory: torch.Tensor | None
    plan: MemoryBackwardPlan


@dataclass(frozen=True)
class _ValidatedForwardInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    initial_memory: torch.Tensor | None
    read_scale: float
    output_final_state: bool
    plan: MemoryForwardPlan


@dataclass(frozen=True)
class _ValidatedBackwardInputs:
    forward: _ValidatedForwardInputs
    output_cotangent: torch.Tensor
    checkpoints: torch.Tensor
    replay_buffer: torch.Tensor
    dstate_carry: torch.Tensor
    dq_acc: torch.Tensor | None
    dk_acc: torch.Tensor | None
    dalpha: torch.Tensor | None
    dv: torch.Tensor | None
    dbeta_domega: torch.Tensor | None
    final_memory_cotangent: torch.Tensor | None
    dinitial_memory: torch.Tensor | None
    initialize_outputs: bool
    plan: MemoryBackwardPlan


def _exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be exactly int")
    return value


def memory_forward_plan(
    *,
    batch_size: int,
    total_tokens: int,
    heads: int,
    key_dim: int,
    value_dim: int,
) -> MemoryForwardPlan:
    """Return the exact selected-BV forward geometry without touching CUDA."""

    batch_size = _exact_int(batch_size, "batch_size")
    total_tokens = _exact_int(total_tokens, "total_tokens")
    heads = _exact_int(heads, "heads")
    key_dim = _exact_int(key_dim, "key_dim")
    value_dim = _exact_int(value_dim, "value_dim")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if total_tokens < 0:
        raise ValueError("total_tokens must be nonnegative")
    if heads <= 0:
        raise ValueError("heads must be positive")
    if key_dim not in SUPPORTED_KEY_DIMS:
        raise ValueError("key_dim must be exactly 64 or 128")
    if value_dim <= 0:
        raise ValueError("value_dim must be positive")
    if value_dim > MAX_VALUE_DIM:
        raise ValueError(f"value_dim must be at most {MAX_VALUE_DIM}")

    chunk_count = (total_tokens + MEMORY_CHUNK_SIZE - 1) // MEMORY_CHUNK_SIZE
    boundary_indices = tuple(
        min(chunk * MEMORY_CHUNK_SIZE, total_tokens)
        for chunk in range(chunk_count + 1)
    )
    segment_lengths = tuple(
        boundary_indices[index + 1] - boundary_indices[index]
        for index in range(chunk_count)
    )
    value_tiles = (value_dim + MEMORY_BLOCK_VALUE - 1) // MEMORY_BLOCK_VALUE
    return MemoryForwardPlan(
        batch_size=batch_size,
        total_tokens=total_tokens,
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        chunk_size=MEMORY_CHUNK_SIZE,
        block_value=MEMORY_BLOCK_VALUE,
        value_tiles=value_tiles,
        chunk_count=chunk_count,
        boundary_indices=boundary_indices,
        segment_lengths=segment_lengths,
        grid=(batch_size, heads, value_tiles),
        output_shape=(batch_size, total_tokens, heads, value_dim),
        checkpoints_shape=(
            chunk_count + 1,
            batch_size,
            heads,
            key_dim,
            value_dim,
        ),
        checkpoints_nbytes=(
            (chunk_count + 1)
            * batch_size
            * heads
            * key_dim
            * value_dim
            * 4
        ),
        final_memory_shape=(batch_size, heads, key_dim, value_dim),
        # The launch also materializes checkpoint zero for an empty sequence.
        forward_launch_count=1,
        num_warps=MEMORY_NUM_WARPS,
        num_stages=MEMORY_NUM_STAGES,
        token_loop_is_runtime=True,
    )


def memory_backward_plan(
    *,
    batch_size: int,
    total_tokens: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    initialize_outputs: bool = True,
    write_dq: bool = True,
    write_dk: bool = True,
    write_dalpha: bool = True,
    write_dv: bool = True,
    write_dbeta: bool = True,
) -> MemoryBackwardPlan:
    """Return exact reverse ordering, storage, and atomic-count metadata."""

    forward = memory_forward_plan(
        batch_size=batch_size,
        total_tokens=total_tokens,
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
    )
    if type(initialize_outputs) is not bool:
        raise TypeError("initialize_outputs must be exactly bool")
    for name, value in (
        ("write_dq", write_dq),
        ("write_dk", write_dk),
        ("write_dalpha", write_dalpha),
        ("write_dv", write_dv),
        ("write_dbeta", write_dbeta),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be exactly bool")
    starts = tuple(reversed(tuple(range(0, total_tokens, MEMORY_CHUNK_SIZE))))
    lengths = tuple(min(MEMORY_CHUNK_SIZE, total_tokens - start) for start in starts)
    checkpoint_indices = tuple(start // MEMORY_CHUNK_SIZE for start in starts)
    programs = batch_size * heads * forward.value_tiles
    segment_launches = len(starts)
    initialization_launches = (
        1 + int(write_dq) + int(write_dk) + int(write_dalpha)
        if initialize_outputs
        else 0
    )
    return MemoryBackwardPlan(
        batch_size=batch_size,
        total_tokens=total_tokens,
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        chunk_size=MEMORY_CHUNK_SIZE,
        block_value=MEMORY_BLOCK_VALUE,
        value_tiles=forward.value_tiles,
        segment_starts=starts,
        segment_lengths=lengths,
        checkpoint_indices=checkpoint_indices,
        grid=forward.grid,
        replay_shape=(
            MEMORY_CHUNK_SIZE + 1,
            batch_size,
            heads,
            key_dim,
            value_dim,
        ),
        replay_nbytes=(
            (MEMORY_CHUNK_SIZE + 1)
            * batch_size
            * heads
            * key_dim
            * value_dim
            * 4
        ),
        carry_shape=forward.final_memory_shape,
        carry_nbytes=batch_size * heads * key_dim * value_dim * 4,
        segment_launch_count=segment_launches,
        initialization_launch_count=initialization_launches,
        total_launch_count=segment_launches + initialization_launches,
        programs_per_segment_launch=programs,
        total_segment_programs=programs * segment_launches,
        logical_atomic_count=(
            (int(write_dq) + int(write_dk) + int(write_dalpha))
            * batch_size
            * total_tokens
            * heads
            * key_dim
            * forward.value_tiles
        ),
        initialize_outputs=initialize_outputs,
        write_dq=write_dq,
        write_dk=write_dk,
        write_dalpha=write_dalpha,
        write_dv=write_dv,
        write_dbeta=write_dbeta,
    )


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _require_contiguous(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_forward_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_memory: torch.Tensor | None,
    read_scale: float | None,
    output_final_state: bool,
    *,
    require_cuda: bool = True,
) -> _ValidatedForwardInputs:
    q = _require_tensor(q, "q")
    k = _require_tensor(k, "k")
    v = _require_tensor(v, "v")
    alpha = _require_tensor(alpha, "alpha")
    beta = _require_tensor(beta, "beta")
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B,T,H,K], got {tuple(q.shape)}")
    B, T, H, K = q.shape
    if v.ndim != 4 or tuple(v.shape[:3]) != (B, T, H):
        raise ValueError(f"v must have shape [B,T,H,V], got {tuple(v.shape)}")
    V = v.shape[-1]
    plan = memory_forward_plan(
        batch_size=B,
        total_tokens=T,
        heads=H,
        key_dim=K,
        value_dim=V,
    )
    expected_shared = (B, T, H, K)
    expected_value = (B, T, H, V)
    for name, tensor, shape in (
        ("k", k, expected_shared),
        ("alpha", alpha, expected_shared),
        ("beta", beta, expected_value),
    ):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")

    if q.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("q, k, and v must all use torch.bfloat16 or torch.float32")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, and v must all have the same BF16 or FP32 dtype")
    if alpha.dtype != torch.float32:
        raise TypeError(f"alpha must have dtype torch.float32, got {alpha.dtype}")
    if beta.dtype != torch.float32:
        raise TypeError(f"beta must have dtype torch.float32, got {beta.dtype}")

    if initial_memory is not None:
        initial_memory = _require_tensor(initial_memory, "initial_memory")
        if tuple(initial_memory.shape) != plan.final_memory_shape:
            raise ValueError(
                "initial_memory must have shape "
                f"{plan.final_memory_shape}, got {tuple(initial_memory.shape)}"
            )
        if initial_memory.dtype != torch.float32:
            raise TypeError(
                "initial_memory must have dtype torch.float32, "
                f"got {initial_memory.dtype}"
            )

    named = [("q", q), ("k", k), ("v", v), ("alpha", alpha), ("beta", beta)]
    if initial_memory is not None:
        named.append(("initial_memory", initial_memory))
    for name, tensor in named:
        _require_contiguous(tensor, name)
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on device {q.device}, got {tensor.device}")

    if read_scale is None:
        resolved_read_scale = K**-0.5
    else:
        if isinstance(read_scale, bool) or type(read_scale) not in {int, float}:
            raise TypeError("read_scale must be a finite Python int or float")
        resolved_read_scale = float(read_scale)
        if not math.isfinite(resolved_read_scale):
            raise ValueError("read_scale must be finite")
    if type(output_final_state) is not bool:
        raise TypeError("output_final_state must be exactly bool")
    if type(require_cuda) is not bool:
        raise TypeError("require_cuda must be exactly bool")
    if require_cuda and q.device.type != "cuda":
        raise ValueError(f"raw Triton memory forward requires CUDA tensors, got device={q.device}")
    return _ValidatedForwardInputs(
        q=q,
        k=k,
        v=v,
        alpha=alpha,
        beta=beta,
        initial_memory=initial_memory,
        read_scale=resolved_read_scale,
        output_final_state=output_final_state,
        plan=plan,
    )


def _storage_byte_range(tensor: torch.Tensor) -> tuple[int, int] | None:
    """Return a contiguous tensor's occupied byte interval, or none if empty."""

    if tensor.numel() == 0:
        return None
    first = (
        tensor.untyped_storage().data_ptr()
        + tensor.storage_offset() * tensor.element_size()
    )
    return first, first + tensor.numel() * tensor.element_size()


def _ranges_overlap(
    left: tuple[int, int] | None,
    right: tuple[int, int] | None,
) -> bool:
    return (
        left is not None
        and right is not None
        and left[0] < right[1]
        and right[0] < left[1]
    )


def _reject_backward_aliases(
    read_only: tuple[tuple[str, torch.Tensor], ...],
    writable: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    read_ranges = tuple(
        (name, _storage_byte_range(tensor)) for name, tensor in read_only
    )
    write_ranges = tuple(
        (name, _storage_byte_range(tensor)) for name, tensor in writable
    )
    for index, (write_name, write_range) in enumerate(write_ranges):
        for other_name, other_range in write_ranges[index + 1 :]:
            if _ranges_overlap(write_range, other_range):
                raise ValueError(f"{write_name} and {other_name} overlap in storage")
        for read_name, read_range in read_ranges:
            if _ranges_overlap(write_range, read_range):
                raise ValueError(f"{write_name} and {read_name} overlap in storage")


def _validate_backward_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_cotangent: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    checkpoints: torch.Tensor,
    replay_buffer: torch.Tensor,
    dstate_carry: torch.Tensor,
    dq_acc: torch.Tensor | None,
    dk_acc: torch.Tensor | None,
    dalpha: torch.Tensor | None,
    dv: torch.Tensor | None,
    dbeta_domega: torch.Tensor | None,
    final_memory_cotangent: torch.Tensor | None,
    dinitial_memory: torch.Tensor | None,
    read_scale: float | None,
    initialize_outputs: bool,
    *,
    require_cuda: bool = True,
) -> _ValidatedBackwardInputs:
    forward = _validate_forward_inputs(
        q,
        k,
        v,
        alpha,
        beta,
        None,
        read_scale,
        False,
        require_cuda=False,
    )
    plan = memory_backward_plan(
        batch_size=forward.plan.batch_size,
        total_tokens=forward.plan.total_tokens,
        heads=forward.plan.heads,
        key_dim=forward.plan.key_dim,
        value_dim=forward.plan.value_dim,
        initialize_outputs=initialize_outputs,
        write_dq=dq_acc is not None,
        write_dk=dk_acc is not None,
        write_dalpha=dalpha is not None,
        write_dv=dv is not None,
        write_dbeta=dbeta_domega is not None,
    )
    output_cotangent = _require_tensor(output_cotangent, "output_cotangent")
    checkpoints = _require_tensor(checkpoints, "checkpoints")
    replay_buffer = _require_tensor(replay_buffer, "replay_buffer")
    dstate_carry = _require_tensor(dstate_carry, "dstate_carry")
    optional_destinations = {
        "dq_acc": dq_acc,
        "dk_acc": dk_acc,
        "dalpha": dalpha,
        "dv": dv,
        "dbeta_domega": dbeta_domega,
    }
    for name, tensor in optional_destinations.items():
        if tensor is not None:
            optional_destinations[name] = _require_tensor(tensor, name)
    dq_acc = optional_destinations["dq_acc"]
    dk_acc = optional_destinations["dk_acc"]
    dalpha = optional_destinations["dalpha"]
    dv = optional_destinations["dv"]
    dbeta_domega = optional_destinations["dbeta_domega"]
    if final_memory_cotangent is not None:
        final_memory_cotangent = _require_tensor(
            final_memory_cotangent, "final_memory_cotangent"
        )
    if dinitial_memory is not None:
        dinitial_memory = _require_tensor(dinitial_memory, "dinitial_memory")

    expected_shapes = {
        "output_cotangent": forward.plan.output_shape,
        "checkpoints": forward.plan.checkpoints_shape,
        "replay_buffer": plan.replay_shape,
        "dstate_carry": plan.carry_shape,
        "dq_acc": tuple(q.shape),
        "dk_acc": tuple(k.shape),
        "dalpha": tuple(alpha.shape),
        "dv": tuple(v.shape),
        "dbeta_domega": tuple(beta.shape),
    }
    by_name: dict[str, torch.Tensor | None] = {
        "output_cotangent": output_cotangent,
        "checkpoints": checkpoints,
        "replay_buffer": replay_buffer,
        "dstate_carry": dstate_carry,
        "dq_acc": dq_acc,
        "dk_acc": dk_acc,
        "dalpha": dalpha,
        "dv": dv,
        "dbeta_domega": dbeta_domega,
    }
    for name, shape in expected_shapes.items():
        tensor = by_name[name]
        if tensor is not None and tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if final_memory_cotangent is not None and (
        tuple(final_memory_cotangent.shape) != plan.carry_shape
    ):
        raise ValueError(
            "final_memory_cotangent must have shape "
            f"{plan.carry_shape}, got {tuple(final_memory_cotangent.shape)}"
        )
    if dinitial_memory is not None and tuple(dinitial_memory.shape) != plan.carry_shape:
        raise ValueError(
            "dinitial_memory must have shape "
            f"{plan.carry_shape}, got {tuple(dinitial_memory.shape)}"
        )

    if output_cotangent.dtype != v.dtype:
        raise TypeError(
            f"output_cotangent must have dtype {v.dtype}, got {output_cotangent.dtype}"
        )
    if dv is not None and dv.dtype != v.dtype:
        raise TypeError(f"dv must have dtype {v.dtype}, got {dv.dtype}")
    for name in (
        "checkpoints",
        "replay_buffer",
        "dstate_carry",
        "dq_acc",
        "dk_acc",
        "dalpha",
        "dbeta_domega",
    ):
        tensor = by_name[name]
        if tensor is not None and tensor.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32")
    for name, tensor in (
        ("final_memory_cotangent", final_memory_cotangent),
        ("dinitial_memory", dinitial_memory),
    ):
        if tensor is not None and tensor.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32")

    all_named: list[tuple[str, torch.Tensor]] = [
        ("q", q),
        ("k", k),
        ("v", v),
        ("output_cotangent", output_cotangent),
        ("alpha", alpha),
        ("beta", beta),
        ("checkpoints", checkpoints),
        ("replay_buffer", replay_buffer),
        ("dstate_carry", dstate_carry),
    ]
    all_named.extend(
        (name, tensor)
        for name, tensor in optional_destinations.items()
        if tensor is not None
    )
    if final_memory_cotangent is not None:
        all_named.append(("final_memory_cotangent", final_memory_cotangent))
    if dinitial_memory is not None:
        all_named.append(("dinitial_memory", dinitial_memory))
    for name, tensor in all_named:
        _require_contiguous(tensor, name)
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on device {q.device}, got {tensor.device}")

    read_only = tuple(
        (name, tensor)
        for name, tensor in all_named
        if name
        in {
            "q",
            "k",
            "v",
            "output_cotangent",
            "alpha",
            "beta",
            "checkpoints",
            "final_memory_cotangent",
        }
    )
    writable = tuple(
        (name, tensor)
        for name, tensor in all_named
        if name
        in {
            "replay_buffer",
            "dstate_carry",
            "dq_acc",
            "dk_acc",
            "dalpha",
            "dv",
            "dbeta_domega",
            "dinitial_memory",
        }
    )
    _reject_backward_aliases(read_only, writable)
    if any(tensor.requires_grad for _, tensor in writable):
        raise ValueError("raw memory writable tensors must not require gradients")
    if not initialize_outputs and final_memory_cotangent is not None:
        raise ValueError(
            "final_memory_cotangent requires initialize_outputs=True"
        )
    if not initialize_outputs and plan.total_tokens == 0 and dinitial_memory is not None:
        raise ValueError(
            "T=0 dinitial_memory requires initialize_outputs=True"
        )
    if type(require_cuda) is not bool:
        raise TypeError("require_cuda must be exactly bool")
    if require_cuda and q.device.type != "cuda":
        raise ValueError(f"raw Triton memory backward requires CUDA tensors, got device={q.device}")
    return _ValidatedBackwardInputs(
        forward=forward,
        output_cotangent=output_cotangent,
        checkpoints=checkpoints,
        replay_buffer=replay_buffer,
        dstate_carry=dstate_carry,
        dq_acc=dq_acc,
        dk_acc=dk_acc,
        dalpha=dalpha,
        dv=dv,
        dbeta_domega=dbeta_domega,
        final_memory_cotangent=final_memory_cotangent,
        dinitial_memory=dinitial_memory,
        initialize_outputs=initialize_outputs,
        plan=plan,
    )


def _require_triton() -> None:
    if triton is None:
        detail = f": {_TRITON_IMPORT_ERROR}" if _TRITON_IMPORT_ERROR else ""
        raise RuntimeError(f"Triton is required for raw recurrent memory{detail}")


if triton is not None:

    @triton.jit(do_not_specialize=["T", "NM", "read_scale"])
    def _value_iso_memory_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        alpha_ptr,
        beta_ptr,
        initial_memory_ptr,
        output_ptr,
        checkpoints_ptr,
        final_memory_ptr,
        B,
        T,
        NM,
        read_scale,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BV: tl.constexpr,
        C: tl.constexpr,
        HAS_INITIAL: tl.constexpr,
        WRITE_FINAL: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        head_index = tl.program_id(1)
        value_tile = tl.program_id(2)
        key_offsets = tl.arange(0, K)
        key_mask = key_offsets < K
        value_offsets = value_tile * BV + tl.arange(0, BV)
        value_mask = value_offsets < V
        state_offsets = key_offsets[:, None] * V + value_offsets[None, :]
        state_mask = value_mask[None, :]
        state_base = (batch_index * H + head_index) * K * V
        if HAS_INITIAL:
            state = tl.load(
                initial_memory_ptr + state_base + state_offsets,
                mask=state_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            state = tl.zeros((K, BV), tl.float32)

        checkpoint_stride = B * H * K * V
        tl.store(
            checkpoints_ptr + state_base + state_offsets,
            state,
            mask=state_mask,
        )

        for chunk_index in tl.range(0, NM, loop_unroll_factor=1):
            chunk_start = chunk_index * C
            chunk_width = tl.minimum(C, T - chunk_start)
            for local_token in tl.range(0, C, loop_unroll_factor=1):
                active = local_token < chunk_width
                token = chunk_start + local_token
                shared_base = (
                    ((batch_index * T + token) * H + head_index) * K
                )
                value_base = (
                    ((batch_index * T + token) * H + head_index) * V
                )
                token_key_mask = active & key_mask
                token_value_mask = active & value_mask
                query = tl.load(
                    q_ptr + shared_base + key_offsets,
                    mask=token_key_mask,
                    other=0.0,
                ).to(tl.float32)
                key = tl.load(
                    k_ptr + shared_base + key_offsets,
                    mask=token_key_mask,
                    other=0.0,
                ).to(tl.float32)
                alpha_value = tl.load(
                    alpha_ptr + shared_base + key_offsets,
                    mask=token_key_mask,
                    other=1.0,
                ).to(tl.float32)
                observation = tl.load(
                    v_ptr + value_base + value_offsets,
                    mask=token_value_mask,
                    other=0.0,
                ).to(tl.float32)
                beta_value = tl.load(
                    beta_ptr + value_base + value_offsets,
                    mask=token_value_mask,
                    other=0.0,
                ).to(tl.float32)

                predicted = alpha_value[:, None] * state
                residual = observation - tl.sum(predicted * key[:, None], axis=0)
                next_state = predicted + key[:, None] * (
                    beta_value * residual
                )[None, :]
                output = read_scale * tl.sum(
                    next_state * query[:, None], axis=0
                )
                tl.store(
                    output_ptr + value_base + value_offsets,
                    output,
                    mask=token_value_mask,
                )
                state = tl.where(active, next_state, state)

            checkpoint_base = (chunk_index + 1) * checkpoint_stride + state_base
            tl.store(
                checkpoints_ptr + checkpoint_base + state_offsets,
                state,
                mask=state_mask,
            )

        if WRITE_FINAL:
            tl.store(
                final_memory_ptr + state_base + state_offsets,
                state,
                mask=state_mask,
            )

    @triton.jit
    def _value_iso_memory_zero_kernel(
        output_ptr,
        element_count,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(output_ptr + offsets, 0.0, mask=offsets < element_count)

    @triton.jit
    def _value_iso_memory_init_dstate_kernel(
        source_ptr,
        dstate_ptr,
        dinitial_ptr,
        element_count,
        COPY_SOURCE: tl.constexpr,
        WRITE_DINITIAL: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < element_count
        if COPY_SOURCE:
            values = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
        else:
            values = tl.zeros((BLOCK_SIZE,), tl.float32)
        tl.store(dstate_ptr + offsets, values, mask=mask)
        if WRITE_DINITIAL:
            tl.store(dinitial_ptr + offsets, values, mask=mask)

    @triton.jit(
        do_not_specialize=[
            "T",
            "segment_start",
            "segment_length",
            "checkpoint_index",
            "read_scale",
        ]
    )
    def _value_iso_memory_backward_segment_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        output_cotangent_ptr,
        alpha_ptr,
        beta_ptr,
        checkpoints_ptr,
        replay_ptr,
        dstate_carry_ptr,
        dq_acc_ptr,
        dk_acc_ptr,
        dalpha_ptr,
        dv_ptr,
        dbeta_domega_ptr,
        dinitial_memory_ptr,
        B,
        T,
        segment_start,
        segment_length,
        checkpoint_index,
        read_scale,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BV: tl.constexpr,
        C: tl.constexpr,
        WRITE_DQ: tl.constexpr,
        WRITE_DK: tl.constexpr,
        WRITE_DALPHA: tl.constexpr,
        WRITE_DV: tl.constexpr,
        WRITE_DBETA: tl.constexpr,
        WRITE_DINITIAL: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        head_index = tl.program_id(1)
        value_tile = tl.program_id(2)
        key_offsets = tl.arange(0, K)
        value_offsets = value_tile * BV + tl.arange(0, BV)
        value_mask = value_offsets < V
        state_offsets = key_offsets[:, None] * V + value_offsets[None, :]
        state_mask = value_mask[None, :]
        state_base = (batch_index * H + head_index) * K * V
        state_stride = B * H * K * V
        checkpoint_base = checkpoint_index * state_stride + state_base
        state = tl.load(
            checkpoints_ptr + checkpoint_base + state_offsets,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            replay_ptr + state_base + state_offsets,
            state,
            mask=state_mask,
        )

        for local_token in tl.range(0, segment_length, loop_unroll_factor=1):
            token = segment_start + local_token
            shared_base = (
                ((batch_index * T + token) * H + head_index) * K
            )
            value_base = (
                ((batch_index * T + token) * H + head_index) * V
            )
            key = tl.load(k_ptr + shared_base + key_offsets).to(tl.float32)
            alpha_value = tl.load(
                alpha_ptr + shared_base + key_offsets
            ).to(tl.float32)
            observation = tl.load(
                v_ptr + value_base + value_offsets,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
            beta_value = tl.load(
                beta_ptr + value_base + value_offsets,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
            predicted = alpha_value[:, None] * state
            residual = observation - tl.sum(predicted * key[:, None], axis=0)
            state = predicted + key[:, None] * (
                beta_value * residual
            )[None, :]
            tl.store(
                replay_ptr
                + (local_token + 1) * state_stride
                + state_base
                + state_offsets,
                state,
                mask=state_mask,
            )

        tl.debug_barrier()
        state_post = tl.load(
            replay_ptr
            + segment_length * state_stride
            + state_base
            + state_offsets,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
        dstate = tl.load(
            dstate_carry_ptr + state_base + state_offsets,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)

        for reverse_offset in tl.range(0, segment_length, loop_unroll_factor=1):
            local_token = segment_length - 1 - reverse_offset
            token = segment_start + local_token
            shared_base = (
                ((batch_index * T + token) * H + head_index) * K
            )
            value_base = (
                ((batch_index * T + token) * H + head_index) * V
            )
            state_prev = tl.load(
                replay_ptr
                + local_token * state_stride
                + state_base
                + state_offsets,
                mask=state_mask,
                other=0.0,
            ).to(tl.float32)
            query = tl.load(q_ptr + shared_base + key_offsets).to(tl.float32)
            key = tl.load(k_ptr + shared_base + key_offsets).to(tl.float32)
            alpha_value = tl.load(
                alpha_ptr + shared_base + key_offsets
            ).to(tl.float32)
            observation = tl.load(
                v_ptr + value_base + value_offsets,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
            beta_value = tl.load(
                beta_ptr + value_base + value_offsets,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
            output_grad = tl.load(
                output_cotangent_ptr + value_base + value_offsets,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)

            dstate += read_scale * query[:, None] * output_grad[None, :]
            dq_value = read_scale * tl.sum(
                state_post * output_grad[None, :], axis=1
            )
            predicted = alpha_value[:, None] * state_prev
            residual = observation - tl.sum(predicted * key[:, None], axis=0)
            weighted = beta_value * residual
            dk_value = tl.sum(dstate * weighted[None, :], axis=1)
            dweighted = tl.sum(dstate * key[:, None], axis=0)
            dbeta_value = dweighted * residual
            dresidual = dweighted * beta_value
            dk_value -= tl.sum(predicted * dresidual[None, :], axis=1)
            dpredicted = dstate - key[:, None] * dresidual[None, :]
            dalpha_value = tl.sum(dpredicted * state_prev, axis=1)
            dstate = alpha_value[:, None] * dpredicted

            if WRITE_DQ:
                tl.atomic_add(dq_acc_ptr + shared_base + key_offsets, dq_value)
            if WRITE_DK:
                tl.atomic_add(dk_acc_ptr + shared_base + key_offsets, dk_value)
            if WRITE_DALPHA:
                tl.atomic_add(
                    dalpha_ptr + shared_base + key_offsets, dalpha_value
                )
            if WRITE_DV:
                tl.store(
                    dv_ptr + value_base + value_offsets,
                    dresidual,
                    mask=value_mask,
                )
            if WRITE_DBETA:
                tl.store(
                    dbeta_domega_ptr + value_base + value_offsets,
                    dbeta_value,
                    mask=value_mask,
                )
            state_post = state_prev

        tl.store(
            dstate_carry_ptr + state_base + state_offsets,
            dstate,
            mask=state_mask,
        )
        if WRITE_DINITIAL:
            tl.store(
                dinitial_memory_ptr + state_base + state_offsets,
                dstate,
                mask=state_mask,
            )

else:  # pragma: no cover - exercised only without Triton installed
    _value_iso_memory_forward_kernel = None
    _value_iso_memory_zero_kernel = None
    _value_iso_memory_init_dstate_kernel = None
    _value_iso_memory_backward_segment_kernel = None


@torch.compiler.disable
def raw_value_iso_memory_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_memory: torch.Tensor | None = None,
    read_scale: float | None = None,
    output_final_state: bool = False,
) -> RawMemoryForwardResult:
    """Launch the selected-BV raw memory forward below autograd."""

    inputs = _validate_forward_inputs(
        q,
        k,
        v,
        alpha,
        beta,
        initial_memory,
        read_scale,
        output_final_state,
    )
    _require_triton()
    assert triton is not None and _value_iso_memory_forward_kernel is not None
    plan = inputs.plan
    output = torch.empty_like(v, memory_format=torch.contiguous_format)
    checkpoints = torch.empty(
        plan.checkpoints_shape, dtype=torch.float32, device=q.device
    )
    final_memory = (
        torch.empty(plan.final_memory_shape, dtype=torch.float32, device=q.device)
        if output_final_state
        else None
    )
    initial_pointer = initial_memory if initial_memory is not None else checkpoints
    final_pointer = final_memory if final_memory is not None else checkpoints
    with torch.cuda.device(q.device):
        _value_iso_memory_forward_kernel[plan.grid](
            q,
            k,
            v,
            alpha,
            beta,
            initial_pointer,
            output,
            checkpoints,
            final_pointer,
            plan.batch_size,
            plan.total_tokens,
            plan.chunk_count,
            inputs.read_scale,
            H=plan.heads,
            K=plan.key_dim,
            V=plan.value_dim,
            BV=plan.block_value,
            C=plan.chunk_size,
            HAS_INITIAL=initial_memory is not None,
            WRITE_FINAL=output_final_state,
            num_warps=plan.num_warps,
            num_stages=plan.num_stages,
        )
    return RawMemoryForwardResult(
        output=output,
        checkpoints=checkpoints,
        final_memory=final_memory,
        plan=plan,
    )


def _zero_memory_buffer(tensor: torch.Tensor) -> None:
    """Launch one accounted initialization for an FP32 accumulator."""

    _require_triton()
    assert triton is not None and _value_iso_memory_zero_kernel is not None
    block_size = 256
    grid = (max(1, triton.cdiv(tensor.numel(), block_size)),)
    _value_iso_memory_zero_kernel[grid](
        tensor,
        tensor.numel(),
        BLOCK_SIZE=block_size,
        num_warps=MEMORY_NUM_WARPS,
        num_stages=MEMORY_NUM_STAGES,
    )


def _initialize_dstate(
    dstate: torch.Tensor,
    final_memory_cotangent: torch.Tensor | None,
    dinitial_memory: torch.Tensor | None,
    *,
    empty_sequence: bool,
) -> None:
    """Initialize the reverse carry and the T=0 identity VJP in one launch."""

    _require_triton()
    assert triton is not None and _value_iso_memory_init_dstate_kernel is not None
    block_size = 256
    source = (
        final_memory_cotangent
        if final_memory_cotangent is not None
        else dstate
    )
    dinitial = dinitial_memory if dinitial_memory is not None else dstate
    grid = (triton.cdiv(dstate.numel(), block_size),)
    _value_iso_memory_init_dstate_kernel[grid](
        source,
        dstate,
        dinitial,
        dstate.numel(),
        COPY_SOURCE=final_memory_cotangent is not None,
        WRITE_DINITIAL=empty_sequence and dinitial_memory is not None,
        BLOCK_SIZE=block_size,
        num_warps=MEMORY_NUM_WARPS,
        num_stages=MEMORY_NUM_STAGES,
    )


@torch.compiler.disable
def raw_value_iso_memory_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_cotangent: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    checkpoints: torch.Tensor,
    replay_buffer: torch.Tensor,
    dstate_carry: torch.Tensor,
    dq_acc: torch.Tensor | None,
    dk_acc: torch.Tensor | None,
    dalpha: torch.Tensor | None,
    dv: torch.Tensor | None,
    dbeta_domega: torch.Tensor | None,
    *,
    final_memory_cotangent: torch.Tensor | None = None,
    dinitial_memory: torch.Tensor | None = None,
    read_scale: float | None = None,
    initialize_outputs: bool = True,
    launch_observer: Callable[[int, int], None] | None = None,
) -> RawMemoryBackwardResult:
    """Replay and reverse one checkpoint segment per descending host launch.

    All storage is caller owned.  Any unrequested gradient destination may be
    ``None``; compile-time write flags then remove both its stores and its
    full-time allocation.  Present shared ``dq_acc``, ``dk_acc``, and
    ``dalpha`` receive FP32 atomic additions; present value-local ``dv`` and
    ``dbeta_domega`` are direct stores.  The latter is deliberately distinct
    from saved ``beta`` and is the future in-place ``domega`` destination.
    """

    inputs = _validate_backward_inputs(
        q,
        k,
        v,
        output_cotangent,
        alpha,
        beta,
        checkpoints,
        replay_buffer,
        dstate_carry,
        dq_acc,
        dk_acc,
        dalpha,
        dv,
        dbeta_domega,
        final_memory_cotangent,
        dinitial_memory,
        read_scale,
        initialize_outputs,
    )
    if launch_observer is not None and not callable(launch_observer):
        raise TypeError("launch_observer must be None or callable")
    _require_triton()
    assert triton is not None
    assert _value_iso_memory_backward_segment_kernel is not None
    plan = inputs.plan

    with torch.cuda.device(q.device):
        if inputs.initialize_outputs:
            if inputs.dq_acc is not None:
                _zero_memory_buffer(inputs.dq_acc)
            if inputs.dk_acc is not None:
                _zero_memory_buffer(inputs.dk_acc)
            if inputs.dalpha is not None:
                _zero_memory_buffer(inputs.dalpha)
            _initialize_dstate(
                inputs.dstate_carry,
                inputs.final_memory_cotangent,
                inputs.dinitial_memory,
                empty_sequence=plan.total_tokens == 0,
            )

        for segment_start, segment_length, checkpoint_index in zip(
            plan.segment_starts,
            plan.segment_lengths,
            plan.checkpoint_indices,
        ):
            dinitial_pointer = (
                inputs.dinitial_memory
                if inputs.dinitial_memory is not None
                else inputs.dstate_carry
            )
            dq_pointer = (
                inputs.dq_acc if inputs.dq_acc is not None else inputs.dstate_carry
            )
            dk_pointer = (
                inputs.dk_acc if inputs.dk_acc is not None else inputs.dstate_carry
            )
            dalpha_pointer = (
                inputs.dalpha if inputs.dalpha is not None else inputs.dstate_carry
            )
            dv_pointer = inputs.dv if inputs.dv is not None else inputs.forward.v
            dbeta_pointer = (
                inputs.dbeta_domega
                if inputs.dbeta_domega is not None
                else inputs.dstate_carry
            )
            _value_iso_memory_backward_segment_kernel[plan.grid](
                q,
                k,
                v,
                inputs.output_cotangent,
                alpha,
                beta,
                inputs.checkpoints,
                inputs.replay_buffer,
                inputs.dstate_carry,
                dq_pointer,
                dk_pointer,
                dalpha_pointer,
                dv_pointer,
                dbeta_pointer,
                dinitial_pointer,
                plan.batch_size,
                plan.total_tokens,
                segment_start,
                segment_length,
                checkpoint_index,
                inputs.forward.read_scale,
                H=plan.heads,
                K=plan.key_dim,
                V=plan.value_dim,
                BV=plan.block_value,
                C=plan.chunk_size,
                WRITE_DQ=plan.write_dq,
                WRITE_DK=plan.write_dk,
                WRITE_DALPHA=plan.write_dalpha,
                WRITE_DV=plan.write_dv,
                WRITE_DBETA=plan.write_dbeta,
                WRITE_DINITIAL=(
                    segment_start == 0 and inputs.dinitial_memory is not None
                ),
                num_warps=MEMORY_NUM_WARPS,
                num_stages=MEMORY_NUM_STAGES,
            )
            if launch_observer is not None:
                launch_observer(segment_start, segment_length)

    return RawMemoryBackwardResult(
        dq_acc=inputs.dq_acc,
        dk_acc=inputs.dk_acc,
        dalpha=inputs.dalpha,
        dv=inputs.dv,
        dbeta_domega=inputs.dbeta_domega,
        dstate_carry=inputs.dstate_carry,
        dinitial_memory=inputs.dinitial_memory,
        plan=plan,
    )


__all__ = [
    "GAIN_CHUNK_SIZE",
    "GAIN_COPY_BLOCK_SIZE",
    "GAIN_NUM_STAGES",
    "GAIN_NUM_WARPS",
    "GainForwardPlan",
    "GainReversePlan",
    "MAX_VALUE_DIM",
    "MEMORY_BLOCK_VALUE",
    "MEMORY_CHUNK_SIZE",
    "MEMORY_NUM_STAGES",
    "MEMORY_NUM_WARPS",
    "MemoryBackwardPlan",
    "MemoryForwardPlan",
    "RawMemoryBackwardResult",
    "RawMemoryForwardResult",
    "RawGainBackwardResult",
    "RawGainForwardResult",
    "gain_forward_plan",
    "gain_reverse_plan",
    "memory_backward_plan",
    "memory_forward_plan",
    "raw_value_iso_memory_backward",
    "raw_value_iso_memory_forward",
    "raw_value_iso_gain_backward",
    "raw_value_iso_gain_forward",
]
