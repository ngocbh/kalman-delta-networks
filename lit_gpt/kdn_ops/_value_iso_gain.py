"""Raw Triton gain stage for value-wise isotropic KDN.

This module deliberately stays below an autograd boundary.  It provides the
three forward launches and the persistent reverse needed by the eventual
single operator Function:

``map (Ng, B*H) -> serial carry (B*H) -> emit (Ng, B*H)``.

The map/carry allocation is physical and intentionally destructive.  It starts
as slot-major ``[A, B, C, D]`` in ``[B,Ng,H,4,V]``; the carry kernel reads all
four coefficients before replacing the dead ``C,D`` slots with the exclusive
homogeneous carry ``n_in,d_in``.  The reverse replays the reciprocal precision
``p_prev=d/n`` into one reusable 64-token buffer, then applies the exact
ratio-first VJP.  All scan arithmetic and outputs are FP32.

Importing the module performs no CUDA query and remains possible when Triton is
not installed.  Kernel compilation and CUDA access occur only in the raw launch
helper.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

try:  # Host-only contract tests must not require a Triton installation.
    import triton
    import triton.language as tl
except (ImportError, OSError) as error:  # pragma: no cover - environment specific
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR: BaseException | None = error
else:
    _TRITON_IMPORT_ERROR = None


GAIN_CHUNK_SIZE = 64
GAIN_NUM_WARPS = 4
GAIN_NUM_STAGES = 1
SUPPORTED_KEY_DIMS = frozenset({64, 128})
MAX_VALUE_DIM = 256
GAIN_COPY_BLOCK_SIZE = 256

SCALE_PYTHON = 0
SCALE_SCALAR_TENSOR = 1
SCALE_PER_HEAD_TENSOR = 2


@dataclass(frozen=True)
class GainForwardPlan:
    """Pure host metadata for the frozen raw gain launch geometry."""

    batch_size: int
    total_tokens: int
    heads: int
    key_dim: int
    value_dim: int
    chunk_size: int
    chunk_count: int
    block_key: int
    block_value: int
    num_warps: int
    num_stages: int
    map_grid: tuple[int, int]
    carry_grid: tuple[int]
    emit_grid: tuple[int, int]
    forward_launch_count: int
    chunk_count_is_runtime: bool
    token_loop_is_runtime: bool
    maps_shape: tuple[int, int, int, int, int]
    statistics_shape: tuple[int, int, int, int]
    beta_shape: tuple[int, int, int, int]
    final_precision_shape: tuple[int, int, int]
    reverse_grid: tuple[int]
    reverse_replay_shape: tuple[int, int, int, int]
    reverse_carry_shape: tuple[int, int, int]
    scale_stream_shape: tuple[int, int]
    reverse_implemented: bool


@dataclass(frozen=True)
class GainReversePlan:
    """Exact conditional Triton launch and ownership metadata for reverse."""

    reverse_grid: tuple[int]
    dc_init_grid: tuple[int] | None
    prior_grid: tuple[int] | None
    initial_copy_grid: tuple[int] | None
    scale_grid: tuple[int] | None
    persistent_launch_count: int
    dc_init_launch_count: int
    prior_reducer_launch_count: int
    prior_copy_launch_count: int
    prior_zero_launch_count: int
    initial_copy_launch_count: int
    scale_reducer_launch_count: int
    scale_zero_launch_count: int
    total_launch_count: int
    replay_shape: tuple[int, int, int, int]
    carry_shape: tuple[int, int, int]
    scale_stream_shape: tuple[int, int] | None
    prior_is_shared: bool
    has_initial_precision: bool
    has_final_precision_cotangent: bool
    return_prior_gradient: bool
    return_initial_precision_gradient: bool
    return_information_scale_gradient: bool
    scale_kind: str


@dataclass(frozen=True)
class RawGainForwardResult:
    """Raw outputs/saves plus the launch choices that produced them."""

    beta: torch.Tensor
    maps_and_carries: torch.Tensor
    statistics: torch.Tensor
    final_precision: torch.Tensor
    plan: GainForwardPlan
    scale_kind: str
    precision_source_kind: str
    initial_batch_stride: int


@dataclass(frozen=True)
class RawGainBackwardResult:
    """Gain gradients and the exact reverse launches that produced them."""

    dk_acc: torch.Tensor | None
    dalpha: torch.Tensor | None
    domega: torch.Tensor | None
    dr: torch.Tensor | None
    dprior: torch.Tensor | None
    dinitial_precision: torch.Tensor | None
    dinformation_scale: torch.Tensor | None
    plan: GainReversePlan


@dataclass(frozen=True)
class _ValidatedReverseInputs:
    forward: _ValidatedForwardInputs
    maps_and_carries: torch.Tensor
    statistics: torch.Tensor
    dbeta_domega: torch.Tensor | None
    dk_acc: torch.Tensor | None
    dalpha: torch.Tensor | None
    dr: torch.Tensor | None
    final_precision_cotangent: torch.Tensor | None
    reverse_plan: GainReversePlan


@dataclass(frozen=True)
class _ValidatedForwardInputs:
    k: torch.Tensor
    alpha: torch.Tensor
    omega: torch.Tensor
    r: torch.Tensor
    prior: torch.Tensor
    initial_precision: torch.Tensor | None
    precision_source: torch.Tensor
    information_scale: float | torch.Tensor
    scale_kind: str
    scale_kind_code: int
    scale_value: float
    scale_tensor: torch.Tensor | None
    precision_source_kind: str
    initial_batch_stride: int
    plan: GainForwardPlan


def _exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be exactly int")
    return value


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def gain_forward_plan(
    *,
    batch_size: int,
    total_tokens: int,
    heads: int,
    key_dim: int,
    value_dim: int,
) -> GainForwardPlan:
    """Return exact scalar-only metadata without importing or querying CUDA."""

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

    chunk_count = (total_tokens + GAIN_CHUNK_SIZE - 1) // GAIN_CHUNK_SIZE
    batch_heads = batch_size * heads
    return GainForwardPlan(
        batch_size=batch_size,
        total_tokens=total_tokens,
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        chunk_size=GAIN_CHUNK_SIZE,
        chunk_count=chunk_count,
        block_key=_next_power_of_two(key_dim),
        block_value=_next_power_of_two(value_dim),
        num_warps=GAIN_NUM_WARPS,
        num_stages=GAIN_NUM_STAGES,
        map_grid=(chunk_count, batch_heads),
        carry_grid=(batch_heads,),
        emit_grid=(chunk_count, batch_heads),
        forward_launch_count=0 if total_tokens == 0 else 3,
        chunk_count_is_runtime=True,
        token_loop_is_runtime=True,
        maps_shape=(batch_size, chunk_count, heads, 4, value_dim),
        statistics_shape=(batch_size, total_tokens, heads, 2),
        beta_shape=(batch_size, total_tokens, heads, value_dim),
        final_precision_shape=(batch_size, heads, value_dim),
        reverse_grid=(batch_heads,),
        reverse_replay_shape=(batch_size, GAIN_CHUNK_SIZE, heads, value_dim),
        reverse_carry_shape=(batch_size, heads, value_dim),
        scale_stream_shape=(batch_size, heads),
        reverse_implemented=True,
    )


def gain_reverse_plan(
    forward_plan: GainForwardPlan,
    *,
    prior_is_shared: bool,
    has_initial_precision: bool,
    has_final_precision_cotangent: bool,
    scale_kind: str,
    return_prior_gradient: bool = False,
    return_initial_precision_gradient: bool = False,
    return_information_scale_gradient: bool = False,
) -> GainReversePlan:
    """Return exact reverse launch metadata for one validated invocation.

    Counts cover every reverse Triton launch, including empty-time copy/zero
    work.  The BHV reverse carry remains scratch and never aliases a returned
    prior or initial-precision gradient.
    """

    if not isinstance(forward_plan, GainForwardPlan):
        raise TypeError("forward_plan must be a GainForwardPlan")
    for name, value in (
        ("prior_is_shared", prior_is_shared),
        ("has_initial_precision", has_initial_precision),
        ("has_final_precision_cotangent", has_final_precision_cotangent),
        ("return_prior_gradient", return_prior_gradient),
        ("return_initial_precision_gradient", return_initial_precision_gradient),
        ("return_information_scale_gradient", return_information_scale_gradient),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be exactly bool")
    if scale_kind not in {"python", "scalar_tensor", "per_head_tensor"}:
        raise ValueError("scale_kind must be python, scalar_tensor, or per_head_tensor")
    if return_initial_precision_gradient and not has_initial_precision:
        raise ValueError(
            "return_initial_precision_gradient requires initial_precision"
        )
    if return_information_scale_gradient and scale_kind == "python":
        raise ValueError(
            "return_information_scale_gradient requires a tensor information_scale"
        )

    nonempty = forward_plan.total_tokens > 0
    scale_gradient = return_information_scale_gradient
    persistent_launch_count = int(nonempty)
    dc_init_launch_count = int(not nonempty)
    prior_reducer_launch_count = int(
        return_prior_gradient and not has_initial_precision and prior_is_shared
    )
    prior_copy_launch_count = int(
        return_prior_gradient and not has_initial_precision and not prior_is_shared
    )
    prior_zero_launch_count = int(
        return_prior_gradient and has_initial_precision
    )
    initial_copy_launch_count = int(return_initial_precision_gradient)
    scale_reducer_launch_count = int(nonempty and scale_gradient)
    scale_zero_launch_count = int(not nonempty and scale_gradient)
    state_elements = (
        forward_plan.batch_size * forward_plan.heads * forward_plan.value_dim
    )
    prior_elements = (
        forward_plan.heads * forward_plan.value_dim
        if prior_is_shared
        else state_elements
    )
    if not nonempty:
        dc_init_grid: tuple[int] | None = (
            (state_elements + GAIN_COPY_BLOCK_SIZE - 1) // GAIN_COPY_BLOCK_SIZE,
        )
    else:
        dc_init_grid = None
    if not return_prior_gradient:
        prior_grid: tuple[int] | None = None
    elif prior_reducer_launch_count:
        prior_grid = (forward_plan.heads,)
    else:
        prior_grid = (
            (prior_elements + GAIN_COPY_BLOCK_SIZE - 1) // GAIN_COPY_BLOCK_SIZE,
        )
    initial_copy_grid = (
        ((state_elements + GAIN_COPY_BLOCK_SIZE - 1) // GAIN_COPY_BLOCK_SIZE,)
        if return_initial_precision_gradient
        else None
    )
    if not scale_gradient:
        scale_grid: tuple[int] | None = None
    elif scale_kind == "scalar_tensor" and nonempty:
        scale_grid = (1,)
    elif scale_kind == "per_head_tensor" and nonempty:
        scale_grid = (forward_plan.heads,)
    elif scale_kind == "scalar_tensor":
        scale_grid = (1,)
    else:
        scale_grid = (
            (forward_plan.heads + GAIN_COPY_BLOCK_SIZE - 1)
            // GAIN_COPY_BLOCK_SIZE,
        )
    return GainReversePlan(
        reverse_grid=forward_plan.reverse_grid,
        dc_init_grid=dc_init_grid,
        prior_grid=prior_grid,
        initial_copy_grid=initial_copy_grid,
        scale_grid=scale_grid,
        persistent_launch_count=persistent_launch_count,
        dc_init_launch_count=dc_init_launch_count,
        prior_reducer_launch_count=prior_reducer_launch_count,
        prior_copy_launch_count=prior_copy_launch_count,
        prior_zero_launch_count=prior_zero_launch_count,
        initial_copy_launch_count=initial_copy_launch_count,
        scale_reducer_launch_count=scale_reducer_launch_count,
        scale_zero_launch_count=scale_zero_launch_count,
        total_launch_count=(
            persistent_launch_count
            + dc_init_launch_count
            + prior_reducer_launch_count
            + prior_copy_launch_count
            + prior_zero_launch_count
            + initial_copy_launch_count
            + scale_reducer_launch_count
            + scale_zero_launch_count
        ),
        replay_shape=forward_plan.reverse_replay_shape,
        carry_shape=forward_plan.reverse_carry_shape,
        scale_stream_shape=(
            forward_plan.scale_stream_shape if scale_gradient and nonempty else None
        ),
        prior_is_shared=prior_is_shared,
        has_initial_precision=has_initial_precision,
        has_final_precision_cotangent=has_final_precision_cotangent,
        return_prior_gradient=return_prior_gradient,
        return_initial_precision_gradient=return_initial_precision_gradient,
        return_information_scale_gradient=return_information_scale_gradient,
        scale_kind=scale_kind,
    )


def _slot_major_map_offset(
    batch: int,
    chunk: int,
    head: int,
    slot: int,
    value: int,
    *,
    chunk_count: int,
    heads: int,
    value_dim: int,
) -> int:
    """Host oracle for physical ``[B,Ng,H,4,V]`` pointer arithmetic."""

    return ((((batch * chunk_count + chunk) * heads + head) * 4 + slot) * value_dim + value)


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _require_shape(tensor: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")


def _require_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _canonicalize_scale(
    information_scale: float | torch.Tensor | None,
    *,
    key_dim: int,
    heads: int,
) -> tuple[float | torch.Tensor, str, int, float, torch.Tensor | None]:
    if information_scale is None:
        information_scale = float(key_dim)
    if isinstance(information_scale, torch.Tensor):
        if information_scale.ndim == 0:
            kind = "scalar_tensor"
            kind_code = SCALE_SCALAR_TENSOR
        elif information_scale.ndim == 1 and tuple(information_scale.shape) == (heads,):
            kind = "per_head_tensor"
            kind_code = SCALE_PER_HEAD_TENSOR
        else:
            raise ValueError(
                "information_scale must be a scalar or have shape "
                f"{(heads,)}, got {tuple(information_scale.shape)}"
            )
        if information_scale.dtype != torch.float32:
            raise TypeError(
                "information_scale must have dtype torch.float32, "
                f"got {information_scale.dtype}"
            )
        _require_contiguous("information_scale", information_scale)
        # Do not read a CUDA tensor on the host.  The runtime value is dead in
        # both tensor specializations and 1.0 supplies a stable launch ABI.
        return information_scale, kind, kind_code, 1.0, information_scale

    if isinstance(information_scale, bool) or type(information_scale) not in {int, float}:
        raise TypeError("information_scale must be a Python int/float or a tensor")
    scale_value = float(information_scale)
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError("Python information_scale must be finite and strictly positive")
    return information_scale, "python", SCALE_PYTHON, scale_value, None


def _select_precision_source(
    prior: torch.Tensor,
    initial_precision: torch.Tensor | None,
    *,
    heads: int,
    value_dim: int,
) -> tuple[torch.Tensor, str, int]:
    """Select the incoming precision without materializing a shared prior."""

    if initial_precision is not None:
        return initial_precision, "initial_precision", heads * value_dim
    if prior.ndim == 2:
        return prior, "shared_prior", 0
    return prior, "batched_prior", heads * value_dim


def _validate_forward_inputs(
    k: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    prior: torch.Tensor,
    initial_precision: torch.Tensor | None = None,
    information_scale: float | torch.Tensor | None = None,
    *,
    require_cuda: bool = True,
) -> _ValidatedForwardInputs:
    """Validate raw metadata before making the final CUDA requirement."""

    k = _require_tensor(k, "k")
    alpha = _require_tensor(alpha, "alpha")
    omega = _require_tensor(omega, "omega")
    r = _require_tensor(r, "r")
    prior = _require_tensor(prior, "prior")
    if k.ndim != 4:
        raise ValueError(f"k must have shape [B,T,H,K], got {tuple(k.shape)}")
    batch_size, total_tokens, heads, key_dim = k.shape
    if omega.ndim != 4 or tuple(omega.shape[:3]) != (
        batch_size,
        total_tokens,
        heads,
    ):
        raise ValueError(
            "omega must have shape [B,T,H,V], got "
            f"{tuple(omega.shape)}"
        )
    plan_value_dim = omega.shape[-1]
    plan = gain_forward_plan(
        batch_size=batch_size,
        total_tokens=total_tokens,
        heads=heads,
        key_dim=key_dim,
        value_dim=plan_value_dim,
    )
    _require_shape(alpha, (batch_size, total_tokens, heads, key_dim), "alpha")
    _require_shape(
        omega,
        (batch_size, total_tokens, heads, plan.value_dim),
        "omega",
    )
    _require_shape(r, (batch_size, total_tokens, heads, plan.value_dim), "r")

    shared_prior = tuple(prior.shape) == (heads, plan.value_dim)
    batched_prior = tuple(prior.shape) == (
        batch_size,
        heads,
        plan.value_dim,
    )
    if not shared_prior and not batched_prior:
        raise ValueError(
            "prior must have shape "
            f"{(heads, plan.value_dim)} or "
            f"{(batch_size, heads, plan.value_dim)}, got {tuple(prior.shape)}"
        )
    if initial_precision is not None:
        initial_precision = _require_tensor(initial_precision, "initial_precision")
        _require_shape(
            initial_precision,
            (batch_size, heads, plan.value_dim),
            "initial_precision",
        )

    if k.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(
            "k must have dtype torch.float32 or torch.bfloat16, "
            f"got {k.dtype}"
        )
    for name, tensor in (
        ("alpha", alpha),
        ("omega", omega),
        ("r", r),
        ("prior", prior),
    ):
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32, got {tensor.dtype}")
    if initial_precision is not None and initial_precision.dtype != torch.float32:
        raise TypeError(
            "initial_precision must have dtype torch.float32, "
            f"got {initial_precision.dtype}"
        )

    for name, tensor in (
        ("k", k),
        ("alpha", alpha),
        ("omega", omega),
        ("r", r),
        ("prior", prior),
    ):
        _require_contiguous(name, tensor)
    if initial_precision is not None:
        _require_contiguous("initial_precision", initial_precision)

    (
        information_scale,
        scale_kind,
        scale_kind_code,
        scale_value,
        scale_tensor,
    ) = _canonicalize_scale(
        information_scale,
        key_dim=key_dim,
        heads=heads,
    )

    named_tensors = [
        ("alpha", alpha),
        ("omega", omega),
        ("r", r),
        ("prior", prior),
    ]
    if initial_precision is not None:
        named_tensors.append(("initial_precision", initial_precision))
    if scale_tensor is not None:
        named_tensors.append(("information_scale", scale_tensor))
    mismatched = [
        f"{name}={tensor.device}"
        for name, tensor in named_tensors
        if tensor.device != k.device
    ]
    if mismatched:
        raise ValueError(
            f"all tensors must share k's device {k.device}; " + ", ".join(mismatched)
        )

    (
        precision_source,
        precision_source_kind,
        initial_batch_stride,
    ) = _select_precision_source(
        prior,
        initial_precision,
        heads=heads,
        value_dim=plan.value_dim,
    )

    if type(require_cuda) is not bool:
        raise TypeError("require_cuda must be exactly bool")
    if require_cuda and k.device.type != "cuda":
        raise ValueError(f"raw Triton gain requires CUDA tensors, got device={k.device}")

    return _ValidatedForwardInputs(
        k=k,
        alpha=alpha,
        omega=omega,
        r=r,
        prior=prior,
        initial_precision=initial_precision,
        precision_source=precision_source,
        information_scale=information_scale,
        scale_kind=scale_kind,
        scale_kind_code=scale_kind_code,
        scale_value=scale_value,
        scale_tensor=scale_tensor,
        precision_source_kind=precision_source_kind,
        initial_batch_stride=initial_batch_stride,
        plan=plan,
    )


def _storage_byte_range(
    tensor: torch.Tensor,
) -> tuple[torch.device, int, int] | None:
    """Return the occupied contiguous byte interval; empty tensors own none."""

    if tensor.numel() == 0:
        return None
    start = tensor.untyped_storage().data_ptr() + tensor.storage_offset() * tensor.element_size()
    return tensor.device, start, start + tensor.numel() * tensor.element_size()


def _byte_ranges_overlap(
    left: tuple[torch.device, int, int] | None,
    right: tuple[torch.device, int, int] | None,
) -> bool:
    if left is None or right is None or left[0] != right[0]:
        return False
    return left[1] < right[2] and right[1] < left[2]


def _reject_reverse_aliases(
    *,
    read_only: tuple[tuple[str, torch.Tensor], ...],
    writable: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    """Reject every overlap except the intentional single mutable beta formal."""

    read_ranges = tuple((name, _storage_byte_range(tensor)) for name, tensor in read_only)
    write_ranges = tuple((name, _storage_byte_range(tensor)) for name, tensor in writable)
    for index, (write_name, write_range) in enumerate(write_ranges):
        for other_name, other_range in write_ranges[index + 1 :]:
            if _byte_ranges_overlap(write_range, other_range):
                raise ValueError(f"{write_name} and {other_name} overlap in storage")
        for read_name, read_range in read_ranges:
            if _byte_ranges_overlap(write_range, read_range):
                raise ValueError(f"{write_name} and {read_name} overlap in storage")


if triton is not None:

    @triton.jit(do_not_specialize=["T", "NG", "scale_value"])
    def _value_iso_gain_map_kernel(
        k_ptr,
        alpha_ptr,
        omega_ptr,
        r_ptr,
        scale_ptr,
        maps_ptr,
        statistics_ptr,
        T,
        NG,
        scale_value,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        CG: tl.constexpr,
        SCALE_KIND: tl.constexpr,
    ):
        chunk_index = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch_index = batch_head // H
        head_index = batch_head % H

        key_offsets = tl.arange(0, BK)
        key_mask = key_offsets < K
        value_offsets = tl.arange(0, BV)
        value_mask = value_offsets < V

        # Triton 3.4 cannot capture ordinary Python globals in a JIT body.
        # These literals are the frozen SCALE_SCALAR_TENSOR/PER_HEAD codes.
        if SCALE_KIND == 1:
            head_scale = tl.load(scale_ptr).to(tl.float32)
        elif SCALE_KIND == 2:
            head_scale = tl.load(scale_ptr + head_index).to(tl.float32)
        else:
            head_scale = scale_value

        map_a = tl.full((BV,), 1.0, tl.float32)
        map_b = tl.full((BV,), 0.0, tl.float32)
        map_c = tl.full((BV,), 0.0, tl.float32)
        map_d = tl.full((BV,), 1.0, tl.float32)
        token_start = chunk_index * CG
        for local_token in tl.range(0, CG, loop_unroll_factor=1):
            token = token_start + local_token
            active = token < T
            key_base = ((batch_index * T + token) * H + head_index) * K
            key = tl.load(
                k_ptr + key_base + key_offsets,
                mask=active & key_mask,
                other=0.0,
            ).to(tl.float32)
            transition = tl.load(
                alpha_ptr + key_base + key_offsets,
                mask=active & key_mask,
                other=0.0,
            ).to(tl.float32)
            a_stat = tl.sum(transition * transition, axis=0) / K
            nu_stat = tl.sum(key * key, axis=0)

            statistics_base = ((batch_index * T + token) * H + head_index) * 2
            tl.store(statistics_ptr + statistics_base, a_stat, mask=active)
            tl.store(statistics_ptr + statistics_base + 1, nu_stat, mask=active)

            value_base = ((batch_index * T + token) * H + head_index) * V
            omega_value = tl.load(
                omega_ptr + value_base + value_offsets,
                mask=active & value_mask,
                other=0.0,
            ).to(tl.float32)
            r_value = tl.load(
                r_ptr + value_base + value_offsets,
                mask=active & value_mask,
                other=1.0,
            ).to(tl.float32)
            u = head_scale * nu_stat / (K * r_value)
            token_a = 1.0 + u * omega_value
            token_b = u * a_stat
            token_c = omega_value
            token_d = a_stat

            next_a = token_a * map_a + token_b * map_c
            next_b = token_a * map_b + token_b * map_d
            next_c = token_c * map_a + token_d * map_c
            next_d = token_c * map_b + token_d * map_d
            magnitude = tl.maximum(
                tl.maximum(tl.abs(next_a), tl.abs(next_b)),
                tl.maximum(tl.abs(next_c), tl.abs(next_d)),
            )
            inverse_magnitude = 1.0 / tl.maximum(magnitude, 1e-30)
            normalized_a = tl.maximum(
                tl.minimum(next_a * inverse_magnitude, 1.0), -1.0
            )
            normalized_b = tl.maximum(
                tl.minimum(next_b * inverse_magnitude, 1.0), -1.0
            )
            normalized_c = tl.maximum(
                tl.minimum(next_c * inverse_magnitude, 1.0), -1.0
            )
            normalized_d = tl.maximum(
                tl.minimum(next_d * inverse_magnitude, 1.0), -1.0
            )
            map_a = tl.where(active, normalized_a, map_a)
            map_b = tl.where(active, normalized_b, map_b)
            map_c = tl.where(active, normalized_c, map_c)
            map_d = tl.where(active, normalized_d, map_d)

        map_base = (((batch_index * NG + chunk_index) * H + head_index) * 4) * V
        tl.store(maps_ptr + map_base + 0 * V + value_offsets, map_a, mask=value_mask)
        tl.store(maps_ptr + map_base + 1 * V + value_offsets, map_b, mask=value_mask)
        tl.store(maps_ptr + map_base + 2 * V + value_offsets, map_c, mask=value_mask)
        tl.store(maps_ptr + map_base + 3 * V + value_offsets, map_d, mask=value_mask)


    @triton.jit(do_not_specialize=["NG"])
    def _value_iso_gain_carry_kernel(
        maps_ptr,
        initial_precision_ptr,
        final_precision_ptr,
        NG,
        H: tl.constexpr,
        V: tl.constexpr,
        BV: tl.constexpr,
        INIT_STRIDE_B: tl.constexpr,
    ):
        batch_head = tl.program_id(0)
        batch_index = batch_head // H
        head_index = batch_head % H
        value_offsets = tl.arange(0, BV)
        value_mask = value_offsets < V

        precision_base = batch_index * INIT_STRIDE_B + head_index * V
        n = tl.load(
            initial_precision_ptr + precision_base + value_offsets,
            mask=value_mask,
            other=1.0,
        ).to(tl.float32)
        d = tl.full((BV,), 1.0, tl.float32)
        initial_magnitude = tl.maximum(tl.abs(n), tl.abs(d))
        initial_inverse_magnitude = 1.0 / tl.maximum(initial_magnitude, 1e-30)
        n = tl.maximum(tl.minimum(n * initial_inverse_magnitude, 1.0), -1.0)
        d = tl.maximum(tl.minimum(d * initial_inverse_magnitude, 1.0), -1.0)

        for chunk_index in tl.range(0, NG, loop_unroll_factor=1):
            map_base = (((batch_index * NG + chunk_index) * H + head_index) * 4) * V
            mA = tl.load(
                maps_ptr + map_base + 0 * V + value_offsets,
                mask=value_mask,
                other=1.0,
            ).to(tl.float32)
            mB = tl.load(
                maps_ptr + map_base + 1 * V + value_offsets,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
            mC = tl.load(
                maps_ptr + map_base + 2 * V + value_offsets,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
            mD = tl.load(
                maps_ptr + map_base + 3 * V + value_offsets,
                mask=value_mask,
                other=1.0,
            ).to(tl.float32)
            next_n = mA * n + mB * d
            next_d = mC * n + mD * d

            # C,D are dead only after all four map coefficients have reached
            # registers and next_n/next_d have been formed.
            tl.store(maps_ptr + map_base + 2 * V + value_offsets, n, mask=value_mask)
            tl.store(maps_ptr + map_base + 3 * V + value_offsets, d, mask=value_mask)

            magnitude = tl.maximum(tl.abs(next_n), tl.abs(next_d))
            inverse_magnitude = 1.0 / tl.maximum(magnitude, 1e-30)
            n = tl.maximum(tl.minimum(next_n * inverse_magnitude, 1.0), -1.0)
            d = tl.maximum(tl.minimum(next_d * inverse_magnitude, 1.0), -1.0)

        final_base = (batch_index * H + head_index) * V
        tl.store(
            final_precision_ptr + final_base + value_offsets,
            n / d,
            mask=value_mask,
        )


    @triton.jit(do_not_specialize=["T", "NG", "scale_value"])
    def _value_iso_gain_emit_kernel(
        omega_ptr,
        r_ptr,
        scale_ptr,
        maps_ptr,
        statistics_ptr,
        beta_ptr,
        T,
        NG,
        scale_value,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BV: tl.constexpr,
        CG: tl.constexpr,
        SCALE_KIND: tl.constexpr,
    ):
        chunk_index = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch_index = batch_head // H
        head_index = batch_head % H
        value_offsets = tl.arange(0, BV)
        value_mask = value_offsets < V

        # Keep the constexpr comparisons literal: Triton 3.4 rejects reads of
        # the equivalent module-level Python constants from a JIT body.
        if SCALE_KIND == 1:
            head_scale = tl.load(scale_ptr).to(tl.float32)
        elif SCALE_KIND == 2:
            head_scale = tl.load(scale_ptr + head_index).to(tl.float32)
        else:
            head_scale = scale_value

        map_base = (((batch_index * NG + chunk_index) * H + head_index) * 4) * V
        n = tl.load(
            maps_ptr + map_base + 2 * V + value_offsets,
            mask=value_mask,
            other=1.0,
        ).to(tl.float32)
        d = tl.load(
            maps_ptr + map_base + 3 * V + value_offsets,
            mask=value_mask,
            other=1.0,
        ).to(tl.float32)

        token_start = chunk_index * CG
        for local_token in tl.range(0, CG, loop_unroll_factor=1):
            token = token_start + local_token
            active = token < T
            statistics_base = ((batch_index * T + token) * H + head_index) * 2
            a_stat = tl.load(
                statistics_ptr + statistics_base, mask=active, other=0.0
            ).to(tl.float32)
            nu_stat = tl.load(
                statistics_ptr + statistics_base + 1, mask=active, other=0.0
            ).to(tl.float32)
            value_base = ((batch_index * T + token) * H + head_index) * V
            omega_value = tl.load(
                omega_ptr + value_base + value_offsets,
                mask=active & value_mask,
                other=0.0,
            ).to(tl.float32)
            r_value = tl.load(
                r_ptr + value_base + value_offsets,
                mask=active & value_mask,
                other=1.0,
            ).to(tl.float32)

            chi = omega_value * n + a_stat * d
            denominator = r_value * n + nu_stat * chi
            beta = chi / denominator
            tl.store(
                beta_ptr + value_base + value_offsets,
                beta,
                mask=active & value_mask,
            )

            u = head_scale * nu_stat / (K * r_value)
            next_n = (1.0 + u * omega_value) * n + (u * a_stat) * d
            next_d = chi
            magnitude = tl.maximum(tl.abs(next_n), tl.abs(next_d))
            inverse_magnitude = 1.0 / tl.maximum(magnitude, 1e-30)
            normalized_n = tl.maximum(
                tl.minimum(next_n * inverse_magnitude, 1.0), -1.0
            )
            normalized_d = tl.maximum(
                tl.minimum(next_d * inverse_magnitude, 1.0), -1.0
            )
            n = tl.where(active, normalized_n, n)
            d = tl.where(active, normalized_d, d)


    @triton.jit(do_not_specialize=["T", "NG", "scale_value"])
    def _value_iso_gain_reverse_kernel(
        k_ptr,
        alpha_ptr,
        omega_ptr,
        r_ptr,
        scale_ptr,
        maps_ptr,
        statistics_ptr,
        dbeta_domega_ptr,
        dk_acc_ptr,
        dalpha_ptr,
        dr_ptr,
        final_dc_ptr,
        replay_ptr,
        dc_ptr,
        ds_stream_ptr,
        T,
        NG,
        scale_value,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        CG: tl.constexpr,
        SCALE_KIND: tl.constexpr,
        HAS_DFINAL: tl.constexpr,
        HAS_DS_STREAM: tl.constexpr,
        HAS_DBETA: tl.constexpr,
        WRITE_DK: tl.constexpr,
        WRITE_DALPHA: tl.constexpr,
        WRITE_DOMEGA: tl.constexpr,
        WRITE_DR: tl.constexpr,
    ):
        """Own one ``(batch, head)`` stream for replay and exact reverse."""

        batch_head = tl.program_id(0)
        batch_index = batch_head // H
        head_index = batch_head % H
        key_offsets = tl.arange(0, BK)
        key_mask = key_offsets < K
        value_offsets = tl.arange(0, BV)
        value_mask = value_offsets < V
        dc_base = (batch_index * H + head_index) * V

        if SCALE_KIND == 1:
            head_scale = tl.load(scale_ptr).to(tl.float32)
        elif SCALE_KIND == 2:
            head_scale = tl.load(scale_ptr + head_index).to(tl.float32)
        else:
            head_scale = scale_value

        if HAS_DFINAL:
            dc = tl.load(
                final_dc_ptr + dc_base + value_offsets,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            dc = tl.zeros((BV,), tl.float32)
        ds_acc = tl.zeros((), tl.float32)

        # NG is a runtime value.  The loop must remain serial and must not be
        # specialized into 64/256 copies at production sequence lengths.
        for reverse_chunk in tl.range(0, NG, loop_unroll_factor=1):
            chunk_index = NG - 1 - reverse_chunk
            token_start = chunk_index * CG
            chunk_width = tl.minimum(CG, T - token_start)
            map_base = (
                ((batch_index * NG + chunk_index) * H + head_index) * 4
            ) * V
            n = tl.load(
                maps_ptr + map_base + 2 * V + value_offsets,
                mask=value_mask,
                other=1.0,
            ).to(tl.float32)
            d = tl.load(
                maps_ptr + map_base + 3 * V + value_offsets,
                mask=value_mask,
                other=1.0,
            ).to(tl.float32)

            # Replay the exact normalized homogeneous forward.  Only the
            # reciprocal precision p_prev=d/n is materialized; raw c=n/d is
            # intentionally never formed because d may decay to zero.
            for local_token in tl.range(0, CG, loop_unroll_factor=1):
                token = token_start + local_token
                active = local_token < chunk_width
                token_value_mask = active & value_mask
                p_prev = d / n
                replay_base = (
                    ((batch_index * CG + local_token) * H + head_index) * V
                )
                tl.store(
                    replay_ptr + replay_base + value_offsets,
                    p_prev,
                    mask=token_value_mask,
                )

                statistics_base = (
                    (batch_index * T + token) * H + head_index
                ) * 2
                a_stat = tl.load(
                    statistics_ptr + statistics_base,
                    mask=active,
                    other=0.0,
                ).to(tl.float32)
                nu_stat = tl.load(
                    statistics_ptr + statistics_base + 1,
                    mask=active,
                    other=0.0,
                ).to(tl.float32)
                value_base = (
                    (batch_index * T + token) * H + head_index
                ) * V
                omega_value = tl.load(
                    omega_ptr + value_base + value_offsets,
                    mask=token_value_mask,
                    other=0.0,
                ).to(tl.float32)
                r_value = tl.load(
                    r_ptr + value_base + value_offsets,
                    mask=token_value_mask,
                    other=1.0,
                ).to(tl.float32)
                chi = omega_value * n + a_stat * d
                u = head_scale * nu_stat / (K * r_value)
                # Replay must use the identical expanded arithmetic order as
                # the forward emit kernel.  The equivalent ``n + u * chi``
                # form accumulates different FP32 rounding across a chunk.
                next_n = (1.0 + u * omega_value) * n + (u * a_stat) * d
                next_d = chi
                magnitude = tl.maximum(tl.abs(next_n), tl.abs(next_d))
                inverse_magnitude = 1.0 / tl.maximum(magnitude, 1e-30)
                normalized_n = tl.maximum(
                    tl.minimum(next_n * inverse_magnitude, 1.0), -1.0
                )
                normalized_d = tl.maximum(
                    tl.minimum(next_d * inverse_magnitude, 1.0), -1.0
                )
                n = tl.where(active, normalized_n, n)
                d = tl.where(active, normalized_d, d)

            # Replay is reused chunk by chunk.  Keep publication and reverse
            # consumption in distinct compiler phases within this program.
            tl.debug_barrier()

            for reverse_offset in tl.range(0, CG, loop_unroll_factor=1):
                local_token = chunk_width - 1 - reverse_offset
                active = reverse_offset < chunk_width
                safe_local_token = tl.maximum(local_token, 0)
                token = token_start + safe_local_token
                token_value_mask = active & value_mask
                replay_base = (
                    ((batch_index * CG + safe_local_token) * H + head_index) * V
                )
                p_prev = tl.load(
                    replay_ptr + replay_base + value_offsets,
                    mask=token_value_mask,
                    other=1.0,
                ).to(tl.float32)

                statistics_base = (
                    (batch_index * T + token) * H + head_index
                ) * 2
                a_stat = tl.load(
                    statistics_ptr + statistics_base,
                    mask=active,
                    other=0.0,
                ).to(tl.float32)
                nu_stat = tl.load(
                    statistics_ptr + statistics_base + 1,
                    mask=active,
                    other=0.0,
                ).to(tl.float32)
                value_base = (
                    (batch_index * T + token) * H + head_index
                ) * V
                omega_value = tl.load(
                    omega_ptr + value_base + value_offsets,
                    mask=token_value_mask,
                    other=0.0,
                ).to(tl.float32)
                r_value = tl.load(
                    r_ptr + value_base + value_offsets,
                    mask=token_value_mask,
                    other=1.0,
                ).to(tl.float32)

                # This is the only load of the mutable beta/omega formal for
                # the token.  Every consumer is formed before the overwrite.
                if HAS_DBETA:
                    dbeta = tl.load(
                        dbeta_domega_ptr + value_base + value_offsets,
                        mask=token_value_mask,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    dbeta = tl.zeros((BV,), tl.float32)
                bhat = a_stat * p_prev + omega_value
                denominator = r_value + nu_stat * bhat
                beta = bhat / denominator
                gamma = r_value / denominator

                # Ratio-first forms avoid explicitly squaring tiny bhat/r.
                db_gain = (dbeta * gamma) / denominator
                dbhat = db_gain - (dc / bhat) / bhat
                dr_value = (
                    -(dbeta * beta) / denominator
                    - (dc / r_value) * (head_scale * nu_stat / (K * r_value))
                )
                dnu_value = (
                    -(dbeta * beta) * beta
                    + dc * head_scale / (K * r_value)
                )
                da_value = dbhat * p_prev
                # Keep the state carry out of dbhat's potentially huge
                # inverse-square intermediate.  This decomposition is
                # algebraically exact but remains finite when bhat is tiny.
                carry_from_dc = (
                    (a_stat * p_prev / bhat) * (p_prev / bhat)
                )
                carry_from_beta = -db_gain * (a_stat * p_prev) * p_prev
                dc_prev = carry_from_dc * dc + carry_from_beta

                da_sum = tl.sum(
                    tl.where(token_value_mask, da_value, 0.0), axis=0
                )
                dnu_sum = tl.sum(
                    tl.where(token_value_mask, dnu_value, 0.0), axis=0
                )
                if HAS_DS_STREAM:
                    ds_value = dc * nu_stat / (K * r_value)
                    ds_acc += tl.sum(
                        tl.where(token_value_mask, ds_value, 0.0), axis=0
                    )
                key_base = (
                    (batch_index * T + token) * H + head_index
                ) * K
                token_key_mask = active & key_mask
                if WRITE_DK:
                    key = tl.load(
                        k_ptr + key_base + key_offsets,
                        mask=token_key_mask,
                        other=0.0,
                    ).to(tl.float32)
                    dk_existing = tl.load(
                        dk_acc_ptr + key_base + key_offsets,
                        mask=token_key_mask,
                        other=0.0,
                    ).to(tl.float32)
                    tl.store(
                        dk_acc_ptr + key_base + key_offsets,
                        dk_existing + 2.0 * key * dnu_sum,
                        mask=token_key_mask,
                    )
                if WRITE_DALPHA:
                    alpha_value = tl.load(
                        alpha_ptr + key_base + key_offsets,
                        mask=token_key_mask,
                        other=0.0,
                    ).to(tl.float32)
                    dalpha_existing = tl.load(
                        dalpha_ptr + key_base + key_offsets,
                        mask=token_key_mask,
                        other=0.0,
                    ).to(tl.float32)
                    tl.store(
                        dalpha_ptr + key_base + key_offsets,
                        dalpha_existing + (2.0 * alpha_value / K) * da_sum,
                        mask=token_key_mask,
                    )
                if WRITE_DR:
                    tl.store(
                        dr_ptr + value_base + value_offsets,
                        dr_value,
                        mask=token_value_mask,
                    )
                if WRITE_DOMEGA:
                    tl.store(
                        dbeta_domega_ptr + value_base + value_offsets,
                        dbhat,
                        mask=token_value_mask,
                    )
                dc = tl.where(active, dc_prev, dc)

            # All replay reads must retire before the next outer iteration
            # reuses the same [B,CG,H,V] slice for an earlier chunk.
            tl.debug_barrier()

        tl.store(dc_ptr + dc_base + value_offsets, dc, mask=value_mask)
        if HAS_DS_STREAM:
            tl.store(ds_stream_ptr + batch_head, ds_acc)


    @triton.jit(do_not_specialize=["B"])
    def _value_iso_gain_prior_reduce_kernel(
        dc_ptr,
        dprior_ptr,
        B,
        H: tl.constexpr,
        V: tl.constexpr,
        BV: tl.constexpr,
    ):
        head_index = tl.program_id(0)
        value_offsets = tl.arange(0, BV)
        value_mask = value_offsets < V
        total = tl.zeros((BV,), tl.float32)
        for batch_index in tl.range(0, B, loop_unroll_factor=1):
            dc_base = (batch_index * H + head_index) * V
            total += tl.load(
                dc_ptr + dc_base + value_offsets,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            dprior_ptr + head_index * V + value_offsets,
            total,
            mask=value_mask,
        )


    @triton.jit(do_not_specialize=["B", "H"])
    def _value_iso_gain_scale_reduce_kernel(
        ds_stream_ptr,
        dscale_ptr,
        B,
        H,
        SCALE_KIND: tl.constexpr,
    ):
        output_index = tl.program_id(0)
        total = tl.zeros((), tl.float32)
        if SCALE_KIND == 1:
            for batch_index in tl.range(0, B, loop_unroll_factor=1):
                for head_index in tl.range(0, H, loop_unroll_factor=1):
                    total += tl.load(
                        ds_stream_ptr + batch_index * H + head_index
                    ).to(tl.float32)
            tl.store(dscale_ptr, total)
        else:
            for batch_index in tl.range(0, B, loop_unroll_factor=1):
                total += tl.load(
                    ds_stream_ptr + batch_index * H + output_index
                ).to(tl.float32)
            tl.store(dscale_ptr + output_index, total)


    @triton.jit
    def _value_iso_gain_copy_or_zero_kernel(
        source_ptr,
        destination_ptr,
        element_count,
        COPY_SOURCE: tl.constexpr,
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
        tl.store(destination_ptr + offsets, values, mask=mask)

else:  # pragma: no cover - exercised only when Triton itself is unavailable
    _value_iso_gain_map_kernel = None
    _value_iso_gain_carry_kernel = None
    _value_iso_gain_emit_kernel = None
    _value_iso_gain_reverse_kernel = None
    _value_iso_gain_prior_reduce_kernel = None
    _value_iso_gain_scale_reduce_kernel = None
    _value_iso_gain_copy_or_zero_kernel = None


def _require_triton() -> None:
    if triton is None:
        detail = f": {_TRITON_IMPORT_ERROR}" if _TRITON_IMPORT_ERROR else ""
        raise RuntimeError(f"Triton is required for raw value-wise gain{detail}")


def _scale_pointer(inputs: _ValidatedForwardInputs, dummy: torch.Tensor) -> torch.Tensor:
    return inputs.scale_tensor if inputs.scale_tensor is not None else dummy


def _validate_workspace_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    tensor = _require_tensor(tensor, name)
    _require_shape(tensor, shape, name)
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32, got {tensor.dtype}")
    _require_contiguous(name, tensor)
    if tensor.device != device:
        raise ValueError(f"{name} must be on device {device}, got {tensor.device}")


def _validate_mutable_reverse_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    _validate_workspace_tensor(tensor, name=name, shape=shape, device=device)
    if tensor.requires_grad:
        raise ValueError(f"{name} is mutable and must not require gradients")
    return tensor


def _validate_reverse_inputs(
    k: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    prior: torch.Tensor,
    maps_and_carries: torch.Tensor,
    statistics: torch.Tensor,
    dbeta_domega: torch.Tensor | None,
    dk_acc: torch.Tensor | None,
    dalpha: torch.Tensor | None,
    dr: torch.Tensor | None,
    initial_precision: torch.Tensor | None = None,
    information_scale: float | torch.Tensor | None = None,
    final_precision_cotangent: torch.Tensor | None = None,
    *,
    return_prior_gradient: bool = False,
    return_initial_precision_gradient: bool = False,
    return_information_scale_gradient: bool = False,
    require_cuda: bool = True,
) -> _ValidatedReverseInputs:
    """Validate the complete raw reverse ABI before requiring CUDA."""

    forward = _validate_forward_inputs(
        k,
        alpha,
        omega,
        r,
        prior,
        initial_precision,
        information_scale,
        require_cuda=False,
    )
    plan = forward.plan
    device = k.device
    _validate_workspace_tensor(
        maps_and_carries,
        name="maps_and_carries",
        shape=plan.maps_shape,
        device=device,
    )
    _validate_workspace_tensor(
        statistics,
        name="statistics",
        shape=plan.statistics_shape,
        device=device,
    )
    destinations = {
        "dbeta_domega": dbeta_domega,
        "dk_acc": dk_acc,
        "dalpha": dalpha,
        "dr": dr,
    }
    destination_shapes = {
        "dbeta_domega": plan.beta_shape,
        "dk_acc": tuple(k.shape),
        "dalpha": tuple(alpha.shape),
        "dr": plan.beta_shape,
    }
    for name, tensor in destinations.items():
        if tensor is not None:
            destinations[name] = _validate_mutable_reverse_tensor(
                tensor,
                name=name,
                shape=destination_shapes[name],
                device=device,
            )
    dbeta_domega = destinations["dbeta_domega"]
    dk_acc = destinations["dk_acc"]
    dalpha = destinations["dalpha"]
    dr = destinations["dr"]
    if final_precision_cotangent is not None:
        final_precision_cotangent = _require_tensor(
            final_precision_cotangent, "final_precision_cotangent"
        )
        _validate_workspace_tensor(
            final_precision_cotangent,
            name="final_precision_cotangent",
            shape=plan.final_precision_shape,
            device=device,
        )

    read_only: list[tuple[str, torch.Tensor]] = [
        ("k", k),
        ("alpha", alpha),
        ("omega", omega),
        ("r", r),
        ("prior", prior),
        ("maps_and_carries", maps_and_carries),
        ("statistics", statistics),
    ]
    if initial_precision is not None:
        read_only.append(("initial_precision", initial_precision))
    if forward.scale_tensor is not None:
        read_only.append(("information_scale", forward.scale_tensor))
    if final_precision_cotangent is not None:
        read_only.append(
            ("final_precision_cotangent", final_precision_cotangent)
        )
    _reject_reverse_aliases(
        read_only=tuple(read_only),
        writable=tuple(
            (name, tensor)
            for name, tensor in destinations.items()
            if tensor is not None
        ),
    )

    reverse_plan = gain_reverse_plan(
        plan,
        prior_is_shared=prior.ndim == 2,
        has_initial_precision=initial_precision is not None,
        has_final_precision_cotangent=final_precision_cotangent is not None,
        scale_kind=forward.scale_kind,
        return_prior_gradient=return_prior_gradient,
        return_initial_precision_gradient=return_initial_precision_gradient,
        return_information_scale_gradient=return_information_scale_gradient,
    )
    if type(require_cuda) is not bool:
        raise TypeError("require_cuda must be exactly bool")
    if require_cuda and device.type != "cuda":
        raise ValueError(f"raw Triton gain reverse requires CUDA tensors, got device={device}")
    return _ValidatedReverseInputs(
        forward=forward,
        maps_and_carries=maps_and_carries,
        statistics=statistics,
        dbeta_domega=dbeta_domega,
        dk_acc=dk_acc,
        dalpha=dalpha,
        dr=dr,
        final_precision_cotangent=final_precision_cotangent,
        reverse_plan=reverse_plan,
    )


def _launch_forward_map(
    inputs: _ValidatedForwardInputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    plan = inputs.plan
    maps = torch.empty(plan.maps_shape, dtype=torch.float32, device=inputs.k.device)
    statistics = torch.empty(
        plan.statistics_shape, dtype=torch.float32, device=inputs.k.device
    )
    if plan.total_tokens == 0:
        return maps, statistics
    _require_triton()
    assert _value_iso_gain_map_kernel is not None
    scale_pointer = _scale_pointer(inputs, statistics)
    with torch.cuda.device(inputs.k.device):
        _value_iso_gain_map_kernel[plan.map_grid](
            inputs.k,
            inputs.alpha,
            inputs.omega,
            inputs.r,
            scale_pointer,
            maps,
            statistics,
            plan.total_tokens,
            plan.chunk_count,
            inputs.scale_value,
            H=plan.heads,
            K=plan.key_dim,
            V=plan.value_dim,
            BK=plan.block_key,
            BV=plan.block_value,
            CG=plan.chunk_size,
            SCALE_KIND=inputs.scale_kind_code,
            num_warps=plan.num_warps,
            num_stages=plan.num_stages,
        )
    return maps, statistics


def _launch_forward_carry(
    inputs: _ValidatedForwardInputs,
    maps: torch.Tensor,
) -> torch.Tensor:
    plan = inputs.plan
    _validate_workspace_tensor(
        maps,
        name="maps",
        shape=plan.maps_shape,
        device=inputs.k.device,
    )
    final_precision = torch.empty(
        plan.final_precision_shape,
        dtype=torch.float32,
        device=inputs.k.device,
    )
    if plan.total_tokens == 0:
        source = inputs.precision_source
        expanded = (
            source.unsqueeze(0).expand(plan.batch_size, -1, -1)
            if source.ndim == 2
            else source
        )
        final_precision.copy_(expanded)
        return final_precision
    _require_triton()
    assert _value_iso_gain_carry_kernel is not None
    with torch.cuda.device(inputs.k.device):
        _value_iso_gain_carry_kernel[plan.carry_grid](
            maps,
            inputs.precision_source,
            final_precision,
            plan.chunk_count,
            H=plan.heads,
            V=plan.value_dim,
            BV=plan.block_value,
            INIT_STRIDE_B=inputs.initial_batch_stride,
            num_warps=plan.num_warps,
            num_stages=plan.num_stages,
        )
    return final_precision


def _launch_forward_emit(
    inputs: _ValidatedForwardInputs,
    maps: torch.Tensor,
    statistics: torch.Tensor,
) -> torch.Tensor:
    plan = inputs.plan
    _validate_workspace_tensor(
        maps,
        name="maps",
        shape=plan.maps_shape,
        device=inputs.k.device,
    )
    _validate_workspace_tensor(
        statistics,
        name="statistics",
        shape=plan.statistics_shape,
        device=inputs.k.device,
    )
    beta = torch.empty(plan.beta_shape, dtype=torch.float32, device=inputs.k.device)
    if plan.total_tokens == 0:
        return beta
    _require_triton()
    assert _value_iso_gain_emit_kernel is not None
    scale_pointer = _scale_pointer(inputs, statistics)
    with torch.cuda.device(inputs.k.device):
        _value_iso_gain_emit_kernel[plan.emit_grid](
            inputs.omega,
            inputs.r,
            scale_pointer,
            maps,
            statistics,
            beta,
            plan.total_tokens,
            plan.chunk_count,
            inputs.scale_value,
            H=plan.heads,
            K=plan.key_dim,
            V=plan.value_dim,
            BV=plan.block_value,
            CG=plan.chunk_size,
            SCALE_KIND=inputs.scale_kind_code,
            num_warps=plan.num_warps,
            num_stages=plan.num_stages,
        )
    return beta


@torch.compiler.disable
def raw_value_iso_gain_forward(
    k: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    prior: torch.Tensor,
    initial_precision: torch.Tensor | None = None,
    information_scale: float | torch.Tensor | None = None,
) -> RawGainForwardResult:
    """Run the raw forward gain kernels without creating an autograd boundary.

    ``k`` is FP32 or BF16 ``[B,T,H,K]``.  ``alpha`` is FP32 with the same
    shape; ``omega`` and ``r`` are FP32 ``[B,T,H,V]``.  ``prior`` is FP32
    ``[H,V]`` or ``[B,H,V]`` and an optional FP32 ``initial_precision``
    ``[B,H,V]`` supersedes it.  The information scale is a positive Python
    scalar, a scalar FP32 tensor, a per-head ``[H]`` FP32 tensor, or ``None``
    (which resolves to K).  Tensor scale values are never read on the host.

    This is a private launch helper.  The future public operator owns the sole
    custom autograd Function and will call this helper from its forward.
    """

    inputs = _validate_forward_inputs(
        k,
        alpha,
        omega,
        r,
        prior,
        initial_precision,
        information_scale,
    )
    maps, statistics = _launch_forward_map(inputs)
    final_precision = _launch_forward_carry(inputs, maps)
    beta = _launch_forward_emit(inputs, maps, statistics)
    return RawGainForwardResult(
        beta=beta,
        maps_and_carries=maps,
        statistics=statistics,
        final_precision=final_precision,
        plan=inputs.plan,
        scale_kind=inputs.scale_kind,
        precision_source_kind=inputs.precision_source_kind,
        initial_batch_stride=inputs.initial_batch_stride,
    )


def _launch_copy_or_zero(
    source: torch.Tensor,
    destination: torch.Tensor,
    *,
    grid: tuple[int],
    copy_source: bool,
) -> None:
    """Launch one accounted copy/zero operation without allocating scratch."""

    _require_triton()
    assert _value_iso_gain_copy_or_zero_kernel is not None
    with torch.cuda.device(destination.device):
        _value_iso_gain_copy_or_zero_kernel[grid](
            source,
            destination,
            destination.numel(),
            COPY_SOURCE=copy_source,
            BLOCK_SIZE=GAIN_COPY_BLOCK_SIZE,
            num_warps=GAIN_NUM_WARPS,
            num_stages=GAIN_NUM_STAGES,
        )


def _launch_gain_reverse(
    inputs: _ValidatedReverseInputs,
    replay: torch.Tensor,
    dc: torch.Tensor,
    ds_stream: torch.Tensor | None,
    dscale: torch.Tensor | None,
) -> None:
    """Launch persistent reverse and its optional fixed-order scale reducer."""

    forward = inputs.forward
    plan = forward.plan
    reverse_plan = inputs.reverse_plan
    _require_triton()
    assert _value_iso_gain_reverse_kernel is not None
    scale_pointer = _scale_pointer(forward, inputs.statistics)
    final_dc_pointer = (
        inputs.final_precision_cotangent
        if inputs.final_precision_cotangent is not None
        else dc
    )
    ds_pointer = ds_stream if ds_stream is not None else dc
    dbeta_pointer = (
        inputs.dbeta_domega if inputs.dbeta_domega is not None else dc
    )
    dk_pointer = inputs.dk_acc if inputs.dk_acc is not None else dc
    dalpha_pointer = inputs.dalpha if inputs.dalpha is not None else dc
    dr_pointer = inputs.dr if inputs.dr is not None else dc
    with torch.cuda.device(forward.k.device):
        _value_iso_gain_reverse_kernel[reverse_plan.reverse_grid](
            forward.k,
            forward.alpha,
            forward.omega,
            forward.r,
            scale_pointer,
            inputs.maps_and_carries,
            inputs.statistics,
            dbeta_pointer,
            dk_pointer,
            dalpha_pointer,
            dr_pointer,
            final_dc_pointer,
            replay,
            dc,
            ds_pointer,
            plan.total_tokens,
            plan.chunk_count,
            forward.scale_value,
            H=plan.heads,
            K=plan.key_dim,
            V=plan.value_dim,
            BK=plan.block_key,
            BV=plan.block_value,
            CG=plan.chunk_size,
            SCALE_KIND=forward.scale_kind_code,
            HAS_DFINAL=inputs.final_precision_cotangent is not None,
            HAS_DS_STREAM=ds_stream is not None,
            HAS_DBETA=inputs.dbeta_domega is not None,
            WRITE_DK=inputs.dk_acc is not None,
            WRITE_DALPHA=inputs.dalpha is not None,
            WRITE_DOMEGA=inputs.dbeta_domega is not None,
            WRITE_DR=inputs.dr is not None,
            num_warps=plan.num_warps,
            num_stages=plan.num_stages,
        )

        if reverse_plan.scale_reducer_launch_count:
            assert _value_iso_gain_scale_reduce_kernel is not None
            assert reverse_plan.scale_grid is not None
            assert ds_stream is not None and dscale is not None
            _value_iso_gain_scale_reduce_kernel[reverse_plan.scale_grid](
                ds_stream,
                dscale,
                plan.batch_size,
                plan.heads,
                SCALE_KIND=forward.scale_kind_code,
                num_warps=plan.num_warps,
                num_stages=plan.num_stages,
            )


def _launch_state_gradient_outputs(
    inputs: _ValidatedReverseInputs,
    dc: torch.Tensor,
    dprior: torch.Tensor | None,
    dinitial_precision: torch.Tensor | None,
) -> None:
    """Materialize distinct prior/initial outputs after dc is complete."""

    forward = inputs.forward
    plan = forward.plan
    reverse_plan = inputs.reverse_plan
    if reverse_plan.prior_reducer_launch_count:
        assert dprior is not None and reverse_plan.prior_grid is not None
        _require_triton()
        assert _value_iso_gain_prior_reduce_kernel is not None
        with torch.cuda.device(forward.k.device):
            _value_iso_gain_prior_reduce_kernel[reverse_plan.prior_grid](
                dc,
                dprior,
                plan.batch_size,
                H=plan.heads,
                V=plan.value_dim,
                BV=plan.block_value,
                num_warps=plan.num_warps,
                num_stages=plan.num_stages,
            )
    elif reverse_plan.prior_copy_launch_count:
        assert dprior is not None and reverse_plan.prior_grid is not None
        _launch_copy_or_zero(
            dc, dprior, grid=reverse_plan.prior_grid, copy_source=True
        )
    elif reverse_plan.prior_zero_launch_count:
        assert dprior is not None and reverse_plan.prior_grid is not None
        _launch_copy_or_zero(
            dprior, dprior, grid=reverse_plan.prior_grid, copy_source=False
        )

    if reverse_plan.initial_copy_launch_count:
        assert dinitial_precision is not None
        assert reverse_plan.initial_copy_grid is not None
        _launch_copy_or_zero(
            dc,
            dinitial_precision,
            grid=reverse_plan.initial_copy_grid,
            copy_source=True,
        )


@torch.compiler.disable
def raw_value_iso_gain_backward(
    k: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    prior: torch.Tensor,
    maps_and_carries: torch.Tensor,
    statistics: torch.Tensor,
    dbeta_domega: torch.Tensor | None,
    dk_acc: torch.Tensor | None,
    dalpha: torch.Tensor | None,
    dr: torch.Tensor | None,
    initial_precision: torch.Tensor | None = None,
    information_scale: float | torch.Tensor | None = None,
    final_precision_cotangent: torch.Tensor | None = None,
    return_prior_gradient: bool = False,
    return_initial_precision_gradient: bool = False,
    return_information_scale_gradient: bool = False,
) -> RawGainBackwardResult:
    """Run the raw persistent gain reverse below an autograd boundary.

    ``dbeta_domega`` is the sole mutable gain-gradient formal: when present,
    the incoming beta cotangent is loaded and the same storage may be
    overwritten with omega's gradient.  ``dk_acc`` and ``dalpha`` are FP32
    accumulation destinations and retain their existing values; ``dr`` is a
    direct-store destination.  Unrequested destinations may be ``None`` and
    are removed by compile-time write flags.  Present destinations are caller
    owned and must be mutually disjoint.
    Prior, initial-precision, and tensor-scale gradients are allocated only
    when their exact ``return_*_gradient`` flag is true.
    """

    inputs = _validate_reverse_inputs(
        k,
        alpha,
        omega,
        r,
        prior,
        maps_and_carries,
        statistics,
        dbeta_domega,
        dk_acc,
        dalpha,
        dr,
        initial_precision,
        information_scale,
        final_precision_cotangent,
        return_prior_gradient=return_prior_gradient,
        return_initial_precision_gradient=return_initial_precision_gradient,
        return_information_scale_gradient=return_information_scale_gradient,
    )
    forward = inputs.forward
    plan = forward.plan
    reverse_plan = inputs.reverse_plan

    dc = torch.empty(
        reverse_plan.carry_shape,
        dtype=torch.float32,
        device=forward.k.device,
    )
    dprior = (
        torch.empty_like(
            forward.prior, memory_format=torch.contiguous_format
        )
        if reverse_plan.return_prior_gradient
        else None
    )
    dinitial_precision = (
        torch.empty_like(
            forward.initial_precision, memory_format=torch.contiguous_format
        )
        if reverse_plan.return_initial_precision_gradient
        else None
    )
    dscale = (
        torch.empty_like(
            forward.scale_tensor, memory_format=torch.contiguous_format
        )
        if reverse_plan.return_information_scale_gradient
        else None
    )

    if plan.total_tokens == 0:
        assert reverse_plan.dc_init_grid is not None
        dc_source = (
            inputs.final_precision_cotangent
            if inputs.final_precision_cotangent is not None
            else dc
        )
        _launch_copy_or_zero(
            dc_source,
            dc,
            grid=reverse_plan.dc_init_grid,
            copy_source=inputs.final_precision_cotangent is not None,
        )
        if reverse_plan.scale_zero_launch_count:
            assert dscale is not None and reverse_plan.scale_grid is not None
            _launch_copy_or_zero(
                dscale,
                dscale,
                grid=reverse_plan.scale_grid,
                copy_source=False,
            )
    else:
        replay = torch.empty(
            reverse_plan.replay_shape,
            dtype=torch.float32,
            device=forward.k.device,
        )
        ds_stream = (
            torch.empty(
                reverse_plan.scale_stream_shape,
                dtype=torch.float32,
                device=forward.k.device,
            )
            if reverse_plan.scale_stream_shape is not None
            else None
        )
        _launch_gain_reverse(inputs, replay, dc, ds_stream, dscale)

    _launch_state_gradient_outputs(
        inputs, dc, dprior, dinitial_precision
    )

    return RawGainBackwardResult(
        dk_acc=inputs.dk_acc,
        dalpha=inputs.dalpha,
        domega=inputs.dbeta_domega,
        dr=inputs.dr,
        dprior=dprior,
        dinitial_precision=dinitial_precision,
        dinformation_scale=dscale,
        plan=reverse_plan,
    )


__all__ = [
    "GAIN_CHUNK_SIZE",
    "GAIN_COPY_BLOCK_SIZE",
    "GAIN_NUM_STAGES",
    "GAIN_NUM_WARPS",
    "GainForwardPlan",
    "GainReversePlan",
    "RawGainForwardResult",
    "RawGainBackwardResult",
    "gain_forward_plan",
    "gain_reverse_plan",
    "raw_value_iso_gain_backward",
    "raw_value_iso_gain_forward",
]
