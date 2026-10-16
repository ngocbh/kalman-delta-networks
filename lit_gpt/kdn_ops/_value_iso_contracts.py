"""Frozen host-side input contract for value-wise isotropic KDN kernels.

This module performs only canonicalization and validation.  In particular, it
contains no recurrence and imports no Triton/CUDA implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

import torch


BACKENDS = frozenset({"auto", "reference", "triton_recurrent", "triton_chunk"})
DTYPE_MODES = frozenset({"oracle", "fp32", "production"})
_TRITON_BACKENDS = frozenset({"triton_recurrent", "triton_chunk"})
_TRITON_KEY_WIDTHS = frozenset({64, 128})
_TRITON_VALUE_WIDTH_MULTIPLE = 8
_TRITON_MAX_VALUE_WIDTH = 256


@dataclass(frozen=True)
class ValidatedInputs:
    """Canonical input bundle shared by the value-Iso-KDN implementations."""

    CUDA_VALUE_VALIDATION_TRANSFER: ClassVar[str] = "value_validation_flags_to_cpu_once"

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    alpha: torch.Tensor
    omega: torch.Tensor
    r: torch.Tensor
    prior: torch.Tensor
    memory: torch.Tensor | None
    precision: torch.Tensor | None
    info_scale: float | torch.Tensor
    read_scale: float
    backend: str
    dtype_mode: str
    compute_dtype: torch.dtype
    B: int
    T: int
    H: int
    K: int
    V: int
    has_initial_state: bool
    prior_was_shared: bool


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _require_shape(tensor: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")


def _require_common_device(
    reference: torch.Tensor, named_tensors: list[tuple[str, torch.Tensor]]
) -> None:
    mismatched = [
        (name, tensor.device)
        for name, tensor in named_tensors
        if tensor.device != reference.device
    ]
    if mismatched:
        details = ", ".join(f"{name}={device}" for name, device in mismatched)
        raise ValueError(
            f"all tensors must share a common device {reference.device}; mismatches: {details}"
        )


def _require_contiguous(named_tensors: list[tuple[str, torch.Tensor]]) -> None:
    for name, tensor in named_tensors:
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def _dtype_profile(
    dtype_mode: str,
    named_tensors: list[tuple[str, torch.Tensor]],
    *,
    info_scale: float | torch.Tensor,
) -> torch.dtype:
    if dtype_mode == "oracle":
        expected = {name: torch.float64 for name, _ in named_tensors}
        compute_dtype = torch.float64
    elif dtype_mode == "fp32":
        expected = {name: torch.float32 for name, _ in named_tensors}
        compute_dtype = torch.float32
    else:
        expected = {
            name: (torch.bfloat16 if name in {"q", "k", "v"} else torch.float32)
            for name, _ in named_tensors
        }
        compute_dtype = torch.float32

    bad = [
        f"{name}={tensor.dtype} (expected {expected[name]})"
        for name, tensor in named_tensors
        if tensor.dtype != expected[name]
    ]
    if bad:
        raise TypeError(f"{dtype_mode} dtype profile violated: " + ", ".join(bad))

    if isinstance(info_scale, torch.Tensor) and info_scale.dtype != compute_dtype:
        raise TypeError(
            f"info_scale must have dtype {compute_dtype} in {dtype_mode} mode, "
            f"got {info_scale.dtype}"
        )
    return compute_dtype


def _python_info_scale(value: object) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise TypeError("info_scale must be a Python int/float or a tensor")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("Python info_scale must be finite and strictly positive")
    return result


def _python_read_scale(value: object) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise TypeError("read_scale must be a finite Python int or float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("read_scale must be finite")
    return result


def _validate_explicit_triton_contract(
    *,
    backend: str,
    dtype_mode: str,
    device: torch.device,
    K: int,
    V: int,
) -> None:
    """Reject inputs for which an explicitly selected Triton backend has no kernel."""

    if backend not in _TRITON_BACKENDS:
        return
    if dtype_mode == "oracle":
        raise ValueError(f"{backend} does not support dtype_mode='oracle'")
    if K not in _TRITON_KEY_WIDTHS:
        raise ValueError(f"{backend} K must be exactly 64 or 128, got {K}")
    if V > _TRITON_MAX_VALUE_WIDTH:
        raise ValueError(
            f"{backend} V must be at most {_TRITON_MAX_VALUE_WIDTH}, got {V}"
        )
    if V % _TRITON_VALUE_WIDTH_MULTIPLE != 0:
        raise ValueError(
            f"{backend} V must be divisible by {_TRITON_VALUE_WIDTH_MULTIPLE}, got {V}"
        )
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            f"{backend} does not support PyTorch deterministic algorithms mode"
        )
    if device.type != "cuda":
        raise ValueError(f"{backend} requires CUDA tensors, got device={device}")


def _value_validation_flags_to_cpu_once(flags: torch.Tensor) -> torch.Tensor:
    """Perform the validation path's sole CUDA-to-host transfer and sync."""

    return flags.to(device="cpu", non_blocking=False)


def _value_validation_flags_need_host_transfer(flags: torch.Tensor) -> bool:
    """Return whether aggregated validation flags need the CUDA host-transfer hook.

    Keeping this metadata-only decision separate makes the one-transfer branch
    exercisable in host tests without misrepresenting a CPU tensor as CUDA.
    """

    return flags.is_cuda


def _validate_tensor_values(
    finite_tensors: list[tuple[str, torch.Tensor]],
    positive_tensors: list[tuple[str, torch.Tensor]],
    nonnegative_tensors: list[tuple[str, torch.Tensor]],
) -> None:
    names: list[str] = []
    flags: list[torch.Tensor] = []
    for name, tensor in finite_tensors:
        names.append(f"{name} must contain only finite values")
        flags.append(torch.isfinite(tensor).all())
    for name, tensor in positive_tensors:
        names.append(f"{name} must be strictly positive")
        flags.append((tensor > 0).all())
    for name, tensor in nonnegative_tensors:
        names.append(f"{name} must be nonnegative")
        flags.append((tensor >= 0).all())

    combined = torch.stack(flags)
    if _value_validation_flags_need_host_transfer(combined):
        combined = _value_validation_flags_to_cpu_once(combined)
    failed = [name for name, passed in zip(names, combined.tolist()) if not passed]
    if failed:
        raise ValueError("; ".join(failed))


def validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    prior: torch.Tensor,
    *,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    info_scale: float | torch.Tensor = 1.0,
    read_scale: float = 1.0,
    backend: str = "auto",
    dtype_mode: str = "oracle",
    validate_values: bool = False,
) -> ValidatedInputs:
    """Validate and canonicalize the frozen value-wise isotropic KDN ABI.

    Structural checks are constant-time metadata checks. Tensor-wide
    finiteness/domain scans are performed only when ``validate_values=True``.
    A shared ``prior [H,V]`` is expanded as a stride-zero view so autograd
    naturally sums its gradient over the batch. Missing recurrent state remains
    absent; the selected operator owns lazy output-state materialization.
    """

    if not isinstance(backend, str) or backend not in BACKENDS:
        choices = ", ".join(sorted(BACKENDS))
        raise ValueError(f"backend must be one of {choices}, got {backend!r}")
    if not isinstance(dtype_mode, str) or dtype_mode not in DTYPE_MODES:
        choices = ", ".join(sorted(DTYPE_MODES))
        raise ValueError(f"dtype_mode must be one of {choices}, got {dtype_mode!r}")
    if not isinstance(validate_values, bool):
        raise TypeError("validate_values must be bool")

    q = _require_tensor(q, "q")
    k = _require_tensor(k, "k")
    v = _require_tensor(v, "v")
    alpha = _require_tensor(alpha, "alpha")
    omega = _require_tensor(omega, "omega")
    r = _require_tensor(r, "r")
    prior = _require_tensor(prior, "prior")

    if q.ndim != 4:
        raise ValueError(f"q must have shape [B,T,H,K], got {tuple(q.shape)}")
    B, T, H, K = q.shape
    if B <= 0 or H <= 0 or K <= 0:
        raise ValueError(f"B, H, and K must be positive; got B={B}, H={H}, K={K}")
    # Tensor extents cannot be negative, but state the T contract explicitly.
    if T < 0:
        raise ValueError(f"T must be nonnegative, got {T}")
    _require_shape(k, (B, T, H, K), "k")
    _require_shape(alpha, (B, T, H, K), "alpha")
    if v.ndim != 4 or tuple(v.shape[:3]) != (B, T, H) or v.shape[-1] <= 0:
        raise ValueError(
            f"v must have shape [B,T,H,V] with V > 0; got {tuple(v.shape)}"
        )
    V = v.shape[-1]
    _require_shape(omega, (B, T, H, V), "omega")
    _require_shape(r, (B, T, H, V), "r")

    shared_prior = tuple(prior.shape) == (H, V)
    if not shared_prior and tuple(prior.shape) != (B, H, V):
        raise ValueError(
            f"prior must have shape {(H, V)} or {(B, H, V)}, got {tuple(prior.shape)}"
        )

    if initial_state is not None:
        if type(initial_state) is not tuple or len(initial_state) != 2:
            raise TypeError("initial_state must be None or a (memory, precision) tensor tuple")
        memory = _require_tensor(initial_state[0], "initial_state memory")
        precision = _require_tensor(initial_state[1], "initial_state precision")
        _require_shape(memory, (B, H, K, V), "initial_state memory")
        _require_shape(precision, (B, H, V), "initial_state precision")
        has_initial_state = True
    else:
        memory = None
        precision = None
        has_initial_state = False

    if isinstance(info_scale, torch.Tensor):
        if info_scale.ndim not in (0, 1) or (
            info_scale.ndim == 1 and tuple(info_scale.shape) != (H,)
        ):
            raise ValueError(
                f"info_scale must be a scalar or have shape {(H,)}, got {tuple(info_scale.shape)}"
            )
    else:
        info_scale = _python_info_scale(info_scale)
    read_scale = _python_read_scale(read_scale)

    tensor_inputs = [
        ("q", q),
        ("k", k),
        ("v", v),
        ("alpha", alpha),
        ("omega", omega),
        ("r", r),
        ("prior", prior),
    ]
    if memory is not None and precision is not None:
        tensor_inputs.extend(
            [
                ("initial_state memory", memory),
                ("initial_state precision", precision),
            ]
        )
    if isinstance(info_scale, torch.Tensor):
        tensor_inputs.append(("info_scale", info_scale))

    _require_common_device(q, tensor_inputs)
    _require_contiguous(tensor_inputs)
    compute_dtype = _dtype_profile(dtype_mode, tensor_inputs, info_scale=info_scale)
    _validate_explicit_triton_contract(
        backend=backend,
        dtype_mode=dtype_mode,
        device=q.device,
        K=K,
        V=V,
    )

    canonical_prior = prior.unsqueeze(0).expand(B, -1, -1) if shared_prior else prior

    if validate_values:
        finite_tensors = [
            ("q", q),
            ("k", k),
            ("v", v),
            ("alpha", alpha),
            ("omega", omega),
            ("r", r),
            ("prior", prior),
        ]
        positive_tensors = [
            ("alpha", alpha),
            ("r", r),
            ("prior", prior),
        ]
        if memory is not None and precision is not None:
            finite_tensors.extend(
                [
                    ("initial_state memory", memory),
                    ("initial_state precision", precision),
                ]
            )
            positive_tensors.append(("initial_state precision", precision))
        if isinstance(info_scale, torch.Tensor):
            finite_tensors.append(("info_scale", info_scale))
            positive_tensors.append(("info_scale", info_scale))
        _validate_tensor_values(
            finite_tensors,
            positive_tensors,
            [("omega", omega)],
        )

    return ValidatedInputs(
        q=q,
        k=k,
        v=v,
        alpha=alpha,
        omega=omega,
        r=r,
        prior=canonical_prior,
        memory=memory,
        precision=precision,
        info_scale=info_scale,
        read_scale=read_scale,
        backend=backend,
        dtype_mode=dtype_mode,
        compute_dtype=compute_dtype,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        has_initial_state=has_initial_state,
        prior_was_shared=shared_prior,
    )


__all__ = ["BACKENDS", "DTYPE_MODES", "ValidatedInputs", "validate_inputs"]
