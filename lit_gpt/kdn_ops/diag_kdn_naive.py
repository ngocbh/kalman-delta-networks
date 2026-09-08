"""Independent PyTorch oracle for Diagonal Kalman Delta Networks.

This module contains only the mathematical gain and memory recurrences. Inputs
are the already-formed query, key, value, retention, process-noise, and
observation-noise tensors; normalization and projection frontends deliberately
remain outside the oracle. The private chunk path uses independent Möbius and
WY decompositions and never calls the tokenwise implementations.
"""
from __future__ import annotations

import math
from numbers import Real

import torch


_FLOAT_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)


def _check_tensor(name: str, value: torch.Tensor, *, ndim: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {tuple(value.shape)}")
    if value.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"{name} must have a floating dtype, got {value.dtype}")


def _check_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} must contain only finite values")


def _compute_dtype(*values: object) -> torch.dtype:
    return (
        torch.float64
        if any(
            isinstance(value, torch.Tensor) and value.dtype == torch.float64
            for value in values
        )
        else torch.float32
    )


def _broadcast_initial_precision(
    initial_precision: Real | torch.Tensor,
    *,
    batch: int,
    heads: int,
    key_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(initial_precision, bool):
        raise TypeError("initial_precision must be a positive real or floating tensor")
    if isinstance(initial_precision, Real):
        value = float(initial_precision)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("initial_precision must be positive and finite")
        return torch.full(
            (batch, heads, key_dim), value, device=device, dtype=dtype
        )
    if not isinstance(initial_precision, torch.Tensor):
        raise TypeError("initial_precision must be a positive real or floating tensor")
    if initial_precision.dtype not in _FLOAT_DTYPES:
        raise TypeError("initial_precision must have a floating dtype")
    if initial_precision.device != device:
        raise ValueError("initial_precision must be on the same device as k")
    if initial_precision.ndim == 0:
        result = initial_precision.to(dtype).expand(batch, heads, key_dim)
    elif initial_precision.shape == (heads,):
        result = initial_precision.to(dtype).view(1, heads, 1).expand(
            batch, heads, key_dim
        )
    elif initial_precision.shape == (batch, heads):
        result = initial_precision.to(dtype).view(batch, heads, 1).expand(
            batch, heads, key_dim
        )
    elif initial_precision.shape == (batch, heads, key_dim):
        result = initial_precision.to(dtype)
    else:
        raise ValueError(
            "initial_precision must be scalar, [H], [B,H], or [B,H,K]; "
            f"got {tuple(initial_precision.shape)}"
        )
    _check_finite("initial_precision", result)
    if not bool((result.detach() > 0).all()):
        raise ValueError("initial_precision must be positive and finite")
    return result


def _broadcast_info_scale(
    info_scale: Real | torch.Tensor | None,
    *,
    heads: int,
    key_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if info_scale is None:
        return torch.full((heads,), float(key_dim), device=device, dtype=dtype)
    if isinstance(info_scale, bool):
        raise TypeError("info_scale must be a positive real or floating scalar/[H] tensor")
    if isinstance(info_scale, Real):
        value = float(info_scale)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("info_scale must be positive and finite")
        return torch.full((heads,), value, device=device, dtype=dtype)
    if not isinstance(info_scale, torch.Tensor):
        raise TypeError("info_scale must be a positive real or floating scalar/[H] tensor")
    if info_scale.dtype not in _FLOAT_DTYPES:
        raise TypeError("info_scale must have a floating dtype")
    if info_scale.device != device:
        raise ValueError("info_scale must be on the same device as k")
    if info_scale.ndim == 0:
        result = info_scale.to(dtype).expand(heads)
    elif info_scale.shape == (heads,):
        result = info_scale.to(dtype)
    else:
        raise ValueError(f"info_scale must be scalar or [H]; got {tuple(info_scale.shape)}")
    _check_finite("info_scale", result)
    if not bool((result.detach() > 0).all()):
        raise ValueError("info_scale must be positive and finite")
    return result


def _validate_gain_inputs(
    k: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
) -> tuple[int, int, int, int]:
    for name, value, ndim in (
        ("k", k, 4),
        ("alpha", alpha, 4),
        ("omega", omega, 4),
        ("r", r, 3),
    ):
        _check_tensor(name, value, ndim=ndim)
    if not (k.shape == alpha.shape == omega.shape):
        raise ValueError(
            "k, alpha, and omega must share shape [B,T,H,K]; "
            f"got k={tuple(k.shape)}, alpha={tuple(alpha.shape)}, "
            f"omega={tuple(omega.shape)}"
        )
    batch, length, heads, key_dim = k.shape
    if min(batch, heads, key_dim) <= 0:
        raise ValueError(
            f"B, H, and K must be positive, got B={batch}, H={heads}, K={key_dim}"
        )
    if r.shape != (batch, length, heads):
        raise ValueError(
            f"r must have shape {(batch, length, heads)}, got {tuple(r.shape)}"
        )
    for name, value in (("alpha", alpha), ("omega", omega), ("r", r)):
        if value.device != k.device:
            raise ValueError(f"{name} must be on the same device as k")
    for name, value in (("k", k), ("alpha", alpha), ("omega", omega), ("r", r)):
        _check_finite(name, value)
    if not bool(((alpha.detach() > 0) & (alpha.detach() <= 1)).all()):
        raise ValueError("alpha must be in the interval (0, 1]")
    if not bool((omega.detach() >= 0).all()):
        raise ValueError("omega must be nonnegative and finite")
    if not bool((r.detach() > 0).all()):
        raise ValueError("r must be positive and finite")
    return batch, length, heads, key_dim


def _prepare_gain_inputs(
    k: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    initial_precision: Real | torch.Tensor,
    info_scale: Real | torch.Tensor | None,
) -> tuple[tuple[int, int, int, int], tuple[torch.Tensor, ...]]:
    shape = _validate_gain_inputs(k, alpha, omega, r)
    batch, _length, heads, key_dim = shape
    dtype = _compute_dtype(k, alpha, omega, r, initial_precision, info_scale)
    precision = _broadcast_initial_precision(
        initial_precision,
        batch=batch,
        heads=heads,
        key_dim=key_dim,
        device=k.device,
        dtype=dtype,
    )
    information_scale = _broadcast_info_scale(
        info_scale,
        heads=heads,
        key_dim=key_dim,
        device=k.device,
        dtype=dtype,
    )
    return shape, (
        k.to(dtype),
        alpha.to(dtype),
        omega.to(dtype),
        r.to(dtype),
        precision,
        information_scale,
    )


def _empty_anchor(*values: object) -> torch.Tensor:
    tensors = tuple(value for value in values if isinstance(value, torch.Tensor))
    if not tensors:
        raise RuntimeError("an empty recurrence requires at least one tensor")
    anchor = tensors[0].sum() * 0.0
    for value in tensors[1:]:
        anchor = anchor + value.sum() * 0.0
    return anchor


def diag_kdn_gain_naive(
    k: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    *,
    info_scale: Real | torch.Tensor | None = None,
    initial_precision: Real | torch.Tensor = 1.0,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Compute the Diagonal KDN vector gain with a sequential covariance scan.

    With one covariance coordinate per key channel::

        p_hat_t = alpha_t**2 * p_{t-1} + omega_t
        kappa_t = p_hat_t * k_t / (r_t + sum_i p_hat_ti * k_ti**2)
        c_t = 1 / p_hat_t + info_scale * k_t**2 / r_t

    ``initial_precision`` is ``c_{-1}``. A per-head ``[H]`` tensor models the
    learned production representation and is broadcast across batch and key
    channels. A returned ``[B,H,K]`` precision can be passed back directly for
    continuation. ``info_scale=None`` means the production value ``K``; unlike
    the isotropic recurrence, the posterior increment has no division by K.
    """
    if type(output_final_state) is not bool:
        raise TypeError("output_final_state must be bool")
    shape, prepared = _prepare_gain_inputs(
        k, alpha, omega, r, initial_precision, info_scale
    )
    _batch, length, _heads, _key_dim = shape
    k_work, alpha_work, omega_work, r_work, precision, information_scale = prepared
    if length == 0:
        anchor = _empty_anchor(
            k_work, alpha_work, omega_work, r_work, precision, information_scale
        )
        gain = k_work + anchor
        final_precision = precision + anchor
        return gain, final_precision if output_final_state else None

    covariance = precision.reciprocal()
    gains: list[torch.Tensor] = []
    scale_by_head = information_scale.view(1, -1, 1)
    with torch.autocast(device_type=k.device.type, enabled=False):
        for token in range(length):
            predicted = alpha_work[:, token].square() * covariance + omega_work[:, token]
            denominator = r_work[:, token, :, None] + (
                predicted * k_work[:, token].square()
            ).sum(dim=-1, keepdim=True)
            gains.append(predicted * k_work[:, token] / denominator)
            information_weight = (
                scale_by_head
                * k_work[:, token].square()
                / r_work[:, token, :, None]
            )
            covariance = predicted / (1.0 + information_weight * predicted)
    final_precision = covariance.reciprocal()
    return (
        torch.stack(gains, dim=1),
        final_precision if output_final_state else None,
    )


def _mobius_apply(matrix: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    numerator = matrix[..., 0, 0] * covariance + matrix[..., 0, 1]
    denominator = matrix[..., 1, 0] * covariance + matrix[..., 1, 1]
    return numerator / denominator


def _chunk_diag_kdn_gain(
    k: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    *,
    info_scale: Real | torch.Tensor | None = None,
    initial_precision: Real | torch.Tensor = 1.0,
    output_final_state: bool = False,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Private gain oracle using per-channel covariance Möbius composition."""
    if type(output_final_state) is not bool:
        raise TypeError("output_final_state must be bool")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    shape, prepared = _prepare_gain_inputs(
        k, alpha, omega, r, initial_precision, info_scale
    )
    batch, length, heads, key_dim = shape
    k_work, alpha_work, omega_work, r_work, precision, information_scale = prepared
    if length == 0:
        anchor = _empty_anchor(
            k_work, alpha_work, omega_work, r_work, precision, information_scale
        )
        gain = k_work + anchor
        final_precision = precision + anchor
        return gain, final_precision if output_final_state else None

    covariance_at_chunk_start = precision.reciprocal()
    scale_by_head = information_scale.view(1, heads, 1)
    gain_parts: list[torch.Tensor] = []
    with torch.autocast(device_type=k.device.type, enabled=False):
        for start in range(0, length, chunk_size):
            stop = min(start + chunk_size, length)
            identity = torch.eye(2, dtype=k_work.dtype, device=k.device)
            prefix = identity.view(1, 1, 1, 2, 2).expand(
                batch, heads, key_dim, 2, 2
            )
            local_gains: list[torch.Tensor] = []
            for token in range(start, stop):
                previous_covariance = _mobius_apply(
                    prefix, covariance_at_chunk_start
                )
                alpha_squared = alpha_work[:, token].square()
                predicted = (
                    alpha_squared * previous_covariance + omega_work[:, token]
                )
                denominator = r_work[:, token, :, None] + (
                    predicted * k_work[:, token].square()
                ).sum(dim=-1, keepdim=True)
                local_gains.append(predicted * k_work[:, token] / denominator)

                information_weight = (
                    scale_by_head
                    * k_work[:, token].square()
                    / r_work[:, token, :, None]
                )
                token_map = torch.stack(
                    (
                        torch.stack((alpha_squared, omega_work[:, token]), dim=-1),
                        torch.stack(
                            (
                                information_weight * alpha_squared,
                                1.0
                                + information_weight * omega_work[:, token],
                            ),
                            dim=-1,
                        ),
                    ),
                    dim=-2,
                )
                prefix = torch.matmul(token_map, prefix)
                prefix = prefix / prefix.abs().amax(
                    dim=(-2, -1), keepdim=True
                ).clamp_min(torch.finfo(prefix.dtype).tiny)
            gain_parts.append(torch.stack(local_gains, dim=1))
            covariance_at_chunk_start = _mobius_apply(
                prefix, covariance_at_chunk_start
            )
    final_precision = covariance_at_chunk_start.reciprocal()
    return (
        torch.cat(gain_parts, dim=1),
        final_precision if output_final_state else None,
    )


def _validate_full_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    initial_memory: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    _check_tensor("q", q, ndim=4)
    _check_tensor("k", k, ndim=4)
    _check_tensor("alpha", alpha, ndim=4)
    _check_tensor("v", v, ndim=4)
    if not (q.shape == k.shape == alpha.shape):
        raise ValueError(
            "q, k, and alpha must share shape [B,T,H,K]; "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, alpha={tuple(alpha.shape)}"
        )
    batch, length, heads, key_dim = _validate_gain_inputs(k, alpha, omega, r)
    if v.shape[:3] != (batch, length, heads):
        raise ValueError(
            f"v must have leading shape {(batch, length, heads)}, "
            f"got {tuple(v.shape[:3])}"
        )
    value_dim = v.shape[-1]
    if value_dim <= 0:
        raise ValueError(f"V must be positive, got V={value_dim}")
    for name, value in (("q", q), ("v", v)):
        if value.device != k.device:
            raise ValueError(f"{name} must be on the same device as k")
        _check_finite(name, value)
    if initial_memory is not None:
        _check_tensor("initial_memory", initial_memory, ndim=4)
        expected = (batch, heads, key_dim, value_dim)
        if initial_memory.shape != expected:
            raise ValueError(
                f"initial_memory must have shape {expected}, "
                f"got {tuple(initial_memory.shape)}"
            )
        if initial_memory.device != k.device:
            raise ValueError("initial_memory must be on the same device as q")
        _check_finite("initial_memory", initial_memory)
    return batch, length, heads, key_dim, value_dim


def _resolve_output_scale(scale: Real | None, key_dim: int) -> float:
    if scale is None:
        return float(key_dim) ** -0.5
    if isinstance(scale, bool) or not isinstance(scale, Real):
        raise TypeError("scale must be a positive finite real or None")
    resolved = float(scale)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError("scale must be positive and finite")
    return resolved


def _prepare_full_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    initial_memory: torch.Tensor | None,
    initial_precision: Real | torch.Tensor,
    info_scale: Real | torch.Tensor | None,
) -> tuple[tuple[int, int, int, int, int], tuple[torch.Tensor, ...]]:
    shape = _validate_full_inputs(q, k, v, alpha, omega, r, initial_memory)
    batch, _length, heads, key_dim, value_dim = shape
    dtype = _compute_dtype(
        q, k, v, alpha, omega, r, initial_memory, initial_precision, info_scale
    )
    precision = _broadcast_initial_precision(
        initial_precision,
        batch=batch,
        heads=heads,
        key_dim=key_dim,
        device=q.device,
        dtype=dtype,
    )
    information_scale = _broadcast_info_scale(
        info_scale,
        heads=heads,
        key_dim=key_dim,
        device=q.device,
        dtype=dtype,
    )
    memory = (
        torch.zeros(
            (batch, heads, key_dim, value_dim), device=q.device, dtype=dtype
        )
        if initial_memory is None
        else initial_memory.to(dtype)
    )
    return shape, (
        q.to(dtype),
        k.to(dtype),
        v.to(dtype),
        alpha.to(dtype),
        omega.to(dtype),
        r.to(dtype),
        memory,
        precision,
        information_scale,
    )


def _memory_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    gain: torch.Tensor,
    memory: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs: list[torch.Tensor] = []
    for token in range(q.shape[1]):
        predicted_memory = alpha[:, token, ..., None] * memory
        innovation = v[:, token] - torch.einsum(
            "bhk,bhkv->bhv", k[:, token], predicted_memory
        )
        memory = predicted_memory + gain[:, token, ..., None] * innovation[
            ..., None, :
        ]
        outputs.append(
            scale * torch.einsum("bhk,bhkv->bhv", q[:, token], memory)
        )
    return torch.stack(outputs, dim=1), memory


def naive_recurrent_diag_kdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    *,
    info_scale: Real | torch.Tensor | None = None,
    scale: Real | None = None,
    initial_memory: torch.Tensor | None = None,
    initial_precision: Real | torch.Tensor = 1.0,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Run tokenwise Diagonal KDN gain, memory, and readout recurrences."""
    if type(output_final_state) is not bool:
        raise TypeError("output_final_state must be bool")
    shape, prepared = _prepare_full_inputs(
        q, k, v, alpha, omega, r, initial_memory, initial_precision, info_scale
    )
    _batch, length, _heads, key_dim, _value_dim = shape
    (
        q_work,
        k_work,
        v_work,
        alpha_work,
        omega_work,
        r_work,
        memory,
        precision,
        information_scale,
    ) = prepared
    output_scale = _resolve_output_scale(scale, key_dim)
    if length == 0:
        anchor = _empty_anchor(
            q_work,
            k_work,
            v_work,
            alpha_work,
            omega_work,
            r_work,
            memory,
            precision,
            information_scale,
        )
        output = (v_work + anchor).to(v.dtype)
        state = (memory + anchor, precision + anchor) if output_final_state else None
        return output, state

    gain, final_precision = diag_kdn_gain_naive(
        k_work,
        alpha_work,
        omega_work,
        r_work,
        info_scale=information_scale,
        initial_precision=precision,
        output_final_state=True,
    )
    assert final_precision is not None
    with torch.autocast(device_type=q.device.type, enabled=False):
        output, memory = _memory_recurrence(
            q_work, k_work, v_work, alpha_work, gain, memory, output_scale
        )
    state = (memory, final_precision) if output_final_state else None
    return output.to(v.dtype), state


def _chunk_memory_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    gain: torch.Tensor,
    memory: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate memory and readout with an independent per-channel WY transform."""
    outputs: list[torch.Tensor] = []
    for start in range(0, q.shape[1], chunk_size):
        stop = min(start + chunk_size, q.shape[1])
        local_length = stop - start
        q_chunk = q[:, start:stop].permute(0, 2, 1, 3) * scale
        k_chunk = k[:, start:stop].permute(0, 2, 1, 3)
        v_chunk = v[:, start:stop].permute(0, 2, 1, 3)
        gain_chunk = gain[:, start:stop].permute(0, 2, 1, 3)
        cumulative_gate = alpha[:, start:stop].log().permute(0, 2, 1, 3).cumsum(dim=2)

        interaction_columns: list[torch.Tensor] = []
        readout_columns: list[torch.Tensor] = []
        for source in range(local_length):
            after = torch.einsum(
                "bhck,bhk->bhc",
                k_chunk[:, :, source + 1 :]
                * (
                    cumulative_gate[:, :, source + 1 :]
                    - cumulative_gate[:, :, source : source + 1]
                ).exp(),
                gain_chunk[:, :, source],
            )
            interaction_columns.append(
                torch.cat(
                    (
                        k_chunk.new_zeros(*k_chunk.shape[:2], source + 1),
                        after,
                    ),
                    dim=-1,
                )
            )
            retained = torch.einsum(
                "bhck,bhk->bhc",
                q_chunk[:, :, source:]
                * (
                    cumulative_gate[:, :, source:]
                    - cumulative_gate[:, :, source : source + 1]
                ).exp(),
                gain_chunk[:, :, source],
            )
            readout_columns.append(
                torch.cat(
                    (
                        q_chunk.new_zeros(*q_chunk.shape[:2], source),
                        retained,
                    ),
                    dim=-1,
                )
            )
        interaction = torch.stack(interaction_columns, dim=-1)
        readout = torch.stack(readout_columns, dim=-1)
        gate_from_start = cumulative_gate.exp()
        transformed_values = torch.linalg.solve_triangular(
            interaction, v_chunk, upper=False, unitriangular=True
        )
        transformed_keys = torch.linalg.solve_triangular(
            interaction,
            gate_from_start * k_chunk,
            upper=False,
            unitriangular=True,
        )
        corrected_values = transformed_values - transformed_keys @ memory
        output_chunk = (
            (q_chunk * gate_from_start) @ memory + readout @ corrected_values
        )
        outputs.append(output_chunk.permute(0, 2, 1, 3))

        final_gate = cumulative_gate[:, :, -1:]
        memory = memory * final_gate[:, :, 0, :, None].exp()
        memory = memory + torch.einsum(
            "bhck,bhcv->bhkv",
            (final_gate - cumulative_gate).exp() * gain_chunk,
            corrected_values,
        )
    return torch.cat(outputs, dim=1), memory


def _naive_chunk_diag_kdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    *,
    info_scale: Real | torch.Tensor | None = None,
    scale: Real | None = None,
    initial_memory: torch.Tensor | None = None,
    initial_precision: Real | torch.Tensor = 1.0,
    output_final_state: bool = False,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Private chunked PyTorch oracle with independent gain and memory scans."""
    if type(output_final_state) is not bool:
        raise TypeError("output_final_state must be bool")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    shape, prepared = _prepare_full_inputs(
        q, k, v, alpha, omega, r, initial_memory, initial_precision, info_scale
    )
    _batch, length, _heads, key_dim, _value_dim = shape
    (
        q_work,
        k_work,
        v_work,
        alpha_work,
        omega_work,
        r_work,
        memory,
        precision,
        information_scale,
    ) = prepared
    output_scale = _resolve_output_scale(scale, key_dim)
    if length == 0:
        anchor = _empty_anchor(
            q_work,
            k_work,
            v_work,
            alpha_work,
            omega_work,
            r_work,
            memory,
            precision,
            information_scale,
        )
        output = (v_work + anchor).to(v.dtype)
        state = (memory + anchor, precision + anchor) if output_final_state else None
        return output, state

    gain, final_precision = _chunk_diag_kdn_gain(
        k_work,
        alpha_work,
        omega_work,
        r_work,
        info_scale=information_scale,
        initial_precision=precision,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    assert final_precision is not None
    with torch.autocast(device_type=q.device.type, enabled=False):
        output, memory = _chunk_memory_recurrence(
            q_work,
            k_work,
            v_work,
            alpha_work,
            gain,
            memory,
            output_scale,
            chunk_size,
        )
    state = (memory, final_precision) if output_final_state else None
    return output.to(v.dtype), state


__all__ = ("diag_kdn_gain_naive", "naive_recurrent_diag_kdn")
