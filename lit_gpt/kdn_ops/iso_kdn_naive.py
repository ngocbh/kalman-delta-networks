"""Independent PyTorch oracle for Isotropic Kalman Delta Networks.

The functions in this module implement the mathematical token recurrence. They
do not import or call the production chunk kernel. Keys are consumed exactly as
provided, so their squared norm appears explicitly in both the Kalman gain and
the posterior information update.
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
        if any(isinstance(value, torch.Tensor) and value.dtype == torch.float64 for value in values)
        else torch.float32
    )


def _broadcast_initial_precision(
    initial_precision: Real | torch.Tensor,
    *,
    batch: int,
    heads: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(initial_precision, bool):
        raise TypeError("initial_precision must be a positive real or floating tensor")
    if isinstance(initial_precision, Real):
        value = float(initial_precision)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("initial_precision must be positive and finite")
        return torch.full((batch, heads), value, device=device, dtype=dtype)
    if not isinstance(initial_precision, torch.Tensor):
        raise TypeError("initial_precision must be a positive real or floating tensor")
    if initial_precision.dtype not in _FLOAT_DTYPES:
        raise TypeError("initial_precision must have a floating dtype")
    if initial_precision.device != device:
        raise ValueError("initial_precision must be on the same device as q")
    if initial_precision.ndim == 0:
        result = initial_precision.to(dtype).expand(batch, heads)
    elif initial_precision.shape == (heads,):
        result = initial_precision.to(dtype).view(1, heads).expand(batch, heads)
    elif initial_precision.shape == (batch, heads):
        result = initial_precision.to(dtype)
    else:
        raise ValueError(
            "initial_precision must be scalar, [H], or [B,H]; "
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
        raise ValueError("info_scale must be on the same device as q")
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
    _check_tensor("k", k, ndim=4)
    _check_tensor("alpha", alpha, ndim=4)
    _check_tensor("omega", omega, ndim=3)
    _check_tensor("r", r, ndim=3)
    if alpha.shape != k.shape:
        raise ValueError(
            "k and alpha must share shape [B,T,H,K]; "
            f"got {tuple(k.shape)} and {tuple(alpha.shape)}"
        )
    batch, length, heads, key_dim = k.shape
    if min(heads, key_dim) <= 0:
        raise ValueError(f"H and K must be positive, got H={heads}, K={key_dim}")
    expected_scalar_shape = (batch, length, heads)
    if omega.shape != expected_scalar_shape:
        raise ValueError(
            f"omega must have shape {expected_scalar_shape}, got {tuple(omega.shape)}"
        )
    if r.shape != expected_scalar_shape:
        raise ValueError(f"r must have shape {expected_scalar_shape}, got {tuple(r.shape)}")
    device = k.device
    for name, value in (("alpha", alpha), ("omega", omega), ("r", r)):
        if value.device != device:
            raise ValueError(f"{name} must be on the same device as k")
    for name, value in (("k", k), ("alpha", alpha), ("omega", omega), ("r", r)):
        _check_finite(name, value)
    if not bool((alpha.detach() > 0).all()):
        raise ValueError("alpha must be positive and finite")
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
) -> tuple[torch.Tensor, ...]:
    batch, _length, heads, key_dim = _validate_gain_inputs(k, alpha, omega, r)
    dtype = _compute_dtype(k, alpha, omega, r, initial_precision, info_scale)
    precision = _broadcast_initial_precision(
        initial_precision,
        batch=batch,
        heads=heads,
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
    return (
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


def iso_kdn_gain_naive(
    k: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    *,
    info_scale: Real | torch.Tensor | None = None,
    initial_precision: Real | torch.Tensor = 1.0,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Compute the Isotropic KDN scalar gain with a sequential precision scan.

    For arbitrary keys, with ``K = k.shape[-1]``::

        a_t       = mean_i(alpha_ti**2)
        b_hat_t   = a_t / c_{t-1} + omega_t
        beta_t    = b_hat_t / (r_t + b_hat_t * ||k_t||**2)
        c_t       = 1 / b_hat_t + info_scale * ||k_t||**2 / (K * r_t)

    ``beta_t`` is the current-token gain and does not use the current posterior
    information scale directly. A shared scale can still affect later gains
    through ``c_{t-1}``. ``info_scale=None`` means the production value ``K``.
    """
    if type(output_final_state) is not bool:
        raise TypeError("output_final_state must be bool")
    batch, length, heads, key_dim = _validate_gain_inputs(k, alpha, omega, r)
    k_work, alpha_work, omega_work, r_work, precision, information_scale = (
        _prepare_gain_inputs(k, alpha, omega, r, initial_precision, info_scale)
    )

    if length == 0:
        anchor = _empty_anchor(
            k_work, alpha_work, omega_work, r_work, precision, information_scale
        )
        beta = k_work.sum(dim=-1) + anchor
        final_precision = precision + anchor
        return beta, final_precision if output_final_state else None

    contraction = alpha_work.square().mean(dim=-1)
    scale_over_key_dim = information_scale.view(1, heads) / float(key_dim)
    betas: list[torch.Tensor] = []
    with torch.autocast(device_type=k.device.type, enabled=False):
        for token in range(length):
            predicted = contraction[:, token] / precision + omega_work[:, token]
            key_norm_squared = k_work[:, token].square().sum(dim=-1)
            beta = predicted / (r_work[:, token] + predicted * key_norm_squared)
            precision = (
                predicted.reciprocal()
                + scale_over_key_dim * key_norm_squared / r_work[:, token]
            )
            betas.append(beta)
    return torch.stack(betas, dim=1), precision if output_final_state else None


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
            f"v must have leading shape {(batch, length, heads)}, got {tuple(v.shape[:3])}"
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
                f"initial_memory must have shape {expected}, got {tuple(initial_memory.shape)}"
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


def _memory_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    memory: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    length = q.shape[1]
    outputs: list[torch.Tensor] = []
    for token in range(length):
        alpha_t = alpha[:, token]
        key_t = k[:, token]
        predicted_memory = alpha_t[..., None] * memory
        innovation = v[:, token] - torch.einsum(
            "bhk,bhkv->bhv", key_t, predicted_memory
        )
        memory = predicted_memory + (
            beta[:, token, ..., None] * key_t
        )[..., None] * innovation[..., None, :]
        outputs.append(
            scale * torch.einsum("bhk,bhkv->bhv", q[:, token], memory)
        )
    return torch.stack(outputs, dim=1), memory


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
        torch.zeros((batch, heads, key_dim, value_dim), device=q.device, dtype=dtype)
        if initial_memory is None
        else initial_memory.to(dtype)
    )
    return shape, (
        q.to(dtype), k.to(dtype), v.to(dtype), alpha.to(dtype),
        omega.to(dtype), r.to(dtype), memory, precision, information_scale,
    )


def naive_recurrent_iso_kdn(
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
    r"""Run the tokenwise Isotropic KDN gain, memory, and readout recurrence.

    ``q``, ``k``, and ``alpha`` are ``[B,T,H,K]``; ``v`` is ``[B,T,H,V]``;
    ``omega`` and ``r`` are ``[B,T,H]``. The function intentionally does not
    normalize q/k. The returned optional state is ``(memory, precision)`` with
    shapes ``[B,H,K,V]`` and ``[B,H]``.
    """
    if type(output_final_state) is not bool:
        raise TypeError("output_final_state must be bool")
    shape, prepared = _prepare_full_inputs(
        q, k, v, alpha, omega, r, initial_memory, initial_precision, info_scale
    )
    batch, length, heads, key_dim, value_dim = shape
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
            q_work, k_work, v_work, alpha_work, omega_work, r_work,
            memory, precision, information_scale,
        )
        output = (v_work + anchor).to(v.dtype)
        final_state = (memory + anchor, precision + anchor) if output_final_state else None
        return output, final_state

    beta, final_precision = iso_kdn_gain_naive(
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
            q_work, k_work, v_work, alpha_work, beta, memory, output_scale
        )
    final_state = (memory, final_precision) if output_final_state else None
    return output.to(v.dtype), final_state


def _mobius_apply(matrix: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    numerator = matrix[..., 0, 0] * covariance + matrix[..., 0, 1]
    denominator = matrix[..., 1, 0] * covariance + matrix[..., 1, 1]
    return numerator / denominator


def _chunk_memory_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    memory: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate memory and readout with an independent WY chunk transform."""
    length = q.shape[1]
    outputs: list[torch.Tensor] = []

    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        local_length = stop - start
        q_chunk = q[:, start:stop].permute(0, 2, 1, 3) * scale
        k_chunk = k[:, start:stop].permute(0, 2, 1, 3)
        v_chunk = v[:, start:stop].permute(0, 2, 1, 3)
        beta_chunk = beta[:, start:stop].permute(0, 2, 1)
        cumulative_gate = alpha[:, start:stop].log().permute(0, 2, 1, 3).cumsum(dim=2)

        # Strictly lower key-key interactions. Constructing only retained rows
        # avoids both masked 0*inf values and assumptions that alpha <= 1.
        key_columns: list[torch.Tensor] = []
        for source in range(local_length):
            before = k_chunk.new_zeros(*k_chunk.shape[:2], source + 1)
            after = torch.einsum(
                "bhck,bhk->bhc",
                k_chunk[:, :, source + 1 :]
                * (
                    cumulative_gate[:, :, source + 1 :]
                    - cumulative_gate[:, :, source : source + 1]
                ).exp(),
                k_chunk[:, :, source],
            )
            key_columns.append(torch.cat((before, after), dim=-1))
        transform = -torch.stack(key_columns, dim=-1) * beta_chunk[..., None]

        # Invert the unit lower-triangular interaction matrix row by row.
        for row in range(1, local_length):
            transform[..., row, :row] = transform[..., row, :row].clone() + (
                transform[..., row, :, None].clone()
                * transform[..., :, :row].clone()
            ).sum(dim=-2)
        identity = torch.eye(
            local_length, dtype=transform.dtype, device=transform.device
        )
        transform = (transform + identity) * beta_chunk[..., None, :]

        transformed_keys = transform @ (cumulative_gate.exp() * k_chunk)
        transformed_values = transform @ v_chunk

        # Lower-triangular query-key interactions, including the diagonal.
        query_columns: list[torch.Tensor] = []
        for source in range(local_length):
            before = q_chunk.new_zeros(*q_chunk.shape[:2], source)
            retained = torch.einsum(
                "bhck,bhk->bhc",
                q_chunk[:, :, source:]
                * (
                    cumulative_gate[:, :, source:]
                    - cumulative_gate[:, :, source : source + 1]
                ).exp(),
                k_chunk[:, :, source],
            )
            query_columns.append(torch.cat((before, retained), dim=-1))
        query_key = torch.stack(query_columns, dim=-1)

        corrected_values = transformed_values - transformed_keys @ memory
        output_chunk = (
            (q_chunk * cumulative_gate.exp()) @ memory
            + query_key @ corrected_values
        )
        outputs.append(output_chunk.permute(0, 2, 1, 3))

        final_gate = cumulative_gate[:, :, -1:]
        memory = memory * final_gate[:, :, 0, :, None].exp()
        memory = memory + (
            ((final_gate - cumulative_gate).exp() * k_chunk).transpose(-1, -2)
            @ corrected_values
        )

    return torch.cat(outputs, dim=1), memory


def _naive_chunk_iso_kdn(
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
    chunk_size: int = 64,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Private chunked PyTorch reference using scalar Möbius map composition."""
    if type(output_final_state) is not bool:
        raise TypeError("output_final_state must be bool")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    shape, prepared = _prepare_full_inputs(
        q, k, v, alpha, omega, r, initial_memory, initial_precision, info_scale
    )
    batch, length, heads, key_dim, _value_dim = shape
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
        final_state = (
            (memory + anchor, precision + anchor) if output_final_state else None
        )
        return output, final_state

    contraction = alpha_work.square().mean(dim=-1)
    key_norm_squared = k_work.square().sum(dim=-1)
    information_per_key_dim = information_scale.view(1, heads) / float(key_dim)
    covariance_at_chunk_start = precision.reciprocal()
    beta_parts: list[torch.Tensor] = []

    with torch.autocast(device_type=q.device.type, enabled=False):
        for start in range(0, length, chunk_size):
            stop = min(start + chunk_size, length)
            identity = torch.eye(2, device=q.device, dtype=q_work.dtype)
            prefix = identity.view(1, 1, 2, 2).expand(batch, heads, 2, 2)
            local_betas: list[torch.Tensor] = []
            for token in range(start, stop):
                previous_covariance = _mobius_apply(prefix, covariance_at_chunk_start)
                predicted = contraction[:, token] * previous_covariance + omega_work[:, token]
                norm2 = key_norm_squared[:, token]
                local_betas.append(predicted / (r_work[:, token] + predicted * norm2))

                information_weight = information_per_key_dim * norm2
                token_map = torch.stack(
                    (
                        torch.stack(
                            (
                                r_work[:, token] * contraction[:, token],
                                r_work[:, token] * omega_work[:, token],
                            ),
                            dim=-1,
                        ),
                        torch.stack(
                            (
                                information_weight * contraction[:, token],
                                r_work[:, token]
                                + information_weight * omega_work[:, token],
                            ),
                            dim=-1,
                        ),
                    ),
                    dim=-2,
                )
                prefix = torch.matmul(token_map, prefix)
                prefix = prefix / prefix.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(
                    torch.finfo(prefix.dtype).tiny
                )
            beta_parts.append(torch.stack(local_betas, dim=1))
            covariance_at_chunk_start = _mobius_apply(prefix, covariance_at_chunk_start)

        beta = torch.cat(beta_parts, dim=1)
        output, memory = _chunk_memory_recurrence(
            q_work,
            k_work,
            v_work,
            alpha_work,
            beta,
            memory,
            output_scale,
            chunk_size,
        )
    final_precision = covariance_at_chunk_start.reciprocal()
    final_state = (memory, final_precision) if output_final_state else None
    return output.to(v.dtype), final_state


__all__ = ("iso_kdn_gain_naive", "naive_recurrent_iso_kdn")
