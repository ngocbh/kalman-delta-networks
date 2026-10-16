"""Pure-PyTorch reference for value-wise isotropic KDN.

The implementation intentionally keeps the uncertainty scan and memory
recurrence simple and sequential.  It is the readable correctness target for
future optimized kernels, not a performance path.
"""

from __future__ import annotations

import torch

from ._value_iso_contracts import validate_inputs


def _head_scale(
    information_scale: float | torch.Tensor,
    *,
    like: torch.Tensor,
) -> float | torch.Tensor:
    """Return a scale broadcastable over ``[B,H,V]`` without detaching it."""

    if not isinstance(information_scale, torch.Tensor):
        return information_scale
    scale = information_scale.to(dtype=like.dtype, device=like.device)
    if scale.ndim == 1:
        return scale.view(1, -1, 1)
    return scale


def _safe_zero_dependency(*tensors: torch.Tensor) -> torch.Tensor:
    """Connect tensors to scalar zero without reading their element values."""

    zero = tensors[0].as_strided((0,), (1,)).sum()
    for tensor in tensors[1:]:
        zero = zero + tensor.as_strided((0,), (1,)).sum()
    return zero


def _value_iso_beta_seq(
    a: torch.Tensor,
    nu: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    initial_precision: torch.Tensor,
    information_scale: float | torch.Tensor,
    key_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Run the value-wise scalar information recurrence.

    ``a`` and ``nu`` have shape ``[B,T,H]``; ``omega`` and ``r`` have shape
    ``[B,T,H,V]``; and ``initial_precision`` has shape ``[B,H,V]``.  The
    current Kalman gain is deliberately independent of ``information_scale``::

        b_hat = a / c_prev + omega
        beta  = b_hat / (r + nu * b_hat)
        c     = 1 / b_hat + information_scale * nu / (K * r)

    The scale calibrates only retained post-write information.  In particular,
    this is not the historical repository IsoKLA convention that also inserts
    its scale into the current-gain denominator.
    """

    if a.ndim != 3:
        raise ValueError(f"a must have shape [B,T,H], got {tuple(a.shape)}")
    if nu.shape != a.shape:
        raise ValueError(
            f"nu must have the same [B,T,H] shape as a; got {tuple(nu.shape)}"
        )
    B, T, H = a.shape
    if omega.ndim != 4 or tuple(omega.shape[:3]) != (B, T, H):
        raise ValueError(
            f"omega must have shape [B,T,H,V], got {tuple(omega.shape)}"
        )
    if r.shape != omega.shape:
        raise ValueError(
            f"r must have the same [B,T,H,V] shape as omega; got {tuple(r.shape)}"
        )
    if initial_precision.shape != (B, H, omega.shape[-1]):
        raise ValueError(
            "initial_precision must have shape "
            f"{(B, H, omega.shape[-1])}, got {tuple(initial_precision.shape)}"
        )
    if isinstance(key_dim, bool) or not isinstance(key_dim, int) or key_dim <= 0:
        raise ValueError("key_dim must be a positive integer")

    if T == 0:
        # Keep the degenerate scan differentiable with ``allow_unused=False``:
        # every tensor input has a connected zero VJP, while the carried
        # precision retains its identity VJP.
        dependencies = [a, nu, omega, r, initial_precision]
        if isinstance(information_scale, torch.Tensor):
            dependencies.append(information_scale.to(a))
        zero = _safe_zero_dependency(*dependencies)
        return omega[:, :0] + zero, initial_precision + zero

    scale = _head_scale(information_scale, like=initial_precision)
    precision = initial_precision
    beta_tokens: list[torch.Tensor] = []
    for token in range(T):
        a_t = a[:, token].unsqueeze(-1)
        nu_t = nu[:, token].unsqueeze(-1)
        b_hat = a_t / precision + omega[:, token]
        beta_tokens.append(b_hat / (r[:, token] + nu_t * b_hat))
        precision = b_hat.reciprocal() + scale * nu_t / (key_dim * r[:, token])
    return torch.stack(beta_tokens, dim=1), precision


def naive_recurrent_value_iso_kdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    prior: torch.Tensor,
    *,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    info_scale: float | torch.Tensor | None = None,
    read_scale: float | None = None,
    dtype_mode: str = "oracle",
    validate_values: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    r"""Token-by-token value-wise isotropic KDN reference.

    Prepared ``q``, ``k``, and ``alpha`` have shape ``[B,T,H,K]``;
    ``v``, ``omega``, and ``r`` have shape ``[B,T,H,V]``.  ``prior`` is an
    initial information state with shape ``[H,V]`` or ``[B,H,V]``.  A supplied
    ``initial_state`` replaces the zero memory and prior precision.

    ``info_scale=None`` resolves to ``K`` as an intentional scalar-Iso
    compatibility/protection calibration.  This is distinct from the paper's
    reverse-KL optimum, ``mu=1``.  ``read_scale=None`` resolves to
    ``K**-0.5``.  Computation is FP64 in oracle mode and FP32 otherwise.  The
    output is cast back to ``v.dtype`` while a requested final memory and
    precision remain in the compute dtype.
    """

    key_dim = q.shape[-1] if isinstance(q, torch.Tensor) and q.ndim == 4 else 1
    resolved_info_scale: float | torch.Tensor = (
        float(key_dim) if info_scale is None else info_scale
    )
    resolved_read_scale = key_dim**-0.5 if read_scale is None else read_scale
    inputs = validate_inputs(
        q,
        k,
        v,
        alpha,
        omega,
        r,
        prior,
        initial_state=initial_state,
        info_scale=resolved_info_scale,
        read_scale=resolved_read_scale,
        backend="reference",
        dtype_mode=dtype_mode,
        validate_values=validate_values,
    )

    compute_dtype = inputs.compute_dtype
    q_hp = inputs.q.to(compute_dtype)
    k_hp = inputs.k.to(compute_dtype)
    alpha_hp = inputs.alpha.to(compute_dtype)
    v_hp = inputs.v.to(compute_dtype)
    omega_hp = inputs.omega.to(compute_dtype)
    r_hp = inputs.r.to(compute_dtype)

    if inputs.has_initial_state:
        assert inputs.memory is not None and inputs.precision is not None
        initial_memory = inputs.memory.to(compute_dtype)
        memory = initial_memory
        initial_precision = inputs.precision.to(compute_dtype)
    else:
        initial_memory = None
        memory = torch.zeros(
            (inputs.B, inputs.H, inputs.K, inputs.V),
            dtype=compute_dtype,
            device=inputs.q.device,
        )
        # A shared prior is a stride-zero expanded view.  Materialize it here,
        # when the operator actually owns output state, so even a T=0 state is
        # valid as the next call's contiguous ``initial_state``.
        initial_precision = inputs.prior.to(compute_dtype).contiguous()

    dependencies = [
        q_hp,
        k_hp,
        v_hp,
        alpha_hp,
        omega_hp,
        r_hp,
        prior.to(compute_dtype),
    ]
    if initial_memory is not None:
        dependencies.extend((initial_memory, initial_precision))
    if isinstance(inputs.info_scale, torch.Tensor):
        dependencies.append(inputs.info_scale.to(compute_dtype))
    zero_dependency = _safe_zero_dependency(*dependencies)

    a = alpha_hp.square().mean(dim=-1)
    nu = k_hp.square().sum(dim=-1)
    beta, final_precision = _value_iso_beta_seq(
        a,
        nu,
        omega_hp,
        r_hp,
        initial_precision,
        inputs.info_scale,
        inputs.K,
    )

    output_tokens: list[torch.Tensor] = []
    for token in range(inputs.T):
        predicted = alpha_hp[:, token, :, :, None] * memory
        predicted_value = (
            predicted * k_hp[:, token, :, :, None]
        ).sum(dim=-2)
        residual = v_hp[:, token] - predicted_value
        memory = predicted + k_hp[:, token, :, :, None] * (
            beta[:, token] * residual
        )[:, :, None, :]
        output_tokens.append(
            inputs.read_scale
            * (memory * q_hp[:, token, :, :, None]).sum(dim=-2)
        )

    if output_tokens:
        output = torch.stack(output_tokens, dim=1)
    else:
        output = v_hp[:, :0]

    # Match the custom reverse contract: every accepted differentiable input
    # has a materialized VJP for each independently selected public result.
    # The empty views above add graph edges without sampling ignored data.
    output = output + zero_dependency
    state = (
        (memory + zero_dependency, final_precision + zero_dependency)
        if output_final_state
        else None
    )
    return output.to(inputs.v.dtype), state


__all__ = ["_value_iso_beta_seq", "naive_recurrent_value_iso_kdn"]
