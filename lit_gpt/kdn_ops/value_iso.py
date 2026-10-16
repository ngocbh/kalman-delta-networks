"""Public dispatch and sole autograd boundary for value-wise isotropic KDN.

The optimized route deliberately composes raw gain and recurrent-memory launch
helpers under one :class:`torch.autograd.Function`.  The pure-PyTorch reference
route remains an ordinary autograd graph and is also the fallback outside the
frozen Triton envelope.
"""

from __future__ import annotations

from typing import Any

import torch

from ._value_iso_contracts import BACKENDS, validate_inputs
from .value_iso_naive import naive_recurrent_value_iso_kdn
from .value_iso_chunk import (
    MEMORY_CHUNK_SIZE,
    raw_value_iso_gain_backward,
    raw_value_iso_gain_forward,
    raw_value_iso_memory_backward,
    raw_value_iso_memory_forward,
)


def _backward_destination_plan(
    needs: tuple[bool, ...],
    *,
    run_memory: bool,
    run_gain: bool,
    has_initial_memory: bool,
    has_initial_precision: bool,
    information_scale_is_tensor: bool,
) -> dict[str, frozenset[str]]:
    """Describe exactly which full-size destinations each raw stage owns."""

    if len(needs) != 12:
        raise ValueError("needs_input_grad must contain exactly 12 entries")
    memory = set()
    gain = set()
    if run_memory:
        for required, name in (
            (needs[0], "dq_acc"),
            (needs[1], "dk_acc"),
            (needs[2], "dv"),
            (needs[3], "dalpha"),
            (run_gain, "dbeta_domega"),
            (needs[7] and has_initial_memory, "dinitial_memory"),
        ):
            if required:
                memory.add(name)
    if run_gain:
        for required, name in (
            (needs[1], "dk_acc"),
            (needs[3], "dalpha"),
            (needs[4], "domega"),
            (needs[5], "dr"),
            (needs[6] and not has_initial_precision, "dprior"),
            (needs[8] and has_initial_precision, "dinitial_precision"),
            (
                needs[9] and information_scale_is_tensor,
                "dinformation_scale",
            ),
        ):
            if required:
                gain.add(name)
    return {"memory": frozenset(memory), "gain": frozenset(gain)}


class _ValueIsoKDNFunction(torch.autograd.Function):
    """The single autograd owner for the composed raw Triton route."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        omega: torch.Tensor,
        r: torch.Tensor,
        prior: torch.Tensor,
        initial_memory: torch.Tensor | None,
        initial_precision: torch.Tensor | None,
        information_scale: float | torch.Tensor,
        read_scale: float,
        output_final_state: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)

        gain = raw_value_iso_gain_forward(
            k,
            alpha,
            omega,
            r,
            prior,
            initial_precision,
            information_scale,
        )
        memory = raw_value_iso_memory_forward(
            q,
            k,
            v,
            alpha,
            gain.beta,
            initial_memory=initial_memory,
            read_scale=read_scale,
            output_final_state=output_final_state,
        )

        needs = ctx.needs_input_grad
        has_initial_state = initial_precision is not None
        nonempty = q.shape[1] > 0
        memory_possible = nonempty and (
            any(needs[index] for index in (0, 1, 2, 3, 4, 5, 7, 8, 9))
            or (needs[6] and not has_initial_state)
        )
        gain_possible = nonempty and (
            any(needs[index] for index in (1, 3, 4, 5, 8, 9))
            or (needs[6] and not has_initial_state)
        )
        saved: list[torch.Tensor] = []
        saved_indices: dict[str, int] = {}

        def save(name: str, tensor: torch.Tensor) -> None:
            saved_indices[name] = len(saved)
            saved.append(tensor)

        if memory_possible:
            for name, tensor in (
                ("q", q),
                ("k", k),
                ("v", v),
                ("alpha", alpha),
                ("beta", gain.beta),
                ("checkpoints", memory.checkpoints),
            ):
                save(name, tensor)
        if gain_possible:
            for name, tensor in (
                ("k", k),
                ("alpha", alpha),
                ("omega", omega),
                ("r", r),
                ("prior", prior),
                ("maps_and_carries", gain.maps_and_carries),
                ("statistics", gain.statistics),
            ):
                if name not in saved_indices:
                    save(name, tensor)
            if initial_precision is not None:
                save("initial_precision", initial_precision)
            if isinstance(information_scale, torch.Tensor):
                save("information_scale", information_scale)
        zero_gradient_inputs = [
            (0, "q", q),
            (1, "k", k),
            (2, "v", v),
            (3, "alpha", alpha),
            (4, "omega", omega),
            (5, "r", r),
            (6, "prior", prior),
        ]
        if initial_memory is not None:
            zero_gradient_inputs.append((7, "initial_memory", initial_memory))
        if initial_precision is not None:
            zero_gradient_inputs.append((8, "initial_precision", initial_precision))
        if isinstance(information_scale, torch.Tensor):
            zero_gradient_inputs.append((9, "information_scale", information_scale))
        for index, name, tensor in zero_gradient_inputs:
            if needs[index] and name not in saved_indices:
                save(name, tensor)

        ctx.save_for_backward(*saved)
        ctx.saved_indices = saved_indices
        ctx.memory_possible = memory_possible
        ctx.gain_possible = gain_possible
        ctx.has_initial_memory = initial_memory is not None
        ctx.has_initial_precision = has_initial_state
        ctx.information_scale_is_tensor = isinstance(
            information_scale, torch.Tensor
        )
        ctx.python_information_scale = (
            None
            if isinstance(information_scale, torch.Tensor)
            else information_scale
        )
        ctx.read_scale = read_scale
        ctx.output_final_state = output_final_state
        ctx.total_tokens = q.shape[1]

        if output_final_state:
            if memory.final_memory is None:
                raise RuntimeError(
                    "raw memory forward omitted the requested final memory"
                )
            return memory.output, memory.final_memory, gain.final_precision
        return memory.output

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor | None
    ) -> tuple[object, ...]:
        if ctx.output_final_state:
            output_cotangent, final_memory_cotangent, final_precision_cotangent = (
                grad_outputs
            )
        else:
            (output_cotangent,) = grad_outputs
            final_memory_cotangent = None
            final_precision_cotangent = None

        saved_tensors = ctx.saved_tensors
        saved_indices: dict[str, int] = ctx.saved_indices

        def saved(name: str) -> torch.Tensor:
            return saved_tensors[saved_indices[name]]

        needs = ctx.needs_input_grad
        (
            need_q,
            need_k,
            need_v,
            need_alpha,
            need_omega,
            need_r,
            need_prior,
            need_initial_memory,
            need_initial_precision,
            need_information_scale,
            _,
            _,
        ) = needs

        has_memory_source = ctx.total_tokens > 0 and (
            output_cotangent is not None or final_memory_cotangent is not None
        )
        run_memory = ctx.memory_possible and has_memory_source
        run_gain = ctx.gain_possible and (
            run_memory or final_precision_cotangent is not None
        )
        write_plan = _backward_destination_plan(
            needs,
            run_memory=run_memory,
            run_gain=run_gain,
            has_initial_memory=ctx.has_initial_memory,
            has_initial_precision=ctx.has_initial_precision,
            information_scale_is_tensor=ctx.information_scale_is_tensor,
        )
        memory_writes = write_plan["memory"]
        gain_writes = write_plan["gain"]

        dq_acc: torch.Tensor | None = None
        dk_acc: torch.Tensor | None = None
        dalpha: torch.Tensor | None = None
        dv: torch.Tensor | None = None
        dbeta_domega: torch.Tensor | None = None
        dinitial_memory: torch.Tensor | None = None

        if run_memory:
            q = saved("q")
            k = saved("k")
            v = saved("v")
            alpha = saved("alpha")
            beta = saved("beta")
            checkpoints = saved("checkpoints")
            if output_cotangent is None:
                output_cotangent = torch.zeros_like(
                    v, memory_format=torch.contiguous_format
                )
            else:
                output_cotangent = output_cotangent.contiguous()
            if final_memory_cotangent is not None:
                final_memory_cotangent = final_memory_cotangent.contiguous()

            replay_buffer = torch.empty(
                (MEMORY_CHUNK_SIZE + 1, *checkpoints.shape[1:]),
                dtype=torch.float32,
                device=q.device,
            )
            dstate_carry = torch.empty(
                checkpoints.shape[1:], dtype=torch.float32, device=q.device
            )
            dq_acc = (
                torch.empty_like(
                    q, dtype=torch.float32, memory_format=torch.contiguous_format
                )
                if "dq_acc" in memory_writes
                else None
            )
            dk_acc = (
                torch.empty_like(
                    k, dtype=torch.float32, memory_format=torch.contiguous_format
                )
                if "dk_acc" in memory_writes
                else None
            )
            dalpha = (
                torch.empty_like(
                    alpha,
                    dtype=torch.float32,
                    memory_format=torch.contiguous_format,
                )
                if "dalpha" in memory_writes
                else None
            )
            dv = (
                torch.empty_like(v, memory_format=torch.contiguous_format)
                if "dv" in memory_writes
                else None
            )
            dbeta_domega = (
                torch.empty_like(
                    beta,
                    dtype=torch.float32,
                    memory_format=torch.contiguous_format,
                )
                if "dbeta_domega" in memory_writes
                else None
            )
            dinitial_memory = (
                torch.empty_like(
                    saved("initial_memory"), memory_format=torch.contiguous_format
                )
                if "dinitial_memory" in memory_writes
                else None
            )
            raw_value_iso_memory_backward(
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
                final_memory_cotangent=final_memory_cotangent,
                dinitial_memory=dinitial_memory,
                read_scale=ctx.read_scale,
                initialize_outputs=True,
            )
            del replay_buffer, dstate_carry

        if run_gain and not run_memory:
            k = saved("k")
            alpha = saved("alpha")
            omega = saved("omega")
            dk_acc = (
                torch.zeros_like(
                    k, dtype=torch.float32, memory_format=torch.contiguous_format
                )
                if "dk_acc" in gain_writes
                else None
            )
            dalpha = (
                torch.zeros_like(
                    alpha,
                    dtype=torch.float32,
                    memory_format=torch.contiguous_format,
                )
                if "dalpha" in gain_writes
                else None
            )
            dbeta_domega = (
                torch.zeros_like(
                    omega,
                    dtype=torch.float32,
                    memory_format=torch.contiguous_format,
                )
                if "domega" in gain_writes
                else None
            )

        domega: torch.Tensor | None = None
        dr: torch.Tensor | None = None
        dprior: torch.Tensor | None = None
        dinitial_precision: torch.Tensor | None = None
        dinformation_scale: torch.Tensor | None = None
        if ctx.total_tokens == 0 and final_precision_cotangent is not None:
            final_precision_cotangent = final_precision_cotangent.contiguous()
            if need_prior and not ctx.has_initial_precision:
                dprior = (
                    final_precision_cotangent.sum(dim=0)
                    if saved("prior").ndim == 2
                    else final_precision_cotangent
                )
            if need_initial_precision and ctx.has_initial_precision:
                dinitial_precision = final_precision_cotangent
        if run_gain:
            k = saved("k")
            alpha = saved("alpha")
            omega = saved("omega")
            r = saved("r")
            prior = saved("prior")
            dr_buffer = (
                torch.empty_like(
                    r, dtype=torch.float32, memory_format=torch.contiguous_format
                )
                if "dr" in gain_writes
                else None
            )
            initial_precision = (
                saved("initial_precision") if ctx.has_initial_precision else None
            )
            information_scale = (
                saved("information_scale")
                if ctx.information_scale_is_tensor
                else ctx.python_information_scale
            )
            if final_precision_cotangent is not None:
                final_precision_cotangent = final_precision_cotangent.contiguous()
            gain_result = raw_value_iso_gain_backward(
                k,
                alpha,
                omega,
                r,
                prior,
                saved("maps_and_carries"),
                saved("statistics"),
                dbeta_domega,
                dk_acc,
                dalpha,
                dr_buffer,
                initial_precision=initial_precision,
                information_scale=information_scale,
                final_precision_cotangent=final_precision_cotangent,
                return_prior_gradient="dprior" in gain_writes,
                return_initial_precision_gradient=(
                    "dinitial_precision" in gain_writes
                ),
                return_information_scale_gradient=(
                    "dinformation_scale" in gain_writes
                ),
            )
            dk_acc = gain_result.dk_acc
            dalpha = gain_result.dalpha
            domega = gain_result.domega if need_omega else None
            dr = gain_result.dr
            dprior = gain_result.dprior
            dinitial_precision = gain_result.dinitial_precision
            dinformation_scale = gain_result.dinformation_scale
            del gain_result, dbeta_domega, dr_buffer

        if dv is None and need_v:
            dv = torch.zeros_like(saved("v"), memory_format=torch.contiguous_format)
        if dalpha is None and need_alpha:
            dalpha = torch.zeros_like(
                saved("alpha"), memory_format=torch.contiguous_format
            )
        if domega is None and need_omega:
            domega = torch.zeros_like(
                saved("omega"), memory_format=torch.contiguous_format
            )
        if dr is None and need_r:
            dr = torch.zeros_like(saved("r"), memory_format=torch.contiguous_format)
        if dprior is None and need_prior:
            dprior = torch.zeros_like(
                saved("prior"), memory_format=torch.contiguous_format
            )
        if dinitial_memory is None and need_initial_memory:
            if ctx.total_tokens == 0 and final_memory_cotangent is not None:
                dinitial_memory = final_memory_cotangent.contiguous()
            else:
                dinitial_memory = torch.zeros_like(
                    saved("initial_memory"), memory_format=torch.contiguous_format
                )
        if dinitial_precision is None and need_initial_precision:
            dinitial_precision = torch.zeros_like(
                saved("initial_precision"), memory_format=torch.contiguous_format
            )
        if dinformation_scale is None and need_information_scale:
            dinformation_scale = torch.zeros_like(
                saved("information_scale"), memory_format=torch.contiguous_format
            )

        dq = None
        if need_q:
            dq = (
                torch.zeros_like(saved("q"), memory_format=torch.contiguous_format)
                if dq_acc is None
                else _cast_fp32_accumulator(dq_acc, saved("q").dtype, "dq")
            )
        del dq_acc
        dk = None
        if need_k:
            dk = (
                torch.zeros_like(saved("k"), memory_format=torch.contiguous_format)
                if dk_acc is None
                else _cast_fp32_accumulator(dk_acc, saved("k").dtype, "dk")
            )
        del dk_acc

        return (
            dq,
            dk,
            dv if need_v else None,
            dalpha if need_alpha else None,
            domega if need_omega else None,
            dr if need_r else None,
            dprior if need_prior else None,
            dinitial_memory if need_initial_memory else None,
            dinitial_precision if need_initial_precision else None,
            dinformation_scale if need_information_scale else None,
            None,
            None,
        )


def _cast_fp32_accumulator(
    accumulator: torch.Tensor, dtype: torch.dtype, name: str
) -> torch.Tensor:
    """Cast one shared accumulator after both raw reverse stages finish."""

    if name not in {"dq", "dk"}:
        raise ValueError(f"unknown accumulator {name!r}")
    return accumulator.to(dtype=dtype)


def _infer_dtype_mode(q: object, k: object, v: object) -> str:
    """Infer the only legal dtype profile from the three activation tensors."""

    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    assert isinstance(q, torch.Tensor)
    assert isinstance(k, torch.Tensor)
    assert isinstance(v, torch.Tensor)
    if q.dtype == k.dtype == v.dtype == torch.float64:
        return "oracle"
    if q.dtype == k.dtype == v.dtype == torch.float32:
        return "fp32"
    if q.dtype == k.dtype == v.dtype == torch.bfloat16:
        return "production"
    raise TypeError(
        "q, k, and v must all use one supported dtype profile: "
        "FP64, FP32, or BF16"
    )


def _auto_backend(q: torch.Tensor, v: torch.Tensor, dtype_mode: str) -> str:
    """Select the optimized route only inside its complete frozen envelope."""

    if torch.are_deterministic_algorithms_enabled():
        return "reference"
    if q.device.type != "cuda" or dtype_mode == "oracle":
        return "reference"
    if q.ndim != 4 or v.ndim != 4:
        return "reference"
    key_dim = q.shape[-1]
    value_dim = v.shape[-1]
    if key_dim not in {64, 128}:
        return "reference"
    if value_dim <= 0 or value_dim > 256 or value_dim % 8 != 0:
        return "reference"
    return "triton_recurrent"


def value_iso_kdn(
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
    information_scale: float | torch.Tensor | None = None,
    read_scale: float | None = None,
    backend: str = "auto",
    validate_values: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Run exact value-wise isotropic KDN with validated eager dispatch.

    ``prior`` keeps its caller-provided ``[H,V]`` or ``[B,H,V]`` layout.  In
    particular, the Triton reverse sees the original contiguous shared prior,
    rather than the validator's stride-zero expanded view, so it can perform
    the required reduction over the batch exactly once.
    """

    if type(output_final_state) is not bool:
        raise TypeError("output_final_state must be exactly bool")
    if not isinstance(backend, str) or backend not in BACKENDS:
        choices = ", ".join(sorted(BACKENDS))
        raise ValueError(f"backend must be one of {choices}, got {backend!r}")

    dtype_mode = _infer_dtype_mode(q, k, v)
    selected_backend = (
        _auto_backend(q, v, dtype_mode) if backend == "auto" else backend
    )
    if selected_backend == "reference":
        return naive_recurrent_value_iso_kdn(
            q,
            k,
            v,
            alpha,
            omega,
            r,
            prior,
            initial_state=initial_state,
            output_final_state=output_final_state,
            info_scale=information_scale,
            read_scale=read_scale,
            dtype_mode=dtype_mode,
            validate_values=validate_values,
        )

    key_dim = q.shape[-1] if q.ndim == 4 else 1
    resolved_information_scale: float | torch.Tensor = (
        float(key_dim) if information_scale is None else information_scale
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
        info_scale=resolved_information_scale,
        read_scale=resolved_read_scale,
        backend=selected_backend,
        dtype_mode=dtype_mode,
        validate_values=validate_values,
    )
    if selected_backend == "triton_chunk":
        raise NotImplementedError(
            "triton_chunk is unavailable because the frozen memory-backend "
            "decision selected triton_recurrent"
        )

    result = _ValueIsoKDNFunction.apply(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.alpha,
        inputs.omega,
        inputs.r,
        prior,
        inputs.memory,
        inputs.precision,
        inputs.info_scale,
        inputs.read_scale,
        output_final_state,
    )
    if output_final_state:
        output, final_memory, final_precision = result
        return output, (final_memory, final_precision)
    assert isinstance(result, torch.Tensor)
    return result, None

__all__ = ["_ValueIsoKDNFunction", "value_iso_kdn"]
