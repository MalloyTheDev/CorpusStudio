"""One home for lowering a sealed :class:`OptimizerSpec` to a concrete optimizer, shared by every worker so
the sealed optimizer can never be PARTLY dropped.

The pre-merge audit of the DPO and pretraining workers found the SAME bug independently: each hand-built its
optimizer and silently dropped sealed fields - the DPO worker replaced ``paged_adamw_8bit`` with a
full-precision ``AdamW`` (both a seal violation and a seq-4096 memory regression), and the pretraining worker
omitted ``optim``/betas/``adam_epsilon`` from its ``TrainingArguments``. Both are the same class of defect: a
worker lowering the seal by hand and forgetting a field. This module makes that lowering a single, tested
place, so a new worker (S4-S9) cannot re-introduce it.

Dependency-light: ``torch`` / ``bitsandbytes`` are imported lazily inside :func:`build_torch_optimizer` only;
:func:`paged_adamw_requested` and :func:`hf_training_arguments_optimizer_kwargs` are PURE (no torch)."""

from __future__ import annotations

from typing import Any


def _impl_value(optimizer_spec: Any) -> Any:
    """The sealed optimizer impl as its string value, tolerant of a bare-string impl (no ``.value``)."""
    impl = getattr(optimizer_spec, "impl", None)
    return getattr(impl, "value", impl)


def paged_adamw_requested(optimizer_spec: Any) -> bool:
    """Whether the SEALED optimizer impl is bitsandbytes paged 8-bit AdamW (vs plain ``adamw_torch``) - a
    PURE read of the sealed enum. Paged 8-bit is the memory-saving optimizer that keeps 4B QLoRA DPO inside
    12 GB at seq 4096, so a worker that silently substitutes a full-precision AdamW both violates the seal
    and erases that headroom."""
    return _impl_value(optimizer_spec) == "paged_adamw_8bit"


def hf_training_arguments_optimizer_kwargs(optimizer_spec: Any) -> dict[str, Any]:
    """The FULL set of ``transformers.TrainingArguments`` optimizer kwargs for a sealed ``OptimizerSpec`` -
    the single place an HF-Trainer worker (pretraining today; SFT can adopt it) lowers the seal, so none of
    impl / betas / epsilon / weight_decay / max_grad_norm / schedule can be omitted (the audit's F3 drift).
    PURE (no torch). ``None`` weight_decay / scheduler / warmup fall back to HF's documented defaults."""
    return {
        "optim": _impl_value(optimizer_spec),
        "learning_rate": optimizer_spec.learning_rate,
        "weight_decay": optimizer_spec.weight_decay or 0.0,
        "adam_beta1": optimizer_spec.adam_beta1,
        "adam_beta2": optimizer_spec.adam_beta2,
        "adam_epsilon": optimizer_spec.adam_epsilon,
        "max_grad_norm": optimizer_spec.max_grad_norm,
        "lr_scheduler_type": optimizer_spec.lr_scheduler or "linear",
        "warmup_ratio": optimizer_spec.warmup_ratio or 0.0,
    }


def build_torch_optimizer(  # pragma: no cover - torch/bitsandbytes integration; proven by a GPU run
    optimizer_spec: Any, params: Any
) -> Any:
    """Build the concrete ``torch`` optimizer a CUSTOM-LOOP worker (DPO today, and future S4-S9 loops that
    do not run through the HF ``Trainer``) needs, honoring the FULL sealed spec - impl (paged 8-bit vs
    ``adamw_torch``), betas, epsilon, weight_decay. Fail-closed on an unknown impl rather than silently
    substituting a default. HF-Trainer workers use :func:`hf_training_arguments_optimizer_kwargs` instead."""
    betas = (optimizer_spec.adam_beta1, optimizer_spec.adam_beta2)
    weight_decay = optimizer_spec.weight_decay or 0.0
    if paged_adamw_requested(optimizer_spec):
        from bitsandbytes.optim import PagedAdamW8bit  # noqa: PLC0415

        return PagedAdamW8bit(
            params, lr=optimizer_spec.learning_rate, betas=betas,
            eps=optimizer_spec.adam_epsilon, weight_decay=weight_decay,
        )
    if _impl_value(optimizer_spec) == "adamw_torch":
        import torch  # noqa: PLC0415

        return torch.optim.AdamW(
            params, lr=optimizer_spec.learning_rate, betas=betas,
            eps=optimizer_spec.adam_epsilon, weight_decay=weight_decay,
        )
    raise ValueError(
        f"unsupported sealed optimizer impl {optimizer_spec.impl!r} "
        "(expected adamw_torch or paged_adamw_8bit)"
    )
