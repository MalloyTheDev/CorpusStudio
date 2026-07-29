"""Execution variants of the ONE canonical training harness (Training Systems P0d, #484).

CorpusStudio has a single training harness. The dense-QLoRA-SFT path is that canonical harness - it has
completed a real training workload and its sealed contract is ``ResolvedExecutionConfiguration``. Other
modes (dense full fine-tune, pretraining, MoE) are EXECUTION VARIANTS of the same harness, not separate
harnesses. This module lets the control plane DESCRIBE those variants and admit them FAIL-CLOSED,
WITHOUT touching the sealed SFT configuration (which stays byte-identical) and WITHOUT a worker change:
a declared variant is not executable until it is separately implemented and workload-verified.

Control-plane only: stdlib + platform contracts, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass

from corpus_studio.platform.common import CONTRACT_VERSION
from corpus_studio.platform.contracts import BackendExecutionVariant
from corpus_studio.platform.enums import ExecutionVariantKind, ExecutionVariantSupport


# The sealed SFT config's task-type lock (``adapter_task_type: Literal["CAUSAL_LM"]``) is FROZEN by
# design (the byte-identical constraint), so there is nothing to sync TO; this constant only
# de-duplicates the value across the envelope table below.
_CAUSAL_LM = "CAUSAL_LM"


class ExecutionVariantRefused(ValueError):
    """A structured, fail-closed refusal from the execution-variant admission gate. The gate never
    silently coerces, downgrades, or substitutes a variant, and never falls back to dense_qlora_sft."""


@dataclass(frozen=True)
class VariantEnvelope:
    """The capability envelope for one execution variant, DERIVED from its kind (never free Booleans on
    a descriptor, so an impossible combination cannot be expressed). It states which of the sealed-SFT
    locks apply to that variant - the sealed ``ResolvedExecutionConfiguration`` itself is unchanged."""

    task_type: str
    requires_causal_lm: bool
    requires_peft_adapter: bool
    allows_full_parameter: bool
    allows_multiple_datasets: bool


# The single source of truth for each variant's rules. dense_qlora_sft mirrors the sealed SFT locks
# (CAUSAL_LM + a PEFT adapter, one dataset, no full-parameter update); the others relax specific locks.
_VARIANT_ENVELOPE: dict[ExecutionVariantKind, VariantEnvelope] = {
    ExecutionVariantKind.dense_qlora_sft: VariantEnvelope(
        task_type=_CAUSAL_LM,
        requires_causal_lm=True,
        requires_peft_adapter=True,
        allows_full_parameter=False,
        allows_multiple_datasets=False,
    ),
    ExecutionVariantKind.dense_full_finetune: VariantEnvelope(
        task_type=_CAUSAL_LM,
        requires_causal_lm=True,
        requires_peft_adapter=False,
        allows_full_parameter=True,
        allows_multiple_datasets=False,
    ),
    ExecutionVariantKind.pretraining: VariantEnvelope(
        task_type=_CAUSAL_LM,
        requires_causal_lm=True,
        requires_peft_adapter=False,
        allows_full_parameter=True,
        allows_multiple_datasets=True,
    ),
    ExecutionVariantKind.moe: VariantEnvelope(
        task_type=_CAUSAL_LM,
        requires_causal_lm=True,
        requires_peft_adapter=False,
        allows_full_parameter=True,
        allows_multiple_datasets=True,
    ),
}

# Derived from the enum's definition order (ascending support), so it cannot drift from the ladder:
# a member added in order automatically extends it, and the two can never disagree.
_SUPPORT_ORDER: dict[ExecutionVariantSupport, int] = {
    member: index for index, member in enumerate(ExecutionVariantSupport)
}


def variant_envelope(kind: ExecutionVariantKind) -> VariantEnvelope:
    """The canonical capability envelope for an execution-variant kind (derived, not stored)."""
    return _VARIANT_ENVELOPE[kind]


def reference_execution_variants() -> tuple[BackendExecutionVariant, ...]:
    """The first-party ``corpus_studio`` backend's declared execution variants. Only dense_qlora_sft is
    ``workload_verified`` - the canonical harness's proven path (real runs preserved under
    ``.../runs/*``; see docs/HOST_STATE.md). The others are contract-expressible but NOT executable."""
    return (
        BackendExecutionVariant(
            backend_id="corpus_studio",
            variant_kind=ExecutionVariantKind.dense_qlora_sft,
            support=ExecutionVariantSupport.workload_verified,
        ),
        BackendExecutionVariant(
            backend_id="corpus_studio",
            variant_kind=ExecutionVariantKind.dense_full_finetune,
            support=ExecutionVariantSupport.contract_validated,
        ),
        BackendExecutionVariant(
            backend_id="corpus_studio",
            variant_kind=ExecutionVariantKind.pretraining,
            support=ExecutionVariantSupport.declared,
        ),
        BackendExecutionVariant(
            backend_id="corpus_studio",
            variant_kind=ExecutionVariantKind.moe,
            support=ExecutionVariantSupport.declared,
        ),
    )


def admit_execution_variant(
    variant: BackendExecutionVariant,
    *,
    required_support: ExecutionVariantSupport = ExecutionVariantSupport.workload_verified,
) -> None:
    """Fail-closed execution admission for an execution-variant descriptor. Raises
    :class:`ExecutionVariantRefused` unless the variant is a known kind, a supported schema version, and
    AT OR ABOVE ``required_support``. It NEVER coerces, downgrades, substitutes, or falls back to
    dense_qlora_sft, and it NEVER returns a worker configuration - the descriptor is not a
    ``ResolvedExecutionConfiguration`` and is never passed to a worker as one.

    In this slice only dense_qlora_sft reaches ``workload_verified`` (the canonical harness's proven
    path), so it is the only variant admitted; every other variant is refused until it is separately
    implemented, measured, and admitted."""
    if variant.contract_version != CONTRACT_VERSION:
        raise ExecutionVariantRefused(
            f"unsupported BackendExecutionVariant schema version '{variant.contract_version}'"
        )
    if variant.variant_kind not in _VARIANT_ENVELOPE:  # pragma: no cover - enum-typed; defensive
        raise ExecutionVariantRefused(f"unknown execution variant kind '{variant.variant_kind}'")
    if _SUPPORT_ORDER[variant.support] < _SUPPORT_ORDER[required_support]:
        raise ExecutionVariantRefused(
            f"execution variant '{variant.variant_kind.value}' is '{variant.support.value}', below the "
            f"required '{required_support.value}' - refused (no fallback to dense_qlora_sft)"
        )
