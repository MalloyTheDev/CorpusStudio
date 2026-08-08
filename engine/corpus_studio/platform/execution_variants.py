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
from corpus_studio.platform.enums import ExecutionVariantKind, ExecutionVariantSupport, TaskType


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
    # Offline DPO (S2b-2): QLoRA preference optimization - a CAUSAL_LM PEFT adapter over a frozen
    # reference model, one preference-pair dataset (no rollout / no reward model).
    ExecutionVariantKind.preference_dpo: VariantEnvelope(
        task_type=_CAUSAL_LM,
        requires_causal_lm=True,
        requires_peft_adapter=True,
        allows_full_parameter=False,
        allows_multiple_datasets=False,
    ),
    # Pairwise reward model (RL slice S5a-1): a QLoRA SEQ_CLS scalar score head over a base, one
    # chosen/rejected dataset. NOT causal-LM (a reward head, not generation), no full-parameter update.
    ExecutionVariantKind.reward_model: VariantEnvelope(
        task_type="SEQ_CLS",
        requires_causal_lm=False,
        requires_peft_adapter=True,
        allows_full_parameter=False,
        allows_multiple_datasets=False,
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
            # workload_verified: the canonical harness. Its intermediate-checkpoint WRITE + exact-lineage
            # RESUME (a FEATURE of this variant, NOT a separate variant) are also workload_verified on the
            # real 7B/seq-4096 workload - Qwen2.5-7B QLoRA nf4 r16, flash SDPA + liger + paged_adamw_8bit;
            # resumed from step 2 with a byte-identical continuation loss - through the managed
            # --subprocess dispatch after #830 wired resume into it (previously in-process only).
            # See docs/HOST_STATE.md.
            support=ExecutionVariantSupport.workload_verified,
        ),
        BackendExecutionVariant(
            backend_id="corpus_studio",
            variant_kind=ExecutionVariantKind.dense_full_finetune,
            # workload_verified: full-parameter SFT of Qwen2.5-0.5B-Instruct (bf16, all params trainable, seq
            # 512, 12 steps) on the RTX 5070, through the first-party FullFinetuneRunner + the supervisor's
            # independent full-model reload-verify - supervisor-admitted full-model success evidence
            # (optimizer + 12 loss/steps 2.28 -> 0.17 + 290/290 trainable tensors observed a materialized
            # gradient + 265/290 changed [the rest are sub-bf16-precision fine-tune updates] +
            # model.safetensors reproduced from bytes; peak 5.01 GiB). A PRODUCT claim, not a sealed IEEE
            # cell. The managed platform-run --subprocess route (a worker wheel + env) is the deployment
            # follow-up, as for pretraining/DPO. See docs/HOST_STATE.md.
            support=ExecutionVariantSupport.workload_verified,
        ),
        BackendExecutionVariant(
            backend_id="corpus_studio",
            variant_kind=ExecutionVariantKind.pretraining,
            # workload_verified: a from-scratch full-parameter 124M GPT-2 trained on the RTX 5070 (seq
            # 1024, 20 steps, peak 3.28 GiB) in the HARDWARE_VERIFIED managed env, through the first-party
            # PretrainingRunner + the supervisor's independent reload-verify - supervisor-admitted
            # PretrainingSuccessEvidence (optimizer + one loss/step + 148/148 tensors changed with observed
            # gradients + model.safetensors verified). Milestone worker wheel sha256 8818d3ad... (source
            # fa6ce97). See docs/HOST_STATE.md.
            support=ExecutionVariantSupport.workload_verified,
        ),
        BackendExecutionVariant(
            backend_id="corpus_studio",
            variant_kind=ExecutionVariantKind.moe,
            support=ExecutionVariantSupport.declared,
        ),
        BackendExecutionVariant(
            backend_id="corpus_studio",
            variant_kind=ExecutionVariantKind.preference_dpo,
            # workload_verified: offline DPO of Qwen3-4B-Instruct-2507 (nf4 QLoRA r16, seq 1024, 15 steps,
            # beta 0.1, sequence-chunked log-prob) on the RTX 5070, through the first-party PreferenceRunner
            # + the supervisor's independent adapter reload-verify - supervisor-admitted
            # PreferenceSuccessEvidence (optimizer + 15 loss/steps 0.6931 -> 0.0739 + reward margin
            # 0.0 -> 31.64 + a frozen reference model + 504/504 LoRA tensors changed with observed gradients
            # + adapter.safetensors reproduced from bytes; peak 5.79 GiB). A PRODUCT claim, not a sealed IEEE
            # cell. The managed platform-run --subprocess route (a DPO worker wheel + env) is the deployment
            # follow-up, exactly as for pretraining. See docs/HOST_STATE.md.
            support=ExecutionVariantSupport.workload_verified,
        ),
        BackendExecutionVariant(
            backend_id="corpus_studio",
            variant_kind=ExecutionVariantKind.reward_model,
            # workload_verified: a pairwise reward model of Qwen2.5-0.5B-Instruct (nf4 QLoRA SEQ_CLS scalar
            # score head, seq 512, 20 steps, Bradley-Terry loss) on the RTX 5070, through the FULL managed
            # execute_run -> RewardRunner -> supervisor independent adapter reload-verify path -
            # supervisor-admitted RewardSuccessEvidence (optimizer + 20 loss/steps 0.7563 -> 0.0 + score
            # margin -0.07 -> 46.65 [chosen 33.52 / rejected -13.13] + the randomly-initialized score head
            # TRAINED: 337/337 tensors changed with observed gradients + adapter.safetensors reproduced from
            # bytes; MEASURED final_fit NATIVE_SAFE, peak 0.95 GiB allocated / 1.2 GiB reserved of 12). The
            # PROMOTION GATE is HELD-OUT pairwise ranking accuracy = 1.0 over 2 held-out pairs (a seeded
            # carve-out), never a falling training loss. Evidence: examples/wbg/runs/reward-bringup-qwen05b/.
            # A PRODUCT claim, not a sealed IEEE cell. The managed platform-run --subprocess route (a reward
            # worker wheel + env) is the deployment follow-up, exactly as for pretraining/DPO. See
            # docs/HOST_STATE.md.
            support=ExecutionVariantSupport.workload_verified,
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


# The requested task -> the execution-variant SHAPE the first-party harness would run it as. Only a task
# whose shape has a workload_verified backend variant is executable; the objective x shape axes are
# co-linear today (SFT == the dense-QLoRA-SFT shape), so this bridge is deliberately minimal and gets
# richer as each real variant lands (S2+). A task with no mapped shape is refused fail-closed.
_TASK_TO_VARIANT_KIND: dict[TaskType, ExecutionVariantKind] = {
    TaskType.sft: ExecutionVariantKind.dense_qlora_sft,
    TaskType.pretraining: ExecutionVariantKind.pretraining,
}

# A preference task is NOT co-linear with one shape: DPO / IPO / KTO / ORPO are distinct objectives, and
# only the QLoRA-DPO objective has a built (PEFT) execution shape. So the preference bridge is keyed on
# the SPECIFIC objective, not the task - a task-level 'preference -> DPO' mapping would mis-claim the
# others. An unrecognized or absent preference objective maps to no shape and is refused fail-closed.
_PREFERENCE_OBJECTIVE_TO_VARIANT: dict[str, ExecutionVariantKind] = {
    "dpo_qlora": ExecutionVariantKind.preference_dpo,
}

# The reward family is like the preference family: reward_model / process_supervision / verifier_training
# are DISTINCT objectives, and only the pairwise reward_model has a built (contract) shape (S5a-1). Keyed
# on the specific objective so the declared-only families are never mis-claimed; unrecognized -> refuse.
_REWARD_OBJECTIVE_TO_VARIANT: dict[str, ExecutionVariantKind] = {
    "reward_model": ExecutionVariantKind.reward_model,
}


def execution_variant_kind_for_task(
    task_type: TaskType, *, is_moe: bool = False, objective_id: str | None = None,
    is_full_parameter: bool = False,
) -> ExecutionVariantKind | None:
    """The execution-variant SHAPE for a (task, objective, topology) request, or ``None`` when no built
    shape maps. Shape is objective x topology, so a MoE model routes to the ``moe`` shape (declared-only)
    REGARDLESS of task - never to a dense shape. A dense SFT maps by whether ALL parameters are trainable
    (full-parameter -> dense_full_finetune, else the QLoRA-adapter dense_qlora_sft); pretraining maps by
    task. A PREFERENCE task maps by its specific ``objective_id`` (only the QLoRA-DPO objective has a built
    shape; DPO/IPO/KTO/ORPO are NOT interchangeable), so a bare preference task without a recognized
    objective maps to nothing. ``None`` -> refuse fail-closed."""
    # A preference task resolves by its specific objective FIRST - before the generic MoE routing - so a
    # MoE preference request (or an absent/unmapped objective) refuses fail-closed instead of being
    # silently routed to the generic ``moe`` shape (there is no built MoE-preference shape).
    if task_type == TaskType.preference:
        if is_moe or not objective_id:
            return None
        return _PREFERENCE_OBJECTIVE_TO_VARIANT.get(objective_id)
    # A reward task, like preference, resolves by its specific objective (only the pairwise reward_model
    # has a built shape); a MoE or unmapped reward objective refuses fail-closed rather than routing to
    # the generic moe shape (there is no built MoE-reward shape).
    if task_type == TaskType.reward:
        if is_moe or not objective_id:
            return None
        return _REWARD_OBJECTIVE_TO_VARIANT.get(objective_id)
    if is_moe:
        return ExecutionVariantKind.moe
    # A dense SFT with all parameters trainable is the full-parameter shape, NOT the QLoRA-adapter one -
    # they have different capability envelopes (full-param update, full-model artifact) and gate separately.
    if task_type == TaskType.sft and is_full_parameter:
        return ExecutionVariantKind.dense_full_finetune
    return _TASK_TO_VARIANT_KIND.get(task_type)


def admit_task_execution_variant(
    task_type: TaskType,
    *,
    is_moe: bool = False,
    objective_id: str | None = None,
    is_full_parameter: bool = False,
    declared_variants: tuple[BackendExecutionVariant, ...],
    required_support: ExecutionVariantSupport = ExecutionVariantSupport.workload_verified,
) -> BackendExecutionVariant:
    """Resolve a requested ``task_type`` + objective + topology to its execution-variant shape and admit
    it fail-closed against the backend's ``declared_variants``. Returns the admitted variant, or raises
    :class:`ExecutionVariantRefused` when the (task, objective, topology) maps to no executable shape,
    the backend does not declare that shape, or the declared support is below ``required_support``. A MoE
    model is routed to the declared-only ``moe`` shape, so it is refused here - never admitted as dense.
    A preference task resolves by its specific ``objective_id`` (only QLoRA-DPO has a built shape); a
    full-parameter dense SFT resolves to ``dense_full_finetune``. Never falls back."""
    kind = execution_variant_kind_for_task(
        task_type, is_moe=is_moe, objective_id=objective_id, is_full_parameter=is_full_parameter
    )
    if kind is None:
        request = f"task '{task_type.value}'"
        if objective_id:
            request += f" objective '{objective_id}'"
        raise ExecutionVariantRefused(
            f"{request} maps to no executable execution variant - the first-party harness implements the "
            f"dense-QLoRA-SFT shape (pretraining/MoE/preference are declared-only or unbuilt); refused"
        )
    declared = next((item for item in declared_variants if item.variant_kind == kind), None)
    if declared is None:
        raise ExecutionVariantRefused(
            f"the backend declares no '{kind.value}' execution variant for task '{task_type.value}'"
        )
    admit_execution_variant(declared, required_support=required_support)
    return declared
