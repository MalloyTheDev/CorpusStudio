"""Pairwise reward-model config-consuming worker: lower a sealed
:class:`ResolvedRewardExecutionConfiguration` to the :func:`run_reward_training` primitive, capture the
formal :class:`RewardExecutionEvidence` via the S5a tracker, save the SEQ_CLS adapter + scalar score head,
measure the held-out pairwise ranking accuracy (the PROMOTION GATE), and seal a proposed
:class:`RewardSuccessEvidence`. The SEQ_CLS sibling of ``preference_worker.run_preference`` - it consumes
the sealed config DIRECTLY (no lossy mirror), and the ``RewardRunner`` + supervisor independently re-verify
before the evidence is admitted.

A reward model is CHEAPER than DPO: no reference model, no [seq x vocab] log-prob. The randomly-initialized
score head trains alongside the LoRA adapter - PEFT keeps it trainable because the adapter is sealed
``task_type=SEQ_CLS`` - and both are saved into the ``reward_model`` artifact family.

``torch`` is lazy-imported inside ``run_reward``; the training path is ``# pragma: no cover`` (proven by a
GPU run). Only the small pure helpers are base-gate tested."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, cast

from corpus_studio.platform.contracts import (
    ResolvedRewardExecutionConfiguration,
    RewardSuccessEvidence,
)


class RewardWorkerError(RuntimeError):
    """A reward worker run that cannot proceed or produce honest evidence (fail-closed)."""


@dataclass
class RewardRunResult:
    """What the reward worker returns to the runner: the run-scoped adapter directory + the proposed
    success evidence the supervisor independently re-verifies before admission."""

    output_dir: str
    success_evidence: RewardSuccessEvidence


def concrete_reward_max_steps(
    execution: ResolvedRewardExecutionConfiguration, train_pair_count: int
) -> int:
    """The sealed schedule as a CONCRETE optimizer-step count over the TRAINING pairs (post held-out
    split). ``max_steps`` is used verbatim; an epoch-scheduled plan is converted here (this primitive
    processes one pair per microbatch, so an epoch is ``ceil(train_pairs / grad_accum)`` steps)."""
    if execution.schedule.max_steps is not None:
        return execution.schedule.max_steps
    grad_accum = execution.batching.fallback_grad_accumulation_steps or 1
    epochs = execution.schedule.num_train_epochs or 1.0
    steps_per_epoch = max(1, -(-train_pair_count // grad_accum))  # ceil
    return max(1, int(steps_per_epoch * epochs))


def run_reward(  # pragma: no cover - optional training-stack integration; proven by a GPU run
    execution: ResolvedRewardExecutionConfiguration,
    *,
    output_dir: str | None = None,
) -> RewardRunResult:
    """Load the sealed nf4 SEQ_CLS base + LoRA score head, tokenize the sealed preference pairs, hold out a
    deterministic seeded ranking-eval split, train via ``run_reward_training``, assemble the formal
    execution evidence, save the adapter + score head, measure held-out pairwise accuracy, and seal the
    proposed success evidence. Refuses ``cpu_toy`` (nf4 requires CUDA; a CPU reward smoke path is a
    follow-up)."""
    from pathlib import Path

    import torch
    from peft import (  # noqa: PLC0415
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        prepare_model_for_kbit_training,
    )
    from transformers import (  # noqa: PLC0415
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    from corpus_studio.importers.jsonl_importer import read_jsonl  # noqa: PLC0415
    from corpus_studio.platform.enums import StageMarker  # noqa: PLC0415
    from corpus_studio.platform.objectives import get_objective  # noqa: PLC0415
    from corpus_studio.training.optimizer_config import build_torch_optimizer  # noqa: PLC0415
    from corpus_studio.training.pretraining_evidence import (  # noqa: PLC0415
        register_full_model_gradient_hooks,
    )
    from corpus_studio.training.reward_evidence import (  # noqa: PLC0415
        RewardExecutionTracker,
        build_reward_success_evidence,
    )
    from corpus_studio.training.trainer import (  # noqa: PLC0415
        TrainerError,
        capture_adapter_export_state,
        capture_trainable_state,
        evaluate_reward_accuracy,
        expected_saved_adapter_config_sha256,
        format_preference_pair,
        reward_heldout_split,
        run_reward_training,
    )

    if execution.runtime_mode != "training":
        raise RewardWorkerError(
            f"the reward worker runs on GPU (runtime_mode='training'); got {execution.runtime_mode!r} - "
            "nf4 4-bit requires CUDA, so a cpu_toy reward smoke path is a separate follow-up."
        )
    objective = get_objective(execution.objective_ref.id)
    if objective is None:
        raise RewardWorkerError(f"unknown sealed reward objective {execution.objective_ref.id!r}")
    out = Path(output_dir or execution.output_dir)

    # --- data: preference pairs from the sealed PreferenceDataPolicy dataset binding ---
    rows = list(read_jsonl(Path(execution.inputs.dataset.location)))
    if not rows:
        raise RewardWorkerError("the sealed preference dataset is empty")
    base_model = execution.inputs.model.location
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pairs = [format_preference_pair(row, tokenizer) for row in rows]

    # A reward model's promotion gate is HELD-OUT pairwise ranking accuracy: carve a reproducible held-out
    # ranking set from the sealed pairs, seeded by the sealed data_seed (the count is recorded as evidence,
    # so the carve-out is never silent). Train on the remainder.
    try:
        train_idx, heldout_idx = reward_heldout_split(len(pairs), data_seed=execution.data_seed)
    except TrainerError as exc:
        raise RewardWorkerError(str(exc)) from exc
    train_pairs = [pairs[index] for index in train_idx]
    heldout_pairs = [pairs[index] for index in heldout_idx]

    # --- model: nf4 SEQ_CLS base (num_labels=1 scalar head) + LoRA from the sealed adapter spec ---
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=execution.bnb_4bit_use_double_quant,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=1, quantization_config=bnb, device_map={"": 0}
    )
    # SEQ_CLS models need a pad id on the config; we pool the score at the explicit last content token, so
    # this only guards any internal length bookkeeping.
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=execution.gradient_checkpointing
    )
    adapter = execution.adapter
    # PEFT reads the "all-linear" sentinel only as a STRING; the sealed config carries it as ["all-linear"]
    # (a list), which PEFT would treat as a literal module name. Lower it exactly as the SFT/DPO trainers do.
    target_modules: Any = (
        adapter.target_modules[0]
        if adapter.target_modules == ["all-linear"]
        else adapter.target_modules
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=adapter.lora_r,
            lora_alpha=adapter.lora_alpha,
            lora_dropout=adapter.lora_dropout,
            bias=adapter.bias,
            task_type=execution.adapter_task_type,  # SEQ_CLS -> PEFT keeps the score head trainable
            target_modules=target_modules,
        ),
    )
    model.train()

    # --- evidence capture: register post-accumulation gradient hooks + snapshot BEFORE training ---
    gradient_tracker = register_full_model_gradient_hooks(model, torch)
    before_trainable = capture_trainable_state(model, torch, stage=StageMarker.adapter_attached)
    before_export = capture_adapter_export_state(
        get_peft_model_state_dict(model), torch, stage=StageMarker.adapter_attached
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    # Build the optimizer the SEAL specifies (impl / betas / eps / weight_decay) from the ONE shared
    # lowering - dropping any sealed field is the "no silent trainer-field filtering" invariant.
    opt = execution.optimizer
    try:
        optimizer = build_torch_optimizer(opt, trainable)
    except ValueError as exc:
        raise RewardWorkerError(str(exc)) from exc
    max_steps = concrete_reward_max_steps(execution, len(train_pairs))
    tracker = RewardExecutionTracker(expected_steps=max_steps, gradients=gradient_tracker)
    tracker.on_train_begin(optimizer)

    # --- train via the reward primitive (a TrainerError is a fail-closed data/finiteness refusal) ---
    try:
        result = run_reward_training(
            model,
            tokenizer,
            train_pairs,
            seq_len=execution.sequence.max_sequence_len,
            margin=execution.reward.margin,
            learning_rate=opt.learning_rate,
            max_steps=max_steps,
            gradient_accumulation_steps=execution.batching.fallback_grad_accumulation_steps or 1,
            max_prompt_length=execution.data.max_prompt_length,
            gradient_checkpointing=execution.gradient_checkpointing,
            max_grad_norm=opt.max_grad_norm,
            optimizer=optimizer,
            truncation_allowed=execution.sequence.truncation_allowed,
        )
    except TrainerError as exc:
        raise RewardWorkerError(f"the reward training primitive refused the run: {exc}") from exc

    # --- replay the per-step evidence into the tracker, snapshot AFTER, and seal the execution evidence ---
    losses = result["losses"]
    margins = result["reward_margins"]
    chosen = result["chosen_rewards"]
    rejected = result["rejected_rewards"]
    for index in range(len(losses)):
        tracker.record_step(
            index + 1,
            loss=losses[index],
            chosen_reward=chosen[index],
            rejected_reward=rejected[index],
            margin=margins[index],
        )
    after_trainable = capture_trainable_state(model, torch, stage=StageMarker.optimizer_step)
    after_export = capture_adapter_export_state(
        get_peft_model_state_dict(model), torch, stage=StageMarker.optimizer_step
    )
    execution_evidence = tracker.finalize(
        steps=len(losses),
        before=before_trainable,
        after=after_trainable,
        before_export=before_export,
        after_export=after_export,
        adapter_config_semantic_sha256=expected_saved_adapter_config_sha256(
            model, cast(Any, types.SimpleNamespace(base_model=base_model))
        ),
        reward_pairs_consumed=len(train_pairs),
    )

    # --- save the PEFT adapter (LoRA + score head) + tokenizer ---
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    # --- measure the PROMOTION GATE: held-out pairwise ranking accuracy (never a falling training loss) ---
    try:
        heldout = evaluate_reward_accuracy(
            model,
            tokenizer,
            heldout_pairs,
            seq_len=execution.sequence.max_sequence_len,
            max_prompt_length=execution.data.max_prompt_length,
            truncation_allowed=execution.sequence.truncation_allowed,
        )
    except TrainerError as exc:
        raise RewardWorkerError(f"the held-out reward evaluation refused the pairs: {exc}") from exc

    return RewardRunResult(
        output_dir=str(out),
        success_evidence=build_reward_success_evidence(
            str(out),
            execution_evidence,
            heldout_pairwise_accuracy=heldout["preference_accuracy"],
            heldout_pairs_evaluated=heldout["n"],
        ),
    )
