"""Offline-DPO (preference) config-consuming worker: lower a sealed
:class:`ResolvedPreferenceExecutionConfiguration` to the #781 ``run_dpo_training`` primitive, capture the
formal :class:`PreferenceExecutionEvidence` via the #814 tracker, save the PEFT adapter, and seal a proposed
:class:`PreferenceSuccessEvidence`. The DPO sibling of ``pretraining_trainer.run_pretraining`` - it consumes
the sealed config DIRECTLY (no lossy mirror), and the ``PreferenceRunner`` + supervisor independently
re-verify before the evidence is admitted.

``torch`` is lazy-imported inside ``run_preference``; the training path is ``# pragma: no cover`` (proven by
a GPU run). Only the small pure helpers are base-gate tested."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from corpus_studio.platform.contracts import (
    PreferenceSuccessEvidence,
    ResolvedPreferenceExecutionConfiguration,
)


class PreferenceWorkerError(RuntimeError):
    """A DPO worker run that cannot proceed or produce honest evidence (fail-closed)."""


@dataclass
class PreferenceRunResult:
    """What the DPO worker returns to the runner: the run-scoped adapter directory + the proposed success
    evidence the supervisor independently re-verifies before admission."""

    output_dir: str
    success_evidence: PreferenceSuccessEvidence


def score_response_eos_for(objective: Any) -> bool:
    """Whether the appended response EOS is a loss-bearing label, derived from the SEALED objective's
    primary mask - never hardcoded. The ``dpo_qlora`` primary mask sets ``include_special_tokens=False``,
    so the EOS is well-formed in the input but labeled ``-100`` (the worker passes ``score_eos=False``)."""
    ref = getattr(objective, "primary_mask_ref", None) or "primary_mask"
    masks = getattr(objective, "token_masks", None) or getattr(objective, "loss_masks", None) or []
    for mask in masks:
        if ref in (getattr(mask, "mask_id", None), getattr(mask, "mask_ref", None)):
            return bool(getattr(mask, "include_special_tokens", False))
    return False


def concrete_max_steps(execution: ResolvedPreferenceExecutionConfiguration, pair_count: int) -> int:
    """The sealed schedule as a CONCRETE optimizer-step count. ``max_steps`` is used verbatim; an
    epoch-scheduled plan is converted here (the worker knows the dataset size - this primitive processes
    one pair per microbatch, so an epoch is ``ceil(pairs / grad_accum)`` steps)."""
    if execution.schedule.max_steps is not None:
        return execution.schedule.max_steps
    grad_accum = execution.batching.fallback_grad_accumulation_steps or 1
    epochs = execution.schedule.num_train_epochs or 1.0
    steps_per_epoch = max(1, -(-pair_count // grad_accum))  # ceil
    return max(1, int(steps_per_epoch * epochs))


def paged_adamw_requested(optimizer_spec: Any) -> bool:
    """Whether the SEALED optimizer impl is bitsandbytes paged 8-bit AdamW (vs plain ``adamw_torch``) - a
    PURE read of the sealed enum. The DPO worker must build the memory-saving optimizer the seal specifies:
    paged 8-bit is what keeps 4B DPO inside 12 GB at seq 4096, so silently substituting a full-precision
    AdamW would both violate the seal and erase that headroom."""
    impl = getattr(optimizer_spec, "impl", None)
    return getattr(impl, "value", impl) == "paged_adamw_8bit"


def run_preference(  # pragma: no cover - optional training-stack integration; proven by a GPU run
    execution: ResolvedPreferenceExecutionConfiguration,
    *,
    output_dir: str | None = None,
) -> PreferenceRunResult:
    """Load the sealed nf4 base + LoRA adapter, tokenize the sealed preference pairs, train via
    ``run_dpo_training``, assemble the formal execution evidence, save the adapter, and seal the proposed
    success evidence. Refuses ``cpu_toy`` (nf4 requires CUDA; a CPU DPO smoke path is a follow-up)."""
    import types
    from pathlib import Path

    import torch
    from peft import (  # noqa: PLC0415
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        prepare_model_for_kbit_training,
    )
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    from corpus_studio.importers.jsonl_importer import read_jsonl  # noqa: PLC0415
    from corpus_studio.platform.enums import StageMarker  # noqa: PLC0415
    from corpus_studio.platform.objectives import get_objective  # noqa: PLC0415
    from corpus_studio.training.pretraining_evidence import (  # noqa: PLC0415
        register_full_model_gradient_hooks,
    )
    from corpus_studio.training.preference_evidence import (  # noqa: PLC0415
        PreferenceExecutionTracker,
        build_preference_success_evidence,
    )
    from corpus_studio.training.trainer import (  # noqa: PLC0415
        TrainerError,
        capture_adapter_export_state,
        capture_trainable_state,
        expected_saved_adapter_config_sha256,
        format_preference_pair,
        run_dpo_training,
    )

    if execution.runtime_mode != "training":
        raise PreferenceWorkerError(
            f"the DPO worker runs on GPU (runtime_mode='training'); got {execution.runtime_mode!r} - "
            "nf4 4-bit requires CUDA, so a cpu_toy DPO smoke path is a separate follow-up."
        )
    objective = get_objective(execution.objective_ref.id)
    if objective is None:
        raise PreferenceWorkerError(f"unknown sealed preference objective {execution.objective_ref.id!r}")
    out = Path(output_dir or execution.output_dir)

    # --- data: preference pairs from the sealed PreferenceDataPolicy dataset binding ---
    rows = list(read_jsonl(Path(execution.inputs.dataset.location)))
    if not rows:
        raise PreferenceWorkerError("the sealed preference dataset is empty")
    base_model = execution.inputs.model.location
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pairs = [format_preference_pair(row, tokenizer) for row in rows]

    # --- model: nf4 base + LoRA adapter from the sealed adapter spec (bias-free reference; frozen head) ---
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=execution.bnb_4bit_use_double_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map={"": 0}
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=execution.gradient_checkpointing
    )
    adapter = execution.adapter
    # PEFT reads the "all-linear" sentinel only as a STRING; the sealed config carries it as ["all-linear"]
    # (a list), which PEFT would treat as a literal module name. Lower it exactly as the SFT trainer does.
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
            task_type=execution.adapter_task_type,
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
    # Build the optimizer the SEAL specifies - impl (paged 8-bit vs adamw_torch), betas, eps, weight_decay -
    # not a plain lr-only AdamW: the sealed paged 8-bit optimizer is what keeps 4B DPO inside 12 GB at seq
    # 4096, and dropping the sealed fields is exactly the "no silent trainer-field filtering" invariant.
    opt = execution.optimizer
    betas = (opt.adam_beta1, opt.adam_beta2)
    weight_decay = opt.weight_decay or 0.0
    if paged_adamw_requested(opt):
        from bitsandbytes.optim import PagedAdamW8bit  # noqa: PLC0415

        optimizer: Any = PagedAdamW8bit(
            trainable, lr=opt.learning_rate, betas=betas, eps=opt.adam_epsilon, weight_decay=weight_decay
        )
    elif getattr(opt.impl, "value", opt.impl) == "adamw_torch":
        optimizer = torch.optim.AdamW(
            trainable, lr=opt.learning_rate, betas=betas, eps=opt.adam_epsilon, weight_decay=weight_decay
        )
    else:
        raise PreferenceWorkerError(
            f"the DPO worker does not support the sealed optimizer impl {opt.impl!r} "
            "(expected adamw_torch or paged_adamw_8bit)"
        )
    max_steps = concrete_max_steps(execution, len(pairs))
    tracker = PreferenceExecutionTracker(expected_steps=max_steps, gradients=gradient_tracker)
    tracker.on_train_begin(optimizer)

    # --- train via the #781 primitive (it OWNS stdout-safe training; the runner wraps the protocol) ---
    # A TrainerError is a fail-closed data/finiteness refusal from the primitive; surface it as the worker's
    # typed error so the PreferenceRunner maps it to a classified RunnerFailure, not the generic catch-all.
    try:
        result = run_dpo_training(
            model,
            tokenizer,
            pairs,
            seq_len=execution.sequence.max_sequence_len,
            chunk_size=execution.preference.sequence_chunk_size,
            beta=execution.preference.beta,
            label_smoothing=execution.preference.label_smoothing,
            average_log_prob=execution.preference.average_log_prob,
            learning_rate=opt.learning_rate,
            max_steps=max_steps,
            gradient_accumulation_steps=execution.batching.fallback_grad_accumulation_steps or 1,
            max_prompt_length=execution.data.max_prompt_length,
            score_response_eos=score_response_eos_for(objective),
            gradient_checkpointing=execution.gradient_checkpointing,
            max_grad_norm=opt.max_grad_norm,
            optimizer=optimizer,
            truncation_allowed=execution.sequence.truncation_allowed,
        )
    except TrainerError as exc:
        raise PreferenceWorkerError(f"the DPO training primitive refused the run: {exc}") from exc

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
        # The SFT helper only reads config.base_model (as a fallback when the PEFT config omits it), so a
        # tiny shim reuses its fail-closed one-adapter validation + canonical hash without a full config.
        adapter_config_semantic_sha256=expected_saved_adapter_config_sha256(
            model, cast(Any, types.SimpleNamespace(base_model=base_model))
        ),
        preference_pairs_consumed=len(pairs),
    )

    # --- save the PEFT adapter + tokenizer, then seal the proposed success evidence ---
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    return PreferenceRunResult(
        output_dir=str(out),
        success_evidence=build_preference_success_evidence(str(out), execution_evidence),
    )
