"""On-policy RL (GRPO) config-consuming worker: lower a sealed
:class:`ResolvedRolloutExecutionConfiguration` to the :func:`run_rollout_training` primitive, capture the
formal :class:`RolloutExecutionEvidence` via the S5b tracker, save the trained POLICY adapter, measure the
held-out mean-reward LIFT under a bounded KL (the PROMOTION GATE), and seal a proposed
:class:`RolloutSuccessEvidence`. The on-policy sibling of ``reward_worker.run_reward``.

Unlike the other workers this loads TWO models: the nf4 CAUSAL_LM policy (+ LoRA) that trains, and the
provenance-bound nf4 SEQ_CLS reward model (served, inference-only) that scores each rollout. Both are loaded
with ``trust_remote_code`` honored from the seal (audit F1). The KL reference is the frozen policy base via
``disable_adapter`` (no third model).

``torch`` is lazy-imported inside ``run_rollout``; the training path is ``# pragma: no cover`` (proven by a
GPU run). Only the small pure helper is base-gate tested."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, cast

from corpus_studio.platform.contracts import (
    ResolvedRolloutExecutionConfiguration,
    RolloutSuccessEvidence,
)


class RolloutWorkerError(RuntimeError):
    """An on-policy RL worker run that cannot proceed or produce honest evidence (fail-closed)."""


@dataclass
class RolloutRunResult:
    """What the rollout worker returns to the runner: the run-scoped policy-adapter directory + the proposed
    success evidence the supervisor independently re-verifies before admission."""

    output_dir: str
    success_evidence: RolloutSuccessEvidence


def concrete_rollout_max_steps(
    execution: ResolvedRolloutExecutionConfiguration, train_prompt_count: int
) -> int:
    """The sealed schedule as a CONCRETE optimizer-step count over the TRAINING prompts (post held-out
    split). ``max_steps`` is used verbatim; an epoch-scheduled plan is converted here (one prompt per
    optimizer step by default, so an epoch is ``ceil(train_prompts / grad_accum)`` steps)."""
    if execution.schedule.max_steps is not None:
        return execution.schedule.max_steps
    grad_accum = execution.batching.fallback_grad_accumulation_steps or 1
    epochs = execution.schedule.num_train_epochs or 1.0
    steps_per_epoch = max(1, -(-train_prompt_count // grad_accum))  # ceil
    return max(1, int(steps_per_epoch * epochs))


def run_rollout(  # pragma: no cover - optional training-stack integration; proven by a GPU run
    execution: ResolvedRolloutExecutionConfiguration,
    *,
    output_dir: str | None = None,
) -> RolloutRunResult:
    """Load the sealed nf4 policy (+ LoRA) and the provenance-bound served reward model, format the sealed
    chat prompts, hold out a deterministic seeded split, train via ``run_rollout_training`` (GRPO), assemble
    the formal execution evidence, save the policy adapter, measure the held-out mean-reward LIFT + max KL,
    and seal the proposed success evidence. Refuses ``cpu_toy`` (nf4 requires CUDA)."""
    from pathlib import Path

    import torch
    from peft import (  # noqa: PLC0415
        LoraConfig,
        PeftModel,
        get_peft_model,
        get_peft_model_state_dict,
        prepare_model_for_kbit_training,
    )
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
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
    from corpus_studio.training.rollout_evidence import (  # noqa: PLC0415
        RolloutExecutionTracker,
        build_rollout_success_evidence,
    )
    from corpus_studio.training.trainer import (  # noqa: PLC0415
        TrainerError,
        _score_reward_branch,
        _seqcls_backbone_and_score_head,
        capture_adapter_export_state,
        capture_trainable_state,
        evaluate_rollout_kl,
        evaluate_rollout_reward,
        expected_saved_adapter_config_sha256,
        reward_heldout_split,
        run_rollout_training,
    )

    if execution.runtime_mode != "training":
        raise RolloutWorkerError(
            f"the rollout worker runs on GPU (runtime_mode='training'); got {execution.runtime_mode!r} - "
            "nf4 4-bit requires CUDA, so a cpu_toy rollout smoke path is a separate follow-up."
        )
    objective = get_objective(execution.objective_ref.id)
    if objective is None:
        raise RolloutWorkerError(f"unknown sealed rollout objective {execution.objective_ref.id!r}")
    out = Path(output_dir or execution.output_dir)
    seq_len = execution.sequence.max_sequence_len
    max_prompt_length = execution.experience.max_prompt_length

    # --- data: chat prompts from the sealed ExperienceSource dataset binding ---
    rows = list(read_jsonl(Path(execution.inputs.dataset.location)))
    if not rows:
        raise RolloutWorkerError("the sealed rollout prompt dataset is empty")
    base_model = execution.inputs.model.location
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=execution.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _format_prompt(row: dict[str, Any]) -> str:
        messages = row.get("messages")
        if not messages:
            raise RolloutWorkerError("each rollout prompt row requires a non-empty 'messages' list")
        return str(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    prompts = [_format_prompt(row) for row in rows]
    try:
        train_idx, heldout_idx = reward_heldout_split(len(prompts), data_seed=execution.data_seed)
    except TrainerError as exc:
        raise RolloutWorkerError(str(exc)) from exc
    train_prompts = [prompts[index] for index in train_idx]
    heldout_prompts = [prompts[index] for index in heldout_idx]

    # --- served reward model: the provenance-bound nf4 SEQ_CLS reward model (inference-only scorer) ---
    reward_source = execution.reward_source
    reward_bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=execution.bnb_4bit_use_double_quant,
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(
        reward_source.reward_base_model, trust_remote_code=execution.trust_remote_code
    )
    if reward_tokenizer.pad_token_id is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_base = AutoModelForSequenceClassification.from_pretrained(
        reward_source.reward_base_model, num_labels=1, quantization_config=reward_bnb,
        device_map={"": 0}, trust_remote_code=execution.trust_remote_code,
    )
    reward_base.config.pad_token_id = reward_tokenizer.pad_token_id
    reward_model = PeftModel.from_pretrained(reward_base, reward_source.reward_adapter_location)
    reward_model.eval()
    reward_backbone, reward_score_head = _seqcls_backbone_and_score_head(reward_model)
    reward_device = next(reward_model.parameters()).device

    def _reward_scorer(prompt: str, completion: str) -> float:
        ids = reward_tokenizer(prompt + completion, add_special_tokens=False)["input_ids"][:seq_len]
        if not ids:
            return 0.0
        with torch.no_grad():
            score = _score_reward_branch(reward_backbone, reward_score_head, ids, len(ids), reward_device)
        return float(score)

    # --- policy: nf4 CAUSAL_LM base + LoRA adapter from the sealed adapter spec (bias-free reference) ---
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=execution.bnb_4bit_use_double_quant,
    )
    policy = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map={"": 0},
        trust_remote_code=execution.trust_remote_code,
    )
    policy = prepare_model_for_kbit_training(
        policy, use_gradient_checkpointing=execution.gradient_checkpointing
    )
    adapter = execution.adapter
    target_modules: Any = (
        adapter.target_modules[0]
        if adapter.target_modules == ["all-linear"]
        else adapter.target_modules
    )
    policy = get_peft_model(
        policy,
        LoraConfig(
            r=adapter.lora_r, lora_alpha=adapter.lora_alpha, lora_dropout=adapter.lora_dropout,
            bias=adapter.bias, task_type=execution.adapter_task_type, target_modules=target_modules,
        ),
    )
    policy.train()

    # --- evidence capture: register gradient hooks + snapshot BEFORE training ---
    gradient_tracker = register_full_model_gradient_hooks(policy, torch)
    before_trainable = capture_trainable_state(policy, torch, stage=StageMarker.adapter_attached)
    before_export = capture_adapter_export_state(
        get_peft_model_state_dict(policy), torch, stage=StageMarker.adapter_attached
    )
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    opt = execution.optimizer
    try:
        optimizer = build_torch_optimizer(opt, trainable)
    except ValueError as exc:
        raise RolloutWorkerError(str(exc)) from exc
    max_steps = concrete_rollout_max_steps(execution, len(train_prompts))
    tracker = RolloutExecutionTracker(expected_steps=max_steps, gradients=gradient_tracker)
    tracker.on_train_begin(optimizer)

    # --- train via the GRPO primitive ---
    try:
        result = run_rollout_training(
            policy,
            tokenizer,
            train_prompts,
            _reward_scorer,
            seq_len=seq_len,
            max_new_tokens=execution.rollout.max_new_tokens,
            group_size=execution.rollout.rollouts_per_prompt,
            max_steps=max_steps,
            sampling_temperature=execution.rollout.sampling_temperature,
            sampling_top_p=execution.rollout.sampling_top_p,
            kl_coefficient=execution.stability.kl_coefficient,
            clip_range=execution.stability.clip_range,
            entropy_bonus=execution.stability.entropy_bonus,
            max_prompt_length=max_prompt_length,
            learning_rate=opt.learning_rate,
            gradient_accumulation_steps=execution.batching.fallback_grad_accumulation_steps or 1,
            max_grad_norm=opt.max_grad_norm,
            optimizer=optimizer,
            gradient_checkpointing=execution.gradient_checkpointing,
        )
    except TrainerError as exc:
        raise RolloutWorkerError(f"the GRPO training primitive refused the run: {exc}") from exc

    # --- replay the per-step evidence into the tracker, snapshot AFTER, seal the execution evidence ---
    for index in range(len(result["losses"])):
        tracker.record_step(
            index + 1,
            loss=result["losses"][index],
            rollouts_sampled=result["rollouts_per_step"][index],
            mean_reward=result["mean_rewards"][index],
            kl_to_reference=result["kls"][index],
            entropy=result["entropies"][index],
            mean_advantage=result["mean_advantages"][index],
        )
    after_trainable = capture_trainable_state(policy, torch, stage=StageMarker.optimizer_step)
    after_export = capture_adapter_export_state(
        get_peft_model_state_dict(policy), torch, stage=StageMarker.optimizer_step
    )
    execution_evidence = tracker.finalize(
        steps=len(result["losses"]),
        before=before_trainable,
        after=after_trainable,
        before_export=before_export,
        after_export=after_export,
        adapter_config_semantic_sha256=expected_saved_adapter_config_sha256(
            policy, cast(Any, types.SimpleNamespace(base_model=base_model))
        ),
    )

    # --- save the trained POLICY adapter + tokenizer ---
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    # --- PROMOTION GATE: held-out mean-reward LIFT (policy vs the frozen reference) under a bounded KL ---
    max_new_tokens = execution.rollout.max_new_tokens
    temperature = execution.rollout.sampling_temperature
    top_p = execution.rollout.sampling_top_p
    try:
        policy_reward = evaluate_rollout_reward(
            policy, tokenizer, heldout_prompts, _reward_scorer, max_new_tokens=max_new_tokens,
            sampling_temperature=temperature, sampling_top_p=top_p,
            max_prompt_length=max_prompt_length, use_reference=False)
        baseline_reward = evaluate_rollout_reward(
            policy, tokenizer, heldout_prompts, _reward_scorer, max_new_tokens=max_new_tokens,
            sampling_temperature=temperature, sampling_top_p=top_p,
            max_prompt_length=max_prompt_length, use_reference=True)
        max_kl = evaluate_rollout_kl(
            policy, tokenizer, heldout_prompts, max_new_tokens=max_new_tokens,
            sampling_temperature=temperature, sampling_top_p=top_p,
            max_prompt_length=max_prompt_length)
    except TrainerError as exc:
        raise RolloutWorkerError(f"the held-out rollout evaluation refused the prompts: {exc}") from exc

    # The KL bound is the sealed adaptive target when set, else the sealed penalty coefficient's implied
    # ceiling; a success that exceeds it is refused by the RolloutSuccessEvidence validator (reward-hacking).
    kl_bound = execution.stability.kl_target or max(execution.stability.kl_coefficient, 1e-3)
    return RolloutRunResult(
        output_dir=str(out),
        success_evidence=build_rollout_success_evidence(
            str(out),
            execution_evidence,
            heldout_prompts_evaluated=policy_reward["n"],
            heldout_baseline_mean_reward=baseline_reward["mean_reward"],
            heldout_policy_mean_reward=policy_reward["mean_reward"],
            heldout_max_kl_to_reference=max_kl,
            kl_bound=kl_bound,
        ),
    )
