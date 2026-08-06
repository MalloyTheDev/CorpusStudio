"""``run_full_finetune`` - the full-parameter supervised fine-tune worker (dense_full_finetune slice 2).

It consumes the sealed :class:`ResolvedFullFinetuneExecutionConfiguration` DIRECTLY (faithful by
construction). It is the FULL-MODEL sibling of the adapter SFT worker: same instruction/chat SFT data, but
ALL parameters train and the artifact is a full model. It reuses the pretraining worker's full-model
machinery verbatim (gradient-observation hooks, the execution tracker, the single-file save, and the
independent success-evidence build) - the ONLY differences from ``run_pretraining`` are ``from_pretrained``
(a real base model, not a random-init config) and an SFT-formatted text dataset (not a packed corpus).

``torch`` + ``transformers`` are lazy-imported; the training loop is ``# pragma: no cover`` (proven by a
run). The pure row-padding helper is base-gate tested. This slice is UNROUTED: ``required_runner_lane``
still refuses a full-finetune plan at execution, so nothing runs it in production and no wheel is needed
until the (gated) promotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from corpus_studio.platform.contracts import (
    PretrainingSuccessEvidence,
    ResolvedFullFinetuneExecutionConfiguration,
)


class FullFinetuneError(RuntimeError):
    """A full-parameter fine-tune the worker cannot honor (fail-closed, a clean typed error)."""


@dataclass
class FullFinetuneRunResult:
    """What the full-finetune worker returns: the run-scoped model directory + the proposed full-model
    success evidence the supervisor independently re-verifies before admission. Full-parameter SFT is
    full-model (like pretraining), so it reuses the full-model :class:`PretrainingSuccessEvidence` shape."""

    output_dir: str
    success_evidence: PretrainingSuccessEvidence


def pad_sft_row(input_ids: list[int], seq_len: int, pad_id: int) -> dict[str, list[int]]:
    """PURE + torch-free. Right-truncate then right-pad ONE tokenized SFT example to ``seq_len`` for a
    fixed-shape batch: ``input_ids`` padded with ``pad_id``, ``labels`` mirroring ``input_ids`` but ``-100``
    on the pad tail (never train on padding), and an ``attention_mask`` over the true content. The current
    first-party SFT trainer trains on the WHOLE sequence (no completion-only mask yet), so labels mirror the
    content verbatim - this worker matches that exactly."""
    content = input_ids[:seq_len]
    n = len(content)
    pad = seq_len - n
    return {
        "input_ids": content + [pad_id] * pad,
        "labels": content + [-100] * pad,
        "attention_mask": [1] * n + [0] * pad,
    }


def run_full_finetune(  # pragma: no cover - torch/transformers integration; proven by a run
    execution: ResolvedFullFinetuneExecutionConfiguration,
    *,
    output_dir: str | None = None,
    cpu_toy: bool = False,
) -> FullFinetuneRunResult:
    """Load the sealed base model at full precision (all parameters trainable), tokenize the sealed SFT
    dataset, train full-parameter via the HF Trainer, capture the full-model execution evidence, save the
    full model, and seal the proposed success evidence. Refuses a quantized config (the contract guarantees
    unquantized, but fail closed anyway)."""
    from pathlib import Path

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    from corpus_studio.importers.jsonl_importer import read_jsonl
    from corpus_studio.platform.enums import StageMarker
    from corpus_studio.training.optimizer_config import hf_training_arguments_optimizer_kwargs
    from corpus_studio.training.pretraining_evidence import (
        PretrainingExecutionTracker,
        register_full_model_gradient_hooks,
    )
    from corpus_studio.training.pretraining_trainer import (
        _build_success_evidence,
        _canonical_config_sha256,
    )
    from corpus_studio.training.trainer import (
        capture_adapter_export_state,
        capture_trainable_state,
        format_example_text,
    )

    if execution.precision.quantized_storage_format.value != "none":
        raise FullFinetuneError(
            "full-parameter fine-tuning must be unquantized; the sealed config is quantized"
        )
    set_seed(execution.seed)
    out = Path(output_dir or execution.output_dir)
    base_model = execution.inputs.model.location

    # --- model + tokenizer: a real base at full precision, ALL parameters trainable (no adapter, no nf4) ---
    dtype = torch.bfloat16 if execution.precision.forward_compute_dtype.value == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- data: the sealed SFT rows, formatted + tokenized to fixed length (whole-sequence loss, per above) ---
    rows = list(read_jsonl(Path(execution.inputs.dataset.location)))
    if not rows:
        raise FullFinetuneError("the sealed full-finetune dataset is empty")
    seq_len = execution.sequence.max_sequence_len
    built = [
        pad_sft_row(
            tokenizer(
                format_example_text(row, execution.data.dataset_format, tokenizer),
                truncation=True, max_length=seq_len, add_special_tokens=True,
            )["input_ids"],
            seq_len,
            tokenizer.pad_token_id,
        )
        for row in rows
    ]
    dataset = Dataset.from_list(built)

    def _collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            key: torch.tensor([feature[key] for feature in features], dtype=torch.long)
            for key in ("input_ids", "labels", "attention_mask")
        }

    if execution.gradient_checkpointing and not cpu_toy:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # --- evidence capture (reused pretraining primitives): hooks + BEFORE snapshot, full-parameter ---
    gradient_tracker = register_full_model_gradient_hooks(model, torch)
    sealed_max_steps = execution.schedule.max_steps
    epoch_mode = sealed_max_steps is None
    sealed_epochs = execution.schedule.num_train_epochs
    tracker = PretrainingExecutionTracker(
        expected_steps=sealed_max_steps if sealed_max_steps is not None else 0,
        gradients=gradient_tracker,
    )

    def _trainable_mapping() -> dict[str, Any]:
        return {name: param for name, param in model.named_parameters() if param.requires_grad}

    before_trainable = capture_trainable_state(model, torch, stage=StageMarker.optimizer_step)
    before_export = capture_adapter_export_state(
        _trainable_mapping(), torch, stage=StageMarker.optimizer_step
    )

    class _EvidenceCallback(TrainerCallback):  # type: ignore[misc]
        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            tracker.on_train_begin(kwargs.get("optimizer"))

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            tracker.on_step_end(int(state.global_step), kwargs.get("optimizer"))

        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
            tracker.on_log(int(state.global_step), logs)

    arguments = TrainingArguments(
        output_dir=str(out),
        max_steps=sealed_max_steps if sealed_max_steps is not None else -1,
        num_train_epochs=float(sealed_epochs) if epoch_mode and sealed_epochs is not None else 1.0,
        per_device_train_batch_size=execution.batching.micro_batch_size,
        gradient_accumulation_steps=execution.batching.fallback_grad_accumulation_steps or 1,
        **hf_training_arguments_optimizer_kwargs(execution.optimizer),
        seed=execution.seed,
        data_seed=execution.data_seed,
        logging_steps=1,
        logging_nan_inf_filter=False,
        save_strategy="no",
        report_to=[],
        use_cpu=cpu_toy,
        gradient_checkpointing=execution.gradient_checkpointing and not cpu_toy,
    )
    trainer = Trainer(
        model=model, args=arguments, train_dataset=dataset,
        data_collator=_collate, callbacks=[_EvidenceCallback()],
    )
    train_output = trainer.train()
    steps = int(getattr(train_output, "global_step", 0) or 0)
    if epoch_mode:
        planned_steps = int(getattr(trainer.state, "max_steps", 0) or 0)
        if planned_steps < 1:
            raise FullFinetuneError("epoch-scheduled full-finetune computed a non-positive step plan")
        tracker.expected_steps = planned_steps

    # --- after training: verify inventory, snapshot AFTER, seal execution evidence BEFORE saving ---
    gradient_tracker.verify_model_inventory(model, stage=StageMarker.optimizer_step)
    after_trainable = capture_trainable_state(model, torch, stage=StageMarker.optimizer_step)
    after_export = capture_adapter_export_state(
        _trainable_mapping(), torch, stage=StageMarker.optimizer_step
    )
    execution_detail = tracker.finalize(
        steps=steps,
        before=before_trainable,
        after=after_trainable,
        before_export=before_export,
        after_export=after_export,
        model_config_semantic_sha256=_canonical_config_sha256(model.config.to_dict()),
    )

    # --- save the FULL model as a single safetensors file + seal the proposed success evidence ---
    out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(out), safe_serialization=True, max_shard_size="1000GB")
    tokenizer.save_pretrained(str(out))
    return FullFinetuneRunResult(
        output_dir=str(out),
        success_evidence=_build_success_evidence(out, execution_detail),
    )
