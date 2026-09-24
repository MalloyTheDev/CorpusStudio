"""``run_full_finetune`` - the full-parameter supervised fine-tune worker (dense_full_finetune slice 2).

It consumes the sealed :class:`ResolvedFullFinetuneExecutionConfiguration` DIRECTLY (faithful by
construction). It is the FULL-MODEL sibling of the adapter SFT worker: same instruction/chat SFT data, but
ALL parameters train and the artifact is a full model. It reuses the pretraining worker's full-model
machinery verbatim (gradient-observation hooks, the execution tracker, the single-file save, and the
independent success-evidence build) - the ONLY differences from ``run_pretraining`` are ``from_pretrained``
(a real base model, not a random-init config) and an SFT-formatted text dataset (not a packed corpus).

The SFT rows are the ones the ``FullFinetuneRunner`` parsed from its single verified read of the sealed
dataset (``training.sealed_inputs``: one stable read, sha256 compared with the seal, the same bytes
parsed), so a changed dataset is refused before any weights load; the worker accepts only rows bound to
its own sealed binding and never reopens the dataset path.

The tokenizer and model are loaded through ``training.sealed_loader``: the tokenizer first, from its OWN
sealed binding and revision (with a sealed chat-template digest checked); then, after the SDPA toggles are
applied and the sealed kernel is probed, the model with the sealed revision, safetensors-only policy,
weight-storage dtype, root device and attention API. Local inputs are re-hashed and the attention API,
placement and storage dtype are observed after the load, placement again after the HF Trainer takes the
model, and training runs inside the exclusive sealed-kernel context. ``cpu_toy`` comes from the sealed
``runtime_mode``, never from the caller. A sealed loader value this worker cannot lower is refused before
anything is imported or loaded.

``torch`` + ``transformers`` are lazy-imported; the training loop is ``# pragma: no cover`` (proven by a
run). The pure row-padding helper is base-gate tested. This slice is UNROUTED: ``required_runner_lane``
still refuses a full-finetune plan at execution, so nothing runs it in production and no wheel is needed
until the (gated) promotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from corpus_studio.platform.contracts import (
    PretrainingSuccessEvidence,
    ResolvedFullFinetuneExecutionConfiguration,
)

if TYPE_CHECKING:
    from corpus_studio.training.sealed_inputs import VerifiedDataset
    from corpus_studio.training.sealed_loader import StageFn


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
    dataset: VerifiedDataset,
    output_dir: str | None = None,
    stage_callback: StageFn | None = None,
) -> FullFinetuneRunResult:
    """Load the sealed base model at full precision (all parameters trainable), tokenize the sealed SFT
    dataset, train full-parameter via the HF Trainer, capture the full-model execution evidence, save the
    full model, and seal the proposed success evidence. Refuses a quantized config (the contract guarantees
    unquantized, but fail closed anyway) and any other sealed loader value it cannot lower.
    ``stage_callback(name, message)`` receives the loader and verification stages."""
    # Rows come only from the runner's single verified read of the sealed dataset (the bytes whose sha256
    # matched the seal). The mutable dataset path is never reopened here, and the binding is checked
    # before anything heavy is imported or loaded.
    from corpus_studio.training.sealed_inputs import (  # noqa: PLC0415
        SealedInputError,
        require_verified_dataset,
    )

    try:
        rows = list(require_verified_dataset(dataset, execution.inputs.dataset).rows)
    except SealedInputError as exc:
        raise FullFinetuneError(str(exc)) from exc
    if not rows:
        raise FullFinetuneError("the sealed full-finetune dataset is empty")
    # The same loader-policy refusal the planner and runner apply, repeated for a direct caller.
    from corpus_studio.platform.execution_config import (  # noqa: PLC0415
        ExecutionConfigurationError,
        verify_loader_policy_supported,
    )

    try:
        verify_loader_policy_supported(execution, lane="full_finetune")
    except ExecutionConfigurationError as exc:
        raise FullFinetuneError(str(exc)) from exc

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
    from corpus_studio.training.sealed_loader import (
        load_sealed_model,
        load_sealed_tokenizer,
        no_stage,
        observe_sealed_placement,
        prepare_sealed_attention,
        sealed_loader_view,
        verify_full_parameter_storage,
    )
    from corpus_studio.training.trainer import (
        ExecutionPlacementDeviation,
        capture_adapter_export_state,
        capture_trainable_state,
        enforced_attention_training_kernel,
        format_example_text,
    )

    if execution.precision.quantized_storage_format.value != "none":
        raise FullFinetuneError(
            "full-parameter fine-tuning must be unquantized; the sealed config is quantized"
        )
    _stage = stage_callback or no_stage
    view = sealed_loader_view(execution)
    # The CPU smoke path is a property of the seal (runtime_mode), not of whoever dispatched the worker.
    cpu_toy = view.cpu_toy
    set_seed(execution.seed)
    out = Path(output_dir or execution.output_dir)

    # --- tokenizer + model: the sealed tokenizer binding, then a real base in the sealed storage dtype,
    # ALL parameters trainable (no adapter, no nf4). The shared SFT helpers lower the sealed revision,
    # safetensors-only policy, storage dtype, root device and attention API (trust_remote_code stays the
    # sealed False), after the SDPA toggles are applied and the sealed kernel is probed. ---
    tokenizer = load_sealed_tokenizer(AutoTokenizer, view, stage=_stage)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        # A base whose tokenizer has neither pad nor eos cannot pad batches - fail closed with a clear
        # reason rather than a cryptic 'NoneType' int error deep in the fixed-length row builder. This
        # guards a malformed base (e.g. a from-scratch tokenizer that declared no eos token).
        raise FullFinetuneError(
            "the base model's tokenizer defines no pad or eos token, so training batches cannot be "
            "padded; the base is unusable (a from-scratch tokenizer must declare an eos token)"
        )
    prepare_sealed_attention(torch, view, stage=_stage)
    model = load_sealed_model(AutoModelForCausalLM, torch, view, stage=_stage)
    verify_full_parameter_storage(model, torch, view, stage=_stage)

    # --- data: the sealed SFT rows (verified above), formatted + tokenized to fixed length (whole-sequence
    # loss, per above) ---
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
    # Named apart from the ``dataset`` parameter (the verified sealed rows) so the two never blur.
    train_dataset = Dataset.from_list(built)

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
        model=model, args=arguments, train_dataset=train_dataset,
        data_collator=_collate, callbacks=[_EvidenceCallback()],
    )
    # The HF Trainer places the model itself: with more than one visible GPU it would replicate it with
    # DataParallel, which is not the sealed single-root placement. Refuse that, then observe the placement
    # the Trainer actually left before the first step.
    if not cpu_toy and trainer.args.n_gpu != 1:
        raise ExecutionPlacementDeviation(
            f"PLACEMENT_DEVIATION: the HF Trainer sees {trainer.args.n_gpu} GPUs and would not keep the "
            f"sealed single-device placement {view.device_map}"
        )
    observe_sealed_placement(trainer.model, view, stage=_stage, label="Trainer-placed model")
    with enforced_attention_training_kernel(torch, view):
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
