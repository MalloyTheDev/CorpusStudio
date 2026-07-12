"""The Unsloth training backend — accelerated 4-bit QLoRA via ``FastLanguageModel`` + a TRL
``SFTTrainer``. A first-class training backend selectable via ``--backend unsloth`` (see
``platform.backends``), NOT an external script.

Honesty: the actual training can only be USER-SMOKE-TESTED — Unsloth needs its own package + a CUDA
GPU (and its fused kernels are flash/sdpa, so the platform routes Blackwell/sm_120 plans to the
first-party math-path trainer instead). So the heavy body here is not runnable in CI; the lazy-import
guard and the config→Unsloth-args mapping ARE unit-tested. Mirrors ``trainer.run_training``'s
signature so the TrainingRunner drives either backend identically.
"""

from __future__ import annotations

from typing import Any

from corpus_studio.training.trainer import (
    ProgressCallback,
    StageCallback,
    TrainerError,
    TrainResult,
    TrainRunConfig,
)

# The Unsloth/PEFT LoRA target modules for a Llama/Qwen-family attention+MLP (Unsloth's default set).
_DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def build_unsloth_kwargs(config: TrainRunConfig) -> dict[str, Any]:
    """The ``FastLanguageModel.from_pretrained`` + ``get_peft_model`` kwargs for a resolved config.
    Pure + unit-tested (no Unsloth import), so the config mapping is verifiable without a GPU."""
    return {
        "from_pretrained": {
            "model_name": config.base_model,
            "max_seq_length": config.sequence_len,
            "load_in_4bit": True,  # Unsloth's headline: 4-bit QLoRA
            "dtype": None,  # auto (bf16 on Ampere+, else fp16)
        },
        "get_peft_model": {
            "r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "target_modules": list(_DEFAULT_TARGET_MODULES),
            "lora_dropout": 0.0,
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "random_state": config.seed,
        },
    }


def run_unsloth_training(
    config: TrainRunConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    stage_callback: StageCallback | None = None,
) -> TrainResult:
    """Train via Unsloth (accelerated 4-bit QLoRA). Lazy-imports Unsloth; raises
    :class:`TrainerError` when it (or a CUDA GPU) is unavailable — the clean "can't run this here"
    signal the platform classifies as ENVIRONMENT_FAILURE. The training itself is user-smoke-tested.
    ``stage_callback`` fires setup milestones during the silent load (see ``trainer.run_training``)."""
    try:
        from unsloth import FastLanguageModel  # noqa: PLC0415 - lazy; the whole point of the backend
    except ImportError as exc:
        raise TrainerError(
            "Unsloth isn't installed (it needs a CUDA GPU): pip install unsloth. On Blackwell "
            "(sm_120) use the corpus_studio backend — Unsloth's fused kernels don't support it yet."
        ) from exc
    return _train_with_unsloth(  # pragma: no cover
        config, FastLanguageModel, progress_callback, stage_callback
    )


def _train_with_unsloth(  # pragma: no cover - needs a CUDA GPU + unsloth; user-smoke-tested only
    config: TrainRunConfig,
    fast_language_model: Any,
    progress_callback: ProgressCallback | None,
    stage_callback: StageCallback | None = None,
) -> TrainResult:
    import contextlib
    import dataclasses
    import inspect
    import sys
    from pathlib import Path

    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    from corpus_studio.importers.jsonl_importer import read_jsonl
    from corpus_studio.training.trainer import (
        build_training_kwargs,
        format_example_text,
    )

    def _stage(name: str, message: str) -> None:
        if stage_callback is not None:
            stage_callback(name, message)

    # Unsloth builds + 4-bit-quantizes + LoRA-wraps the model in two calls.
    kwargs = build_unsloth_kwargs(config)
    model, tokenizer = fast_language_model.from_pretrained(**kwargs["from_pretrained"])
    _stage("model_loaded", f"loaded {config.base_model} (Unsloth 4-bit)")
    model = fast_language_model.get_peft_model(model, **kwargs["get_peft_model"])
    _stage("adapter_attached", "LoRA adapter attached")

    rows = list(read_jsonl(Path(config.dataset_path)))
    texts = [format_example_text(row, config.dataset_format, tokenizer) for row in rows]
    dataset = Dataset.from_list([{"text": text} for text in texts if text])
    if len(dataset) == 0:
        raise TrainerError("The dataset produced no usable training rows.")

    # Same TRL-version robustness as the first-party trainer: SFTConfig renamed max_seq_length→max_length
    # and SFTTrainer renamed tokenizer→processing_class across versions; adapt to whichever the installed
    # TRL exposes rather than pinning a single one.
    raw_kwargs = build_training_kwargs(config)
    valid_fields = {f.name for f in dataclasses.fields(SFTConfig)}
    if "max_seq_length" not in valid_fields and "max_seq_length" in raw_kwargs:
        seq_len = raw_kwargs.pop("max_seq_length")
        if "max_length" in valid_fields:
            raw_kwargs["max_length"] = seq_len
    sft_config = SFTConfig(**{k: v for k, v in raw_kwargs.items() if k in valid_fields})

    class _Progress(TrainerCallback):  # type: ignore[misc]
        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> None:
            if progress_callback is None:
                return
            loss = state.log_history[-1].get("loss") if state.log_history else None
            progress_callback(int(state.global_step), int(state.max_steps or 0), loss)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": dataset,
        "callbacks": [_Progress()],
    }
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    output_dir = config.output_dir
    with contextlib.redirect_stdout(sys.stderr):
        trainer = SFTTrainer(**trainer_kwargs)
        _stage("optimizer_created", "Unsloth SFT trainer ready — starting training")
        result = trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    return TrainResult(
        output_dir=output_dir,
        adapter_path=output_dir,
        base_model=config.base_model,
        cpu_toy=False,
        steps=int(getattr(result, "global_step", 0)),
        final_loss=getattr(result, "training_loss", None),
        checkpoints=[str(p) for p in sorted(Path(output_dir).glob("checkpoint-*"))],
    )
