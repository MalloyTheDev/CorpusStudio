"""First-party SFT/QLoRA trainer (the opt-in ``[train]`` extra).

Runs training **in-process** so CorpusStudio can go config → *train* → adapter without the user
bringing an external trainer. Every heavy import (torch / transformers / peft / trl / datasets /
bitsandbytes) is **lazy** — done inside :func:`run_training`, never at module load — so importing this
module in the dependency-light engine costs nothing and never fails when the extra is absent.

Two paths share one code route:

* **GPU 4-bit QLoRA** — the real run: NF4 bitsandbytes quantization + LoRA on the config's base model.
* **CPU toy** (``cpu_toy=True``) — a tiny model, no quantization, a few steps, so the *plumbing* is
  provable without a GPU (this is what makes the trainer verifiable in CI-less environments).

The pure helpers (config load, row→text formatting, the LoRA/SFT arg mapping, run-plan resolution)
carry no heavy imports and are unit-tested with the deps mocked; :func:`run_training` itself is
verified via the CPU toy path.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from corpus_studio.importers.jsonl_importer import read_jsonl
from corpus_studio.training.environment import INSTALL_HINT, probe_training_runtime

# Tiny Llama-architecture model for the CPU toy path. The CPU toy builds the model FROM CONFIG
# (random weights — no weights download, so it works offline/behind a firewall), and only the small
# config + tokenizer are fetched (once). nn.Linear layers → LoRA "all-linear" works, matching the real
# Qwen family. It is a smoke test of the pipeline, not a model meant to learn the task.
TINY_TOY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

ProgressCallback = Callable[[int, int, float | None], None]


class TrainerError(Exception):
    """Raised for an unrunnable request (runtime missing, bad config) — a clean CLI exit, not a crash."""


class TrainRunConfig(BaseModel):
    """Resolved inputs for one training run (from a CorpusStudio training config + overrides)."""

    base_model: str
    dataset_path: str
    output_dir: str = "output"
    dataset_format: str = "instruction"
    sequence_len: int = Field(default=4096, gt=0)
    lora_r: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    micro_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    seed: int = Field(default=42, ge=0)
    cpu_toy: bool = False
    max_steps: int | None = None


class TrainResult(BaseModel):
    output_dir: str
    adapter_path: str
    base_model: str
    cpu_toy: bool
    steps: int = 0
    final_loss: float | None = None
    checkpoints: list[str] = Field(default_factory=list)


def _parse_config_text(text: str) -> dict[str, Any]:
    """Parse a training config that may be JSON **or** YAML. The ``corpus_studio`` target renders JSON,
    but a user can point ``train-run`` at a YAML config (e.g. one named ``*.yaml``, or an axolotl-style
    file) — so a valid config never dies on format. Tries JSON first (fast, no dep); on failure falls
    back to YAML (PyYAML ships with the ``[train]`` deps that ``train-run`` needs anyway)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # noqa: PLC0415 - lazy; PyYAML is a transitive [train] dep (transformers/datasets).
        except ImportError as exc:
            raise TrainerError(
                "The training config is not valid JSON, and PyYAML isn't available to parse it as YAML. "
                "Use a JSON config (the corpus_studio target emits one) or install pyyaml."
            ) from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise TrainerError(f"The training config is neither valid JSON nor valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise TrainerError("The training config must be a JSON object / YAML mapping of fields.")
    return data


def load_run_config_from_file(
    config_path: Path | str,
    *,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    base_model: str | None = None,
    cpu_toy: bool = False,
    max_steps: int | None = None,
) -> TrainRunConfig:
    """Build a :class:`TrainRunConfig` from a CorpusStudio training config (the ``training-config``
    output: base_model / dataset_path / format / sequence_len / lora_* / seed …), applying overrides.
    Accepts JSON or YAML so a config never dies on format.

    The CPU toy path forces a tiny model + a short sequence + a few steps so it runs in seconds,
    unless the caller overrides them explicitly."""
    data: dict[str, Any] = _parse_config_text(Path(config_path).read_text(encoding="utf-8-sig"))

    resolved_base = base_model or data.get("base_model") or ""
    resolved_dataset = dataset_path or data.get("dataset_path") or ""
    seq = int(data.get("sequence_len", 4096))
    steps = max_steps

    if cpu_toy:
        resolved_base = base_model or TINY_TOY_MODEL
        seq = min(seq, 128)
        steps = max_steps if max_steps is not None else 3

    if not resolved_base:
        raise TrainerError("No base model: pass --base-model or a config with 'base_model'.")
    if not resolved_dataset:
        raise TrainerError("No dataset: pass --dataset-path or a config with 'dataset_path'.")

    return TrainRunConfig(
        base_model=resolved_base,
        dataset_path=resolved_dataset,
        output_dir=output_dir or data.get("output_dir", "output"),
        dataset_format=str(data.get("format", "instruction")),
        sequence_len=seq,
        lora_r=int(data.get("lora_r", 16)),
        lora_alpha=int(data.get("lora_alpha", 32)),
        micro_batch_size=int(data.get("micro_batch_size", 1)),
        gradient_accumulation_steps=int(data.get("gradient_accumulation_steps", 8)),
        learning_rate=float(data.get("learning_rate", 2e-4)),
        seed=int(data.get("seed", 42)),
        cpu_toy=cpu_toy,
        max_steps=steps,
    )


def format_example_text(row: dict, dataset_format: str, tokenizer: Any | None = None) -> str:
    """Format one dataset row into a single training-text string.

    ``chat`` uses the model's chat template when a tokenizer is supplied (the correct format for a
    chat model), else a simple ``role: content`` join. ``instruction`` uses an Alpaca-style template.
    Returns "" for an empty/unusable row (the caller drops it)."""
    if dataset_format == "chat":
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            return ""
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return str(tokenizer.apply_chat_template(messages, tokenize=False))
            except Exception:  # noqa: BLE001 - a template failure falls back to the plain join.
                pass
        return "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}" for m in messages if isinstance(m, dict)
        )

    instruction = str(row.get("instruction", "")).strip()
    extra_input = str(row.get("input", "")).strip()
    output = str(row.get("output", "")).strip()
    if not instruction and not output:
        return ""
    prompt = instruction + (f"\n\n{extra_input}" if extra_input else "")
    return f"### Instruction:\n{prompt}\n\n### Response:\n{output}"


def build_lora_kwargs(config: TrainRunConfig) -> dict[str, Any]:
    """peft ``LoraConfig`` kwargs. ``target_modules='all-linear'`` targets every linear layer, so it
    works across architectures (tiny Llama toy → real Qwen) without a per-model module list."""
    return {
        "r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": "all-linear",
    }


def build_training_kwargs(config: TrainRunConfig) -> dict[str, Any]:
    """TRL ``SFTConfig`` kwargs from the run config. No W&B (``report_to=[]``); logs every step so the
    progress callback fires; saves by steps so checkpoints appear for the launcher."""
    kwargs: dict[str, Any] = {
        "output_dir": config.output_dir,
        "per_device_train_batch_size": config.micro_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": 50,
        "report_to": [],
        "dataset_text_field": "text",
        "max_seq_length": config.sequence_len,
        "disable_tqdm": True,  # we emit our own [step/total] progress
    }
    if config.max_steps is not None:
        kwargs["max_steps"] = config.max_steps
    else:
        kwargs["num_train_epochs"] = 1
    if config.cpu_toy:
        # Force CPU + no half precision so the toy runs on a machine with no GPU.
        kwargs["use_cpu"] = True
        kwargs["bf16"] = False
        kwargs["fp16"] = False
    return kwargs


def resolve_run_plan(config: TrainRunConfig, report: Any) -> dict[str, Any]:
    """Decide device + quantization from the runtime report, or raise a clean :class:`TrainerError`.

    CPU toy → CPU, no quantization (needs only ``cpu_toy_ready``). Real run → CUDA + 4-bit
    (needs the full ``ready`` set: deps + GPU + bitsandbytes)."""
    if config.cpu_toy:
        if not report.cpu_toy_ready:
            raise TrainerError(
                "CPU toy training needs torch + transformers + peft + trl + datasets + accelerate. "
                f"{INSTALL_HINT}"
            )
        return {"device": "cpu", "quantize": False}

    if not report.ready:
        raise TrainerError(
            "A real QLoRA run needs the full [train] runtime + a CUDA GPU + bitsandbytes. "
            f"Run 'corpus-studio train-check' to see what's missing. {INSTALL_HINT} "
            "(or pass --cpu-toy to smoke-test the pipeline on CPU)."
        )
    return {"device": "cuda", "quantize": True}


def _list_checkpoints(output_dir: Path) -> list[str]:
    if not output_dir.is_dir():
        return []
    return sorted(str(p) for p in output_dir.glob("checkpoint-*") if p.is_dir())


def run_training(config: TrainRunConfig, *, progress_callback: ProgressCallback | None = None) -> TrainResult:
    """Run the training. Lazy-imports the heavy stack; verified via the CPU toy path (a real GPU QLoRA
    can only be user-smoke-tested). Raises :class:`TrainerError` if the runtime can't run the request."""
    plan = resolve_run_plan(config, probe_training_runtime())
    quantize: bool = plan["quantize"]

    import torch  # noqa: PLC0415 - intentionally lazy heavy imports.
    from datasets import Dataset  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainerCallback,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    set_seed(config.seed)

    # SECURITY: never execute a downloaded model repo's custom code. trust_remote_code defaults False,
    # but we set it explicitly (defence-in-depth; guards against a future default change). A model that
    # genuinely needs custom code is a deliberate, separate decision — not something a fetched repo can
    # trigger silently.
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.cpu_toy:
        # Build from config (random weights) — no weights download, so the smoke test runs offline.
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(config.base_model, trust_remote_code=False)
        )
    else:
        model_kwargs: dict[str, Any] = {"trust_remote_code": False}
        if quantize:
            from transformers import BitsAndBytesConfig  # noqa: PLC0415

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
        else:
            # torch_dtype (not dtype) for compat across transformers 4.44+ and 5.x (the [train] floor).
            model_kwargs["torch_dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    if quantize:
        from peft import prepare_model_for_kbit_training  # noqa: PLC0415

        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(**build_lora_kwargs(config)))

    rows = list(read_jsonl(Path(config.dataset_path)))
    texts = [format_example_text(row, config.dataset_format, tokenizer) for row in rows]
    dataset = Dataset.from_list([{"text": text} for text in texts if text])
    if len(dataset) == 0:
        raise TrainerError("The dataset produced no usable training rows.")

    # Adapt to the installed TRL's SFTConfig: `max_seq_length` was renamed to `max_length`, so map it
    # to whichever the class actually has and drop any field it doesn't accept (robust across versions).
    import dataclasses  # noqa: PLC0415

    raw_kwargs = build_training_kwargs(config)
    valid_fields = {f.name for f in dataclasses.fields(SFTConfig)}
    if "max_seq_length" not in valid_fields and "max_seq_length" in raw_kwargs:
        seq_len = raw_kwargs.pop("max_seq_length")
        if "max_length" in valid_fields:
            raw_kwargs["max_length"] = seq_len
    sft_config = SFTConfig(**{k: v for k, v in raw_kwargs.items() if k in valid_fields})

    class _ProgressCallback(TrainerCallback):  # type: ignore[misc]
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if progress_callback is None:
                return
            loss = None
            if state.log_history:
                loss = state.log_history[-1].get("loss")
            progress_callback(int(state.global_step), int(state.max_steps or 0), loss)

    # The tokenizer arg was renamed `tokenizer` -> `processing_class`; pass it under whichever exists.
    import inspect  # noqa: PLC0415

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": dataset,
        "callbacks": [_ProgressCallback()],
    }
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    # Training frameworks print metrics/tqdm to STDOUT — and transformers can log to stdout during
    # SAVE too — so redirect the whole train+save block to stderr. Only the CLI's final JSON echo then
    # writes to stdout, keeping it the pure JSON result the desktop/WBG parses. Progress reaches stderr.
    with contextlib.redirect_stdout(sys.stderr):
        train_output = trainer.train()

        output_dir = Path(config.output_dir)
        trainer.save_model(str(output_dir))  # saves the LoRA adapter
        tokenizer.save_pretrained(str(output_dir))

        metrics = getattr(train_output, "metrics", {}) or {}
        steps = int(getattr(trainer.state, "global_step", 0) or 0)

        # Self-documenting run: write a model card next to the adapter (recipe + honesty + base-model
        # license reminder). Best-effort — the card is a convenience and must NEVER fail a completed run.
        try:
            from corpus_studio.training.model_card import build_model_card, write_model_card  # noqa: PLC0415

            write_model_card(
                output_dir,
                build_model_card(
                    output_dir,
                    base_model=config.base_model,
                    training_config={
                        "format": config.dataset_format,
                        "sequence_len": config.sequence_len,
                        "learning_rate": config.learning_rate,
                        "seed": config.seed,
                    },
                    train_result={"steps": steps, "final_loss": metrics.get("train_loss"), "cpu_toy": config.cpu_toy},
                ),
            )
        except Exception:  # noqa: BLE001 - never let card-writing break a finished training run.
            pass

    return TrainResult(
        output_dir=str(output_dir),
        adapter_path=str(output_dir),
        base_model=config.base_model,
        cpu_toy=config.cpu_toy,
        steps=steps,
        final_loss=metrics.get("train_loss"),
        checkpoints=_list_checkpoints(output_dir),
    )
