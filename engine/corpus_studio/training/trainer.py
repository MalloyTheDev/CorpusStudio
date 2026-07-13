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
# (stage_name, message) — platform-agnostic strings so the trainer stays decoupled from the platform
# enums. Fires at setup milestones so a supervisor sees progress during the long silent model-load.
StageCallback = Callable[[str, str], None]


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
    # Attention backend passed to from_pretrained (e.g. "eager" / "sdpa" / "flash_attention_2").
    # None = auto: honor transformers' default, but on Blackwell (sm_120) the fused flash/mem-efficient
    # SDPA kernels deadlock on the first backward, so the trainer forces the math SDPA path there.
    attn_implementation: str | None = None
    # --- memory / spill-avoidance levers (opt-in) ---
    # The optimizer TRL/transformers uses. "paged_adamw_8bit" (bitsandbytes) pages optimizer state to
    # host RAM under pressure via CUDA managed memory — a spike-safe SAFE-SPILL for the optimizer
    # (verified on a real 5070 under WSL2). For QLoRA the optimizer is small (LoRA params only), so this
    # is more crash-safety than a big saving; the base 4-bit model dominates VRAM.
    optim: str = "adamw_torch"
    # Fuse the cross-entropy loss (Liger) so the full-vocab logits are never materialized — removes the
    # ~2.5 GB fp32 logits spike at long sequence_len + large vocab (the real long-seq memory lever).
    # Requires the `liger-kernel` package; ignored gracefully by the SFTConfig-field filter if the
    # installed TRL/transformers predates `use_liger_kernel`. NOTE: Liger uses Triton kernels — its
    # support on Blackwell/sm_120 is not yet verified here.
    use_liger: bool = False
    # --- checkpoint retention (disk-control) ---
    # How often to checkpoint (was hardcoded to 50). Each checkpoint is the adapter + resume state
    # (optimizer/RNG) — hundreds of MB for QLoRA, not a full base model — but with no cap they
    # accumulate indefinitely over a long run.
    save_steps: int = Field(default=50, gt=0)
    # Keep only the N most recent checkpoints (passed to TRL's SFTConfig). Default 3 keeps resume
    # capability while preventing unbounded checkpoint growth; None keeps every checkpoint.
    save_total_limit: int | None = Field(default=3, ge=1)


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
    attn_implementation: str | None = None,
    optim: str | None = None,
    use_liger: bool | None = None,
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
        attn_implementation=attn_implementation or data.get("attn_implementation"),
        optim=optim or str(data.get("optim", "adamw_torch")),
        use_liger=use_liger if use_liger is not None else bool(data.get("use_liger", False)),
        save_steps=int(data.get("save_steps", 50)),
        save_total_limit=data.get("save_total_limit", 3),  # int or null (keep all)
    )


def format_example_text(row: dict, dataset_format: str, tokenizer: Any | None = None) -> str:
    """Format one dataset row into a single training-text string.

    ``chat`` uses the model's chat template when a tokenizer is supplied (the correct format for a
    chat model), else a simple ``role: content`` join. ``instruction`` uses an Alpaca-style template.
    ``trace`` renders a reasoning trace (prompt + ``<think>reasoning</think>`` + answer — see
    ``training.traces``). Returns "" for an empty/unusable row (the caller drops it)."""
    if dataset_format == "trace":
        from corpus_studio.training.traces import format_trace, trace_from_row  # noqa: PLC0415

        return format_trace(trace_from_row(row), tokenizer=tokenizer)
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
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,  # cap checkpoint accumulation (None = keep all)
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
        # Force CPU + no half precision so the toy runs on a machine with no GPU. The paged optimizer
        # (bitsandbytes) and Liger (Triton) are CUDA-only, so the toy path never uses them — it would
        # crash on a GPU-less machine, defeating the point of the smoke test.
        kwargs["use_cpu"] = True
        kwargs["bf16"] = False
        kwargs["fp16"] = False
        kwargs["optim"] = "adamw_torch"
    else:
        kwargs["optim"] = config.optim
        if config.use_liger:
            # Filtered out by valid_fields below if the installed TRL/transformers predates it — so a
            # request for Liger on an older stack degrades to the normal (unfused) loss, never a crash.
            kwargs["use_liger_kernel"] = True
    return kwargs


class TruncationReport(BaseModel):
    """How badly a ``sequence_len`` truncates a dataset — the guardrail against silently training on
    cut-off examples (a real WBG bug: seq_len 1536 truncated 100% of ~2.2k-token examples, cutting the
    end of every assistant/output and teaching the model to emit incomplete JSON)."""

    n_examples: int
    n_truncated: int
    pct_truncated: float
    seq_len: int
    max_tokens: int
    median_tokens: int
    # the smallest sequence_len that would truncate NOTHING (== the longest example)
    seq_len_for_zero_truncation: int

    @property
    def truncates(self) -> bool:
        return self.n_truncated > 0


def analyze_truncation(token_lengths: list[int], seq_len: int) -> TruncationReport:
    """PURE + unit-tested. Given tokenized example lengths + the training ``seq_len``, report how many
    are truncated (their tails — including the model's output — cut off) and the seq_len that keeps
    them whole."""
    lengths = sorted(token_lengths)
    n = len(lengths)
    n_trunc = sum(1 for x in lengths if x > seq_len)
    return TruncationReport(
        n_examples=n,
        n_truncated=n_trunc,
        pct_truncated=(100.0 * n_trunc / n) if n else 0.0,
        seq_len=seq_len,
        max_tokens=lengths[-1] if lengths else 0,
        median_tokens=lengths[n // 2] if lengths else 0,
        seq_len_for_zero_truncation=lengths[-1] if lengths else seq_len,
    )


def truncation_warning(report: TruncationReport) -> str | None:
    """The operator-facing warning for a truncating ``seq_len`` — or None when nothing is cut."""
    if not report.truncates:
        return None
    return (
        f"[WARNING] TRUNCATION: {report.n_truncated}/{report.n_examples} examples "
        f"({report.pct_truncated:.0f}%) exceed sequence_len={report.seq_len} and will be CUT — the end "
        f"of each (including the assistant/output) is lost, so the model learns incomplete outputs. "
        f"Longest example is {report.max_tokens} tokens; raise sequence_len to "
        f">= {report.seq_len_for_zero_truncation} to keep every example whole (costs more VRAM), or "
        "shorten/split the data."
    )


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


# Blackwell (RTX 50-series) is sm_120 → compute-capability major 12. The fused FLASH SDPA backward
# deadlocks there ONLY under the native-Windows WDDM driver model — the canonical rule lives in
# corpus_studio.platform.host_platform.flash_sdpa_deadlocks; kept as a plain check here so the trainer
# stays decoupled from the platform layer (which sits above it).
_BLACKWELL_CAPABILITY_MAJOR = 12


def resolve_attention_implementation(
    explicit: str | None,
    device_capability_major: int | None,
    *,
    native_windows: bool = False,
) -> tuple[str | None, bool]:
    """Decide the attention backend for ``from_pretrained``. PURE + unit-tested.

    Returns ``(attn_implementation, disable_flash_sdp)``:
    * an **explicit** choice (config / CLI ``--attn-implementation``) always wins, with no SDP toggling
      — the user owns it;
    * else on **native Windows (WDDM) + Blackwell** (sm_120) → ``(None, True)``: keep transformers'
      default SDPA but signal the caller to DISABLE the fused flash SDP backend — the only kernel that
      deadlocks there (mem-efficient + math are safe), so SDPA uses a non-deadlocking kernel;
    * else ``(None, False)`` — no change. Crucially this includes **WSL and bare Linux on Blackwell**:
      the deadlock is a Windows WDDM property, NOT an sm_120 kernel bug (verified on a real 5070 under
      WSL2 that fused flash + mem-efficient SDPA both run), so flash stays enabled there — the whole
      point of running training under WSL. ``native_windows`` (``sys.platform == "win32"``, which is
      False under WSL) is what distinguishes the two; unknown/False hosts leave the kernel enabled.
    """
    if explicit:
        return explicit, False
    blackwell = device_capability_major is not None and device_capability_major >= _BLACKWELL_CAPABILITY_MAJOR
    if native_windows and blackwell:
        return None, True
    return None, False


def _list_checkpoints(output_dir: Path) -> list[str]:
    if not output_dir.is_dir():
        return []
    return sorted(str(p) for p in output_dir.glob("checkpoint-*") if p.is_dir())


def run_training(
    config: TrainRunConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    stage_callback: StageCallback | None = None,
) -> TrainResult:
    """Run the training. Lazy-imports the heavy stack; verified via the CPU toy path (a real GPU QLoRA
    can only be user-smoke-tested). Raises :class:`TrainerError` if the runtime can't run the request.
    ``stage_callback(name, message)`` fires at setup milestones (model_loaded / quantized /
    adapter_attached / optimizer_created) so the long SILENT model-load window emits real progress a
    supervisor can see — the honest alternative to a liveness heartbeat."""
    plan = resolve_run_plan(config, probe_training_runtime())
    quantize: bool = plan["quantize"]

    def _stage(name: str, message: str) -> None:
        if stage_callback is not None:
            stage_callback(name, message)

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

        # Attention backend. On **native Windows (WDDM) + Blackwell (sm_120)** the fused FLASH SDPA
        # kernel DEADLOCKS on the first backward — verified on a real 5070: bnb 4-bit, the mem-efficient
        # kernel, and the math path are all fine; only flash hangs, and only under WDDM. So disable
        # ONLY flash, and ONLY there. Under **WSL / bare Linux** the SAME sm_120 flash kernel runs fine
        # (verified on the 5070 under WSL2 — O(seq) memory, ~1000x faster than the math fallback), so it
        # is left ENABLED — that is the reason to train under WSL. `sys.platform == "win32"` is True only
        # on native Windows (WSL Python reports "linux"). An explicit --attn-implementation always wins.
        try:
            capability_major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else None
        except Exception:  # noqa: BLE001 - a probe failure just means "don't special-case".
            capability_major = None
        native_windows = sys.platform == "win32"
        attn_impl, disable_flash_sdp = resolve_attention_implementation(
            config.attn_implementation, capability_major, native_windows=native_windows
        )
        if disable_flash_sdp:
            try:
                torch.backends.cuda.enable_flash_sdp(False)  # the only SDPA kernel that hangs on WDDM+sm_120
                print(
                    "[note] Native Windows + Blackwell GPU (sm_120) detected — disabled the fused FLASH "
                    "SDPA kernel (it deadlocks on the first backward under the Windows WDDM driver). SDPA "
                    "uses mem-efficient/math; the model's masked attention runs on the math kernel (more "
                    "VRAM), so lower sequence_len if it spills to system RAM. Running under WSL keeps "
                    "flash enabled (O(seq) memory). --attn-implementation overrides.",
                    file=sys.stderr,
                )
            except Exception:  # noqa: BLE001 - if the backend toggle is unavailable, fall back to eager.
                attn_impl = "eager"
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

        model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    _stage("model_loaded", f"loaded {config.base_model}")  # the end of the long silent load window
    if quantize:
        from peft import prepare_model_for_kbit_training  # noqa: PLC0415

        model = prepare_model_for_kbit_training(model)
        _stage("quantized", "prepared for 4-bit k-bit training")
    model = get_peft_model(model, LoraConfig(**build_lora_kwargs(config)))
    _stage("adapter_attached", "LoRA adapter attached")

    rows = list(read_jsonl(Path(config.dataset_path)))
    texts = [format_example_text(row, config.dataset_format, tokenizer) for row in rows]
    dataset = Dataset.from_list([{"text": text} for text in texts if text])
    if len(dataset) == 0:
        raise TrainerError("The dataset produced no usable training rows.")

    # TRUNCATION GUARDRAIL: tokenize a sample and WARN if sequence_len would cut examples. Silent
    # truncation trains the model on incomplete outputs (the WBG bug: seq_len 1536 truncated 100% of
    # ~2.2k-token examples → the model learned to emit unterminated JSON). A warning only — never fails
    # the run — and it's a cheap sample so it doesn't slow a large dataset.
    try:
        sample = [t for t in texts if t][:256]
        lengths = [len(tokenizer(t)["input_ids"]) for t in sample]
        warning = truncation_warning(analyze_truncation(lengths, config.sequence_len))
        if warning:
            print(warning, file=sys.stderr)
    except Exception:  # noqa: BLE001 - a guardrail must never break a run it was meant to protect.
        pass

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
    _stage("optimizer_created", "SFT trainer + optimizer ready — starting training")
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
