"""Merge a trained LoRA adapter into its base model — with a fallback for small-VRAM cards.

After ``train-run`` produces a LoRA adapter, you often want a single standalone model. Merging a 7B
in fp16 needs ~14 GB, which won't fit a 12 GB card, so this tries a chain and falls back gracefully:

* **gpu** — load the base in fp16 on the GPU, merge, save. Fast; needs the VRAM.
* **cpu** — load + merge on CPU/RAM. Slower, no VRAM limit — the answer when the GPU merge OOMs.
* **adapter-only** — don't merge at all; keep the adapter separate and serve base+adapter (peft) at
  inference. The always-available fallback (smallest footprint, no merge cost).

``strategy="auto"`` walks gpu → cpu → adapter-only, catching each failure (OOM etc.) and trying the
next. All heavy imports (torch/transformers/peft) are lazy — inside :func:`merge_adapter`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from corpus_studio.training.environment import probe_training_runtime

MergeProgress = Callable[[str], None]
VALID_STRATEGIES = ("auto", "gpu", "cpu", "adapter-only")


class MergeError(Exception):
    """Raised when no merge strategy could produce a model (a clean CLI exit, not a crash)."""


class MergeResult(BaseModel):
    strategy: str  # gpu | cpu | adapter-only — the one that actually ran
    merged: bool
    output_path: str  # the merged-model dir, or the adapter dir when adapter-only
    base_model: str
    notes: list[str] = Field(default_factory=list)


def resolve_merge_plan(strategy: str, gpu_available: bool) -> list[str]:
    """The ordered strategies to try. ``auto`` = gpu (if present) → cpu → adapter-only; an explicit
    strategy is tried alone. adapter-only is always the terminal fallback for ``auto``."""
    if strategy not in VALID_STRATEGIES:
        raise MergeError(f"Unknown merge strategy '{strategy}'; expected one of {VALID_STRATEGIES}.")
    if strategy != "auto":
        return [strategy]
    plan = ["gpu", "cpu", "adapter-only"] if gpu_available else ["cpu", "adapter-only"]
    return plan


def base_model_from_adapter(adapter_path: Path | str, override: str | None = None) -> str:
    """The base model to merge into: an explicit override, else ``base_model_name_or_path`` from the
    adapter's ``adapter_config.json``."""
    if override:
        return override
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        raise MergeError(f"No adapter_config.json in {adapter_path}; pass --base-model explicitly.")
    try:
        base = json.loads(config_path.read_text(encoding="utf-8")).get("base_model_name_or_path")
    except (json.JSONDecodeError, OSError) as exc:
        raise MergeError(f"Could not read {config_path}: {exc}") from exc
    if not base:
        raise MergeError("adapter_config.json has no base_model_name_or_path; pass --base-model.")
    return str(base)


def serving_instructions(base_model: str, adapter_path: Path | str) -> str:
    """How to serve base + adapter without merging (the adapter-only path)."""
    return (
        "Adapter-only (not merged): serve the base model with the LoRA adapter applied at load time, "
        f"e.g. peft.PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained('{base_model}'), "
        f"'{adapter_path}'). vLLM/TGI also accept the adapter directly."
    )


def merge_adapter(
    adapter_path: Path | str,
    *,
    base_model: str | None = None,
    output_dir: Path | str | None = None,
    strategy: str = "auto",
    progress: MergeProgress | None = None,
) -> MergeResult:
    """Merge the adapter into its base, walking the resolved plan and falling back on failure. Verified
    on GPU + CPU for a small model; the OOM→CPU fallback is exercised by the plan/try structure."""
    adapter_path = Path(adapter_path)
    base = base_model_from_adapter(adapter_path, base_model)
    out = Path(output_dir) if output_dir else adapter_path.parent / "merged"
    plan = resolve_merge_plan(strategy, probe_training_runtime().gpu.available)

    notes: list[str] = []
    for attempt in plan:
        if progress is not None:
            progress(f"merge attempt: {attempt}")
        if attempt == "adapter-only":
            notes.append(serving_instructions(base, adapter_path))
            return MergeResult(
                strategy="adapter-only",
                merged=False,
                output_path=str(adapter_path),
                base_model=base,
                notes=notes,
            )
        try:
            _merge_on_device(base, adapter_path, out, device=("cuda" if attempt == "gpu" else "cpu"))
            return MergeResult(
                strategy=attempt, merged=True, output_path=str(out), base_model=base, notes=notes
            )
        except Exception as exc:  # noqa: BLE001 - any failure (incl. CUDA OOM) → try the next strategy.
            notes.append(f"{attempt} merge failed ({type(exc).__name__}: {exc}); trying the next option.")
            continue

    raise MergeError("All merge strategies failed. " + " | ".join(notes))


def _merge_on_device(base_model: str, adapter_path: Path, output_dir: Path, device: str) -> None:
    """Load the fp16 base on ``device``, apply + merge the adapter, save the merged model + tokenizer.
    Lazy heavy imports; raises on OOM (caught by the fallback in :func:`merge_adapter`)."""
    import torch  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    # SECURITY: trust_remote_code=False so merging an adapter never executes the base repo's code.
    # torch_dtype (not dtype) for compat across transformers 4.44+ and 5.x.
    load_kwargs: dict[str, Any] = {"torch_dtype": torch.float16, "trust_remote_code": False}
    if device == "cuda":
        load_kwargs["device_map"] = {"": 0}
    base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    merged = PeftModel.from_pretrained(base, str(adapter_path)).merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir))
    # Prefer the adapter dir's tokenizer (saved by train-run), else the base's.
    tokenizer_source = adapter_path if (adapter_path / "tokenizer_config.json").exists() else base_model
    AutoTokenizer.from_pretrained(str(tokenizer_source), trust_remote_code=False).save_pretrained(str(output_dir))
