"""Adapter model-card generation for the first-party trainer (pure — no heavy imports).

After ``train-run`` writes a LoRA adapter, a *model card* documents the training **recipe** so the
result is self-describing and handoff-ready: the base model (and the reminder that ITS license governs
what you may do with the adapter), the LoRA hyper-parameters (read from the adapter's own
``adapter_config.json``), the training settings, and the run provenance when available.

The card is deliberately honest: a completed run is not a quality signal, and a CPU-toy run produces a
smoke-test artifact, not a usable model. Everything here is pure string/JSON work — no torch/peft — so
it renders anywhere and is unit-tested without the ``[train]`` extra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_adapter_config(adapter_dir: Path | str) -> dict[str, Any] | None:
    """The adapter's ``adapter_config.json`` (base model + LoRA params), or None when absent/unreadable."""
    path = Path(adapter_dir) / "adapter_config.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _format_target_modules(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value) if value is not None else "?"


def _evaluation_lines(evaluation: dict[str, Any] | None) -> list[str]:
    """Render the Evaluation section - honestly. No eval attached => 'Not evaluated' (null-with-reason),
    never a fabricated pass. When attached, report the metric + measured score + the exact decode, and
    caveat that it is a signal on THIS dataset/metric, not a comprehensive quality guarantee."""
    if not evaluation:
        return [
            "",
            "## Evaluation",
            "",
            "- **Not evaluated.** No evaluation report is attached to this card. Run "
            "`corpus-studio eval-run <data> <schema> --model ... --output-path <report>.json`, then "
            "`corpus-studio model-card <adapter> --eval-report <report>.json`.",
        ]
    metric = evaluation.get("metric") or "unknown"
    tested = evaluation.get("examples_tested")
    failed = evaluation.get("failed_examples") or 0
    average = evaluation.get("average_score")
    settings = evaluation.get("run_settings") or {}

    lines = ["", "## Evaluation", "", f"- **Metric:** `{metric}`"]
    if isinstance(average, (int, float)) and not isinstance(average, bool) and tested is not None:
        failed_note = f" ({failed} failed)" if failed else ""
        lines.append(f"- **Score:** {float(average):.2f} average over {tested} example(s){failed_note}.")
    else:
        # an unavailable score is null-with-reason, never a fabricated 0
        lines.append("- **Score:** unavailable (the report carries no average_score).")

    model_name = settings.get("model")
    if model_name:
        temperature = settings.get("temperature")
        greedy = "greedy" if temperature in (0, 0.0) else f"temperature {temperature}"
        lines.append(
            f"- **Decoded as:** model `{model_name}`, {greedy}, seed {settings.get('seed')}, up to "
            f"{settings.get('max_output_tokens')} tokens (recorded for reproducibility)."
        )
    lines.append(
        "- This is a measured signal on THIS dataset + metric, not a comprehensive quality guarantee: a "
        "keyword-overlap score is lexical recall (not correctness) and a schema-conformance score is "
        "structural completeness (not meaning)."
    )
    return lines


def build_model_card(
    adapter_dir: Path | str,
    *,
    base_model: str | None = None,
    training_config: dict[str, Any] | None = None,
    train_result: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> str:
    """Render a Markdown model card for a trained LoRA adapter.

    Reads the LoRA hyper-parameters from the adapter's own ``adapter_config.json`` (so the card matches
    what was actually trained). ``base_model`` overrides the recorded base; ``training_config`` (the
    CorpusStudio training config), ``train_result`` (steps/loss/cpu_toy), and ``provenance`` (dataset
    fingerprint / config hash / engine) are folded in when supplied. Pure — reads only local files."""
    adapter_dir = Path(adapter_dir)
    adapter_config = read_adapter_config(adapter_dir) or {}
    training_config = training_config or {}
    train_result = train_result or {}
    provenance = provenance or {}

    resolved_base = (
        base_model
        or adapter_config.get("base_model_name_or_path")
        or training_config.get("base_model")
        or "unknown"
    )
    cpu_toy = bool(train_result.get("cpu_toy"))

    lines: list[str] = [f"# Model card: {adapter_dir.name} (LoRA adapter)", ""]
    lines.append(
        "A LoRA adapter trained with Corpus Studio's first-party trainer. This card documents the "
        "training **recipe** — it is NOT an evaluation of the model's quality. Evaluate before use."
    )
    if generated_at:
        lines += ["", f"_Generated {generated_at}._"]

    if cpu_toy:
        lines += [
            "",
            "> ⚠ **CPU toy smoke test.** This adapter came from the `--cpu-toy` path: a tiny model, a "
            "few steps, on CPU. It proves the training pipeline runs; it is **not a usable model**. "
            "Discard it and run a real GPU QLoRA for anything real.",
        ]

    lines += ["", "## Base model", ""]
    lines.append(f"- **Base:** `{resolved_base}`")
    lines.append(
        "- **License:** the *base model's* license governs what you may do with this adapter and any "
        "merged model — training on data you can read does not grant redistribution rights to the base. "
        "`corpus-studio model-fetch <repo>` reports a base model's license; prefer a permissive "
        "(MIT/Apache/BSD) base and verify before distributing."
    )

    lines += ["", "## Adapter (LoRA)", ""]
    peft_type = adapter_config.get("peft_type", "LORA")
    task_type = adapter_config.get("task_type", "CAUSAL_LM")
    lines.append(f"- Type: {peft_type} / {task_type}")
    if adapter_config:
        lines.append(
            f"- r: {adapter_config.get('r', '?')}, alpha: {adapter_config.get('lora_alpha', '?')}, "
            f"dropout: {adapter_config.get('lora_dropout', '?')}"
        )
        lines.append(f"- Target modules: {_format_target_modules(adapter_config.get('target_modules'))}")
    else:
        lines.append("- (no adapter_config.json found — hyper-parameters unavailable)")

    lines += ["", "## Training", ""]
    if training_config:
        lines.append(f"- Format: {training_config.get('format', '?')}")
        lines.append(
            f"- Sequence length: {training_config.get('sequence_len', '?')}, "
            f"learning rate: {training_config.get('learning_rate', '?')}, "
            f"seed: {training_config.get('seed', '?')}"
        )
    if train_result:
        loss = train_result.get("final_loss")
        loss_text = f"{loss:.4f}" if isinstance(loss, (int, float)) else "n/a"
        lines.append(f"- Steps: {train_result.get('steps', '?')}, final train loss: {loss_text}")
    lines.append(f"- Mode: {'CPU toy (smoke test — not a usable model)' if cpu_toy else '4-bit QLoRA'}")

    if provenance:
        lines += ["", "## Provenance (reproducibility)", ""]
        fingerprint = provenance.get("dataset_fingerprint")
        if fingerprint:
            rows = provenance.get("dataset_row_count")
            lines.append(f"- Dataset fingerprint: `{fingerprint}`" + (f" ({rows} rows)" if rows else ""))
        if provenance.get("config_sha256"):
            lines.append(f"- Config hash: `{provenance['config_sha256']}`")
        if provenance.get("engine_version"):
            platform = provenance.get("platform")
            lines.append(f"- Engine: {provenance['engine_version']}" + (f" on {platform}" if platform else ""))

    lines += _evaluation_lines(evaluation)

    lines += [
        "",
        "## Serving",
        "",
        "- Merge into the base with `corpus-studio train-merge <adapter-dir>` (auto → GPU / CPU-offload "
        "/ adapter-only), or serve the base + adapter unmerged (peft / vLLM / TGI accept the adapter).",
        "",
        "## Honesty",
        "",
        "- A completed run is not a quality signal — see the Evaluation section (or run an evaluation "
        "suite) before promoting this adapter.",
        "- Reproducibility depends on the pinned seed + dataset fingerprint above; changing the data or "
        "config changes the result.",
    ]

    return "\n".join(lines) + "\n"


def write_model_card(adapter_dir: Path | str, card_markdown: str, *, filename: str = "MODEL_CARD.md") -> Path:
    """Write ``card_markdown`` to ``<adapter_dir>/<filename>`` and return the path."""
    output = Path(adapter_dir) / filename
    output.write_text(card_markdown, encoding="utf-8")
    return output
