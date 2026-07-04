"""Arena orchestration: run a prompt suite across model backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from corpus_studio.arena.models import ArenaPrompt, ArenaReport, ArenaResponse, build_arena_report
from corpus_studio.importers.jsonl_importer import read_jsonl
from corpus_studio.model_backends.base import BackendGenerateRequest, ModelBackend
from corpus_studio.model_backends.retry import BACKEND_ERROR_TYPES, format_backend_error


def load_prompt_suite(path: Path) -> list[ArenaPrompt]:
    """Load a prompt suite from JSONL rows: ``{prompt, id?, system?}``.

    Ids are assigned positionally (``prompt-1`` ...) when absent. Rows without a
    non-empty ``prompt`` are skipped.
    """

    prompts: list[ArenaPrompt] = []
    for index, row in enumerate(read_jsonl(path), start=1):
        if not isinstance(row, dict):
            continue
        text = str(row.get("prompt", "")).strip()
        if not text:
            continue
        prompt_id = str(row.get("id") or f"prompt-{index}")
        system = row.get("system")
        prompts.append(
            ArenaPrompt(
                id=prompt_id,
                prompt=text,
                system=str(system) if system not in (None, "") else None,
            )
        )
    return prompts


def _request_for(prompt: ArenaPrompt) -> BackendGenerateRequest:
    if prompt.system:
        return BackendGenerateRequest(
            messages=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.prompt},
            ]
        )
    return BackendGenerateRequest(prompt=prompt.prompt)


def run_arena(
    prompts: list[ArenaPrompt],
    model_backends: list[tuple[str, ModelBackend]],
    limit: int | None = None,
    generated_at: str | None = None,
) -> ArenaReport:
    """Run each prompt through each model and collect responses side by side."""

    selected = prompts[:limit] if limit is not None else prompts
    responses: list[ArenaResponse] = []
    for model, backend in model_backends:
        for prompt in selected:
            try:
                response = backend.generate(_request_for(prompt))
                responses.append(
                    ArenaResponse(prompt_id=prompt.id, model=model, text=response.text)
                )
            except BACKEND_ERROR_TYPES as exc:
                # Isolate one model/prompt failure: record it and keep going so a
                # single outage never discards the whole comparison.
                responses.append(
                    ArenaResponse(
                        prompt_id=prompt.id,
                        model=model,
                        text="",
                        error=format_backend_error(exc),
                    )
                )

    models = [model for model, _ in model_backends]
    return build_arena_report(selected, models, responses, generated_at)


def responses_for_prompt(report: ArenaReport, prompt_id: str) -> dict[str, Any]:
    """Collect one prompt's responses keyed by model (for side-by-side display)."""

    return {r.model: r.text for r in report.responses if r.prompt_id == prompt_id}
