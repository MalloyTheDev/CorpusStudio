"""Generate reasoning traces from prompts via any ``ModelBackend`` — backend-agnostic:

* **self-distillation** — point it at your own model (Ollama) so it reasons about its own tasks;
* **teacher** — a bigger local model, or an API model (openai_compatible), writes the reasoning.

The pipeline is PURE over an injected ``generate_fn`` (messages → text), so it is testable without a
network, and it **self-filters**: a generated trace is kept only if it carries real reasoning and does
not FAIL the reasoning-quality gate. Accepted traces are ``dataset_format=trace`` rows ready to train.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from corpus_studio.training.traces import (
    DEFAULT_THINK_CLOSE,
    DEFAULT_THINK_OPEN,
    Trace,
    TraceQualityReport,
    _pick,
    _split_think,
    trace_quality,
)

# messages -> completion text. The seam that makes the pipeline pure + backend-agnostic.
GenerateFn = Callable[[list[dict[str, str]]], str]

DEFAULT_REASONING_SYSTEM = (
    "You are a careful reasoner. Work through the problem step by step INSIDE <think> and </think> "
    "tags, then give your final answer AFTER the closing </think> tag. The final answer must stand on "
    "its own and must NOT repeat the reasoning."
)

_PROMPT_KEYS = ("prompt", "instruction", "question", "input", "query", "task")


class TraceGenerationResult(BaseModel):
    prompt: str
    trace: Trace | None = None
    quality: TraceQualityReport | None = None
    accepted: bool = False
    reason: str = ""


def prompt_from_row(row: dict[str, Any]) -> str:
    """Extract the generation prompt from a corpus row — a flat prompt field, or the concatenated
    non-assistant chat turns."""
    flat = _pick(row, _PROMPT_KEYS)
    if flat:
        return flat
    messages = row.get("messages")
    if isinstance(messages, list):
        return "\n".join(
            str(m.get("content", "")).strip()
            for m in messages
            if isinstance(m, dict) and m.get("role") != "assistant"
        ).strip()
    return ""


def build_reasoning_messages(prompt: str, system: str = DEFAULT_REASONING_SYSTEM) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def parse_generated_trace(
    prompt: str,
    completion: str,
    *,
    think_open: str = DEFAULT_THINK_OPEN,
    think_close: str = DEFAULT_THINK_CLOSE,
) -> Trace:
    thinking, answer = _split_think(completion, think_open, think_close)
    return Trace(prompt=prompt, thinking=thinking, answer=answer)


def generate_trace(
    prompt: str,
    generate_fn: GenerateFn,
    *,
    system: str = DEFAULT_REASONING_SYSTEM,
    require_thinking: bool = True,
) -> TraceGenerationResult:
    """Generate one trace for a prompt — PURE over ``generate_fn``. Kept only if it has an answer,
    (when ``require_thinking``) a real reasoning trace, and does not FAIL the quality gate. A backend
    error rejects the item, never aborts the batch."""
    try:
        completion = generate_fn(build_reasoning_messages(prompt, system))
    except Exception as exc:  # noqa: BLE001 - one prompt's failure must not kill the whole generation.
        return TraceGenerationResult(prompt=prompt, accepted=False, reason=f"generation error: {exc}")

    trace = parse_generated_trace(prompt, completion)
    quality = trace_quality(trace)
    if not trace.answer:
        return TraceGenerationResult(prompt=prompt, trace=trace, quality=quality, reason="no answer")
    if require_thinking and not trace.thinking.strip():
        return TraceGenerationResult(prompt=prompt, trace=trace, quality=quality, reason="no reasoning produced")
    if quality.status == "fail":
        return TraceGenerationResult(
            prompt=prompt, trace=trace, quality=quality, reason="quality: " + "; ".join(quality.issues)
        )
    return TraceGenerationResult(prompt=prompt, trace=trace, quality=quality, accepted=True)


def generate_traces(
    prompts: list[str],
    generate_fn: GenerateFn,
    *,
    system: str = DEFAULT_REASONING_SYSTEM,
    require_thinking: bool = True,
) -> list[TraceGenerationResult]:
    """Generate a trace per prompt (PURE over ``generate_fn``)."""
    return [
        generate_trace(p, generate_fn, system=system, require_thinking=require_thinking) for p in prompts
    ]


def backend_generate_fn(
    backend: Any, *, max_tokens: int | None = None, temperature: float | None = None, top_p: float | None = None
) -> GenerateFn:
    """Adapt a ``ModelBackend`` into a ``generate_fn`` for the pure pipeline (the only network seam)."""
    from corpus_studio.model_backends.base import BackendGenerateRequest  # noqa: PLC0415

    def _fn(messages: list[dict[str, str]]) -> str:
        response = backend.generate(
            BackendGenerateRequest(
                messages=messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p
            )
        )
        return response.text

    return _fn
