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

from pydantic import BaseModel, Field

from corpus_studio.training.traces import (
    DEFAULT_THINK_CLOSE,
    DEFAULT_THINK_OPEN,
    Trace,
    TraceQualityReport,
    _pick,
    _split_think,
    trace_quality,
)
from corpus_studio.platform.trace_records import canonical_sha256, text_sha256


class GeneratedCompletion(BaseModel):
    """Provider-neutral completion evidence retained without credentials or raw response bodies."""

    text: str
    model_name: str | None = None
    response_sha256: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# messages -> completion text/evidence. The seam that makes the pipeline pure + backend-agnostic.
GenerateFn = Callable[[list[dict[str, str]]], str | GeneratedCompletion]

DEFAULT_REASONING_SYSTEM = (
    "You are a careful reasoner. Work through the problem step by step INSIDE <think> and </think> "
    "tags, then give your final answer AFTER the closing </think> tag. The final answer must stand on "
    "its own and must NOT repeat the reasoning."
)

_PROMPT_KEYS = ("prompt", "instruction", "question", "input", "query", "task")


class TraceGenerationResult(BaseModel):
    prompt: str
    context: list[dict[str, str]] = Field(default_factory=list)
    request_messages: list[dict[str, str]] = Field(default_factory=list)
    trace: Trace | None = None
    quality: TraceQualityReport | None = None
    response_sha256: str | None = None
    response_model: str | None = None
    response_metadata: dict[str, Any] = Field(default_factory=dict)
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


def context_from_row(row: dict[str, Any]) -> list[dict[str, str]]:
    """Preserve structured context for generation, trimming only a trailing assistant target."""

    raw = row.get("messages")
    if isinstance(raw, list):
        context: list[dict[str, str]] = []
        for message in raw:
            if not isinstance(message, dict):
                return []
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant", "tool"} or not isinstance(content, str):
                return []
            item = {"role": str(role), "content": content}
            for key in ("name", "tool_call_id"):
                if isinstance(message.get(key), str) and message[key]:
                    item[key] = str(message[key])
            context.append(item)
        while context and context[-1]["role"] == "assistant":
            context.pop()
        if context:
            return context
    flat = _pick(row, _PROMPT_KEYS)
    return [{"role": "user", "content": flat}] if flat else []


def build_reasoning_messages(
    prompt: str | list[dict[str, str]],
    system: str = DEFAULT_REASONING_SYSTEM,
) -> list[dict[str, str]]:
    context = (
        [{"role": "user", "content": prompt}]
        if isinstance(prompt, str)
        else [dict(message) for message in prompt]
    )
    return [{"role": "system", "content": system}, *context]


def parse_generated_trace(
    prompt: str | list[dict[str, str]],
    completion: str,
    *,
    think_open: str = DEFAULT_THINK_OPEN,
    think_close: str = DEFAULT_THINK_CLOSE,
) -> Trace:
    thinking, answer = _split_think(completion, think_open, think_close)
    if isinstance(prompt, str):
        return Trace(prompt=prompt, thinking=thinking, answer=answer)
    return Trace(messages=[dict(message) for message in prompt], thinking=thinking, answer=answer)


def generate_trace(
    prompt: str | list[dict[str, str]],
    generate_fn: GenerateFn,
    *,
    system: str = DEFAULT_REASONING_SYSTEM,
    require_thinking: bool = True,
) -> TraceGenerationResult:
    """Generate one trace for a prompt — PURE over ``generate_fn``. Kept only if it has an answer,
    (when ``require_thinking``) a real reasoning trace, and does not FAIL the quality gate. A backend
    error rejects the item, never aborts the batch."""
    context = (
        [{"role": "user", "content": prompt}]
        if isinstance(prompt, str)
        else [dict(message) for message in prompt]
    )
    prompt_text = "\n".join(message.get("content", "") for message in context).strip()
    request_messages = build_reasoning_messages(context, system)
    try:
        generated = generate_fn(request_messages)
    except Exception as exc:  # noqa: BLE001 - one prompt's failure must not kill the whole generation.
        return TraceGenerationResult(
            prompt=prompt_text,
            context=context,
            request_messages=request_messages,
            accepted=False,
            reason=f"generation error: {exc}",
        )

    if isinstance(generated, GeneratedCompletion):
        completion = generated.text
        response_sha256 = generated.response_sha256
        response_model = generated.model_name
        response_metadata = generated.metadata
    else:
        completion = generated
        response_sha256 = text_sha256(generated)
        response_model = None
        response_metadata = {}

    try:
        trace = parse_generated_trace(context, completion)
    except ValueError as exc:
        return TraceGenerationResult(
            prompt=prompt_text,
            context=context,
            request_messages=request_messages,
            response_sha256=response_sha256,
            response_model=response_model,
            response_metadata=response_metadata,
            accepted=False,
            reason=str(exc),
        )
    quality = trace_quality(trace)

    def _result(*, reason: str = "", accepted: bool = False) -> TraceGenerationResult:
        return TraceGenerationResult(
            prompt=prompt_text,
            context=context,
            request_messages=request_messages,
            trace=trace,
            quality=quality,
            response_sha256=response_sha256,
            response_model=response_model,
            response_metadata=response_metadata,
            accepted=accepted,
            reason=reason,
        )

    if not trace.answer:
        return _result(reason="no answer")
    if require_thinking and not trace.thinking.strip():
        return _result(reason="no reasoning produced")
    if quality.status == "fail":
        return _result(reason="quality: " + "; ".join(quality.issues))
    return _result(accepted=True)


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

    def _fn(messages: list[dict[str, str]]) -> GeneratedCompletion:
        response = backend.generate(
            BackendGenerateRequest(
                messages=messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p
            )
        )
        raw = response.raw if isinstance(response.raw, dict) else None
        metadata: dict[str, Any] = {}
        raw_response_sha256 = canonical_sha256(raw) if raw is not None else None
        if raw is not None:
            for key in (
                "id",
                "created",
                "model",
                "done",
                "done_reason",
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
                "usage",
                "system_fingerprint",
            ):
                if key in raw:
                    metadata[key] = raw[key]
            metadata["raw_response_sha256"] = raw_response_sha256
        reported_model = raw.get("model") if raw is not None else None
        resolved_model = (
            reported_model.strip()
            if isinstance(reported_model, str) and reported_model.strip()
            else response.model_name.strip()
        )
        return GeneratedCompletion(
            text=response.text,
            model_name=resolved_model,
            response_sha256=canonical_sha256(
                {
                    "text": response.text,
                    "model_name": resolved_model,
                    "raw_response_sha256": raw_response_sha256,
                }
            ),
            metadata=metadata,
        )

    return _fn
