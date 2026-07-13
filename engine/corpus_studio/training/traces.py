"""Reasoning-trace ("thinking trace") support — the data substrate for training reasoning models.

A *trace* pairs a prompt with an explicit REASONING process and a final ANSWER. Rendered for training
with the reasoning wrapped in ``<think>…</think>`` before the answer (the DeepSeek-R1 / Qwen-QwQ
convention), it teaches a model to reason-then-answer. The paired **no-think** rendering (answer only —
the same example without the reasoning shown) is the baseline, so one corpus can train or ablate both.

Pure stdlib + pydantic — no torch. The trainer's ``format_example_text`` dispatches the ``trace``
dataset format here; the ``trace-validate`` CLI checks a trace corpus before a run.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from pydantic import BaseModel

# The reasoning delimiters. Default to the widely-adopted <think>…</think> tags; overridable so a
# corpus can match a base model whose chat template already reserves different markers.
DEFAULT_THINK_OPEN = "<think>"
DEFAULT_THINK_CLOSE = "</think>"

# Field aliases we accept from a raw row — reasoning corpora in the wild use many names.
_PROMPT_KEYS = ("prompt", "instruction", "question", "input", "query")
_THINKING_KEYS = ("thinking", "reasoning", "rationale", "cot", "thought", "reasoning_content")
_ANSWER_KEYS = ("answer", "output", "response", "solution", "final", "completion")


class Trace(BaseModel):
    """One reasoning example: an optional flat ``prompt`` OR chat ``messages`` for the context, an
    explicit ``thinking`` trace, and the final ``answer``."""

    prompt: str = ""
    thinking: str = ""
    answer: str = ""
    messages: list[dict[str, Any]] | None = None  # chat context (system+user); the assistant turn is built


def _pick(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def trace_from_row(row: dict[str, Any]) -> Trace:
    """Build a :class:`Trace` from a JSONL row, tolerant of common field aliases. If the row is chat
    (messages) and the last message is the assistant's, its content is split into thinking/answer when
    it already carries ``<think>`` tags — so an existing reasoning corpus round-trips."""
    from corpus_studio.platform.trace_records import (  # noqa: PLC0415
        is_trace_record_row,
        legacy_trace_from_record,
        parse_trace_record,
    )

    if is_trace_record_row(row):
        return legacy_trace_from_record(parse_trace_record(row))

    messages = row.get("messages") if isinstance(row.get("messages"), list) else None
    thinking = _pick(row, _THINKING_KEYS)
    answer = _pick(row, _ANSWER_KEYS)

    # A chat corpus may embed the reasoning inside the final assistant turn as <think>…</think>answer.
    if messages and not (thinking or answer):
        last = messages[-1] if messages else {}
        if isinstance(last, dict) and last.get("role") == "assistant":
            content = str(last.get("content", ""))
            thinking, answer = _split_think(content)
            if thinking or answer:
                messages = messages[:-1]  # the assistant turn is rebuilt from thinking+answer

    return Trace(
        prompt=_pick(row, _PROMPT_KEYS),
        thinking=thinking,
        answer=answer,
        messages=messages,
    )


def _split_think(
    content: str,
    open_tag: str = DEFAULT_THINK_OPEN,
    close_tag: str = DEFAULT_THINK_CLOSE,
) -> tuple[str, str]:
    """Strict legacy/import parser for ``<think>reasoning</think>answer``.

    Structured :class:`TraceRecord` segments never contain delimiter markup. At an import boundary,
    no tags means an answer-only baseline; any partial, repeated, reversed, or nested tag structure
    is rejected instead of being silently flattened into misleading training data.
    """
    open_count = content.count(open_tag)
    close_count = content.count(close_tag)
    if not open_count and not close_count:
        return "", content.strip()
    if open_count != 1 or close_count != 1:
        raise ValueError("malformed reasoning markup: expected exactly one <think>...</think> pair")
    open_index = content.index(open_tag)
    close_index = content.index(close_tag)
    if close_index < open_index + len(open_tag):
        raise ValueError("malformed reasoning markup: closing tag precedes the opening tag")
    head = content[:open_index]
    if head.strip():
        raise ValueError("malformed reasoning markup: content before the opening <think> tag")
    reasoning = content[open_index + len(open_tag) : close_index]
    answer = content[close_index + len(close_tag) :]
    if open_tag in reasoning or close_tag in reasoning:
        raise ValueError("malformed reasoning markup: nested reasoning tags are not supported")
    return reasoning.strip(), answer.strip()


def answer_for_scoring(
    model_output: str, *, think_open: str = DEFAULT_THINK_OPEN, think_close: str = DEFAULT_THINK_CLOSE
) -> tuple[str, bool]:
    """Strip a reasoning model's ``<think>…</think>`` block for evaluation, returning
    ``(answer, had_thinking)`` — so a scorer compares the model's ANSWER to the reference, not its
    reasoning (which would corrupt the score). No tags → ``(text_unchanged, False)``."""
    try:
        thinking, answer = _split_think(model_output, think_open, think_close)
    except ValueError:
        # Evaluation must not crash on malformed model output. Leave it unchanged so the scorer sees
        # the actual failure instead of accidentally stripping or repairing it.
        return model_output.strip(), False
    return answer, bool(thinking)


def format_trace(
    trace: Trace,
    *,
    show_thinking: bool = True,
    think_open: str = DEFAULT_THINK_OPEN,
    think_close: str = DEFAULT_THINK_CLOSE,
    tokenizer: Any | None = None,
) -> str:
    """Render a trace to one training-text string.

    ``show_thinking=True`` → prompt + ``<think>reasoning</think>`` + answer (trains the reasoning).
    ``show_thinking=False`` → prompt + answer (the no-think baseline). Returns "" for an empty trace.
    """
    if not trace.answer and not (show_thinking and trace.thinking):
        return ""
    completion = (
        f"{think_open}\n{trace.thinking}\n{think_close}\n\n{trace.answer}"
        if show_thinking and trace.thinking
        else trace.answer
    )

    if trace.messages:
        full = [*trace.messages, {"role": "assistant", "content": completion}]
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return str(tokenizer.apply_chat_template(full, tokenize=False))
            except Exception:  # noqa: BLE001 - a template failure falls back to the plain join.
                pass
        return "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}" for m in full if isinstance(m, dict)
        )

    return f"### Instruction:\n{trace.prompt}\n\n### Response:\n{completion}"


class TraceValidation(BaseModel):
    """A trace's structural + basic-quality verdict — the guardrail for a reasoning corpus."""

    valid: bool
    errors: list[str]
    has_thinking: bool
    thinking_chars: int
    answer_chars: int


def validate_trace(trace: Trace) -> TraceValidation:
    """PURE + unit-tested. Structural checks for a trace: an answer and a prompt/context must exist;
    the reasoning must not be a verbatim copy of the answer (no real reasoning); a trace intended to
    train reasoning must actually have a non-empty thinking trace."""
    errors: list[str] = []
    if not trace.answer:
        errors.append("missing answer")
    if not trace.prompt and not trace.messages:
        errors.append("missing prompt/messages")
    if trace.thinking and trace.answer and trace.thinking.strip() == trace.answer.strip():
        errors.append("thinking is identical to the answer — no real reasoning")
    return TraceValidation(
        valid=not errors,
        errors=errors,
        has_thinking=bool(trace.thinking.strip()),
        thinking_chars=len(trace.thinking),
        answer_chars=len(trace.answer),
    )


# --- reasoning-quality gate (beyond structure) ---------------------------------------------------

_STATUS_RANK = {"pass": 0, "warn": 1, "fail": 2}


def _worse(a: str, b: str) -> str:
    return a if _STATUS_RANK[a] >= _STATUS_RANK[b] else b


class TraceQualityFinding(BaseModel):
    code: str
    severity: str
    location: str
    message: str


class TraceQualityReport(BaseModel):
    """A trace's REASONING-quality verdict — the checks that a structurally-valid trace can still
    fail: a leaked answer, malformed tags, or a token-thin "reasoning" that teaches nothing.
    ``status`` is pass / warn / fail; a reasoning corpus should gate on ``fail``."""

    status: str
    issues: list[str]
    findings: list[TraceQualityFinding]
    thinking_chars: int
    answer_chars: int
    thinking_to_answer_ratio: float


def _normalized_exact(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def trace_quality(
    trace: Trace, *, min_thinking_chars: int = 24, min_ratio: float = 0.15, leak_min_chars: int = 40
) -> TraceQualityReport:
    """PURE + unit-tested. Reasoning-quality checks that structural validation (``validate_trace``)
    doesn't cover:

    * **answer leaked in the prompt** — a substantial answer appearing verbatim in the prompt means the
      task already contains its answer (data leakage) → ``fail``;
    * **stray ``<think>`` tags in the answer** — malformed/unsplit reasoning markers → ``fail``;
    * **token-thin reasoning** — a thinking trace far shorter than the answer (or trivially short) is
      not real reasoning → ``warn``.
    """
    findings: list[TraceQualityFinding] = []
    status = "pass"

    prompt_text = trace.prompt or " ".join(
        str(m.get("content", "")) for m in (trace.messages or []) if isinstance(m, dict)
    )
    answer = trace.answer.strip()
    thinking = trace.thinking.strip()

    normalized_answer = _normalized_exact(answer)
    normalized_prompt = _normalized_exact(prompt_text)
    if (
        normalized_answer
        and len(normalized_answer) >= leak_min_chars
        and normalized_answer in normalized_prompt
    ):
        findings.append(
            TraceQualityFinding(
                code="answer_leak_normalized_exact",
                severity="block",
                location="context",
                message="answer leakage: answer appears in the prompt after Unicode/case/whitespace normalization",
            )
        )
        status = _worse(status, "fail")
    if DEFAULT_THINK_OPEN in trace.answer or DEFAULT_THINK_CLOSE in trace.answer:
        findings.append(
            TraceQualityFinding(
                code="stray_think_markup",
                severity="block",
                location="final_answer",
                message="stray <think> tags remain in the answer (malformed/unsplit)",
            )
        )
        status = _worse(status, "fail")

    ratio = (len(thinking) / len(answer)) if answer else 0.0
    if thinking and len(thinking) < min_thinking_chars:
        findings.append(
            TraceQualityFinding(
                code="reasoning_too_short",
                severity="warning",
                location="reasoning",
                message=f"reasoning is trivially short ({len(thinking)} chars) - not substantial",
            )
        )
        status = _worse(status, "warn")
    elif thinking and answer and ratio < min_ratio:
        findings.append(
            TraceQualityFinding(
                code="reasoning_answer_ratio_low",
                severity="warning",
                location="reasoning",
                message=f"reasoning is very short vs the answer (ratio {ratio:.2f})",
            )
        )
        status = _worse(status, "warn")

    return TraceQualityReport(
        status=status,
        issues=[item.message for item in findings],
        findings=findings,
        thinking_chars=len(thinking),
        answer_chars=len(answer),
        thinking_to_answer_ratio=round(ratio, 3),
    )
