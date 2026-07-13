"""Reasoning-trace ("thinking trace") support — the data substrate for training reasoning models.

A *trace* pairs a prompt with an explicit REASONING process and a final ANSWER. Rendered for training
with the reasoning wrapped in ``<think>…</think>`` before the answer (the DeepSeek-R1 / Qwen-QwQ
convention), it teaches a model to reason-then-answer. The paired **no-think** rendering (answer only —
the same example without the reasoning shown) is the baseline, so one corpus can train or ablate both.

Pure stdlib + pydantic — no torch. The trainer's ``format_example_text`` dispatches the ``trace``
dataset format here; the ``trace-validate`` CLI checks a trace corpus before a run.
"""

from __future__ import annotations

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


def _split_think(content: str, open_tag: str = DEFAULT_THINK_OPEN, close_tag: str = DEFAULT_THINK_CLOSE) -> tuple[str, str]:
    """Split ``<think>reasoning</think>answer`` → (reasoning, answer). No tags → ("", content)."""
    if open_tag in content and close_tag in content:
        head, _, rest = content.partition(open_tag)
        reasoning, _, answer = rest.partition(close_tag)
        return reasoning.strip(), (head + answer).strip()
    return "", content.strip()


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
