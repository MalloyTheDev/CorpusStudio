"""Reasoning-trace substrate — the data layer for training reasoning models (prompt +
``<think>reasoning</think>`` + answer), plus the no-think baseline + validation guardrail."""

from corpus_studio.training.traces import (
    Trace,
    _split_think,
    format_trace,
    trace_from_row,
    validate_trace,
)
from corpus_studio.training.trainer import format_example_text


def test_trace_from_row_field_aliases():
    trace = trace_from_row({"question": "2+2?", "reasoning": "add two and two", "solution": "4"})
    assert trace.prompt == "2+2?" and trace.thinking == "add two and two" and trace.answer == "4"


def test_trace_from_row_splits_embedded_think_in_chat():
    # A chat corpus with the reasoning already inside the assistant turn as <think>…</think>answer.
    row = {
        "messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "<think>add them</think>4"},
        ]
    }
    trace = trace_from_row(row)
    assert trace.thinking == "add them" and trace.answer == "4"
    assert trace.messages == [{"role": "user", "content": "2+2?"}]  # assistant turn stripped (rebuilt)


def test_split_think():
    assert _split_think("<think>r</think>a") == ("r", "a")
    assert _split_think("no tags here") == ("", "no tags here")


def test_format_trace_shows_reasoning():
    out = format_trace(Trace(prompt="Q", thinking="reason", answer="A"), show_thinking=True)
    assert "<think>" in out and "reason" in out and out.rstrip().endswith("A")


def test_format_trace_no_think_baseline_hides_reasoning():
    out = format_trace(Trace(prompt="Q", thinking="reason", answer="A"), show_thinking=False)
    assert "<think>" not in out and "reason" not in out and "A" in out


def test_format_trace_empty_is_dropped():
    assert format_trace(Trace()) == ""


def test_format_trace_messages_builds_the_assistant_turn():
    trace = Trace(messages=[{"role": "user", "content": "Q"}], thinking="reason", answer="A")
    out = format_trace(trace)  # no tokenizer → plain role:content join
    assert "assistant:" in out and "<think>" in out and "A" in out


def test_validate_trace_ok():
    verdict = validate_trace(Trace(prompt="Q", thinking="r", answer="A"))
    assert verdict.valid and verdict.has_thinking


def test_validate_trace_missing_answer():
    verdict = validate_trace(Trace(prompt="Q", thinking="r"))
    assert not verdict.valid and "missing answer" in verdict.errors


def test_validate_trace_thinking_equals_answer_is_not_reasoning():
    verdict = validate_trace(Trace(prompt="Q", thinking="same", answer="same"))
    assert not verdict.valid and any("no real reasoning" in e for e in verdict.errors)


def test_format_example_text_dispatches_the_trace_format():
    out = format_example_text({"instruction": "Q", "reasoning": "r", "output": "A"}, "trace")
    assert "<think>" in out and "r" in out and "A" in out
