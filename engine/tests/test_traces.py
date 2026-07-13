"""Reasoning-trace substrate — the data layer for training reasoning models (prompt +
``<think>reasoning</think>`` + answer), plus the no-think baseline + validation guardrail."""

import pytest

from corpus_studio.training.traces import (
    Trace,
    _split_think,
    answer_for_scoring,
    format_trace,
    trace_from_row,
    trace_quality,
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


def test_split_think_rejects_partial_or_repeated_markup():
    with pytest.raises(ValueError, match="exactly one"):
        _split_think("<think>reasoning only")
    with pytest.raises(ValueError, match="exactly one"):
        _split_think("<think>one<think>two</think>A")
    with pytest.raises(ValueError, match="content before"):
        _split_think("FINAL-FIRST<think>post-hoc rationale</think>")


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


# ---- reasoning quality gate --------------------------------------------------


def test_trace_quality_passes_a_real_derivation():
    trace = Trace(prompt="What is 17*23?", thinking="17*23 = 17*20 + 17*3 = 340 + 51 = 391", answer="391")
    gate = trace_quality(trace)
    assert gate.status == "pass" and not gate.issues


def test_trace_quality_fails_answer_leaked_in_prompt():
    answer = "The Ostervaal Registry decides what counts as a legitimate self."
    trace = Trace(
        prompt=f"Explain this: {answer}",
        thinking="the registry gatekeeps personhood through accreditation, review, and registered materials",
        answer=answer,
    )
    gate = trace_quality(trace)
    assert gate.status == "fail" and any("leakage" in i for i in gate.issues)


def test_trace_quality_fails_stray_think_tags_in_answer():
    trace = Trace(prompt="Q", thinking="reason about it carefully, step by step", answer="ans <think>oops</think>")
    gate = trace_quality(trace)
    assert gate.status == "fail" and any("stray" in i for i in gate.issues)


def test_trace_quality_warns_on_token_thin_reasoning():
    trace = Trace(prompt="Q", thinking="yes", answer="A long, detailed answer far longer than the reasoning.")
    gate = trace_quality(trace)
    assert gate.status == "warn" and any("short" in i for i in gate.issues)


def test_trace_quality_no_think_baseline_is_not_a_failure():
    # A no-think example (answer only) is legitimate — the quality gate must not fail it.
    assert trace_quality(Trace(prompt="Q", answer="A")).status == "pass"


# ---- answer_for_scoring (trace-aware eval) -----------------------------------


def test_answer_for_scoring_strips_the_reasoning():
    answer, had = answer_for_scoring("<think>let me work it out</think>\n\nThe answer is 391.")
    assert answer == "The answer is 391." and had is True


def test_answer_for_scoring_no_tags_returns_text_unchanged():
    answer, had = answer_for_scoring("Just a plain answer, no reasoning.")
    assert answer == "Just a plain answer, no reasoning." and had is False
