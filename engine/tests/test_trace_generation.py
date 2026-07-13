"""Backend-agnostic trace generation — pure over an injected generate_fn, self-filtering on the
reasoning-quality gate. No network: every backend call is a fake function."""

from corpus_studio.training.trace_generation import (
    build_reasoning_messages,
    generate_trace,
    generate_traces,
    parse_generated_trace,
    prompt_from_row,
)


def test_build_reasoning_messages_elicits_think_tags():
    messages = build_reasoning_messages("solve 2+2")
    assert messages[0]["role"] == "system" and "<think>" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "solve 2+2"}


def test_parse_generated_trace_splits_think_from_answer():
    trace = parse_generated_trace("Q", "<think>reasoning here</think>\n\nfinal answer")
    assert trace.thinking == "reasoning here" and trace.answer == "final answer"


def test_generate_trace_accepts_a_reasoned_answer():
    result = generate_trace("What is 17*23?", lambda m: "<think>17*20+17*3 = 340+51 = 391</think>\n\n391")
    assert result.accepted and result.trace.thinking and result.trace.answer == "391"


def test_generate_trace_rejects_no_reasoning():
    result = generate_trace("Q?", lambda m: "391")  # no <think> tags at all
    assert not result.accepted and "no reasoning" in result.reason


def test_generate_trace_rejects_no_answer():
    result = generate_trace("Q?", lambda m: "<think>thinking but nothing after</think>")
    assert not result.accepted and "no answer" in result.reason


def test_generate_trace_rejects_a_quality_failure():
    # answer (>=40 chars) is verbatim in the prompt → leakage → quality FAIL → rejected.
    answer = "The Ostervaal Registry decides what a legitimate self is."
    result = generate_trace(answer, lambda m: f"<think>reasoning about accreditation and review</think>{answer}")
    assert not result.accepted and "quality" in result.reason


def test_generate_trace_survives_a_backend_error():
    def boom(_messages):
        raise RuntimeError("backend down")

    result = generate_trace("Q?", boom)
    assert not result.accepted and "generation error" in result.reason


def test_generate_traces_batch_isolates_per_prompt_failures():
    calls = {"n": 0}

    def flaky(_messages):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("transient")
        return "<think>real reasoning here</think>ans"

    results = generate_traces(["a", "b", "c"], flaky)
    assert len(results) == 3
    assert results[0].accepted and not results[1].accepted and results[2].accepted


def test_prompt_from_row_aliases_and_messages():
    assert prompt_from_row({"question": "2+2?"}) == "2+2?"
    assert (
        prompt_from_row({"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]})
        == "sys\nhi"
    )
    # The assistant turn is excluded (that's what we're generating).
    assert prompt_from_row({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}) == "q"
    assert prompt_from_row({}) == ""
