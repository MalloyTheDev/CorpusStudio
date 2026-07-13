"""Backend-agnostic trace generation — pure over an injected generate_fn, self-filtering on the
reasoning-quality gate. No network: every backend call is a fake function."""

from corpus_studio.training.trace_generation import (
    GeneratedCompletion,
    backend_generate_fn,
    build_reasoning_messages,
    context_from_row,
    generate_trace,
    generate_traces,
    parse_generated_trace,
    prompt_from_row,
)
from corpus_studio.model_backends.base import BackendGenerateResponse
from corpus_studio.platform.trace_records import canonical_sha256


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


def test_generate_trace_rejects_malformed_reasoning_markup():
    result = generate_trace("Q?", lambda m: "<think>one<think>two</think>A")
    assert not result.accepted and "malformed reasoning markup" in result.reason

    result = generate_trace("Q?", lambda m: "answer first<think>post-hoc rationale</think>")
    assert not result.accepted and "content before" in result.reason


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


def test_context_from_row_preserves_roles_and_only_trims_trailing_target():
    context = context_from_row(
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "prior turn"},
                {"role": "user", "content": "next"},
                {"role": "assistant", "content": "target"},
            ]
        }
    )
    assert [message["role"] for message in context] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert context[-1]["content"] == "next"


def test_generate_trace_retains_response_evidence():
    result = generate_trace(
        "Q?",
        lambda messages: GeneratedCompletion(
            text="<think>A detailed reasoning process for the result.</think>A",
            model_name="model@revision",
            response_sha256="a" * 64,
            metadata={"usage": {"completion_tokens": 10}},
        ),
    )
    assert result.accepted
    assert result.response_model == "model@revision"
    assert result.response_sha256 == "a" * 64
    assert result.response_metadata["usage"]["completion_tokens"] == 10


def test_backend_response_hash_binds_text_model_and_raw_evidence():
    class FakeBackend:
        def generate(self, request):
            return BackendGenerateResponse(
                text="completion text",
                model_name="requested-alias",
                raw={"model": "resolved-model", "id": "response-1", "secret_body": "not stored"},
            )

    completion = backend_generate_fn(FakeBackend())([{"role": "user", "content": "Q"}])
    raw_hash = canonical_sha256(
        {"model": "resolved-model", "id": "response-1", "secret_body": "not stored"}
    )
    assert isinstance(completion, GeneratedCompletion)
    assert completion.model_name == "resolved-model"
    assert completion.metadata["raw_response_sha256"] == raw_hash
    assert "secret_body" not in completion.metadata
    assert completion.response_sha256 == canonical_sha256(
        {
            "text": "completion text",
            "model_name": "resolved-model",
            "raw_response_sha256": raw_hash,
        }
    )
