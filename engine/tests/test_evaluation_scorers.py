import pytest

from corpus_studio.evaluation.evaluator import (
    EvaluationRunConfig,
    extract_evaluation_examples,
    run_evaluation,
)
from corpus_studio.evaluation.scorers import (
    KeywordOverlapScorer,
    LlmJudgeScorer,
    build_eval_judge_prompt,
    parse_eval_judgment,
)
from corpus_studio.evaluation.scoring import score_text_overlap
from corpus_studio.model_backends.base import BackendGenerateResponse
from corpus_studio.providers.policy import (
    ProviderPolicy,
    ProviderPolicyError,
    ProviderRole,
)


class FakeJudgeBackend:
    """Records requests and returns a canned judge response."""

    def __init__(self, text: str):
        self.text = text
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return BackendGenerateResponse(text=self.text, model_name="fake-judge")


def _evaluator_policy(can: bool) -> ProviderPolicy:
    return ProviderPolicy(
        provider_id="test",
        allowed_roles=[ProviderRole.EVALUATOR] if can else [],
    )


# ---- keyword-overlap scorer --------------------------------------------------


def test_keyword_overlap_scorer_matches_helper_and_labels_metric():
    scorer = KeywordOverlapScorer()
    assert scorer.metric == "keyword_overlap"
    result = scorer.score("prompt", "a function calls itself", "a function calls itself now")
    assert result.score == score_text_overlap("a function calls itself", "a function calls itself now")
    assert result.rationale is None


# ---- judge prompt + parsing --------------------------------------------------


def test_build_eval_judge_prompt_is_hardened_and_carries_fields():
    prompt = build_eval_judge_prompt("PROMPT_X", "REFERENCE_Y", "ANSWER_Z")
    assert "untrusted" in prompt
    assert "PROMPT_X" in prompt and "REFERENCE_Y" in prompt and "ANSWER_Z" in prompt
    assert "ONLY JSON" in prompt


@pytest.mark.parametrize(
    "text,expected_score",
    [
        ('{"score": 82, "rationale": "close paraphrase"}', 82.0),
        ("noise {\"score\": 150} noise", 100.0),   # clamped high
        ('{"score": -5}', 0.0),                      # clamped low
    ],
)
def test_parse_eval_judgment_clamps_valid_scores(text, expected_score):
    assert parse_eval_judgment(text).score == expected_score


@pytest.mark.parametrize(
    "text",
    [
        "the model did fine",           # prose, no JSON
        '{"rationale": "no score key"}',  # JSON without a numeric score
        '{"score": "high"}',              # non-numeric score
    ],
)
def test_parse_eval_judgment_flags_unparseable_without_crashing(text):
    result = parse_eval_judgment(text)
    assert result.score == 0.0
    assert result.rationale is not None
    assert result.rationale.startswith("judge_unparseable")


# ---- LLM judge scorer + policy ----------------------------------------------


def test_llm_judge_scorer_scores_and_keeps_rationale():
    backend = FakeJudgeBackend('{"score": 91, "rationale": "correct and complete"}')
    scorer = LlmJudgeScorer(backend, "judge-model", policy=_evaluator_policy(True))
    assert scorer.metric == "llm_judge"
    result = scorer.score("Explain X.", "X is Y.", "X means Y.")
    assert result.score == 91.0
    assert result.rationale == "correct and complete"
    assert backend.requests, "the judge backend should have been called"


def test_llm_judge_scorer_refuses_non_evaluator_provider():
    backend = FakeJudgeBackend('{"score": 91}')
    with pytest.raises(ProviderPolicyError):
        LlmJudgeScorer(backend, "judge-model", policy=_evaluator_policy(False))


# ---- run_evaluation wiring ---------------------------------------------------


class FakeModelBackend:
    def generate(self, request):
        return BackendGenerateResponse(text="X means Y.", model_name="fake-local")


def test_run_evaluation_with_judge_scorer_sets_metric_and_rationale():
    rows = [{"instruction": "Explain X.", "input": "", "output": "X is Y."}]
    examples = extract_evaluation_examples(rows, "instruction")
    judge = LlmJudgeScorer(
        FakeJudgeBackend('{"score": 88, "rationale": "equivalent meaning"}'),
        "judge-model",
        policy=_evaluator_policy(True),
    )

    report = run_evaluation(
        EvaluationRunConfig(dataset="d", model="fake-local", schema_id="instruction", score_threshold=70.0),
        examples,
        FakeModelBackend(),
        scorer=judge,
    )

    assert report.metric == "llm_judge"
    assert report.results[0].score == 88.0
    assert report.results[0].rationale == "equivalent meaning"
    assert report.results[0].passed is True


def test_run_evaluation_defaults_to_keyword_overlap_metric():
    rows = [{"instruction": "Explain X.", "input": "", "output": "X is Y."}]
    examples = extract_evaluation_examples(rows, "instruction")
    report = run_evaluation(
        EvaluationRunConfig(dataset="d", model="fake-local", schema_id="instruction"),
        examples,
        FakeModelBackend(),
    )
    assert report.metric == "keyword_overlap"
    assert report.results[0].rationale is None


class FlakyModelBackend:
    """Fails on the second example only, then works again."""

    def __init__(self):
        self.calls = 0

    def generate(self, request):
        from urllib.error import URLError

        self.calls += 1
        if self.calls == 2:
            raise URLError("connection reset")
        return BackendGenerateResponse(text="X is Y.", model_name="fake-local")


def test_run_evaluation_isolates_a_backend_failure_per_example():
    rows = [
        {"instruction": "Explain A.", "input": "", "output": "A is 1."},
        {"instruction": "Explain B.", "input": "", "output": "B is 2."},
        {"instruction": "Explain C.", "input": "", "output": "C is 3."},
    ]
    examples = extract_evaluation_examples(rows, "instruction")
    report = run_evaluation(
        EvaluationRunConfig(dataset="d", model="fake-local", schema_id="instruction"),
        examples,
        FlakyModelBackend(),
    )

    # Every example is accounted for; the failure did not abort the run.
    assert report.examples_tested == 3
    failed = [r for r in report.results if r.notes == "backend_error"]
    assert len(failed) == 1
    assert failed[0].error and "URLError" in failed[0].error
    assert failed[0].score == 0.0 and failed[0].passed is False
    assert failed[0].model_output == ""
    # The failure surfaces as a distinct reason in the report summary.
    assert any(s.reason == "backend_error" for s in report.failure_reason_summary)
