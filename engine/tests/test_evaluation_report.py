import json

from corpus_studio.evaluation.evaluator import (
    EvaluationRunConfig,
    build_report_from_outputs,
    extract_evaluation_examples,
    run_evaluation,
)
from corpus_studio.evaluation.reports import EvaluationExampleResult, EvaluationReport
from corpus_studio.model_backends.base import BackendGenerateResponse


def test_evaluation_report_serializes_cleanly():
    result = EvaluationExampleResult(
        example_id="example-1",
        prompt="Explain recursion.",
        expected_output="Recursion is a function calling itself.",
        model_output="A function calls itself to solve a smaller problem.",
        score=80.0,
        passed=True,
        tags=["recursion"],
        manual_score=90.0,
        manual_notes="Human reviewer accepted this answer.",
    )
    report = EvaluationReport.from_results(
        dataset="coding_tutor_v0.1",
        model="qwen2.5-coder:7b",
        results=[result],
    )

    payload = json.loads(report.model_dump_json())

    assert payload["dataset"] == "coding_tutor_v0.1"
    assert payload["examples_tested"] == 1
    assert payload["average_score"] == 80.0
    assert payload["manually_scored_examples"] == 1
    assert payload["average_manual_score"] == 90.0
    assert payload["failed_examples"] == 0
    assert payload["tag_summary"] == [
        {
            "tag": "recursion",
            "examples": 1,
            "failed_examples": 0,
            "average_score": 80.0,
        }
    ]
    assert payload["failure_reason_summary"] == []
    assert payload["score_band_summary"] == [
        {
            "band": "70-84",
            "examples": 1,
            "failed_examples": 0,
            "average_score": 80.0,
        }
    ]
    assert payload["run_settings"] is None
    assert payload["results"][0]["manual_notes"] == "Human reviewer accepted this answer."


def test_evaluation_report_summarizes_tags_failure_reasons_and_score_bands():
    report = EvaluationReport.from_results(
        dataset="coding_tutor_v0.1",
        model="local-model",
        results=[
            EvaluationExampleResult(
                example_id="row-1",
                prompt="Explain loops.",
                expected_output="A loop repeats work.",
                model_output="A loop repeats work.",
                score=100.0,
                passed=True,
                tags=["loops"],
            ),
            EvaluationExampleResult(
                example_id="row-2",
                prompt="Explain recursion.",
                expected_output="A function calls itself.",
                model_output="Code repeats somehow.",
                score=40.0,
                passed=False,
                tags=["recursion"],
                notes="Weak explanation.",
            ),
            EvaluationExampleResult(
                example_id="row-3",
                prompt="Explain classes.",
                expected_output="A class groups data and behavior.",
                model_output="A thing.",
                score=20.0,
                passed=False,
            ),
        ],
    )

    assert report.tag_summary[0].tag == "recursion"
    assert report.tag_summary[0].examples == 1
    assert report.tag_summary[0].failed_examples == 1
    assert report.tag_summary[1].tag == "untagged"
    assert report.failure_reason_summary[0].reason == "score_below_threshold"
    assert report.failure_reason_summary[0].failed_examples == 1
    assert report.failure_reason_summary[1].reason == "Weak explanation."
    assert report.score_band_summary[0].band == "0-49"
    assert report.score_band_summary[0].examples == 2
    assert report.score_band_summary[0].failed_examples == 2
    assert report.score_band_summary[0].average_score == 30.0
    assert report.score_band_summary[1].band == "85-100"
    assert report.score_band_summary[1].examples == 1


def test_build_report_from_outputs_uses_placeholder_scoring_without_network():
    config = EvaluationRunConfig(
        dataset="coding_tutor_v0.1",
        model="local-model",
        schema_id="instruction",
        score_threshold=60.0,
        tags=["recursion"],
    )

    report = build_report_from_outputs(
        config,
        [
            (
                "Explain recursion.",
                "Recursion is a function calling itself.",
                "Recursion is when a function calls itself.",
            )
        ],
    )

    assert report.examples_tested == 1
    assert report.failed_examples == 0
    assert report.results[0].passed
    assert report.tag_summary[0].tag == "recursion"
    assert report.score_band_summary[0].band == "70-84"
    assert report.run_settings is not None
    assert report.run_settings.schema_id == "instruction"
    assert report.run_settings.score_threshold == 60.0


class FakeBackend:
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return BackendGenerateResponse(
            text="Recursion is when a function calls itself.",
            model_name="fake-local",
        )


def test_run_evaluation_uses_backend_and_scores_instruction_rows_without_real_network():
    rows = [
        {
            "instruction": "Explain recursion.",
            "input": "",
            "output": "Recursion is a function calling itself.",
            "tags": ["recursion"],
        }
    ]
    examples = extract_evaluation_examples(rows, "instruction")
    backend = FakeBackend()

    report = run_evaluation(
        EvaluationRunConfig(
            dataset="coding_tutor",
            model="fake-local",
            schema_id="instruction",
            backend="ollama",
            base_url="http://localhost:11434",
            limit=1,
            score_threshold=60.0,
            timeout_seconds=30,
        ),
        examples,
        backend,
    )

    assert report.examples_tested == 1
    assert report.failed_examples == 0
    assert report.weak_tags == []
    assert report.run_settings is not None
    assert report.run_settings.backend == "ollama"
    assert report.run_settings.base_url == "http://localhost:11434"
    assert report.run_settings.limit == 1
    assert report.run_settings.timeout_seconds == 30
    assert backend.requests[0].prompt == "Explain recursion."


def test_extract_evaluation_examples_builds_chat_request_from_last_assistant_message():
    rows = [
        {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What is recursion?"},
                {"role": "assistant", "content": "A function calling itself."},
            ]
        }
    ]

    examples = extract_evaluation_examples(rows, "chat")

    assert examples[0].expected_output == "A function calling itself."
    assert examples[0].messages == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is recursion?"},
    ]
