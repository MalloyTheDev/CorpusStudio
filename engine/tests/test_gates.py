from pathlib import Path

from corpus_studio.evaluation.reports import EvaluationExampleResult, EvaluationReport
from corpus_studio.gates.models import GateReport, GateScope, GateStatus, GateThresholds
from corpus_studio.gates.runner import (
    load_gate_report,
    run_dataset_gates,
    run_evaluation_gate,
    run_export_gates,
    run_split_gate,
    save_gate_report,
)

CLEAN_ROWS = [
    {"instruction": "Explain recursion in detail.", "output": "Recursion is when a function calls itself to solve subproblems."},
    {"instruction": "Describe binary search clearly.", "output": "Binary search halves a sorted range each step to find a value."},
]


def test_dataset_gates_pass_on_clean_rows():
    report = run_dataset_gates(CLEAN_ROWS, "instruction")
    assert report.overall_status == GateStatus.PASS
    assert report.block_count == 0
    assert {result.gate_id for result in report.results} == {
        "input_present",
        "schema",
        "quality",
        "pii",
    }


def test_empty_dataset_warns_not_silently_passes():
    report = run_dataset_gates([], "instruction")
    assert report.overall_status == GateStatus.WARN
    present = next(r for r in report.results if r.gate_id == "input_present")
    assert present.status == GateStatus.WARN


def test_empty_export_blocks():
    report = run_export_gates([], "instruction")
    assert report.overall_status == GateStatus.BLOCK


def test_export_warns_not_blocks_on_exact_duplicates():
    row = {"instruction": "Explain recursion in detail.", "output": "A function calls itself on subproblems."}
    report = run_export_gates([row, row], "instruction")
    quality = next(r for r in report.results if r.gate_id == "quality")
    assert quality.status == GateStatus.WARN
    assert report.overall_status != GateStatus.BLOCK


def test_schema_gate_counts_rows_not_errors():
    # One empty row against 'instruction' yields 2 errors but is 1 failing row.
    report = run_dataset_gates([{}], "instruction")
    schema = next(r for r in report.results if r.gate_id == "schema")
    assert "1 row(s) fail" in schema.message


def test_schema_gate_blocks_invalid_rows():
    rows = [{"instruction": "missing output"}]  # required 'output' missing
    report = run_dataset_gates(rows, "instruction")
    schema_result = next(r for r in report.results if r.gate_id == "schema")
    assert schema_result.status == GateStatus.BLOCK
    assert report.overall_status == GateStatus.BLOCK


def test_quality_gate_blocks_exact_duplicates():
    row = {"instruction": "Explain recursion in detail.", "output": "A function calls itself to solve subproblems."}
    report = run_dataset_gates([row, row], "instruction")
    quality = next(r for r in report.results if r.gate_id == "quality")
    assert quality.status == GateStatus.BLOCK
    assert "exact duplicate" in quality.observed


def test_pii_gate_blocks_on_secret():
    rows = [
        {
            "instruction": "Use this config",
            "output": "set AWS_KEY=AKIAIOSFODNN7EXAMPLE in your environment now",
        }
    ]
    report = run_dataset_gates(rows, "instruction")
    pii = next(r for r in report.results if r.gate_id == "pii")
    assert pii.status == GateStatus.BLOCK
    assert report.blocked is True


def test_pii_gate_warns_on_email_only():
    rows = [
        {"instruction": "Contact form", "output": "Reach the maintainer at someone@example.com for help."},
    ]
    report = run_dataset_gates(rows, "instruction")
    pii = next(r for r in report.results if r.gate_id == "pii")
    assert pii.status == GateStatus.WARN


def test_leakage_gate_blocks_when_rows_span_splits():
    shared = {"instruction": "Shared row appears twice.", "output": "It leaks across splits and inflates eval."}
    report = run_split_gate([shared], [shared], [])
    leakage = report.results[0]
    assert leakage.status == GateStatus.BLOCK
    assert report.overall_status == GateStatus.BLOCK


def test_leakage_gate_passes_when_splits_disjoint():
    report = run_split_gate([CLEAN_ROWS[0]], [CLEAN_ROWS[1]], [])
    assert report.overall_status == GateStatus.PASS


def test_export_gate_blocks_on_pii():
    rows = [{"instruction": "leak", "output": "token sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ABCD here"}]
    report = run_export_gates(rows, "instruction")
    assert report.scope == GateScope.EXPORT
    assert report.blocked is True


def _eval_report(avg: float, failed: int, tested: int) -> EvaluationReport:
    results = [
        EvaluationExampleResult(
            example_id=f"e{i}",
            prompt="p",
            expected_output="e",
            model_output="m",
            score=avg,
            passed=i >= failed,
        )
        for i in range(tested)
    ]
    return EvaluationReport(
        dataset="d",
        model="m",
        examples_tested=tested,
        average_score=avg,
        failed_examples=failed,
        results=results,
    )


def test_eval_gate_blocks_below_threshold():
    report = run_evaluation_gate(_eval_report(avg=40.0, failed=8, tested=10))
    assert report.overall_status == GateStatus.BLOCK


def test_eval_gate_passes_above_threshold():
    report = run_evaluation_gate(_eval_report(avg=85.0, failed=1, tested=10))
    assert report.overall_status == GateStatus.PASS


def test_custom_thresholds_relax_quality_gate():
    row = {"instruction": "Explain recursion in detail.", "output": "A function calls itself."}
    lenient = GateThresholds(max_exact_duplicates=5)
    report = run_dataset_gates([row, row], "instruction", thresholds=lenient)
    quality = next(r for r in report.results if r.gate_id == "quality")
    assert quality.status != GateStatus.BLOCK


def test_gate_report_serializes_and_reloads(tmp_path: Path):
    report = run_dataset_gates(CLEAN_ROWS, "instruction", generated_at="2026-07-02T00:00:00Z")
    path = save_gate_report(tmp_path, report)
    assert path.name.startswith("dataset-")  # scope + target discriminator

    reloaded = load_gate_report(path)
    assert reloaded.scope == GateScope.DATASET
    assert reloaded.generated_at == "2026-07-02T00:00:00Z"
    assert reloaded.overall_status == report.overall_status
    assert [r.gate_id for r in reloaded.results] == [r.gate_id for r in report.results]
