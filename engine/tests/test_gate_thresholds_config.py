import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.gates.models import (
    GATE_THRESHOLDS_FILENAME,
    GateThresholds,
    load_gate_thresholds,
)

runner = CliRunner()


def _write_thresholds(project: Path, data: dict) -> None:
    (project / GATE_THRESHOLDS_FILENAME).write_text(json.dumps(data), encoding="utf-8")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


# --- loader ------------------------------------------------------------------

def test_no_file_returns_defaults(tmp_path: Path):
    assert load_gate_thresholds(tmp_path) == GateThresholds()


def test_partial_override_merges_over_defaults(tmp_path: Path):
    _write_thresholds(tmp_path, {"max_regression_score_drop": 10.0})
    thresholds = load_gate_thresholds(tmp_path)
    assert thresholds.max_regression_score_drop == 10.0
    assert thresholds.min_eval_average_score == 70.0  # untouched default


def test_unknown_keys_are_ignored(tmp_path: Path):
    _write_thresholds(tmp_path, {"block_exact_duplicates": False, "bogus_key": 999})
    thresholds = load_gate_thresholds(tmp_path)
    assert thresholds.block_exact_duplicates is False
    assert not hasattr(thresholds, "bogus_key")


def test_corrupt_file_falls_back_to_defaults(tmp_path: Path):
    (tmp_path / GATE_THRESHOLDS_FILENAME).write_text("{ not json", encoding="utf-8")
    assert load_gate_thresholds(tmp_path) == GateThresholds()


def test_wrong_type_value_falls_back_to_defaults(tmp_path: Path):
    _write_thresholds(tmp_path, {"max_regression_score_drop": "not-a-number"})
    assert load_gate_thresholds(tmp_path) == GateThresholds()


def test_non_object_json_falls_back(tmp_path: Path):
    (tmp_path / GATE_THRESHOLDS_FILENAME).write_text("[1, 2, 3]", encoding="utf-8")
    assert load_gate_thresholds(tmp_path) == GateThresholds()


def test_bom_prefixed_file_is_honored(tmp_path: Path):
    # Notepad / PowerShell save UTF-8 with a BOM; the override must still apply.
    (tmp_path / GATE_THRESHOLDS_FILENAME).write_bytes(
        b"\xef\xbb\xbf" + json.dumps({"max_exact_duplicates": 3}).encode("utf-8")
    )
    assert load_gate_thresholds(tmp_path).max_exact_duplicates == 3


@pytest.mark.parametrize(
    "override",
    [
        {"max_exact_duplicates": -5},  # negative would invert the duplicate gate
        {"max_regression_score_drop": -2.0},  # negative would invert the regression gate
        {"min_eval_pass_rate": 5.0},  # a fraction > 1 makes the gate impossible to pass
        {"min_eval_average_score": float("nan")},  # NaN silently disables the score gate
    ],
)
def test_out_of_range_override_falls_back_to_defaults(tmp_path: Path, override: dict):
    # json.dumps emits bare NaN (valid to Python's json.loads); either way the
    # loader must reject a semantically-broken value and keep strict defaults.
    (tmp_path / GATE_THRESHOLDS_FILENAME).write_text(json.dumps(override), encoding="utf-8")
    assert load_gate_thresholds(tmp_path) == GateThresholds()


def test_direct_construction_rejects_bad_values():
    with pytest.raises(ValidationError):
        GateThresholds(max_exact_duplicates=-1)
    with pytest.raises(ValidationError):
        GateThresholds(max_regression_score_drop=math.nan)
    with pytest.raises(ValidationError):
        GateThresholds(min_eval_pass_rate=1.5)


# --- CLI: show + end-to-end verdict flip -------------------------------------

def test_cli_gate_thresholds_shows_effective(tmp_path: Path):
    _write_thresholds(tmp_path, {"max_low_information": 25})
    result = runner.invoke(app, ["gate-thresholds", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["max_low_information"] == 25


def test_gate_run_without_project_dir_warns_when_config_present(tmp_path: Path):
    # A config sits next to the input but --project-dir is omitted: the override
    # must NOT be applied (verdict stays at strict defaults) and a stderr note must
    # surface the ignored file so it does not fail invisibly.
    rows = tmp_path / "rows.jsonl"
    row = {"instruction": "Explain recursion clearly.", "output": "A function calls itself on subproblems."}
    _write_rows(rows, [row, row])  # exact duplicate -> block under defaults
    _write_thresholds(tmp_path, {"block_exact_duplicates": False})

    result = runner.invoke(app, ["gate-run", str(rows), "instruction"])
    assert result.exit_code == 0, result.output
    # The note goes to stderr; the JSON report stays clean on stdout (safe to redirect).
    assert json.loads(result.stdout)["overall_status"] == "block"  # override NOT applied
    assert "not applied" in result.stderr.lower()


def test_report_records_effective_thresholds(tmp_path: Path):
    rows = tmp_path / "rows.jsonl"
    row = {"instruction": "Explain recursion clearly.", "output": "A function calls itself on subproblems."}
    _write_rows(rows, [row, row])
    _write_thresholds(tmp_path, {"block_exact_duplicates": False})

    result = runner.invoke(app, ["gate-run", str(rows), "instruction", "--project-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    # The saved verdict carries the thresholds that produced it (reproducible).
    assert report["thresholds"]["block_exact_duplicates"] is False


def test_project_threshold_flips_gate_verdict(tmp_path: Path):
    # Two identical rows -> exact duplicate. With the default it BLOCKs; a project
    # override (block_exact_duplicates=false) downgrades it to a WARN.
    rows = tmp_path / "rows.jsonl"
    row = {"instruction": "Explain recursion clearly.", "output": "A function calls itself on subproblems."}
    _write_rows(rows, [row, row])

    default_gate = runner.invoke(app, ["gate-run", str(rows), "instruction"])
    assert json.loads(default_gate.output)["overall_status"] == "block"

    _write_thresholds(tmp_path, {"block_exact_duplicates": False})
    relaxed = runner.invoke(app, ["gate-run", str(rows), "instruction", "--project-dir", str(tmp_path)])
    assert json.loads(relaxed.output)["overall_status"] == "warn"


def test_project_threshold_relaxes_regression_gate(tmp_path: Path):
    from corpus_studio.evaluation.reports import EvaluationReport
    from corpus_studio.training.run_registry import TrainingRunRecord, save_run_record

    def report(model, avg):
        return EvaluationReport(dataset="d", model=model, examples_tested=10, average_score=avg, failed_examples=1)

    (tmp_path / "before.json").write_text(report("base-model", 80).model_dump_json(), encoding="utf-8")
    (tmp_path / "after.json").write_text(report("trained", 75).model_dump_json(), encoding="utf-8")
    save_run_record(tmp_path, TrainingRunRecord(
        run_id="20260702T180000-a", created_at="t", updated_at="t", status="succeeded",
        base_model="base-model", before_eval_path=str(tmp_path / "before.json"),
        after_eval_path=str(tmp_path / "after.json"), after_eval_model="trained",
    ))

    # 5-point drop blocks at the default tolerance (2.0)...
    strict = runner.invoke(app, ["training-run-gate", str(tmp_path), "--run-id", "20260702T180000-a"])
    assert json.loads(strict.output)["overall_status"] == "block"

    # ...but a project tolerance of 10.0 lets it pass.
    _write_thresholds(tmp_path, {"max_regression_score_drop": 10.0})
    relaxed = runner.invoke(app, ["training-run-gate", str(tmp_path), "--run-id", "20260702T180000-a"])
    assert json.loads(relaxed.output)["overall_status"] == "pass"
