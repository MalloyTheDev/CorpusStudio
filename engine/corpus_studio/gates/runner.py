"""Gate runner: compute inputs from existing logic and assemble GateReports."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import re

from corpus_studio.evaluation.reports import EvaluationReport
from collections.abc import Callable

from corpus_studio.gates.basic_gates import (
    artifact_integrity_gate,
    eval_score_gate,
    input_present_gate,
    leakage_gate,
    pii_gate,
    quality_gate,
    regression_gate,
    schema_gate,
)
from corpus_studio.gates.models import (
    GateReport,
    GateResult,
    GateScope,
    GateStatus,
    GateThresholds,
)
from corpus_studio.quality.basic_quality import build_basic_quality_report
from corpus_studio.splitters.leakage import detect_split_leakage
from corpus_studio.validators.basic_validator import validate_jsonl_row
from corpus_studio.validators.results import ValidationReport

GATE_REPORTS_DIRNAME = "gate_reports"


def _validate_rows(rows: list[dict[str, Any]], schema_id: str) -> ValidationReport:
    report = ValidationReport(valid=True, schema_id=schema_id)
    for row_number, row in enumerate(rows, start=1):
        report.checked_rows += 1
        report.errors.extend(validate_jsonl_row(row, schema_id, row_number))
    report.valid = len(report.errors) == 0
    return report


def run_dataset_gates(
    rows: list[dict[str, Any]],
    schema_id: str,
    thresholds: GateThresholds | None = None,
    target: str = "dataset",
    generated_at: str | None = None,
) -> GateReport:
    thresholds = thresholds or GateThresholds()
    validation = _validate_rows(rows, schema_id)
    quality = build_basic_quality_report(rows)
    results = [
        input_present_gate(len(rows), GateScope.DATASET, block_when_empty=False),
        schema_gate(validation, GateScope.DATASET),
        quality_gate(quality, thresholds, GateScope.DATASET),
        pii_gate(quality, thresholds, GateScope.DATASET),
    ]
    return GateReport.build(GateScope.DATASET, target, results, generated_at, thresholds=thresholds)


def run_export_gates(
    rows: list[dict[str, Any]],
    schema_id: str,
    thresholds: GateThresholds | None = None,
    target: str = "export",
    generated_at: str | None = None,
) -> GateReport:
    """Export gate: block on empty input, schema, or PII failure; warn on quality.

    Quality issues (duplicates, low-information) warn rather than block on export
    because the export command has a dedicated cleaning pass.
    """

    base = thresholds or GateThresholds()
    export_thresholds = base.model_copy(update={"block_exact_duplicates": False})
    validation = _validate_rows(rows, schema_id)
    quality = build_basic_quality_report(rows)
    results = [
        input_present_gate(len(rows), GateScope.EXPORT, block_when_empty=True),
        schema_gate(validation, GateScope.EXPORT),
        pii_gate(quality, export_thresholds, GateScope.EXPORT),
        quality_gate(quality, export_thresholds, GateScope.EXPORT),
    ]
    return GateReport.build(GateScope.EXPORT, target, results, generated_at, thresholds=export_thresholds)


def run_split_gate(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    target: str = "split",
    generated_at: str | None = None,
) -> GateReport:
    leakage = detect_split_leakage(train, validation, test)
    return GateReport.build(
        GateScope.SPLIT, target, [leakage_gate(leakage, GateScope.SPLIT)], generated_at
    )


def run_evaluation_gate(
    report: EvaluationReport,
    thresholds: GateThresholds | None = None,
    target: str = "evaluation_report",
    generated_at: str | None = None,
) -> GateReport:
    thresholds = thresholds or GateThresholds()
    return GateReport.build(
        GateScope.EVALUATION_REPORT,
        target,
        [eval_score_gate(report, thresholds, GateScope.EVALUATION_REPORT)],
        generated_at,
        thresholds=thresholds,
    )


def _slug(text: str) -> str:
    """Filesystem-safe discriminator from a target (usually a file path)."""

    base = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(text).name).strip("_")
    return base or "target"


def run_training_run_gate(
    record: Any,
    load_report: Callable[[str], EvaluationReport | None],
    thresholds: GateThresholds | None = None,
    generated_at: str | None = None,
) -> GateReport:
    """Regression-gate a training run using its linked before/after eval reports.

    ``record`` is a TrainingRunRecord (duck-typed: before_eval_path,
    after_eval_path, after_eval_model, base_model, run_id). Provenance holds only
    when the after-eval declared a model that is not the base model.
    """

    thresholds = thresholds or GateThresholds()
    before, after, provenance_ok = _regression_inputs(record, load_report)
    result = regression_gate(before, after, thresholds, provenance_ok, GateScope.TRAINING_RUN)
    return GateReport.build(
        GateScope.TRAINING_RUN, record.run_id, [result], generated_at, thresholds=thresholds
    )


def _regression_inputs(
    run: Any, load_report: Callable[[str], EvaluationReport | None]
) -> tuple[EvaluationReport | None, EvaluationReport | None, bool]:
    """Resolve (before, after, provenance_ok) for a run's linked eval reports.

    Provenance holds only when the after-eval declared a model that is not the
    base model AND the recorded ``after_eval_model`` label matches the linked
    report's own ``model`` — so a base-vs-base comparison, or an after-eval report
    that actually evaluated a different model than the label claims, is not trusted.
    """

    before = load_report(run.before_eval_path) if run.before_eval_path else None
    after = load_report(run.after_eval_path) if run.after_eval_path else None
    provenance_ok = True
    if after is not None:
        if not run.after_eval_model:
            provenance_ok = False  # after-eval target not recorded
        elif not run.base_model:
            provenance_ok = False  # base model unrecorded -> can't verify the target
        elif run.after_eval_model == run.base_model:
            provenance_ok = False  # after-eval targeted the base model
        elif run.after_eval_model != after.model:
            # The operator-supplied label disagrees with the model the linked report
            # actually evaluated — the "trained-vs-base" claim cannot be trusted (a
            # base-vs-base eval could be relabeled as the trained model to spoof a pass).
            provenance_ok = False
    return before, after, provenance_ok


def run_artifact_gate(
    artifact: Any,
    integrity: str,
    run: Any,
    load_report: Callable[[str], EvaluationReport | None],
    thresholds: GateThresholds | None = None,
    generated_at: str | None = None,
) -> GateReport:
    """Promote gate for a model artifact (the enforcement point for 'keep').

    Blocks when the artifact integrity is missing/modified OR the source run
    regressed; warns on unverified linkage / a missing source run.
    """

    thresholds = thresholds or GateThresholds()
    results = [artifact_integrity_gate(integrity, GateScope.MODEL_ARTIFACT)]

    if run is None:
        results.append(
            GateResult(
                gate_id="regression",
                name="Training regression",
                scope=GateScope.MODEL_ARTIFACT,
                status=GateStatus.WARN,
                observed="source run record not found",
                expected="a source run with before/after evaluations",
                message="Cannot assess regression: the artifact's source run record is missing.",
                repair="Keep the source training run record so promotion can be judged.",
            )
        )
    else:
        before, after, provenance_ok = _regression_inputs(run, load_report)
        results.append(
            regression_gate(before, after, thresholds, provenance_ok, GateScope.MODEL_ARTIFACT)
        )

    return GateReport.build(
        GateScope.MODEL_ARTIFACT, artifact.artifact_id, results, generated_at, thresholds=thresholds
    )


def save_gate_report(project_dir: Path | str, report: GateReport) -> Path:
    """Write a gate report to gate_reports/<scope>-<target>.json (atomic).

    The target is part of the filename so gating different files in the same
    scope does not silently clobber earlier reports.
    """

    directory = Path(project_dir) / GATE_REPORTS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report.scope.value}-{_slug(report.target)}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_gate_report(path: Path | str) -> GateReport:
    return GateReport.model_validate_json(Path(path).read_text(encoding="utf-8"))
