"""Gate runner: compute inputs from existing logic and assemble GateReports."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from corpus_studio.evaluation.reports import EvaluationReport
from corpus_studio.gates.basic_gates import (
    eval_score_gate,
    leakage_gate,
    pii_gate,
    quality_gate,
    schema_gate,
)
from corpus_studio.gates.models import GateReport, GateScope, GateThresholds
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
        schema_gate(validation, GateScope.DATASET),
        quality_gate(quality, thresholds, GateScope.DATASET),
        pii_gate(quality, thresholds, GateScope.DATASET),
    ]
    return GateReport.build(GateScope.DATASET, target, results, generated_at)


def run_export_gates(
    rows: list[dict[str, Any]],
    schema_id: str,
    thresholds: GateThresholds | None = None,
    target: str = "export",
    generated_at: str | None = None,
) -> GateReport:
    """Export gate: block on schema or PII failure; warn on quality issues."""

    thresholds = thresholds or GateThresholds()
    validation = _validate_rows(rows, schema_id)
    quality = build_basic_quality_report(rows)
    results = [
        schema_gate(validation, GateScope.EXPORT),
        pii_gate(quality, thresholds, GateScope.EXPORT),
        quality_gate(quality, thresholds, GateScope.EXPORT),
    ]
    return GateReport.build(GateScope.EXPORT, target, results, generated_at)


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
    )


def save_gate_report(project_dir: Path | str, report: GateReport) -> Path:
    """Write a gate report to project-local gate_reports/<scope>.json (atomic)."""

    directory = Path(project_dir) / GATE_REPORTS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report.scope.value}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_gate_report(path: Path | str) -> GateReport:
    return GateReport.model_validate_json(Path(path).read_text(encoding="utf-8"))
