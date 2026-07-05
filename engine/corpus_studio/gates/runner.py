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
    chat_structure_gate,
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


def run_chat_gates(
    rows: list[dict[str, Any]],
    schema_id: str = "chat",
    thresholds: GateThresholds | None = None,
    target: str = "chat",
    generated_at: str | None = None,
) -> GateReport:
    """Chat-suite gate: is a chat dataset structurally sound to train on? Combines input
    presence, per-message schema validation, and conversation-SEQUENCE structure. Verdicts
    structure, never semantic quality."""

    thresholds = thresholds or GateThresholds()
    validation = _validate_rows(rows, schema_id)
    results = [
        input_present_gate(len(rows), GateScope.CHAT_SUITE, block_when_empty=True),
        schema_gate(validation, GateScope.CHAT_SUITE),
        chat_structure_gate(rows, thresholds, GateScope.CHAT_SUITE),
    ]
    return GateReport.build(GateScope.CHAT_SUITE, target, results, generated_at, thresholds=thresholds)


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
    # Export blocks only on empty/schema/PII; quality counts warn regardless of the
    # project's dataset-scope block knobs (the export command has a cleaning pass).
    export_thresholds = base.model_copy(
        update={
            "block_exact_duplicates": False,
            "block_normalized_duplicates": False,
            "block_low_information": False,
        }
    )
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

    Trust boundary (honest scope): this catches STRUCTURAL / accidental mislabeling
    — a base-vs-base comparison, or a label that disagrees with the report's own
    ``model`` field. It is NOT tamper-proof. The eval reports are local, user-owned
    JSON and ``after.model`` is a self-declared field, so a user who edits their own
    report can make provenance pass. Truly binding provenance to the scored outputs
    would require the report to attest which model generated them (a signature or the
    hashed generations), which the local-first engine deliberately does not collect.
    Provenance is therefore an honesty aid against accidents, not a security control
    against a user determined to fool their own gate.
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


class PromoteBlockedError(Exception):
    """Raised when the promote gate blocks keeping an artifact. Carries the gate report."""

    def __init__(self, report: GateReport):
        self.report = report
        super().__init__("Promote gate blocked keeping this artifact.")


def promote_artifact(
    project_dir: Path,
    artifact_id: str,
    now: str = "",
    thresholds: GateThresholds | None = None,
) -> tuple[Any, GateReport]:
    """Keep (promote) an artifact ONLY when the promote gate passes — the single, authoritative
    enforcement point so every caller (CLI, desktop, script) is gated, not just the UI.

    Loads byte-exact integrity + the source run + its eval reports, runs the promote gate, and
    raises :class:`PromoteBlockedError` (carrying the report) on a BLOCK instead of writing
    ``kept``. Returns ``(updated_record, gate_report)`` on success.
    """

    from corpus_studio.training.artifact_registry import (
        artifact_content_integrity,
        artifact_path,
        load_artifact_record,
        update_artifact_status,
    )
    from corpus_studio.training.run_registry import load_run_record, record_path

    path = artifact_path(project_dir, artifact_id)
    if not path.exists():
        raise ValueError(f"No artifact '{artifact_id}'.")
    artifact = load_artifact_record(path)
    integrity = artifact_content_integrity(artifact)  # byte-exact at the enforcement point

    run = None
    run_path = record_path(project_dir, artifact.run_id)
    if run_path.exists():
        try:
            run = load_run_record(run_path)
        except Exception:  # noqa: BLE001 - a corrupt run record just means no regression context
            run = None

    def load_report(report_path: str) -> EvaluationReport | None:
        try:
            return EvaluationReport.model_validate_json(
                Path(report_path).read_text(encoding="utf-8")
            )
        except Exception:  # noqa: BLE001 - a missing/corrupt report is simply "no link"
            return None

    report = run_artifact_gate(
        artifact, integrity, run, load_report, thresholds=thresholds, generated_at=now
    )
    if report.overall_status == GateStatus.BLOCK:
        raise PromoteBlockedError(report)

    updated = update_artifact_status(project_dir, artifact_id, "kept", now=now)
    return updated, report


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
