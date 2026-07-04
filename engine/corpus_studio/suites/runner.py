"""File-driven evaluation-suite runner (v1.3 M1).

Pure orchestration over the existing ``run_evaluation`` + ``run_evaluation_gate`` —
NO new eval/scoring logic, NO registry. ``run_suite`` is pure over an injected
``evaluate_case`` (mirroring how the training-run gate takes an injected report
loader), so the roll-up / verdict / isolation are fully unit-testable offline.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from corpus_studio.evaluation.reports import EvaluationReport
from corpus_studio.gates.models import GateStatus, GateThresholds, worst_status
from corpus_studio.gates.runner import run_evaluation_gate
from corpus_studio.suites.models import (
    SuiteCase,
    SuiteCaseResult,
    SuiteCaseStatus,
    SuiteDefinition,
    SuiteMetricRollup,
    SuiteReport,
)
from corpus_studio.versions.version_registry import fingerprint_dataset

SUITE_REPORTS_DIRNAME = "suite_reports"

_DEFAULT_MIN_SCORE = 70.0
_DEFAULT_MIN_PASS_RATE = 0.5

_STATUS_FROM_GATE: dict[GateStatus, SuiteCaseStatus] = {
    GateStatus.PASS: "pass",
    GateStatus.WARN: "warn",
    GateStatus.BLOCK: "block",
}
# An errored case blocks the suite (a case we could not evaluate is not a pass).
_GATE_FROM_STATUS = {
    "pass": GateStatus.PASS,
    "warn": GateStatus.WARN,
    "block": GateStatus.BLOCK,
    "error": GateStatus.BLOCK,
}


def load_suite_definition(path: Path | str) -> SuiteDefinition:
    """Read + validate a suite JSON file. Raises ValueError for any bad definition."""

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return SuiteDefinition.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid suite definition: {exc}") from exc


def _pass_rate(report: EvaluationReport) -> float:
    if report.examples_tested <= 0:
        return 0.0
    return (report.examples_tested - report.failed_examples) / report.examples_tested


def run_suite(
    definition: SuiteDefinition,
    evaluate_case: Callable[[SuiteCase], EvaluationReport],
    generated_at: str | None = None,
) -> SuiteReport:
    """Run each case through the injected evaluator + the evaluation gate, isolating per-case
    failures, and assemble a per-metric-honest SuiteReport."""

    results: list[SuiteCaseResult] = []
    for case in definition.cases:
        # Honest record of WHAT ran (a case pins a mutable path); best-effort, never faked.
        try:
            fingerprint, _ = fingerprint_dataset(case.dataset_path)
        except Exception:  # noqa: BLE001 - fingerprint is advisory honesty, must not abort a case
            fingerprint = None

        thresholds = GateThresholds(
            min_eval_average_score=case.min_score if case.min_score is not None else _DEFAULT_MIN_SCORE,
            min_eval_pass_rate=case.min_pass_rate if case.min_pass_rate is not None else _DEFAULT_MIN_PASS_RATE,
        )

        try:
            report = evaluate_case(case)
            gate = run_evaluation_gate(report, thresholds, target=case.name)
            results.append(
                SuiteCaseResult(
                    case=case.name,
                    model=case.model,
                    metric=case.metric,
                    dataset_fingerprint=fingerprint,
                    examples_tested=report.examples_tested,
                    average_score=report.average_score,
                    pass_rate=round(_pass_rate(report), 4),
                    gate=gate,
                    status=_STATUS_FROM_GATE[gate.overall_status],
                )
            )
        except Exception as exc:  # noqa: BLE001 - per-case isolation: one dead backend can't abort the suite
            results.append(
                SuiteCaseResult(
                    case=case.name,
                    model=case.model,
                    metric=case.metric,
                    dataset_fingerprint=fingerprint,
                    error=str(exc) or type(exc).__name__,
                    status="error",
                )
            )

    return SuiteReport(
        suite=definition.name,
        generated_at=generated_at,
        cases=results,
        per_metric=_rollup(results),
        overall_status=worst_status([_GATE_FROM_STATUS[result.status] for result in results]),
        summary=_summary(definition.name, results),
    )


def _rollup(results: list[SuiteCaseResult]) -> list[SuiteMetricRollup]:
    field_for_status = {"pass": "passed", "warn": "warned", "block": "blocked", "error": "errored"}
    counts: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = counts.setdefault(result.metric, {"passed": 0, "warned": 0, "blocked": 0, "errored": 0})
        bucket[field_for_status[result.status]] += 1
    return [
        SuiteMetricRollup(metric=metric, total=sum(bucket.values()), **bucket)
        for metric, bucket in sorted(counts.items())
    ]


def _summary(name: str, results: list[SuiteCaseResult]) -> str:
    total = len(results)
    tally = {"pass": 0, "warn": 0, "block": 0, "error": 0}
    for result in results:
        tally[result.status] += 1
    parts = [f"{tally['pass']} pass"]
    if tally["warn"]:
        parts.append(f"{tally['warn']} warn")
    if tally["block"]:
        parts.append(f"{tally['block']} block")
    if tally["error"]:
        parts.append(f"{tally['error']} error")
    return f"Suite '{name}': {total} case(s) — " + ", ".join(parts) + "."


def save_suite_report(project_dir: Path | str, report: SuiteReport) -> Path:
    """Atomically write the suite report to suite_reports/<suite>.json (name is validated
    to filesystem-safe characters at definition load)."""

    directory = Path(project_dir) / SUITE_REPORTS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report.suite}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path
