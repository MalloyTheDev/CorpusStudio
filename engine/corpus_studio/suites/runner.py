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
    SuiteHistoryEntry,
    SuiteMetricRollup,
    SuiteReport,
)
from corpus_studio.versions.version_registry import fingerprint_dataset

SUITE_REPORTS_DIRNAME = "suite_reports"
SUITE_HISTORY_DIRNAME = "history"
SUITE_HISTORY_LIMIT = 200  # keep the most recent N runs per suite; older points are pruned

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
        # Honest record of WHAT ran. For a path case, fingerprint the file; for a
        # version-pinned case, the reproducibility record is version_id (echoed below) —
        # the eval runs the version's VERIFIED reconstruction.
        fingerprint: str | None = None
        if case.dataset_path:
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
                    version_id=case.version_id,
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
                    version_id=case.version_id,
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


def _is_safe_suite_name(suite_name: str) -> bool:
    """A suite name must be a single filesystem-safe segment — no separators or traversal — so a
    caller-supplied name can never point the history file outside suite_reports/history/."""
    name = suite_name.strip()
    return bool(name) and name not in {".", ".."} and name == Path(name).name


def suite_history_path(project_dir: Path | str, suite_name: str) -> Path:
    return Path(project_dir) / SUITE_REPORTS_DIRNAME / SUITE_HISTORY_DIRNAME / f"{suite_name}.jsonl"


def load_suite_history(project_dir: Path | str, suite_name: str) -> list[SuiteHistoryEntry]:
    """Load a suite's run history (oldest → newest). Missing file → empty; a corrupt line is skipped
    rather than crashing the trend."""
    if not _is_safe_suite_name(suite_name):
        return []
    path = suite_history_path(project_dir, suite_name)
    if not path.exists():
        return []
    entries: list[SuiteHistoryEntry] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entries.append(SuiteHistoryEntry.model_validate_json(stripped))
        except ValidationError:
            continue
    return entries


def append_suite_history(project_dir: Path | str, report: SuiteReport) -> Path:
    """Append this run to the suite's history (suite_reports/history/<suite>.jsonl), keeping the most
    recent :data:`SUITE_HISTORY_LIMIT` points. Atomic rewrite so a crash can't truncate the history."""
    path = suite_history_path(project_dir, report.suite)
    path.parent.mkdir(parents=True, exist_ok=True)

    entries = load_suite_history(project_dir, report.suite)
    entries.append(SuiteHistoryEntry.from_report(report))
    trimmed = entries[-SUITE_HISTORY_LIMIT:]

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(entry.model_dump_json() for entry in trimmed) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path
