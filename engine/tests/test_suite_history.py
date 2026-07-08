"""Tests for suite run-history + trend tracking (issue #190)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import corpus_studio.suites.runner as suite_runner
from corpus_studio.cli import app
from corpus_studio.gates.models import GateStatus
from corpus_studio.suites.models import SuiteHistoryEntry, SuiteMetricRollup, SuiteReport
from corpus_studio.suites.runner import append_suite_history, load_suite_history

_runner = CliRunner()


def _report(status: GateStatus, passed: int, blocked: int = 0, at: str = "2026-07-08T00:00:00Z") -> SuiteReport:
    return SuiteReport(
        suite="demo",
        generated_at=at,
        overall_status=status,
        summary="s",
        per_metric=[
            SuiteMetricRollup(
                metric="keyword_overlap",
                total=passed + blocked,
                passed=passed,
                warned=0,
                blocked=blocked,
                errored=0,
            )
        ],
    )


def test_from_report_aggregates_per_metric_without_folding() -> None:
    report = SuiteReport(
        suite="d",
        overall_status=GateStatus.BLOCK,
        per_metric=[
            SuiteMetricRollup(metric="a", total=3, passed=2, warned=1, blocked=0, errored=0),
            SuiteMetricRollup(metric="b", total=2, passed=1, warned=0, blocked=1, errored=0),
        ],
    )
    entry = SuiteHistoryEntry.from_report(report)
    assert (entry.total, entry.passed, entry.warned, entry.blocked) == (5, 3, 1, 1)
    assert entry.overall_status == GateStatus.BLOCK


def test_append_then_load_accumulates_oldest_to_newest(tmp_path: Path) -> None:
    append_suite_history(tmp_path, _report(GateStatus.PASS, 5, at="t1"))
    append_suite_history(tmp_path, _report(GateStatus.BLOCK, 3, blocked=2, at="t2"))

    history = load_suite_history(tmp_path, "demo")
    assert [h.generated_at for h in history] == ["t1", "t2"]
    assert history[-1].overall_status == GateStatus.BLOCK
    assert history[-1].blocked == 2


def test_history_is_capped_to_the_most_recent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(suite_runner, "SUITE_HISTORY_LIMIT", 3)
    for i in range(5):
        append_suite_history(tmp_path, _report(GateStatus.PASS, i, at=f"t{i}"))

    history = load_suite_history(tmp_path, "demo")
    assert [h.generated_at for h in history] == ["t2", "t3", "t4"]


def test_load_history_rejects_unsafe_suite_name(tmp_path: Path) -> None:
    assert load_suite_history(tmp_path, "../../etc/passwd") == []
    assert load_suite_history(tmp_path, "missing") == []


def test_suite_history_cli_emits_json(tmp_path: Path) -> None:
    append_suite_history(tmp_path, _report(GateStatus.PASS, 4, at="t1"))
    result = _runner.invoke(app, ["suite-history", "demo", "--project-dir", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    assert '"passed": 4' in result.output
    assert '"total": 4' in result.output
