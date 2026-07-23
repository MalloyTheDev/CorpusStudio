"""Tests for review-feedback (scripts/loop/review.py, controller slice 6).

Pins the CLEAN vs CHANGES_REQUESTED verdict, findings -> correction tasks appended with a dependency on
the reviewed task, idempotent re-review (no duplicate corrections), fail-closed on a bad reviewer, the
verdict -> Observation mapping, and the CHANGES_REQUESTED -> RESCHEDULE -> ASSIGN loop-back.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from loop.controller import Decision, LoopState, Observation, Phase, apply  # noqa: E402
from loop.review import (  # noqa: E402
    Finding,
    ReviewError,
    ReviewVerdict,
    review,
    review_observation,
)
from loop.tasks import decompose  # noqa: E402


def _reviewer(findings: list[Finding]):
    def run(_state: LoopState) -> list[Finding]:
        return findings
    return run


def test_clean_review_advances() -> None:
    state = LoopState()
    result = review(state, _reviewer([Finding(id="f1", summary="nit", accepted=False)]))
    assert result.verdict is ReviewVerdict.CLEAN
    assert result.correction_task_ids == [] and state.task_graph == []
    assert review_observation(result) is Observation.SUCCESS


def test_accepted_findings_become_correction_tasks_depending_on_the_reviewed_task() -> None:
    state = LoopState()
    decompose(state, [{"id": "impl", "allowed_paths": ["engine/"]}])
    result = review(state, _reviewer([
        Finding(id="bug1", summary="null deref", suggested_fix="guard None", allowed_paths=["engine/x.py"]),
        Finding(id="nit1", summary="cosmetic", accepted=False),
    ]), reviewed_task_id="impl")
    assert result.verdict is ReviewVerdict.CHANGES_REQUESTED
    assert result.correction_task_ids == ["fix-bug1"]  # only the accepted finding
    fix = next(t for t in state.task_graph if t["id"] == "fix-bug1")
    assert fix["depends_on"] == ["impl"] and fix["allowed_paths"] == ["engine/x.py"]
    assert fix["success_criteria"] == ["guard None"]
    assert review_observation(result) is Observation.CHANGES_REQUESTED


def test_re_review_is_idempotent() -> None:
    state = LoopState()
    decompose(state, [{"id": "impl"}])
    finding = Finding(id="bug1", summary="x")
    review(state, _reviewer([finding]), reviewed_task_id="impl")
    review(state, _reviewer([finding]), reviewed_task_id="impl")  # again
    assert sum(1 for t in state.task_graph if t["id"] == "fix-bug1") == 1  # not duplicated


def test_correction_is_standalone_when_reviewed_task_is_unknown() -> None:
    state = LoopState()  # empty graph
    review(state, _reviewer([Finding(id="bug1", summary="x")]), reviewed_task_id="ghost")
    fix = next(t for t in state.task_graph if t["id"] == "fix-bug1")
    assert fix["depends_on"] == []  # no dangling dependency on a non-existent task


def test_reviewer_returning_non_findings_fails_closed() -> None:
    state = LoopState()
    with pytest.raises(ReviewError):
        review(state, lambda _s: ["not a finding"])  # type: ignore[arg-type,list-item]
    with pytest.raises(ReviewError):
        review(state, lambda _s: "nope")  # type: ignore[arg-type,return-value]


def test_changes_requested_routes_back_to_assign() -> None:
    state = LoopState(current_phase=Phase.REVIEW)
    t = apply(state, Observation.CHANGES_REQUESTED, note="review requested changes")
    assert t.decision is Decision.RESCHEDULE and state.current_phase is Phase.ASSIGN
