"""Tests for L8 self-correction (scripts/loop/completeness.py).

Pins the completeness critic (all-met -> complete; unmet -> gaps; empty -> unproven; fail-closed), the
verdict -> Observation mapping, gaps -> correction tasks, and the cross-goal learning ledger (seed prior
dead ends, record outcome, round-trip, fail-closed on a malformed ledger).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from loop.completeness import (  # noqa: E402
    CompletenessError,
    Criterion,
    check_completeness,
    completeness_correction_tasks,
    completeness_observation,
    record_outcome,
    seed_known_dead_ends,
)
from loop.controller import LoopState, Observation, Phase  # noqa: E402
from loop.tasks import parse_tasks  # noqa: E402


def _critic(criteria: list[Criterion]):
    return lambda _state: criteria


# --------------------------------------------------------------------------- the critic


def test_all_criteria_met_is_complete() -> None:
    verdict = check_completeness(LoopState(), _critic([Criterion("c1", "scorer registered", met=True),
                                                       Criterion("c2", "gate green", met=True)]))
    assert verdict.complete and verdict.unmet == []
    assert completeness_observation(verdict) is Observation.SUCCESS


def test_unmet_criteria_is_incomplete_and_folds_to_gaps() -> None:
    verdict = check_completeness(LoopState(), _critic([Criterion("c1", "registered", met=True),
                                                       Criterion("c2", "docs written", met=False)]))
    assert not verdict.complete and [c.id for c in verdict.unmet] == ["c2"]
    assert completeness_observation(verdict) is Observation.CHANGES_REQUESTED
    tasks = parse_tasks(completeness_correction_tasks(verdict))  # valid task graph
    assert tasks[0].id == "meet-c2" and tasks[0].success_criteria == ["docs written"]


def test_no_criteria_is_unproven_not_complete() -> None:
    # A goal with nothing to check must NOT be declared complete (a green gate is not 'done').
    verdict = check_completeness(LoopState(), _critic([]))
    assert not verdict.complete and "unproven" in verdict.note


def test_critic_returning_non_criteria_fails_closed() -> None:
    with pytest.raises(CompletenessError):
        check_completeness(LoopState(), lambda _s: ["not a criterion"])  # type: ignore[arg-type,list-item]


# --------------------------------------------------------------------------- cross-goal ledger


def test_seed_loads_prior_dead_ends(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps([{"goal": "g0", "failed_approaches": ["sha256:x", "sha256:y"]}]))
    state = LoopState(goal="g1")
    assert seed_known_dead_ends(state, ledger) == 2
    assert state.failed_approaches == ["sha256:x", "sha256:y"]  # g1 starts knowing g0's dead ends


def test_seed_on_missing_ledger_is_a_noop(tmp_path: Path) -> None:
    assert seed_known_dead_ends(LoopState(), tmp_path / "nope.json") == 0


def test_record_then_seed_round_trips(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    done = LoopState(goal="g1", goal_id="id1", current_phase=Phase.FINALIZE,
                     failed_approaches=["sha256:deadend"])
    record_outcome(done, ledger, lessons=["prefer chunked CE"])
    assert list(ledger.parent.glob("*.tmp-*")) == []  # atomic, no temp leak
    entry = json.loads(ledger.read_text())[0]
    assert entry["goal"] == "g1" and entry["outcome"] == "FINALIZE" and entry["lessons"] == ["prefer chunked CE"]
    # a fresh goal seeds from it
    fresh = LoopState(goal="g2")
    assert seed_known_dead_ends(fresh, ledger) == 1 and fresh.failed_approaches == ["sha256:deadend"]


def test_malformed_ledger_fails_closed(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{not json")
    with pytest.raises(CompletenessError):
        seed_known_dead_ends(LoopState(), ledger)
    ledger.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(CompletenessError):
        seed_known_dead_ends(LoopState(), ledger)
