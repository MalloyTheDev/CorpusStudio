"""Tests for the loop driver (scripts/loop/driver.py, controller slice 3 - completes L4).

Pins the action emitter, one-cycle advance (executor phases vs the cs_assure verify phases), a full
happy-path run to FINALIZE, a bounded failure run that STOPs, the hard step cap (independent of the
attempt budget), and per-cycle persistence - all via injected executor/observer callbacks (no real gate).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from loop.controller import LoopState, Observation, Phase, Transition, apply  # noqa: E402
from loop.driver import (  # noqa: E402
    Directive,
    LoopDriverError,
    advance,
    next_directive,
    run,
)
from loop.store import load  # noqa: E402


def _executor(observation: Observation):
    def run_executor(_directive: Directive) -> Observation:
        return observation
    return run_executor


def _observer(observation: Observation):
    def observe(state: LoopState, _repo: Path, _base: str) -> Transition:
        return apply(state, observation, note="fake-observer")
    return observe


# --------------------------------------------------------------------------- action emitter


def test_next_directive_reports_phase_action_and_budget() -> None:
    state = LoopState(current_phase=Phase.RECON, success_criteria=["gate green"],
                      budgets={"total_attempts": 3, "max_attempts": 20})
    d = next_directive(state)
    assert d.phase == "RECON" and not d.terminal and "Inspect" in d.action
    assert d.budget_remaining == 17 and d.success_criteria == ["gate green"]


def test_next_directive_is_terminal_marked_when_state_terminal() -> None:
    state = LoopState(current_phase=Phase.FINALIZE, termination_reason="done")
    d = next_directive(state)
    assert d.terminal and d.termination_reason == "done"


def test_next_directive_surfaces_active_task_allowed_paths() -> None:
    state = LoopState(current_phase=Phase.EXECUTE,
                      task_graph=[{"status": "active", "allowed_paths": ["scripts/loop/", "x.py"]}])
    assert next_directive(state).allowed_paths == ["scripts/loop/", "x.py"]


# --------------------------------------------------------------------------- one cycle


def test_advance_uses_executor_at_a_judgment_phase() -> None:
    state = LoopState(current_phase=Phase.RECON)
    t = advance(state, REPO_ROOT, _executor(Observation.SUCCESS), observer=_observer(Observation.SUCCESS))
    assert t is not None and state.current_phase is Phase.DEFINE_SUCCESS  # advanced along the spine


def test_advance_uses_the_observer_at_verify_phases() -> None:
    # At OBSERVE the loop reads the repo itself - the executor is NOT consulted (it would raise here).
    def boom(_d: Directive) -> Observation:
        raise AssertionError("executor must not be called at a verify phase")

    state = LoopState(current_phase=Phase.OBSERVE)
    advance(state, REPO_ROOT, boom, observer=_observer(Observation.SUCCESS))
    assert state.current_phase is Phase.DIAGNOSE


def test_advance_on_terminal_state_is_a_noop() -> None:
    state = LoopState(current_phase=Phase.STOPPED, termination_reason="stopped")
    assert advance(state, REPO_ROOT, _executor(Observation.SUCCESS), observer=_observer(Observation.SUCCESS)) is None


def test_advance_rejects_a_non_observation_executor_result() -> None:
    state = LoopState(current_phase=Phase.RECON)
    with pytest.raises(LoopDriverError):
        advance(state, REPO_ROOT, lambda _d: "SUCCESS", observer=_observer(Observation.SUCCESS))  # type: ignore[arg-type,return-value]


def test_advance_persists_when_a_store_path_is_given(tmp_path: Path) -> None:
    state = LoopState(current_phase=Phase.RECON)
    path = tmp_path / "loop.json"
    advance(state, REPO_ROOT, _executor(Observation.SUCCESS), observer=_observer(Observation.SUCCESS),
            store_path=path)
    assert load(path).current_phase is Phase.DEFINE_SUCCESS  # the cycle was persisted


# --------------------------------------------------------------------------- full runs


def test_run_drives_a_clean_goal_to_finalize() -> None:
    state = LoopState(goal="ship it", current_phase=Phase.RECEIVE_GOAL)
    run(state, REPO_ROOT, _executor(Observation.SUCCESS), observer=_observer(Observation.SUCCESS))
    assert state.current_phase is Phase.FINALIZE and state.termination_reason == "completion criteria satisfied"


def test_run_stops_when_the_attempt_budget_is_exhausted() -> None:
    # A gate that keeps failing (no fingerprint) is bounded by the attempt budget -> STOPPED, not forever.
    state = LoopState(current_phase=Phase.EXECUTE, budgets={"total_attempts": 0, "max_attempts": 3})
    run(state, REPO_ROOT, _executor(Observation.SUCCESS), observer=_observer(Observation.TEST_REGRESSION))
    assert state.current_phase is Phase.STOPPED


def test_run_hard_step_cap_stops_independently_of_the_budget() -> None:
    # Even with a huge budget, the driver's hard step cap prevents an unbounded spin.
    state = LoopState(current_phase=Phase.EXECUTE, budgets={"total_attempts": 0, "max_attempts": 100000})
    run(state, REPO_ROOT, _executor(Observation.SUCCESS), observer=_observer(Observation.TEST_REGRESSION),
        max_steps=5)
    assert state.current_phase is Phase.STOPPED and "hard step cap" in (state.termination_reason or "")


def test_run_escalates_to_a_human_on_authorization_required() -> None:
    # A human-gated observation at a verify phase ends the ordinary loop at ESCALATED.
    state = LoopState(current_phase=Phase.OBSERVE)
    run(state, REPO_ROOT, _executor(Observation.SUCCESS),
        observer=_observer(Observation.AUTHORIZATION_REQUIRED))
    assert state.current_phase is Phase.ESCALATED
