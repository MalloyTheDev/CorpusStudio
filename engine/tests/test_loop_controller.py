"""Tests for the bounded autonomous-loop controller kernel (scripts/loop/controller.py, L3 -> L4 s1).

These pin the routing brain: each failure class routes to its own decision, the loop is BOUNDED (a
budget cap and a repeated-failure guard), it FAILS CLOSED (an unrouted observation escalates), and
human-gated observations escalate rather than proceed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from loop.controller import (  # noqa: E402
    _ROUTE,
    Decision,
    LoopState,
    Observation,
    Phase,
    apply,
    attempt_fingerprint,
    route,
)


def test_route_table_covers_every_observation() -> None:
    # Completeness: every taxonomy member has an explicit route, so the fail-closed default is only a
    # safety net for a future member added without a route (never a silent advance).
    assert set(_ROUTE) == set(Observation)


def test_success_advances_along_the_pipeline() -> None:
    state = LoopState(current_phase=Phase.RECON)
    t = route(state, Observation.SUCCESS)
    assert t.decision is Decision.ADVANCE and t.next_phase is Phase.DEFINE_SUCCESS


def test_success_at_verify_finalizes() -> None:
    state = LoopState(current_phase=Phase.VERIFY)
    t = route(state, Observation.SUCCESS)
    assert t.next_phase is Phase.FINALIZE and t.termination_reason == "completion criteria satisfied"


def test_each_failure_routes_to_its_own_decision() -> None:
    cases = {
        Observation.TEST_REGRESSION: Decision.REVISE,
        Observation.TYPE_FAILURE: Decision.REVISE,
        Observation.CONTRACT_DRIFT: Decision.REVISE,
        Observation.WRONG_PLAN: Decision.REPLAN,
        Observation.OWNERSHIP_COLLISION: Decision.RESCHEDULE,
        Observation.ENVIRONMENT_FAILURE: Decision.ESCALATE,
        Observation.POLICY_BLOCK: Decision.ESCALATE,
        Observation.AUTHORIZATION_REQUIRED: Decision.ESCALATE,
        Observation.WORKER_LINEAGE_IMPACT: Decision.ENTER_WORKER_WORKFLOW,
    }
    for observation, expected in cases.items():
        state = LoopState(current_phase=Phase.DIAGNOSE)
        assert route(state, observation).decision is expected, observation


def test_authorization_required_escalates_not_proceeds() -> None:
    # Human approval is RETAINED: a credential/dangerous/irreversible signal escalates immediately.
    state = LoopState(current_phase=Phase.EXECUTE)
    t = route(state, Observation.AUTHORIZATION_REQUIRED)
    assert t.decision is Decision.ESCALATE and t.next_phase is Phase.ESCALATED


def test_worker_lineage_leaves_the_ordinary_loop() -> None:
    state = LoopState(current_phase=Phase.DIAGNOSE)
    t = route(state, Observation.WORKER_LINEAGE_IMPACT)
    assert t.decision is Decision.ENTER_WORKER_WORKFLOW and t.next_phase is Phase.ESCALATED


def test_budget_exhaustion_stops_instead_of_looping() -> None:
    state = LoopState(current_phase=Phase.DIAGNOSE, budgets={"total_attempts": 5, "max_attempts": 5})
    t = route(state, Observation.TEST_REGRESSION)  # would REVISE, but the budget is spent
    assert t.decision is Decision.STOP and t.next_phase is Phase.STOPPED


def test_repeated_failed_approach_is_not_retried() -> None:
    fp = attempt_fingerprint("pytest::test_x FAILED assert 1==2", "patch: bump timeout")
    state = LoopState(current_phase=Phase.DIAGNOSE, failed_approaches=[fp])
    # Same failure + same patch -> do NOT retry; demand a new approach.
    t = route(state, Observation.TEST_REGRESSION, fingerprint=fp)
    assert t.decision is Decision.ESCALATE and "new approach" in t.note
    # A DIFFERENT patch for the same failure is still allowed to try.
    fresh = attempt_fingerprint("pytest::test_x FAILED assert 1==2", "patch: fix the off-by-one")
    assert route(state, Observation.TEST_REGRESSION, fingerprint=fresh).decision is Decision.REVISE


def test_apply_mutates_state_charges_budget_and_records_deadend() -> None:
    fp = attempt_fingerprint("mypy: incompatible type", "patch: add cast")
    state = LoopState(current_phase=Phase.DIAGNOSE)
    t = apply(state, Observation.TYPE_FAILURE, fingerprint=fp, note="tried a cast")
    assert t.decision is Decision.REVISE and state.current_phase is Phase.EXECUTE
    assert state.budgets["total_attempts"] == 1          # a re-entering decision charges the budget
    assert fp in state.failed_approaches                 # the failed approach is remembered
    assert state.observations[-1]["observation"] == "TYPE_FAILURE"
    assert state.observations[-1]["note"] == "tried a cast"


def test_success_does_not_charge_budget_or_record_deadend() -> None:
    state = LoopState(current_phase=Phase.RECON)
    apply(state, Observation.SUCCESS, fingerprint="sha256:whatever")
    assert state.budgets["total_attempts"] == 0 and state.failed_approaches == []


def test_terminal_phase_is_sticky() -> None:
    state = LoopState(current_phase=Phase.FINALIZE, termination_reason="done")
    t = route(state, Observation.TEST_REGRESSION)
    assert t.decision is Decision.STOP and state.current_phase is Phase.FINALIZE


def test_attempt_fingerprint_is_deterministic_and_prefixed() -> None:
    a = attempt_fingerprint("f", "p")
    assert a == attempt_fingerprint("f", "p") and a.startswith("sha256:")
    assert a != attempt_fingerprint("f", "p2")


# --------------------------------------------------------------------------- audit hardening (2026-07-23)


def test_degenerate_budget_fails_closed_not_unbounded() -> None:
    # A missing / zero / garbage max_attempts must STOP (fail-closed), never disable the cap and loop
    # forever. Regression for the audit's headline fail-open finding.
    for budgets in ({}, {"max_attempts": 0}, {"total_attempts": 3}, {"max_attempts": "oops"},
                    {"max_attempts": -1}):
        state = LoopState(current_phase=Phase.DIAGNOSE, budgets=dict(budgets))
        t = route(state, Observation.TEST_REGRESSION)
        assert t.decision is Decision.STOP and t.next_phase is Phase.STOPPED, budgets


def test_repeated_approach_guard_covers_replan_and_reschedule() -> None:
    # The dead-end guard is not REVISE-only: a repeated (failure, approach) fingerprint on
    # REPLAN/RESCHEDULE escalates too, matching the module's documented BOUNDED invariant.
    for obs in (Observation.WRONG_PLAN, Observation.OWNERSHIP_COLLISION, Observation.TEST_REGRESSION):
        fp = attempt_fingerprint("same failure", "same approach")
        state = LoopState(current_phase=Phase.DIAGNOSE, failed_approaches=[fp])
        assert route(state, obs, fingerprint=fp).decision is Decision.ESCALATE, obs


def test_replan_records_fingerprint_so_an_identical_repeat_escalates() -> None:
    # apply() records re-entering fingerprints (aligned with the guard) so a second identical REPLAN is
    # caught instead of grinding to the budget - and recording is scoped to re-entering decisions.
    fp = attempt_fingerprint("wrong plan X", "same replan")
    state = LoopState(current_phase=Phase.DIAGNOSE)
    assert apply(state, Observation.WRONG_PLAN, fingerprint=fp).decision is Decision.REPLAN
    assert fp in state.failed_approaches
    state.current_phase = Phase.DIAGNOSE  # the executor replanned identically; back at diagnose
    assert apply(state, Observation.WRONG_PLAN, fingerprint=fp).decision is Decision.ESCALATE


def test_apply_is_idempotent_on_a_terminal_state() -> None:
    # Feeding observations to a FINALIZED loop must not keep growing the durable lists (the entering
    # transition still commits; replays are true no-ops).
    state = LoopState(current_phase=Phase.VERIFY)
    apply(state, Observation.SUCCESS)  # VERIFY -> FINALIZE, records the entering transition
    assert state.current_phase is Phase.FINALIZE
    n_obs = len(state.observations)
    for _ in range(5):
        apply(state, Observation.TEST_REGRESSION, fingerprint="sha256:x")
    assert len(state.observations) == n_obs and state.failed_approaches == []
    assert state.current_phase is Phase.FINALIZE
