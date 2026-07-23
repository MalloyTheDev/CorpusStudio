"""Tests for the observe/verify wiring (scripts/loop/observe.py, controller slice 2b).

Pins the MECHANICAL classifier (cs_assure evidence -> one Observation) across every mapping case, the
gate-order precedence, the human-gated/worker precedence on a green gate, fail-closed on unusable
evidence, and the observe_and_apply integration - all via an injected cs_assure runner (no slow gate).
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

from loop.controller import Decision, LoopState, Observation, Phase  # noqa: E402
from loop.observe import (  # noqa: E402
    LoopObserveError,
    classify_observation,
    observe,
    observe_and_apply,
)


def _step(name: str, passed: bool, exit_code: int = 0, timed_out: bool = False) -> dict:
    return {"name": name, "passed": passed, "exit_code": exit_code, "timed_out": timed_out}


def _payload(gate_passed: bool, steps: list[dict], obligations: tuple[str, ...] = ()) -> dict:
    return {
        "gate_passed": gate_passed,
        "gate_steps": steps,
        "fired_obligations": [{"id": o} for o in obligations],
        "change_set_fingerprint": "cs:abc123",
    }


_GREEN = [_step("ruff", True), _step("mypy", True), _step("pytest", True)]


# --------------------------------------------------------------------------- pure classifier


def test_all_green_no_obligations_is_success() -> None:
    obs, _ = classify_observation(_payload(True, _GREEN))
    assert obs is Observation.SUCCESS


def test_gate_red_maps_to_the_first_failing_step_in_order() -> None:
    cases = {
        "ruff": Observation.SYNTAX_FAILURE,
        "mypy": Observation.TYPE_FAILURE,
        "pytest": Observation.TEST_REGRESSION,
    }
    for failing, expected in cases.items():
        steps = [_step(n, n != failing, exit_code=0 if n != failing else 1) for n in ("ruff", "mypy", "pytest")]
        obs, _ = classify_observation(_payload(False, steps))
        assert obs is expected, failing
    # ruff + pytest both fail -> ruff wins (first in gate order).
    steps = [_step("ruff", False, 1), _step("mypy", True), _step("pytest", False, 1)]
    assert classify_observation(_payload(False, steps))[0] is Observation.SYNTAX_FAILURE


def test_timed_out_step_is_environment_failure_even_when_gate_red() -> None:
    steps = [_step("ruff", True), _step("mypy", True), _step("pytest", False, timed_out=True)]
    obs, reason = classify_observation(_payload(False, steps))
    assert obs is Observation.ENVIRONMENT_FAILURE and "timed out" in reason


def test_green_gate_human_gated_obligations_require_authorization() -> None:
    for ob in ("assurance-self-modify", "sealed-research"):
        obs, _ = classify_observation(_payload(True, _GREEN, obligations=(ob,)))
        assert obs is Observation.AUTHORIZATION_REQUIRED, ob


def test_green_gate_worker_closure_is_worker_lineage_impact() -> None:
    obs, _ = classify_observation(_payload(True, _GREEN, obligations=("worker-closure",)))
    assert obs is Observation.WORKER_LINEAGE_IMPACT


def test_human_gated_takes_precedence_over_worker_closure() -> None:
    obs, _ = classify_observation(_payload(True, _GREEN, obligations=("worker-closure", "sealed-research")))
    assert obs is Observation.AUTHORIZATION_REQUIRED


def test_green_gate_with_doclint_findings_is_contract_drift() -> None:
    obs, _ = classify_observation(_payload(True, _GREEN), {"finding_count": 3})
    assert obs is Observation.CONTRACT_DRIFT
    # zero / malformed finding counts do not manufacture drift.
    assert classify_observation(_payload(True, _GREEN), {"finding_count": 0})[0] is Observation.SUCCESS
    assert classify_observation(_payload(True, _GREEN), {"finding_count": "x"})[0] is Observation.SUCCESS


def test_non_dict_verify_payload_fails_closed() -> None:
    with pytest.raises(LoopObserveError):
        classify_observation(["not", "a", "dict"])


# --------------------------------------------------------------------------- runner + integration


def _runner(verify_record: dict, doclint: dict | None = None, *, verify_stdout: str | None = None):
    def run(_repo_root: Path, *argv: str) -> tuple[int, str, str]:
        if argv and argv[0] == "verify":
            return (0, verify_stdout if verify_stdout is not None else json.dumps(verify_record), "")
        if argv and argv[0] == "doclint":
            return (0, json.dumps(doclint or {"finding_count": 0}), "")
        return (2, "", "unexpected argv")
    return run


def test_observe_returns_result_with_evidence() -> None:
    record = {"record_digest": "sha256:rec1", "payload": _payload(True, _GREEN)}
    result = observe(REPO_ROOT, "main", run_cs_assure=_runner(record))
    assert result.observation is Observation.SUCCESS and result.gate_passed
    assert result.record_digest == "sha256:rec1" and result.change_set_fingerprint == "cs:abc123"


def test_observe_fails_closed_on_unparseable_verify_output() -> None:
    with pytest.raises(LoopObserveError):
        observe(REPO_ROOT, "main", run_cs_assure=_runner({}, verify_stdout="cs_assure: Refusal\n"))


def test_observe_and_apply_drives_the_controller_and_records_evidence() -> None:
    steps = [_step("ruff", True), _step("mypy", True), _step("pytest", False, 1)]
    record = {"record_digest": "sha256:recX", "payload": _payload(False, steps)}
    state = LoopState(current_phase=Phase.DIAGNOSE)
    t = observe_and_apply(state, REPO_ROOT, "main", run_cs_assure=_runner(record))
    assert t.decision is Decision.REVISE and state.current_phase is Phase.EXECUTE
    assert "sha256:recX" in state.assurance_records           # sealed evidence recorded on the state
    assert len(state.failed_approaches) == 1                  # a fingerprint was formed + recorded


def test_observe_and_apply_success_advances_without_a_fingerprint() -> None:
    record = {"record_digest": "sha256:ok", "payload": _payload(True, _GREEN)}
    state = LoopState(current_phase=Phase.DIAGNOSE)
    t = observe_and_apply(state, REPO_ROOT, "main", run_cs_assure=_runner(record))
    assert t.decision is Decision.ADVANCE and state.failed_approaches == []
    assert "sha256:ok" in state.assurance_records
