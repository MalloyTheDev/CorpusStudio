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
    evidence_entry,
    observe,
    observe_and_apply,
    record_evidence,
)


def _step(name: str, passed: bool, exit_code: int = 0, timed_out: bool = False) -> dict:
    return {"name": name, "passed": passed, "exit_code": exit_code, "timed_out": timed_out}


def _payload(gate_passed: bool, steps: list[dict], obligations: tuple[str, ...] = (),
             workspace_stable: bool = True) -> dict:
    return {
        "gate_passed": gate_passed,
        "gate_steps": steps,
        "fired_obligations": [{"id": o} for o in obligations],
        "change_set_fingerprint": "cs:abc123",
        "workspace_stable": workspace_stable,
    }


def _record(payload: dict, digest: str = "sha256:rec") -> dict:
    """A validly-typed sealed WorkspaceVerification envelope (v2) around a payload, for observe() tests."""
    return {"record_type": "workspace_verification", "schema_version": 2,
            "record_digest": digest, "payload": payload}


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


def test_green_gate_loop_controller_self_modify_requires_authorization() -> None:
    # Hardening finding A: a change to the loop controller fires loop-controller-self-modify; the OBSERVE
    # plane must escalate it (was: it fell through to SUCCESS, letting the loop admit a change to itself).
    obs, reason = classify_observation(_payload(True, _GREEN, obligations=("loop-controller-self-modify",)))
    assert obs is Observation.AUTHORIZATION_REQUIRED and "loop-controller-self-modify" in reason


def test_observe_and_integrate_share_one_human_gated_set_pinned_to_policy() -> None:
    # The two planes must never disagree about what needs a human, and every id must be a real obligation.
    import loop.integrate as integrate_mod  # noqa: PLC0415
    import loop.observe as observe_mod  # noqa: PLC0415
    from loop.controller import HUMAN_GATED_OBLIGATIONS  # noqa: PLC0415
    assert observe_mod._HUMAN_GATED is HUMAN_GATED_OBLIGATIONS
    assert integrate_mod._HUMAN_GATED is HUMAN_GATED_OBLIGATIONS
    policy = json.loads((SCRIPTS_DIR / "assurance" / "policy" / "obligations.json").read_text())
    policy_ids = {o["id"] for o in policy["obligations"]}
    assert HUMAN_GATED_OBLIGATIONS <= policy_ids  # no human-gated id can be a typo / removed obligation


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


def test_non_bool_gate_passed_is_read_as_red() -> None:
    # A truthy non-bool gate_passed (e.g. the string "false") must NOT read as green - only True is green.
    payload = {"gate_passed": "false", "gate_steps": [_step("pytest", False, 1)], "fired_obligations": []}
    obs, _ = classify_observation(payload)
    assert obs is Observation.TEST_REGRESSION


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
    record = _record(_payload(True, _GREEN), "sha256:rec1")
    result = observe(REPO_ROOT, "main", run_cs_assure=_runner(record))
    assert result.observation is Observation.SUCCESS and result.gate_passed
    assert result.record_digest == "sha256:rec1" and result.change_set_fingerprint == "cs:abc123"


def test_observe_fails_closed_on_unparseable_verify_output() -> None:
    with pytest.raises(LoopObserveError):
        observe(REPO_ROOT, "main", run_cs_assure=_runner({}, verify_stdout="cs_assure: Refusal\n"))


def test_observe_and_apply_drives_the_controller_and_records_evidence() -> None:
    steps = [_step("ruff", True), _step("mypy", True), _step("pytest", False, 1)]
    record = _record(_payload(False, steps), "sha256:recX")
    state = LoopState(current_phase=Phase.DIAGNOSE)
    t = observe_and_apply(state, REPO_ROOT, "main", run_cs_assure=_runner(record))
    assert t.decision is Decision.REVISE and state.current_phase is Phase.EXECUTE
    assert "sha256:recX" in state.assurance_records           # sealed evidence recorded on the state
    assert len(state.failed_approaches) == 1                  # a fingerprint was formed + recorded


def test_observe_and_apply_success_advances_without_a_fingerprint() -> None:
    record = _record(_payload(True, _GREEN), "sha256:ok")
    state = LoopState(current_phase=Phase.DIAGNOSE)
    t = observe_and_apply(state, REPO_ROOT, "main", run_cs_assure=_runner(record))
    assert t.decision is Decision.ADVANCE and state.failed_approaches == []
    assert "sha256:ok" in state.assurance_records


# --------------------------------------------------------------------------- structured evidence (#6 slice 2)


def test_observe_and_apply_records_structured_evidence_for_a_green_gate() -> None:
    state = LoopState(current_phase=Phase.DIAGNOSE)
    observe_and_apply(state, REPO_ROOT, "main", run_cs_assure=_runner(_record(_payload(True, _GREEN), "sha256:g")))
    index = state.review_state["evidence"]
    assert index == [{"record_type": "workspace_verification", "predicate": "WORKSPACE_GATE_GREEN",
                      "subject_fingerprint": "cs:abc123", "digest": "sha256:g"}]


def test_a_non_bool_gate_passed_records_no_green_evidence() -> None:
    # Hardening finding C: classify reads a string "false" as RED (`is True`); the ObservationResult field
    # must AGREE (also `is True`), so a red/malformed gate can never record GATE_GREEN completion evidence.
    payload = {"gate_passed": "false", "gate_steps": _GREEN, "fired_obligations": [],
               "change_set_fingerprint": "cs:x", "workspace_stable": True}
    result = observe(REPO_ROOT, "main", run_cs_assure=_runner(_record(payload)))
    assert result.gate_passed is False               # not bool("false") == True
    assert evidence_entry(result) is None            # so NO green evidence is emitted


def test_a_red_gate_records_no_completion_evidence() -> None:
    from loop.observe import ObservationResult
    red = ObservationResult(Observation.TEST_REGRESSION, "gate red", gate_passed=False,
                            record_digest="sha256:r", change_set_fingerprint="cs:x",
                            record_type="workspace_verification")
    assert evidence_entry(red) is None  # a red gate is not completion evidence
    state = LoopState()
    record_evidence(state, red)
    assert state.review_state.get("evidence", []) == []


def test_record_evidence_dedupes_repeat_observations() -> None:
    state = LoopState(current_phase=Phase.DIAGNOSE)
    runner = _runner(_record(_payload(True, _GREEN), "sha256:g"))
    observe_and_apply(state, REPO_ROOT, "main", run_cs_assure=runner)
    observe_and_apply(state, REPO_ROOT, "main", run_cs_assure=runner)  # same evidence again
    assert len(state.review_state["evidence"]) == 1  # deduped by digest+predicate


# --------------------------------------------------------------------------- record validation (#3)


def test_observe_refuses_a_verify_exit_of_2() -> None:
    # exit >= 2 is a fail-closed refusal that emits NO record - never read leftover stdout as evidence.
    def refuse(_r: Path, *argv: str) -> tuple[int, str, str]:
        return (2, json.dumps(_record(_payload(True, _GREEN))), "GateError: ...")  # stdout ignored
    with pytest.raises(LoopObserveError, match="refused"):
        observe(REPO_ROOT, "main", run_cs_assure=refuse)


def test_observe_accepts_a_red_gate_exit_of_1() -> None:
    # exit 1 is a RED gate, which is still a VALID record (a not-clean result, not a refusal).
    steps = [_step("ruff", True), _step("mypy", True), _step("pytest", False, 1)]
    def red(_r: Path, *argv: str) -> tuple[int, str, str]:
        if argv and argv[0] == "verify":
            return (1, json.dumps(_record(_payload(False, steps))), "")
        return (0, json.dumps({"finding_count": 0}), "")
    assert observe(REPO_ROOT, "main", run_cs_assure=red).observation is Observation.TEST_REGRESSION


def test_observe_rejects_a_wrong_record_type_or_schema() -> None:
    bad_type = dict(_record(_payload(True, _GREEN)), record_type="change_set")
    with pytest.raises(LoopObserveError, match="record_type"):
        observe(REPO_ROOT, "main", run_cs_assure=_runner(bad_type))
    bad_ver = dict(_record(_payload(True, _GREEN)), schema_version=1)  # pre-workspace_stable
    with pytest.raises(LoopObserveError, match="schema_version"):
        observe(REPO_ROOT, "main", run_cs_assure=_runner(bad_ver))


def test_observe_rejects_a_mutated_workspace() -> None:
    # workspace_stable=false -> the workspace changed during the gate -> the record is not trustworthy.
    record = _record(_payload(True, _GREEN, workspace_stable=False))
    with pytest.raises(LoopObserveError, match="workspace_stable"):
        observe(REPO_ROOT, "main", run_cs_assure=_runner(record))


def test_observe_rejects_a_missing_change_set_fingerprint() -> None:
    payload = _payload(True, _GREEN)
    del payload["change_set_fingerprint"]
    with pytest.raises(LoopObserveError, match="change_set_fingerprint"):
        observe(REPO_ROOT, "main", run_cs_assure=_runner(_record(payload)))


def test_observe_surfaces_an_unavailable_doclint_not_silently_clean() -> None:
    # A doclint that could not run must NOT be read as "docs clean" - it is surfaced explicitly.
    def gh(_r: Path, *argv: str) -> tuple[int, str, str]:
        if argv and argv[0] == "verify":
            return (0, json.dumps(_record(_payload(True, _GREEN))), "")
        return (2, "boom", "doclint refused")  # doclint UNAVAILABLE
    result = observe(REPO_ROOT, "main", run_cs_assure=gh)
    assert result.observation is Observation.SUCCESS  # advisory: still advances...
    assert "UNVERIFIED" in result.reason               # ...but honestly, not "docs clean"
