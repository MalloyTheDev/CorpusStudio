"""Tests for the integrated loop (scripts/loop/orchestrate.py, the capstone).

Drives the WHOLE loop through injected fakes and pins the corrected composition (post capstone-audit):
the INTEGRATE merge boundary (HOLD on unsettled CI, escalate self-modify/worker/dangerous/uncomputable,
merge only authorized product), multi-agent wave DRAINING, and fail-closed handling (unobservable repo ->
ESCALATED, non-Observation executor -> error, non-object cs_assure JSON -> no crash).
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
from loop.orchestrate import LoopContext, LoopOrchestrateError, run_loop, step  # noqa: E402
from loop.router import AgentResult  # noqa: E402
from loop.tasks import decompose  # noqa: E402


def _cs_assure(*, verify_green: bool = True, obligations: tuple[str, ...] = (),
               changed: tuple[str, ...] = (), impact_fail: bool = False, changeset_out: str | None = None):
    steps = [{"name": n, "passed": verify_green, "exit_code": 0 if verify_green else 1, "timed_out": False}
             for n in ("ruff", "mypy", "pytest")]
    records = {
        "verify": {"record_digest": "sha256:v", "payload": {
            "gate_passed": verify_green, "gate_steps": steps,
            "fired_obligations": [{"id": o} for o in obligations], "change_set_fingerprint": "cs:x"}},
        "changeset": {"payload": {"changed_paths": [{"path": p} for p in changed]}},
        "impact": {"payload": {"fired_obligations": [{"id": o} for o in obligations]}},
        "doclint": {"finding_count": 0},
    }

    def run(_repo: Path, *argv: str) -> tuple[int, str, str]:
        cmd = argv[0] if argv else ""
        if cmd == "impact" and impact_fail:
            return (2, "", "impact refused")
        if cmd == "changeset" and changeset_out is not None:
            return (0, changeset_out, "")
        return (0, json.dumps(records.get(cmd, {})), "")
    return run


def _executor(task_dicts: list[dict] | None = None):
    def run(state: LoopState, _directive) -> Observation:
        if state.current_phase is Phase.DECOMPOSE and task_dicts is not None:
            decompose(state, task_dicts)
        return Observation.SUCCESS
    return run


def _gh(*, ci: str = "pass", merge_ok: bool = True):
    def run(*argv: str) -> tuple[int, str, str]:
        if len(argv) >= 2 and argv[1] == "merge":
            return (0, "merged", "") if merge_ok else (1, "", "merge conflict")
        return (0, json.dumps([{"name": "pytest", "bucket": ci}]), "")
    return run


def _ctx(**kw) -> LoopContext:
    kw.setdefault("executor", _executor())
    kw.setdefault("run_cs_assure", _cs_assure())
    return LoopContext(repo_root=REPO_ROOT, **kw)


# --------------------------------------------------------------------------- full run + merge boundary


def test_full_integrated_loop_merges_and_reaches_finalize() -> None:
    state = LoopState(goal="add scorer", current_phase=Phase.RECEIVE_GOAL)
    ctx = _ctx(executor=_executor([{"id": "impl", "allowed_paths": ["engine/"]}]),
               reviewer=lambda _s: [], gh_runner=_gh(ci="pass"), pr_ref="1")
    run_loop(state, ctx)
    assert state.current_phase is Phase.FINALIZE
    assert "sha256:v" in state.assurance_records
    assert any("merged" in o.get("note", "") for o in state.observations)  # the PR was actually merged
    assert next(t for t in state.task_graph if t["id"] == "impl")["status"] == "DONE"  # #4: not left PENDING


def test_single_agent_drains_a_multi_task_graph_before_finalizing() -> None:
    # A single-agent loop with several tasks (one dependent) must finalize with EVERY task DONE - no task
    # is left PENDING when the goal finalizes (external review #4).
    state = LoopState(goal="multi", current_phase=Phase.RECEIVE_GOAL)
    ctx = _ctx(executor=_executor([{"id": "a", "allowed_paths": ["engine/"]},
                                   {"id": "b", "allowed_paths": ["scripts/"]},
                                   {"id": "c", "allowed_paths": ["docs/"], "depends_on": ["a"]}]),
               reviewer=lambda _s: [], gh_runner=_gh(ci="pass"), pr_ref="1")
    run_loop(state, ctx)
    assert state.current_phase is Phase.FINALIZE
    assert all(t["status"] == "DONE" for t in state.task_graph)  # a, b, and the dependent c all DONE


def test_integrate_holds_on_unsettled_ci_never_advancing_past_the_gate() -> None:
    state = LoopState(current_phase=Phase.INTEGRATE)
    t = step(state, _ctx(gh_runner=_gh(ci="pending"), pr_ref="1"))
    assert t.decision is Decision.HOLD and state.current_phase is Phase.INTEGRATE  # did NOT advance to VERIFY


def test_run_loop_pauses_on_hold_rather_than_stopping(tmp_path: Path) -> None:
    state = LoopState(current_phase=Phase.INTEGRATE)
    run_loop(state, _ctx(gh_runner=_gh(ci="pending"), pr_ref="1"), max_steps=5)
    assert state.current_phase is Phase.INTEGRATE and not state.is_terminal  # paused, not STOPPED


def test_integrate_dangerous_escalates_even_on_green_ci() -> None:
    state = LoopState(current_phase=Phase.INTEGRATE)
    t = step(state, _ctx(gh_runner=_gh(ci="pass"), pr_ref="1", dangerous=True))
    assert t.decision is Decision.ESCALATE and state.current_phase is Phase.ESCALATED


def test_integrate_self_modify_escalates_at_the_merge_gate() -> None:
    state = LoopState(current_phase=Phase.INTEGRATE)
    t = step(state, _ctx(gh_runner=_gh(ci="pass"), pr_ref="1",
                         run_cs_assure=_cs_assure(obligations=("assurance-self-modify",))))
    assert t.decision is Decision.ESCALATE


def test_integrate_fails_closed_when_obligations_are_uncomputable() -> None:
    # cs_assure impact refuses (exit 2) -> must NOT be read as 'no obligations' and auto-merge.
    state = LoopState(current_phase=Phase.INTEGRATE)
    t = step(state, _ctx(gh_runner=_gh(ci="pass"), pr_ref="1", run_cs_assure=_cs_assure(impact_fail=True)))
    assert t.decision is Decision.ESCALATE and state.current_phase is Phase.ESCALATED


def test_integrate_merges_an_authorized_product_change() -> None:
    state = LoopState(current_phase=Phase.INTEGRATE)
    t = step(state, _ctx(gh_runner=_gh(ci="pass", merge_ok=True), pr_ref="1",
                         run_cs_assure=_cs_assure(obligations=("contracts",))))
    assert t.decision is Decision.ADVANCE and "merged" in state.observations[-1]["note"]


def test_integrate_ci_failure_routes_to_fix() -> None:
    state = LoopState(current_phase=Phase.INTEGRATE)
    t = step(state, _ctx(gh_runner=_gh(ci="fail"), pr_ref="1"))
    assert t.decision is Decision.REVISE  # CI red -> back to EXECUTE


# --------------------------------------------------------------------------- task lifecycle


def test_multi_agent_execute_drains_the_whole_graph() -> None:
    ran: list[str] = []

    def runner(task):
        ran.append(task.id)
        return AgentResult(task.id, Observation.SUCCESS, changed_paths=[])

    state = LoopState(current_phase=Phase.EXECUTE)
    decompose(state, [{"id": "a", "allowed_paths": ["engine/"]}, {"id": "b", "allowed_paths": ["scripts/"]},
                      {"id": "c", "allowed_paths": ["docs/"], "depends_on": ["a"]}])
    t = step(state, _ctx(multi_agent=True, agent_runner=runner))
    assert set(ran) == {"a", "b", "c"}  # every task actually ran (incl. the dependent one, across waves)
    assert t.decision is Decision.ADVANCE and all(x["status"] == "DONE" for x in state.task_graph)


def test_multi_agent_stuck_graph_escalates() -> None:
    # A task whose dep FAILED can never become ready -> the drain is stuck -> escalate, not a false SUCCESS.
    def runner(task):
        obs = Observation.SUCCESS if task.id == "a" else Observation.TEST_REGRESSION
        return AgentResult(task.id, obs, changed_paths=[])

    state = LoopState(current_phase=Phase.EXECUTE)
    decompose(state, [{"id": "a", "allowed_paths": ["engine/"]},
                      {"id": "b", "allowed_paths": ["scripts/"]},
                      {"id": "c", "allowed_paths": ["docs/"], "depends_on": ["b"]}])
    t = step(state, _ctx(multi_agent=True, agent_runner=runner))
    assert t.decision is not Decision.ADVANCE  # b failed -> not a clean advance


# --------------------------------------------------------------------------- fail-closed


def test_unobservable_repo_escalates_not_crashes() -> None:
    def bad(_r: Path, *argv: str) -> tuple[int, str, str]:
        if argv and argv[0] == "verify":
            return (0, json.dumps({"record_digest": "x", "payload": "NOT-AN-OBJECT"}), "")
        return (0, "{}", "")
    state = LoopState(current_phase=Phase.OBSERVE)
    t = step(state, _ctx(run_cs_assure=bad))
    assert state.current_phase is Phase.ESCALATED and t.decision is Decision.ESCALATE


def test_non_observation_executor_fails_closed() -> None:
    state = LoopState(current_phase=Phase.RECON)
    with pytest.raises(LoopOrchestrateError):
        step(state, _ctx(executor=lambda _s, _d: "SUCCESS"))  # type: ignore[arg-type,return-value]


def test_observe_tolerates_non_object_changeset_json() -> None:
    # A non-object top-level changeset ('[]') must not crash docs-freshness (advisory -> []).
    state = LoopState(current_phase=Phase.OBSERVE)
    t = step(state, _ctx(run_cs_assure=_cs_assure(verify_green=True, changeset_out="[]")))
    assert t.decision is Decision.ADVANCE


def test_invalid_decompose_replans() -> None:
    def bad(state: LoopState, _d) -> Observation:
        state.task_graph = [{"id": "a", "depends_on": ["ghost"]}]
        return Observation.SUCCESS
    state = LoopState(current_phase=Phase.DECOMPOSE)
    assert step(state, _ctx(executor=bad)).decision is Decision.REPLAN


def test_observe_red_gate_revises() -> None:
    state = LoopState(current_phase=Phase.OBSERVE)
    assert step(state, _ctx(run_cs_assure=_cs_assure(verify_green=False))).decision is Decision.REVISE


def test_observe_flags_stale_docs_as_contract_drift() -> None:
    state = LoopState(current_phase=Phase.OBSERVE)
    step(state, _ctx(run_cs_assure=_cs_assure(verify_green=True, changed=("scripts/loop/x.py",))))
    assert state.observations[-1]["observation"] == "CONTRACT_DRIFT"


def test_step_persists_when_a_store_path_is_set(tmp_path: Path) -> None:
    from loop.store import load
    path = tmp_path / "loop.json"
    state = LoopState(current_phase=Phase.RECON)
    step(state, _ctx(store_path=path))
    assert load(path).current_phase is Phase.DEFINE_SUCCESS


# --------------------------------------------------------------------------- L8 completeness critic


def test_verify_finalizes_only_when_success_criteria_are_met() -> None:
    from loop.completeness import Criterion, CriterionKind
    # A DETERMINISTIC criterion citing a SEALED assurance record (pre-seeded here) lets VERIFY finalize.
    state = LoopState(current_phase=Phase.VERIFY, assurance_records=["sha256:proof"])
    t = step(state, _ctx(critic=lambda _s: [Criterion("c1", "scorer works",
                         kind=CriterionKind.DETERMINISTIC, met=True, evidence="sha256:proof")]))
    assert t.decision is Decision.ADVANCE and state.current_phase is Phase.FINALIZE


def test_verify_escalates_on_a_bare_model_judgment() -> None:
    # A green gate + a MODEL_JUDGMENT 'met' is not an autonomous finalize - it escalates for human authority.
    from loop.completeness import Criterion
    state = LoopState(current_phase=Phase.VERIFY)
    t = step(state, _ctx(critic=lambda _s: [Criterion("c1", "looks done", met=True)]))
    assert t.decision is Decision.ESCALATE and state.current_phase is Phase.ESCALATED


def test_verify_does_not_finalize_a_green_gate_with_unmet_criteria() -> None:
    # A green gate is not 'done' - an unmet goal criterion routes back to work the gap (self-correction).
    from loop.completeness import Criterion, CriterionKind
    state = LoopState(current_phase=Phase.VERIFY)
    t = step(state, _ctx(critic=lambda _s: [Criterion("c1", "docs written",
                         kind=CriterionKind.DETERMINISTIC, met=False)]))
    assert t.decision is not Decision.ADVANCE and state.current_phase is not Phase.FINALIZE
    assert any(task["id"] == "meet-c1" for task in state.task_graph)  # the gap became a correction task


def test_multi_agent_completeness_gap_is_executor_handled_not_delegated() -> None:
    # An unbounded completeness task (empty allowed_paths) must run via the executor, NOT a bounded agent
    # (which would breach on every edit and deadlock the gap).
    from loop.tasks import decompose
    calls: list[str] = []

    def runner(task):
        calls.append(task.id)
        return AgentResult(task.id, Observation.SUCCESS, changed_paths=[])

    state = LoopState(current_phase=Phase.EXECUTE)
    decompose(state, [{"id": "meet-c1", "allowed_paths": []}])
    t = step(state, _ctx(multi_agent=True, agent_runner=runner))
    assert calls == []  # the agent runner was NOT invoked for the unbounded task
    assert t.decision is Decision.ADVANCE
    assert next(x for x in state.task_graph if x["id"] == "meet-c1")["status"] == "DONE"


def test_one_executor_result_closes_only_one_unbounded_task() -> None:
    # Two unbounded (self-owned) completeness tasks + one executor SUCCESS must close exactly ONE - a
    # single result cannot mark several tasks DONE.
    from loop.tasks import decompose
    state = LoopState(current_phase=Phase.EXECUTE)
    decompose(state, [{"id": "meet-a", "allowed_paths": []}, {"id": "meet-b", "allowed_paths": []}])
    step(state, _ctx(multi_agent=True,
                     agent_runner=lambda t: AgentResult(t.id, Observation.SUCCESS, changed_paths=[])))
    done = [t["id"] for t in state.task_graph if t["status"] == "DONE"]
    assert done == ["meet-a"]  # exactly one, not both


def test_critic_that_raises_escalates_not_crashes() -> None:
    def boom(_s: LoopState):
        raise RuntimeError("LLM judge timed out")
    state = LoopState(current_phase=Phase.VERIFY)
    t = step(state, _ctx(critic=boom))
    assert state.current_phase is Phase.ESCALATED and t.decision is Decision.ESCALATE


def test_run_loop_seeds_and_records_the_learning_ledger(tmp_path: Path) -> None:
    from loop.completeness import Criterion, CriterionKind
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps([{"failed_approaches": ["sha256:prior-dead-end"]}]))
    state = LoopState(goal="g1", current_phase=Phase.RECEIVE_GOAL)
    # A DETERMINISTIC criterion citing the verify record ("sha256:v") the OBSERVE step seals -> it finalizes.
    run_loop(state, _ctx(executor=_executor([{"id": "impl", "allowed_paths": ["engine/"]}]),
                         reviewer=lambda _s: [], gh_runner=_gh(ci="pass"), pr_ref="1",
                         critic=lambda _s: [Criterion("c", "done", kind=CriterionKind.DETERMINISTIC,
                                                      met=True, evidence="sha256:v")], ledger_path=ledger))
    assert state.current_phase is Phase.FINALIZE  # finalized on evidence-bound completion
    assert "sha256:prior-dead-end" in state.failed_approaches  # seeded from the ledger
    assert json.loads(ledger.read_text())[-1]["goal"] == "g1"  # this goal recorded for the next
