"""Tests for the integrated loop (scripts/loop/orchestrate.py, the capstone).

Drives the WHOLE loop through injected fakes (executor / reviewer / agent runner / gh / cs_assure) and
pins that each phase dispatches to its module: decompose+validate, assign, execute (single + wave),
observe (cs_assure + docs-freshness + task-close), review, integrate (CI + merge gate), verify.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from loop.controller import Decision, LoopState, Observation, Phase  # noqa: E402
from loop.orchestrate import LoopContext, run_loop, step  # noqa: E402
from loop.router import AgentResult  # noqa: E402
from loop.tasks import decompose  # noqa: E402


def _cs_assure(*, verify_green: bool = True, obligations: tuple[str, ...] = (), changed: tuple[str, ...] = ()):
    steps = [{"name": n, "passed": verify_green, "exit_code": 0 if verify_green else 1, "timed_out": False}
             for n in ("ruff", "mypy", "pytest")]
    verify = {"record_digest": "sha256:v", "payload": {
        "gate_passed": verify_green, "gate_steps": steps,
        "fired_obligations": [{"id": o} for o in obligations], "change_set_fingerprint": "cs:x"}}
    changeset = {"payload": {"changed_paths": [{"path": p} for p in changed]}}
    impact = {"payload": {"fired_obligations": [{"id": o} for o in obligations]}}

    def run(_repo: Path, *argv: str) -> tuple[int, str, str]:
        cmd = argv[0] if argv else ""
        return (0, json.dumps({"verify": verify, "changeset": changeset, "impact": impact,
                               "doclint": {"finding_count": 0}}.get(cmd, {})), "")
    return run


def _executor(task_dicts: list[dict] | None = None):
    def run(state: LoopState, _directive) -> Observation:
        if state.current_phase is Phase.DECOMPOSE and task_dicts is not None:
            decompose(state, task_dicts)
        return Observation.SUCCESS
    return run


def _gh(green: bool = True):
    def run(*_argv: str) -> tuple[int, str, str]:
        return (0, json.dumps([{"name": "pytest", "bucket": "pass" if green else "fail"}]), "")
    return run


def test_full_integrated_loop_reaches_finalize() -> None:
    state = LoopState(goal="add scorer", current_phase=Phase.RECEIVE_GOAL)
    ctx = LoopContext(repo_root=REPO_ROOT, executor=_executor([{"id": "impl", "allowed_paths": ["engine/"]}]),
                      reviewer=lambda _s: [], gh_runner=_gh(green=True), pr_ref="1",
                      run_cs_assure=_cs_assure(verify_green=True))
    run_loop(state, ctx)
    assert state.current_phase is Phase.FINALIZE
    assert any(t["id"] == "impl" and t["status"] == "DONE" for t in state.task_graph)  # task closed
    assert "sha256:v" in state.assurance_records  # cs_assure evidence recorded on the state


def test_invalid_decompose_replans() -> None:
    def bad(state: LoopState, _d) -> Observation:
        state.task_graph = [{"id": "a", "depends_on": ["ghost"]}]  # dangling dependency
        return Observation.SUCCESS
    state = LoopState(current_phase=Phase.DECOMPOSE)
    t = step(state, LoopContext(repo_root=REPO_ROOT, executor=bad, run_cs_assure=_cs_assure()))
    assert t.decision is Decision.REPLAN and state.current_phase is Phase.PLAN


def test_observe_red_gate_revises() -> None:
    state = LoopState(current_phase=Phase.OBSERVE)
    t = step(state, LoopContext(repo_root=REPO_ROOT, executor=_executor(),
                                run_cs_assure=_cs_assure(verify_green=False)))
    assert t.decision is Decision.REVISE and state.current_phase is Phase.EXECUTE


def test_observe_flags_stale_docs_as_contract_drift() -> None:
    # Green gate but loop code changed without its doc -> the OBSERVE handler folds in docs-freshness.
    state = LoopState(current_phase=Phase.OBSERVE)
    step(state, LoopContext(repo_root=REPO_ROOT, executor=_executor(),
                            run_cs_assure=_cs_assure(verify_green=True, changed=("scripts/loop/x.py",))))
    assert state.observations[-1]["observation"] == "CONTRACT_DRIFT"


def test_integrate_self_modify_escalates_at_the_merge_gate() -> None:
    state = LoopState(current_phase=Phase.INTEGRATE)
    t = step(state, LoopContext(repo_root=REPO_ROOT, executor=_executor(), gh_runner=_gh(green=True),
                                pr_ref="1", run_cs_assure=_cs_assure(obligations=("assurance-self-modify",))))
    assert t.decision is Decision.ESCALATE and state.current_phase is Phase.ESCALATED


def test_multi_agent_execute_dispatches_a_parallel_wave() -> None:
    state = LoopState(current_phase=Phase.EXECUTE)
    decompose(state, [{"id": "a", "allowed_paths": ["engine/"]}, {"id": "b", "allowed_paths": ["scripts/"]}])
    ctx = LoopContext(repo_root=REPO_ROOT, executor=_executor(), multi_agent=True,
                      agent_runner=lambda task: AgentResult(task.id, Observation.SUCCESS, changed_paths=[]),
                      run_cs_assure=_cs_assure())
    t = step(state, ctx)
    assert t.decision is Decision.ADVANCE  # wave all-success -> advance
    assert all(x["status"] == "DONE" for x in state.task_graph)  # both tasks dispatched + done


def test_step_persists_when_a_store_path_is_set(tmp_path: Path) -> None:
    from loop.store import load
    path = tmp_path / "loop.json"
    state = LoopState(current_phase=Phase.RECON)
    step(state, LoopContext(repo_root=REPO_ROOT, executor=_executor(), run_cs_assure=_cs_assure(),
                            store_path=path))
    assert load(path).current_phase is Phase.DEFINE_SUCCESS
