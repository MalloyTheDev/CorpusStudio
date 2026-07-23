"""Tests for the agent router (scripts/loop/router.py, controller slice 5 - L6 coordination core).

Pins parallel-safe wave selection, the <=10 fan-out cap (enforced in the router, not trusted to the
caller), file-ownership boundary ENFORCEMENT (an out-of-lane edit is a POLICY_BLOCK), dispatch status
bookkeeping + evidence, and the worst-case wave aggregation - all via an injected agent runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from loop.controller import LoopState, Observation  # noqa: E402
from loop.router import (  # noqa: E402
    MAX_FANOUT,
    AgentResult,
    WaveOutcome,
    aggregate_observation,
    check_boundary,
    dispatch_wave,
    select_wave,
    within_boundary,
)
from loop.tasks import Task, TaskStatus, decompose, parse_tasks  # noqa: E402


def _t(tid: str, deps: list[str] | None = None, paths: list[str] | None = None, status: str = "PENDING") -> dict:
    return {"id": tid, "depends_on": deps or [], "allowed_paths": paths or [], "status": status}


def _runner(results: dict[str, AgentResult]):
    def run(task: Task) -> AgentResult:
        return results.get(task.id, AgentResult(task.id, Observation.SUCCESS, changed_paths=[]))
    return run


# --------------------------------------------------------------------------- boundary enforcement


def test_within_and_check_boundary() -> None:
    assert within_boundary("engine/tests/x.py", ["engine/"])
    assert not within_boundary("scripts/loop/x.py", ["engine/"])
    task = Task(id="a", allowed_paths=["engine/corpus_studio/"])
    assert check_boundary(task, ["engine/corpus_studio/eval.py"]) == []
    assert check_boundary(task, ["engine/corpus_studio/eval.py", "docs/x.md"]) == ["docs/x.md"]
    # a task with NO declared boundary owns nothing -> any edit is a breach (fail-closed).
    assert check_boundary(Task(id="b"), ["anything/at/all.py"]) == ["anything/at/all.py"]


def test_boundary_rejects_dotdot_traversal() -> None:
    # A '..'-escaped path must NEVER score as in-lane, even when a prefix looks like the allowed dir.
    assert not within_boundary("engine/../scripts/loop/evil.py", ["engine/"])
    assert not within_boundary("/etc/passwd", ["engine/"])
    task = Task(id="a", allowed_paths=["engine/"])
    assert check_boundary(task, ["engine/ok.py", "engine/../scripts/evil.py"]) == ["engine/../scripts/evil.py"]


# --------------------------------------------------------------------------- wave selection + cap


def test_select_wave_is_parallel_safe() -> None:
    tasks = parse_tasks([_t("a", paths=["engine/"]), _t("b", paths=["engine/x.py"]), _t("c", paths=["scripts/"])])
    wave = {t.id for t in select_wave(tasks)}
    assert wave == {"a", "c"}  # b conflicts with a (both touch engine/) -> held back this wave


def test_select_wave_excludes_conflicts_with_active_tasks() -> None:
    tasks = parse_tasks([_t("a", paths=["engine/"], status="ACTIVE"),
                         _t("b", paths=["engine/x.py"]), _t("c", paths=["scripts/"])])
    assert {t.id for t in select_wave(tasks)} == {"c"}  # b conflicts with in-flight a


def test_select_wave_enforces_the_fanout_cap() -> None:
    # 15 disjoint ready tasks -> the router still dispatches at most MAX_FANOUT (10).
    tasks = parse_tasks([_t(f"t{i}", paths=[f"area{i}/"]) for i in range(15)])
    assert len(select_wave(tasks)) == MAX_FANOUT
    assert len(select_wave(tasks, max_agents=3)) == 3          # a smaller request is honored
    assert len(select_wave(tasks, max_agents=100)) == MAX_FANOUT  # a larger request is still capped


# --------------------------------------------------------------------------- dispatch + enforcement


def test_dispatch_marks_status_and_records_evidence() -> None:
    state = LoopState()
    decompose(state, [_t("a", paths=["engine/"]), _t("b", paths=["scripts/"])])
    outcomes = dispatch_wave(state, _runner({
        "a": AgentResult("a", Observation.SUCCESS, changed_paths=["engine/x.py"], evidence="sha256:a"),
        "b": AgentResult("b", Observation.TEST_REGRESSION, changed_paths=["scripts/y.py"]),
    }))
    by_id = {o.task_id: o for o in outcomes}
    assert by_id["a"].status is TaskStatus.DONE and by_id["b"].status is TaskStatus.FAILED
    graph = {t["id"]: t for t in state.task_graph}
    assert graph["a"]["status"] == "DONE" and graph["a"]["evidence"] == ["sha256:a"]
    assert graph["b"]["status"] == "FAILED"


def test_progress_leaves_a_task_pending_not_done() -> None:
    # A PROGRESS agent result means "moved forward, not done" -> the task stays PENDING (re-dispatchable),
    # never DONE (no completion claim).
    state = LoopState()
    decompose(state, [_t("a", paths=["engine/"])])
    outcomes = dispatch_wave(state, _runner({"a": AgentResult("a", Observation.PROGRESS, changed_paths=[])}))
    assert outcomes[0].status is TaskStatus.PENDING
    assert next(t for t in state.task_graph if t["id"] == "a")["status"] == "PENDING"


def test_dispatch_rejects_a_boundary_breach_as_policy_block() -> None:
    state = LoopState()
    decompose(state, [_t("a", paths=["engine/"])])
    # the agent CLAIMS success but edited outside its lane -> rejected regardless.
    outcomes = dispatch_wave(state, _runner({
        "a": AgentResult("a", Observation.SUCCESS, changed_paths=["engine/x.py", "scripts/loop/evil.py"]),
    }))
    assert outcomes[0].observation is Observation.POLICY_BLOCK
    assert outcomes[0].status is TaskStatus.FAILED and "outside its boundary" in outcomes[0].reason


# --------------------------------------------------------------------------- aggregation


def test_aggregate_worst_case_wins() -> None:
    ok = WaveOutcome("a", Observation.SUCCESS, TaskStatus.DONE, "")
    fail = WaveOutcome("b", Observation.TEST_REGRESSION, TaskStatus.FAILED, "")
    breach = WaveOutcome("c", Observation.POLICY_BLOCK, TaskStatus.FAILED, "")
    assert aggregate_observation([ok, ok])[0] is Observation.SUCCESS
    assert aggregate_observation([ok, fail])[0] is Observation.TEST_REGRESSION
    assert aggregate_observation([ok, fail, breach])[0] is Observation.POLICY_BLOCK
    assert aggregate_observation([])[0] is Observation.PROGRESS  # nothing ready -> advance, not "done"
