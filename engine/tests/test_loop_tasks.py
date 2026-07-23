"""Tests for the task graph + ownership model (scripts/loop/tasks.py, controller slice 4).

Pins graph validation (fail-closed on dup id / dangling / self / cyclic deps / bad status / unsafe
path), the ready/blocked derivation, the allowed-path OVERLAP check (the L6 ownership boundary), next
assignable selection that respects active-task boundaries, and the LoopState integration helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from loop.controller import LoopState  # noqa: E402
from loop.tasks import (  # noqa: E402
    LoopTaskError,
    Task,
    TaskStatus,
    assign_next,
    decompose,
    is_complete,
    parse_tasks,
    path_conflicts,
    paths_overlap,
    ready_tasks,
    set_status,
    tasks_conflict,
)


def _t(tid: str, deps: list[str] | None = None, paths: list[str] | None = None,
       status: str = "PENDING") -> dict:
    return {"id": tid, "depends_on": deps or [], "allowed_paths": paths or [], "status": status}


# --------------------------------------------------------------------------- validation (fail-closed)


def test_task_dict_round_trips() -> None:
    d = _t("a", deps=["b"], paths=["engine/x.py"])
    d["id"] = "a"
    assert Task.from_dict(Task.from_dict({**d, "depends_on": []}).to_dict()).id == "a"


def test_parse_rejects_duplicate_ids() -> None:
    with pytest.raises(LoopTaskError, match="duplicate"):
        parse_tasks([_t("a"), _t("a")])


def test_parse_rejects_dangling_and_self_dependency() -> None:
    with pytest.raises(LoopTaskError, match="unknown task"):
        parse_tasks([_t("a", deps=["ghost"])])
    with pytest.raises(LoopTaskError, match="itself"):
        parse_tasks([_t("a", deps=["a"])])


def test_parse_rejects_dependency_cycles() -> None:
    with pytest.raises(LoopTaskError, match="cycle"):
        parse_tasks([_t("a", deps=["b"]), _t("b", deps=["a"])])
    with pytest.raises(LoopTaskError, match="cycle"):
        parse_tasks([_t("a", deps=["c"]), _t("b", deps=["a"]), _t("c", deps=["b"])])


def test_parse_rejects_bad_status_and_unsafe_paths() -> None:
    with pytest.raises(LoopTaskError, match="status"):
        parse_tasks([_t("a", status="BOGUS")])
    with pytest.raises(LoopTaskError, match="repo-relative"):
        parse_tasks([_t("a", paths=["/etc/passwd"])])
    with pytest.raises(LoopTaskError, match="repo-relative"):
        parse_tasks([_t("a", paths=["../secrets"])])


# --------------------------------------------------------------------------- ready / blocked


def test_ready_requires_all_deps_done() -> None:
    tasks = parse_tasks([_t("impl"), _t("test", deps=["impl"])])
    assert {t.id for t in ready_tasks(tasks)} == {"impl"}  # test is not ready until impl is DONE
    tasks[0].status = TaskStatus.DONE
    assert {t.id for t in ready_tasks(tasks)} == {"test"}


def test_ready_excludes_active_and_done() -> None:
    tasks = parse_tasks([_t("a", status="ACTIVE"), _t("b", status="DONE"), _t("c")])
    assert {t.id for t in ready_tasks(tasks)} == {"c"}


# --------------------------------------------------------------------------- ownership boundary


def test_paths_overlap_semantics() -> None:
    assert paths_overlap("scripts/loop/", "scripts/loop/controller.py")   # dir contains file
    assert paths_overlap("scripts/", "scripts/loop/")                     # ancestor dir
    assert paths_overlap("a/b.py", "a/b.py")                              # identical
    assert not paths_overlap("scripts/loop/controller.py", "scripts/loop/store.py")  # sibling files
    assert not paths_overlap("engine/", "scripts/")                       # disjoint trees


def test_path_conflicts_finds_overlapping_task_pairs() -> None:
    tasks = parse_tasks([
        _t("engine", paths=["engine/corpus_studio/"]),
        _t("tests", paths=["engine/tests/"]),
        _t("wide", paths=["engine/"]),          # overlaps BOTH of the above
    ])
    conflicts = {frozenset(p) for p in path_conflicts(tasks)}
    assert frozenset({"wide", "engine"}) in conflicts
    assert frozenset({"wide", "tests"}) in conflicts
    assert frozenset({"engine", "tests"}) not in conflicts  # sibling subtrees are disjoint


def test_next_assignable_respects_active_task_boundaries() -> None:
    # 'a' is active on engine/; a ready 'b' that also touches engine/ must NOT be assigned concurrently,
    # but a ready 'c' on scripts/ can be.
    tasks = parse_tasks([
        _t("a", paths=["engine/"], status="ACTIVE"),
        _t("b", paths=["engine/x.py"]),
        _t("c", paths=["scripts/loop/"]),
    ])
    from loop.tasks import next_assignable
    assert next_assignable(tasks).id == "c"  # skips b (conflicts with active a), picks c


# --------------------------------------------------------------------------- LoopState integration


def test_decompose_installs_a_valid_graph_and_rejects_a_bad_one() -> None:
    state = LoopState()
    decompose(state, [_t("impl"), _t("test", deps=["impl"])])
    assert [t["id"] for t in state.task_graph] == ["impl", "test"]
    with pytest.raises(LoopTaskError):
        decompose(state, [_t("x", deps=["missing"])])


def test_assign_next_marks_active_and_set_status_completes() -> None:
    state = LoopState()
    decompose(state, [_t("impl", paths=["engine/"]), _t("test", deps=["impl"], paths=["engine/tests/"])])
    picked = assign_next(state)
    assert picked is not None and picked.id == "impl"
    assert next(t for t in state.task_graph if t["id"] == "impl")["status"] == "ACTIVE"
    # test is not assignable yet (impl not done); complete impl, then test becomes assignable.
    assert assign_next(state) is None
    set_status(state, "impl", TaskStatus.DONE, evidence="sha256:rec")
    assert assign_next(state).id == "test"
    assert next(t for t in state.task_graph if t["id"] == "impl")["evidence"] == ["sha256:rec"]


def test_is_complete_when_all_terminal() -> None:
    tasks = parse_tasks([_t("a", status="DONE"), _t("b", status="FAILED")])
    assert is_complete(tasks)
    assert not is_complete(parse_tasks([_t("a", status="DONE"), _t("b")]))


def test_tasks_conflict_pairwise() -> None:
    a = Task(id="a", allowed_paths=["engine/"])
    b = Task(id="b", allowed_paths=["engine/tests/x.py"])
    c = Task(id="c", allowed_paths=["scripts/"])
    assert tasks_conflict(a, b) and not tasks_conflict(a, c)
