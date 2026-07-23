"""The task graph + ownership model (controller slice 4 - the L6 groundwork).

A goal decomposes into OWNED tasks. This module is the data model + the pure graph operations the
DECOMPOSE / ASSIGN phases need: validate a graph (fail-closed), compute which tasks are ready, detect
ownership (allowed-path) COLLISIONS between tasks, and pick the next assignable task without violating a
boundary. The file-ownership overlap check is the primitive that lets slice 5 run agents in parallel
without two of them editing the same files.

The graph lives in ``LoopState.task_graph`` as a list of plain dicts (so it round-trips through
:mod:`loop.store`); :class:`Task` is the typed view. Same stdlib-only / fail-closed discipline as the
rest of the loop: a malformed graph (bad id, dangling/​cyclic dependency, bad status, unsafe path)
raises :class:`LoopTaskError` rather than being silently accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from loop.controller import LoopState


class TaskStatus(str, Enum):
    """The persisted lifecycle of a task. READY / BLOCKED are DERIVED (from deps), not stored."""

    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    DONE = "DONE"
    FAILED = "FAILED"


class LoopTaskError(Exception):
    """The task graph is malformed (bad id / dangling or cyclic dependency / bad status / unsafe path)."""


@dataclass
class Task:
    """One owned unit of work and its ownership boundary."""

    id: str
    description: str = ""
    owner: str = "self"
    allowed_paths: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    retry_budget: int = 3
    status: TaskStatus = TaskStatus.PENDING
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "description": self.description, "owner": self.owner,
            "allowed_paths": list(self.allowed_paths), "depends_on": list(self.depends_on),
            "success_criteria": list(self.success_criteria), "retry_budget": self.retry_budget,
            "status": self.status.value, "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: Any) -> Task:
        if not isinstance(data, dict):
            raise LoopTaskError(f"task is not an object (got {type(data).__name__})")
        task_id = data.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise LoopTaskError("task has no non-empty string 'id'")
        status_value = data.get("status", TaskStatus.PENDING.value)
        try:
            status = TaskStatus(status_value)
        except ValueError as exc:
            raise LoopTaskError(f"task {task_id!r}: invalid status {status_value!r}") from exc
        paths = data.get("allowed_paths", []) or []
        if not isinstance(paths, list):
            raise LoopTaskError(f"task {task_id!r}: allowed_paths must be a list")
        for p in paths:
            if not isinstance(p, str) or not p:
                raise LoopTaskError(f"task {task_id!r}: allowed path {p!r} must be a non-empty string")
            if PurePosixPath(p).is_absolute() or "\\" in p or ".." in PurePosixPath(p).parts:
                raise LoopTaskError(f"task {task_id!r}: allowed path {p!r} must be repo-relative (no absolute/'..'/'\\')")
        deps = data.get("depends_on", []) or []
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            raise LoopTaskError(f"task {task_id!r}: depends_on must be a list of ids")
        return cls(
            id=task_id, description=str(data.get("description", "")), owner=str(data.get("owner", "self")),
            allowed_paths=list(paths), depends_on=list(deps),
            success_criteria=list(data.get("success_criteria", []) or []),
            retry_budget=int(data.get("retry_budget", 3)), status=status,
            evidence=list(data.get("evidence", []) or []),
        )


def parse_tasks(task_dicts: Any) -> list[Task]:
    """Validate a task graph and return the typed tasks. Fail-closed: unique ids, every dependency
    references an existing task, and NO dependency cycle."""
    if not isinstance(task_dicts, list):
        raise LoopTaskError("task graph is not a list")
    tasks = [Task.from_dict(d) for d in task_dicts]
    by_id: dict[str, Task] = {}
    for task in tasks:
        if task.id in by_id:
            raise LoopTaskError(f"duplicate task id {task.id!r}")
        by_id[task.id] = task
    for task in tasks:
        for dep in task.depends_on:
            if dep not in by_id:
                raise LoopTaskError(f"task {task.id!r} depends on unknown task {dep!r}")
            if dep == task.id:
                raise LoopTaskError(f"task {task.id!r} depends on itself")
    _check_acyclic(by_id)
    return tasks


def _check_acyclic(by_id: dict[str, Task]) -> None:
    # Kahn-style resolution (ITERATIVE - no recursion, so no RecursionError on a deep chain): repeatedly
    # resolve tasks all of whose dependencies are already resolved; if any remain, they form a cycle.
    remaining = dict(by_id)
    resolved: set[str] = set()
    while True:
        newly = [tid for tid, task in remaining.items()
                 if all(dep in resolved for dep in task.depends_on)]
        if not newly:
            break
        for tid in newly:
            resolved.add(tid)
            del remaining[tid]
    if remaining:
        raise LoopTaskError(f"dependency cycle among tasks {sorted(remaining)}")


def is_ready(task: Task, by_id: dict[str, Task]) -> bool:
    """A PENDING task whose every dependency is DONE - assignable now."""
    return task.status is TaskStatus.PENDING and all(
        by_id[d].status is TaskStatus.DONE for d in task.depends_on)


def is_blocked(task: Task, by_id: dict[str, Task]) -> bool:
    """A PENDING task with a FAILED dependency - it can never become ready as-is."""
    return task.status is TaskStatus.PENDING and any(
        by_id[d].status is TaskStatus.FAILED for d in task.depends_on)


def ready_tasks(tasks: list[Task]) -> list[Task]:
    by_id = {t.id: t for t in tasks}
    return [t for t in tasks if is_ready(t, by_id)]


def _covers(parent: str, child: str) -> bool:
    p = PurePosixPath(parent.rstrip("/"))
    c = PurePosixPath(child.rstrip("/"))
    return c == p or p in c.parents


def paths_overlap(a: str, b: str) -> bool:
    """Two allowed paths overlap if either is the same file/dir as, or an ancestor of, the other."""
    return _covers(a, b) or _covers(b, a)


def tasks_conflict(t1: Task, t2: Task) -> bool:
    """Two tasks conflict if any of their allowed paths overlap - they could edit the same file. A task
    with NO declared ownership (empty ``allowed_paths``) owns an undeclared lane and conflicts with
    EVERYTHING, so it is never co-scheduled in parallel (fail-closed - an undeclared boundary must not be
    treated as 'owns nothing')."""
    if not t1.allowed_paths or not t2.allowed_paths:
        return True
    return any(paths_overlap(pa, pb) for pa in t1.allowed_paths for pb in t2.allowed_paths)


def path_conflicts(tasks: list[Task]) -> list[tuple[str, str]]:
    """All conflicting (unordered) task-id pairs by allowed-path overlap - the ownership-boundary check
    that must be empty before two tasks run in parallel (slice 5)."""
    conflicts: list[tuple[str, str]] = []
    for i, a in enumerate(tasks):
        for b in tasks[i + 1:]:
            if tasks_conflict(a, b):
                conflicts.append((a.id, b.id))
    return conflicts


def next_assignable(tasks: list[Task]) -> Task | None:
    """The first ready task whose ownership boundary does NOT overlap any currently-ACTIVE task, so a
    new assignment never contends with work already in flight."""
    active = [t for t in tasks if t.status is TaskStatus.ACTIVE]
    for task in ready_tasks(tasks):
        if not any(tasks_conflict(task, act) for act in active):
            return task
    return None


def is_complete(tasks: list[Task]) -> bool:
    """Every task reached a terminal status (DONE or FAILED) - the graph has nothing left to run."""
    return bool(tasks) and all(t.status in (TaskStatus.DONE, TaskStatus.FAILED) for t in tasks)


# --------------------------------------------------------------------------- LoopState integration


def decompose(state: LoopState, task_dicts: list[dict[str, Any]]) -> list[Task]:
    """Validate a proposed task graph and install it on the loop state (normalised). Raises
    :class:`LoopTaskError` on any malformed graph, so a bad DECOMPOSE never becomes the plan."""
    tasks = parse_tasks(task_dicts)
    state.task_graph = [t.to_dict() for t in tasks]
    return tasks


def assign_next(state: LoopState) -> Task | None:
    """Pick the next assignable task, mark it ACTIVE on the state, and return it (None if nothing is
    ready). The chosen task's ``allowed_paths`` then flow into the driver's next directive."""
    tasks = parse_tasks(state.task_graph)
    task = next_assignable(tasks)
    if task is None:
        return None
    task.status = TaskStatus.ACTIVE
    state.task_graph = [t.to_dict() for t in tasks]
    return task


def set_status(state: LoopState, task_id: str, status: TaskStatus, *, evidence: str | None = None) -> None:
    """Update one task's status on the state (e.g. mark the active task DONE/FAILED after a cycle),
    optionally attaching an evidence reference (an assurance record digest)."""
    tasks = parse_tasks(state.task_graph)
    by_id = {t.id: t for t in tasks}
    if task_id not in by_id:
        raise LoopTaskError(f"cannot set status of unknown task {task_id!r}")
    by_id[task_id].status = status
    if evidence is not None:
        by_id[task_id].evidence.append(evidence)
    state.task_graph = [t.to_dict() for t in tasks]
