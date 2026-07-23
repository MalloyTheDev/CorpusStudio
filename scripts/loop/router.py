"""The agent router: coordinated multi-agent execution with strict file-ownership boundaries
(controller slice 5 - the coordination core of Level 6).

Slice 4 gave the ownership model; this turns it into actual coordination. Given a task graph it selects a
parallel-SAFE wave (ready tasks with pairwise-disjoint ownership, none overlapping an in-flight task),
CAPS the fan-out at 10 agents (a standing hard rule), dispatches one agent per task, ENFORCES that each
agent stayed inside its declared ``allowed_paths`` (an out-of-lane edit is rejected as a boundary breach),
and folds the wave down to a single loop Observation.

The actual agent SPAWN is an effect (the Agent/Workflow tool) that cannot run inside a stdlib module, so
- exactly like :mod:`loop.observe` and :mod:`loop.driver` - the spawn is an INJECTED ``runner`` callback
(task -> :class:`AgentResult`). This module is the deterministic coordination logic around it: selection,
the cap, boundary enforcement, status bookkeeping, and aggregation. Same stdlib-only discipline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Callable

from loop.controller import LoopState, Observation
from loop.tasks import (
    Task,
    TaskStatus,
    parse_tasks,
    ready_tasks,
    set_status,
    status_for,
    tasks_conflict,
)

# Standing hard cap on concurrent agents - never dispatch a wave wider than this, regardless of request.
MAX_FANOUT = 10

# Observations that mean the delegated task actually succeeded.
_OK = frozenset({Observation.SUCCESS, Observation.PROGRESS})

AgentRunner = Callable[[Task], "AgentResult"]


@dataclass(frozen=True)
class AgentResult:
    """What an injected agent runner reports back for one task."""

    task_id: str
    observation: Observation                       # the agent's own result (SUCCESS / a failure class)
    changed_paths: list[str] = field(default_factory=list)  # what it actually edited (for enforcement)
    evidence: str | None = None                    # an assurance record digest, if any
    note: str = ""


@dataclass(frozen=True)
class WaveOutcome:
    """The router's verdict for one dispatched task (after boundary enforcement)."""

    task_id: str
    observation: Observation
    status: TaskStatus
    reason: str


def _covers(parent: str, child: str) -> bool:
    p = PurePosixPath(parent.rstrip("/"))
    c = PurePosixPath(child.rstrip("/"))
    return c == p or p in c.parents


def within_boundary(changed: str, allowed_paths: list[str]) -> bool:
    """True if ``changed`` falls under some allowed path (i.e. the agent stayed in its lane). A changed
    path that is absolute, uses a backslash, or contains a ``..`` segment is ALWAYS out of bounds
    (fail-closed): a traversal like ``engine/../scripts/x`` must never be scored as in-lane, mirroring
    the same rejection tasks.Task applies to declared paths."""
    p = PurePosixPath(changed)
    if p.is_absolute() or "\\" in changed or ".." in p.parts:
        return False
    return any(_covers(a, changed) for a in allowed_paths)


def check_boundary(task: Task, changed_paths: list[str]) -> list[str]:
    """The changed paths that fall OUTSIDE the task's ownership boundary (empty result = the agent
    stayed in its lane). A task with NO declared ``allowed_paths`` owns nothing, so ANY edit it makes is
    a breach (fail-closed) - an agent must declare its lane to be trusted with edits."""
    if not task.allowed_paths:
        return list(changed_paths)
    return [c for c in changed_paths if not within_boundary(c, task.allowed_paths)]


def select_wave(tasks: list[Task], *, max_agents: int = MAX_FANOUT) -> list[Task]:
    """A parallel-safe wave of ready tasks: pairwise non-conflicting ownership, none overlapping an
    ACTIVE task, deterministically ordered, and never wider than ``min(max_agents, MAX_FANOUT)`` - the
    fan-out cap is enforced HERE so a caller can never exceed it."""
    limit = max(0, min(max_agents, MAX_FANOUT))
    active = [t for t in tasks if t.status is TaskStatus.ACTIVE]
    wave: list[Task] = []
    for task in ready_tasks(tasks):
        if len(wave) >= limit:
            break
        if any(tasks_conflict(task, other) for other in active):
            continue  # would contend with in-flight work
        if any(tasks_conflict(task, chosen) for chosen in wave):
            continue  # would contend with a sibling already in this wave
        wave.append(task)
    return wave


def dispatch_wave(state: LoopState, runner: AgentRunner, *,
                  max_agents: int = MAX_FANOUT) -> list[WaveOutcome]:
    """Select a parallel-safe wave, mark it ACTIVE, run each task via ``runner``, ENFORCE each agent's
    ownership boundary, and record the resulting DONE/FAILED status + evidence on the loop state. An
    agent that edits outside its ``allowed_paths`` is a POLICY_BLOCK -> FAILED (escalate), whatever it
    claimed. Returns one :class:`WaveOutcome` per dispatched task."""
    tasks = parse_tasks(state.task_graph)
    wave = select_wave(tasks, max_agents=max_agents)
    by_id = {t.id: t for t in tasks}
    for task in wave:
        by_id[task.id].status = TaskStatus.ACTIVE
    state.task_graph = [t.to_dict() for t in tasks]

    outcomes: list[WaveOutcome] = []
    for task in wave:
        result = runner(task)
        outside = check_boundary(task, result.changed_paths)
        if outside:
            observation = Observation.POLICY_BLOCK
            status = TaskStatus.FAILED
            reason = f"agent for {task.id!r} edited outside its boundary: {sorted(outside)}"
        else:
            observation = result.observation
            status = status_for(observation)  # SUCCESS->DONE, PROGRESS->PENDING, else FAILED (shared)
            reason = result.note or f"agent for {task.id!r} -> {observation.value}"
        set_status(state, task.id, status, evidence=result.evidence)
        outcomes.append(WaveOutcome(task.id, observation, status, reason))
    return outcomes


def aggregate_observation(outcomes: list[WaveOutcome]) -> tuple[Observation, str]:
    """Reduce a dispatched wave to ONE loop observation, worst-case first: a boundary breach
    (POLICY_BLOCK) beats any other failure, which beats success. An empty wave means nothing was ready -
    PROGRESS, so the loop advances to re-plan/assign rather than treating "nothing to do" as done."""
    if not outcomes:
        return Observation.PROGRESS, "no task was ready to dispatch this wave"
    if any(o.observation is Observation.POLICY_BLOCK for o in outcomes):
        return Observation.POLICY_BLOCK, "a delegated agent breached its ownership boundary"
    failures = [o for o in outcomes if o.observation not in _OK]
    if failures:
        return failures[0].observation, f"{len(failures)}/{len(outcomes)} delegated task(s) failed"
    return Observation.SUCCESS, f"all {len(outcomes)} delegated task(s) succeeded within their boundaries"
