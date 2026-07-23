"""Multi-goal campaign orchestration (controller slice 10, an L8 mechanism).

A campaign runs a set of goals - each as its own :func:`loop.orchestrate.run_loop` - sharing one
cross-goal dead-end memory ledger, so a backlog is driven end to end and known dead ends carry across
goals. Goals form a DEPENDENCY DAG: they are TOPOLOGICALLY scheduled (a goal runs once every prerequisite
has finalized, regardless of input order), a goal whose prerequisite did not finalize is skipped, and a
dependency cycle is rejected fail-closed.

Scope note (honesty): this is single-repository, single-working-tree orchestration - goals share the
repo/branch/PR/context and are NOT yet isolated per goal (separate branch/worktree/PR). It is a mechanism,
not production-complete campaign isolation. stdlib-only, fail-closed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from loop.controller import LoopState, Phase
from loop.orchestrate import LoopContext, run_loop

# A goal_id is used to name a per-goal state file, so it must be a safe filename component (no path
# separators, no '..', no absolute/traversal) - never interpolate a raw id into a filesystem path.
_SAFE_GOAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class CampaignError(Exception):
    """The campaign is malformed (unsafe / duplicate / dangling / cyclic goal ids) - fail-closed."""


@dataclass
class Goal:
    goal: str
    goal_id: str
    depends_on: list[str] = field(default_factory=list)  # goal_ids that must FINALIZE before this runs


@dataclass(frozen=True)
class GoalOutcome:
    goal_id: str
    goal: str
    final_phase: str
    termination_reason: str | None
    finalized: bool


def _validate(goals: list[Goal]) -> None:
    ids: set[str] = set()
    for g in goals:
        if not _SAFE_GOAL_ID.match(g.goal_id):
            raise CampaignError(f"unsafe goal_id {g.goal_id!r} (must match {_SAFE_GOAL_ID.pattern})")
        if g.goal_id in ids:
            raise CampaignError(f"duplicate goal_id {g.goal_id!r}")
        ids.add(g.goal_id)
    for g in goals:
        for dep in g.depends_on:
            if dep not in ids:
                raise CampaignError(f"goal {g.goal_id!r} depends on unknown goal {dep!r}")
            if dep == g.goal_id:
                raise CampaignError(f"goal {g.goal_id!r} depends on itself")
    # Cycle detection (iterative Kahn - no recursion): resolve goals whose deps are all resolved; if any
    # remain, they form a cycle.
    by_id = {g.goal_id: g for g in goals}
    remaining = dict(by_id)
    resolved: set[str] = set()
    while True:
        newly = [i for i, g in remaining.items() if all(d in resolved for d in g.depends_on)]
        if not newly:
            break
        for i in newly:
            resolved.add(i)
            del remaining[i]
    if remaining:
        raise CampaignError(f"dependency cycle among goals {sorted(remaining)}")


def _goal_store(store_dir: Path | None, goal_id: str) -> Path | None:
    return (store_dir / f"{goal_id}.json") if store_dir is not None else None


def run_campaign(goals: list[Goal], ctx: LoopContext, *, store_dir: Path | None = None,
                 max_steps: int = 200, stop_on_escalate: bool = True) -> list[GoalOutcome]:
    """TOPOLOGICALLY schedule + run the goals (its own loop, isolated state, shared learning ledger),
    regardless of input order: a goal runs once every dependency has FINALIZED; a goal whose dependency
    did NOT finalize is SKIPPED; an ESCALATION halts the campaign when ``stop_on_escalate``. Returns one
    outcome per goal (input order). Fail-closed on a malformed goal graph (unsafe id / dup / dangling /
    self / cyclic dependency)."""
    _validate(goals)
    outcomes: dict[str, GoalOutcome] = {}
    finalized: set[str] = set()
    failed: set[str] = set()  # escalated / stopped / skipped -> its dependents can never become ready
    halted = False

    while not halted and len(outcomes) < len(goals):
        pending = [g for g in goals if g.goal_id not in outcomes]
        ready = [g for g in pending if all(d in finalized for d in g.depends_on)]
        blocked = [g for g in pending if any(d in failed for d in g.depends_on)]
        for g in blocked:  # a dependency failed -> this goal can never run
            outcomes[g.goal_id] = GoalOutcome(g.goal_id, g.goal, "SKIPPED",
                                              "an upstream goal did not finalize", False)
            failed.add(g.goal_id)
        if not ready:
            break  # nothing runnable (remaining are all blocked, now skipped)
        goal = ready[0]  # deterministic: the first input-order ready goal
        state = LoopState(goal=goal.goal, goal_id=goal.goal_id, current_phase=Phase.RECEIVE_GOAL)
        goal_ctx = replace(ctx, store_path=_goal_store(store_dir, goal.goal_id))
        run_loop(state, goal_ctx, max_steps=max_steps)  # shares ctx.ledger_path -> cross-goal memory
        is_final = state.current_phase is Phase.FINALIZE
        (finalized if is_final else failed).add(goal.goal_id)
        outcomes[goal.goal_id] = GoalOutcome(goal.goal_id, goal.goal, state.current_phase.value,
                                             state.termination_reason, is_final)
        if not is_final and state.current_phase is Phase.ESCALATED and stop_on_escalate:
            halted = True  # a hard blocker halts the campaign - the rest wait for a human

    for g in goals:  # anything unrun (campaign halted / unreachable) is reported SKIPPED
        outcomes.setdefault(g.goal_id, GoalOutcome(g.goal_id, g.goal, "SKIPPED",
                                                   "not reached (campaign halted or blocked)", False))
    return [outcomes[g.goal_id] for g in goals]
