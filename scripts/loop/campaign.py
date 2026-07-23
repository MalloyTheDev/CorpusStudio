"""Multi-goal campaign orchestration (controller slice 10, an L8 mechanism).

A campaign runs a set of goals - each as its own :func:`loop.orchestrate.run_loop` - sharing one
cross-goal dead-end memory ledger, so a backlog is driven end to end and known dead ends carry across
goals. Goals form a DEPENDENCY DAG: they are TOPOLOGICALLY scheduled (a goal runs once every prerequisite
has finalized, regardless of input order), a goal whose prerequisite did not finalize is skipped, and a
dependency cycle is rejected fail-closed.

Isolation: a ``context_for(goal)`` factory (injected, like every other effect) lets a runtime give each
goal its OWN :class:`LoopContext` - separate branch / worktree / PR / state file - so goals do not share a
working tree. Without a factory the goals share one base context (only their per-goal state files differ);
the actual branch/worktree/PR creation is the factory's job (a runtime effect), not this module's. A goal
whose per-goal state file already exists is RESUMED (continued, or reported if already finished) rather
than restarted, so a crashed/paused campaign picks up where it left off. stdlib-only, fail-closed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from loop.controller import LoopState, Phase
from loop.orchestrate import LoopContext, run_loop
from loop.store import LoopStateError, load

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


# A runtime supplies each goal's isolated LoopContext (its own branch / worktree / PR / state file).
ContextFactory = Callable[[Goal], LoopContext]


def _goal_context(goal: Goal, ctx: LoopContext | None, context_for: ContextFactory | None,
                  store_dir: Path | None) -> LoopContext:
    """The LoopContext for one goal: from the injected ``context_for`` factory (isolated per goal) if given,
    else the shared base ``ctx`` with a per-goal state file. Fail-closed on a misbehaving factory."""
    if context_for is not None:
        try:
            goal_ctx = context_for(goal)
        except Exception as exc:  # noqa: BLE001 - the injected factory is untrusted; fail closed
            raise CampaignError(f"context_for({goal.goal_id!r}) raised {type(exc).__name__}: {exc}") from exc
        if not isinstance(goal_ctx, LoopContext):
            raise CampaignError(
                f"context_for({goal.goal_id!r}) must return a LoopContext, got {type(goal_ctx).__name__}")
        return goal_ctx
    assert ctx is not None  # run_campaign guarantees ctx or context_for is present
    return replace(ctx, store_path=_goal_store(store_dir, goal.goal_id))


def _resume_or_new(goal: Goal, goal_ctx: LoopContext) -> LoopState:
    """RESUME a goal from its persisted state file if one exists (so a crashed/paused campaign continues),
    else start it fresh. Fail-closed on an unreadable or mismatched resume state."""
    path = goal_ctx.store_path
    if path is not None and path.exists():
        try:
            state = load(path)
        except LoopStateError as exc:
            raise CampaignError(f"goal {goal.goal_id!r} has an unreadable resume state ({path}): {exc}") from exc
        if state.goal_id and state.goal_id != goal.goal_id:
            raise CampaignError(f"resume state at {path} is goal {state.goal_id!r}, not {goal.goal_id!r}")
        return state
    return LoopState(goal=goal.goal, goal_id=goal.goal_id, current_phase=Phase.RECEIVE_GOAL)


def run_campaign(goals: list[Goal], ctx: LoopContext | None = None, *,
                 context_for: ContextFactory | None = None, store_dir: Path | None = None,
                 max_steps: int = 200, stop_on_escalate: bool = True) -> list[GoalOutcome]:
    """TOPOLOGICALLY schedule + run the goals (each its own loop + isolated context, shared learning
    ledger), regardless of input order: a goal runs once every dependency has FINALIZED; a goal whose
    dependency did NOT finalize is SKIPPED; an ESCALATION halts the campaign when ``stop_on_escalate``.
    Each goal's context comes from the injected ``context_for`` factory (isolated branch/worktree/PR/state)
    or, absent one, the shared ``ctx`` with a per-goal state file. A goal with an existing state file is
    RESUMED. Returns one outcome per goal (input order). Fail-closed on a malformed goal graph or a
    misbehaving factory."""
    if ctx is None and context_for is None:
        raise CampaignError("run_campaign needs a base ctx or a context_for factory")
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
        goal_ctx = _goal_context(goal, ctx, context_for, store_dir)
        state = _resume_or_new(goal, goal_ctx)
        if not state.is_terminal:
            run_loop(state, goal_ctx, max_steps=max_steps)  # shares ledger_path -> cross-goal memory
        # else: a resumed, already-finished goal - report it; do NOT re-run or double-record the ledger.
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
