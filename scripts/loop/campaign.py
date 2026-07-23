"""Multi-goal campaign orchestration (controller slice 10, the last L8 piece).

A campaign runs a QUEUE of goals - each as its own :func:`loop.orchestrate.run_loop` - sharing one
cross-goal learning ledger, so a backlog is driven end to end and effort compounds across goals instead
of resetting. Because every goal's loop seeds from and records to the same ``ctx.ledger_path``, a dead
end one goal exhausted is known to the next; and goals may declare dependencies, so a goal whose
prerequisite did not finalize is skipped rather than run blindly.

This is a thin, deterministic orchestration over the already-hardened run_loop: it owns the goal order,
per-goal state isolation, the dependency gate, and the stop-on-blocker policy. stdlib-only, fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from loop.controller import LoopState, Phase
from loop.orchestrate import LoopContext, run_loop


class CampaignError(Exception):
    """The campaign is malformed (duplicate / dangling goal ids) - fail-closed."""


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
        if not g.goal_id:
            raise CampaignError("every goal needs a non-empty goal_id")
        if g.goal_id in ids:
            raise CampaignError(f"duplicate goal_id {g.goal_id!r}")
        ids.add(g.goal_id)
    for g in goals:
        for dep in g.depends_on:
            if dep not in ids:
                raise CampaignError(f"goal {g.goal_id!r} depends on unknown goal {dep!r}")
            if dep == g.goal_id:
                raise CampaignError(f"goal {g.goal_id!r} depends on itself")


def _goal_store(store_dir: Path | None, goal_id: str) -> Path | None:
    return (store_dir / f"{goal_id}.json") if store_dir is not None else None


def run_campaign(goals: list[Goal], ctx: LoopContext, *, store_dir: Path | None = None,
                 max_steps: int = 200, stop_on_escalate: bool = True) -> list[GoalOutcome]:
    """Run each goal in order (its own loop, isolated state, shared learning ledger). A goal whose
    dependency did not FINALIZE is SKIPPED; a goal that ESCALATES halts the campaign for human attention
    when ``stop_on_escalate`` (a hard blocker should not be stepped over silently). Returns per-goal
    outcomes. Fail-closed: a malformed goal list (duplicate / dangling / self dependency) raises."""
    _validate(goals)
    outcomes: list[GoalOutcome] = []
    finalized: set[str] = set()
    for goal in goals:
        if any(dep not in finalized for dep in goal.depends_on):
            outcomes.append(GoalOutcome(goal.goal_id, goal.goal, "SKIPPED",
                                        "an upstream goal did not finalize", False))
            continue
        state = LoopState(goal=goal.goal, goal_id=goal.goal_id, current_phase=Phase.RECEIVE_GOAL)
        goal_ctx = replace(ctx, store_path=_goal_store(store_dir, goal.goal_id))
        run_loop(state, goal_ctx, max_steps=max_steps)  # shares ctx.ledger_path -> cross-goal learning
        is_final = state.current_phase is Phase.FINALIZE
        if is_final:
            finalized.add(goal.goal_id)
        outcomes.append(GoalOutcome(goal.goal_id, goal.goal, state.current_phase.value,
                                    state.termination_reason, is_final))
        if not is_final and state.current_phase is Phase.ESCALATED and stop_on_escalate:
            break  # a hard blocker halts the campaign - the rest wait for a human
    return outcomes
