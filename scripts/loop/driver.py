"""The driver: the top-level runnable cycle that ties the controller, store, and observer into a loop
(controller slice 3 - completes Level 4, the executable single-agent loop).

Slices 1-2 gave the pieces: the routing brain (:mod:`loop.controller`), durable state (:mod:`loop.store`),
and the assurance observer (:mod:`loop.observe`). This module is the cycle that runs them:

    while not terminal:
        directive = next_directive(state)     # what the executor should do in this phase + constraints
        observation = <executor acts>  OR  <cs_assure observes>   (mechanical at the verify phases)
        apply(state, observation)             # route + transition
        persist(state)

Because the EXECUTOR here is the LLM/agent (a Python loop cannot run the reasoning), the driver is
parameterised by an ``executor`` callback: for the judgment phases it hands the executor a
:class:`Directive` and takes back the executor's :class:`Observation`; for the VERIFY phases
(OBSERVE / VERIFY) it does NOT ask the executor - it reads repository state itself via the assurance
plane (:func:`loop.observe.observe_and_apply`). Fully deterministic given its callbacks; stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from loop.controller import LoopState, Observation, Phase, Transition, apply
from loop.observe import observe_and_apply
from loop.store import save
from loop.tasks import TaskStatus

# The phases where the loop reads repository state ITSELF (via cs_assure) rather than asking the
# executor - the mechanical observation points.
_VERIFY_PHASES: frozenset[Phase] = frozenset({Phase.OBSERVE, Phase.VERIFY})

# Per-phase instruction for the executor. Deliberately imperative + minimal; the task graph (slice 4)
# will supply the concrete target + allowed paths.
_PHASE_ACTION: dict[Phase, str] = {
    Phase.RECEIVE_GOAL: "Restate the goal and its scope in one line.",
    Phase.RECON: "Inspect the repository and current state relevant to the goal (read-only).",
    Phase.DEFINE_SUCCESS: "Write explicit, checkable success criteria for the goal.",
    Phase.PLAN: "Produce a bounded plan to reach the success criteria.",
    Phase.DECOMPOSE: "Break the plan into owned tasks, each with allowed paths + success criteria.",
    Phase.ASSIGN: "Assign the next ready task to an owner (self or a delegated agent).",
    Phase.EXECUTE: "Make the smallest coherent change for the active task.",
    Phase.OBSERVE: "The loop runs the assurance gate (cs_assure verify + doclint) and reads the result.",
    Phase.DIAGNOSE: "Classify the observation; the controller routes advance / revise / replan / escalate.",
    Phase.REVIEW: "Review the change against the success criteria (adversarially).",
    Phase.INTEGRATE: "Integrate the reviewed change (commit / open PR).",
    Phase.VERIFY: "The loop verifies the completion criteria end-to-end (cs_assure verify).",
}

# Callback types. The executor turns a Directive into its judged Observation. The observer reads the
# repo via the assurance plane and routes it (defaults to observe_and_apply).
Executor = Callable[["Directive"], Observation]
Observer = Callable[[LoopState, Path, str], Transition]


class LoopDriverError(Exception):
    """The executor returned something that is not an Observation (fail-closed)."""


@dataclass(frozen=True)
class Directive:
    """What the executor should do next, and the constraints on doing it."""

    phase: str
    action: str
    terminal: bool
    budget_remaining: int
    allowed_paths: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    termination_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase, "action": self.action, "terminal": self.terminal,
            "budget_remaining": self.budget_remaining, "allowed_paths": self.allowed_paths,
            "success_criteria": self.success_criteria, "termination_reason": self.termination_reason,
        }


def _remaining(budgets: dict[str, Any]) -> int:
    try:
        return max(0, int(budgets.get("max_attempts", 0)) - int(budgets.get("total_attempts", 0)))
    except (TypeError, ValueError):
        return 0


def _active_task_paths(state: LoopState) -> list[str]:
    """Allowed paths for the currently-active task, if the task graph names one (slice 4 populates it;
    until then this is empty = unconstrained)."""
    for task in state.task_graph:
        if isinstance(task, dict) and task.get("status") == TaskStatus.ACTIVE.value:
            paths = task.get("allowed_paths")
            if isinstance(paths, list):
                return [p for p in paths if isinstance(p, str)]
    return []


def next_directive(state: LoopState) -> Directive:
    """Emit the next action for the executor, given the current phase + constraints. PURE."""
    phase = state.current_phase
    if state.is_terminal:
        return Directive(phase=phase.value, action="(loop is terminal - nothing to do)", terminal=True,
                         budget_remaining=_remaining(state.budgets),
                         success_criteria=list(state.success_criteria),
                         termination_reason=state.termination_reason)
    return Directive(
        phase=phase.value,
        action=_PHASE_ACTION.get(phase, "Proceed."),
        terminal=False,
        budget_remaining=_remaining(state.budgets),
        allowed_paths=_active_task_paths(state),
        success_criteria=list(state.success_criteria),
    )


def advance(state: LoopState, repo_root: Path, executor: Executor, *, base: str = "main",
            observer: Observer = observe_and_apply, store_path: Path | None = None) -> Transition | None:
    """Run ONE loop cycle. Returns the taken Transition, or None if the state was already terminal.
    At a VERIFY phase the loop observes the repo itself (cs_assure); otherwise it asks the executor."""
    if state.is_terminal:
        return None
    directive = next_directive(state)
    if state.current_phase in _VERIFY_PHASES:
        transition = observer(state, repo_root, base)
    else:
        observation = executor(directive)
        if not isinstance(observation, Observation):
            raise LoopDriverError(
                f"executor returned {type(observation).__name__}, not an Observation "
                f"(phase {state.current_phase.value})")
        transition = apply(state, observation, note=f"executor@{state.current_phase.value}")
    if store_path is not None:
        save(state, store_path)
    return transition


def run(state: LoopState, repo_root: Path, executor: Executor, *, base: str = "main",
        max_steps: int = 100, observer: Observer = observe_and_apply,
        store_path: Path | None = None) -> LoopState:
    """Drive the loop to a terminal phase (FINALIZE / ESCALATED / STOPPED) or until ``max_steps`` cycles
    - a hard safety cap INDEPENDENT of the attempt budget, so a mis-set budget can never spin forever.
    Persists after each cycle when ``store_path`` is given, so a crash resumes mid-loop."""
    steps = 0
    while not state.is_terminal and steps < max_steps:
        advance(state, repo_root, executor, base=base, observer=observer, store_path=store_path)
        steps += 1
    if not state.is_terminal:
        # Hit the hard cap without a controller-driven terminal state: stop fail-closed, OVERWRITING any
        # stale termination_reason (a resumed non-terminal state may carry a leftover reason - the cap
        # must fire regardless, or run() could return a non-terminal state and the caller spins forever).
        state.current_phase = Phase.STOPPED
        state.termination_reason = f"driver hard step cap ({max_steps}) reached"
        if store_path is not None:
            save(state, store_path)
    return state
