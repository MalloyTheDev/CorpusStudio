"""The capstone: one integrated loop that composes every module (controller slice 8).

Slices 1-7 built the parts (controller/store/observe/driver/tasks/router/review/integrate/docs) and proved
each in isolation. This wires them into ONE runnable loop where each phase does its real work:

    DECOMPOSE  -> the executor proposes a task graph; tasks.parse_tasks validates it (bad graph -> WRONG_PLAN)
    ASSIGN     -> tasks.assign_next picks the next ready task (its ownership boundary flows to the directive)
    EXECUTE    -> single executor, OR router.dispatch_wave for a parallel-safe multi-agent wave
    OBSERVE    -> observe (cs_assure) + docs.stale_docs (docs-freshness); on success, close the active task
    REVIEW     -> review.review folds findings into correction tasks (or the executor reviews)
    INTEGRATE  -> integrate.observe_ci + merge_gate (product auto-merges; self-modify/worker/danger escalate)
    VERIFY     -> observe (cs_assure) end-to-end
    (goal/recon/define/plan/diagnose) -> the executor (the LLM) does the reasoning

Every EFFECT is an injected callback on :class:`LoopContext` (executor / reviewer / agent runner / gh /
cs_assure), so the whole integrated loop runs deterministically in tests without any of them. stdlib-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from loop.controller import LoopState, Observation, Phase, Transition, apply, attempt_fingerprint
from loop.docs import DEFAULT_COUPLINGS, DocCoupling, docs_observation, stale_docs
from loop.driver import Directive, next_directive
from loop.integrate import GhRunner, ci_observation, merge_gate, observe_ci
from loop.observe import CsAssureRunner, _run_cs_assure, observe
from loop.review import Reviewer, review, review_observation
from loop.router import AgentRunner, aggregate_observation, dispatch_wave
from loop.store import save
from loop.tasks import LoopTaskError, TaskStatus, assign_next, is_complete, parse_tasks, set_status

# The executor (the LLM/agent) acts ON the state (it may set the task graph, make edits) and returns its
# judged Observation - a richer signature than driver.Executor, which only sees the directive.
PhaseExecutor = Callable[[LoopState, Directive], Observation]


@dataclass
class LoopContext:
    """The injected effects + config for one integrated loop run."""

    repo_root: Path
    executor: PhaseExecutor
    base: str = "main"
    reviewer: Reviewer | None = None
    agent_runner: AgentRunner | None = None
    gh_runner: GhRunner | None = None
    pr_ref: str | None = None
    dangerous: bool = False
    multi_agent: bool = False
    couplings: tuple[DocCoupling, ...] = DEFAULT_COUPLINGS
    run_cs_assure: CsAssureRunner = _run_cs_assure
    store_path: Path | None = None


@dataclass(frozen=True)
class PhaseResult:
    observation: Observation
    note: str
    fingerprint: str | None = None
    evidence: str | None = None


def _changed_paths(ctx: LoopContext) -> list[str]:
    """Best-effort change set for docs-freshness. A changeset failure yields [] (docs-freshness is
    advisory - its absence must not block the mechanical observation)."""
    try:
        _code, out, _err = ctx.run_cs_assure(ctx.repo_root, "changeset", "--base", ctx.base)
        payload = json.loads(out).get("payload", {})
    except (ValueError, RecursionError, KeyError, TypeError):
        return []
    entries = payload.get("changed_paths") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []
    return [c["path"] for c in entries if isinstance(c, dict) and isinstance(c.get("path"), str)]


def _fired_obligations(ctx: LoopContext) -> list[str]:
    """The obligation ids the change fires (via cs_assure impact) - the input to the merge gate."""
    try:
        _code, out, _err = ctx.run_cs_assure(ctx.repo_root, "impact", "--base", ctx.base)
        payload = json.loads(out).get("payload", {})
    except (ValueError, RecursionError, KeyError, TypeError):
        return []
    fired = payload.get("fired_obligations") if isinstance(payload, dict) else None
    if not isinstance(fired, list):
        return []
    return [o["id"] for o in fired if isinstance(o, dict) and isinstance(o.get("id"), str)]


def _active_task_id(state: LoopState) -> str | None:
    for task in state.task_graph:
        if isinstance(task, dict) and task.get("status") == TaskStatus.ACTIVE.value:
            tid = task.get("id")
            return tid if isinstance(tid, str) else None
    return None


def _dispatch(state: LoopState, ctx: LoopContext) -> PhaseResult:
    phase = state.current_phase
    directive = next_directive(state)

    if phase is Phase.DECOMPOSE:
        observation = ctx.executor(state, directive)  # the executor is expected to set state.task_graph
        try:
            parse_tasks(state.task_graph)
        except LoopTaskError as exc:
            return PhaseResult(Observation.WRONG_PLAN, f"invalid task graph: {exc}")
        return PhaseResult(observation, "decomposed into a valid task graph")

    if phase is Phase.ASSIGN:
        if not state.task_graph:
            return PhaseResult(Observation.SUCCESS, "no task graph; proceeding single-track")
        if is_complete(parse_tasks(state.task_graph)):
            return PhaseResult(Observation.SUCCESS, "all tasks complete")
        task = assign_next(state)
        if task is not None:
            return PhaseResult(Observation.SUCCESS, f"assigned task {task.id!r}")
        return PhaseResult(Observation.OWNERSHIP_COLLISION, "no task ready (blocked/contended); rescheduling")

    if phase is Phase.EXECUTE:
        if ctx.multi_agent and ctx.agent_runner is not None and state.task_graph:
            outcomes = dispatch_wave(state, ctx.agent_runner)
            observation, note = aggregate_observation(outcomes)
            return PhaseResult(observation, note)
        return PhaseResult(ctx.executor(state, directive), "executed the active task")

    if phase in (Phase.OBSERVE, Phase.VERIFY):
        result = observe(ctx.repo_root, ctx.base, run_cs_assure=ctx.run_cs_assure)
        observation, reason = result.observation, result.reason
        if phase is Phase.OBSERVE:
            gaps = stale_docs(_changed_paths(ctx), ctx.couplings)
            if observation is Observation.SUCCESS and gaps:
                observation, reason = docs_observation(gaps)
            if observation is Observation.SUCCESS:  # the active task passed the gate -> close it
                active = _active_task_id(state)
                if active is not None:
                    set_status(state, active, TaskStatus.DONE, evidence=result.record_digest)
        fingerprint = None
        if observation not in (Observation.SUCCESS, Observation.PROGRESS) and result.change_set_fingerprint:
            fingerprint = attempt_fingerprint(f"{observation.value}:{reason}", result.change_set_fingerprint)
        return PhaseResult(observation, reason, fingerprint=fingerprint, evidence=result.record_digest)

    if phase is Phase.REVIEW:
        if ctx.reviewer is not None:
            review_result = review(state, ctx.reviewer, reviewed_task_id=_active_task_id(state))
            return PhaseResult(review_observation(review_result), f"review: {review_result.verdict.value}")
        return PhaseResult(ctx.executor(state, directive), "reviewed the change")

    if phase is Phase.INTEGRATE:
        if ctx.gh_runner is not None and ctx.pr_ref is not None:
            status = observe_ci(ctx.gh_runner, ctx.pr_ref)
            observation, reason = ci_observation(status)
            if observation is Observation.SUCCESS:
                gate = merge_gate(_fired_obligations(ctx), dangerous=ctx.dangerous)
                return PhaseResult(gate.observation, gate.reason)
            return PhaseResult(observation, reason)
        return PhaseResult(ctx.executor(state, directive), "integrated the change")

    # RECEIVE_GOAL / RECON / DEFINE_SUCCESS / PLAN / DIAGNOSE - the executor reasons.
    return PhaseResult(ctx.executor(state, directive), directive.action)


def step(state: LoopState, ctx: LoopContext) -> Transition | None:
    """Run ONE integrated cycle: dispatch the current phase to its module/effect, record any sealed
    evidence, route through the controller, and persist. Returns None if already terminal."""
    if state.is_terminal:
        return None
    result = _dispatch(state, ctx)
    if result.evidence is not None and result.evidence not in state.assurance_records:
        state.assurance_records.append(result.evidence)
    transition = apply(state, result.observation, fingerprint=result.fingerprint, note=result.note)
    if ctx.store_path is not None:
        save(state, ctx.store_path)
    return transition


def run_loop(state: LoopState, ctx: LoopContext, *, max_steps: int = 200) -> LoopState:
    """Drive the fully-integrated loop to a terminal phase or a HARD step cap (independent of the attempt
    budget). Persists after each cycle when a store_path is set (crash-resumable)."""
    steps = 0
    while not state.is_terminal and steps < max_steps:
        step(state, ctx)
        steps += 1
    if not state.is_terminal:
        state.current_phase = Phase.STOPPED
        state.termination_reason = f"orchestrator hard step cap ({max_steps}) reached"
        if ctx.store_path is not None:
            save(state, ctx.store_path)
    return state
