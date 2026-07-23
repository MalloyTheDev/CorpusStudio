"""The capstone: one integrated loop that composes every module (controller slice 8).

Slices 1-7 built the parts (controller/store/observe/driver/tasks/router/review/integrate/docs) and proved
each in isolation. This wires them into ONE runnable loop where each phase does its real work:

    DECOMPOSE  -> the executor proposes a task graph; tasks.parse_tasks validates it (bad graph -> WRONG_PLAN)
    ASSIGN     -> a lightweight pass-through (the task graph is drained at EXECUTE / owned by the router)
    EXECUTE    -> single executor, OR router.dispatch_wave DRAINED across waves for multi-agent work
    OBSERVE    -> observe (cs_assure) + docs.stale_docs (docs-freshness)
    REVIEW     -> review.review folds findings into correction tasks (or the executor reviews)
    INTEGRATE  -> integrate.observe_ci: HOLD while CI is unsettled, escalate/merge via merge_gate on green
    VERIFY     -> observe (cs_assure) end-to-end
    (goal/recon/define/plan/diagnose) -> the executor (the LLM) does the reasoning

Every EFFECT is an injected callback on :class:`LoopContext` (executor / reviewer / agent runner / gh /
cs_assure). FAIL-CLOSED throughout: an unusable assurance plane escalates (never crashes the loop), an
uncomputable obligation set blocks the merge (never auto-merges), and CI that has not settled HOLDs at the
merge gate instead of advancing past it. stdlib-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from loop.completeness import (
    CompletenessError,
    CompletenessVerdict,
    Critic,
    check_completeness,
    completeness_correction_tasks,
    completeness_observation,
    record_outcome,
    seed_known_dead_ends,
)
from loop.controller import (
    Decision,
    LoopState,
    Observation,
    Phase,
    Transition,
    apply,
    attempt_fingerprint,
)
from loop.docs import DEFAULT_COUPLINGS, DocCoupling, docs_observation, stale_docs
from loop.driver import Directive, next_directive
from loop.integrate import GhRunner, ci_observation, merge_gate, observe_ci
from loop.observe import CsAssureRunner, LoopObserveError, _run_cs_assure, observe
from loop.review import Reviewer, review, review_observation
from loop.router import AgentRunner, aggregate_observation, dispatch_wave
from loop.store import save
from loop.tasks import LoopTaskError, TaskStatus, parse_tasks, ready_tasks, set_status

# The executor (the LLM/agent) acts ON the state (it may set the task graph, make edits) and returns its
# judged Observation - a richer signature than driver.Executor, which only sees the directive.
PhaseExecutor = Callable[[LoopState, Directive], Observation]

_MAX_WAVES = 100  # a hard bound on multi-agent wave draining, independent of the loop step cap.


class LoopOrchestrateError(Exception):
    """A phase handler produced a non-Observation, or an effect returned uncomputable evidence."""


@dataclass
class LoopContext:
    """The injected effects + config for one integrated loop run."""

    repo_root: Path
    executor: PhaseExecutor
    base: str = "main"
    reviewer: Reviewer | None = None
    critic: Critic | None = None
    agent_runner: AgentRunner | None = None
    gh_runner: GhRunner | None = None
    pr_ref: str | None = None
    dangerous: bool = False
    multi_agent: bool = False
    couplings: tuple[DocCoupling, ...] = DEFAULT_COUPLINGS
    run_cs_assure: CsAssureRunner = _run_cs_assure
    store_path: Path | None = None
    ledger_path: Path | None = None  # cross-goal learning ledger (seed at start, record at terminal)


@dataclass(frozen=True)
class PhaseResult:
    observation: Observation
    note: str
    fingerprint: str | None = None
    evidence: str | None = None


def _payload(out: str) -> dict:
    """Parse a cs_assure record's payload, tolerating any non-object shape (fail-closed to {})."""
    try:
        data = json.loads(out)
    except (ValueError, RecursionError):
        return {}
    if not isinstance(data, dict):
        return {}
    payload = data.get("payload")
    return payload if isinstance(payload, dict) else {}


def _changed_paths(ctx: LoopContext) -> list[str]:
    """Best-effort change set for docs-freshness (advisory - any failure yields [])."""
    try:
        _code, out, _err = ctx.run_cs_assure(ctx.repo_root, "changeset", "--base", ctx.base)
    except OSError:
        return []
    entries = _payload(out).get("changed_paths")
    if not isinstance(entries, list):
        return []
    return [c["path"] for c in entries if isinstance(c, dict) and isinstance(c.get("path"), str)]


def _fired_obligations(ctx: LoopContext) -> list[str]:
    """The obligation ids the change fires (via cs_assure impact) - the input to the merge gate. FAIL-
    CLOSED: a non-zero exit or an unusable record raises, so an UNCOMPUTABLE obligation set can never be
    conflated with 'no obligations fired' at the merge button."""
    code, out, err = ctx.run_cs_assure(ctx.repo_root, "impact", "--base", ctx.base)
    if code != 0:
        raise LoopOrchestrateError(f"cs_assure impact refused (exit {code}): {err.strip()[:120]}")
    try:
        data = json.loads(out)
    except (ValueError, RecursionError) as exc:
        raise LoopOrchestrateError(f"cs_assure impact produced no usable JSON: {exc}") from exc
    payload = data.get("payload") if isinstance(data, dict) else None
    if not isinstance(payload, dict):
        raise LoopOrchestrateError("cs_assure impact record has no payload object")
    fired = payload.get("fired_obligations")
    if not isinstance(fired, list):
        return []
    return [o["id"] for o in fired if isinstance(o, dict) and isinstance(o.get("id"), str)]


def _append_completeness_tasks(state: LoopState, verdict: CompletenessVerdict) -> None:
    """Fold unmet success criteria into correction tasks on the graph. Validate PER-TASK so one bad or
    duplicate criterion does not discard the whole batch."""
    existing = {t.get("id") for t in state.task_graph if isinstance(t, dict)}
    graph = list(state.task_graph)
    for task in completeness_correction_tasks(verdict):
        if task["id"] in existing:
            continue
        try:
            parse_tasks([*graph, task])  # incremental validation
        except LoopTaskError:
            continue  # skip an invalid task, keep the rest
        graph.append(task)
        existing.add(task["id"])
    state.task_graph = [t.to_dict() for t in parse_tasks(graph)]


def _all_done(state: LoopState) -> bool:
    return bool(state.task_graph) and all(
        isinstance(t, dict) and t.get("status") == TaskStatus.DONE.value for t in state.task_graph)


def _execute(state: LoopState, ctx: LoopContext, directive: Directive) -> PhaseResult:
    if ctx.multi_agent and ctx.agent_runner is not None and state.task_graph:
        # A ready task with NO declared ownership lane (empty allowed_paths) - e.g. an L8 completeness
        # gap - is the loop's OWN work: the router cannot enforce a boundary for it (every edit would be
        # a breach), so run it through the executor, never a bounded agent. Lane'd tasks go to the wave.
        unbounded_ready = [t for t in ready_tasks(parse_tasks(state.task_graph)) if not t.allowed_paths]
        if unbounded_ready:
            observation = ctx.executor(state, directive)
            status = (TaskStatus.DONE if observation in (Observation.SUCCESS, Observation.PROGRESS)
                      else TaskStatus.FAILED)
            for task in unbounded_ready:
                set_status(state, task.id, status)
            return PhaseResult(observation, "executed self-owned (unbounded) correction work")
        # DRAIN: dispatch waves until no ready task remains (deps unlock across waves), stopping on a
        # failure. The router marks each wave ACTIVE->DONE/FAILED; we never close a task no agent ran.
        outcomes = []
        for _ in range(_MAX_WAVES):
            wave = dispatch_wave(state, ctx.agent_runner)
            if not wave:
                break
            outcomes.extend(wave)
            if any(o.status is TaskStatus.FAILED for o in wave):
                break
        observation, note = aggregate_observation(outcomes)
        if observation in (Observation.SUCCESS, Observation.PROGRESS) and not _all_done(state):
            return PhaseResult(Observation.POLICY_BLOCK,
                               "task graph is stuck: ready tasks exhausted but not all DONE")
        return PhaseResult(observation, note)
    return PhaseResult(ctx.executor(state, directive), "executed the change")


def _integrate(state: LoopState, ctx: LoopContext, directive: Directive) -> PhaseResult:
    if ctx.gh_runner is None or ctx.pr_ref is None:
        return PhaseResult(ctx.executor(state, directive), "integrated the change")
    observation, reason = ci_observation(observe_ci(ctx.gh_runner, ctx.pr_ref))
    if observation is Observation.PROGRESS:
        # CI has not settled (pending / not yet reported) - HOLD at INTEGRATE; never advance PAST the
        # merge gate on an unsettled CI (that would FINALIZE unmerged / merge before CI validates).
        return PhaseResult(Observation.HOLD, f"CI not settled: {reason}")
    if observation is not Observation.SUCCESS:
        return PhaseResult(observation, reason)  # CI failing -> route to fix
    try:
        fired = _fired_obligations(ctx)
    except LoopOrchestrateError as exc:
        return PhaseResult(Observation.AUTHORIZATION_REQUIRED, f"obligations uncomputable; escalating: {exc}")
    gate = merge_gate(fired, dangerous=ctx.dangerous)
    if not gate.authorized:
        return PhaseResult(gate.observation, gate.reason)  # self-modify / worker / dangerous -> escalate
    code, _out, err = ctx.gh_runner("pr", "merge", ctx.pr_ref, "--squash")
    if code == 0:
        return PhaseResult(Observation.SUCCESS, "merged (authorized product change)")
    return PhaseResult(Observation.TEST_REGRESSION, f"authorized merge failed: {err.strip()[:120]}")


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
        return PhaseResult(Observation.SUCCESS, "ready to execute")

    if phase is Phase.EXECUTE:
        return _execute(state, ctx, directive)

    if phase in (Phase.OBSERVE, Phase.VERIFY):
        result = observe(ctx.repo_root, ctx.base, run_cs_assure=ctx.run_cs_assure)
        observation, reason = result.observation, result.reason
        if phase is Phase.OBSERVE:
            gaps = stale_docs(_changed_paths(ctx), ctx.couplings)
            if observation is Observation.SUCCESS and gaps:
                observation, reason = docs_observation(gaps)
        elif phase is Phase.VERIFY and observation is Observation.SUCCESS and ctx.critic is not None:
            # L8 self-correction: a green gate is not 'done' - the GOAL's success criteria must be MET.
            verdict = check_completeness(state, ctx.critic)
            if not verdict.complete:
                _append_completeness_tasks(state, verdict)
                observation, reason = completeness_observation(verdict), verdict.note
        fingerprint = None
        if observation not in (Observation.SUCCESS, Observation.PROGRESS) and result.change_set_fingerprint:
            fingerprint = attempt_fingerprint(f"{observation.value}:{reason}", result.change_set_fingerprint)
        return PhaseResult(observation, reason, fingerprint=fingerprint, evidence=result.record_digest)

    if phase is Phase.REVIEW:
        if ctx.reviewer is not None:
            # Goal-level review (the loop reviews the whole change, not a single task).
            review_result = review(state, ctx.reviewer, reviewed_task_id=None)
            return PhaseResult(review_observation(review_result), f"review: {review_result.verdict.value}")
        return PhaseResult(ctx.executor(state, directive), "reviewed the change")

    if phase is Phase.INTEGRATE:
        return _integrate(state, ctx, directive)

    # RECEIVE_GOAL / RECON / DEFINE_SUCCESS / PLAN / DIAGNOSE - the executor reasons.
    return PhaseResult(ctx.executor(state, directive), directive.action)


def step(state: LoopState, ctx: LoopContext) -> Transition | None:
    """Run ONE integrated cycle: dispatch the current phase to its module/effect, record any sealed
    evidence, route through the controller, and persist. Returns None if already terminal. FAIL-CLOSED:
    an unusable assurance plane escalates to ESCALATED; a non-Observation handler result is a hard error."""
    if state.is_terminal:
        return None
    try:
        result = _dispatch(state, ctx)
    except (LoopObserveError, CompletenessError) as exc:
        # An unusable assurance plane OR a misbehaving/erroring completeness critic escalates to a human
        # (persisted) - it never crashes the loop mid-run.
        state.current_phase = Phase.ESCALATED
        state.termination_reason = f"unrecoverable: {exc}"
        if ctx.store_path is not None:
            save(state, ctx.store_path)
        return Transition(Decision.ESCALATE, Phase.ESCALATED, state.termination_reason,
                          "escalated: assurance plane / critic unusable")
    if not isinstance(result.observation, Observation):
        raise LoopOrchestrateError(
            f"handler for {state.current_phase.value} returned {type(result.observation).__name__}, "
            "not an Observation")
    if result.evidence is not None and result.evidence not in state.assurance_records:
        state.assurance_records.append(result.evidence)
    transition = apply(state, result.observation, fingerprint=result.fingerprint, note=result.note)
    if ctx.store_path is not None:
        save(state, ctx.store_path)
    return transition


def run_loop(state: LoopState, ctx: LoopContext, *, max_steps: int = 200) -> LoopState:
    """Drive the fully-integrated loop to a terminal phase, a HOLD (paused on an external condition such
    as CI - the caller re-invokes when it may change), or a HARD step cap (independent of the attempt
    budget). Persists after each cycle when a store_path is set (crash-resumable)."""
    if ctx.ledger_path is not None:
        # CROSS-GOAL LEARNING: seed this loop with prior goals' dead ends. A malformed ledger escalates
        # (fail-closed) rather than running blind.
        try:
            seed_known_dead_ends(state, ctx.ledger_path)
        except CompletenessError as exc:
            state.current_phase = Phase.ESCALATED
            state.termination_reason = f"malformed learning ledger: {exc}"
            return state
    steps = 0
    held = False
    while not state.is_terminal and steps < max_steps:
        transition = step(state, ctx)
        steps += 1
        if transition is not None and transition.decision is Decision.HOLD:
            held = True  # paused waiting on an external condition; not a terminal state
            break
    if not state.is_terminal and not held:
        state.current_phase = Phase.STOPPED
        state.termination_reason = f"orchestrator hard step cap ({max_steps}) reached"
        if ctx.store_path is not None:
            save(state, ctx.store_path)
    if state.is_terminal and ctx.ledger_path is not None:
        record_outcome(state, ctx.ledger_path)  # this goal's dead ends feed the next
    return state
