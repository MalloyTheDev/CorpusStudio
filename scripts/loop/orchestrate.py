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
from loop.integrate import (
    GhRunner,
    IntegrateError,
    ci_observation,
    head_bound_merge,
    merge_gate,
    observe_ci,
)
from loop.observe import CsAssureRunner, LoopObserveError, _run_cs_assure, observe, record_evidence
from loop.review import Reviewer, review, review_observation
from loop.router import AgentRunner, PathVerifier, aggregate_observation, dispatch_wave
from loop.store import save
from loop.tasks import LoopTaskError, TaskStatus, parse_tasks, ready_tasks, set_status, status_for

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
    # An INDEPENDENT worktree-diff source for boundary enforcement. When set, a delegated agent's ownership
    # boundary is checked against the real diff, not its self-reported changed_paths. None = trust-based.
    verify_paths: PathVerifier | None = None
    gh_runner: GhRunner | None = None
    pr_ref: str | None = None
    dangerous: bool = False
    # The commit the loop validated + intends to merge (what `impact` analyzed). If set, INTEGRATE refuses
    # to merge a PR whose observed head is not this commit - so the human-review gate (computed locally)
    # can never be bound to a different remote head than the one being merged. None = single-writer trust.
    expected_head: str | None = None
    # CI checks that MUST have reported green before a merge (e.g. ("python-engine", "assurance")). An
    # all-green rollup still MISSING one of these is treated as not-yet-settled (HOLD), so the loop never
    # merges in the window before a required check has created its run. Empty = trust whatever reported.
    # Names must EXACTLY match the check names gh reports; a name that never matches HOLDs indefinitely
    # (fail-safe: never merges) and the HOLD reason names the missing check so the misconfig is visible.
    required_checks: tuple[str, ...] = ()
    multi_agent: bool = False
    couplings: tuple[DocCoupling, ...] = DEFAULT_COUPLINGS
    run_cs_assure: CsAssureRunner = _run_cs_assure
    store_path: Path | None = None
    ledger_path: Path | None = None  # cross-goal learning ledger (seed at start, record at terminal)

    def __post_init__(self) -> None:
        # verify_paths only takes effect on the DELEGATED multi-agent EXECUTE path. Setting it while
        # multi_agent is off would silently do nothing (the single-agent executor runs unbounded), so an
        # operator could believe boundary verification is active when it is not - fail LOUD instead.
        if self.verify_paths is not None and not self.multi_agent:
            raise LoopOrchestrateError("verify_paths requires multi_agent=True (it only bounds delegated agents)")


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


@dataclass(frozen=True)
class PolicyAssessment:
    """The merge-gate input from ``cs_assure impact``: the fired obligations, whether the TRUSTED-BASE
    policy was available (a candidate-only assessment must not authorize an autonomous merge, since the
    candidate could have weakened the policy unseen), and the effective policy digest that was assessed."""

    fired_obligations: list[dict[str, str]]
    base_policy_available: bool
    effective_policy_digest: str | None


def _impact_assessment(ctx: LoopContext) -> PolicyAssessment:
    """The change's policy assessment (via cs_assure impact): fired obligations as ``{id, severity}`` plus
    the trusted-base-policy availability. FAIL-CLOSED: a non-zero exit, an unusable record, or a malformed
    fired-obligations list raises, so an UNCOMPUTABLE assessment is never conflated with 'safe to merge';
    a MISSING / non-True base_policy_available reads as NOT available (a candidate-only assessment)."""
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
        # A well-formed impact record ALWAYS has a (possibly empty) list here. A missing / non-list field
        # is a malformed / schema-drifted record - fail CLOSED (do not conflate 'uncomputable' with 'none
        # fired' at the merge button), like the payload check above.
        raise LoopOrchestrateError("cs_assure impact record has no fired_obligations list")
    items: list[dict[str, str]] = []
    for o in fired:
        if not isinstance(o, dict) or not isinstance(o.get("id"), str):
            raise LoopOrchestrateError(f"cs_assure impact has a malformed fired obligation: {o!r}")
        items.append({"id": o["id"], "severity": str(o.get("severity", ""))})
    provenance = data.get("provenance") if isinstance(data, dict) else None
    digest = provenance.get("policy_digest") if isinstance(provenance, dict) else None
    return PolicyAssessment(
        fired_obligations=items,
        base_policy_available=payload.get("base_policy_available") is True,  # missing/non-True -> not available
        effective_policy_digest=digest if isinstance(digest, str) else None,
    )


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


def _verify_completeness(state: LoopState, ctx: LoopContext, observation: Observation,
                         reason: str) -> tuple[Observation, str]:
    """VERIFY on a green gate: a green gate is NOT 'done' - prove goal COMPLETION. MANDATORY: with no
    completeness evaluator, completion is UNPROVEN -> escalate (a green gate is never the implicit
    definition of done). With a critic, the typed evidence-bound criteria must all be met; any gap folds
    into correction tasks (CHANGES_REQUESTED) or escalates a residual human decision. Returns the
    (observation, reason) - the passed-through green result when completion is proven."""
    if ctx.critic is None:
        return (Observation.AUTHORIZATION_REQUIRED,
                "gate green but no completeness evaluator - goal completion is unproven")
    verdict = check_completeness(state, ctx.critic)
    if not verdict.complete:
        _append_completeness_tasks(state, verdict)
        return completeness_observation(verdict), verdict.note
    return observation, reason  # completion proven - keep the gate's green result


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
            # ONE task per executor call: a single executor result must NOT close several tasks (the
            # remaining unbounded tasks are drained on subsequent EXECUTE cycles). Mark it ACTIVE and
            # recompute the directive so the executor is explicitly given the task it is run for; then map
            # its result via the shared status_for (SUCCESS->DONE, PROGRESS->PENDING, else FAILED).
            task = unbounded_ready[0]
            set_status(state, task.id, TaskStatus.ACTIVE)
            observation = ctx.executor(state, next_directive(state))
            set_status(state, task.id, status_for(observation))
            return PhaseResult(observation, f"executed self-owned task {task.id!r}")
        # DRAIN: dispatch waves until no ready task remains (deps unlock across waves), stopping on a
        # failure. The router marks each wave ACTIVE->DONE/FAILED; we never close a task no agent ran.
        outcomes = []
        for _ in range(_MAX_WAVES):
            wave = dispatch_wave(state, ctx.agent_runner, verify_paths=ctx.verify_paths)
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

    if state.task_graph:
        # SINGLE-AGENT: execute each currently-ready task via the executor, binding execution to that task
        # (mark it ACTIVE, recompute the directive) and marking it via status_for. Newly-unblocked or
        # still-in-PROGRESS tasks are handled on the next EXECUTE cycle (CHANGES_REQUESTED -> re-assign),
        # so the loop NEVER advances / finalizes with a task left PENDING (external review #4).
        for task in ready_tasks(parse_tasks(state.task_graph)):
            set_status(state, task.id, TaskStatus.ACTIVE)
            observation = ctx.executor(state, next_directive(state))
            status = status_for(observation)
            set_status(state, task.id, status)
            if status is TaskStatus.FAILED:
                return PhaseResult(observation, f"task {task.id!r} failed")
        if _all_done(state):
            return PhaseResult(Observation.SUCCESS, "all tasks executed")
        remaining = sorted(t.id for t in parse_tasks(state.task_graph) if t.status is not TaskStatus.DONE)
        return PhaseResult(Observation.CHANGES_REQUESTED, f"{len(remaining)} task(s) remain: {remaining}")

    return PhaseResult(ctx.executor(state, directive), "executed the change")


def _integrate(state: LoopState, ctx: LoopContext, directive: Directive) -> PhaseResult:
    if ctx.gh_runner is None or ctx.pr_ref is None:
        return PhaseResult(ctx.executor(state, directive), "integrated the change")
    snapshot = observe_ci(ctx.gh_runner, ctx.pr_ref)  # ONE read: checks + the head they ran against
    observation, reason = ci_observation(snapshot.status, required=frozenset(ctx.required_checks))
    if observation is Observation.PROGRESS:
        # CI has not settled (pending / not yet reported / a required check missing) - HOLD at INTEGRATE;
        # never advance PAST the merge gate on an unsettled CI (that would merge before CI validates).
        return PhaseResult(Observation.HOLD, f"CI not settled: {reason}")
    if observation is not Observation.SUCCESS:
        return PhaseResult(observation, reason)  # CI failing -> route to fix
    if ctx.expected_head is not None and snapshot.head_sha is not None \
            and snapshot.head_sha != ctx.expected_head:
        # The remote PR head is NOT the commit we validated locally (the merge-gate obligations were
        # computed against ctx.expected_head's tree). Do NOT merge a head whose obligations we never
        # evaluated - HOLD to re-sync (re-observe + re-run impact on the new head).
        return PhaseResult(Observation.HOLD,
                           f"PR head {snapshot.head_sha[:12]} != validated commit {ctx.expected_head[:12]}; re-syncing")
    try:
        impact = _impact_assessment(ctx)
    except LoopOrchestrateError as exc:
        return PhaseResult(Observation.AUTHORIZATION_REQUIRED, f"obligations uncomputable; escalating: {exc}")
    if not impact.base_policy_available:
        # The TRUSTED-BASE policy could not be loaded (shallow clone / no merge base / read failure), so
        # this is a CANDIDATE-ONLY assessment: the candidate could have weakened the policy unseen. A
        # candidate-only assessment must NOT authorize an autonomous merge - escalate for a human (the only
        # exception is an explicit bootstrap reviewed under the repo's pre-existing controls, done off-band).
        return PhaseResult(Observation.AUTHORIZATION_REQUIRED,
                           "trusted-base policy unavailable; a candidate-only assessment cannot authorize an autonomous merge")
    gate = merge_gate(impact.fired_obligations, dangerous=ctx.dangerous)
    if not gate.authorized:
        return PhaseResult(gate.observation, gate.reason)  # self-modify / worker / policy-gated -> escalate
    # Merge BOUND to the exact head CI validated: a commit pushed since is not merged blind (it HOLDs to
    # re-observe the new head), so we never merge a diff CI never saw.
    merge_obs, merge_reason = head_bound_merge(ctx.gh_runner, ctx.pr_ref, snapshot.head_sha)
    return PhaseResult(merge_obs, merge_reason)


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
        record_evidence(state, result)  # structured evidence for the semantic completeness check
        observation, reason = result.observation, result.reason
        if phase is Phase.OBSERVE:
            gaps = stale_docs(_changed_paths(ctx), ctx.couplings)
            if observation is Observation.SUCCESS and gaps:
                observation, reason = docs_observation(gaps)
        elif phase is Phase.VERIFY and observation is Observation.SUCCESS:
            observation, reason = _verify_completeness(state, ctx, observation, reason)
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
    except (LoopObserveError, CompletenessError, IntegrateError, OSError) as exc:
        # An unusable assurance plane, a misbehaving completeness critic, an unusable CI/gh read at
        # INTEGRATE, OR an injected effect that RAISES (e.g. a subprocess spawn failure - OSError:
        # EAGAIN/EMFILE/ENOMEM - from gh_runner / run_cs_assure) escalates to a human (persisted). It
        # fails CLOSED (no merge) and lands durably at ESCALATED, never as an uncaught crash mid-run.
        state.current_phase = Phase.ESCALATED
        state.termination_reason = f"unrecoverable: {type(exc).__name__}: {exc}"
        if ctx.store_path is not None:
            save(state, ctx.store_path)
        return Transition(Decision.ESCALATE, Phase.ESCALATED, state.termination_reason,
                          "escalated: assurance plane / critic / CI read / effect unusable")
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
