"""The deterministic controller kernel: phases, the failure taxonomy, and the bounded transition.

Design invariants (mirror the assurance kernel so the two planes share a discipline):
  * stdlib-only; runs under any ``python3``; imports nothing from ``corpus_studio`` and no torch.
  * every transition is a PURE function of ``(state, observation)`` - deterministic and testable; no
    wall-clock, randomness, or I/O in the transition itself.
  * FAIL-CLOSED: an unrecognised observation ESCALATES; it never silently advances the loop.
  * BOUNDED: a total-attempt budget caps the loop (a non-positive / missing / unparseable cap fails
    CLOSED, it does NOT mean "unlimited"), and a re-entering decision that repeats a known-failed
    ``(failure, approach)`` fingerprint escalates instead of retrying - "same failure, same approach
    -> do not retry".
  * human approval is RETAINED: ``AUTHORIZATION_REQUIRED`` / ``POLICY_BLOCK`` escalate to a human; a
    worker-lineage impact leaves the ordinary loop for the (human-gated) worker workflow.

This is slice 1 (L3 -> L4): the routing brain. Action emission bound to a task graph, agent ownership,
and the ``cs_assure`` observe/verify wiring are later slices; the durable :class:`LoopState` already
carries the fields those slices fill in so its on-disk shape does not churn.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    """The operational state machine. FINALIZE / ESCALATED / STOPPED are terminal."""

    RECEIVE_GOAL = "RECEIVE_GOAL"
    RECON = "RECON"
    DEFINE_SUCCESS = "DEFINE_SUCCESS"
    PLAN = "PLAN"
    DECOMPOSE = "DECOMPOSE"
    ASSIGN = "ASSIGN"
    EXECUTE = "EXECUTE"
    OBSERVE = "OBSERVE"
    DIAGNOSE = "DIAGNOSE"
    REVIEW = "REVIEW"
    INTEGRATE = "INTEGRATE"
    VERIFY = "VERIFY"
    FINALIZE = "FINALIZE"      # terminal: success, completion criteria met
    ESCALATED = "ESCALATED"    # terminal: needs a human (blocker / authorization / worker workflow)
    STOPPED = "STOPPED"        # terminal: budget exhausted or a repeated-failure dead end


# The forward pipeline used when a step SUCCEEDS. The loop is not purely linear - DIAGNOSE branches -
# but a clean step advances one position along this spine.
_PIPELINE: tuple[Phase, ...] = (
    Phase.RECEIVE_GOAL, Phase.RECON, Phase.DEFINE_SUCCESS, Phase.PLAN, Phase.DECOMPOSE,
    Phase.ASSIGN, Phase.EXECUTE, Phase.OBSERVE, Phase.DIAGNOSE, Phase.REVIEW,
    Phase.INTEGRATE, Phase.VERIFY, Phase.FINALIZE,
)
TERMINAL: frozenset[Phase] = frozenset({Phase.FINALIZE, Phase.ESCALATED, Phase.STOPPED})


class Observation(str, Enum):
    """The CLASSIFIED result of an executed step - the failure taxonomy plus the two non-failures.

    The executor (LLM/agent) runs the step and reports a raw result; classification into one of these
    is what lets the controller ROUTE each failure differently instead of blindly re-running."""

    SUCCESS = "SUCCESS"                              # the step met its success criteria
    PROGRESS = "PROGRESS"                            # forward movement, not yet complete
    SYNTAX_FAILURE = "SYNTAX_FAILURE"
    TYPE_FAILURE = "TYPE_FAILURE"
    TEST_REGRESSION = "TEST_REGRESSION"
    DEPENDENCY_FAILURE = "DEPENDENCY_FAILURE"        # a missing/incompatible package - code-fixable
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"      # broken venv / host / GPU - usually not code-fixable
    CONTRACT_DRIFT = "CONTRACT_DRIFT"                # schemas/TS/count assertions out of sync
    WRONG_HYPOTHESIS = "WRONG_HYPOTHESIS"            # the fix theory was wrong; a new one is needed
    WRONG_PLAN = "WRONG_PLAN"                        # the decomposition itself is wrong
    OWNERSHIP_COLLISION = "OWNERSHIP_COLLISION"      # two tasks/agents contend for the same files
    POLICY_BLOCK = "POLICY_BLOCK"                    # an obligation/honesty invariant forbids this
    AUTHORIZATION_REQUIRED = "AUTHORIZATION_REQUIRED"  # credential / dangerous / irreversible / release
    WORKER_LINEAGE_IMPACT = "WORKER_LINEAGE_IMPACT"  # worker bytes changed -> fresh wheel/env workflow
    NONDETERMINISTIC_FAILURE = "NONDETERMINISTIC_FAILURE"  # flaky; a bounded retry may pass
    CHANGES_REQUESTED = "CHANGES_REQUESTED"          # a reviewer found real issues -> correction tasks
    HOLD = "HOLD"                                    # wait on an external condition (e.g. CI) - do NOT advance


# Obligations that REQUIRE A HUMAN before a change can be admitted - the loop can NEVER self-admit these
# (.claude/rules/{assurance,loop-controller}-self-modify.md). Single source of truth shared by observe
# (OBSERVE/VERIFY classification) and integrate (the merge gate) so the two planes cannot drift; a test
# pins every id to the obligations policy. Note: observe routes ``worker-closure`` to its own
# WORKER_LINEAGE_IMPACT signal (still human-gated, just a distinct label), so it peels that one off before
# mapping the rest to AUTHORIZATION_REQUIRED - the SET is shared, the per-id labelling is each plane's.
HUMAN_GATED_OBLIGATIONS: frozenset[str] = frozenset({
    "sealed-research", "assurance-self-modify", "worker-closure", "loop-controller-self-modify",
})


class Decision(str, Enum):
    """How the controller routes an observation."""

    ADVANCE = "ADVANCE"                              # step succeeded -> next phase on the spine
    REVISE = "REVISE"                                # recoverable -> new hypothesis, re-EXECUTE
    REPLAN = "REPLAN"                                # the plan is wrong -> back to PLAN
    RESCHEDULE = "RESCHEDULE"                        # ownership conflict -> re-ASSIGN
    ESCALATE = "ESCALATE"                            # hard blocker / authorization -> human
    STOP = "STOP"                                    # budget exhausted / repeated dead end
    ENTER_WORKER_WORKFLOW = "ENTER_WORKER_WORKFLOW"  # leave the ordinary loop for the worker workflow
    HOLD = "HOLD"                                    # stay in the CURRENT phase, waiting on an external condition


# The base routing table: taxonomy -> decision BEFORE the budget/retry guards apply. A missing key is
# treated as ESCALATE by :func:`route` (fail-closed), so adding an Observation without a route cannot
# silently advance the loop.
_ROUTE: dict[Observation, Decision] = {
    Observation.SUCCESS: Decision.ADVANCE,
    Observation.PROGRESS: Decision.ADVANCE,
    Observation.SYNTAX_FAILURE: Decision.REVISE,
    Observation.TYPE_FAILURE: Decision.REVISE,
    Observation.TEST_REGRESSION: Decision.REVISE,
    Observation.DEPENDENCY_FAILURE: Decision.REVISE,
    Observation.CONTRACT_DRIFT: Decision.REVISE,
    Observation.WRONG_HYPOTHESIS: Decision.REVISE,
    Observation.NONDETERMINISTIC_FAILURE: Decision.REVISE,
    Observation.WRONG_PLAN: Decision.REPLAN,
    Observation.OWNERSHIP_COLLISION: Decision.RESCHEDULE,
    Observation.CHANGES_REQUESTED: Decision.RESCHEDULE,
    Observation.ENVIRONMENT_FAILURE: Decision.ESCALATE,
    Observation.POLICY_BLOCK: Decision.ESCALATE,
    Observation.AUTHORIZATION_REQUIRED: Decision.ESCALATE,
    Observation.WORKER_LINEAGE_IMPACT: Decision.ENTER_WORKER_WORKFLOW,
    Observation.HOLD: Decision.HOLD,
}

# Decisions that re-enter the loop (and therefore consume budget / are subject to the retry guard).
_RETRYING: frozenset[Decision] = frozenset({Decision.REVISE, Decision.REPLAN, Decision.RESCHEDULE})


def attempt_fingerprint(failure_signature: str, patch_signature: str) -> str:
    """A stable id for "this failure, addressed this way". Two attempts with the same fingerprint are
    the same dead end; the controller refuses to retry one it has already seen fail."""
    payload = f"{failure_signature}\x00{patch_signature}".encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass
class LoopState:
    """The durable loop runtime. Slice 1 fills the control fields; the task-graph / agent / evidence
    fields exist now so later slices populate them without changing the on-disk shape."""

    goal: str = ""
    goal_id: str = ""
    success_criteria: list[str] = field(default_factory=list)
    current_phase: Phase = Phase.RECEIVE_GOAL
    task_graph: list[dict] = field(default_factory=list)
    active_agents: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    failed_approaches: list[str] = field(default_factory=list)  # attempt fingerprints that failed
    budgets: dict = field(default_factory=lambda: {"total_attempts": 0, "max_attempts": 20})
    assurance_records: list[str] = field(default_factory=list)  # sealed cs_assure record digests
    review_state: dict = field(default_factory=dict)
    blockers: list[dict] = field(default_factory=list)
    termination_reason: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.current_phase in TERMINAL


@dataclass(frozen=True)
class Transition:
    """The pure result of routing one observation: what to decide, where to go, and why."""

    decision: Decision
    next_phase: Phase
    termination_reason: str | None
    note: str


def _as_int(value: object, default: int) -> int:
    """Coerce a durable/free-form budget value to int, defaulting on anything unparseable - a garbage
    budget must degrade to a fail-closed number, never raise out of the pure transition."""
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return default


def route(state: LoopState, observation: Observation, *, fingerprint: str | None = None) -> Transition:
    """Route one classified observation to a decision + next phase. PURE: no I/O, no mutation of
    ``state``. Apply the budget guard (a non-positive/missing/unusable cap or used >= cap -> STOP), then
    the repeated-approach guard (a re-entering decision that repeats a known-failed (failure, approach)
    fingerprint -> ESCALATE, because grinding the same approach is a dead end)."""
    if state.is_terminal:
        return Transition(Decision.STOP, state.current_phase, state.termination_reason,
                          "already terminal; no further transition")

    base = _ROUTE.get(observation, Decision.ESCALATE)  # fail-closed default
    if observation not in _ROUTE:
        return Transition(Decision.ESCALATE, Phase.ESCALATED,
                          f"unclassified observation {observation!r}",
                          "no route for observation; escalating fail-closed")

    if base in _RETRYING:
        max_attempts = _as_int(state.budgets.get("max_attempts"), 0)
        used = _as_int(state.budgets.get("total_attempts"), 0)
        # A non-positive / missing / unparseable cap is NOT "no cap" - it fails CLOSED (STOP). An
        # unbounded loop is exactly what this controller exists to prevent.
        if max_attempts <= 0 or used >= max_attempts:
            stop_reason = "attempt budget exhausted" if max_attempts > 0 else "no usable attempt budget"
            return Transition(Decision.STOP, Phase.STOPPED, stop_reason,
                              f"{used}/{max_attempts} attempts; stopping (fail-closed), not looping")
        # Repeated-approach guard: ANY re-entering decision (REVISE/REPLAN/RESCHEDULE) that repeats a
        # known-failed (failure, approach) fingerprint is a dead end -> escalate instead of grinding.
        if fingerprint is not None and fingerprint in state.failed_approaches:
            return Transition(Decision.ESCALATE, Phase.ESCALATED, "repeated failed approach",
                              "same failure + same approach already failed; a new approach is required")

    next_phase, reason, note = _phase_for(state.current_phase, base, observation)
    return Transition(base, next_phase, reason, note)


def _phase_for(current: Phase, decision: Decision, observation: Observation) -> tuple[Phase, str | None, str]:
    """Map a decision to the next phase (and terminal reason, if any)."""
    if decision is Decision.ADVANCE:
        nxt = _advance(current)
        if nxt is Phase.FINALIZE:
            return Phase.FINALIZE, "completion criteria satisfied", "advancing to FINALIZE"
        return nxt, None, f"advancing {current.value} -> {nxt.value}"
    if decision is Decision.REVISE:
        return Phase.EXECUTE, None, "revise hypothesis and re-execute"
    if decision is Decision.REPLAN:
        return Phase.PLAN, None, "plan is wrong; replanning"
    if decision is Decision.RESCHEDULE:
        return Phase.ASSIGN, None, "re-entering task assignment (a collision or new correction work to schedule)"
    if decision is Decision.HOLD:
        # Stay in the current phase, waiting on an external condition (e.g. CI). Does NOT charge the
        # retry budget - it is waiting, not retrying - so the caller polls / re-invokes when it may change.
        return current, None, "holding for an external condition"
    if decision is Decision.ENTER_WORKER_WORKFLOW:
        return Phase.ESCALATED, "worker-lineage impact -> worker workflow (human-gated)", \
            "worker bytes changed; leaving the ordinary loop"
    if decision is Decision.STOP:
        return Phase.STOPPED, "stopped", "stopping"
    # ESCALATE
    return Phase.ESCALATED, f"escalated on {observation.value}", "escalating to a human"


def _advance(current: Phase) -> Phase:
    idx = _PIPELINE.index(current)
    return _PIPELINE[min(idx + 1, len(_PIPELINE) - 1)]


def apply(state: LoopState, observation: Observation, *, fingerprint: str | None = None,
          note: str = "") -> Transition:
    """Route the observation AND commit it to ``state`` (the one mutating entry point): append the
    observation, charge the attempt budget for a re-entering decision, record a failed fingerprint,
    move to the next phase, and set the terminal reason. Returns the taken :class:`Transition`."""
    transition = route(state, observation, fingerprint=fingerprint)
    # Idempotent on a terminal state: route() already reports "no further transition", so a replay must
    # not keep growing the durable observation / failed-approach lists. The transition that FIRST enters
    # a terminal phase still commits, because current_phase is not yet terminal at that point.
    if state.is_terminal:
        return transition
    state.observations.append({
        "observation": observation.value,
        "decision": transition.decision.value,
        "from_phase": state.current_phase.value,
        "to_phase": transition.next_phase.value,
        "fingerprint": fingerprint,
        "note": note or transition.note,
    })
    if transition.decision in _RETRYING:
        state.budgets["total_attempts"] = _as_int(state.budgets.get("total_attempts"), 0) + 1
    # Record the dead-end fingerprint for exactly the decisions the guard consults (the re-entering
    # ones), so recording and the repeated-approach guard stay aligned - no REPLAN/RESCHEDULE entry can
    # pollute a later REVISE, and no re-entering dead end goes unrecorded.
    if transition.decision in _RETRYING and fingerprint is not None:
        if fingerprint not in state.failed_approaches:
            state.failed_approaches.append(fingerprint)
    state.current_phase = transition.next_phase
    if transition.termination_reason is not None and state.current_phase in TERMINAL:
        state.termination_reason = transition.termination_reason
    return transition
