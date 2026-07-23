"""CI / PR continuation (controller slice 7 - completes Level 7, CI/review/integration autonomy).

The outer loop past a local green gate: open a PR, OBSERVE CI, diagnose a CI failure into an Observation
so the loop patches + re-pushes, and MERGE only WHEN AUTHORIZED - a product change may auto-merge on
standing authorization, but a self-modifying / sealed-research / dangerous change ESCALATES to a human
(the BOOTSTRAP_SELF_MODIFIED boundary and the honesty invariants reach all the way to the merge button).

Deterministic core here (parse CI checks -> status -> Observation; the merge-authorization gate); the
``gh`` calls themselves are an INJECTED runner, exactly like every other effect in the loop. stdlib-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from loop.controller import Observation

# gh's check "bucket" (or a raw state/conclusion) normalised to pass / fail / pending.
_PASS = frozenset({"pass", "success", "neutral", "skipping", "skipped"})
_FAIL = frozenset({"fail", "failure", "cancel", "cancelled", "canceled", "timed_out",
                   "action_required", "startup_failure", "error"})

# A failing CI check name -> the observation, so a CI failure routes like its local-gate equivalent.
_CHECK_HINTS: tuple[tuple[tuple[str, ...], Observation], ...] = (
    (("ruff", "lint"), Observation.SYNTAX_FAILURE),
    (("mypy", "typecheck", "types"), Observation.TYPE_FAILURE),
    (("pytest", "test", "cov"), Observation.TEST_REGRESSION),
    (("web", "build", "tsc", "npm", "node"), Observation.DEPENDENCY_FAILURE),
)
# Obligations that ALWAYS forbid an autonomous merge - the change must be admitted by an INDEPENDENT human
# (the candidate/loop cannot self-verify a change to itself). worker-closure is here too: a worker-lineage
# change needs a fresh wheel/env via the human-gated worker workflow, the same invariant observe.py
# enforces by routing worker-closure to WORKER_LINEAGE_IMPACT. loop-controller-self-modify is here because
# a change to the loop controller must not be admitted by the loop's OWN merge gate (its rule #666).
_HUMAN_GATED = frozenset({
    "sealed-research", "assurance-self-modify", "worker-closure", "loop-controller-self-modify",
})
# Blocking obligations a green PRODUCT change discharges WITHOUT a human: they are satisfied by
# regeneration + the CI diff gate (contracts) or by test-enforced invariants (evaluation-honesty). Every
# OTHER blocking obligation defaults to human review (fail-closed), so a NEW blocking obligation added to
# the policy escalates automatically instead of silently auto-merging.
_AUTO_MERGEABLE = frozenset({"contracts", "evaluation-honesty"})
# Severities that do NOT by themselves gate a merge. Any obligation whose severity is not one of these
# (blocking, unknown, or absent) and is not on the auto-mergeable allowlist escalates - so a missing or
# non-canonical severity fails CLOSED (escalate) instead of silently authorizing.
_LOW_SEVERITY = frozenset({"info", "advisory"})

GhRunner = Callable[..., "tuple[int, str, str]"]


class IntegrateError(Exception):
    """gh produced no usable check data (fail-closed)."""


@dataclass(frozen=True)
class CiStatus:
    state: str  # "passing" | "failing" | "pending" | "none"
    passed: int
    total: int
    failing_checks: list[str] = field(default_factory=list)
    passing_checks: list[str] = field(default_factory=list)  # names of checks that reported and passed


@dataclass(frozen=True)
class CiSnapshot:
    """One CONSISTENT read of a PR's CI: the parsed check status AND the exact head commit those checks
    ran against, from a single ``gh pr view`` call - so a merge can be bound to that head (race-safe)."""

    status: CiStatus
    head_sha: str | None  # the commit the checks are for; None if gh did not report it


@dataclass(frozen=True)
class MergeGate:
    authorized: bool
    observation: Observation
    reason: str


def _bucket(check: dict[str, Any]) -> str:
    raw = str(check.get("bucket") or check.get("state") or check.get("conclusion")
              or check.get("status") or "").lower()
    if raw in _PASS:
        return "pass"
    if raw in _FAIL:
        return "fail"
    return "pending"


def parse_ci_checks(checks: Any) -> CiStatus:
    """Reduce a gh checks list to one status. A failure anywhere dominates, then pending, then pass;
    no checks at all is 'none' (nothing to satisfy). Fail-closed on a non-list payload."""
    if not isinstance(checks, list):
        raise IntegrateError(f"CI checks payload is not a list (got {type(checks).__name__})")
    items = [c for c in checks if isinstance(c, dict)]
    failing_names: list[str] = []
    passing_names: list[str] = []
    pending = 0
    for check in items:
        bucket = _bucket(check)
        # `gh pr checks` names a check `name`; a `statusCheckRollup` StatusContext uses `context`.
        name = str(check.get("name") or check.get("context") or "?")
        if bucket == "fail":
            failing_names.append(name)
        elif bucket == "pass":
            passing_names.append(name)
        else:
            pending += 1
    if not items:
        state = "none"
    elif failing_names:
        state = "failing"
    elif pending:
        state = "pending"
    else:
        state = "passing"
    return CiStatus(state=state, passed=len(passing_names), total=len(items),
                    failing_checks=failing_names, passing_checks=passing_names)


def ci_observation(status: CiStatus, *, required: "frozenset[str]" = frozenset()) -> tuple[Observation, str]:
    """Map a CI status to a loop Observation. Pending is PROGRESS (keep observing, do not treat 'not
    done' as done); a failure is classified by check name so it routes like the local gate. If
    ``required`` names are given, an all-green rollup that is still MISSING a required check (it has not
    reported yet) is PROGRESS, not SUCCESS - so the loop never merges before every required check ran."""
    if status.state == "none":
        # No checks reported YET (the window before Actions creates check runs, or a required check that
        # has not reported) must not be read as 'done' - that would merge before CI validates the diff.
        # Keep observing; a genuinely CI-less repo is the caller's explicit opt-in, not this default.
        return Observation.PROGRESS, "no CI checks have reported yet (keep observing)"
    if status.state == "pending":
        return Observation.PROGRESS, f"CI in progress ({status.passed}/{status.total} settled)"
    if status.state == "passing":
        missing = sorted(required - set(status.passing_checks))
        if missing:  # a required check has not reported/passed yet - do NOT read a partial rollup as done
            return Observation.PROGRESS, f"required CI check(s) not yet green: {missing}"
        return Observation.SUCCESS, f"CI green ({status.passed}/{status.total})"
    for name in status.failing_checks:
        low = name.lower()
        for hints, observation in _CHECK_HINTS:
            if any(h in low for h in hints):
                return observation, f"CI check {name!r} failed"
    return Observation.TEST_REGRESSION, f"CI failing: {sorted(status.failing_checks)}"


def merge_gate(fired_obligations: Any, *, dangerous: bool = False) -> MergeGate:
    """Decide whether the loop may merge autonomously - RISK DERIVED FROM POLICY, not from the caller.
    FAIL-CLOSED: a fired obligation forbids an autonomous merge unless it is provably safe - i.e. it
    escalates if it is a known human-gated one, OR it is NOT on the candidate-satisfiable allowlist and
    its severity is not a low (info/advisory) one. So a blocking obligation, an obligation with an unknown
    or MISSING severity, or a malformed entry all escalate by default; only a no-obligation change or one
    whose obligations are all allowlisted / low-severity auto-merges. ``dangerous`` is a ONE-WAY caller
    override - it can only FORCE escalation (belt-and-suspenders), never authorize a policy-gated merge."""
    if dangerous:
        return MergeGate(False, Observation.AUTHORIZATION_REQUIRED,
                         "caller forced human authorization (dangerous / irreversible / release)")
    gated: list[str] = []
    for o in fired_obligations or []:
        if isinstance(o, str):
            oid, severity = o, ""
        elif isinstance(o, dict) and isinstance(o.get("id"), str):
            oid, severity = o["id"], str(o.get("severity", ""))
        else:  # an unparseable obligation entry: we cannot assess its risk -> fail closed (escalate)
            return MergeGate(False, Observation.AUTHORIZATION_REQUIRED,
                             f"malformed fired obligation {o!r}; cannot assess merge risk (escalating)")
        if oid in _HUMAN_GATED or (oid not in _AUTO_MERGEABLE and severity.lower() not in _LOW_SEVERITY):
            gated.append(oid)
    if gated:
        return MergeGate(False, Observation.AUTHORIZATION_REQUIRED,
                         f"policy-gated, needs independent human review (no self-merge): "
                         f"{', '.join(sorted(set(gated)))}")
    return MergeGate(True, Observation.SUCCESS, "product change - auto-mergeable on standing authorization")


def observe_ci(gh_runner: GhRunner, pr_ref: str) -> CiSnapshot:
    """ONE consistent read of the PR via ``gh pr view`` - the head commit AND the checks that ran against
    it - so a later merge can be bound to that exact head (race-safe). Fail-closed: a non-zero ``gh`` exit
    (an auth/network/not-found error, distinct from a mere failing check) or unusable output raises."""
    code, out, err = gh_runner("pr", "view", pr_ref, "--json", "headRefOid,statusCheckRollup")
    if code != 0:
        raise IntegrateError(f"gh pr view failed (exit {code}); stderr: {err.strip()[:200]}")
    try:
        data = json.loads(out)
    except (ValueError, RecursionError) as exc:
        raise IntegrateError(f"gh pr view produced no usable JSON ({exc}); stderr: {err.strip()[:200]}") from exc
    if not isinstance(data, dict):
        raise IntegrateError(f"gh pr view payload is not an object (got {type(data).__name__})")
    raw_sha = data.get("headRefOid")
    head_sha = raw_sha if isinstance(raw_sha, str) and raw_sha else None
    # A PR with no checks yet reports statusCheckRollup: null - normalise to [] (state 'none' -> HOLD),
    # never a crash: parse_ci_checks fails closed on a non-list, but null here is 'no checks', not an error.
    rollup = data.get("statusCheckRollup")
    return CiSnapshot(status=parse_ci_checks(rollup if rollup is not None else []), head_sha=head_sha)


# gh's stderr when ``--match-head-commit`` refuses because the PR head advanced since we observed CI.
_HEAD_MOVED_HINTS = ("head branch was modified", "not the most recent", "match-head-commit",
                     "head of", "changed since", "stale", "expected head", "no longer matches")


def head_bound_merge(gh_runner: GhRunner, pr_ref: str, head_sha: str | None) -> tuple[Observation, str]:
    """Merge the PR BOUND to the exact head the CI ran against (``--match-head-commit``), so a commit
    pushed AFTER we observed CI is never merged unvalidated (closes the observe->merge race). Returns:

      * ``SUCCESS`` - merged.
      * ``HOLD`` - the head moved (a new commit landed since CI); re-observe the new head, do not merge
        blind. Also HOLD when the head is unknown, so we never merge without a binding (fail-closed).
      * ``TEST_REGRESSION`` - a genuine merge failure (conflict / permissions).
    """
    if not head_sha:
        return Observation.HOLD, "cannot determine the PR head to bind a race-safe merge; re-observing"
    code, _out, err = gh_runner("pr", "merge", pr_ref, "--squash", "--match-head-commit", head_sha)
    if code == 0:
        return Observation.SUCCESS, f"merged (authorized product change, head-bound to {head_sha[:12]})"
    low = err.lower()
    if any(hint in low for hint in _HEAD_MOVED_HINTS):
        return (Observation.HOLD,
                f"PR head moved since CI (race-safe merge refused); re-observing: {err.strip()[:120]}")
    return Observation.TEST_REGRESSION, f"authorized merge failed: {err.strip()[:120]}"
