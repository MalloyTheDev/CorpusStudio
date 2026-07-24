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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

from loop.controller import HUMAN_GATED_OBLIGATIONS, Observation

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
# (the candidate/loop cannot self-verify a change to itself). This is the SHARED source of truth with
# observe.py (imported, not re-declared, so the two planes cannot drift): worker-closure needs a fresh
# wheel/env via the human-gated worker workflow, and loop-controller-self-modify must not be admitted by
# the loop's OWN merge gate.
_HUMAN_GATED = HUMAN_GATED_OBLIGATIONS
# Severities that do NOT by themselves gate a merge. Any obligation whose severity is not one of these
# (blocking, unknown, or absent) escalates unless a resolution record proves it discharged - so a missing
# or non-canonical severity fails CLOSED (escalate) instead of silently authorizing.
_LOW_SEVERITY = frozenset({"info", "advisory"})

# An obligation-resolution record proves a specific blocking obligation was ACTUALLY discharged for a
# specific change set, by a TRUSTED authority - not merely that it is "usually CI-satisfiable" (re-review
# #14: obligation IDENTITY must not be equated with "this obligation was satisfied for this commit"). A
# resolution is an injected plain dict (the loop is stdlib-only; a producing runtime seals it separately):
#   {"obligation_id": str, "status": "RESOLVED", "subject_fingerprint": <change-set fp>, "authority": str}
_RESOLVED = "RESOLVED"
# The candidate CANNOT self-resolve an obligation: only these authorities are trusted to discharge one.
# (A candidate-controlled result would let a change vouch for itself - the same reason self-modify is
# human-gated.) Kept deliberately small + fail-closed; a producing runtime widens it under review.
_TRUSTED_RESOLUTION_AUTHORITIES = frozenset({"trusted-base-ci", "independent-human-review"})

# The verdict a fired obligation gets at the gate (single source of truth for the serialized labels, so
# merge_gate / GateEvaluation.to_record / tests / consumers never drift). Only *_SATISFIED_* clear the gate.
_DISP_HUMAN_GATED = "HUMAN_GATED"    # a human-gated obligation: never dischargeable by the loop
_DISP_LOW_SEVERITY = "LOW_SEVERITY"  # info/advisory: does not gate a merge
_DISP_RESOLVED = "RESOLVED"          # a blocking obligation with a trusted, current resolution record
_DISP_UNRESOLVED = "UNRESOLVED"      # a blocking obligation with no such record -> escalate
_DISP_MALFORMED = "MALFORMED"        # an unparseable obligation entry -> escalate

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
class ObligationVerdict:
    """One blocking obligation's disposition at the merge gate."""

    obligation_id: str
    disposition: str  # HUMAN_GATED | LOW_SEVERITY | RESOLVED | UNRESOLVED | MALFORMED
    satisfied: bool


@dataclass(frozen=True)
class GateEvaluation:
    """The FINAL merge-gate record (re-review #14): the change set the decision bound to, a per-obligation
    verdict, and the overall authorization. A merge is authorized only if EVERY fired obligation is
    satisfied - human-gated ones never are, and a blocking one is satisfied only by a trusted, current
    resolution record. Serializable for a producing runtime to seal alongside the merge."""

    subject_fingerprint: str
    authorized: bool
    verdicts: tuple[ObligationVerdict, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "subject_fingerprint": self.subject_fingerprint,
            "authorized": self.authorized,
            "verdicts": [
                {"obligation_id": v.obligation_id, "disposition": v.disposition, "satisfied": v.satisfied}
                for v in self.verdicts
            ],
        }


@dataclass(frozen=True)
class MergeGate:
    authorized: bool
    observation: Observation
    reason: str
    evaluation: "GateEvaluation | None" = None


def _resolution_supports(
    obligation_id: str, subject_fingerprint: str, resolutions: "Iterable[Mapping[str, Any]] | None"
) -> bool:
    """True iff some resolution proves ``obligation_id`` was discharged for THIS change set: RESOLVED
    (status), CURRENT + APPLICABLE_TO_HEAD (its ``subject_fingerprint`` equals the one the gate is bound
    to), and TRUSTED (a trusted authority, never the candidate). A blank bound fingerprint matches nothing
    (fail-closed: an unbound gate can accept no resolution). Malformed resolution ENTRIES are ignored; a
    ``resolutions`` value that is not a sequence at all is treated as empty (fail-closed: escalate rather
    than iterate a dict's keys or crash the loop on a non-iterable)."""
    if not subject_fingerprint or not isinstance(resolutions, (list, tuple)):
        return False
    for r in resolutions:
        if (isinstance(r, Mapping)
                and r.get("obligation_id") == obligation_id
                and r.get("status") == _RESOLVED
                and r.get("subject_fingerprint") == subject_fingerprint
                and r.get("authority") in _TRUSTED_RESOLUTION_AUTHORITIES):
            return True
    return False


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


def merge_gate(fired_obligations: Any, *, resolutions: "Iterable[Mapping[str, Any]] | None" = None,
               subject_fingerprint: str = "", dangerous: bool = False) -> MergeGate:
    """Decide whether the loop may merge autonomously - RISK DERIVED FROM POLICY + EVIDENCE, not identity.
    FAIL-CLOSED. Each fired obligation is dispositioned: a known human-gated one always escalates; a
    low-severity (info/advisory) one does not gate; ANY OTHER blocking obligation is satisfied ONLY by a
    trusted, current resolution record proving it was discharged for THIS change set (re-review #14 - no
    obligation is auto-mergeable on identity alone, so 'usually CI-satisfiable' is never equated with
    'satisfied for this commit'). A blocking obligation with no such resolution, an unknown/missing
    severity, or a malformed entry all escalate. Returns a :class:`GateEvaluation` recording every
    verdict. ``dangerous`` is a ONE-WAY caller override - it only FORCES escalation, never authorizes."""
    if dangerous:
        return MergeGate(False, Observation.AUTHORIZATION_REQUIRED,
                         "caller forced human authorization (dangerous / irreversible / release)",
                         GateEvaluation(subject_fingerprint, False, ()))
    verdicts: list[ObligationVerdict] = []
    gated: list[str] = []
    for o in fired_obligations or []:
        if isinstance(o, str):
            oid, severity = o, ""
        elif isinstance(o, dict) and isinstance(o.get("id"), str):
            oid, severity = o["id"], str(o.get("severity", ""))
        else:  # an unparseable obligation entry: we cannot assess its risk -> fail closed (escalate)
            verdicts.append(ObligationVerdict(repr(o), _DISP_MALFORMED, False))
            return MergeGate(False, Observation.AUTHORIZATION_REQUIRED,
                             f"malformed fired obligation {o!r}; cannot assess merge risk (escalating)",
                             GateEvaluation(subject_fingerprint, False, tuple(verdicts)))
        if oid in _HUMAN_GATED:
            verdicts.append(ObligationVerdict(oid, _DISP_HUMAN_GATED, False))
            gated.append(oid)
        elif severity.lower() in _LOW_SEVERITY:
            verdicts.append(ObligationVerdict(oid, _DISP_LOW_SEVERITY, True))
        elif _resolution_supports(oid, subject_fingerprint, resolutions):
            verdicts.append(ObligationVerdict(oid, _DISP_RESOLVED, True))
        else:
            verdicts.append(ObligationVerdict(oid, _DISP_UNRESOLVED, False))
            gated.append(oid)
    evaluation = GateEvaluation(subject_fingerprint, not gated, tuple(verdicts))
    if gated:
        return MergeGate(False, Observation.AUTHORIZATION_REQUIRED,
                         f"policy-gated, needs independent human review (no self-merge): "
                         f"{', '.join(sorted(set(gated)))}", evaluation)
    return MergeGate(True, Observation.SUCCESS,
                     "product change - every blocking obligation resolved for this commit", evaluation)


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
