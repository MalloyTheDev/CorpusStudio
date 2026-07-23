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
# Obligations that forbid an autonomous merge - the change must be admitted by a human. worker-closure
# is here too: a worker-lineage change needs a fresh wheel/env via the human-gated worker workflow, the
# same invariant observe.py enforces by routing worker-closure to WORKER_LINEAGE_IMPACT.
_HUMAN_GATED = frozenset({"sealed-research", "assurance-self-modify", "worker-closure"})

GhRunner = Callable[..., "tuple[int, str, str]"]


class IntegrateError(Exception):
    """gh produced no usable check data (fail-closed)."""


@dataclass(frozen=True)
class CiStatus:
    state: str  # "passing" | "failing" | "pending" | "none"
    passed: int
    total: int
    failing_checks: list[str] = field(default_factory=list)


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
    passing = failing = pending = 0
    failing_names: list[str] = []
    for check in items:
        bucket = _bucket(check)
        if bucket == "fail":
            failing += 1
            failing_names.append(str(check.get("name", "?")))
        elif bucket == "pass":
            passing += 1
        else:
            pending += 1
    if not items:
        state = "none"
    elif failing:
        state = "failing"
    elif pending:
        state = "pending"
    else:
        state = "passing"
    return CiStatus(state=state, passed=passing, total=len(items), failing_checks=failing_names)


def ci_observation(status: CiStatus) -> tuple[Observation, str]:
    """Map a CI status to a loop Observation. Pending is PROGRESS (keep observing, do not treat 'not
    done' as done); a failure is classified by check name so it routes like the local gate."""
    if status.state == "none":
        # No checks reported YET (the window before Actions creates check runs, or a required check that
        # has not reported) must not be read as 'done' - that would merge before CI validates the diff.
        # Keep observing; a genuinely CI-less repo is the caller's explicit opt-in, not this default.
        return Observation.PROGRESS, "no CI checks have reported yet (keep observing)"
    if status.state == "pending":
        return Observation.PROGRESS, f"CI in progress ({status.passed}/{status.total} settled)"
    if status.state == "passing":
        return Observation.SUCCESS, f"CI green ({status.passed}/{status.total})"
    for name in status.failing_checks:
        low = name.lower()
        for hints, observation in _CHECK_HINTS:
            if any(h in low for h in hints):
                return observation, f"CI check {name!r} failed"
    return Observation.TEST_REGRESSION, f"CI failing: {sorted(status.failing_checks)}"


def merge_gate(fired_obligations: Any, *, dangerous: bool = False) -> MergeGate:
    """Decide whether the loop may merge autonomously. A self-modifying / sealed-research change, or any
    caller-flagged dangerous/irreversible/release action, is NOT auto-mergeable - it escalates to a human
    (AUTHORIZATION_REQUIRED). Everything else is a product change, mergeable on standing authorization."""
    fired = {o for o in (fired_obligations or []) if isinstance(o, str)}
    if dangerous:
        return MergeGate(False, Observation.AUTHORIZATION_REQUIRED,
                         "dangerous / irreversible / release - human authorization required")
    human = sorted(fired & _HUMAN_GATED)
    if human:
        return MergeGate(False, Observation.AUTHORIZATION_REQUIRED,
                         f"{', '.join(human)} requires independent human review (no self-merge)")
    return MergeGate(True, Observation.SUCCESS, "product change - auto-mergeable on standing authorization")


def observe_ci(gh_runner: GhRunner, pr_ref: str) -> CiStatus:
    """Run ``gh pr checks`` via the injected runner and parse it. Fail-closed on unusable output."""
    _code, out, err = gh_runner("pr", "checks", pr_ref, "--json", "name,state,bucket")
    try:
        checks = json.loads(out)
    except (ValueError, RecursionError) as exc:
        raise IntegrateError(f"gh pr checks produced no usable JSON ({exc}); stderr: {err.strip()[:200]}") from exc
    return parse_ci_checks(checks)
