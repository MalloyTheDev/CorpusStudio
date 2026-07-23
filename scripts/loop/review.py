"""Review-feedback: turn a reviewer's findings into executor work (controller slice 6 - first half of L7).

A review must not be a dead end. This module runs a reviewer over the active change and folds its
ACCEPTED findings into CORRECTION TASKS appended to the task graph (each depending on the reviewed
task), so review feedback becomes scheduled work rather than comments - and a CLEAN review lets the loop
proceed to INTEGRATE. The verdict maps to a loop Observation: CLEAN -> SUCCESS (advance), else ->
CHANGES_REQUESTED (route back to ASSIGN and work the corrections).

As with the rest of the loop, the reviewer itself is an INJECTED callback (the real reviewer is a
read-only agent - e.g. the assurance-reviewer subagent - an effect that cannot run in a stdlib module);
this module is the deterministic mechanism around it. Fail-closed + stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from loop.controller import LoopState, Observation
from loop.tasks import LoopTaskError, parse_tasks


class ReviewError(Exception):
    """The reviewer returned something that is not a list of Findings (fail-closed)."""


class ReviewVerdict(str, Enum):
    CLEAN = "CLEAN"                          # no accepted findings -> the change may INTEGRATE
    CHANGES_REQUESTED = "CHANGES_REQUESTED"  # accepted findings became correction tasks


@dataclass(frozen=True)
class Finding:
    """One review finding. ``accepted`` is the adversarial-verify verdict - only accepted findings
    become correction tasks (a refuted finding is recorded but not actioned)."""

    id: str
    summary: str
    severity: str = "med"                   # high | med | low - informational, does not route
    suggested_fix: str = ""
    allowed_paths: list[str] = field(default_factory=list)  # the correction task's ownership boundary
    accepted: bool = True


@dataclass(frozen=True)
class ReviewResult:
    verdict: ReviewVerdict
    findings: list[Finding]
    accepted: list[Finding]
    correction_task_ids: list[str]


Reviewer = Callable[[LoopState], "list[Finding]"]


def _correction_id(reviewed_task_id: str | None, finding: Finding) -> str:
    # Namespace by the reviewed task so per-round-local finding ids cannot collide across rounds.
    return f"fix-{reviewed_task_id or 'goal'}-{finding.id}"


def _correction_task(finding: Finding, deps: list[str], task_id: str) -> dict[str, object]:
    return {
        "id": task_id,
        "description": f"Address review finding: {finding.summary}",
        "owner": "self",
        "allowed_paths": list(finding.allowed_paths),
        "depends_on": list(deps),
        "success_criteria": [finding.suggested_fix] if finding.suggested_fix else [],
        "status": "PENDING",
    }


def review(state: LoopState, reviewer: Reviewer, *, reviewed_task_id: str | None = None) -> ReviewResult:
    """Run the reviewer over the loop state and fold ACCEPTED findings into correction tasks appended to
    the graph (deps on ``reviewed_task_id`` when it names an existing task). Idempotent: a finding that
    already has a correction task is not duplicated. Fail-closed on a non-Finding reviewer result and on
    any resulting invalid graph."""
    findings_raw = reviewer(state)
    if not isinstance(findings_raw, list) or not all(isinstance(f, Finding) for f in findings_raw):
        raise ReviewError("reviewer must return a list[Finding]")
    findings = list(findings_raw)
    accepted = [f for f in findings if f.accepted]
    if not accepted:
        return ReviewResult(ReviewVerdict.CLEAN, findings, [], [])

    existing_ids = {t.get("id") for t in state.task_graph if isinstance(t, dict)}
    # Dedup against corrections THIS loop actually created (tracked in review_state), not arbitrary
    # existing task ids - so an unrelated task that happens to share the id never silently swallows a
    # finding. A collision with a truly-existing id is still skipped, but it is surfaced (below).
    created = state.review_state.get("correction_ids", [])
    if not isinstance(created, list):
        created = []
    deps: list[str] = [reviewed_task_id] if reviewed_task_id is not None and reviewed_task_id in existing_ids else []
    new_tasks: list[dict[str, object]] = []
    new_ids: list[str] = []
    blocked: list[str] = []
    for finding in accepted:
        tid = _correction_id(reviewed_task_id, finding)
        if tid in created or tid in new_ids:
            continue  # already scheduled this correction (idempotent re-review)
        if tid in existing_ids:
            blocked.append(tid)  # a foreign task already owns this id - do not conflate; surface it
            continue
        new_tasks.append(_correction_task(finding, deps, tid))
        new_ids.append(tid)

    if blocked:
        raise ReviewError(f"correction id(s) already exist as unrelated tasks: {sorted(blocked)}")
    if new_tasks:
        try:
            tasks = parse_tasks(list(state.task_graph) + new_tasks)  # fail-closed validation
        except LoopTaskError as exc:
            raise ReviewError(f"review corrections form an invalid task graph: {exc}") from exc
        state.task_graph = [t.to_dict() for t in tasks]
        state.review_state["correction_ids"] = created + new_ids
    return ReviewResult(ReviewVerdict.CHANGES_REQUESTED, findings, accepted, new_ids)


def review_observation(result: ReviewResult) -> Observation:
    """CLEAN -> SUCCESS (the loop advances to INTEGRATE); otherwise CHANGES_REQUESTED (route back to
    ASSIGN to schedule the correction tasks)."""
    return Observation.SUCCESS if result.verdict is ReviewVerdict.CLEAN else Observation.CHANGES_REQUESTED
