"""Long-horizon self-correction (controller slice 9, Level 8).

Two capabilities that make the loop a self-correcting, long-horizon system rather than a one-shot pass:

  * COMPLETENESS CRITIC - at VERIFY, do not FINALIZE just because the gate is green; check the GOAL's
    own success criteria are actually MET, and require the RIGHT KIND of evidence for each. A criterion
    is TYPED (:class:`CriterionKind`): a DETERMINISTIC or DOMAIN_AUTHORITY criterion counts as met only if
    it cites evidence BOUND to a sealed assurance record (a model asserting ``met=True`` with no bound
    digest is NOT proven -> it becomes a correction task); a MODEL_JUDGMENT criterion is the model's own
    opinion and can NEVER by itself close an autonomous finalize -> it needs human authority; a
    HUMAN_APPROVAL criterion is met only by a RECORDED human authorization (the critic cannot self-grant).
    An unmet criterion is a CHANGES_REQUESTED that folds into a correction task (keep working the gap); a
    criterion that is "met" only on model opinion / awaits a human is AUTHORIZATION_REQUIRED (escalate the
    residual human decision). So the loop never autonomously declares a goal done on a bare model claim.
  * CROSS-GOAL LEARNING - a durable ledger of prior goals' dead ends (failed-approach fingerprints).
    Seeding a new loop from it means the same (failure, approach) another goal already exhausted is
    recognised as a dead end immediately, so effort accumulates across goals/sessions instead of resetting.

Same discipline as the rest of the loop: the critic itself is an INJECTED callback (the real judge is the
LLM / an agent); this module is the deterministic mechanism. stdlib-only, fail-closed.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from loop.controller import LoopState, Observation
from loop.locking import FileLock, LockError


_LEDGER_MAX_ENTRIES = 500  # bound the cross-goal ledger so it cannot grow without limit


class CompletenessError(Exception):
    """The critic returned a non-Criterion result / raised, or the ledger is malformed (fail-closed)."""


class CriterionKind(str, Enum):
    """How a success criterion is PROVEN - which determines what evidence lets it close an AUTONOMOUS
    finalize. The default is the WEAKEST (``MODEL_JUDGMENT``), so an untyped criterion is never silently
    trusted as machine-proven."""

    DETERMINISTIC = "DETERMINISTIC"        # a re-runnable check (gate/test/digest); needs BOUND evidence
    DOMAIN_AUTHORITY = "DOMAIN_AUTHORITY"  # an authoritative offline record (cs_assure); needs BOUND evidence
    MODEL_JUDGMENT = "MODEL_JUDGMENT"      # the LLM's opinion; NEVER enough alone -> human authority
    HUMAN_APPROVAL = "HUMAN_APPROVAL"      # met only by a RECORDED human authorization (grant == id)


@dataclass(frozen=True)
class Criterion:
    """One goal success criterion: its KIND (how it is proven), whether the critic judged it met, and the
    evidence that must SUPPORT it. For DETERMINISTIC / DOMAIN_AUTHORITY there are two evidence modes:

      * SEMANTIC (preferred): name ``required_record_type`` AND ``required_predicate`` (and optionally a
        ``subject_fingerprint`` to pin the subject). The criterion is met only if the loop's recorded
        evidence holds an entry of that record type, asserting that predicate, about that subject - so a
        workspace-verification digest can never satisfy a 'docs complete' claim. The predicate is
        REQUIRED here (an empty predicate is rejected at construction, not treated as a wildcard: a
        wildcard would reintroduce the 'any record of this type counts' vagueness this mode removes).
      * DIGEST (the weaker slice-1 fallback, when no ``required_record_type`` is set): ``evidence`` must be
        a digest present in the loop's sealed assurance records - proves the digest was recorded, but not
        that it SUPPORTS this specific claim.

    For HUMAN_APPROVAL, ``met`` is IGNORED - the criterion is met only by a recorded authorization whose
    grant equals this criterion's ``id``."""

    id: str
    description: str
    kind: CriterionKind = CriterionKind.MODEL_JUDGMENT
    met: bool = False
    evidence: str = ""                 # digest-membership fallback (used only when required_record_type == "")
    required_record_type: str = ""     # SEMANTIC: the evidence record type that must exist (e.g. workspace_verification)
    required_predicate: str = ""       # ...asserting this predicate (e.g. WORKSPACE_GATE_GREEN); required if type is set
    subject_fingerprint: str = ""      # ...about this subject (a change-set fingerprint); "" = any subject

    def __post_init__(self) -> None:
        # Reject silently-unsatisfiable / no-op semantic criteria LOUDLY at construction (the critic path
        # turns this into a fail-closed escalation) rather than letting a goal never complete for a reason
        # nobody can see:
        #   - a record type with no predicate would match only an empty-predicate entry, which the loop
        #     never records (it always asserts a concrete predicate) -> unsatisfiable;
        #   - a pinned subject with no record type does nothing (the digest fallback ignores it) -> a no-op
        #     the author almost certainly did not intend.
        if self.required_record_type and not self.required_predicate:
            raise ValueError(
                f"criterion {self.id!r} sets required_record_type={self.required_record_type!r} but no "
                "required_predicate; a semantic criterion must pin the predicate it asserts")
        if self.subject_fingerprint and not self.required_record_type:
            raise ValueError(
                f"criterion {self.id!r} pins subject_fingerprint without a required_record_type; "
                "subject pinning only constrains semantic (record-type) matching")


@dataclass(frozen=True)
class CompletenessVerdict:
    complete: bool
    unmet: list[Criterion]            # not proven / met=False -> correction tasks (keep working the gap)
    needs_authority: list[Criterion]  # model-judged-met or human-approval-pending -> escalate to a human
    note: str


# The critic judges each success criterion met/unmet for the current state (the LLM/agent, injected).
Critic = Callable[[LoopState], "list[Criterion]"]


def _evidence_is_bound(evidence: str, state: LoopState) -> bool:
    """True only if ``evidence`` is a non-empty digest present in the loop's SEALED assurance records - so
    a deterministic/authority criterion cannot score as met on a digest the loop never actually recorded."""
    return bool(evidence) and evidence in state.assurance_records


def _evidence_index(state: LoopState) -> list[dict[str, Any]]:
    """The loop's STRUCTURED evidence entries ({record_type, predicate, subject_fingerprint, digest}),
    recorded by observe when it seals a green gate. Distinct from the flat digest list in assurance_records."""
    index = state.review_state.get("evidence")
    return [e for e in index if isinstance(e, dict)] if isinstance(index, list) else []


def _evidence_supports(criterion: Criterion, state: LoopState) -> bool:
    """Does the loop's recorded evidence SUPPORT this criterion? A criterion that names a
    ``required_record_type`` is matched SEMANTICALLY - the evidence index must hold an entry of that record
    type, asserting ``required_predicate``, about ``subject_fingerprint`` (if the criterion pins one) - so a
    record of the wrong type / predicate / subject cannot satisfy the claim. A criterion with NO semantic
    fields falls back to digest MEMBERSHIP (the weaker slice-1 mode)."""
    if criterion.required_record_type:
        return any(
            e.get("record_type") == criterion.required_record_type
            and e.get("predicate") == criterion.required_predicate
            and (not criterion.subject_fingerprint
                 or e.get("subject_fingerprint") == criterion.subject_fingerprint)
            for e in _evidence_index(state)
        )
    return _evidence_is_bound(criterion.evidence, state)


def _human_granted(criterion: Criterion, state: LoopState) -> bool:
    """True if a human authorization for this criterion has been RECORDED (``cs_loop authorize --grant``
    stores ``{grant, note}`` on ``review_state['authorizations']``). The critic cannot self-grant."""
    auths = state.review_state.get("authorizations", [])
    if not isinstance(auths, list):
        return False
    return any(isinstance(a, dict) and a.get("grant") == criterion.id for a in auths)


def check_completeness(state: LoopState, critic: Critic) -> CompletenessVerdict:
    """Run the critic and reduce it to an EVIDENCE-BOUND, KIND-AWARE verdict (fail-closed on a non-Criterion
    result). A goal with NO declared success criteria is not complete - a human must define/approve 'done'.
    Each criterion is classed by :class:`CriterionKind`:

      * DETERMINISTIC / DOMAIN_AUTHORITY - met only if the critic says met AND cites evidence bound to a
        sealed assurance record; otherwise it is UNMET (asserted without proof -> a correction task).
      * MODEL_JUDGMENT - met=True is the model's opinion; it can never by ITSELF close an autonomous
        finalize -> NEEDS_AUTHORITY, unless a human has RATIFIED it with a recorded grant (grant == id),
        which counts it met. met=False -> UNMET.
      * HUMAN_APPROVAL - met only by a recorded authorization (grant == id); otherwise NEEDS_AUTHORITY.

    A criterion awaiting authority is thus satisfied by ``cs_loop authorize --grant <criterion-id>`` - the
    universal "a human ratifies this specific criterion" mechanism that resolves the escalation.

    ``complete`` requires every criterion met with sufficient evidence and none awaiting human authority."""
    try:
        criteria = critic(state)
    except CompletenessError:
        raise
    except Exception as exc:  # noqa: BLE001 - the injected critic (LLM/agent) is untrusted; fail closed
        raise CompletenessError(f"critic raised {type(exc).__name__}: {exc}") from exc
    if not isinstance(criteria, list) or not all(isinstance(c, Criterion) for c in criteria):
        raise CompletenessError("critic must return a list[Criterion]")
    if not criteria:
        return CompletenessVerdict(
            False, [], [], "no success criteria were evaluated; a human must define/approve what 'done' means")
    unmet: list[Criterion] = []
    needs_authority: list[Criterion] = []
    for c in criteria:
        # Normalize the kind through the enum: a raw string ("DETERMINISTIC") resolves to its member, and
        # an UNKNOWN / typo / None kind FAILS CLOSED here rather than falling into a weaker branch - never
        # give an unrecognized requirement the model-judgment (grant-satisfiable, no-evidence) treatment.
        try:
            kind = CriterionKind(c.kind)
        except ValueError as exc:
            raise CompletenessError(f"criterion {c.id!r} has an unknown kind {c.kind!r}") from exc
        if kind is CriterionKind.HUMAN_APPROVAL:
            if not _human_granted(c, state):
                needs_authority.append(c)  # awaits a recorded human grant; the critic cannot self-grant
            # else: met by a recorded authorization (counted as met - falls through)
        elif kind in (CriterionKind.DETERMINISTIC, CriterionKind.DOMAIN_AUTHORITY):
            # met ONLY if exactly True (a truthy non-bool must not score) AND the recorded evidence
            # SEMANTICALLY supports it (right record type / predicate / subject), or - for a bare-digest
            # criterion - the cited digest was recorded.
            if c.met is True and _evidence_supports(c, state):
                continue
            unmet.append(c)  # asserted without supporting evidence is NOT proven -> work it
        else:  # MODEL_JUDGMENT (the only remaining valid kind)
            if c.met is not True:
                unmet.append(c)
            elif _human_granted(c, state):
                continue  # a human RATIFIED the model's judgment (grant == id) -> met
            else:
                needs_authority.append(c)  # model opinion awaiting human ratification
    complete = not unmet and not needs_authority
    if complete:
        note = f"all {len(criteria)} success criteria met with sufficient evidence"
    elif unmet:
        note = f"{len(unmet)}/{len(criteria)} success criteria unmet"
        if needs_authority:
            note += f"; {len(needs_authority)} await human authority"
    else:
        note = (f"{len(needs_authority)}/{len(criteria)} success criteria await human authority "
                "(model-judged or human-approval); an autonomous finalize is not permitted")
    return CompletenessVerdict(complete, unmet, needs_authority, note)


def completeness_observation(verdict: CompletenessVerdict) -> Observation:
    """Complete -> SUCCESS (the loop may FINALIZE). Unmet gaps -> CHANGES_REQUESTED (work them first).
    Otherwise the only thing left is a human decision -> AUTHORIZATION_REQUIRED (escalate)."""
    if verdict.complete:
        return Observation.SUCCESS
    if verdict.unmet:
        return Observation.CHANGES_REQUESTED
    return Observation.AUTHORIZATION_REQUIRED


def completeness_correction_tasks(verdict: CompletenessVerdict) -> list[dict[str, Any]]:
    """Turn each unmet criterion into a correction task (a goal-completion gap to close). Criteria awaiting
    human authority are NOT turned into tasks - they are a human decision, surfaced via escalation."""
    return [
        {
            "id": f"meet-{c.id}",
            "description": f"Meet unmet success criterion: {c.description}",
            "owner": "self",
            "allowed_paths": [],
            "depends_on": [],
            "success_criteria": [c.description],
            "status": "PENDING",
        }
        for c in verdict.unmet
    ]


# --------------------------------------------------------------------------- cross-goal learning ledger


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise CompletenessError(f"cross-goal ledger is unreadable/malformed ({path}): {exc}") from exc
    if not isinstance(data, list):
        raise CompletenessError(f"cross-goal ledger root must be a list ({path})")
    return [e for e in data if isinstance(e, dict)]


def seed_known_dead_ends(state: LoopState, ledger_path: Path) -> int:
    """Seed ``state.failed_approaches`` with prior goals' failed-approach fingerprints, so a (failure,
    approach) another goal already exhausted is recognised as a dead end immediately. Only EXACT repeats
    are blocked (the fingerprint includes the approach), so a genuinely new approach is still allowed.
    Returns the number of fingerprints seeded."""
    seeded = 0
    for entry in _load_ledger(ledger_path):
        approaches = entry.get("failed_approaches", [])
        if not isinstance(approaches, list):
            continue  # a non-list failed_approaches (e.g. a scalar) must not be iterated / seeded
        for fingerprint in approaches:
            if isinstance(fingerprint, str) and fingerprint not in state.failed_approaches:
                state.failed_approaches.append(fingerprint)
                seeded += 1
    return seeded


def record_outcome(state: LoopState, ledger_path: Path, *, lessons: list[str] | None = None,
                   lock_timeout: float = 10.0) -> None:
    """Append this goal's outcome + dead ends to the cross-goal ledger, so the next goal starts from
    accumulated experience. The read-modify-write is guarded by a cross-process :class:`FileLock` so two
    concurrent campaigns (or a campaign + a standalone run) sharing one ledger cannot LOSE each other's
    append (a read-then-replace race). The write itself stays atomic (temp + os.replace). Fail-closed: if
    the ledger cannot be locked within ``lock_timeout`` the outcome is NOT recorded (raises)."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)  # the lockfile lives beside the ledger
    try:
        with FileLock(ledger_path, timeout=lock_timeout):
            entries = _load_ledger(ledger_path)
            entries.append({
                "goal": state.goal,
                "goal_id": state.goal_id,
                "outcome": state.current_phase.value,
                "termination_reason": state.termination_reason,
                "failed_approaches": copy.deepcopy(list(state.failed_approaches)),
                "lessons": list(lessons or []),
            })
            if len(entries) > _LEDGER_MAX_ENTRIES:
                entries = entries[-_LEDGER_MAX_ENTRIES:]  # bound the ledger; keep the most recent goals
            tmp = ledger_path.with_name(f"{ledger_path.name}.tmp-{os.getpid()}")
            tmp.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, ledger_path)
    except LockError as exc:
        raise CompletenessError(f"could not lock the learning ledger ({ledger_path}): {exc}") from exc
