"""Tests for L8 self-correction (scripts/loop/completeness.py).

Pins the completeness critic (all-met -> complete; unmet -> gaps; empty -> unproven; fail-closed), the
verdict -> Observation mapping, gaps -> correction tasks, and the cross-goal learning ledger (seed prior
dead ends, record outcome, round-trip, fail-closed on a malformed ledger).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from loop.completeness import (  # noqa: E402
    CompletenessError,
    Criterion,
    CriterionKind,
    check_completeness,
    completeness_correction_tasks,
    completeness_observation,
    record_outcome,
    seed_known_dead_ends,
)
from loop.controller import LoopState, Observation, Phase  # noqa: E402
from loop.tasks import parse_tasks  # noqa: E402


def _critic(criteria: list[Criterion]):
    return lambda _state: criteria


def _proven(cid: str, desc: str, digest: str) -> Criterion:
    return Criterion(cid, desc, kind=CriterionKind.DETERMINISTIC, met=True, evidence=digest)


def _state_with_evidence(*entries: dict) -> LoopState:
    state = LoopState()
    state.review_state["evidence"] = list(entries)
    return state


def _gate_green_evidence(subject: str = "cs:x", digest: str = "sha256:v") -> dict:
    return {"record_type": "workspace_verification", "predicate": "WORKSPACE_GATE_GREEN",
            "subject_fingerprint": subject, "digest": digest}


# --------------------------------------------------------------------------- semantic evidence (#6 slice 2)


def test_semantic_criterion_is_met_by_matching_evidence() -> None:
    state = _state_with_evidence(_gate_green_evidence())
    crit = Criterion("gate", "the workspace gate is green", kind=CriterionKind.DETERMINISTIC, met=True,
                     required_record_type="workspace_verification", required_predicate="WORKSPACE_GATE_GREEN")
    assert check_completeness(state, _critic([crit])).complete
    # ...and with the subject pinned + matching
    pinned = Criterion("gate", "green for this change", kind=CriterionKind.DETERMINISTIC, met=True,
                       required_record_type="workspace_verification", required_predicate="WORKSPACE_GATE_GREEN",
                       subject_fingerprint="cs:x")
    assert check_completeness(state, _critic([pinned])).complete


def test_a_verification_digest_cannot_satisfy_an_unrelated_claim() -> None:
    # The reviewer's exact case: a 'docs complete' criterion must NOT be satisfied by a recorded
    # workspace-verification digest - the record type / predicate must actually MATCH the claim.
    state = _state_with_evidence(_gate_green_evidence())
    docs = Criterion("docs", "documentation is complete", kind=CriterionKind.DETERMINISTIC, met=True,
                     required_record_type="docs_check", required_predicate="DOCS_COMPLETE")
    verdict = check_completeness(state, _critic([docs]))
    assert not verdict.complete and [c.id for c in verdict.unmet] == ["docs"]


def test_semantic_criterion_rejects_a_wrong_predicate_or_subject() -> None:
    state = _state_with_evidence(_gate_green_evidence(subject="cs:x"))
    wrong_pred = Criterion("c", "d", kind=CriterionKind.DETERMINISTIC, met=True,
                           required_record_type="workspace_verification", required_predicate="SOMETHING_ELSE")
    assert not check_completeness(state, _critic([wrong_pred])).complete
    wrong_subj = Criterion("c", "d", kind=CriterionKind.DETERMINISTIC, met=True,
                           required_record_type="workspace_verification", required_predicate="WORKSPACE_GATE_GREEN",
                           subject_fingerprint="cs:OTHER")
    assert not check_completeness(state, _critic([wrong_subj])).complete


def test_bare_digest_criterion_still_uses_membership_fallback() -> None:
    # Back-compat: a criterion with NO semantic fields falls back to slice-1 digest membership.
    state = LoopState(assurance_records=["sha256:v"])
    assert check_completeness(state, _critic([_proven("c", "done", "sha256:v")])).complete
    assert not check_completeness(state, _critic([_proven("c", "done", "sha256:UNRECORDED")])).complete


# --------------------------------------------------------------------------- the typed, evidence-bound critic


def test_deterministic_criteria_with_bound_evidence_are_complete() -> None:
    # A DETERMINISTIC criterion counts as met only when its evidence is a SEALED assurance record.
    state = LoopState(assurance_records=["sha256:a", "sha256:b"])
    verdict = check_completeness(state, _critic([_proven("c1", "scorer registered", "sha256:a"),
                                                 _proven("c2", "gate green", "sha256:b")]))
    assert verdict.complete and verdict.unmet == [] and verdict.needs_authority == []
    assert completeness_observation(verdict) is Observation.SUCCESS


def test_deterministic_met_without_bound_evidence_is_unmet() -> None:
    # met=True but the cited digest was never recorded -> NOT proven -> a correction task, not completion.
    state = LoopState(assurance_records=["sha256:a"])
    verdict = check_completeness(state, _critic([_proven("c1", "gate green", "sha256:UNRECORDED")]))
    assert not verdict.complete and [c.id for c in verdict.unmet] == ["c1"]
    assert completeness_observation(verdict) is Observation.CHANGES_REQUESTED
    assert parse_tasks(completeness_correction_tasks(verdict))[0].id == "meet-c1"


def test_model_judgment_alone_cannot_autonomously_finalize() -> None:
    # A MODEL_JUDGMENT criterion (the default kind) that the model calls met is the model's OPINION: it
    # needs human authority, so the loop escalates rather than finalizing on a bare model claim.
    verdict = check_completeness(LoopState(), _critic([Criterion("c1", "looks good", met=True)]))
    assert not verdict.complete and [c.id for c in verdict.needs_authority] == ["c1"]
    assert completeness_observation(verdict) is Observation.AUTHORIZATION_REQUIRED
    assert completeness_correction_tasks(verdict) == []  # authority gaps are NOT auto-fixable tasks


def test_human_approval_is_met_only_by_a_recorded_grant() -> None:
    crit = Criterion("release-signoff", "human approves release", kind=CriterionKind.HUMAN_APPROVAL, met=True)
    pending = check_completeness(LoopState(), _critic([crit]))  # met=True is IGNORED without a grant
    assert not pending.complete and [c.id for c in pending.needs_authority] == ["release-signoff"]
    assert completeness_observation(pending) is Observation.AUTHORIZATION_REQUIRED
    granted = LoopState()
    granted.review_state["authorizations"] = [{"grant": "release-signoff", "note": "approved"}]
    assert check_completeness(granted, _critic([crit])).complete  # the recorded grant satisfies it


def test_model_judgment_is_ratified_by_a_recorded_grant() -> None:
    # The escalate -> `cs_loop authorize --grant <id>` -> finalize resolution: a human grant matching the
    # criterion id ratifies the model's judgment, so the loop can then complete.
    crit = Criterion("looks-good", "the model judges it done", met=True)  # default MODEL_JUDGMENT
    assert not check_completeness(LoopState(), _critic([crit])).complete  # opinion alone: escalate
    ratified = LoopState()
    ratified.review_state["authorizations"] = [{"grant": "looks-good", "note": "I agree"}]
    assert check_completeness(ratified, _critic([crit])).complete  # human ratified -> complete


def test_unmet_takes_precedence_over_authority_in_routing() -> None:
    # With BOTH an unmet gap and an authority gap, route CHANGES_REQUESTED first (work what you can),
    # escalating the residual human decision only once nothing is autonomously fixable.
    verdict = check_completeness(LoopState(), _critic([
        Criterion("c1", "docs written", kind=CriterionKind.DETERMINISTIC, met=False),
        Criterion("c2", "looks good", met=True)]))
    assert [c.id for c in verdict.unmet] == ["c1"] and [c.id for c in verdict.needs_authority] == ["c2"]
    assert completeness_observation(verdict) is Observation.CHANGES_REQUESTED


def test_no_criteria_is_unproven_not_complete() -> None:
    # A goal with nothing to check must NOT be declared complete (a green gate is not 'done').
    verdict = check_completeness(LoopState(), _critic([]))
    assert not verdict.complete and "define/approve" in verdict.note


def test_unknown_criterion_kind_fails_closed() -> None:
    # An unrecognized `kind` must NOT silently fall into the (weakest) model-judgment branch - it fails
    # closed, so a typo'd requirement can never be satisfied without the evidence its real kind demands.
    bad = Criterion("c1", "x", met=True)
    object.__setattr__(bad, "kind", "NOT_A_REAL_KIND")  # unrecognized value on the frozen dataclass
    with pytest.raises(CompletenessError, match="unknown kind"):
        check_completeness(LoopState(), _critic([bad]))


def test_raw_string_kind_normalizes_to_its_enum() -> None:
    # A valid kind given as a raw string still enforces that kind's evidence rule (here: bound evidence).
    raw = Criterion("c1", "x", met=True, evidence="sha256:a")
    object.__setattr__(raw, "kind", "DETERMINISTIC")  # a raw str, not the enum member
    state = LoopState(assurance_records=["sha256:a"])
    assert check_completeness(state, _critic([raw])).complete  # normalized -> deterministic + bound -> met


def test_critic_returning_non_criteria_fails_closed() -> None:
    with pytest.raises(CompletenessError):
        check_completeness(LoopState(), lambda _s: ["not a criterion"])  # type: ignore[arg-type,list-item]


def test_critic_that_raises_fails_closed() -> None:
    def boom(_s: LoopState) -> list[Criterion]:
        raise RuntimeError("LLM judge timed out")
    with pytest.raises(CompletenessError, match="critic raised"):
        check_completeness(LoopState(), boom)


def test_non_bool_met_is_not_treated_as_met() -> None:
    # A truthy non-bool `met` (e.g. the string "yes") must NOT score a deterministic criterion as met.
    state = LoopState(assurance_records=["sha256:a"])
    verdict = check_completeness(state, _critic(
        [Criterion("c1", "x", kind=CriterionKind.DETERMINISTIC, met="yes", evidence="sha256:a")]))  # type: ignore[arg-type]
    assert not verdict.complete and [c.id for c in verdict.unmet] == ["c1"]


def test_seed_ignores_a_non_list_failed_approaches(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps([{"failed_approaches": "sha256:not-a-list"}]))  # scalar, not a list
    state = LoopState()
    assert seed_known_dead_ends(state, ledger) == 0 and state.failed_approaches == []


# --------------------------------------------------------------------------- cross-goal ledger


def test_seed_loads_prior_dead_ends(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps([{"goal": "g0", "failed_approaches": ["sha256:x", "sha256:y"]}]))
    state = LoopState(goal="g1")
    assert seed_known_dead_ends(state, ledger) == 2
    assert state.failed_approaches == ["sha256:x", "sha256:y"]  # g1 starts knowing g0's dead ends


def test_seed_on_missing_ledger_is_a_noop(tmp_path: Path) -> None:
    assert seed_known_dead_ends(LoopState(), tmp_path / "nope.json") == 0


def test_record_then_seed_round_trips(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    done = LoopState(goal="g1", goal_id="id1", current_phase=Phase.FINALIZE,
                     failed_approaches=["sha256:deadend"])
    record_outcome(done, ledger, lessons=["prefer chunked CE"])
    assert list(ledger.parent.glob("*.tmp-*")) == []  # atomic, no temp leak
    entry = json.loads(ledger.read_text())[0]
    assert entry["goal"] == "g1" and entry["outcome"] == "FINALIZE" and entry["lessons"] == ["prefer chunked CE"]
    # a fresh goal seeds from it
    fresh = LoopState(goal="g2")
    assert seed_known_dead_ends(fresh, ledger) == 1 and fresh.failed_approaches == ["sha256:deadend"]


def test_record_outcome_fails_closed_when_the_ledger_is_locked(tmp_path: Path) -> None:
    # A concurrent holder of the ledger lock must make record_outcome FAIL rather than lose its append to a
    # read-modify-write race - it raises within lock_timeout instead of clobbering the other writer.
    from loop.locking import FileLock
    ledger = tmp_path / "ledger.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    held = FileLock(ledger, timeout=5).acquire()
    try:
        with pytest.raises(CompletenessError, match="could not lock"):
            record_outcome(LoopState(goal="g", current_phase=Phase.FINALIZE), ledger, lock_timeout=0.2)
    finally:
        held.release()
    # once the lock is free, it records normally
    record_outcome(LoopState(goal="g", current_phase=Phase.FINALIZE), ledger, lock_timeout=5)
    assert json.loads(ledger.read_text())[-1]["goal"] == "g"


def test_malformed_ledger_fails_closed(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{not json")
    with pytest.raises(CompletenessError):
        seed_known_dead_ends(LoopState(), ledger)
    ledger.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(CompletenessError):
        seed_known_dead_ends(LoopState(), ledger)
