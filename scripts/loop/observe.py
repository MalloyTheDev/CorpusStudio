"""Wire the assurance plane (``cs_assure``) into the controller's OBSERVE step (controller slice 2b).

This is the seam where the two planes meet: after the executor does work, the controller must turn "what
happened to the repository" into ONE :class:`Observation` so :func:`loop.controller.apply` can route it.
This module runs ``cs_assure verify`` (+ ``doclint``) and MECHANICALLY classifies the sealed result.

The fact/judgment seam is explicit: this classifier only emits the observations that are MECHANICALLY
determinable from the assurance evidence -

  * gate red    -> the first failing step:  ruff->SYNTAX_FAILURE, mypy->TYPE_FAILURE, pytest->TEST_REGRESSION
  * a timed-out gate step                 -> ENVIRONMENT_FAILURE (a hang is not a code result)
  * gate green + self-modify/sealed-research fired -> AUTHORIZATION_REQUIRED (human review to admit)
  * gate green + worker-closure fired     -> WORKER_LINEAGE_IMPACT (leave the ordinary loop)
  * gate green + doc-trust findings        -> CONTRACT_DRIFT (docs out of sync)
  * otherwise                              -> SUCCESS

The JUDGMENT observations (WRONG_PLAN / WRONG_HYPOTHESIS / OWNERSHIP_COLLISION / POLICY_BLOCK /
NONDETERMINISTIC / DEPENDENCY_FAILURE) are NOT guessed here - they are the executor's (the LLM's) to
assign, layered on top of this mechanical baseline. Same stdlib-only / fail-closed discipline as the rest.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from loop.controller import (
    HUMAN_GATED_OBLIGATIONS,
    LoopState,
    Observation,
    Transition,
    apply,
    attempt_fingerprint,
)

# Gate step name -> the observation for its failure, in the order the gate runs them (first failure wins).
_GATE_ORDER: tuple[tuple[str, Observation], ...] = (
    ("ruff", Observation.SYNTAX_FAILURE),
    ("mypy", Observation.TYPE_FAILURE),
    ("pytest", Observation.TEST_REGRESSION),
)
# Obligations that, on an otherwise-green gate, require a human before the change can be admitted. SHARED
# with integrate's merge gate (loop.controller.HUMAN_GATED_OBLIGATIONS) so the OBSERVE plane and the merge
# gate can never disagree about what needs a human - a drift here previously let a loop-controller change
# read SUCCESS at OBSERVE. worker-closure is handled by its own WORKER_LINEAGE_IMPACT branch below (still
# human-gated, distinct label), so it is peeled off before the AUTHORIZATION_REQUIRED mapping.
_HUMAN_GATED = HUMAN_GATED_OBLIGATIONS

# The verify CLI exit contract (scripts/cs_assure.py): 0 = green gate, 1 = RED gate (still a valid record),
# >=2 = a fail-closed REFUSAL that emits NO record. So a record is trustworthy only for exit 0 or 1.
_VALID_VERIFY_EXITS = frozenset({0, 1})
# The sealed WorkspaceVerification contract (assurance.verification), mirrored here so the loop stays
# stdlib-only and does not import the assurance library. workspace_stable + the post-gate resnapshot were
# added in payload schema v2, so a record must be >= v2 to carry the stability guarantee this loop needs.
_VERIFY_RECORD_TYPE = "workspace_verification"
_MIN_VERIFY_SCHEMA = 2

# Explicit doc-trust authority states - a doclint that could NOT run must never be silently read as clean.
_DOCLINT_CLEAN = "CLEAN"
_DOCLINT_FINDINGS = "FINDINGS"
_DOCLINT_UNAVAILABLE = "UNAVAILABLE"

# A cs_assure runner: (repo_root, *argv) -> (returncode, stdout, stderr). Injectable for testing.
CsAssureRunner = Callable[..., "tuple[int, str, str]"]


class LoopObserveError(Exception):
    """cs_assure produced no usable evidence (refusal / unparseable output) - fail closed."""


@dataclass(frozen=True)
class ObservationResult:
    """The mechanical observation plus the evidence it was derived from."""

    observation: Observation
    reason: str
    gate_passed: bool
    record_digest: str | None
    change_set_fingerprint: str | None
    record_type: str | None = None  # the validated record's type (for the semantic evidence index)


def classify_observation(verify_payload: Any, doclint_payload: Any = None) -> tuple[Observation, str]:
    """Map a sealed ``verify`` payload (+ optional ``doclint`` payload) to one mechanical Observation.
    PURE. Fail-closed: a non-dict verify payload raises :class:`LoopObserveError`."""
    if not isinstance(verify_payload, dict):
        raise LoopObserveError(f"verify payload is not an object (got {type(verify_payload).__name__})")
    steps = [s for s in (verify_payload.get("gate_steps") or []) if isinstance(s, dict)]
    obligations = {o["id"] for o in (verify_payload.get("fired_obligations") or [])
                   if isinstance(o, dict) and isinstance(o.get("id"), str)}
    # Fail-closed on a non-bool: only a literal True is a green gate (a truthy "false"/1/[...] is red).
    gate_passed = verify_payload.get("gate_passed") is True

    # A hung step is an environment/infra problem, not a code result - surface it even if the gate is red.
    for step in steps:
        if step.get("timed_out"):
            return Observation.ENVIRONMENT_FAILURE, f"gate step {step.get('name')!r} timed out"

    # Gate RED -> fix the code first: classify the FIRST failing step in gate order.
    if not gate_passed:
        by_name = {s.get("name"): s for s in steps}
        for name, observation in _GATE_ORDER:
            failing = by_name.get(name)
            if failing is not None and not failing.get("passed", True):
                return observation, f"gate step {name!r} failed (exit {failing.get('exit_code')})"
        return Observation.TEST_REGRESSION, "gate failed with no identifiable step"

    # Gate GREEN -> the change is code-clean; surface human-gated / structural signals in priority order.
    # worker-closure has its own WORKER_LINEAGE_IMPACT signal, so peel it off; the rest of the shared
    # human-gated set (sealed-research / assurance-self-modify / loop-controller-self-modify) escalates.
    human = sorted((obligations & _HUMAN_GATED) - {"worker-closure"})
    if human:
        return Observation.AUTHORIZATION_REQUIRED, f"gate green but {', '.join(human)} requires human review"
    if "worker-closure" in obligations:
        return Observation.WORKER_LINEAGE_IMPACT, "gate green but the change touches worker-execution bytes"
    findings = _finding_count(doclint_payload)
    if findings > 0:
        return Observation.CONTRACT_DRIFT, f"gate green but {findings} doc-trust finding(s); docs out of sync"
    return Observation.SUCCESS, "gate green, no blocking obligations, docs clean"


def _finding_count(doclint_payload: Any) -> int:
    """The doclint finding count, tolerating a non-dict / non-int payload (-> 0)."""
    if not isinstance(doclint_payload, dict):
        return 0
    try:
        return int(doclint_payload.get("finding_count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _run_cs_assure(repo_root: Path, *argv: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "cs_assure.py"), *argv],
        cwd=str(repo_root), capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _parse(stdout: str, stderr: str, what: str) -> dict[str, Any]:
    try:
        data = json.loads(stdout)
    except (ValueError, RecursionError) as exc:
        raise LoopObserveError(f"cs_assure {what} produced no usable JSON ({exc}); stderr: {stderr.strip()[:200]}") from exc
    if not isinstance(data, dict):
        raise LoopObserveError(f"cs_assure {what} output is not an object")
    return data


def _validate_verify_record(record: dict[str, Any], payload: dict[str, Any]) -> None:
    """Fail-closed structural validation of a sealed WorkspaceVerification BEFORE it is trusted. The loop
    must consume a VALIDATED record, not raw JSON: a wrong record_type, an unsupported schema, a missing
    change-set fingerprint, or a workspace that MUTATED during the gate (workspace_stable != True ->
    CHANGE_SET_MUTATED_DURING_VERIFICATION, so the gate does not apply to a stable state) each raise.

    (Note: cryptographic record_digest re-derivation is deliberately NOT done here - it would couple the
    stdlib-only loop to the assurance canonical-JSON library; it belongs behind a future cs_assure
    ``verify-record`` subcommand that preserves the CLI boundary.)"""
    rtype = record.get("record_type")
    if rtype != _VERIFY_RECORD_TYPE:
        raise LoopObserveError(f"verify record_type {rtype!r} is not {_VERIFY_RECORD_TYPE!r}")
    version = record.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < _MIN_VERIFY_SCHEMA:
        raise LoopObserveError(f"unsupported verify schema_version {version!r} (need >= {_MIN_VERIFY_SCHEMA})")
    fingerprint = payload.get("change_set_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise LoopObserveError("verify record has no change_set_fingerprint")
    if payload.get("workspace_stable") is not True:
        raise LoopObserveError(
            "verify record workspace_stable is not True (the workspace mutated during the gate - "
            "CHANGE_SET_MUTATED_DURING_VERIFICATION); the gate does not apply to a stable change set")


def observe(repo_root: Path, base: str = "main", *,
            run_cs_assure: CsAssureRunner = _run_cs_assure) -> ObservationResult:
    """Run the assurance plane and classify the result. Fail-closed: if cs_assure refuses or emits no
    usable record, raise :class:`LoopObserveError` (the caller escalates - an unobservable repo is not
    a silent SUCCESS)."""
    code, out, err = run_cs_assure(repo_root, "verify", "--base", base)
    if code not in _VALID_VERIFY_EXITS:
        # exit >= 2 is a fail-closed refusal that emits NO record - never read a leftover/partial stdout
        # as evidence. (exit 0/1 both carry a valid record: a green vs a red gate.)
        raise LoopObserveError(f"cs_assure verify refused (exit {code}); stderr: {err.strip()[:200]}")
    record = _parse(out, err, "verify")
    payload = record.get("payload")
    if not isinstance(payload, dict):
        raise LoopObserveError("cs_assure verify record has no payload object")
    _validate_verify_record(record, payload)

    # Doc-trust is ADVISORY, but its status must be EXPLICIT: a doclint that could not run must not be
    # silently read as "no findings" (clean). CLEAN / FINDINGS / UNAVAILABLE are distinguished.
    doclint_payload: dict[str, Any] | None = None
    doclint_status = _DOCLINT_CLEAN
    try:
        _dc, dout, derr = run_cs_assure(repo_root, "doclint", "--format", "json")
        doclint_payload = _parse(dout, derr, "doclint")
        doclint_status = _DOCLINT_FINDINGS if _finding_count(doclint_payload) > 0 else _DOCLINT_CLEAN
    except LoopObserveError:
        doclint_status = _DOCLINT_UNAVAILABLE  # NOT silently clean - surfaced honestly in the reason below

    observation, reason = classify_observation(payload, doclint_payload)
    if doclint_status == _DOCLINT_UNAVAILABLE and observation is Observation.SUCCESS:
        reason = f"{reason}; NOTE docs UNVERIFIED (doclint unavailable)"
    return ObservationResult(
        observation=observation,
        reason=reason,
        # STRICT (matches classify_observation): only a real ``True`` is green. A truthy non-bool (the
        # string "false", 1, a non-empty list) must NOT record GATE_GREEN completion evidence for a gate
        # that classify read as RED - the two paths must agree, and both fail closed.
        gate_passed=payload.get("gate_passed") is True,
        record_digest=record.get("record_digest"),
        change_set_fingerprint=payload.get("change_set_fingerprint"),
        record_type=record.get("record_type"),
    )


# The predicate a GREEN workspace gate asserts - the semantic claim a DETERMINISTIC criterion matches on.
GATE_GREEN_PREDICATE = "WORKSPACE_GATE_GREEN"


def evidence_entry(result: ObservationResult) -> dict[str, Any] | None:
    """A STRUCTURED evidence entry for a GREEN-gate observation (else None): the record type + the
    predicate it asserts + the subject (change-set fingerprint) it is about + its digest. This is the
    semantic input the completeness layer matches criteria against - so a criterion is proven by evidence
    of the RIGHT TYPE about the RIGHT SUBJECT, not by any digest that happens to be recorded."""
    if not (result.gate_passed and result.record_digest and result.change_set_fingerprint):
        return None
    return {
        "record_type": result.record_type or "workspace_verification",
        "predicate": GATE_GREEN_PREDICATE,
        "subject_fingerprint": result.change_set_fingerprint,
        "digest": result.record_digest,
    }


def record_evidence(state: LoopState, result: ObservationResult) -> None:
    """Append the structured evidence entry for a green gate to the loop's evidence index (deduped by
    digest+predicate). A non-green observation records nothing (a red gate is not completion evidence)."""
    entry = evidence_entry(result)
    if entry is None:
        return
    index = state.review_state.setdefault("evidence", [])
    if isinstance(index, list) and not any(
            isinstance(e, dict) and e.get("digest") == entry["digest"]
            and e.get("predicate") == entry["predicate"] for e in index):
        index.append(entry)


def observe_and_apply(state: LoopState, repo_root: Path, base: str = "main", *,
                      run_cs_assure: CsAssureRunner = _run_cs_assure) -> Transition:
    """The integration point: OBSERVE the repo via cs_assure, record the sealed evidence on the loop
    state, and route it through the controller. The retry fingerprint is (observation+reason, change-set)
    so the SAME failure on the SAME change set is recognised as a repeated dead end."""
    result = observe(repo_root, base, run_cs_assure=run_cs_assure)
    if result.record_digest is not None and result.record_digest not in state.assurance_records:
        state.assurance_records.append(result.record_digest)
    record_evidence(state, result)  # structured evidence for the semantic completeness check
    fingerprint: str | None = None
    if result.observation not in (Observation.SUCCESS, Observation.PROGRESS) and result.change_set_fingerprint:
        fingerprint = attempt_fingerprint(f"{result.observation.value}:{result.reason}",
                                          result.change_set_fingerprint)
    return apply(state, result.observation, fingerprint=fingerprint, note=result.reason)
