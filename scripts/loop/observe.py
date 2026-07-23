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

from loop.controller import LoopState, Observation, Transition, apply, attempt_fingerprint

# Gate step name -> the observation for its failure, in the order the gate runs them (first failure wins).
_GATE_ORDER: tuple[tuple[str, Observation], ...] = (
    ("ruff", Observation.SYNTAX_FAILURE),
    ("mypy", Observation.TYPE_FAILURE),
    ("pytest", Observation.TEST_REGRESSION),
)
# Obligations that, on an otherwise-green gate, require a human before the change can be admitted.
_HUMAN_GATED = frozenset({"sealed-research", "assurance-self-modify"})

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
    human = sorted(obligations & _HUMAN_GATED)
    if human:
        return Observation.AUTHORIZATION_REQUIRED, f"gate green but {', '.join(human)} requires human review"
    if "worker-closure" in obligations:
        return Observation.WORKER_LINEAGE_IMPACT, "gate green but the change touches worker-execution bytes"
    findings = 0
    if isinstance(doclint_payload, dict):
        try:
            findings = int(doclint_payload.get("finding_count", 0) or 0)
        except (TypeError, ValueError):
            findings = 0
    if findings > 0:
        return Observation.CONTRACT_DRIFT, f"gate green but {findings} doc-trust finding(s); docs out of sync"
    return Observation.SUCCESS, "gate green, no blocking obligations, docs clean"


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


def observe(repo_root: Path, base: str = "main", *,
            run_cs_assure: CsAssureRunner = _run_cs_assure) -> ObservationResult:
    """Run the assurance plane and classify the result. Fail-closed: if cs_assure refuses or emits no
    usable record, raise :class:`LoopObserveError` (the caller escalates - an unobservable repo is not
    a silent SUCCESS)."""
    _code, out, err = run_cs_assure(repo_root, "verify", "--base", base)
    record = _parse(out, err, "verify")
    payload = record.get("payload")
    if not isinstance(payload, dict):
        raise LoopObserveError("cs_assure verify record has no payload object")

    doclint_payload: dict[str, Any] | None = None
    try:
        _dc, dout, derr = run_cs_assure(repo_root, "doclint", "--format", "json")
        doclint_payload = _parse(dout, derr, "doclint")
    except LoopObserveError:
        doclint_payload = None  # doc-trust is advisory; its absence must not block the observation

    observation, reason = classify_observation(payload, doclint_payload)
    return ObservationResult(
        observation=observation,
        reason=reason,
        gate_passed=bool(payload.get("gate_passed", False)),
        record_digest=record.get("record_digest"),
        change_set_fingerprint=payload.get("change_set_fingerprint"),
    )


def observe_and_apply(state: LoopState, repo_root: Path, base: str = "main", *,
                      run_cs_assure: CsAssureRunner = _run_cs_assure) -> Transition:
    """The integration point: OBSERVE the repo via cs_assure, record the sealed evidence on the loop
    state, and route it through the controller. The retry fingerprint is (observation+reason, change-set)
    so the SAME failure on the SAME change set is recognised as a repeated dead end."""
    result = observe(repo_root, base, run_cs_assure=run_cs_assure)
    if result.record_digest is not None and result.record_digest not in state.assurance_records:
        state.assurance_records.append(result.record_digest)
    fingerprint: str | None = None
    if result.observation not in (Observation.SUCCESS, Observation.PROGRESS) and result.change_set_fingerprint:
        fingerprint = attempt_fingerprint(f"{result.observation.value}:{result.reason}",
                                          result.change_set_fingerprint)
    return apply(state, result.observation, fingerprint=fingerprint, note=result.reason)
