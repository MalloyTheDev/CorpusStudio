"""Workspace-verification engine for the assurance loop (Phase 8, first slice).

``cs_assure verify`` RUNS the repo's declared workspace gate (a policy of no-shell ``argv`` steps -
ruff / mypy / pytest) itself, then seals a :class:`WorkspaceVerification` record binding the REAL
per-step exit codes to the current change set. The point is honesty: "I verified" stops being a
narrated claim and becomes evidence cs_assure produced - it ran the gate, here are the exit codes,
here is the change set they apply to. It also folds in the obligations that fired on the same change
set (from the impact policy) so a completion record shows both "the gate is green" AND "these
obligations fired - discharge them".

HONESTY CEILING (enforced by naming, not by overclaiming): a green result is ``completion_level ==
"WORKSPACE_GATE"`` and NOTHING more. A green workspace gate is NOT proven fit, NOT commit / PR /
release / sealed-research readiness, and NOT CI (CI additionally enforces coverage + the web job).
The record never says "verified" / "fit" / "release-ready" / "sealed"; it records exit codes and a
workspace-level verdict. Discharging a fired obligation is a HUMAN act - the record lists what fired,
it never asserts an obligation was discharged.

This record is honestly a MEASUREMENT (``provenance.is_measurement`` is true): the sealed payload is
the per-step exit codes as OBSERVED on this host/toolchain, not a pure function of the tree - a flaky
test or a bumped tool can flip ``gate_passed`` on a byte-identical tree. No wall-clock timestamps or
timing-laden output are sealed, so a re-run on the same host+toolchain reproduces it, but that is a
weaker promise than the deterministic change-set/impact records make. A step that cannot be launched
at all (missing interpreter, embedded-NUL argv) fails CLOSED as :class:`GateError` (exit 2); a step
that runs and returns a non-expected code is a red gate (exit 1); a step that TIMES OUT is bucketed
red too (``passed=False`` + ``timed_out=True`` disambiguates it) - it ran but overran, which is
nearer a failing run than an un-runnable one.

Stdlib-only; no-shell (every step is an ``argv`` list, never a shell string); reuses the kernel
verbatim (change set, canonical JSON, the sealed envelope) and the impact policy loader/matcher.
"""

from __future__ import annotations

import subprocess  # noqa: S404 - fixed-argv gate steps only; never a shell string.
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from assurance import KERNEL_VERSION
from assurance.canonical_json import sha256_of_bytes
from assurance.git_state import AssuranceError, discover_git_context
from assurance.obligations import DEFAULT_POLICY_RELPATH, load_effective_policy, match_obligations
from assurance.records import RECORD_SCHEMA_VERSION, build_change_set_record, seal_record

GATE_SCHEMA_VERSION = 1
VERIFICATION_RECORD_TYPE = "workspace_verification"
VERIFICATION_SCHEMA_VERSION = 1
DEFAULT_GATE_RELPATH = "scripts/assurance/policy/gate.json"
COMPLETION_LEVEL = "WORKSPACE_GATE"
_DEFAULT_STEP_TIMEOUT_S = 3600

_ALLOWED_GATE_TOP_KEYS = frozenset({"schema_version", "description", "steps"})
_ALLOWED_STEP_KEYS = frozenset({"name", "argv", "cwd", "expected_exit"})
_TIMEOUT_EXIT = -1  # sentinel exit code recorded for a step that timed out (never a real exit code)


class GateError(AssuranceError):
    """The verification gate spec is malformed, or a gate step could not be launched (exit 2)."""


@dataclass(frozen=True)
class GateStep:
    """One declared gate step: a no-shell argv run from a repo-relative cwd, expecting an exit code."""

    name: str
    argv: tuple[str, ...]
    cwd: str
    expected_exit: int


@dataclass(frozen=True)
class LoadedGate:
    """A validated gate spec plus the digest of its exact on-disk bytes (for record lineage)."""

    steps: tuple[GateStep, ...]  # declaration order preserved
    digest: str
    schema_version: int
    relpath: str


@dataclass(frozen=True)
class StepResult:
    """The outcome of running one gate step: the real exit code and whether it matched expectation."""

    name: str
    expected_exit: int
    exit_code: int
    passed: bool
    timed_out: bool


def _validate_relpath(what: str, value: object) -> str:
    """Return a repo-relative POSIX path (files stay in the tree); fail closed otherwise. '.' is ok."""
    if not isinstance(value, str) or not value:
        raise GateError(f"{what} is empty/non-string")
    if "\\" in value:
        raise GateError(f"{what} {value!r} contains a backslash (use POSIX '/')")
    if value.startswith("/"):
        raise GateError(f"{what} {value!r} must be repo-relative (no leading '/')")
    for segment in value.split("/"):
        if segment == ".." or (segment == "" and value != "."):
            raise GateError(f"{what} {value!r} has an empty or '..' path segment")
    return value


def parse_gate(raw: dict[str, Any]) -> list[GateStep]:
    """Validate a parsed gate document into steps. Fails closed on any defect."""
    if not isinstance(raw, dict):
        raise GateError("gate spec root is not an object")
    unknown = set(raw) - _ALLOWED_GATE_TOP_KEYS
    if unknown:
        raise GateError(f"gate spec has unknown top-level keys: {sorted(unknown)}")
    if raw.get("schema_version") != GATE_SCHEMA_VERSION:
        raise GateError(f"gate schema_version must be {GATE_SCHEMA_VERSION}, got {raw.get('schema_version')!r}")
    entries = raw.get("steps")
    if not isinstance(entries, list) or not entries:
        raise GateError("gate spec has no non-empty 'steps' list")
    steps: list[GateStep] = []
    seen_names: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise GateError(f"gate step[{index}] is not an object")
        keys = set(entry)
        if keys != set(_ALLOWED_STEP_KEYS):
            missing = sorted(_ALLOWED_STEP_KEYS - keys)
            extra = sorted(keys - _ALLOWED_STEP_KEYS)
            raise GateError(f"gate step[{index}] key mismatch (missing={missing}, unknown={extra})")
        name = entry["name"]
        if not isinstance(name, str) or not name:
            raise GateError(f"gate step[{index}] has an empty/non-string name")
        if name in seen_names:
            raise GateError(f"gate step name {name!r} is used more than once (duplicate)")
        seen_names.add(name)
        argv = entry["argv"]
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) and a for a in argv):
            raise GateError(f"gate step {name!r} argv must be a non-empty list of non-empty strings")
        expected_exit = entry["expected_exit"]
        # Bound to a real process exit code (0..255). A negative value would be a SIGNAL-death code
        # (e.g. -11 SIGSEGV), so declaring one would let a CRASHING step read as "expected" == green;
        # it also removes any aliasing with the -1 timeout sentinel.
        if not isinstance(expected_exit, int) or isinstance(expected_exit, bool) or not 0 <= expected_exit <= 255:
            raise GateError(f"gate step {name!r} expected_exit must be an int in 0..255")
        cwd = _validate_relpath(f"gate step {name!r} cwd", entry["cwd"])
        steps.append(GateStep(name=name, argv=tuple(argv), cwd=cwd, expected_exit=expected_exit))
    return steps


def load_gate(root: Path, gate_relpath: str = DEFAULT_GATE_RELPATH) -> LoadedGate:
    """Read + validate the on-disk gate spec under ``root``; fail closed on any read/parse defect."""
    import json  # noqa: PLC0415 - local to keep the module import surface minimal.

    path = root / gate_relpath
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise GateError(f"gate spec could not be read ({gate_relpath}): {exc}") from exc
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        # RecursionError (deeply-nested JSON) is NOT a ValueError - catch it so a hostile gate spec
        # fails CLOSED (exit 2), never as a bare traceback + exit 1.
        raise GateError(f"gate spec is not valid UTF-8 JSON ({gate_relpath}): {exc}") from exc
    steps = parse_gate(raw)
    return LoadedGate(
        steps=tuple(steps),
        digest=sha256_of_bytes(raw_bytes),
        schema_version=GATE_SCHEMA_VERSION,
        relpath=gate_relpath,
    )


def run_gate(root: Path, gate: LoadedGate, timeout: int = _DEFAULT_STEP_TIMEOUT_S) -> list[StepResult]:
    """Run each gate step (no-shell argv, from its declared cwd) and capture the real exit code.

    A step that runs and returns a non-expected code is a red gate (``passed=False``). A step that
    times out is also ``passed=False`` (``timed_out=True``, sentinel exit). A step that cannot be
    LAUNCHED at all (missing interpreter/binary) fails CLOSED as :class:`GateError` - the gate could
    not be evaluated, which must never masquerade as either pass or fail.
    """
    results: list[StepResult] = []
    for step in gate.steps:
        workdir = root / step.cwd
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv list, no shell, trusted in-repo policy.
                list(step.argv),
                cwd=str(workdir),
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            # A step that ran too long is NOT a pass (distinct sentinel + timed_out flag).
            results.append(StepResult(step.name, step.expected_exit, _TIMEOUT_EXIT, False, True))
            continue
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            # Could-not-EVALUATE (missing/unrunnable interpreter, an embedded-NUL argv/cwd, or any
            # other launch/run error) fails CLOSED as exit 2. It must never masquerade as a red gate
            # (exit 1) or escape as a bare traceback - that is the exact false signal verify prevents.
            raise GateError(f"gate step {step.name!r} could not be launched or run ({exc}); cannot verify") from exc
        results.append(
            StepResult(
                name=step.name,
                expected_exit=step.expected_exit,
                exit_code=proc.returncode,
                passed=proc.returncode == step.expected_exit,
                timed_out=False,
            )
        )
    return results


def build_verification_record(
    *,
    start_dir: Path,
    scope: str = "workspace",
    base_ref: str = "main",
    gate_relpath: str = DEFAULT_GATE_RELPATH,
    policy_relpath: str = DEFAULT_POLICY_RELPATH,
    timeout: int = _DEFAULT_STEP_TIMEOUT_S,
) -> dict[str, Any]:
    """Run the gate + map obligations on the current change set, and seal a WorkspaceVerification.

    ``gate_passed`` is workspace-level ONLY (see the module honesty ceiling). ``fired_obligations``
    are those the impact policy fires on this change set - listed for the human to discharge, never
    asserted discharged.
    """
    ctx = discover_git_context(start_dir)
    gate = load_gate(ctx.root, gate_relpath)
    policy, base_policy_available = load_effective_policy(ctx, base_ref, policy_relpath)
    change_set = build_change_set_record(start_dir=start_dir, scope=scope, base_ref=base_ref)
    cs_payload = change_set["payload"]
    cs_provenance = change_set["provenance"]
    changed_paths = [cp["path"] for cp in cs_payload["changed_paths"]]
    fired, _unmatched = match_obligations(changed_paths, list(policy.obligations))

    # Run the gate LAST (after the read-only steps above), so a malformed policy/change set fails
    # closed before we spend minutes running pytest.
    step_results = run_gate(ctx.root, gate, timeout)
    gate_passed = all(step.passed for step in step_results)

    payload = {
        "scope": scope,
        "completion_level": COMPLETION_LEVEL,
        "base_oid": cs_payload["base_oid"],
        "change_set_fingerprint": cs_payload["fingerprint"],
        "changed_path_count": cs_payload["changed_path_count"],
        "gate_passed": gate_passed,
        "gate_step_count": len(step_results),
        "gate_passed_count": sum(1 for step in step_results if step.passed),
        "gate_steps": [
            {
                "name": step.name,
                "expected_exit": step.expected_exit,
                "exit_code": step.exit_code,
                "passed": step.passed,
                "timed_out": step.timed_out,
            }
            for step in sorted(step_results, key=lambda s: s.name)
        ],
        "obligation_count": len(fired),
        "base_policy_available": base_policy_available,
        "fired_obligations": [
            {
                "id": f["id"],
                "severity": f["severity"],
                "obligation": f["obligation"],
                "source": f["source"],
                "trigger_path_count": f["trigger_path_count"],
            }
            for f in fired  # already sorted by id
        ],
    }
    provenance = {
        "tool": "cs_assure",
        "tool_version": KERNEL_VERSION,
        "record_kernel_schema_version": RECORD_SCHEMA_VERSION,
        "subcommand": "verify",
        "base_ref": cs_provenance["base_ref"],
        "base_oid": cs_provenance["base_oid"],
        "head_oid": cs_provenance["head_oid"],
        "is_shallow": cs_provenance["is_shallow"],
        "change_set_digest": change_set["record_digest"],
        "gate_path": gate.relpath,
        "gate_schema_version": gate.schema_version,
        "gate_digest": gate.digest,
        "policy_path": policy.relpath,
        "policy_digest": policy.digest,
        # The gate exit codes are observed on this host/toolchain (flaky-test/tool-version dependent),
        # not a pure function of the tree - labelled honestly, matching status.py.
        "is_measurement": True,
    }
    return seal_record(VERIFICATION_RECORD_TYPE, VERIFICATION_SCHEMA_VERSION, payload, provenance)
