"""Impact -> Obligations engine for the assurance loop (Phase 4).

Given a change set (the Phase-1 kernel's ``ChangeSetRecord``), this maps each changed repository
path against a machine policy bundle (``scripts/assurance/policy/obligations.json``) and emits a
sealed :class:`ImpactAssessment` record listing which OBLIGATIONS fire and which changed paths
triggered each. It answers "given what changed, what must I now do?" - turning the manual
classify-every-change step into a deterministic checklist.

It is **OBSERVATION-ONLY**: it reports fired obligations for the engineer/agent to discharge; it
never enforces, gates, auto-fixes, blocks, runs verification, or emits evidence. ``severity`` is
descriptive metadata - a ``blocking`` and an ``advisory`` obligation are reported identically. There
is no ``--strict`` mode and no severity-driven exit; gating is a separate future phase.

Scope of the matcher: it flags only the obligations' DECLARED globs. It does NOT compute a worker
runtime-import closure or any reachability graph (that is the worker-closure rule's symbol-level
trace / a later phase); the fired obligation TEXT is what tells the human to perform that trace.

Stdlib-only; reuses the kernel verbatim (canonical JSON, the change-set builder, the sealed
four-field envelope + ``record_digest``). Malformed policy fails closed as :class:`PolicyError`
(an ``AssuranceError`` the CLI maps to exit 2).
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from assurance import KERNEL_VERSION
from assurance.canonical_json import sha256_digest, sha256_of_bytes
from assurance.git_state import (
    AssuranceError,
    GitContext,
    discover_git_context,
    merge_base,
    read_committed_file,
)
from assurance.records import RECORD_SCHEMA_VERSION, build_change_set_record, seal_record

_SEVERITY_RANK = {"info": 0, "advisory": 1, "blocking": 2}

POLICY_SCHEMA_VERSION = 1
IMPACT_RECORD_TYPE = "impact_assessment"
IMPACT_SCHEMA_VERSION = 1
DEFAULT_POLICY_RELPATH = "scripts/assurance/policy/obligations.json"
VALID_SEVERITIES = ("info", "advisory", "blocking")

_ALLOWED_TOP_KEYS = frozenset({"schema_version", "description", "severities", "obligations"})
_ALLOWED_OBLIGATION_KEYS = frozenset({"id", "globs", "obligation", "authority", "severity", "source"})
# Catch-alls that would fire on every change - refused so the policy stays specific and honest.
_UNBOUNDED_GLOBS = frozenset({"*", "**", "/**"})


class PolicyError(AssuranceError):
    """The obligations policy bundle is malformed or missing (fail-closed, CLI exit 2)."""


class ImpactError(AssuranceError):
    """An impact-assessment refusal not covered by a more specific error (fail-closed, exit 2)."""


@dataclass(frozen=True)
class Obligation:
    """One policy obligation: which path globs trigger it and what it obliges (data, not behavior)."""

    id: str
    globs: tuple[str, ...]
    obligation: str
    authority: str
    severity: str
    source: str


@dataclass(frozen=True)
class LoadedPolicy:
    """A validated policy bundle plus the digest of its exact on-disk bytes (for record lineage)."""

    obligations: tuple[Obligation, ...]  # sorted by id
    digest: str  # sha256: of the raw policy file bytes
    schema_version: int
    relpath: str
    obligation_count: int


def _validate_glob(ob_id: str, glob: object) -> str:
    """Return a repo-relative POSIX glob or fail closed. Rejects absolute / traversal / catch-alls."""
    if not isinstance(glob, str) or not glob:
        raise PolicyError(f"obligation {ob_id!r} has an empty/non-string glob")
    if "\\" in glob:
        raise PolicyError(f"obligation {ob_id!r} glob {glob!r} contains a backslash (use POSIX '/')")
    if glob.startswith("/"):
        raise PolicyError(f"obligation {ob_id!r} glob {glob!r} must be repo-relative (no leading '/')")
    if glob in _UNBOUNDED_GLOBS:
        raise PolicyError(f"obligation {ob_id!r} glob {glob!r} is an unbounded catch-all")
    if "**" in glob and not glob.endswith("/**"):
        raise PolicyError(
            f"obligation {ob_id!r} glob {glob!r}: '**' is only supported as a trailing '/**'"
        )
    for segment in glob.split("/"):
        if segment in ("", ".", ".."):
            raise PolicyError(f"obligation {ob_id!r} glob {glob!r} has an empty / '.' / '..' segment")
        if ":" in segment:
            raise PolicyError(f"obligation {ob_id!r} glob {glob!r} segment {segment!r} contains ':'")
    return glob


def parse_policy(raw: dict[str, Any]) -> list[Obligation]:
    """Validate a parsed policy document into obligations (sorted by id). Fails closed on any defect."""
    if not isinstance(raw, dict):
        raise PolicyError("obligations policy root is not an object")
    unknown = set(raw) - _ALLOWED_TOP_KEYS
    if unknown:
        raise PolicyError(f"obligations policy has unknown top-level keys: {sorted(unknown)}")
    if raw.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise PolicyError(
            f"obligations policy schema_version must be {POLICY_SCHEMA_VERSION}, "
            f"got {raw.get('schema_version')!r}"
        )
    entries = raw.get("obligations")
    if not isinstance(entries, list) or not entries:
        raise PolicyError("obligations policy has no non-empty 'obligations' list")
    obligations: list[Obligation] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PolicyError(f"obligation[{index}] is not an object")
        keys = set(entry)
        if keys != set(_ALLOWED_OBLIGATION_KEYS):
            missing = sorted(_ALLOWED_OBLIGATION_KEYS - keys)
            extra = sorted(keys - _ALLOWED_OBLIGATION_KEYS)
            raise PolicyError(f"obligation[{index}] key mismatch (missing={missing}, unknown={extra})")
        ob_id = entry["id"]
        if not isinstance(ob_id, str) or not ob_id:
            raise PolicyError(f"obligation[{index}] has an empty/non-string id")
        if ob_id in seen_ids:
            raise PolicyError(f"obligation id {ob_id!r} is registered more than once (duplicate)")
        seen_ids.add(ob_id)
        severity = entry["severity"]
        if severity not in VALID_SEVERITIES:
            raise PolicyError(
                f"obligation {ob_id!r}: invalid severity {severity!r} (allowed: {VALID_SEVERITIES})"
            )
        globs = entry["globs"]
        if not isinstance(globs, list) or not globs:
            raise PolicyError(f"obligation {ob_id!r} has an empty/non-list 'globs'")
        validated_globs = tuple(_validate_glob(ob_id, g) for g in globs)
        for field in ("obligation", "authority", "source"):
            if not isinstance(entry[field], str) or not entry[field]:
                raise PolicyError(f"obligation {ob_id!r} has an empty/non-string {field!r}")
        obligations.append(
            Obligation(
                id=ob_id,
                globs=validated_globs,
                obligation=entry["obligation"],
                authority=entry["authority"],
                severity=severity,
                source=entry["source"],
            )
        )
    obligations.sort(key=lambda o: o.id)
    return obligations


def load_policy(root: Path, policy_relpath: str = DEFAULT_POLICY_RELPATH) -> LoadedPolicy:
    """Read + validate the on-disk policy under ``root``; fail closed on any read/parse/schema defect."""
    path = root / policy_relpath
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise PolicyError(f"obligations policy could not be read ({policy_relpath}): {exc}") from exc
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        # RecursionError (deeply-nested JSON) is NOT a ValueError - catch it so a hostile policy fails
        # CLOSED (exit 2), never as a bare traceback + exit 1.
        raise PolicyError(f"obligations policy is not valid UTF-8 JSON ({policy_relpath}): {exc}") from exc
    obligations = parse_policy(raw)
    return LoadedPolicy(
        obligations=tuple(obligations),
        digest=sha256_of_bytes(raw_bytes),
        schema_version=POLICY_SCHEMA_VERSION,
        relpath=policy_relpath,
        obligation_count=len(obligations),
    )


def load_base_policy(ctx: GitContext, base_ref: str,
                     policy_relpath: str = DEFAULT_POLICY_RELPATH) -> LoadedPolicy | None:
    """The TRUSTED policy as committed at the merge-base of HEAD and ``base_ref`` (the last reviewed +
    merged point). Returns None when the trusted base is unavailable - no merge base, the policy file was
    absent there, a git error, or a malformed base policy - so the caller can honestly flag that the
    no-weakening guarantee could not be applied (candidate-only) rather than crash or silently trust the
    candidate. Never touches the working tree."""
    try:
        base_commit = merge_base(ctx, base_ref)
        raw_bytes = read_committed_file(ctx, base_commit, policy_relpath)
    except AssuranceError:
        return None
    if raw_bytes is None:
        return None
    try:
        obligations = parse_policy(json.loads(raw_bytes.decode("utf-8")))
    except (ValueError, UnicodeDecodeError, RecursionError, PolicyError):
        return None
    return LoadedPolicy(
        obligations=tuple(obligations),
        digest=sha256_of_bytes(raw_bytes),
        schema_version=POLICY_SCHEMA_VERSION,
        relpath=policy_relpath,
        obligation_count=len(obligations),
    )


def _stronger_severity(a: str, b: str) -> str:
    return a if _SEVERITY_RANK.get(a, 0) >= _SEVERITY_RANK.get(b, 0) else b


def union_policy(candidate: LoadedPolicy, base: LoadedPolicy) -> LoadedPolicy:
    """Effective policy = candidate UNION trusted-base. A candidate may STRENGTHEN (add obligations, add
    globs, raise severity) but never WEAKEN: every base obligation is preserved (a candidate cannot remove
    it), a shared id keeps the UNION of globs and the STRONGER severity, and a candidate-only obligation is
    added. So the fired-obligation list can never be shrunk below the trusted base by editing the policy.
    The digest binds BOTH source digests so the sealed record is honest about the effective policy used."""
    by_base = {o.id: o for o in base.obligations}
    by_cand = {o.id: o for o in candidate.obligations}
    merged: list[Obligation] = []
    for ob_id in sorted(set(by_base) | set(by_cand)):
        b, c = by_base.get(ob_id), by_cand.get(ob_id)
        if b is not None and c is not None:
            merged.append(replace(c, globs=tuple(sorted(set(b.globs) | set(c.globs))),
                                  severity=_stronger_severity(b.severity, c.severity)))
        else:
            merged.append(b if b is not None else c)  # type: ignore[arg-type]
    digest = sha256_digest({"effective_policy_of": {"candidate": candidate.digest, "base": base.digest}})
    return LoadedPolicy(obligations=tuple(merged), digest=digest, schema_version=candidate.schema_version,
                        relpath=candidate.relpath, obligation_count=len(merged))


def load_effective_policy(ctx: GitContext, base_ref: str,
                          policy_relpath: str = DEFAULT_POLICY_RELPATH) -> tuple[LoadedPolicy, bool]:
    """Return (effective_policy, base_policy_available). When the trusted base is available, the effective
    policy is candidate UNION base (a candidate may strengthen, never weaken); otherwise it is the
    candidate policy with base_policy_available=False so the record flags the un-applied guarantee."""
    candidate = load_policy(ctx.root, policy_relpath)
    base = load_base_policy(ctx, base_ref, policy_relpath)
    if base is None:
        return candidate, False
    return union_policy(candidate, base), True


def glob_matches(glob: str, path: str) -> bool:
    """Boundary-correct match of a repo-relative POSIX ``path`` against a policy ``glob``.

    ``dir/**`` matches the directory itself or anything strictly under it (never a sibling that
    merely shares the prefix). Other wildcards use case-sensitive ``fnmatchcase`` - note this means a
    ``*`` DOES cross ``/`` (fnmatch semantics), so a bare ``*``-glob would over-match; the shipped
    policy therefore uses only literal paths, ``dir/**``, and precise ``file_*.py``-style globs where
    over-crossing has no real target. A literal glob is exact-path equality, so a delete of that exact
    path still matches.
    """
    if glob.endswith("/**"):
        prefix = glob[:-3]
        return path == prefix or path.startswith(prefix + "/")
    if any(ch in glob for ch in "*?["):
        return fnmatch.fnmatchcase(path, glob)
    return path == glob


def match_obligations(
    changed_paths: list[str], obligations: list[Obligation]
) -> tuple[list[dict[str, Any]], int]:
    """Map changed paths onto obligations.

    Returns ``(fired, unmatched_path_count)`` where ``fired`` is the list of fired-obligation dicts
    (sorted by id; each with its sorted ``triggers``) and ``unmatched_path_count`` is the number of
    changed paths that fired no obligation. An obligation and a path are many-to-many: one path may
    fire several obligations, and every fired obligation is reported.
    """
    fired: list[dict[str, Any]] = []
    matched_paths: set[str] = set()
    for obligation in obligations:
        triggers: list[dict[str, Any]] = []
        for path in changed_paths:
            hit = sorted(g for g in obligation.globs if glob_matches(g, path))
            if hit:
                triggers.append({"path": path, "globs": hit})
                matched_paths.add(path)
        if triggers:
            triggers.sort(key=lambda t: t["path"])
            fired.append(
                {
                    "id": obligation.id,
                    "severity": obligation.severity,
                    "authority": obligation.authority,
                    "obligation": obligation.obligation,
                    "source": obligation.source,
                    "trigger_path_count": len(triggers),
                    "triggers": triggers,
                }
            )
    fired.sort(key=lambda f: f["id"])
    unmatched_path_count = len(set(changed_paths)) - len(matched_paths)
    return fired, unmatched_path_count


def build_impact_assessment(
    *,
    start_dir: Path,
    scope: str = "workspace",
    base_ref: str = "main",
    policy_relpath: str = DEFAULT_POLICY_RELPATH,
) -> dict[str, Any]:
    """Build a sealed ImpactAssessment: the change set mapped onto the policy's obligations.

    The change set is REFERENCED (by fingerprint + record digest + base/head oids), not re-embedded;
    the meaningful subset - the paths that fired an obligation - lives in each obligation's
    ``triggers``. The policy is the EFFECTIVE policy: the candidate working-tree policy UNIONed with the
    trusted merge-base policy, so a change that edits the policy can STRENGTHEN it but can never WEAKEN
    (remove / shrink) a trusted-base obligation to escape firing it. ``base_policy_available`` records
    whether that union could be applied (false = candidate-only, when no trusted base is reachable).
    """
    ctx = discover_git_context(start_dir)
    policy, base_policy_available = load_effective_policy(ctx, base_ref, policy_relpath)
    change_set = build_change_set_record(start_dir=start_dir, scope=scope, base_ref=base_ref)
    cs_payload = change_set["payload"]
    cs_provenance = change_set["provenance"]
    changed_paths = [cp["path"] for cp in cs_payload["changed_paths"]]
    fired, unmatched_path_count = match_obligations(changed_paths, list(policy.obligations))

    # Fired obligations are a pure function of (change-set applicability, policy bytes): no new state
    # scan exists to fingerprint, so the applicability key is the composite of those two, preserving
    # the kernel's applicability-vs-integrity split (record_digest still covers the whole envelope).
    applicability_key = sha256_digest(
        {"change_set_fingerprint": cs_payload["fingerprint"], "policy_digest": policy.digest}
    )
    payload = {
        "scope": scope,
        "base_oid": cs_payload["base_oid"],
        "change_set_fingerprint": cs_payload["fingerprint"],
        "changed_path_count": cs_payload["changed_path_count"],
        "obligation_count": len(fired),
        "unmatched_path_count": unmatched_path_count,
        "applicability_key": applicability_key,
        "base_policy_available": base_policy_available,
        "fired_obligations": fired,
    }
    provenance = {
        "tool": "cs_assure",
        "tool_version": KERNEL_VERSION,
        "record_kernel_schema_version": RECORD_SCHEMA_VERSION,
        "subcommand": "impact",
        "base_ref": cs_provenance["base_ref"],
        "base_oid": cs_provenance["base_oid"],
        "head_oid": cs_provenance["head_oid"],
        "is_shallow": cs_provenance["is_shallow"],
        "change_set_digest": change_set["record_digest"],
        "policy_path": policy.relpath,
        "policy_schema_version": policy.schema_version,
        "policy_digest": policy.digest,
        "policy_obligation_count": policy.obligation_count,
        "base_policy_available": base_policy_available,
    }
    return seal_record(IMPACT_RECORD_TYPE, IMPACT_SCHEMA_VERSION, payload, provenance)
