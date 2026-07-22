"""Immutable assurance records + the Phase-1 ChangeSetRecord builder.

Every immutable assurance record shares one envelope::

    {"record_type", "schema_version", "payload", "provenance", "record_digest"}

``record_digest`` is the ``sha256:`` digest of the canonical envelope with the ``record_digest``
field itself EXCLUDED - so a record's integrity can be re-verified by dropping that one field and
re-canonicalizing. Records are never mutated in place; any repository mutation would create a new
``ChangeSetRecord`` root (that lifecycle belongs to a later phase - Phase 1 only builds the root).

A ``ChangeSetRecord`` describes what changed between two source views:
  * base view    = the tree at ``merge-base(HEAD, --base)`` (read from the object store),
  * candidate view = the local working tree (Phase 1 implements the ``workspace`` scope only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from assurance import KERNEL_VERSION
from assurance.canonical_json import sha256_digest
from assurance.fingerprint import changed_path_record, compute_fingerprint
from assurance.git_state import (
    AssuranceError,
    changed_tracked_paths,
    discover_git_context,
    merge_base,
    resolve_commit,
    untracked_paths,
)
from assurance.source_views import FileState, GitTreeSourceView, WorkspaceSourceView

RECORD_SCHEMA_VERSION = 1
CHANGE_SET_RECORD_TYPE = "change_set"
# Phase 1 implements the workspace scope only. index / head / merge_candidate are a later phase.
SUPPORTED_SCOPES = ("workspace",)


class ChangeSetError(AssuranceError):
    """A change set cannot be produced for the requested state."""


class ScopeNotImplemented(ChangeSetError):
    """A Git scope other than the Phase-1 ``workspace`` scope was requested."""


class ChangeSetUnstable(ChangeSetError):
    """The working tree changed during collection; two snapshots of the same state disagree."""


def seal_record(
    record_type: str, schema_version: int, payload: Any, provenance: Any
) -> dict[str, Any]:
    """Return the sealed record envelope: the four fields plus a ``record_digest`` over them."""
    envelope = {
        "record_type": record_type,
        "schema_version": schema_version,
        "payload": payload,
        "provenance": provenance,
    }
    return {**envelope, "record_digest": sha256_digest(envelope)}


def verify_record(record: dict[str, Any]) -> bool:
    """Re-derive the record digest (excluding the field itself) and compare it to the claim."""
    if "record_digest" not in record:
        return False
    envelope = {key: value for key, value in record.items() if key != "record_digest"}
    return sha256_digest(envelope) == record["record_digest"]


def _states_equal(left: FileState | None, right: FileState | None) -> bool:
    if left is None or right is None:
        return left is right
    return left == right


def _collect_change_set(
    root: Path, scope: str, base_oid: str
) -> tuple[str, list[dict[str, Any]]]:
    """One snapshot pass: return ``(fingerprint, sorted changed-path records)``."""
    base_view = GitTreeSourceView(root, base_oid)
    candidate_view = WorkspaceSourceView(root)
    candidate_paths = set(changed_tracked_paths(root, base_oid)) | set(untracked_paths(root))
    records: list[dict[str, Any]] = []
    for path in sorted(candidate_paths):
        base_state = base_view.state(path)
        candidate_state = candidate_view.state(path)
        if _states_equal(base_state, candidate_state):
            continue  # defensive: only emit genuinely differing states
        records.append(changed_path_record(path, base_state, candidate_state))
    fingerprint = compute_fingerprint(scope, base_oid, records)
    return fingerprint, records


def build_change_set_record(
    *,
    start_dir: Path,
    scope: str = "workspace",
    base_ref: str = "main",
    _between_passes: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Build a sealed ChangeSetRecord for the selected Git state.

    Two independent snapshot passes are taken and their fingerprints compared; if they disagree
    the working tree moved during collection and the kernel fails closed with
    :class:`ChangeSetUnstable` rather than emit an internally inconsistent record. ``_between_passes``
    is a test-only hook invoked between the two passes.
    """
    if scope not in SUPPORTED_SCOPES:
        raise ScopeNotImplemented(
            f"scope {scope!r} is not implemented in the Phase-1 kernel "
            f"(supported: {', '.join(SUPPORTED_SCOPES)})"
        )
    ctx = discover_git_context(start_dir)
    base_commit = resolve_commit(ctx.root, base_ref)
    base_oid = merge_base(ctx, base_commit)

    fingerprint, records = _collect_change_set(ctx.root, scope, base_oid)
    if _between_passes is not None:
        _between_passes()
    second_fingerprint, _second_records = _collect_change_set(ctx.root, scope, base_oid)
    if fingerprint != second_fingerprint:
        raise ChangeSetUnstable(
            "the working tree changed during change-set collection (two snapshots disagree); "
            "re-run against a quiescent tree"
        )

    payload = {
        "scope": scope,
        "base_view": "merge_base",
        "base_oid": base_oid,
        "candidate_view": "workspace",
        "changed_path_count": len(records),
        "changed_paths": records,
        "fingerprint": fingerprint,
    }
    provenance = {
        "tool": "cs_assure",
        "tool_version": KERNEL_VERSION,
        "record_kernel_schema_version": RECORD_SCHEMA_VERSION,
        "base_ref": base_ref,
        "base_oid": base_oid,
        "head_oid": ctx.head_oid,
        "is_shallow": ctx.is_shallow,
    }
    return seal_record(CHANGE_SET_RECORD_TYPE, RECORD_SCHEMA_VERSION, payload, provenance)
