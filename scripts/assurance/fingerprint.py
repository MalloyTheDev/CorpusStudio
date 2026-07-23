"""State-based change-set fingerprint (Phase 1).

The fingerprint is the *applicability key* of a change set: a deterministic digest over
``(scope, base_oid, the sorted per-path base/candidate FileStates)``. It is INDEPENDENT of the
requested ref name, the tool version, and any capture time - so two runs over the same content
against the same base commit produce the same fingerprint (stability), while any single changed
byte or mode change produces a different one (invalidation).

This is deliberately separate from a record's integrity digest (``records.record_digest``),
which covers the whole envelope including provenance. Integrity answers "are these bytes intact";
the fingerprint answers "does this change set still describe the same content-level change" - the
trust model's *record applicability* vs *record integrity* distinction.
"""

from __future__ import annotations

from typing import Any

from assurance.canonical_json import sha256_digest
from assurance.source_views import FileState


def changed_path_record(
    path: str, base_state: FileState | None, candidate_state: FileState | None
) -> dict[str, Any]:
    """Build the deterministic per-path record. ``None`` on a side means absent (added/deleted)."""
    return {
        "path": path,
        "base": base_state.to_record() if base_state is not None else None,
        "candidate": candidate_state.to_record() if candidate_state is not None else None,
    }


def fingerprint_input(scope: str, base_oid: str, changed_paths: list[dict[str, Any]]) -> dict[str, Any]:
    """The exact object the fingerprint digests. ``changed_paths`` must already be sorted by path."""
    return {"scope": scope, "base_oid": base_oid, "paths": changed_paths}


def compute_fingerprint(scope: str, base_oid: str, changed_paths: list[dict[str, Any]]) -> str:
    """Return the ``sha256:`` fingerprint of the change set (applicability key)."""
    return sha256_digest(fingerprint_input(scope, base_oid, changed_paths))
