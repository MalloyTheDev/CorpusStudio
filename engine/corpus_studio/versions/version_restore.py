"""Reconstruct a dataset version's rows from the row store (v1.0.3).

Rebuilds the exact rows of a version, in manifest order, from the content-addressed
store, and verifies fidelity by re-fingerprinting: because ``exact_row_signature``
is idempotent on canonical rows, a faithful reconstruction MUST reproduce the
version's recorded ``content_fingerprint``. That turns "trust me, this is your old
data" into a proof.

The rows are rebuilt in **canonical** form (sorted keys), so a restore is not
byte-identical to the original file — the fingerprint match proves the rows are
*semantically* identical, which is the honest promise.

This module is pure (no file writes). The CLI performs the atomic write to an
``--output`` path and never touches ``examples.jsonl`` — in-place restore is a
desktop operation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from corpus_studio.exporters.cleaning import exact_row_signature


class RestoreResult(BaseModel):
    version_id: str
    rows_written: int
    verified: bool
    verify_skipped: bool
    output_path: str


def reconstruct_and_verify(
    manifest_ids: list[str],
    rows_by_id: dict[str, Any],
    expected_fingerprint: str | None,
) -> tuple[list[str], str, bool, list[str]]:
    """Rebuild the version's JSONL lines from its ordered manifest and the store.

    Returns ``(lines, computed_fingerprint, matches, missing_ids)``:

    - ``lines`` — canonical JSONL (one row per manifest occurrence, in order),
      or ``[]`` when any row is missing (a partial restore is never emitted).
    - ``computed_fingerprint`` — SHA-256 over the ordered per-row
      ``exact_row_signature``, identical to how the version was fingerprinted;
      meaningful only when ``missing_ids`` is empty.
    - ``matches`` — ``True`` iff ``expected_fingerprint`` is set and equals the
      computed one (a faithful, verifiable restore).
    - ``missing_ids`` — every manifest id absent from ``rows_by_id`` (all-or-nothing).
    """

    missing_ids: list[str] = []
    lines: list[str] = []
    digest = hashlib.sha256()
    fed = 0
    for row_id in manifest_ids:
        if row_id not in rows_by_id:
            missing_ids.append(row_id)
            continue
        row = rows_by_id[row_id]
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
        if fed:
            digest.update(b"\n")
        digest.update(exact_row_signature(row).encode("utf-8"))
        fed += 1

    computed = digest.hexdigest()
    if missing_ids:
        # Never emit a partial reconstruction; the computed digest is discarded.
        return [], computed, False, missing_ids

    matches = expected_fingerprint is not None and computed == expected_fingerprint
    return lines, computed, matches, missing_ids


def reconstruct_version_lines(project_dir: Any, version_id: str) -> list[str]:
    """Reconstruct + VERIFY a stored dataset version's rows as canonical JSONL lines.

    Raises FileNotFoundError (no such version) or ValueError (corrupt record, no stored
    rows, missing rows, or fingerprint mismatch) — the reconstruction is all-or-nothing.
    Reused by both `dataset-version-restore` (via the CLI) and version-pinned suite cases,
    so a pinned case evaluates the *verified* version, not a mutable path.
    """

    from pathlib import Path

    from corpus_studio.versions.row_store import load_rows_by_id
    from corpus_studio.versions.version_registry import (
        load_row_manifest,
        load_version_record,
        record_path,
    )

    project = Path(project_dir)
    path = record_path(project, version_id)
    if not path.exists():
        raise FileNotFoundError(f"No dataset version '{version_id}'.")

    record = load_version_record(path)
    manifest = load_row_manifest(project, version_id)
    if not record.rows_stored or manifest is None:
        raise ValueError(
            f"Version '{version_id}' has no stored rows; recapture with "
            "dataset-version-create --store-rows to pin it."
        )

    rows_by_id = load_rows_by_id(project, set(manifest))
    lines, _computed, matches, missing_ids = reconstruct_and_verify(
        manifest, rows_by_id, record.content_fingerprint
    )
    if missing_ids:
        raise ValueError(f"Version '{version_id}' is missing {len(missing_ids)} row(s) from the store.")
    if not matches:
        raise ValueError(f"Version '{version_id}' failed fingerprint verification.")
    return lines
