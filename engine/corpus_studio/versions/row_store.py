"""Content-addressed, deduped row store for dataset versions (v1.0.2).

Row bodies for captured versions live once in a project-local, append-only,
content-addressed JSONL blob at ``dataset_versions/row_store.jsonl`` — one line
per UNIQUE row: ``{"row_id": <sha256>, "row": <canonical row>}``. Identical rows
across versions are stored once. A per-version ordered *manifest* of row_ids
(see ``version_registry``) references rows here, which is what makes diff and
(later) restore possible.

The stored row is the **canonical** form (``sort_keys=True``, the same
``exact_row_signature`` shape used for identity), so diff and a future restore
normalize key order and whitespace — they reconstruct the same rows in order,
not a byte-identical file.

Hard constraint: this module writes only under ``dataset_versions/``. It never
touches ``examples.jsonl`` or any weight file. There is no GC of orphaned blobs
(rows referenced by no version) in this slice — a store only grows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from corpus_studio.exporters.cleaning import exact_row_signature
from corpus_studio.versions.version_registry import DATASET_VERSION_REGISTRY_DIRNAME

ROW_STORE_FILENAME = "row_store.jsonl"

# Identity algorithm tag: row_id = sha256(exact_row_signature). Versioned so a
# future normalized (near-duplicate) identity is additive, never a silent
# reinterpretation of an already-stored manifest.
ROW_MANIFEST_ALGO = "sha256-exact-v1"


def row_id(row: Any) -> str:
    """Stable content id for a row: sha256 of its canonical exact signature."""

    return hashlib.sha256(exact_row_signature(row).encode("utf-8")).hexdigest()


def store_line(row_id_value: str, row: Any) -> str:
    """The exact on-disk line for one stored row (newline-terminated). Canonical
    (sorted keys) so the store matches the identity signature; ``ensure_ascii``
    off keeps non-ASCII rows human-inspectable."""

    return json.dumps(
        {"row_id": row_id_value, "row": row}, ensure_ascii=False, sort_keys=True
    ) + "\n"


def row_store_path(project_dir: Path | str) -> Path:
    return Path(project_dir) / DATASET_VERSION_REGISTRY_DIRNAME / ROW_STORE_FILENAME


def load_row_id_set(project_dir: Path | str) -> set[str]:
    """The set of row_ids already in the store. Tolerant: skips blank/torn lines;
    a missing store is an empty set."""

    path = row_store_path(project_dir)
    if not path.exists():
        return set()
    ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except ValueError:
                    continue  # skip a torn/partial line
                if isinstance(entry, dict) and isinstance(entry.get("row_id"), str):
                    ids.add(entry["row_id"])
    except OSError:
        return ids
    return ids


def append_rows(
    project_dir: Path | str,
    rows: Iterable[tuple[str, Any]],
    existing_ids: set[str],
) -> int:
    """Append only rows whose id is not already present; returns the count newly
    written. Streams (opens the file lazily, writes per row — never buffers row
    bodies). ``existing_ids`` is updated in place so within-call duplicates are
    written once. Content-addressed => a duplicate/torn line is harmless."""

    path = row_store_path(project_dir)
    handle = None
    written = 0
    try:
        for rid, row in rows:
            if rid in existing_ids:
                continue
            existing_ids.add(rid)
            if handle is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open("a", encoding="utf-8")
            handle.write(store_line(rid, row))
            written += 1
    finally:
        if handle is not None:
            handle.close()
    return written


def load_rows_by_id(project_dir: Path | str, ids: set[str]) -> dict[str, Any]:
    """Return ``{row_id: row}`` for the requested ids found in the store. Tolerant
    of blank/torn lines; ids not present are simply omitted (an orphaned manifest
    entry degrades to 'missing', it does not crash)."""

    path = row_store_path(project_dir)
    if not path.exists() or not ids:
        return {}
    found: dict[str, Any] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except ValueError:
                    continue
                if not isinstance(entry, dict):
                    continue
                entry_id = entry.get("row_id")
                if isinstance(entry_id, str) and entry_id in ids and entry_id not in found:
                    found[entry_id] = entry.get("row")
                    if len(found) == len(ids):
                        break
    except OSError:
        return found
    return found
