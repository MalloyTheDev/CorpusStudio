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
touches ``examples.jsonl`` or any weight file. Rows referenced by no version
manifest are pruned only by row-store GC (``versions/gc.py``). Every append runs
under the version-store lock (``versions/store_lock.py``) so GC can never rewrite
the store underneath it; readers here stay lock-free.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator

from corpus_studio.exporters.cleaning import exact_row_signature
from corpus_studio.versions.store_lock import version_store_lock
from corpus_studio.versions.version_registry import DATASET_VERSION_REGISTRY_DIRNAME

ROW_STORE_FILENAME = "row_store.jsonl"

# Identity algorithm tag: row_id = sha256(exact_row_signature). Versioned so a
# future normalized (near-duplicate) identity is additive, never a silent
# reinterpretation of an already-stored manifest.
ROW_MANIFEST_ALGO = "sha256-exact-v1"

_UTF8_BOM = b"\xef\xbb\xbf"
# JSON's own insignificant whitespace (RFC 8259): stripping only these cannot change what a line
# parses to, and a CRLF terminator is still tolerated.
_JSON_WHITESPACE = b" \t\r\n"


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


def _iter_store_entries(handle: BinaryIO) -> Iterator[dict[str, Any]]:
    """Yield every JSON-object line of an open row store, in order.

    Reads bytes and splits on ``\\n`` only (never on U+2028/U+2029/U+0085, which ``store_line``
    writes raw inside a row), then decodes each line on its own, so one damaged line never hides
    the rows around it: a blank line, a torn/partial line, a line that is not valid UTF-8 (a tear
    inside a multi-byte character), or a line that is not a JSON object is skipped. A UTF-8 BOM on
    the first line is tolerated, matching GC's classification. Raises ``OSError``."""

    first = True
    for raw in handle:
        if first:
            first = False
            if raw.startswith(_UTF8_BOM):
                raw = raw[len(_UTF8_BOM) :]
        stripped = raw.strip(_JSON_WHITESPACE)
        if not stripped:
            continue
        try:
            # UnicodeDecodeError is a ValueError too: an undecodable line is skipped like a torn one.
            entry = json.loads(stripped.decode("utf-8"))
        except ValueError:
            continue
        if isinstance(entry, dict):
            yield entry


def load_row_id_set(project_dir: Path | str) -> set[str]:
    """The set of row_ids already in the store. Tolerant: skips blank, torn and
    undecodable lines; a missing store is an empty set."""

    path = row_store_path(project_dir)
    if not path.exists():
        return set()
    ids: set[str] = set()
    try:
        with path.open("rb") as handle:
            for entry in _iter_store_entries(handle):
                entry_id = entry.get("row_id")
                if isinstance(entry_id, str):
                    ids.add(entry_id)
    except OSError:
        return ids
    return ids


def terminate_torn_tail(path: Path) -> bool:
    """Newline-terminate a store whose last line is partial (a writer died mid-append).

    Appending straight after such a fragment would glue the next row onto it, making
    that row unreadable while its version still claims it is stored. Terminating it
    leaves the fragment as its own unclassifiable line, which readers skip and GC
    never prunes. The repair is byte-level: the fragment is never decoded, so a tear
    inside a multi-byte UTF-8 character is handled like any other (the readers skip
    that undecodable line). The store is opened for writing only when a repair is
    needed, so an intact store that is readable but not writable is left alone.
    Returns True when a newline was added; a missing or empty store is left alone.
    Raises ``OSError``. Call only under the version-store lock."""

    try:
        with path.open("rb") as reader:
            end = reader.seek(0, os.SEEK_END)
            if end == 0:
                return False
            reader.seek(end - 1)
            if reader.read(1) == b"\n":
                return False
    except FileNotFoundError:
        return False
    with path.open("ab") as writer:
        writer.write(b"\n")
        writer.flush()
        os.fsync(writer.fileno())
    return True


def append_rows(
    project_dir: Path | str,
    rows: Iterable[tuple[str, Any]],
    existing_ids: set[str],
) -> int:
    """Append only rows whose id is not already present; returns the count newly
    written. Streams (opens the file lazily, writes per row - never buffers row
    bodies). ``existing_ids`` is updated in place so within-call duplicates are
    written once. Content-addressed => a duplicate/torn line is harmless.

    Runs under the version-store lock (reentrant). The appended rows stay prunable
    by GC until a manifest names them, so a caller that publishes one must hold the
    lock across both steps."""

    path = row_store_path(project_dir)
    handle = None
    written = 0
    with version_store_lock(project_dir, operation="row-store append"):
        try:
            for rid, row in rows:
                if rid in existing_ids:
                    continue
                existing_ids.add(rid)
                if handle is None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    terminate_torn_tail(path)
                    handle = path.open("a", encoding="utf-8")
                handle.write(store_line(rid, row))
                written += 1
        finally:
            if handle is not None:
                handle.close()
    return written


def load_rows_by_id(project_dir: Path | str, ids: set[str]) -> dict[str, Any]:
    """Return ``{row_id: row}`` for the requested ids found in the store. Tolerant
    of blank, torn and undecodable lines; ids not present are simply omitted (an
    orphaned manifest entry degrades to 'missing', it does not crash)."""

    path = row_store_path(project_dir)
    if not path.exists() or not ids:
        return {}
    found: dict[str, Any] = {}
    try:
        with path.open("rb") as handle:
            for entry in _iter_store_entries(handle):
                entry_id = entry.get("row_id")
                if isinstance(entry_id, str) and entry_id in ids and entry_id not in found:
                    found[entry_id] = entry.get("row")
                    if len(found) == len(ids):
                        break
    except OSError:
        return found
    return found
