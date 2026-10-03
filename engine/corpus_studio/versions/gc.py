"""Row-store garbage collection (issue #197).

The content-addressed row store (``dataset_versions/row_store.jsonl``) keeps one line per unique row
ever captured. Over many captures it accumulates rows that no surviving version references. GC prunes
exactly those, and **never** a row referenced by any version manifest.

Safety is the whole point, so this is deliberately fail-closed:

* The live set is the union of row-ids across **every** ``*.rows`` manifest file, read directly from
  disk - not derived from version *records*. A corrupt or missing record therefore cannot make a
  still-referenced row look prunable; a manifest whose record is missing (a publication interrupted
  before its commit point) stays live.
* If any manifest file can't be read, :func:`collect_referenced_row_ids` lets the ``OSError``
  propagate so the caller **aborts** rather than prune on an incomplete picture. A manifest that
  reads but is not trustworthy (not UTF-8, a line that is not a sha256 row id, or an id count that
  disagrees with its record) raises :class:`IncompleteReferenceScanError` for the same reason.
* A row-store line that can't be parsed into a ``row_id`` is **kept**, never pruned - GC only removes
  lines it can positively identify as unreferenced. That includes a line that is not valid UTF-8 (a
  capture killed mid-append can cut the store inside a multi-byte character): the store is decoded
  with ``surrogateescape``, so such a line is never refused or classified - the store's readers skip
  it too - and is written back byte for byte (like every kept line, only edge JSON whitespace such
  as a CRLF terminator is normalized).
* The store and the manifests are split on ``\\n`` only, never with ``str.splitlines()``, which also
  breaks on U+2028, U+2029 and U+0085. ``json.dumps(ensure_ascii=False)`` writes those raw inside row
  strings and the store's readers (``load_row_id_set`` / ``load_rows_by_id``) keep them inside the
  row; splitting on them would cut a referenced row into unparseable fragments that the rewritten
  store no longer holds as that row.
* Concurrency (#859): the whole GC (manifest scan, store read, replace) runs under the
  version-store lock (``versions/store_lock.py``), which every store append and manifest publication
  also holds. A capture can therefore never append and publish between GC's scan and its replace,
  so rows of a version published while GC runs are never pruned or dropped by the replace. The
  replace is a unique, fsynced temp file plus ``os.replace``: a GC that dies leaves the original
  store intact.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel

from corpus_studio.storage.examples_writer import atomic_write_lines
from corpus_studio.versions.row_store import row_store_path
from corpus_studio.versions.store_lock import version_store_lock
from corpus_studio.versions.version_registry import (
    ROW_MANIFEST_SUFFIX,
    load_version_record,
    registry_dir,
)

# sha256-exact-v1 row ids: 64 lowercase hex characters (hashlib hexdigest). A future identity
# algorithm must extend this check, or GC (correctly) refuses its manifests.
_ROW_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
# JSON's own insignificant whitespace (RFC 8259): stripping only these (never Unicode separators)
# cannot change what a line parses to, and leaves a kept unclassifiable line's content intact.
_JSON_WHITESPACE = " \t\r\n"


class RowStoreGcRefusedError(RuntimeError):
    """GC refused to run or to replace the store; nothing was pruned."""


class IncompleteReferenceScanError(RowStoreGcRefusedError):
    """A manifest could not be trusted as a complete list of referenced rows; nothing was pruned."""


class RowStoreGcResult(BaseModel):
    referenced_row_ids: int = 0  # unique row-ids referenced by all manifests (the live set)
    scanned_rows: int = 0  # store lines with a valid row_id
    kept_rows: int = 0  # referenced rows kept (unclassifiable lines are always preserved, not counted)
    pruned_rows: int = 0  # unreferenced rows removed
    dry_run: bool = False


def _read_lines(path: Path, *, errors: str = "strict") -> list[str]:
    """``path`` decoded as UTF-8 (BOM-tolerant) and split on ``\\n`` ONLY.

    Decoded from bytes, so no universal-newline translation happens either: a stray ``\\r`` inside
    a line stays in that line (a CRLF terminator is dropped by the caller's whitespace strip).
    ``\\n`` is a single byte that never occurs inside a multi-byte sequence, so splitting after a
    ``surrogateescape`` decode cuts the same lines as splitting the raw bytes. Raises ``OSError`` if
    unreadable and, with the default strict ``errors``, ``UnicodeDecodeError`` if not UTF-8."""

    return path.read_bytes().decode("utf-8-sig", errors).split("\n")


def _row_id_of(line: str) -> str | None:
    """The ``row_id`` a stripped store line declares, or ``None`` when GC cannot classify it.

    A line holding lone surrogates came from bytes that are not valid UTF-8 (the store is decoded
    with ``surrogateescape``). The store's readers skip such a line, so GC never classifies it
    either: it stays unclassifiable, and therefore kept."""

    try:
        line.encode("utf-8")
    except UnicodeEncodeError:
        return None
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        return None
    if isinstance(entry, dict) and isinstance(entry.get("row_id"), str):
        return str(entry["row_id"])
    return None


def _check_manifest_against_record(manifest_file: Path, id_count: int) -> None:
    """Refuse a manifest shorter or longer than its committed record declares.

    The record is the publication's commit point and carries ``stored_row_count`` (one manifest line
    per captured row, duplicates included), so a mismatch means the manifest is torn or was edited.
    A missing or unreadable record gives nothing to check against; the manifest then stays live as
    read, which can only keep rows, never prune one it names."""

    try:
        record = load_version_record(manifest_file.with_suffix(".json"))
    except Exception:  # noqa: BLE001 - no trustworthy record: keep the manifest live as read.
        return
    if record.rows_stored and record.stored_row_count != id_count:
        raise IncompleteReferenceScanError(
            f"manifest '{manifest_file.name}' lists {id_count} row id(s) but its record declares "
            f"{record.stored_row_count} (torn or edited manifest); nothing was pruned"
        )


def collect_referenced_row_ids(project_dir: Path | str) -> set[str]:
    """Union of row-ids across every version manifest - the set GC must keep.

    Reads the manifest files directly. If a manifest can't be read the ``OSError`` propagates, and
    a manifest that is not valid UTF-8, holds a line that is not a sha256 row id, or disagrees with
    its record's row count raises :class:`IncompleteReferenceScanError`: the caller must abort
    rather than prune on incomplete information.
    """
    live: set[str] = set()
    directory = registry_dir(project_dir)
    if not directory.exists():
        return live
    for manifest_file in sorted(directory.glob(f"*{ROW_MANIFEST_SUFFIX}")):
        try:
            lines = _read_lines(manifest_file)  # OSError propagates -> abort
        except UnicodeDecodeError as exc:
            raise IncompleteReferenceScanError(
                f"manifest '{manifest_file.name}' is not valid UTF-8; nothing was pruned"
            ) from exc
        ids: list[str] = []
        for number, line in enumerate(lines, start=1):
            row_id = line.strip(_JSON_WHITESPACE)
            if not row_id:
                continue
            if not _ROW_ID_PATTERN.fullmatch(row_id):
                raise IncompleteReferenceScanError(
                    f"manifest '{manifest_file.name}' line {number} is not a sha256 row id "
                    "(torn or corrupt manifest); nothing was pruned"
                )
            ids.append(row_id)
        _check_manifest_against_record(manifest_file, len(ids))
        live.update(ids)
    return live


def gc_row_store(project_dir: Path | str, dry_run: bool = False) -> RowStoreGcResult:
    """Prune row-store rows not referenced by any version manifest. Atomic rewrite; ``dry_run`` reports
    what would change without writing.

    Runs entirely under the version-store lock, so it never races a capture or publication; raises
    :class:`~corpus_studio.versions.store_lock.VersionStoreBusyError` (or its base
    ``VersionStoreLockError``) when the lock cannot be acquired. Raises ``OSError`` if a manifest is
    unreadable, :class:`IncompleteReferenceScanError` if a manifest cannot be trusted, and
    :class:`RowStoreGcRefusedError` if the store cannot be replaced. In every one of those cases
    nothing is pruned. Store lines that are not valid UTF-8 are not a refusal: they are kept byte for
    byte like any other unclassifiable line."""
    if not registry_dir(project_dir).is_dir():
        # No version store at all: nothing to prune, and no lock file is created as a side effect.
        return RowStoreGcResult(dry_run=dry_run)
    with version_store_lock(project_dir, operation="row-store GC"):
        return _gc_row_store_locked(project_dir, dry_run)


def _gc_row_store_locked(project_dir: Path | str, dry_run: bool) -> RowStoreGcResult:
    referenced = collect_referenced_row_ids(project_dir)  # abort-on-unreadable is intentional

    path = row_store_path(project_dir)
    if not path.exists():
        return RowStoreGcResult(referenced_row_ids=len(referenced), dry_run=dry_run)

    # BOM-tolerant, so a BOM-prefixed store isn't misread; split on "\n" only, like the store's
    # readers. surrogateescape: an undecodable line survives the decode and the rewrite unchanged.
    lines = _read_lines(path, errors="surrogateescape")

    kept: list[str] = []
    scanned = 0
    pruned = 0
    for line in lines:
        stripped = line.strip(_JSON_WHITESPACE)
        if not stripped:
            continue  # blank line: carries no row, safe to drop

        row_id = _row_id_of(stripped)
        if row_id is None:
            # Can't identify this line - KEEP it. GC never prunes what it can't classify.
            kept.append(stripped)
            continue

        scanned += 1
        if row_id in referenced:
            kept.append(stripped)
        else:
            pruned += 1

    if pruned and not dry_run:
        try:
            atomic_write_lines(path, kept, errors="surrogateescape")
        except OSError as exc:
            raise RowStoreGcRefusedError(
                f"the row store could not be replaced ({exc}); nothing was pruned"
            ) from exc

    return RowStoreGcResult(
        referenced_row_ids=len(referenced),
        scanned_rows=scanned,
        kept_rows=scanned - pruned,
        pruned_rows=pruned,
        dry_run=dry_run,
    )
