"""Durable, project-local dataset version records (v1.0 Dataset Version History).

A dataset version is a lightweight *lineage anchor*: it records the identity of
the project's dataset at a moment in time — ``row_count`` plus a streaming
SHA-256 ``content_fingerprint`` over the ordered per-row exact signatures — and
pins the artifacts that co-existed with it (training runs, model artifacts, an
eval report, a gate report). The record JSON itself stores no row bodies (eval
scores, base model, and integrity are all resolved live in the version card).
As of v1.0.2, :func:`capture_dataset` (with ``store_rows``) also writes each row
to a content-addressed store plus an ordered per-version manifest, which powers
``dataset-version-diff`` and restore (see ``row_store`` / ``version_diff`` /
``version_restore``).

Concurrency and crash safety (#859): the row store and the manifests are one
shared structure that row-store GC also rewrites, so every store append and every
manifest publication runs under the version-store lock (``store_lock``), and
:func:`publish_dataset_version` holds it across the whole capture -> manifest ->
record sequence. The interrupted states that sequence can leave behind are all
recoverable without manual repair:

* died mid-append: complete orphan rows plus possibly a torn last line, which can
  end inside a multi-byte UTF-8 character. The next capture newline-terminates the
  torn line first, at the byte level (so it cannot swallow the next row), and every
  store reader skips an undecodable or torn line, so capture, diff and restore keep
  working; GC prunes the orphans and keeps the unclassifiable fragment.
* died after the append, before the manifest rename: orphan rows (and possibly a
  uniquely named temp file that nothing reads). GC prunes the orphans; it may do
  so only under the lock, which proves no publication is in flight.
* died after the manifest, before the record: a record-less manifest. GC keeps its
  rows (never prunes on a guess); the version is invisible to list/restore.
* a failed capture rolls the store back to its pre-capture size, never beyond the
  current end (it never extends the file).

Records are per-version inspectable JSON under ``dataset_versions/`` (mutable
metadata like label/links => a per-record file, never a JSONL append log).
``version_id`` is timestamp-prefixed so listing is chronological without an
index file.

Hard constraint: this module only READS ``examples.jsonl`` and writes only under
``dataset_versions/`` (records, manifests, the row store, and the lock file). It
never moves, copies, or deletes the dataset or any weight file.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, Field

from corpus_studio.storage.examples_writer import atomic_write_lines

# Single source of the per-row exact signature (json.dumps sort_keys, compact),
# reused verbatim so version identity matches cleaning/quality/leakage exactly.
from corpus_studio.exporters.cleaning import exact_row_signature
from corpus_studio.importers.jsonl_importer import read_jsonl

DATASET_VERSION_REGISTRY_DIRNAME = "dataset_versions"

# Version tag for the fingerprint algorithm so a future order-insensitive or
# normalized variant is additive (new tag) and never silently reinterprets an
# already-stored fingerprint.
FINGERPRINT_ALGO = "sha256-ordered-exact-v1"
ROW_SIGNATURE_EXACT = "exact"

# Per-version ordered row-id manifest sidecar: dataset_versions/<version_id>.rows
ROW_MANIFEST_SUFFIX = ".rows"

# current_integrity values — computed live (record vs disk), never stored.
MATCHES = "matches"
DRIFTED = "drifted"
UNREADABLE = "unreadable"

_VALID_VERSION_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class DatasetVersionRecord(BaseModel):
    """A durable lineage anchor for the dataset at a point in time.

    Everything derivable (eval scores, base model, artifact integrity) is
    resolved live in the version card — never stored here — so a record can
    never drift from the state it points at.
    """

    version_id: str
    created_at: str
    updated_at: str
    label: str = ""
    # manual_add | import_commit | pre_training | manual (free text; not validated)
    trigger: str = ""
    row_count: int = 0
    # 64-char sha256 hex, or None when examples.jsonl was absent/unreadable at
    # capture time (so a fingerprint is only ever an affirmative claim of state).
    content_fingerprint: str | None = None
    fingerprint_algo: str = FINGERPRINT_ALGO
    row_signature_kind: str = ROW_SIGNATURE_EXACT
    source_run_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    # Absolute path — eval reports live outside the project dir; resolved live.
    eval_report_path: str | None = None
    # Dataset-scope gate report inside the project's gate_reports/.
    gate_report_path: str | None = None
    notes: str = ""
    # v1.0.2 row store: whether this version stored its row bodies (=> diffable).
    # Tolerant defaults so pre-v1.0.2 records load as "no rows stored".
    rows_stored: bool = False
    stored_row_count: int = 0
    row_manifest_algo: str | None = None


def _slug(version_id: str) -> str:
    # No strip(): every char in the validated alphabet (including a leading or
    # trailing '_') is preserved, so distinct valid version_ids never collapse to
    # the same filename. For a validated id this is the identity; the ``or`` only
    # guards an empty string (which validation already rejects on the save path).
    return re.sub(r"[^A-Za-z0-9._-]+", "_", version_id) or "version"


def mint_version_id(timestamp_compact: str, suffix: str) -> str:
    """Chronologically-sortable id, e.g. '20260702T183000-004217-9af3c1'.

    ``list_version_records`` sorts on the id string, so chronological ordering
    holds only when ids are **fixed width** (a fixed-width timestamp prefix and a
    fixed-width suffix). The CLI guarantees this (zero-padded microseconds + a
    fixed-length random token); a caller minting variable-width suffixes must not
    rely on list ordering being chronological within the same second.
    """

    return f"{timestamp_compact}-{suffix}"


def registry_dir(project_dir: Path | str) -> Path:
    return Path(project_dir) / DATASET_VERSION_REGISTRY_DIRNAME


def record_path(project_dir: Path | str, version_id: str) -> Path:
    return registry_dir(project_dir) / f"{_slug(version_id)}.json"


def fingerprint_dataset(examples_path: Path | str) -> tuple[str | None, int]:
    """One pass over ``examples.jsonl`` → ``(content_fingerprint, row_count)``.

    The fingerprint is an **order-sensitive** SHA-256 fed line-by-line with the
    canonical per-row ``exact_row_signature`` joined by newlines (streams in
    O(1) memory). Order-sensitivity is deliberate: rows have no stable id today,
    so identity is "these exact rows in this exact order"; a set would silently
    make it order-insensitive and defeat drift detection.

    Returns ``(None, 0)`` — never raises — when the dataset is missing or
    unreadable (including a malformed JSON line), so integrity never cries wolf.
    An existing but empty dataset returns the sha256 of empty input with count 0.
    """

    path = Path(examples_path)
    if not path.exists():
        return None, 0
    digest = hashlib.sha256()
    count = 0
    try:
        for row in read_jsonl(path):
            if count:
                digest.update(b"\n")
            digest.update(exact_row_signature(row).encode("utf-8"))
            count += 1
    except (OSError, ValueError, RecursionError):
        # ValueError covers a malformed line (json.JSONDecodeError) and bad bytes
        # (UnicodeDecodeError); RecursionError covers pathologically nested JSON.
        # An unreadable dataset yields no fingerprint, never a partial/wrong one.
        return None, 0
    return digest.hexdigest(), count


def compute_content_fingerprint(examples_path: Path | str) -> str | None:
    """Order-sensitive SHA-256 of the dataset, or None if missing/unreadable."""

    return fingerprint_dataset(examples_path)[0]


def integrity_from_fingerprints(stored: str | None, live: str | None) -> str:
    """Compare a record's stored fingerprint to a freshly computed live one.

    ``unreadable`` when either side is absent (nothing to compare), else
    ``matches`` / ``drifted``.
    """

    if stored is None or live is None:
        return UNREADABLE
    return MATCHES if stored == live else DRIFTED


def current_integrity(record: DatasetVersionRecord, examples_path: Path | str) -> str:
    """Live integrity of a version vs the current dataset (never persisted)."""

    return integrity_from_fingerprints(
        record.content_fingerprint, compute_content_fingerprint(examples_path)
    )


def save_version_record(project_dir: Path | str, record: DatasetVersionRecord) -> Path:
    """Atomically write a version record (unique fsynced temp + os.replace).

    ``version_id`` must match ``[A-Za-z0-9._-]+`` so the slugged filename is
    injective (distinct ids can never collapse to the same file and silently
    overwrite one another).
    """

    if not _VALID_VERSION_ID.match(record.version_id):
        raise ValueError(
            f"Invalid version_id '{record.version_id}': must match [A-Za-z0-9._-]+."
        )
    path = registry_dir(project_dir) / f"{_slug(record.version_id)}.json"
    atomic_write_lines(path, [record.model_dump_json(indent=2)])
    return path


def load_version_record(path: Path | str) -> DatasetVersionRecord:
    return DatasetVersionRecord.model_validate_json(Path(path).read_text(encoding="utf-8"))


def list_version_records(project_dir: Path | str) -> list[DatasetVersionRecord]:
    """All records, newest first (version_id is chronological). Corrupt files skipped."""

    directory = registry_dir(project_dir)
    if not directory.exists():
        return []
    records: list[DatasetVersionRecord] = []
    seen: set[str] = set()
    for path in directory.glob("*.json"):
        try:
            record = load_version_record(path)
        except Exception:  # noqa: BLE001 - a corrupt record must not break listing.
            continue
        if record.version_id in seen:
            continue  # tolerate a duplicate file (first wins)
        seen.add(record.version_id)
        records.append(record)
    records.sort(key=lambda record: record.version_id, reverse=True)
    return records


# --- v1.0.2: row-id manifest sidecar + single-pass capture -------------------


def manifest_path(project_dir: Path | str, version_id: str) -> Path:
    return registry_dir(project_dir) / f"{_slug(version_id)}{ROW_MANIFEST_SUFFIX}"


def save_row_manifest(project_dir: Path | str, version_id: str, row_ids: list[str]) -> Path:
    """Atomically write the ordered row-id manifest (one id per line; unique fsynced
    temp + os.replace, so a crash leaves either no manifest or a complete one).

    Publishing a manifest pins rows against GC, so it runs under the version-store
    lock (reentrant: :func:`publish_dataset_version` already holds it across the
    capture that stored those rows)."""

    from corpus_studio.versions.store_lock import version_store_lock

    path = registry_dir(project_dir) / f"{_slug(version_id)}{ROW_MANIFEST_SUFFIX}"
    with version_store_lock(project_dir, operation="manifest publication"):
        atomic_write_lines(path, list(row_ids))
    return path


def load_row_manifest(project_dir: Path | str, version_id: str) -> list[str] | None:
    """The ordered row-ids for a version, or ``None`` if no manifest exists (a
    pre-v1.0.2 record, or one captured with ``--no-store-rows``). An existing but
    empty manifest returns ``[]`` (captured with storage, 0 rows)."""

    path = manifest_path(project_dir, version_id)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return [line.strip() for line in text.splitlines() if line.strip()]


class DatasetCapture(BaseModel):
    """Result of one streaming capture pass.

    ``content_fingerprint`` is ``None`` (lists empty) when the dataset was
    missing/unreadable. ``rows_stored`` is True only when row storage was
    requested AND fully succeeded; if the store could not be written, the
    fingerprint is still returned but ``rows_stored`` is False (a fingerprint-only
    version), so a store I/O error never masquerades as an unreadable dataset.
    """

    content_fingerprint: str | None = None
    row_count: int = 0
    row_ids: list[str] = Field(default_factory=list)
    new_rows_stored: int = 0
    rows_stored: bool = False


def _truncate_row_store(store_target: Path | None, size: int | None) -> None:
    """Best-effort rollback: shrink the row store back to its pre-capture size so a
    failed/partial capture leaves nothing new on disk.

    A ``size`` of ``None`` means the pre-capture size could not be determined, so the
    rollback is SKIPPED: truncating to a guessed 0 would destroy every previously-stored
    version's rows. Leaving the newly-appended (content-addressed) rows in place is safe:
    they are unreferenced orphans that row-store GC prunes later.

    The store is only ever SHRUNK. ``os.truncate`` to a size past the current end would
    EXTEND the file with NUL bytes, and the next append would be glued onto that NUL run
    and become unreadable. The version-store lock keeps GC from shrinking the store
    during a capture; this guard keeps the rollback safe even if something outside the
    protocol did."""

    if store_target is None or size is None:
        return
    try:
        if store_target.stat().st_size > size:
            os.truncate(store_target, size)
    except OSError:
        pass


def capture_dataset(
    examples_path: Path | str, project_dir: Path | str, *, store_rows: bool
) -> DatasetCapture:
    """Single streaming pass over ``examples.jsonl`` producing the identity + rows.

    In ONE read it (1) feeds the content fingerprint digest with the exact,
    ordered per-row signatures - byte-for-byte identical to
    :func:`fingerprint_dataset` - (2) computes each row_id, and (3) when
    ``store_rows`` appends any not-yet-stored row to the shared content-addressed
    store. Because everything derives from the same iteration, the returned
    fingerprint and ordered ``row_ids`` can never desync.

    Failure handling keeps two domains separate so the report is always honest:
    an **unreadable dataset** (missing/malformed/bad bytes) returns an empty
    capture; a **row-store I/O failure** on an otherwise-readable dataset still
    returns the real fingerprint with ``rows_stored=False``. On either failure the
    store is rolled back to its pre-capture size, so a failed capture leaves
    nothing new on disk. Dataset and store I/O problems never raise.

    With ``store_rows`` the pass runs under the version-store lock (reentrant) and
    raises :class:`~corpus_studio.versions.store_lock.VersionStoreLockError` (nothing
    written) when that lock cannot be acquired. Rows appended here are unreferenced,
    and therefore prunable by GC, until a manifest names them: a caller that
    publishes a manifest for this capture MUST hold the lock across the capture AND
    the publication. :func:`publish_dataset_version` does exactly that.
    """

    path = Path(examples_path)
    if not path.exists():
        return DatasetCapture()
    if not store_rows:
        return _capture_pass(path, project_dir, store_rows=False)

    from corpus_studio.versions.store_lock import version_store_lock

    with version_store_lock(project_dir, operation="dataset capture"):
        return _capture_pass(path, project_dir, store_rows=True)


def _capture_pass(path: Path, project_dir: Path | str, *, store_rows: bool) -> DatasetCapture:
    """The body of :func:`capture_dataset`; with ``store_rows`` the caller holds the
    version-store lock, so no GC or other append can move the store under it."""

    from corpus_studio.versions.row_store import (
        load_row_id_set,
        row_store_path,
        store_line,
        terminate_torn_tail,
    )

    store_target: Path | None = None
    # 0 => no pre-existing store, so a rollback truncates the file we create back to
    # empty. None => the store exists but its size is UNKNOWN (stat failed); rollback
    # must then be skipped rather than guess 0 and wipe every prior version's rows.
    store_start_size: int | None = 0
    existing: set[str] = set()
    store_failed = False
    if store_rows:
        store_target = row_store_path(project_dir)
        try:
            # A writer that died mid-append can leave a partial last line; appending
            # straight after it would glue our first row onto the fragment.
            terminate_torn_tail(store_target)
        except OSError:
            # The store cannot be made safe to append to: store nothing and record
            # a fingerprint-only version rather than risk an unreadable first row.
            store_failed = True
        existing = load_row_id_set(project_dir)
        if store_target.exists():
            try:
                store_start_size = store_target.stat().st_size
            except OSError:
                store_start_size = None

    digest = hashlib.sha256()
    row_ids: list[str] = []
    new_stored = 0
    store_handle = None
    dataset_unreadable = False
    count = 0
    try:
        for row in read_jsonl(path):
            signature = exact_row_signature(row)
            if count:
                digest.update(b"\n")
            digest.update(signature.encode("utf-8"))
            rid = hashlib.sha256(signature.encode("utf-8")).hexdigest()
            row_ids.append(rid)
            count += 1
            if store_rows and not store_failed and rid not in existing:
                existing.add(rid)
                try:
                    if store_handle is None:
                        assert store_target is not None  # store_rows guarantees a target
                        store_target.parent.mkdir(parents=True, exist_ok=True)
                        store_handle = store_target.open("a", encoding="utf-8")
                    store_handle.write(store_line(rid, row))
                    new_stored += 1
                except OSError:
                    # A store I/O failure must NOT null the fingerprint of a
                    # readable dataset: stop storing, keep computing identity.
                    store_failed = True
    except (OSError, ValueError, RecursionError):
        dataset_unreadable = True
    finally:
        if store_handle is not None:
            try:
                store_handle.flush()
                os.fsync(store_handle.fileno())
                store_handle.close()
            except OSError:
                # A flush/fsync/close failure means buffered rows may NOT have reached
                # disk - treat the store write as failed so rows_stored is not a false
                # promise and the store is rolled back below.
                store_failed = True

    # Roll the store back on any failure so a failed/partial capture stores
    # nothing on disk (not just in the return value).
    if store_rows and (dataset_unreadable or store_failed):
        _truncate_row_store(store_target, store_start_size)

    if dataset_unreadable:
        return DatasetCapture()

    return DatasetCapture(
        content_fingerprint=digest.hexdigest(),
        row_count=count,
        row_ids=row_ids,
        new_rows_stored=0 if store_failed else new_stored,
        rows_stored=store_rows and not store_failed,
    )


def publish_dataset_version(
    project_dir: Path | str,
    *,
    label: str = "",
    trigger: str = "manual",
    store_rows: bool = True,
    source_run_ids: Sequence[str] = (),
    artifact_ids: Sequence[str] = (),
    eval_report_path: str | None = None,
    gate_report_path: str | None = None,
) -> tuple[DatasetVersionRecord, DatasetCapture]:
    """Capture examples.jsonl and publish it as a new dataset version.

    The single publication path (the CLI ``dataset-version-create``, import-commit,
    the examples-mutation undo, and in-place restore's undo all come through here):
    capture -> mint id -> save manifest -> save record, as ONE critical section under
    the version-store lock when rows are stored, so row-store GC can never prune the
    appended rows before the manifest that pins them is published. The record save is
    the commit point. Returns ``(record, capture)``.

    Raises :class:`~corpus_studio.versions.store_lock.VersionStoreBusyError` (or its
    base :class:`~corpus_studio.versions.store_lock.VersionStoreLockError`) before
    anything is written when the lock cannot be acquired. Reads examples.jsonl;
    writes only under dataset_versions/.
    """

    import secrets
    from contextlib import AbstractContextManager, nullcontext
    from datetime import datetime, timezone

    from corpus_studio.versions.row_store import ROW_MANIFEST_ALGO
    from corpus_studio.versions.store_lock import version_store_lock

    project = Path(project_dir)
    # A fingerprint-only version appends no rows and publishes no manifest, so it has
    # nothing to serialize against GC.
    guard: AbstractContextManager[None] = (
        version_store_lock(project, operation="dataset version capture")
        if store_rows
        else nullcontext()
    )
    with guard:
        capture = capture_dataset(project / "examples.jsonl", project, store_rows=store_rows)
        rows_stored = capture.rows_stored
        now_dt = datetime.now(timezone.utc)
        # A random token breaks ties: the wall clock can be too coarse to advance
        # between two in-process creates (esp. on Windows), and a pure-timestamp id
        # would collide and silently overwrite the earlier version's file.
        version_id = mint_version_id(
            now_dt.strftime("%Y%m%dT%H%M%S"), f"{now_dt.microsecond:06d}-{secrets.token_hex(3)}"
        )
        now_iso = now_dt.isoformat()
        record = DatasetVersionRecord(
            version_id=version_id,
            created_at=now_iso,
            updated_at=now_iso,
            label=label,
            trigger=trigger,
            row_count=capture.row_count,
            content_fingerprint=capture.content_fingerprint,
            source_run_ids=list(source_run_ids),
            artifact_ids=list(artifact_ids),
            eval_report_path=eval_report_path,
            gate_report_path=gate_report_path,
            rows_stored=rows_stored,
            stored_row_count=capture.row_count if rows_stored else 0,
            row_manifest_algo=ROW_MANIFEST_ALGO if rows_stored else None,
        )
        # The ordered manifest (it references the store) goes before the record; the
        # record save is the commit point.
        if rows_stored:
            save_row_manifest(project, version_id, capture.row_ids)
        save_version_record(project, record)
    return record, capture


def create_dataset_version(
    project_dir: Path | str,
    *,
    label: str = "",
    trigger: str = "manual",
    store_rows: bool = True,
) -> DatasetVersionRecord:
    """Capture examples.jsonl as a new dataset version and persist it.

    A plain version (no run/artifact/gate linkage) through
    :func:`publish_dataset_version` - notably in-place restore's undo capture. Raises
    :class:`~corpus_studio.versions.store_lock.VersionStoreLockError` (nothing written)
    when the version-store lock cannot be acquired. Returns the saved record.
    """

    record, _capture = publish_dataset_version(
        project_dir, label=label, trigger=trigger, store_rows=store_rows
    )
    return record
