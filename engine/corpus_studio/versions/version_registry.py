"""Durable, project-local dataset version records (v1.0 Dataset Version History).

A dataset version is a lightweight *lineage anchor*: it records the identity of
the project's dataset at a moment in time — ``row_count`` plus a streaming
SHA-256 ``content_fingerprint`` over the ordered per-row exact signatures — and
pins the artifacts that co-existed with it (training runs, model artifacts, an
eval report, a gate report). It stores **no row bodies**: diff and restore need
stable per-row identity the current line-number-only storage cannot cheaply
provide, so they are deferred (v1.0.2+). Nothing derivable is stored; eval
scores, base model, and integrity are resolved live in the version card.

Records are per-version inspectable JSON under ``dataset_versions/`` (mutable
metadata like label/links => a per-record file, never a JSONL append log).
``version_id`` is timestamp-prefixed so listing is chronological without an
index file.

Hard constraint: this module only READS ``examples.jsonl`` and writes JSON under
``dataset_versions/``. It never moves, copies, or deletes the dataset or any
weight file.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from pydantic import BaseModel, Field

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
    """Atomically write a version record (temp + os.replace).

    ``version_id`` must match ``[A-Za-z0-9._-]+`` so the slugged filename is
    injective (distinct ids can never collapse to the same file and silently
    overwrite one another).
    """

    if not _VALID_VERSION_ID.match(record.version_id):
        raise ValueError(
            f"Invalid version_id '{record.version_id}': must match [A-Za-z0-9._-]+."
        )
    directory = registry_dir(project_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_slug(record.version_id)}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
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
