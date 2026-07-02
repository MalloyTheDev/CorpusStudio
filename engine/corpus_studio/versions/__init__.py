"""Dataset Version History & Lineage (v1.0).

A dataset version is a lightweight lineage anchor: it records the *identity* of
the project's dataset at a moment in time (row count + a streaming SHA-256
fingerprint over the ordered per-row exact signatures) plus pinned links to the
artifacts that co-existed with it (training runs, model artifacts, an eval
report, a gate report). As of v1.0.2, capture also writes each row to a
content-addressed, deduped row store plus a per-version ordered manifest, which
powers a read-only ``dataset-version-diff``; only restore-to-version is deferred.
"""

from corpus_studio.versions.version_registry import (
    DATASET_VERSION_REGISTRY_DIRNAME,
    DRIFTED,
    FINGERPRINT_ALGO,
    MATCHES,
    ROW_MANIFEST_SUFFIX,
    ROW_SIGNATURE_EXACT,
    UNREADABLE,
    DatasetCapture,
    DatasetVersionRecord,
    capture_dataset,
    compute_content_fingerprint,
    current_integrity,
    fingerprint_dataset,
    integrity_from_fingerprints,
    list_version_records,
    load_row_manifest,
    load_version_record,
    manifest_path,
    mint_version_id,
    record_path,
    registry_dir,
    save_row_manifest,
    save_version_record,
)
from corpus_studio.versions.row_store import (
    ROW_MANIFEST_ALGO,
    ROW_STORE_FILENAME,
    append_rows,
    load_row_id_set,
    load_rows_by_id,
    row_id,
    row_store_path,
    store_line,
)
from corpus_studio.versions.version_card import (
    DatasetVersionCard,
    build_version_card,
    render_version_card_markdown,
)
from corpus_studio.versions.version_diff import (
    DatasetVersionDiff,
    diff_manifests,
    render_dataset_version_diff_markdown,
)
from corpus_studio.versions.version_restore import (
    RestoreResult,
    reconstruct_and_verify,
)

__all__ = [
    "DATASET_VERSION_REGISTRY_DIRNAME",
    "DRIFTED",
    "FINGERPRINT_ALGO",
    "MATCHES",
    "ROW_MANIFEST_ALGO",
    "ROW_MANIFEST_SUFFIX",
    "ROW_SIGNATURE_EXACT",
    "ROW_STORE_FILENAME",
    "UNREADABLE",
    "DatasetCapture",
    "DatasetVersionCard",
    "DatasetVersionDiff",
    "DatasetVersionRecord",
    "RestoreResult",
    "append_rows",
    "build_version_card",
    "capture_dataset",
    "compute_content_fingerprint",
    "current_integrity",
    "diff_manifests",
    "reconstruct_and_verify",
    "fingerprint_dataset",
    "integrity_from_fingerprints",
    "list_version_records",
    "load_row_id_set",
    "load_row_manifest",
    "load_rows_by_id",
    "load_version_record",
    "manifest_path",
    "mint_version_id",
    "record_path",
    "registry_dir",
    "render_dataset_version_diff_markdown",
    "render_version_card_markdown",
    "row_id",
    "row_store_path",
    "save_row_manifest",
    "save_version_record",
    "store_line",
]
