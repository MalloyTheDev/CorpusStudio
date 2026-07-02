"""Dataset Version History & Lineage (v1.0).

A dataset version is a lightweight lineage anchor: it records the *identity* of
the project's dataset at a moment in time (row count + a streaming SHA-256
fingerprint over the ordered per-row exact signatures) plus pinned links to the
artifacts that co-existed with it (training runs, model artifacts, an eval
report, a gate report). It stores no row bodies — diff and restore need stable
row identity the current storage cannot cheaply provide and are deferred.
"""

from corpus_studio.versions.version_registry import (
    DATASET_VERSION_REGISTRY_DIRNAME,
    DRIFTED,
    FINGERPRINT_ALGO,
    MATCHES,
    ROW_SIGNATURE_EXACT,
    UNREADABLE,
    DatasetVersionRecord,
    compute_content_fingerprint,
    current_integrity,
    fingerprint_dataset,
    integrity_from_fingerprints,
    list_version_records,
    load_version_record,
    mint_version_id,
    record_path,
    registry_dir,
    save_version_record,
)
from corpus_studio.versions.version_card import (
    DatasetVersionCard,
    build_version_card,
    render_version_card_markdown,
)

__all__ = [
    "DATASET_VERSION_REGISTRY_DIRNAME",
    "DRIFTED",
    "FINGERPRINT_ALGO",
    "MATCHES",
    "ROW_SIGNATURE_EXACT",
    "UNREADABLE",
    "DatasetVersionRecord",
    "DatasetVersionCard",
    "build_version_card",
    "compute_content_fingerprint",
    "current_integrity",
    "fingerprint_dataset",
    "integrity_from_fingerprints",
    "list_version_records",
    "load_version_record",
    "mint_version_id",
    "record_path",
    "registry_dir",
    "render_version_card_markdown",
    "save_version_record",
]
