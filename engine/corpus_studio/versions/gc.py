"""Row-store garbage collection (issue #197).

The content-addressed row store (``dataset_versions/row_store.jsonl``) keeps one line per unique row
ever captured. Over many captures it accumulates rows that no surviving version references. GC prunes
exactly those, and **never** a row referenced by any version manifest.

Safety is the whole point, so this is deliberately fail-closed:

* The live set is the union of row-ids across **every** ``*.rows`` manifest file, read directly from
  disk — not derived from version *records*. A corrupt or missing record therefore cannot make a
  still-referenced row look prunable.
* If any manifest file can't be read, :func:`collect_referenced_row_ids` lets the ``OSError``
  propagate so the caller **aborts** rather than prune on an incomplete picture.
* A row-store line that can't be parsed into a ``row_id`` is **kept**, never pruned — GC only removes
  lines it can positively identify as unreferenced.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel

from corpus_studio.versions.row_store import row_store_path
from corpus_studio.versions.version_registry import ROW_MANIFEST_SUFFIX, registry_dir


class RowStoreGcResult(BaseModel):
    referenced_row_ids: int = 0  # unique row-ids referenced by all manifests (the live set)
    scanned_rows: int = 0  # store lines with a valid row_id
    kept_rows: int = 0  # referenced rows kept (unclassifiable lines are always preserved, not counted)
    pruned_rows: int = 0  # unreferenced rows removed
    dry_run: bool = False


def collect_referenced_row_ids(project_dir: Path | str) -> set[str]:
    """Union of row-ids across every version manifest — the set GC must keep.

    Reads the manifest files directly. If a manifest can't be read the ``OSError`` propagates: the
    caller must abort rather than prune on incomplete information.
    """
    live: set[str] = set()
    directory = registry_dir(project_dir)
    if not directory.exists():
        return live
    for manifest_file in sorted(directory.glob(f"*{ROW_MANIFEST_SUFFIX}")):
        text = manifest_file.read_text(encoding="utf-8")  # OSError propagates → caller aborts
        for line in text.splitlines():
            row_id = line.strip()
            if row_id:
                live.add(row_id)
    return live


def gc_row_store(project_dir: Path | str, dry_run: bool = False) -> RowStoreGcResult:
    """Prune row-store rows not referenced by any version manifest. Atomic rewrite; ``dry_run`` reports
    what would change without writing. Raises ``OSError`` (does NOT prune) if a manifest is unreadable."""
    referenced = collect_referenced_row_ids(project_dir)  # abort-on-unreadable is intentional

    path = row_store_path(project_dir)
    if not path.exists():
        return RowStoreGcResult(referenced_row_ids=len(referenced), dry_run=dry_run)

    kept: list[str] = []
    scanned = 0
    pruned = 0
    # BOM-tolerant read, matching the store's other readers, so a BOM-prefixed store isn't misread.
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped:
            continue  # blank line: carries no row, safe to drop

        row_id: str | None = None
        try:
            entry = json.loads(stripped)
        except json.JSONDecodeError:
            entry = None
        if isinstance(entry, dict) and isinstance(entry.get("row_id"), str):
            row_id = entry["row_id"]

        if row_id is None:
            # Can't identify this line — KEEP it. GC never prunes what it can't classify.
            kept.append(stripped)
            continue

        scanned += 1
        if row_id in referenced:
            kept.append(stripped)
        else:
            pruned += 1

    if pruned and not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("".join(k + "\n" for k in kept), encoding="utf-8")
        os.replace(tmp, path)

    return RowStoreGcResult(
        referenced_row_ids=len(referenced),
        scanned_rows=scanned,
        kept_rows=scanned - pruned,
        pruned_rows=pruned,
        dry_run=dry_run,
    )
