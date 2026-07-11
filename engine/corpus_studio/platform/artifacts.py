"""ArtifactManifest persistence + two-tier integrity — platform slice 5.

Slice 4's :class:`~corpus_studio.platform.runners.TrainingRunner` returns a lightweight
``ProducedArtifact`` (id / kind / path); this turns that into a durable, integrity-checked
:class:`ArtifactManifest`. The two-tier model (grounded in ``training.artifact_registry``): a cheap
size+mtime **fingerprint** powers the fast liveness check, and a byte-exact sha256 **content_hash**
powers the promote GATE. The platform NEVER moves / copies / deletes the underlying weights — it only
references and re-checks them, so a deleted (``missing``) or overwritten (``modified``) artifact is
caught rather than silently trusted.

Dependency-light: the file-integrity helpers are reused from ``training.artifact_registry`` (pure
``hashlib`` / ``os``, no torch) via a lazy import, so importing this module stays cheap.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from corpus_studio.platform.contracts import ArtifactManifest

_ARTIFACT_KINDS = frozenset(
    {"adapter", "checkpoint", "merged_model", "gguf", "onnx", "quantized", "other"}
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_artifact_manifest(
    *,
    artifact_id: str,
    path: str,
    run_id: str,
    kind: str = "adapter",
    base_model: str | None = None,
    reload_verified: bool = False,
    notes: str = "",
    now: str | None = None,
) -> ArtifactManifest:
    """Fingerprint + content-hash the weights at ``path`` and wrap them in a durable
    :class:`ArtifactManifest` (status ``candidate``). ``current_integrity`` is ``ok`` when the weights
    were readable at build time and ``missing`` when nothing hashable was found; the live re-check
    (``ok`` / ``modified`` / ``missing``) is :func:`recheck_artifact_integrity`. An unknown ``kind``
    is recorded as ``other`` rather than rejected."""
    from corpus_studio.training.artifact_registry import (  # noqa: PLC0415 - pure, torch-free
        compute_content_hash,
        compute_fingerprint,
    )

    stamp = now or _now_iso()
    fingerprint = compute_fingerprint(path)
    content_hash = compute_content_hash(path)
    return ArtifactManifest.model_validate(
        {
            "artifact_id": artifact_id,
            "producer_run_ref": {"id": run_id},
            "created_at": stamp,
            "updated_at": stamp,
            "kind": kind if kind in _ARTIFACT_KINDS else "other",
            "path": path,
            "status": "candidate",
            "integrity": {
                "cheap_fingerprint": fingerprint,
                "content_hash": content_hash,
                "current_integrity": "ok" if fingerprint is not None else "missing",
            },
            "reload_verified": reload_verified,
            "base_model": base_model,
            "notes": notes,
        }
    )


def recheck_artifact_integrity(
    manifest: ArtifactManifest, *, now: str | None = None
) -> ArtifactManifest:
    """Re-verify an artifact's weights against its recorded integrity and return a manifest with a
    live ``current_integrity``. Prefers the byte-exact ``content_hash`` (the promote gate); falls back
    to the cheap fingerprint when no content hash was stored. ``missing`` when the weights are gone,
    ``modified`` when they changed since the manifest was built, ``ok`` when they match."""
    from corpus_studio.training.artifact_registry import (  # noqa: PLC0415 - pure, torch-free
        compute_content_hash,
        compute_fingerprint,
    )

    stored = manifest.integrity
    if stored is None:
        return manifest

    current_fingerprint = compute_fingerprint(manifest.path)
    if current_fingerprint is None:
        status = "missing"
    elif stored.content_hash is not None:
        status = "ok" if compute_content_hash(manifest.path) == stored.content_hash else "modified"
    elif stored.cheap_fingerprint is not None:
        status = "ok" if current_fingerprint == stored.cheap_fingerprint else "modified"
    else:
        status = "unknown"

    rechecked = stored.model_copy(update={"current_integrity": status})
    return manifest.model_copy(update={"integrity": rechecked, "updated_at": now or _now_iso()})


def write_artifact_manifest(manifest: ArtifactManifest, out_dir: str | Path) -> Path:
    """Crash-safe write of an ArtifactManifest to ``<out_dir>/artifacts/<artifact_id>.json`` (temp file
    then ``os.replace`` — the same durability convention as the RunManifest)."""
    directory = Path(out_dir) / "artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"{manifest.artifact_id}.json"
    tmp = directory / f".{manifest.artifact_id}.{os.getpid()}.tmp"
    tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, final)
    return final
