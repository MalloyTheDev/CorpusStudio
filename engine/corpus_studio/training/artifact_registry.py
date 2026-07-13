"""Durable, project-local model artifact records (v0.9 Weight Registry).

Tracks the weights a training run produced (LoRA adapters, promoted checkpoints)
as first-class objects the user can keep or reject. The engine NEVER moves,
copies, or deletes the user's weight files — records only *reference* paths.

The headline feature is **path integrity**: a record stores a cheap fingerprint
(size + mtime of a key descriptor file, never a hash of multi-GB weights) at
register time, and on load re-checks the path so a record can never quietly
point at deleted (`missing`) or overwritten (`modified`) weights.

Nothing derivable from the source run is stored — base_model and eval scores are
resolved live through ``run_id`` at display time so they cannot drift.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from pydantic import BaseModel

from corpus_studio.training.run_registry import record_path as run_record_path

ARTIFACT_REGISTRY_DIRNAME = "model_artifacts"

CANDIDATE = "candidate"
KEPT = "kept"
REJECTED = "rejected"
ARTIFACT_STATUSES = frozenset({CANDIDATE, KEPT, REJECTED})

OK = "ok"
MISSING = "missing"
MODIFIED = "modified"

# Descriptor files that identify an adapter/model directory, in priority order.
_DESCRIPTOR_FILES = ("adapter_config.json", "config.json")
# Weight files whose bytes ARE the model — top-level within an artifact directory.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".pth", ".onnx", ".h5")
_VALID_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class ModelArtifactRecord(BaseModel):
    artifact_id: str
    run_id: str
    created_at: str
    updated_at: str
    path: str
    kind: str = "adapter"
    status: str = CANDIDATE
    # Cheap size+mtime fingerprint (descriptor + weight files) — the fast path used by the
    # artifact LIST. Catches overwrite / resize / add / remove without reading weight bytes.
    fingerprint: str | None = None
    # sha256 over the weight file BYTES — the byte-exact check used by the promote gate and the
    # weight card (the enforcement/decision points). None for records registered before it
    # existed (they fall back to the cheap check).
    content_hash: str | None = None
    notes: str = ""


def normalize_artifact_path(path: str) -> str:
    """Absolute, OS-canonical path used for identity and storage."""

    return os.path.normcase(os.path.abspath(str(path)))


def make_artifact_id(run_id: str, path: str) -> str:
    """Deterministic id in (run_id, normalized path) so re-register is idempotent."""

    digest = hashlib.sha1(normalize_artifact_path(path).encode("utf-8")).hexdigest()[:8]
    return f"{run_id}-{digest}"


def _descriptor_file(directory: Path) -> Path | None:
    for name in _DESCRIPTOR_FILES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    files = sorted(entry for entry in directory.iterdir() if entry.is_file())
    return files[0] if files else None


def _weight_files(directory: Path) -> list[Path]:
    """Every weight file under an artifact directory (RECURSIVE, sorted by relative path for
    determinism). Recursion + the broadened suffix set (.safetensors/.bin/.gguf/.pt/.pth/.onnx/
    .h5) mean a swapped SHARD in a subdir, or an .onnx/.h5 weight, is covered — not just a
    top-level .safetensors. Callers key each file by its path RELATIVE to ``directory``, so a
    top-level-only artifact hashes byte-identically to before (relative path == basename) and
    only previously-under-covered layouts change."""

    return sorted(
        (
            entry
            for entry in directory.rglob("*")
            if entry.is_file() and entry.suffix.lower() in _WEIGHT_SUFFIXES
        ),
        key=lambda entry: entry.relative_to(directory).as_posix(),
    )


def _rel_key(file: Path, base: Path) -> str:
    """Stable per-file key: the path relative to the artifact dir (POSIX slashes) so nested
    files with the same basename can't collide; falls back to the name for a single-file
    artifact (where ``file`` is not under ``base``)."""

    try:
        return file.relative_to(base).as_posix()
    except ValueError:
        return file.name


def compute_fingerprint(path: str) -> str | None:
    """Cheap, deterministic fingerprint: never reads weight bytes (only ``stat``).

    File -> ``"size:mtime_ns"``.
    Dir  -> the descriptor file AND every weight file found RECURSIVELY as
    ``"relpath=size:mtime_ns"`` joined by ``;`` — so an overwritten / resized / added /
    removed weight (including a nested shard or an .onnx/.h5 file) is detected (a byte-swap
    that preserves size AND mtime is not; use :func:`compute_content_hash` for that). Returns
    ``None`` when nothing is readable.
    """

    target = Path(path)
    try:
        if target.is_file():
            stat = target.stat()
            return f"{stat.st_size}:{stat.st_mtime_ns}"
        if target.is_dir():
            parts: list[str] = []
            descriptor = _descriptor_file(target)
            if descriptor is not None:
                stat = descriptor.stat()
                parts.append(f"{_rel_key(descriptor, target)}={stat.st_size}:{stat.st_mtime_ns}")
            for weight in _weight_files(target):
                stat = weight.stat()
                parts.append(f"{_rel_key(weight, target)}={stat.st_size}:{stat.st_mtime_ns}")
            return ";".join(parts) if parts else None
    except OSError:
        return None
    return None


def compute_content_hash(path: str) -> str | None:
    """Byte-exact sha256 over the weight file BYTES (their names + contents).

    Unlike :func:`compute_fingerprint` this READS the weights, so it detects a byte-swap
    that preserves size and mtime. It is therefore only used at the promote gate and the
    weight card — never on the list (which would re-read every artifact's weights on each
    refresh). Falls back to the descriptor when a directory has no weight files. Returns
    ``None`` when nothing is readable.
    """

    target = Path(path)
    try:
        if target.is_file():
            files = [target]
        elif target.is_dir():
            files = _weight_files(target)
            if not files:
                descriptor = _descriptor_file(target)
                files = [descriptor] if descriptor is not None else []
        else:
            return None
        if not files:
            return None
        return _content_hash_files(target, files)
    except OSError:
        return None


def _content_hash_files(base: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for file in files:
        digest.update(_rel_key(file, base).encode("utf-8"))
        digest.update(b"\0")
        with file.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def compute_weight_content_hash(path: str) -> str | None:
    """Hash actual recognized weight bytes only; never fall back to a descriptor.

    Training-success gates use this stricter helper. The general artifact registry keeps
    :func:`compute_content_hash`'s descriptor fallback for legacy/non-weight artifacts.
    """

    target = Path(path)
    try:
        if target.is_file():
            files = [target] if target.suffix.lower() in _WEIGHT_SUFFIXES else []
        elif target.is_dir():
            files = _weight_files(target)
        else:
            return None
        return _content_hash_files(target, files) if files else None
    except OSError:
        return None


def artifact_integrity(record: ModelArtifactRecord) -> str:
    """Cheap integrity (size+mtime) against current disk state — used by the LIST. Fast."""

    if not os.path.exists(record.path):
        return MISSING
    if record.fingerprint is None:
        return OK  # never fingerprinted at register time; do not raise a false alarm
    current = compute_fingerprint(record.path)
    if current is None:
        # We HAD a fingerprint but can't compute one now (weights/descriptor gone while the
        # path still exists) — that is a change, not "ok".
        return MODIFIED
    return OK if current == record.fingerprint else MODIFIED


def artifact_content_integrity(record: ModelArtifactRecord) -> str:
    """Byte-exact integrity (reads weight bytes) — used by the promote GATE and weight CARD.

    Detects even a change that preserves size and mtime. Legacy records without a stored
    ``content_hash`` fall back to the cheap check (so they still catch size/mtime changes and
    don't cry wolf)."""

    if not os.path.exists(record.path):
        return MISSING
    if record.content_hash is None:
        return artifact_integrity(record)
    current = compute_content_hash(record.path)
    if current is None:
        return MODIFIED
    return OK if current == record.content_hash else MODIFIED


def registry_dir(project_dir: Path | str) -> Path:
    return Path(project_dir) / ARTIFACT_REGISTRY_DIRNAME


def artifact_path(project_dir: Path | str, artifact_id: str) -> Path:
    return registry_dir(project_dir) / f"{artifact_id}.json"


def save_artifact_record(project_dir: Path | str, record: ModelArtifactRecord) -> Path:
    if not _VALID_ARTIFACT_ID.match(record.artifact_id):
        raise ValueError(f"Invalid artifact_id '{record.artifact_id}'.")
    directory = registry_dir(project_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record.artifact_id}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_artifact_record(path: Path | str) -> ModelArtifactRecord:
    return ModelArtifactRecord.model_validate_json(Path(path).read_text(encoding="utf-8"))


def register_artifact(
    project_dir: Path | str,
    run_id: str,
    path: str,
    kind: str = "adapter",
    notes: str = "",
    now: str = "",
) -> ModelArtifactRecord:
    """Register (or idempotently update) an artifact for a run.

    Requires the source run to exist. Re-registering the same run+path preserves
    ``created_at`` and ``status`` and refreshes ``updated_at`` + ``fingerprint``.
    """

    if not run_record_path(project_dir, run_id).exists():
        raise ValueError(f"No training run '{run_id}' to attach an artifact to.")

    normalized = normalize_artifact_path(path)
    artifact_id = make_artifact_id(run_id, normalized)
    existing_path = artifact_path(project_dir, artifact_id)

    created_at = now
    status = CANDIDATE
    keep_notes = notes
    if existing_path.exists():
        try:
            existing = load_artifact_record(existing_path)
            created_at = existing.created_at
            status = existing.status
            keep_notes = notes or existing.notes
        except Exception:  # noqa: BLE001 - a corrupt prior record is replaced.
            pass

    record = ModelArtifactRecord(
        artifact_id=artifact_id,
        run_id=run_id,
        created_at=created_at,
        updated_at=now,
        path=normalized,
        kind=kind or "adapter",
        status=status,
        fingerprint=compute_fingerprint(normalized),
        content_hash=compute_content_hash(normalized),
        notes=keep_notes,
    )
    save_artifact_record(project_dir, record)
    return record


def list_artifacts(project_dir: Path | str) -> list[tuple[ModelArtifactRecord, str]]:
    """All artifacts (newest first) paired with their computed integrity."""

    directory = registry_dir(project_dir)
    if not directory.exists():
        return []
    seen: set[str] = set()
    records: list[ModelArtifactRecord] = []
    for path in directory.glob("*.json"):
        try:
            record = load_artifact_record(path)
        except Exception:  # noqa: BLE001 - skip a corrupt record.
            continue
        if record.artifact_id in seen:
            continue  # tolerate a duplicate file (first wins)
        seen.add(record.artifact_id)
        records.append(record)
    records.sort(key=lambda record: record.artifact_id, reverse=True)
    return [(record, artifact_integrity(record)) for record in records]


def update_artifact_status(
    project_dir: Path | str, artifact_id: str, status: str, now: str = ""
) -> ModelArtifactRecord:
    if status not in ARTIFACT_STATUSES:
        raise ValueError(f"Unknown artifact status '{status}'.")
    path = artifact_path(project_dir, artifact_id)
    if not path.exists():
        raise ValueError(f"No artifact '{artifact_id}'.")
    record = load_artifact_record(path)
    updated = record.model_copy(update={"status": status, "updated_at": now})
    save_artifact_record(project_dir, updated)
    return updated
