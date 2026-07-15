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

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path

from corpus_studio.platform.contracts import (
    AdapterExportStateEvidence,
    ArtifactManifest,
    ResolvedExecutionConfiguration,
)

_ARTIFACT_KINDS = frozenset(
    {"adapter", "checkpoint", "merged_model", "gguf", "onnx", "quantized", "other"}
)
_MAX_ADAPTER_CONFIG_BYTES = 1024 * 1024


@dataclass(frozen=True)
class SealedAdapterEvidence:
    safetensors_sha256: str
    adapter_config_sha256: str
    tensor_state_sha256: str
    adapter_config_semantic_sha256: str
    tensor_names: tuple[str, ...]
    target_modules: tuple[str, ...]


_SET_LIKE_CONFIG_FIELDS = frozenset(
    {"exclude_modules", "modules_to_save", "target_modules", "target_parameters"}
)
_LORA_TENSOR = re.compile(
    r"^(?P<module>.+)\.lora_(?P<side>A|B)(?:\.[^.]+)?\.weight$"
)
_WEIGHT_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".gguf", ".onnx", ".pt", ".pth", ".safetensors"}
)
# Explicitly classified, benign trainer metadata permitted in the sealed adapter directory. TRL /
# transformers Trainer.save_model writes ``training_args.bin`` (a serialized ``TrainingArguments``) next
# to the adapter; its ``.bin`` suffix is NOT a model-weight payload. It is admitted ONLY at the artifact
# root, by exact basename, as a bounded single-hard-link regular file, and is NEVER deserialized here
# (it is not a source of training truth); its bytes are covered by the artifact content hash like every
# other file. Every OTHER ``.bin`` (and every real weight payload) stays fail-closed.
_ROOT_AUXILIARY_METADATA_FILES = frozenset({"training_args.bin"})
_MAX_AUXILIARY_METADATA_BYTES = 1 << 20  # 1 MiB; a real training_args.bin is a few KiB.


def _semantic_json_value(value: object, *, field_name: str | None = None) -> object:
    if isinstance(value, Enum):
        return _semantic_json_value(value.value, field_name=field_name)
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_json_value(item, field_name=str(key))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_semantic_json_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (list, tuple)):
        normalized = [_semantic_json_value(item) for item in value]
        if field_name in _SET_LIKE_CONFIG_FIELDS:
            return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
        return normalized
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"adapter config contains non-JSON semantic value {type(value).__name__}")


def canonical_adapter_config_sha256(config: Mapping[str, object]) -> str:
    """Canonical identity for every serialized PEFT config field, including future fields."""

    normalized = _semantic_json_value(config)
    return hashlib.sha256(
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _stable_bounded_file_bytes(path: Path, *, limit: int) -> bytes:
    """Read one regular non-link file and prove the pathname still names the opened inode."""

    try:
        before_path = path.lstat()
        if not stat.S_ISREG(before_path.st_mode) or path.is_symlink():
            raise ValueError(f"{path.name} must be a regular non-link file")
        with path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            payload = handle.read(limit + 1)
            opened_after = os.fstat(handle.fileno())
        after_path = path.lstat()
    except OSError as exc:
        raise ValueError(f"{path.name} could not be read") from exc
    identities = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        for item in (before_path, opened_before, opened_after, after_path)
    }
    if len(identities) != 1:
        raise ValueError(f"{path.name} changed while it was read")
    if not payload or len(payload) > limit:
        raise ValueError(f"{path.name} size is invalid")
    return payload


def _validate_root_auxiliary_metadata(path: Path) -> None:
    """Fail-closed structural checks for an explicitly permitted root auxiliary metadata file.

    It must be a bounded, single-hard-link regular file (never a symlink, hard link, or special file).
    It is NEVER opened for deserialization here - only its link status and size are inspected; its bytes
    are covered by the artifact content hash (which detects any later tamper)."""

    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ValueError("adapter artifact auxiliary metadata is not a regular file")
    if info.st_nlink != 1:
        raise ValueError("adapter artifact auxiliary metadata is hard-linked")
    if info.st_size > _MAX_AUXILIARY_METADATA_BYTES:
        raise ValueError("adapter artifact auxiliary metadata exceeds the permitted size")


def _validate_adapter_tree(root: Path) -> None:
    """Reject links, checkpoint payloads, and any second model-weight format recursively.

    A file is classified by NAME, not by extension alone: an explicitly permitted root auxiliary
    metadata file (``training_args.bin``) is admitted under the narrow policy in
    :func:`_validate_root_auxiliary_metadata`; every other weight-suffixed or model-weight-named file
    (a second ``.safetensors``, ``pytorch_model*``, ``model*.bin``, ``optimizer.pt``, a nested/arbitrary
    ``.bin``, ...) stays fail-closed."""

    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise ValueError("adapter artifact directory is unavailable") from exc
    if not stat.S_ISDIR(root_stat.st_mode) or root.is_symlink():
        raise ValueError("adapter artifact must be a regular, non-link directory")
    resolved_root = root.resolve(strict=True)
    try:
        for current_raw, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            current = Path(current_raw)
            for name in sorted(dirnames):
                candidate = current / name
                if candidate.is_symlink() or not stat.S_ISDIR(candidate.lstat().st_mode):
                    raise ValueError("adapter artifact contains a linked or irregular directory")
                if name.startswith("checkpoint-"):
                    raise ValueError("adapter artifact contains an intermediate checkpoint")
                candidate.resolve(strict=True).relative_to(resolved_root)
            for name in sorted(filenames):
                candidate = current / name
                if candidate.is_symlink() or not stat.S_ISREG(candidate.lstat().st_mode):
                    raise ValueError("adapter artifact contains a linked or irregular file")
                candidate.resolve(strict=True).relative_to(resolved_root)
                relative = candidate.relative_to(root).as_posix()
                if relative == "adapter_model.safetensors":
                    continue
                if relative in _ROOT_AUXILIARY_METADATA_FILES:
                    # Explicitly classified benign metadata at the artifact ROOT only (``relative`` has
                    # no path separator). A nested ``dir/training_args.bin`` is not in the set and falls
                    # through to the weight-payload rejection below.
                    _validate_root_auxiliary_metadata(candidate)
                    continue
                if (
                    candidate.suffix.lower() in _WEIGHT_SUFFIXES
                    or name.startswith("adapter_model.")
                    or name.startswith("pytorch_model")
                ):
                    raise ValueError(
                        "adapter artifact contains an alternate or nested model-weight payload"
                    )
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("adapter artifact"):
            raise
        raise ValueError("adapter artifact tree is unsafe or changed during validation") from exc


def validate_sealed_adapter_artifact(
    path: str | Path,
    execution: ResolvedExecutionConfiguration,
    export_evidence: AdapterExportStateEvidence,
) -> SealedAdapterEvidence:
    """Validate the exact usable PEFT adapter payload against the sealed adapter policy."""

    root = Path(path)
    _validate_adapter_tree(root)
    safetensors_path = root / "adapter_model.safetensors"
    config_path = root / "adapter_config.json"
    for label, candidate in (
        ("adapter Safetensors", safetensors_path),
        ("adapter config", config_path),
    ):
        if not candidate.is_file() or candidate.is_symlink():
            raise ValueError(f"{label} is missing, non-regular, or link-like")
        try:
            candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"{label} escapes the adapter artifact") from exc
    from corpus_studio.platform.parameter_accounting import (  # noqa: PLC0415
        ParameterAccountingError,
        validate_safetensors_tensor_file,
    )

    try:
        tensor_state = validate_safetensors_tensor_file(safetensors_path)
    except ParameterAccountingError as exc:
        raise ValueError(f"adapter Safetensors is invalid: {exc}") from exc
    if (
        tensor_state.tensor_state_sha256 != export_evidence.after_sha256
        or list(tensor_state.tensor_names) != export_evidence.tensor_names
    ):
        raise ValueError("saved adapter tensor state differs from the trained export state")

    raw_config = _stable_bounded_file_bytes(config_path, limit=_MAX_ADAPTER_CONFIG_BYTES)

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"adapter config repeats key {key!r}")
            result[key] = value
        return result

    try:
        config = json.loads(
            raw_config.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("adapter config is not valid bounded JSON") from exc
    if not isinstance(config, dict):
        raise ValueError("adapter config must be a JSON object")

    sealed = execution.adapter
    expected: dict[str, object] = {
        "peft_type": "LORA",
        "task_type": execution.adapter_task_type,
        "r": sealed.lora_r,
        "lora_alpha": sealed.lora_alpha,
        "bias": sealed.bias,
    }
    for field_name, expected_value in expected.items():
        if config.get(field_name) != expected_value:
            raise ValueError(f"adapter config field {field_name!r} differs from the seal")
    dropout = config.get("lora_dropout")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValueError("adapter config lora_dropout is invalid")
    if float(dropout) != sealed.lora_dropout:
        raise ValueError("adapter config lora_dropout differs from the seal")
    raw_targets = config.get("target_modules")
    if (
        not isinstance(raw_targets, list)
        or not raw_targets
        or any(not isinstance(item, str) or not item for item in raw_targets)
        or len(raw_targets) != len(set(raw_targets))
    ):
        raise ValueError("adapter config target_modules are invalid")

    tensor_sides: dict[str, set[str]] = {}
    for tensor_name in tensor_state.tensor_names:
        match = _LORA_TENSOR.fullmatch(tensor_name)
        if match is None:
            raise ValueError("adapter Safetensors contains non-LoRA trainable state")
        module = match.group("module")
        tensor_sides.setdefault(module, set()).add(match.group("side"))
    if any(sides != {"A", "B"} for sides in tensor_sides.values()):
        raise ValueError("adapter Safetensors does not contain complete LoRA A/B tensor pairs")
    config_targets = sorted(raw_targets)
    matched_targets: set[str] = set()
    for module in tensor_sides:
        matches = [
            target
            for target in config_targets
            if module == target or module.endswith("." + target)
        ]
        if len(matches) != 1:
            raise ValueError("adapter config targets do not uniquely cover saved LoRA modules")
        matched_targets.add(matches[0])
    if matched_targets != set(config_targets):
        raise ValueError("adapter config contains a target with no saved LoRA tensor pair")
    if sealed.target_modules != ["all-linear"] and config_targets != sealed.target_modules:
        raise ValueError("adapter config target_modules differ from the explicit seal")

    if config.get("base_model_name_or_path") != execution.inputs.model.location:
        raise ValueError("adapter config base-model linkage differs from the seal")
    if config.get("inference_mode") is not True:
        raise ValueError("saved adapter config must be inference-mode reloadable")
    package_versions = {
        item.name: item.version for item in execution.trainer_interface.package_versions
    }
    if package_versions.get("peft") is not None and config.get("peft_version") != package_versions["peft"]:
        raise ValueError("adapter config PEFT version differs from the sealed runtime")

    config_semantic_sha256 = canonical_adapter_config_sha256(config)
    if config_semantic_sha256 != export_evidence.adapter_config_semantic_sha256:
        raise ValueError("saved adapter config semantics differ from the trainer-bound config")
    _validate_adapter_tree(root)
    return SealedAdapterEvidence(
        safetensors_sha256=tensor_state.content_sha256,
        adapter_config_sha256=hashlib.sha256(raw_config).hexdigest(),
        tensor_state_sha256=tensor_state.tensor_state_sha256,
        adapter_config_semantic_sha256=config_semantic_sha256,
        tensor_names=tensor_state.tensor_names,
        target_modules=tuple(config_targets),
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
    metadata_hash = None
    if kind == "adapter":
        try:
            config_bytes = _stable_bounded_file_bytes(
                Path(path) / "adapter_config.json",
                limit=_MAX_ADAPTER_CONFIG_BYTES,
            )
        except ValueError:
            pass
        else:
            metadata_hash = hashlib.sha256(config_bytes).hexdigest()
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
                "metadata_hash": metadata_hash,
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
    metadata_matches = True
    if stored.metadata_hash is not None:
        try:
            current_metadata = hashlib.sha256(
                _stable_bounded_file_bytes(
                    Path(manifest.path) / "adapter_config.json",
                    limit=_MAX_ADAPTER_CONFIG_BYTES,
                )
            ).hexdigest()
        except ValueError:
            metadata_matches = False
        else:
            metadata_matches = current_metadata == stored.metadata_hash

    if current_fingerprint is None:
        status = "missing"
    elif not metadata_matches:
        status = "modified"
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
