"""Dependency-light, offline inspection for model and tokenizer snapshots.

This module reads bounded metadata and inventories local files without importing torch,
transformers, tokenizers, or repository-provided Python. It never follows a symlink/junction and
never authorizes custom code. Network retrieval remains the separate model-fetch operation.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal

from pydantic import Field, model_validator

from .common import ContractModel, License, Ref
from .contracts import (
    CompatibilityCheck,
    DescriptorFile,
    DescriptorSource,
    DescriptorVerification,
    DimensionEvidence,
    EmbeddingVocabulary,
    ModelDescriptor,
    ModelTokenizerCompatibility,
    ParameterAccountingReport,
    ParameterComponent,
    ParameterCount,
    ParameterCountHandling,
    ParameterRepresentation,
    SpecialToken,
    TokenizerDescriptor,
    TrustRequirement,
)
from .enums import (
    CompatibilityStatus,
    CountHandling,
    DescriptorFileRole,
    EvidenceKind,
    ModelAttentionType,
    ModelExecutionKind,
    ModelFormat,
    ModelSourceKind,
    ModelTaskClass,
    ParameterCountKind,
    PositionalEncoding,
    QuantizationMode,
    TokenizerFormat,
    VerificationOutcome,
)
from .moe_inspector import inspect_moe_topology

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_INVENTORY_FILES = 100_000
HF_MODEL_MAX_LENGTH_SENTINEL = 1_000_000_000

_WEIGHT_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".jit",
    ".torchscript",
    ".npy",
    ".npz",
}
_IGNORED_DIRECTORIES = {".git", "__pycache__"}

HashStatus = Literal["verified", "not_requested", "unreadable", "skipped_unsafe"]
SerializationRisk = Literal["safe", "pickle", "executable_code", "archive", "unknown"]


class ModelInspectionError(RuntimeError):
    """A bounded, user-facing static inspection failure."""


class ModelInspectionBundle(ContractModel):
    model: ModelDescriptor
    tokenizer: TokenizerDescriptor | None = None
    compatibility: ModelTokenizerCompatibility | None = None
    parameter_accounting: ParameterAccountingReport | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_parameter_accounting_model(self) -> ModelInspectionBundle:
        if self.parameter_accounting is None:
            return self
        from .parameter_accounting import verify_parameter_accounting_hash

        if not verify_parameter_accounting_hash(self.parameter_accounting):
            raise ValueError("parameter-accounting report hash mismatch")
        model_ref = self.parameter_accounting.model_ref
        expected_hash = self.model.source.snapshot_sha256 or self.model.inventory_sha256
        expected_algo = (
            "sha256"
            if self.model.source.snapshot_sha256 is not None
            else "sha256-ordered-exact-v1"
            if self.model.inventory_sha256 is not None
            else None
        )
        if model_ref.id != self.model.model_id or (
            expected_hash is not None
            and (
                model_ref.hash is None
                or model_ref.hash.algo != expected_algo
                or model_ref.hash.value != expected_hash
            )
        ):
            raise ValueError("parameter accounting must reference the bundled model revision")
        return self


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(now: Callable[[], datetime]) -> str:
    return now().astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_link_like(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction and is_junction())


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _safe_root(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        raise ModelInspectionError(f"snapshot path does not exist: {candidate}")
    if not candidate.is_dir():
        raise ModelInspectionError(f"snapshot path is not a directory: {candidate}")
    if _is_link_like(candidate):
        raise ModelInspectionError(f"snapshot root cannot be a symlink or junction: {candidate}")
    return candidate.resolve(strict=True)


def _sha256_file_stable(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        before = path.stat()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise ModelInspectionError(f"cannot hash {path.name}: {exc}") from exc
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ModelInspectionError(f"source changed while hashing: {path.name}")
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_role(relative: str) -> DescriptorFileRole:
    name = Path(relative).name.lower()
    suffix = Path(relative).suffix.lower()
    if suffix == ".py":
        return DescriptorFileRole.custom_code
    if name == "config.json":
        return DescriptorFileRole.config
    if name == "generation_config.json":
        return DescriptorFileRole.generation_config
    if name == "tokenizer_config.json":
        return DescriptorFileRole.tokenizer_config
    if name in {"special_tokens_map.json", "added_tokens.json"}:
        return DescriptorFileRole.special_tokens
    if name.endswith(".index.json") and ("weight" in name or "model" in name):
        return DescriptorFileRole.weight_index
    if suffix in _WEIGHT_SUFFIXES:
        return DescriptorFileRole.weights
    if name in {"tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json", "merges.txt"} or (
        suffix == ".tiktoken"
    ):
        return DescriptorFileRole.tokenizer
    if name.startswith("readme"):
        return DescriptorFileRole.model_card
    if name.startswith("license") or name.startswith("copying"):
        return DescriptorFileRole.license
    return DescriptorFileRole.other


def _model_format(path: str) -> ModelFormat | None:
    suffix = Path(path).suffix.lower()
    if suffix == ".safetensors":
        return ModelFormat.safetensors
    if suffix in {".bin", ".pt", ".pth", ".ckpt"}:
        return ModelFormat.pytorch_pickle
    if suffix == ".gguf":
        return ModelFormat.gguf
    if suffix == ".onnx":
        return ModelFormat.onnx
    if suffix in {".jit", ".torchscript"}:
        return ModelFormat.torchscript
    if suffix in {".npy", ".npz"}:
        return ModelFormat.numpy
    return None


def _serialization_risk(path: str) -> SerializationRisk:
    suffix = Path(path).suffix.lower()
    if suffix in {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}:
        return "pickle"
    if suffix == ".py":
        return "executable_code"
    if suffix in {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z"}:
        return "archive"
    if suffix in {
        ".json",
        ".txt",
        ".md",
        ".safetensors",
        ".gguf",
        ".onnx",
        ".model",
        ".npy",
        ".npz",
    }:
        return "safe"
    return "unknown"


def _inventory(
    root: Path,
    *,
    hash_weights: bool,
) -> tuple[list[DescriptorFile], list[str], bool, str]:
    records: list[DescriptorFile] = []
    warnings: list[str] = []
    complete = True

    for current_raw, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(current_raw)
        kept_dirs: list[str] = []
        for name in sorted(dirnames):
            candidate = current / name
            relative = candidate.relative_to(root).as_posix()
            if name in _IGNORED_DIRECTORIES:
                warnings.append(f"ignored non-model directory: {relative}")
                continue
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                warnings.append(f"skipped unreadable directory: {relative}")
                complete = False
                continue
            if _is_link_like(candidate) or not _is_within(resolved, root):
                warnings.append(f"skipped unsafe linked directory: {relative}")
                complete = False
                continue
            kept_dirs.append(name)
        dirnames[:] = kept_dirs

        for name in sorted(filenames):
            if len(records) >= MAX_INVENTORY_FILES:
                raise ModelInspectionError(
                    f"snapshot exceeds the {MAX_INVENTORY_FILES} file inspection limit"
                )
            candidate = current / name
            relative = candidate.relative_to(root).as_posix()
            role = _file_role(relative)
            file_format = _model_format(relative)
            risk = _serialization_risk(relative)
            try:
                stat = candidate.lstat()
            except OSError:
                warnings.append(f"could not stat file: {relative}")
                complete = False
                continue

            if _is_link_like(candidate):
                records.append(
                    DescriptorFile(
                        path=relative,
                        role=role,
                        size_bytes=stat.st_size,
                        format=file_format,
                        hash_status="skipped_unsafe",
                        serialization_risk=risk,
                        is_link=True,
                    )
                )
                warnings.append(f"did not follow linked file: {relative}")
                complete = False
                continue

            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                warnings.append(f"could not resolve file: {relative}")
                complete = False
                continue
            if not _is_within(resolved, root):
                warnings.append(f"skipped path outside snapshot: {relative}")
                complete = False
                continue

            should_hash = role != DescriptorFileRole.weights or hash_weights
            digest: str | None = None
            hash_status: HashStatus = "not_requested"
            if should_hash:
                try:
                    digest = _sha256_file_stable(candidate)
                    hash_status = "verified"
                except ModelInspectionError as exc:
                    warnings.append(str(exc))
                    hash_status = "unreadable"
                    complete = False
            records.append(
                DescriptorFile(
                    path=relative,
                    role=role,
                    size_bytes=stat.st_size,
                    format=file_format,
                    hash_status=hash_status,
                    sha256=digest,
                    serialization_risk=risk,
                )
            )

    records.sort(key=lambda item: item.path)
    canonical = json.dumps(
        [record.model_dump(mode="json") for record in records],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return records, warnings, complete, _sha256_bytes(canonical)


def _inventory_hash(files: list[DescriptorFile], relative: str) -> str | None:
    for item in files:
        if item.path == relative and item.hash_status == "verified":
            return item.sha256
    return None


def _load_json(
    root: Path,
    relative: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any] | None:
    candidate = root / relative
    if not candidate.exists():
        return None
    if _is_link_like(candidate):
        raise ModelInspectionError(f"refusing linked metadata file: {relative}")
    resolved = candidate.resolve(strict=True)
    if not _is_within(resolved, root):
        raise ModelInspectionError(f"metadata path escapes snapshot: {relative}")
    before = candidate.stat()
    size = before.st_size
    if size > MAX_JSON_BYTES:
        raise ModelInspectionError(
            f"metadata file exceeds the {MAX_JSON_BYTES}-byte limit: {relative}"
    )
    try:
        with candidate.open("rb") as handle:
            content = handle.read(MAX_JSON_BYTES + 1)
        if len(content) > MAX_JSON_BYTES:
            raise ModelInspectionError(
                f"metadata file exceeds the {MAX_JSON_BYTES}-byte limit: {relative}"
            )
        after = candidate.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ModelInspectionError(f"source changed while reading metadata: {relative}")
        digest = _sha256_bytes(content)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ModelInspectionError(f"source changed after inventory: {relative}")
        value = json.loads(content.decode("utf-8"))
    except ModelInspectionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelInspectionError(f"invalid JSON metadata in {relative}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelInspectionError(f"metadata root must be an object: {relative}")
    return value


def _resolved_commit(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.lower()
    if 7 <= len(lowered) <= 64 and all(char in "0123456789abcdef" for char in lowered):
        return lowered
    raise ModelInspectionError("--resolved-commit must be a 7-64 character hexadecimal commit")


def _source(
    root: Path,
    *,
    repository: str | None,
    requested_revision: str | None,
    resolved_commit: str | None,
    snapshot_sha256: str | None,
) -> DescriptorSource:
    commit = _resolved_commit(resolved_commit)
    return DescriptorSource(
        kind=ModelSourceKind.huggingface if repository else ModelSourceKind.local,
        repository=repository,
        requested_revision=requested_revision,
        resolved_revision=commit,
        resolved_commit=commit,
        revision_pinned=commit is not None,
        local_path=str(root),
        snapshot_sha256=snapshot_sha256,
        evidence_source="static_local_inspection",
    )


def _trust(files: list[DescriptorFile], *configs: dict[str, Any] | None) -> TrustRequirement:
    code_files = sorted(
        item.path for item in files if item.role == DescriptorFileRole.custom_code
    )
    auto_map: dict[str, Any] = {}
    for config in configs:
        if config and isinstance(config.get("auto_map"), dict):
            auto_map.update(config["auto_map"])
    custom_required = bool(code_files or auto_map)
    notes = (
        [
            "Custom code was detected. Static inspection did not execute it; a separate exact-revision "
            "approval and isolated worker environment are required."
        ]
        if custom_required
        else []
    )
    return TrustRequirement(
        custom_code_required=custom_required,
        approval_required=custom_required,
        isolated_execution_required=custom_required,
        custom_code_files=code_files,
        detected_auto_map=auto_map,
        notes=notes,
    )


def _integrity_outcome(files: list[DescriptorFile], complete: bool) -> VerificationOutcome:
    if any(item.hash_status == "unreadable" for item in files):
        return VerificationOutcome.failed
    if complete and files and all(item.hash_status == "verified" for item in files):
        return VerificationOutcome.passed
    if files and any(item.hash_status == "verified" for item in files):
        return VerificationOutcome.partial
    return VerificationOutcome.not_checked


def _verification(
    *,
    files: list[DescriptorFile],
    complete: bool,
    metadata_present: bool,
    license_present: bool,
    trust: TrustRequirement,
    warnings: list[str],
    captured_at: str,
) -> DescriptorVerification:
    return DescriptorVerification(
        metadata=(
            VerificationOutcome.passed
            if metadata_present
            else VerificationOutcome.partial
        ),
        integrity=_integrity_outcome(files, complete),
        license=(
            VerificationOutcome.partial
            if license_present
            else VerificationOutcome.not_checked
        ),
        custom_code_policy=(
            VerificationOutcome.partial
            if trust.custom_code_required
            else VerificationOutcome.passed
        ),
        inspected_at=captured_at,
        inspector="corpus_studio.platform.model_inspector/v2",
        warnings=sorted(set(warnings)),
    )


def _license(config: dict[str, Any] | None, files: list[DescriptorFile]) -> License | None:
    declared = config.get("license") if config else None
    if isinstance(declared, str) and declared.strip():
        return License(name=declared.strip(), source="declared")
    if any(item.role == DescriptorFileRole.license for item in files):
        return License(source="unknown")
    return None


def _task_classes(config: dict[str, Any]) -> list[ModelTaskClass]:
    text = " ".join(
        [str(config.get("model_type", ""))]
        + [str(item) for item in config.get("architectures", []) if isinstance(item, str)]
    ).lower()
    found: set[ModelTaskClass] = set()
    mappings = (
        ("causallm", ModelTaskClass.causal_lm),
        ("maskedlm", ModelTaskClass.masked_lm),
        ("conditionalgeneration", ModelTaskClass.seq2seq_lm),
        ("sequenceclassification", ModelTaskClass.classification),
        ("reward", ModelTaskClass.reward_model),
        ("vision", ModelTaskClass.vision),
        ("audio", ModelTaskClass.speech),
        ("multimodal", ModelTaskClass.multimodal),
    )
    for needle, value in mappings:
        if needle in text:
            found.add(value)
    return sorted(found or {ModelTaskClass.unknown}, key=lambda item: item.value)


def _attention_type(config: dict[str, Any]) -> ModelAttentionType:
    declared = str(config.get("attention_type", "")).lower()
    for item in ModelAttentionType:
        if declared == item.value:
            return item
    if isinstance(config.get("sliding_window"), int) and config["sliding_window"] > 0:
        return ModelAttentionType.sliding_window
    model_type = str(config.get("model_type", "")).lower()
    if "mamba" in model_type or "state_space" in model_type:
        return ModelAttentionType.state_space
    return ModelAttentionType.unknown


def _positional_encoding(config: dict[str, Any]) -> PositionalEncoding:
    if "rope_theta" in config or "rope_scaling" in config:
        return PositionalEncoding.rope
    if config.get("alibi") is True:
        return PositionalEncoding.alibi
    declared = str(config.get("position_embedding_type", "")).lower()
    for item in PositionalEncoding:
        if declared == item.value:
            return item
    return PositionalEncoding.unknown


def _dimension(config: dict[str, Any], *keys: str) -> DimensionEvidence | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return DimensionEvidence(value=value, source=f"config.{key}", evidence=EvidenceKind.declared)
    return None


def _quantization(config: dict[str, Any]) -> tuple[QuantizationMode | None, dict[str, Any]]:
    raw = config.get("quantization_config")
    if not isinstance(raw, dict):
        return None, {}
    method = str(raw.get("quant_method") or raw.get("quantization_method") or "").lower()
    aliases = {"bitsandbytes": "int4" if raw.get("load_in_4bit") else "int8"}
    method = aliases.get(method, method)
    try:
        return QuantizationMode(method), raw
    except ValueError:
        return None, raw


def _parameter_representation(
    config: dict[str, Any],
    files: list[DescriptorFile],
) -> ParameterRepresentation:
    quantization, details = _quantization(config)
    dtype_raw = config.get("torch_dtype") or config.get("dtype")
    dtype = str(dtype_raw) if dtype_raw is not None else None
    by_format: dict[ModelFormat, list[str]] = {}
    for item in files:
        if item.role == DescriptorFileRole.weights and item.format is not None:
            by_format.setdefault(item.format, []).append(item.path)
    components = [
        ParameterComponent(
            component_id=f"weights-{format_.value}",
            format=format_,
            storage_dtype=dtype,
            quantization=quantization,
            quantization_details=details,
            file_refs=sorted(paths),
        )
        for format_, paths in sorted(by_format.items(), key=lambda item: item[0].value)
    ]
    counts: list[ParameterCount] = []
    for key in ("num_parameters", "n_params", "parameter_count"):
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            counts.append(
                ParameterCount(
                    kind=ParameterCountKind.logical,
                    value=value,
                    scope="model",
                    measurement_window="static_model",
                    source=f"config.{key}",
                    evidence=EvidenceKind.declared,
                    handling=ParameterCountHandling(
                        tied=CountHandling.unknown,
                        shared=CountHandling.unknown,
                        replicated=CountHandling.unknown,
                        generated=CountHandling.unknown,
                        quantized=CountHandling.unknown,
                    ),
                )
            )
            break
    return ParameterRepresentation(
        kind=ModelExecutionKind.unknown,
        components=components,
        counts=counts,
    )


def inspect_model(
    path: str | Path,
    *,
    model_id: str,
    tokenizer_id: str | None = None,
    repository: str | None = None,
    requested_revision: str | None = None,
    resolved_commit: str | None = None,
    hash_weights: bool = False,
    now: Callable[[], datetime] = _utcnow,
) -> ModelDescriptor:
    """Inspect a local model snapshot without importing or executing model code."""

    root = _safe_root(path)
    files, warnings, complete, inventory_sha256 = _inventory(
        root, hash_weights=hash_weights
    )
    config_sha256 = _inventory_hash(files, "config.json")
    config = _load_json(
        root,
        "config.json",
        expected_sha256=config_sha256,
    ) or {}
    if not config and not any(item.role == DescriptorFileRole.weights for item in files):
        raise ModelInspectionError("snapshot has neither config.json nor recognized weight files")

    trust = _trust(files, config)
    formats = sorted(
        {
            item.format
            for item in files
            if item.role == DescriptorFileRole.weights and item.format is not None
        },
        key=lambda item: item.value,
    )
    if ModelFormat.pytorch_pickle in formats:
        warnings.append(
            "Pickle-based model weights were found. Loading them can execute code; prefer safetensors."
        )
    if repository and resolved_commit is None:
        warnings.append("Repository revision is not pinned to an immutable commit.")
    if trust.custom_code_required:
        warnings.extend(trust.notes)

    vocab = _dimension(config, "vocab_size")
    vocabulary = EmbeddingVocabulary(
        declared_vocab_size=vocab,
        input_embedding_rows=vocab,
        output_head_rows=vocab,
        tied_embeddings=(
            config.get("tie_word_embeddings")
            if isinstance(config.get("tie_word_embeddings"), bool)
            else None
        ),
    )
    context = _dimension(
        config, "max_position_embeddings", "n_positions", "seq_length", "max_sequence_length"
    )
    topology = inspect_moe_topology(config, config_sha256=config_sha256)
    warnings.extend(topology.inspection.warnings)
    representation = _parameter_representation(config, files).model_copy(
        update={"kind": topology.execution_kind}
    )
    timestamp = _timestamp(now)
    snapshot_hash = (
        inventory_sha256
        if complete and files and all(item.hash_status == "verified" for item in files)
        else None
    )
    model_license = _license(config, files)
    return ModelDescriptor(
        model_id=model_id,
        source=_source(
            root,
            repository=repository,
            requested_revision=requested_revision,
            resolved_commit=resolved_commit,
            snapshot_sha256=snapshot_hash,
        ),
        architectures=sorted(
            {str(item) for item in config.get("architectures", []) if isinstance(item, str)}
        ),
        model_family=(
            str(config["model_type"]) if isinstance(config.get("model_type"), str) else None
        ),
        task_classes=_task_classes(config),
        formats=formats or [ModelFormat.unknown],
        parameters=representation,
        topology=topology,
        vocabulary=vocabulary,
        context_window=context,
        tokenizer_ref=Ref(id=tokenizer_id) if tokenizer_id else None,
        attention_type=_attention_type(config),
        positional_encoding=_positional_encoding(config),
        license=model_license,
        trust=trust,
        files=files,
        inventory_complete=complete,
        storage_size_bytes=sum(item.size_bytes for item in files),
        inventory_sha256=inventory_sha256,
        verification=_verification(
            files=files,
            complete=complete,
            metadata_present=bool(config),
            license_present=model_license is not None,
            trust=trust,
            warnings=warnings,
            captured_at=timestamp,
        ),
        captured_at=timestamp,
        notes=sorted(set(warnings)),
    )


def _token_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return None


def _tokenizer_vocabulary(
    tokenizer_json: dict[str, Any] | None,
    tokenizer_config: dict[str, Any],
) -> tuple[int | None, int | None, int | None, dict[str, int], int, set[str]]:
    ids: dict[str, int] = {}
    base_count: int | None = None
    if tokenizer_json:
        model = tokenizer_json.get("model")
        vocab = model.get("vocab") if isinstance(model, dict) else None
        if isinstance(vocab, dict):
            for token, token_id in vocab.items():
                if isinstance(token, str) and isinstance(token_id, int) and token_id >= 0:
                    ids[token] = token_id
            base_count = len(ids)
        elif isinstance(vocab, list):
            for index, token in enumerate(vocab):
                if isinstance(token, str):
                    ids[token] = index
            base_count = len(ids)
    if base_count is None:
        declared = tokenizer_config.get("vocab_size")
        if isinstance(declared, int) and not isinstance(declared, bool) and declared >= 0:
            base_count = declared

    base_ids = set(ids.values())
    added_entries: set[tuple[str, int]] = set()
    if tokenizer_json and isinstance(tokenizer_json.get("added_tokens"), list):
        for item in tokenizer_json["added_tokens"]:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            token_id = item.get("id")
            if isinstance(content, str) and isinstance(token_id, int) and token_id >= 0:
                added_entries.add((content, token_id))
                ids[content] = token_id

    max_id = max(ids.values()) if ids else None
    effective = max_id + 1 if max_id is not None else base_count
    added_count = len({token_id for _, token_id in added_entries if token_id not in base_ids})
    return base_count, effective, max_id, ids, added_count, {
        content for content, _ in added_entries
    }


def _special_tokens(
    tokenizer_config: dict[str, Any],
    special_map: dict[str, Any],
    token_ids: dict[str, int],
    added_contents: set[str],
) -> list[SpecialToken]:
    found: dict[tuple[str, str, int | None], SpecialToken] = {}
    merged = {**special_map, **tokenizer_config}
    for key, raw in merged.items():
        if key.endswith("_token") and key != "additional_special_tokens":
            content = _token_content(raw)
            if content is None:
                continue
            role = key.removesuffix("_token")
            explicit_id = merged.get(f"{key}_id")
            token_id = explicit_id if isinstance(explicit_id, int) else token_ids.get(content)
            token = SpecialToken(
                role=role,
                content=content,
                token_id=token_id,
                added=content in added_contents,
            )
            found[(token.role, token.content, token.token_id)] = token
    additional = merged.get("additional_special_tokens")
    if isinstance(additional, list):
        for raw in additional:
            content = _token_content(raw)
            if content is None:
                continue
            token = SpecialToken(
                role="additional_special_token",
                content=content,
                token_id=token_ids.get(content),
                added=content in added_contents,
            )
            found[(token.role, token.content, token.token_id)] = token
    return sorted(
        found.values(),
        key=lambda item: (item.role, item.content, item.token_id if item.token_id is not None else -1),
    )


def _chat_template(value: Any) -> str | list[dict[str, Any]] | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    return None


def inspect_tokenizer(
    path: str | Path,
    *,
    tokenizer_id: str,
    repository: str | None = None,
    requested_revision: str | None = None,
    resolved_commit: str | None = None,
    hash_weights: bool = False,
    now: Callable[[], datetime] = _utcnow,
) -> TokenizerDescriptor:
    """Inspect a local tokenizer snapshot without loading its implementation."""

    root = _safe_root(path)
    files, warnings, complete, inventory_sha256 = _inventory(
        root, hash_weights=hash_weights
    )
    tokenizer_json = _load_json(
        root,
        "tokenizer.json",
        expected_sha256=_inventory_hash(files, "tokenizer.json"),
    )
    tokenizer_config = _load_json(
        root,
        "tokenizer_config.json",
        expected_sha256=_inventory_hash(files, "tokenizer_config.json"),
    ) or {}
    special_map = _load_json(
        root,
        "special_tokens_map.json",
        expected_sha256=_inventory_hash(files, "special_tokens_map.json"),
    ) or {}
    if not tokenizer_json and not tokenizer_config and not any(
        item.role in {DescriptorFileRole.tokenizer, DescriptorFileRole.tokenizer_config}
        for item in files
    ):
        raise ModelInspectionError("snapshot has no recognized tokenizer metadata")

    (
        base_vocab,
        effective_vocab,
        max_id,
        token_ids,
        added_count,
        added_contents,
    ) = _tokenizer_vocabulary(tokenizer_json, tokenizer_config)
    trust = _trust(files, tokenizer_json, tokenizer_config)
    if repository and resolved_commit is None:
        warnings.append("Tokenizer repository revision is not pinned to an immutable commit.")
    if trust.custom_code_required:
        warnings.extend(trust.notes)

    tokenizer_format = TokenizerFormat.unknown
    if tokenizer_json is not None:
        tokenizer_format = TokenizerFormat.tokenizers_json
    elif any(Path(item.path).name in {"tokenizer.model", "spiece.model"} for item in files):
        tokenizer_format = TokenizerFormat.sentencepiece
    elif any(Path(item.path).suffix.lower() == ".tiktoken" for item in files):
        tokenizer_format = TokenizerFormat.tiktoken
    elif trust.custom_code_required:
        tokenizer_format = TokenizerFormat.custom

    max_length: DimensionEvidence | None = None
    raw_max_length = tokenizer_config.get("model_max_length")
    if isinstance(raw_max_length, int) and not isinstance(raw_max_length, bool):
        if 0 <= raw_max_length < HF_MODEL_MAX_LENGTH_SENTINEL:
            max_length = DimensionEvidence(
                value=raw_max_length,
                source="tokenizer_config.model_max_length",
                evidence=EvidenceKind.declared,
            )
        elif raw_max_length >= HF_MODEL_MAX_LENGTH_SENTINEL:
            warnings.append(
                "Ignored Hugging Face model_max_length sentinel; tokenizer limit remains unknown."
            )

    template = _chat_template(tokenizer_config.get("chat_template"))
    template_hash = None
    if template is not None:
        rendered = (
            template
            if isinstance(template, str)
            else json.dumps(template, sort_keys=True, separators=(",", ":"))
        )
        template_hash = _sha256_bytes(rendered.encode("utf-8"))

    timestamp = _timestamp(now)
    snapshot_hash = (
        inventory_sha256
        if complete and files and all(item.hash_status == "verified" for item in files)
        else None
    )
    normalizer = tokenizer_json.get("normalizer") if tokenizer_json else None
    pre_tokenizer = tokenizer_json.get("pre_tokenizer") if tokenizer_json else None
    return TokenizerDescriptor(
        tokenizer_id=tokenizer_id,
        source=_source(
            root,
            repository=repository,
            requested_revision=requested_revision,
            resolved_commit=resolved_commit,
            snapshot_sha256=snapshot_hash,
        ),
        format=tokenizer_format,
        implementation_class=(
            str(tokenizer_config["tokenizer_class"])
            if isinstance(tokenizer_config.get("tokenizer_class"), str)
            else None
        ),
        base_vocabulary_size=base_vocab,
        added_token_count=added_count,
        effective_vocabulary_size=effective_vocab,
        max_token_id=max_id,
        model_max_length=max_length,
        special_tokens=_special_tokens(
            tokenizer_config, special_map, token_ids, added_contents
        ),
        chat_template=template,
        chat_template_sha256=template_hash,
        normalization=normalizer if isinstance(normalizer, dict) else None,
        pre_tokenization=pre_tokenizer if isinstance(pre_tokenizer, dict) else None,
        trust=trust,
        files=files,
        inventory_complete=complete,
        storage_size_bytes=sum(item.size_bytes for item in files),
        inventory_sha256=inventory_sha256,
        verification=_verification(
            files=files,
            complete=complete,
            metadata_present=bool(tokenizer_json or tokenizer_config),
            license_present=any(item.role == DescriptorFileRole.license for item in files),
            trust=trust,
            warnings=warnings,
            captured_at=timestamp,
        ),
        captured_at=timestamp,
        notes=sorted(set(warnings)),
    )


def check_model_tokenizer_compatibility(
    model: ModelDescriptor,
    tokenizer: TokenizerDescriptor,
) -> ModelTokenizerCompatibility:
    """Perform static, evidence-labelled compatibility checks without loading either artifact."""

    checks: list[CompatibilityCheck] = []
    warnings: list[str] = []
    hard_incompatible = False
    resize_input = False
    resize_output = False
    required_rows = tokenizer.effective_vocabulary_size

    if model.tokenizer_ref is None:
        checks.append(
            CompatibilityCheck(
                check="tokenizer-link",
                outcome=VerificationOutcome.not_checked,
                message="Model descriptor has no tokenizer_ref.",
            )
        )
    elif model.tokenizer_ref.id != tokenizer.tokenizer_id:
        hard_incompatible = True
        checks.append(
            CompatibilityCheck(
                check="tokenizer-link",
                outcome=VerificationOutcome.failed,
                evidence=f"expected {model.tokenizer_ref.id}, got {tokenizer.tokenizer_id}",
                message="Tokenizer identity does not match the model linkage.",
            )
        )
    else:
        checks.append(
            CompatibilityCheck(
                check="tokenizer-link",
                outcome=VerificationOutcome.passed,
                evidence=tokenizer.tokenizer_id,
            )
        )

    input_rows = (
        model.vocabulary.input_embedding_rows.value
        if model.vocabulary.input_embedding_rows is not None
        else None
    )
    declared_output_rows = (
        model.vocabulary.output_head_rows.value
        if model.vocabulary.output_head_rows is not None
        else None
    )
    output_rows = declared_output_rows
    output_rows_inferred_from_tying = False
    if output_rows is None and model.vocabulary.tied_embeddings is True:
        output_rows = input_rows
        output_rows_inferred_from_tying = output_rows is not None
    if required_rows is None or input_rows is None:
        checks.append(
            CompatibilityCheck(
                check="input-vocabulary",
                outcome=VerificationOutcome.not_checked,
                message="Effective tokenizer size or input embedding rows are unknown.",
            )
        )
    elif required_rows > input_rows:
        resize_input = True
        checks.append(
            CompatibilityCheck(
                check="input-vocabulary",
                outcome=VerificationOutcome.failed,
                evidence=f"{required_rows} tokenizer rows > {input_rows} input rows",
                message="Input embeddings require an explicit resize operation.",
                remediation=f"Create a new artifact with at least {required_rows} input rows.",
            )
        )
    else:
        checks.append(
            CompatibilityCheck(
                check="input-vocabulary",
                outcome=VerificationOutcome.passed,
                evidence=f"{required_rows} <= {input_rows}",
            )
        )

    if required_rows is None or output_rows is None:
        checks.append(
            CompatibilityCheck(
                check="output-vocabulary",
                outcome=VerificationOutcome.not_checked,
                message="Effective tokenizer size or output head rows are unknown.",
            )
        )
    elif required_rows > output_rows:
        resize_output = True
        checks.append(
            CompatibilityCheck(
                check="output-vocabulary",
                outcome=VerificationOutcome.failed,
                evidence=f"{required_rows} tokenizer rows > {output_rows} output rows",
                message="Output head requires an explicit resize operation.",
                remediation=f"Create a new artifact with at least {required_rows} output rows.",
            )
        )
    else:
        checks.append(
            CompatibilityCheck(
                check="output-vocabulary",
                outcome=VerificationOutcome.passed,
                evidence=(
                    f"{required_rows} <= {output_rows} (inferred from tied input embeddings)"
                    if output_rows_inferred_from_tying
                    else f"{required_rows} <= {output_rows}"
                ),
            )
        )

    special_ids = [
        item.token_id for item in tokenizer.special_tokens if item.token_id is not None
    ]
    if input_rows is None or not special_ids:
        checks.append(
            CompatibilityCheck(
                check="special-token-ids",
                outcome=VerificationOutcome.not_checked,
                message="Special-token IDs or embedding rows are unavailable.",
            )
        )
    elif max(special_ids) >= input_rows:
        resize_input = True
        if model.vocabulary.tied_embeddings is True or (
            output_rows is not None and max(special_ids) >= output_rows
        ):
            resize_output = True
        required_rows = max(required_rows or 0, max(special_ids) + 1)
        checks.append(
            CompatibilityCheck(
                check="special-token-ids",
                outcome=VerificationOutcome.failed,
                evidence=f"max special token id {max(special_ids)} >= {input_rows} input rows",
                message="Special-token IDs exceed the model vocabulary.",
            )
        )
    else:
        checks.append(
            CompatibilityCheck(
                check="special-token-ids",
                outcome=VerificationOutcome.passed,
                evidence=f"max special token id {max(special_ids)} < {input_rows}",
            )
        )

    model_context = model.context_window.value if model.context_window else None
    tokenizer_context = (
        tokenizer.model_max_length.value if tokenizer.model_max_length else None
    )
    if model_context is None or tokenizer_context is None:
        checks.append(
            CompatibilityCheck(
                check="context-window",
                outcome=VerificationOutcome.not_checked,
                message="Model or tokenizer context evidence is unknown.",
            )
        )
    else:
        checks.append(
            CompatibilityCheck(
                check="context-window",
                outcome=VerificationOutcome.passed,
                evidence=f"model={model_context}, tokenizer={tokenizer_context}",
            )
        )
        if tokenizer_context != model_context:
            warnings.append(
                "Model and tokenizer context declarations differ; the smaller runtime limit governs."
            )

    if model.trust.custom_code_required or tokenizer.trust.custom_code_required:
        checks.append(
            CompatibilityCheck(
                check="custom-code-policy",
                outcome=VerificationOutcome.not_checked,
                message="Custom code requires separate exact-revision approval and isolated execution.",
            )
        )
    else:
        checks.append(
            CompatibilityCheck(
                check="custom-code-policy",
                outcome=VerificationOutcome.passed,
            )
        )

    if hard_incompatible:
        status = CompatibilityStatus.incompatible
    elif resize_input or resize_output:
        status = CompatibilityStatus.resize_required
    elif any(item.outcome == VerificationOutcome.not_checked for item in checks):
        status = CompatibilityStatus.unverified
    else:
        status = CompatibilityStatus.compatible

    return ModelTokenizerCompatibility(
        model_ref=Ref(id=model.model_id),
        tokenizer_ref=Ref(id=tokenizer.tokenizer_id),
        status=status,
        checks=checks,
        required_embedding_rows=required_rows,
        resize_input_embeddings=resize_input,
        resize_output_head=resize_output,
        warnings=warnings,
    )


def inspect_model_bundle(
    model_path: str | Path,
    *,
    model_id: str,
    tokenizer_path: str | Path | None = None,
    tokenizer_id: str | None = None,
    repository: str | None = None,
    requested_revision: str | None = None,
    resolved_commit: str | None = None,
    tokenizer_repository: str | None = None,
    tokenizer_requested_revision: str | None = None,
    tokenizer_resolved_commit: str | None = None,
    hash_weights: bool = False,
    parameter_accounting: bool = False,
    now: Callable[[], datetime] = _utcnow,
) -> ModelInspectionBundle:
    """Inspect a model and optional tokenizer and return one composable JSON bundle."""

    if tokenizer_path is not None and tokenizer_id is None:
        raise ModelInspectionError("--tokenizer-id is required with --tokenizer")
    if tokenizer_path is None and any(
        value is not None
        for value in (
            tokenizer_repository,
            tokenizer_requested_revision,
            tokenizer_resolved_commit,
        )
    ):
        raise ModelInspectionError("tokenizer source options require --tokenizer")
    model = inspect_model(
        model_path,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        repository=repository,
        requested_revision=requested_revision,
        resolved_commit=resolved_commit,
        hash_weights=hash_weights,
        now=now,
    )
    accounting = None
    if parameter_accounting:
        from .parameter_accounting import build_model_parameter_accounting

        accounting = build_model_parameter_accounting(
            model,
            snapshot_root=model_path,
            now=now,
        )
    if tokenizer_path is None or tokenizer_id is None:
        return ModelInspectionBundle(
            model=model,
            parameter_accounting=accounting,
            warnings=model.notes,
        )
    same_snapshot = _safe_root(model_path) == _safe_root(tokenizer_path)
    tokenizer_source_explicit = any(
        value is not None
        for value in (
            tokenizer_repository,
            tokenizer_requested_revision,
            tokenizer_resolved_commit,
        )
    )
    if same_snapshot and not tokenizer_source_explicit:
        tokenizer_repository = repository
        tokenizer_requested_revision = requested_revision
        tokenizer_resolved_commit = resolved_commit
    tokenizer = inspect_tokenizer(
        tokenizer_path,
        tokenizer_id=tokenizer_id,
        repository=tokenizer_repository,
        requested_revision=tokenizer_requested_revision,
        resolved_commit=tokenizer_resolved_commit,
        hash_weights=hash_weights,
        now=now,
    )
    compatibility = check_model_tokenizer_compatibility(model, tokenizer)
    tokenizer = tokenizer.model_copy(
        update={"model_compatibility": [compatibility]}
    )
    return ModelInspectionBundle(
        model=model,
        tokenizer=tokenizer,
        compatibility=compatibility,
        parameter_accounting=accounting,
        warnings=sorted(set(model.notes + tokenizer.notes + compatibility.warnings)),
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_link_like(path.parent):
        raise ModelInspectionError(f"output directory cannot be a symlink or junction: {path.parent}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_inspection_bundle(
    bundle: ModelInspectionBundle,
    out_dir: str | Path,
) -> list[Path]:
    """Atomically persist portable descriptor records. The inspected snapshot is never mutated."""

    root = Path(out_dir)
    if root.exists() and (not root.is_dir() or _is_link_like(root)):
        raise ModelInspectionError(f"output path must be a regular directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    outputs = [root / f"{bundle.model.model_id}.model.json"]
    _atomic_write_json(outputs[0], bundle.model.model_dump(mode="json"))
    if bundle.tokenizer is not None:
        tokenizer_path = root / f"{bundle.tokenizer.tokenizer_id}.tokenizer.json"
        _atomic_write_json(tokenizer_path, bundle.tokenizer.model_dump(mode="json"))
        outputs.append(tokenizer_path)
    if bundle.compatibility is not None:
        compatibility_path = root / (
            f"{bundle.model.model_id}--{bundle.compatibility.tokenizer_ref.id}.compatibility.json"
        )
        _atomic_write_json(
            compatibility_path, bundle.compatibility.model_dump(mode="json")
        )
        outputs.append(compatibility_path)
    if bundle.parameter_accounting is not None:
        from .parameter_accounting import (
            ParameterAccountingError,
            write_parameter_accounting_report,
        )

        accounting_path = root / f"{bundle.model.model_id}.parameter-accounting.json"
        try:
            write_parameter_accounting_report(bundle.parameter_accounting, accounting_path)
        except ParameterAccountingError as exc:
            raise ModelInspectionError(str(exc)) from exc
        outputs.append(accounting_path)
    return outputs
