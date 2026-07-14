"""Shared primitive definitions referenced by every platform contract.

A single source of truth for cross-cutting primitives (hashes, timestamps, semver, resolved package
locks, memory metrics, token stats) so hashing/timestamp/versioning conventions stay identical as
the engine is progressively re-implemented (Python → Rust). Grounded in the engine's existing
conventions: sha256 hex fingerprints (versions/version_registry, training/artifact_registry,
training/provenance), ISO-8601 UTC timestamps (storage/project), importlib.metadata package
versions (training/environment). Pure — no heavy imports.
"""

from __future__ import annotations

import secrets
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "1.0.0"

# 64-char lowercase sha256 hex; the engine's dataset content fingerprint + artifact content_hash.
SHA256_PATTERN = r"^[0-9a-f]{64}$"
CONTRACT_VERSION_LITERAL = Literal["1.0.0"]

HashAlgo = Literal["sha256", "sha256-ordered-exact-v1", "blake3", "none"]
PackageSource = Literal["pypi", "wheel", "sdist", "conda", "vcs", "local", "unknown"]
RecordIntegrity = Literal["verified", "failed", "missing", "unknown"]
LicenseSource = Literal["declared", "model_card", "dataset_card", "user_asserted", "unknown"]


class ContractModel(BaseModel):
    """Base for every contract type: reject unknown fields so a stale producer/consumer is caught
    rather than silently dropping data (matches the schemas' ``additionalProperties: false``)."""

    model_config = ConfigDict(extra="forbid")


class HashRef(ContractModel):
    """An algorithm-tagged digest. The engine emits sha256 today; the algo tag makes a future
    migration additive (cf. versions/version_registry.FINGERPRINT_ALGO)."""

    algo: HashAlgo = "sha256"
    # Digest value, or null when the target was absent/unreadable at capture (an affirmative claim).
    value: str | None = None


class Ref(ContractModel):
    """A stable reference to another contract instance by id, optionally pinned to a content hash so
    the reference cannot silently re-point."""

    id: str = Field(min_length=1)
    hash: HashRef | None = None


class PackageLock(ContractModel):
    """A resolved dependency and its install provenance.

    ``hash`` seals the installed distribution's RECORD metadata when that evidence is available; it
    is not mislabelled as the original wheel hash. ``direct_url`` and ``artifact`` preserve the
    stronger source identity pip exposes for direct/VCS/local installs. ``dependencies`` is the
    installed metadata dependency graph, not a second resolver.
    """

    name: str = Field(min_length=1)
    # PEP 503-normalized name. Empty only for legacy records written before source-evidence v2.
    normalized_name: str = ""
    # Installed version string, or null when the distribution is absent (environment stores None).
    version: str | None = None
    hash: HashRef | None = None
    source: PackageSource = "unknown"
    source_index_url: str | None = None
    direct_url: str | None = None
    artifact: str | None = None
    artifact_hash: HashRef | None = None
    installer: str | None = None
    requested: bool | None = None
    direct: bool | None = None
    editable: bool | None = None
    vcs_repository: str | None = None
    vcs_commit: str | None = None
    source_evidence_reason: str | None = None
    # RECORD metadata identity and installed-file verification are distinct claims. ``hash`` above
    # remains the canonical digest of the RECORD text; these fields say whether every hash-bearing
    # entry in that RECORD still matches the installed bytes.
    record_integrity: RecordIntegrity = "unknown"
    record_entries: int | None = Field(default=None, ge=0)
    record_verified_entries: int | None = Field(default=None, ge=0)
    record_failed_entries: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)


class License(ContractModel):
    """License metadata for a dataset, base model, or produced artifact. The engine reminds users
    the BASE model's license governs a produced adapter (training/model_card)."""

    spdx_id: str | None = None
    name: str | None = None
    url: str | None = None
    # Whether outputs may be redistributed; null = unverified.
    redistributable: bool | None = None
    source: LicenseSource = "unknown"


class MemoryMetrics(ContractModel):
    """The full memory-signature block sampled during a run. Distinguishes PyTorch's allocator view,
    raw CUDA device memory, and OS-level residency (``dedicated`` vs ``shared`` GPU memory) so a
    Windows/WDDM spill to shared memory is VISIBLE rather than hidden inside 'used VRAM'. Grounded in
    gpu_probe.GpuMemory + the estimators note that torch.max_memory_allocated counts the WDDM spill.
    """

    torch_allocated_bytes: int | None = Field(default=None, ge=0)
    torch_reserved_bytes: int | None = Field(default=None, ge=0)
    torch_peak_allocated_bytes: int | None = Field(default=None, ge=0)
    torch_peak_reserved_bytes: int | None = Field(default=None, ge=0)
    cuda_device_used_bytes: int | None = Field(default=None, ge=0)
    cuda_device_free_bytes: int | None = Field(default=None, ge=0)
    # OS-reported dedicated GPU residency (Windows WDDM 'Dedicated', Linux resident VRAM).
    dedicated_gpu_bytes: int | None = Field(default=None, ge=0)
    # OS-reported shared (system-RAM-backed) GPU memory — the WDDM spill lane. Non-zero during
    # training is an accidental-spill signal.
    shared_gpu_bytes: int | None = Field(default=None, ge=0)
    system_ram_used_bytes: int | None = Field(default=None, ge=0)
    process_rss_bytes: int | None = Field(default=None, ge=0)


class TokenStats(ContractModel):
    """Token accounting for a dataset/run. Extends estimators.TokenBudgetEstimate with the
    supervised/prompt/completion/padding breakdown the platform requires. 'natural' = tokens in raw
    content; 'collated' = after chat-template rendering + packing; 'supervised' = tokens the loss is
    computed over (completion tokens under a completion-only mask)."""

    method: str = "heuristic"
    exact: bool = False
    example_count: int = Field(ge=0)
    sequence_len: int = Field(default=0, ge=0)
    natural_tokens: int | None = Field(default=None, ge=0)
    collated_tokens: int | None = Field(default=None, ge=0)
    # Tokens the loss is actually computed over — drives the RunPlan gradient-accumulation target.
    supervised_tokens: int | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    padding_tokens: int | None = Field(default=None, ge=0)
    tokens_per_epoch: int | None = Field(default=None, ge=0)
    mean_tokens_per_example: float | None = Field(default=None, ge=0)
    max_tokens_in_example: int | None = Field(default=None, ge=0)
    examples_over_sequence_len: int | None = Field(default=None, ge=0)
    # True asserts NO row exceeds sequence_len: a no-content-dropped guarantee.
    no_truncation: bool = False


# A free-form, JSON-serializable mapping used where a contract folds in an opaque snapshot/params
# block (e.g. RunPlan.training_config_snapshot, RunEvent.payload).
JsonObject = dict[str, Any]


def new_uuid7_id(prefix: str, *, timestamp_ms: int | None = None) -> str:
    """Return a time-sortable, collision-resistant runtime identity.

    Python 3.11/3.12 do not expose :func:`uuid.uuid7`, so construct the RFC 9562 bit layout directly:
    a 48-bit Unix-millisecond timestamp, version 7, the RFC variant, and 74 cryptographic random bits.
    The optional timestamp exists only for deterministic format tests; randomness is never injectable.
    """

    if not prefix or not prefix.replace("_", "-").replace("-", "").isalnum():
        raise ValueError("runtime identity prefix must contain only letters, digits, '-' or '_'")
    millis = int(time.time() * 1000) if timestamp_ms is None else timestamp_ms
    if not 0 <= millis < 1 << 48:
        raise ValueError("UUIDv7 timestamp must fit in 48 bits")
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)
    value = (millis << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    return f"{prefix}-{uuid.UUID(int=value)}"
