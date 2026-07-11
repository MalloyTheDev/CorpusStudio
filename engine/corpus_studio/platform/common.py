"""Shared primitive definitions referenced by every platform contract.

A single source of truth for cross-cutting primitives (hashes, timestamps, semver, resolved package
locks, memory metrics, token stats) so hashing/timestamp/versioning conventions stay identical as
the engine is progressively re-implemented (Python → Rust). Grounded in the engine's existing
conventions: sha256 hex fingerprints (versions/version_registry, training/artifact_registry,
training/provenance), ISO-8601 UTC timestamps (storage/project), importlib.metadata package
versions (training/environment). Pure — no heavy imports.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "1.0.0"

# 64-char lowercase sha256 hex; the engine's dataset content fingerprint + artifact content_hash.
SHA256_PATTERN = r"^[0-9a-f]{64}$"
CONTRACT_VERSION_LITERAL = Literal["1.0.0"]

HashAlgo = Literal["sha256", "sha256-ordered-exact-v1", "blake3", "none"]
PackageSource = Literal["pypi", "wheel", "sdist", "conda", "vcs", "local", "unknown"]
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
    """A resolved dependency: distribution name → installed version + optional wheel/artifact hash.
    Grounded in environment.probe_training_runtime (importlib.metadata.version, no import)."""

    name: str = Field(min_length=1)
    # Installed version string, or null when the distribution is absent (environment stores None).
    version: str | None = None
    hash: HashRef | None = None
    source: PackageSource = "unknown"


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
