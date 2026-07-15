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

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = "1.0.0"

# 64-char lowercase sha256 hex; the engine's dataset content fingerprint + artifact content_hash.
SHA256_PATTERN = r"^[0-9a-f]{64}$"
CONTRACT_VERSION_LITERAL = Literal["1.0.0"]

HashAlgo = Literal["sha256", "sha256-ordered-exact-v1", "blake3", "none"]
PackageSource = Literal["pypi", "wheel", "sdist", "conda", "vcs", "local", "unknown"]
RecordIntegrity = Literal["verified", "failed", "missing", "unknown"]
RecordCountSemantics = Literal["all_record_rows_v2"]
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

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "properties": {"record_integrity": {"const": "verified"}},
                        "required": ["record_integrity"],
                    },
                    "then": {
                        "required": [
                            "version",
                            "hash",
                            "record_entries",
                            "record_verified_entries",
                            "record_failed_entries",
                        ],
                        "properties": {
                            "version": {"not": {"type": "null"}},
                            "hash": {
                                "type": "object",
                                "properties": {
                                    "algo": {"const": "sha256"},
                                    "value": {
                                        "type": "string",
                                        "pattern": SHA256_PATTERN,
                                    },
                                },
                                "required": ["value"],
                            },
                            "record_entries": {"type": "integer", "minimum": 1},
                            "record_verified_entries": {
                                "type": "integer",
                                "minimum": 1,
                            },
                            "record_failed_entries": {"maxItems": 0},
                        },
                    },
                },
                {
                    "if": {
                        "properties": {
                            "record_count_semantics": {"const": "all_record_rows_v2"}
                        },
                        "required": ["record_count_semantics"],
                    },
                    "then": {
                        "properties": {
                            "record_integrity": {"const": "verified"},
                            "installed_files_hash": {
                                "type": "object",
                                "properties": {
                                    "algo": {"const": "sha256"},
                                    "value": {
                                        "type": "string",
                                        "pattern": SHA256_PATTERN,
                                    },
                                },
                                "required": ["value"],
                            },
                            "installed_file_count": {
                                "type": "integer",
                                "minimum": 1,
                            },
                        },
                        "required": [
                            "record_integrity",
                            "installed_files_hash",
                            "installed_file_count",
                        ],
                    },
                },
            ]
        },
    )

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
    # remains the canonical digest of the RECORD text. Manager <=1.2 used
    # ``record_verified_entries`` for hash-bearing rows only; manager 1.3 explicitly tags the new
    # all-row meaning so preserved evidence remains parseable without silently changing semantics.
    record_integrity: RecordIntegrity = "unknown"
    record_count_semantics: RecordCountSemantics | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "Explicit all-RECORD-row count meaning. Missing means preserved legacy hash-bearing-row "
            "counts and is not admissible for new health, planning, or execution."
        ),
    )
    record_entries: int | None = Field(
        default=None,
        ge=0,
        description="Number of regular installed files named by the distribution RECORD; positive when record_integrity is verified.",
    )
    record_verified_entries: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Verified row count under record_count_semantics; manager <=1.2 counted only "
            "hash-bearing rows, while all_record_rows_v2 equals record_entries."
        ),
    )
    record_failed_entries: list[str] = Field(default_factory=list)
    # Deterministic digest of every regular file named by RECORD, including generated unhashed pyc
    # files. This complements (and does not replace) the distribution-provided RECORD digest above.
    installed_files_hash: HashRef | None = None
    installed_file_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of files sealed by installed_files_hash; equals record_entries when record_integrity is verified.",
    )
    dependencies: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _verified_record_evidence_is_complete(self) -> PackageLock:
        """Validate complete new evidence while retaining truthful legacy documents.

        RECORD legitimately leaves its own row and generated bytecode unhashed.  The manager still
        reads and hashes those files into ``installed_files_hash``; consequently a successful row is
        verified by either its RECORD digest or that installed-tree digest. Manager 1.3 labels that
        changed counter meaning explicitly. A missing label retains the manager <=1.2 meaning for
        reconstruction only; admission separately requires the v2 label.
        """

        entries = self.record_entries
        verified = self.record_verified_entries
        installed = self.installed_file_count
        if entries is not None and verified is not None and verified > entries:
            raise ValueError("record_verified_entries cannot exceed record_entries")
        if self.record_count_semantics is not None and self.record_integrity != "verified":
            raise ValueError("all-row RECORD count semantics require verified integrity")
        if self.record_integrity != "verified":
            return self
        if self.version is None:
            raise ValueError("an absent package cannot carry verified RECORD evidence")
        if entries is None or entries <= 0:
            raise ValueError("verified RECORD evidence requires positive record_entries")
        if verified is None or verified <= 0:
            raise ValueError("verified RECORD evidence requires positive verified-entry counts")
        if self.record_count_semantics == "all_record_rows_v2" and verified != entries:
            raise ValueError(
                "all-row verified RECORD evidence requires "
                "record_verified_entries == record_entries"
            )
        if self.record_failed_entries:
            raise ValueError("verified RECORD evidence cannot carry failed entries")
        for label, digest in (("RECORD", self.hash),):
            if (
                digest is None
                or digest.algo != "sha256"
                or digest.value is None
                or len(digest.value) != 64
                or any(character not in "0123456789abcdef" for character in digest.value)
            ):
                raise ValueError(f"verified {label} evidence requires an exact SHA-256 digest")
        if self.record_count_semantics == "all_record_rows_v2":
            if installed != entries:
                raise ValueError(
                    "all-row verified RECORD evidence requires "
                    "installed_file_count == record_entries"
                )
            digest = self.installed_files_hash
            if (
                digest is None
                or digest.algo != "sha256"
                or digest.value is None
                or len(digest.value) != 64
                or any(character not in "0123456789abcdef" for character in digest.value)
            ):
                raise ValueError(
                    "verified installed files evidence requires an exact SHA-256 digest"
                )
        elif (self.installed_files_hash is None) != (installed is None):
            raise ValueError(
                "legacy installed-file evidence must provide both its digest and count"
            )
        elif installed is not None:
            if installed != entries:
                raise ValueError(
                    "legacy installed_file_count must equal record_entries when present"
                )
            digest = self.installed_files_hash
            if (
                digest is None
                or digest.algo != "sha256"
                or digest.value is None
                or len(digest.value) != 64
                or any(character not in "0123456789abcdef" for character in digest.value)
            ):
                raise ValueError(
                    "legacy installed files evidence requires an exact SHA-256 digest"
                )
        return self

    def has_complete_record_count_evidence(self) -> bool:
        """Whether this package carries the manager-1.3 all-row admission meaning."""

        return (
            self.record_integrity == "verified"
            and self.record_count_semantics == "all_record_rows_v2"
            and self.record_entries is not None
            and self.record_entries > 0
            and self.record_verified_entries == self.record_entries
            and self.installed_file_count == self.record_entries
            and not self.record_failed_entries
            and self.version is not None
            and self.hash is not None
            and self.hash.algo == "sha256"
            and self.hash.value is not None
            and len(self.hash.value) == 64
            and all(character in "0123456789abcdef" for character in self.hash.value)
            and self.installed_files_hash is not None
            and self.installed_files_hash.algo == "sha256"
            and self.installed_files_hash.value is not None
            and len(self.installed_files_hash.value) == 64
            and all(
                character in "0123456789abcdef"
                for character in self.installed_files_hash.value
            )
        )


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
