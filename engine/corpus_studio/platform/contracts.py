"""The CorpusStudio platform contracts — the language-neutral boundary between the (Python → Rust)
platform core, the Python AI worker(s), and the UI shell.

Each root contract is a pydantic model carrying a ``contract_version`` and grounded in an existing
engine model (see each class docstring). These models are the canonical source of truth; the
language-neutral JSON Schemas the Rust core / Avalonia / Tauri consume are generated FROM them by
:mod:`corpus_studio.platform.schema_export`. Pure — importing this pulls no torch.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .common import (
    CONTRACT_VERSION_LITERAL,
    SHA256_PATTERN,
    ContractModel,
    HashRef,
    JsonObject,
    License,
    MemoryMetrics,
    PackageLock,
    Ref,
    TokenStats,
)
from .enums import (
    AdapterMethod,
    AllocatorPolicy,
    AttentionImpl,
    CheckpointImpl,
    CompatibilityStatus,
    CompileMode,
    CountHandling,
    DependencyLayer,
    DescriptorFileRole,
    DeviceKind,
    EvidenceKind,
    EnvironmentState,
    ExportFormat,
    FailureTaxonomy,
    FitClass,
    LossImpl,
    MemoryResidencyModel,
    ModelAttentionType,
    ModelExecutionKind,
    ModelFormat,
    ModelSourceKind,
    ModelTaskClass,
    OffloadStrategy,
    OperatingSystem,
    Optimizer,
    ObjectiveArtifactKind,
    ObjectiveCompatibilityStatus,
    ObjectiveDatasetAvailability,
    ObjectiveExecutionKind,
    ObjectiveExposureTracking,
    ObjectiveKind,
    ObjectiveLabelKind,
    ObjectiveLossComponentKind,
    ObjectiveLossMaskKind,
    ObjectiveOptimizerClock,
    ObjectiveResumeMode,
    ObjectiveSelectionMode,
    ObjectiveUpdateScope,
    ObjectiveVerificationStatus,
    ParameterCountKind,
    PositionalEncoding,
    PrecisionMode,
    QuantizationMode,
    RecipeVerification,
    StageMarker,
    StorageInterface,
    StorageRole,
    StorageSuitability,
    TaskType,
    TokenizerFormat,
    TrainerTarget,
    VerificationOutcome,
)

_ID = r"^[A-Za-z0-9._-]+$"
_RELATIVE_DESCRIPTOR_PATH = (
    r"^(?:[^/\\:.][^/\\:]*|\.[^./\\:][^/\\:]*)"
    r"(?:/(?:[^/\\:.][^/\\:]*|\.[^./\\:][^/\\:]*))*$"
)
ObjectiveCapability = Annotated[str, Field(pattern=_ID)]


def _validate_descriptor_path(value: str) -> str:
    """Require a portable relative POSIX path with no traversal or drive prefix."""

    if "\\" in value:
        raise ValueError("descriptor paths must use POSIX separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or value != path.as_posix()
        or any(part in {".", ".."} or ":" in part for part in path.parts)
    ):
        raise ValueError("descriptor paths must be safe relative paths")
    return value


# --------------------------------------------------------------------------------------------------
# FitClassification / FailureRecord — the machine-actionable spill-vs-OOM vocabulary (both NEW).
# --------------------------------------------------------------------------------------------------
class FitClassification(ContractModel):
    """The planner/calibrator verdict on whether a resolved RunPlan fits the target environment, and
    HOW: a native fit, a deliberately-offloaded fit, or an ACCIDENTAL spill (the silent WDDM/unified
    paging that looks frozen but crawls at 10-25x). NEW — the engine emits only a coarse warn/pass
    VRAM band (preflight.gpu_memory, _VRAM_SAFETY_MARGIN_GB)."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    classification: FitClass
    estimated_peak_bytes: int | None = Field(default=None, ge=0)
    device_capacity_bytes: int | None = Field(default=None, ge=0)
    # capacity - estimated_peak. Negative predicts a spill/OOM (engine keeps a 1.5 GB margin).
    headroom_bytes: int | None = None
    attention_path: AttentionImpl | None = None
    rationale: str = ""


class FailureRecord(ContractModel):
    """A structured, classified terminal outcome for a run, capability probe, or export. The
    taxonomy turns 'it died' into an actionable category — a real OOM vs a KERNEL_STALL (the sm_120
    fused-attention deadlock) vs an ACCIDENTAL_SPILL vs a CONTROLLED_OFFLOAD. NEW."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    run_id: str | None = None
    taxonomy: FailureTaxonomy
    stage: StageMarker | None = None
    exit_code: int | None = None
    signal: str | None = None
    message: str
    detail: str | None = None
    exception_type: str | None = None
    detected_at: str | None = None
    fit_at_failure: FitClassification | None = None
    # Non-zero shared_gpu_bytes here is the ACCIDENTAL_SPILL fingerprint.
    memory_at_failure: MemoryMetrics | None = None
    remediation: str | None = None
    # True when set by crash reconciliation (a 'running' record whose pid was dead → INTERRUPTED).
    reconciled: bool = False


# --------------------------------------------------------------------------------------------------
# ProjectManifest — formalizes storage/project.DatasetProject.
# --------------------------------------------------------------------------------------------------
class SplitSettings(ContractModel):
    """Verbatim from storage/project.SplitSettings. Seed pins deterministic splitting."""

    train_ratio: float = Field(default=0.9, ge=0, le=1)
    validation_ratio: float = Field(default=0.05, ge=0, le=1)
    seed: int = Field(default=42, ge=0)


class ProjectRegistries(ContractModel):
    """Project-relative directories the engine already uses for durable records. Pointers only;
    contents are resolved live."""

    dataset_versions_dir: str = "dataset_versions"
    training_runs_dir: str = "training_runs"
    model_artifacts_dir: str = "model_artifacts"
    gate_reports_dir: str = "gate_reports"
    gate_thresholds_file: str = "gate_thresholds.json"


class ProjectManifest(ContractModel):
    """The top-level descriptor of a project/workspace. Formalizes storage/project.DatasetProject
    (project.json) + SplitSettings, promoting it to the workspace-primary manifest a UI shell opens;
    everything else resolves live from the referenced registries."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = Field(min_length=1)
    schema_id: str = Field(pattern=r"^[a-z0-9_]+$")
    created_at: str | None = None
    updated_at: str | None = None
    split_settings: SplitSettings = Field(default_factory=SplitSettings)
    dataset_path: str = "examples.jsonl"
    registries: ProjectRegistries = Field(default_factory=ProjectRegistries)
    labels: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------------------
# DatasetManifest — identity + lineage. Formalizes versions/version_registry + adds lineage.
# --------------------------------------------------------------------------------------------------
class DatasetSourceArtifact(ContractModel):
    kind: Literal["dataset_version", "imported_file", "hf_dataset", "generated", "external"] = (
        "imported_file"
    )
    ref: str
    hash: HashRef | None = None
    license: License | None = None


class DatasetTransformStep(ContractModel):
    """One ordered transform (import → clean → redact → split → collate). Grounded in the engine's
    cleaning/redaction sidecar manifests (cli.py)."""

    step: str
    tool: str
    tool_version: str | None = None
    params: JsonObject = Field(default_factory=dict)
    manifest_ref: str | None = None
    input_hash: HashRef | None = None
    output_hash: HashRef | None = None


class DatasetGeneration(ContractModel):
    """Set when rows were synthesized/distilled from a teacher model — the reproducible recipe."""

    teacher_model: str
    teacher_model_version: str | None = None
    prompt_version: str | None = None
    prompt_hash: HashRef | None = None
    random_seed: int | None = None
    decoding: JsonObject = Field(default_factory=dict)


class DatasetLineage(ContractModel):
    source_artifacts: list[DatasetSourceArtifact] = Field(default_factory=list)
    transformation_pipeline: list[DatasetTransformStep] = Field(default_factory=list)
    tool_versions: dict[str, str | None] = Field(default_factory=dict)
    generation: DatasetGeneration | None = None
    random_seed: int | None = None


class DatasetRowHashes(ContractModel):
    """Per-row content addressing. Reuses capture_dataset's sha256(exact_row_signature) row ids."""

    algo: str = "sha256-exact-row-signature-v1"
    rows_stored: bool = False
    stored_row_count: int = Field(default=0, ge=0)
    row_manifest_ref: str | None = None
    ids: list[str] | None = None


class DatasetLinks(ContractModel):
    """Live cross-references resolved by the version card. Presence/integrity resolved live."""

    source_run_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    eval_report_ref: str | None = None
    gate_report_ref: str | None = None


class DatasetManifest(ContractModel):
    """Full identity + LINEAGE of a dataset version. Content identity is verbatim from
    versions/version_registry.DatasetVersionRecord; row hashes reuse the per-row sha256 of
    exact_row_signature; token stats extend estimators; the transformation-pipeline lineage is NEW.
    """

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    version_id: str = Field(pattern=_ID)
    project_id: str = ""
    schema_id: str = ""
    label: str = ""
    trigger: str = ""
    created_at: str | None = None
    updated_at: str | None = None
    row_count: int = Field(ge=0)
    # Order-sensitive sha256 over the ordered per-row signatures; null when absent at capture.
    content_fingerprint: str | None = Field(default=..., pattern=SHA256_PATTERN)
    fingerprint_algo: str = "sha256-ordered-exact-v1"
    row_signature_kind: str = "exact"
    output_artifact_hash: HashRef | None = None
    token_stats: TokenStats | None = None
    row_hashes: DatasetRowHashes = Field(default_factory=DatasetRowHashes)
    lineage: DatasetLineage = Field(default_factory=DatasetLineage)
    license: License | None = None
    links: DatasetLinks = Field(default_factory=DatasetLinks)
    notes: str = ""


# --------------------------------------------------------------------------------------------------
# ModelDescriptor / TokenizerDescriptor - static, dependency-light model identity and compatibility.
# The inspector never imports model code or a heavy ML framework. MoE execution remains future work,
# but these contracts avoid dense-only assumptions from their first version.
# --------------------------------------------------------------------------------------------------
class DescriptorSource(ContractModel):
    """Identity of the requested source and the immutable revision actually inspected.

    requested_revision is user intent. resolved_revision/resolved_commit are evidence. They are
    intentionally separate so a mutable branch name is never misreported as a pinned snapshot.
    """

    kind: ModelSourceKind
    repository: str | None = None
    requested_revision: str | None = None
    resolved_revision: str | None = None
    resolved_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{7,64}$")
    revision_pinned: bool = False
    local_path: str | None = None
    artifact_ref: Ref | None = None
    snapshot_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    evidence_source: str = "user_input"

    @model_validator(mode="after")
    def _validate_identity(self) -> DescriptorSource:
        if self.kind == ModelSourceKind.huggingface and not self.repository:
            raise ValueError("huggingface sources require repository identity")
        if self.kind == ModelSourceKind.local and not self.local_path:
            raise ValueError("local sources require local_path")
        if self.kind == ModelSourceKind.artifact and self.artifact_ref is None:
            raise ValueError("artifact sources require artifact_ref")
        if self.revision_pinned != (self.resolved_commit is not None):
            raise ValueError("revision_pinned is true only with an immutable resolved_commit")
        return self


class DescriptorFile(ContractModel):
    """One safe, portable inventory entry. sha256 is separate from hash_status so skipped hashing
    and unreadable content are never confused with a verified empty digest."""

    path: str = Field(min_length=1, pattern=_RELATIVE_DESCRIPTOR_PATH)
    role: DescriptorFileRole = DescriptorFileRole.other
    size_bytes: int = Field(ge=0)
    format: ModelFormat | None = None
    hash_status: Literal["verified", "not_requested", "unreadable", "skipped_unsafe"] = (
        "not_requested"
    )
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    serialization_risk: Literal[
        "safe", "pickle", "executable_code", "archive", "unknown"
    ] = "unknown"
    is_link: bool = False

    @field_validator("path")
    @classmethod
    def _portable_path(cls, value: str) -> str:
        return _validate_descriptor_path(value)

    @model_validator(mode="after")
    def _validate_hash_evidence(self) -> DescriptorFile:
        if self.hash_status == "verified" and self.sha256 is None:
            raise ValueError("verified files require sha256")
        if self.hash_status != "verified" and self.sha256 is not None:
            raise ValueError("sha256 may only be set when hash_status is verified")
        if self.is_link and self.hash_status != "skipped_unsafe":
            raise ValueError("links must be recorded as skipped_unsafe and never followed")
        return self


class ParameterCountHandling(ContractModel):
    tied: CountHandling = CountHandling.unknown
    shared: CountHandling = CountHandling.unknown
    replicated: CountHandling = CountHandling.unknown
    generated: CountHandling = CountHandling.unknown
    quantized: CountHandling = CountHandling.unknown
    optimizer_shadows: CountHandling = CountHandling.not_applicable
    decompressed_caches: CountHandling = CountHandling.not_applicable


class ParameterCount(ContractModel):
    """One explicitly scoped count. There is deliberately no scalar parameter_count field."""

    kind: ParameterCountKind
    value: int = Field(ge=0)
    unit: Literal["coordinates", "elements", "parameters"] = "coordinates"
    scope: str = Field(min_length=1)
    measurement_window: str = Field(min_length=1)
    source: str = Field(min_length=1)
    evidence: EvidenceKind
    handling: ParameterCountHandling = Field(default_factory=ParameterCountHandling)
    notes: str = ""


class ParameterComponent(ContractModel):
    """A representation component such as shared weights, router, experts, or an adapter.

    Stored dtype is a raw representation string, not PrecisionMode (which describes run compute and
    includes values such as tf32/mixed_bf16 that are not on-disk dtypes).
    """

    component_id: str = Field(pattern=_ID)
    scope: Literal[
        "all", "embedding", "shared", "router", "expert_group", "output_head", "adapter", "other"
    ] = "all"
    format: ModelFormat
    storage_dtype: str | None = None
    quantization: QuantizationMode | None = None
    quantization_details: JsonObject = Field(default_factory=dict)
    file_refs: list[str] = Field(default_factory=list)

    @field_validator("file_refs")
    @classmethod
    def _portable_file_refs(cls, values: list[str]) -> list[str]:
        normalized = [_validate_descriptor_path(value) for value in values]
        if normalized != sorted(set(normalized)):
            raise ValueError("parameter component file_refs must be sorted and unique")
        return normalized


class ParameterRepresentation(ContractModel):
    kind: ModelExecutionKind = ModelExecutionKind.unknown
    components: list[ParameterComponent] = Field(default_factory=list)
    counts: list[ParameterCount] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_parameter_axes(self) -> ParameterRepresentation:
        component_ids = [item.component_id for item in self.components]
        if component_ids != sorted(component_ids) or len(component_ids) != len(set(component_ids)):
            raise ValueError("parameter components must be sorted by unique component_id")
        count_keys = [
            (item.kind.value, item.scope, item.measurement_window) for item in self.counts
        ]
        if count_keys != sorted(count_keys) or len(count_keys) != len(set(count_keys)):
            raise ValueError("parameter counts must be sorted and unique by kind/scope/window")
        return self


class SemanticRouting(ContractModel):
    """The learned semantic selection policy. Physical placement is not represented here."""

    router_type: str = Field(min_length=1)
    routing_unit: Literal["token", "sequence", "layer", "request", "custom"] = "token"
    selection_policy: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    capacity_factor: float | None = Field(default=None, gt=0)
    routing_noise: str | None = None
    metadata_source: str = Field(min_length=1)


class ExpertGroup(ContractModel):
    group_id: str = Field(pattern=_ID)
    layer_indices: list[int] = Field(default_factory=list)
    expert_count: int = Field(ge=1)
    experts_per_token: int | None = Field(default=None, ge=1)
    shared_expert_count: int | None = Field(default=None, ge=0)
    heterogeneous: bool = False
    expert_identity_scheme: str | None = None
    expert_registry_ref: Ref | None = None

    @model_validator(mode="after")
    def _validate_expert_counts(self) -> ExpertGroup:
        if self.experts_per_token is not None and self.experts_per_token > self.expert_count:
            raise ValueError("experts_per_token cannot exceed expert_count")
        if self.shared_expert_count is not None and self.shared_expert_count > self.expert_count:
            raise ValueError("shared_expert_count cannot exceed expert_count")
        if self.layer_indices != sorted(set(self.layer_indices)):
            raise ValueError("layer_indices must be sorted and unique")
        return self


class ModelTopology(ContractModel):
    execution_kind: ModelExecutionKind = ModelExecutionKind.unknown
    semantic_routing: SemanticRouting | None = None
    expert_groups: list[ExpertGroup] = Field(default_factory=list)
    # Placement/prefetch/residency decisions belong to the future RunPlan physical scheduler.
    physical_scheduler_owner: Literal["run_plan"] = "run_plan"

    @model_validator(mode="after")
    def _validate_topology(self) -> ModelTopology:
        group_ids = [item.group_id for item in self.expert_groups]
        if group_ids != sorted(group_ids) or len(group_ids) != len(set(group_ids)):
            raise ValueError("expert_groups must be sorted by unique group_id")
        if self.execution_kind == ModelExecutionKind.dense and (
            self.semantic_routing is not None or self.expert_groups
        ):
            raise ValueError("dense topology cannot declare semantic routing or expert groups")
        return self


class TrustRequirement(ContractModel):
    """Static trust findings only. This descriptor can never authorize custom-code execution."""

    trust_remote_code: Literal[False] = False
    custom_code_required: bool = False
    approval_required: bool = False
    isolated_execution_required: bool = False
    custom_code_files: list[str] = Field(default_factory=list)
    detected_auto_map: JsonObject = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @field_validator("custom_code_files")
    @classmethod
    def _portable_code_files(cls, values: list[str]) -> list[str]:
        normalized = [_validate_descriptor_path(value) for value in values]
        if normalized != sorted(set(normalized)):
            raise ValueError("custom_code_files must be sorted and unique")
        return normalized

    @model_validator(mode="after")
    def _fail_closed_custom_code(self) -> TrustRequirement:
        if self.custom_code_required and not (
            self.approval_required and self.isolated_execution_required
        ):
            raise ValueError("custom code requires approval and isolated execution")
        if not self.custom_code_required and (
            self.approval_required or self.isolated_execution_required or self.custom_code_files
        ):
            raise ValueError("custom-code controls require custom_code_required")
        if self.notes != sorted(set(self.notes)):
            raise ValueError("trust notes must be sorted and unique")
        return self


class DescriptorVerification(ContractModel):
    """Independent evidence axes. Integrity never implies compatibility or hardware support."""

    metadata: VerificationOutcome = VerificationOutcome.not_checked
    integrity: VerificationOutcome = VerificationOutcome.not_checked
    license: VerificationOutcome = VerificationOutcome.not_checked
    custom_code_policy: VerificationOutcome = VerificationOutcome.not_checked
    inspected_at: str | None = None
    inspector: str | None = None
    evidence_refs: list[Ref] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_evidence_lists(self) -> DescriptorVerification:
        evidence_ids = [item.id for item in self.evidence_refs]
        if evidence_ids != sorted(set(evidence_ids)):
            raise ValueError("verification evidence_refs must be sorted and unique")
        if self.warnings != sorted(set(self.warnings)):
            raise ValueError("verification warnings must be sorted and unique")
        return self


class DimensionEvidence(ContractModel):
    value: int = Field(ge=0)
    source: str = Field(min_length=1)
    evidence: EvidenceKind


class EmbeddingVocabulary(ContractModel):
    declared_vocab_size: DimensionEvidence | None = None
    input_embedding_rows: DimensionEvidence | None = None
    output_head_rows: DimensionEvidence | None = None
    tied_embeddings: bool | None = None

    @model_validator(mode="after")
    def _validate_tied_rows(self) -> EmbeddingVocabulary:
        if (
            self.tied_embeddings is True
            and self.input_embedding_rows is not None
            and self.output_head_rows is not None
            and self.input_embedding_rows.value != self.output_head_rows.value
        ):
            raise ValueError("tied embeddings require matching input and output row counts")
        return self


class BackendCompatibilityEntry(ContractModel):
    backend_ref: Ref
    environment_ref: Ref | None = None
    status: Literal["compatible", "incompatible", "unverified"] = "unverified"
    capability_report_ref: Ref | None = None
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_support_evidence(self) -> BackendCompatibilityEntry:
        if self.reasons != sorted(set(self.reasons)):
            raise ValueError("backend compatibility reasons must be sorted and unique")
        if self.status == "compatible" and (
            self.environment_ref is None or self.capability_report_ref is None
        ):
            raise ValueError("compatible backend entries require environment and capability evidence")
        if self.status == "incompatible" and not self.reasons:
            raise ValueError("incompatible backend entries require reasons")
        return self


class CompatibilityCheck(ContractModel):
    check: str = Field(pattern=_ID)
    outcome: VerificationOutcome
    evidence: str | None = None
    message: str = ""
    remediation: str | None = None


class ModelTokenizerCompatibility(ContractModel):
    model_ref: Ref
    tokenizer_ref: Ref
    status: CompatibilityStatus
    checks: list[CompatibilityCheck] = Field(default_factory=list)
    required_embedding_rows: int | None = Field(default=None, ge=0)
    resize_input_embeddings: bool = False
    resize_output_head: bool = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_compatibility_evidence(self) -> ModelTokenizerCompatibility:
        check_ids = [item.check for item in self.checks]
        if len(check_ids) != len(set(check_ids)):
            raise ValueError("compatibility checks must have unique ids")
        if self.warnings != sorted(set(self.warnings)):
            raise ValueError("compatibility warnings must be sorted and unique")
        outcomes = {item.outcome for item in self.checks}
        if self.status == CompatibilityStatus.compatible and (
            not self.checks or outcomes != {VerificationOutcome.passed}
        ):
            raise ValueError("compatible status requires one or more passed checks and no gaps")
        if self.status == CompatibilityStatus.resize_required and not (
            self.required_embedding_rows is not None
            and (self.resize_input_embeddings or self.resize_output_head)
            and VerificationOutcome.failed in outcomes
        ):
            raise ValueError("resize_required needs failed size evidence and an explicit resize")
        if self.status == CompatibilityStatus.incompatible and VerificationOutcome.failed not in outcomes:
            raise ValueError("incompatible status requires a failed check")
        if self.status == CompatibilityStatus.unverified and VerificationOutcome.not_checked not in outcomes:
            raise ValueError("unverified status requires a not_checked evidence gap")
        return self


class ModelDescriptor(ContractModel):
    """Static identity, representation, integrity, and compatibility surface for one model snapshot.

    The descriptor is safe to build in the torch-free control plane. It does not claim that the model
    can load, train, fit a device, or execute custom code.
    """

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    model_id: str = Field(pattern=_ID)
    source: DescriptorSource
    artifact_role: Literal[
        "base", "adapter", "merged", "checkpoint", "quantized", "converted", "other", "unknown"
    ] = "unknown"
    architectures: list[str] = Field(default_factory=list)
    model_family: str | None = None
    task_classes: list[ModelTaskClass] = Field(
        default_factory=lambda: [ModelTaskClass.unknown], min_length=1
    )
    formats: list[ModelFormat] = Field(
        default_factory=lambda: [ModelFormat.unknown], min_length=1
    )
    parameters: ParameterRepresentation = Field(default_factory=ParameterRepresentation)
    topology: ModelTopology = Field(default_factory=ModelTopology)
    vocabulary: EmbeddingVocabulary = Field(default_factory=EmbeddingVocabulary)
    context_window: DimensionEvidence | None = None
    tokenizer_ref: Ref | None = None
    attention_type: ModelAttentionType = ModelAttentionType.unknown
    positional_encoding: PositionalEncoding = PositionalEncoding.unknown
    license: License | None = None
    trust: TrustRequirement = Field(default_factory=TrustRequirement)
    files: list[DescriptorFile] = Field(default_factory=list)
    inventory_complete: bool = False
    storage_size_bytes: int = Field(default=0, ge=0)
    inventory_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    backend_compatibility: list[BackendCompatibilityEntry] = Field(default_factory=list)
    verification: DescriptorVerification = Field(default_factory=DescriptorVerification)
    captured_at: str | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_model_inventory(self) -> ModelDescriptor:
        if self.architectures != sorted(set(self.architectures)):
            raise ValueError("architectures must be sorted and unique")
        task_values = [item.value for item in self.task_classes]
        if task_values != sorted(set(task_values)):
            raise ValueError("task_classes must be sorted and unique")
        format_values = [item.value for item in self.formats]
        if format_values != sorted(set(format_values)):
            raise ValueError("formats must be sorted and unique")
        paths = [item.path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("model file inventory must be sorted by unique path")
        if self.storage_size_bytes != sum(item.size_bytes for item in self.files):
            raise ValueError("storage_size_bytes must equal the recorded file sizes")
        if self.source.snapshot_sha256 is not None and (
            not self.inventory_complete
            or not self.files
            or any(item.hash_status != "verified" for item in self.files)
        ):
            raise ValueError("snapshot_sha256 requires a complete, fully hashed inventory")
        component_formats = {item.format for item in self.parameters.components}
        if not component_formats.issubset(set(self.formats)):
            raise ValueError("formats must include every parameter component format")
        backend_keys = [
            (item.backend_ref.id, item.environment_ref.id if item.environment_ref else "")
            for item in self.backend_compatibility
        ]
        if backend_keys != sorted(set(backend_keys)):
            raise ValueError("backend_compatibility must be sorted and unique")
        inventory_paths = set(paths)
        component_refs = {
            file_ref
            for component in self.parameters.components
            for file_ref in component.file_refs
        }
        if not component_refs.issubset(inventory_paths):
            raise ValueError("parameter component file_refs must exist in the model inventory")
        if not set(self.trust.custom_code_files).issubset(inventory_paths):
            raise ValueError("custom_code_files must exist in the model inventory")
        if self.notes != sorted(set(self.notes)):
            raise ValueError("model notes must be sorted and unique")
        return self


class SpecialToken(ContractModel):
    role: str = Field(min_length=1)
    content: str
    token_id: int | None = Field(default=None, ge=0)
    added: bool = False


class TokenizerDescriptor(ContractModel):
    """Static tokenizer identity and structure. Exact encode/decode behavior needs a later functional
    probe in an isolated capability environment; inspection alone does not claim it."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    tokenizer_id: str = Field(pattern=_ID)
    source: DescriptorSource
    format: TokenizerFormat = TokenizerFormat.unknown
    implementation_class: str | None = None
    base_vocabulary_size: int | None = Field(default=None, ge=0)
    added_token_count: int | None = Field(default=None, ge=0)
    effective_vocabulary_size: int | None = Field(default=None, ge=0)
    max_token_id: int | None = Field(default=None, ge=0)
    model_max_length: DimensionEvidence | None = None
    special_tokens: list[SpecialToken] = Field(default_factory=list)
    chat_template: str | list[JsonObject] | None = None
    chat_template_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    normalization: JsonObject | None = None
    pre_tokenization: JsonObject | None = None
    model_compatibility: list[ModelTokenizerCompatibility] = Field(default_factory=list)
    trust: TrustRequirement = Field(default_factory=TrustRequirement)
    files: list[DescriptorFile] = Field(default_factory=list)
    inventory_complete: bool = False
    storage_size_bytes: int = Field(default=0, ge=0)
    inventory_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    verification: DescriptorVerification = Field(default_factory=DescriptorVerification)
    captured_at: str | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_tokenizer_inventory(self) -> TokenizerDescriptor:
        paths = [item.path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("tokenizer file inventory must be sorted by unique path")
        if self.storage_size_bytes != sum(item.size_bytes for item in self.files):
            raise ValueError("storage_size_bytes must equal the recorded file sizes")
        if self.source.snapshot_sha256 is not None and (
            not self.inventory_complete
            or not self.files
            or any(item.hash_status != "verified" for item in self.files)
        ):
            raise ValueError("snapshot_sha256 requires a complete, fully hashed inventory")
        token_keys = [(item.role, item.content, item.token_id) for item in self.special_tokens]
        if len(token_keys) != len(set(token_keys)) or token_keys != sorted(
            token_keys,
            key=lambda item: (item[0], item[1], item[2] if item[2] is not None else -1),
        ):
            raise ValueError("special_tokens must be deterministically sorted")
        compatibility_keys = [
            (item.model_ref.id, item.tokenizer_ref.id) for item in self.model_compatibility
        ]
        if compatibility_keys != sorted(set(compatibility_keys)):
            raise ValueError("model_compatibility must be sorted and unique")
        if not set(self.trust.custom_code_files).issubset(set(paths)):
            raise ValueError("custom_code_files must exist in the tokenizer inventory")
        if (
            self.base_vocabulary_size is not None
            and self.effective_vocabulary_size is not None
            and self.effective_vocabulary_size < self.base_vocabulary_size
        ):
            raise ValueError("effective_vocabulary_size cannot be below base_vocabulary_size")
        if self.effective_vocabulary_size is not None and self.max_token_id is not None:
            if self.max_token_id >= self.effective_vocabulary_size:
                raise ValueError("max_token_id must be below effective_vocabulary_size")
        if (self.chat_template is None) != (self.chat_template_sha256 is None):
            raise ValueError("chat_template and chat_template_sha256 must be set together")
        if self.notes != sorted(set(self.notes)):
            raise ValueError("tokenizer notes must be sorted and unique")
        return self


# --------------------------------------------------------------------------------------------------
# TrainingObjective / ObjectiveCompatibilityReport - objective semantics, independent of a backend.
# --------------------------------------------------------------------------------------------------
class ObjectiveDatasetField(ContractModel):
    name: str = Field(pattern=_ID)
    field_type: str | None = None
    semantic_role: str = Field(min_length=1)


class ObjectiveDatasetVariant(ContractModel):
    """One accepted dataset shape. ``planned`` means the shape is specified but CorpusStudio does
    not yet ship a matching built-in schema; registry presence never turns that into support."""

    schema_id: str = Field(pattern=_ID)
    availability: ObjectiveDatasetAvailability
    schema_version: str | None = None
    dataset_format: str | None = None
    required_fields: list[ObjectiveDatasetField] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_fields(self) -> ObjectiveDatasetVariant:
        names = [item.name for item in self.required_fields]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("objective dataset fields must be sorted by unique name")
        return self


class ObjectiveDatasetInput(ContractModel):
    role: Literal["train", "validation", "teacher", "preference", "evaluation"]
    variants: list[ObjectiveDatasetVariant] = Field(min_length=1)
    row_validation_required: bool = True
    split_isolation_required: bool = True
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_variants(self) -> ObjectiveDatasetInput:
        keys = [(item.schema_id, item.schema_version or "") for item in self.variants]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("objective dataset variants must be sorted and unique")
        if self.notes != sorted(set(self.notes)):
            raise ValueError("objective dataset notes must be sorted and unique")
        return self


class ObjectiveLabelConstruction(ContractModel):
    label_id: str = Field(pattern=_ID)
    kind: ObjectiveLabelKind
    source_fields: list[str] = Field(default_factory=list)
    construction: str = Field(min_length=1)
    ignore_index: int | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_label(self) -> ObjectiveLabelConstruction:
        if self.source_fields != sorted(set(self.source_fields)):
            raise ValueError("label source_fields must be sorted and unique")
        if self.notes != sorted(set(self.notes)):
            raise ValueError("label notes must be sorted and unique")
        return self


class ObjectiveLossMask(ContractModel):
    mask_id: str = Field(pattern=_ID)
    kind: ObjectiveLossMaskKind
    source_fields: list[str] = Field(default_factory=list)
    include_padding: bool = False
    include_special_tokens: bool = False
    empty_mask_action: Literal["reject", "skip", "zero_loss", "not_applicable"] = "reject"
    construction: str = Field(min_length=1)

    @field_validator("source_fields")
    @classmethod
    def _sorted_mask_fields(cls, values: list[str]) -> list[str]:
        if values != sorted(set(values)):
            raise ValueError("loss-mask source_fields must be sorted and unique")
        return values


class ObjectiveLossComponent(ContractModel):
    component_id: str = Field(pattern=_ID)
    kind: ObjectiveLossComponentKind
    construction: str = Field(min_length=1)
    label_ref: str | None = Field(default=None, pattern=_ID)
    mask_ref: str | None = Field(default=None, pattern=_ID)
    default_weight: float | None = Field(default=1.0, ge=0)
    reduction: Literal["mean", "sum", "token_mean", "pair_mean", "none"] = "mean"


class ObjectiveModelRequirement(ContractModel):
    task_classes: list[ModelTaskClass] = Field(min_length=1)
    execution_kinds: list[ModelExecutionKind] = Field(min_length=1)
    requires_tokenizer: bool = True
    requires_output_head: bool = True
    requires_reference_model: bool = False
    requires_reward_head: bool = False
    requires_multimodal_projector: bool = False
    custom_code_policy: Literal["forbid", "isolated_approval", "backend_defined"] = (
        "isolated_approval"
    )

    @model_validator(mode="after")
    def _validate_model_requirements(self) -> ObjectiveModelRequirement:
        task_values = [item.value for item in self.task_classes]
        if task_values != sorted(set(task_values)):
            raise ValueError("objective model task_classes must be sorted and unique")
        execution_values = [item.value for item in self.execution_kinds]
        if execution_values != sorted(set(execution_values)):
            raise ValueError("objective model execution_kinds must be sorted and unique")
        if ModelExecutionKind.unknown in self.execution_kinds:
            raise ValueError("unknown is an evidence gap, not an accepted execution kind")
        return self


class ObjectiveUpdatePolicy(ContractModel):
    """What may change, separate from where those components are physically resident.

    This is the MoE-safe semantic update policy. Placement, prefetch, and device scheduling remain
    future RunPlan responsibilities.
    """

    scopes: list[ObjectiveUpdateScope] = Field(min_length=1)
    selection_mode: ObjectiveSelectionMode
    stable_expert_identity: Literal["not_required", "when_expert_scoped", "required"]
    exposure_tracking: ObjectiveExposureTracking
    optimizer_clock: ObjectiveOptimizerClock
    update_window_definition: str = Field(min_length=1)
    starvation_gate_required_when_expert_scoped: bool = False
    routing_collapse_gate_required_when_routed: bool = False
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_sparse_update_policy(self) -> ObjectiveUpdatePolicy:
        values = [item.value for item in self.scopes]
        if values != sorted(set(values)):
            raise ValueError("objective update scopes must be sorted and unique")
        scopes = set(self.scopes)
        expert_scopes = {ObjectiveUpdateScope.selected_experts, ObjectiveUpdateScope.all_experts}
        if self.selection_mode == ObjectiveSelectionMode.none:
            if scopes != {ObjectiveUpdateScope.none}:
                raise ValueError("selection_mode none requires the none update scope")
        else:
            if ObjectiveUpdateScope.none in scopes:
                raise ValueError("none cannot be combined with trainable update scopes")
            if self.selection_mode == ObjectiveSelectionMode.adapter_only and scopes != {
                ObjectiveUpdateScope.adapters
            }:
                raise ValueError("adapter_only requires exactly the adapters scope")
            if self.selection_mode == ObjectiveSelectionMode.router_only and scopes != {
                ObjectiveUpdateScope.router
            }:
                raise ValueError("router_only requires exactly the router scope")
            if self.selection_mode in {
                ObjectiveSelectionMode.selected_experts,
                ObjectiveSelectionMode.routed_experts,
            } and scopes != {ObjectiveUpdateScope.selected_experts}:
                raise ValueError("expert selection requires exactly the selected_experts scope")
            if self.selection_mode == ObjectiveSelectionMode.task_head_only and (
                ObjectiveUpdateScope.task_head not in scopes
                or not scopes.issubset(
                    {ObjectiveUpdateScope.task_head, ObjectiveUpdateScope.projector}
                )
            ):
                raise ValueError("task_head_only allows only task_head and optional projector scopes")
            if (
                self.selection_mode == ObjectiveSelectionMode.all
                and ObjectiveUpdateScope.all_parameters in scopes
                and scopes != {ObjectiveUpdateScope.all_parameters}
            ):
                raise ValueError("all_parameters cannot be combined with narrower update scopes")
        if scopes.intersection(expert_scopes):
            if self.stable_expert_identity == "not_required":
                raise ValueError("expert-scoped updates require stable expert identity")
            if self.exposure_tracking not in {
                ObjectiveExposureTracking.per_expert,
                ObjectiveExposureTracking.router_and_expert,
            }:
                raise ValueError("expert-scoped updates require per-expert exposure tracking")
            if self.optimizer_clock != ObjectiveOptimizerClock.per_expert:
                raise ValueError("expert-scoped updates require per-expert optimizer clocks")
        if self.notes != sorted(set(self.notes)):
            raise ValueError("objective update notes must be sorted and unique")
        return self


class ObjectiveBackendRequirement(ContractModel):
    """Semantic backend requirements. Backend IDs never belong in an objective definition.

    A backend may match any listed loss, adaptation, and quantization mode; every listed objective
    capability token is required. Static matches remain declarations until a capability report proves
    the same objective tokens on the selected environment.
    """

    task_type: TaskType | None = None
    loss_impls: list[LossImpl] = Field(default_factory=list)
    adaptation_methods: list[AdapterMethod] = Field(default_factory=list)
    quantization_modes: list[QuantizationMode] = Field(default_factory=list)
    objective_capabilities: list[ObjectiveCapability] = Field(default_factory=list)
    functional_probe_required: bool = True
    hardware_verification_required: bool = True

    @model_validator(mode="after")
    def _validate_backend_requirements(self) -> ObjectiveBackendRequirement:
        for field_name in ("loss_impls", "adaptation_methods", "quantization_modes"):
            items = getattr(self, field_name)
            values = [item.value for item in items]
            if values != sorted(set(values)):
                raise ValueError(f"{field_name} must be sorted and unique")
        if self.objective_capabilities != sorted(set(self.objective_capabilities)):
            raise ValueError("objective_capabilities must be sorted and unique")
        return self


class ObjectiveArtifactExpectation(ContractModel):
    kind: ObjectiveArtifactKind
    required: bool = True
    component_scoped: bool = False
    condition: str | None = None
    description: str = Field(min_length=1)


class ObjectiveResumeSemantics(ContractModel):
    mode: ObjectiveResumeMode
    required_state: list[str] = Field(default_factory=list)
    component_scoped_resume: bool = False
    non_exact_resume_creates_lineage: bool = True
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_resume(self) -> ObjectiveResumeSemantics:
        if self.required_state != sorted(set(self.required_state)):
            raise ValueError("resume required_state must be sorted and unique")
        if self.notes != sorted(set(self.notes)):
            raise ValueError("resume notes must be sorted and unique")
        if self.mode == ObjectiveResumeMode.not_applicable and self.required_state:
            raise ValueError("not-applicable resume cannot require state")
        return self


class ObjectiveEvaluationRequirements(ContractModel):
    before_run: bool = False
    during_run: bool = False
    after_run: bool = True
    holdout_required: bool = True
    gate_required: bool = True
    metrics: list[str] = Field(default_factory=list)
    expert_system_metrics: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_evaluation(self) -> ObjectiveEvaluationRequirements:
        if self.metrics != sorted(set(self.metrics)):
            raise ValueError("evaluation metrics must be sorted and unique")
        if self.expert_system_metrics != sorted(set(self.expert_system_metrics)):
            raise ValueError("expert-system metrics must be sorted and unique")
        return self


class ObjectiveHardwareImplications(ContractModel):
    compute: Literal["none", "low", "medium", "high", "unknown"] = "unknown"
    device_memory: Literal["none", "low", "medium", "high", "unknown"] = "unknown"
    host_memory: Literal["none", "low", "medium", "high", "unknown"] = "unknown"
    storage_io: Literal["none", "low", "medium", "high", "unknown"] = "unknown"
    communication: Literal["none", "low", "medium", "high", "unknown"] = "unknown"
    # Only a measured run may prove fit; an objective definition never makes that claim.
    fit_claim: Literal["none"] = "none"
    implications: list[str] = Field(default_factory=list)

    @field_validator("implications")
    @classmethod
    def _sorted_implications(cls, values: list[str]) -> list[str]:
        if values != sorted(set(values)):
            raise ValueError("hardware implications must be sorted and unique")
        return values


class ObjectiveVerification(ContractModel):
    definition: ObjectiveVerificationStatus = ObjectiveVerificationStatus.declared
    implementation: ObjectiveVerificationStatus = ObjectiveVerificationStatus.not_verified
    hardware: ObjectiveVerificationStatus = ObjectiveVerificationStatus.not_verified
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_verification(self) -> ObjectiveVerification:
        if self.evidence_refs != sorted(set(self.evidence_refs)):
            raise ValueError("objective verification evidence_refs must be sorted and unique")
        proven = {
            ObjectiveVerificationStatus.functional_verified,
            ObjectiveVerificationStatus.hardware_verified,
        }
        if (self.implementation in proven or self.hardware in proven) and not self.evidence_refs:
            raise ValueError("verified objective claims require evidence_refs")
        definition_proven = {
            ObjectiveVerificationStatus.contract_validated,
            ObjectiveVerificationStatus.functional_verified,
            ObjectiveVerificationStatus.hardware_verified,
        }
        implementation_proven = {
            ObjectiveVerificationStatus.functional_verified,
            ObjectiveVerificationStatus.hardware_verified,
        }
        if self.implementation in implementation_proven and self.definition not in definition_proven:
            raise ValueError("functional implementation proof requires a validated definition")
        if (
            self.hardware == ObjectiveVerificationStatus.hardware_verified
            and self.implementation not in implementation_proven
        ):
            raise ValueError("hardware proof requires functionally verified implementation")
        return self


class TrainingObjective(ContractModel):
    """A versioned semantic objective, deliberately independent from trainer implementation.

    The objective hash seals the canonical definition. Registry helpers verify it; deserializing an
    arbitrary contract alone does not imply that its hash or execution claims are trusted.
    """

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    objective_id: str = Field(pattern=_ID)
    objective_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    objective_hash: str = Field(pattern=SHA256_PATTERN)
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    kind: ObjectiveKind
    execution_kind: ObjectiveExecutionKind
    coarse_task_type: TaskType | None = None
    dataset_inputs: list[ObjectiveDatasetInput] = Field(default_factory=list)
    labels: list[ObjectiveLabelConstruction] = Field(default_factory=list)
    loss_masks: list[ObjectiveLossMask] = Field(default_factory=list)
    loss_components: list[ObjectiveLossComponent] = Field(default_factory=list)
    model_requirement: ObjectiveModelRequirement
    adaptation_methods: list[AdapterMethod] = Field(default_factory=list)
    update_policy: ObjectiveUpdatePolicy
    backend_requirement: ObjectiveBackendRequirement
    expected_artifacts: list[ObjectiveArtifactExpectation] = Field(default_factory=list)
    resume: ObjectiveResumeSemantics
    evaluation: ObjectiveEvaluationRequirements
    hardware: ObjectiveHardwareImplications
    limitations: list[str] = Field(default_factory=list)
    verification: ObjectiveVerification = Field(default_factory=ObjectiveVerification)

    @model_validator(mode="after")
    def _validate_objective(self) -> TrainingObjective:
        input_roles = [item.role for item in self.dataset_inputs]
        if input_roles != sorted(input_roles) or len(input_roles) != len(set(input_roles)):
            raise ValueError("dataset_inputs must be sorted by unique role")
        for field_name, id_attr in (
            ("labels", "label_id"),
            ("loss_masks", "mask_id"),
            ("loss_components", "component_id"),
        ):
            items = getattr(self, field_name)
            ids = [getattr(item, id_attr) for item in items]
            if ids != sorted(ids) or len(ids) != len(set(ids)):
                raise ValueError(f"{field_name} must be sorted by unique id")
        label_ids = {item.label_id for item in self.labels}
        mask_ids = {item.mask_id for item in self.loss_masks}
        for component in self.loss_components:
            if component.label_ref is not None and component.label_ref not in label_ids:
                raise ValueError("loss component references an unknown label")
            if component.mask_ref is not None and component.mask_ref not in mask_ids:
                raise ValueError("loss component references an unknown mask")
        adaptation_values = [item.value for item in self.adaptation_methods]
        if adaptation_values != sorted(set(adaptation_values)):
            raise ValueError("adaptation_methods must be sorted and unique")
        artifact_values = [item.kind.value for item in self.expected_artifacts]
        if artifact_values != sorted(set(artifact_values)):
            raise ValueError("expected_artifacts must be sorted by unique kind")
        if self.limitations != sorted(set(self.limitations)):
            raise ValueError("objective limitations must be sorted and unique")
        if self.execution_kind == ObjectiveExecutionKind.training:
            if not self.loss_components or not self.labels:
                raise ValueError("training objectives require label and loss construction")
            if self.update_policy.selection_mode == ObjectiveSelectionMode.none:
                raise ValueError("training objectives require a trainable update policy")
        elif self.update_policy.selection_mode != ObjectiveSelectionMode.none:
            raise ValueError("non-training objectives cannot declare trainable updates")
        return self


class ObjectiveCompatibilityAxis(ContractModel):
    status: ObjectiveCompatibilityStatus
    reasons: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_axis(self) -> ObjectiveCompatibilityAxis:
        if self.reasons != sorted(set(self.reasons)):
            raise ValueError("compatibility reasons must be sorted and unique")
        if self.evidence != sorted(set(self.evidence)):
            raise ValueError("compatibility evidence must be sorted and unique")
        if self.status == ObjectiveCompatibilityStatus.incompatible and not self.reasons:
            raise ValueError("incompatible axes require reasons")
        if self.status in {
            ObjectiveCompatibilityStatus.declared_compatible,
            ObjectiveCompatibilityStatus.verified_compatible,
        } and not self.evidence:
            raise ValueError("compatible axes require evidence")
        return self


class ObjectiveCompatibilityReport(ContractModel):
    """Independent compatibility axes. A static backend declaration can earn only
    ``declared_compatible``; a matching functional capability report is required for verified
    compatibility."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    objective_ref: Ref
    objective_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    dataset_schema_id: str | None = None
    dataset_schema_version: str | None = None
    model_id: str | None = None
    backend_id: str | None = None
    capability_environment_ref: Ref | None = None
    dataset: ObjectiveCompatibilityAxis
    model: ObjectiveCompatibilityAxis
    backend: ObjectiveCompatibilityAxis
    overall_status: ObjectiveCompatibilityStatus
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_overall(self) -> ObjectiveCompatibilityReport:
        statuses = {self.dataset.status, self.model.status, self.backend.status}
        if statuses == {ObjectiveCompatibilityStatus.not_applicable}:
            expected = ObjectiveCompatibilityStatus.not_applicable
        elif ObjectiveCompatibilityStatus.incompatible in statuses:
            expected = ObjectiveCompatibilityStatus.incompatible
        elif ObjectiveCompatibilityStatus.unverified in statuses:
            expected = ObjectiveCompatibilityStatus.unverified
        elif ObjectiveCompatibilityStatus.declared_compatible in statuses:
            expected = ObjectiveCompatibilityStatus.declared_compatible
        else:
            expected = ObjectiveCompatibilityStatus.verified_compatible
        if self.overall_status != expected:
            raise ValueError(f"overall_status must be {expected.value} for the axis statuses")
        if self.notes != sorted(set(self.notes)):
            raise ValueError("compatibility report notes must be sorted and unique")
        return self


# --------------------------------------------------------------------------------------------------
# EnvironmentProfile — the hashable host + software signature. Formalizes + extends environment.py.
# --------------------------------------------------------------------------------------------------
class EnvHost(ContractModel):
    os: OperatingSystem
    os_detail: str = ""
    # wddm / linux_dedicated / unified_memory — the most decisive field for spill-vs-OOM.
    memory_residency_model: MemoryResidencyModel = MemoryResidencyModel.unknown
    python_version: str = ""
    hostname_hash: str | None = None


class EnvCpu(ContractModel):
    model: str = ""
    physical_cores: int | None = Field(default=None, ge=0)
    logical_cores: int | None = Field(default=None, ge=0)
    instruction_sets: list[str] = Field(default_factory=list)


class EnvRam(ContractModel):
    total_bytes: int | None = Field(default=None, ge=0)
    available_bytes: int | None = Field(default=None, ge=0)


class GpuPcie(ContractModel):
    gen: int | None = Field(default=None, ge=1)
    width: int | None = None


class GpuDevice(ContractModel):
    """One accelerator. Grounded in environment.GpuInfo + gpu_probe.GpuMemory."""

    index: int = Field(ge=0)
    kind: DeviceKind
    name: str
    vram_total_bytes: int | None = Field(default=None, ge=0)
    vram_free_bytes: int | None = Field(default=None, ge=0)
    compute_capability: str | None = None
    # >=12 → Blackwell, forced onto the math-attention path (environment._capability_major).
    compute_capability_major: int | None = None
    supported_dtypes: list[PrecisionMode] = Field(default_factory=list)
    pcie: GpuPcie | None = None


class AcceleratorRuntime(ContractModel):
    kind: DeviceKind | None = None
    driver_version: str | None = None
    cuda_runtime_version: str | None = None
    cuda_driver_version: str | None = None
    rocm_version: str | None = None
    mps_available: bool | None = None
    nvidia_smi_available: bool = False


class EnvStorage(ContractModel):
    scratch_path: str | None = None
    free_bytes: int | None = Field(default=None, ge=0)
    kind: Literal["nvme", "ssd", "hdd", "network", "unknown"] = "unknown"


class EnvironmentProfile(ContractModel):
    """The full, hashable SIGNATURE of a host + software environment. Formalizes + greatly extends
    environment.probe_training_runtime (package versions + GpuInfo), gpu_probe.probe_gpu_memory, and
    provenance.RunProvenance. A RunManifest/RunPlan references a profile by ``environment_signature``
    so a result is always tied to the exact environment that produced it."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    environment_signature: str = Field(pattern=SHA256_PATTERN)
    captured_at: str | None = None
    engine_version: str = ""
    host: EnvHost
    cpu: EnvCpu | None = None
    ram: EnvRam | None = None
    gpus: list[GpuDevice] = Field(default_factory=list)
    accelerator_runtime: AcceleratorRuntime | None = None
    storage: EnvStorage | None = None
    packages: list[PackageLock] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------------------
# StorageProfile — the storage topology + per-role path suitability. NEW (EnvStorage was a stub).
# --------------------------------------------------------------------------------------------------
class StorageDevice(ContractModel):
    """One characterized storage location, from a dependency-light, NON-destructive probe (mount +
    capacity + cheaply-discoverable device attributes). The heavy metrics the spec envisions —
    measured sequential/random throughput, SMART/NVMe endurance, temperature — are deliberately absent
    here: they require a bounded benchmark or a privileged SMART read (a later, consent-gated slice).
    Unknown fields stay ``None``/``unknown`` — an honest gap, never a guessed value."""

    mount_point: str = Field(min_length=1)
    filesystem: str = ""
    interface: StorageInterface = StorageInterface.unknown
    total_bytes: int | None = Field(default=None, ge=0)
    free_bytes: int | None = Field(default=None, ge=0)
    # Cheaply-discoverable flags (removable/rotational from GetDriveType / /sys/block); None = unknown.
    removable: bool | None = None
    rotational: bool | None = None
    # True when the mount point is inside a known cloud-sync client's folder (a sync client will
    # re-upload every checkpoint/offload write and thrash the disk).
    cloud_synced: bool | None = None
    # True when this is a Windows host drive seen from WSL through /mnt (drvfs/9p). Access crosses the
    # NTFS translation layer — slow for small-file-heavy roles (venv/repo) and for high I/O.
    wsl_host_drive: bool | None = None
    device_name: str | None = None
    notes: list[str] = Field(default_factory=list)


class StorageRoleAssessment(ContractModel):
    """The PER-ROLE verdict for a candidate path: can it play this role, and if not, WHY. The reasons
    are the safe-spill guardrail's human-readable justification (USB bridge / synced folder / free-space
    margin / inside the source repo / rotational disk)."""

    role: StorageRole
    path: str
    suitability: StorageSuitability
    device_mount_point: str | None = None
    interface: StorageInterface = StorageInterface.unknown
    free_bytes: int | None = Field(default=None, ge=0)
    required_free_bytes: int | None = Field(default=None, ge=0)
    reasons: list[str] = Field(default_factory=list)


class StorageProfile(ContractModel):
    """The host's storage topology + optional per-role path assessments — the input the run planner
    needs to assign offload/checkpoint/scratch paths SAFELY (§11/§20). Standalone (not folded into
    EnvironmentProfile) so it never perturbs the ``environment_signature``. NEW: the engine has no
    storage detection today (EnvStorage was a scratch_path/free_bytes/kind stub)."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    captured_at: str | None = None
    devices: list[StorageDevice] = Field(default_factory=list)
    assessments: list[StorageRoleAssessment] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------------------
# BackendManifest / CapabilityReport — static declaration vs measured host verdict.
# --------------------------------------------------------------------------------------------------
class DependencyRequirement(ContractModel):
    name: str
    specifier: str | None = None
    optional: bool = False
    reason: str | None = None


class DependencyConflict(ContractModel):
    packages: list[str] = Field(min_length=1)
    condition: str
    severity: Literal["block", "warn"] = "block"


class CheckpointSemantics(ContractModel):
    contains: list[
        Literal[
            "adapter_weights",
            "base_weights",
            "optimizer_state",
            "lr_scheduler",
            "rng_state",
            "trainer_state",
        ]
    ] = Field(default_factory=list)
    resumable: bool = False
    reload_verifiable: bool = False


class ExportCompatibilityEntry(ContractModel):
    format: ExportFormat
    serves_in: list[str] = Field(default_factory=list)


class KnownFailureMode(ContractModel):
    """Pre-declared hazards tagged with a taxonomy so the core can warn/refuse up front. The
    canonical example: fused attention deadlocks on sm_120."""

    taxonomy: FailureTaxonomy
    condition: str
    description: str = ""
    mitigation: str | None = None


class TelemetryHook(ContractModel):
    stage: StageMarker
    metrics: list[str] = Field(default_factory=list)


class BackendManifest(ContractModel):
    """A backend worker's STATIC self-declaration of everything it can do — the core reads this to
    decide which backend can even attempt a RunPlan, BEFORE dispatch. Mostly NEW; generalizes the
    inference model_backends.base.ModelBackend Protocol + training/compatibility into a declarable
    manifest for a TRAINING backend."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    backend_id: str = Field(min_length=1)
    display_name: str = ""
    backend_version: str
    trainer_target: TrainerTarget | None = None
    supported_os: list[OperatingSystem] = Field(min_length=1)
    supported_devices: list[DeviceKind] = Field(min_length=1)
    required_compute_capability: str | None = None
    model_families: list[str] = Field(default_factory=list)
    task_types: list[TaskType] = Field(min_length=1)
    precision_modes: list[PrecisionMode] = Field(default_factory=list)
    quantization_modes: list[QuantizationMode] = Field(default_factory=list)
    adapter_methods: list[AdapterMethod] = Field(default_factory=list)
    attention_impls: list[AttentionImpl] = Field(default_factory=list)
    loss_impls: list[LossImpl] = Field(default_factory=list)
    # Semantic objective capabilities (for example causal_lm_sft or adapter_qlora). This is still a
    # STATIC declaration; only the matching field in EffectiveCapabilities can prove it on a host.
    objective_capabilities: list[ObjectiveCapability] = Field(default_factory=list)
    checkpoint_impls: list[CheckpointImpl] = Field(default_factory=list)
    optimizers: list[Optimizer] = Field(default_factory=list)
    offload_strategies: list[OffloadStrategy] = Field(default_factory=list)
    compile_modes: list[CompileMode] = Field(default_factory=list)
    export_formats: list[ExportFormat] = Field(default_factory=list)
    dependency_requirements: list[DependencyRequirement] = Field(default_factory=list)
    dependency_conflicts: list[DependencyConflict] = Field(default_factory=list)
    environment_lock_ref: Ref | None = None
    checkpoint_semantics: CheckpointSemantics | None = None
    export_compatibility: list[ExportCompatibilityEntry] = Field(default_factory=list)
    known_failure_modes: list[KnownFailureMode] = Field(default_factory=list)
    capability_probes: list[str] = Field(default_factory=list)
    telemetry_hooks: list[TelemetryHook] = Field(default_factory=list)

    @field_validator("objective_capabilities")
    @classmethod
    def _sorted_objective_capabilities(cls, values: list[str]) -> list[str]:
        if values != sorted(set(values)):
            raise ValueError("objective_capabilities must be sorted and unique")
        return values


class ProbeResult(ContractModel):
    probe: str
    outcome: FailureTaxonomy
    detail: str | None = None
    measured: JsonObject = Field(default_factory=dict)


class EffectiveCapabilities(ContractModel):
    """The intersection of what a backend DECLARES and what PROVED to work on this host. The planner
    resolves a RunPlan against this, not the raw BackendManifest."""

    precision_modes: list[PrecisionMode] = Field(default_factory=list)
    quantization_modes: list[QuantizationMode] = Field(default_factory=list)
    attention_impls: list[AttentionImpl] = Field(default_factory=list)
    adapter_methods: list[AdapterMethod] = Field(default_factory=list)
    objective_capabilities: list[ObjectiveCapability] = Field(default_factory=list)

    @field_validator("objective_capabilities")
    @classmethod
    def _sorted_effective_objective_capabilities(cls, values: list[str]) -> list[str]:
        if values != sorted(set(values)):
            raise ValueError("effective objective_capabilities must be sorted and unique")
        return values


class CapabilityReport(ContractModel):
    """The DYNAMIC, measured counterpart of a BackendManifest: probe results against a specific
    EnvironmentProfile. Formalizes environment.probe_training_runtime (ready/cpu_toy_ready/
    bitsandbytes_ok/notes) and generalizes it to per-probe outcomes tagged with FailureTaxonomy."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    backend_id: str
    backend_version: str | None = None
    environment_ref: Ref
    generated_at: str | None = None
    readiness: Literal["ready", "cpu_toy_only", "not_ready"]
    bitsandbytes_ok: bool = False
    installed_packages: list[PackageLock] = Field(default_factory=list)
    missing_packages: list[str] = Field(default_factory=list)
    probe_results: list[ProbeResult] = Field(default_factory=list)
    effective_capabilities: EffectiveCapabilities | None = None
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------------------
# Environment Manager — recipes, resolution preview, locks, descriptors, health. NEW (Phase 2).
# The 3-layer dependency model: a lightweight always-installable control plane, opt-in capability
# profiles, and ISOLATED per-backend worker environments (heavy frameworks pin conflicting builds).
# --------------------------------------------------------------------------------------------------
class PythonRuntime(ContractModel):
    """A discovered Python executable that can potentially create an isolated worker environment.

    Discovery never assumes the control-plane interpreter is the only installation. Compatibility is
    an explicit verdict against the selected recipe, while ``venv_available`` proves the stdlib venv
    module can be located without creating anything.
    """

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    runtime_id: str = Field(pattern=_ID)
    executable: str = Field(min_length=1)
    version: str = ""
    implementation: str = ""
    architecture: str = ""
    platform: str = ""
    os: OperatingSystem = OperatingSystem.unknown
    is_virtual_environment: bool = False
    venv_available: bool = False
    compatible: bool = False
    incompatibility_reasons: list[str] = Field(default_factory=list)


class EnvironmentRecipe(ContractModel):
    """A declarative, platform/CUDA-aware recipe for building one isolated environment — the WHAT to
    install, not the act of installing. A recipe is only a declaration: ``verification`` says whether
    it has ever produced a working environment (declared → hardware_verified). Grounded in the engine's
    real optional extras (pyproject ``[train]`` / ``[parquet]`` / ``[tokenizer]``)."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    recipe_id: str = Field(pattern=_ID)
    display_name: str = ""
    layer: DependencyLayer
    description: str = ""
    # The backend_id (backend_worker layer) or capability-profile name this recipe provisions.
    target: str = ""
    python_requires: str = ">=3.10"
    dependency_requirements: list[DependencyRequirement] = Field(default_factory=list)
    # PyPI index by default; a CUDA/ROCm build overrides it per accelerator tag (see cuda_index_urls).
    default_index_url: str | None = None
    extra_index_urls: list[str] = Field(default_factory=list)
    # Accelerator tag → wheel index (e.g. "cu128" → the PyTorch cu128 index). The resolver picks one by
    # the host's detected CUDA, so a Blackwell host gets cu128 wheels, a CPU host the cpu index.
    cuda_index_urls: dict[str, str] = Field(default_factory=dict)
    requires_cuda: bool = False
    # A dependency that builds a native extension needs a compiler toolchain present (e.g. DeepSpeed).
    requires_native_build: bool = False
    min_compute_capability: str | None = None
    supported_os: list[OperatingSystem] = Field(default_factory=list)
    known_conflicts: list[DependencyConflict] = Field(default_factory=list)
    capability_probes: list[str] = Field(default_factory=list)
    verification: RecipeVerification = RecipeVerification.declared
    notes: list[str] = Field(default_factory=list)


class InstallStep(ContractModel):
    """One bounded, argv-structured install command — NEVER a shell string, so an untrusted package or
    index name can't inject a command (mirrors the no-shell trainer-launch invariant). ``argv[0]`` is
    the executable; the rest are literal arguments."""

    phase: Literal["create_venv", "upgrade_pip", "install", "verify"]
    description: str = ""
    argv: list[str] = Field(min_length=1)
    working_directory: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=1800, ge=1)
    expected_outputs: list[str] = Field(default_factory=list)
    network_required: bool = False
    native_build_expected: bool = False


class DependencyResolution(ContractModel):
    """The resolved PREVIEW of provisioning a recipe on a specific host — the exact argv steps, the
    chosen wheel index, and the disk/network cost — for explicit user confirmation BEFORE anything is
    installed. Pure/derivable; no environment is created to produce it. NEW."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    recipe_ref: Ref
    environment_ref: Ref | None = None
    runtime: PythonRuntime | None = None
    environment_root: str | None = None
    manager_version: str = ""
    python_version: str = ""
    os: OperatingSystem = OperatingSystem.unknown
    # The accelerator tag the resolver selected (e.g. "cu128", "cpu") + the index it maps to.
    accelerator_tag: str = "cpu"
    resolved_index_urls: list[str] = Field(default_factory=list)
    install_steps: list[InstallStep] = Field(default_factory=list)
    # Rough, explicitly-heuristic size estimates (download + on-disk installed footprint).
    estimated_download_bytes: int | None = Field(default=None, ge=0)
    estimated_disk_bytes: int | None = Field(default=None, ge=0)
    # A recipe the host cannot satisfy (unmet python_requires / unsupported OS / cuda required, absent)
    # resolves with resolvable=False and the blocking reasons; warnings are non-blocking caveats.
    resolvable: bool = True
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # Hash over the canonical resolution body, including concrete argv/runtime/root. Creation requires
    # the caller to echo this exact hash, proving the executed plan is the reviewed plan.
    resolution_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)


class EnvironmentCommandRecord(ContractModel):
    """Durable evidence for one bounded, argv-only creation/install/probe command."""

    command_id: str = Field(pattern=_ID)
    phase: Literal[
        "create_venv", "upgrade_pip", "install", "lock", "import_probe",
        "verify", "dependency_probe", "functional_probe", "hardware_probe", "health_probe",
    ]
    argv: list[str] = Field(min_length=1)
    working_directory: str
    environment: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(ge=1)
    expected_outputs: list[str] = Field(default_factory=list)
    started_at: str
    finished_at: str
    exit_code: int | None = None
    timed_out: bool = False
    cancelled: bool = False
    stdout_path: str | None = None
    stderr_path: str | None = None
    native_build_occurred: bool = False
    failure: FailureRecord | None = None


class EnvironmentInstallation(ContractModel):
    """Recoverable journal for one environment creation attempt."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    installation_id: str = Field(pattern=_ID)
    environment_ref: Ref
    recipe_ref: Ref
    resolution_ref: Ref
    state: EnvironmentState = EnvironmentState.installing
    started_at: str
    finished_at: str | None = None
    commands: list[EnvironmentCommandRecord] = Field(default_factory=list)
    failure: FailureRecord | None = None
    retry_requires_recreate: bool = False


class EnvironmentLock(ContractModel):
    """The exact, reproducible record of what an environment actually contains — the post-install
    counterpart of a recipe (which is only intent). ``packages`` are the resolved installs with
    versions + hashes; ``lock_hash`` seals the set for drift detection. NEW."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    lock_id: str = Field(pattern=_ID)
    recipe_ref: Ref
    created_at: str | None = None
    manager_version: str = ""
    runtime: PythonRuntime | None = None
    python_version: str = ""
    platform_tag: str = ""
    architecture: str = ""
    implementation: str = ""
    torch_version: str | None = None
    torch_build: str | None = None
    cuda_runtime_version: str | None = None
    compute_capability: str | None = None
    index_urls: list[str] = Field(default_factory=list)
    packages: list[PackageLock] = Field(default_factory=list)
    # sha256 over the canonical lock body (runtime, recipe, sources, metadata hashes, and packages).
    lock_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)


class EnvironmentDescriptor(ContractModel):
    """A managed, ISOLATED environment instance. Its ``root_path`` is the isolation boundary — the
    Environment Manager only ever installs into this env's own interpreter, never another's, so one
    backend can't corrupt another's runtime. NEW."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    env_id: str = Field(pattern=_ID)
    recipe_ref: Ref
    layer: DependencyLayer
    root_path: str = ""
    python_executable: str = ""
    state: EnvironmentState = EnvironmentState.not_installed
    lock_ref: Ref | None = None
    resolution_ref: Ref | None = None
    installation_ref: Ref | None = None
    managed_by: str = "CorpusStudio"
    manager_version: str = ""
    created_at: str | None = None
    updated_at: str | None = None
    notes: list[str] = Field(default_factory=list)


class EnvironmentHealthReport(ContractModel):
    """The live health of a managed environment: its state, drift vs the recorded lock, and probe
    outcomes. ``drift_detected`` means the installed set no longer matches the lock (a package changed
    under the env). NEW."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    environment_ref: Ref
    recipe_ref: Ref | None = None
    lock_ref: Ref | None = None
    state: EnvironmentState
    python_version: str = ""
    checked_at: str | None = None
    installed_packages: list[PackageLock] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    drifted_packages: list[str] = Field(default_factory=list)
    changed_package_sources: list[str] = Field(default_factory=list)
    drift_detected: bool = False
    recipe_drift_detected: bool = False
    lock_mismatch: bool = False
    interpreter_missing: bool = False
    environment_missing: bool = False
    hardware_mismatch: bool = False
    cuda_mismatch: bool = False
    probe_results: list[ProbeResult] = Field(default_factory=list)
    failure: FailureRecord | None = None
    remediation: str | None = None


# --------------------------------------------------------------------------------------------------
# RunPlan — the immutable, fully-resolved execution plan.
# --------------------------------------------------------------------------------------------------
class AdapterSpec(ContractModel):
    method: AdapterMethod
    lora_r: int | None = Field(default=None, ge=1)
    lora_alpha: int | None = Field(default=None, ge=1)
    lora_dropout: float | None = Field(default=None, ge=0, le=1)
    target_modules: list[str] | None = None


class OptimizerSpec(ContractModel):
    impl: Optimizer
    learning_rate: float = Field(gt=0)
    weight_decay: float | None = Field(default=None, ge=0)
    lr_scheduler: str | None = None
    warmup_ratio: float | None = Field(default=None, ge=0, le=1)


class SequenceSpec(ContractModel):
    """Sequence handling. Buckets let variable-length rows batch efficiently; the max bucket ==
    the trainer's sequence_len (config_templates.sequence_len, default 4096)."""

    max_sequence_len: int = Field(default=4096, ge=1)
    buckets: list[int] = Field(default_factory=list)
    packing: bool = False
    # When false, a plan whose dataset has examples_over_sequence_len>0 is invalid.
    truncation_allowed: bool = True


class BatchingSpec(ContractModel):
    """The accumulation TARGET is expressed in SUPERVISED TOKENS, not microbatch count. A
    token-target-CAPABLE backend accumulates until summed supervised tokens >= target and normalizes
    the loss by supervised tokens across the window, so the effective batch is invariant to sequence
    length + padding. The first-party ``corpus_studio`` reference trainer is NOT token-target-capable
    today — it honors ``fallback_grad_accumulation_steps`` (a fixed microbatch count); the token
    target is the contract a future token-aware backend would satisfy."""

    micro_batch_size: int = Field(default=1, ge=1)
    supervised_token_accumulation_target: int = Field(ge=1)
    # Legacy count for backends that cannot honor a token target; advisory when the target is set.
    fallback_grad_accumulation_steps: int | None = Field(default=None, ge=1)


class CheckpointPolicy(ContractModel):
    impl: CheckpointImpl
    cadence_optimizer_steps: int | None = Field(default=None, ge=1)
    cadence_seconds: int | None = Field(default=None, ge=1)
    keep_last: int | None = Field(default=None, ge=1)
    # Reload each checkpoint and assert integrity (feeds ArtifactManifest.reload_verified).
    reload_verify: bool = False


class EvalSchedule(ContractModel):
    before_run: bool = True
    after_run: bool = True
    every_optimizer_steps: int | None = Field(default=None, ge=1)
    suite_ref: str | None = None


class ExportSpec(ContractModel):
    format: ExportFormat
    output_dir: str = "output"


class RunPlan(ContractModel):
    """The IMMUTABLE, fully-resolved execution plan the core dispatches to a worker: no ambiguity is
    left for the worker to decide. Formalizes + hardens config_templates.TrainingConfigTemplate. Key
    upgrades: attention_backend defaults to math on Blackwell; the accumulation target is in
    SUPERVISED TOKENS; ``plan_hash`` seals immutability (a change means a NEW plan)."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    plan_id: str = Field(min_length=1)
    # sha256 over the canonicalized plan body — the immutability seal (cf. provenance.config_sha256).
    plan_hash: str = Field(pattern=SHA256_PATTERN)
    created_at: str | None = None
    backend_ref: Ref
    environment_ref: Ref
    dataset_ref: Ref
    task_type: TaskType
    base_model: str = Field(min_length=1)
    precision: PrecisionMode
    quantization: QuantizationMode
    adapter: AdapterSpec
    optimizer: OptimizerSpec
    loss_impl: LossImpl
    # MUST be math/eager/sdpa when the target GPU is compute_capability_major>=12 (Blackwell).
    attention_backend: AttentionImpl
    sequence: SequenceSpec
    batching: BatchingSpec
    checkpoint_policy: CheckpointPolicy
    offload_strategy: OffloadStrategy = OffloadStrategy.none
    allocator_policy: AllocatorPolicy = AllocatorPolicy.default
    compile_mode: CompileMode = CompileMode.none
    gradient_checkpointing: bool = True
    eval_schedule: EvalSchedule = Field(default_factory=EvalSchedule)
    export: ExportSpec
    seed: int = Field(default=42, ge=0)
    # The exact rendered trainer config folded in verbatim for byte-exact reproducibility.
    training_config_snapshot: JsonObject = Field(default_factory=dict)


# --------------------------------------------------------------------------------------------------
# ArtifactManifest — a produced weight artifact. Formalizes artifact_registry.ModelArtifactRecord.
# --------------------------------------------------------------------------------------------------
class ArtifactIntegrity(ContractModel):
    """Two-tier integrity: cheap size+mtime fingerprint powers the fast LIST; content_hash (sha256
    over weight bytes) powers the promote GATE. ``current_integrity`` is computed LIVE at read."""

    cheap_fingerprint: str | None = None
    content_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    current_integrity: Literal["ok", "missing", "modified", "unknown"] = "unknown"


class ArtifactManifest(ContractModel):
    """A first-class record of a weight artifact a run produced. Formalizes
    artifact_registry.ModelArtifactRecord + its two-tier integrity model. The platform NEVER
    moves/copies/deletes the underlying weights — the manifest only references + re-checks them."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    artifact_id: str = Field(pattern=_ID)
    producer_run_ref: Ref
    created_at: str | None = None
    updated_at: str | None = None
    kind: Literal[
        "adapter", "checkpoint", "merged_model", "gguf", "onnx", "quantized", "other"
    ] = "adapter"
    path: str
    status: Literal["candidate", "kept", "rejected"] = "candidate"
    integrity: ArtifactIntegrity | None = None
    # NEW: whether the producing backend reloaded these weights and asserted equivalence.
    reload_verified: bool = False
    # Resolved LIVE from the source run at display time, never stored.
    base_model: str | None = None
    notes: str = ""


# --------------------------------------------------------------------------------------------------
# EvaluationResult — formalizes evaluation/reports + gates + adds the as-served-vs-raw axis.
# --------------------------------------------------------------------------------------------------
class EvalTarget(ContractModel):
    model: str
    artifact_ref: str | None = None
    run_ref: str | None = None
    phase: Literal["before", "after", "standalone"] = "standalone"


class EvalDataset(ContractModel):
    name: str = ""
    version_ref: str | None = None
    dataset_fingerprint: str | None = Field(default=None, pattern=SHA256_PATTERN)


class EvalMetric(ContractModel):
    """The scorer AND what it measures — so a number is never treated as quality without
    qualification (reports.EvaluationReport.metric honesty note)."""

    name: Literal["keyword_overlap", "llm_judge", "exact_match", "pass_rate", "custom"] = (
        "keyword_overlap"
    )
    measures: str = ""
    judge_model: str | None = None
    score_threshold: float | None = None


class AsServed(ContractModel):
    """How the model was actually served — the RAW-vs-AS-SERVED axis. Two evals of the 'same' model
    differ if quantization/adapter/template/decoding differ."""

    backend: str | None = None
    precision: PrecisionMode | None = None
    quantization: QuantizationMode | None = None
    adapter_applied: bool | None = None
    chat_template_applied: bool | None = None
    decoding: JsonObject = Field(default_factory=dict)


class EvalSummary(ContractModel):
    examples_tested: int = Field(ge=0)
    average_score: float
    failed_examples: int = Field(default=0, ge=0)
    pass_rate: float | None = Field(default=None, ge=0, le=1)
    weak_tags: list[str] = Field(default_factory=list)
    average_manual_score: float | None = None


class EvalGate(ContractModel):
    """Grounded in gates/models.GateReport/GateStatus (pass/warn/block, counts, effective
    thresholds behind the verdict for reproducibility)."""

    overall_status: Literal["pass", "warn", "block"]
    pass_count: int = Field(default=0, ge=0)
    warn_count: int = Field(default=0, ge=0)
    block_count: int = Field(default=0, ge=0)
    min_eval_average_score: float | None = None
    min_eval_pass_rate: float | None = None
    max_regression_score_drop: float | None = None


class EvaluationResult(ContractModel):
    """The outcome of evaluating a model/dataset, with an explicit as-served vs raw distinction so a
    number is never presented as a quality signal without saying what produced it. Formalizes
    evaluation/reports.EvaluationReport + gates/models.GateReport."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    eval_id: str = Field(min_length=1)
    generated_at: str | None = None
    target: EvalTarget
    dataset: EvalDataset | None = None
    metric: EvalMetric
    as_served: AsServed | None = None
    summary: EvalSummary
    gate: EvalGate | None = None
    # Presenting a before/after delta requires this to be null (weight_card provenance guard).
    provenance_caveat: str | None = None
    report_ref: str | None = None


# --------------------------------------------------------------------------------------------------
# RunEvent — the streamed telemetry envelope (NEW).
# --------------------------------------------------------------------------------------------------
class EventMetrics(ContractModel):
    """Present on metric/heartbeat events. All optional — a worker emits what it can sample. The
    memory block + step_time make the WDDM spill (10-25x slowdown, non-zero shared bytes) visible."""

    memory: MemoryMetrics | None = None
    gpu_utilization: float | None = Field(default=None, ge=0, le=100)
    memory_controller_utilization: float | None = Field(default=None, ge=0, le=100)
    power_watts: float | None = Field(default=None, ge=0)
    temperature_c: float | None = None
    pcie_tx_bytes_per_sec: int | None = Field(default=None, ge=0)
    pcie_rx_bytes_per_sec: int | None = Field(default=None, ge=0)
    step_time_seconds: float | None = Field(default=None, ge=0)
    tokens_per_sec: float | None = Field(default=None, ge=0)
    # The honest training rate that ignores padding, paired with the plan's supervised-token target.
    supervised_tokens_per_sec: float | None = Field(default=None, ge=0)
    loss: float | None = None
    grad_norm: float | None = Field(default=None, ge=0)
    learning_rate: float | None = Field(default=None, ge=0)


class RunEvent(ContractModel):
    """One envelope in the structured telemetry stream a worker emits for a run — the RunEvent half
    of the immutable-RunPlan-in / RunEvent-stream-out worker protocol. NEW; the engine has no
    streaming telemetry today (run_registry is a durable per-run record, not an event stream)."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    event_type: Literal[
        "stage",
        "metric",
        "log",
        "warning",
        "checkpoint_written",
        "eval_result",
        "artifact_produced",
        "heartbeat",
        "terminal",
    ]
    run_id: str = Field(min_length=1)
    # Monotonic per-run sequence number so a consumer can order/dedupe a resumed stream.
    seq: int = Field(ge=0)
    emitted_at: str
    stage: StageMarker | None = None
    microstep: int | None = Field(default=None, ge=0)
    optimizer_step: int | None = Field(default=None, ge=0)
    epoch: float | None = Field(default=None, ge=0)
    message: str | None = None
    metrics: EventMetrics | None = None
    # A live fit reclassification when the memory signature shifts (e.g. NATIVE_TIGHT → SPILL).
    fit: FitClassification | None = None
    payload: JsonObject | None = None


# --------------------------------------------------------------------------------------------------
# RunManifest — the durable run instance. Formalizes run_registry.TrainingRunRecord.
# --------------------------------------------------------------------------------------------------
class RunProcessInfo(ContractModel):
    """Process identity so a recycled pid is never mistaken for a live run. A 'running' record whose
    pid is not alive reconciles to 'interrupted' (run_registry.reconcile_running_records)."""

    pid: int | None = None
    process_started_at: str | None = None
    argv: list[str] = Field(default_factory=list)
    exit_code: int | None = None


class RunEvaluationLink(ContractModel):
    before_eval_ref: str | None = None
    after_eval_ref: str | None = None
    # The model/adapter the after-eval actually targeted (run_registry.after_eval_model).
    after_eval_model: str | None = None


class RunReproducibility(ContractModel):
    """Embedded reproducibility manifest (provenance.RunProvenance) for a self-contained audit."""

    dataset_fingerprint: str | None = Field(default=None, pattern=SHA256_PATTERN)
    dataset_row_count: int = Field(default=0, ge=0)
    config_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    engine_version: str = ""
    platform: str = ""
    python_version: str = ""


class RunManifest(ContractModel):
    """A single run INSTANCE: the crash-safe durable record of one execution of a RunPlan.
    Formalizes run_registry.TrainingRunRecord almost field-for-field + its state machine (terminal =
    {succeeded, failed, cancelled, interrupted}; a dead-pid 'running' record reconciles to
    interrupted)."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    run_id: str = Field(pattern=_ID)
    plan_ref: Ref
    environment_ref: Ref | None = None
    dataset_ref: Ref | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    state: Literal[
        "prepared", "running", "succeeded", "failed", "cancelled", "interrupted"
    ] = "prepared"
    base_model: str = ""
    target: str = ""
    process: RunProcessInfo | None = None
    output_dir: str = ""
    checkpoints: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    evaluation: RunEvaluationLink | None = None
    reproducibility: RunReproducibility | None = None
    # Present on abnormal termination (state in {failed, interrupted}).
    failure: FailureRecord | None = None
    # Post-run fit reconciliation from observed peak memory (planned NATIVE_SAFE, or a spill?).
    final_fit: FitClassification | None = None
    notes: str = ""


# --------------------------------------------------------------------------------------------------
# WorkerProtocol — the versioned core↔worker message envelope (NEW).
# --------------------------------------------------------------------------------------------------
class HelloBody(ContractModel):
    """Worker→core handshake: who I am + what I can do."""

    worker_id: str
    backend: BackendManifest
    environment: EnvironmentProfile | None = None


class CapabilityProbeRequestBody(ContractModel):
    """Core→worker: run these probes on this host and reply with a CapabilityReport."""

    probes: list[str] = Field(default_factory=list)
    environment_ref: Ref | None = None


class RunDispatchBody(ContractModel):
    """Core→worker: execute this immutable plan (passed by value so the worker needs no shared
    state)."""

    run_id: str = Field(min_length=1)
    plan: RunPlan
    heartbeat_interval_seconds: int = Field(default=30, ge=1)


class RunAcceptedBody(ContractModel):
    run_id: str
    pid: int | None = None
    process_started_at: str | None = None


class RunControlBody(ContractModel):
    run_id: str
    action: Literal["cancel", "pause", "resume", "checkpoint_now"]


class HeartbeatBody(ContractModel):
    run_id: str
    stage: StageMarker | None = None
    optimizer_step: int | None = Field(default=None, ge=0)
    pid_alive: bool = True


class TerminalResultBody(ContractModel):
    """Worker→core: the run ended. A FailureRecord is present iff the outcome was not PASS."""

    run_id: str
    outcome: FailureTaxonomy
    run_manifest: RunManifest | None = None
    artifacts: list[ArtifactManifest] = Field(default_factory=list)
    final_eval: EvaluationResult | None = None
    failure: FailureRecord | None = None


WorkerMessageType = Literal[
    "hello",
    "capability_probe_request",
    "capability_report",
    "run_dispatch",
    "run_accepted",
    "run_rejected",
    "run_control",
    "event",
    "heartbeat",
    "terminal_result",
    "failure",
]

# The body model that a given message `type` selects. Consumers parse `body` with this map; the
# envelope keeps `body` loose so the wire stays forward-compatible.
WORKER_BODY_BY_TYPE: dict[str, type[ContractModel]] = {
    "hello": HelloBody,
    "capability_probe_request": CapabilityProbeRequestBody,
    "capability_report": CapabilityReport,
    "run_dispatch": RunDispatchBody,
    "run_accepted": RunAcceptedBody,
    "run_rejected": FailureRecord,
    "run_control": RunControlBody,
    "event": RunEvent,
    "heartbeat": HeartbeatBody,
    "terminal_result": TerminalResultBody,
    "failure": FailureRecord,
}


class WorkerMessage(ContractModel):
    """The versioned envelope for the core↔worker channel — realizes the 'immutable RunPlan IN,
    structured RunEvent stream OUT' boundary. NEW; no worker/core protocol exists in the engine
    today (the desktop owns the trainer process directly). ``protocol_version`` lets the two sides
    negotiate compatibility independently of any single contract's version. The body shape is
    selected by ``type`` (see :data:`WORKER_BODY_BY_TYPE`)."""

    protocol_version: str = Field(pattern=r"^\d+\.\d+\.\d+([-+].+)?$")
    message_id: str = Field(min_length=1)
    correlation_id: str | None = None
    direction: Literal["core_to_worker", "worker_to_core"]
    sent_at: str | None = None
    type: WorkerMessageType
    body: JsonObject | None = None
