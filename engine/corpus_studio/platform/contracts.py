"""The CorpusStudio platform contracts — the language-neutral boundary between the (Python → Rust)
platform core, the Python AI worker(s), and the UI shell.

Each root contract is a pydantic model carrying a ``contract_version`` and grounded in an existing
engine model (see each class docstring). These models are the canonical source of truth; the
language-neutral JSON Schemas the Rust core / Avalonia / Tauri consume are generated FROM them by
:mod:`corpus_studio.platform.schema_export`. Pure — importing this pulls no torch.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

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
    AttentionKernel,
    CheckpointImpl,
    CommunicationBackend,
    CompatibilityStatus,
    CompileMode,
    CountHandling,
    DependencyLayer,
    DescriptorFileRole,
    DeviceKind,
    EvidenceKind,
    ExecutionVerificationRequirement,
    EnvironmentState,
    ExportFormat,
    FailureTaxonomy,
    FitClass,
    LossImpl,
    MemoryTier,
    MemoryResidencyModel,
    ModelAttentionType,
    ModelAttentionApi,
    ModelExecutionKind,
    ModelFormat,
    ModelSourceKind,
    ModelTaskClass,
    OffloadMechanism,
    OffloadStrategy,
    OffloadTrigger,
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
    ParameterAccountingProfile,
    ParameterAccountingStatus,
    ParameterCountKind,
    ParameterEvidenceSourceKind,
    ParameterGapReason,
    ParameterIdentityBasis,
    ParameterObservationCoverage,
    ParameterScopeKind,
    ParameterValueRelation,
    ParameterWindowKind,
    ParallelismKind,
    PhysicalStateKind,
    PlacementMode,
    PlacementRole,
    PositionalEncoding,
    PrecisionMode,
    QuantizationMode,
    RecipeVerification,
    RouteMissAction,
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
ParameterId = Annotated[str, Field(pattern=_ID)]


def _is_digest_value(value: str | None) -> bool:
    return bool(
        value
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_pinned_ref(ref: Ref) -> bool:
    return bool(
        ref.hash is not None
        and ref.hash.algo != "none"
        and _is_digest_value(ref.hash.value)
    )


def _canonical_contract_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
# TraceRecord - versioned reasoning/tool/process evidence. Structured records never use <think>
# markup internally; that syntax is confined to legacy import and training rendering boundaries.
# --------------------------------------------------------------------------------------------------
class TraceSource(ContractModel):
    """Pinned identity of the exact source row used to create a trace.

    A source is either a hash-pinned DatasetManifest reference or a hash-pinned imported artifact.
    ``source_row_id`` reuses the engine's sha256(exact_row_signature) row identity so trace lineage
    and dataset-version lineage agree rather than inventing a second row fingerprint.
    """

    dataset_ref: Ref | None = None
    artifact_ref: str | None = Field(default=None, min_length=1)
    artifact_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    source_row_id: str = Field(pattern=SHA256_PATTERN)
    source_row_index: int | None = Field(default=None, ge=1)
    row_id_algo: Literal["sha256-exact-row-signature-v1"] = "sha256-exact-row-signature-v1"

    @model_validator(mode="after")
    def _validate_source(self) -> TraceSource:
        has_dataset = self.dataset_ref is not None
        has_artifact = self.artifact_ref is not None or self.artifact_sha256 is not None
        if has_dataset == has_artifact:
            raise ValueError("trace source requires exactly one pinned dataset or imported artifact")
        if self.dataset_ref is not None and not _is_pinned_ref(self.dataset_ref):
            raise ValueError("trace dataset_ref must be hash-pinned")
        if (self.artifact_ref is None) != (self.artifact_sha256 is None):
            raise ValueError("trace artifact_ref and artifact_sha256 must be set together")
        return self


class TraceMessage(ContractModel):
    message_id: str = Field(pattern=_ID)
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)
    name: str | None = Field(default=None, min_length=1)
    tool_call_id: str | None = Field(default=None, pattern=_ID)

    @model_validator(mode="after")
    def _validate_message(self) -> TraceMessage:
        if not self.content.strip():
            raise ValueError("trace context content cannot be blank")
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("tool context messages require tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid on tool context messages")
        return self


class TraceToolCall(ContractModel):
    call_id: str = Field(pattern=_ID)
    tool_name: str = Field(min_length=1)
    tool_version: str | None = None
    arguments: JsonObject = Field(default_factory=dict)
    argument_schema_ref: Ref | None = None


class TraceToolResult(ContractModel):
    call_id: str = Field(pattern=_ID)
    status: Literal["success", "error", "denied"]
    content: str | None = None
    content_ref: Ref | None = None
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    truncated: bool = False
    error: str | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> TraceToolResult:
        if self.content is None and self.content_ref is None:
            raise ValueError("tool results require inline content or a content_ref")
        if self.status in {"error", "denied"} and not self.error:
            raise ValueError("error/denied tool results require an error explanation")
        if self.status == "success" and self.error is not None:
            raise ValueError("successful tool results cannot carry an error")
        return self


class TraceTrainingSignal(ContractModel):
    """Optional per-segment supervision without implying the content is factual ground truth."""

    target: bool = False
    label: str | float | bool | None = None
    reward: float | None = None
    weight: float = Field(default=1.0, ge=0)
    verifier_ref: Ref | None = None


class TraceSegment(ContractModel):
    segment_id: str = Field(pattern=_ID)
    sequence: int = Field(ge=0)
    kind: Literal[
        "reasoning",
        "action",
        "tool_call",
        "tool_result",
        "observation",
        "verifier",
        "final_answer",
    ]
    actor: Literal["system", "user", "assistant", "tool", "verifier", "human"]
    origin: Literal["model", "tool", "human", "imported", "derived"]
    verification: Literal[
        "unverified", "human_verified", "tool_verified", "verifier_accepted", "rejected"
    ] = "unverified"
    content: str | None = None
    content_ref: Ref | None = None
    tool_call: TraceToolCall | None = None
    tool_result: TraceToolResult | None = None
    training_signal: TraceTrainingSignal | None = None
    evidence_refs: list[Ref] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_segment(self) -> TraceSegment:
        if self.kind == "tool_call":
            if self.tool_call is None or self.tool_result is not None:
                raise ValueError("tool_call segments require only tool_call payload")
            if self.actor != "assistant" or self.origin not in {"model", "human", "imported"}:
                raise ValueError("tool_call segments must be authored by assistant/model, human, or import")
        elif self.kind == "tool_result":
            if self.tool_result is None or self.tool_call is not None:
                raise ValueError("tool_result segments require only tool_result payload")
            if self.actor != "tool" or self.origin != "tool":
                raise ValueError("tool_result segments must preserve the tool boundary")
        else:
            if self.tool_call is not None or self.tool_result is not None:
                raise ValueError("tool payloads are only valid on tool_call/tool_result segments")
            if self.content is None and self.content_ref is None:
                raise ValueError(f"{self.kind} segments require content or content_ref")
        if self.content is not None:
            if not self.content.strip():
                raise ValueError("inline trace segment content cannot be blank")
            if "<think>" in self.content or "</think>" in self.content:
                raise ValueError("structured trace segments cannot contain <think> markup")
        evidence_keys = [
            (item.id, item.hash.value if item.hash and item.hash.value else "")
            for item in self.evidence_refs
        ]
        if evidence_keys != sorted(set(evidence_keys)):
            raise ValueError("trace segment evidence_refs must be sorted and unique")
        return self


class TracePolicyDecision(ContractModel):
    action: Literal["generate-trace"] = "generate-trace"
    allowed: bool
    policy_source: str = Field(min_length=1)
    policy_sha256: str = Field(pattern=SHA256_PATTERN)
    human_review_required: bool = True
    captured_at: str = Field(min_length=1)


class TraceProducer(ContractModel):
    """Who produced the trace and the reproducibility/policy evidence available at capture time."""

    kind: Literal["human", "model", "imported", "observed"]
    tool: str = Field(min_length=1)
    tool_version: str | None = Field(default=None, min_length=1)
    backend: str | None = Field(default=None, min_length=1)
    provider_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    provider_kind: str | None = Field(default=None, min_length=1)
    requested_model_id: str | None = Field(default=None, min_length=1)
    model_id: str | None = Field(default=None, min_length=1)
    route_id: str | None = Field(default=None, min_length=1)
    model_ref: Ref | None = None
    prompt_template_version: str | None = Field(default=None, min_length=1)
    prompt_template_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    request_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    response_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    response_metadata: JsonObject = Field(default_factory=dict)
    decoding: JsonObject = Field(default_factory=dict)
    seed: int | None = None
    policy_decision: TracePolicyDecision | None = None
    policy_snapshot: JsonObject | None = None

    @model_validator(mode="after")
    def _validate_producer(self) -> TraceProducer:
        if self.kind == "model":
            required = {
                "backend": self.backend,
                "provider_id": self.provider_id,
                "requested_model_id": self.requested_model_id,
                "model_id": self.model_id,
                "prompt_template_sha256": self.prompt_template_sha256,
                "request_sha256": self.request_sha256,
                "response_sha256": self.response_sha256,
                "policy_decision": self.policy_decision,
                "policy_snapshot": self.policy_snapshot,
            }
            missing = sorted(
                name
                for name, value in required.items()
                if value is None or (isinstance(value, str) and not value.strip())
            )
            if missing:
                raise ValueError(f"model trace producers require: {', '.join(missing)}")
            assert self.policy_decision is not None
            if not self.policy_decision.allowed:
                raise ValueError("model-produced trace records require an allowed generation policy")
            if not self.policy_decision.human_review_required:
                raise ValueError("model-produced trace records must require human review")
        elif self.policy_decision is not None or self.policy_snapshot is not None:
            raise ValueError("provider policy evidence is only valid for model trace producers")
        return self


class TraceValidationFinding(ContractModel):
    code: str = Field(pattern=r"^[a-z0-9_]+$")
    severity: Literal["warning", "block"]
    location: str = Field(min_length=1)
    message: str = Field(min_length=1)


class TraceValidationEvidence(ContractModel):
    validator: str = Field(min_length=1)
    validator_version: str = Field(min_length=1)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    checked_at: str = Field(min_length=1)
    status: Literal["pass", "warn", "block"]
    findings: list[TraceValidationFinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_findings(self) -> TraceValidationEvidence:
        keys = [(item.code, item.location, item.message) for item in self.findings]
        if keys != sorted(set(keys)):
            raise ValueError("trace validation findings must be sorted and unique")
        severities = {item.severity for item in self.findings}
        expected = "block" if "block" in severities else "warn" if "warning" in severities else "pass"
        if self.status != expected:
            raise ValueError("trace validation status must match the strongest finding")
        return self


class TraceReview(ContractModel):
    status: Literal["pending", "approved", "rejected"] = "pending"
    reviewer: str | None = None
    reviewed_at: str | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_review(self) -> TraceReview:
        if self.notes != sorted(set(self.notes)):
            raise ValueError("trace review notes must be sorted and unique")
        if self.status == "pending":
            if self.reviewer is not None or self.reviewed_at is not None:
                raise ValueError("pending trace reviews cannot identify a reviewer or review time")
        elif not self.reviewer or not self.reviewer.strip() or not self.reviewed_at:
            raise ValueError("approved/rejected trace reviews require reviewer and reviewed_at")
        return self


class TraceRecord(ContractModel):
    """A hash-sealed reasoning/tool/process record whose review gate is separate from validation.

    Heuristic validation and human approval do not promote generated reasoning to ground truth.
    Model/imported reasoning segments must remain explicitly ``unverified``.
    """

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    trace_id: str = Field(pattern=_ID)
    trace_hash: str = Field(pattern=SHA256_PATTERN)
    created_at: str = Field(min_length=1)
    trace_kind: Literal[
        "reasoning", "tool_use", "agent", "process_supervision", "verifier", "mixed"
    ] = "reasoning"
    source: TraceSource
    context: list[TraceMessage] = Field(min_length=1)
    segments: list[TraceSegment] = Field(min_length=1)
    producer: TraceProducer
    validation: TraceValidationEvidence
    review: TraceReview = Field(default_factory=TraceReview)
    parent_trace_refs: list[Ref] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_trace_record(self) -> TraceRecord:
        context_ids = [item.message_id for item in self.context]
        if len(context_ids) != len(set(context_ids)):
            raise ValueError("trace context message_ids must be unique")

        sequences = [item.sequence for item in self.segments]
        if sequences != list(range(len(self.segments))):
            raise ValueError("trace segment sequence must be contiguous and ordered from zero")
        segment_ids = [item.segment_id for item in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("trace segment_ids must be unique")
        answers = [item for item in self.segments if item.kind == "final_answer"]
        if len(answers) != 1 or self.segments[-1].kind != "final_answer":
            raise ValueError("trace records require exactly one final_answer as the last segment")

        calls: dict[str, int] = {}
        resolved: set[str] = set()
        for segment in self.segments:
            if segment.tool_call is not None:
                call_id = segment.tool_call.call_id
                if call_id in calls:
                    raise ValueError("trace tool call_ids must be unique")
                calls[call_id] = segment.sequence
            if segment.tool_result is not None:
                call_id = segment.tool_result.call_id
                if call_id not in calls or calls[call_id] >= segment.sequence:
                    raise ValueError("every tool result must follow its matching tool call")
                if call_id in resolved:
                    raise ValueError("each tool call can have at most one result")
                resolved.add(call_id)
        if set(calls) != resolved:
            raise ValueError("every trace tool call requires exactly one later result")
        if self.trace_kind == "tool_use" and not calls:
            raise ValueError("tool_use traces require at least one tool call/result pair")

        for segment in self.segments:
            if (
                segment.kind == "reasoning"
                and segment.origin in {"model", "imported"}
                and segment.verification != "unverified"
            ):
                raise ValueError("generated/imported reasoning must remain explicitly unverified")

        parent_keys: list[tuple[str, str]] = []
        for parent in self.parent_trace_refs:
            if not _is_pinned_ref(parent):
                raise ValueError("parent_trace_refs must be hash-pinned")
            assert parent.hash is not None and parent.hash.value is not None
            parent_keys.append((parent.id, parent.hash.value))
        if parent_keys != sorted(set(parent_keys)):
            raise ValueError("parent_trace_refs must be sorted and unique")
        if self.tags != sorted(set(self.tags)):
            raise ValueError("trace tags must be sorted and unique")
        if self.notes != sorted(set(self.notes)):
            raise ValueError("trace notes must be sorted and unique")
        return self


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


class ParameterScope(ContractModel):
    """Stable coordinate universe for an authoritative parameter observation.

    Runtime addresses are never identities. Sparse scopes carry stable expert IDs, and every scope
    is tied to one exact model reference plus a named coordinate universe.
    """

    scope_id: ParameterId
    kind: ParameterScopeKind
    model_ref: Ref
    coordinate_universe_id: ParameterId
    coordinate_universe_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    component_ids: list[ParameterId] = Field(default_factory=list)
    expert_ids: list[ParameterId] = Field(default_factory=list)
    device_id: str | None = None
    memory_tier: MemoryTier | None = None
    definition: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_scope(self) -> ParameterScope:
        if self.component_ids != sorted(set(self.component_ids)):
            raise ValueError("parameter scope component_ids must be sorted and unique")
        if self.expert_ids != sorted(set(self.expert_ids)):
            raise ValueError("parameter scope expert_ids must be sorted and unique")
        if self.kind in {ParameterScopeKind.expert_group, ParameterScopeKind.expert_set}:
            if not self.expert_ids or self.coordinate_universe_sha256 is None:
                raise ValueError(
                    "expert parameter scopes require stable expert IDs and a coordinate-universe hash"
                )
        if self.kind == ParameterScopeKind.device_residency and (
            not self.device_id or self.memory_tier is None
        ):
            raise ValueError("device-residency scopes require device_id and memory_tier")
        return self


class ParameterWindow(ContractModel):
    """The exact computation or scheduling window a count describes."""

    window_id: ParameterId
    kind: ParameterWindowKind
    definition: str = Field(min_length=1)
    plan_ref: Ref | None = None
    run_ref: Ref | None = None
    sequence_id: ParameterId | None = None
    token_index: int | None = Field(default=None, ge=0)
    event_seq_start: int | None = Field(default=None, ge=0)
    event_seq_end: int | None = Field(default=None, ge=0)
    microstep_start: int | None = Field(default=None, ge=0)
    microstep_end: int | None = Field(default=None, ge=0)
    optimizer_step_start: int | None = Field(default=None, ge=0)
    optimizer_step_end: int | None = Field(default=None, ge=0)
    captured_at: str | None = None

    @model_validator(mode="after")
    def _validate_window(self) -> ParameterWindow:
        anchor_present = self.plan_ref is not None or self.run_ref is not None
        if self.kind != ParameterWindowKind.static_snapshot and not anchor_present:
            raise ValueError("dynamic parameter windows require a plan_ref or run_ref")
        if self.kind == ParameterWindowKind.token and (
            self.sequence_id is None or self.token_index is None
        ):
            raise ValueError("token windows require sequence_id and token_index")
        if self.kind == ParameterWindowKind.sequence and self.sequence_id is None:
            raise ValueError("sequence windows require sequence_id")
        if self.kind == ParameterWindowKind.microbatch and self.microstep_start is None:
            raise ValueError("microbatch windows require microstep_start")
        if self.kind == ParameterWindowKind.optimizer_window and (
            self.optimizer_step_start is None or self.optimizer_step_end is None
        ):
            raise ValueError("optimizer windows require optimizer step bounds")
        if self.kind == ParameterWindowKind.run and self.run_ref is None:
            raise ValueError("run windows require run_ref")
        for start, end, label in (
            (self.event_seq_start, self.event_seq_end, "event sequence"),
            (self.microstep_start, self.microstep_end, "microstep"),
            (self.optimizer_step_start, self.optimizer_step_end, "optimizer step"),
        ):
            if end is not None and start is None:
                raise ValueError(f"{label} end requires a start")
            if start is not None and end is not None and end < start:
                raise ValueError(f"{label} end cannot precede its start")
        return self


class ParameterEvidenceSource(ContractModel):
    kind: ParameterEvidenceSourceKind
    producer: ParameterId
    producer_version: str = Field(min_length=1)
    method: str = Field(min_length=1)
    captured_at: str | None = None
    source_ref: Ref
    environment_ref: Ref | None = None
    backend_ref: Ref | None = None


class ParameterObservation(ContractModel):
    """One evidence-bearing parameter count. Unknown evidence is represented as a gap, never zero."""

    observation_id: ParameterId
    kind: ParameterCountKind
    value: int = Field(ge=0)
    unit: Literal["coordinates", "elements", "parameters"] = "coordinates"
    scope: ParameterScope
    window: ParameterWindow
    evidence: Literal[
        EvidenceKind.measured,
        EvidenceKind.estimated,
        EvidenceKind.declared,
    ]
    source: ParameterEvidenceSource
    coverage: ParameterObservationCoverage
    value_relation: ParameterValueRelation
    identity_basis: ParameterIdentityBasis
    handling: ParameterCountHandling
    definition: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("evidence", mode="before")
    @classmethod
    def _known_evidence(cls, value: object) -> object:
        if value == EvidenceKind.unknown or value == EvidenceKind.unknown.value:
            raise ValueError("unknown parameter evidence must be represented as a gap")
        return value

    @model_validator(mode="after")
    def _validate_observation(self) -> ParameterObservation:
        if self.kind != ParameterCountKind.effective and self.unit != "coordinates":
            raise ValueError("authoritative parameter axes use coordinate units")
        if (
            self.kind == ParameterCountKind.resident
            and self.scope.kind != ParameterScopeKind.device_residency
        ):
            raise ValueError("resident observations require a device-residency scope")
        if self.coverage != ParameterObservationCoverage.complete and (
            self.value_relation == ParameterValueRelation.exact
        ):
            raise ValueError("partial or sampled observations cannot claim an exact value")
        if (
            self.evidence == EvidenceKind.estimated
            and self.value_relation == ParameterValueRelation.exact
        ):
            raise ValueError("estimated parameter evidence cannot claim an exact value")
        if self.assumptions != sorted(set(self.assumptions)):
            raise ValueError("parameter observation assumptions must be sorted and unique")
        if self.evidence == EvidenceKind.measured:
            if self.source.captured_at is None or not _is_pinned_ref(
                self.source.source_ref
            ):
                raise ValueError(
                    "measured parameter evidence requires capture time and a hash-pinned source"
                )
        required_windows: dict[ParameterCountKind, set[ParameterWindowKind]] = {
            ParameterCountKind.logical: {ParameterWindowKind.static_snapshot},
            ParameterCountKind.active_token: {ParameterWindowKind.token},
            ParameterCountKind.active_sequence: {ParameterWindowKind.sequence},
            ParameterCountKind.touched_window: {
                ParameterWindowKind.token,
                ParameterWindowKind.sequence,
                ParameterWindowKind.microbatch,
                ParameterWindowKind.optimizer_window,
                ParameterWindowKind.run,
            },
            ParameterCountKind.resident: {ParameterWindowKind.instant},
            ParameterCountKind.updated_window: {
                ParameterWindowKind.optimizer_window,
                ParameterWindowKind.run,
            },
            ParameterCountKind.exposed_window: {
                ParameterWindowKind.token,
                ParameterWindowKind.sequence,
                ParameterWindowKind.microbatch,
                ParameterWindowKind.optimizer_window,
                ParameterWindowKind.run,
            },
        }
        accepted = required_windows.get(self.kind)
        if accepted is not None and self.window.kind not in accepted:
            raise ValueError(
                f"{self.kind.value} observations require one of "
                f"{sorted(item.value for item in accepted)} windows"
            )
        if (
            self.evidence == EvidenceKind.measured
            and self.window.kind == ParameterWindowKind.instant
            and self.window.captured_at is None
        ):
            raise ValueError("measured instant observations require window captured_at")
        return self


class ParameterEvidenceGap(ContractModel):
    gap_id: ParameterId
    kind: ParameterCountKind
    scope: ParameterScope
    window: ParameterWindow
    reason: ParameterGapReason
    explanation: str = Field(min_length=1)
    resolution: str = Field(min_length=1)


class ParameterConflict(ContractModel):
    conflict_id: ParameterId
    observation_ids: list[ParameterId] = Field(min_length=2)
    reason_code: ParameterId
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_conflict(self) -> ParameterConflict:
        if self.observation_ids != sorted(set(self.observation_ids)):
            raise ValueError("parameter conflict observation_ids must be sorted and unique")
        return self


_PARAMETER_REQUIRED_KINDS: dict[ParameterAccountingProfile, set[ParameterCountKind]] = {
    ParameterAccountingProfile.model_static: {ParameterCountKind.logical},
    ParameterAccountingProfile.training_plan: {
        ParameterCountKind.logical,
        ParameterCountKind.active_token,
        ParameterCountKind.touched_window,
        ParameterCountKind.resident,
        ParameterCountKind.updated_window,
        ParameterCountKind.exposed_window,
    },
    ParameterAccountingProfile.training_runtime: {
        ParameterCountKind.logical,
        ParameterCountKind.active_token,
        ParameterCountKind.touched_window,
        ParameterCountKind.resident,
        ParameterCountKind.updated_window,
        ParameterCountKind.exposed_window,
    },
    ParameterAccountingProfile.inference_runtime: {
        ParameterCountKind.logical,
        ParameterCountKind.active_token,
        ParameterCountKind.touched_window,
        ParameterCountKind.resident,
    },
    ParameterAccountingProfile.checkpoint: {
        ParameterCountKind.logical,
        ParameterCountKind.updated_window,
    },
    ParameterAccountingProfile.evaluation: {
        ParameterCountKind.logical,
        ParameterCountKind.active_sequence,
        ParameterCountKind.touched_window,
        ParameterCountKind.resident,
    },
}


def required_parameter_kinds(
    profile: ParameterAccountingProfile,
) -> set[ParameterCountKind]:
    """Return a copy of the minimum evidence axes for one accounting profile."""

    return set(_PARAMETER_REQUIRED_KINDS[profile])


class ParameterAccountingReport(ContractModel):
    """Hash-sealed, auditable reconciliation of parameter evidence for one exact model context."""

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    report_id: ParameterId
    report_hash: str = Field(pattern=SHA256_PATTERN)
    generated_at: str
    profile: ParameterAccountingProfile
    status: ParameterAccountingStatus
    model_ref: Ref
    plan_ref: Ref | None = None
    run_ref: Ref | None = None
    artifact_refs: list[Ref] = Field(default_factory=list)
    evaluation_refs: list[Ref] = Field(default_factory=list)
    parent_report_refs: list[Ref] = Field(default_factory=list)
    observations: list[ParameterObservation] = Field(default_factory=list)
    gaps: list[ParameterEvidenceGap] = Field(default_factory=list)
    conflicts: list[ParameterConflict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @staticmethod
    def _ref_key(ref: Ref) -> tuple[str, str, str]:
        return (
            ref.id,
            ref.hash.algo if ref.hash is not None else "",
            (ref.hash.value or "") if ref.hash is not None else "",
        )

    @staticmethod
    def _scope_key(observation: ParameterObservation) -> tuple[str, str, str, str]:
        scope = observation.scope
        return (
            scope.model_ref.id,
            scope.coordinate_universe_id,
            scope.coordinate_universe_sha256 or "",
            scope.scope_id,
        )

    @staticmethod
    def _handling_complete(observation: ParameterObservation) -> bool:
        return CountHandling.unknown not in {
            observation.handling.tied,
            observation.handling.shared,
            observation.handling.replicated,
            observation.handling.generated,
            observation.handling.quantized,
            observation.handling.optimizer_shadows,
            observation.handling.decompressed_caches,
        }

    def _qualifies(self, observation: ParameterObservation) -> bool:
        if not self._handling_complete(observation):
            return False
        if observation.identity_basis in {
            ParameterIdentityBasis.unknown,
            ParameterIdentityBasis.stored_tensor_elements,
        }:
            return False
        runtime_profiles = {
            ParameterAccountingProfile.training_runtime,
            ParameterAccountingProfile.inference_runtime,
            ParameterAccountingProfile.checkpoint,
            ParameterAccountingProfile.evaluation,
        }
        if self.profile in runtime_profiles and observation.kind != ParameterCountKind.logical:
            return (
                observation.evidence == EvidenceKind.measured
                and observation.coverage == ParameterObservationCoverage.complete
                and observation.value_relation == ParameterValueRelation.exact
            )
        if self.profile == ParameterAccountingProfile.training_plan:
            return (
                observation.coverage == ParameterObservationCoverage.complete
                and observation.value_relation
                in {ParameterValueRelation.exact, ParameterValueRelation.estimate}
            )
        if self.profile == ParameterAccountingProfile.model_static:
            return (
                observation.evidence == EvidenceKind.measured
                and observation.coverage == ParameterObservationCoverage.complete
                and observation.value_relation == ParameterValueRelation.exact
            )
        return (
            observation.coverage == ParameterObservationCoverage.complete
            and observation.value_relation == ParameterValueRelation.exact
        )

    @model_validator(mode="after")
    def _validate_accounting_report(self) -> ParameterAccountingReport:
        for refs, label in (
            (self.artifact_refs, "artifact_refs"),
            (self.evaluation_refs, "evaluation_refs"),
            (self.parent_report_refs, "parent_report_refs"),
        ):
            keys = [self._ref_key(ref) for ref in refs]
            if keys != sorted(set(keys)):
                raise ValueError(f"{label} must be sorted and unique")
        observation_ids = [item.observation_id for item in self.observations]
        if observation_ids != sorted(observation_ids) or len(observation_ids) != len(
            set(observation_ids)
        ):
            raise ValueError("parameter observations must be sorted by unique observation_id")
        gap_ids = [item.gap_id for item in self.gaps]
        if gap_ids != sorted(gap_ids) or len(gap_ids) != len(set(gap_ids)):
            raise ValueError("parameter gaps must be sorted by unique gap_id")
        conflict_ids = [item.conflict_id for item in self.conflicts]
        if conflict_ids != sorted(conflict_ids) or len(conflict_ids) != len(set(conflict_ids)):
            raise ValueError("parameter conflicts must be sorted by unique conflict_id")
        if self.notes != sorted(set(self.notes)):
            raise ValueError("parameter accounting notes must be sorted and unique")
        scopes_by_id: dict[str, ParameterScope] = {}
        windows_by_id: dict[str, ParameterWindow] = {}

        def register_evidence_shape(scope: ParameterScope, window: ParameterWindow) -> None:
            if scope.model_ref != self.model_ref:
                raise ValueError("every parameter scope must reference the report model")
            previous_scope = scopes_by_id.get(scope.scope_id)
            if previous_scope is not None and previous_scope.model_dump(
                exclude={"definition"}
            ) != scope.model_dump(exclude={"definition"}):
                raise ValueError("a parameter scope_id must have one definition per report")
            scopes_by_id[scope.scope_id] = scope
            previous_window = windows_by_id.get(window.window_id)
            if previous_window is not None and previous_window.model_dump(
                exclude={"definition"}
            ) != window.model_dump(exclude={"definition"}):
                raise ValueError("a parameter window_id must have one definition per report")
            windows_by_id[window.window_id] = window

        for observation in self.observations:
            register_evidence_shape(observation.scope, observation.window)
        for gap in self.gaps:
            register_evidence_shape(gap.scope, gap.window)
        if self.profile == ParameterAccountingProfile.training_plan and self.plan_ref is None:
            raise ValueError("training-plan accounting requires plan_ref")
        if self.profile in {
            ParameterAccountingProfile.training_runtime,
            ParameterAccountingProfile.inference_runtime,
        } and self.run_ref is None:
            raise ValueError("runtime accounting requires run_ref")
        if self.profile == ParameterAccountingProfile.checkpoint and (
            not self.artifact_refs or not all(_is_pinned_ref(ref) for ref in self.artifact_refs)
        ):
            raise ValueError("checkpoint accounting requires hash-pinned artifact_refs")
        if self.profile == ParameterAccountingProfile.evaluation and (
            not self.evaluation_refs or not all(_is_pinned_ref(ref) for ref in self.evaluation_refs)
        ):
            raise ValueError("evaluation accounting requires hash-pinned evaluation_refs")
        if not all(_is_pinned_ref(ref) for ref in self.parent_report_refs):
            raise ValueError("parent_report_refs must be hash-pinned")

        runtime_profiles = {
            ParameterAccountingProfile.training_runtime,
            ParameterAccountingProfile.inference_runtime,
            ParameterAccountingProfile.checkpoint,
            ParameterAccountingProfile.evaluation,
        }
        def validate_window_anchor(window: ParameterWindow) -> None:
            if window.kind == ParameterWindowKind.static_snapshot:
                return
            if self.profile in runtime_profiles and window.run_ref != self.run_ref:
                raise ValueError(
                    "dynamic runtime evidence must reference the report run_ref"
                )
            if (
                self.profile == ParameterAccountingProfile.training_plan
                and window.plan_ref != self.plan_ref
            ):
                raise ValueError(
                    "dynamic training-plan evidence must reference the report plan_ref"
                )

        for observation in self.observations:
            validate_window_anchor(observation.window)
        for gap in self.gaps:
            if gap.window.kind == ParameterWindowKind.static_snapshot:
                continue
            validate_window_anchor(gap.window)

        known_observations = set(observation_ids)
        for conflict in self.conflicts:
            if not set(conflict.observation_ids).issubset(known_observations):
                raise ValueError("parameter conflicts must reference observations in the report")

        gaps_by_kind = {item.kind for item in self.gaps}
        for kind in required_parameter_kinds(self.profile):
            candidates = [item for item in self.observations if item.kind == kind]
            if not any(self._qualifies(item) for item in candidates) and kind not in gaps_by_kind:
                raise ValueError(f"unproven required parameter axis '{kind.value}' needs a gap")

        conflict_sets = [set(item.observation_ids) for item in self.conflicts]
        exact_groups: dict[tuple[object, ...], list[ParameterObservation]] = {}
        for observation in self.observations:
            if (
                observation.coverage == ParameterObservationCoverage.complete
                and observation.value_relation == ParameterValueRelation.exact
            ):
                key = (
                    observation.kind.value,
                    *self._scope_key(observation),
                    observation.window.window_id,
                    observation.unit,
                )
                exact_groups.setdefault(key, []).append(observation)
        required_pairs: set[frozenset[str]] = set()
        for group in exact_groups.values():
            if len({item.value for item in group}) > 1:
                for left in group:
                    for right in group:
                        if left.value != right.value:
                            required_pairs.add(frozenset({left.observation_id, right.observation_id}))

        logical_by_scope: dict[tuple[str, str, str, str], list[ParameterObservation]] = {}
        for observation in self.observations:
            if (
                observation.kind == ParameterCountKind.logical
                and observation.coverage == ParameterObservationCoverage.complete
                and observation.value_relation == ParameterValueRelation.exact
                and observation.unit == "coordinates"
            ):
                logical_by_scope.setdefault(self._scope_key(observation), []).append(observation)
        for observation in self.observations:
            if (
                observation.kind in {
                    ParameterCountKind.active_token,
                    ParameterCountKind.active_sequence,
                    ParameterCountKind.touched_window,
                    ParameterCountKind.resident,
                    ParameterCountKind.updated_window,
                    ParameterCountKind.exposed_window,
                }
                and observation.coverage == ParameterObservationCoverage.complete
                and observation.value_relation == ParameterValueRelation.exact
                and observation.unit == "coordinates"
            ):
                for logical in logical_by_scope.get(self._scope_key(observation), []):
                    if observation.value > logical.value:
                        required_pairs.add(
                            frozenset({observation.observation_id, logical.observation_id})
                        )

        for left in self.observations:
            for right in self.observations:
                if not all(
                    item.coverage == ParameterObservationCoverage.complete
                    and item.value_relation == ParameterValueRelation.exact
                    and item.unit == "coordinates"
                    for item in (left, right)
                ):
                    continue
                if self._scope_key(left) != self._scope_key(right):
                    continue
                if left.window.window_id != right.window.window_id:
                    continue
                if (
                    left.kind == ParameterCountKind.updated_window
                    and right.kind == ParameterCountKind.touched_window
                    and left.value > right.value
                ):
                    required_pairs.add(frozenset({left.observation_id, right.observation_id}))
        token_observations = [
            item
            for item in self.observations
            if item.kind == ParameterCountKind.active_token
            and item.coverage == ParameterObservationCoverage.complete
            and item.value_relation == ParameterValueRelation.exact
            and item.unit == "coordinates"
        ]
        sequence_observations = [
            item
            for item in self.observations
            if item.kind == ParameterCountKind.active_sequence
            and item.coverage == ParameterObservationCoverage.complete
            and item.value_relation == ParameterValueRelation.exact
            and item.unit == "coordinates"
        ]
        for token in token_observations:
            for sequence in sequence_observations:
                if (
                    self._scope_key(token) == self._scope_key(sequence)
                    and token.window.sequence_id == sequence.window.sequence_id
                    and token.value > sequence.value
                ):
                    required_pairs.add(frozenset({token.observation_id, sequence.observation_id}))
        for pair in required_pairs:
            if not any(pair.issubset(conflict_set) for conflict_set in conflict_sets):
                raise ValueError(
                    "conflicting or algebraically impossible observations require a conflict record"
                )

        expected_status = (
            ParameterAccountingStatus.conflicting
            if self.conflicts
            else ParameterAccountingStatus.incomplete
            if self.gaps
            else ParameterAccountingStatus.complete
        )
        if self.status != expected_status:
            raise ValueError(f"parameter accounting status must be {expected_status.value}")
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
    details: JsonObject = Field(default_factory=dict)


class TopologyInspection(ContractModel):
    """Evidence for a bounded static topology classification.

    This is config metadata evidence only. It cannot authorize model code, prove that a backend can
    load the snapshot, or claim inference/training/hardware support.
    """

    status: Literal[
        "not_checked",
        "no_recognized_moe_evidence",
        "detected",
        "incomplete",
        "unsupported_family",
    ] = "not_checked"
    method: Literal["not_checked", "static_config_v1"] = "not_checked"
    family: str | None = None
    config_file: str | None = None
    config_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    evidence_paths: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence_level: Literal["not_checked", "static_metadata_only"] = "not_checked"
    runtime_capability: Literal["unverified"] = "unverified"

    @field_validator("config_file")
    @classmethod
    def _portable_config_file(cls, value: str | None) -> str | None:
        return _validate_descriptor_path(value) if value is not None else None

    @model_validator(mode="after")
    def _validate_inspection(self) -> TopologyInspection:
        if self.evidence_paths != sorted(set(self.evidence_paths)):
            raise ValueError("topology evidence_paths must be sorted and unique")
        if self.warnings != sorted(set(self.warnings)):
            raise ValueError("topology warnings must be sorted and unique")
        if self.method == "not_checked":
            if (
                self.status != "not_checked"
                or any(
                    (
                        self.family,
                        self.config_file,
                        self.config_sha256,
                        self.evidence_paths,
                        self.warnings,
                    )
                )
                or self.evidence_level != "not_checked"
            ):
                raise ValueError("not-checked topology inspection cannot carry static evidence")
            return self
        if self.status == "not_checked":
            raise ValueError("static topology inspection requires an explicit result status")
        if self.evidence_level != "static_metadata_only":
            raise ValueError("static topology inspection must be labeled static_metadata_only")
        if self.config_file is None or self.config_sha256 is None:
            raise ValueError("static topology inspection requires a hash-pinned config file")
        if self.status in {"detected", "incomplete", "unsupported_family"} and not self.family:
            raise ValueError(f"{self.status} topology inspection requires a model family")
        if self.status == "detected" and not self.evidence_paths:
            raise ValueError("detected topology requires config evidence paths")
        return self


class ExpertGroup(ContractModel):
    """One routed-expert layout repeated at each listed layer.

    expert_count is the total logical expert identities per listed layer. routed_expert_count and
    always-active shared_expert_count partition that total. experts_per_token counts only routed
    experts selected per token; neither it nor any other field here is a parameter count.
    """

    group_id: str = Field(pattern=_ID)
    layer_namespace: str = Field(default="decoder", pattern=_ID)
    component_path: str | None = Field(default=None, min_length=1)
    layer_indices: list[int] = Field(default_factory=list)
    expert_count: int = Field(ge=1)
    routed_expert_count: int | None = Field(default=None, ge=1)
    experts_per_token: int | None = Field(default=None, ge=1)
    shared_expert_count: int | None = Field(default=None, ge=0)
    heterogeneous: bool = False
    expert_identity_scheme: str | None = None
    expert_registry_ref: Ref | None = None
    metadata_sources: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_expert_counts(cls, value: Any) -> Any:
        """Accept pre-Phase-8 expert groups and normalize their implicit routed partition."""

        if not isinstance(value, Mapping):
            return value
        migrated = dict(value)
        shared = migrated.get("shared_expert_count")
        if shared is None:
            shared = 0
            migrated["shared_expert_count"] = shared
        if migrated.get("routed_expert_count") is None:
            total = migrated.get("expert_count")
            if (
                isinstance(total, int)
                and not isinstance(total, bool)
                and isinstance(shared, int)
                and not isinstance(shared, bool)
            ):
                migrated["routed_expert_count"] = total - shared
        return migrated

    @model_validator(mode="after")
    def _validate_expert_counts(self) -> ExpertGroup:
        if self.routed_expert_count is None or self.shared_expert_count is None:
            raise ValueError("expert groups require a routed/shared expert partition")
        if self.expert_count != self.routed_expert_count + self.shared_expert_count:
            raise ValueError("expert_count must equal routed plus shared expert counts")
        if self.experts_per_token is not None and self.experts_per_token > self.routed_expert_count:
            raise ValueError("experts_per_token cannot exceed routed_expert_count")
        if self.layer_indices != sorted(set(self.layer_indices)):
            raise ValueError("layer_indices must be sorted and unique")
        if self.metadata_sources != sorted(set(self.metadata_sources)):
            raise ValueError("expert metadata_sources must be sorted and unique")
        return self


class ExpertTopologyCounts(ContractModel):
    """Derived structural counts across one full model pass for one token.

    The unit is expert instances, not parameter coordinates. These values therefore never substitute
    for N_logical/N_active in ParameterAccountingReport.
    """

    unit: Literal["expert_instances"] = "expert_instances"
    moe_layer_count: int = Field(ge=1)
    routed_expert_instances: int = Field(ge=1)
    shared_expert_instances: int = Field(ge=0)
    logical_expert_instances: int = Field(ge=1)
    active_routed_expert_instances_per_token: int = Field(ge=1)
    active_shared_expert_instances_per_token: int = Field(ge=0)
    active_expert_instances_per_token: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_totals(self) -> ExpertTopologyCounts:
        if self.logical_expert_instances != (
            self.routed_expert_instances + self.shared_expert_instances
        ):
            raise ValueError("logical expert instances must equal routed plus shared instances")
        if self.active_expert_instances_per_token != (
            self.active_routed_expert_instances_per_token
            + self.active_shared_expert_instances_per_token
        ):
            raise ValueError(
                "active expert instances must equal active routed plus shared instances"
            )
        if self.active_routed_expert_instances_per_token > self.routed_expert_instances:
            raise ValueError(
                "active routed expert instances cannot exceed logical routed instances"
            )
        if self.active_shared_expert_instances_per_token > self.shared_expert_instances:
            raise ValueError(
                "active shared expert instances cannot exceed logical shared instances"
            )
        return self


class ModelTopology(ContractModel):
    execution_kind: ModelExecutionKind = ModelExecutionKind.unknown
    semantic_routing: SemanticRouting | None = None
    expert_groups: list[ExpertGroup] = Field(default_factory=list)
    expert_counts: ExpertTopologyCounts | None = None
    inspection: TopologyInspection = Field(default_factory=TopologyInspection)
    # Placement/prefetch/residency decisions belong to the RunPlan physical-execution layer.
    physical_scheduler_owner: Literal["run_plan"] = "run_plan"

    @model_validator(mode="after")
    def _validate_topology(self) -> ModelTopology:
        group_ids = [item.group_id for item in self.expert_groups]
        if group_ids != sorted(group_ids) or len(group_ids) != len(set(group_ids)):
            raise ValueError("expert_groups must be sorted by unique group_id")
        if self.execution_kind in {ModelExecutionKind.dense, ModelExecutionKind.unknown} and (
            self.semantic_routing is not None
            or self.expert_groups
            or self.expert_counts is not None
        ):
            raise ValueError("dense/unknown topology cannot declare expert routing or counts")
        if self.semantic_routing is not None and self.semantic_routing.top_k is not None:
            if any(
                group.experts_per_token != self.semantic_routing.top_k
                for group in self.expert_groups
            ):
                raise ValueError("global router top_k must match every expert group")
        if self.inspection.status == "detected":
            if self.execution_kind not in {
                ModelExecutionKind.sparse,
                ModelExecutionKind.mixture_of_experts,
                ModelExecutionKind.conditional,
                ModelExecutionKind.hybrid,
            }:
                raise ValueError("detected expert topology requires an expert-bearing execution kind")
            if (
                self.semantic_routing is None
                or not self.expert_groups
                or self.expert_counts is None
                or any(group.component_path is None for group in self.expert_groups)
            ):
                raise ValueError(
                    "detected MoE topology requires routing, component-scoped groups, and counts"
                )
        elif (
            self.inspection.method == "static_config_v1"
            and self.execution_kind != ModelExecutionKind.unknown
        ):
            raise ValueError("non-detected static inspection must leave execution_kind unknown")
        if self.expert_counts is not None:
            if not self.expert_groups or any(
                not group.layer_indices or group.experts_per_token is None
                for group in self.expert_groups
            ):
                raise ValueError("expert counts require complete layer and top-k group structure")
            layers = {
                (group.layer_namespace, layer)
                for group in self.expert_groups
                for layer in group.layer_indices
            }
            routed = sum(
                len(group.layer_indices) * int(group.routed_expert_count or 0)
                for group in self.expert_groups
            )
            shared = sum(
                len(group.layer_indices) * int(group.shared_expert_count or 0)
                for group in self.expert_groups
            )
            logical = sum(
                len(group.layer_indices) * group.expert_count for group in self.expert_groups
            )
            active_routed = sum(
                len(group.layer_indices) * int(group.experts_per_token or 0)
                for group in self.expert_groups
            )
            active_shared = shared
            expected = (
                len(layers),
                routed,
                shared,
                logical,
                active_routed,
                active_shared,
                active_routed + active_shared,
            )
            actual = (
                self.expert_counts.moe_layer_count,
                self.expert_counts.routed_expert_instances,
                self.expert_counts.shared_expert_instances,
                self.expert_counts.logical_expert_instances,
                self.expert_counts.active_routed_expert_instances_per_token,
                self.expert_counts.active_shared_expert_instances_per_token,
                self.expert_counts.active_expert_instances_per_token,
            )
            if actual != expected:
                raise ValueError("expert topology counts do not match the declared expert groups")
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
        if (
            self.parameters.kind != ModelExecutionKind.unknown
            and self.topology.execution_kind != ModelExecutionKind.unknown
            and self.parameters.kind != self.topology.execution_kind
        ) or (
            self.topology.inspection.status == "detected"
            and self.parameters.kind != self.topology.execution_kind
        ):
            raise ValueError("parameter representation kind must match topology execution_kind")
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
    default_weight: float | None = Field(default=None, ge=0)
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
    separate RunPlan physical-execution responsibilities.
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
    # >=12 means Blackwell. Native-Windows/WDDM forces math; other platforms remain probe-gated.
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
    attention_kernels: list[AttentionKernel] = Field(default_factory=list)
    loss_impls: list[LossImpl] = Field(default_factory=list)
    # Semantic objective capabilities (for example causal_lm_sft or adapter_qlora). This is still a
    # STATIC declaration; only the matching field in EffectiveCapabilities can prove it on a host.
    objective_capabilities: list[ObjectiveCapability] = Field(default_factory=list)
    checkpoint_impls: list[CheckpointImpl] = Field(default_factory=list)
    optimizers: list[Optimizer] = Field(default_factory=list)
    execution_contract_versions: list[str] = Field(default_factory=list)
    trainer_fields: list[str] = Field(default_factory=list)
    trainer_init_fields: list[str] = Field(default_factory=list)
    offload_strategies: list[OffloadStrategy] = Field(default_factory=list)
    placement_tiers: list[MemoryTier] = Field(default_factory=list)
    placement_modes: list[PlacementMode] = Field(default_factory=list)
    parallelism_kinds: list[ParallelismKind] = Field(default_factory=list)
    communication_backends: list[CommunicationBackend] = Field(default_factory=list)
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

    @field_validator(
        "attention_kernels",
        "offload_strategies",
        "placement_tiers",
        "placement_modes",
        "parallelism_kinds",
        "communication_backends",
    )
    @classmethod
    def _sorted_physical_capabilities(cls, values: list[Any]) -> list[Any]:
        if values != sorted(set(values), key=lambda item: item.value):
            raise ValueError("physical capability lists must be sorted and unique")
        return values

    @field_validator(
        "execution_contract_versions",
        "trainer_fields",
        "trainer_init_fields",
    )
    @classmethod
    def _sorted_backend_tokens(cls, values: list[str]) -> list[str]:
        if values != sorted(set(values)):
            raise ValueError("backend token lists must be sorted and unique")
        return values


class ExecutionCapabilityCombination(ContractModel):
    """One execution tuple demonstrated together by a bounded functional probe.

    Independent successes on precision, quantization, adapter, optimizer, loss, attention, and
    checkpoint axes are diagnostic only. The planner may seal a run only from one of these complete
    tuples, preventing a union of unrelated probes from becoming a fictional capability.
    """

    runtime_mode: Literal["training", "cpu_toy"]
    device: DeviceKind
    precision: PrecisionMode
    quantization: QuantizationMode
    adapter_method: AdapterMethod
    attention_impl: AttentionImpl
    attention_kernel: AttentionKernel
    optimizer: Optimizer
    loss_impl: LossImpl
    checkpoint_impl: CheckpointImpl
    export_format: ExportFormat
    execution_contract_version: str = Field(pattern=_ID)
    probe: str = Field(pattern=_ID)

    def canonical_key(self) -> tuple[str, ...]:
        return (
            self.runtime_mode,
            self.device.value,
            self.precision.value,
            self.quantization.value,
            self.adapter_method.value,
            self.attention_impl.value,
            self.attention_kernel.value,
            self.optimizer.value,
            self.loss_impl.value,
            self.checkpoint_impl.value,
            self.export_format.value,
            self.execution_contract_version,
            self.probe,
        )


class ProbeResult(ContractModel):
    probe: str
    outcome: FailureTaxonomy
    detail: str | None = None
    measured: JsonObject = Field(default_factory=dict)
    proves: dict[str, list[str]] = Field(default_factory=dict)
    execution_combinations: list[ExecutionCapabilityCombination] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_evidence(self) -> ProbeResult:
        for axis, values in self.proves.items():
            if values != sorted(set(values)):
                raise ValueError(f"probe proof axis {axis!r} must be sorted and unique")
        keys = [item.canonical_key() for item in self.execution_combinations]
        if keys != sorted(set(keys)):
            raise ValueError("probe execution combinations must be sorted and unique")
        if any(item.probe != self.probe for item in self.execution_combinations):
            raise ValueError("an execution combination must name the probe that emitted it")
        if self.outcome != FailureTaxonomy.PASS and (
            self.proves or self.execution_combinations
        ):
            raise ValueError("only a passing probe may carry capability evidence")
        return self


class EffectiveCapabilities(ContractModel):
    """The intersection of what a backend DECLARES and what PROVED to work on this host. The planner
    resolves a RunPlan against this, not the raw BackendManifest."""

    precision_modes: list[PrecisionMode] = Field(default_factory=list)
    quantization_modes: list[QuantizationMode] = Field(default_factory=list)
    attention_impls: list[AttentionImpl] = Field(default_factory=list)
    attention_kernels: list[AttentionKernel] = Field(default_factory=list)
    adapter_methods: list[AdapterMethod] = Field(default_factory=list)
    loss_impls: list[LossImpl] = Field(default_factory=list)
    optimizers: list[Optimizer] = Field(default_factory=list)
    checkpoint_impls: list[CheckpointImpl] = Field(default_factory=list)
    execution_contract_versions: list[str] = Field(default_factory=list)
    execution_combinations: list[ExecutionCapabilityCombination] = Field(default_factory=list)
    trainer_fields: list[str] = Field(default_factory=list)
    trainer_init_fields: list[str] = Field(default_factory=list)
    objective_capabilities: list[ObjectiveCapability] = Field(default_factory=list)
    offload_strategies: list[OffloadStrategy] = Field(default_factory=list)
    placement_tiers: list[MemoryTier] = Field(default_factory=list)
    placement_modes: list[PlacementMode] = Field(default_factory=list)
    parallelism_kinds: list[ParallelismKind] = Field(default_factory=list)
    communication_backends: list[CommunicationBackend] = Field(default_factory=list)

    @field_validator("objective_capabilities")
    @classmethod
    def _sorted_effective_objective_capabilities(cls, values: list[str]) -> list[str]:
        if values != sorted(set(values)):
            raise ValueError("effective objective_capabilities must be sorted and unique")
        return values

    @field_validator(
        "attention_kernels",
        "offload_strategies",
        "placement_tiers",
        "placement_modes",
        "parallelism_kinds",
        "communication_backends",
    )
    @classmethod
    def _sorted_effective_physical_capabilities(cls, values: list[Any]) -> list[Any]:
        if values != sorted(set(values), key=lambda item: item.value):
            raise ValueError("effective physical capability lists must be sorted and unique")
        return values

    @field_validator(
        "execution_contract_versions",
        "trainer_fields",
        "trainer_init_fields",
    )
    @classmethod
    def _sorted_effective_tokens(cls, values: list[str]) -> list[str]:
        if values != sorted(set(values)):
            raise ValueError("effective token lists must be sorted and unique")
        return values

    @field_validator("execution_combinations")
    @classmethod
    def _sorted_execution_combinations(
        cls, values: list[ExecutionCapabilityCombination]
    ) -> list[ExecutionCapabilityCombination]:
        keys = [item.canonical_key() for item in values]
        if keys != sorted(set(keys)):
            raise ValueError("execution combinations must be sorted and unique")
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

    @model_validator(mode="after")
    def _validate_effective_evidence(self) -> CapabilityReport:
        effective = self.effective_capabilities
        # Legacy/imported reports may predate embedded proof payloads. They remain readable for
        # inspection and objective compatibility, but the execution planner separately requires a
        # matching passing ProbeResult for every exact execution combination it consumes.
        if effective is None or not self.probe_results:
            return self
        passing = [item for item in self.probe_results if item.outcome == FailureTaxonomy.PASS]
        proven: dict[str, set[str]] = {}
        combinations: list[ExecutionCapabilityCombination] = []
        for result in passing:
            for axis, values in result.proves.items():
                proven.setdefault(axis, set()).update(values)
            combinations.extend(result.execution_combinations)
        axis_fields = {
            "precision": "precision_modes",
            "quantization": "quantization_modes",
            "attention": "attention_impls",
            "attention_kernel": "attention_kernels",
            "adapter": "adapter_methods",
            "loss": "loss_impls",
            "optimizer": "optimizers",
            "checkpoint": "checkpoint_impls",
            "trainer_field": "trainer_fields",
            "trainer_init_field": "trainer_init_fields",
            "objective": "objective_capabilities",
            "offload": "offload_strategies",
            "placement_tier": "placement_tiers",
            "placement_mode": "placement_modes",
            "parallelism": "parallelism_kinds",
            "communication_backend": "communication_backends",
        }
        for axis, field_name in axis_fields.items():
            actual = [
                value.value if hasattr(value, "value") else value
                for value in getattr(effective, field_name)
            ]
            expected = sorted(proven.get(axis, set()))
            if actual != expected:
                raise ValueError(
                    f"effective {field_name} does not equal passing probe evidence"
                )
        actual_combinations = [item.canonical_key() for item in effective.execution_combinations]
        expected_combinations = sorted(item.canonical_key() for item in combinations)
        if actual_combinations != expected_combinations:
            raise ValueError("effective execution combinations do not equal passing probe evidence")
        trainer_surface_passed = any(item.probe == "trainer_contract" for item in passing)
        expected_contracts = ["1.0.0"] if trainer_surface_passed and combinations else []
        if effective.execution_contract_versions != expected_contracts:
            raise ValueError("execution contract support lacks its conjunctive probe evidence")
        expected_readiness = (
            "ready"
            if any(item.runtime_mode == "training" for item in combinations)
            else "cpu_toy_only"
            if any(item.runtime_mode == "cpu_toy" for item in combinations)
            else "not_ready"
        )
        if self.readiness != expected_readiness:
            raise ValueError("readiness does not match complete execution-combination evidence")
        bnb_passed = any(item.probe == "bnb_4bit_load" for item in passing)
        if self.bitsandbytes_ok != bnb_passed:
            raise ValueError("bitsandbytes_ok does not match the NF4 probe outcome")
        return self


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
    bias: Literal["none", "all", "lora_only"] | None = None


class OptimizerSpec(ContractModel):
    impl: Optimizer
    learning_rate: float = Field(gt=0)
    weight_decay: float | None = Field(default=None, ge=0)
    adam_beta1: float = Field(default=0.9, ge=0, lt=1)
    adam_beta2: float = Field(default=0.999, ge=0, lt=1)
    adam_epsilon: float = Field(default=1e-8, gt=0)
    max_grad_norm: float = Field(default=1.0, ge=0)
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


class ExecutionInputBinding(ContractModel):
    """One immutable input consumed by the worker.

    Local inputs pin the exact bytes (a stable file or directory digest). Hugging Face inputs pin an
    immutable repository commit; a branch or tag is never sufficient execution identity.
    """

    kind: Literal["dataset", "model", "tokenizer"]
    ref: Ref
    source: Literal["local_file", "local_directory", "huggingface"]
    location: str = Field(min_length=1)
    resolved_revision: str | None = Field(
        default=None,
        pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$",
    )
    content_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_immutable_input(self) -> ExecutionInputBinding:
        if not _is_pinned_ref(self.ref):
            raise ValueError("execution input refs must be hash-pinned")
        if self.source == "huggingface":
            if self.kind == "dataset" or self.resolved_revision is None:
                raise ValueError("Hugging Face model/tokenizer inputs require an immutable commit")
        elif self.content_sha256 is None:
            raise ValueError("local execution inputs require a stable content digest")
        if self.kind == "dataset" and self.source != "local_file":
            raise ValueError("the current dense trainer consumes a pinned local dataset file")
        return self


class ExecutionInputs(ContractModel):
    dataset: ExecutionInputBinding
    model: ExecutionInputBinding
    tokenizer: ExecutionInputBinding

    @model_validator(mode="after")
    def _validate_kinds(self) -> ExecutionInputs:
        if (
            self.dataset.kind != "dataset"
            or self.model.kind != "model"
            or self.tokenizer.kind != "tokenizer"
        ):
            raise ValueError("execution input slots must carry their matching input kind")
        return self


class PrecisionExecutionPolicy(ContractModel):
    """The numerical representation of each material training state.

    ``weight_storage_dtype`` describes an unquantized frozen base; quantized bases use
    ``quantized_storage_format`` instead. ``master_weight_dtype`` describes the trainable adapter
    parameters. An 8-bit optimizer may use quantized primary state plus FP32 auxiliary tensors.
    """

    weight_storage_dtype: PrecisionMode | None = None
    quantized_storage_format: QuantizationMode = QuantizationMode.none
    dequantization_dtype: PrecisionMode
    forward_compute_dtype: PrecisionMode
    gradient_dtype: PrecisionMode
    optimizer_state_dtype: PrecisionMode | QuantizationMode
    optimizer_auxiliary_dtype: PrecisionMode = PrecisionMode.fp32
    master_weight_dtype: PrecisionMode | None = None

    @model_validator(mode="after")
    def _validate_storage_precision(self) -> PrecisionExecutionPolicy:
        quantized = self.quantized_storage_format != QuantizationMode.none
        if quantized == (self.weight_storage_dtype is not None):
            raise ValueError(
                "quantized weights omit weight_storage_dtype; unquantized weights require it"
            )
        return self


class AttentionExecutionPolicy(ContractModel):
    """Exact model attention API plus the one runtime kernel that is permitted."""

    model_attention_api: ModelAttentionApi
    effective_backend_required: AttentionKernel
    flash_sdp_enabled: bool
    mem_efficient_sdp_enabled: bool
    math_sdp_enabled: bool
    flash_attention_package: PackageLock | None = None
    kernel_probe_ref: Ref
    evidence_kind: Literal["functional_probe", "cpu_reference"]
    safety_mandate: str | None = None
    verification_requirement: ExecutionVerificationRequirement = (
        ExecutionVerificationRequirement.require_verified
    )
    fallback_policy: Literal["refuse"] = "refuse"

    @model_validator(mode="after")
    def _validate_attention_policy(self) -> AttentionExecutionPolicy:
        if not _is_pinned_ref(self.kernel_probe_ref):
            raise ValueError("attention kernel_probe_ref must be hash-pinned")
        toggles = (
            self.flash_sdp_enabled,
            self.mem_efficient_sdp_enabled,
            self.math_sdp_enabled,
        )
        required = self.effective_backend_required
        expected: tuple[bool, bool, bool]
        if required == AttentionKernel.torch_sdpa_flash:
            expected = (True, False, False)
        elif required == AttentionKernel.torch_sdpa_mem_efficient:
            expected = (False, True, False)
        elif required == AttentionKernel.torch_sdpa_math:
            expected = (False, False, True)
        else:
            # Eager/external implementations do not dispatch through PyTorch SDPA. Keep only the
            # safe math fallback globally enabled for unrelated framework operations.
            expected = (False, False, True)
        if toggles != expected:
            raise ValueError("SDPA toggles must permit exactly the required attention backend")
        if self.model_attention_api == ModelAttentionApi.sdpa and required not in {
            AttentionKernel.torch_sdpa_flash,
            AttentionKernel.torch_sdpa_mem_efficient,
            AttentionKernel.torch_sdpa_math,
        }:
            raise ValueError("the sdpa model API requires one exact PyTorch SDPA kernel")
        if self.model_attention_api == ModelAttentionApi.eager and required != AttentionKernel.eager:
            raise ValueError("the eager model API requires the eager backend")
        if (
            self.model_attention_api == ModelAttentionApi.xformers
            and required != AttentionKernel.xformers
        ):
            raise ValueError("the xformers model API requires the xformers backend")
        external = {
            ModelAttentionApi.flash_attention_2: AttentionKernel.flash_attention_2,
            ModelAttentionApi.flash_attention_3: AttentionKernel.flash_attention_3,
        }
        if self.model_attention_api in external:
            if (
                required != external[self.model_attention_api]
                or self.flash_attention_package is None
                or self.flash_attention_package.version is None
            ):
                raise ValueError("external FlashAttention requires its exact package and kernel")
        elif self.flash_attention_package is not None:
            raise ValueError("flash_attention_package is only valid for an external FlashAttention API")
        return self


class DeviceMapEntry(ContractModel):
    module: str
    device: str = Field(min_length=1)

    @field_validator("device")
    @classmethod
    def _explicit_device(cls, value: str) -> str:
        if value == "auto":
            raise ValueError("device placement must be explicit; 'auto' is forbidden")
        return value


class TrainingSchedule(ContractModel):
    max_steps: int | None = Field(default=None, ge=1)
    num_train_epochs: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _one_stop_condition(self) -> TrainingSchedule:
        if (self.max_steps is None) == (self.num_train_epochs is None):
            raise ValueError("exactly one of max_steps or num_train_epochs must be sealed")
        return self


class TrainingDataPolicy(ContractModel):
    dataset_format: Literal["instruction", "chat", "trace"]
    formatter_id: str = Field(min_length=1)
    formatter_sha256: str = Field(pattern=SHA256_PATTERN)
    chat_template_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    truncation_policy: Literal["refuse", "allow"] = "refuse"
    truncation_analysis: Literal["full_pinned_dataset"] = "full_pinned_dataset"
    packing: bool = False
    dataset_text_field: str = "text"

    @model_validator(mode="after")
    def _chat_template_required(self) -> TrainingDataPolicy:
        if (self.dataset_format == "chat") != (self.chat_template_sha256 is not None):
            raise ValueError("chat datasets require one exact chat-template digest")
        return self


class TrainerInterfacePolicy(ContractModel):
    """Version- and field-exact adapter to the installed TRL/Transformers surface."""

    package_versions: list[PackageLock] = Field(min_length=1)
    required_sft_config_fields: list[str] = Field(min_length=1)
    sequence_length_field: Literal["max_seq_length", "max_length"]
    tokenizer_parameter: Literal["tokenizer", "processing_class"]
    logging_steps: int = Field(default=1, ge=1)
    report_to: list[str] = Field(default_factory=list)
    disable_tqdm: bool = True

    @model_validator(mode="after")
    def _deterministic_interface(self) -> TrainerInterfacePolicy:
        names = [item.name for item in self.package_versions]
        if names != sorted(set(names)):
            raise ValueError("trainer package versions must be sorted by unique name")
        if any(item.version is None for item in self.package_versions):
            raise ValueError("trainer package versions must be exact")
        if self.required_sft_config_fields != sorted(set(self.required_sft_config_fields)):
            raise ValueError("required trainer fields must be sorted and unique")
        if self.sequence_length_field not in self.required_sft_config_fields:
            raise ValueError("the exact sequence-length field must be required")
        if self.report_to != sorted(set(self.report_to)):
            raise ValueError("report destinations must be sorted and unique")
        return self


class ResolvedExecutionConfiguration(ContractModel):
    """The hash-sealed configuration consumed directly by an isolated training worker.

    It contains every execution-affecting default. Workers may refuse it, but may not fill in,
    filter, reinterpret, or override semantic fields after this configuration is sealed.
    """

    contract_version: CONTRACT_VERSION_LITERAL = "1.0.0"
    configuration_id: str = Field(pattern=_ID)
    configuration_hash: str = Field(pattern=SHA256_PATTERN)
    backend_ref: Ref
    environment_ref: Ref
    environment_binding: Literal["profile_snapshot", "managed_lock"]
    capability_report_ref: Ref
    inputs: ExecutionInputs
    objective_ref: Ref
    runtime_mode: Literal["training", "cpu_toy"]
    precision: PrecisionExecutionPolicy
    attention: AttentionExecutionPolicy
    device_map: list[DeviceMapEntry] = Field(min_length=1)
    adapter: AdapterSpec
    optimizer: OptimizerSpec
    loss_impl: LossImpl
    sequence: SequenceSpec
    batching: BatchingSpec
    checkpoint_policy: CheckpointPolicy
    schedule: TrainingSchedule
    data: TrainingDataPolicy
    trainer_interface: TrainerInterfacePolicy
    export_format: ExportFormat
    trust_remote_code: Literal[False] = False
    use_safetensors: Literal[True] = True
    bnb_4bit_use_double_quant: bool
    adapter_task_type: Literal["CAUSAL_LM"] = "CAUSAL_LM"
    save_strategy: Literal["steps"] = "steps"
    gradient_checkpointing: bool = True
    output_dir: str = Field(min_length=1)
    # The sealed path is a ROOT, not the final trainer directory. Every execution derives its own
    # collision-free adapter/checkpoint directory from the fresh run_id under this layout.
    output_layout: Literal["run_scoped_v1"] = "run_scoped_v1"
    seed: int = Field(default=42, ge=0)
    data_seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def _validate_resolved_configuration(self) -> ResolvedExecutionConfiguration:
        for label, ref in (
            ("backend_ref", self.backend_ref),
            ("environment_ref", self.environment_ref),
            ("capability_report_ref", self.capability_report_ref),
            ("objective_ref", self.objective_ref),
        ):
            if not _is_pinned_ref(ref):
                raise ValueError(f"resolved execution {label} must be hash-pinned")
        environment_hash = self.environment_ref.hash
        assert environment_hash is not None and environment_hash.value is not None
        if (
            self.environment_binding == "profile_snapshot"
            and self.environment_ref.id != environment_hash.value
        ):
            raise ValueError("profile-snapshot environment identity must be its content hash")
        if (
            self.environment_binding == "managed_lock"
            and self.environment_ref.id == environment_hash.value
        ):
            raise ValueError("managed-lock environment identity must name the managed environment")
        keys = [item.module for item in self.device_map]
        if keys != sorted(set(keys)) or "" not in keys:
            raise ValueError("device_map must be sorted, unique, and bind the root module")
        if self.runtime_mode == "cpu_toy" and self.device_map != [
            DeviceMapEntry(module="", device="cpu")
        ]:
            raise ValueError("cpu_toy execution must bind the entire model to CPU")
        if self.adapter.method in {
            AdapterMethod.lora,
            AdapterMethod.qlora,
            AdapterMethod.dora,
        } and any(
            value is None
            for value in (
                self.adapter.lora_r,
                self.adapter.lora_alpha,
                self.adapter.lora_dropout,
                self.adapter.target_modules,
                self.adapter.bias,
            )
        ):
            raise ValueError("LoRA-family execution must seal every adapter default")
        if self.adapter.method in {
            AdapterMethod.lora,
            AdapterMethod.qlora,
            AdapterMethod.dora,
        } and self.precision.master_weight_dtype is None:
            raise ValueError("LoRA-family execution must seal the trainable master-weight dtype")
        if (
            self.precision.quantized_storage_format == QuantizationMode.none
            and self.precision.weight_storage_dtype
            != self.precision.forward_compute_dtype
        ):
            raise ValueError(
                "the first-party unquantized trainer requires weight and forward dtypes to match"
            )
        if (
            self.precision.quantized_storage_format != QuantizationMode.none
            and self.precision.dequantization_dtype
            != self.precision.forward_compute_dtype
        ):
            raise ValueError(
                "the first-party quantized trainer requires dequantization and forward dtypes to match"
            )
        if self.batching.fallback_grad_accumulation_steps is None:
            raise ValueError("the first-party trainer requires exact gradient accumulation")
        expected_token_target = (
            self.sequence.max_sequence_len
            * self.batching.micro_batch_size
            * self.batching.fallback_grad_accumulation_steps
        )
        if self.batching.supervised_token_accumulation_target != expected_token_target:
            raise ValueError(
                "the fixed-microbatch trainer requires its advisory token target to be derived exactly"
            )
        if self.sequence.buckets:
            raise ValueError("the first-party trainer does not implement sequence buckets")
        if self.sequence.packing != self.data.packing:
            raise ValueError("sequence and data packing policies must match")
        if any(
            value is None
            for value in (
                self.optimizer.weight_decay,
                self.optimizer.lr_scheduler,
                self.optimizer.warmup_ratio,
                self.checkpoint_policy.cadence_optimizer_steps,
            )
        ):
            raise ValueError("the first-party trainer requires all optimizer/checkpoint defaults")
        if self.checkpoint_policy.cadence_seconds is not None:
            raise ValueError("the first-party trainer does not implement time-based checkpoints")
        if self.checkpoint_policy.reload_verify:
            raise ValueError("the first-party trainer does not implement checkpoint reload verification")
        if self.checkpoint_policy.impl != CheckpointImpl.adapter_only:
            raise ValueError("the first-party trainer implements adapter-only checkpoints")
        quantized = self.precision.quantized_storage_format != QuantizationMode.none
        if not quantized and self.bnb_4bit_use_double_quant:
            raise ValueError("double quantization is invalid for unquantized execution")
        if self.export_format != ExportFormat.adapter_peft:
            raise ValueError("the first-party resolved executor emits PEFT adapters only")
        external_package = self.attention.flash_attention_package
        if external_package is not None:
            exact_packages = {
                (item.name.lower(), item.version)
                for item in self.trainer_interface.package_versions
            }
            if (external_package.name.lower(), external_package.version) not in exact_packages:
                raise ValueError(
                    "external FlashAttention must be included in exact trainer package versions"
                )
        return self


class EvalSchedule(ContractModel):
    before_run: bool = True
    after_run: bool = True
    every_optimizer_steps: int | None = Field(default=None, ge=1)
    suite_ref: str | None = None


class ExportSpec(ContractModel):
    format: ExportFormat
    output_dir: str = "output"


class PlannedStorageBinding(ContractModel):
    """The exact StorageProfile assessment accepted by a plan. ``marginal``/``unknown`` are usable
    only when that same verdict is explicitly recorded in ``accepted_suitability``; ``unsuitable`` is
    always refused."""

    role: StorageRole
    path: str = Field(min_length=1)
    assessment: StorageRoleAssessment
    accepted_suitability: StorageSuitability = StorageSuitability.suitable

    @model_validator(mode="after")
    def _validate_binding(self) -> PlannedStorageBinding:
        if self.assessment.role != self.role or self.assessment.path != self.path:
            raise ValueError("storage binding role/path must match its embedded assessment")
        if self.assessment.suitability == StorageSuitability.unsuitable:
            raise ValueError("unsuitable storage cannot be sealed into a RunPlan")
        if self.accepted_suitability != self.assessment.suitability:
            raise ValueError(
                "accepted_suitability must explicitly match the embedded storage verdict"
            )
        return self


class PhysicalResource(ContractModel):
    """One planned physical tier/device. This is scheduling intent, never measured residency."""

    resource_id: ParameterId
    tier: MemoryTier
    device_kind: DeviceKind | None = None
    device_id: str | None = None
    storage: PlannedStorageBinding | None = None

    @model_validator(mode="after")
    def _validate_resource(self) -> PhysicalResource:
        storage_tiers = {MemoryTier.nvme, MemoryTier.sata, MemoryTier.remote}
        if self.tier == MemoryTier.unknown:
            raise ValueError("a resolved physical resource cannot use the unknown memory tier")
        if self.tier == MemoryTier.gpu:
            if self.device_kind not in {
                DeviceKind.cuda,
                DeviceKind.rocm,
                DeviceKind.mps,
                DeviceKind.xpu,
            } or not self.device_id:
                raise ValueError("GPU resources require an accelerator kind and stable device_id")
            if self.storage is not None:
                raise ValueError("GPU resources cannot carry a storage binding")
        elif self.tier in {MemoryTier.pinned_ram, MemoryTier.pageable_ram}:
            if self.device_kind != DeviceKind.cpu or not self.device_id:
                raise ValueError("RAM resources require device_kind=cpu and a stable device_id")
            if self.storage is not None:
                raise ValueError("RAM resources cannot carry a storage binding")
        elif self.tier in storage_tiers:
            if self.storage is None:
                raise ValueError("storage-tier resources require a bound StorageProfile assessment")
            if self.device_kind is not None or self.device_id is not None:
                raise ValueError("storage-tier resources use their storage binding, not a device_id")
        return self


class PhysicalScopeSelector(ContractModel):
    """Select planned state by stable logical identity. Empty identity lists mean nothing, never an
    inferred dense model. ``whole_model`` is the explicit dense-safe fallback for unknown topology."""

    whole_model: bool = False
    parameter_scope_ids: list[ParameterId] = Field(default_factory=list)
    component_ids: list[ParameterId] = Field(default_factory=list)
    expert_ids: list[ParameterId] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_selector(self) -> PhysicalScopeSelector:
        for label, values in (
            ("parameter_scope_ids", self.parameter_scope_ids),
            ("component_ids", self.component_ids),
            ("expert_ids", self.expert_ids),
        ):
            if values != sorted(set(values)):
                raise ValueError(f"{label} must be sorted and unique")
        has_scoped_identity = bool(
            self.parameter_scope_ids or self.component_ids or self.expert_ids
        )
        if self.whole_model == has_scoped_identity:
            raise ValueError(
                "physical selectors require either whole_model or stable scoped identities"
            )
        return self

    def identity_key(self) -> tuple[object, ...]:
        return (
            self.whole_model,
            tuple(self.parameter_scope_ids),
            tuple(self.component_ids),
            tuple(self.expert_ids),
        )


class StatePlacement(ContractModel):
    placement_id: ParameterId
    state: PhysicalStateKind
    selector: PhysicalScopeSelector
    resource_id: ParameterId
    role: PlacementRole
    source_placement_id: ParameterId | None = None
    shard_group_id: ParameterId | None = None
    shard_index: int | None = Field(default=None, ge=0)
    shard_count: int | None = Field(default=None, ge=2)

    @model_validator(mode="after")
    def _validate_placement(self) -> StatePlacement:
        if self.role in {PlacementRole.replica, PlacementRole.cache}:
            if self.source_placement_id is None:
                raise ValueError("replica/cache placements require source_placement_id")
        elif self.source_placement_id is not None:
            raise ValueError("authoritative/shard placements cannot name source_placement_id")
        shard_fields = (self.shard_group_id, self.shard_index, self.shard_count)
        if self.role == PlacementRole.shard:
            if any(value is None for value in shard_fields):
                raise ValueError("shard placements require group, index, and count")
            if self.shard_index is not None and self.shard_count is not None:
                if self.shard_index >= self.shard_count:
                    raise ValueError("shard_index must be less than shard_count")
        elif any(value is not None for value in shard_fields):
            raise ValueError("only shard placements may carry shard metadata")
        return self


class OffloadRule(ContractModel):
    rule_id: ParameterId
    state: PhysicalStateKind
    selector: PhysicalScopeSelector
    source_resource_id: ParameterId
    target_resource_id: ParameterId
    mechanism: OffloadMechanism
    trigger: OffloadTrigger
    prefetch_policy: Literal["none", "static", "layer_window", "route_prediction", "heat_based"] = (
        "none"
    )
    eviction_policy: Literal["none", "lru", "lfu", "layer_window", "heat_based"] = "none"
    route_miss_action: RouteMissAction = RouteMissAction.fail

    @model_validator(mode="after")
    def _validate_rule(self) -> OffloadRule:
        if self.source_resource_id == self.target_resource_id:
            raise ValueError("offload source and target resources must differ")
        return self


class RankBinding(ContractModel):
    rank: int = Field(ge=0)
    resource_id: ParameterId
    node_id: ParameterId = "local"
    local_rank: int = Field(default=0, ge=0)


class ParallelGroup(ContractModel):
    group_id: ParameterId
    kind: ParallelismKind
    ranks: list[int] = Field(min_length=2)
    communication_backend: CommunicationBackend
    parameter_scope_ids: list[ParameterId] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_group(self) -> ParallelGroup:
        if self.ranks != sorted(set(self.ranks)):
            raise ValueError("parallel group ranks must be sorted and unique")
        if self.communication_backend == CommunicationBackend.none:
            raise ValueError("multi-rank groups require an explicit communication backend")
        if self.parameter_scope_ids != sorted(set(self.parameter_scope_ids)):
            raise ValueError("parallel group parameter_scope_ids must be sorted and unique")
        if self.kind == ParallelismKind.expert and not self.parameter_scope_ids:
            raise ValueError("expert-parallel groups require stable parameter scope IDs")
        return self


class ParallelismSpec(ContractModel):
    """Explicit rank/group topology. Groups may overlap across axes, so the contract never assumes
    that data x tensor x pipeline x expert degrees form one universal product."""

    world_size: int = Field(default=1, ge=1)
    ranks: list[RankBinding] = Field(min_length=1)
    groups: list[ParallelGroup] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_parallelism(self) -> ParallelismSpec:
        rank_ids = [item.rank for item in self.ranks]
        if rank_ids != list(range(self.world_size)):
            raise ValueError("rank bindings must be ordered and cover exactly 0..world_size-1")
        local_keys = [(item.node_id, item.local_rank) for item in self.ranks]
        if len(local_keys) != len(set(local_keys)):
            raise ValueError("local ranks must be unique within each node")
        group_ids = [item.group_id for item in self.groups]
        if group_ids != sorted(set(group_ids)):
            raise ValueError("parallel groups must be sorted by unique group_id")
        valid_ranks = set(rank_ids)
        if any(not set(group.ranks).issubset(valid_ranks) for group in self.groups):
            raise ValueError("parallel group ranks must reference bound ranks")
        if self.world_size == 1 and self.groups:
            raise ValueError("a single-rank plan cannot declare parallel groups")
        groups_by_kind: dict[ParallelismKind, list[ParallelGroup]] = {}
        for group in self.groups:
            groups_by_kind.setdefault(group.kind, []).append(group)
        for kind, groups in groups_by_kind.items():
            members = [rank for group in groups for rank in group.ranks]
            if len(members) != len(set(members)) or set(members) != valid_ranks:
                raise ValueError(
                    f"{kind.value} groups must partition all ranks without overlap"
                )
        return self


class PhysicalExecutionSpec(ContractModel):
    """The physical scheduler input, kept separate from learned semantic routing. Every field is
    planned intent sealed by RunPlan; it is not runtime residency or fit evidence."""

    evidence_status: Literal["planned_not_measured"] = "planned_not_measured"
    route_fidelity: Literal["preserve_or_fail", "declared_semantic_fallback"] = (
        "preserve_or_fail"
    )
    semantic_fallback_policy_ref: Ref | None = None
    storage_profile_ref: Ref | None = None
    resources: list[PhysicalResource] = Field(min_length=1)
    placements: list[StatePlacement] = Field(min_length=1)
    offload_rules: list[OffloadRule] = Field(default_factory=list)
    parallelism: ParallelismSpec

    def requires_parameter_accounting(self) -> bool:
        """Whether stable parameter/component/expert identities must be resolved from evidence."""

        return any(
            not selector.whole_model
            for selector in [
                *(item.selector for item in self.placements),
                *(item.selector for item in self.offload_rules),
            ]
        ) or any(group.parameter_scope_ids for group in self.parallelism.groups)

    @model_validator(mode="after")
    def _validate_physical_execution(self) -> PhysicalExecutionSpec:
        resource_ids = [item.resource_id for item in self.resources]
        if resource_ids != sorted(set(resource_ids)):
            raise ValueError("physical resources must be sorted by unique resource_id")
        resources = {item.resource_id: item for item in self.resources}
        placement_ids = [item.placement_id for item in self.placements]
        if placement_ids != sorted(set(placement_ids)):
            raise ValueError("state placements must be sorted by unique placement_id")
        rule_ids = [item.rule_id for item in self.offload_rules]
        if rule_ids != sorted(set(rule_ids)):
            raise ValueError("offload rules must be sorted by unique rule_id")
        for placement in self.placements:
            if placement.resource_id not in resources:
                raise ValueError("state placement references an unknown physical resource")
        for rank in self.parallelism.ranks:
            resource = resources.get(rank.resource_id)
            if resource is None:
                raise ValueError("rank binding references an unknown physical resource")
            if resource.tier not in {
                MemoryTier.gpu,
                MemoryTier.pinned_ram,
                MemoryTier.pageable_ram,
            }:
                raise ValueError("rank bindings require a compute resource, not storage")

        placements_by_id = {item.placement_id: item for item in self.placements}
        grouped: dict[tuple[object, ...], list[StatePlacement]] = {}
        for placement in self.placements:
            key = (placement.state, *placement.selector.identity_key())
            grouped.setdefault(key, []).append(placement)
            if placement.source_placement_id is not None:
                source_placement = placements_by_id.get(placement.source_placement_id)
                if source_placement is None or source_placement.role not in {
                    PlacementRole.authoritative,
                    PlacementRole.shard,
                }:
                    raise ValueError("replica/cache source must be an authoritative placement or shard")
                if (
                    source_placement.state != placement.state
                    or source_placement.selector != placement.selector
                ):
                    raise ValueError("replica/cache source must cover the same state and selector")
        for placements in grouped.values():
            authoritative = [
                item for item in placements if item.role == PlacementRole.authoritative
            ]
            shards = [item for item in placements if item.role == PlacementRole.shard]
            if shards:
                if authoritative:
                    raise ValueError("state scope cannot be both authoritative and sharded")
                groups = {item.shard_group_id for item in shards}
                counts = {item.shard_count for item in shards}
                indices = {item.shard_index for item in shards}
                if len(groups) != 1 or len(counts) != 1:
                    raise ValueError("authoritative shards require one coherent shard group")
                shard_count = next(iter(counts))
                if shard_count is None or indices != set(range(shard_count)):
                    raise ValueError("authoritative shard placements must cover every shard index")
            elif len(authoritative) != 1:
                raise ValueError("every planned state scope requires exactly one authoritative placement")

        for rule in self.offload_rules:
            source_resource = resources.get(rule.source_resource_id)
            target = resources.get(rule.target_resource_id)
            if source_resource is None or target is None:
                raise ValueError("offload rules must reference known resources")
            if source_resource.tier == target.tier or target.tier == MemoryTier.gpu:
                raise ValueError("offload rules must move state to a different non-GPU tier")
            source_placements = [
                item
                for item in self.placements
                if item.state == rule.state
                and item.selector == rule.selector
                and item.resource_id == rule.source_resource_id
                and item.role in {PlacementRole.authoritative, PlacementRole.shard}
            ]
            if not source_placements:
                raise ValueError("offload source must match an authoritative placement or shard")
            if rule.mechanism == OffloadMechanism.cpu_copy and target.tier not in {
                MemoryTier.pinned_ram,
                MemoryTier.pageable_ram,
            }:
                raise ValueError("cpu_copy offload requires a RAM target")
            if rule.mechanism == OffloadMechanism.cuda_unified_memory and target.tier not in {
                MemoryTier.pinned_ram,
                MemoryTier.pageable_ram,
            }:
                raise ValueError("cuda_unified_memory offload requires a RAM target")
            if rule.mechanism == OffloadMechanism.nvme_io and target.tier not in {
                MemoryTier.nvme,
                MemoryTier.sata,
            }:
                raise ValueError("nvme_io offload requires an NVMe or SATA target")

        storage_resources = [item for item in self.resources if item.storage is not None]
        if storage_resources:
            if self.storage_profile_ref is None or not _is_pinned_ref(self.storage_profile_ref):
                raise ValueError("storage-backed physical plans require a hash-pinned StorageProfile ref")
        elif self.storage_profile_ref is not None:
            raise ValueError("storage_profile_ref requires a storage-backed physical resource")

        semantic_fallback = any(
            rule.route_miss_action == RouteMissAction.semantic_fallback
            for rule in self.offload_rules
        )
        if semantic_fallback:
            if self.route_fidelity != "declared_semantic_fallback" or (
                self.semantic_fallback_policy_ref is None
                or not _is_pinned_ref(self.semantic_fallback_policy_ref)
            ):
                raise ValueError(
                    "semantic route fallback requires a declared, hash-pinned model-policy ref"
                )
        elif self.route_fidelity != "preserve_or_fail" or self.semantic_fallback_policy_ref is not None:
            raise ValueError("physical scheduling cannot alter semantic routing without an explicit rule")
        return self


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
    # Deprecated compatibility summary. New plans carry the exact resources/rules below; ZeRO values
    # are not treated as physical evidence or as a substitute for explicit parallel groups.
    offload_strategy: OffloadStrategy = OffloadStrategy.none
    allocator_policy: AllocatorPolicy = AllocatorPolicy.default
    compile_mode: CompileMode = CompileMode.none
    gradient_checkpointing: bool = True
    eval_schedule: EvalSchedule = Field(default_factory=EvalSchedule)
    export: ExportSpec
    seed: int = Field(default=42, ge=0)
    # Legacy compatibility surface. New plans leave this empty and execute only the independently
    # sealed, typed resolved_execution contract below.
    training_config_snapshot: JsonObject = Field(default_factory=dict)
    resolved_execution: ResolvedExecutionConfiguration | None = None
    # Pins the parameter evidence the planner consumed; it does not manufacture missing counts.
    parameter_accounting_ref: Ref | None = None
    # ``None`` identifies a legacy plan. The planner always emits a fully resolved spec for new plans.
    physical_execution: PhysicalExecutionSpec | None = None

    @model_validator(mode="after")
    def _validate_physical_plan(self) -> RunPlan:
        if self.parameter_accounting_ref is not None and not _is_pinned_ref(
            self.parameter_accounting_ref
        ):
            raise ValueError("parameter_accounting_ref must pin the sealed report hash")
        execution = self.resolved_execution
        if execution is not None:
            expected_hash = _canonical_contract_sha256(
                execution.model_dump(mode="json", exclude={"configuration_hash"})
            )
            if execution.configuration_hash != expected_hash:
                raise ValueError("resolved_execution configuration_hash does not match its body")
            if self.training_config_snapshot:
                raise ValueError("new resolved plans cannot carry a second trainer-config authority")
            if execution.backend_ref != self.backend_ref:
                raise ValueError("resolved execution backend_ref must match the RunPlan")
            if execution.environment_ref != self.environment_ref:
                raise ValueError("resolved execution environment_ref must match the RunPlan")
            if execution.inputs.dataset.ref != self.dataset_ref:
                raise ValueError("resolved execution dataset ref must match the RunPlan")
            if execution.inputs.model.location != self.base_model:
                raise ValueError("resolved execution model location must match base_model")
            if execution.precision.forward_compute_dtype != self.precision:
                raise ValueError("resolved forward precision must match the RunPlan summary")
            if execution.precision.quantized_storage_format != self.quantization:
                raise ValueError("resolved quantization must match the RunPlan summary")
            for label, resolved, summary in (
                ("adapter", execution.adapter, self.adapter),
                ("optimizer", execution.optimizer, self.optimizer),
                ("loss", execution.loss_impl, self.loss_impl),
                ("sequence", execution.sequence, self.sequence),
                ("batching", execution.batching, self.batching),
                ("checkpoint", execution.checkpoint_policy, self.checkpoint_policy),
            ):
                if resolved != summary:
                    raise ValueError(f"resolved {label} policy must match the RunPlan summary")
            attention_summary = {
                AttentionKernel.eager: AttentionImpl.eager,
                AttentionKernel.torch_sdpa_math: AttentionImpl.math,
                AttentionKernel.torch_sdpa_flash: AttentionImpl.sdpa,
                AttentionKernel.torch_sdpa_mem_efficient: AttentionImpl.sdpa,
                AttentionKernel.flash_attention_2: AttentionImpl.flash_attention_2,
                AttentionKernel.flash_attention_3: AttentionImpl.flash_attention_3,
                AttentionKernel.xformers: AttentionImpl.xformers,
            }[execution.attention.effective_backend_required]
            if attention_summary != self.attention_backend:
                raise ValueError("resolved attention policy must match the RunPlan summary")
            if execution.seed != self.seed or (
                execution.gradient_checkpointing != self.gradient_checkpointing
            ):
                raise ValueError("resolved seed/checkpointing must match the RunPlan summary")
            if (
                execution.export_format != self.export.format
                or execution.output_dir != self.export.output_dir
            ):
                raise ValueError("resolved export format/output must match the RunPlan summary")
        if self.physical_execution is None:
            return self
        if (
            self.physical_execution.requires_parameter_accounting()
            and self.parameter_accounting_ref is None
        ):
            raise ValueError(
                "scope-specific physical planning requires a hash-pinned parameter-accounting report"
            )
        has_rules = bool(self.physical_execution.offload_rules)
        if has_rules != (self.offload_strategy != OffloadStrategy.none):
            raise ValueError(
                "offload_strategy must agree with the explicit physical offload rules"
            )
        if execution is not None:
            planned_devices = {item.device for item in execution.device_map}
            resource_devices = {
                "cpu" if item.device_id == "cpu:0" else item.device_id
                for item in self.physical_execution.resources
                if item.device_id is not None
            }
            if not planned_devices.issubset(resource_devices):
                raise ValueError("resolved device_map references an unplanned physical resource")
        return self


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
    parameter_accounting_ref: Ref | None = None
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
    parameter_accounting_ref: Ref | None = None


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
    parameter_observations: list[ParameterObservation] = Field(default_factory=list)

    @field_validator("parameter_observations")
    @classmethod
    def _sorted_parameter_observations(
        cls, values: list[ParameterObservation]
    ) -> list[ParameterObservation]:
        ids = [item.observation_id for item in values]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("event parameter observations must be sorted by unique observation_id")
        return values


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
    parameter_accounting_refs: list[Ref] = Field(default_factory=list)
    evaluation: RunEvaluationLink | None = None
    reproducibility: RunReproducibility | None = None
    # Present on abnormal termination (state in {failed, interrupted}).
    failure: FailureRecord | None = None
    # Post-run fit reconciliation from observed peak memory (planned NATIVE_SAFE, or a spill?).
    final_fit: FitClassification | None = None
    notes: str = ""

    @field_validator("parameter_accounting_refs")
    @classmethod
    def _sorted_parameter_accounting_refs(cls, values: list[Ref]) -> list[Ref]:
        keys = [
            (
                item.id,
                item.hash.algo if item.hash is not None else "",
                (item.hash.value or "") if item.hash is not None else "",
            )
            for item in values
        ]
        if keys != sorted(set(keys)):
            raise ValueError("parameter_accounting_refs must be sorted and unique")
        return values


# --------------------------------------------------------------------------------------------------
# WorkerProtocol — the versioned core↔worker message envelope (NEW).
# --------------------------------------------------------------------------------------------------
class HelloBody(ContractModel):
    """Worker→core handshake: who I am + what I can do."""

    worker_id: str = Field(min_length=1)
    backend: BackendManifest
    # The exact environment identity the worker process is running inside. Managed environments use
    # their immutable lock hash; an unmanaged probe/demo environment may carry only its stable id.
    environment_ref: Ref
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
    run_id: str = Field(min_length=1)
    pid: int | None = None
    process_started_at: str | None = None
    execution_configuration_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)


class RunControlBody(ContractModel):
    run_id: str = Field(min_length=1)
    action: Literal["cancel", "pause", "resume", "checkpoint_now"]


class HeartbeatBody(ContractModel):
    run_id: str = Field(min_length=1)
    stage: StageMarker | None = None
    optimizer_step: int | None = Field(default=None, ge=0)
    pid_alive: bool = True


class TerminalResultBody(ContractModel):
    """Worker→core: the run ended. A FailureRecord is present iff the outcome was not PASS."""

    run_id: str = Field(min_length=1)
    outcome: FailureTaxonomy
    run_manifest: RunManifest
    artifacts: list[ArtifactManifest] = Field(default_factory=list)
    final_eval: EvaluationResult | None = None
    failure: FailureRecord | None = None

    @model_validator(mode="after")
    def _terminal_consistency(self) -> TerminalResultBody:
        if self.run_manifest.run_id != self.run_id:
            raise ValueError("terminal_result run_id must match run_manifest.run_id")
        if self.outcome == FailureTaxonomy.PASS:
            if self.failure is not None or self.run_manifest.failure is not None:
                raise ValueError("PASS terminal_result cannot carry a failure")
            if self.run_manifest.state != "succeeded":
                raise ValueError("PASS terminal_result requires a succeeded run_manifest")
        else:
            if self.failure is None:
                raise ValueError("non-PASS terminal_result requires a failure")
            if self.failure.taxonomy != self.outcome:
                raise ValueError("terminal_result outcome must match failure taxonomy")
            if self.failure.run_id not in {None, self.run_id}:
                raise ValueError("terminal_result failure run_id must match run_id")
            if self.run_manifest.state == "succeeded":
                raise ValueError("non-PASS terminal_result cannot carry a succeeded run_manifest")
        if self.run_manifest.failure != self.failure:
            raise ValueError("terminal_result failure must match run_manifest.failure")
        return self


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
WORKER_PROTOCOL_VERSION: Literal["2.0.0"] = "2.0.0"

# The body model that a given message `type` selects. The envelope retains a language-neutral JSON
# object on the wire, while WorkerMessage validates it against this map before accepting the message.
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

WorkerBody = (
    HelloBody
    | CapabilityProbeRequestBody
    | CapabilityReport
    | RunDispatchBody
    | RunAcceptedBody
    | FailureRecord
    | RunControlBody
    | RunEvent
    | HeartbeatBody
    | TerminalResultBody
)


class WorkerMessage(ContractModel):
    """The versioned envelope for the core↔worker channel — realizes the 'immutable RunPlan IN,
    structured RunEvent stream OUT' boundary. Protocol 2.0 uses a mandatory worker-first identity
    handshake. ``protocol_version`` evolves independently of any single contract's version. The body
    union is language-neutral and must match ``type`` (see :data:`WORKER_BODY_BY_TYPE`)."""

    protocol_version: Literal["2.0.0"]
    message_id: str = Field(min_length=1)
    correlation_id: str | None = None
    direction: Literal["core_to_worker", "worker_to_core"]
    sent_at: str | None = None
    type: WorkerMessageType
    body: WorkerBody

    @model_validator(mode="after")
    def _direction_and_body_match_type(self) -> WorkerMessage:
        core_to_worker = {"capability_probe_request", "run_dispatch", "run_control"}
        expected_direction = (
            "core_to_worker" if self.type in core_to_worker else "worker_to_core"
        )
        if self.direction != expected_direction:
            raise ValueError(
                f"message type {self.type!r} requires direction {expected_direction!r}"
            )
        body_payload = self.body.model_dump(mode="json")
        WORKER_BODY_BY_TYPE[self.type].model_validate(body_payload)
        if self.type in {"run_rejected", "failure"}:
            failure = FailureRecord.model_validate(body_payload)
            if failure.taxonomy == FailureTaxonomy.PASS:
                raise ValueError(f"{self.type} cannot carry PASS taxonomy")
        return self
