"""The CorpusStudio platform contracts — the language-neutral boundary between the (Python → Rust)
platform core, the Python AI worker(s), and the UI shell.

Each root contract is a pydantic model carrying a ``contract_version`` and grounded in an existing
engine model (see each class docstring). These models are the canonical source of truth; the
language-neutral JSON Schemas the Rust core / Avalonia / Tauri consume are generated FROM them by
:mod:`corpus_studio.platform.schema_export`. Pure — importing this pulls no torch.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

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
    CompileMode,
    DeviceKind,
    ExportFormat,
    FailureTaxonomy,
    FitClass,
    LossImpl,
    MemoryResidencyModel,
    OffloadStrategy,
    OperatingSystem,
    Optimizer,
    PrecisionMode,
    QuantizationMode,
    StageMarker,
    TaskType,
    TrainerTarget,
)

_ID = r"^[A-Za-z0-9._-]+$"


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
