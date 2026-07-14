/* GENERATED from docs/contracts/EnvironmentInstallation.schema.json — do not edit. Run: npm run gen:contracts */

/**
 * @minItems 1
 */
export type Argv = [string, ...string[]];
export type Cancelled = boolean;
export type CommandId = string;
export type ExitCode = number | null;
export type ExpectedOutputs = string[];
export type ContractVersion = "1.0.0";
export type Detail = string | null;
export type DetectedAt = string | null;
export type ExceptionType = string | null;
export type ExitCode1 = number | null;
/**
 * ``math``/``eager`` is forced on native-Windows/WDDM Blackwell sm_120 because the fused flash
 * kernel deadlocks there. Other platforms require their own functional capability result; WSL
 * evidence is not bare-Linux proof.
 */
export type AttentionImpl =
  "math" | "eager" | "sdpa" | "flash_attention_2" | "flash_attention_3" | "mem_efficient" | "xformers";
/**
 * The fit verdict. ``NATIVE_*`` = fully resident. ``CONTROLLED_*`` = a deliberate, planned
 * offload (acceptable, slower). ``ACCIDENTAL_*`` / ``THRASHING`` = an unplanned spill the platform
 * did silently (the failure mode the engine warns about). ``FAIL`` = will not run.
 */
export type FitClass =
  | "PLANNED_UNPROVEN"
  | "NATIVE_SAFE"
  | "NATIVE_TIGHT"
  | "NATIVE_UNPROVEN"
  | "MARGINAL"
  | "CONTROLLED_ACTIVATION_OFFLOAD"
  | "CONTROLLED_OPTIMIZER_OFFLOAD"
  | "CONTROLLED_PARAMETER_OFFLOAD"
  | "ACCIDENTAL_UNIFIED_MEMORY_PAGING"
  | "ACCIDENTAL_WDDM_SPILL"
  | "THRASHING"
  | "FAIL";
export type ContractVersion1 = "1.0.0";
export type DeviceCapacityBytes = number | null;
export type EstimatedPeakBytes = number | null;
export type HeadroomBytes = number | null;
export type Rationale = string;
export type CudaDeviceFreeBytes = number | null;
export type CudaDeviceUsedBytes = number | null;
export type DedicatedGpuBytes = number | null;
export type ProcessRssBytes = number | null;
export type SharedGpuBytes = number | null;
export type SystemRamUsedBytes = number | null;
export type TorchAllocatedBytes = number | null;
export type TorchPeakAllocatedBytes = number | null;
export type TorchPeakReservedBytes = number | null;
export type TorchReservedBytes = number | null;
export type Message = string;
export type Reconciled = boolean;
export type Remediation = string | null;
export type RunId = string | null;
export type Signal = string | null;
/**
 * Ordered lifecycle stage of a run, launch → export. A RunEvent carries the stage it belongs to
 * so a consumer can render a precise progress spine and localize a failure to the exact stage.
 */
export type StageMarker =
  | "process_start"
  | "env_loaded"
  | "cuda_init"
  | "execution_config_verified"
  | "attention_policy_applied"
  | "placement_verified"
  | "placement_deviation"
  | "model_loaded"
  | "quantized"
  | "adapter_attached"
  | "optimizer_created"
  | "batch_materialized"
  | "forward"
  | "loss"
  | "backward"
  | "optimizer_step"
  | "checkpoint"
  | "reload"
  | "evaluation"
  | "export";
/**
 * Terminal outcome category. ``PASS`` is included so the same enum classifies a completed
 * probe/run, not only failures. Grounded in the exact hazards the engine documents: the sm_120
 * fused-attention deadlock (KERNEL_STALL), the WDDM silent spill (ACCIDENTAL_SPILL vs a clean
 * OOM), and env/dependency mismatches (ENVIRONMENT_FAILURE).
 */
export type FailureTaxonomy =
  | "PASS"
  | "FAIL"
  | "OOM"
  | "TIMEOUT"
  | "KERNEL_STALL"
  | "NUMERICAL_FAILURE"
  | "CHECKPOINT_FAILURE"
  | "ENVIRONMENT_FAILURE"
  | "UNSUPPORTED_CONFIGURATION"
  | "ACCIDENTAL_SPILL"
  | "CONTROLLED_OFFLOAD";
export type FinishedAt = string;
export type NativeBuildOccurred = boolean;
export type Phase =
  | "create_venv"
  | "upgrade_pip"
  | "install"
  | "lock"
  | "import_probe"
  | "verify"
  | "inventory"
  | "dependency_probe"
  | "functional_probe"
  | "hardware_probe"
  | "capability_probe"
  | "health_probe";
export type StartedAt = string;
export type StderrPath = string | null;
export type StdoutPath = string | null;
export type TimedOut = boolean;
export type TimeoutSeconds = number;
export type WorkingDirectory = string;
export type Commands = EnvironmentCommandRecord[];
export type ContractVersion2 = "1.0.0";
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type FinishedAt1 = string | null;
export type InstallationId = string;
export type ArtifactFilename = string | null;
export type ConfiguredIndexUrls = string[];
export type Direct = boolean | null;
export type DirectUrl = string | null;
export type Editable = boolean | null;
export type InstallerCommandId = string | null;
export type NormalizedName = string;
export type Requested = boolean | null;
export type Source = "pypi" | "wheel" | "sdist" | "conda" | "vcs" | "local" | "unknown";
export type SourceEvidenceReason = string | null;
export type SourceIndexUrl = string | null;
export type VcsCommit = string | null;
export type VcsRepository = string | null;
export type Version = string;
export type PackageInstallEvidence = PackageInstallEvidence1[];
export type Architecture = string;
export type CapturedAt = string | null;
export type ComputeCapability = string | null;
export type ContractVersion3 = "1.0.0";
export type CudaRuntimeVersion = string | null;
export type EvidenceHash = string;
export type EvidenceId = string;
export type Implementation = string;
export type IndexUrls = string[];
export type PackageInstallEvidence2 = PackageInstallEvidence1[];
export type Artifact = string | null;
export type Dependencies = string[];
export type Direct1 = boolean | null;
export type DirectUrl1 = string | null;
export type Editable1 = boolean | null;
export type Installer = string | null;
export type Name = string;
export type NormalizedName1 = string;
export type RecordEntries = number | null;
export type RecordFailedEntries = string[];
export type RecordIntegrity = "verified" | "failed" | "missing" | "unknown";
export type RecordVerifiedEntries = number | null;
export type Requested1 = boolean | null;
export type Source1 = "pypi" | "wheel" | "sdist" | "conda" | "vcs" | "local" | "unknown";
export type SourceEvidenceReason1 = string | null;
export type SourceIndexUrl1 = string | null;
export type VcsCommit1 = string | null;
export type VcsRepository1 = string | null;
export type Version1 = string | null;
export type Packages = PackageLock[];
export type PlatformTag = string;
export type PythonVersion = string;
export type Architecture1 = string;
export type Compatible = boolean;
export type ContractVersion4 = "1.0.0";
export type Executable = string;
export type Implementation1 = string;
export type IncompatibilityReasons = string[];
export type IsVirtualEnvironment = boolean;
export type OperatingSystem = "windows" | "wsl" | "linux" | "macos" | "unknown";
export type Platform = string;
export type RuntimeId = string;
export type VenvAvailable = boolean;
export type Version2 = string;
export type TorchBuild = string | null;
export type TorchVersion = string | null;
export type ContractVersion5 = "1.0.0";
export type DistributionName = string;
export type Filename = string;
export type NormalizedName2 = string;
export type Path = string;
export type SizeBytes = number;
export type Version3 = string;
export type ContractVersion6 = "1.0.0";
export type EvidenceHash1 = string;
export type EvidenceId1 = string;
export type BackwardDurationSeconds = number | null;
export type BaselineGpuAllocatedBytes = number;
export type BaselineGpuReservedBytes = number;
export type BaselineHostRssBytes = number;
export type BaselineNvidiaSmiProcessBytes = number | null;
export type DurationSeconds = number;
export type ForwardDurationSeconds = number | null;
export type GpuAllocatorScope = "pytorch_cuda_allocator_process";
export type GpuDeviceScope = "nvidia_smi_current_process" | "unavailable";
export type GpuPowerWatts = number | null;
export type GpuTemperatureCelsius = number | null;
export type HostMemoryScope = "current_process_rss";
export type OptimizerStepDurationSeconds = number | null;
export type PeakGpuAllocatedBytes = number;
export type PeakGpuReservedBytes = number;
export type PeakHostRssBytes = number;
export type PeakNvidiaSmiProcessBytes = number | null;
export type ProfileSignature = string;
export type AttentionApi = "sdpa";
export type ComputeDtype = "bf16";
export type Device = "cuda:0";
export type DoubleQuantization = true;
export type AdapterMethod =
  "none" | "lora" | "qlora" | "dora" | "ia3" | "full_finetune" | "prompt_tuning" | "prefix_tuning";
/**
 * The exact attention implementation an execution policy permits at runtime.
 */
export type AttentionKernel =
  | "eager"
  | "torch_sdpa_math"
  | "torch_sdpa_flash"
  | "torch_sdpa_mem_efficient"
  | "flash_attention_2"
  | "flash_attention_3"
  | "xformers";
export type CheckpointImpl = "full_state" | "adapter_only" | "sharded" | "distcp" | "safetensors";
export type DeviceKind = "cuda" | "rocm" | "mps" | "xpu" | "cpu";
export type ExecutionContractVersion = string;
export type ExportFormat =
  "adapter_peft" | "merged_safetensors" | "merged_fp16" | "gguf" | "onnx" | "awq" | "gptq" | "mlx";
export type LossImpl = "cross_entropy" | "liger_fused_ce" | "chunked_ce" | "dpo" | "orpo" | "kto" | "ipo" | "reward_bt";
export type Optimizer =
  | "adamw_torch"
  | "adamw_torch_fused"
  | "adamw_8bit"
  | "adamw_bnb_8bit"
  | "paged_adamw_8bit"
  | "paged_adamw_32bit"
  | "adafactor"
  | "lion"
  | "sgd";
export type PrecisionMode = "fp32" | "tf32" | "fp16" | "bf16" | "fp8" | "mixed_bf16" | "mixed_fp16";
export type Probe = string;
export type QuantizationMode = "none" | "int8" | "int4" | "nf4" | "fp4" | "gptq" | "awq" | "hqq";
export type RuntimeMode = "training" | "cpu_toy";
export type FlashSdpEnabled = boolean;
export type GradientCheckpointing = true;
export type MathSdpEnabled = boolean;
export type MemoryEfficientSdpEnabled = false;
export type Optimizer1 = "adamw_torch";
export type Probe1 = "cuda_qlora_math_execution" | "cuda_qlora_sdpa_flash_execution";
export type Quantization = "nf4";
export type RequireAdapterRoundTrip = true;
export type RequiredDistributions = string[];
export type TargetModules = "all-linear";
export type Detail1 = string | null;
export type ExecutionCombinations = ExecutionCapabilityCombination[];
export type Probe2 = string;
export type ProbeResults = ProbeResult[];
export type RetryRequiresRecreate = boolean;
export type StartedAt1 = string;
/**
 * The lifecycle state of a managed environment. The escalation is deliberate — "installed" is
 * NEVER "supported": a package importing (IMPORTABLE) is not proof a kernel runs
 * (FUNCTIONAL_PROBE_PASSED), which is not proof the hardware supports it (HARDWARE_VERIFIED). Only
 * HARDWARE_VERIFIED earns "supported". The terminal-degraded states record WHY an env is unusable.
 */
export type EnvironmentState =
  | "NOT_INSTALLED"
  | "INSTALLING"
  | "INSTALLED_UNCHECKED"
  | "IMPORTABLE"
  | "DEPENDENCY_PROBE_PASSED"
  | "FUNCTIONAL_PROBE_PASSED"
  | "HARDWARE_VERIFIED"
  | "DEGRADED"
  | "INCOMPATIBLE"
  | "DRIFTED"
  | "BROKEN";

/**
 * Recoverable journal for one environment creation attempt.
 */
export interface EnvironmentInstallation {
  commands?: Commands;
  contract_version?: ContractVersion2;
  environment_ref: Ref;
  failure?: FailureRecord | null;
  finished_at?: FinishedAt1;
  installation_id: InstallationId;
  package_install_evidence?: PackageInstallEvidence;
  post_probe_inventory?: InstalledEnvironmentEvidence | null;
  pre_probe_inventory?: InstalledEnvironmentEvidence | null;
  probe_evidence?: EnvironmentProbeEvidence | null;
  probe_results?: ProbeResults;
  recipe_ref: Ref;
  resolution_ref: Ref;
  retry_requires_recreate?: RetryRequiresRecreate;
  started_at: StartedAt1;
  state?: EnvironmentState;
  worker_artifact?: WorkerArtifactIdentity | null;
}
/**
 * Durable evidence for one bounded, argv-only creation/install/probe command.
 */
export interface EnvironmentCommandRecord {
  argv: Argv;
  cancelled?: Cancelled;
  command_id: CommandId;
  environment?: Environment;
  exit_code?: ExitCode;
  expected_outputs?: ExpectedOutputs;
  failure?: FailureRecord | null;
  finished_at: FinishedAt;
  native_build_occurred?: NativeBuildOccurred;
  phase: Phase;
  started_at: StartedAt;
  stderr_path?: StderrPath;
  stdout_path?: StdoutPath;
  timed_out?: TimedOut;
  timeout_seconds: TimeoutSeconds;
  working_directory: WorkingDirectory;
}
export interface Environment {
  [k: string]: string;
}
/**
 * A structured, classified terminal outcome for a run, capability probe, or export. The
 * taxonomy turns 'it died' into an actionable category — a real OOM vs a KERNEL_STALL (the sm_120
 * fused-attention deadlock) vs an ACCIDENTAL_SPILL vs a CONTROLLED_OFFLOAD. NEW.
 */
export interface FailureRecord {
  contract_version?: ContractVersion;
  detail?: Detail;
  detected_at?: DetectedAt;
  exception_type?: ExceptionType;
  exit_code?: ExitCode1;
  fit_at_failure?: FitClassification | null;
  memory_at_failure?: MemoryMetrics | null;
  message: Message;
  reconciled?: Reconciled;
  remediation?: Remediation;
  run_id?: RunId;
  signal?: Signal;
  stage?: StageMarker | null;
  taxonomy: FailureTaxonomy;
}
/**
 * The planner/calibrator verdict on whether a resolved RunPlan fits the target environment, and
 * HOW: a native fit, a deliberately-offloaded fit, or an ACCIDENTAL spill (the silent WDDM/unified
 * paging that looks frozen but crawls at 10-25x). NEW — the engine emits only a coarse warn/pass
 * VRAM band (preflight.gpu_memory, _VRAM_SAFETY_MARGIN_GB).
 */
export interface FitClassification {
  attention_path?: AttentionImpl | null;
  classification: FitClass;
  contract_version?: ContractVersion1;
  device_capacity_bytes?: DeviceCapacityBytes;
  estimated_peak_bytes?: EstimatedPeakBytes;
  headroom_bytes?: HeadroomBytes;
  rationale?: Rationale;
}
/**
 * The full memory-signature block sampled during a run. Distinguishes PyTorch's allocator view,
 * raw CUDA device memory, and OS-level residency (``dedicated`` vs ``shared`` GPU memory) so a
 * Windows/WDDM spill to shared memory is VISIBLE rather than hidden inside 'used VRAM'. Grounded in
 * gpu_probe.GpuMemory + the estimators note that torch.max_memory_allocated counts the WDDM spill.
 */
export interface MemoryMetrics {
  cuda_device_free_bytes?: CudaDeviceFreeBytes;
  cuda_device_used_bytes?: CudaDeviceUsedBytes;
  dedicated_gpu_bytes?: DedicatedGpuBytes;
  process_rss_bytes?: ProcessRssBytes;
  shared_gpu_bytes?: SharedGpuBytes;
  system_ram_used_bytes?: SystemRamUsedBytes;
  torch_allocated_bytes?: TorchAllocatedBytes;
  torch_peak_allocated_bytes?: TorchPeakAllocatedBytes;
  torch_peak_reserved_bytes?: TorchPeakReservedBytes;
  torch_reserved_bytes?: TorchReservedBytes;
}
/**
 * A stable reference to another contract instance by id, optionally pinned to a content hash so
 * the reference cannot silently re-point.
 */
export interface Ref {
  hash?: HashRef | null;
  id: Id;
}
/**
 * An algorithm-tagged digest. The engine emits sha256 today; the algo tag makes a future
 * migration additive (cf. versions/version_registry.FINGERPRINT_ALGO).
 */
export interface HashRef {
  algo?: Algo;
  value?: Value;
}
/**
 * Sanitized pip-install provenance retained separately from installed-file inspection.
 */
export interface PackageInstallEvidence1 {
  artifact_filename?: ArtifactFilename;
  artifact_hash?: HashRef | null;
  configured_index_urls?: ConfiguredIndexUrls;
  direct?: Direct;
  direct_url?: DirectUrl;
  editable?: Editable;
  installer_command_id?: InstallerCommandId;
  normalized_name: NormalizedName;
  requested?: Requested;
  source?: Source;
  source_evidence_reason?: SourceEvidenceReason;
  source_index_url?: SourceIndexUrl;
  vcs_commit?: VcsCommit;
  vcs_repository?: VcsRepository;
  version: Version;
}
/**
 * A pre- or post-probe installed-state inventory; explicitly not a final environment lock.
 */
export interface InstalledEnvironmentEvidence {
  architecture?: Architecture;
  captured_at?: CapturedAt;
  compute_capability?: ComputeCapability;
  contract_version?: ContractVersion3;
  cuda_runtime_version?: CudaRuntimeVersion;
  evidence_hash: EvidenceHash;
  evidence_id: EvidenceId;
  implementation?: Implementation;
  index_urls?: IndexUrls;
  package_install_evidence?: PackageInstallEvidence2;
  packages?: Packages;
  platform_tag?: PlatformTag;
  python_version?: PythonVersion;
  recipe_ref: Ref;
  runtime?: PythonRuntime | null;
  torch_build?: TorchBuild;
  torch_version?: TorchVersion;
  worker_artifact?: WorkerArtifactIdentity | null;
}
/**
 * A resolved dependency and its install provenance.
 *
 * ``hash`` seals the installed distribution's RECORD metadata when that evidence is available; it
 * is not mislabelled as the original wheel hash. ``direct_url`` and ``artifact`` preserve the
 * stronger source identity pip exposes for direct/VCS/local installs. ``dependencies`` is the
 * installed metadata dependency graph, not a second resolver.
 */
export interface PackageLock {
  artifact?: Artifact;
  artifact_hash?: HashRef | null;
  dependencies?: Dependencies;
  direct?: Direct1;
  direct_url?: DirectUrl1;
  editable?: Editable1;
  hash?: HashRef | null;
  installer?: Installer;
  name: Name;
  normalized_name?: NormalizedName1;
  record_entries?: RecordEntries;
  record_failed_entries?: RecordFailedEntries;
  record_integrity?: RecordIntegrity;
  record_verified_entries?: RecordVerifiedEntries;
  requested?: Requested1;
  source?: Source1;
  source_evidence_reason?: SourceEvidenceReason1;
  source_index_url?: SourceIndexUrl1;
  vcs_commit?: VcsCommit1;
  vcs_repository?: VcsRepository1;
  version?: Version1;
}
/**
 * A discovered Python executable that can potentially create an isolated worker environment.
 *
 * Discovery never assumes the control-plane interpreter is the only installation. Compatibility is
 * an explicit verdict against the selected recipe, while ``venv_available`` proves the stdlib venv
 * module can be located without creating anything.
 */
export interface PythonRuntime {
  architecture?: Architecture1;
  compatible?: Compatible;
  contract_version?: ContractVersion4;
  executable: Executable;
  implementation?: Implementation1;
  incompatibility_reasons?: IncompatibilityReasons;
  is_virtual_environment?: IsVirtualEnvironment;
  os?: OperatingSystem;
  platform?: Platform;
  runtime_id: RuntimeId;
  venv_available?: VenvAvailable;
  version?: Version2;
}
/**
 * Immutable identity of the exact wheel executed by a managed backend worker.
 *
 * A mutable checkout is not a worker identity. The plan binds a concrete wheel before mutation;
 * the post-install lock binds the same wheel and the installed distribution evidence.
 */
export interface WorkerArtifactIdentity {
  content_hash: HashRef;
  contract_version?: ContractVersion5;
  distribution_name: DistributionName;
  filename: Filename;
  metadata_hash?: HashRef | null;
  normalized_name: NormalizedName2;
  path: Path;
  size_bytes: SizeBytes;
  version: Version3;
}
/**
 * Hash-sealed evidence that a required complete execution tuple passed as a unit.
 */
export interface EnvironmentProbeEvidence {
  capability_report_hash: HashRef;
  contract_version?: ContractVersion6;
  evidence_hash: EvidenceHash1;
  evidence_id: EvidenceId1;
  memory: ProbeMemoryEvidence;
  profile_signature: ProfileSignature;
  required_spec: QloraExecutionProbeSpec;
  tuple_result: ProbeResult;
}
/**
 * Measured resource evidence for one bounded probe, with scopes kept explicit.
 */
export interface ProbeMemoryEvidence {
  backward_duration_seconds?: BackwardDurationSeconds;
  baseline_gpu_allocated_bytes: BaselineGpuAllocatedBytes;
  baseline_gpu_reserved_bytes: BaselineGpuReservedBytes;
  baseline_host_rss_bytes: BaselineHostRssBytes;
  baseline_nvidia_smi_process_bytes?: BaselineNvidiaSmiProcessBytes;
  duration_seconds: DurationSeconds;
  forward_duration_seconds?: ForwardDurationSeconds;
  gpu_allocator_scope?: GpuAllocatorScope;
  gpu_device_scope?: GpuDeviceScope;
  gpu_power_watts?: GpuPowerWatts;
  gpu_temperature_celsius?: GpuTemperatureCelsius;
  host_memory_scope?: HostMemoryScope;
  optimizer_step_duration_seconds?: OptimizerStepDurationSeconds;
  peak_gpu_allocated_bytes: PeakGpuAllocatedBytes;
  peak_gpu_reserved_bytes: PeakGpuReservedBytes;
  peak_host_rss_bytes: PeakHostRssBytes;
  peak_nvidia_smi_process_bytes?: PeakNvidiaSmiProcessBytes;
}
/**
 * The exact complete QLoRA tuple a readiness environment must prove as one operation.
 *
 * Math and flash tuples are independent identities. A math-only seal is never a flash claim, and
 * independent flash/bitsandbytes/optimizer probes cannot be unioned into a complete capability.
 */
export interface QloraExecutionProbeSpec {
  attention_api?: AttentionApi;
  compute_dtype?: ComputeDtype;
  device?: Device;
  double_quantization?: DoubleQuantization;
  execution_combination: ExecutionCapabilityCombination;
  flash_sdp_enabled?: FlashSdpEnabled;
  gradient_checkpointing?: GradientCheckpointing;
  math_sdp_enabled?: MathSdpEnabled;
  memory_efficient_sdp_enabled?: MemoryEfficientSdpEnabled;
  optimizer?: Optimizer1;
  probe?: Probe1;
  quantization?: Quantization;
  require_adapter_round_trip?: RequireAdapterRoundTrip;
  required_distributions?: RequiredDistributions;
  target_modules?: TargetModules;
}
/**
 * One execution tuple demonstrated together by a bounded functional probe.
 *
 * Independent successes on precision, quantization, adapter, optimizer, loss, attention, and
 * checkpoint axes are diagnostic only. The planner may seal a run only from one of these complete
 * tuples, preventing a union of unrelated probes from becoming a fictional capability.
 */
export interface ExecutionCapabilityCombination {
  adapter_method: AdapterMethod;
  attention_impl: AttentionImpl;
  attention_kernel: AttentionKernel;
  checkpoint_impl: CheckpointImpl;
  device: DeviceKind;
  execution_contract_version: ExecutionContractVersion;
  export_format: ExportFormat;
  loss_impl: LossImpl;
  optimizer: Optimizer;
  precision: PrecisionMode;
  probe: Probe;
  quantization: QuantizationMode;
  runtime_mode: RuntimeMode;
}
export interface ProbeResult {
  detail?: Detail1;
  execution_combinations?: ExecutionCombinations;
  measured?: Measured;
  outcome: FailureTaxonomy;
  probe: Probe2;
  proves?: Proves;
}
export interface Measured {
  [k: string]: unknown;
}
export interface Proves {
  [k: string]: string[];
}
