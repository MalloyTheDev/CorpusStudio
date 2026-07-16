/* GENERATED from docs/contracts/EnvironmentLock.schema.json — do not edit. Run: npm run gen:contracts */

export type Architecture = string;
export type ComputeCapability = string | null;
export type ContractVersion = "1.0.0";
export type CreatedAt = string | null;
export type CudaRuntimeVersion = string | null;
export type Implementation = string;
export type IndexUrls = string[];
export type LockHash = string | null;
export type LockId = string;
export type ManagerVersion = string;
export type ArtifactFilename = string | null;
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
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
export type Artifact = string | null;
export type Dependencies = string[];
export type Direct1 = boolean | null;
export type DirectUrl1 = string | null;
export type Editable1 = boolean | null;
/**
 * Number of files sealed by installed_files_hash; equals record_entries when record_integrity is verified.
 */
export type InstalledFileCount = number | null;
export type Installer = string | null;
export type Name = string;
export type NormalizedName1 = string;
/**
 * Explicit all-RECORD-row count meaning. Missing means preserved legacy hash-bearing-row counts and is not admissible for new health, planning, or execution.
 */
export type RecordCountSemantics = "all_record_rows_v2" | null;
/**
 * Number of regular installed files named by the distribution RECORD; positive when record_integrity is verified.
 */
export type RecordEntries = number | null;
export type RecordFailedEntries = string[];
export type RecordIntegrity = "verified" | "failed" | "missing" | "unknown";
/**
 * Verified row count under record_count_semantics; manager <=1.2 counted only hash-bearing rows, while all_record_rows_v2 equals record_entries.
 */
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
export type ContractVersion1 = "1.0.0";
export type EvidenceHash = string;
export type EvidenceId = string;
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
/**
 * Seals both 4-bit dequantization compute and forward activation autocast; complete probe evidence must observe this value for compute_dtype and forward_autocast.
 */
export type ComputeDtype = "bf16";
export type Device = "cuda:0";
export type DoubleQuantization = true;
export type AdapterMethod =
  "none" | "lora" | "qlora" | "dora" | "ia3" | "full_finetune" | "prompt_tuning" | "prefix_tuning";
/**
 * ``math``/``eager`` is forced on native-Windows/WDDM Blackwell sm_120 because the fused flash
 * kernel deadlocks there. Other platforms require their own functional capability result; WSL
 * evidence is not bare-Linux proof.
 */
export type AttentionImpl =
  "math" | "eager" | "sdpa" | "flash_attention_2" | "flash_attention_3" | "mem_efficient" | "xformers";
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
export type Detail = string | null;
export type ExecutionCombinations = ExecutionCapabilityCombination[];
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
  | "GRADIENT_FAILURE"
  | "LOSS_EVIDENCE_FAILURE"
  | "OPTIMIZER_FAILURE"
  | "UPDATE_FAILURE"
  | "ARTIFACT_FAILURE"
  | "CHECKPOINT_FAILURE"
  | "ENVIRONMENT_FAILURE"
  | "UNSUPPORTED_CONFIGURATION"
  | "ACCIDENTAL_SPILL"
  | "CONTROLLED_OFFLOAD";
export type Probe2 = string;
export type PythonVersion = string;
export type Id = string;
export type RequiredGitAncestor = string | null;
export type Architecture1 = string;
export type Compatible = boolean;
export type ContractVersion2 = "1.0.0";
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
export type ContractVersion3 = "1.0.0";
export type DistributionName = string;
export type Filename = string;
export type NormalizedName2 = string;
export type Path = string;
export type SizeBytes = number;
export type Version3 = string;

/**
 * The exact, reproducible record of what an environment actually contains — the post-install
 * counterpart of a recipe (which is only intent). ``packages`` are the resolved installs with
 * versions + hashes; ``lock_hash`` seals the set for drift detection. NEW.
 */
export interface EnvironmentLock {
  architecture?: Architecture;
  compute_capability?: ComputeCapability;
  contract_version?: ContractVersion;
  created_at?: CreatedAt;
  cuda_runtime_version?: CudaRuntimeVersion;
  implementation?: Implementation;
  index_urls?: IndexUrls;
  lock_hash?: LockHash;
  lock_id: LockId;
  manager_version?: ManagerVersion;
  package_install_evidence?: PackageInstallEvidence;
  packages?: Packages;
  platform_tag?: PlatformTag;
  probe_evidence?: EnvironmentProbeEvidence | null;
  python_version?: PythonVersion;
  recipe_ref: Ref;
  required_git_ancestor?: RequiredGitAncestor;
  resolution_ref?: Ref | null;
  runtime?: PythonRuntime | null;
  torch_build?: TorchBuild;
  torch_version?: TorchVersion;
  worker_artifact?: WorkerArtifactIdentity | null;
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
 * An algorithm-tagged digest. The engine emits sha256 today; the algo tag makes a future
 * migration additive (cf. versions/version_registry.FINGERPRINT_ALGO).
 */
export interface HashRef {
  algo?: Algo;
  value?: Value;
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
  installed_file_count?: InstalledFileCount;
  installed_files_hash?: HashRef | null;
  installer?: Installer;
  name: Name;
  normalized_name?: NormalizedName1;
  record_count_semantics?: RecordCountSemantics;
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
 * Hash-sealed evidence that a required complete execution tuple passed as a unit.
 */
export interface EnvironmentProbeEvidence {
  capability_report_hash: HashRef;
  contract_version?: ContractVersion1;
  evidence_hash: EvidenceHash;
  evidence_id: EvidenceId;
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
  detail?: Detail;
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
/**
 * A stable reference to another contract instance by id, optionally pinned to a content hash so
 * the reference cannot silently re-point.
 */
export interface Ref {
  hash?: HashRef | null;
  id: Id;
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
  contract_version?: ContractVersion2;
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
  contract_version?: ContractVersion3;
  distribution_name: DistributionName;
  filename: Filename;
  metadata_hash?: HashRef | null;
  normalized_name: NormalizedName2;
  path: Path;
  size_bytes: SizeBytes;
  version: Version3;
}
