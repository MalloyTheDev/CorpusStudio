/* GENERATED from docs/contracts/DependencyResolution.schema.json — do not edit. Run: npm run gen:contracts */

export type AcceleratorTag = string;
export type BlockingReasons = string[];
export type ContractVersion = "1.0.0";
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type EnvironmentRoot = string | null;
export type EstimatedDiskBytes = number | null;
export type EstimatedDownloadBytes = number | null;
/**
 * @minItems 1
 */
export type Argv = [string, ...string[]];
export type ConfiguredIndexUrls = string[];
export type Description = string;
export type EvidencePath = string | null;
export type ExpectedOutputs = string[];
export type NativeBuildExpected = boolean;
export type NetworkRequired = boolean;
export type Phase = "create_venv" | "upgrade_pip" | "install" | "verify";
export type TimeoutSeconds = number;
export type WorkingDirectory = string | null;
export type InstallSteps = InstallStep[];
export type ManagerVersion = string;
export type OperatingSystem = "windows" | "wsl" | "linux" | "macos" | "unknown";
export type PythonVersion = string;
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
export type RequiredGitAncestor = string | null;
export type ResolutionHash = string | null;
export type Resolvable = boolean;
export type ResolvedIndexUrls = string[];
export type Architecture = string;
export type Compatible = boolean;
export type ContractVersion1 = "1.0.0";
export type Executable = string;
export type Implementation = string;
export type IncompatibilityReasons = string[];
export type IsVirtualEnvironment = boolean;
export type OperatingSystem1 = "windows" | "wsl" | "linux" | "macos" | "unknown";
export type Platform = string;
export type RuntimeId = string;
export type VenvAvailable = boolean;
export type Version = string;
export type Warnings = string[];
export type ContractVersion2 = "1.0.0";
export type DistributionName = string;
export type Filename = string;
export type NormalizedName = string;
export type Path = string;
export type SizeBytes = number;
export type Version1 = string;
export type WorkerSourceCommit = string | null;

/**
 * The resolved PREVIEW of provisioning a recipe on a specific host — the exact argv steps, the
 * chosen wheel index, and the disk/network cost — for explicit user confirmation BEFORE anything is
 * installed. Pure/derivable; no environment is created to produce it. NEW.
 */
export interface DependencyResolution {
  accelerator_tag?: AcceleratorTag;
  blocking_reasons?: BlockingReasons;
  contract_version?: ContractVersion;
  environment_ref?: Ref | null;
  environment_root?: EnvironmentRoot;
  estimated_disk_bytes?: EstimatedDiskBytes;
  estimated_download_bytes?: EstimatedDownloadBytes;
  install_steps?: InstallSteps;
  manager_version?: ManagerVersion;
  os?: OperatingSystem;
  python_version?: PythonVersion;
  recipe_ref: Ref;
  required_execution_probe?: QloraExecutionProbeSpec | null;
  required_git_ancestor?: RequiredGitAncestor;
  resolution_hash?: ResolutionHash;
  resolvable?: Resolvable;
  resolved_index_urls?: ResolvedIndexUrls;
  runtime?: PythonRuntime | null;
  warnings?: Warnings;
  worker_artifact?: WorkerArtifactIdentity | null;
  worker_source_commit?: WorkerSourceCommit;
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
 * One bounded, argv-structured install command — NEVER a shell string, so an untrusted package or
 * index name can't inject a command (mirrors the no-shell trainer-launch invariant). ``argv[0]`` is
 * the executable; the rest are literal arguments.
 */
export interface InstallStep {
  argv: Argv;
  configured_index_urls?: ConfiguredIndexUrls;
  description?: Description;
  environment?: Environment;
  evidence_path?: EvidencePath;
  expected_outputs?: ExpectedOutputs;
  native_build_expected?: NativeBuildExpected;
  network_required?: NetworkRequired;
  phase: Phase;
  timeout_seconds?: TimeoutSeconds;
  working_directory?: WorkingDirectory;
}
export interface Environment {
  [k: string]: string;
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
/**
 * A discovered Python executable that can potentially create an isolated worker environment.
 *
 * Discovery never assumes the control-plane interpreter is the only installation. Compatibility is
 * an explicit verdict against the selected recipe, while ``venv_available`` proves the stdlib venv
 * module can be located without creating anything.
 */
export interface PythonRuntime {
  architecture?: Architecture;
  compatible?: Compatible;
  contract_version?: ContractVersion1;
  executable: Executable;
  implementation?: Implementation;
  incompatibility_reasons?: IncompatibilityReasons;
  is_virtual_environment?: IsVirtualEnvironment;
  os?: OperatingSystem1;
  platform?: Platform;
  runtime_id: RuntimeId;
  venv_available?: VenvAvailable;
  version?: Version;
}
/**
 * Immutable identity of the exact wheel executed by a managed backend worker.
 *
 * A mutable checkout is not a worker identity. The plan binds a concrete wheel before mutation;
 * the post-install lock binds the same wheel and the installed distribution evidence.
 */
export interface WorkerArtifactIdentity {
  content_hash: HashRef;
  contract_version?: ContractVersion2;
  distribution_name: DistributionName;
  filename: Filename;
  metadata_hash?: HashRef | null;
  normalized_name: NormalizedName;
  path: Path;
  size_bytes: SizeBytes;
  version: Version1;
}
