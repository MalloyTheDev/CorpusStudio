/* GENERATED from docs/contracts/EnvironmentRecipe.schema.json — do not edit. Run: npm run gen:contracts */

export type BootstrapPipVersion = string | null;
export type CapabilityProbes = string[];
export type ContractVersion = "1.0.0";
export type DefaultIndexUrl = string | null;
export type Name = string;
export type Optional = boolean;
export type Reason = string | null;
export type Specifier = string | null;
export type DependencyRequirements = DependencyRequirement[];
export type Description = string;
export type DisplayName = string;
export type ExtraIndexUrls = string[];
export type Condition = string;
/**
 * @minItems 1
 */
export type Packages = [string, ...string[]];
export type Severity = "block" | "warn";
export type KnownConflicts = DependencyConflict[];
/**
 * The three dependency layers. The CONTROL PLANE stays lightweight + always installable (opening
 * CorpusStudio must never require CUDA/DeepSpeed/an ML framework); CAPABILITY profiles are opt-in
 * feature stacks added to the core process with graceful fallback; BACKEND_WORKER environments are
 * isolated per-framework runtimes (heavy frameworks pin conflicting torch/CUDA/xformers builds and
 * cannot coexist — they talk to the core via the WorkerMessage protocol, never by import).
 */
export type DependencyLayer = "control_plane" | "capability" | "backend_worker";
export type MinComputeCapability = string | null;
export type Notes = string[];
export type PythonRequires = string;
export type RecipeId = string;
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
export type Probe1 = "cuda_qlora_math_execution" | "cuda_qlora_sdpa_flash_execution" | "cuda_qlora_liger_execution";
export type Quantization = "nf4";
export type RequireAdapterRoundTrip = true;
export type RequiredDistributions = string[];
export type TargetModules = "all-linear";
export type RequiresCuda = boolean;
export type RequiresNativeBuild = boolean;
export type RequiresWorkerWheel = boolean;
export type OperatingSystem = "windows" | "wsl" | "linux" | "macos" | "unknown";
export type SupportedOs = OperatingSystem[];
export type Target = string;
/**
 * How far a recipe has been proven — the recipe-level twin of EnvironmentState. A recipe is a
 * DECLARATION of what to install; this says whether that declaration has ever produced a working
 * environment, and at what level. ``declared`` = we can render the install plan but have not built +
 * verified it; higher tiers require actual evidence (a real install / probe / hardware run).
 */
export type RecipeVerification = "declared" | "build_verified" | "functional_verified" | "hardware_verified";

/**
 * A declarative, platform/CUDA-aware recipe for building one isolated environment — the WHAT to
 * install, not the act of installing. A recipe is only a declaration: ``verification`` says whether
 * it has ever produced a working environment (declared → hardware_verified). Grounded in the engine's
 * real optional extras (pyproject ``[train]`` / ``[parquet]`` / ``[tokenizer]``).
 */
export interface EnvironmentRecipe {
  bootstrap_pip_version?: BootstrapPipVersion;
  capability_probes?: CapabilityProbes;
  contract_version?: ContractVersion;
  cuda_index_urls?: CudaIndexUrls;
  default_index_url?: DefaultIndexUrl;
  dependency_requirements?: DependencyRequirements;
  description?: Description;
  display_name?: DisplayName;
  extra_index_urls?: ExtraIndexUrls;
  known_conflicts?: KnownConflicts;
  layer: DependencyLayer;
  min_compute_capability?: MinComputeCapability;
  notes?: Notes;
  python_requires?: PythonRequires;
  recipe_id: RecipeId;
  required_execution_probe?: QloraExecutionProbeSpec | null;
  requires_cuda?: RequiresCuda;
  requires_native_build?: RequiresNativeBuild;
  requires_worker_wheel?: RequiresWorkerWheel;
  supported_os?: SupportedOs;
  target?: Target;
  verification?: RecipeVerification;
}
export interface CudaIndexUrls {
  [k: string]: string;
}
export interface DependencyRequirement {
  name: Name;
  optional?: Optional;
  reason?: Reason;
  specifier?: Specifier;
}
export interface DependencyConflict {
  condition: Condition;
  packages: Packages;
  severity?: Severity;
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
