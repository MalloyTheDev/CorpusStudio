/* GENERATED from docs/contracts/BackendManifest.schema.json — do not edit. Run: npm run gen:contracts */

export type AdapterMethod =
  "none" | "lora" | "qlora" | "dora" | "ia3" | "full_finetune" | "prompt_tuning" | "prefix_tuning";
export type AdapterMethods = AdapterMethod[];
/**
 * ``math``/``eager`` is forced on native-Windows/WDDM Blackwell sm_120 because the fused flash
 * kernel deadlocks there. Other platforms require their own functional capability result; WSL
 * evidence is not bare-Linux proof.
 */
export type AttentionImpl =
  "math" | "eager" | "sdpa" | "flash_attention_2" | "flash_attention_3" | "mem_efficient" | "xformers";
export type AttentionImpls = AttentionImpl[];
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
export type AttentionKernels = AttentionKernel[];
export type BackendId = string;
export type BackendVersion = string;
export type CapabilityProbes = string[];
export type CheckpointImpl = "full_state" | "adapter_only" | "sharded" | "distcp" | "safetensors";
export type CheckpointImpls = CheckpointImpl[];
export type Contains = (
  "adapter_weights" | "base_weights" | "optimizer_state" | "lr_scheduler" | "rng_state" | "trainer_state"
)[];
export type ReloadVerifiable = boolean;
export type Resumable = boolean;
export type CommunicationBackend = "none" | "nccl" | "gloo" | "mpi" | "ucc" | "backend_native";
export type CommunicationBackends = CommunicationBackend[];
export type CompileMode = "none" | "eager" | "reduce_overhead" | "max_autotune" | "aot_inductor";
export type CompileModes = CompileMode[];
export type ContractVersion = "1.0.0";
export type Condition = string;
/**
 * @minItems 1
 */
export type Packages = [string, ...string[]];
export type Severity = "block" | "warn";
export type DependencyConflicts = DependencyConflict[];
export type Name = string;
export type Optional = boolean;
export type Reason = string | null;
export type Specifier = string | null;
export type DependencyRequirements = DependencyRequirement[];
export type DisplayName = string;
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type ExecutionContractVersions = string[];
export type ExportFormat =
  "adapter_peft" | "merged_safetensors" | "merged_fp16" | "gguf" | "onnx" | "awq" | "gptq" | "mlx";
export type ServesIn = string[];
export type ExportCompatibility = ExportCompatibilityEntry[];
export type ExportFormats = ExportFormat[];
export type Condition1 = string;
export type Description = string;
export type Mitigation = string | null;
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
export type KnownFailureModes = KnownFailureMode[];
export type LossImpl = "cross_entropy" | "liger_fused_ce" | "chunked_ce" | "dpo" | "orpo" | "kto" | "ipo" | "reward_bt";
export type LossImpls = LossImpl[];
export type ModelFamilies = string[];
export type ObjectiveCapabilities = string[];
/**
 * The ``controlled_*`` values are the deliberate, planned counterparts of the accidental spills
 * in :class:`FitClass`.
 */
export type OffloadStrategy =
  | "none"
  | "controlled_activation_offload"
  | "controlled_optimizer_offload"
  | "controlled_parameter_offload"
  | "cpu_offload"
  | "disk_offload"
  | "deepspeed_zero2"
  | "deepspeed_zero3";
export type OffloadStrategies = OffloadStrategy[];
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
export type Optimizers = Optimizer[];
export type ParallelismKind = "data" | "tensor" | "pipeline" | "expert" | "sequence" | "context";
export type ParallelismKinds = ParallelismKind[];
export type PlacementMode =
  "single_resource" | "identity_scoped" | "replicated" | "sharded" | "tiered" | "expert_scoped";
export type PlacementModes = PlacementMode[];
/**
 * A physical state tier. A RunPlan names the intended tier; only runtime evidence may claim
 * actual residency there.
 */
export type MemoryTier = "gpu" | "pinned_ram" | "pageable_ram" | "nvme" | "sata" | "remote" | "unknown";
export type PlacementTiers = MemoryTier[];
export type PrecisionMode = "fp32" | "tf32" | "fp16" | "bf16" | "fp8" | "mixed_bf16" | "mixed_fp16";
export type PrecisionModes = PrecisionMode[];
export type QuantizationMode = "none" | "int8" | "int4" | "nf4" | "fp4" | "gptq" | "awq" | "hqq";
export type QuantizationModes = QuantizationMode[];
export type RequiredComputeCapability = string | null;
/**
 * @minItems 1
 */
export type SupportedDevices = [DeviceKind, ...DeviceKind[]];
export type DeviceKind = "cuda" | "rocm" | "mps" | "xpu" | "cpu";
/**
 * @minItems 1
 */
export type SupportedOs = [OperatingSystem, ...OperatingSystem[]];
export type OperatingSystem = "windows" | "wsl" | "linux" | "macos" | "unknown";
/**
 * @minItems 1
 */
export type TaskTypes = [TaskType, ...TaskType[]];
export type TaskType =
  | "sft"
  | "pretraining"
  | "preference"
  | "reward"
  | "classification"
  | "embedding"
  | "multimodal"
  | "evaluation"
  | "distillation"
  | "grpo";
export type Metrics = string[];
/**
 * Ordered lifecycle stage of a run, launch → export. A RunEvent carries the stage it belongs to
 * so a consumer can render a precise progress spine and localize a failure to the exact stage.
 */
export type StageMarker =
  | "process_start"
  | "dataset_verification"
  | "execution_config_verified"
  | "env_loaded"
  | "cuda_init"
  | "tokenizer_load"
  | "dataset_formatting"
  | "truncation_analysis"
  | "attention_policy_applied"
  | "model_load"
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
export type TelemetryHooks = TelemetryHook[];
export type TrainerFields = string[];
export type TrainerInitFields = string[];
/**
 * Verbatim from config_templates.TrainingConfigTarget.
 */
export type TrainerTarget =
  "corpus_studio" | "axolotl_yaml" | "trl_config" | "unsloth_script" | "huggingface_trainer" | "llama_factory";

/**
 * A backend worker's STATIC self-declaration of everything it can do — the core reads this to
 * decide which backend can even attempt a RunPlan, BEFORE dispatch. Mostly NEW; generalizes the
 * inference model_backends.base.ModelBackend Protocol + training/compatibility into a declarable
 * manifest for a TRAINING backend.
 */
export interface BackendManifest {
  adapter_methods?: AdapterMethods;
  attention_impls?: AttentionImpls;
  attention_kernels?: AttentionKernels;
  backend_id: BackendId;
  backend_version: BackendVersion;
  capability_probes?: CapabilityProbes;
  checkpoint_impls?: CheckpointImpls;
  checkpoint_semantics?: CheckpointSemantics | null;
  communication_backends?: CommunicationBackends;
  compile_modes?: CompileModes;
  contract_version?: ContractVersion;
  dependency_conflicts?: DependencyConflicts;
  dependency_requirements?: DependencyRequirements;
  display_name?: DisplayName;
  environment_lock_ref?: Ref | null;
  execution_contract_versions?: ExecutionContractVersions;
  export_compatibility?: ExportCompatibility;
  export_formats?: ExportFormats;
  known_failure_modes?: KnownFailureModes;
  loss_impls?: LossImpls;
  model_families?: ModelFamilies;
  objective_capabilities?: ObjectiveCapabilities;
  offload_strategies?: OffloadStrategies;
  optimizers?: Optimizers;
  parallelism_kinds?: ParallelismKinds;
  placement_modes?: PlacementModes;
  placement_tiers?: PlacementTiers;
  precision_modes?: PrecisionModes;
  quantization_modes?: QuantizationModes;
  required_compute_capability?: RequiredComputeCapability;
  supported_devices: SupportedDevices;
  supported_os: SupportedOs;
  task_types: TaskTypes;
  telemetry_hooks?: TelemetryHooks;
  trainer_fields?: TrainerFields;
  trainer_init_fields?: TrainerInitFields;
  trainer_target?: TrainerTarget | null;
}
export interface CheckpointSemantics {
  contains?: Contains;
  reload_verifiable?: ReloadVerifiable;
  resumable?: Resumable;
}
export interface DependencyConflict {
  condition: Condition;
  packages: Packages;
  severity?: Severity;
}
export interface DependencyRequirement {
  name: Name;
  optional?: Optional;
  reason?: Reason;
  specifier?: Specifier;
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
export interface ExportCompatibilityEntry {
  format: ExportFormat;
  serves_in?: ServesIn;
}
/**
 * Pre-declared hazards tagged with a taxonomy so the core can warn/refuse up front. The
 * canonical example: fused attention deadlocks on sm_120.
 */
export interface KnownFailureMode {
  condition: Condition1;
  description?: Description;
  mitigation?: Mitigation;
  taxonomy: FailureTaxonomy;
}
export interface TelemetryHook {
  metrics?: Metrics;
  stage: StageMarker;
}
