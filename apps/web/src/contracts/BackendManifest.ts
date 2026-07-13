/* GENERATED from docs/contracts/BackendManifest.schema.json — do not edit. Run: npm run gen:contracts */

export type AdapterMethod =
  "none" | "lora" | "qlora" | "dora" | "ia3" | "full_finetune" | "prompt_tuning" | "prefix_tuning";
export type AdapterMethods = AdapterMethod[];
/**
 * ``math``/``eager`` is forced on Blackwell sm_120 — the fused flash/mem-efficient kernels
 * deadlock on the first backward (training/environment.py, estimators.py) — at a large activation
 * VRAM cost.
 */
export type AttentionImpl =
  "math" | "eager" | "sdpa" | "flash_attention_2" | "flash_attention_3" | "mem_efficient" | "xformers";
export type AttentionImpls = AttentionImpl[];
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
  | "env_loaded"
  | "cuda_init"
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
  backend_id: BackendId;
  backend_version: BackendVersion;
  capability_probes?: CapabilityProbes;
  checkpoint_impls?: CheckpointImpls;
  checkpoint_semantics?: CheckpointSemantics | null;
  compile_modes?: CompileModes;
  contract_version?: ContractVersion;
  dependency_conflicts?: DependencyConflicts;
  dependency_requirements?: DependencyRequirements;
  display_name?: DisplayName;
  environment_lock_ref?: Ref | null;
  export_compatibility?: ExportCompatibility;
  export_formats?: ExportFormats;
  known_failure_modes?: KnownFailureModes;
  loss_impls?: LossImpls;
  model_families?: ModelFamilies;
  objective_capabilities?: ObjectiveCapabilities;
  offload_strategies?: OffloadStrategies;
  optimizers?: Optimizers;
  precision_modes?: PrecisionModes;
  quantization_modes?: QuantizationModes;
  required_compute_capability?: RequiredComputeCapability;
  supported_devices: SupportedDevices;
  supported_os: SupportedOs;
  task_types: TaskTypes;
  telemetry_hooks?: TelemetryHooks;
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
