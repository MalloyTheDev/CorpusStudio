/* GENERATED from docs/contracts/RunPlan.schema.json — do not edit. Run: npm run gen:contracts */

export type LoraAlpha = number | null;
export type LoraDropout = number | null;
export type LoraR = number | null;
export type AdapterMethod =
  "none" | "lora" | "qlora" | "dora" | "ia3" | "full_finetune" | "prompt_tuning" | "prefix_tuning";
export type TargetModules = string[] | null;
export type AllocatorPolicy = "default" | "expandable_segments" | "max_split_size" | "garbage_collection";
/**
 * ``math``/``eager`` is forced on Blackwell sm_120 — the fused flash/mem-efficient kernels
 * deadlock on the first backward (training/environment.py, estimators.py) — at a large activation
 * VRAM cost.
 */
export type AttentionImpl =
  "math" | "eager" | "sdpa" | "flash_attention_2" | "flash_attention_3" | "mem_efficient" | "xformers";
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type BaseModel = string;
export type FallbackGradAccumulationSteps = number | null;
export type MicroBatchSize = number;
export type SupervisedTokenAccumulationTarget = number;
export type CadenceOptimizerSteps = number | null;
export type CadenceSeconds = number | null;
export type CheckpointImpl = "full_state" | "adapter_only" | "sharded" | "distcp" | "safetensors";
export type KeepLast = number | null;
export type ReloadVerify = boolean;
export type CompileMode = "none" | "eager" | "reduce_overhead" | "max_autotune" | "aot_inductor";
export type ContractVersion = "1.0.0";
export type CreatedAt = string | null;
export type AfterRun = boolean;
export type BeforeRun = boolean;
export type EveryOptimizerSteps = number | null;
export type SuiteRef = string | null;
export type ExportFormat =
  "adapter_peft" | "merged_safetensors" | "merged_fp16" | "gguf" | "onnx" | "awq" | "gptq" | "mlx";
export type OutputDir = string;
export type GradientCheckpointing = boolean;
export type LossImpl = "cross_entropy" | "liger_fused_ce" | "chunked_ce" | "dpo" | "orpo" | "kto" | "ipo" | "reward_bt";
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
export type LearningRate = number;
export type LrScheduler = string | null;
export type WarmupRatio = number | null;
export type WeightDecay = number | null;
export type PlanHash = string;
export type PlanId = string;
export type PrecisionMode = "fp32" | "tf32" | "fp16" | "bf16" | "fp8" | "mixed_bf16" | "mixed_fp16";
export type QuantizationMode = "none" | "int8" | "int4" | "nf4" | "fp4" | "gptq" | "awq" | "hqq";
export type Seed = number;
export type Buckets = number[];
export type MaxSequenceLen = number;
export type Packing = boolean;
export type TruncationAllowed = boolean;
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

/**
 * The IMMUTABLE, fully-resolved execution plan the core dispatches to a worker: no ambiguity is
 * left for the worker to decide. Formalizes + hardens config_templates.TrainingConfigTemplate. Key
 * upgrades: attention_backend defaults to math on Blackwell; the accumulation target is in
 * SUPERVISED TOKENS; ``plan_hash`` seals immutability (a change means a NEW plan).
 */
export interface RunPlan {
  adapter: AdapterSpec;
  allocator_policy?: AllocatorPolicy;
  attention_backend: AttentionImpl;
  backend_ref: Ref;
  base_model: BaseModel;
  batching: BatchingSpec;
  checkpoint_policy: CheckpointPolicy;
  compile_mode?: CompileMode;
  contract_version?: ContractVersion;
  created_at?: CreatedAt;
  dataset_ref: Ref;
  environment_ref: Ref;
  eval_schedule?: EvalSchedule;
  export: ExportSpec;
  gradient_checkpointing?: GradientCheckpointing;
  loss_impl: LossImpl;
  offload_strategy?: OffloadStrategy;
  optimizer: OptimizerSpec;
  parameter_accounting_ref?: Ref | null;
  plan_hash: PlanHash;
  plan_id: PlanId;
  precision: PrecisionMode;
  quantization: QuantizationMode;
  seed?: Seed;
  sequence: SequenceSpec;
  task_type: TaskType;
  training_config_snapshot?: TrainingConfigSnapshot;
}
export interface AdapterSpec {
  lora_alpha?: LoraAlpha;
  lora_dropout?: LoraDropout;
  lora_r?: LoraR;
  method: AdapterMethod;
  target_modules?: TargetModules;
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
 * The accumulation TARGET is expressed in SUPERVISED TOKENS, not microbatch count. A
 * token-target-CAPABLE backend accumulates until summed supervised tokens >= target and normalizes
 * the loss by supervised tokens across the window, so the effective batch is invariant to sequence
 * length + padding. The first-party ``corpus_studio`` reference trainer is NOT token-target-capable
 * today — it honors ``fallback_grad_accumulation_steps`` (a fixed microbatch count); the token
 * target is the contract a future token-aware backend would satisfy.
 */
export interface BatchingSpec {
  fallback_grad_accumulation_steps?: FallbackGradAccumulationSteps;
  micro_batch_size?: MicroBatchSize;
  supervised_token_accumulation_target: SupervisedTokenAccumulationTarget;
}
export interface CheckpointPolicy {
  cadence_optimizer_steps?: CadenceOptimizerSteps;
  cadence_seconds?: CadenceSeconds;
  impl: CheckpointImpl;
  keep_last?: KeepLast;
  reload_verify?: ReloadVerify;
}
export interface EvalSchedule {
  after_run?: AfterRun;
  before_run?: BeforeRun;
  every_optimizer_steps?: EveryOptimizerSteps;
  suite_ref?: SuiteRef;
}
export interface ExportSpec {
  format: ExportFormat;
  output_dir?: OutputDir;
}
export interface OptimizerSpec {
  impl: Optimizer;
  learning_rate: LearningRate;
  lr_scheduler?: LrScheduler;
  warmup_ratio?: WarmupRatio;
  weight_decay?: WeightDecay;
}
/**
 * Sequence handling. Buckets let variable-length rows batch efficiently; the max bucket ==
 * the trainer's sequence_len (config_templates.sequence_len, default 4096).
 */
export interface SequenceSpec {
  buckets?: Buckets;
  max_sequence_len?: MaxSequenceLen;
  packing?: Packing;
  truncation_allowed?: TruncationAllowed;
}
export interface TrainingConfigSnapshot {
  [k: string]: unknown;
}
