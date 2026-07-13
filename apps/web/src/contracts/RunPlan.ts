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
export type EvidenceStatus = "planned_not_measured";
export type EvictionPolicy = "none" | "lru" | "lfu" | "layer_window" | "heat_based";
export type OffloadMechanism = "cpu_copy" | "cuda_unified_memory" | "nvme_io" | "backend_native";
export type PrefetchPolicy = "none" | "static" | "layer_window" | "route_prediction" | "heat_based";
/**
 * What the physical scheduler does when requested state is not ready. Semantic fallback is
 * never implicit: it requires a separately pinned learned-policy reference.
 */
export type RouteMissAction = "wait" | "defer" | "fail" | "semantic_fallback";
export type RuleId = string;
export type ComponentIds = string[];
export type ExpertIds = string[];
export type ParameterScopeIds = string[];
export type WholeModel = boolean;
export type SourceResourceId = string;
export type PhysicalStateKind = "parameters" | "gradients" | "optimizer_state" | "activations";
export type TargetResourceId = string;
export type OffloadTrigger = "static" | "memory_pressure" | "ahead_of_use" | "after_use";
export type OffloadRules = OffloadRule[];
export type CommunicationBackend = "none" | "nccl" | "gloo" | "mpi" | "ucc" | "backend_native";
export type GroupId = string;
export type ParallelismKind = "data" | "tensor" | "pipeline" | "expert" | "sequence" | "context";
export type ParameterScopeIds1 = string[];
/**
 * @minItems 2
 */
export type Ranks = [number, number, ...number[]];
export type Groups = ParallelGroup[];
/**
 * @minItems 1
 */
export type Ranks1 = [RankBinding, ...RankBinding[]];
export type LocalRank = number;
export type NodeId = string;
export type Rank = number;
export type ResourceId = string;
export type WorldSize = number;
/**
 * @minItems 1
 */
export type Placements = [StatePlacement, ...StatePlacement[]];
export type PlacementId = string;
export type ResourceId1 = string;
export type PlacementRole = "authoritative" | "shard" | "replica" | "cache";
export type ShardCount = number | null;
export type ShardGroupId = string | null;
export type ShardIndex = number | null;
export type SourcePlacementId = string | null;
/**
 * @minItems 1
 */
export type Resources = [PhysicalResource, ...PhysicalResource[]];
export type DeviceId = string | null;
export type DeviceKind = "cuda" | "rocm" | "mps" | "xpu" | "cpu";
export type ResourceId2 = string;
/**
 * The per-role verdict for a candidate path. ``unsuitable`` is a hard no (data-loss or
 * thrash-to-a-halt risk); ``marginal`` will work but degrade (e.g. an HDD for offload); ``unknown``
 * when detection couldn't characterize the device (honest, never a false ``suitable``).
 */
export type StorageSuitability = "suitable" | "marginal" | "unsuitable" | "unknown";
export type DeviceMountPoint = string | null;
export type FreeBytes = number | null;
/**
 * How a storage device attaches. The interface — not just free space — decides whether a device
 * can sustain the heavy sequential + random writes of optimizer/parameter offload and checkpointing.
 * A USB bridge or a network mount will thrash under sustained offload even with terabytes free.
 */
export type StorageInterface = "nvme_pcie" | "sata_ssd" | "hdd" | "usb" | "network" | "virtual" | "unknown";
export type Path = string;
export type Reasons = string[];
export type RequiredFreeBytes = number | null;
/**
 * The role a path plays in a run. Roles differ in access pattern: ``optimizer_offload`` /
 * ``parameter_offload`` / ``scratch`` / ``checkpoints`` are WRITE-heavy; ``model_cache`` /
 * ``dataset_cache`` are read-LATENCY-sensitive during load; ``source_repo`` / ``python_env`` are
 * thousands of SMALL files touched on every process start (an import over a USB bridge or a WSL
 * ``/mnt`` mount stalls); ``archive`` just wants capacity. A path's suitability is judged PER ROLE (a
 * USB SSD is fine for ``archive``, poor for ``model_cache``, unfit for ``optimizer_offload``).
 */
export type StorageRole =
  | "os"
  | "source_repo"
  | "python_env"
  | "model_cache"
  | "dataset_cache"
  | "checkpoints"
  | "scratch"
  | "optimizer_offload"
  | "parameter_offload"
  | "artifacts"
  | "archive"
  | "logs";
/**
 * The per-role verdict for a candidate path. ``unsuitable`` is a hard no (data-loss or
 * thrash-to-a-halt risk); ``marginal`` will work but degrade (e.g. an HDD for offload); ``unknown``
 * when detection couldn't characterize the device (honest, never a false ``suitable``).
 */
export type StorageSuitability1 = "suitable" | "marginal" | "unsuitable" | "unknown";
export type Path1 = string;
/**
 * A physical state tier. A RunPlan names the intended tier; only runtime evidence may claim
 * actual residency there.
 */
export type MemoryTier = "gpu" | "pinned_ram" | "pageable_ram" | "nvme" | "sata" | "remote" | "unknown";
export type RouteFidelity = "preserve_or_fail" | "declared_semantic_fallback";
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
  physical_execution?: PhysicalExecutionSpec | null;
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
 * The physical scheduler input, kept separate from learned semantic routing. Every field is
 * planned intent sealed by RunPlan; it is not runtime residency or fit evidence.
 */
export interface PhysicalExecutionSpec {
  evidence_status?: EvidenceStatus;
  offload_rules?: OffloadRules;
  parallelism: ParallelismSpec;
  placements: Placements;
  resources: Resources;
  route_fidelity?: RouteFidelity;
  semantic_fallback_policy_ref?: Ref | null;
  storage_profile_ref?: Ref | null;
}
export interface OffloadRule {
  eviction_policy?: EvictionPolicy;
  mechanism: OffloadMechanism;
  prefetch_policy?: PrefetchPolicy;
  route_miss_action?: RouteMissAction;
  rule_id: RuleId;
  selector: PhysicalScopeSelector;
  source_resource_id: SourceResourceId;
  state: PhysicalStateKind;
  target_resource_id: TargetResourceId;
  trigger: OffloadTrigger;
}
/**
 * Select planned state by stable logical identity. Empty identity lists mean nothing, never an
 * inferred dense model. ``whole_model`` is the explicit dense-safe fallback for unknown topology.
 */
export interface PhysicalScopeSelector {
  component_ids?: ComponentIds;
  expert_ids?: ExpertIds;
  parameter_scope_ids?: ParameterScopeIds;
  whole_model?: WholeModel;
}
/**
 * Explicit rank/group topology. Groups may overlap across axes, so the contract never assumes
 * that data x tensor x pipeline x expert degrees form one universal product.
 */
export interface ParallelismSpec {
  groups?: Groups;
  ranks: Ranks1;
  world_size?: WorldSize;
}
export interface ParallelGroup {
  communication_backend: CommunicationBackend;
  group_id: GroupId;
  kind: ParallelismKind;
  parameter_scope_ids?: ParameterScopeIds1;
  ranks: Ranks;
}
export interface RankBinding {
  local_rank?: LocalRank;
  node_id?: NodeId;
  rank: Rank;
  resource_id: ResourceId;
}
export interface StatePlacement {
  placement_id: PlacementId;
  resource_id: ResourceId1;
  role: PlacementRole;
  selector: PhysicalScopeSelector;
  shard_count?: ShardCount;
  shard_group_id?: ShardGroupId;
  shard_index?: ShardIndex;
  source_placement_id?: SourcePlacementId;
  state: PhysicalStateKind;
}
/**
 * One planned physical tier/device. This is scheduling intent, never measured residency.
 */
export interface PhysicalResource {
  device_id?: DeviceId;
  device_kind?: DeviceKind | null;
  resource_id: ResourceId2;
  storage?: PlannedStorageBinding | null;
  tier: MemoryTier;
}
/**
 * The exact StorageProfile assessment accepted by a plan. ``marginal``/``unknown`` are usable
 * only when that same verdict is explicitly recorded in ``accepted_suitability``; ``unsuitable`` is
 * always refused.
 */
export interface PlannedStorageBinding {
  accepted_suitability?: StorageSuitability;
  assessment: StorageRoleAssessment;
  path: Path1;
  role: StorageRole;
}
/**
 * The PER-ROLE verdict for a candidate path: can it play this role, and if not, WHY. The reasons
 * are the safe-spill guardrail's human-readable justification (USB bridge / synced folder / free-space
 * margin / inside the source repo / rotational disk).
 */
export interface StorageRoleAssessment {
  device_mount_point?: DeviceMountPoint;
  free_bytes?: FreeBytes;
  interface?: StorageInterface;
  path: Path;
  reasons?: Reasons;
  required_free_bytes?: RequiredFreeBytes;
  role: StorageRole;
  suitability: StorageSuitability1;
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
