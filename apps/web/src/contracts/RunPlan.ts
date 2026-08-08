/* GENERATED from docs/contracts/RunPlan.schema.json — do not edit. Run: npm run gen:contracts */

export type Bias = ("none" | "all" | "lora_only") | null;
export type LoraAlpha = number | null;
export type LoraDropout = number | null;
export type LoraR = number | null;
export type AdapterMethod =
  "none" | "lora" | "qlora" | "dora" | "ia3" | "full_finetune" | "prompt_tuning" | "prefix_tuning";
export type TargetModules = string[] | null;
export type AllocatorGcThreshold = number | null;
export type AllocatorMaxSplitSizeMb = number | null;
export type AllocatorPolicy = "default" | "expandable_segments" | "max_split_size" | "garbage_collection";
/**
 * ``math``/``eager`` is forced on native-Windows/WDDM Blackwell sm_120 because the fused flash
 * kernel deadlocks there. Other platforms require their own functional capability result; WSL
 * evidence is not bare-Linux proof.
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
  "adapter_peft" | "reward_model" | "merged_safetensors" | "merged_fp16" | "gguf" | "onnx" | "awq" | "gptq" | "mlx";
export type OutputDir = string;
export type GradientCheckpointing = boolean;
export type LossImpl =
  "cross_entropy" | "liger_fused_ce" | "chunked_ce" | "dpo" | "orpo" | "kto" | "ipo" | "reward_bt" | "grpo";
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
export type AdamBeta1 = number;
export type AdamBeta2 = number;
export type AdamEpsilon = number;
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
export type MaxGradNorm = number;
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
export type AdapterTaskType = "CAUSAL_LM";
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
export type EvidenceKind = "functional_probe" | "cpu_reference";
export type FallbackPolicy = "refuse";
export type Artifact = string | null;
export type Dependencies = string[];
export type Direct = boolean | null;
export type DirectUrl = string | null;
export type Editable = boolean | null;
/**
 * Number of files sealed by installed_files_hash; equals record_entries when record_integrity is verified.
 */
export type InstalledFileCount = number | null;
export type Installer = string | null;
export type Name = string;
export type NormalizedName = string;
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
export type Requested = boolean | null;
export type Source = "pypi" | "wheel" | "sdist" | "conda" | "vcs" | "local" | "unknown";
export type SourceEvidenceReason = string | null;
export type SourceIndexUrl = string | null;
export type VcsCommit = string | null;
export type VcsRepository = string | null;
export type Version = string | null;
export type FlashSdpEnabled = boolean;
export type MathSdpEnabled = boolean;
export type MemEfficientSdpEnabled = boolean;
/**
 * The model-loader API selected before execution.
 *
 * This is deliberately separate from :class:`AttentionKernel`: ``sdpa`` is an API that can
 * dispatch to several materially different PyTorch kernels.
 */
export type ModelAttentionApi = "eager" | "sdpa" | "flash_attention_2" | "flash_attention_3" | "xformers";
export type SafetyMandate = string | null;
/**
 * Whether the planner may seal a capability that lacks functional evidence.
 */
export type ExecutionVerificationRequirement = "require_verified" | "allow_unverified";
export type Bnb4BitUseDoubleQuant = boolean;
export type ConfigurationHash = string;
export type ConfigurationId = string;
export type ContractVersion1 = "1.0.0";
export type ChatTemplateSha256 = string | null;
export type DatasetFormat = "instruction" | "chat" | "trace";
export type DatasetTextField = string;
export type FormatterId = string;
export type FormatterSha256 = string;
export type Packing = boolean;
export type TruncationAnalysis = "full_pinned_dataset";
export type TruncationPolicy = "refuse" | "allow";
export type DataSeed = number;
/**
 * @minItems 1
 */
export type DeviceMap = [DeviceMapEntry, ...DeviceMapEntry[]];
export type Device = string;
export type Module = string;
export type EnvironmentBinding = "profile_snapshot" | "managed_lock";
export type GradientCheckpointing1 = boolean;
export type ContentSha256 = string | null;
export type Kind = "dataset" | "model" | "tokenizer";
export type Location = string;
export type ResolvedRevision = string | null;
export type Source1 = "local_file" | "local_directory" | "huggingface";
export type OutputDir1 = string;
export type OutputLayout = "run_scoped_v1";
export type PrecisionMode1 = "fp32" | "tf32" | "fp16" | "bf16" | "fp8" | "mixed_bf16" | "mixed_fp16";
export type OptimizerStateDtype = PrecisionMode | QuantizationMode;
export type QuantizationMode1 = "none" | "int8" | "int4" | "nf4" | "fp4" | "gptq" | "awq" | "hqq";
export type RuntimeMode = "training" | "cpu_toy";
export type SaveStrategy = "no" | "steps";
export type MaxSteps = number | null;
export type NumTrainEpochs = number | null;
export type Seed = number;
export type Buckets = number[];
export type MaxSequenceLen = number;
export type Packing1 = boolean;
export type TruncationAllowed = boolean;
export type DisableTqdm = boolean;
export type LoggingNanInfFilter = false | null;
export type LoggingSteps = 1;
export type LoggingStrategy = "steps" | null;
/**
 * @minItems 1
 */
export type PackageVersions = [PackageLock, ...PackageLock[]];
export type ReportTo = string[];
/**
 * @minItems 1
 */
export type RequiredSftConfigFields = [string, ...string[]];
export type SequenceLengthField = "max_seq_length" | "max_length";
export type TokenizerParameter = "tokenizer" | "processing_class";
export type TrustRemoteCode = false;
export type UseSafetensors = true;
export type AdapterTaskType1 = "CAUSAL_LM";
export type Bnb4BitUseDoubleQuant1 = boolean;
export type ConfigurationHash1 = string;
export type ConfigurationId1 = string;
export type ContractVersion2 = "1.0.0";
export type DataSeed1 = number;
/**
 * @minItems 1
 */
export type DeviceMap1 = [DeviceMapEntry, ...DeviceMapEntry[]];
export type EnvironmentBinding1 = "profile_snapshot" | "managed_lock";
export type GradientCheckpointing2 = boolean;
export type OutputDir2 = string;
export type OutputLayout1 = "run_scoped_v1";
export type RuntimeMode1 = "training" | "cpu_toy";
export type SaveStrategy1 = "no" | "steps";
export type Seed1 = number;
export type TrustRemoteCode1 = false;
export type UseSafetensors1 = true;
export type AdapterTaskType2 = "CAUSAL_LM";
export type Bnb4BitUseDoubleQuant2 = boolean;
export type ConfigurationHash2 = string;
export type ConfigurationId2 = string;
export type ContractVersion3 = "1.0.0";
export type ChatTemplateSha2561 = string | null;
export type ContractVersion4 = "1.0.0";
export type DataSeed2 = number;
export type FormatterId1 = string;
export type FormatterSha2561 = string;
export type MaxLength = number;
export type MaxPromptLength = number;
export type PairSchema = "chosen_rejected" | "preference_pair";
export type SchemaId = string;
export type SchemaSha256 = string;
export type SchemaVersion = string;
export type TruncationPolicy1 = "refuse" | "allow";
export type DataSeed3 = number;
/**
 * @minItems 1
 */
export type DeviceMap2 = [DeviceMapEntry, ...DeviceMapEntry[]];
export type EnvironmentBinding2 = "profile_snapshot" | "managed_lock";
export type GradientCheckpointing3 = boolean;
export type OutputDir3 = string;
export type OutputLayout2 = "run_scoped_v1";
export type AverageLogProb = boolean;
export type Beta = number;
export type LabelSmoothing = number;
export type LossType = "sigmoid";
export type Objective = "dpo";
export type Mode = "frozen_base";
export type PrecomputeRefLogProbs = boolean;
export type SequenceChunkSize = number;
export type RuntimeMode2 = "training" | "cpu_toy";
export type SaveStrategy2 = "no" | "steps";
export type Seed2 = number;
export type TrustRemoteCode2 = false;
export type UseSafetensors2 = true;
export type ConfigurationHash3 = string;
export type ConfigurationId3 = string;
export type ContractVersion5 = "1.0.0";
export type ContractVersion6 = "1.0.0";
export type DataSeed4 = number;
export type DocumentBoundaries = boolean;
export type Epochs = number | null;
export type GlobalBatchSize = number;
export type Packing2 = "none" | "concat_and_split" | "best_fit";
export type ContentSha2561 = string;
export type Location1 = string;
export type RowCount = number;
export type ShardId = string;
export type Source2 = string;
export type TokenCount = number;
export type Shards = PretrainingShard[];
export type Streaming = boolean;
export type TokenBudget = number | null;
export type DataSeed5 = number;
/**
 * @minItems 1
 */
export type DeviceMap3 = [DeviceMapEntry, ...DeviceMapEntry[]];
export type EnvironmentBinding3 = "profile_snapshot" | "managed_lock";
export type GradientCheckpointing4 = boolean;
export type EntrySymbol = string;
export type InterfaceVersion = "custom_decoder_v1";
export type TrustRemoteCode3 = false;
export type VettingVerdict = "admitted";
export type InitSeed = number | null;
export type InitializerRange = number | null;
export type MaxPositionEmbeddings = number | null;
export type Mode1 = "random" | "continued";
export type ResetDataCursor = boolean;
export type ResetLrScheduler = boolean;
export type ResetOptimizer = boolean;
export type VocabSize = number | null;
export type OutputDir4 = string;
export type OutputLayout3 = "run_scoped_v1";
export type RuntimeMode3 = "training" | "cpu_toy";
export type Seed3 = number;
export type Algorithm = ("bpe" | "unigram" | "wordpiece") | null;
export type MinFrequency = number | null;
export type Mode2 = "train" | "import" | "freeze";
export type SpecialTokens = string[] | null;
export type TokenizerContentSha256 = string | null;
export type TokenizerLocation = string | null;
export type VocabSize1 = number | null;
export type TrustRemoteCode4 = false;
export type UseSafetensors3 = true;
export type AdapterTaskType3 = "SEQ_CLS";
export type Bnb4BitUseDoubleQuant3 = boolean;
export type ConfigurationHash4 = string;
export type ConfigurationId4 = string;
export type ContractVersion7 = "1.0.0";
export type DataSeed6 = number;
/**
 * @minItems 1
 */
export type DeviceMap4 = [DeviceMapEntry, ...DeviceMapEntry[]];
export type EnvironmentBinding4 = "profile_snapshot" | "managed_lock";
export type GradientCheckpointing5 = boolean;
export type OutputDir5 = string;
export type OutputLayout4 = "run_scoped_v1";
export type Family = "pairwise";
export type LossType1 = "bradley_terry";
export type Margin = number;
export type OutputDirection = "higher_is_better";
export type ScorePooling = "last_token";
export type RuntimeMode4 = "training" | "cpu_toy";
export type SaveStrategy3 = "no" | "steps";
export type Seed4 = number;
export type TrustRemoteCode5 = false;
export type UseSafetensors4 = true;
export type AdapterTaskType4 = "CAUSAL_LM";
export type Bnb4BitUseDoubleQuant4 = boolean;
export type ConfigurationHash5 = string;
export type ConfigurationId5 = string;
export type ContractVersion8 = "1.0.0";
export type DataSeed7 = number;
/**
 * @minItems 1
 */
export type DeviceMap5 = [DeviceMapEntry, ...DeviceMapEntry[]];
export type EnvironmentBinding5 = "profile_snapshot" | "managed_lock";
export type ChatTemplateSha2562 = string | null;
export type ContractVersion9 = "1.0.0";
export type DataSeed8 = number;
export type FormatterId2 = string;
export type FormatterSha2562 = string;
export type MaxPromptLength1 = number;
export type Mode3 = "on_policy";
export type SchemaId1 = string;
export type SchemaSha2561 = string;
export type SchemaVersion1 = string;
export type TruncationPolicy2 = "refuse" | "allow";
export type GradientCheckpointing6 = boolean;
export type OutputDir6 = string;
export type OutputLayout5 = "run_scoped_v1";
export type Algorithm1 = "grpo" | "ppo";
export type ContractVersion10 = "1.0.0";
export type UseCritic = boolean;
export type ContractVersion11 = "1.0.0";
export type HigherIsBetter = boolean;
export type Kind1 = "served_reward_model" | "verifier" | "process_reward" | "rlaif_judge";
export type ContractVersion12 = "1.0.0";
export type DecodePolicy = "sanctioned_worker_decode";
export type MaxNewTokens = number;
export type RolloutsPerPrompt = number;
export type SamplingTemperature = number;
export type SamplingTopP = number;
export type RuntimeMode5 = "training" | "cpu_toy";
export type SaveStrategy4 = "no" | "steps";
export type Seed5 = number;
export type AdvantageNormalization = boolean;
export type ClipRange = number;
export type ContractVersion13 = "1.0.0";
export type EntropyBonus = number;
export type KlCoefficient = number;
export type KlTarget = number | null;
export type TrustRemoteCode6 = false;
export type UseSafetensors5 = true;
export type Seed6 = number;
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
  allocator_gc_threshold?: AllocatorGcThreshold;
  allocator_max_split_size_mb?: AllocatorMaxSplitSizeMb;
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
  resolved_execution?: ResolvedExecutionConfiguration | null;
  resolved_full_finetune_execution?: ResolvedFullFinetuneExecutionConfiguration | null;
  resolved_preference_execution?: ResolvedPreferenceExecutionConfiguration | null;
  resolved_pretraining_execution?: ResolvedPretrainingExecutionConfiguration | null;
  resolved_reward_execution?: ResolvedRewardExecutionConfiguration | null;
  resolved_rollout_execution?: ResolvedRolloutExecutionConfiguration | null;
  seed?: Seed6;
  sequence: SequenceSpec;
  task_type: TaskType;
  training_config_snapshot?: TrainingConfigSnapshot;
}
export interface AdapterSpec {
  bias?: Bias;
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
  adam_beta1?: AdamBeta1;
  adam_beta2?: AdamBeta2;
  adam_epsilon?: AdamEpsilon;
  impl: Optimizer;
  learning_rate: LearningRate;
  lr_scheduler?: LrScheduler;
  max_grad_norm?: MaxGradNorm;
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
 * The hash-sealed configuration consumed directly by an isolated training worker.
 *
 * It contains every execution-affecting default. Workers may refuse it, but may not fill in,
 * filter, reinterpret, or override semantic fields after this configuration is sealed.
 */
export interface ResolvedExecutionConfiguration {
  adapter: AdapterSpec;
  adapter_task_type?: AdapterTaskType;
  attention: AttentionExecutionPolicy;
  backend_ref: Ref;
  batching: BatchingSpec;
  bnb_4bit_use_double_quant: Bnb4BitUseDoubleQuant;
  capability_report_ref: Ref;
  checkpoint_policy: CheckpointPolicy;
  configuration_hash: ConfigurationHash;
  configuration_id: ConfigurationId;
  contract_version?: ContractVersion1;
  data: TrainingDataPolicy;
  data_seed?: DataSeed;
  device_map: DeviceMap;
  environment_binding: EnvironmentBinding;
  environment_ref: Ref;
  export_format: ExportFormat;
  gradient_checkpointing?: GradientCheckpointing1;
  inputs: ExecutionInputs;
  loss_impl: LossImpl;
  objective_ref: Ref;
  optimizer: OptimizerSpec;
  output_dir: OutputDir1;
  output_layout?: OutputLayout;
  precision: PrecisionExecutionPolicy;
  runtime_mode: RuntimeMode;
  save_strategy?: SaveStrategy;
  schedule: TrainingSchedule;
  seed?: Seed;
  sequence: SequenceSpec;
  trainer_interface: TrainerInterfacePolicy;
  trust_remote_code?: TrustRemoteCode;
  use_safetensors?: UseSafetensors;
}
/**
 * Exact model attention API plus the one runtime kernel that is permitted.
 */
export interface AttentionExecutionPolicy {
  effective_backend_required: AttentionKernel;
  evidence_kind: EvidenceKind;
  fallback_policy?: FallbackPolicy;
  flash_attention_package?: PackageLock | null;
  flash_sdp_enabled: FlashSdpEnabled;
  kernel_probe_ref: Ref;
  math_sdp_enabled: MathSdpEnabled;
  mem_efficient_sdp_enabled: MemEfficientSdpEnabled;
  model_attention_api: ModelAttentionApi;
  safety_mandate?: SafetyMandate;
  verification_requirement?: ExecutionVerificationRequirement;
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
  direct?: Direct;
  direct_url?: DirectUrl;
  editable?: Editable;
  hash?: HashRef | null;
  installed_file_count?: InstalledFileCount;
  installed_files_hash?: HashRef | null;
  installer?: Installer;
  name: Name;
  normalized_name?: NormalizedName;
  record_count_semantics?: RecordCountSemantics;
  record_entries?: RecordEntries;
  record_failed_entries?: RecordFailedEntries;
  record_integrity?: RecordIntegrity;
  record_verified_entries?: RecordVerifiedEntries;
  requested?: Requested;
  source?: Source;
  source_evidence_reason?: SourceEvidenceReason;
  source_index_url?: SourceIndexUrl;
  vcs_commit?: VcsCommit;
  vcs_repository?: VcsRepository;
  version?: Version;
}
export interface TrainingDataPolicy {
  chat_template_sha256?: ChatTemplateSha256;
  dataset_format: DatasetFormat;
  dataset_text_field?: DatasetTextField;
  formatter_id: FormatterId;
  formatter_sha256: FormatterSha256;
  packing?: Packing;
  truncation_analysis?: TruncationAnalysis;
  truncation_policy?: TruncationPolicy;
}
export interface DeviceMapEntry {
  device: Device;
  module: Module;
}
export interface ExecutionInputs {
  dataset: ExecutionInputBinding;
  model: ExecutionInputBinding;
  tokenizer: ExecutionInputBinding;
}
/**
 * One immutable input consumed by the worker.
 *
 * Local inputs pin the exact bytes (a stable file or directory digest). Hugging Face inputs pin an
 * immutable repository commit; a branch or tag is never sufficient execution identity.
 */
export interface ExecutionInputBinding {
  content_sha256?: ContentSha256;
  kind: Kind;
  location: Location;
  ref: Ref;
  resolved_revision?: ResolvedRevision;
  source: Source1;
}
/**
 * The numerical representation of each material training state.
 *
 * ``weight_storage_dtype`` describes an unquantized frozen base; quantized bases use
 * ``quantized_storage_format`` instead. ``master_weight_dtype`` describes the trainable adapter
 * parameters. An 8-bit optimizer may use quantized primary state plus FP32 auxiliary tensors.
 */
export interface PrecisionExecutionPolicy {
  dequantization_dtype: PrecisionMode;
  forward_compute_dtype: PrecisionMode;
  gradient_dtype: PrecisionMode;
  master_weight_dtype?: PrecisionMode | null;
  optimizer_auxiliary_dtype?: PrecisionMode1;
  optimizer_state_dtype: OptimizerStateDtype;
  quantized_storage_format?: QuantizationMode1;
  weight_storage_dtype?: PrecisionMode | null;
}
export interface TrainingSchedule {
  max_steps?: MaxSteps;
  num_train_epochs?: NumTrainEpochs;
}
/**
 * Sequence handling. Buckets let variable-length rows batch efficiently; the max bucket ==
 * the trainer's sequence_len (config_templates.sequence_len, default 4096).
 */
export interface SequenceSpec {
  buckets?: Buckets;
  max_sequence_len?: MaxSequenceLen;
  packing?: Packing1;
  truncation_allowed?: TruncationAllowed;
}
/**
 * Version- and field-exact adapter to the installed TRL/Transformers surface.
 */
export interface TrainerInterfacePolicy {
  disable_tqdm?: DisableTqdm;
  logging_nan_inf_filter?: LoggingNanInfFilter;
  logging_steps?: LoggingSteps;
  logging_strategy?: LoggingStrategy;
  package_versions: PackageVersions;
  report_to?: ReportTo;
  required_sft_config_fields: RequiredSftConfigFields;
  sequence_length_field: SequenceLengthField;
  tokenizer_parameter: TokenizerParameter;
}
/**
 * The hash-sealed configuration for a FULL-PARAMETER supervised fine-tune - the full-model sibling of
 * the adapter-only :class:`ResolvedExecutionConfiguration`. Same SFT data + objective path, but ALL model
 * parameters are trainable and the artifact is a full model (merged safetensors + full-state checkpoints),
 * so it is its OWN sealed config: the SFT config's validator hard-requires a PEFT adapter + adapter-only
 * export, which a full fine-tune cannot satisfy. Consumed directly by the first-party full-parameter
 * worker; refused at execution until dense_full_finetune is workload_verified.
 */
export interface ResolvedFullFinetuneExecutionConfiguration {
  adapter: AdapterSpec;
  adapter_task_type?: AdapterTaskType1;
  attention: AttentionExecutionPolicy;
  backend_ref: Ref;
  batching: BatchingSpec;
  bnb_4bit_use_double_quant: Bnb4BitUseDoubleQuant1;
  capability_report_ref: Ref;
  checkpoint_policy: CheckpointPolicy;
  configuration_hash: ConfigurationHash1;
  configuration_id: ConfigurationId1;
  contract_version?: ContractVersion2;
  data: TrainingDataPolicy;
  data_seed?: DataSeed1;
  device_map: DeviceMap1;
  environment_binding: EnvironmentBinding1;
  environment_ref: Ref;
  export_format: ExportFormat;
  gradient_checkpointing?: GradientCheckpointing2;
  inputs: ExecutionInputs;
  loss_impl: LossImpl;
  objective_ref: Ref;
  optimizer: OptimizerSpec;
  output_dir: OutputDir2;
  output_layout?: OutputLayout1;
  precision: PrecisionExecutionPolicy;
  runtime_mode: RuntimeMode1;
  save_strategy?: SaveStrategy1;
  schedule: TrainingSchedule;
  seed?: Seed1;
  sequence: SequenceSpec;
  trainer_interface: TrainerInterfacePolicy;
  trust_remote_code?: TrustRemoteCode1;
  use_safetensors?: UseSafetensors1;
}
/**
 * The hash-sealed configuration for an offline preference-optimization (DPO) run - the sibling of
 * :class:`ResolvedExecutionConfiguration` for the ``preference_dpo`` execution variant.
 *
 * The dense-QLoRA-SFT seal is byte-locked two ways (its own ``configuration_hash`` AND a committed
 * semantic golden over its full field set), so DPO's execution semantics live on THIS separate
 * contract, never as new fields on the SFT config. It reuses every shared execution sub-spec
 * (placement / precision / attention / adapter / optimizer / sequence / batching / checkpoint /
 * schedule / trainer interface) and adds only what DPO needs: a :class:`PreferenceDataPolicy` (never
 * the SFT ``TrainingDataPolicy``) and a :class:`PreferenceOptimizationSpec` (beta / loss / reference).
 *
 * Carried on ``RunPlan.resolved_preference_execution`` (a plan holds EITHER the SFT config OR this
 * one, never both). The matching adapter-based ``dpo_qlora`` objective exists and the admission gate
 * binds the ``preference_dpo`` variant to it; the resolver (a later control-plane step) populates
 * ``objective_ref`` with that objective's sealed identity - the pure contract cannot dereference the
 * registry itself. What remains gated is EXECUTION: the ``DPOTrainer`` branch, a workload-verified 4B
 * run, and the milestone wheel that promotes ``preference_dpo`` to ``workload_verified``.
 * ``trainer_interface`` is reused as an execution-shaped placeholder; the exact ``DPOConfig`` trainer
 * surface (distinct from ``SFTConfig``) is sealed with the ``DPOTrainer`` branch, since the contract is
 * not yet executable.
 */
export interface ResolvedPreferenceExecutionConfiguration {
  adapter: AdapterSpec;
  adapter_task_type?: AdapterTaskType2;
  attention: AttentionExecutionPolicy;
  backend_ref: Ref;
  batching: BatchingSpec;
  bnb_4bit_use_double_quant: Bnb4BitUseDoubleQuant2;
  capability_report_ref: Ref;
  checkpoint_policy: CheckpointPolicy;
  configuration_hash: ConfigurationHash2;
  configuration_id: ConfigurationId2;
  contract_version?: ContractVersion3;
  data: PreferenceDataPolicy;
  data_seed?: DataSeed3;
  device_map: DeviceMap2;
  environment_binding: EnvironmentBinding2;
  environment_ref: Ref;
  export_format: ExportFormat;
  gradient_checkpointing?: GradientCheckpointing3;
  inputs: ExecutionInputs;
  objective_ref: Ref;
  optimizer: OptimizerSpec;
  output_dir: OutputDir3;
  output_layout?: OutputLayout2;
  precision: PrecisionExecutionPolicy;
  preference: PreferenceOptimizationSpec;
  runtime_mode: RuntimeMode2;
  save_strategy?: SaveStrategy2;
  schedule: TrainingSchedule;
  seed?: Seed2;
  sequence: SequenceSpec;
  trainer_interface: TrainerInterfacePolicy;
  trust_remote_code?: TrustRemoteCode2;
  use_safetensors?: UseSafetensors2;
}
/**
 * Additive, dense/MoE-safe preference-pair data policy (S2 / DPO), PARALLEL to the SFT-only
 * ``TrainingDataPolicy`` - never reuse the SFT contract for preference pairs. It seals the RESOLVED
 * dataset schema identity - ``schema_id`` + ``schema_version`` + ``schema_sha256`` (the content digest
 * of the resolved schema) - so a consumer fails closed on a row-layout change even when a project-local
 * schema shadows the builtin and edits fields without bumping the version; plus the pair render layout,
 * formatter + chat template, and the DPO prompt/response length budget - so a preference run formats
 * every pair identically and refuses (never silently truncates) an over-length prompt or response. The
 * reference model + DPO loss hyperparameters live on the DPO execution seal (a separate worker slice),
 * not here - this is only the data contract.
 */
export interface PreferenceDataPolicy {
  chat_template_sha256?: ChatTemplateSha2561;
  contract_version?: ContractVersion4;
  data_seed?: DataSeed2;
  formatter_id: FormatterId1;
  formatter_sha256: FormatterSha2561;
  max_length: MaxLength;
  max_prompt_length: MaxPromptLength;
  pair_schema?: PairSchema;
  schema_id: SchemaId;
  schema_sha256: SchemaSha256;
  schema_version: SchemaVersion;
  truncation_policy?: TruncationPolicy1;
}
/**
 * The offline preference-optimization loss (TRL ``DPOConfig``): the KL strength ``beta``, the
 * ``loss_type`` variant, conservative ``label_smoothing`` for noisy preferences, and the frozen
 * reference binding. Kept off :class:`PreferenceDataPolicy` (which is only the data contract) so the
 * data policy stays reusable by a non-DPO preference method later.
 */
export interface PreferenceOptimizationSpec {
  average_log_prob?: AverageLogProb;
  beta?: Beta;
  label_smoothing?: LabelSmoothing;
  loss_type?: LossType;
  objective?: Objective;
  reference_model: ReferenceModelBinding;
  sequence_chunk_size?: SequenceChunkSize;
}
/**
 * The frozen reference policy an offline DPO run scores its trainable policy against.
 *
 * Only ``frozen_base`` is sealed in this slice: the reference IS the same quantized base with the
 * trainable PEFT adapter disabled, so TRL computes reference log-probs by turning the adapter off and
 * no second model is loaded - nothing beyond the already-sealed base/adapter needs an execution-input
 * binding. Chaining a previously-trained adapter as the reference (a future ``prior_adapter`` mode) is
 * deferred to the worker slice, where it must carry its OWN immutable ``ExecutionInputBinding``
 * (location + content digest) so its bytes are covered by the execution-input verification path - a
 * bare hash ref would not be loadable or verifiable. ``precompute_ref_log_probs`` caches the reference
 * log-probs once (a memory/throughput trade the worker honors verbatim).
 */
export interface ReferenceModelBinding {
  mode?: Mode;
  precompute_ref_log_probs?: PrecomputeRefLogProbs;
}
/**
 * The hash-sealed configuration for a from-scratch / continued PRETRAINING run - the sibling of
 * :class:`ResolvedExecutionConfiguration` for the ``pretraining`` execution variant.
 *
 * The dense-QLoRA-SFT seal is byte-locked (its own ``configuration_hash`` AND a committed semantic
 * golden), so pretraining semantics live on THIS separate contract, never as new fields on the SFT
 * config. Unlike SFT/DPO this is a FULL-PARAMETER causal-LM run: there is no adapter, no 4-bit base,
 * and no single dataset file. The three input kinds are captured by method sub-specs rather than the
 * SFT-shaped ``ExecutionInputs`` (which fail-closed requires one local dataset file + a model-weights
 * binding, neither of which a from-scratch run has): the model by a :class:`ModelInitializationSpec`
 * (random from a config, or a continued checkpoint), the tokenizer by a :class:`TokenizerSourceSpec`,
 * and the corpus by the sharded :class:`PretrainingDataPolicy`.
 *
 * Carried on ``RunPlan.resolved_pretraining_execution`` (a plan carries exactly one of the SFT /
 * preference / pretraining configs). What remains gated is EXECUTION: the pretraining worker loop
 * (``from_config`` init, corpus streaming, packing, per-rank cursor), a workload-verified run, and the
 * milestone wheel that promotes ``pretraining`` to ``workload_verified``. ``trainer_interface`` is an
 * execution-shaped placeholder until that worker seals the exact trainer surface.
 */
export interface ResolvedPretrainingExecutionConfiguration {
  attention: AttentionExecutionPolicy;
  backend_ref: Ref;
  batching: BatchingSpec;
  capability_report_ref: Ref;
  checkpoint_policy: CheckpointPolicy;
  configuration_hash: ConfigurationHash3;
  configuration_id: ConfigurationId3;
  contract_version?: ContractVersion5;
  data: PretrainingDataPolicy;
  data_seed?: DataSeed5;
  device_map: DeviceMap3;
  environment_binding: EnvironmentBinding3;
  environment_ref: Ref;
  export_format: ExportFormat;
  gradient_checkpointing?: GradientCheckpointing4;
  init: ModelInitializationSpec;
  loss_impl: LossImpl;
  objective_ref: Ref;
  optimizer: OptimizerSpec;
  output_dir: OutputDir4;
  output_layout?: OutputLayout3;
  precision: PrecisionExecutionPolicy;
  runtime_mode: RuntimeMode3;
  schedule: TrainingSchedule;
  seed?: Seed3;
  sequence: SequenceSpec;
  tokenizer_source: TokenizerSourceSpec;
  trainer_interface: TrainerInterfacePolicy;
  trust_remote_code?: TrustRemoteCode4;
  use_safetensors?: UseSafetensors3;
}
/**
 * Additive, dense/MoE-safe pretraining data policy (#487), PARALLEL to the SFT-only
 * ``TrainingDataPolicy`` - never reuse the SFT contract for a sharded / streamed / mixture-weighted
 * corpus. It declares a content-hashed shard set, streaming, per-source mixture weights, document
 * boundaries, pretraining packing, a seeded deterministic global order, and a stop condition (token
 * budget and/or epochs) so a run stops at the budget and never silently truncates. The runtime
 * per-rank data cursor + streaming resume is a separate (worker) slice.
 */
export interface PretrainingDataPolicy {
  contract_version?: ContractVersion6;
  data_seed: DataSeed4;
  document_boundaries?: DocumentBoundaries;
  epochs?: Epochs;
  global_batch_size: GlobalBatchSize;
  mixture_weights?: MixtureWeights;
  packing?: Packing2;
  shards: Shards;
  streaming?: Streaming;
  token_budget?: TokenBudget;
}
export interface MixtureWeights {
  [k: string]: number;
}
/**
 * One content-hashed corpus shard in a :class:`PretrainingDataPolicy`: a stable id + location, its
 * row and token counts, its sha256, and the mixture source it belongs to. The token count feeds the
 * token budget; the sha256 pins the exact bytes so a resumed stream reads the same shard.
 */
export interface PretrainingShard {
  content_sha256: ContentSha2561;
  location: Location1;
  row_count: RowCount;
  shard_id: ShardId;
  source?: Source2;
  token_count: TokenCount;
}
/**
 * How a PRETRAINING run instantiates its model. From-scratch has NO source weights: it builds a
 * model from an architecture config with reproducible random init (the worker's ``from_config`` path,
 * never ``from_pretrained``), pinned by ``architecture_ref`` + ``vocab_size`` + ``init_seed``.
 * Continued pretraining loads a hash-pinned ``source_checkpoint_ref`` and states explicitly what is
 * reset (optimizer / lr scheduler / data cursor) vs carried. Dense/MoE-safe: this seals init INTENT
 * and assumes no dense-specific model shape.
 */
export interface ModelInitializationSpec {
  architecture_ref?: Ref | null;
  custom_code?: CustomModelCodeSpec | null;
  init_seed?: InitSeed;
  initializer_range?: InitializerRange;
  max_position_embeddings?: MaxPositionEmbeddings;
  mode: Mode1;
  reset_data_cursor?: ResetDataCursor;
  reset_lr_scheduler?: ResetLrScheduler;
  reset_optimizer?: ResetOptimizer;
  source_checkpoint_ref?: Ref | null;
  vocab_size?: VocabSize;
}
/**
 * A hash-pinned, ADMITTED local custom-block bundle for a from-scratch run - the mode-3 'your own
 * model code' path (your own IMPLEMENTATION, the only not-borrowed mode). It seals WHICH exact bytes
 * (``code_bundle_ref``) an ADMITTED :class:`ModelCodeVettingReport` (``vetting_ref``) screened, plus the
 * entry class + interface. This path NEVER uses HF ``trust_remote_code`` (``Literal[False]``); the
 * module is loaded locally, by path, from the pinned bundle.
 *
 * Sealing this admits the design AT PLANNING; a static screen is not a safety proof, so EXECUTION stays
 * gated behind the (later) worker sandbox exactly as pretraining itself is refused at the worker today.
 * Both refs must be hash-pinned so admission binds to specific bytes and cannot silently re-point.
 */
export interface CustomModelCodeSpec {
  code_bundle_ref: Ref;
  entry_symbol: EntrySymbol;
  interface_version: InterfaceVersion;
  trust_remote_code?: TrustRemoteCode3;
  vetting_ref: Ref;
  vetting_verdict: VettingVerdict;
}
/**
 * How the tokenizer is obtained, frozen by hash BEFORE any token is consumed (a tokenizer change
 * invalidates all downstream token accounting). ``train`` builds a NEW tokenizer from a corpus sample
 * (a new subsystem the worker slice implements); ``import`` / ``freeze`` pin an existing tokenizer by
 * its content digest exactly as the SFT path does today.
 */
export interface TokenizerSourceSpec {
  algorithm?: Algorithm;
  min_frequency?: MinFrequency;
  mode: Mode2;
  special_tokens?: SpecialTokens;
  tokenizer_content_sha256?: TokenizerContentSha256;
  tokenizer_location?: TokenizerLocation;
  vocab_size?: VocabSize1;
}
/**
 * The hash-sealed configuration for a pairwise reward-model run - the sibling of
 * :class:`ResolvedExecutionConfiguration` for the ``reward_model`` execution variant (RL slice S5a-1).
 *
 * Like the DPO seal, it reuses every shared execution sub-spec (placement / precision / attention /
 * adapter / optimizer / sequence / batching / checkpoint / schedule / trainer interface) and reuses the
 * :class:`PreferenceDataPolicy` (a reward model trains on the SAME chosen/rejected pairs), adding only
 * a :class:`RewardModelingSpec`. It binds the declared ``reward_model`` objective. Two things differ
 * from every policy config: the adapter task type is ``SEQ_CLS`` (a scalar score head, not
 * ``CAUSAL_LM``), and the export format is ``reward_model`` (a new artifact family, not an adapter).
 *
 * Carried on ``RunPlan.resolved_reward_execution`` (a plan holds EXACTLY ONE execution config). The
 * contract + resolver are the control plane; EXECUTION (the reward-head trainer branch + a
 * workload-verified run + the promoting wheel) remains gated - ``reward_model`` stays
 * ``contract_validated`` until a measured run promotes it.
 */
export interface ResolvedRewardExecutionConfiguration {
  adapter: AdapterSpec;
  adapter_task_type?: AdapterTaskType3;
  attention: AttentionExecutionPolicy;
  backend_ref: Ref;
  batching: BatchingSpec;
  bnb_4bit_use_double_quant: Bnb4BitUseDoubleQuant3;
  capability_report_ref: Ref;
  checkpoint_policy: CheckpointPolicy;
  configuration_hash: ConfigurationHash4;
  configuration_id: ConfigurationId4;
  contract_version?: ContractVersion7;
  data: PreferenceDataPolicy;
  data_seed?: DataSeed6;
  device_map: DeviceMap4;
  environment_binding: EnvironmentBinding4;
  environment_ref: Ref;
  export_format: ExportFormat;
  gradient_checkpointing?: GradientCheckpointing5;
  inputs: ExecutionInputs;
  objective_ref: Ref;
  optimizer: OptimizerSpec;
  output_dir: OutputDir5;
  output_layout?: OutputLayout4;
  precision: PrecisionExecutionPolicy;
  reward: RewardModelingSpec;
  runtime_mode: RuntimeMode4;
  save_strategy?: SaveStrategy3;
  schedule: TrainingSchedule;
  seed?: Seed4;
  sequence: SequenceSpec;
  trainer_interface: TrainerInterfacePolicy;
  trust_remote_code?: TrustRemoteCode5;
  use_safetensors?: UseSafetensors4;
}
/**
 * The reward-model training shape for a PAIRWISE (Bradley-Terry) reward model - the sibling of
 * :class:`PreferenceOptimizationSpec` for the ``reward_model`` execution variant. Kept off
 * :class:`PreferenceDataPolicy` (reused verbatim as the chosen/rejected data contract) so the data
 * policy stays method-agnostic. A reward model's output is a NEW artifact family: a scalar SCORE HEAD
 * over the base (a SEQ_CLS ``num_labels=1`` projection), NOT a policy adapter and NOT causal-LM
 * generation.
 *
 * Only the pairwise Bradley-Terry family is admitted here; scalar-pointwise / process / outcome /
 * generative-verifier reward families are DISTINCT (different output artifacts + eval semantics) and are
 * added as their own specs + objectives, never silently under this seal (each gates independently).
 */
export interface RewardModelingSpec {
  family?: Family;
  loss_type?: LossType1;
  margin?: Margin;
  output_direction?: OutputDirection;
  score_pooling?: ScorePooling;
}
/**
 * The hash-sealed configuration for an on-policy RL run - the sibling of
 * :class:`ResolvedExecutionConfiguration` for the ``on_policy_rl`` execution variant (RL slice S5b,
 * gated L1 design #839).
 *
 * Like the reward seal, it reuses every shared execution sub-spec (placement / precision / attention /
 * adapter / optimizer / sequence / batching / checkpoint / schedule / trainer interface) and adds the
 * on-policy specs: a :class:`RolloutSpec` (generation), an :class:`ExperienceSource` (on-policy prompt
 * stream), a :class:`RewardSourceRef` (what scores rollouts), a :class:`StabilityController`
 * (KL/entropy/clip), and a :class:`PolicyOptimizationSpec` (GRPO now, PPO in S5c). Unlike reward it
 * trains a CAUSAL_LM POLICY adapter (``adapter_task_type='CAUSAL_LM'``) and exports an ``adapter_peft``
 * artifact - a policy, not a score head.
 *
 * Carried on ``RunPlan.resolved_rollout_execution`` (a plan holds EXACTLY ONE execution config). The
 * contract is the control plane; EXECUTION (the rollout+reward+GRPO worker + a workload-verified run)
 * stays gated - ``on_policy_rl`` remains ``contract_validated`` until a measured run promotes it.
 */
export interface ResolvedRolloutExecutionConfiguration {
  adapter: AdapterSpec;
  adapter_task_type?: AdapterTaskType4;
  attention: AttentionExecutionPolicy;
  backend_ref: Ref;
  batching: BatchingSpec;
  bnb_4bit_use_double_quant: Bnb4BitUseDoubleQuant4;
  capability_report_ref: Ref;
  checkpoint_policy: CheckpointPolicy;
  configuration_hash: ConfigurationHash5;
  configuration_id: ConfigurationId5;
  contract_version?: ContractVersion8;
  data_seed?: DataSeed7;
  device_map: DeviceMap5;
  environment_binding: EnvironmentBinding5;
  environment_ref: Ref;
  experience: ExperienceSource;
  export_format: ExportFormat;
  gradient_checkpointing?: GradientCheckpointing6;
  inputs: ExecutionInputs;
  objective_ref: Ref;
  optimizer: OptimizerSpec;
  output_dir: OutputDir6;
  output_layout?: OutputLayout5;
  policy_optimization: PolicyOptimizationSpec;
  precision: PrecisionExecutionPolicy;
  reward_source: RewardSourceRef;
  rollout: RolloutSpec;
  runtime_mode: RuntimeMode5;
  save_strategy?: SaveStrategy4;
  schedule: TrainingSchedule;
  seed?: Seed5;
  sequence: SequenceSpec;
  stability: StabilityController;
  trainer_interface: TrainerInterfacePolicy;
  trust_remote_code?: TrustRemoteCode6;
  use_safetensors?: UseSafetensors5;
}
/**
 * The on-policy experience buffer + the PROMPT dataset identity it draws from (S5b). ``mode`` is
 * ``on_policy``: completions are generated FRESH from the current policy each iteration (a streaming
 * source distinct from a static dataset - it ties to the G2 data-cursor gap), never replayed
 * (off-policy replay is a later variant). It seals the resolved PROMPT-dataset schema identity
 * (``schema_id`` + ``schema_version`` + ``schema_sha256``, the content digest) + the prompt formatter +
 * the prompt length budget, so an over-length prompt is refused (never silently truncated), exactly like
 * the preference / SFT data policies. Prompts only - NOT chosen/rejected pairs.
 */
export interface ExperienceSource {
  chat_template_sha256?: ChatTemplateSha2562;
  contract_version?: ContractVersion9;
  data_seed?: DataSeed8;
  formatter_id: FormatterId2;
  formatter_sha256: FormatterSha2562;
  max_prompt_length: MaxPromptLength1;
  mode?: Mode3;
  schema_id: SchemaId1;
  schema_sha256: SchemaSha2561;
  schema_version: SchemaVersion1;
  truncation_policy?: TruncationPolicy2;
}
/**
 * The on-policy optimization algorithm (S5b). GRPO (group-relative advantage) needs NO critic - it is
 * the cheaper shape that fits the 12 GB envelope like DPO/reward did; PPO (a clipped surrogate with a
 * value head) is the S5c follow-up. The reference model is the frozen base reached via
 * ``disable_adapter`` (the DPO pattern), so no separate reference weights are stored.
 */
export interface PolicyOptimizationSpec {
  algorithm?: Algorithm1;
  contract_version?: ContractVersion10;
  use_critic?: UseCritic;
}
/**
 * The hash-pinned reference to what scores each rollout (S5b). A reward model produced by the S5a
 * reward vertical is the primary source, served for inference-only scoring; rule / verifier rewards and
 * an RLAIF judge (Evaluation Studio's judge under the provider policy) are the declared alternatives.
 * The reference is hash-pinned so a run cannot silently swap the reward function. Admissibility (e.g. a
 * served reward model must itself be ``workload_verified``) is enforced by the resolver + runner, not
 * here.
 */
export interface RewardSourceRef {
  contract_version?: ContractVersion11;
  higher_is_better?: HigherIsBetter;
  kind: Kind1;
  reward_ref: Ref;
}
/**
 * The GENERATION phase of an on-policy RL run (S5b): how completions are sampled from the current
 * policy to form the experience the update is computed over. Sampling MUST be stochastic (temperature > 0)
 * - a greedy rollout collapses the group and yields a zero-variance GRPO advantage. ``rollouts_per_prompt``
 * is the GRPO group size (>= 2 so the group-relative advantage is defined). Generation runs on the
 * sanctioned worker decode path, never an unsanctioned generation path.
 */
export interface RolloutSpec {
  contract_version?: ContractVersion12;
  decode_policy?: DecodePolicy;
  max_new_tokens: MaxNewTokens;
  rollouts_per_prompt: RolloutsPerPrompt;
  sampling_temperature: SamplingTemperature;
  sampling_top_p: SamplingTopP;
}
/**
 * The sealed on-policy stability controls (S5b) - the guardrails that keep an on-policy update from
 * reward-hacking or collapsing. The KL-to-reference penalty anchors the policy to the frozen base; the
 * entropy bonus preserves exploration; advantage normalization stabilizes the group-relative signal; the
 * clip range bounds the per-step policy change. Sealed like any execution field - no silent defaults.
 */
export interface StabilityController {
  advantage_normalization?: AdvantageNormalization;
  clip_range: ClipRange;
  contract_version?: ContractVersion13;
  entropy_bonus?: EntropyBonus;
  kl_coefficient: KlCoefficient;
  kl_target?: KlTarget;
}
export interface TrainingConfigSnapshot {
  [k: string]: unknown;
}
