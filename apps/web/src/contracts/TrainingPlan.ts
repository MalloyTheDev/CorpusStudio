/* GENERATED from docs/contracts/TrainingPlan.schema.json — do not edit. Run: npm run gen:contracts */

export type CheckpointImpl = "full_state" | "adapter_only" | "sharded" | "distcp" | "safetensors";
export type EvaluationProfile = string | null;
export type Framework = string;
export type DeviceKind = "cuda" | "rocm" | "mps" | "xpu" | "cpu";
export type ModelTopology = "dense" | "moe";
export type ObjectiveId = string;
export type Orchestrator = string;
export type ParallelismKind = "data" | "tensor" | "pipeline" | "expert" | "sequence" | "context";
export type Parallelism = ParallelismKind[];
export type PrecisionMode = "fp32" | "tf32" | "fp16" | "bf16" | "fp8" | "mixed_bf16" | "mixed_fp16";
export type Preset = string | null;
export type QuantizationMode = "none" | "int8" | "int4" | "nf4" | "fp4" | "gptq" | "awq" | "hqq";
export type AdapterMethod =
  "none" | "lora" | "qlora" | "dora" | "ia3" | "full_finetune" | "prompt_tuning" | "prefix_tuning";
export type ContractVersion = "1.0.0";
export type CreatedAt = string | null;
export type AdamBeta1 = number;
export type AdamBeta2 = number;
export type AdamEpsilon = number;
export type AdapterMethod1 = string | null;
export type AllocatorGcThreshold = number | null;
export type AllocatorMaxSplitSizeMb = number | null;
export type AllocatorPolicy = string;
export type AllowCpuToy = boolean;
export type ArchitectureRefId = string | null;
export type ArchitectureRefSha256 = string | null;
export type AttentionBackend = string | null;
export type Backend = string;
export type BaseModel = string;
export type ChatTemplateSha256 = string | null;
export type CheckpointKeepLast = number | null;
export type CheckpointSteps = number | null;
export type CustomCodeBundleRefId = string | null;
export type CustomCodeBundleRefSha256 = string | null;
export type CustomCodeEntrySymbol = string | null;
export type CustomCodeInterfaceVersion = string | null;
export type CustomCodeVettingRefId = string | null;
export type CustomCodeVettingRefSha256 = string | null;
export type DataSeed = number | null;
export type DatasetContentSha256 = string | null;
export type DatasetFormat = string;
export type DatasetPath = string;
export type ExportFormat = string;
export type GradientAccumulationSteps = number;
export type InitInitializerRange = number | null;
export type InitMode = string | null;
export type InitSeed = number | null;
export type InitVocabSize = number | null;
export type LearningRate = number;
export type LoraAlpha = number;
export type LoraBias = string;
export type LoraDropout = number;
export type LoraR = number;
export type LoraTargetModules = string[];
export type LrScheduler = string;
export type MaxGradNorm = number;
export type MaxSteps = number | null;
export type MicroBatchSize = number;
export type ModelContentSha256 = string | null;
export type ModelRevision = string | null;
export type NumTrainEpochs = number;
export type ObjectiveId1 = string | null;
export type Optim = string;
export type OutputDir = string;
export type PreferenceBeta = number;
export type PreferenceLabelSmoothing = number;
export type PreferenceMaxPromptLength = number | null;
export type Seed = number;
export type SequenceLen = number;
export type SourceCheckpointRefId = string | null;
export type SourceCheckpointRefSha256 = string | null;
export type SupervisedTokenAccumulationTarget = number | null;
export type TaskType = string;
export type TokenizerAlgorithm = string | null;
export type TokenizerContentSha256 = string | null;
export type TokenizerMinFrequency = number | null;
export type TokenizerRevision = string | null;
export type TokenizerSourceMode = string | null;
export type TokenizerSpecialTokens = string[] | null;
export type TokenizerVocabSize = number | null;
export type TruncationAllowed = boolean;
export type UseLiger = boolean;
export type VerificationRequirement = string;
export type WarmupRatio = number;
export type WeightDecay = number;
export type PlanIntentId = string;

/**
 * A pre-resolution, user-facing composition of the training registries + free parameters (Training
 * Systems P0b, #482). It lowers into one-or-more RunPlans via the planner; it is NOT an execution
 * authority. Two invariants: (1) it carries NO ``plan_hash`` / ``configuration_hash``-sealed field -
 * RunPlan and ResolvedExecutionConfiguration remain the sole sealing authority; (2) any cross-
 * dimension compatibility check on it is an early UX pre-check, never the authoritative gate.
 */
export interface TrainingPlan {
  composition: TrainingPlanComposition;
  contract_version?: ContractVersion;
  created_at?: CreatedAt;
  parameters: TrainingPlanParameters;
  plan_intent_id: PlanIntentId;
}
/**
 * One entry selected per registry dimension (the architecture doc's registries). Pre-resolution
 * intent, dense-default and MoE-compatible - it never assumes dense execution. These are capability
 * SELECTIONS (refs into the thin registries), not sealed execution fields; the ``parameters`` block
 * carries the executable knobs and the compatibility pre-check validates the two are consistent.
 */
export interface TrainingPlanComposition {
  checkpoint_strategy?: CheckpointImpl;
  evaluation_profile?: EvaluationProfile;
  framework?: Framework;
  hardware_target?: DeviceKind;
  model_topology?: ModelTopology;
  objective_id: ObjectiveId;
  orchestrator?: Orchestrator;
  parallelism?: Parallelism;
  precision?: PrecisionMode;
  preset?: Preset;
  quantization?: QuantizationMode;
  update_method?: AdapterMethod;
}
/**
 * The free (numeric / identity) knobs of a training run - a serializable mirror of the planner's
 * ``PlannerConstraints`` user intent. The resolver lowers these verbatim into a RunPlan, so this is
 * the executable-knob source of truth; the registry SELECTIONS in ``TrainingPlanComposition`` are the
 * higher-level capability view. A drift test keeps this mirror field-for-field with
 * ``PlannerConstraints`` so the resolver can reproduce a direct build's ``plan_hash`` exactly.
 */
export interface TrainingPlanParameters {
  adam_beta1?: AdamBeta1;
  adam_beta2?: AdamBeta2;
  adam_epsilon?: AdamEpsilon;
  adapter_method?: AdapterMethod1;
  allocator_gc_threshold?: AllocatorGcThreshold;
  allocator_max_split_size_mb?: AllocatorMaxSplitSizeMb;
  allocator_policy?: AllocatorPolicy;
  allow_cpu_toy?: AllowCpuToy;
  architecture_ref_id?: ArchitectureRefId;
  architecture_ref_sha256?: ArchitectureRefSha256;
  attention_backend?: AttentionBackend;
  backend?: Backend;
  base_model: BaseModel;
  chat_template_sha256?: ChatTemplateSha256;
  checkpoint_keep_last?: CheckpointKeepLast;
  checkpoint_steps?: CheckpointSteps;
  custom_code_bundle_ref_id?: CustomCodeBundleRefId;
  custom_code_bundle_ref_sha256?: CustomCodeBundleRefSha256;
  custom_code_entry_symbol?: CustomCodeEntrySymbol;
  custom_code_interface_version?: CustomCodeInterfaceVersion;
  custom_code_vetting_ref_id?: CustomCodeVettingRefId;
  custom_code_vetting_ref_sha256?: CustomCodeVettingRefSha256;
  data_seed?: DataSeed;
  dataset_content_sha256?: DatasetContentSha256;
  dataset_format?: DatasetFormat;
  dataset_path: DatasetPath;
  export_format?: ExportFormat;
  gradient_accumulation_steps?: GradientAccumulationSteps;
  init_initializer_range?: InitInitializerRange;
  init_mode?: InitMode;
  init_seed?: InitSeed;
  init_vocab_size?: InitVocabSize;
  learning_rate?: LearningRate;
  lora_alpha?: LoraAlpha;
  lora_bias?: LoraBias;
  lora_dropout?: LoraDropout;
  lora_r?: LoraR;
  lora_target_modules?: LoraTargetModules;
  lr_scheduler?: LrScheduler;
  max_grad_norm?: MaxGradNorm;
  max_steps?: MaxSteps;
  micro_batch_size?: MicroBatchSize;
  model_content_sha256?: ModelContentSha256;
  model_revision?: ModelRevision;
  num_train_epochs?: NumTrainEpochs;
  objective_id?: ObjectiveId1;
  optim?: Optim;
  output_dir?: OutputDir;
  preference_beta?: PreferenceBeta;
  preference_label_smoothing?: PreferenceLabelSmoothing;
  preference_max_prompt_length?: PreferenceMaxPromptLength;
  seed?: Seed;
  sequence_len?: SequenceLen;
  source_checkpoint_ref_id?: SourceCheckpointRefId;
  source_checkpoint_ref_sha256?: SourceCheckpointRefSha256;
  supervised_token_accumulation_target?: SupervisedTokenAccumulationTarget;
  task_type?: TaskType;
  tokenizer_algorithm?: TokenizerAlgorithm;
  tokenizer_content_sha256?: TokenizerContentSha256;
  tokenizer_min_frequency?: TokenizerMinFrequency;
  tokenizer_revision?: TokenizerRevision;
  tokenizer_source_mode?: TokenizerSourceMode;
  tokenizer_special_tokens?: TokenizerSpecialTokens;
  tokenizer_vocab_size?: TokenizerVocabSize;
  truncation_allowed?: TruncationAllowed;
  use_liger?: UseLiger;
  verification_requirement?: VerificationRequirement;
  warmup_ratio?: WarmupRatio;
  weight_decay?: WeightDecay;
}
