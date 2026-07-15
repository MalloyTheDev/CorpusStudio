/* GENERATED from docs/contracts/ResolvedExecutionConfiguration.schema.json — do not edit. Run: npm run gen:contracts */

export type Bias = ("none" | "all" | "lora_only") | null;
export type LoraAlpha = number | null;
export type LoraDropout = number | null;
export type LoraR = number | null;
export type AdapterMethod =
  "none" | "lora" | "qlora" | "dora" | "ia3" | "full_finetune" | "prompt_tuning" | "prefix_tuning";
export type TargetModules = string[] | null;
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
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
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
export type Id = string;
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
export type FallbackGradAccumulationSteps = number | null;
export type MicroBatchSize = number;
export type SupervisedTokenAccumulationTarget = number;
export type Bnb4BitUseDoubleQuant = boolean;
export type CadenceOptimizerSteps = number | null;
export type CadenceSeconds = number | null;
export type CheckpointImpl = "full_state" | "adapter_only" | "sharded" | "distcp" | "safetensors";
export type KeepLast = number | null;
export type ReloadVerify = boolean;
export type ConfigurationHash = string;
export type ConfigurationId = string;
export type ContractVersion = "1.0.0";
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
export type ExportFormat =
  "adapter_peft" | "merged_safetensors" | "merged_fp16" | "gguf" | "onnx" | "awq" | "gptq" | "mlx";
export type GradientCheckpointing = boolean;
export type ContentSha256 = string | null;
export type Kind = "dataset" | "model" | "tokenizer";
export type Location = string;
export type ResolvedRevision = string | null;
export type Source1 = "local_file" | "local_directory" | "huggingface";
export type LossImpl = "cross_entropy" | "liger_fused_ce" | "chunked_ce" | "dpo" | "orpo" | "kto" | "ipo" | "reward_bt";
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
export type OutputDir = string;
export type OutputLayout = "run_scoped_v1";
export type PrecisionMode = "fp32" | "tf32" | "fp16" | "bf16" | "fp8" | "mixed_bf16" | "mixed_fp16";
export type PrecisionMode1 = "fp32" | "tf32" | "fp16" | "bf16" | "fp8" | "mixed_bf16" | "mixed_fp16";
export type OptimizerStateDtype = PrecisionMode | QuantizationMode;
export type QuantizationMode = "none" | "int8" | "int4" | "nf4" | "fp4" | "gptq" | "awq" | "hqq";
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
  contract_version?: ContractVersion;
  data: TrainingDataPolicy;
  data_seed?: DataSeed;
  device_map: DeviceMap;
  environment_binding: EnvironmentBinding;
  environment_ref: Ref;
  export_format: ExportFormat;
  gradient_checkpointing?: GradientCheckpointing;
  inputs: ExecutionInputs;
  loss_impl: LossImpl;
  objective_ref: Ref;
  optimizer: OptimizerSpec;
  output_dir: OutputDir;
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
export interface AdapterSpec {
  bias?: Bias;
  lora_alpha?: LoraAlpha;
  lora_dropout?: LoraDropout;
  lora_r?: LoraR;
  method: AdapterMethod;
  target_modules?: TargetModules;
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
/**
 * An algorithm-tagged digest. The engine emits sha256 today; the algo tag makes a future
 * migration additive (cf. versions/version_registry.FINGERPRINT_ALGO).
 */
export interface HashRef {
  algo?: Algo;
  value?: Value;
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
