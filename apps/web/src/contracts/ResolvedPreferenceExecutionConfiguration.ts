/* GENERATED from docs/contracts/ResolvedPreferenceExecutionConfiguration.schema.json — do not edit. Run: npm run gen:contracts */

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
export type ContractVersion1 = "1.0.0";
export type DataSeed = number;
export type FormatterId = string;
export type FormatterSha256 = string;
export type MaxLength = number;
export type MaxPromptLength = number;
export type PairSchema = "chosen_rejected" | "preference_pair";
export type TruncationPolicy = "refuse" | "allow";
export type DataSeed1 = number;
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
export type Beta = number;
export type LabelSmoothing = number;
export type LossType = "sigmoid" | "hinge" | "ipo";
export type Objective = "dpo";
export type Mode = "frozen_base" | "prior_adapter";
export type PrecomputeRefLogProbs = boolean;
export type RuntimeMode = "training" | "cpu_toy";
export type SaveStrategy = "no" | "steps";
export type MaxSteps = number | null;
export type NumTrainEpochs = number | null;
export type Seed = number;
export type Buckets = number[];
export type MaxSequenceLen = number;
export type Packing = boolean;
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
 * NOT yet consumed by a worker: the ``DPOTrainer`` branch, a workload-verified 4B run, and the
 * milestone wheel that promotes ``preference_dpo`` to ``workload_verified`` are a separate, gated
 * slice. This contract only makes a DPO run EXPRESSIBLE and hash-sealable so the architecture is
 * reviewable before any worker byte changes.
 */
export interface ResolvedPreferenceExecutionConfiguration {
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
  data: PreferenceDataPolicy;
  data_seed?: DataSeed1;
  device_map: DeviceMap;
  environment_binding: EnvironmentBinding;
  environment_ref: Ref;
  export_format: ExportFormat;
  gradient_checkpointing?: GradientCheckpointing;
  inputs: ExecutionInputs;
  objective_ref: Ref;
  optimizer: OptimizerSpec;
  output_dir: OutputDir;
  output_layout?: OutputLayout;
  precision: PrecisionExecutionPolicy;
  preference: PreferenceOptimizationSpec;
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
/**
 * Additive, dense/MoE-safe preference-pair data policy (S2 / DPO), PARALLEL to the SFT-only
 * ``TrainingDataPolicy`` - never reuse the SFT contract for preference pairs. It pins the exact
 * chosen/rejected pair schema + formatter + chat template + the DPO prompt/response length budget, so a
 * preference run formats every pair identically and refuses (never silently truncates) an over-length
 * prompt or response. The reference model + DPO loss hyperparameters live on the DPO execution seal
 * (a separate worker slice), not here - this is only the data contract.
 */
export interface PreferenceDataPolicy {
  chat_template_sha256?: ChatTemplateSha256;
  contract_version?: ContractVersion1;
  data_seed?: DataSeed;
  formatter_id: FormatterId;
  formatter_sha256: FormatterSha256;
  max_length: MaxLength;
  max_prompt_length: MaxPromptLength;
  pair_schema?: PairSchema;
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
/**
 * The offline preference-optimization loss (TRL ``DPOConfig``): the KL strength ``beta``, the
 * ``loss_type`` variant, conservative ``label_smoothing`` for noisy preferences, and the frozen
 * reference binding. Kept off :class:`PreferenceDataPolicy` (which is only the data contract) so the
 * data policy stays reusable by a non-DPO preference method later.
 */
export interface PreferenceOptimizationSpec {
  beta?: Beta;
  label_smoothing?: LabelSmoothing;
  loss_type?: LossType;
  objective?: Objective;
  reference_model: ReferenceModelBinding;
}
/**
 * The frozen reference policy an offline DPO run scores its trainable policy against.
 *
 * ``frozen_base`` is the standard QLoRA-DPO path: the reference IS the same quantized base with the
 * trainable adapter disabled, so TRL computes reference log-probs by turning the PEFT adapter off and
 * no second model is loaded. ``prior_adapter`` instead pins an explicit, previously-trained adapter as
 * the reference. ``precompute_ref_log_probs`` caches the reference log-probs once (a memory/throughput
 * trade the worker honors verbatim).
 */
export interface ReferenceModelBinding {
  adapter_ref?: Ref | null;
  mode?: Mode;
  precompute_ref_log_probs?: PrecomputeRefLogProbs;
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
  packing?: Packing;
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
