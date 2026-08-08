/* GENERATED from docs/contracts/ResolvedRolloutExecutionConfiguration.schema.json — do not edit. Run: npm run gen:contracts */

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
export type DataSeed = number;
/**
 * @minItems 1
 */
export type DeviceMap = [DeviceMapEntry, ...DeviceMapEntry[]];
export type Device = string;
export type Module = string;
export type EnvironmentBinding = "profile_snapshot" | "managed_lock";
export type ChatTemplateSha256 = string | null;
export type ContractVersion1 = "1.0.0";
export type DataSeed1 = number;
export type FormatterId = string;
export type FormatterSha256 = string;
export type MaxPromptLength = number;
export type Mode = "on_policy";
export type SchemaId = string;
export type SchemaSha256 = string;
export type SchemaVersion = string;
export type TruncationPolicy = "refuse" | "allow";
export type ExportFormat =
  "adapter_peft" | "reward_model" | "merged_safetensors" | "merged_fp16" | "gguf" | "onnx" | "awq" | "gptq" | "mlx";
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
export type Algorithm = "grpo" | "ppo";
export type ContractVersion2 = "1.0.0";
export type UseCritic = boolean;
export type PrecisionMode = "fp32" | "tf32" | "fp16" | "bf16" | "fp8" | "mixed_bf16" | "mixed_fp16";
export type PrecisionMode1 = "fp32" | "tf32" | "fp16" | "bf16" | "fp8" | "mixed_bf16" | "mixed_fp16";
export type OptimizerStateDtype = PrecisionMode | QuantizationMode;
export type QuantizationMode = "none" | "int8" | "int4" | "nf4" | "fp4" | "gptq" | "awq" | "hqq";
export type QuantizationMode1 = "none" | "int8" | "int4" | "nf4" | "fp4" | "gptq" | "awq" | "hqq";
export type ContractVersion3 = "1.0.0";
export type HigherIsBetter = boolean;
export type Kind1 = "served_reward_model" | "verifier" | "process_reward" | "rlaif_judge";
export type RewardAdapterLocation = string | null;
export type RewardBaseModel = string | null;
export type ContractVersion4 = "1.0.0";
export type DecodePolicy = "sanctioned_worker_decode";
export type MaxNewTokens = number;
export type RolloutsPerPrompt = number;
export type SamplingTemperature = number;
export type SamplingTopP = number;
export type RuntimeMode = "training" | "cpu_toy";
export type SaveStrategy = "no" | "steps";
export type MaxSteps = number | null;
export type NumTrainEpochs = number | null;
export type Seed = number;
export type Buckets = number[];
export type MaxSequenceLen = number;
export type Packing = boolean;
export type TruncationAllowed = boolean;
export type AdvantageNormalization = boolean;
export type ClipRange = number;
export type ContractVersion5 = "1.0.0";
export type EntropyBonus = number;
export type KlCoefficient = number;
export type KlTarget = number | null;
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
  data_seed?: DataSeed;
  device_map: DeviceMap;
  environment_binding: EnvironmentBinding;
  environment_ref: Ref;
  experience: ExperienceSource;
  export_format: ExportFormat;
  gradient_checkpointing?: GradientCheckpointing;
  inputs: ExecutionInputs;
  objective_ref: Ref;
  optimizer: OptimizerSpec;
  output_dir: OutputDir;
  output_layout?: OutputLayout;
  policy_optimization: PolicyOptimizationSpec;
  precision: PrecisionExecutionPolicy;
  reward_source: RewardSourceRef;
  rollout: RolloutSpec;
  runtime_mode: RuntimeMode;
  save_strategy?: SaveStrategy;
  schedule: TrainingSchedule;
  seed?: Seed;
  sequence: SequenceSpec;
  stability: StabilityController;
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
export interface DeviceMapEntry {
  device: Device;
  module: Module;
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
  chat_template_sha256?: ChatTemplateSha256;
  contract_version?: ContractVersion1;
  data_seed?: DataSeed1;
  formatter_id: FormatterId;
  formatter_sha256: FormatterSha256;
  max_prompt_length: MaxPromptLength;
  mode?: Mode;
  schema_id: SchemaId;
  schema_sha256: SchemaSha256;
  schema_version: SchemaVersion;
  truncation_policy?: TruncationPolicy;
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
 * The on-policy optimization algorithm (S5b). GRPO (group-relative advantage) needs NO critic - it is
 * the cheaper shape that fits the 12 GB envelope like DPO/reward did; PPO (a clipped surrogate with a
 * value head) is the S5c follow-up. The reference model is the frozen base reached via
 * ``disable_adapter`` (the DPO pattern), so no separate reference weights are stored.
 */
export interface PolicyOptimizationSpec {
  algorithm?: Algorithm;
  contract_version?: ContractVersion2;
  use_critic?: UseCritic;
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
 * The hash-pinned reference to what scores each rollout (S5b). A reward model produced by the S5a
 * reward vertical is the primary source, served for inference-only scoring; rule / verifier rewards and
 * an RLAIF judge (Evaluation Studio's judge under the provider policy) are the declared alternatives.
 * ``reward_ref`` hash-pins the reward ADAPTER (its safetensors digest) so a run cannot silently swap the
 * reward function.
 *
 * A ``served_reward_model`` additionally seals (a) the LOADABLE identity the worker reconstructs the
 * scorer from - ``reward_base_model`` (the base the SEQ_CLS reward adapter sits on) + ``reward_adapter_location``
 * (the reward adapter directory) - and (b) ``provenance_manifest_ref``, the reward run's ``RunManifest``
 * whose supervisor-admitted ``reward_success_evidence`` PROVES the source came from a workload_verified
 * reward run. The resolver / runner verify that provenance; this contract only requires it be present +
 * pinned. Non-served kinds leave the served-model fields unset.
 */
export interface RewardSourceRef {
  contract_version?: ContractVersion3;
  higher_is_better?: HigherIsBetter;
  kind: Kind1;
  provenance_manifest_ref?: Ref | null;
  reward_adapter_location?: RewardAdapterLocation;
  reward_base_model?: RewardBaseModel;
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
  contract_version?: ContractVersion4;
  decode_policy?: DecodePolicy;
  max_new_tokens: MaxNewTokens;
  rollouts_per_prompt: RolloutsPerPrompt;
  sampling_temperature: SamplingTemperature;
  sampling_top_p: SamplingTopP;
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
 * The sealed on-policy stability controls (S5b) - the guardrails that keep an on-policy update from
 * reward-hacking or collapsing. The KL-to-reference penalty anchors the policy to the frozen base; the
 * entropy bonus preserves exploration; advantage normalization stabilizes the group-relative signal; the
 * clip range bounds the per-step policy change. Sealed like any execution field - no silent defaults.
 */
export interface StabilityController {
  advantage_normalization?: AdvantageNormalization;
  clip_range: ClipRange;
  contract_version?: ContractVersion5;
  entropy_bonus?: EntropyBonus;
  kl_coefficient: KlCoefficient;
  kl_target?: KlTarget;
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
