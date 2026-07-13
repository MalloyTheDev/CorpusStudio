/* GENERATED from docs/contracts/TrainingObjective.schema.json — do not edit. Run: npm run gen:contracts */

export type AdapterMethod =
  "none" | "lora" | "qlora" | "dora" | "ia3" | "full_finetune" | "prompt_tuning" | "prefix_tuning";
export type AdaptationMethods = AdapterMethod[];
export type AdaptationMethods1 = AdapterMethod[];
export type FunctionalProbeRequired = boolean;
export type HardwareVerificationRequired = boolean;
export type LossImpl = "cross_entropy" | "liger_fused_ce" | "chunked_ce" | "dpo" | "orpo" | "kto" | "ipo" | "reward_bt";
export type LossImpls = LossImpl[];
export type ObjectiveCapabilities = string[];
export type QuantizationMode = "none" | "int8" | "int4" | "nf4" | "fp4" | "gptq" | "awq" | "hqq";
export type QuantizationModes = QuantizationMode[];
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
export type ContractVersion = "1.0.0";
export type Notes = string[];
export type Role = "train" | "validation" | "teacher" | "preference" | "evaluation";
export type RowValidationRequired = boolean;
export type SplitIsolationRequired = boolean;
/**
 * @minItems 1
 */
export type Variants = [ObjectiveDatasetVariant, ...ObjectiveDatasetVariant[]];
export type ObjectiveDatasetAvailability = "builtin" | "structural" | "planned";
export type DatasetFormat = string | null;
/**
 * @minItems 1
 */
export type RequiredFields = [ObjectiveDatasetField, ...ObjectiveDatasetField[]];
export type FieldType = string | null;
export type Name = string;
export type SemanticRole = string;
export type SchemaId = string;
export type SchemaVersion = string | null;
export type DatasetInputs = ObjectiveDatasetInput[];
export type Description = string;
export type DisplayName = string;
export type AfterRun = boolean;
export type BeforeRun = boolean;
export type DuringRun = boolean;
export type ExpertSystemMetrics = string[];
export type GateRequired = boolean;
export type HoldoutRequired = boolean;
export type Metrics = string[];
export type ObjectiveExecutionKind = "training" | "evaluation" | "artifact_operation";
export type ComponentScoped = boolean;
export type Condition = string | null;
export type Description1 = string;
export type ObjectiveArtifactKind =
  | "full_model"
  | "adapter"
  | "checkpoint"
  | "optimizer_state"
  | "scheduler_state"
  | "trainer_state"
  | "reward_model"
  | "embedding_model"
  | "reranker_model"
  | "classifier_model"
  | "multimodal_model"
  | "distillation_student"
  | "verifier_model"
  | "evaluation_result"
  | "merged_model"
  | "converted_model"
  | "quantized_model"
  | "routing_state"
  | "expert_shards"
  | "provenance_manifest";
export type Required = boolean;
export type ExpectedArtifacts = ObjectiveArtifactExpectation[];
export type Communication = "none" | "low" | "medium" | "high" | "unknown";
export type Compute = "none" | "low" | "medium" | "high" | "unknown";
export type DeviceMemory = "none" | "low" | "medium" | "high" | "unknown";
export type FitClaim = "none";
export type HostMemory = "none" | "low" | "medium" | "high" | "unknown";
export type Implications = string[];
export type StorageIo = "none" | "low" | "medium" | "high" | "unknown";
/**
 * The semantic family of a training objective.
 *
 * Concrete registry entries remain more specific than this coarse family. For example, LoRA and
 * QLoRA are distinct entries with different adaptation and hardware requirements while both are
 * supervised-fine-tuning objectives.
 */
export type ObjectiveKind =
  | "pretraining"
  | "supervised_fine_tuning"
  | "preference_optimization"
  | "reward_modeling"
  | "distillation"
  | "process_supervision"
  | "verifier_training"
  | "tool_use"
  | "embedding"
  | "reranking"
  | "classification"
  | "multimodal"
  | "evaluation"
  | "merge"
  | "conversion"
  | "quantization";
export type Construction = string;
export type IgnoreIndex = number | null;
export type ObjectiveLabelKind =
  | "next_token"
  | "completion_tokens"
  | "response_tokens"
  | "preference_pair"
  | "scalar_reward"
  | "class_id"
  | "contrastive_pair"
  | "teacher_distribution"
  | "teacher_sequence"
  | "trace_step"
  | "verifier_target"
  | "multimodal_target"
  | "none";
export type LabelId = string;
export type Notes1 = string[];
export type SourceFields = string[];
export type Labels = ObjectiveLabelConstruction[];
export type Limitations = string[];
export type ComponentId = string;
export type Construction1 = string;
export type DefaultWeight = number | null;
export type ObjectiveLossComponentKind =
  | "cross_entropy"
  | "preference"
  | "odds_ratio"
  | "reward_pairwise"
  | "classification"
  | "contrastive"
  | "ranking"
  | "knowledge_distillation"
  | "sequence_distillation"
  | "logit_distillation"
  | "rationale_distillation"
  | "process_supervision"
  | "verifier"
  | "router_auxiliary"
  | "load_balancing"
  | "router_z_loss"
  | "entropy"
  | "overflow"
  | "specialization"
  | "none";
export type LabelRef = string | null;
export type MaskRef = string | null;
export type Reduction = "mean" | "sum" | "token_mean" | "pair_mean" | "none";
export type LossComponents = ObjectiveLossComponent[];
export type Construction2 = string;
export type EmptyMaskAction = "reject" | "skip" | "zero_loss" | "not_applicable";
export type IncludePadding = boolean;
export type IncludeSpecialTokens = boolean;
export type ObjectiveLossMaskKind =
  | "all_non_padding"
  | "completion_only"
  | "response_only"
  | "chosen_rejected"
  | "labeled_positions"
  | "trace_steps"
  | "multimodal_target"
  | "none"
  | "custom";
export type MaskId = string;
export type SourceFields1 = string[];
export type LossMasks = ObjectiveLossMask[];
export type CustomCodePolicy = "forbid" | "isolated_approval" | "backend_defined";
/**
 * @minItems 1
 */
export type ExecutionKinds = [ModelExecutionKind, ...ModelExecutionKind[]];
export type ModelExecutionKind = "dense" | "sparse" | "mixture_of_experts" | "conditional" | "hybrid" | "unknown";
export type RequiresMultimodalProjector = boolean;
export type RequiresOutputHead = boolean;
export type RequiresReferenceModel = boolean;
export type RequiresRewardHead = boolean;
export type RequiresTokenizer = boolean;
/**
 * @minItems 1
 */
export type TaskClasses = [ModelTaskClass, ...ModelTaskClass[]];
export type ModelTaskClass =
  | "causal_lm"
  | "masked_lm"
  | "seq2seq_lm"
  | "classification"
  | "embedding"
  | "reranker"
  | "reward_model"
  | "vision"
  | "speech"
  | "multimodal"
  | "custom"
  | "unknown";
export type ObjectiveHash = string;
export type ObjectiveId = string;
export type ObjectiveVersion = string;
export type ComponentScopedResume = boolean;
export type ObjectiveResumeMode = "exact" | "fork_only" | "restart_only" | "not_applicable";
export type NonExactResumeCreatesLineage = boolean;
export type Notes2 = string[];
export type RequiredState = string[];
export type ObjectiveExposureTracking =
  "none" | "per_component" | "per_expert" | "router_and_expert" | "backend_defined";
export type Notes3 = string[];
export type ObjectiveOptimizerClock = "none" | "global" | "per_component" | "per_expert" | "backend_defined";
export type RoutingCollapseGateRequiredWhenRouted = boolean;
/**
 * @minItems 1
 */
export type Scopes = [ObjectiveUpdateScope, ...ObjectiveUpdateScope[]];
export type ObjectiveUpdateScope =
  | "all_parameters"
  | "shared_parameters"
  | "embeddings"
  | "output_head"
  | "adapters"
  | "router"
  | "selected_experts"
  | "all_experts"
  | "projector"
  | "task_head"
  | "none";
export type ObjectiveSelectionMode =
  | "all"
  | "adapter_only"
  | "router_only"
  | "selected_experts"
  | "routed_experts"
  | "task_head_only"
  | "none"
  | "backend_defined";
export type StableExpertIdentity = "not_required" | "when_expert_scoped" | "required";
export type StarvationGateRequiredWhenExpertScoped = boolean;
export type UpdateWindowDefinition = string;
export type ObjectiveVerificationStatus =
  "not_verified" | "declared" | "contract_validated" | "functional_verified" | "hardware_verified" | "not_applicable";
export type EvidenceRefs = string[];
export type ObjectiveVerificationStatus1 =
  "not_verified" | "declared" | "contract_validated" | "functional_verified" | "hardware_verified" | "not_applicable";
export type ObjectiveVerificationStatus2 =
  "not_verified" | "declared" | "contract_validated" | "functional_verified" | "hardware_verified" | "not_applicable";

/**
 * A versioned semantic objective, deliberately independent from trainer implementation.
 *
 * The objective hash seals the canonical definition. Registry helpers verify it; deserializing an
 * arbitrary contract alone does not imply that its hash or execution claims are trusted.
 */
export interface TrainingObjective {
  adaptation_methods?: AdaptationMethods;
  backend_requirement: ObjectiveBackendRequirement;
  coarse_task_type?: TaskType | null;
  contract_version?: ContractVersion;
  dataset_inputs?: DatasetInputs;
  description: Description;
  display_name: DisplayName;
  evaluation: ObjectiveEvaluationRequirements;
  execution_kind: ObjectiveExecutionKind;
  expected_artifacts?: ExpectedArtifacts;
  hardware: ObjectiveHardwareImplications;
  kind: ObjectiveKind;
  labels?: Labels;
  limitations?: Limitations;
  loss_components?: LossComponents;
  loss_masks?: LossMasks;
  model_requirement: ObjectiveModelRequirement;
  objective_hash: ObjectiveHash;
  objective_id: ObjectiveId;
  objective_version: ObjectiveVersion;
  resume: ObjectiveResumeSemantics;
  update_policy: ObjectiveUpdatePolicy;
  verification?: ObjectiveVerification;
}
/**
 * Semantic backend requirements. Backend IDs never belong in an objective definition.
 *
 * A backend may match any listed loss, adaptation, and quantization mode; every listed objective
 * capability token is required. Static matches remain declarations until a capability report proves
 * the same objective tokens on the selected environment.
 */
export interface ObjectiveBackendRequirement {
  adaptation_methods?: AdaptationMethods1;
  functional_probe_required?: FunctionalProbeRequired;
  hardware_verification_required?: HardwareVerificationRequired;
  loss_impls?: LossImpls;
  objective_capabilities?: ObjectiveCapabilities;
  quantization_modes?: QuantizationModes;
  task_type?: TaskType | null;
}
export interface ObjectiveDatasetInput {
  notes?: Notes;
  role: Role;
  row_validation_required?: RowValidationRequired;
  split_isolation_required?: SplitIsolationRequired;
  variants: Variants;
}
/**
 * One accepted dataset shape. ``planned`` means the shape is specified but CorpusStudio does
 * not yet ship a matching built-in schema; registry presence never turns that into support.
 */
export interface ObjectiveDatasetVariant {
  availability: ObjectiveDatasetAvailability;
  dataset_format?: DatasetFormat;
  required_fields: RequiredFields;
  schema_id: SchemaId;
  schema_version?: SchemaVersion;
}
export interface ObjectiveDatasetField {
  field_type?: FieldType;
  name: Name;
  semantic_role: SemanticRole;
}
export interface ObjectiveEvaluationRequirements {
  after_run?: AfterRun;
  before_run?: BeforeRun;
  during_run?: DuringRun;
  expert_system_metrics?: ExpertSystemMetrics;
  gate_required?: GateRequired;
  holdout_required?: HoldoutRequired;
  metrics?: Metrics;
}
export interface ObjectiveArtifactExpectation {
  component_scoped?: ComponentScoped;
  condition?: Condition;
  description: Description1;
  kind: ObjectiveArtifactKind;
  required?: Required;
}
export interface ObjectiveHardwareImplications {
  communication?: Communication;
  compute?: Compute;
  device_memory?: DeviceMemory;
  fit_claim?: FitClaim;
  host_memory?: HostMemory;
  implications?: Implications;
  storage_io?: StorageIo;
}
export interface ObjectiveLabelConstruction {
  construction: Construction;
  ignore_index?: IgnoreIndex;
  kind: ObjectiveLabelKind;
  label_id: LabelId;
  notes?: Notes1;
  source_fields?: SourceFields;
}
export interface ObjectiveLossComponent {
  component_id: ComponentId;
  construction: Construction1;
  default_weight?: DefaultWeight;
  kind: ObjectiveLossComponentKind;
  label_ref?: LabelRef;
  mask_ref?: MaskRef;
  reduction?: Reduction;
}
export interface ObjectiveLossMask {
  construction: Construction2;
  empty_mask_action?: EmptyMaskAction;
  include_padding?: IncludePadding;
  include_special_tokens?: IncludeSpecialTokens;
  kind: ObjectiveLossMaskKind;
  mask_id: MaskId;
  source_fields?: SourceFields1;
}
export interface ObjectiveModelRequirement {
  custom_code_policy?: CustomCodePolicy;
  execution_kinds: ExecutionKinds;
  requires_multimodal_projector?: RequiresMultimodalProjector;
  requires_output_head?: RequiresOutputHead;
  requires_reference_model?: RequiresReferenceModel;
  requires_reward_head?: RequiresRewardHead;
  requires_tokenizer?: RequiresTokenizer;
  task_classes: TaskClasses;
}
export interface ObjectiveResumeSemantics {
  component_scoped_resume?: ComponentScopedResume;
  mode: ObjectiveResumeMode;
  non_exact_resume_creates_lineage?: NonExactResumeCreatesLineage;
  notes?: Notes2;
  required_state?: RequiredState;
}
/**
 * What may change, separate from where those components are physically resident.
 *
 * This is the MoE-safe semantic update policy. Placement, prefetch, and device scheduling remain
 * separate RunPlan physical-execution responsibilities.
 */
export interface ObjectiveUpdatePolicy {
  exposure_tracking: ObjectiveExposureTracking;
  notes?: Notes3;
  optimizer_clock: ObjectiveOptimizerClock;
  routing_collapse_gate_required_when_routed?: RoutingCollapseGateRequiredWhenRouted;
  scopes: Scopes;
  selection_mode: ObjectiveSelectionMode;
  stable_expert_identity: StableExpertIdentity;
  starvation_gate_required_when_expert_scoped?: StarvationGateRequiredWhenExpertScoped;
  update_window_definition: UpdateWindowDefinition;
}
export interface ObjectiveVerification {
  definition?: ObjectiveVerificationStatus;
  evidence_refs?: EvidenceRefs;
  hardware?: ObjectiveVerificationStatus1;
  implementation?: ObjectiveVerificationStatus2;
}
