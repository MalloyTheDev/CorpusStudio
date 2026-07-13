/* GENERATED from docs/contracts/ModelDescriptor.schema.json — do not edit. Run: npm run gen:contracts */

export type Architectures = string[];
export type ArtifactRole =
  "base" | "adapter" | "merged" | "checkpoint" | "quantized" | "converted" | "other" | "unknown";
export type ModelAttentionType =
  "full" | "sliding_window" | "block_sparse" | "linear" | "state_space" | "hybrid" | "custom" | "unknown";
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type Reasons = string[];
export type Status = "compatible" | "incompatible" | "unverified";
export type BackendCompatibility = BackendCompatibilityEntry[];
export type CapturedAt = string | null;
export type EvidenceKind = "measured" | "estimated" | "declared" | "unknown";
export type Source = string;
export type Value1 = number;
export type ContractVersion = "1.0.0";
export type ModelFormat =
  "safetensors" | "pytorch_pickle" | "gguf" | "onnx" | "torchscript" | "numpy" | "other" | "unknown";
export type HashStatus = "verified" | "not_requested" | "unreadable" | "skipped_unsafe";
export type IsLink = boolean;
export type Path = string;
export type DescriptorFileRole =
  | "config"
  | "weights"
  | "weight_index"
  | "tokenizer"
  | "tokenizer_config"
  | "special_tokens"
  | "generation_config"
  | "model_card"
  | "license"
  | "custom_code"
  | "other";
export type SerializationRisk = "safe" | "pickle" | "executable_code" | "archive" | "unknown";
export type Sha256 = string | null;
export type SizeBytes = number;
export type Files = DescriptorFile[];
/**
 * @minItems 1
 */
export type Formats = [ModelFormat, ...ModelFormat[]];
export type InventoryComplete = boolean;
export type InventorySha256 = string | null;
export type Name = string | null;
export type Redistributable = boolean | null;
export type Source1 = "declared" | "model_card" | "dataset_card" | "user_asserted" | "unknown";
export type SpdxId = string | null;
export type Url = string | null;
export type ModelFamily = string | null;
export type ModelId = string;
export type Notes = string[];
export type ComponentId = string;
export type FileRefs = string[];
export type QuantizationMode = "none" | "int8" | "int4" | "nf4" | "fp4" | "gptq" | "awq" | "hqq";
export type Scope = "all" | "embedding" | "shared" | "router" | "expert_group" | "output_head" | "adapter" | "other";
export type StorageDtype = string | null;
export type Components = ParameterComponent[];
export type CountHandling =
  "included" | "excluded" | "deduplicated" | "represented_separately" | "not_applicable" | "unknown";
export type CountHandling1 =
  "included" | "excluded" | "deduplicated" | "represented_separately" | "not_applicable" | "unknown";
export type CountHandling2 =
  "included" | "excluded" | "deduplicated" | "represented_separately" | "not_applicable" | "unknown";
export type CountHandling3 =
  "included" | "excluded" | "deduplicated" | "represented_separately" | "not_applicable" | "unknown";
export type CountHandling4 =
  "included" | "excluded" | "deduplicated" | "represented_separately" | "not_applicable" | "unknown";
export type CountHandling5 =
  "included" | "excluded" | "deduplicated" | "represented_separately" | "not_applicable" | "unknown";
export type CountHandling6 =
  "included" | "excluded" | "deduplicated" | "represented_separately" | "not_applicable" | "unknown";
/**
 * Distinct parameter quantities required for dense-safe and MoE-safe accounting.
 */
export type ParameterCountKind =
  | "logical"
  | "active_token"
  | "active_sequence"
  | "touched_window"
  | "resident"
  | "updated_window"
  | "exposed_window"
  | "effective";
export type MeasurementWindow = string;
export type Notes1 = string;
export type Scope1 = string;
export type Source2 = string;
export type Unit = "coordinates" | "elements" | "parameters";
export type Value2 = number;
export type Counts = ParameterCount[];
export type ModelExecutionKind = "dense" | "sparse" | "mixture_of_experts" | "conditional" | "hybrid" | "unknown";
export type PositionalEncoding = "rope" | "alibi" | "absolute" | "relative" | "none" | "custom" | "unknown";
export type EvidenceSource = string;
/**
 * Where a model/tokenizer identity originated. A local snapshot may still carry a repository
 * and revision, but network retrieval is never implied by this value.
 */
export type ModelSourceKind = "local" | "huggingface" | "ollama" | "artifact" | "generated" | "external" | "unknown";
export type LocalPath = string | null;
export type Repository = string | null;
export type RequestedRevision = string | null;
export type ResolvedCommit = string | null;
export type ResolvedRevision = string | null;
export type RevisionPinned = boolean;
export type SnapshotSha256 = string | null;
export type StorageSizeBytes = number;
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
export type ModelExecutionKind1 = "dense" | "sparse" | "mixture_of_experts" | "conditional" | "hybrid" | "unknown";
export type ExpertCount = number;
export type ExpertIdentityScheme = string | null;
export type ExpertsPerToken = number | null;
export type GroupId = string;
export type Heterogeneous = boolean;
export type LayerIndices = number[];
export type SharedExpertCount = number | null;
export type ExpertGroups = ExpertGroup[];
export type PhysicalSchedulerOwner = "run_plan";
export type CapacityFactor = number | null;
export type MetadataSource = string;
export type RouterType = string;
export type RoutingNoise = string | null;
export type RoutingUnit = "token" | "sequence" | "layer" | "request" | "custom";
export type SelectionPolicy = string;
export type TopK = number | null;
export type ApprovalRequired = boolean;
export type CustomCodeFiles = string[];
export type CustomCodeRequired = boolean;
export type IsolatedExecutionRequired = boolean;
export type Notes2 = string[];
export type TrustRemoteCode = false;
/**
 * One independent descriptor evidence axis. Integrity, compatibility, functional behavior,
 * and hardware support must never be collapsed into a misleading linear level.
 */
export type VerificationOutcome = "not_checked" | "passed" | "failed" | "partial" | "not_applicable";
export type EvidenceRefs = Ref[];
export type InspectedAt = string | null;
export type Inspector = string | null;
/**
 * One independent descriptor evidence axis. Integrity, compatibility, functional behavior,
 * and hardware support must never be collapsed into a misleading linear level.
 */
export type VerificationOutcome1 = "not_checked" | "passed" | "failed" | "partial" | "not_applicable";
/**
 * One independent descriptor evidence axis. Integrity, compatibility, functional behavior,
 * and hardware support must never be collapsed into a misleading linear level.
 */
export type VerificationOutcome2 = "not_checked" | "passed" | "failed" | "partial" | "not_applicable";
/**
 * One independent descriptor evidence axis. Integrity, compatibility, functional behavior,
 * and hardware support must never be collapsed into a misleading linear level.
 */
export type VerificationOutcome3 = "not_checked" | "passed" | "failed" | "partial" | "not_applicable";
export type Warnings = string[];
export type TiedEmbeddings = boolean | null;

/**
 * Static identity, representation, integrity, and compatibility surface for one model snapshot.
 *
 * The descriptor is safe to build in the torch-free control plane. It does not claim that the model
 * can load, train, fit a device, or execute custom code.
 */
export interface ModelDescriptor {
  architectures?: Architectures;
  artifact_role?: ArtifactRole;
  attention_type?: ModelAttentionType;
  backend_compatibility?: BackendCompatibility;
  captured_at?: CapturedAt;
  context_window?: DimensionEvidence | null;
  contract_version?: ContractVersion;
  files?: Files;
  formats?: Formats;
  inventory_complete?: InventoryComplete;
  inventory_sha256?: InventorySha256;
  license?: License | null;
  model_family?: ModelFamily;
  model_id: ModelId;
  notes?: Notes;
  parameters?: ParameterRepresentation;
  positional_encoding?: PositionalEncoding;
  source: DescriptorSource;
  storage_size_bytes?: StorageSizeBytes;
  task_classes?: TaskClasses;
  tokenizer_ref?: Ref | null;
  topology?: ModelTopology;
  trust?: TrustRequirement;
  verification?: DescriptorVerification;
  vocabulary?: EmbeddingVocabulary;
}
export interface BackendCompatibilityEntry {
  backend_ref: Ref;
  capability_report_ref?: Ref | null;
  environment_ref?: Ref | null;
  reasons?: Reasons;
  status?: Status;
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
export interface DimensionEvidence {
  evidence: EvidenceKind;
  source: Source;
  value: Value1;
}
/**
 * One safe, portable inventory entry. sha256 is separate from hash_status so skipped hashing
 * and unreadable content are never confused with a verified empty digest.
 */
export interface DescriptorFile {
  format?: ModelFormat | null;
  hash_status?: HashStatus;
  is_link?: IsLink;
  path: Path;
  role?: DescriptorFileRole;
  serialization_risk?: SerializationRisk;
  sha256?: Sha256;
  size_bytes: SizeBytes;
}
/**
 * License metadata for a dataset, base model, or produced artifact. The engine reminds users
 * the BASE model's license governs a produced adapter (training/model_card).
 */
export interface License {
  name?: Name;
  redistributable?: Redistributable;
  source?: Source1;
  spdx_id?: SpdxId;
  url?: Url;
}
export interface ParameterRepresentation {
  components?: Components;
  counts?: Counts;
  kind?: ModelExecutionKind;
}
/**
 * A representation component such as shared weights, router, experts, or an adapter.
 *
 * Stored dtype is a raw representation string, not PrecisionMode (which describes run compute and
 * includes values such as tf32/mixed_bf16 that are not on-disk dtypes).
 */
export interface ParameterComponent {
  component_id: ComponentId;
  file_refs?: FileRefs;
  format: ModelFormat;
  quantization?: QuantizationMode | null;
  quantization_details?: QuantizationDetails;
  scope?: Scope;
  storage_dtype?: StorageDtype;
}
export interface QuantizationDetails {
  [k: string]: unknown;
}
/**
 * One explicitly scoped count. There is deliberately no scalar parameter_count field.
 */
export interface ParameterCount {
  evidence: EvidenceKind;
  handling?: ParameterCountHandling;
  kind: ParameterCountKind;
  measurement_window: MeasurementWindow;
  notes?: Notes1;
  scope: Scope1;
  source: Source2;
  unit?: Unit;
  value: Value2;
}
export interface ParameterCountHandling {
  decompressed_caches?: CountHandling;
  generated?: CountHandling1;
  optimizer_shadows?: CountHandling2;
  quantized?: CountHandling3;
  replicated?: CountHandling4;
  shared?: CountHandling5;
  tied?: CountHandling6;
}
/**
 * Identity of the requested source and the immutable revision actually inspected.
 *
 * requested_revision is user intent. resolved_revision/resolved_commit are evidence. They are
 * intentionally separate so a mutable branch name is never misreported as a pinned snapshot.
 */
export interface DescriptorSource {
  artifact_ref?: Ref | null;
  evidence_source?: EvidenceSource;
  kind: ModelSourceKind;
  local_path?: LocalPath;
  repository?: Repository;
  requested_revision?: RequestedRevision;
  resolved_commit?: ResolvedCommit;
  resolved_revision?: ResolvedRevision;
  revision_pinned?: RevisionPinned;
  snapshot_sha256?: SnapshotSha256;
}
export interface ModelTopology {
  execution_kind?: ModelExecutionKind1;
  expert_groups?: ExpertGroups;
  physical_scheduler_owner?: PhysicalSchedulerOwner;
  semantic_routing?: SemanticRouting | null;
}
export interface ExpertGroup {
  expert_count: ExpertCount;
  expert_identity_scheme?: ExpertIdentityScheme;
  expert_registry_ref?: Ref | null;
  experts_per_token?: ExpertsPerToken;
  group_id: GroupId;
  heterogeneous?: Heterogeneous;
  layer_indices?: LayerIndices;
  shared_expert_count?: SharedExpertCount;
}
/**
 * The learned semantic selection policy. Physical placement is not represented here.
 */
export interface SemanticRouting {
  capacity_factor?: CapacityFactor;
  metadata_source: MetadataSource;
  router_type: RouterType;
  routing_noise?: RoutingNoise;
  routing_unit?: RoutingUnit;
  selection_policy: SelectionPolicy;
  top_k?: TopK;
}
/**
 * Static trust findings only. This descriptor can never authorize custom-code execution.
 */
export interface TrustRequirement {
  approval_required?: ApprovalRequired;
  custom_code_files?: CustomCodeFiles;
  custom_code_required?: CustomCodeRequired;
  detected_auto_map?: DetectedAutoMap;
  isolated_execution_required?: IsolatedExecutionRequired;
  notes?: Notes2;
  trust_remote_code?: TrustRemoteCode;
}
export interface DetectedAutoMap {
  [k: string]: unknown;
}
/**
 * Independent evidence axes. Integrity never implies compatibility or hardware support.
 */
export interface DescriptorVerification {
  custom_code_policy?: VerificationOutcome;
  evidence_refs?: EvidenceRefs;
  inspected_at?: InspectedAt;
  inspector?: Inspector;
  integrity?: VerificationOutcome1;
  license?: VerificationOutcome2;
  metadata?: VerificationOutcome3;
  warnings?: Warnings;
}
export interface EmbeddingVocabulary {
  declared_vocab_size?: DimensionEvidence | null;
  input_embedding_rows?: DimensionEvidence | null;
  output_head_rows?: DimensionEvidence | null;
  tied_embeddings?: TiedEmbeddings;
}
