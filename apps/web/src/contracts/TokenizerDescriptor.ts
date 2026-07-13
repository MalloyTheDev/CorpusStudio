/* GENERATED from docs/contracts/TokenizerDescriptor.schema.json — do not edit. Run: npm run gen:contracts */

export type AddedTokenCount = number | null;
export type BaseVocabularySize = number | null;
export type CapturedAt = string | null;
export type ChatTemplate =
  | string
  | {
      [k: string]: unknown;
    }[]
  | null;
export type ChatTemplateSha256 = string | null;
export type ContractVersion = "1.0.0";
export type EffectiveVocabularySize = number | null;
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
export type TokenizerFormat = "tokenizers_json" | "sentencepiece" | "tiktoken" | "custom" | "unknown";
export type ImplementationClass = string | null;
export type InventoryComplete = boolean;
export type InventorySha256 = string | null;
export type MaxTokenId = number | null;
export type Check = string;
export type Evidence = string | null;
export type Message = string;
/**
 * One independent descriptor evidence axis. Integrity, compatibility, functional behavior,
 * and hardware support must never be collapsed into a misleading linear level.
 */
export type VerificationOutcome = "not_checked" | "passed" | "failed" | "partial" | "not_applicable";
export type Remediation = string | null;
export type Checks = CompatibilityCheck[];
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type RequiredEmbeddingRows = number | null;
export type ResizeInputEmbeddings = boolean;
export type ResizeOutputHead = boolean;
export type CompatibilityStatus = "compatible" | "resize_required" | "incompatible" | "unverified";
export type Warnings = string[];
export type ModelCompatibility = ModelTokenizerCompatibility[];
export type EvidenceKind = "measured" | "estimated" | "declared" | "unknown";
export type Source = string;
export type Value1 = number;
export type Normalization = {
  [k: string]: unknown;
} | null;
export type Notes = string[];
export type PreTokenization = {
  [k: string]: unknown;
} | null;
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
export type Added = boolean;
export type Content = string;
export type Role = string;
export type TokenId = number | null;
export type SpecialTokens = SpecialToken[];
export type StorageSizeBytes = number;
export type TokenizerId = string;
export type ApprovalRequired = boolean;
export type CustomCodeFiles = string[];
export type CustomCodeRequired = boolean;
export type IsolatedExecutionRequired = boolean;
export type Notes1 = string[];
export type TrustRemoteCode = false;
/**
 * One independent descriptor evidence axis. Integrity, compatibility, functional behavior,
 * and hardware support must never be collapsed into a misleading linear level.
 */
export type VerificationOutcome1 = "not_checked" | "passed" | "failed" | "partial" | "not_applicable";
export type EvidenceRefs = Ref[];
export type InspectedAt = string | null;
export type Inspector = string | null;
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
/**
 * One independent descriptor evidence axis. Integrity, compatibility, functional behavior,
 * and hardware support must never be collapsed into a misleading linear level.
 */
export type VerificationOutcome4 = "not_checked" | "passed" | "failed" | "partial" | "not_applicable";
export type Warnings1 = string[];

/**
 * Static tokenizer identity and structure. Exact encode/decode behavior needs a later functional
 * probe in an isolated capability environment; inspection alone does not claim it.
 */
export interface TokenizerDescriptor {
  added_token_count?: AddedTokenCount;
  base_vocabulary_size?: BaseVocabularySize;
  captured_at?: CapturedAt;
  chat_template?: ChatTemplate;
  chat_template_sha256?: ChatTemplateSha256;
  contract_version?: ContractVersion;
  effective_vocabulary_size?: EffectiveVocabularySize;
  files?: Files;
  format?: TokenizerFormat;
  implementation_class?: ImplementationClass;
  inventory_complete?: InventoryComplete;
  inventory_sha256?: InventorySha256;
  max_token_id?: MaxTokenId;
  model_compatibility?: ModelCompatibility;
  model_max_length?: DimensionEvidence | null;
  normalization?: Normalization;
  notes?: Notes;
  pre_tokenization?: PreTokenization;
  source: DescriptorSource;
  special_tokens?: SpecialTokens;
  storage_size_bytes?: StorageSizeBytes;
  tokenizer_id: TokenizerId;
  trust?: TrustRequirement;
  verification?: DescriptorVerification;
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
export interface ModelTokenizerCompatibility {
  checks?: Checks;
  model_ref: Ref;
  required_embedding_rows?: RequiredEmbeddingRows;
  resize_input_embeddings?: ResizeInputEmbeddings;
  resize_output_head?: ResizeOutputHead;
  status: CompatibilityStatus;
  tokenizer_ref: Ref;
  warnings?: Warnings;
}
export interface CompatibilityCheck {
  check: Check;
  evidence?: Evidence;
  message?: Message;
  outcome: VerificationOutcome;
  remediation?: Remediation;
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
export interface SpecialToken {
  added?: Added;
  content: Content;
  role: Role;
  token_id?: TokenId;
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
  notes?: Notes1;
  trust_remote_code?: TrustRemoteCode;
}
export interface DetectedAutoMap {
  [k: string]: unknown;
}
/**
 * Independent evidence axes. Integrity never implies compatibility or hardware support.
 */
export interface DescriptorVerification {
  custom_code_policy?: VerificationOutcome1;
  evidence_refs?: EvidenceRefs;
  inspected_at?: InspectedAt;
  inspector?: Inspector;
  integrity?: VerificationOutcome2;
  license?: VerificationOutcome3;
  metadata?: VerificationOutcome4;
  warnings?: Warnings1;
}
