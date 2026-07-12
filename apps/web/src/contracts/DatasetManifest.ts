/* GENERATED from docs/contracts/DatasetManifest.schema.json — do not edit. Run: npm run gen:contracts */

export type ContentFingerprint = string | null;
export type ContractVersion = "1.0.0";
export type CreatedAt = string | null;
export type FingerprintAlgo = string;
export type Label = string;
export type Name = string | null;
export type Redistributable = boolean | null;
export type Source = "declared" | "model_card" | "dataset_card" | "user_asserted" | "unknown";
export type SpdxId = string | null;
export type Url = string | null;
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type PromptVersion = string | null;
export type RandomSeed = number | null;
export type TeacherModel = string;
export type TeacherModelVersion = string | null;
export type RandomSeed1 = number | null;
export type Kind = "dataset_version" | "imported_file" | "hf_dataset" | "generated" | "external";
export type Ref = string;
export type SourceArtifacts = DatasetSourceArtifact[];
export type ManifestRef = string | null;
export type Step = string;
export type Tool = string;
export type ToolVersion = string | null;
export type TransformationPipeline = DatasetTransformStep[];
export type ArtifactIds = string[];
export type EvalReportRef = string | null;
export type GateReportRef = string | null;
export type SourceRunIds = string[];
export type Notes = string;
export type ProjectId = string;
export type RowCount = number;
export type Algo1 = string;
export type Ids = string[] | null;
export type RowManifestRef = string | null;
export type RowsStored = boolean;
export type StoredRowCount = number;
export type RowSignatureKind = string;
export type SchemaId = string;
export type CollatedTokens = number | null;
export type CompletionTokens = number | null;
export type Exact = boolean;
export type ExampleCount = number;
export type ExamplesOverSequenceLen = number | null;
export type MaxTokensInExample = number | null;
export type MeanTokensPerExample = number | null;
export type Method = string;
export type NaturalTokens = number | null;
export type NoTruncation = boolean;
export type PaddingTokens = number | null;
export type PromptTokens = number | null;
export type SequenceLen = number;
export type SupervisedTokens = number | null;
export type TokensPerEpoch = number | null;
export type Trigger = string;
export type UpdatedAt = string | null;
export type VersionId = string;

/**
 * Full identity + LINEAGE of a dataset version. Content identity is verbatim from
 * versions/version_registry.DatasetVersionRecord; row hashes reuse the per-row sha256 of
 * exact_row_signature; token stats extend estimators; the transformation-pipeline lineage is NEW.
 */
export interface DatasetManifest {
  content_fingerprint: ContentFingerprint;
  contract_version?: ContractVersion;
  created_at?: CreatedAt;
  fingerprint_algo?: FingerprintAlgo;
  label?: Label;
  license?: License | null;
  lineage?: DatasetLineage;
  links?: DatasetLinks;
  notes?: Notes;
  output_artifact_hash?: HashRef | null;
  project_id?: ProjectId;
  row_count: RowCount;
  row_hashes?: DatasetRowHashes;
  row_signature_kind?: RowSignatureKind;
  schema_id?: SchemaId;
  token_stats?: TokenStats | null;
  trigger?: Trigger;
  updated_at?: UpdatedAt;
  version_id: VersionId;
}
/**
 * License metadata for a dataset, base model, or produced artifact. The engine reminds users
 * the BASE model's license governs a produced adapter (training/model_card).
 */
export interface License {
  name?: Name;
  redistributable?: Redistributable;
  source?: Source;
  spdx_id?: SpdxId;
  url?: Url;
}
export interface DatasetLineage {
  generation?: DatasetGeneration | null;
  random_seed?: RandomSeed1;
  source_artifacts?: SourceArtifacts;
  tool_versions?: ToolVersions;
  transformation_pipeline?: TransformationPipeline;
}
/**
 * Set when rows were synthesized/distilled from a teacher model — the reproducible recipe.
 */
export interface DatasetGeneration {
  decoding?: Decoding;
  prompt_hash?: HashRef | null;
  prompt_version?: PromptVersion;
  random_seed?: RandomSeed;
  teacher_model: TeacherModel;
  teacher_model_version?: TeacherModelVersion;
}
export interface Decoding {
  [k: string]: unknown;
}
/**
 * An algorithm-tagged digest. The engine emits sha256 today; the algo tag makes a future
 * migration additive (cf. versions/version_registry.FINGERPRINT_ALGO).
 */
export interface HashRef {
  algo?: Algo;
  value?: Value;
}
export interface DatasetSourceArtifact {
  hash?: HashRef | null;
  kind?: Kind;
  license?: License | null;
  ref: Ref;
}
export interface ToolVersions {
  [k: string]: string | null;
}
/**
 * One ordered transform (import → clean → redact → split → collate). Grounded in the engine's
 * cleaning/redaction sidecar manifests (cli.py).
 */
export interface DatasetTransformStep {
  input_hash?: HashRef | null;
  manifest_ref?: ManifestRef;
  output_hash?: HashRef | null;
  params?: Params;
  step: Step;
  tool: Tool;
  tool_version?: ToolVersion;
}
export interface Params {
  [k: string]: unknown;
}
/**
 * Live cross-references resolved by the version card. Presence/integrity resolved live.
 */
export interface DatasetLinks {
  artifact_ids?: ArtifactIds;
  eval_report_ref?: EvalReportRef;
  gate_report_ref?: GateReportRef;
  source_run_ids?: SourceRunIds;
}
/**
 * Per-row content addressing. Reuses capture_dataset's sha256(exact_row_signature) row ids.
 */
export interface DatasetRowHashes {
  algo?: Algo1;
  ids?: Ids;
  row_manifest_ref?: RowManifestRef;
  rows_stored?: RowsStored;
  stored_row_count?: StoredRowCount;
}
/**
 * Token accounting for a dataset/run. Extends estimators.TokenBudgetEstimate with the
 * supervised/prompt/completion/padding breakdown the platform requires. 'natural' = tokens in raw
 * content; 'collated' = after chat-template rendering + packing; 'supervised' = tokens the loss is
 * computed over (completion tokens under a completion-only mask).
 */
export interface TokenStats {
  collated_tokens?: CollatedTokens;
  completion_tokens?: CompletionTokens;
  exact?: Exact;
  example_count: ExampleCount;
  examples_over_sequence_len?: ExamplesOverSequenceLen;
  max_tokens_in_example?: MaxTokensInExample;
  mean_tokens_per_example?: MeanTokensPerExample;
  method?: Method;
  natural_tokens?: NaturalTokens;
  no_truncation?: NoTruncation;
  padding_tokens?: PaddingTokens;
  prompt_tokens?: PromptTokens;
  sequence_len?: SequenceLen;
  supervised_tokens?: SupervisedTokens;
  tokens_per_epoch?: TokensPerEpoch;
}
