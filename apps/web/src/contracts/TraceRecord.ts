/* GENERATED from docs/contracts/TraceRecord.schema.json — do not edit. Run: npm run gen:contracts */

/**
 * @minItems 1
 */
export type Context = [TraceMessage, ...TraceMessage[]];
export type Content = string;
export type MessageId = string;
export type Name = string | null;
export type Role = "system" | "user" | "assistant" | "tool";
export type ToolCallId = string | null;
export type ContractVersion = "1.0.0";
export type CreatedAt = string;
export type Notes = string[];
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type ParentTraceRefs = Ref[];
export type Backend = string | null;
export type Kind = "human" | "model" | "imported" | "observed";
export type ModelId = string | null;
export type Action = "generate-trace";
export type Allowed = boolean;
export type CapturedAt = string;
export type HumanReviewRequired = boolean;
export type PolicySha256 = string;
export type PolicySource = string;
export type PolicySnapshot = {
  [k: string]: unknown;
} | null;
export type PromptTemplateSha256 = string | null;
export type PromptTemplateVersion = string | null;
export type ProviderId = string | null;
export type ProviderKind = string | null;
export type RequestSha256 = string | null;
export type RequestedModelId = string | null;
export type ResponseSha256 = string | null;
export type RouteId = string | null;
export type Seed = number | null;
export type Tool = string;
export type ToolVersion = string | null;
export type Notes1 = string[];
export type ReviewedAt = string | null;
export type Reviewer = string | null;
export type Status = "pending" | "approved" | "rejected";
/**
 * @minItems 1
 */
export type Segments = [TraceSegment, ...TraceSegment[]];
export type Actor = "system" | "user" | "assistant" | "tool" | "verifier" | "human";
export type Content1 = string | null;
export type EvidenceRefs = Ref[];
export type Kind1 = "reasoning" | "action" | "tool_call" | "tool_result" | "observation" | "verifier" | "final_answer";
export type Origin = "model" | "tool" | "human" | "imported" | "derived";
export type SegmentId = string;
export type Sequence = number;
export type CallId = string;
export type ToolName = string;
export type ToolVersion1 = string | null;
export type CallId1 = string;
export type Content2 = string | null;
export type ContentSha256 = string;
export type Error = string | null;
export type Status1 = "success" | "error" | "denied";
export type Truncated = boolean;
export type Label = string | number | boolean | null;
export type Reward = number | null;
export type Target = boolean;
export type Weight = number;
export type Verification = "unverified" | "human_verified" | "tool_verified" | "verifier_accepted" | "rejected";
export type ArtifactRef = string | null;
export type ArtifactSha256 = string | null;
export type RowIdAlgo = "sha256-exact-row-signature-v1";
export type SourceRowId = string;
export type SourceRowIndex = number | null;
export type Tags = string[];
export type TraceHash = string;
export type TraceId = string;
export type TraceKind = "reasoning" | "tool_use" | "agent" | "process_supervision" | "verifier" | "mixed";
export type CheckedAt = string;
export type ConfigSha256 = string;
export type Code = string;
export type Location = string;
export type Message = string;
export type Severity = "warning" | "block";
export type Findings = TraceValidationFinding[];
export type Status2 = "pass" | "warn" | "block";
export type Validator = string;
export type ValidatorVersion = string;

/**
 * A hash-sealed reasoning/tool/process record whose review gate is separate from validation.
 *
 * Heuristic validation and human approval do not promote generated reasoning to ground truth.
 * Model/imported reasoning segments must remain explicitly ``unverified``.
 */
export interface TraceRecord {
  context: Context;
  contract_version?: ContractVersion;
  created_at: CreatedAt;
  notes?: Notes;
  parent_trace_refs?: ParentTraceRefs;
  producer: TraceProducer;
  review?: TraceReview;
  segments: Segments;
  source: TraceSource;
  tags?: Tags;
  trace_hash: TraceHash;
  trace_id: TraceId;
  trace_kind?: TraceKind;
  validation: TraceValidationEvidence;
}
export interface TraceMessage {
  content: Content;
  message_id: MessageId;
  name?: Name;
  role: Role;
  tool_call_id?: ToolCallId;
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
 * Who produced the trace and the reproducibility/policy evidence available at capture time.
 */
export interface TraceProducer {
  backend?: Backend;
  decoding?: Decoding;
  kind: Kind;
  model_id?: ModelId;
  model_ref?: Ref | null;
  policy_decision?: TracePolicyDecision | null;
  policy_snapshot?: PolicySnapshot;
  prompt_template_sha256?: PromptTemplateSha256;
  prompt_template_version?: PromptTemplateVersion;
  provider_id?: ProviderId;
  provider_kind?: ProviderKind;
  request_sha256?: RequestSha256;
  requested_model_id?: RequestedModelId;
  response_metadata?: ResponseMetadata;
  response_sha256?: ResponseSha256;
  route_id?: RouteId;
  seed?: Seed;
  tool: Tool;
  tool_version?: ToolVersion;
}
export interface Decoding {
  [k: string]: unknown;
}
export interface TracePolicyDecision {
  action?: Action;
  allowed: Allowed;
  captured_at: CapturedAt;
  human_review_required?: HumanReviewRequired;
  policy_sha256: PolicySha256;
  policy_source: PolicySource;
}
export interface ResponseMetadata {
  [k: string]: unknown;
}
export interface TraceReview {
  notes?: Notes1;
  reviewed_at?: ReviewedAt;
  reviewer?: Reviewer;
  status?: Status;
}
export interface TraceSegment {
  actor: Actor;
  content?: Content1;
  content_ref?: Ref | null;
  evidence_refs?: EvidenceRefs;
  kind: Kind1;
  origin: Origin;
  segment_id: SegmentId;
  sequence: Sequence;
  tool_call?: TraceToolCall | null;
  tool_result?: TraceToolResult | null;
  training_signal?: TraceTrainingSignal | null;
  verification?: Verification;
}
export interface TraceToolCall {
  argument_schema_ref?: Ref | null;
  arguments?: Arguments;
  call_id: CallId;
  tool_name: ToolName;
  tool_version?: ToolVersion1;
}
export interface Arguments {
  [k: string]: unknown;
}
export interface TraceToolResult {
  call_id: CallId1;
  content?: Content2;
  content_ref?: Ref | null;
  content_sha256: ContentSha256;
  error?: Error;
  status: Status1;
  truncated?: Truncated;
}
/**
 * Optional per-segment supervision without implying the content is factual ground truth.
 */
export interface TraceTrainingSignal {
  label?: Label;
  reward?: Reward;
  target?: Target;
  verifier_ref?: Ref | null;
  weight?: Weight;
}
/**
 * Pinned identity of the exact source row used to create a trace.
 *
 * A source is either a hash-pinned DatasetManifest reference or a hash-pinned imported artifact.
 * ``source_row_id`` reuses the engine's sha256(exact_row_signature) row identity so trace lineage
 * and dataset-version lineage agree rather than inventing a second row fingerprint.
 */
export interface TraceSource {
  artifact_ref?: ArtifactRef;
  artifact_sha256?: ArtifactSha256;
  dataset_ref?: Ref | null;
  row_id_algo?: RowIdAlgo;
  source_row_id: SourceRowId;
  source_row_index?: SourceRowIndex;
}
export interface TraceValidationEvidence {
  checked_at: CheckedAt;
  config_sha256: ConfigSha256;
  findings?: Findings;
  status: Status2;
  validator: Validator;
  validator_version: ValidatorVersion;
}
export interface TraceValidationFinding {
  code: Code;
  location: Location;
  message: Message;
  severity: Severity;
}
