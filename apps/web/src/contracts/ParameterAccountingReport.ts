/* GENERATED from docs/contracts/ParameterAccountingReport.schema.json — do not edit. Run: npm run gen:contracts */

export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type ArtifactRefs = Ref[];
export type ConflictId = string;
export type Explanation = string;
/**
 * @minItems 2
 */
export type ObservationIds = [string, string, ...string[]];
export type ReasonCode = string;
export type Conflicts = ParameterConflict[];
export type ContractVersion = "1.0.0";
export type EvaluationRefs = Ref[];
export type Explanation1 = string;
export type GapId = string;
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
export type ParameterGapReason =
  | "missing_observation"
  | "unknown_handling"
  | "unpinned_model"
  | "unstructured_claim"
  | "stored_elements_not_logical"
  | "incomplete_inventory"
  | "unsupported_format"
  | "malformed_evidence"
  | "changed_during_read"
  | "runtime_instrumentation_required"
  | "estimated_only"
  | "measured_evidence_required"
  | "conflicting_evidence"
  | "incomparable_evidence";
export type Resolution = string;
export type ComponentIds = string[];
export type CoordinateUniverseId = string;
export type CoordinateUniverseSha256 = string | null;
export type Definition = string;
export type DeviceId = string | null;
export type ExpertIds = string[];
export type ParameterScopeKind =
  | "model"
  | "component_set"
  | "shared"
  | "router"
  | "expert_group"
  | "expert_set"
  | "adapter"
  | "embedding"
  | "output_head"
  | "device_residency"
  | "custom";
/**
 * A physical state tier. A RunPlan names the intended tier; only runtime evidence may claim
 * actual residency there.
 */
export type MemoryTier = "gpu" | "pinned_ram" | "pageable_ram" | "nvme" | "sata" | "remote" | "unknown";
export type ScopeId = string;
export type CapturedAt = string | null;
export type Definition1 = string;
export type EventSeqEnd = number | null;
export type EventSeqStart = number | null;
export type ParameterWindowKind =
  "static_snapshot" | "token" | "sequence" | "instant" | "microbatch" | "optimizer_window" | "run";
export type MicrostepEnd = number | null;
export type MicrostepStart = number | null;
export type OptimizerStepEnd = number | null;
export type OptimizerStepStart = number | null;
export type SequenceId = string | null;
export type TokenIndex = number | null;
export type WindowId = string;
export type Gaps = ParameterEvidenceGap[];
export type GeneratedAt = string;
export type Notes = string[];
export type Assumptions = string[];
export type ParameterObservationCoverage = "complete" | "sampled" | "partial";
export type Definition2 = string;
export type Evidence = "measured" | "estimated" | "declared";
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
export type ParameterIdentityBasis =
  | "independent_coordinates"
  | "stored_tensor_elements"
  | "optimizer_addressable_coordinates"
  | "runtime_identity_set"
  | "topology_formula"
  | "declared_definition"
  | "unknown";
export type Notes1 = string;
export type ObservationId = string;
export type CapturedAt1 = string | null;
export type ParameterEvidenceSourceKind =
  | "model_config"
  | "model_descriptor"
  | "safetensors_header"
  | "planner"
  | "backend_worker"
  | "checkpoint_inventory"
  | "evaluation_runtime"
  | "user_supplied";
export type Method = string;
export type Producer = string;
export type ProducerVersion = string;
export type Unit = "coordinates" | "elements" | "parameters";
export type Value1 = number;
export type ParameterValueRelation = "exact" | "estimate" | "lower_bound" | "upper_bound";
export type Observations = ParameterObservation[];
export type ParentReportRefs = Ref[];
/**
 * The evidence set a report is trying to complete.
 */
export type ParameterAccountingProfile =
  "model_static" | "training_plan" | "training_runtime" | "inference_runtime" | "checkpoint" | "evaluation";
export type ReportHash = string;
export type ReportId = string;
export type ParameterAccountingStatus = "complete" | "incomplete" | "conflicting";

/**
 * Hash-sealed, auditable reconciliation of parameter evidence for one exact model context.
 */
export interface ParameterAccountingReport {
  artifact_refs?: ArtifactRefs;
  conflicts?: Conflicts;
  contract_version?: ContractVersion;
  evaluation_refs?: EvaluationRefs;
  gaps?: Gaps;
  generated_at: GeneratedAt;
  model_ref: Ref;
  notes?: Notes;
  observations?: Observations;
  parent_report_refs?: ParentReportRefs;
  plan_ref?: Ref | null;
  profile: ParameterAccountingProfile;
  report_hash: ReportHash;
  report_id: ReportId;
  run_ref?: Ref | null;
  status: ParameterAccountingStatus;
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
export interface ParameterConflict {
  conflict_id: ConflictId;
  explanation: Explanation;
  observation_ids: ObservationIds;
  reason_code: ReasonCode;
}
export interface ParameterEvidenceGap {
  explanation: Explanation1;
  gap_id: GapId;
  kind: ParameterCountKind;
  reason: ParameterGapReason;
  resolution: Resolution;
  scope: ParameterScope;
  window: ParameterWindow;
}
/**
 * Stable coordinate universe for an authoritative parameter observation.
 *
 * Runtime addresses are never identities. Sparse scopes carry stable expert IDs, and every scope
 * is tied to one exact model reference plus a named coordinate universe.
 */
export interface ParameterScope {
  component_ids?: ComponentIds;
  coordinate_universe_id: CoordinateUniverseId;
  coordinate_universe_sha256?: CoordinateUniverseSha256;
  definition: Definition;
  device_id?: DeviceId;
  expert_ids?: ExpertIds;
  kind: ParameterScopeKind;
  memory_tier?: MemoryTier | null;
  model_ref: Ref;
  scope_id: ScopeId;
}
/**
 * The exact computation or scheduling window a count describes.
 */
export interface ParameterWindow {
  captured_at?: CapturedAt;
  definition: Definition1;
  event_seq_end?: EventSeqEnd;
  event_seq_start?: EventSeqStart;
  kind: ParameterWindowKind;
  microstep_end?: MicrostepEnd;
  microstep_start?: MicrostepStart;
  optimizer_step_end?: OptimizerStepEnd;
  optimizer_step_start?: OptimizerStepStart;
  plan_ref?: Ref | null;
  run_ref?: Ref | null;
  sequence_id?: SequenceId;
  token_index?: TokenIndex;
  window_id: WindowId;
}
/**
 * One evidence-bearing parameter count. Unknown evidence is represented as a gap, never zero.
 */
export interface ParameterObservation {
  assumptions?: Assumptions;
  coverage: ParameterObservationCoverage;
  definition: Definition2;
  evidence: Evidence;
  handling: ParameterCountHandling;
  identity_basis: ParameterIdentityBasis;
  kind: ParameterCountKind;
  notes?: Notes1;
  observation_id: ObservationId;
  scope: ParameterScope;
  source: ParameterEvidenceSource;
  unit?: Unit;
  value: Value1;
  value_relation: ParameterValueRelation;
  window: ParameterWindow;
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
export interface ParameterEvidenceSource {
  backend_ref?: Ref | null;
  captured_at?: CapturedAt1;
  environment_ref?: Ref | null;
  kind: ParameterEvidenceSourceKind;
  method: Method;
  producer: Producer;
  producer_version: ProducerVersion;
  source_ref: Ref;
}
