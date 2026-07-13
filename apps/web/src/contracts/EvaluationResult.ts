/* GENERATED from docs/contracts/EvaluationResult.schema.json — do not edit. Run: npm run gen:contracts */

export type AdapterApplied = boolean | null;
export type Backend = string | null;
export type ChatTemplateApplied = boolean | null;
export type PrecisionMode = "fp32" | "tf32" | "fp16" | "bf16" | "fp8" | "mixed_bf16" | "mixed_fp16";
export type QuantizationMode = "none" | "int8" | "int4" | "nf4" | "fp4" | "gptq" | "awq" | "hqq";
export type ContractVersion = "1.0.0";
export type DatasetFingerprint = string | null;
export type Name = string;
export type VersionRef = string | null;
export type EvalId = string;
export type BlockCount = number;
export type MaxRegressionScoreDrop = number | null;
export type MinEvalAverageScore = number | null;
export type MinEvalPassRate = number | null;
export type OverallStatus = "pass" | "warn" | "block";
export type PassCount = number;
export type WarnCount = number;
export type GeneratedAt = string | null;
export type JudgeModel = string | null;
export type Measures = string;
export type Name1 = "keyword_overlap" | "llm_judge" | "exact_match" | "pass_rate" | "custom";
export type ScoreThreshold = number | null;
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type ProvenanceCaveat = string | null;
export type ReportRef = string | null;
export type AverageManualScore = number | null;
export type AverageScore = number;
export type ExamplesTested = number;
export type FailedExamples = number;
export type PassRate = number | null;
export type WeakTags = string[];
export type ArtifactRef = string | null;
export type Model = string;
export type Phase = "before" | "after" | "standalone";
export type RunRef = string | null;

/**
 * The outcome of evaluating a model/dataset, with an explicit as-served vs raw distinction so a
 * number is never presented as a quality signal without saying what produced it. Formalizes
 * evaluation/reports.EvaluationReport + gates/models.GateReport.
 */
export interface EvaluationResult {
  as_served?: AsServed | null;
  contract_version?: ContractVersion;
  dataset?: EvalDataset | null;
  eval_id: EvalId;
  gate?: EvalGate | null;
  generated_at?: GeneratedAt;
  metric: EvalMetric;
  parameter_accounting_ref?: Ref | null;
  provenance_caveat?: ProvenanceCaveat;
  report_ref?: ReportRef;
  summary: EvalSummary;
  target: EvalTarget;
}
/**
 * How the model was actually served — the RAW-vs-AS-SERVED axis. Two evals of the 'same' model
 * differ if quantization/adapter/template/decoding differ.
 */
export interface AsServed {
  adapter_applied?: AdapterApplied;
  backend?: Backend;
  chat_template_applied?: ChatTemplateApplied;
  decoding?: Decoding;
  precision?: PrecisionMode | null;
  quantization?: QuantizationMode | null;
}
export interface Decoding {
  [k: string]: unknown;
}
export interface EvalDataset {
  dataset_fingerprint?: DatasetFingerprint;
  name?: Name;
  version_ref?: VersionRef;
}
/**
 * Grounded in gates/models.GateReport/GateStatus (pass/warn/block, counts, effective
 * thresholds behind the verdict for reproducibility).
 */
export interface EvalGate {
  block_count?: BlockCount;
  max_regression_score_drop?: MaxRegressionScoreDrop;
  min_eval_average_score?: MinEvalAverageScore;
  min_eval_pass_rate?: MinEvalPassRate;
  overall_status: OverallStatus;
  pass_count?: PassCount;
  warn_count?: WarnCount;
}
/**
 * The scorer AND what it measures — so a number is never treated as quality without
 * qualification (reports.EvaluationReport.metric honesty note).
 */
export interface EvalMetric {
  judge_model?: JudgeModel;
  measures?: Measures;
  name?: Name1;
  score_threshold?: ScoreThreshold;
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
export interface EvalSummary {
  average_manual_score?: AverageManualScore;
  average_score: AverageScore;
  examples_tested: ExamplesTested;
  failed_examples?: FailedExamples;
  pass_rate?: PassRate;
  weak_tags?: WeakTags;
}
export interface EvalTarget {
  artifact_ref?: ArtifactRef;
  model: Model;
  phase?: Phase;
  run_ref?: RunRef;
}
