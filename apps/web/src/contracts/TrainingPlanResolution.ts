/* GENERATED from docs/contracts/TrainingPlanResolution.schema.json — do not edit. Run: npm run gen:contracts */

export type CheckpointImpl = "full_state" | "adapter_only" | "sharded" | "distcp" | "safetensors";
export type EvaluationProfile = string | null;
export type Framework = string;
export type DeviceKind = "cuda" | "rocm" | "mps" | "xpu" | "cpu";
export type ModelTopology = "dense" | "moe";
export type ObjectiveId = string;
export type Orchestrator = string;
export type ParallelismKind = "data" | "tensor" | "pipeline" | "expert" | "sequence" | "context";
export type Parallelism = ParallelismKind[];
export type PrecisionMode = "fp32" | "tf32" | "fp16" | "bf16" | "fp8" | "mixed_bf16" | "mixed_fp16";
export type Preset = string | null;
export type QuantizationMode = "none" | "int8" | "int4" | "nf4" | "fp4" | "gptq" | "awq" | "hqq";
export type AdapterMethod =
  "none" | "lora" | "qlora" | "dora" | "ia3" | "full_finetune" | "prompt_tuning" | "prefix_tuning";
export type ContractVersion = "1.0.0";
export type PlanIntentId = string;
export type Code = string;
export type Message = string;
export type Severity = "info" | "warning";
export type PrecheckFindings = TrainingPlanCompatibilityFinding[];
export type ResolvedAt = string | null;
/**
 * @minItems 1
 */
export type RunPlanRefs = [Ref, ...Ref[]];
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;

/**
 * The result of lowering a :class:`TrainingPlan` into one-or-more RunPlans (Training Systems P0b,
 * #482). It references the resolved RunPlan(s) by their sealed plan identity, carries the advisory
 * pre-check findings, and links back to the intent. It has NO sealing authority of its own:
 * ``RunPlan.plan_hash`` remains the execution seal, and every ref here must carry it.
 */
export interface TrainingPlanResolution {
  composition: TrainingPlanComposition;
  contract_version?: ContractVersion;
  plan_intent_id: PlanIntentId;
  precheck_findings?: PrecheckFindings;
  resolved_at?: ResolvedAt;
  run_plan_refs: RunPlanRefs;
}
/**
 * One entry selected per registry dimension (the architecture doc's registries). Pre-resolution
 * intent, dense-default and MoE-compatible - it never assumes dense execution. These are capability
 * SELECTIONS (refs into the thin registries), not sealed execution fields; the ``parameters`` block
 * carries the executable knobs and the compatibility pre-check validates the two are consistent.
 */
export interface TrainingPlanComposition {
  checkpoint_strategy?: CheckpointImpl;
  evaluation_profile?: EvaluationProfile;
  framework?: Framework;
  hardware_target?: DeviceKind;
  model_topology?: ModelTopology;
  objective_id: ObjectiveId;
  orchestrator?: Orchestrator;
  parallelism?: Parallelism;
  precision?: PrecisionMode;
  preset?: Preset;
  quantization?: QuantizationMode;
  update_method?: AdapterMethod;
}
/**
 * One advisory cross-dimension compatibility finding from the UX pre-check. Advisory ONLY - it
 * never gates resolution or execution; the authoritative gate stays the planner's declared-and-proven
 * capability check plus the exact ``ExecutionCapabilityCombination``-in-a-passing-probe match.
 */
export interface TrainingPlanCompatibilityFinding {
  code: Code;
  message: Message;
  severity?: Severity;
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
