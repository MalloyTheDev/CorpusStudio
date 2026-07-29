/* GENERATED from docs/contracts/OrchestratorAdapter.schema.json — do not edit. Run: npm run gen:contracts */

/**
 * How a training backend is integrated (docs/TRAINING_BACKEND_REGISTRY.md). Classification only -
 * nothing is installed, probed, or run by declaring a class.
 */
export type BackendCandidateClass =
  "first_party" | "managed_adapter" | "config_export_only" | "research_only" | "defer" | "reject";
export type ConfigGenerator = string;
export type ContractVersion = "1.0.0";
export type DisplayName = string;
export type FailureParser = string;
export type FrameworkRef = string;
export type Launcher = string;
export type OrchestratorId = string;
export type ProgressParser = string;
export type ContractVersion1 = "1.0.0";
/**
 * How much network / download a backend's isolated worker process is DECLARED to need. Graduated so
 * a stricter assurance tier can refuse a broader scope.
 */
export type AccessScope = "none" | "allowlisted" | "unrestricted";
/**
 * How much network / download a backend's isolated worker process is DECLARED to need. Graduated so
 * a stricter assurance tier can refuse a broader scope.
 */
export type AccessScope1 = "none" | "allowlisted" | "unrestricted";
export type SecurityBoundaries = string[];
export type TrustRemoteCode = boolean;
/**
 * A coarse, single-value capability-support ROLLUP - "how far has this capability been proven,
 * end to end?" - that COEXISTS with the shipped multi-axis ladders (ObjectiveVerification's 3 axes,
 * VerificationOutcome, RecipeVerification), which remain the authority. A SupportLevel never says
 * WHICH axis is proven: the projection onto it (``support_level.project_objective_verification``) is
 * a lossy PARTIAL mapping that CARRIES not_applicable / partial / not_checked rather than inventing a
 * level. "installed" is never "supported" - a higher state requires the evidence its name implies,
 * never a lower one plus optimism (the repo-wide "a completed step != proven fit" invariant).
 */
export type SupportLevel =
  | "declared"
  | "config_generation_only"
  | "installed"
  | "probed"
  | "workload_verified"
  | "production_supported"
  | "refused";

/**
 * The training-loop driver (HF/TRL, Unsloth, torchtune, Axolotl, Megatron, ...) bound to a
 * ``FrameworkBackend`` - split out of the conflated ``BackendManifest`` (P0c, #483). It declares its
 * config generator / launcher / progress + failure parsers and its ``BackendSecurityPosture``, which
 * the planner can refuse by assurance tier. Additive + append-only; ``BackendManifest`` is untouched.
 */
export interface OrchestratorAdapter {
  candidate_class?: BackendCandidateClass;
  config_generator?: ConfigGenerator;
  contract_version?: ContractVersion;
  display_name?: DisplayName;
  failure_parser?: FailureParser;
  framework_ref: FrameworkRef;
  launcher?: Launcher;
  orchestrator_id: OrchestratorId;
  progress_parser?: ProgressParser;
  security_posture?: BackendSecurityPosture;
  support_level?: SupportLevel;
}
/**
 * A backend's DECLARED security posture - what its isolated worker process is allowed to do. The
 * TrainingPlan resolver REFUSES a posture that exceeds the run's assurance tier (fail-closed; see
 * ``training_plan.resolve_training_plan`` -> ``backend_registry.refuse_backend_security``). ``trust_remote_code`` is arbitrary code execution:
 * it is declarable ON PURPOSE (so a managed adapter cannot enable it invisibly inside its own
 * process) and refused at every tier. ``network_access`` / ``download_access`` tighten as the tier
 * hardens. This is separate from the sealed reference execution path, which keeps its own
 * ``trust_remote_code`` hard-lock (``Literal[False]``).
 */
export interface BackendSecurityPosture {
  contract_version?: ContractVersion1;
  download_access?: AccessScope;
  network_access?: AccessScope1;
  security_boundaries?: SecurityBoundaries;
  trust_remote_code?: TrustRemoteCode;
}
