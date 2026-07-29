/* GENERATED from docs/contracts/FrameworkBackend.schema.json — do not edit. Run: npm run gen:contracts */

/**
 * How a training backend is integrated (docs/TRAINING_BACKEND_REGISTRY.md). Classification only -
 * nothing is installed, probed, or run by declaring a class.
 */
export type BackendCandidateClass =
  "first_party" | "managed_adapter" | "config_export_only" | "research_only" | "defer" | "reject";
export type ContractVersion = "1.0.0";
export type DisplayName = string;
export type FrameworkId = string;
export type FrameworkVersion = string;
export type Name = string | null;
export type Redistributable = boolean | null;
export type Source = "declared" | "model_card" | "dataset_card" | "user_asserted" | "unknown";
export type SpdxId = string | null;
export type Url = string | null;
export type ModelTopologies = ("dense" | "moe")[];
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
export type DeviceKind = "cuda" | "rocm" | "mps" | "xpu" | "cpu";
export type SupportedDevices = DeviceKind[];
export type OperatingSystem = "windows" | "wsl" | "linux" | "macos" | "unknown";
export type SupportedOs = OperatingSystem[];

/**
 * The compute substrate a training run executes on (PyTorch / JAX / TF-Keras / MLX) - split out of
 * the conflated ``BackendManifest`` (P0c, #483). Additive + append-only; dense-default and MoE-safe.
 * Every capability carries a ``SupportLevel``; "installed" is never "supported".
 */
export interface FrameworkBackend {
  candidate_class?: BackendCandidateClass;
  contract_version?: ContractVersion;
  display_name?: DisplayName;
  framework_id: FrameworkId;
  framework_version?: FrameworkVersion;
  license?: License | null;
  model_topologies?: ModelTopologies;
  support_level?: SupportLevel;
  supported_devices?: SupportedDevices;
  supported_os?: SupportedOs;
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
