/* GENERATED from docs/contracts/EnvironmentRecipe.schema.json — do not edit. Run: npm run gen:contracts */

export type CapabilityProbes = string[];
export type ContractVersion = "1.0.0";
export type DefaultIndexUrl = string | null;
export type Name = string;
export type Optional = boolean;
export type Reason = string | null;
export type Specifier = string | null;
export type DependencyRequirements = DependencyRequirement[];
export type Description = string;
export type DisplayName = string;
export type ExtraIndexUrls = string[];
export type Condition = string;
/**
 * @minItems 1
 */
export type Packages = [string, ...string[]];
export type Severity = "block" | "warn";
export type KnownConflicts = DependencyConflict[];
/**
 * The three dependency layers. The CONTROL PLANE stays lightweight + always installable (opening
 * CorpusStudio must never require CUDA/DeepSpeed/an ML framework); CAPABILITY profiles are opt-in
 * feature stacks added to the core process with graceful fallback; BACKEND_WORKER environments are
 * isolated per-framework runtimes (heavy frameworks pin conflicting torch/CUDA/xformers builds and
 * cannot coexist — they talk to the core via the WorkerMessage protocol, never by import).
 */
export type DependencyLayer = "control_plane" | "capability" | "backend_worker";
export type MinComputeCapability = string | null;
export type Notes = string[];
export type PythonRequires = string;
export type RecipeId = string;
export type RequiresCuda = boolean;
export type RequiresNativeBuild = boolean;
export type OperatingSystem = "windows" | "wsl" | "linux" | "macos" | "unknown";
export type SupportedOs = OperatingSystem[];
export type Target = string;
/**
 * How far a recipe has been proven — the recipe-level twin of EnvironmentState. A recipe is a
 * DECLARATION of what to install; this says whether that declaration has ever produced a working
 * environment, and at what level. ``declared`` = we can render the install plan but have not built +
 * verified it; higher tiers require actual evidence (a real install / probe / hardware run).
 */
export type RecipeVerification = "declared" | "build_verified" | "functional_verified" | "hardware_verified";

/**
 * A declarative, platform/CUDA-aware recipe for building one isolated environment — the WHAT to
 * install, not the act of installing. A recipe is only a declaration: ``verification`` says whether
 * it has ever produced a working environment (declared → hardware_verified). Grounded in the engine's
 * real optional extras (pyproject ``[train]`` / ``[parquet]`` / ``[tokenizer]``).
 */
export interface EnvironmentRecipe {
  capability_probes?: CapabilityProbes;
  contract_version?: ContractVersion;
  cuda_index_urls?: CudaIndexUrls;
  default_index_url?: DefaultIndexUrl;
  dependency_requirements?: DependencyRequirements;
  description?: Description;
  display_name?: DisplayName;
  extra_index_urls?: ExtraIndexUrls;
  known_conflicts?: KnownConflicts;
  layer: DependencyLayer;
  min_compute_capability?: MinComputeCapability;
  notes?: Notes;
  python_requires?: PythonRequires;
  recipe_id: RecipeId;
  requires_cuda?: RequiresCuda;
  requires_native_build?: RequiresNativeBuild;
  supported_os?: SupportedOs;
  target?: Target;
  verification?: RecipeVerification;
}
export interface CudaIndexUrls {
  [k: string]: string;
}
export interface DependencyRequirement {
  name: Name;
  optional?: Optional;
  reason?: Reason;
  specifier?: Specifier;
}
export interface DependencyConflict {
  condition: Condition;
  packages: Packages;
  severity?: Severity;
}
