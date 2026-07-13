/* GENERATED from docs/contracts/EnvironmentLock.schema.json — do not edit. Run: npm run gen:contracts */

export type Architecture = string;
export type ComputeCapability = string | null;
export type ContractVersion = "1.0.0";
export type CreatedAt = string | null;
export type CudaRuntimeVersion = string | null;
export type Implementation = string;
export type IndexUrls = string[];
export type LockHash = string | null;
export type LockId = string;
export type ManagerVersion = string;
export type Artifact = string | null;
export type Dependencies = string[];
export type DirectUrl = string | null;
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Installer = string | null;
export type Name = string;
export type Requested = boolean | null;
export type Source = "pypi" | "wheel" | "sdist" | "conda" | "vcs" | "local" | "unknown";
export type Version = string | null;
export type Packages = PackageLock[];
export type PlatformTag = string;
export type PythonVersion = string;
export type Id = string;
export type Architecture1 = string;
export type Compatible = boolean;
export type ContractVersion1 = "1.0.0";
export type Executable = string;
export type Implementation1 = string;
export type IncompatibilityReasons = string[];
export type IsVirtualEnvironment = boolean;
export type OperatingSystem = "windows" | "wsl" | "linux" | "macos" | "unknown";
export type Platform = string;
export type RuntimeId = string;
export type VenvAvailable = boolean;
export type Version1 = string;
export type TorchBuild = string | null;
export type TorchVersion = string | null;

/**
 * The exact, reproducible record of what an environment actually contains — the post-install
 * counterpart of a recipe (which is only intent). ``packages`` are the resolved installs with
 * versions + hashes; ``lock_hash`` seals the set for drift detection. NEW.
 */
export interface EnvironmentLock {
  architecture?: Architecture;
  compute_capability?: ComputeCapability;
  contract_version?: ContractVersion;
  created_at?: CreatedAt;
  cuda_runtime_version?: CudaRuntimeVersion;
  implementation?: Implementation;
  index_urls?: IndexUrls;
  lock_hash?: LockHash;
  lock_id: LockId;
  manager_version?: ManagerVersion;
  packages?: Packages;
  platform_tag?: PlatformTag;
  python_version?: PythonVersion;
  recipe_ref: Ref;
  runtime?: PythonRuntime | null;
  torch_build?: TorchBuild;
  torch_version?: TorchVersion;
}
/**
 * A resolved dependency and its install provenance.
 *
 * ``hash`` seals the installed distribution's RECORD metadata when that evidence is available; it
 * is not mislabelled as the original wheel hash. ``direct_url`` and ``artifact`` preserve the
 * stronger source identity pip exposes for direct/VCS/local installs. ``dependencies`` is the
 * installed metadata dependency graph, not a second resolver.
 */
export interface PackageLock {
  artifact?: Artifact;
  dependencies?: Dependencies;
  direct_url?: DirectUrl;
  hash?: HashRef | null;
  installer?: Installer;
  name: Name;
  requested?: Requested;
  source?: Source;
  version?: Version;
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
 * A stable reference to another contract instance by id, optionally pinned to a content hash so
 * the reference cannot silently re-point.
 */
export interface Ref {
  hash?: HashRef | null;
  id: Id;
}
/**
 * A discovered Python executable that can potentially create an isolated worker environment.
 *
 * Discovery never assumes the control-plane interpreter is the only installation. Compatibility is
 * an explicit verdict against the selected recipe, while ``venv_available`` proves the stdlib venv
 * module can be located without creating anything.
 */
export interface PythonRuntime {
  architecture?: Architecture1;
  compatible?: Compatible;
  contract_version?: ContractVersion1;
  executable: Executable;
  implementation?: Implementation1;
  incompatibility_reasons?: IncompatibilityReasons;
  is_virtual_environment?: IsVirtualEnvironment;
  os?: OperatingSystem;
  platform?: Platform;
  runtime_id: RuntimeId;
  venv_available?: VenvAvailable;
  version?: Version1;
}
