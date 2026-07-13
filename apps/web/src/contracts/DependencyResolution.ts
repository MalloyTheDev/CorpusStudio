/* GENERATED from docs/contracts/DependencyResolution.schema.json — do not edit. Run: npm run gen:contracts */

export type AcceleratorTag = string;
export type BlockingReasons = string[];
export type ContractVersion = "1.0.0";
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type EnvironmentRoot = string | null;
export type EstimatedDiskBytes = number | null;
export type EstimatedDownloadBytes = number | null;
/**
 * @minItems 1
 */
export type Argv = [string, ...string[]];
export type Description = string;
export type ExpectedOutputs = string[];
export type NativeBuildExpected = boolean;
export type NetworkRequired = boolean;
export type Phase = "create_venv" | "upgrade_pip" | "install" | "verify";
export type TimeoutSeconds = number;
export type WorkingDirectory = string | null;
export type InstallSteps = InstallStep[];
export type ManagerVersion = string;
export type OperatingSystem = "windows" | "wsl" | "linux" | "macos" | "unknown";
export type PythonVersion = string;
export type ResolutionHash = string | null;
export type Resolvable = boolean;
export type ResolvedIndexUrls = string[];
export type Architecture = string;
export type Compatible = boolean;
export type ContractVersion1 = "1.0.0";
export type Executable = string;
export type Implementation = string;
export type IncompatibilityReasons = string[];
export type IsVirtualEnvironment = boolean;
export type OperatingSystem1 = "windows" | "wsl" | "linux" | "macos" | "unknown";
export type Platform = string;
export type RuntimeId = string;
export type VenvAvailable = boolean;
export type Version = string;
export type Warnings = string[];

/**
 * The resolved PREVIEW of provisioning a recipe on a specific host — the exact argv steps, the
 * chosen wheel index, and the disk/network cost — for explicit user confirmation BEFORE anything is
 * installed. Pure/derivable; no environment is created to produce it. NEW.
 */
export interface DependencyResolution {
  accelerator_tag?: AcceleratorTag;
  blocking_reasons?: BlockingReasons;
  contract_version?: ContractVersion;
  environment_ref?: Ref | null;
  environment_root?: EnvironmentRoot;
  estimated_disk_bytes?: EstimatedDiskBytes;
  estimated_download_bytes?: EstimatedDownloadBytes;
  install_steps?: InstallSteps;
  manager_version?: ManagerVersion;
  os?: OperatingSystem;
  python_version?: PythonVersion;
  recipe_ref: Ref;
  resolution_hash?: ResolutionHash;
  resolvable?: Resolvable;
  resolved_index_urls?: ResolvedIndexUrls;
  runtime?: PythonRuntime | null;
  warnings?: Warnings;
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
 * One bounded, argv-structured install command — NEVER a shell string, so an untrusted package or
 * index name can't inject a command (mirrors the no-shell trainer-launch invariant). ``argv[0]`` is
 * the executable; the rest are literal arguments.
 */
export interface InstallStep {
  argv: Argv;
  description?: Description;
  environment?: Environment;
  expected_outputs?: ExpectedOutputs;
  native_build_expected?: NativeBuildExpected;
  network_required?: NetworkRequired;
  phase: Phase;
  timeout_seconds?: TimeoutSeconds;
  working_directory?: WorkingDirectory;
}
export interface Environment {
  [k: string]: string;
}
/**
 * A discovered Python executable that can potentially create an isolated worker environment.
 *
 * Discovery never assumes the control-plane interpreter is the only installation. Compatibility is
 * an explicit verdict against the selected recipe, while ``venv_available`` proves the stdlib venv
 * module can be located without creating anything.
 */
export interface PythonRuntime {
  architecture?: Architecture;
  compatible?: Compatible;
  contract_version?: ContractVersion1;
  executable: Executable;
  implementation?: Implementation;
  incompatibility_reasons?: IncompatibilityReasons;
  is_virtual_environment?: IsVirtualEnvironment;
  os?: OperatingSystem1;
  platform?: Platform;
  runtime_id: RuntimeId;
  venv_available?: VenvAvailable;
  version?: Version;
}
