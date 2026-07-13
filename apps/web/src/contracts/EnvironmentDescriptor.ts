/* GENERATED from docs/contracts/EnvironmentDescriptor.schema.json — do not edit. Run: npm run gen:contracts */

export type ContractVersion = "1.0.0";
export type CreatedAt = string | null;
export type EnvId = string;
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
/**
 * The three dependency layers. The CONTROL PLANE stays lightweight + always installable (opening
 * CorpusStudio must never require CUDA/DeepSpeed/an ML framework); CAPABILITY profiles are opt-in
 * feature stacks added to the core process with graceful fallback; BACKEND_WORKER environments are
 * isolated per-framework runtimes (heavy frameworks pin conflicting torch/CUDA/xformers builds and
 * cannot coexist — they talk to the core via the WorkerMessage protocol, never by import).
 */
export type DependencyLayer = "control_plane" | "capability" | "backend_worker";
export type ManagedBy = string;
export type ManagerVersion = string;
export type Notes = string[];
export type PythonExecutable = string;
export type RootPath = string;
/**
 * The lifecycle state of a managed environment. The escalation is deliberate — "installed" is
 * NEVER "supported": a package importing (IMPORTABLE) is not proof a kernel runs
 * (FUNCTIONAL_PROBE_PASSED), which is not proof the hardware supports it (HARDWARE_VERIFIED). Only
 * HARDWARE_VERIFIED earns "supported". The terminal-degraded states record WHY an env is unusable.
 */
export type EnvironmentState =
  | "NOT_INSTALLED"
  | "INSTALLING"
  | "INSTALLED_UNCHECKED"
  | "IMPORTABLE"
  | "DEPENDENCY_PROBE_PASSED"
  | "FUNCTIONAL_PROBE_PASSED"
  | "HARDWARE_VERIFIED"
  | "DEGRADED"
  | "INCOMPATIBLE"
  | "DRIFTED"
  | "BROKEN";
export type UpdatedAt = string | null;

/**
 * A managed, ISOLATED environment instance. Its ``root_path`` is the isolation boundary — the
 * Environment Manager only ever installs into this env's own interpreter, never another's, so one
 * backend can't corrupt another's runtime. NEW.
 */
export interface EnvironmentDescriptor {
  contract_version?: ContractVersion;
  created_at?: CreatedAt;
  env_id: EnvId;
  installation_ref?: Ref | null;
  layer: DependencyLayer;
  lock_ref?: Ref | null;
  managed_by?: ManagedBy;
  manager_version?: ManagerVersion;
  notes?: Notes;
  python_executable?: PythonExecutable;
  recipe_ref: Ref;
  resolution_ref?: Ref | null;
  root_path?: RootPath;
  state?: EnvironmentState;
  updated_at?: UpdatedAt;
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
