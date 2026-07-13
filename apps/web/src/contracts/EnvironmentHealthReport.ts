/* GENERATED from docs/contracts/EnvironmentHealthReport.schema.json — do not edit. Run: npm run gen:contracts */

export type ChangedPackageSources = string[];
export type CheckedAt = string | null;
export type ContractVersion = "1.0.0";
export type CudaMismatch = boolean;
export type DriftDetected = boolean;
export type DriftedPackages = string[];
export type EnvironmentMissing = boolean;
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type ContractVersion1 = "1.0.0";
export type Detail = string | null;
export type DetectedAt = string | null;
export type ExceptionType = string | null;
export type ExitCode = number | null;
/**
 * ``math``/``eager`` is forced on Blackwell sm_120 — the fused flash/mem-efficient kernels
 * deadlock on the first backward (training/environment.py, estimators.py) — at a large activation
 * VRAM cost.
 */
export type AttentionImpl =
  "math" | "eager" | "sdpa" | "flash_attention_2" | "flash_attention_3" | "mem_efficient" | "xformers";
/**
 * The fit verdict. ``NATIVE_*`` = fully resident. ``CONTROLLED_*`` = a deliberate, planned
 * offload (acceptable, slower). ``ACCIDENTAL_*`` / ``THRASHING`` = an unplanned spill the platform
 * did silently (the failure mode the engine warns about). ``FAIL`` = will not run.
 */
export type FitClass =
  | "PLANNED_UNPROVEN"
  | "NATIVE_SAFE"
  | "NATIVE_TIGHT"
  | "NATIVE_UNPROVEN"
  | "MARGINAL"
  | "CONTROLLED_ACTIVATION_OFFLOAD"
  | "CONTROLLED_OPTIMIZER_OFFLOAD"
  | "CONTROLLED_PARAMETER_OFFLOAD"
  | "ACCIDENTAL_UNIFIED_MEMORY_PAGING"
  | "ACCIDENTAL_WDDM_SPILL"
  | "THRASHING"
  | "FAIL";
export type ContractVersion2 = "1.0.0";
export type DeviceCapacityBytes = number | null;
export type EstimatedPeakBytes = number | null;
export type HeadroomBytes = number | null;
export type Rationale = string;
export type CudaDeviceFreeBytes = number | null;
export type CudaDeviceUsedBytes = number | null;
export type DedicatedGpuBytes = number | null;
export type ProcessRssBytes = number | null;
export type SharedGpuBytes = number | null;
export type SystemRamUsedBytes = number | null;
export type TorchAllocatedBytes = number | null;
export type TorchPeakAllocatedBytes = number | null;
export type TorchPeakReservedBytes = number | null;
export type TorchReservedBytes = number | null;
export type Message = string;
export type Reconciled = boolean;
export type Remediation = string | null;
export type RunId = string | null;
export type Signal = string | null;
/**
 * Ordered lifecycle stage of a run, launch → export. A RunEvent carries the stage it belongs to
 * so a consumer can render a precise progress spine and localize a failure to the exact stage.
 */
export type StageMarker =
  | "process_start"
  | "env_loaded"
  | "cuda_init"
  | "model_loaded"
  | "quantized"
  | "adapter_attached"
  | "optimizer_created"
  | "batch_materialized"
  | "forward"
  | "loss"
  | "backward"
  | "optimizer_step"
  | "checkpoint"
  | "reload"
  | "evaluation"
  | "export";
/**
 * Terminal outcome category. ``PASS`` is included so the same enum classifies a completed
 * probe/run, not only failures. Grounded in the exact hazards the engine documents: the sm_120
 * fused-attention deadlock (KERNEL_STALL), the WDDM silent spill (ACCIDENTAL_SPILL vs a clean
 * OOM), and env/dependency mismatches (ENVIRONMENT_FAILURE).
 */
export type FailureTaxonomy =
  | "PASS"
  | "FAIL"
  | "OOM"
  | "TIMEOUT"
  | "KERNEL_STALL"
  | "NUMERICAL_FAILURE"
  | "CHECKPOINT_FAILURE"
  | "ENVIRONMENT_FAILURE"
  | "UNSUPPORTED_CONFIGURATION"
  | "ACCIDENTAL_SPILL"
  | "CONTROLLED_OFFLOAD";
export type HardwareMismatch = boolean;
export type Artifact = string | null;
export type Dependencies = string[];
export type DirectUrl = string | null;
export type Installer = string | null;
export type Name = string;
export type Requested = boolean | null;
export type Source = "pypi" | "wheel" | "sdist" | "conda" | "vcs" | "local" | "unknown";
export type Version = string | null;
export type InstalledPackages = PackageLock[];
export type InterpreterMissing = boolean;
export type LockMismatch = boolean;
export type MissingRequirements = string[];
export type Detail1 = string | null;
export type Probe = string;
export type ProbeResults = ProbeResult[];
export type PythonVersion = string;
export type RecipeDriftDetected = boolean;
export type Remediation1 = string | null;
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

/**
 * The live health of a managed environment: its state, drift vs the recorded lock, and probe
 * outcomes. ``drift_detected`` means the installed set no longer matches the lock (a package changed
 * under the env). NEW.
 */
export interface EnvironmentHealthReport {
  changed_package_sources?: ChangedPackageSources;
  checked_at?: CheckedAt;
  contract_version?: ContractVersion;
  cuda_mismatch?: CudaMismatch;
  drift_detected?: DriftDetected;
  drifted_packages?: DriftedPackages;
  environment_missing?: EnvironmentMissing;
  environment_ref: Ref;
  failure?: FailureRecord | null;
  hardware_mismatch?: HardwareMismatch;
  installed_packages?: InstalledPackages;
  interpreter_missing?: InterpreterMissing;
  lock_mismatch?: LockMismatch;
  lock_ref?: Ref | null;
  missing_requirements?: MissingRequirements;
  probe_results?: ProbeResults;
  python_version?: PythonVersion;
  recipe_drift_detected?: RecipeDriftDetected;
  recipe_ref?: Ref | null;
  remediation?: Remediation1;
  state: EnvironmentState;
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
 * A structured, classified terminal outcome for a run, capability probe, or export. The
 * taxonomy turns 'it died' into an actionable category — a real OOM vs a KERNEL_STALL (the sm_120
 * fused-attention deadlock) vs an ACCIDENTAL_SPILL vs a CONTROLLED_OFFLOAD. NEW.
 */
export interface FailureRecord {
  contract_version?: ContractVersion1;
  detail?: Detail;
  detected_at?: DetectedAt;
  exception_type?: ExceptionType;
  exit_code?: ExitCode;
  fit_at_failure?: FitClassification | null;
  memory_at_failure?: MemoryMetrics | null;
  message: Message;
  reconciled?: Reconciled;
  remediation?: Remediation;
  run_id?: RunId;
  signal?: Signal;
  stage?: StageMarker | null;
  taxonomy: FailureTaxonomy;
}
/**
 * The planner/calibrator verdict on whether a resolved RunPlan fits the target environment, and
 * HOW: a native fit, a deliberately-offloaded fit, or an ACCIDENTAL spill (the silent WDDM/unified
 * paging that looks frozen but crawls at 10-25x). NEW — the engine emits only a coarse warn/pass
 * VRAM band (preflight.gpu_memory, _VRAM_SAFETY_MARGIN_GB).
 */
export interface FitClassification {
  attention_path?: AttentionImpl | null;
  classification: FitClass;
  contract_version?: ContractVersion2;
  device_capacity_bytes?: DeviceCapacityBytes;
  estimated_peak_bytes?: EstimatedPeakBytes;
  headroom_bytes?: HeadroomBytes;
  rationale?: Rationale;
}
/**
 * The full memory-signature block sampled during a run. Distinguishes PyTorch's allocator view,
 * raw CUDA device memory, and OS-level residency (``dedicated`` vs ``shared`` GPU memory) so a
 * Windows/WDDM spill to shared memory is VISIBLE rather than hidden inside 'used VRAM'. Grounded in
 * gpu_probe.GpuMemory + the estimators note that torch.max_memory_allocated counts the WDDM spill.
 */
export interface MemoryMetrics {
  cuda_device_free_bytes?: CudaDeviceFreeBytes;
  cuda_device_used_bytes?: CudaDeviceUsedBytes;
  dedicated_gpu_bytes?: DedicatedGpuBytes;
  process_rss_bytes?: ProcessRssBytes;
  shared_gpu_bytes?: SharedGpuBytes;
  system_ram_used_bytes?: SystemRamUsedBytes;
  torch_allocated_bytes?: TorchAllocatedBytes;
  torch_peak_allocated_bytes?: TorchPeakAllocatedBytes;
  torch_peak_reserved_bytes?: TorchPeakReservedBytes;
  torch_reserved_bytes?: TorchReservedBytes;
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
export interface ProbeResult {
  detail?: Detail1;
  measured?: Measured;
  outcome: FailureTaxonomy;
  probe: Probe;
}
export interface Measured {
  [k: string]: unknown;
}
