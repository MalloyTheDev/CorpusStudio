/* GENERATED from docs/contracts/EnvironmentInstallation.schema.json — do not edit. Run: npm run gen:contracts */

/**
 * @minItems 1
 */
export type Argv = [string, ...string[]];
export type Cancelled = boolean;
export type CommandId = string;
export type ExitCode = number | null;
export type ExpectedOutputs = string[];
export type ContractVersion = "1.0.0";
export type Detail = string | null;
export type DetectedAt = string | null;
export type ExceptionType = string | null;
export type ExitCode1 = number | null;
/**
 * ``math``/``eager`` is forced on native-Windows/WDDM Blackwell sm_120 because the fused flash
 * kernel deadlocks there. Other platforms require their own functional capability result; WSL
 * evidence is not bare-Linux proof.
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
export type ContractVersion1 = "1.0.0";
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
export type FinishedAt = string;
export type NativeBuildOccurred = boolean;
export type Phase =
  | "create_venv"
  | "upgrade_pip"
  | "install"
  | "lock"
  | "import_probe"
  | "verify"
  | "dependency_probe"
  | "functional_probe"
  | "hardware_probe"
  | "health_probe";
export type StartedAt = string;
export type StderrPath = string | null;
export type StdoutPath = string | null;
export type TimedOut = boolean;
export type TimeoutSeconds = number;
export type WorkingDirectory = string;
export type Commands = EnvironmentCommandRecord[];
export type ContractVersion2 = "1.0.0";
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type FinishedAt1 = string | null;
export type InstallationId = string;
export type RetryRequiresRecreate = boolean;
export type StartedAt1 = string;
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
 * Recoverable journal for one environment creation attempt.
 */
export interface EnvironmentInstallation {
  commands?: Commands;
  contract_version?: ContractVersion2;
  environment_ref: Ref;
  failure?: FailureRecord | null;
  finished_at?: FinishedAt1;
  installation_id: InstallationId;
  recipe_ref: Ref;
  resolution_ref: Ref;
  retry_requires_recreate?: RetryRequiresRecreate;
  started_at: StartedAt1;
  state?: EnvironmentState;
}
/**
 * Durable evidence for one bounded, argv-only creation/install/probe command.
 */
export interface EnvironmentCommandRecord {
  argv: Argv;
  cancelled?: Cancelled;
  command_id: CommandId;
  environment?: Environment;
  exit_code?: ExitCode;
  expected_outputs?: ExpectedOutputs;
  failure?: FailureRecord | null;
  finished_at: FinishedAt;
  native_build_occurred?: NativeBuildOccurred;
  phase: Phase;
  started_at: StartedAt;
  stderr_path?: StderrPath;
  stdout_path?: StdoutPath;
  timed_out?: TimedOut;
  timeout_seconds: TimeoutSeconds;
  working_directory: WorkingDirectory;
}
export interface Environment {
  [k: string]: string;
}
/**
 * A structured, classified terminal outcome for a run, capability probe, or export. The
 * taxonomy turns 'it died' into an actionable category — a real OOM vs a KERNEL_STALL (the sm_120
 * fused-attention deadlock) vs an ACCIDENTAL_SPILL vs a CONTROLLED_OFFLOAD. NEW.
 */
export interface FailureRecord {
  contract_version?: ContractVersion;
  detail?: Detail;
  detected_at?: DetectedAt;
  exception_type?: ExceptionType;
  exit_code?: ExitCode1;
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
  contract_version?: ContractVersion1;
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
