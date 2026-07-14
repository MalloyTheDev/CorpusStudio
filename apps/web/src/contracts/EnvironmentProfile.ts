/* GENERATED from docs/contracts/EnvironmentProfile.schema.json — do not edit. Run: npm run gen:contracts */

export type CudaDriverVersion = string | null;
export type CudaRuntimeVersion = string | null;
export type DriverVersion = string | null;
export type DeviceKind = "cuda" | "rocm" | "mps" | "xpu" | "cpu";
export type MpsAvailable = boolean | null;
export type NvidiaSmiAvailable = boolean;
export type RocmVersion = string | null;
export type CapturedAt = string | null;
export type ContractVersion = "1.0.0";
export type InstructionSets = string[];
export type LogicalCores = number | null;
export type Model = string;
export type PhysicalCores = number | null;
export type EngineVersion = string;
export type EnvironmentSignature = string;
export type ComputeCapability = string | null;
export type ComputeCapabilityMajor = number | null;
export type Index = number;
export type Name = string;
export type Gen = number | null;
export type Width = number | null;
export type PrecisionMode = "fp32" | "tf32" | "fp16" | "bf16" | "fp8" | "mixed_bf16" | "mixed_fp16";
export type SupportedDtypes = PrecisionMode[];
export type VramFreeBytes = number | null;
export type VramTotalBytes = number | null;
export type Gpus = GpuDevice[];
export type HostnameHash = string | null;
/**
 * How the platform maps device memory. ``wddm`` (Windows) silently spills overflow to shared
 * system RAM and thrashes over PCIe; ``linux_dedicated`` models a hard-OOM boundary rather than a
 * WDDM spill; ``unified_memory`` is Apple MPS / integrated shared memory. This is a planning model,
 * not a measured-fit claim.
 */
export type MemoryResidencyModel = "wddm" | "linux_dedicated" | "unified_memory" | "unknown";
export type OperatingSystem = "windows" | "wsl" | "linux" | "macos" | "unknown";
export type OsDetail = string;
export type PythonVersion = string;
export type Notes = string[];
export type Artifact = string | null;
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Dependencies = string[];
export type Direct = boolean | null;
export type DirectUrl = string | null;
export type Editable = boolean | null;
export type InstalledFileCount = number | null;
export type Installer = string | null;
export type Name1 = string;
export type NormalizedName = string;
export type RecordEntries = number | null;
export type RecordFailedEntries = string[];
export type RecordIntegrity = "verified" | "failed" | "missing" | "unknown";
export type RecordVerifiedEntries = number | null;
export type Requested = boolean | null;
export type Source = "pypi" | "wheel" | "sdist" | "conda" | "vcs" | "local" | "unknown";
export type SourceEvidenceReason = string | null;
export type SourceIndexUrl = string | null;
export type VcsCommit = string | null;
export type VcsRepository = string | null;
export type Version = string | null;
export type Packages = PackageLock[];
export type AvailableBytes = number | null;
export type TotalBytes = number | null;
export type FreeBytes = number | null;
export type Kind = "nvme" | "ssd" | "hdd" | "network" | "unknown";
export type ScratchPath = string | null;

/**
 * The full, hashable SIGNATURE of a host + software environment. Formalizes + greatly extends
 * environment.probe_training_runtime (package versions + GpuInfo), gpu_probe.probe_gpu_memory, and
 * provenance.RunProvenance. A RunManifest/RunPlan references a profile by ``environment_signature``
 * so a result is always tied to the exact environment that produced it.
 */
export interface EnvironmentProfile {
  accelerator_runtime?: AcceleratorRuntime | null;
  captured_at?: CapturedAt;
  contract_version?: ContractVersion;
  cpu?: EnvCpu | null;
  engine_version?: EngineVersion;
  environment_signature: EnvironmentSignature;
  gpus?: Gpus;
  host: EnvHost;
  notes?: Notes;
  packages?: Packages;
  ram?: EnvRam | null;
  storage?: EnvStorage | null;
}
export interface AcceleratorRuntime {
  cuda_driver_version?: CudaDriverVersion;
  cuda_runtime_version?: CudaRuntimeVersion;
  driver_version?: DriverVersion;
  kind?: DeviceKind | null;
  mps_available?: MpsAvailable;
  nvidia_smi_available?: NvidiaSmiAvailable;
  rocm_version?: RocmVersion;
}
export interface EnvCpu {
  instruction_sets?: InstructionSets;
  logical_cores?: LogicalCores;
  model?: Model;
  physical_cores?: PhysicalCores;
}
/**
 * One accelerator. Grounded in environment.GpuInfo + gpu_probe.GpuMemory.
 */
export interface GpuDevice {
  compute_capability?: ComputeCapability;
  compute_capability_major?: ComputeCapabilityMajor;
  index: Index;
  kind: DeviceKind;
  name: Name;
  pcie?: GpuPcie | null;
  supported_dtypes?: SupportedDtypes;
  vram_free_bytes?: VramFreeBytes;
  vram_total_bytes?: VramTotalBytes;
}
export interface GpuPcie {
  gen?: Gen;
  width?: Width;
}
export interface EnvHost {
  hostname_hash?: HostnameHash;
  memory_residency_model?: MemoryResidencyModel;
  os: OperatingSystem;
  os_detail?: OsDetail;
  python_version?: PythonVersion;
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
  artifact_hash?: HashRef | null;
  dependencies?: Dependencies;
  direct?: Direct;
  direct_url?: DirectUrl;
  editable?: Editable;
  hash?: HashRef | null;
  installed_file_count?: InstalledFileCount;
  installed_files_hash?: HashRef | null;
  installer?: Installer;
  name: Name1;
  normalized_name?: NormalizedName;
  record_entries?: RecordEntries;
  record_failed_entries?: RecordFailedEntries;
  record_integrity?: RecordIntegrity;
  record_verified_entries?: RecordVerifiedEntries;
  requested?: Requested;
  source?: Source;
  source_evidence_reason?: SourceEvidenceReason;
  source_index_url?: SourceIndexUrl;
  vcs_commit?: VcsCommit;
  vcs_repository?: VcsRepository;
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
export interface EnvRam {
  available_bytes?: AvailableBytes;
  total_bytes?: TotalBytes;
}
export interface EnvStorage {
  free_bytes?: FreeBytes;
  kind?: Kind;
  scratch_path?: ScratchPath;
}
