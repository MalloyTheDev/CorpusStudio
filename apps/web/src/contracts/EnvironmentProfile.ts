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
 * system RAM and thrashes over PCIe; ``linux_dedicated`` hard-OOMs instead; ``unified_memory`` is
 * Apple MPS / integrated shared memory. The single most decisive field for spill-vs-OOM.
 */
export type MemoryResidencyModel = "wddm" | "linux_dedicated" | "unified_memory" | "unknown";
export type OperatingSystem = "windows" | "linux" | "macos" | "unknown";
export type OsDetail = string;
export type PythonVersion = string;
export type Notes = string[];
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Name1 = string;
export type Source = "pypi" | "wheel" | "sdist" | "conda" | "vcs" | "local" | "unknown";
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
 * A resolved dependency: distribution name → installed version + optional wheel/artifact hash.
 * Grounded in environment.probe_training_runtime (importlib.metadata.version, no import).
 */
export interface PackageLock {
  hash?: HashRef | null;
  name: Name1;
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
export interface EnvRam {
  available_bytes?: AvailableBytes;
  total_bytes?: TotalBytes;
}
export interface EnvStorage {
  free_bytes?: FreeBytes;
  kind?: Kind;
  scratch_path?: ScratchPath;
}
