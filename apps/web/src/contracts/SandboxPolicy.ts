/* GENERATED from docs/contracts/SandboxPolicy.schema.json — do not edit. Run: npm run gen:contracts */

export type ContractVersion = "1.0.0";
export type GpuDevices = string[];
export type NetworkIsolated = true;
export type NoNewPrivileges = true;
export type ReadonlyRoot = true;
export type RlimitAddressSpaceBytes = number | null;
export type RlimitCpuSeconds = number | null;
export type RlimitOpenFiles = number | null;
export type RlimitProcesses = number | null;
export type WritablePaths = string[];

/**
 * The OS-level containment policy for executing VETTED-BUT-UNTRUSTED custom-block code (mode 3, slice
 * 3). A static screen is not a safety proof; this policy is the real blast-radius limit. Three invariants
 * are TYPE-LOCKED so an untrusted-code sandbox can never be weakened: no network, a read-only root, and
 * no-new-privileges. ``writable_paths`` are the only rw exceptions (the run-scoped output dir);
 * ``gpu_devices`` is an HONEST, documented hole - a GPU training block must reach the CUDA devices, so a
 * GPU workload is blast-radius-limited, NOT fully isolated. Enforced by the sandbox launcher; a host with
 * no usable backend refuses to run custom code (fail-closed). ``bubblewrap`` is the primary backend, with
 * an ``unshare`` + rlimits fallback that is weaker (no filesystem confinement).
 */
export interface SandboxPolicy {
  contract_version?: ContractVersion;
  gpu_devices?: GpuDevices;
  network_isolated?: NetworkIsolated;
  no_new_privileges?: NoNewPrivileges;
  readonly_root?: ReadonlyRoot;
  rlimit_address_space_bytes?: RlimitAddressSpaceBytes;
  rlimit_cpu_seconds?: RlimitCpuSeconds;
  rlimit_open_files?: RlimitOpenFiles;
  rlimit_processes?: RlimitProcesses;
  writable_paths?: WritablePaths;
}
