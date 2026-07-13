/* GENERATED from docs/contracts/StorageProfile.schema.json — do not edit. Run: npm run gen:contracts */

export type DeviceMountPoint = string | null;
export type FreeBytes = number | null;
/**
 * How a storage device attaches. The interface — not just free space — decides whether a device
 * can sustain the heavy sequential + random writes of optimizer/parameter offload and checkpointing.
 * A USB bridge or a network mount will thrash under sustained offload even with terabytes free.
 */
export type StorageInterface = "nvme_pcie" | "sata_ssd" | "hdd" | "usb" | "network" | "virtual" | "unknown";
export type Path = string;
export type Reasons = string[];
export type RequiredFreeBytes = number | null;
/**
 * The role a path plays in a run. Roles differ in access pattern: ``optimizer_offload`` /
 * ``parameter_offload`` / ``scratch`` / ``checkpoints`` are WRITE-heavy; ``model_cache`` /
 * ``dataset_cache`` are read-LATENCY-sensitive during load; ``source_repo`` / ``python_env`` are
 * thousands of SMALL files touched on every process start (an import over a USB bridge or a WSL
 * ``/mnt`` mount stalls); ``archive`` just wants capacity. A path's suitability is judged PER ROLE (a
 * USB SSD is fine for ``archive``, poor for ``model_cache``, unfit for ``optimizer_offload``).
 */
export type StorageRole =
  | "os"
  | "source_repo"
  | "python_env"
  | "model_cache"
  | "dataset_cache"
  | "checkpoints"
  | "scratch"
  | "optimizer_offload"
  | "parameter_offload"
  | "artifacts"
  | "archive"
  | "logs";
/**
 * The per-role verdict for a candidate path. ``unsuitable`` is a hard no (data-loss or
 * thrash-to-a-halt risk); ``marginal`` will work but degrade (e.g. an HDD for offload); ``unknown``
 * when detection couldn't characterize the device (honest, never a false ``suitable``).
 */
export type StorageSuitability = "suitable" | "marginal" | "unsuitable" | "unknown";
export type Assessments = StorageRoleAssessment[];
export type CapturedAt = string | null;
export type ContractVersion = "1.0.0";
export type CloudSynced = boolean | null;
export type DeviceName = string | null;
export type Filesystem = string;
export type FreeBytes1 = number | null;
/**
 * How a storage device attaches. The interface — not just free space — decides whether a device
 * can sustain the heavy sequential + random writes of optimizer/parameter offload and checkpointing.
 * A USB bridge or a network mount will thrash under sustained offload even with terabytes free.
 */
export type StorageInterface1 = "nvme_pcie" | "sata_ssd" | "hdd" | "usb" | "network" | "virtual" | "unknown";
export type MountPoint = string;
export type Notes = string[];
export type Removable = boolean | null;
export type Rotational = boolean | null;
export type TotalBytes = number | null;
export type WslHostDrive = boolean | null;
export type Devices = StorageDevice[];
export type Notes1 = string[];

/**
 * The host's storage topology + optional per-role path assessments — the input the run planner
 * needs to assign offload/checkpoint/scratch paths SAFELY (§11/§20). Standalone (not folded into
 * EnvironmentProfile) so it never perturbs the ``environment_signature``. NEW: the engine has no
 * storage detection today (EnvStorage was a scratch_path/free_bytes/kind stub).
 */
export interface StorageProfile {
  assessments?: Assessments;
  captured_at?: CapturedAt;
  contract_version?: ContractVersion;
  devices?: Devices;
  notes?: Notes1;
}
/**
 * The PER-ROLE verdict for a candidate path: can it play this role, and if not, WHY. The reasons
 * are the safe-spill guardrail's human-readable justification (USB bridge / synced folder / free-space
 * margin / inside the source repo / rotational disk).
 */
export interface StorageRoleAssessment {
  device_mount_point?: DeviceMountPoint;
  free_bytes?: FreeBytes;
  interface?: StorageInterface;
  path: Path;
  reasons?: Reasons;
  required_free_bytes?: RequiredFreeBytes;
  role: StorageRole;
  suitability: StorageSuitability;
}
/**
 * One characterized storage location, from a dependency-light, NON-destructive probe (mount +
 * capacity + cheaply-discoverable device attributes). The heavy metrics the spec envisions —
 * measured sequential/random throughput, SMART/NVMe endurance, temperature — are deliberately absent
 * here: they require a bounded benchmark or a privileged SMART read (a later, consent-gated slice).
 * Unknown fields stay ``None``/``unknown`` — an honest gap, never a guessed value.
 */
export interface StorageDevice {
  cloud_synced?: CloudSynced;
  device_name?: DeviceName;
  filesystem?: Filesystem;
  free_bytes?: FreeBytes1;
  interface?: StorageInterface1;
  mount_point: MountPoint;
  notes?: Notes;
  removable?: Removable;
  rotational?: Rotational;
  total_bytes?: TotalBytes;
  wsl_host_drive?: WslHostDrive;
}
