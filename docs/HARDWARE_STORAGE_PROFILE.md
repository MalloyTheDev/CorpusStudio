# Hardware & Storage Profile

Corpus Studio plans training runs against **real** hardware, not assumptions. The
[`EnvironmentProfile`](contracts/EnvironmentProfile.schema.json) already characterizes the OS, CPU,
RAM, and GPUs (see [`PLATFORM_RUN.md`](PLATFORM_RUN.md)). This document covers the **storage** half:
the `StorageProfile` contract and the per-role *safe-spill guardrail* that decides whether a path is
fit to hold checkpoints, scratch, or an optimizer/parameter **offload** file.

This matters because offloading is how a model that doesn't fit in VRAM still trains (CPU/NVMe
offload, DeepSpeed ZeRO, activation checkpointing). Offload turns a disk into part of the training
loop — and putting that file on a **USB drive, a cloud-sync folder, a nearly-full disk, or inside the
source repository** is a run-halting (or data-losing) mistake. The planner needs to *know the disk*
before it can *plan the offload*.

## What is measured (and what is honestly not)

Detection is **dependency-light and non-destructive** — mount points, capacity, and cheaply
discoverable device attributes only. There is **no benchmark and no privileged SMART read**, so the
following stay honestly absent (a later, consent-gated slice):

- measured sequential / random throughput and latency;
- SMART / NVMe endurance (`data_units_written`, TBW, percentage-used);
- device temperature.

Anything detection can't cheaply determine is `null` / `unknown` — never a guessed number.

| Field | Source | Notes |
|---|---|---|
| `mount_point`, `filesystem` | `/proc/mounts` (Linux/WSL), `GetVolumeInformationW` (Windows) | |
| `total_bytes`, `free_bytes` | `shutil.disk_usage` | cross-platform |
| `interface` | `/sys/block/*/…` (Linux), `GetDriveTypeW` (Windows) | `nvme_pcie` / `sata_ssd` / `hdd` / `usb` / `network` / `virtual` / `unknown` |
| `removable`, `rotational` | `/sys/block/<disk>/removable`, `queue/rotational` (Linux) | `null` when not determinable (e.g. SSD-vs-HDD on Windows needs WMI) |
| `cloud_synced` | path-component match against known sync-client folders | Dropbox / OneDrive / Google Drive / iCloud / pCloud / Nextcloud / … |

### WSL

Under WSL, `/mnt/c` and `drvfs`/`9p` mounts are the **Windows host drives seen through a translation
layer** — the real device attributes aren't visible from Linux. Those devices are reported with
`interface = virtual` and a note saying so, rather than a fabricated verdict.

## Storage roles

A path's suitability is judged **per role**, because roles differ in write intensity and durability
needs. A USB drive is fine for `archive`, unfit for `optimizer_offload`.

| Role | Write pattern | Advisory free-space floor |
|---|---|---|
| `optimizer_offload`, `parameter_offload` | sustained, heavy | ~20 GB |
| `checkpoints` | periodic, large | ~20 GB |
| `scratch` | sustained | ~10 GB |
| `model_cache` | bursty (downloads) | ~30 GB |
| `dataset_cache` | bursty | ~5 GB |
| `artifacts` | write-once | ~10 GB |
| `os`, `source_repo`, `archive`, `logs` | light | 1–5 GB |

The free-space floors are **advisory heuristics** — a floor to catch a nearly-full disk, not a precise
per-run requirement (the planner refines with a real VRAM/offload estimate).

## The suitability verdict

For each `(role, path)` the guardrail returns `suitable` / `marginal` / `unsuitable` / `unknown` with
human-readable reasons. A single **unsuitable** reason (data-loss or thrash-to-a-halt risk) makes the
whole verdict unsuitable; otherwise a **marginal** reason (works but degraded) wins over suitable.
When the device can't be characterized the verdict is **unknown** — never a false `suitable`.

| Condition | High-write roles* | Offload roles |
|---|---|---|
| Cloud-sync folder | **unsuitable** — a sync client re-uploads every write | |
| Inside the source repository | **unsuitable** — generated state must not pollute source | |
| USB / removable device | **unsuitable** — can't sustain the write traffic | |
| Network mount | **unsuitable** — latency/reliability unfit | |
| Rotational disk (HDD) | | **marginal** — I/O-bound; prefer internal NVMe |
| Free space below the role floor | **unsuitable** | **unsuitable** |
| Internal NVMe with headroom | **suitable** | **suitable** |

\* high-write = `checkpoints`, `scratch`, `optimizer_offload`, `parameter_offload`.

## CLI

```
corpus-studio platform-storage                                   # characterize all devices
corpus-studio platform-storage --path /mnt/nvme/offload --role optimizer_offload
corpus-studio platform-storage --path ./ck                       # assess across offload + checkpoint roles
corpus-studio platform-storage --json --out ./profile            # StorageProfile.json
```

Example (real Windows host):

```
Platform storage
  devices:
    C:\              NTFS     unknown     free 533.7 GB / 1999.3 GB
    F:\              NTFS     unknown     free 1525.8 GB / 2000.4 GB
  assessments:
    optimizer_offload    UNSUITABLE - inside the source repository - generated run state must not pollute source
```

## Recommended Linux topology (a recommendation, not a requirement)

For a dedicated Linux training box, a sound layout is:

- **OS + applications** on a normal SSD;
- **source repositories** on OS or development storage;
- **internal PCIe NVMe** for the active model cache, checkpoints, scratch, and offload;
- **larger / slower storage** for archives and completed artifacts.

Corpus Studio never *enforces* this — it detects what you have and tells you, per role, whether a
chosen path is safe.

## Where this fits

`StorageProfile` is a **standalone control-plane contract** (not folded into `EnvironmentProfile`, so
it never perturbs the `environment_signature`). It is the input the run planner needs to assign
offload/checkpoint/scratch paths safely — the prerequisite for the offload-planning work
(DeepSpeed/FSDP/NVMe offload) on the roadmap. See [`ROADMAP.md`](ROADMAP.md).
