# Host State — Native-Linux RTX 5070 Workstation

**Last verified:** 2026-07-14 (Environment Manager health report `checked_at`
2026-07-14T03:29:28Z; GPU + paths re-checked the same day).

This file records the *verified* runtime facts of the machine CorpusStudio currently runs
on. It supersedes the Windows `C:`/`F:` host descriptions in older docs for **"where you
are."** The prior native-Windows/WDDM (and separately labeled WSL) evidence is **preserved
as history** in [`HANDOFF.md`](../HANDOFF.md) and [`CURRENT_STATE.md`](CURRENT_STATE.md) —
it is not deleted or restated as Linux.

> **Verification level.** Every value below is read directly from the OS, `nvidia-smi`, and
> the Environment Manager registry / lock / health report on this host. What a
> `HARDWARE_VERIFIED` environment does and does **not** prove is spelled out under
> "Verification boundary" — do not read it as a training-workload result.

## Host

| Fact | Value |
|---|---|
| Platform | Linux 6.8.0-134-generic — Ubuntu 24.04.4 LTS, x86-64 |
| Repository (active runtime) | `/mnt/training-nvme/repos/CorpusStudio` |
| Engine control-plane venv | `/mnt/training-nvme/repos/CorpusStudio/engine/.venv` — CPython 3.12.3 (dependency-light core + `[dev]`, torch-free) |
| CLI entrypoint | `engine/.venv/bin/corpus-studio` (equivalently `.venv/bin/python -m corpus_studio.cli`) |
| Linux training filesystem | `/mnt/training-nvme` |
| Windows `C:` drive (mount) | `/mnt/windows-c` — read-write filesystem mount; history-only project policy; e.g. former `C:\CorpusStudio` → `/mnt/windows-c/CorpusStudio` |
| Windows Projects / `F:` drive (mount) | `/mnt/windows-f` — read-write filesystem mount; history-only project policy; e.g. former `F:\CorpusStudio` → `/mnt/windows-f/CorpusStudio` |

The active runtime is the native-Linux NVMe checkout under `/mnt/training-nvme/...`. The old
Windows `C:` and `F:` copies are still visible as read-write `/mnt/windows-c` and `/mnt/windows-f`
filesystem mounts. Read-write is an OS mount fact, not permission to use them as development roots:
they are stale fallbacks that will drift, so **do not work from or write to them.**

## GPU

| Fact | Value |
|---|---|
| Device | NVIDIA GeForce RTX 5070, 12227 MiB (~12 GB) |
| Architecture | Blackwell — compute capability 12.0 (sm_120) |
| Driver | 595.71.05 |
| CUDA (driver-reported) | 13.2 |

## Managed backend environment — `backend-corpus-studio`

The reference training-worker environment (the isolated Layer-3 backend of the three-layer
dependency model — see [`ENVIRONMENT_MANAGER.md`](ENVIRONMENT_MANAGER.md)) is built and
probed on this host.

| Fact | Value |
|---|---|
| Env id | `backend-corpus-studio` |
| Layer | `backend_worker` |
| Root | `/mnt/training-nvme/corpusstudio/xdg-data/corpusstudio/environment-manager/environments/backend-corpus-studio` |
| Manager root | `/mnt/training-nvme/corpusstudio/xdg-data/corpusstudio/environment-manager` |
| Managed interpreter | `<root>/bin/python` — CPython 3.12.3 |
| **State** | **`HARDWARE_VERIFIED`** |
| Drift detected | `false` |
| Recipe ref | `backend-corpus-studio` / `sha256:7fd0c05d…ca94c4` |
| Resolution ref | `resolution-d2c32667f525c17b84d9` / `sha256:d2c32667…55638b` |
| Installation journal | `install-d53b77b4cf9e44c99ab3` |
| Lock ref | `lock-dbc528f0167a2ec0ccfa` |
| Lock digest | `sha256:dbc528f0167a2ec0ccfa42d46ce86c9061be126e02aa557ae5ef5741788a8045` |
| Created / verified | 2026-07-14T03:25:51Z / 2026-07-14T03:29:15Z |
| Owner marker | `.corpusstudio-owner.json` (`corpus-studio-managed-environment-v1`, this manager root) |

**Probe results** (`EnvironmentHealthReport`, `checked_at` 2026-07-14T03:29:28Z):

| Probe | Outcome |
|---|---|
| `reference_backend_imports` | PASS |
| `pip_check` | PASS |
| `reference_backend_functional` | PASS |
| `reference_backend_hardware` | PASS |

**Pinned stack** (`EnvironmentLock` `lock-dbc528f0…`):

| Package | Version |
|---|---|
| torch | 2.11.0+cu128 (build `70d99e99…`) |
| CUDA runtime (wheel) | 12.8 |
| compute capability | 12.0 |
| transformers | 5.13.1 |
| peft | 0.19.1 |
| trl | 1.8.0 |
| bitsandbytes | 0.49.2 |
| accelerate | 1.14.0 |
| datasets | 5.0.0 |

(84 distributions are locked in total; the table lists the training-relevant ones.)

### Readiness-v2 status

`backend-corpus-studio-readiness-v2` is a separate exact-pinned recipe and replacement environment
plan. It does not replace, mutate, or reinterpret the environment above. As of this record, the
readiness-v2 environment has **not** been created or installed and therefore has no lock or health
state. Its plan binds a concrete `corpus-studio-engine` wheel and requires the complete QLoRA tuple
described in [`ENVIRONMENT_MANAGER.md`](ENVIRONMENT_MANAGER.md) before a lock can be sealed. Creating
it still requires separate explicit authorization.

## Verification boundary — what `HARDWARE_VERIFIED` does and does NOT prove

`HARDWARE_VERIFIED` is the **Environment Manager** evidence level, not a training-run result.
Per [`ENVIRONMENT_MANAGER.md`](ENVIRONMENT_MANAGER.md), the passing
`reference_backend_hardware` probe proves the managed interpreter can, on this GPU: allocate
CUDA memory, read compute capability, produce a BF16 signal, construct a bitsandbytes 4-bit
layer, run a **minimal** GPU forward/backward, and execute the safe **math** SDPA attention
path.

That is **real native-Linux GPU evidence** — the old "until the Linux NVMe is installed in the
final RTX 5070 machine, do not claim native-Linux" precondition is now satisfied *for the
environment probe*. It is **not** proof of any of the following, which remain unverified and
must not be claimed from this state alone:

- **Full-sequence 7B training success** — the probe is a minimal kernel check, not a real
  workload. (The `~10.8 GB @ seq 1024` / `~13.8 GB @ 2048` VRAM ceiling on record was measured
  on native-Windows/WDDM; native-Linux *workload* VRAM behavior is not yet measured.)
- **DeepSpeed / FSDP / CPU / NVMe offload** — no offload backend is implemented; only the dense
  `backend-corpus-studio` reference exists.
- **Real offload fit, PCIe/NVMe throughput, sustained-write endurance** — the NVMe has not been
  benchmarked (`platform-storage` is non-destructive and reads no SMART data).
- **Bare-Linux FlashAttention for the real workload** — the probe verified the *math* path; the
  fused/flash path is not claimed here. The native-Windows/WDDM deadlock reason for forcing math
  does not apply on this Linux host, but that is the *absence of a blocker*, not a positive
  flash-attention result.
- **MoE runtime capability** — static inspection only (Phase 8); no MoE execution.

"Installed ≠ supported" and "a completed step ≠ proven fit" both still hold: a passing
environment hardware probe is the *precondition* for the GPU-workload bring-up steps in
[`HANDOFF.md`](../HANDOFF.md) §7, not their completion.

## Re-verifying this state

From the engine control-plane venv:

```bash
cd /mnt/training-nvme/repos/CorpusStudio/engine
.venv/bin/corpus-studio env-status backend-corpus-studio --refresh --json
.venv/bin/corpus-studio env-probe  backend-corpus-studio --json
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
```

A changed recipe, runtime, installed package, source, or CUDA / compute-capability signal
flips the environment out of `HARDWARE_VERIFIED` into a `DRIFTED` / `DEGRADED` / `BROKEN`
state; re-probe before trusting this file.
