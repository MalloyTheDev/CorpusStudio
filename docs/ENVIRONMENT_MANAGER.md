# Environment Manager

The Environment Manager implements the three-layer dependency architecture described in
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md): a lightweight control plane, opt-in capability
packs, and isolated backend-worker environments. Heavy frameworks do not share the control-plane
interpreter or each other's dependency graph.

> **Implemented status:** the create-to-remove lifecycle supports the legacy
> `backend-corpus-studio` rollback recipe, the exact-pinned math
> `backend-corpus-studio-readiness-v2` recipe, and the exact-pinned flash
> `backend-corpus-studio-readiness-flash-v1` recipe. Flash readiness is Linux-only and seals only
> after the complete forced-flash QLoRA tuple (`cuda_qlora_sdpa_flash_execution`) with math and
> mem-efficient SDPA disabled; the tuple uses CUDA bf16 autocast so attention dtypes match real
> TRL/PEFT QLoRA training (float32 residual after k-bit prep is not accepted as flash proof).
> Declaring the recipe or generating a sealed plan does not prove the environment. The lifecycle is
> covered in default CI by fake installers and CPU-only probes. On the current native-Linux host,
> the managed `backend-corpus-studio`, `backend-corpus-studio-readiness-v2`, and
> `backend-corpus-studio-readiness-flash-v1` environments have preserved `HARDWARE_VERIFIED` evidence
> for their respective exact probe tuples (legacy minimal hardware probe; readiness-v2 complete math
> QLoRA tuple; readiness-flash-v1 complete forced-flash QLoRA tuple with bf16 autocast). Manager 1.2
> preserves the math rollback identity but requires replacement of the manager-1.1 flash instance
> before a new health claim. Sealed flash-v1 digests are recorded in [`HOST_STATE.md`](HOST_STATE.md).
> Rebuilding remains an
> explicit network-using operation. These environment results do not verify a real 7B workload,
> offload, or full-sequence flash training.

## Dependency layers

| Layer | Purpose | Installation boundary |
|---|---|---|
| `control_plane` | contracts, projects, policy, planning, registries, CLI/UI communication | base CorpusStudio interpreter |
| `capability` | optional stable in-process features such as exact tokenization or Parquet | control-plane interpreter |
| `backend_worker` | torch/CUDA/framework stacks that can conflict | one owned venv per backend environment |

Only the three CorpusStudio worker recipes (legacy, readiness-v2 math, and readiness-flash-v1) have
side-effectful creation implementations. DeepSpeed, FSDP, Axolotl, LLaMA-Factory, Unsloth, and MoE
runtimes are not added here.

## Evidence levels

Environment state is deliberately monotonic until a probe fails or drift is found:

```text
NOT_INSTALLED -> INSTALLING -> INSTALLED_UNCHECKED -> IMPORTABLE ->
DEPENDENCY_PROBE_PASSED -> FUNCTIONAL_PROBE_PASSED -> HARDWARE_VERIFIED
                                                   \-> DEGRADED / INCOMPATIBLE / DRIFTED / BROKEN
```

`INSTALLED_UNCHECKED` is not a support claim. CPU forward/backward and checkpoint reload earn
`FUNCTIONAL_PROBE_PASSED`; they do not earn GPU support. The legacy recipe retains its existing
minimal hardware-probe meaning. Readiness-v2 additionally requires one complete probe to prove BF16,
NF4 with double quantization, math-only SDPA toggles, exact CUDA placement, QLoRA adapter insertion,
finite forward loss, backward, AdamW update, and adapter safetensors reload as one tuple. Flash
readiness-v1 is a separate complete probe identity: the same QLoRA tuple forced onto
`SDPBackend.FLASH_ATTENTION` with flash enabled and math/mem-efficient disabled, with no automatic
dispatch fallback. Independent passing probes cannot be unioned into either complete support claim.
On native-Windows Blackwell, the known-deadlocking fused SDPA path is refused for the standalone
flash probe; readiness flash environments are Linux-only. `HARDWARE_VERIFIED` is evidence for that
exact environment-level tuple, not backend-wide, workload, 7B, offload, external `flash_attention_2`,
distributed, or MoE support.

## Runtime discovery and plan review

`env-runtimes` probes more than the current interpreter. Each `PythonRuntime` records its executable,
Python version, implementation, architecture, platform, whether it is already a venv, whether the
stdlib `venv` module is available, and compatibility reasons.

`env-plan` is non-mutating. For an actionable backend plan it resolves:

- the selected base interpreter and deterministic environment root;
- exact argv arrays, working directories, a small explicit non-secret environment, timeouts, expected
  outputs, and whether each step uses the network;
- explicit PyPI and accelerator-specific PyTorch indexes (`cu128` for the Blackwell reference host);
- for the legacy recipe, the reviewed local CorpusStudio worker source; for readiness recipes, a
  concrete `corpus-studio-engine` wheel whose size, METADATA digest, and byte hash are part of the plan;
- estimated download and installed sizes;
- a recipe digest and `resolution_hash` over the concrete plan.

Pip runs with `--isolated`, `--no-input`, an explicit `PIP_CONFIG_FILE` null device, and explicit
indexes. Host pip configuration cannot add a source, and the manager never silently retries from a
different source.

Example review flow on the current native-Linux host:

```bash
cd /mnt/training-nvme/repos/CorpusStudio
engine/.venv/bin/corpus-studio env-runtimes --recipe backend-corpus-studio
engine/.venv/bin/corpus-studio env-plan backend-corpus-studio \
  --env-id backend-corpus-studio \
  --runtime /usr/bin/python3 \
  --accelerator cu128
```

Readiness recipes require an already-built local wheel and can write the canonical plan without
creating an environment. Math readiness-v2 remains the safety/rollback baseline; flash readiness is a
separate environment id and must not mutate it:

```bash
engine/.venv/bin/corpus-studio env-plan backend-corpus-studio-readiness-v2 \
  --env-id backend-corpus-studio-readiness-v2 \
  --runtime /usr/bin/python3 \
  --accelerator cu128 \
  --worker-wheel /mnt/training-nvme/artifacts/corpusstudio-worker/<commit>/<wheel>.whl \
  --manager-root /mnt/training-nvme/corpusstudio/xdg-data/corpusstudio/environment-manager \
  --out /mnt/training-nvme/artifacts/corpusstudio-worker/<commit>/DependencyResolution.json

engine/.venv/bin/corpus-studio env-plan backend-corpus-studio-readiness-flash-v1 \
  --env-id backend-corpus-studio-readiness-flash-v1 \
  --runtime /usr/bin/python3 \
  --accelerator cu128 \
  --worker-wheel /mnt/training-nvme/artifacts/corpusstudio-worker/<commit>/<wheel>.whl \
  --manager-root /mnt/training-nvme/corpusstudio/xdg-data/corpusstudio/environment-manager \
  --out /mnt/training-nvme/artifacts/corpusstudio-worker/<commit>/DependencyResolution.flash-v1.json
```

The plan prints its exact `resolution hash`. Creation requires that same value and the same planning
options:

```bash
engine/.venv/bin/corpus-studio env-create backend-corpus-studio \
  --env-id backend-corpus-studio \
  --runtime /usr/bin/python3 \
  --accelerator cu128 \
  --confirm <resolution-hash>
```

This second command performs network package installation. Do not run it until the displayed indexes,
size, target path, environment, and argv have been reviewed. A changed recipe, runtime, root, command,
environment variable, or manager version changes the hash and invalidates confirmation.

## Durable state and logs

The default user-owned root is `%LOCALAPPDATA%\CorpusStudio\environment-manager` on Windows and the
XDG data directory on Linux/macOS. It has two separate areas:

```text
environment-manager/
  .locks/                        # persistent manager/per-environment lock files; never delete
  environments/<env-id>/          # the venv; contains .corpusstudio-owner.json
  registry/<env-id>/
    EnvironmentDescriptor.json
    EnvironmentHealthReport.json
    installations/<attempt-id>.json
    locks/<lock-id>.json
    logs/<attempt-id>/*.stdout.log|*.stderr.log
```

Registry writes use temp-file plus atomic replacement. A failed, timed-out, or cancelled command keeps
its descriptor, command journal, logs, structured `FailureRecord`, and partial owned environment as
`BROKEN`. A completed command sequence whose required probe fails may instead be honestly `DEGRADED`
or `INCOMPATIBLE`; it still has no lock and explicitly requires recreation. The manager does not hide
either case with a different source or an automatic destructive retry.

Each installation command record includes argv, cwd, explicit environment, timeout, expected outputs,
timestamps, exit code, stdout/stderr paths, native-build evidence, and failure details.

## Lock ordering, probes, and drift

The manager does not use a pre-install hash as a final environment lock. Its deterministic order is:

1. resolve the immutable recipe and canonical plan;
2. verify the echoed plan hash before environment mutation;
3. create the owned environment and execute the reviewed argv;
4. capture sanitized pip reports and a pre-probe installed-file inventory;
5. run import, dependency, CPU, GPU, and recipe-required complete-tuple probes;
6. capture a post-probe inventory and refuse sealing if the environment changed;
7. seal the `EnvironmentLock` only after all required evidence passes;
8. recompute live package, worker-artifact, hardware, and lock drift against that sealed state.

Lifecycle mutation is protected across processes by a bounded manager lock followed by an exclusive
per-environment lease. Create, recreate, and removal hold both locks for their complete transaction;
health/capability checks and managed planning hold the environment lease for a consistent evidence
snapshot. `platform-run` keeps the same lease from its pre-dispatch health check until the worker has
terminated, so a concurrent remove or recreate cannot invalidate the interpreter beneath a live run.
Lock acquisition fails with an explicit `TIMEOUT`; stale lock files are harmless bookkeeping because
the operating-system lock is released on process exit, and the files must not be deleted to break
contention.

A failed complete probe leaves `lock_ref` absent. It may produce an honest `INCOMPATIBLE` or
`DEGRADED` installation record, marked `retry_requires_recreate=true`, but never a final lock or
`HARDWARE_VERIFIED` state.

The sealed `EnvironmentLock` contains:

- Python executable/version/implementation/platform/architecture;
- normalized package names, exact versions, sanitized index/direct/VCS evidence, artifact filenames
  and hashes where pip can prove them, plus an explicit reason when the source remains unknown;
- installed `RECORD` metadata hashes, verification of every SHA-256-bearing installed file, a
  manager-computed tree digest over every regular file named by `RECORD` (including generated,
  unhashed bytecode), and dependency metadata. Every site-package file must be owned by one such
  record; unrecorded files, duplicate normalized distributions, and symlinks fail closed. An entry
  must resolve inside the managed environment and may not use an absolute or escaping path;
- torch build, CUDA runtime, and compute capability;
- recipe and resolution identities, selected indexes, exact worker wheel identity, complete-probe
  evidence, manager version, timestamp, and a canonical lock digest.

Pip report entries fail closed when malformed or when normalized distribution names collide. Artifact
hosts are matched exactly to configured index hosts (with the explicit PyPI file-host mapping); URL
substrings are never source evidence. URLs are sanitized before durable storage: credentials, query
parameters, fragments, signed URLs, and private-index secrets are not retained. Allocator memory, `nvidia-smi`
current-process memory, and host RSS are labeled as separate scopes. They are measurements of the
bounded probe, not model-fit or parameter-residency claims.

Readiness wheel inspection is bounded and fail-closed before mutation: the filename, root dist-info
directory, unique METADATA identity, unique RECORD, archive members, per-member SHA-256/size, and
expanded size must agree. Pip's observed artifact hash is compared with that reviewed identity before
any newly installed interpreter process is started. Installed inventories run with `-I -S`, use
bounded metadata reads and streaming file hashes without processing `.pth` files, reject unrecorded
site-package files, and compare every immutable worker payload member with the reviewed wheel. Only
after the parent validates that non-executable evidence does a second isolated process import torch.
Health and capability probes are bracketed by fresh inventories; a probe-side package/file mutation
cannot be returned as healthy.

Manager 1.2 preserves the sealed 1.1 lock digests. In particular, the pre-autocast-field
readiness-v2 math evidence remains a narrow, untouched rollback identity during health checks. New
math or flash creations must emit the stronger measured configuration, including BF16 forward
autocast, forced kernel/toggles, bounded probe shape, and an adapter-state equality check; the legacy
exception is never accepted for flash or for a new creation.
The current manager-1.1 flash lock predates the adapter-state equality observation. Its lock and
probe-evidence digests remain valid historical evidence, but manager 1.2 intentionally does not treat
it as a rollback exception; replacement is required before it can receive a manager-1.2 health claim.

Probe categories remain separate:

- **import:** CorpusStudio worker, torch, transformers, PEFT, TRL, Accelerate, Datasets, bitsandbytes;
- **dependency:** `python -m pip check`;
- **functional:** tiny CPU forward/backward plus checkpoint save/reload;
- **hardware:** CUDA availability/allocation, compute capability, BF16 signal, bitsandbytes 4-bit
  construction, minimal GPU forward/backward, optional-kernel flags, and math SDPA execution.

`env-probe` snapshots before imports and again after probes and detects missing roots/interpreters/locks,
normalized name collisions, package addition/removal/version/installed-RECORD or installed-file-tree
changes, observable direct/VCS/index source changes, worker-wheel changes, recipe drift, lock tampering,
broken imports, functional failures, and CUDA/compute-capability changes. Index provenance is
install-time evidence: reinstalling identical
bytes from a different index without PEP 610 metadata cannot be reconstructed from installed files
alone, so the health report does not invent that claim.

```bash
cd /mnt/training-nvme/repos/CorpusStudio
engine/.venv/bin/corpus-studio env-status [<env-id>] [--refresh] [--json]
engine/.venv/bin/corpus-studio env-probe <env-id> [--json]
engine/.venv/bin/corpus-studio env-lock <env-id>
```

## Safe removal, immutable identities, and failed-attempt recovery

Removal requires both path containment under the manager's `environments` directory and a matching
ownership marker. It also requires the exact environment ID as confirmation. Registry evidence and
the logical environment identity are retained; a later `env-create` cannot silently reuse that ID.

```bash
cd /mnt/training-nvme/repos/CorpusStudio
engine/.venv/bin/corpus-studio env-remove backend-corpus-studio \
  --confirm backend-corpus-studio
```

`env-recreate` is recovery for an **unsealed failed attempt** only. It is intentionally two
confirmations: the new plan hash and the exact old environment ID. Once any environment has a sealed
lock, in-place recreation is refused even after removal. Create its replacement under a new ID, check
the new lock, then explicitly move callers to that identity; this preserves rollback and historical
evidence instead of overwriting its meaning.

```bash
cd /mnt/training-nvme/repos/CorpusStudio
engine/.venv/bin/corpus-studio env-recreate backend-corpus-studio-failed-attempt \
  --env-id backend-corpus-studio-failed-attempt \
  --confirm <new-resolution-hash> \
  --confirm-remove backend-corpus-studio-failed-attempt
```

An arbitrary directory, an unmarked directory, a marker owned by another manager root, or a path that
escapes containment is refused.

## RunPlan association

`RunPlan.environment_ref` can now carry the environment ID plus the immutable lock hash. It never
depends on the mutable venv path. When `--environment` is selected, `platform-plan` builds its
EnvironmentProfile and CapabilityReport inside that managed interpreter, so a lightweight control
plane does not need the training stack installed merely to plan for its isolated worker.

```bash
cd /mnt/training-nvme/repos/CorpusStudio
engine/.venv/bin/corpus-studio platform-plan ... --environment backend-corpus-studio
engine/.venv/bin/corpus-studio platform-run RunPlan.json --subprocess
```

Before dispatch or resume, `platform-run` performs live health/drift checks and verifies the plan's
environment ID, lock hash, and functional state. A managed plan must use `--subprocess`; the worker is
launched with the managed interpreter so training cannot silently fall back into the control plane.

## Contracts and clients

The lifecycle uses the root contracts `PythonRuntime`, `EnvironmentRecipe`, `DependencyResolution`,
`EnvironmentInstallation`, `EnvironmentLock`, `EnvironmentDescriptor`, and
`EnvironmentHealthReport`, with explicit nested `InstalledEnvironmentEvidence` records for pre- and
post-probe inventories. Pydantic remains the source of truth; deterministic JSON Schemas under
[`contracts/`](contracts/) and committed TypeScript types under `apps/web/src/contracts/` are generated
from it. CI regenerates and diffs both layers.

## Verification boundary and deferred work

Default CI proves command construction, confirmation seals, path containment, ownership, atomic
records, timeout/cancellation, failure recovery, lock generation, CPU probes, drift, bounded
cross-process lifecycle exclusion, sealed-identity preservation, RunPlan pinning, and
managed-interpreter dispatch using fakes and temp directories.

Current-host evidence covers three distinct managed environments: the legacy minimal hardware tuple,
the readiness-v2 complete math QLoRA tuple, and the readiness-flash-v1 tiny forced-flash QLoRA tuple.
The first real 0.5B flash smoke failed placement verification before adapter insertion and completed
zero optimizer steps; a separate placement-only diagnostic observed all loaded parameters and buffers
on `cuda:0`. It does not claim a complete 7B training run, a real flash optimizer step, sequence-4096
stability, sustained throughput, production checkpoint behavior, DeepSpeed/FSDP, CPU/NVMe parameter
or optimizer offload, MoE execution, or resource-elastic expert paging.

Also not claimed by this slice:

- in-place package repair (`env-recreate` only replaces an unsealed failed attempt; sealed
  replacements use a new ID);
- side-effectful creation for capability packs or any backend other than the three CorpusStudio worker
  recipes;
- container, conda, `uv`, or remote environment providers.
