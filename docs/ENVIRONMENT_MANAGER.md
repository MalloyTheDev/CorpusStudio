# Environment Manager

The Environment Manager implements the three-layer dependency architecture described in
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md): a lightweight control plane, opt-in capability
packs, and isolated backend-worker environments. Heavy frameworks do not share the control-plane
interpreter or each other's dependency graph.

> **Implemented status:** the complete create-to-remove lifecycle is implemented for the
> `backend-corpus-studio` reference backend. Other recipes can still be inspected, but their presence
> does not make their creation or hardware path supported. The lifecycle is covered in default CI by
> fake installers and CPU-only probes. Building the new managed CUDA environment on a real GPU remains
> an explicit, network-using verification operation; it is not run automatically.

## Dependency layers

| Layer | Purpose | Installation boundary |
|---|---|---|
| `control_plane` | contracts, projects, policy, planning, registries, CLI/UI communication | base CorpusStudio interpreter |
| `capability` | optional stable in-process features such as exact tokenization or Parquet | control-plane interpreter |
| `backend_worker` | torch/CUDA/framework stacks that can conflict | one owned venv per backend environment |

Only `backend-corpus-studio` has a side-effectful creation implementation in this slice. DeepSpeed,
FSDP, Axolotl, LLaMA-Factory, and MoE runtimes are not added here.

## Evidence levels

Environment state is deliberately monotonic until a probe fails or drift is found:

```text
NOT_INSTALLED -> INSTALLING -> INSTALLED_UNCHECKED -> IMPORTABLE ->
DEPENDENCY_PROBE_PASSED -> FUNCTIONAL_PROBE_PASSED -> HARDWARE_VERIFIED
                                                   \-> DEGRADED / INCOMPATIBLE / DRIFTED / BROKEN
```

`INSTALLED_UNCHECKED` is not a support claim. CPU forward/backward and checkpoint reload earn
`FUNCTIONAL_PROBE_PASSED`; they do not earn GPU support. `HARDWARE_VERIFIED` requires the managed
interpreter to prove CUDA allocation, 4-bit layer construction, a minimal GPU forward/backward, and
the safe math-attention path. On native-Windows Blackwell, the probe never executes the known
deadlocking fused SDPA path.

## Runtime discovery and plan review

`env-runtimes` probes more than the current interpreter. Each `PythonRuntime` records its executable,
Python version, implementation, architecture, platform, whether it is already a venv, whether the
stdlib `venv` module is available, and compatibility reasons.

`env-plan` is non-mutating. For an actionable backend plan it resolves:

- the selected base interpreter and deterministic environment root;
- exact argv arrays, working directories, a small explicit non-secret environment, timeouts, expected
  outputs, and whether each step uses the network;
- explicit PyPI and accelerator-specific PyTorch indexes (`cu128` for the Blackwell reference host);
- a non-editable install of the reviewed local CorpusStudio worker source into the isolated venv;
- estimated download and installed sizes;
- a recipe digest and `resolution_hash` over the concrete plan.

Pip runs with `--isolated`, `--no-input`, an explicit `PIP_CONFIG_FILE` null device, and explicit
indexes. Host pip configuration cannot add a source, and the manager never silently retries from a
different source.

Example review flow (PowerShell):

```powershell
corpus-studio env-runtimes --recipe backend-corpus-studio
corpus-studio env-plan backend-corpus-studio `
  --env-id backend-corpus-studio `
  --runtime C:\CorpusStudio\engine\.venv\Scripts\python.exe `
  --accelerator cu128
```

The plan prints its exact `resolution hash`. Creation requires that same value and the same planning
options:

```powershell
corpus-studio env-create backend-corpus-studio `
  --env-id backend-corpus-studio `
  --runtime C:\CorpusStudio\engine\.venv\Scripts\python.exe `
  --accelerator cu128 `
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
  environments/<env-id>/          # the venv; contains .corpusstudio-owner.json
  registry/<env-id>/
    EnvironmentDescriptor.json
    EnvironmentHealthReport.json
    installations/<attempt-id>.json
    locks/<lock-id>.json
    logs/<attempt-id>/*.stdout.log|*.stderr.log
```

Registry writes use temp-file plus atomic replacement. A failed, timed-out, or cancelled attempt keeps
its descriptor, command journal, logs, structured `FailureRecord`, and partial owned environment. It
is `BROKEN` and explicitly requires recreation; the manager does not hide the failure with a different
source or an automatic destructive retry.

Each installation command record includes argv, cwd, explicit environment, timeout, expected outputs,
timestamps, exit code, stdout/stderr paths, native-build evidence, and failure details.

## Lock, probes, and drift

After installation, the managed interpreter emits an `EnvironmentLock` containing:

- Python executable/version/implementation/platform/architecture;
- exact installed distribution versions;
- pip installer, direct URL and wheel/source identity where available;
- installed `RECORD` metadata hashes and dependency metadata;
- torch build, CUDA runtime, and compute capability;
- recipe digest, selected indexes, manager version, timestamp, and a canonical lock digest.

Probe categories remain separate:

- **import:** CorpusStudio worker, torch, transformers, PEFT, TRL, Accelerate, Datasets, bitsandbytes;
- **dependency:** `python -m pip check`;
- **functional:** tiny CPU forward/backward plus checkpoint save/reload;
- **hardware:** CUDA availability/allocation, compute capability, BF16 signal, bitsandbytes 4-bit
  construction, minimal GPU forward/backward, optional-kernel flags, and math SDPA execution.

`env-probe` re-snapshots the live environment and detects missing roots/interpreters/locks, package
addition/removal/version/hash changes, source changes, recipe drift, lock tampering, broken imports,
functional failures, and CUDA/compute-capability changes.

```powershell
corpus-studio env-status [<env-id>] [--refresh] [--json]
corpus-studio env-probe <env-id> [--json]
corpus-studio env-lock <env-id>
```

## Safe removal and recreation

Removal requires both path containment under the manager's `environments` directory and a matching
ownership marker. It also requires the exact environment ID as confirmation. Registry evidence is
retained.

```powershell
corpus-studio env-remove backend-corpus-studio --confirm backend-corpus-studio
```

Recreation is intentionally two confirmations: the new plan hash and the exact old environment ID.

```powershell
corpus-studio env-recreate backend-corpus-studio `
  --confirm <new-resolution-hash> `
  --confirm-remove backend-corpus-studio
```

An arbitrary directory, an unmarked directory, a marker owned by another manager root, or a path that
escapes containment is refused.

## RunPlan association

`RunPlan.environment_ref` can now carry the environment ID plus the immutable lock hash. It never
depends on the mutable venv path. When `--environment` is selected, `platform-plan` builds its
EnvironmentProfile and CapabilityReport inside that managed interpreter, so a lightweight control
plane does not need the training stack installed merely to plan for its isolated worker.

```powershell
corpus-studio platform-plan ... --environment backend-corpus-studio
corpus-studio platform-run RunPlan.json --subprocess
```

Before dispatch or resume, `platform-run` performs live health/drift checks and verifies the plan's
environment ID, lock hash, and functional state. A managed plan must use `--subprocess`; the worker is
launched with the managed interpreter so training cannot silently fall back into the control plane.

## Contracts and clients

The lifecycle uses the root contracts `PythonRuntime`, `EnvironmentRecipe`, `DependencyResolution`,
`EnvironmentInstallation`, `EnvironmentLock`, `EnvironmentDescriptor`, and
`EnvironmentHealthReport`. Pydantic remains the source of truth; deterministic JSON Schemas under
[`contracts/`](contracts/) and committed TypeScript types under `apps/web/src/contracts/` are generated
from it. CI regenerates and diffs both layers.

## Verification boundary and deferred work

Default CI proves command construction, confirmation seals, path containment, ownership, atomic
records, timeout/cancellation, failure recovery, lock generation, CPU probes, drift, safe
remove/recreate, RunPlan pinning, and managed-interpreter dispatch using fakes and temp directories.

Not claimed by this slice:

- a newly downloaded real CUDA environment or real-GPU result from this branch (network confirmation
  was not supplied);
- in-place package repair (recreate is the safe supported recovery path);
- side-effectful creation for capability packs or any backend other than `backend-corpus-studio`;
- container, conda, `uv`, or remote environment providers.
