# Environment Manager

The Environment Manager realizes the **three-layer dependency architecture**
([`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §2): a lightweight, always-installable control
plane; opt-in capability profiles; and **isolated per-backend worker environments**. It is the gate
before heavy backends (DeepSpeed / FSDP / Unsloth / multimodal) can be added honestly — those pin
conflicting `torch` / CUDA / `xformers` builds and cannot coexist in one Python environment.

> **Status:** this slice ships the **substrate** — the recipe registry + the install *preview*
> resolver. It creates nothing and mutates nothing. Actually building a venv and running the
> installers (the side-effectful half — `env-create` / health / drift / lock) is the next slice, which
> needs real-machine verification.

## The three layers

| Layer | What it is | Where it installs |
|---|---|---|
| **control plane** | the dependency-light CorpusStudio core — no CUDA, no ML framework | the base interpreter |
| **capability** | opt-in feature stacks (tokenization, columnar data, …) with graceful fallback | **into the control-plane interpreter** (augments the core process) |
| **backend_worker** | isolated per-framework training runtimes | its **own** venv (`root_path` is the isolation boundary) |

Opening CorpusStudio requires only the control plane. A capability profile adds capability to the core
process. A backend worker never shares an environment with another backend — so one backend can't
corrupt another's runtime.

## Recipes

An `EnvironmentRecipe` is a **declaration** of what to install — not the act of installing. The
built-in recipes are grounded in the engine's real optional extras:

| Recipe | Layer | Deps | Verification |
|---|---|---|---|
| `control-plane` | control_plane | pydantic, typer | hardware_verified |
| `capability-tokenization` | capability | tiktoken | hardware_verified |
| `capability-model-tokenizer` | capability | tokenizers | hardware_verified |
| `capability-data` | capability | pyarrow | hardware_verified |
| `backend-corpus-studio` | backend_worker | torch, transformers, peft, trl, accelerate, datasets, bitsandbytes | hardware_verified (ran on a real RTX 5070) |
| `backend-unsloth` | backend_worker | unsloth | **declared** (happy path not verified on our Blackwell host) |

`verification` is the recipe-level honesty tier — **`declared` means we can render the install plan but
have not built + probed it.** Heavier backends (DeepSpeed / FSDP / Axolotl / LLaMA-Factory / MoE)
arrive with their own backend slices, so a recipe is never claimed before it can be built and probed.

## Environment states — "installed" ≠ "supported"

```
NOT_INSTALLED → INSTALLING → INSTALLED_UNCHECKED → IMPORTABLE →
DEPENDENCY_PROBE_PASSED → FUNCTIONAL_PROBE_PASSED → HARDWARE_VERIFIED
                                                    ↘ DEGRADED / INCOMPATIBLE / DRIFTED / BROKEN
```

A package importing (`IMPORTABLE`) is not proof a kernel runs (`FUNCTIONAL_PROBE_PASSED`), which is not
proof the hardware supports it (`HARDWARE_VERIFIED`). Only `HARDWARE_VERIFIED` earns "supported".

## Install preview (the resolver)

`resolve_dependencies(recipe, os, accelerator_tag, python_version)` renders the exact install plan
**without installing anything** — for explicit confirmation first. It is:

- **CUDA-aware** — picks the PyTorch wheel index by the host's accelerator (a Blackwell 5070 → the
  `cu128` index); installs torch from its own index so the rest still resolve from PyPI.
- **argv-structured, never a shell string** — each `InstallStep.argv` is a list, executed without a
  shell, so an untrusted package/index name can't inject a command (mirrors the no-shell
  trainer-launch invariant). Environment markers like `; platform_system != 'Darwin'` stay safely
  inside a single argv token.
- **layer-aware** — a backend recipe creates its own venv; a capability recipe installs into the
  control-plane interpreter.
- **honest about feasibility** — `resolvable = False` (with reasons) when the host can't satisfy the
  recipe: Python floor unmet, unsupported OS, or a CUDA-required recipe on a CUDA-less host. Warnings
  are non-blocking caveats (CPU torch, macOS bitsandbytes skip, native-build compiler need).

## CLI

```
corpus-studio env-recipes [--layer control_plane|capability|backend_worker] [--json]
corpus-studio env-plan <recipe-id> [--accelerator cu128|cpu|…] [--python 3.12] [--json]
```

Example on a real RTX 5070 (Blackwell) host — the accelerator is detected, not assumed:

```
Install plan: backend-corpus-studio  [backend_worker]
  host: windows  |  accelerator: cu128  |  python 3.12.10
  resolvable: True
  estimated download: 2.85 GB  |  on disk: 6.55 GB
  steps:
    [create_venv] <BASE_PYTHON> -m venv <ENV_ROOT>
    [upgrade_pip] <ENV_ROOT>\Scripts\python.exe -m pip install --upgrade pip
    [install] <ENV_ROOT>\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu128 torch>=2.1
    [install] <ENV_ROOT>\Scripts\python.exe -m pip install transformers>=4.44 peft>=0.11 …
```

The `<BASE_PYTHON>` / `<ENV_ROOT>` placeholders are substituted with real paths when a future
`env-create` acts on the plan.

## Contracts

`EnvironmentRecipe`, `DependencyResolution` (+ `InstallStep`), `EnvironmentLock`,
`EnvironmentDescriptor`, `EnvironmentHealthReport` — all versioned, exported as language-neutral JSON
Schema under [`contracts/`](contracts/). `RunPlan` will reference an environment by hash in a later
slice so a result is always tied to the exact environment that produced it.

## Deferred to the next slice

Environment **creation** and management — discover compatible Python runtimes; create the isolated
venv; run the bounded argv installers with explicit confirmation; record the exact `EnvironmentLock`
(package + source + hash); import / functional / hardware probes → `EnvironmentHealthReport`; drift
detection; repair/recreate; associate the environment hash with each `RunPlan`. Those are
side-effectful and need real-machine verification, so they land as their own vertical slice.
