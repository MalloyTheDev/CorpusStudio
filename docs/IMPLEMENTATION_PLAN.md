# Implementation Plan — training-and-execution frontier

This is the forward-looking engineering plan for the **training-and-execution frontier** of CorpusStudio -
the local-first, end-to-end AI development ecosystem and IDE with seven co-equal product areas (see
[`PRODUCT_AREAS.md`](PRODUCT_AREAS.md)). It evolves the current Python control plane (a mature
dataset-to-model-to-evaluation control plane with a hardware-aware run lifecycle) toward the full
lifecycle. The **target platform architecture — a Rust authoritative core plus isolated Python ML
workers — is a separate tracked epic**; this plan covers the Python-side lifecycle/training contracts,
which are the highest-priority engineering gate (not the product's whole purpose). It is **planning**, not
a record of shipped work — for what actually works today, [`CURRENT_STATE.md`](CURRENT_STATE.md) is
the source of truth. Nothing here is claimed as implemented unless it links to a shipped feature.

## 1. Grounded architecture audit (2026-07-14)

The **run lifecycle is largely built.** The `engine/corpus_studio/platform/` package is a real,
torch-free contract substrate + lifecycle, not a plan on paper. Hardware evidence has two distinct,
non-interchangeable levels:

- **28 root versioned contracts** (`platform/contracts.py`) → deterministic language-neutral JSON
  Schema (`docs/contracts/`) → generated TypeScript client types.
- **profile → plan (hash-sealed) → predict-fit → run → measure-fit → account-for-artifacts**, with a
  backend-manifest registry (`corpus_studio`, `unsloth`), a watchdog (measured fit + spill/stall), and
  a supervised subprocess worker that can **kill a hung run**. The pre-Phase-9B lifecycle ran on a real
  RTX 5070 under native Windows/WDDM. Separately, the current native-Linux host's managed
  `backend-corpus-studio` environment passed its minimal hardware probe. The new effective-execution
  path on a real workload, full-sequence 7B behavior, and every real offload path remain unverified.

Against the full-platform vision, the frontier is the **input side**. Ranked gaps:

| Gap | Status | Notes |
|---|---|---|
| **Storage & offload profiling** | ✅ **done** (this slice) | `StorageProfile` + role suitability — see [`HARDWARE_STORAGE_PROFILE.md`](HARDWARE_STORAGE_PROFILE.md) |
| **Environment Manager + dependency profiles** | ✅ **reference lifecycle shipped** | §2 below — sealed creation/lock/probe/drift/recreate for `backend-corpus-studio`; other backends remain separate future slices |
| Model & Tokenizer Lab (`ModelDescriptor`/`TokenizerDescriptor`) | ✅ **contract + static-inspection foundation shipped** | offline identity/inventory/trust/tokenizer compatibility plus allowlisted static MoE topology evidence; loading, editing/training, and runtime support proof remain future work — see [`MODEL_TOKENIZER_CONTRACTS.md`](MODEL_TOKENIZER_CONTRACTS.md) |
| `TrainingObjective` as a distinct registry | ✅ **contract + registry foundation shipped** | 29 sealed objective definitions + conservative dataset/model/backend evidence checker; RunPlan integration and trainer implementation remain future work — see [`TRAINING_OBJECTIVES.md`](TRAINING_OBJECTIVES.md) |
| MoE-safe parameter accounting | ✅ **contract + static/reconciliation foundation shipped** | Sealed reports, bounded descriptor/safetensors evidence, typed worker-observation seam, explicit gaps/conflicts, and lifecycle refs; backend runtime instrumentation remains future work — see [`PARAMETER_ACCOUNTING.md`](PARAMETER_ACCOUNTING.md) |
| Physical `RunPlan` planning | ✅ **contract + planner foundation shipped** | Concrete resources/placements/offload/ranks/groups, sealed evidence refs, static + probed capability gates, and legacy-hash compatibility; built-in workers still refuse non-trivial physical execution — see [`RUN_PLAN_PHYSICAL_EXECUTION.md`](RUN_PLAN_PHYSICAL_EXECUTION.md) |
| Full versioned `TraceRecord` | ✅ **contract + workflow foundation shipped** | sealed source/context/segments/producer/validation/review, migration/generation/review/training gates, built-in trace draft schema; graphical Trace Studio, tool execution, and tool/process trainers remain future work — see [`TRACE_RECORDS.md`](TRACE_RECORDS.md) |
| Static MoE model inspection | ✅ **allowlisted config foundation shipped** | hash-pinned Mixtral/Qwen2-MoE/DeepSeek V2/V3 structure and expert-instance counts; runtime, backend, parameter-residency, and fit proof remain future work — see [`MOE_MODEL_INSPECTION.md`](MOE_MODEL_INSPECTION.md) |
| Isolated backend worker boundary | ✅ **protocol 2.0 + conformance foundation shipped** | backend-manifest and environment/lock identity before dispatch; typed direction/body, correlation/order/event/terminal lineage enforcement; fake workers only, not a new training backend — see [`BACKEND_WORKER_PROTOCOL.md`](BACKEND_WORKER_PROTOCOL.md) |
| Dataset mixtures + transformation graph | partial | `DatasetManifest.lineage` is the seed |
| Offload execution (DeepSpeed/FSDP/NVMe) | planned | the physical contract is shipped; each mechanism still needs an isolated backend plus functional and hardware proof |

## 2. Dependency & runtime architecture (reference foundation shipped)

**"Dependency-light" describes the CONTROL PLANE, not the whole product.** A full AI platform needs
complete data, model, training, evaluation, multimodal, conversion, serving, storage, and
observability stacks. The mistake to avoid is forcing them all into one Python environment. The
architecture has **three dependency layers**:

### Layer 1 — Control plane (stays lightweight + highly reliable)

Contracts, projects, policies, planning, environment management, run supervision, artifact lineage,
UI communication. **Opening Corpus Studio must never require CUDA, DeepSpeed, FlashAttention, or any
ML framework.** This is the existing dependency-light engine core — the property is preserved, not
weakened.

### Layer 2 — Capability profiles (installable feature stacks)

Opt-in extras that add real capability to the *core process* for a feature area, each with graceful
fallback when absent. Today the engine already does this for a few (`tokenizer`, `model-tokenizer`,
`parquet`, `train`); the plan formalizes the full set:

`core-extended` · `data` · `documents` · `tokenization` · `model` · `evaluation` · `traces` ·
`multimodal` · `observability` · `conversion` · `serving`

### Layer 3 — Backend worker environments (isolated, per-framework)

Heavy training/serving frameworks get **separate runtime environments** — they cannot coexist in one
env (Unsloth, Axolotl, DeepSpeed, FSDP each pin conflicting torch/transformers/xformers/CUDA builds).
The main process talks to them through the **versioned `WorkerMessage` protocol**, never by importing
them. Protocol 2.0 requires the worker to prove its exact static backend-manifest digest and
environment/lock ref before the core sends a RunPlan, then enforces the run stream fail-closed:

`backend-corpus-studio` · `backend-trl` · `backend-deepspeed` · `backend-fsdp` · `backend-unsloth` ·
`backend-axolotl` · `backend-llamafactory` · (serving / conversion workers)

> **The current `[train]` extra is the single-environment anti-pattern.** It bundles
> torch+transformers+trl+peft+bitsandbytes into one env. It stays as the *reference*
> `backend-corpus-studio` environment, but adding DeepSpeed/Unsloth/Axolotl must create **isolated**
> environments, not pile into `[train]`.

A convenience full-development profile may exist, but **production execution must support isolated
backend environments** — never one uncontrolled `[everything]` env as the primary architecture.

### 2.1 Environment Manager (the deliverable of this phase)

A first-class Environment Manager now ships with versioned contracts:

- `PythonRuntime` · `EnvironmentRecipe` · `DependencyResolution` · `EnvironmentInstallation` ·
  `EnvironmentLock` · `EnvironmentDescriptor` · `EnvironmentHealthReport`

Responsibilities:

- discover compatible Python runtimes;
- create isolated environments under a deterministic per-user manager root;
- select a platform/CUDA-aware **recipe** (wheel source, native-build needs, GPU-arch compat);
- **preview** install commands + disk/network requirements → **explicit user confirmation**;
- run installs as **bounded argv installers** (no shell interpolation) — mirrors the existing
  no-shell trainer-launch invariant;
- record exact package / source / hash (`EnvironmentLock`);
- apply **backend-specific dependency constraints**;
- run import → dependency → functional → hardware probes (reusing `platform/probes.py`);
- detect **drift**; safely remove/recreate; export a reproducibility lock;
- **associate an environment hash with each `RunPlan`** (extends `RunPlan.environment_ref`);
- **prevent one backend from modifying another backend's runtime.**

The side-effectful implementation is intentionally scoped to the `backend-corpus-studio` reference
worker. Other declared recipes are not installable merely because they can be previewed. Default CI
uses fake installers and CPU-only functional probes; a new real CUDA environment requires explicit
network confirmation and separate hardware verification. See
[`ENVIRONMENT_MANAGER.md`](ENVIRONMENT_MANAGER.md).

### 2.2 Environment states — "installed" ≠ "supported"

```
NOT_INSTALLED → INSTALLING → INSTALLED_UNCHECKED → IMPORTABLE →
DEPENDENCY_PROBE_PASSED → FUNCTIONAL_PROBE_PASSED → HARDWARE_VERIFIED
                                                    ↘ DEGRADED / INCOMPATIBLE / DRIFTED / BROKEN
```

This is the storage/capability honesty discipline applied to environments: a package importing is not
proof a kernel runs, which is not proof a real workload is supported. `HARDWARE_VERIFIED` supports
only the exact environment-level probe tuple that passed; it does not promote the backend, a 7B run,
offload, FlashAttention, FSDP/DeepSpeed, or MoE execution to supported.

### 2.3 Versioning strategy — no frozen versions without compatibility testing

- **broad supported ranges** in package metadata;
- **backend-specific constraints** (per worker env);
- **tested lockfiles** per supported platform/runtime;
- **actual installed manifests** (`EnvironmentLock`, exact hashes);
- **functional capability reports** (`CapabilityReport`, already shipped).

Account explicitly for: PyTorch/CUDA wheel-source selection, native-extension builds, OS differences,
GPU-architecture compatibility, and packages whose support can't be expressed via ordinary extras.

## 3. Phased roadmap

Each phase is a coherent, tested, non-destabilizing vertical slice. Contracts and tests land with (or
before) implementation; docs record only implemented facts. This is the **foundational order** — MoE
*execution* can come later, but **dense-only assumptions must be removed from the foundational
contracts as they are built** (see §5 and [`MOE_ARCHITECTURE.md`](MOE_ARCHITECTURE.md)).

| # | Deliverable | Notes |
|---|---|---|
| **0** | ✅ Platform contracts + lifecycle (profile→plan→fit→run→artifact→watchdog→subprocess) | shipped |
| **1** | ✅ **StorageProfile** — `StorageDevice` / volume / path-role assessment | shipped (this slice) |
| **2** | ✅ **Environment Manager + isolated backend runtimes** (3-layer deps, §2) — reference `backend-corpus-studio` creation, command journal, lock, probes, drift, safe remove/recreate, and RunPlan pinning shipped ([`ENVIRONMENT_MANAGER.md`](ENVIRONMENT_MANAGER.md)) | new frameworks still require one isolated, verified backend slice each |
| **3** | ✅ General **`ModelDescriptor` + `TokenizerDescriptor`** foundation + static local inspection | component-scoped representation, scoped count records, semantic routing separate from physical scheduling; the Phase 8 allowlisted topology parser now fills this substrate without runtime claims |
| **4** | ✅ **`TrainingObjective` registry** (objective distinct from backend) | 29 sealed definitions; explicit labels/masks/loss components, router/expert update policy, artifacts/resume/eval/hardware implications, and independent compatibility evidence ([`TRAINING_OBJECTIVES.md`](TRAINING_OBJECTIVES.md)) |
| **5** | ✅ **Dense-safe + MoE-safe parameter-accounting evidence foundation** (`N_logical`/`N_active`/`N_resident`/`N_touched`/`N_updated`/`N_exposed`) | hash-sealed report + strict scopes/windows/sources, bounded static producer, typed runtime reconciliation, gaps/conflicts, and planning/telemetry/checkpoint/evaluation refs shipped; actual workers must still emit measured coordinates ([`PARAMETER_ACCOUNTING.md`](PARAMETER_ACCOUNTING.md)) |
| **6** | ✅ **Immutable `RunPlan` expansion** (offload/placement/parallelism representable) | explicit physical specification, evidence pinning, capability gates, tamper checks, and honest singleton-only runner support shipped ([`RUN_PLAN_PHYSICAL_EXECUTION.md`](RUN_PLAN_PHYSICAL_EXECUTION.md)) |
| **7** | ✅ Generalized **`TraceRecord`** + Trace Studio engine/authoring foundation | versioned contract, generated clients, legacy adapter, fail-closed generation, explicit review, trainer gate, and desktop-selectable trace draft schema shipped; dedicated graphical surface remains future work ([`TRACE_RECORDS.md`](TRACE_RECORDS.md)) |
| **8** | ✅ **Static MoE model inspection** (parse allowlisted existing-MoE configs; report structural expert-instance counts) | static metadata only; runtime capability remains unverified ([`MOE_MODEL_INSPECTION.md`](MOE_MODEL_INSPECTION.md)) |
| **9A** | ✅ **Identity-bound worker protocol + fake-worker conformance** | protocol 2.0 handshake/state machine, backend and environment/lock binding, managed recipe-target checks, process-tree termination, execution-entry seal checks, legacy-plan migration boundary; no new runtime capability ([`BACKEND_WORKER_PROTOCOL.md`](BACKEND_WORKER_PROTOCOL.md)) |
| **9B** | ✅ **Effective execution truth** for the first-party dense trainer | separately hash-sealed resolved configuration; immutable inputs; explicit precision/attention/device placement/trainer defaults; exact capability/interface admission; no post-seal semantic overrides or silent field filtering; worker hash echo + runtime deviation refusal ([`EFFECTIVE_EXECUTION_CONFIGURATION.md`](EFFECTIVE_EXECUTION_CONFIGURATION.md)) |
| 9C | Additional **dense** training backends (one isolated env at a time: TRL → DeepSpeed → FSDP → Unsloth → Axolotl) | begin only after 9B closes intent-to-runtime drift; each backend requires its own managed recipe, functional probes, and eligible-host measurements |
| 10 | **Existing-model MoE fine-tuning** (router and/or selected experts, verified backend) | one backend + family first |
| 11 | **Full MoE training + expert parallelism** (exposure clocks, starvation/collapse gates, all-to-all, distributed ckpt) | |
| 12 | **Resource-elastic VRAM/RAM/NVMe expert runtime** (`N_resident << N_active << N_logical`) | measured, not claimed |

**Ordering rationale:** (a) the Environment Manager (Phase 2) is the gate before any heavy backend —
those need isolated, capability-probed environments to be added honestly; (b) `ModelDescriptor` /
`TrainingObjective` / parameter-accounting / `RunPlan` (Phases 3–6) are the foundational contracts that
**must be MoE-safe when written**, because retrofitting sparse semantics into dense-assuming contracts
later would force a disruptive redesign of the model, optimizer, checkpoint, telemetry, and artifact
systems all at once. Static topology inspection is now shipped; MoE execution (10–12) still comes
later, and the inspection result is never promoted into runtime proof. Phase 9B closed the first-party
intent-to-worker gap at the contract boundary. Another backend must independently declare, prove, and
enforce the same semantic axes; inheriting the first-party proof is forbidden.

### Phase 9B execution-truth closure

A 2026-07-13 source audit identified drift between the sealed plan and the first-party trainer. Phase
9B closes that drift for newly generated first-party plans:

- the plan carries one independently sealed `ResolvedExecutionConfiguration` with immutable input and
  objective/environment/backend/capability refs;
- attention is an exact model API + kernel + three-SDPA-toggle policy, and QLoRA precision is
  represented per material state;
- readiness requires one passing complete execution tuple (runtime/device + precision +
  quantization + adapter + attention + optimizer + loss + checkpoint + export), so unrelated probe
  successes cannot be unioned into a support claim;
- the model device map is explicit (`auto` is invalid), every semantic trainer/LoRA/checkpoint/data
  default is sealed, and exact optimizer/loss/trainer-interface capabilities are admitted fail-closed;
- the runner cannot mutate semantics or switch to echo after sealing; the worker echoes the effective
  hash before model loading and refuses package, input, attention, precision, or placement drift;
- chat-template failure blocks and truncation analysis covers the complete pinned dataset unless an
  explicit allow policy was sealed;
- backend identity fixes the runner lane, every execution receives a fresh UUIDv7 run ID, output is
  isolated beneath a run-scoped directory, and success requires optimizer-step plus real adapter
  weight evidence. The shipping desktop no longer exposes the unsealed direct-trainer path.

This is control-plane and fake/unit-test evidence, not new GPU-workload proof. The current native-Linux
host now supplies the separate prerequisite environment result: `backend-corpus-studio` passed its
minimal hardware-probe tuple. The Phase-9B workload must still run through the sealed enforcement path
before that path is called workload-verified. DeepSpeed, FSDP, CPU/NVMe offload, and MoE execution
remain unimplemented.

## 4. Invariants preserved throughout

The dependency-light **control plane**, single-writer dataset protection, provider fail-closed policy,
provenance/gate honesty, no-shell argv execution, "a completed step ≠ proven fit", no silent target
truncation, and "installed ≠ supported" all carry forward unchanged. No working functionality is
deleted without migration coverage; no on-disk format changes silently; no irreversible migration
without backup/rollback.

## 5. Dense-safe / MoE-safe foundational contracts (binding constraint)

**No new foundational contract may assume dense execution.** Even while the implementation stays
dense/QLoRA-oriented in early phases, `ModelDescriptor`, `TrainingObjective`, `ArtifactManifest`,
`RunPlan`, checkpoint, telemetry, and evaluation contracts must be built so that Mixture-of-Experts,
sparse models, and conditional computation slot in **without a redesign**. Specifically, no foundational
contract may assume: every parameter active per token; all parameters resident; one optimizer clock;
one monolithic checkpoint; one global parameter count; routing == physical placement; equal expert
exposure; sparse activation ⇒ sparse data movement; or sparse activation ⇒ reduced wall-clock. The full
requirement — parameter accounting, expert identity, semantic-router-vs-physical-scheduler separation,
sparse optimizer clocks, exposure gates, MoE-aware fit prediction, sharded transactional checkpoints,
expert lineage, and the A–G MoE phase plan — is specified in
[`MOE_ARCHITECTURE.md`](MOE_ARCHITECTURE.md).
