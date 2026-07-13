# Implementation Plan — Full AI Lifecycle Platform

This is the forward-looking engineering plan for evolving Corpus Studio from its current state (a
mature dataset-to-model-to-evaluation control plane with a hardware-aware run lifecycle) toward the
complete local-first AI engineering platform described in the product vision. It is **planning**, not
a record of shipped work — for what actually works today, [`CURRENT_STATE.md`](CURRENT_STATE.md) is
the source of truth. Nothing here is claimed as implemented unless it links to a shipped feature.

## 1. Grounded architecture audit (2026-07-13)

The **run lifecycle is largely built and hardware-verified.** The `engine/corpus_studio/platform/`
package is a real, torch-free contract substrate + lifecycle, not a plan on paper:

- **14 versioned contracts** (`platform/contracts.py`) → language-neutral JSON Schema (`docs/contracts/`).
- **profile → plan (hash-sealed) → predict-fit → run → measure-fit → account-for-artifacts**, with a
  multi-backend registry (`corpus_studio`, `unsloth`), a watchdog (measured fit + spill/stall), and a
  supervised subprocess worker that can **kill a hung run**. Verified end-to-end on a real RTX 5070.

Against the full-platform vision, the frontier is the **input side**. Ranked gaps:

| Gap | Status | Notes |
|---|---|---|
| **Storage & offload profiling** | ✅ **done** (this slice) | `StorageProfile` + role suitability — see [`HARDWARE_STORAGE_PROFILE.md`](HARDWARE_STORAGE_PROFILE.md) |
| **Environment Manager + dependency profiles** | ⏭️ **next foundational phase** | §2 below — the prerequisite for isolated multi-backend execution |
| Model & Tokenizer Lab (`ModelDescriptor`/`TokenizerDescriptor`) | planned | no contracts yet |
| `TrainingObjective` as a distinct registry | partial | today only `TaskType` enum + `RunPlan` |
| Full versioned `TraceRecord` | partial | lightweight `training/traces.py` exists |
| Dataset mixtures + transformation graph | partial | `DatasetManifest.lineage` is the seed |
| Offload planning (DeepSpeed/FSDP/NVMe) | planned | **depends on** StorageProfile ✅ **and** the Environment Manager |

## 2. Dependency & runtime architecture (correction — the next foundational phase)

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
The main process talks to them through the **versioned `WorkerMessage` protocol** (already shipped),
never by importing them:

`backend-corpus-studio` · `backend-trl` · `backend-deepspeed` · `backend-fsdp` · `backend-unsloth` ·
`backend-axolotl` · `backend-llamafactory` · (serving / conversion workers)

> **The current `[train]` extra is the single-environment anti-pattern.** It bundles
> torch+transformers+trl+peft+bitsandbytes into one env. It stays as the *reference*
> `backend-corpus-studio` environment, but adding DeepSpeed/Unsloth/Axolotl must create **isolated**
> environments, not pile into `[train]`.

A convenience full-development profile may exist, but **production execution must support isolated
backend environments** — never one uncontrolled `[everything]` env as the primary architecture.

### 2.1 Environment Manager (the deliverable of this phase)

A first-class Environment Manager with versioned contracts:

- `EnvironmentDescriptor` · `DependencyRequirement` · `DependencyResolution` · `InstalledPackage` ·
  `EnvironmentLock` · `CapabilityProbeResult` · `EnvironmentHealthReport` · `EnvironmentRecipe`

Responsibilities:

- discover compatible Python runtimes;
- create isolated environments (`~/.corpusstudio/environments/<backend>/`);
- select a platform/CUDA-aware **recipe** (wheel source, native-build needs, GPU-arch compat);
- **preview** install commands + disk/network requirements → **explicit user confirmation**;
- run installs as **bounded argv installers** (no shell interpolation) — mirrors the existing
  no-shell trainer-launch invariant;
- record exact package / source / hash (`EnvironmentLock`);
- apply **backend-specific dependency constraints**;
- run import → dependency → functional → hardware probes (reusing `platform/probes.py`);
- detect **drift**; **repair/recreate**; export a reproducibility lock;
- **associate an environment hash with each `RunPlan`** (extends `RunPlan.environment_ref`);
- **prevent one backend from modifying another backend's runtime.**

### 2.2 Environment states — "installed" ≠ "supported"

```
NOT_INSTALLED → INSTALLING → INSTALLED_UNCHECKED → IMPORTABLE →
DEPENDENCY_PROBE_PASSED → FUNCTIONAL_PROBE_PASSED → HARDWARE_VERIFIED
                                                    ↘ DEGRADED / INCOMPATIBLE / DRIFTED / BROKEN
```

This is the storage/capability honesty discipline applied to environments: a package importing is not
proof a kernel runs, which is not proof the hardware supports it. Only `HARDWARE_VERIFIED` earns
"supported."

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
| **2** | 🔨 **Environment Manager + isolated backend runtimes** (3-layer deps, §2) — **substrate shipped** (recipe registry + install-preview resolver, [`ENVIRONMENT_MANAGER.md`](ENVIRONMENT_MANAGER.md)); env creation/health/drift/lock is the next slice | gate before DeepSpeed/FSDP/multimodal |
| 3 | General **`ModelDescriptor` + `TokenizerDescriptor`** | **must be MoE-safe from the start** (§1 of MoE doc) |
| 4 | **`TrainingObjective` registry** (objective distinct from backend) | must express router-vs-expert training |
| 5 | **Dense-safe + MoE-safe parameter accounting** (`N_logical`/`N_active`/`N_resident`/`N_touched`/`N_updated`/`N_exposed`) | no runtime needed — contracts only |
| 6 | **Immutable `RunPlan` expansion** (offload/placement/parallelism representable) | uses StorageProfile + Env Manager |
| 7 | Generalized **`TraceRecord`** + Trace Studio | |
| 8 | **MoE model inspection** (detect/parse existing MoE; report logical/active/expert counts) | inference-only OK if labeled |
| 9 | Additional **dense** training backends (one isolated env at a time: TRL → DeepSpeed → FSDP → Unsloth → Axolotl) | |
| 10 | **Existing-model MoE fine-tuning** (router and/or selected experts, verified backend) | one backend + family first |
| 11 | **Full MoE training + expert parallelism** (exposure clocks, starvation/collapse gates, all-to-all, distributed ckpt) | |
| 12 | **Resource-elastic VRAM/RAM/NVMe expert runtime** (`N_resident << N_active << N_logical`) | measured, not claimed |

**Ordering rationale:** (a) the Environment Manager (Phase 2) is the gate before any heavy backend —
those need isolated, capability-probed environments to be added honestly; (b) `ModelDescriptor` /
`TrainingObjective` / parameter-accounting / `RunPlan` (Phases 3–6) are the foundational contracts that
**must be MoE-safe when written**, because retrofitting sparse semantics into dense-assuming contracts
later would force a disruptive redesign of the model, optimizer, checkpoint, telemetry, and artifact
systems all at once. MoE execution (8, 10–12) comes later; the contracts do not.

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
