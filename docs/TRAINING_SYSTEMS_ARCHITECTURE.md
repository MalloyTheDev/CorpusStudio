# Training Systems Architecture

**Status: architecture proposal for review. Docs-only. No training code, dependency, model, dataset, GPU,
or research action is part of this document.**

CorpusStudio is a **local-first, end-to-end AI development ecosystem and IDE** (see
[`PRODUCT_AREAS.md`](PRODUCT_AREAS.md), [`PRODUCT_SPEC.md`](PRODUCT_SPEC.md)); this doc is the internal
design of its **Training Studio** area (one of seven co-equal product areas). It expands the training
surface from "fine-tuning support" into a complete, **pluggable model-development system** that spans
from-scratch pretraining, continued
pretraining, full-parameter fine-tuning, adapter/PEFT fine-tuning, preference and RL post-training,
distillation, dense and MoE architectures, single-device and distributed execution, and multiple
framework and orchestrator adapters.

It does **not** redesign the foundational contracts. Most already exist and are dense-safe / MoE-safe by
construction (the [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §5 mandate: *no new foundational
contract may assume dense execution*). The work here is **additive**: expose the existing dimensions as
independent registries, add the few missing contracts (pretraining data, framework/orchestrator split,
support-level ladder), and compose them through a `TrainingPlan`.

The native-Linux 7B IEEE paper is a **sealed-research overlay** (see
[`PRODUCT_VS_RESEARCH.md`](PRODUCT_VS_RESEARCH.md)); it is one *consumer* of this architecture at the
SEALED_RESEARCH assurance level, never its definition.

## 1. The architectural rule: compose, do not enumerate

Do **not** model every capability combination as one backend ID. A single `unsloth-qlora-qwen` "backend"
is a dead end - the combination space is `objective x topology x update x framework x orchestrator x
parallelism x precision x hardware`, which is millions of cells.

Instead, define **independent registries** and let a `TrainingPlan` compose one entry from each:

```
objective   = pretraining
topology     = moe
update       = full_parameter
framework    = pytorch
orchestrator = megatron
parallelism   = [dp, tp, ep]
precision     = bf16
```

Presets such as `unsloth-qlora-qwen` are **convenience profiles** that pin one entry per dimension - not
foundational backend identities.

## 2. The eleven registries

Each registry is an independent, versioned, dependency-light set of entries. Every entry carries a
**support level** (§4). "Exists today" is grounded in `engine/corpus_studio/platform/contracts.py` and
`enums.py`.

| # | Registry | What an entry is | Today |
|---|---|---|---|
| 1 | **TrainingObjective** | what is optimized (loss/label/mask/update semantics) | **shipped** - 29 sealed defs, `TrainingObjective` contract, `ObjectiveKind` (16 families incl. pretraining, preference, reward, distillation, process-supervision, embedding, multimodal). See [`TRAINING_OBJECTIVES.md`](TRAINING_OBJECTIVES.md). |
| 2 | **ModelTopology** | dense vs MoE structure + expert layout | **contract shipped** - `ModelTopology`, `ExpertGroup`, `ExpertTopologyCounts`, `SemanticRouting`; no runtime training claim. Registry surface: **planned**. |
| 3 | **UpdateMethod** | which parameters change + how | **partial** - `AdapterMethod` enum + `ObjectiveUpdateScope` (all_parameters / adapters / router / selected_experts / all_experts). Full independent registry with per-method deps/hardware: **planned**. |
| 4 | **FrameworkBackend** | compute substrate (PyTorch / JAX / TF-Keras / MLX) | **new** - today only PyTorch is assumed. Must be split out of `BackendManifest`. |
| 5 | **OrchestratorAdapter** | the training-loop driver (HF/TRL, Unsloth, torchtune, Axolotl, Megatron, ...) | **new** - today conflated into the single reference backend. |
| 6 | **ParallelismStrategy** | dp/tp/pp/ep/sp/cp degrees + placement | **contract shipped** - `ParallelismKind`, `ParallelGroup`, `RankBinding`, `PhysicalExecutionSpec`, `StatePlacement`, `OffloadRule`. See [`RUN_PLAN_PHYSICAL_EXECUTION.md`](RUN_PLAN_PHYSICAL_EXECUTION.md). Named strategy presets: **planned**. |
| 7 | **PrecisionAndQuantization** | compute/param/quant dtypes | **shipped enums** - `PrecisionMode`, `QuantizationMode`, `PrecisionExecutionPolicy`. Registry surface: **planned**. |
| 8 | **HardwareTarget** | CUDA / ROCm / Metal-MLX / XLA-TPU / CPU | **partial** - `DeviceKind`, `EnvironmentProfile`, `AcceleratorRuntime`; only CUDA is workload-touched. |
| 9 | **CheckpointStrategy** | monolithic vs sharded/resharding + resume | **contract shipped** - `CheckpointImpl`, `CheckpointManifest`, `SealedTrainingState`, `ResumeLineage`. See [`CHECKPOINT_RESUME.md`](CHECKPOINT_RESUME.md). Pretraining **data cursor** is a gap (§6). |
| 10 | **EvaluationProfile** | how a run is scored/gated | **contract shipped** - `EvaluationResult`, `EvalGate`, `EvalMetric`, `EvalTarget`. See [`EVALUATION_STUDIO.md`](EVALUATION_STUDIO.md). Named profiles: **planned**. |
| 11 | **TrainingPreset** | a named convenience profile pinning one entry per dimension | **new** - e.g. `unsloth-qlora-qwen`. |

## 3. The TrainingPlan composition layer

Three layers, from user intent to sealed execution (only the top is new):

- **TrainingPlan** (new, user-facing) - composes one entry per registry + free parameters. Validates
  cross-dimension compatibility *before* resolution (an objective's `ObjectiveModelRequirement` vs the
  topology; a parallelism strategy vs the hardware target; an update method vs the framework).
- **RunPlan** (shipped) - the immutable, fully-resolved plan the core dispatches. Already binds
  `task_type`, precision, quantization, adapter, optimizer, loss, attention, sequence, batching,
  checkpoint policy, offload, and a `PhysicalExecutionSpec`.
- **ResolvedExecutionConfiguration** (shipped) - the separately hash-sealed exact execution truth the
  worker echoes and refuses to deviate from. See
  [`EFFECTIVE_EXECUTION_CONFIGURATION.md`](EFFECTIVE_EXECUTION_CONFIGURATION.md).

**Authority + the Rust-core target.** "The core" that seals plans, verifies, and admits results is the
authoritative control plane. The **target architecture is a Rust authoritative core + isolated Python ML
workers** ("Rust owns truth; Python computes ML and returns evidence"; a worker result is a candidate until
the core admits it); the control plane is Python today and the migration is gated/incremental. The
composition model below is unchanged by that migration - only which language owns the authority moves.

`TrainingPlan` composes; it does not replace `RunPlan`. Resolution lowers a `TrainingPlan` into
**one or more** `RunPlan`s (one per objective/stage; `RunPlan.task_type` is single-valued over the rich
`TaskType` enum) for a chosen `(FrameworkBackend, OrchestratorAdapter)` pair. Two invariants keep
`TrainingPlan` from becoming a second planning authority: (1) it carries **no** `plan_hash` /
`configuration_hash`-sealed execution field - it is pre-resolution intent only; `RunPlan` retains sole
sealing authority and `ResolvedExecutionConfiguration` is independently sealed with an explicit
single-trainer-config-authority guard. (2) Its cross-dimension compatibility check is an **early UX
pre-check**, never the authoritative gate - the authoritative gate stays the planner's declared-and-proven
capability check plus the `ExecutionCapabilityCombination`-in-a-passing-probe match (which is why proving
one capability tuple never implies another; see §6).

## 4. Support levels (installed is never supported)

Every capability - a registry entry, a backend manifest claim, a preset - carries exactly one state:

| State | Meaning |
|---|---|
| `DECLARED` | claimed in a manifest; no code path yet |
| `CONFIG_GENERATION_ONLY` | can emit a valid config/launcher for an external tool; CorpusStudio does not run it |
| `INSTALLED` | dependencies resolve and import in a managed env |
| `PROBED` | a functional capability probe passed (import + tiny construct), **not** a workload |
| `WORKLOAD_VERIFIED` | a real bounded workload passed on real hardware (measured, not predicted) |
| `PRODUCTION_SUPPORTED` | workload-verified + telemetry + checkpoint/resume + failure handling + docs |
| `REFUSED` | known-incompatible or unsafe on this host/stack; fail-closed |

`INSTALLED` never implies `PROBED` or `WORKLOAD_VERIFIED`. This is the honesty invariant that already
governs the repo (`ObjectiveVerificationStatus`, `RecipeVerification`, "a completed step != proven fit",
"installed != supported").

`SupportLevel` does **not** replace the shipped verification vocabulary. The committed ladders are
**multi-axis**, not linear: `ObjectiveVerificationStatus` has 6 members (incl. `not_applicable`) and is
applied on three independent axes (`ObjectiveVerification`: definition / implementation / hardware);
`VerificationOutcome` carries `partial` / `not_checked` and its docstring explicitly forbids collapsing
integrity/compatibility/functional/hardware into "a misleading linear level." `SupportLevel` is therefore
an **additive, coarse capability-support rollup that COEXISTS** with those ladders (which remain the
authority). The mapping from them is a **lossy partial projection**, defined only for the proven axes;
`not_applicable` / `partial` / `not_checked` are **carried, not projected**; `REFUSED`,
`CONFIG_GENERATION_ONLY`, `INSTALLED`, and `PRODUCTION_SUPPORTED` are net-new states. A single
`SupportLevel` never states *which* axis is proven - the multi-axis records do. See §6 gap G3.

The **default** framework/orchestrator/preset for any workload is **evidence-selected**: it must reach
`WORKLOAD_VERIFIED` on the actual host stack. Unsloth (or any project) is **not** the default until its
exact stack is probed and workload-verified.

## 5. Backend manifest (what each backend declares)

A backend = a `(FrameworkBackend, OrchestratorAdapter)` binding. Its static manifest declares (shipped
fields are in `BackendManifest`; **new** fields are additive):

objectives; topologies*(new)*; update methods; model families; hardware; precision; quantization;
parallelism; checkpoint/resume; artifact formats; telemetry support; required dependencies; license*(new)*;
security boundaries*(new)*; config generator*(new)*; launcher*(new)*; progress parser*(new)*; failure
parser*(new)*; capability probes; known incompatibilities.

Detail + the full inventory classification is in
[`TRAINING_BACKEND_REGISTRY.md`](TRAINING_BACKEND_REGISTRY.md).

## 6. Contract-gap inventory (existing dense/SFT-only assumptions -> required changes)

All changes are **additive and MoE-safe by construction**; none is a foundational redesign.

| ID | Existing assumption / gap | Required contract change |
|---|---|---|
| **G1** | `TrainingDataPolicy.dataset_format` is `Literal["instruction","chat","trace"]` - **SFT-only** | Add a pretraining/corpus data contract: shards, streaming, mixture weights, document boundaries, deterministic sample order, token budget. See [`PRETRAINING_ARCHITECTURE.md`](PRETRAINING_ARCHITECTURE.md). |
| **G2** | `SealedTrainingState` is single-cursor / single-rank / `epoch`-required (validated), captures one opaque `sampler.pt`; `TrainingSchedule` allows only `max_steps` XOR epochs (no token budget) | Extend the checkpoint contract for streaming: a **data cursor** (shard id + intra-shard offset + consumed tokens + mixture RNG + packing-buffer), **per-rank** cursors (a shape change, not a scalar add), epoch-optional-for-streaming, a new `CheckpointFileEntry.role`, and a **token-budget** stop condition on `TrainingSchedule`. See [`PRETRAINING_ARCHITECTURE.md`](PRETRAINING_ARCHITECTURE.md). |
| **G3** | The shipped verification ladders are **multi-axis** (`ObjectiveVerificationStatus` 6-state on 3 axes; `VerificationOutcome` with `partial`/`not_checked` + an explicit anti-collapse rule; `RecipeVerification`) and lack `CONFIG_GENERATION_ONLY` / `INSTALLED` / `PRODUCTION_SUPPORTED` / `REFUSED` | Add a `SupportLevel` enum (§4) that **coexists** with the ladders (not a replacement); a **lossy partial** projection that carries `not_applicable`/`partial`/`not_checked`. Never collapse the three axes into one linear level. |
| **G4** | `BackendManifest` conflates framework + orchestrator + recipe (`backend_id` + `trainer_target`) and has **no field to declare or refuse** a backend that enables `trust_remote_code` / network / model-download inside its isolated process (security gap) | Split `FrameworkBackend` / `OrchestratorAdapter`; add `model_topologies`, `license`, `security_boundaries`, a `trust_remote_code`/network posture (refusable by assurance tier), `config_generator`, `launcher`, `progress_parser`, `failure_parser`. Because `backend_manifest_digest` hashes the whole manifest into `backend_ref`, the split lands as an **append-only new backend identity/version**, never a mutation of the sealed reference manifest. |
| **G5** | No user-facing composition object; `RunPlan.task_type` is a single value and topology/orchestrator are implicit | Add `TrainingPlan` (§3) that composes the 11 registries and validates cross-dimension compatibility before resolution. |
| **G6** | Registries 2,3,6,7,8,10 exist only as enums/contracts, not as exposed entry sets with support levels | Add thin registry surfaces (dependency-light, sealed, like the `TrainingObjective` registry). |
| **G7** | No `TrainingPreset` contract | Add a preset that pins one entry per dimension + records the support level of the whole composition. |
| **G8** | The **sealed execution/export seal** (`ResolvedExecutionConfiguration`, `TrainerInterfacePolicy`, `BackendManifest.trainer_fields`, `ExecutionInputBinding`, worker `artifacts.py`) is hard-locked to first-party dense-QLoRA-SFT via `Literal`/validator locks (`adapter_task_type=CAUSAL_LM`; `export_format==adapter_peft` else raise; `checkpoint_impl==adapter_only` else raise; single `local_file` dataset; one `adapter_model.safetensors`) | Additive but a **contract change** (not "implementation only"): **backend-scoped** resolved-execution variants (or per-backend resolved-config contracts) to express pretraining / full-parameter / MoE; new `CheckpointFileEntry` roles (`expert_shard`/`routing_state`) + a reshard-plan / EP-degree binding on `CheckpointBoundIdentities` (exact `plan_hash` match refuses an EP-degree change today); align `ArtifactManifest.kind` with `ObjectiveArtifactKind` (`expert_shards`). Belongs in **P0**. The composition authority (`RunPlan`/`ResolvedExecutionConfiguration`) is unchanged; only the seal's validators specialize. |

**What already holds (no change needed):** objective families (incl. pretraining/preference/reward/
distillation), `ParallelismKind` (dp/tp/pp/ep/sp/cp), `ModelTopology`/experts, `ObjectiveUpdateScope`
(full-parameter + adapters + router + experts), per-expert optimizer clocks, exposure tracking,
router/load-balancing/z-loss/distillation loss components, `ParameterAccountingReport` (N_logical/
N_active/N_resident), `PhysicalExecutionSpec`, and the telemetry/eval/checkpoint suites.

## 7. MoE (architecture-neutral, no single dense block)

MoE training fields (expert count, top-k, shared experts, layer placement, expert dims, router type/
score/dtype, capacity/token-dropping, load-balancing/z-loss, DP/TP/PP/CP/EP degrees, expert placement,
expert checkpoint shards/resharding, router/expert telemetry, expert utilization, dropped/overflow
tokens, dead/starved experts, per-expert gradient/update coverage, expert-contribution evaluation) are
specified in [`MOE_TRAINING_ARCHITECTURE.md`](MOE_TRAINING_ARCHITECTURE.md). **No foundational contract
may assume one dense parameter block** - this is already enforced by the existing MoE-safe contracts and
[`MOE_ARCHITECTURE.md`](MOE_ARCHITECTURE.md).

## 8. Training specialists (the multi-layer plugin plan)

The product-first skill (`.claude/skills/corpus-studio`) gains a **Training Systems** area, split into
specialists that load by task. These are a *plan of record*; no skill files are created by this doc.

- **Training Systems Router** - picks the objective/topology/update/framework/orchestrator/parallelism
  composition, resolves a `TrainingPlan`, and routes to the specialist below. Owns support-level honesty.
- **Pretraining** - corpus data contracts, tokenizer train/import/freeze, shards/streaming, token budget,
  data-cursor resume. See [`PRETRAINING_ARCHITECTURE.md`](PRETRAINING_ARCHITECTURE.md).
- **Fine-Tuning and PEFT** - full-parameter + adapter methods; today's reference path.
- **Post-Training and RL** - preference optimization, reward modeling, RL (TRL/verl/OpenRLHF/NeMo-RL).
- **MoE and Parallelism** - expert routing/balancing, EP/TP/PP, expert shards. See
  [`MOE_TRAINING_ARCHITECTURE.md`](MOE_TRAINING_ARCHITECTURE.md).
- **Backend Integration** - `FrameworkBackend`/`OrchestratorAdapter` manifests, probes, launchers,
  progress/failure parsers. See [`TRAINING_BACKEND_REGISTRY.md`](TRAINING_BACKEND_REGISTRY.md).
- **Checkpoint and Resume** - exact lineage, sharded/resharding, data cursor. See
  [`CHECKPOINT_RESUME.md`](CHECKPOINT_RESUME.md).
- **Training Evaluation** - per-objective eval profiles, gates, MoE expert-contribution metrics.

The **sealed-research overlay** (`research/ieee-linux-training`) remains a separate opt-in specialist and
is unaffected by this expansion.

## 9. Implementation sequence (no implementation PR until this is reviewed)

| Phase | Deliverable | Exit at |
|---|---|---|
| **P0** | Contracts + capability registry + authority cleanup: `SupportLevel` (coexisting rollup + lossy-partial mapping), `TrainingPlan` (no sealed fields; pre-check only), `FrameworkBackend`/`OrchestratorAdapter` split (append-only backend identity), **backend-scoped resolved-execution variants**, backend **security posture** (`trust_remote_code`/network/license, refusable by assurance tier), backend-manifest fields, thin registries. Docs + schemas + tests only. | contracts validated, schemas regenerated |
| **P1** | Dense pretraining on a small model (corpus data contract, token budget, validation loss, data-cursor resume) | `WORKLOAD_VERIFIED` (small) |
| **P2** | Continued pretraining (init-from-checkpoint, mixture reweight) | `WORKLOAD_VERIFIED` |
| **P3** | Full-parameter fine-tuning | `WORKLOAD_VERIFIED` |
| **P4** | Single-device small MoE **semantic** validation (routing, balancing, per-expert coverage) | `PROBED` -> `WORKLOAD_VERIFIED` (small) |
| **P5** | Distributed FSDP/DeepSpeed (multi-GPU dense) | `WORKLOAD_VERIFIED` |
| **P6** | Multi-device MoE expert parallelism (EP + shards/resharding) | `WORKLOAD_VERIFIED` |
| **P7** | Additional adapters: JAX, Keras, MLX, external orchestrators (Axolotl / LLaMA-Factory / torchtune / verl) | per-adapter `PROBED`/`CONFIG_GENERATION_ONLY` |

Each phase is a separate reviewed slice; each capability advances the support ladder only on measured
evidence. No phase is claimed until it is workload-verified.

## 10. Related documents

- [`PRETRAINING_ARCHITECTURE.md`](PRETRAINING_ARCHITECTURE.md) - pretraining first-class contracts.
- [`MOE_TRAINING_ARCHITECTURE.md`](MOE_TRAINING_ARCHITECTURE.md) - MoE training contracts.
- [`TRAINING_BACKEND_REGISTRY.md`](TRAINING_BACKEND_REGISTRY.md) - backend manifest + inventory.
- [`TRAINING_OBJECTIVES.md`](TRAINING_OBJECTIVES.md) - the shipped objective registry.
- [`RUN_PLAN_PHYSICAL_EXECUTION.md`](RUN_PLAN_PHYSICAL_EXECUTION.md) - parallelism/placement contract.
- [`PARAMETER_ACCOUNTING.md`](PARAMETER_ACCOUNTING.md) - dense/MoE-safe parameter evidence.
- [`MOE_ARCHITECTURE.md`](MOE_ARCHITECTURE.md) - the dense-safe foundational-contract mandate.
- [`PRODUCT_VS_RESEARCH.md`](PRODUCT_VS_RESEARCH.md) - standard / verified / sealed-research boundary.
