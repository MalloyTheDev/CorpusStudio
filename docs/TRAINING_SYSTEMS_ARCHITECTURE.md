# Training Systems Architecture

**Status: architecture proposal for review. Docs-only. No training code, dependency, model, dataset, GPU,
or research action is part of this document.**

CorpusStudio is a local-first AI dataset and model-development **application** (see
[`PRODUCT_SPEC.md`](PRODUCT_SPEC.md)). This doc expands the training surface from "fine-tuning support"
into a complete, **pluggable model-development system** that spans from-scratch pretraining, continued
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
| 10 | **EvaluationProfile** | how a run is scored/gated | **contract shipped** - `EvaluationResult`, `EvalGate`, `EvalMetric`, `EvalTarget`. See [`EVALUATION_LAB.md`](EVALUATION_LAB.md). Named profiles: **planned**. |
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

`TrainingPlan` composes; it does not replace `RunPlan`. Resolution lowers a `TrainingPlan` into a
`RunPlan` for a chosen `(FrameworkBackend, OrchestratorAdapter)` pair.

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
"installed != supported"). This 7-state ladder is a **superset** of the shipped 5-state
`ObjectiveVerificationStatus` and needs one additive `SupportLevel` enum + a mapping (§6, gap G3).

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
| **G2** | `SealedTrainingState` captures a sampler cursor for a finite in-memory SFT dataset (`sampler_state_captured`, `consumed_microsteps`), not a streaming-corpus position | Add a **data cursor** (shard id + intra-shard offset + consumed tokens) to the checkpoint contract for pretraining/streaming. |
| **G3** | Two overlapping verification ladders (`ObjectiveVerificationStatus` 5-state, `RecipeVerification`) and no `CONFIG_GENERATION_ONLY` / `INSTALLED` / `PRODUCTION_SUPPORTED` / `REFUSED` | Add one `SupportLevel` enum (§4) + a total mapping from the existing ladders. |
| **G4** | `BackendManifest` conflates framework + orchestrator + recipe and omits 7 declared fields | Split `FrameworkBackend` / `OrchestratorAdapter`; add `model_topologies`, `license`, `security_boundaries`, `config_generator`, `launcher`, `progress_parser`, `failure_parser`. |
| **G5** | No user-facing composition object; `RunPlan.task_type` is a single value and topology/orchestrator are implicit | Add `TrainingPlan` (§3) that composes the 11 registries and validates cross-dimension compatibility before resolution. |
| **G6** | Registries 2,3,6,7,8,10 exist only as enums/contracts, not as exposed entry sets with support levels | Add thin registry surfaces (dependency-light, sealed, like the `TrainingObjective` registry). |
| **G7** | No `TrainingPreset` contract | Add a preset that pins one entry per dimension + records the support level of the whole composition. |
| **G8** | Only one dense PyTorch reference backend is implemented; MoE training and distributed execution are contract-only | No contract change - these are **implementation** milestones (§9), each gated by workload verification. |

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
| **P0** | Contracts + capability registry: `SupportLevel`, `TrainingPlan`, `FrameworkBackend`/`OrchestratorAdapter` split, backend-manifest fields, thin registries. Docs + schemas + tests only. | contracts validated, schemas regenerated |
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
