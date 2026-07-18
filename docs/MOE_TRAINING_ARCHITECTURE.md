# MoE Training Architecture

**Status: architecture proposal for review. Docs-only. No MoE training code, dependency, model, dataset,
GPU, or research action is part of this document.** Part of
[`TRAINING_SYSTEMS_ARCHITECTURE.md`](TRAINING_SYSTEMS_ARCHITECTURE.md).

This is the **training** side of Mixture-of-Experts. [`MOE_ARCHITECTURE.md`](MOE_ARCHITECTURE.md) states
the foundational-contract mandate (*no new foundational contract may assume dense execution*) and
[`MOE_MODEL_INSPECTION.md`](MOE_MODEL_INSPECTION.md) covers static topology inspection. Here we specify
the **architecture-neutral training fields** so MoE training composes through the same `TrainingPlan` as
dense training, with no assumption of one dense parameter block.

The relevant foundations already exist: `ModelTopology` / `ExpertGroup` / `ExpertTopologyCounts` /
`SemanticRouting`, `ObjectiveUpdateScope` (`router`, `selected_experts`, `all_experts`),
`ObjectiveOptimizerClock` (`per_expert`), `ObjectiveExposureTracking` (`per_expert`,
`router_and_expert`), the `ObjectiveLossComponentKind` set (`router_auxiliary`, `load_balancing`,
`router_z_loss`, `specialization`, `overflow`), `ParallelismKind.expert`, and the
`ParameterAccountingReport` coordinates (`N_logical`/`N_active`/`N_resident`/`N_touched`/`N_updated`/
`N_exposed`).

## 1. Architecture-neutral topology fields

Declared per model, never hard-coded to one implementation:

- **expert count** (logical) and **active top-k** per token;
- **shared experts** (always-on) count;
- **MoE layer placement** (which decoder layers are MoE vs dense);
- **expert dimensions** (intermediate size per expert, which may differ from dense FFN);
- **router type** (token-choice / expert-choice / hash / learned), **score function** (softmax/sigmoid),
  and **router dtype** (often fp32 for stability, independent of compute precision);
- **capacity** factor and **token-dropping** policy (drop / overflow-to-shared / no-drop);
- **load-balancing** loss and **router z-loss** weights.

## 2. Parallelism and placement

- **DP, TP, PP, CP, and EP degrees** (`ParallelismKind` already has data/tensor/pipeline/context/expert)
  composed independently; a `TrainingPlan` may declare `parallelism=[dp,tp,ep]`.
- **expert placement** - which experts live on which ranks (expert-parallel groups), via the shipped
  `ParallelGroup` / `RankBinding` / `StatePlacement`.
- Communication via `CommunicationBackend` (nccl/gloo/mpi/ucc).

## 3. Expert checkpoint shards and resharding

- **expert checkpoint shards** - a checkpoint is not one monolithic file; experts are sharded across
  ranks (`ObjectiveArtifactKind.expert_shards`, `routing_state`). `CheckpointManifest` already lists
  per-file entries with hashes.
- **resharding** - resume under a different EP degree must remap expert shards deterministically, with a
  recorded reshard plan; exact-lineage rules from [`CHECKPOINT_RESUME.md`](CHECKPOINT_RESUME.md) hold.

## 4. Router and expert telemetry (per-step evidence)

A MoE training claim must be bound to what the router and experts actually did:

- **expert utilization** - tokens routed per expert (distribution, not just a mean);
- **dropped / overflow tokens** - counts per step (a dropped-token rate is a first-class metric, not a
  hidden loss of signal);
- **dead / starved experts** - experts receiving ~zero tokens or ~zero gradient;
- **per-expert gradient and update coverage** - did every expert that should update actually receive a
  gradient and an optimizer step this window (ties to `N_updated` / `N_touched` and per-expert clocks);
- **load-balancing / z-loss values** as separate telemetry channels.

These extend the telemetry summary with MoE channels; the null-not-fabricated rule applies (an
unavailable per-expert count is null with a typed reason, never a plausible zero).

## 5. Evaluation

- **expert-contribution evaluation** - measure each expert's marginal contribution / specialization, and
  whether routing generalizes to held-out data, via named MoE `EvaluationProfile` entries
  (`EvaluationResult` / `EvalMetric` already exist).

## 6. Parameter accounting

MoE parameter claims use the existing `ParameterAccountingReport` coordinates so a "7B active / 47B
logical" statement is evidence, not a scalar: `N_logical` (all experts), `N_active` (top-k per token),
`N_resident` (loaded on device), `N_touched` / `N_updated` / `N_exposed`. See
[`PARAMETER_ACCOUNTING.md`](PARAMETER_ACCOUNTING.md). **No foundational contract may collapse these into
one dense parameter count.**

## 7. What is implemented vs planned

| Capability | Support level |
|---|---|
| MoE-safe topology / update-scope / optimizer-clock / exposure / loss-component contracts | contract shipped (`DECLARED`) |
| Static MoE topology inspection (hash-pinned allowlist) | shipped (inspection only) |
| Router/expert **training** telemetry channels | **planned (P4)** |
| Single-device small-MoE semantic validation (routing/balancing/coverage) | **planned (P4)** |
| Expert-parallel multi-device training + shards/resharding | **planned (P6)** |
| Expert-contribution evaluation profiles | **planned** |

No MoE **runtime training capability** is claimed by any static contract or inspection. Every MoE
training capability advances the support ladder only on measured evidence.
