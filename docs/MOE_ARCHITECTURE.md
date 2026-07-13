# Mixture-of-Experts, Sparse Models & Conditional Computation

**Status: architectural requirement, not yet implemented.** No MoE runtime exists today; the platform
is dense/QLoRA-oriented. This document is the binding design constraint so that when MoE arrives it is
*not* a disruptive redesign. Its single most important rule:

> **No new foundational contract may assume dense execution.** MoE support must be designed into the
> `ModelDescriptor`, `TrainingObjective`, `RunPlan`, checkpoint, telemetry, artifact, and evaluation
> contracts **now**, even though full MoE *training* lands in a later phase.

MoE is **not** `is_moe: bool` and **not** just another trainer label. It changes the meaning of
parameter count, memory, optimizer steps, checkpoint consistency, placement, throughput, evaluation,
artifact identity, routing telemetry, storage offload, and distributed execution.

## Dense assumptions to remove from foundational contracts

No foundational contract may assume:

- every parameter is active for every token;
- every parameter is resident simultaneously;
- all parameters share one optimizer clock;
- every checkpoint is one monolithic weight set;
- one global parameter count is sufficient;
- routing is identical to physical placement;
- every expert receives equal training exposure;
- sparse activation guarantees sparse data movement;
- sparse activation guarantees reduced wall-clock cost.

Each of these maps to a specific contract that must stay neutral — see the checklist in §"Impact on
existing contracts".

## 1. Parameter accounting — one count is a lie

An MoE model must **never** be reported with a single `parameter_count`. Foundational parameter
accounting introduces distinct, explicitly-scoped quantities:

| Quantity | Meaning |
|---|---|
| `N_logical` | all independently addressable model coordinates |
| `N_active_token` | coordinates participating in one token's computation |
| `N_active_sequence` | coordinates participating in one sequence's computation (where calculable) |
| `N_touched_window` | unique coordinates/expert blocks touched during a declared scheduling window |
| `N_resident` | coordinates resident in GPU memory at a measurement point |
| `N_updated_window` | coordinates actually changed by optimizer actions in the window |
| `N_exposed_window` | coordinates that received valid routing/training opportunities |
| `N_effective` | optional measured effective-capacity — **never** substituted for addressable coordinates without a declared definition |

The intended research relationship is `N_resident << N_active << N_logical` (e.g. a research target of
~30B logical / 2–4B active / 50–200M resident). **These are configurable research targets, never
hardcoded Corpus Studio limits.**

Each count must declare: unit; scope; measurement window; source; measured-vs-estimated; and its
handling of tied / shared / replicated / generated / quantized parameters, optimizer shadows, and
decompressed caches. Do **not** say "independent degrees of freedom" when only optimizer-addressable
coordinates were measured.

The Phase 5 `ParameterAccountingReport` foundation now implements this distinction with stable
coordinate scopes, structured windows, pinned evidence sources, coverage/value relations, explicit
identity bases, gaps, and comparable-evidence conflicts. Static descriptor/safetensors evidence and a
typed `RunEvent` reconciliation seam ship; current backend workers do not yet claim full runtime
measurement. See [`PARAMETER_ACCOUNTING.md`](PARAMETER_ACCOUNTING.md).

## 2. Coordinate & expert identity

A stable **expert registry** maps `expert_id → (model revision, layer, expert index, tensor set,
authoritative state location, representation, ownership, replication group, shard group, version)`.
Every checkpoint, route event, optimizer update, and artifact references **stable expert identity**,
never a transient device address. This prevents double-counting from aliases, tied tensors, shared
experts, replicas, decompressed copies, GPU caches, checkpoint replicas, optimizer shadows, generated
weights, and factorized reconstructions.

## 3. Semantic router vs physical scheduler (the critical separation)

Two systems, never conflated:

- **Semantic router** — *which experts are semantically appropriate for this token* (router logits,
  top-k, gating weights, routing noise, capacity constraints, load-balancing loss, specialization,
  fallback). This is part of the **learned algorithm**.
- **Physical scheduler** — *where those experts live, when to load them, what transfer schedule is
  feasible* (placement, residency, prefetch, eviction, buffering, transfer scheduling, expert
  batching, cache policy, execution order, compute/transfer overlap, recovery).

**Storage location or current residency must never silently redefine the learning algorithm.** The
platform may optimize execution (prefetch, caching, residency-aware tie-breaking, load-aware routing
penalties, route batching, expert grouping, NVMe/RAM tiering) — but must **record when the physical
scheduler altered, deferred, approximated, or rejected the router's preferred assignment**: preferred
route, executed route, reason, affected tokens, delay, fallback, estimated + measured impact.
Hardware-aware routing is allowed only as an **explicit declared model policy**, never an invisible
runtime substitution.

## 4. Expert residency & tiered state

Expert state spans a memory hierarchy (GPU / pinned RAM / pageable RAM / local NVMe / SATA / archive /
remote). For each expert or shard, record: authoritative location; cached locations; dirty/clean;
representation + precision per tier; version; last access/update; residency duration; transfer count;
bytes transferred; cache hits/misses; prefetch success; eviction reason. Placement policies (static,
LRU/LFU, predicted-next prefetch, route-cluster caching, layer-window residency, heat-based promotion,
bandwidth/cost-budgeted) must be **inspectable and benchmarked** — this is where §Storage
([`HARDWARE_STORAGE_PROFILE.md`](HARDWARE_STORAGE_PROFILE.md)) and the physical `RunPlan` contract
([`RUN_PLAN_PHYSICAL_EXECUTION.md`](RUN_PLAN_PHYSICAL_EXECUTION.md)) meet MoE. The current contract can
represent these choices; no built-in worker yet executes or measures the tiered path.

## 5. Sparse optimizer semantics — a global step is insufficient

Define multiple clocks: global token / microbatch / optimizer; router optimizer; per-expert exposure,
gradient, and optimizer-update clocks; per-expert and per-shard versions. Per expert, record: routed
tokens; valid training tokens; backward contributions; optimizer updates; last update; staleness;
accumulated gradient age; skipped-update reason; optimizer-state residency + version. **A global step
count is not proof that every expert was trained.**

## 6. Exposure integrity & starvation

Track per-expert exposure (tokens/sequences/domains routed, updates, time-since-update, gradient norm,
parameter delta, utilization distribution, routing entropy, load imbalance, starvation/overflow/
dropped-token/fallback/cold-expert rates). Configurable **gates**: routing collapse, dead experts,
starvation, imbalance, overflow, token dropping, stale state, routed-but-never-updated,
updated-without-exposure, never-loaded experts. Exposure policy is **declared before training**;
evaluation data must not manipulate routing exposure unless that is an explicit, leakage-protected part
of the experiment.

## 7. Loss & router telemetry — never one opaque scalar

Represent routing losses separately (primary task, router aux, load-balancing, z-loss, entropy reg,
overflow penalty, specialization reg, locality penalty). Telemetry: router-logit stats, top-k margins,
routing entropy, expert load histograms, per-layer utilization, overflow, capacity utilization, aux
loss, router + expert gradient norms, expert parameter-delta norms. **Physical-locality metrics stay
separate from semantic-quality metrics.**

## 8. MoE-aware fit prediction — never from active-param count alone

Predict dense/shared + router residency; active + unique expert weight volume per window; expert cache
working set; optimizer working set; expected transfers, bytes/token, bytes/step, cache hit rate,
all-to-all traffic, imbalance, overflow, NVMe/RAM/PCIe traffic, write volume, checkpoint volume,
prefetch slack, stalls, cold-start cost. **Two models with equal active parameters can have radically
different unique footprint, routing locality, transfer volume, and wall-clock cost.**

## 9. MoE backend capabilities & environments

`BackendManifest` gains explicit MoE capabilities (loads MoE; trains router; trains experts; full-param
MoE; PEFT-over-MoE; expert adapters; expert/tensor/data/pipeline parallelism; CPU/NVMe expert offload;
optimizer offload; prefetch; caching; expert checkpoint sharding; distributed restore; token dropping;
aux losses; heterogeneous experts; dynamic expert add/remove; inference-only). **Each claimed
capability requires a functional probe or an explicit `UNVERIFIED` state — loading for inference does
NOT prove training support.**

MoE lives in **isolated backend worker environments** (per the 3-layer dependency model in
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)) — potential recipes `backend-moe-reference`,
`backend-moe-deepspeed`, `backend-moe-distributed`, `backend-moe-offload`, `backend-moe-inference`.
Each records torch/CUDA/compiler/GPU-arch/comm-backend/native-build status + distributed / all-to-all /
grouped-expert-kernel / checkpoint-compat / offload probes + an exact lock. **Installed ≠ MoE-capable.**

## 10. Checkpointing & transactionality

An MoE checkpoint may be dense/shared weights + router + expert shards + optimizer shards (router +
expert) + routing stats + expert clocks + placement manifest + scheduler state + RNG + data position +
distributed topology. **Complete only when all required components share one consistent logical
version** (temp dir → per-shard hashes → manifest → expected-shard inventory → completion marker →
atomic promotion → rollback; partial-checkpoint detection). Never restore a router from one version
with expert shards from another without explicit compatibility validation. Support complete / expert-
only / router-only restore, expert replacement, dense→MoE init, MoE→dense distillation lineage, partial
recovery, resharding — **every non-exact restore produces a new forked lineage record.**

## 11. Artifact lineage & evaluation

Artifact types extend to MoE base / router / expert collection / expert shard / expert adapter /
placement manifest / sparse optimizer state / dense→MoE conversion / MoE→dense distillation / pruned /
expanded / merged / routing-calibration. The graph preserves parent model + router parent + expert
parents + dataset versions + expert identities + run + checkpoint source + routing config + topology +
backend + hashes + eval links.

Evaluation covers **both** model quality (held-out/domain/robustness/calibration/regression vs dense &
sparse baselines) **and** expert-system behavior (utilization, load balance, entropy, specialization,
stability, overflow, dropping, dead/stale experts, route sensitivity, cache-locality). Causal
expert-contribution tests (revert one expert's delta, replace with init, disable, cohort-replace,
bounded routing randomization, freeze-router-vs-experts). **Do not claim an expert "learned" merely
because it was selected — require evidence that expert-state changes affected held-out behavior.**

## 12. Consumer-hardware / resource-elastic MoE

Represent constraints independently (VRAM, RAM, NVMe capacity + sustained bandwidth, PCIe, CPU/GPU
compute, energy, thermal, write endurance, wall-clock). Support planning `N_resident << N_active <<
N_logical` experiments (pageable experts, layer-window execution, prefetch, cache reuse, route
batching, bounded-staleness updates, compressed/reduced-state optimizers, RAM/NVMe tiering, conditional
depth, adaptive activation) — **research hypotheses until measured**. Never claim a 30B-logical MoE is
practical on a consumer GPU merely because 2–4B params are active per token; the run must **measure**
unique expert state touched, bytes transferred, cache amplification, optimizer traffic, update
coverage, staleness, throughput, energy, wall-clock, held-out improvement, and recovery correctness.

## Impact on existing contracts (the dense-safe checklist)

| Existing contract | Change to stay dense-safe |
|---|---|
| `ModelDescriptor` ✅ foundation | component-scoped representation + scoped multi-count records + optional semantic routing/expert groups; physical scheduling remains owned by `RunPlan`. The static inspector leaves execution kind unknown rather than guessing dense/MoE; actual MoE parsing is Phase B. |
| `RunPlan` ✅ foundation | explicit resources, state placements, expert/component/parameter selectors, offload rules, rank bindings, and parallel groups are hash-sealed; semantic routing remains separate and non-trivial built-in execution is refused rather than inferred |
| `CheckpointPolicy` / checkpoint | must allow multi-component sharded checkpoints + a completion manifest (§10) |
| `RunEvent` / `EventMetrics` | routing/expert telemetry channels; locality metrics separate from quality |
| `ArtifactManifest` | expert/router/shard artifact kinds + expert-identity lineage (§11) |
| `EvaluationResult` | router/expert-behavior axis alongside model quality (§11) |
| `BackendManifest` | MoE capability flags, each functionally probed (§9) |
| `TrainingObjective` ✅ foundation | semantic update scopes distinguish router/selected experts/all experts/adapters/shared components; expert-scoped policies require stable identity + per-expert exposure and carry optimizer-clock/starvation/collapse requirements; separately keyed loss components allow future router/load-balance/z-loss terms without conflating physical scheduling |
| `ParameterAccountingReport` ✅ foundation | separate logical/active/resident/touched/updated/exposed axes; stable expert/coordinate identity; structured token/sequence/run/instant windows; explicit measured/estimated/gap/conflict status; bytes never masquerade as resident coordinates |

## Phased MoE plan

- **A — Dense-safe foundational contracts** (in progress; `ModelDescriptor`, `TrainingObjective`,
  parameter-accounting, and physical `RunPlan` foundations shipped, no MoE runtime): ensure
  `ArtifactManifest`, checkpoint, and telemetry also avoid dense-only assumptions.
- **B — MoE model inspection**: detect existing MoE architectures; parse expert/router structure;
  report logical/active/shared/routed/expert counts; tokenizer compat. Inference-only is acceptable if
  **labeled**.
- **C — Existing-model MoE fine-tuning**: load a supported MoE model; train router and/or selected
  experts through a verified backend; MoE-aware checkpoints + telemetry. One backend, one family first.
- **D — Full expert training**: full expert + router optimization; exposure accounting; sparse
  optimizer clocks; starvation/collapse gates; exact resume.
- **E — Expert parallelism**: multi-GPU expert-parallel execution; all-to-all telemetry; distributed
  checkpointing; topology-aware fit planning.
- **F — Consumer-resource expert tiering**: VRAM/RAM/NVMe residency; prefetch; eviction;
  bounded-staleness updates; traffic accounting; transactional recovery.
- **G — Custom MoE construction**: build from architecture descriptors; init routers/experts; valid
  dense→MoE conversion; train from random init; evaluate vs declared dense + sparse baselines.

Each phase is a complete, tested vertical slice.

## Acceptance criteria

MoE support is not complete until: logical/active/resident/touched/exposed/updated counts are distinct;
stable expert identities exist; semantic routing is separated from physical scheduling; per-expert
exposure + update clocks are tracked; starvation and routing collapse are detectable; expert shards
checkpoint atomically; partial checkpoints are rejected or explicitly recovered; backend capabilities
are functionally probed; expert placement + offload appear in `RunPlan`; bytes moved + unique expert
footprint are measured; evaluation tests whether learned expert state affects held-out behavior; dense
assumptions are removed from foundational contracts; and **all public claims state the precise
verification level**. Until then, label the capability **partial / experimental / inference-only** as
appropriate.
