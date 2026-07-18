# Mixture-of-Experts, Sparse Models & Conditional Computation

**Status: architectural requirement with the static-inspection foundation implemented.** No MoE
runtime exists today; the platform remains dense/QLoRA-oriented for execution. Phase B can now parse a
narrow allowlist of hash-pinned config topology, but that is not loadability, backend, fit, or hardware
proof. This document remains the binding design constraint so later MoE execution is *not* a disruptive
redesign. Its single most important rule:

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
| `ModelDescriptor` ✅ foundation | component-scoped representation + scoped multi-count records + semantic routing/expert groups; physical scheduling remains owned by `RunPlan`. Phase B now recognizes only complete Mixtral/Qwen2-MoE/DeepSeek V2/V3 static config mappings, binds the config hash/evidence paths, and otherwise leaves execution kind unknown. Structural expert-instance totals are not parameter-coordinate or runtime evidence. |
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
- **B — Static MoE model inspection** ✅ foundation shipped: parse a narrow allowlist of existing-MoE
  config metadata; report routed/shared/logical and active-per-token expert-instance structure;
  preserve tokenizer compatibility; fail closed on incomplete or unsupported families. Evidence is
  labeled `static_metadata_only`, and runtime capability remains `unverified` (see
  [Static MoE Model Inspection](#static-moe-model-inspection)).
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

---

## Static MoE Model Inspection

_Consolidated from the former `docs/MOE_MODEL_INSPECTION.md`._

CorpusStudio Phase 8 adds bounded, dependency-light inspection of expert topology declared by a local
model snapshot's `config.json`. The source of truth is the `ModelTopology` surface in
[`engine/corpus_studio/platform/contracts.py`](../engine/corpus_studio/platform/contracts.py); the pure
family parser lives in
[`engine/corpus_studio/platform/moe_inspector.py`](../engine/corpus_studio/platform/moe_inspector.py)
and is called by `model-inspect`.

This is **static metadata evidence only**. It does not load a model, import torch or Transformers,
execute repository code, test inference or training, prove backend compatibility, measure parameter
residency, predict fit, or establish any MoE runtime capability.

### Evidence states

Every `ModelTopology` carries a `TopologyInspection` record:

| Status | Meaning |
|---|---|
| `not_checked` | No verified `config.json` digest was available, so no classification ran. |
| `no_recognized_moe_evidence` | A verified config was inspected, but no allowlisted MoE family or MoE-like signal was recognized. This is not dense proof. |
| `detected` | One allowlisted family supplied a complete, internally valid structural mapping. |
| `incomplete` | The family is allowlisted, but required fields are missing, malformed, out of bounds, or contradictory. Execution kind stays unknown. |
| `unsupported_family` | MoE-like metadata is present for a family the parser does not map. Execution kind stays unknown. |

Static results bind `config_file`, its SHA-256 digest, the exact config paths used, parser method
`static_config_v1`, and `evidence_level: static_metadata_only`. `runtime_capability` is always
`unverified`.

### Allowlisted families

The first slice intentionally recognizes only mappings with narrow, testable rules:

| `model_type` | Required topology fields | Layer rule | Shared experts |
|---|---|---|---|
| `mixtral` | `num_hidden_layers`, `num_local_experts`, `num_experts_per_tok` | every decoder layer | none declared by this mapping |
| `qwen2_moe` | `num_hidden_layers`, `num_experts`, `num_experts_per_tok`, `decoder_sparse_step`, `shared_expert_intermediate_size`; optional `mlp_only_layers` defaults to `[]` | `(layer + 1) % decoder_sparse_step == 0`, excluding `mlp_only_layers` | one always-active shared expert per sparse layer |
| `deepseek_v2` / `deepseek_v3` | `num_hidden_layers`, `first_k_dense_replace`, `n_routed_experts`, `n_shared_experts`, `num_experts_per_tok`; optional `moe_layer_freq` defaults to compatibility value `1` | `layer >= first_k_dense_replace` and `layer % moe_layer_freq == 0` | the declared `n_shared_experts` per MoE layer |

Counts and layer indices are bounded before materializing structures. Boolean values are rejected as
integers, top-k cannot exceed routed experts, duplicate or out-of-range layer indices fail closed, and
an allowlisted config that resolves to no expert layer remains incomplete. Unsupported families are
not guessed from similar field names.

The mappings follow the corresponding upstream Transformers configurations and decoder-layer
construction:

- [Mixtral configuration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/configuration_mixtral.py)
- [Canonical Qwen1.5-MoE config](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/raw/main/config.json), [Qwen2-MoE configuration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_moe/configuration_qwen2_moe.py), and [decoder implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py)
- [DeepSeek V2 configuration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v2/configuration_deepseek_v2.py) and [official layer-frequency implementation history](https://huggingface.co/deepseek-ai/DeepSeek-V2/commit/44f7caf9e95112ea265b13f837bd43d520253548)
- [DeepSeek V3 configuration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/configuration_deepseek_v3.py) and [decoder implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py)

Additional families require their own explicit parser and fixtures. Similar names or keys are not
enough to claim a topology.

### Structural counts are not parameter counts

`ExpertGroup` records a component-scoped layout:

- stable group, layer namespace, canonical component path, and sorted layer indices;
- total logical expert identities per layer;
- routed and always-active shared expert counts that exactly partition that total;
- routed experts selected per token; and
- the config paths used as metadata evidence.

`ExpertTopologyCounts` derives model-wide **expert-instance** totals: MoE layers, routed/shared/logical
expert instances, and routed/shared/total active expert instances for one token's full model pass.
Its unit is literally `expert_instances`.

Phase 8 also makes the routed/shared partition explicit. Newly emitted records always carry concrete
`routed_expert_count` and `shared_expert_count` values. The schema continues to accept pre-Phase-8
groups that omitted the routed count or serialized a null shared count; validation normalizes those
records to `routed = expert_count - shared` and `shared = 0` when absent. Re-serialize a validated
legacy descriptor to persist the normalized form.

These values never become `N_logical`, `N_active_token`, or `N_resident` parameter coordinates.
Top-k says how many expert identities routing selects; it does not say how many independent parameter
coordinates those experts contain, where they reside, whether weights are tied, or whether a backend
executed them. `ParameterAccountingReport` therefore retains active/resident gaps until an exact,
hash-pinned producer supplies coordinate evidence.

### CLI

From the active repository:

```bash
cd /mnt/training-nvme/repos/CorpusStudio
engine/.venv/bin/python -m corpus_studio.cli model-inspect /mnt/training-nvme/models/mixtral --json
```

The human view reports the classification, family, expert-instance counts, and the explicit statements
`Resident experts: unknown - runtime measurement required` and `Execution support: not evaluated`.
JSON output and atomically written descriptor files carry the complete evidence record.

### Deliberate non-claims

This slice does not verify:

- model loading, inference, fine-tuning, or full training;
- backend or kernel support;
- router quality, expert utilization, exposure, starvation, or collapse;
- active, touched, updated, exposed, or resident parameter-coordinate counts;
- GPU/RAM/NVMe placement, offload, parallelism, throughput, or fit; or
- native-Linux RTX 5070 or any other hardware behavior.

Those require isolated backend contracts, functional probes, typed runtime telemetry, and—where
hardware is involved—measurements on the current native-Linux host.

---

## MoE Training Architecture

_Consolidated from the former `docs/MOE_TRAINING_ARCHITECTURE.md`._

**Status: architecture proposal for review. Docs-only. No MoE training code, dependency, model, dataset,
GPU, or research action is part of this document.** Part of
[`TRAINING_SYSTEMS_ARCHITECTURE.md`](TRAINING_SYSTEMS_ARCHITECTURE.md).

This is the **training** side of Mixture-of-Experts. This document states
the foundational-contract mandate (*no new foundational contract may assume dense execution*) and
the [Static MoE Model Inspection](#static-moe-model-inspection) section covers static topology inspection. Here we specify
the **architecture-neutral training fields** so MoE training composes through the same `TrainingPlan` as
dense training, with no assumption of one dense parameter block.

The relevant foundations already exist: `ModelTopology` / `ExpertGroup` / `ExpertTopologyCounts` /
`SemanticRouting`, `ObjectiveUpdateScope` (`router`, `selected_experts`, `all_experts`),
`ObjectiveOptimizerClock` (`per_expert`), `ObjectiveExposureTracking` (`per_expert`,
`router_and_expert`), the `ObjectiveLossComponentKind` set (`router_auxiliary`, `load_balancing`,
`router_z_loss`, `specialization`, `overflow`), `ParallelismKind.expert`, and the
`ParameterAccountingReport` coordinates (`N_logical`/`N_active`/`N_resident`/`N_touched`/`N_updated`/
`N_exposed`).

### 1. Architecture-neutral topology fields

Declared per model, never hard-coded to one implementation:

- **expert count** (logical) and **active top-k** per token;
- **shared experts** (always-on) count;
- **MoE layer placement** (which decoder layers are MoE vs dense);
- **expert dimensions** (intermediate size per expert, which may differ from dense FFN);
- **router type** (token-choice / expert-choice / hash / learned), **score function** (softmax/sigmoid),
  and **router dtype** (often fp32 for stability, independent of compute precision);
- **capacity** factor and **token-dropping** policy (drop / overflow-to-shared / no-drop);
- **load-balancing** loss and **router z-loss** weights.

### 2. Parallelism and placement

- **DP, TP, PP, CP, and EP degrees** (`ParallelismKind` already has data/tensor/pipeline/context/expert)
  composed independently; a `TrainingPlan` may declare `parallelism=[dp,tp,ep]`.
- **expert placement** - which experts live on which ranks (expert-parallel groups), via the shipped
  `ParallelGroup` / `RankBinding` / `StatePlacement`. See
  [`RUN_PLAN_PHYSICAL_EXECUTION.md`](RUN_PLAN_PHYSICAL_EXECUTION.md) (the authoritative parallelism/placement
  home; the current first-party worker refuses non-trivial specs fail-closed until an isolated worker
  implements them).
- Communication via `CommunicationBackend` (nccl/gloo/mpi/ucc).

### 3. Expert checkpoint shards and resharding

- **expert checkpoint shards** - a checkpoint is not one monolithic file; `CheckpointManifest.files` is
  already a multi-file hashed list. But `CheckpointFileEntry.role` has **no** `expert_shard` /
  `routing_state` role today (they fall into `other`), and a resumable checkpoint currently **mandates** a
  single optimizer-state file - so sharded expert/optimizer state needs new roles.
- **resharding** - resume under a different EP degree must remap expert shards deterministically with a
  recorded reshard plan. This is **not free today**: `CheckpointBoundIdentities` carries no
  EP-degree / world-size / reshard-plan field and requires an **exact `plan_hash` match**, so an EP-degree
  change currently **fails closed** rather than resharding. Adding the roles + a reshard-plan/EP-degree
  binding is additive (see [`TRAINING_SYSTEMS_ARCHITECTURE.md`](TRAINING_SYSTEMS_ARCHITECTURE.md) G8); the
  exact-lineage discipline from [`CHECKPOINT_RESUME.md`](CHECKPOINT_RESUME.md) still governs.

### 4. Router and expert telemetry (per-step evidence)

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

### 5. Evaluation

- **expert-contribution evaluation** - measure each expert's marginal contribution / specialization, and
  whether routing generalizes to held-out data, via named MoE `EvaluationProfile` entries
  (`EvaluationResult` / `EvalMetric` already exist).

### 6. Parameter accounting

MoE parameter claims use the existing `ParameterAccountingReport` coordinates so a "7B active / 47B
logical" statement is evidence, not a scalar: `N_logical` (all experts), `N_active` (top-k per token),
`N_resident` (loaded on device), `N_touched` / `N_updated` / `N_exposed`. See
[`PARAMETER_ACCOUNTING.md`](PARAMETER_ACCOUNTING.md). **No foundational contract may collapse these into
one dense parameter count.**

### 7. What is implemented vs planned

| Capability | Support level |
|---|---|
| MoE-safe topology / update-scope / optimizer-clock / exposure / loss-component contracts | contract shipped (`DECLARED`) |
| Static MoE topology inspection (hash-pinned allowlist) | shipped (inspection only) |
| Router/expert **training** telemetry channels | **planned (P4)** |
| Single-device small-MoE semantic validation (routing/balancing/coverage) | **planned (P4)** |
| Expert-parallel multi-device training + shards/resharding (new `CheckpointFileEntry` roles + reshard/EP-degree binding) | **planned (P6)** |
| MoE / full-model export (new worker exporter + `ArtifactManifest.kind` alignment with `ObjectiveArtifactKind`; today one `adapter_model.safetensors` only) | **planned** |
| Expert-contribution evaluation profiles | **planned** |

No MoE **runtime training capability** is claimed by any static contract or inspection. Every MoE
training capability advances the support ladder only on measured evidence.
