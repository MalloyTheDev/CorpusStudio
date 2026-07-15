# RunPlan physical execution contract

Phase 6 makes physical scheduling explicit without claiming that today's reference trainer can
execute every represented topology. The pydantic source is `platform/contracts.py`; generated JSON
Schema and TypeScript remain the language-neutral boundary.

## What a new RunPlan seals

`platform-plan` now always emits `physical_execution` for a new plan:

- `resources` names concrete GPU, pinned/pageable RAM, NVMe, SATA, or remote tiers;
- `placements` maps parameters, gradients, optimizer state, or activations to those resources by
  whole-model identity or stable parameter/component/expert IDs;
- `offload_rules` names the source, target, mechanism, trigger, prefetch/eviction policy, and route
  miss behavior;
- `parallelism` binds every rank to a compute resource and records explicit data/tensor/pipeline/
  expert/sequence/context groups plus their communication backend;
- `storage_profile_ref` pins the exact storage snapshot used by a storage-backed plan;
- `evidence_status = planned_not_measured` prevents planned placement from masquerading as measured
  `N_resident` evidence.

The default currently supported plan is intentionally narrow: one explicit CPU or CUDA resource,
rank 0, one whole-model parameter placement, no offload rules, and no distributed groups. The legacy
`offload_strategy` field remains as a compatibility summary, but it must agree with the explicit rules
on every new expanded plan. ZeRO-2/3 names in that legacy enum are not treated as a substitute for
placement or parallel-group records.

## Semantic routing stays separate

A physical selector contains identities, never router probabilities, top-k, or learned routing
policy. The default route fidelity is `preserve_or_fail`: unavailable state may wait, defer, or fail,
but it may not silently select a resident expert instead.

`semantic_fallback` is representable only when the plan also carries a separately hash-pinned model
policy reference. That records a declared learned-policy choice; storage location still does not
silently redefine the algorithm.

## Parameter and storage evidence

Use a sealed Phase 5 report when a physical spec names parameter scopes, components, or experts:

```powershell
corpus-studio platform-plan `
  --base-model Qwen/Qwen2.5-7B-Instruct `
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 `
  --dataset .\data\examples.jsonl `
  --physical-spec .\PhysicalExecutionSpec.json `
  --parameter-accounting-report .\ParameterAccountingReport.json `
  --out .\plan
```

The planner revalidates the report, verifies its canonical hash, checks that every selected ID exists,
and pins the report by `(report_id, report_hash)`. It copies no inferred parameter totals into the
plan, and missing runtime axes remain gaps.

Storage-backed resources embed the exact per-role `StorageRoleAssessment` and require the matching
content-hashed `StorageProfile`:

```powershell
corpus-studio platform-plan ... `
  --physical-spec .\PhysicalExecutionSpec.json `
  --storage-profile .\StorageProfile.json
```

`unsuitable` is always refused. A `marginal` or `unknown` assessment must already record that exact
accepted verdict in the physical spec and also requires `--allow-marginal-storage` or
`--allow-unknown-storage`. This is explicit risk acceptance, not a promoted suitability claim. The
storage profiler remains non-destructive and does not prove throughput.

## Backend and fit honesty

`BackendManifest` and `EffectiveCapabilities` now have separate placement-tier, placement-mode,
offload, parallelism, and communication axes. A non-trivial physical plan needs both:

1. a backend static declaration; and
2. a passing functional probe for every requested token.

The built-in workers currently declare only single-resource GPU placement and have no proven offload,
distributed, or communication capability. Therefore the contract can represent tiered/offloaded/MoE
plans, while the planner and runner honestly refuse to execute them today. A paged optimizer is not
relabeled as proven controlled offload merely because it can page under pressure.

A selector that names parameter, component, or expert IDs requests the separate `identity_scoped`
placement capability; one device alone does not prove that a worker will honor those IDs. The planner
validates and pins the sealed parameter report first, then still refuses the plan unless that scoped
behavior is both declared and functionally proven.

The current VRAM calibrator handles only the singleton path. Any non-trivial physical spec returns
`PLANNED_UNPROVEN` with no peak or native-residency claim until a physical-plan estimator exists.
Existing training runners also fail with `UNSUPPORTED_CONFIGURATION` before importing or invoking a
trainer if they cannot consume the spec.

For the supported singleton training path, measured memory is still not proof of successful fit. A
native fit receives `proven=true` only after the exact output path, recognized adapter bytes,
artifact integrity, canonical before/after trainable-state change, materialized-gradient coverage, a
real optimizer, one finite loss for every completed sealed step, finite final tensors, exact
trained-to-saved PEFT tensor/config identity, and durable terminal admission all pass. The raw peak is
carried in success evidence so the subprocess parent can reconstruct the fit instead of trusting a
child classification. A failed non-spilling run remains `NATIVE_UNPROVEN`; only an actually measured
spill retains a spill classification.

## Immutability and compatibility

`plan_hash` covers the fully defaulted physical spec on every new plan. `platform-run` and the worker
both recompute and verify the canonical hash before execution, so editing a rank, device, selector, or
offload rule invalidates the plan.

Older persisted plans remain readable with `physical_execution = null`. Their historical hash payload
omits that absent field, preserving verification compatibility. They should be replanned before using
new physical-scheduling features.

## Deliberately not claimed

- no DeepSpeed/FSDP/expert-parallel worker was added;
- no CPU/NVMe offload path was hardware verified;
- no placement record proves actual residency, bytes moved, cache behavior, or fit;
- no physical scheduler currently changes semantic routing;
- no MoE topology is inferred by this phase.

Those execution and measurement steps remain later isolated-backend and MoE phases.
