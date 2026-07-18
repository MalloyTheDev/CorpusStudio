# Parameter Accounting

CorpusStudio represents parameter quantities as evidence, not as one model-size scalar. The source of
truth is the `ParameterAccountingReport` contract in
[`engine/corpus_studio/platform/contracts.py`](../engine/corpus_studio/platform/contracts.py); the
dependency-light producers and reconciler live in
[`engine/corpus_studio/platform/parameter_accounting.py`](../engine/corpus_studio/platform/parameter_accounting.py).

This foundation is dense-safe and MoE-safe. Static MoE topology inspection now supplies structural
expert identities for an allowlisted family set, but it does not claim that a current backend measures
every parameter axis or makes the resource-elastic research target practical.

## Distinct quantities

| Contract kind | Short name | Meaning |
|---|---|---|
| `logical` | `N_logical` | Independently addressable coordinates in an exact model universe. |
| `active_token` | `N_active_token` | Coordinates participating in one declared token computation. |
| `active_sequence` | `N_active_sequence` | Unique coordinates participating in one declared sequence. |
| `touched_window` | `N_touched_window` | Unique coordinates touched during an explicit scheduling window. |
| `resident` | `N_resident` | Coordinates resident on one named device/tier at a measurement instant. |
| `updated_window` | `N_updated_window` | Coordinates actually changed by optimizer actions in a window. |
| `exposed_window` | `N_exposed_window` | Coordinates receiving valid routing/training opportunities in a window. |
| `effective` | `N_effective` | An optional defined effective-capacity quantity; never substituted for an addressable count. |

`N_resident << N_active << N_logical` is a research target, not a validator rule. Dense models may
have equal values, estimates can be incomparable, and sparse activation does not prove low data
movement or fast execution.

## Evidence shape

Every `ParameterObservation` carries:

- a stable `ParameterScope` tied to an exact model reference and coordinate-universe ID;
- a structured `ParameterWindow` (static snapshot, token, sequence, instant, microbatch, optimizer
  window, or run) rather than a free-form timing label;
- a producer, version, method, capture time, and source reference;
- coverage (`complete`, `partial`, or `sampled`) and value relation (`exact`, `estimate`, or bound);
- an identity basis, so stored tensor elements are not mislabeled as independent coordinates;
- explicit handling for tied, shared, replicated, generated, quantized, optimizer-shadow, and cache
  state.

Measured observations require a capture time and hash-pinned source. Partial or sampled evidence
cannot claim an exact value. Expert scopes require sorted stable expert IDs plus a coordinate-universe
hash; transient device addresses are never identities. Unknown is a `ParameterEvidenceGap`, never a
numeric zero. A measured zero remains valid evidence when its source is pinned.

## Static model evidence

`build_model_parameter_accounting()` consumes a `ModelDescriptor` and optionally the corresponding
local snapshot:

1. Descriptor/config counts retain their original `declared`, `estimated`, or `measured` evidence;
   their source reference hash binds the complete extracted `ModelDescriptor` semantics, not only the
   underlying weight snapshot.
2. An optional safetensors reader parses at most a 16 MiB header per file, validates shapes, dtypes,
   offsets, duplicate keys, duplicate tensor identities, and path containment, and never deserializes
   tensor data or imports model code. Multiple shards require one content-pinned
   `model.safetensors.index.json` whose tensor-to-shard map exactly covers the observed headers.
3. If descriptor weight hashes exist, each source file is streamed through SHA-256 in the same stable
   file-open used for inspection. This is integrity verification, not model loading.
4. A header total becomes exact logical evidence only when the weight inventory is complete and
   content-verified, handling is fully resolved, tensor identities do not overlap, and a declared or
   measured (not merely estimated) descriptor count agrees. Otherwise it remains stored-element
   evidence with explicit gaps.

Supplying a different or changed snapshot cannot silently attach evidence to the descriptor's model
revision. Malformed, oversized, linked, overlapping, incomplete, or changing inputs produce bounded
gaps rather than a guessed count.

Static [`ExpertTopologyCounts`](MOE_ARCHITECTURE.md) are deliberately outside this parameter
coordinate algebra. They count expert **instances** across layers and expert identities selected per
token; they do not reveal how many independent coordinates each expert contains, which coordinates
are tied/shared, or where any coordinate resides. A router top-k therefore never manufactures
`N_active_token` or `N_resident`. Unless exact coordinate evidence exists, the corresponding
`ParameterAccountingReport` axes remain explicit gaps.

## Runtime reconciliation

Workers can emit typed observations through `RunEvent.metrics.parameter_observations`. The
`reconcile_parameter_accounting_events()` seam:

- verifies the parent report seal;
- requires one sorted, unique run stream;
- anchors every dynamic observation to that report's `run_ref`;
- requires complete, exact, measured evidence for runtime axes;
- keeps parent-report lineage hash-pinned;
- derives explicit conflicts only for comparable coordinate universes and windows.

Validated contradictions include same-key exact disagreement, a comparable dynamic count exceeding
`N_logical`, `N_updated_window > N_touched_window` in the same exact window, and
`N_active_token > N_active_sequence` for the same sequence. There is deliberately no universal
`N_updated <= N_exposed` rule: exposure and optimizer action can describe different semantics.

`MemoryMetrics` byte counters are not converted into `N_resident`. A worker must emit coordinate
identity evidence for residency; otherwise the report stays incomplete.

## Report status and lifecycle links

Reports are canonically SHA-256 sealed and have one of three statuses:

- `complete`: every axis required by the selected profile has qualifying evidence and no gap;
- `incomplete`: at least one required or integrity/identity fact remains a gap;
- `conflicting`: comparable authoritative observations contradict one another.

Profiles cover static models, training plans, training runtime, inference runtime, checkpoints, and
evaluation. `RunPlan`, `RunManifest`, `ArtifactManifest`, and `EvaluationResult` now carry report
references; `RunEvent` carries the typed observations. These links make the boundary available across
the lifecycle without manufacturing backend measurements.

Current limitation: the control-plane contract, static producer, event loader, and reconciler ship;
the existing dense training workers do not yet emit the full runtime observation set. Until a backend
adds pinned coordinate instrumentation, runtime reports correctly retain gaps. Static family
detection does not change that requirement.

## CLI

Produce static evidence during inspection:

```bash
cd /mnt/training-nvme/repos/CorpusStudio
engine/.venv/bin/python -m corpus_studio.cli model-inspect /mnt/training-nvme/models/tiny \
  --hash-weights --parameter-accounting --out /mnt/training-nvme/model-records
```

Produce a report from a saved `ModelDescriptor`:

```bash
engine/.venv/bin/python -m corpus_studio.cli parameter-account \
  /mnt/training-nvme/model-records/tiny.model.json --snapshot /mnt/training-nvme/models/tiny \
  --out /mnt/training-nvme/model-records/tiny.parameter-accounting.json --json
```

Reconcile typed worker events against a sealed parent report:

```bash
engine/.venv/bin/python -m corpus_studio.cli parameter-account \
  /mnt/training-nvme/model-records/tiny.parameter-accounting.json \
  --events /mnt/training-nvme/runs/run-1/events.jsonl --profile training_runtime \
  --out /mnt/training-nvme/runs/run-1/parameter-accounting.json
```

Checkpoint and evaluation profiles additionally require hash-pinned `--artifact-ref ID@SHA256` or
`--evaluation-ref ID@SHA256` lineage. All output writes are atomic.

## Generated boundary

`ParameterAccountingReport` is a root JSON Schema with a generated TypeScript module. Contract edits
must regenerate both layers:

```bash
cd /mnt/training-nvme/repos/CorpusStudio/engine
.venv/bin/python -c "from corpus_studio.platform.schema_export import export_json_schemas; export_json_schemas('../docs/contracts')"
cd /mnt/training-nvme/repos/CorpusStudio/apps/web
npm run gen:contracts
```
