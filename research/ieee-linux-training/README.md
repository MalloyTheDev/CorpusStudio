# Native-Linux RTX 5070 training study

This directory preregisters the CorpusStudio native-Linux measurement program before any new benchmark result is collected. Protocol version `1.0.0` was frozen on 2026-07-14. The machine-readable matrix is the source of truth when prose and configuration differ.

The study asks how forced PyTorch math and flash SDPA behave for matched QLoRA workloads on one RTX 5070, how feasibility changes through sequence length 4096, how independently verified backends compare, and how LoRA differs from QLoRA in selected cells.

Files:

- [PROTOCOL.md](PROTOCOL.md) defines the design, controls, execution order, stopping rules, exclusions, and claim boundaries.
- [HYPOTHESES.md](HYPOTHESES.md) separates confirmatory hypotheses from exploratory analyses.
- [EXPERIMENT_MATRIX.yaml](EXPERIMENT_MATRIX.yaml) is the immutable, machine-readable factor and trial specification.
- [EXPERIMENT_MATRIX.v1.4.0.json](EXPERIMENT_MATRIX.v1.4.0.json) is the complete, hash-sealed
  **current** effective matrix (after amendment 0004).
  [EXPERIMENT_MATRIX.v1.3.0.json](EXPERIMENT_MATRIX.v1.3.0.json) (amendment 0003),
  [EXPERIMENT_MATRIX.v1.2.0.json](EXPERIMENT_MATRIX.v1.2.0.json) (amendment 0002) and
  [EXPERIMENT_MATRIX.v1.1.0.json](EXPERIMENT_MATRIX.v1.1.0.json) (amendment 0001) are retained as
  frozen historical documents.
- [METRICS.md](METRICS.md) defines measurement windows, formulas, units, and summaries.
- [FAILURE_TAXONOMY.md](FAILURE_TAXONOMY.md) defines success and terminal classifications.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) defines the identity chain, evidence layout, and replay requirements.
- [amendments/](amendments/) contains append-only protocol amendments, manifests, and reserved
  historical identities.
- [validate_protocol.py](validate_protocol.py) reconstructs and verifies the exact effective matrix;
  it can also reject reused candidate identities and rehash preserved host evidence.

## Effective protocol version

Protocol version 1.0.0 remains the immutable base specification. Amendment
[0001](amendments/0001-2026-07-15-manager-1.3-blue-green-identities.md) established effective protocol
version 1.1.0 (manager-1.3 v4 blue/green identities) and is now a frozen historical document. Amendment
[0002](amendments/0002-2026-07-15-post-audit-v5-identities.md) supersedes it with effective protocol
version **1.2.0**: because two post-audit worker corrections (#444 sealed-precision restore and #445
AdamW-CPU-scalar-step / attention-cleanup fixes) changed the worker execution bytes after the v4 wheel
was built, every v4 identity is now historical. 1.2.0 replaces the v1 placeholder first-party
environment IDs with fresh **v5** blue/green identities, binds the audited worker source commit
`df86db5`, extends the reserved-identity registry to
[v2](amendments/RESERVED_IDENTITIES.v2.json) (append-only over v1, adding all v4 identities), and
defines the ~500-output full-training phase separately from feasibility. The validator enforces the
append-only reservation and hash-binds the frozen 0001 amendment. The complete versioned JSON matrix
is the effective study specification; no prior environment, RunPlan, failed run, or result is
relabeled.

Amendment [0003](amendments/0003-2026-07-16-v6-worker-lineage-telemetry-and-artifact-corrections.md)
supersedes 0002 with effective protocol version **1.3.0**: the v5 bring-up produced the study's first
real GPU training then failed at export, and the two worker-child corrections #461 (narrow
`training_args.bin` admission) and #462 (complete paper telemetry) changed worker execution bytes, so a
fresh **v6** blue/green lineage is required. 1.3.0 replaces the v5 environment IDs with
`backend-corpus-studio-research-{math,flash}-v6`, requires the worker source to descend from `af28be9`,
and extends the reserved-identity registry to
[v3](amendments/RESERVED_IDENTITIES.v3.json) (append-only over v2, reserving every fully instantiated
v5 identity). The scientific tuple (Qwen2.5-0.5B, chat, seq 256, mb 1, ga 1, 12 steps, QLoRA
r16/alpha32) is unchanged. **Result (2026-07-16):** both matched v6 0.5B smokes SUCCEEDED -
`V6_MATH_AND_FLASH_BRINGUP_PASS` (12 steps each, adapter admitted, measured `NATIVE_SAFE`,
`scientifically_complete=True`); one honestly-recorded non-blocking token-throughput observer gap
(`tokens/sec = 0.0`). See [`docs/HOST_STATE.md`](../../docs/HOST_STATE.md) v6 section.

Amendment [0004](amendments/0004-2026-07-16-v7-worker-lineage-token-throughput-observer.md)
supersedes 0003 with effective protocol version **1.4.0**: the v6 token-throughput `0.0` is
reclassified as **UNAVAILABLE (null), not a measured zero** - the #462 collate-fn observer never fired
because the accelerate-prepared `DataLoaderShard` bypasses a `.collate_fn` reassignment on the pinned
stack. The fix (PR #466, merge `25c901ec`) observes `inputs` at `training_step`, emits raw per-step
token counts, and gates `scientific_throughput_complete` / `paper_performance_complete` separately from
resource completeness. Because that observer runs in the worker child it changes worker execution bytes,
so a fresh **v7** blue/green lineage is required. 1.4.0 replaces the v6 environment IDs with
`backend-corpus-studio-research-{math,flash}-v7`, requires the worker source to descend from `25c901ec`,
and extends the reserved-identity registry to
[v4](amendments/RESERVED_IDENTITIES.v4.json) (append-only over v3, reserving every fully instantiated
v6 identity). The scientific tuple is unchanged; a v7 pass now additionally requires valid token
accounting (positive non-padding and supervised counts every measured step, rates equal to observed
tokens / duration, `paper_performance_complete=true`).

## Evidence boundary at preregistration

Historical evidence is context, not a benchmark result:

- A tiny environment-level forced-flash QLoRA tuple passed.
- A real Qwen2.5 0.5B load reached placement verification.
- The first real smoke failed before adapter insertion and completed zero optimizer steps.
- A separate placement-only diagnostic observed actual singleton `cuda:0` parameter residency.
- No real optimizer step has passed through `platform-run`.
- Sequence length 4096 remains unverified.

These facts do not establish workload feasibility, comparative performance, model fit, or model quality.

## Change control

After the first new feasibility or characterization result exists, this protocol is append-only. Any change requires a dated, versioned amendment that states the reason, affected cells, whether results were visible, and whether the change is prospective or creates a separately labeled follow-up. Primary results are always reported under the protocol version that generated them.

Large environments, model snapshots, datasets, telemetry, logs, and raw results belong under `/mnt/training-nvme`, not in the source checkout. Only small reviewable specifications, tooling, summaries, and publication artifacts belong here.
