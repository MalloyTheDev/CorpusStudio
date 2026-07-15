# Native-Linux RTX 5070 training study

This directory preregisters the CorpusStudio native-Linux measurement program before any new benchmark result is collected. Protocol version `1.0.0` was frozen on 2026-07-14. The machine-readable matrix is the source of truth when prose and configuration differ.

The study asks how forced PyTorch math and flash SDPA behave for matched QLoRA workloads on one RTX 5070, how feasibility changes through sequence length 4096, how independently verified backends compare, and how LoRA differs from QLoRA in selected cells.

Files:

- [PROTOCOL.md](PROTOCOL.md) defines the design, controls, execution order, stopping rules, exclusions, and claim boundaries.
- [HYPOTHESES.md](HYPOTHESES.md) separates confirmatory hypotheses from exploratory analyses.
- [EXPERIMENT_MATRIX.yaml](EXPERIMENT_MATRIX.yaml) is the immutable, machine-readable factor and trial specification.
- [EXPERIMENT_MATRIX.v1.2.0.json](EXPERIMENT_MATRIX.v1.2.0.json) is the complete, hash-sealed
  **current** effective matrix (after amendment 0002).
  [EXPERIMENT_MATRIX.v1.1.0.json](EXPERIMENT_MATRIX.v1.1.0.json) (amendment 0001) is retained as a
  frozen historical document.
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
