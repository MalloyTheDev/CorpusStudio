# Native-Linux RTX 5070 training study

This directory preregisters the CorpusStudio native-Linux measurement program before any new benchmark result is collected. Protocol version `1.0.0` was frozen on 2026-07-14. The machine-readable matrix is the source of truth when prose and configuration differ.

The study asks how forced PyTorch math and flash SDPA behave for matched QLoRA workloads on one RTX 5070, how feasibility changes through sequence length 4096, how independently verified backends compare, and how LoRA differs from QLoRA in selected cells.

Files:

- [PROTOCOL.md](PROTOCOL.md) defines the design, controls, execution order, stopping rules, exclusions, and claim boundaries.
- [HYPOTHESES.md](HYPOTHESES.md) separates confirmatory hypotheses from exploratory analyses.
- [EXPERIMENT_MATRIX.yaml](EXPERIMENT_MATRIX.yaml) is the immutable, machine-readable factor and trial specification.
- [EXPERIMENT_MATRIX.v1.1.0.json](EXPERIMENT_MATRIX.v1.1.0.json) is the complete, hash-sealed
  effective matrix after amendment 0001.
- [METRICS.md](METRICS.md) defines measurement windows, formulas, units, and summaries.
- [FAILURE_TAXONOMY.md](FAILURE_TAXONOMY.md) defines success and terminal classifications.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) defines the identity chain, evidence layout, and replay requirements.
- [amendments/](amendments/) contains append-only protocol amendments, manifests, and reserved
  historical identities.
- [validate_protocol.py](validate_protocol.py) reconstructs and verifies the exact effective matrix;
  it can also reject reused candidate identities and rehash preserved host evidence.

## Effective protocol version

Protocol version 1.0.0 remains the immutable base specification. Amendment
[0001](amendments/0001-2026-07-15-manager-1.3-blue-green-identities.md) establishes effective
protocol version 1.1.0 prospectively for new work. It changes only the two first-party managed
environment IDs at the workload-semantics layer, replacing the v1 placeholders with new manager-1.3
v4 blue/green identities. It also adds fail-closed manager, worker-success, study-identity, and
historical non-reuse admission records. The complete versioned JSON matrix is the effective study
specification; no prior environment, RunPlan, failed run, or result is relabeled.

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
