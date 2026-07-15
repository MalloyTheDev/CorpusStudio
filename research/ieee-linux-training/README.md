# Native-Linux RTX 5070 training study

This directory preregisters the CorpusStudio native-Linux measurement program before any new benchmark result is collected. Protocol version `1.0.0` was frozen on 2026-07-14. The machine-readable matrix is the source of truth when prose and configuration differ.

The study asks how forced PyTorch math and flash SDPA behave for matched QLoRA workloads on one RTX 5070, how feasibility changes through sequence length 4096, how independently verified backends compare, and how LoRA differs from QLoRA in selected cells.

Files:

- [PROTOCOL.md](PROTOCOL.md) defines the design, controls, execution order, stopping rules, exclusions, and claim boundaries.
- [HYPOTHESES.md](HYPOTHESES.md) separates confirmatory hypotheses from exploratory analyses.
- [EXPERIMENT_MATRIX.yaml](EXPERIMENT_MATRIX.yaml) is the immutable, machine-readable factor and trial specification.
- [METRICS.md](METRICS.md) defines measurement windows, formulas, units, and summaries.
- [FAILURE_TAXONOMY.md](FAILURE_TAXONOMY.md) defines success and terminal classifications.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) defines the identity chain, evidence layout, and replay requirements.

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
