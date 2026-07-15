# Preregistered hypotheses

These hypotheses apply to the exact controlled cells in protocol `1.0.0`. The trial is the replication unit. Failure and feasibility outcomes are analyzed separately from successful-cell performance; missing successful measurements are never treated as zero.

## Confirmatory hypotheses

### H1: forced flash versus forced math SDPA

For matched successful first-party QLoRA cells:

- H1a: forced flash SDPA has lower median measured optimizer-step time than forced math SDPA.
- H1b: forced flash SDPA has higher total non-padding tokens per second than forced math SDPA.
- H1c: the paths differ in peak CUDA allocated and reserved memory. This memory hypothesis is two-sided because checkpointed layer-boundary state may dominate kernel workspace.
- H1d: forced flash SDPA has lower integrated energy per 1,000 non-padding tokens. Instantaneous power may be higher; energy, not power alone, is the efficiency outcome.

The null for each metric is no paired difference at the trial-summary level. Directional H1a, H1b, and H1d are considered supported only when all three paired trial differences have the predicted sign and the descriptive estimate agrees. No p-value threshold is used with three trials.

### H2: sequence scaling

Within a fixed model and attention path:

- median optimizer-step time and peak memory are nondecreasing as sequence length increases;
- throughput need not be monotonic because fixed overhead and kernel utilization compete;
- feasibility is nonincreasing with sequence length, subject to preserving any nonmonotonic observed result rather than correcting it.

Sequence 4096 is tested directly for every primary model/path cell unless an exact sealed planner contradiction refuses it before model loading.

### H3: model scaling

At matched sequence and attention path, increasing nominal model size increases median step time and peak process memory. Feasibility is expected to become more constrained from 0.5B to 3B to 7B.

### H4: maximum feasible sequence

Forced flash SDPA is expected to support an equal or greater maximum feasible primary sequence length than forced math SDPA for a matched model. A tie, refusal, timeout, OOM, or counterexample is retained as the result.

## Secondary hypotheses

### H5: LoRA versus QLoRA

At selected matched 0.5B and 3B cells, QLoRA is expected to reduce peak process GPU memory relative to LoRA. Step-time and energy differences are treated as two-sided because quantization overhead can trade memory for compute cost. This benchmark makes no quality or convergence hypothesis.

### H6: independently verified backends

A backend may differ in step time, throughput, memory, and energy only after it independently proves the same sealed semantic tuple. Unsloth comparisons are gated, secondary, and two-sided. Registration, import success, or a backend-specific toy run is insufficient admission evidence.

## Exploratory analyses

The following are explicitly exploratory: adaptive sequence-boundary refinement; load and preflight cost; correlations among power, utilization, clocks, and temperature; variance changes with sequence length; failure-stage distributions; and any later fused-loss or additional-backend ablation. Exploratory results are labeled and cannot replace preregistered primary cells.

## Non-hypotheses

The study does not test output quality, convergence, generalization, long-run reliability, Windows/WSL behavior, external flash-attn, offload, MoE execution, or hardware other than the registered host. A successful bounded trial cannot support those claims.
