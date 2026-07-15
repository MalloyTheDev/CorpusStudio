# Measurement harness (platform Section 11)

The measurement harness captures the raw telemetry a training run needs to be both *engineering-
diagnosable* and *usable as paper data*, and derives one reviewable summary from it. It is wired into
the authoritative `platform-run` path so its records cannot disagree with the run's own
`RunManifest` / `RunEvent` evidence.

It is **torch-free** (importing `corpus_studio.platform.telemetry` pulls no torch and no third-party
runtime dependency) and does no GPU work of its own beyond a read-only `nvidia-smi` status query. The
metric definitions are the research study's [`research/ieee-linux-training/METRICS.md`](../research/ieee-linux-training/METRICS.md);
this document describes the mechanism.

## Two halves, one rule

- The **sampler** (`TelemetrySampler`) takes raw GPU + host samples on a fixed cadence and appends each
  as one `TelemetrySample` JSON line to `TelemetrySamples.jsonl` **as it is taken**.
- The **aggregator** (`summarize_run_telemetry`) reads that raw series plus the durable `RunEvent`
  stream and the authoritative `RunManifest`, and **derives** one `RunTelemetrySummary`. CSV, tables,
  and plot series all render from that derived object.

The rule: **raw is authoritative and is written before any summary; the summary binds its raw sources
by sha256; a telemetry gap is never zero-filled and never converts a workload success into paper
data.**

## Raw records (written into the run directory)

Every record lives under the run-scoped directory `<out>/runs/<run_id>/`:

| File | Written by | Contents |
| --- | --- | --- |
| `RunManifest.json` | supervisor | the authoritative terminal run instance (state, failure taxonomy, success evidence) |
| `RunEvents.jsonl` | supervisor (in-path) | the durable, append-only `RunEvent` stream (stages, per-step metrics, terminal) - one JSON line each, flushed per line so a torn tail loses at most one trailing line |
| `TelemetrySamples.jsonl` | sampler | append-only `TelemetrySample` lines (GPU + host + memory, monotonic + UTC timestamps, phase, source) |
| `RunTelemetrySummary.json` | aggregator | the single derived summary (written **after** the raw records, atomically) |

The durable `RunEvents.jsonl` is written by both the in-process supervisor (`execute_run`) and the
subprocess supervisor (`execute_run_subprocess`), so the raw event stream survives a crash and is the
same truth the in-memory `SupervisedRun.events` carries.

## What is captured

- **Identity** (`TelemetryIdentity`): study / protocol / amendment / effective-matrix / cell / trial /
  repository-commit / worker-wheel / environment-lock / capability / execution-probe / plan /
  execution-configuration / model / tokenizer / chat-template / dataset / run / sequence-view. The
  aggregator fills what the plan and manifest expose; the caller supplies the study-level identity that
  the plan does not carry, via an overlay.
- **Step evidence** (from the durable event stream + manifest success evidence): exact optimizer step,
  finite per-step loss (every step, warm-up included), step time, non-padding and supervised
  tokens/second, optimizer-steps/minute, observed gradient tensor count, changed adapter tensor count,
  trainable-state before/after hashes.
- **GPU** (`GpuTelemetrySample` -> `GpuTelemetrySummary`): utilization, memory-controller utilization,
  power, temperature (with starting and whole-run maxima), graphics/memory clocks, performance state,
  throttle reasons, and the memory signature (allocator / CUDA / dedicated / shared) via `MemoryMetrics`.
- **Host** (`HostTelemetrySample` -> `HostTelemetrySummary`): worker process-tree RSS, system RAM
  used/available, swap used (+ delta), CPU utilization, cumulative disk read/write (+ deltas).
- **Energy** (`EnergyIntegration`): GPU energy by the trapezoidal rule over adjacent power samples,
  `E = sum(0.5*(P_i+P_{i+1})*(t_{i+1}-t_i))`, with intervals crossing the measured-window boundary
  linearly clipped and power linearly interpolated at the boundary; joules per measured step,
  energy per 1000 non-padding tokens, time-weighted mean / median / max power, and window coverage.
- **Sampling cadence** (`SamplingCadence`): requested interval plus the *observed* median/min/max
  inter-sample interval, so a claimed 200 ms rate is checked against reality.
- **Measurement overhead** (`MeasurementOverhead`): the sampler's own cumulative busy time and its
  fraction of wall time, so telemetry overhead is quantified rather than assumed negligible.
- **Outcome** (`RunOutcomeSummary`): copied verbatim from the manifest (state, training-success,
  failure taxonomy/stage, measured fit) - never re-derived, so it cannot disagree with the run.

## Warm-up vs measured window

Warm-up optimizer steps (the study's steps 1-2) are kept distinct from the measured steady-state
window (steps 3-12). Warm-up samples are retained and counted but excluded from measured aggregates and
from the measured-window energy integral. The supervisor marks the sampler's phase from the
authoritative event stream (a step's number decides warm-up vs measured), never from a wall-clock
guess.

## Scientific completeness (a gap never fakes paper data)

`ScientificCompleteness.scientifically_complete` is `False` whenever a required paper field was not
captured - step losses, step time, GPU power, GPU memory, run energy, host process-tree RSS, or the
required identity lineage (repository commit, worker wheel, environment lock, plan hash, execution
configuration hash, run id). A run can be a genuine workload **success** and still be reported as *not
scientifically complete*; the run's terminal state is never altered by a telemetry gap, and missing
driver fields stay `null` rather than being zero-filled.

## CLI

```bash
# Capture raw telemetry during a run and derive the summary (requires --out).
corpus-studio platform-run <plan.json> --subprocess --out <run-root> \
  --telemetry --telemetry-interval-ms 200

# Re-derive a summary from a completed run's raw records (idempotent, no GPU).
corpus-studio telemetry-summarize <run-root>/runs/<run_id> --plan <plan.json> --table
corpus-studio telemetry-summarize <run-root>/runs/<run_id> --csv --no-write
```

`platform-run --telemetry` mints the run id up front so the sampler and supervisor share the run
directory, samples across the run, and then writes `RunTelemetrySummary.json` derived from the raw
records. `telemetry-summarize` derives (and by default writes) the summary purely from
`RunManifest.json` + `RunEvents.jsonl` + `TelemetrySamples.jsonl`; `--csv` / `--table` render the same
derived object. It reports on stderr when a run is not scientifically complete.

## Cross-trial statistics

`combine_trial_values` computes the descriptive statistics and the two-sided 95% Student-t interval for
a per-trial mean over several per-run summaries. For the planned `n=3` design it uses
`t_(0.975,2) = 4.3026527299`; with fewer than three successful trials it returns the available values
and no confirmatory interval (never a fabricated one). Optimizer steps are never treated as independent
trials.

## Deliberate scope boundaries (this slice)

- **Plots are emitted as plot-ready data series** (`power_series`, `step_loss_series`), not rasterized
  images: the control plane stays dependency-light and a downstream tool renders the image from the
  same raw records the energy integral uses.
- **Subprocess-mode host process-tree RSS** roots the sampler's default probe at the worker child pid
  (`set_root_pid`). GPU and global host metrics are captured directly; the child-tree RSS rooting is
  validated on the first live GPU smoke, not fabricated here.
- **The default GPU/host probes are only smoke-tested off-GPU** (they are fail-soft and must not
  raise); their driver-parsing accuracy is confirmed on a live run. All aggregation, energy, statistics,
  completeness, and rendering logic is covered by deterministic synthetic-fixture tests.
