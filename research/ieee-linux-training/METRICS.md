# Metrics and statistical definitions

All raw values retain their native timestamp and unit. Aggregation converts units explicitly and never transcribes measured values by hand.

## Measurement windows

Each characterization trial has 12 optimizer steps. Steps 1 and 2 are warm-up. Steps 3 through 12 form the ten-step steady-state measurement window. The unit of replication is the trial.

The following phases are timed separately: immutable-input verification, dataset formatting, truncation analysis, model load, k-bit preparation, adapter insertion, optimizer creation, warm-up steps, measured steps, adapter serialization, and post-run verification. Preflight and load time are not included in steady-state throughput.

Telemetry uses monotonic host timestamps. The baseline interval ends at worker process start; the run interval starts at process start and ends at the authenticated terminal event/process exit; the measured interval spans the boundaries of optimizer steps 3 through 12. Samples outside the run interval are retained but excluded from run energy integration.

## Primary performance metrics

- Optimizer-step time (seconds): elapsed monotonic time from the start of the first microstep belonging to an optimizer step through completion of that optimizer update and gradient reset. Report all ten measured steps and the per-trial median.
- Total non-padding tokens per second: sum of attention-mask tokens consumed in the measured window divided by measured-window seconds.
- Supervised tokens per second: sum of labels not equal to the ignore index in the measured window divided by measured-window seconds.
- Samples per second: completed samples in the measured window divided by measured-window seconds.
- Optimizer steps per minute: `60 * measured_optimizer_steps / measured_window_seconds`.

Token counts come from the materialized batch, not `sequence_length * batch_size` when padding is present. Effective tokens per optimizer step and supervised tokens per optimizer step are stored for every step.

## Memory metrics

For each source, report baseline, measured-window maximum, whole-run maximum, and delta from baseline where meaningful:

- CUDA allocated bytes from the process allocator;
- CUDA reserved bytes from the process allocator;
- NVIDIA process memory bytes from `nvidia-smi` for the registered GPU UUID and worker process tree;
- host resident-set bytes for the worker process tree;
- host swap-used bytes and delta;
- explicit offload or spill observations.

CUDA peak statistics are reset at the declared measured-window boundary. Whole-run peaks are collected separately so model-load and steady-state claims cannot be confused. Allocator values are not interpreted as parameter residency, and `nvidia-smi` values are not interpreted as model fit.

## Power, energy, thermal, and utilization metrics

`nvidia-smi` is sampled every 200 ms where the driver supports it. Samples bind to exact GPU index and UUID. Record GPU utilization, memory-controller utilization when available, power draw, temperature, graphics and memory clocks when available, and process memory.

Energy is integrated with the trapezoidal rule over adjacent power samples:

`E_joules = sum(0.5 * (P_i + P_(i+1)) * (t_(i+1) - t_i))`

Intervals that cross a measurement-window boundary are linearly clipped at the boundary. Report joules per measured optimizer step and:

`energy_per_1000_tokens = 1000 * E_joules / measured_non_padding_tokens`

Also report starting temperature, maximum run temperature, median measured-window power, maximum power, and time-weighted mean power. Missing driver fields remain null with an availability reason; they are never zero-filled.

## Training and integrity metrics

Record loss for every step, finite-loss status, gradient finite status, adapter dtype and placement evidence, optimizer-state placement, exact optimizer-step count, adapter byte count, adapter content hash, artifact-manifest hash, worker terminal state, process exit, attention observations, environment drift, and post-run GPU memory release.

A metric record is valid only when linked to repository commit, worker wheel, environment lock, capability report, execution probe, RunPlan, resolved execution configuration, model, tokenizer, chat template, dataset, cell, trial, run, and telemetry hashes.

## Feasibility and maximum sequence

A feasibility cell passes only under the success definition in `FAILURE_TAXONOMY.md`. Maximum feasible sequence is the greatest preregistered length with a passing three-step smoke and all three successful characterization trials under the common configuration. Also report the greatest length with smoke success alone; do not conflate the two.

A nonmonotonic result is reported as observed. No interpolation or inference fills unrun lengths.

## Summaries and comparisons

For each successful cell and metric, report individual trial summaries, mean, median, sample standard deviation (`n-1` denominator), minimum, maximum, sample count, and a two-sided 95% Student-t confidence interval for the trial mean. For `n=3`, use `t_(0.975,2) = 4.3026527299`:

`CI = mean +/- 4.3026527299 * sample_sd / sqrt(3)`

With fewer than three successful planned trials, report the available values and no confirmatory confidence interval. Do not treat optimizer steps as independent trials.

For matched math/flash trials, compute both `flash - math` and `flash / math` by trial index before summarizing. A ratio is null when its denominator is zero or unavailable. Failure comparisons use counts and exact classifications, not numeric imputation.

## Rounding and presentation

Raw JSON and CSV retain full collected precision. Tables may round time to three decimals, throughput to two decimals, memory to whole MiB, power to two watts, energy to two joules, and temperature to one degree C, while retaining links to raw values. Calculations always use unrounded inputs.
