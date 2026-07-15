# Telemetry scientific-completeness - field-by-field root cause

**Purpose.** Before spending another GPU result, identify the *exact* reason each required paper
telemetry field was missing from the preserved v5 chat math run, so the fix is targeted and not a
guess. This is the evidence the corrective telemetry PR is built on.

**Evidence base (read-only, not mutated):**
- Preserved failed run: `.../v5-bringup-713fde1/out-math-chat/runs/run-019f67e6-8d14-782a-8b10-ecdb4e74a6e0/`
  (`RunTelemetrySummary.json`, `RunEvents.jsonl`, `TelemetrySamples.jsonl`, `RunManifest.json`).
- Source read at commit `4078825` (post-#461).

## As-observed completeness (verbatim from the preserved summary)

```
scientifically_complete: false
missing_required_paper_fields:
  - gpu.memory
  - identity.execution_configuration_hash
  - identity.repository_commit
  - identity.worker_wheel_sha256
  - step.step_time_seconds
telemetry_degraded: true   degraded_sample_count: 77
```

`REQUIRED_PAPER_FIELDS` (telemetry.py:73) has 12 entries. Seven were already satisfied by this run:
`step.step_losses`, `gpu.power_watts`, `energy.run_joules`, **`host.process_tree_rss`**,
`identity.environment_lock_hash`, `identity.plan_hash`, `identity.run_id`. Five were missing.

### Two corrections to earlier assumptions (verified against the raw records)

1. **`host.process_tree_rss` is NOT missing.** The raw samples carry `process_tree_rss_bytes` in
   77/77 samples; the summary aggregates it correctly (`host.process_tree_rss.whole_run_max
   .process_rss_bytes = 2534219776`, baseline `82542592`). It is not in the missing list. The earlier
   belief that host RSS was dropped confused the *null GPU sub-fields* inside that
   `MemoryWindowSummary` (`torch_allocated_bytes` etc., null because the parent has no torch) with the
   RSS value itself (populated). **No RSS fix is needed.**
2. **Swap is already present and is not a required paper field.** The host summary carries
   `swap_used_bytes` and `swap_used_delta_bytes` (raw swap in 77/77 samples); `REQUIRED_PAPER_FIELDS`
   contains no swap entry. **No swap fix is needed for completeness** (a small hardening is optional).

## Field-by-field root cause

### 1. `gpu.memory` - MISSING (real gap)

`_present(summary, "gpu.memory")` (telemetry.py) requires
`gpu.memory.whole_run_max is not None`. `_gpu_summary` derives that window **only** from
`TelemetrySample.memory` (`mem_all = [s.memory for s in all_samples if s.memory is not None]`). Every
sample's `.memory` was null, so `mem_all` was empty -> `whole_run_max = None`.

**Why every sample's `.memory` is null:** the parent `TelemetrySampler` (runs in the control-plane
subprocess-supervisor process, torch-free venv) sources memory from
`watchdog.sample_gpu_memory()` (watchdog.py:50), whose first act is `import torch`; on `ImportError`
it returns `None`. So the parent, by construction, can never populate `.memory`. `probe_gpu`
(telemetry.py) then appends `"gpu_memory"` to `probe_unavailable` (observed 77/77).

The CUDA-owning **worker child** *does* sample memory (via `RunWatchdog.memory_sampler` for the peak /
final-fit reconciliation), but that value is never emitted per-sample into the telemetry stream nor
per-step into `EventMetrics.memory`; it only feeds `ctx.measured_peak`.

**Root cause:** the only field the completeness check reads is populated by a torch probe running in
the torch-free parent. Two independent, honest sources now feed it:
- **Parent / driver-side (device memory) - FIXED:** `probe_gpu` now also queries
  `--query-gpu=memory.used,memory.total` and folds the driver device used/free into
  `TelemetrySample.memory` (`_nvidia_smi_device_memory` + `_combine_memory`, MiB -> bytes). This is
  real in the torch-free parent, so `gpu.memory.whole_run_max` is populated; null (never zero) when
  nvidia-smi is absent.
- **Worker-side (torch allocator memory) - FIXED:** the runner's per-step `_progress` (which runs in
  the CUDA-owning child) samples `self.memory_sampler()` and emits it on `EventMetrics.memory`, so the
  torch allocator view is persisted per step into `RunEvents.jsonl` - the authoritative process
  allocator evidence the driver cannot give.

`MemoryMetrics` already carries both families (`cuda_device_*` and `torch_*`); no contract change.

### 2. `step.step_time_seconds` - MISSING (real gap)

`_step_rows` reads `metrics.step_time_seconds` off each `event_type == "metric"` RunEvent. The trainer
callback `on_log` (trainer.py:1979) records **loss only**; `on_step_end` (trainer.py:1953) verifies
optimizer identity/sequence but captures **no timestamp**. The platform bridge
`_progress(step, total, loss)` (runners.py:217) emits `EventMetrics(loss=loss)` with no timing, because
`ProgressCallback = Callable[[int, int, float | None], None]` (trainer.py:54) has no slot for it. So
`metrics.step_time_seconds` was `None` for every step -> `step.step_time_seconds = None`.

**Root cause:** wall-clock per-step duration is never measured at the optimizer-step boundary. **FIXED
runner-side (no trainer change):** the runner's `_progress` runs in the CUDA-owning child and fires
once per optimizer step (sealed `logging_steps=1`); it records `time.monotonic()` and emits the delta
to the previous boundary as `EventMetrics.step_time_seconds`. Step 1 has no prior boundary, so it is
null (honest); measured steps (3-12) all carry a real duration, so `step.step_time_seconds` is
populated.

### 3. token throughput (`nonpadding_tokens_per_second`, `supervised_tokens_per_second`) - MISSING

Not in `REQUIRED_PAPER_FIELDS` (so it did not gate completeness), but explicitly requested. Same
mechanism as step timing: the trainer never counts tokens and the callback cannot carry them, so
`EventMetrics.tokens_per_sec / supervised_tokens_per_sec` are always `None`.

**Root cause:** no per-step token accounting. **FIXED (guarded, degrade-to-null):** a
`_TokenObservingSFTTrainer` wraps the sealed dataloader's `collate_fn` in a strict pass-through that
only *reads* each collated microbatch (`count_batch_tokens`: non-padding = attention-mask ones,
supervised = labels != -100) into a `_TokenAccumulator`; the step-boundary callback flushes the
per-step total (correct because the sealed single-worker dataloader has no background prefetch) through
a separate `token_callback`, and the runner derives `tokens_per_sec = tokens / step_time_seconds` into
`EventMetrics`. The batch object is returned unchanged and every fault is swallowed, so token
accounting can never alter or fail training - it degrades to null token evidence. The pure counter and
accumulator are unit-tested; the collator wiring lives in the pragma-`no cover` `run_training` and is
first exercised end-to-end on the managed GPU smoke (null-token fallback if the wrapper is a no-op on
a given TRL/accelerate build). Token throughput is not a `REQUIRED_PAPER_FIELD`, so completeness never
depends on it.

### 4. `identity.execution_configuration_hash` - MISSING (real gap)

`identity_from_plan` (telemetry.py:664) does
`getattr(execution, "execution_configuration_hash", None)`. `ResolvedExecutionConfiguration`
(contracts.py:3569) has **no such attribute**, so the default `None` always fires. The value is not
stored on the plan or manifest; it is a *computed* hash (canonical function
`execution_configuration_hash_for(config)` at execution_config.py:94, torch-free, already used by the
worker/runner/supervisor at verify time).

**Root cause:** the summary read a non-existent attribute instead of the real field. **FIXED:**
`identity_from_plan` now reads `execution.configuration_hash` (the sealed value; equal to a fresh
`execution_configuration_hash_for` recomputation by `verify_execution_configuration_hash`).

### 5. `identity.repository_commit` and `identity.worker_wheel_sha256` - MISSING (real gap)

Both are **overlay-only** in `identity_from_plan` (never derived from plan or manifest). The
`platform-run` command (cli.py:675) calls `summarize_run_telemetry(..., plan=plan)` with **no
`identity_overlay`**, so both stay `None` on every live run. The values exist at run time in the
sealed backend/environment manifest (wheel sha256) and the pinned worker source (repo commit).

**Root cause:** no identity overlay is threaded from the sealed backend manifest into the summary.
**FIXED:** `platform-run` now builds a `TelemetryIdentity` overlay from the resolved environment lock
(`worker_identity_overlay`): `worker_wheel_sha256 = lock.worker_artifact.content_hash.value`, and
`repository_commit` read best-effort from the wheel's `BUILD_PROVENANCE.json` `source_commit` sidecar
(null, never fabricated, if absent). The overlay merge also now guards the manifest-authoritative
resume-lineage fields so an overlay's default `resumed=False` can never clobber a resumed trial.

## Worker-identity impact

The step-time + worker-allocator emission (`platform/runners.py::_progress`) and the token observer
(`training/trainer.py`) run inside the managed child, so per the `WORKER_IMPACT_proof.md` reasoning
they change worker execution bytes. This is intentional and consistent with the v6 lineage decision:
the artifact fix (#461) already made a new worker wheel mandatory, and this telemetry work is folded
into the *same* v6 lineage so a single new wheel carries both. The identity fixes
(`identity_from_plan`, `worker_identity_overlay`, the cli overlay threading) and the parent/driver
device-memory probe (`probe_gpu`) are control-plane summary/probe changes.

## What shipped

- `platform/telemetry.py`: `identity_from_plan` reads the real `configuration_hash`; overlay merge
  guards resume lineage; `worker_identity_overlay` + `_build_provenance_source_commit`; `probe_gpu`
  device memory via `_nvidia_smi_device_memory` + `_combine_memory`.
- `cli.py`: threads the worker-artifact overlay into `summarize_run_telemetry`.
- `platform/runners.py`: `_progress` emits per-step wall time + worker allocator memory + derived
  token rates.
- `training/trainer.py`: `count_batch_tokens` / `_flatten_ints` / `_TokenAccumulator` (unit-tested) +
  a guarded `_TokenObservingSFTTrainer` + step-boundary `token_callback` flush.
- Tests: token counting, identity/overlay/device-memory unit tests, a runner per-step-emission test,
  and an end-to-end `scientifically_complete=true`-from-plan+overlay fixture.

**No honesty invariant is weakened.** Every added field is emitted from a real measurement, is null
(never zero-filled) when its source is unavailable, and `scientifically_complete` still becomes true
only when the required fields are genuinely present.
