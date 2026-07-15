# Protocol: native-Linux consumer-GPU training

Protocol version: `1.0.0`

Preregistered: 2026-07-14

Study ID: `cs-ieee-linux-training-v1`

## Scope and research questions

This is a single-host systems characterization of bounded adapter training. It is not a model-quality study.

1. On the registered native-Linux RTX 5070 host, how do forced PyTorch math SDPA and forced PyTorch flash SDPA differ in optimizer-step time, throughput, CUDA memory, process GPU memory, host RSS, power, energy, temperature, and maximum feasible sequence length?
2. How does QLoRA feasibility and performance change at sequence lengths 512, 1024, 2048, 3072, and 4096?
3. How do independently verified training backends compare when model, tokenizer, dataset, adapter, optimizer, attention meaning, batching, and seeds are matched?
4. How do LoRA and QLoRA differ in the selected secondary cells?
5. Which registered model and sequence combinations are feasible within 12,227 MiB of VRAM under the exact sealed configurations?

A refusal, OOM, stall, timeout, or unsupported result at sequence length 4096 is a reportable result. Shorter-sequence success is never extrapolated to 4096.

## Study stages

The stages are ordered. Results from a later stage are not collected until its execution path has independent capability evidence and a passing three-step production-path smoke.

1. Close workload liveness, checkpoint, and environment-concurrency blockers.
2. Build matched first-party math and flash environments from identical worker-wheel bytes and matched package versions.
3. Pass one separately sealed three-step 0.5B smoke in each first-party environment.
4. Implement and validate the sealed-plan experiment harness.
5. Execute the primary first-party QLoRA matrix.
6. Implement and verify Unsloth as a real isolated backend before admitting any Unsloth cell.
7. Execute the selected LoRA-versus-QLoRA secondary matrix.
8. Generate deterministic aggregates and paper-ready outputs from immutable raw evidence.

## Independent variables

Primary matrix:

- model: Qwen2.5 Instruct 0.5B, 3B, or 7B;
- sequence length: 512, 1024, 2048, 3072, or 4096;
- forced attention path: PyTorch math SDPA or PyTorch flash SDPA.

The primary matrix contains 30 cells. Adapter method is QLoRA in every primary cell.

Secondary matrix:

- model: 0.5B or 3B;
- sequence length: 1024 or 4096;
- adapter method: LoRA or QLoRA;
- execution path: first-party math, first-party flash, and Unsloth only where independently supported.

The 7B LoRA cell receives a predicted-fit decision first and is attempted only if the exact sealed environment and planner classify the bounded run as viable. A refusal remains in the matrix.

## Controlled variables

Within every comparison, the following are byte-identical or semantically identical and recorded: model snapshot and file hashes; tokenizer and chat-template hash; dataset bytes, row order, rendered examples, masks, and labels; sequence and padding; packing and truncation policy; microbatch and gradient accumulation; adapter rank, alpha, dropout, bias, and targets; optimizer, scheduler, learning rate, loss, and gradient clipping; gradient checkpointing; seeds; checkpoint policy; worker wheel; package versions; driver; CUDA runtime; output-layout semantics; and telemetry configuration.

Math and flash cells may differ only in environment lock, attention kernel and toggles, and the capability evidence necessarily bound to that environment. The environments must use the same worker-wheel bytes and matched package versions. Batch size is not changed between math and flash for a comparison.

The license-cleared source corpus and deterministic sequence views are frozen and hash-pinned before the first new matrix result. A view is keyed by model-tokenizer hash and declared sequence length, and every materialized example has exactly that non-padding input length without runtime truncation or packing. Math, flash, and admitted backend/method comparisons for the same model and sequence consume byte-identical views in the same seeded order. Views differ across sequence length only by the prospectively defined deterministic materialization needed to make sequence length the manipulated variable; content is not adapted in response to performance results.

## Sealed training configuration

The primary configuration is SFT with QLoRA, NF4, nested/double quantization, BF16 quantization compute, BF16 forward autocast, `r=16`, `alpha=32`, dropout `0.05`, no bias, and the `all-linear` target policy. It uses `adamw_torch`, learning rate `2e-4`, beta1 `0.9`, beta2 `0.999`, epsilon `1e-8`, zero weight decay, maximum gradient norm `1.0`, a linear scheduler, and no optimizer warmup. Gradient checkpointing is enabled. Offload, compile, packing, truncation, remote code, external flash-attn, and attention fallback are disabled.

No before-run, during-run, or after-run model-quality evaluation is part of a performance trial. Dataset shuffling is deterministic from the sealed data seed, and the same order is used by paired execution paths.

The default microbatch is one and gradient accumulation is one. If an exact pre-run fit analysis requires a different common accumulation value, it must be declared in a prospective protocol amendment and applied to both kernels for the affected comparison; changing accumulation after observing a workload result is prohibited.

Short trials use no intermediate checkpoints. Execution is blocked until the sealed first-party path supports a checkpoint-free strategy. Any planned run expected to exceed 30 minutes is blocked until exact sealed resume semantics exist or an explicit long-run blocker is recorded.

## Cell and trial procedure

Every new model, sequence, and execution-path combination follows this order:

1. Verify a clean audited repository commit and build identity.
2. Verify pre-run environment health, exact lock, capability tuple, immutable inputs, GPU identity, GPU idle state, and safety headroom.
3. Stop loaded Ollama models and verify that no unrelated GPU workload is active.
4. Generate a fresh sealed three-step feasibility RunPlan for the exact environment lock.
5. Run it once. Success requires exactly three optimizer steps, finite loss, terminal worker success, valid adapter and manifest hashes, no fallback or spill, and clean post-run environment health.
6. If feasibility passes, generate the characterization RunPlan. Each characterization trial executes 12 optimizer steps: the first two are warm-up steps and the final ten are the measured window.
7. Run three independent characterization trials using the preregistered seed pairs. Preserve every trial and failure before aggregation.
8. Verify post-run environment health, GPU memory release, terminal state, artifacts, and evidence hashes.

A fresh run ID and run-scoped output directory are required for every execution. A RunPlan is never reused across a new environment lock, worker wheel, model snapshot, dataset hash, or semantics-bearing change.

## Ordering, warm-up, and cooldown

Models run in ascending nominal size and primary sequence lengths run in ascending order for operational safety. Within each matched math/flash pair, kernel order is counterbalanced by `(model_index + sequence_index + trial_index) mod 2`: even means math first; odd means flash first. The feasibility smoke uses trial index zero. Trial seed pairs are matched across kernels.

The first two optimizer steps in each characterization trial are the only performance warm-up exclusion. Model load, preflight, and adapter preparation are measured separately and never folded into steady-state step metrics.

Before each GPU execution, temperature must be at most 45 C and the GPU must have no unrelated compute process. If the host does not cool to the threshold within ten minutes, the execution is deferred and the wait is recorded; the threshold is not relaxed silently.

## Repeats and statistical summaries

The trial is the unit of replication; optimizer steps are repeated observations within a trial, not independent replicates. A successful characterization cell has three trials. Every trial, its ten measured step values, mean, median, sample standard deviation, minimum, maximum, sample count, and two-sided 95% Student-t confidence interval are reported. Kernel comparisons also report matched absolute differences and ratios by trial index. With only three trials, confidence intervals are descriptive and no claim relies on a significance threshold.

No failed result is imputed, replaced with zero, or removed from the matrix. Aggregation begins only after all planned trials or classified failures for the compared cells exist.

## Safety and stopping rules

Only one GPU workload runs at a time. Stop and preserve evidence on any of the following:

- GPU temperature reaches 85 C;
- host process-tree RSS reaches the lower of 56 GiB or 90% of detected physical memory;
- swap use grows by at least 256 MiB from the pre-run baseline and remains nondecreasing for 30 seconds;
- the GPU is unresponsive for three consecutive telemetry polls and no legitimate progress event occurs;
- the sealed supervisor reports a legitimate silence timeout;
- one optimizer step exceeds 15 minutes unless that duration was prospectively declared as the measured subject;
- total trial wall time reaches 60 minutes;
- OOM, numerical failure, kernel stall, environment drift, identity mismatch, artifact-integrity failure, unexpected offload/spill, or attention fallback occurs.

There is no automatic retry for semantic or workload failures. One retry is permitted only for an identified infrastructure-only interruption, using the identical sealed plan. The original attempt is retained and linked to the replacement.

## Exclusions

Only the two preregistered warm-up optimizer steps are excluded from steady-state performance summaries. Telemetry samples strictly before process start or after terminal process exit are excluded from trial integration and retained as baseline/post-run evidence. No valid slow step, thermal result, failure, outlier, or unsupported cell is discarded. Instrumentation-corrupt trials remain preserved and are classified; a replacement requires a versioned rationale and identical workload identity.

Adaptive sequence-boundary lengths such as 1280, 1536, 2560, or 3584 may be run only after the fixed ladder and are labeled exploratory. They never replace a primary matrix cell.

## Claim boundaries

Claims apply only to the recorded native-Linux host, RTX 5070 GPU, driver/CUDA stack, package locks, immutable local inputs, sealed configurations, and bounded step counts. Results do not prove Windows, WSL, macOS, external flash-attn, offload, MoE, other GPUs, other package stacks, full fine-tuning convergence, model quality, or long-duration stability.

A passing environment probe proves only its exact tiny tuple. A passing three-step smoke proves bounded production-path feasibility. A completed 12-step characterization trial remains a bounded workload measurement, not full training. Predicted fit is not measured fit.

## Amendments

After the first new result, protocol text and the primary matrix are not edited in place. Create `amendments/NNNN-YYYY-MM-DD-<slug>.md`, increment the protocol version, and record: author; timestamp; reason; exact changed fields; affected cells; evidence already visible; prospective or retrospective status; compatibility impact; and whether the analysis is primary or exploratory. Earlier protocol versions and result bindings remain available.
