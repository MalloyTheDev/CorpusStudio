# Failure taxonomy and success criteria

The platform `FailureTaxonomy` is the top-level vocabulary. Research reason codes add detail without changing the platform outcome. The first terminal cause observed in the authenticated event stream is primary; consequential failures are retained as secondary observations.

## Success

`PASS` requires all of the following:

- exact planned optimizer-step count;
- finite loss and gradients at every required step;
- authenticated terminal worker success and clean process exit;
- forced attention path observed with prohibited paths disabled and no fallback;
- exact model, tokenizer, dataset, environment, worker, plan, and configuration identities;
- valid placement, quantization, adapter, optimizer-state, and precision observations;
- required adapter Safetensors and artifact manifest with valid content hashes;
- no accidental spill, offload, environment drift, safety stop, or integrity failure;
- post-run environment health and GPU memory-release checks complete.

A clean exit without these facts is `FAIL`, not `PASS`.

## Platform outcomes

| Outcome | Required interpretation | Example reason codes |
|---|---|---|
| `OOM` | CUDA or host allocation failure attributable to the sealed workload. | `CUDA_ALLOC_OOM`, `HOST_MEMORY_OOM` |
| `TIMEOUT` | Declared wall-time or per-step limit reached while the GPU and progress channel remain responsive. | `TRIAL_WALL_LIMIT`, `STEP_WALL_LIMIT`, `COOLDOWN_DEFERRED` |
| `KERNEL_STALL` | No legitimate progress plus an unresponsive/stalled kernel or supervisor silence classification supported by evidence. | `GPU_UNRESPONSIVE`, `SUPERVISOR_SILENCE`, `ATTENTION_KERNEL_STALL` |
| `NUMERICAL_FAILURE` | Nonfinite loss, gradients, optimizer state, or invalid numerical output. | `NONFINITE_LOSS`, `NONFINITE_GRADIENT`, `NONFINITE_OPTIMIZER_STATE` |
| `CHECKPOINT_FAILURE` | A required checkpoint or adapter serialization/reload/integrity operation fails. Short trials request no intermediate checkpoint. | `ADAPTER_SAVE_FAILED`, `SAFETENSORS_INVALID`, `ARTIFACT_HASH_MISMATCH` |
| `ENVIRONMENT_FAILURE` | Environment identity, health, dependency, package source, driver/CUDA, or worker identity is invalid. | `PRE_RUN_DRIFT`, `POST_RUN_DRIFT`, `LOCK_MISMATCH`, `WORKER_MISMATCH`, `PACKAGE_DRIFT` |
| `UNSUPPORTED_CONFIGURATION` | The planner, capability gate, or sealed worker correctly refuses the exact tuple. | `CAPABILITY_MISSING`, `PLANNER_REFUSAL`, `PLACEMENT_DEVIATION`, `BACKEND_NOT_ADMITTED` |
| `ACCIDENTAL_SPILL` | Unplanned host/unified-memory paging or offload occurs. | `UNIFIED_MEMORY_PAGING`, `UNDECLARED_CPU_OFFLOAD`, `DISK_OFFLOAD_DETECTED` |
| `CONTROLLED_OFFLOAD` | Declared offload is observed. Primary cells require `offload=none`, so this is not a primary-cell pass. | `DECLARED_PARAMETER_OFFLOAD`, `DECLARED_OPTIMIZER_OFFLOAD` |
| `FAIL` | A terminal failure not truthfully covered above. It must carry a specific reason and evidence. | `IDENTITY_MISMATCH`, `ATTENTION_FALLBACK`, `STEP_COUNT_MISMATCH`, `INSTRUMENTATION_FAILURE`, `SAFETY_STOP` |

## Safety-stop mapping

- GPU temperature at 85 C: `FAIL/SAFETY_STOP_GPU_TEMPERATURE`.
- Host process-tree RSS at the lower of 56 GiB or 90% of detected physical memory: `FAIL/SAFETY_STOP_HOST_RSS` unless an allocation failure already establishes `OOM`.
- Sustained swap growth: `ACCIDENTAL_SPILL/SUSTAINED_SWAP_GROWTH`.
- GPU unresponsive for three polls with no progress: `KERNEL_STALL/GPU_UNRESPONSIVE`.
- Per-step or trial wall limit with responsive progress: `TIMEOUT`.
- Artifact or input hash mismatch: `FAIL/ARTIFACT_INTEGRITY` or `ENVIRONMENT_FAILURE` when the mismatched artifact is the environment/worker.

## Stage

Every terminal classification records the last verified stage: admission, immutable-input verification, dataset formatting, truncation analysis, model load, placement verification, k-bit preparation, adapter insertion, post-adapter verification, optimizer creation, warm-up, forward, loss, backward, optimizer step, adapter save, manifest verification, or post-run health.

## Retry and incomplete evidence

OOM, stall, drift, identity, integrity, numerical, unsupported, and safety outcomes are never automatically retried. One identified infrastructure-only interruption may be retried once with the identical sealed plan; the original remains in the raw results and the replacement links to it.

If telemetry or terminal evidence is incomplete, the trial cannot pass. Classify the known primary failure when supported; otherwise use `FAIL/INCOMPLETE_EVIDENCE`. Unknown and null are not synonyms for zero or success.
