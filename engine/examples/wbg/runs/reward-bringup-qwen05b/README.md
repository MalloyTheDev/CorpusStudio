# Reward-model workload_verified bring-up (RL slice S5a PR 3c-2)

Durable evidence promoting the `reward_model` execution variant from `contract_validated` to
**`workload_verified`** on this host. A pairwise reward model trained through the **full managed
`execute_run` -> `RewardRunner` -> supervisor independent adapter reload-verify** path on the RTX 5070.

## What ran

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`, nf4 QLoRA **SEQ_CLS** (a scalar score head over the base).
- **Objective / loss:** `reward_model` / Bradley-Terry pairwise `-log sigmoid(s_chosen - s_rejected - margin)`.
- **Data:** the chosen/rejected preference pairs used for the DPO bring-up (12 pairs). The worker carves a
  deterministic seeded held-out split (by the sealed `data_seed`): **10 train / 2 held-out**.
- **Schedule:** seq 512, 20 optimizer steps, one pair per microbatch.
- **Path:** `execute_run(plan, RewardRunner())` - the same admission every other variant uses. The runner
  reports worker-proposed evidence; the supervisor independently reloads the saved adapter, re-hashes it,
  and compares to the sealed trained export state before admitting.

## Measured result (this preserved run)

| signal | value |
| --- | --- |
| terminal state | `succeeded` (supervisor-admitted) |
| optimizer steps | 20 (matches the sealed schedule) |
| loss | 0.7563 -> 0.0 (Bradley-Terry) |
| score margin | -0.07 -> 46.65 (chosen 33.52 / rejected -13.13) |
| trainable tensors | 337 / 337 changed **and** observed a gradient (LoRA + the randomly-initialized score head) |
| adapter re-verify | `adapter_bytes_verified = True` (sha `b6c5c713...` reproduced from bytes) |
| **PROMOTION GATE** | **held-out pairwise ranking accuracy = 1.0 over 2 held-out pairs** (never a falling training loss) |
| measured fit | `NATIVE_SAFE` - peak 0.95 GiB allocated / 1.2 GiB reserved of 12.3 GiB (11.1 GiB free) |

The randomly-initialized SEQ_CLS score head (`score.weight` reported MISSING at load, newly initialized)
is trained because the adapter is sealed `adapter_task_type=SEQ_CLS`, so PEFT keeps it in `modules_to_save`.

## Honesty scope

A **PRODUCT** claim, not a sealed IEEE cell. Reward modeling is cheaper than DPO - no reference model, no
`[seq x vocab]` log-prob. The managed `platform-run --subprocess` route (a reward worker wheel + a sealed
env) is the deployment follow-up, exactly as for pretraining/DPO. See `docs/HOST_STATE.md`.

## Files

- `runs/run-reward-sealed-0001/RunManifest.json` - the supervisor-written terminal record (carries
  `reward_success_evidence` + the measured `final_fit`).
- `runs/run-reward-sealed-0001/RunEvents.jsonl` - the run's event stream.
- `reward-bringup.RunPlan.json` - the sealed `RunPlan` (its `resolved_reward_execution` is what the worker
  consumed).

The multi-MB adapter weights are not preserved here - the RunManifest + ArtifactManifest carry the sha256
digests that identify them.
