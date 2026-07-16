# Amendment 0004 - v7 worker lineage (token-throughput observer + throughput-honesty corrections)

- **Amendment id:** `cs-ieee-linux-training-amendment-0004`
- **Study:** `cs-ieee-linux-training-v1`
- **Base protocol version:** 1.0.0
- **Effective protocol version:** 1.4.0 (supersedes 1.3.0)
- **Status:** prospective (authored before any v7 wheel build, environment, plan, or run)
- **Authored:** 2026-07-16
- **Analysis role:** primary

This amendment is **append-only**. It does not edit `PROTOCOL.md`, `EXPERIMENT_MATRIX.yaml`, amendments
0001-0003, effective matrices 1.1.0-1.3.0, or `RESERVED_IDENTITIES.v1`/`.v2`/`.v3` in place. It adds
effective matrix **1.4.0**, reserved-identity registry **v4** (a strict append-only superset of v3),
and this narrative + manifest. The prior amendment 0003 is bound by exact raw-file hash in the manifest
`supersedes` block, so the amendment chain stays ordered and 0003 stays provably byte-frozen.

## Why this amendment exists

The v6 0.5B bring-up (effective matrix 1.3.0, worker wheel
`bdc32196203539cbeb9078ce2317fb41d2a30abe68f7e94bc0fa290a97f414d4`, worker source
`73b756c49da0f03203ebd05dfb5528805b0fd280`) produced the study's **first fully admitted end-to-end
success** on the native-Linux RTX 5070. The visible, preserved facts:

1. Both arms completed **12 QLoRA optimizer steps** with monotonically decreasing loss, serialized a
   clean LoRA adapter that was **admitted**, and measured **`NATIVE_SAFE`** - forced `torch_sdpa_math`
   (`run-019f688c-67c0-77cf-82e2-477f52fab76f`) and forced `torch_sdpa_flash`
   (`run-019f6892-3a54-7922-8e10-d138ee7e77ce`). **Verdict: `V6_MATH_AND_FLASH_BRINGUP_PASS`.**
2. One honestly-recorded gap: `nonpadding_tokens_per_second` and `supervised_tokens_per_second` read
   `0.0` on both runs.

**The recorded `0.0` cannot be treated as a measured zero.** A real supervised optimizer step
necessarily processes tokens, so the semantically correct value is **unavailable / null**, not zero. Its
durable classification is `TOKEN_THROUGHPUT_UNAVAILABLE_OBSERVER_MISSED_BATCHES` (sidecar preserved under
`.../evidence/v6-smoke-73b756c/`). The v6 raw records are **preserved unchanged**; this amendment neither
edits nor reinterprets them - it only carries the corrected classification forward and reserves the v6
identities.

## Root cause (proven on the pinned stack)

On the pinned stack (trl 1.8.0 / transformers 5.13.1 / accelerate 1.14.0 / datasets 5.0.0 /
torch 2.11.0+cu128), `transformers.Trainer._get_dataloader` returns an accelerate-prepared
`DataLoaderShard` whose base loader captured `collate_fn` at `accelerator.prepare` time. The #462 (v6)
observer reassigned `.collate_fn` on the returned shard, which is silently bypassed, so the observer
never fired and both token counts flushed `0`. A reproducible CPU experiment on the pinned stack
confirmed: the collate-wrap observed **0** batches; observing `inputs` at `training_step` observed the
real collated batch with positive supervised counts.

## The correction and why it forces a fresh worker lineage

The fix landed on `main` as **PR #466** (merge commit
`25c901ec85fd6f6303eff6c3dd81938afe328a2b`):

- Token accounting now observes `inputs` at `SFTTrainer.training_step` - the trainer's own
  un-bypassable consumption boundary - with a pure, vectorized, read-only count (non-padding =
  `attention_mask != 0`; supervised = `labels != -100`). It never moves, mutates, or retains the batch,
  and can never disturb training.
- Each optimizer step emits raw `nonpadding_tokens` / `supervised_tokens` / `observed_microbatches`;
  rates are derived from those counts. `observed_microbatches == 0` yields **null (unavailable)** counts,
  never a fabricated `0.0`.
- A throughput-validity gate plus separable completeness dimensions
  (`scientific_resource_complete` / `scientific_throughput_complete` / `paper_performance_complete`)
  keep a workload success with incomplete instrumentation distinct from a paper-ready measurement;
  `scientifically_complete` is kept equal to the resource dimension for backward compatibility. The fix
  was verified on the real pinned stack (`INTEGRATION_PASS`; per-step supervised counts 40/39/42;
  collate-wrap fired 0).

**The token observer runs inside the managed worker child** (`worker.py -> execute_run` ->
runner/trainer `training_step`). It therefore changes worker execution bytes. Because the v6
environments are sealed to the immutable wheel `bdc32196...`, they cannot be patched in place and must
not be recreated under the same ids. **A fresh v7 wheel, v7 environments, v7 plans, and v7 runs are
required.**

## What 1.4.0 changes

- `first_party_execution_paths[first-party-math].environment_id` ->
  `backend-corpus-studio-research-math-v7`
- `first_party_execution_paths[first-party-flash].environment_id` ->
  `backend-corpus-studio-research-flash-v7`
- `schema_version` / `protocol_version` -> `1.4.0`
- `worker_success_admission` gains the token-observation boundary, the null-not-zero rule, the
  rate-equals-observed-tokens-over-duration rule, and `paper_performance_complete` as a paper-promotion
  requirement.

The v7 worker source must **descend from `25c901ec85fd6f6303eff6c3dd81938afe328a2b`** (the merged token
fix, which descends from the v6 floor `73b756c`, the 0003 floor `af28be9`, and the `df86db5` floor). To
avoid an impossible self-referential commit pin, the amendment does not pin an exact v7 commit; it
requires that the final wheel source descend from `25c901ec`, that the exact final post-amendment source
commit be recorded in the wheel evidence, that the exact wheel sha-256 be recorded in each environment
and trial, and that historical worker-wheel reuse is prohibited.

## What 1.4.0 does NOT change (the scientific tuple is preserved)

Qwen2.5-0.5B bring-up model; chat fixture; sequence length 256; microbatch 1; gradient accumulation 1;
12 optimizer steps; QLoRA r16 / alpha 32 / dropout 0.05 / bias none / all-linear; NF4 + double
quantization; BF16 compute; FP32 adapter weights and materialized gradients; `adamw_torch`;
checkpoint-free; no offload; no compile; no truncation; no packing; forced math versus forced flash. The
only differences between the matched math and flash arms remain the attention capability/probe, the SDP
toggles, and the environment identity.

## Reserved identities

`RESERVED_IDENTITIES.v4.json` is an append-only strict superset of v3. Beyond every v1-v5 identity it
already carried, it reserves the fully instantiated v6 identities: the v6 worker wheel
(`bdc32196...`) and source commit (`73b756c...`); the `math-v6`/`flash-v6` environment ids and lock
hashes; both v6 matched chat plan ids, plan hashes, execution-configuration ids and hashes; both
dispatched v6 run ids; both admitted v6 adapter artifact ids; and the v6 output and evidence roots. No
v1-v6 identity may be reused.

## v7 complete-pass requirement (throughput now gated)

A v7 pass requires every v6 success criterion (12 completed steps per arm, decreasing loss, adapter
admitted, measured `NATIVE_SAFE`) **plus** valid token accounting: positive non-padding AND supervised
token counts observed on **every** measured optimizer step, a positive step duration, token rates that
equal observed tokens / observed duration, token-normalized energy, `scientific_throughput_complete` and
`paper_performance_complete` both true. Missing observation is reported as null with a typed reason,
never a plausible-looking zero.

## Preregistration boundary (unchanged)

Feasibility characterization remains distinct from the gated full-training phase over the
~500-output corpus; that phase is not authorized here. This amendment authorizes only a fresh v7 0.5B
feasibility bring-up (one math smoke, one conditional flash smoke). No 7B model is loaded and the
sequence-length ladder is not started.
