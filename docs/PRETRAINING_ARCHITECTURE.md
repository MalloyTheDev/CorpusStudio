# Pretraining Architecture

**Status: architecture proposal for review. Docs-only. No pretraining code, dependency, model, dataset,
GPU, or corpus action is part of this document.** Part of
[`TRAINING_SYSTEMS_ARCHITECTURE.md`](TRAINING_SYSTEMS_ARCHITECTURE.md).

Pretraining (from-scratch and continued) is a **first-class objective**, not an SFT config with a
different mask. The `ObjectiveKind.pretraining` family already exists; the missing pieces are the
**data, budget, and resume** contracts. Today's `TrainingDataPolicy` is SFT-only
(`dataset_format: Literal["instruction","chat","trace"]`, a single in-memory `examples.jsonl`), which
cannot express a sharded, streamed, mixture-weighted corpus. Do **not** reuse the SFT-only contract where
pretraining semantics differ.

## 1. Initialization

A pretraining run declares exactly one init mode:

- **random initialization** - architecture config + init scheme + seed; no source weights. The
  `ModelDescriptor` source is a config, not a checkpoint.
- **continued-pretraining initialization** - init from an existing checkpoint (bound by content hash),
  with an explicit statement of what is reset (optimizer, scheduler, data cursor) vs carried.

Both are content-addressed and recorded in the plan's immutable inputs.

## 2. Tokenizer lifecycle

Pretraining may **train** a tokenizer, not just consume one. Three explicit modes, each frozen before any
token is consumed (a tokenizer change invalidates all downstream token accounting):

- **train** - corpus sample -> algorithm/vocab-size/special-tokens -> new tokenizer; sealed by hash.
- **import** - an existing tokenizer bound by `tokenizer_content_sha256` (as SFT does today).
- **freeze** - the exact tokenizer identity that produced the token stream, pinned into the plan and
  every checkpoint (`CheckpointBoundIdentities.tokenizer_ref` already exists).

See [`MODEL_TOKENIZER_CONTRACTS.md`](MODEL_TOKENIZER_CONTRACTS.md).

## 3. Corpus data contract (the G1 gap)

A new `PretrainingDataPolicy` (additive; parallel to `TrainingDataPolicy`) declares:

- **shards** - an ordered, content-hashed shard set (path/id + row/token counts + sha256 per shard).
- **streaming** - shard iteration without full materialization; bounded memory; single-writer per shard.
- **data-mixture weights** - per-source sampling weights, with the exact realized mixture recorded as
  evidence (planned vs consumed).
- **document boundaries** - explicit document separators / attention-reset policy so packing never leaks
  across documents unless declared.
- **sequence packing** - pretraining packing semantics (concat-and-split vs best-fit), distinct from the
  SFT `packing: bool`.
- **deterministic sample order** - a seeded, reproducible global order across shards (a `data_seed`
  already exists in `CheckpointBoundIdentities`), so a resume reproduces the exact stream.
- **token accounting** - exact non-padding / supervised / total token counts per step and cumulative.
- **global batch + token budget** - global batch size (across all data-parallel ranks) and a target
  total-token budget; the run stops at the budget, never silently truncates.

## 4. Optimization

Pretraining optimization is declared explicitly (no product defaults substituted for research/pretraining
choices): optimizer, LR schedule, warmup, gradient clipping, weight decay, and precision. These reuse the
shipped `OptimizerSpec` / `TrainingSchedule` / `PrecisionExecutionPolicy` contracts; the pretraining
objective's `ObjectiveUpdatePolicy` is `all_parameters` with a global optimizer clock (dense) or a
per-expert clock (MoE, see [`MOE_TRAINING_ARCHITECTURE.md`](MOE_TRAINING_ARCHITECTURE.md)).

## 5. Checkpoint / resume with a data cursor (the G2 gap)

`SealedTrainingState` today places a checkpoint on the optimizer-step / microstep / epoch timeline for a
finite dataset. Streaming pretraining additionally requires a **data cursor**:

- shard id + intra-shard offset (or a resumable iterator state),
- cumulative consumed tokens,
- the realized mixture position,

so a resume continues from the **exact** token, never an approximate one. This extends the checkpoint
contract; the exact-lineage rules in [`CHECKPOINT_RESUME.md`](CHECKPOINT_RESUME.md) (bitwise-equivalent
resume) apply unchanged.

## 6. Evidence (what a pretraining run must prove)

- **validation loss** on a held-out shard on the declared schedule.
- **consumed-data evidence** - which shards/documents/tokens were actually consumed (planned vs realized
  mixture), so a claim is bound to what the model actually saw.
- **compute / memory / power / energy** - via the shipped telemetry suite (`RunTelemetrySummary`,
  `EnergyIntegration`, `EventMetrics`), with token-throughput at the un-bypassable consumption boundary
  (the v6/v7 lesson: unavailable is null, never a fabricated zero).
- **exact token accounting** every measured step (positive non-padding + supervised counts; rates ==
  observed tokens / duration).

## 7. What is implemented vs planned

| Capability | Support level |
|---|---|
| `ObjectiveKind.pretraining` family + label/mask/update semantics | contract shipped (`DECLARED`) |
| `PretrainingDataPolicy` (shards/streaming/mixture/boundaries/budget) | **planned (P1)** |
| Data-cursor checkpoint extension | **planned (P1)** |
| Tokenizer train/import/freeze | import shipped; train **planned** |
| Dense small-model pretraining run | **planned (P1)**, gated by `WORKLOAD_VERIFIED` |
| Continued pretraining | **planned (P2)** |
| Consumed-data + validation-loss evidence | telemetry suite shipped; pretraining wiring **planned** |

No pretraining workload is claimed until it is workload-verified on real hardware.
