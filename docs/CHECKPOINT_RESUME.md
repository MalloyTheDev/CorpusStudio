# Exact checkpoint + resume lineage (#440)

A long first-party run must be resumable with **exact** lineage or not at all. Two halves realize this:

- **Control plane, torch-free (`platform/checkpoint.py`)** - a hash-sealed `CheckpointManifest`,
  byte-integrity verification that fails closed (including symlink / hard-link / traversal defense), an
  exact resume-request pin, and a resume admission that only proceeds against a byte-identical, fully
  compatible target `RunPlan`.
- **Execution engine, torch-lazy (`training/checkpoint_io.py`)** - the worker half that serializes the
  full resumable state into a run-scoped temp directory, seals a complete manifest, atomically publishes
  the directory, and on resume verifies everything BEFORE loading any tensor, restores the adapter
  weights onto the live parameters, and rebuilds the optimizer over those live parameters.

**Equivalence level (proven).** A real-torch CPU integration
(`tests/test_training_checkpoint_integration.py`) runs N uninterrupted steps, then K steps + checkpoint
+ a **fresh-process** resume of the remaining N-K steps, in three separate processes. Under a controlled
deterministic configuration (fixed seed, single intra-op thread, `use_deterministic_algorithms(True)`,
and full RNG-stream restore) the resumed run reproduces the uninterrupted run **bitwise** - identical
final parameters and identical per-step losses for the shared step numbers, with the first resumed
optimizer step continuing from exactly K+1. This is the strongest defensible level for this
configuration; it demonstrates exact state restoration and is **not** a claim of bitwise equivalence on
GPU, where non-deterministic reductions can perturb the low bits even with correct state restored.

**What stays checkpoint-free.** A short run and the in-process SFTTrainer body remain checkpoint-free
and byte-identical to before - checkpoints are written only under a sealed checkpoint-enabled policy
(`save_strategy="steps"` with a positive optimizer-step cadence), resolved by
`resolve_checkpoint_execution_policy`. Binding the engine into the worker's long-run SFTTrainer step /
sampler / optimizer resume is the first-authorized-GPU-run integration; the engine itself is fully
proven above.

## The sealed checkpoint manifest

`CheckpointManifest` (a root contract; `docs/contracts/CheckpointManifest.schema.json`) is the crash-safe
record of one checkpoint:

- **`complete`** - the atomic completion marker. It is `False` until every file is hashed and the
  manifest sealed and written (temp file -> `fsync` -> `os.replace`), so a torn write is never mistaken
  for a resumable checkpoint.
- **`checkpoint_manifest_hash`** - the canonical sha256 of the manifest body with the hash field
  removed (same convention as the RunPlan seal). `parent_checkpoint_hash` chains lineage to the parent
  by that same digest.
- **`bound`** (`CheckpointBoundIdentities`) - everything a resumed run must match: RunPlan hash,
  resolved execution-configuration hash, backend, managed environment lock, worker wheel, model,
  tokenizer, dataset, chat template / formatter, objective, and the seeds. A mismatch on any of these
  makes the checkpoint inadmissible.
- **`state`** (`SealedTrainingState`) - the resumable training state described without torch: which of
  optimizer / scheduler / scaler / RNG / sampler were captured (each backed by a file), the RNG
  algorithm, and the exact position on the timeline - epoch, global optimizer step, microstep within
  the step, gradient-accumulation width, and consumed microsteps - so a resume continues from the
  precise position, never an approximate one.
- **`files`** (`CheckpointFileEntry[]`) - every required file pinned by canonical relative path,
  role, sha256, and size; sorted and unique, and the optimizer state is mandatory.

## Verification (fail closed)

`platform/checkpoint.py` is torch-free. `verify_checkpoint_integrity(dir)` fails closed with a typed
`CheckpointError.reason` on:

| reason | condition |
| --- | --- |
| `missing_file` | no manifest, or a required checkpoint file is absent |
| `malformed` | the manifest is not valid JSON / not a valid `CheckpointManifest` |
| `incomplete` | the manifest is not marked `complete` (the write did not finish) |
| `hash_mismatch` | the manifest body no longer matches its sealed hash (tamper) |
| `unsafe_path` | a member is a symlink, a hard link, or resolves outside the checkpoint directory |
| `external_change` | a file's bytes or size changed since it was sealed |

`verify_matches_request(manifest, request)` pins the on-disk checkpoint to the exact id + sealed hash
the dispatch named, so an individually-valid but swapped checkpoint is refused. `verify_resumable_into(
manifest, plan)` then fails closed (`reason="incompatible"`) unless every plan-derivable bound identity
- plan hash, execution-configuration hash, environment lock, model/tokenizer/dataset, objective, seeds,
and backend - matches the target run exactly. The engine additionally re-verifies the worker-only
identities (worker wheel, formatter, chat template) it can derive, and asserts the resumed optimizer
owns exactly the model's live trainable parameters. `admit_resume(plan, dir, resumed_run_id=...)` runs
the integrity + compatibility checks and returns the `ResumeLineage` a **fresh** resumed run records;
it refuses to reuse the source run's id.

## Lineage

A resumed run mints a fresh run id and a run-scoped output directory, and records `ResumeLineage`
(parent run id + parent checkpoint id + parent checkpoint hash + the global step it resumed from) on
its `RunManifest`. An ordinary from-scratch run has `resume_lineage = null`; no parent run, checkpoint,
or output is ever mutated or relabeled.

## CLI

```bash
# Verify a checkpoint's completion marker + per-file byte integrity (fails closed, non-zero on any gap).
corpus-studio checkpoint-verify <checkpoint-dir>

# Also verify it is a compatible resume source for a specific target plan.
corpus-studio checkpoint-verify <checkpoint-dir> --plan ./plan/RunPlan.json
```

`checkpoint-verify` never resumes or executes anything - it only verifies. The execution engine
(`training/checkpoint_io.py`) is what writes and restores checkpoints; a resumed trial is also marked
on `TelemetryIdentity` (`resumed`, `parent_run_id`, `resumed_from_global_step`) so paper aggregation
never conflates it with an uninterrupted one.
