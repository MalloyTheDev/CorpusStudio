# Exact checkpoint + resume lineage (#440)

A long first-party run must be resumable with **exact** lineage or not at all. This is the
control-plane, torch-free design and its verifier: a hash-sealed `CheckpointManifest`, byte-integrity
verification that fails closed, and a resume admission that only proceeds against a byte-identical,
fully compatible target `RunPlan`.

**This does not enable automatic resume.** Intermediate checkpoints stay disabled in the trainer and
execution stays checkpoint-free. First-party runs expected to exceed 30 minutes remain blocked until a
**separately reviewed trainer change** consumes a `CheckpointResumeRequest` and restores the sealed
state. Until then this is the reviewed design plus its verifier - no checkpoint is written or reused
automatically, and no run's checkpoint-free behavior changes.

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
| `external_change` | a file's bytes or size changed since it was sealed |

`verify_resumable_into(manifest, plan)` then fails closed (`reason="incompatible"`) unless every
plan-derivable bound identity - plan hash, execution-configuration hash, environment lock,
model/tokenizer/dataset, objective, seeds, and backend - matches the target run exactly. The manifest's
worker-only fields (worker wheel, formatter/chat-template bytes) are re-verified by the worker when it
restores the bytes. `admit_resume(plan, dir, resumed_run_id=...)` runs both checks and returns the
`ResumeLineage` a **fresh** resumed run records; it refuses to reuse the source run's id.

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

`checkpoint-verify` never resumes or executes anything - it only verifies. Resume remains blocked
until the separately reviewed trainer change lands.
