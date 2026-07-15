# Amendment 0002: post-audit v5 blue/green environment identities

- Amendment ID: cs-ieee-linux-training-amendment-0002
- Base protocol version: 1.0.0
- Effective protocol version: 1.2.0
- Supersedes: amendment 0001 (effective protocol version 1.1.0)
- Author: CorpusStudio maintainers
- Timestamp: 2026-07-15T15:30:00Z
- Status: prospective
- Analysis role: primary
- Effective matrix: ../EXPERIMENT_MATRIX.v1.2.0.json
- Machine manifest: 0002-2026-07-15-post-audit-v5-identities.manifest.json
- Reserved historical identities: RESERVED_IDENTITIES.v2.json
- Validator: ../validate_protocol.py

## Reason

Amendment 0001 (effective matrix 1.1.0) bound the manager-1.3 `...-v4` first-party environments. Two
subsequent worker corrections make every `...-v4` identity historical and inadmissible for a new run:

1. **Post-#444 / #445 worker correction (merged to `main` at `df86db5`).** The v4 math attempt failed
   before optimizer step 1 because pinned TRL recast the sealed FP32 QLoRA parameters to BF16 during
   `SFTTrainer` construction; #444 restores the sealed master dtype after construction. A follow-up
   adversarial audit (#445, commit `c1322b5d82854dfc76408d2a550b32366ea7d14d`) then found the next
   blocker sitting immediately behind #444: `verify_optimizer_state_precision` rejected torch's own
   CPU-resident 0-dim AdamW `step` counter (default `adamw_torch`, the only sealed optimizer) as a
   placement deviation, which would have failed optimizer step 1 of every real run; and the enforced
   attention-kernel cleanup could rewrite a real `GRADIENT_FAILURE`/`OPTIMIZER_FAILURE` as an
   environment error. #445 corrects both with CPU/unit evidence only.
2. The worker execution bytes therefore changed again after the v4 wheel was built. Reusing,
   recreating, renaming, or relabeling any `...-v4` (or earlier) environment, lock, plan, run, or wheel
   would break the identity chain and the study's evidence boundary.

A new worker wheel and new immutable blue/green environment locks are required. This amendment
allocates the unused v5 identities and reserves every prior (v1-v4) identity as non-reusable.

## Exact changes

The effective matrix is the complete UTF-8 JSON document `EXPERIMENT_MATRIX.v1.2.0.json`, reconstructed
by the validator from the immutable base matrix (`EXPERIMENT_MATRIX.yaml`, 1.0.0) plus the exact,
single-match changes and additive fields in this amendment's machine manifest. Relative to the base
matrix it makes these changes:

| Field | Base value | Effective 1.2.0 value |
| --- | --- | --- |
| schema_version | 1.0.0 | 1.2.0 |
| protocol_version | 1.0.0 | 1.2.0 |
| first_party_execution_paths[id=first-party-math].environment_id | backend-corpus-studio-research-math-v1 | backend-corpus-studio-research-math-v5 |
| first_party_execution_paths[id=first-party-flash].environment_id | backend-corpus-studio-research-flash-v1 | backend-corpus-studio-research-flash-v5 |
| immutable_bindings | prior bindings only | adds effective-matrix, reserved-identity, and worker-source-commit bindings |
| top-level evidence controls | absent | adds amendment, effective-specification, manager-1.3 admission, worker-success admission, historical non-reuse, affected-scope, prior-amendment lineage, preserved-v4-evidence, and full-training-phase records |

No attention policy, package-selection rule, workload field, model, tokenizer, dataset, adapter,
precision, optimizer, scheduler, seed, step count, telemetry, failure, exclusion, statistics, or
analysis rule changes from the base matrix. The exact repository commit, wheel bytes, package pins,
lock hashes, and capability/probe hashes remain required per environment or trial through the base
`immutable_bindings` (which already require `capability_report_hash` and `execution_probe_hash` per
trial).

## Bound worker source

- Final audited source commit for the v5 worker: `df86db53e294a6e15b724c586f7016a1c9fdac00`.
- Introducing fix commit (required ancestor): `c1322b5d82854dfc76408d2a550b32366ea7d14d` (#445).
- The v5 wheel must be built from `main` after this amendment merges. That commit adds only
  `research/` files and does not touch `engine/`, so its `engine/` bytes are identical to
  `df86db5`; the wheel therefore realizes the audited worker source above.
- `worker_success_admission.required_git_ancestor` is set to `df86db5` so the wheel source must
  descend from the fully audited main (which includes #444 and #445).

## Reserved historical identities (v2)

`RESERVED_IDENTITIES.v2.json` is append-only over `RESERVED_IDENTITIES.v1.json`: every v1-v3 identity
it enumerated remains reserved, and the following v4 identities are added so no v5 evidence may reuse
them:

- environments `backend-corpus-studio-research-math-v4`, `backend-corpus-studio-research-flash-v4`;
- lock hashes `14750ec5...a62a8`, `9f599070...7fd30`;
- worker wheel `f8b03634...12a92`;
- plans `plan-019f650d-...` / `plan-019f650e-...` and their hashes;
- executions `...-execution` and their hashes;
- run `run-019f6518-3927-7d73-b106-15f385b61415`;
- output path `.../phase3-qwen25-05b-matched-v4`;
- the v4 environment, plan, and production-smoke evidence roots.

The v4 math/flash capability-report hashes are recorded in the reserved set's
`superseded_environment_evidence` for preservation; the sealed lock and wheel reservations plus the
per-trial `capability_report_hash`/`execution_probe_hash` bindings already prevent capability/probe
reuse. The validator enforces the append-only property (`_validate_reserved_superset`) and binds the
frozen 0001 amendment by hash (`_validate_supersession`).

## Deterministic identity and validation

The effective matrix identity is the SHA-256 of its committed raw bytes (recorded in the machine
manifest). The manifest binds the immutable base protocol and matrix, this narrative, the complete
effective matrix, the reserved-identity set, and the validator by SHA-256, and records the superseded
0001 amendment's manifest, narrative, effective-matrix, and reserved-identity hashes. The validator
reconstructs the effective document from the base plus exact single-match changes and additive fields,
refuses a base-hash mismatch, duplicate key, selector cardinality other than one, old-value mismatch,
unknown effective change, stale affected-cell count, malformed reserved set, dropped prior reserved
identity, or a superseded-prior-amendment hash mismatch. On this host `--verify-host-evidence`
additionally rehashes the preserved source manifests, including the v4 evidence `SHA256SUMS` files.

Every evidence bundle for a v5 environment plan or RunPlan, and every trial manifest, must record the
exact effective-matrix 1.2.0 hash and be set-disjoint from `RESERVED_IDENTITIES.v2.json`.

## Affected cells and execution stages

The environment identity substitution affects the same scope as amendment 0001 (nothing has been
executed as a paper cell since):

- both first-party 0.5B bring-up smoke paths;
- all 30 not-yet-executed primary cells (3 models x 5 sequence lengths x 1 adapter method x 2
  first-party paths);
- all 16 not-yet-executed first-party secondary adapter cells (2 models x 2 sequence lengths x 2
  adapter methods x 2 first-party paths); and
- any later adaptive follow-up that selects a first-party path.

The Unsloth execution path is unchanged. No primary or secondary cell has been executed. The two new
v5 environments must be built from the same v5 wheel bytes and matched package artifacts, retaining
their separately sealed math and forced-`torch_sdpa_flash` capability tuples.

## Sequence length 4096 and the 500-output full-training phase

- The base matrix already sets `primary_matrix.sequence_4096_attempt_required: true`; sequence length
  4096 remains an explicit attempted cell for both first-party paths. A cell that fails (OOM, stall,
  timeout, thermal/RSS safety stop) is recorded with its quantitative failure evidence and taxonomy -
  it is never omitted and never imputed as missing data.
- The `full_training_phase` record defines the user's ~500-output full-training arm as a distinct
  phase, gated until the feasibility ladder identifies a feasible configuration and preregistered
  before any full run. Feasibility characterization (12-step trials) and the 500-output full-training
  arm are never conflated: bring-up smokes and characterization trials are not paper full-training
  results.

## Evidence visible before this change

Disclosed to avoid outcome-dependent ambiguity:

- The manager-1.3 v4 math plan was dispatched once (`run-019f6518-3927-7d73-b106-15f385b61415`); it
  verified plan/execution/lock identities, forced `torch_sdpa_math`, singleton CUDA placement, NF4,
  QLoRA insertion, and a real optimizer, then **failed before optimizer step 1** with `GRADIENT_FAILURE`
  at `backward` (a BF16 materialized gradient under the sealed FP32 policy). It completed zero optimizer
  steps, wrote no artifact/checkpoint/output, kept `drift_detected=false`, and released VRAM.
- The paired v4 flash plan was **never dispatched**.
- No real optimizer step has passed through `platform-run`; sequence length 4096 remains unverified;
  no paper-matrix cell exists. These are non-paper bring-up records preserved under their original IDs
  and hashes.

## Prospective status and compatibility

Protocol 1.0.0 and its base matrix, and amendment 0001 with effective matrix 1.1.0, remain available
unchanged as frozen historical documents. Protocol 1.2.0 uses the complete versioned effective matrix;
it does not reinterpret any 1.0.0 or 1.1.0 result. This amendment is prospective and does not by itself
authorize building the v5 wheel, creating the v5 environments, or dispatching any GPU workload.
