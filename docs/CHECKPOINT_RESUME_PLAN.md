# Checkpoint / Resume — Implementation Plan (#486, building on #440)

**Status:** design-complete, execution-gated. Authored 2026-07-31.
**Goal:** production checkpoint + exact resume for a first-party `platform-run` training run — a long run
survives interruption and continues with exact lineage, or fails closed. This is Training Systems **P2**
(production resume via `platform-run`).

> **The headline: this is ~70% built and already proven bitwise on CPU.** #440 landed the hard parts —
> the torch-free seal/verify/admit control plane, the torch worker save/restore/coordinator I/O, the
> contracts, and a 3-process integration test that proves a fresh-interpreter resume reproduces
> uninterrupted training **bit-for-bit** on CPU. What remains is **wiring** (the production trainer loop
> and the resume orchestration), a **gate lift**, and the **measured GPU evidence** — not new algorithms.

---

## 1. Verified inventory — what already exists (do NOT rebuild)

| Component | Where | Status |
|---|---|---|
| Torch-free seal / integrity verify / resume admission | `platform/checkpoint.py` — `seal_checkpoint_manifest`, `verify_checkpoint_integrity`, `verify_resumable_into`, `admit_resume` | **Done + tested** (`test_platform_checkpoint.py`) |
| Worker save / restore / cadence driver | `training/checkpoint_io.py` — `save_checkpoint`, `restore_checkpoint`, `CheckpointCoordinator`, `capture/restore_rng_state`, `assert_optimizer_over_live_params` | **Done + tested** (`test_training_checkpoint_io.py`) |
| End-to-end resume equivalence proof | `tests/_checkpoint_reference.py` + `test_training_checkpoint_integration.py` | **Done** — 3 processes (uninterrupted N / checkpoint-at-K / fresh-process resume) prove **bitwise** param + per-step-loss equivalence on deterministic CPU |
| Contracts | `CheckpointPolicy` (cadence/keep_last/impl + validation), `CheckpointResumeRequest`, `CheckpointManifest`, `SealedTrainingState`, `CheckpointBoundIdentities`, `ResumeLineage`, `CheckpointFileEntry` | **Done** |
| Sealed config carries the policy | `ResolvedExecutionConfiguration.checkpoint_policy` + `RunPlan.checkpoint_policy` | **Done** — cadence is inside the seal |
| Dispatch body carries the resume | `RunDispatchBody.resume: CheckpointResumeRequest | None` | **Done** |
| Verify-only CLI | `checkpoint-verify` (verifies a checkpoint is a compatible resume source; never executes) | **Done** |

**Key invariants already enforced by the built code:** exact-lineage-or-nothing (`verify_resumable_into`
compares every plan-derivable identity + the sealed execution-config hash); fail-closed on partial /
corrupt / incomplete / externally-changed / symlinked / hard-linked / escaping-path members; a resume
**mints a fresh run id** and reads the parent **read-only** (never mutates it); the resumed optimizer is
rebuilt over the **live restored parameters** (object-identity checked).

## 2. The precise gap — three unwired seams

1. **The production trainer loop does not use the coordinator.** `training/trainer.py` only *mentions*
   `checkpoint_io` in a docstring (`trainer.py:314-324`); `run_training` never instantiates a
   `CheckpointCoordinator`, never calls `maybe_checkpoint`, and never calls `restore_checkpoint`. The
   proven wiring lives only in the reference trainer.
2. **`admit_resume` is called nowhere.** The resume *orchestration* — verify a checkpoint, mint a fresh
   run, rebuild the identical target plan, dispatch with `RunDispatchBody.resume` populated, record the
   `ResumeLineage` — does not exist yet.
3. **The runner refuses cadence today.** `platform/runners.py:592-601` fails closed on any
   `checkpoint_policy` with a cadence/keep_last: *"sealed intermediate checkpoints are unsupported until
   exact resume compatibility ... sealed resume support exists."* That gate is lifted only once the
   trainer actually supports it. (`platform-plan` also has no `--checkpoint-cadence` flag yet, so no
   run can request intermediate checkpoints.)

## 3. Phased plan

The work splits cleanly by **what changes worker-execution bytes** (→ a fresh sealed wheel + GPU) and
what is pure control plane (→ CI-green + fake-worker testable, no hardware).

### Phase A — Control-plane resume orchestration (NON-gated; I can do this now)

No worker bytes change. Every piece is torch-free and testable with fake workers + the existing
`admit_resume`. Setting a cadence and dispatching a resume are **not** contract changes — the fields
already exist in the seal and the dispatch body.

- **A1 — `platform-plan --checkpoint-cadence <steps> [--keep-last <n>]`.** Populate the existing sealed
  `CheckpointPolicy` (planner `checkpoint_policy`, `planner.py:1141`) so a run can *request* intermediate
  checkpoints. Setting cadence>0 legitimately changes the sealed execution-config hash (a checkpointed
  run is a different config than a checkpoint-free one) — expected, not fragility.
- **A2 — resume orchestration (`platform-resume`).** A new control-plane command/function: load + fully
  verify the parent checkpoint via `admit_resume`; mint a **fresh** run id; rebuild the **byte-identical**
  target `RunPlan` (required — `verify_resumable_into` compares the execution-config hash exactly);
  construct the `RunDispatchBody` with `resume` populated (`checkpoint_id` + sealed manifest hash +
  dir); persist the returned `ResumeLineage` on the resumed run record.
- **A3 — record `ResumeLineage` on the run record.** Add a `resume_lineage` field to the run-registry
  record (a run-record field, not a sealed contract) so lineage is queryable and the resumed run
  declares its parent + `resumed_from_global_step`.
- **A4 — tests (fake-worker / unit, torch-free):** `admit_resume` happy path + every fail-closed reason
  (incomplete / hash_mismatch / external_change / unsafe_path / incompatible / parent-id reuse);
  orchestration mints a fresh id, refuses parent-id reuse, attaches lineage; `--checkpoint-cadence`
  round-trips into the sealed config + `verify_resumable_into` accepts the rebuilt plan; an
  incompatible-plan resume is refused.

**Deliverable:** one or two CI-green control-plane PRs I can author + self-merge under the standing rule.
This makes the *whole* resume flow real and tested end-to-end **except** the trainer executing it.

### Phase B — Worker trainer-loop wiring (GATED: worker bytes → v10 wheel)

Merge + wheel build require authorization (worker-closure + wheel-rebuild gates).

> **Correction after reading the real trainer (2026-07-31):** this is **not** a mechanical port of the
> hand-rolled reference. `trainer.run_training` wraps **TRL `SFTTrainer`** (constructed at
> `trainer.py:3068`, `.train()` at `:3110`), which owns optimizer creation, the step loop, the data
> cursor, and RNG timing. The reference proves the *algorithm* on a hand-rolled loop; wiring it into
> `SFTTrainer` is a real integration with three concrete decisions that only a torch/GPU run can
> validate - so this phase is developed **against a real run when the window opens**, not authored blind.

- **B1 — checkpoint WRITING via a `TrainerCallback`.** Add a callback (mirroring the existing
  `_ProgressCallback` at `trainer.py:2979`) whose `on_step_end` calls
  `CheckpointCoordinator.maybe_checkpoint(...)` with `get_peft_model_state_dict(model)` as the adapter
  state, the callback's `optimizer`/`lr_scheduler`, `capture_rng_state(torch)`, and the data cursor.
  Wrinkle: the coordinator's `bound` identities come from `bound_identities_from_plan(RunPlan)`, but
  `run_training` only has the flattened `TrainRunConfig` - so the worker must also receive (or
  reconstruct) the `RunPlan`, plus fill the worker-only bound fields (worker-wheel / formatter /
  chat-template sha).
- **B2 — RESUME injection.** `SFTTrainer.train()` builds its own optimizer unless the constructor is
  given `optimizers=(optimizer, lr_scheduler)`. The defensible approach: `restore_checkpoint(...)` the
  adapter onto the model **before** constructing the trainer, then pass the restored
  `optimizers=(...)`; the data-cursor skip and RNG timing still need validation against a real
  `SFTTrainer` run. **Open decision (needs a steer + a run):** use our sealed `checkpoint_io` format
  with `optimizers=` injection (reuses the built+tested I/O) **vs** seal HF-native checkpoints and use
  `trainer.train(resume_from_checkpoint=...)` (reuses HF's proven step-skip, but our I/O becomes a seal
  wrapper). Recommendation: the former (it keeps the CPU-bitwise-proven `checkpoint_io` as the trust
  anchor), pending the first real run.
- **B3 — lift the two guards** (`runners.py:592-601` **and** `trainer._require_checkpoint_free_execution`
  at `:2683`) once B1/B2 land, plus the deferred `platform-plan --checkpoint-cadence/--keep-last` flag.
- **B4 — v10 PRODUCT wheel + env re-seal.** `trainer.py` + `runners.py` are worker-execution code, so
  their bytes changing forces a fresh sealed wheel (v9 PRODUCT → **v10**) and a re-sealed managed
  environment (new lock/id). The **v8 sealed-research lineage is untouched** — this is the product chain.

The already-tested control-plane scaffolding for this phase is real: `resolve_checkpoint_execution_policy`
(`trainer.py:339`) already classifies the sealed policy into an enabled/cadence/keep-last decision, and
`train_config_from_resolved` already threads `save_steps`/`save_total_limit` from the checkpoint policy.

### Phase C — Measured GPU evidence + SupportLevel promotion (GATED: GPU auth)

- **C1 — GPU smoke on the real workload** (7B QLoRA, seq-4096, the winning flash+liger+paged config):
  train to K, checkpoint, kill the process, resume in a fresh interpreter, verify continuation to N, and
  a full run to completion under the checkpoint cadence.
- **C2 — promote the capability's SupportLevel to `workload_verified` ONLY on this evidence.** CPU-bitwise
  is *not* GPU-verified; "a completed step ≠ proven fit"; "installed ≠ supported." Until C1 passes on
  this host, resume stays `probed`/design-proven, not `workload_verified`.
- **C3 — record the evidence** (resumed run records + `docs/HOST_STATE.md`), following the existing
  measured-run evidence pattern.

## 4. Robustness / invariants honored

- **Exact lineage or nothing.** `verify_resumable_into` requires byte-identical plan identity + sealed
  execution-config hash; any drift is an incompatible resume. Fail-closed on every corrupt / partial /
  incomplete / tampered / unsafe-member checkpoint.
- **Fresh run id, read-only parent.** A resume never reuses/mutates the parent run or checkpoint; the
  coordinator prunes only *this* run's own written checkpoints and keeps the freshest `keep_last`.
- **Sealed-hash honesty.** Cadence lives inside the sealed config; the resume target rebuilds the
  identical plan, so a resume can only continue a run whose exact execution semantics match.
- **Worker-closure discipline.** Only Phase B touches worker bytes → exactly one lineage bump (v10) +
  one env re-seal, batched (B1+B2 together), never incremental. Control-plane Phase A ships independently.
- **Dense-first, MoE-additive.** Today's checkpoint is `adapter_only` dense QLoRA. `CheckpointFileEntry.role`
  is a free-form role string, so per-expert / expert-shard checkpoint roles (MoE, gap **G8**) are a purely
  additive later slice — **no foundational contract assumes dense execution**.
- **No silent capability claims.** The resume capability is promoted to `workload_verified` only by a
  measured GPU resume (Phase C), never by CI, contracts, or the CPU proof alone.

## 5. What I can start now vs what needs you

- **DONE:** Phase A (resume-preparation orchestration + lineage record + fake-worker tests) shipped in
  PR #771. The `platform-plan --checkpoint-cadence` flag was deferred to Phase B (it advertises an
  unrunnable option until the trainer honors cadence).
- **GATED (your authorization):** Phase B (the `SFTTrainer` wiring) + the v10 wheel build/env re-seal;
  Phase C GPU run + the SupportLevel promotion. **Revised recommendation:** because the `SFTTrainer`
  integration (§3 Phase B) is not a mechanical port and cannot be validated in the torch-free lane, Phase
  B is best **built against a real run when the hardware window opens** rather than authored blind - a
  blind draft would be intricate, unexecutable, and likely rewritten against the first real run. The
  design above is deliberately concrete so that build is de-risked and fast.

## 6. Risks + mitigations

| Risk | Mitigation |
|---|---|
| GPU reductions are non-deterministic — bitwise resume is not guaranteed on GPU | Phase C proves **state restoration + loss continuity**, not bitwise; the bitwise claim stays CPU-scoped (as the integration test already documents) |
| **`SFTTrainer` owns the optimizer/loop/data** — the reference's hand-rolled resume does not transfer directly | B2 injects the restored optimizer via the constructor `optimizers=(...)` and restores adapter weights before `.train()`; the data-cursor skip + RNG timing are validated against a real `SFTTrainer` run before the capability is claimed (build B against a run, not blind) |
| The real data pipeline's sampler/packing cursor differs from the reference | B1 captures the *actual* sampler/dataloader cursor via the coordinator's `sampler_state`; C1 asserts the restored cursor resumes at the exact next sample |
| `keep_last` pruning races a reader | Coordinator only removes directories it wrote and always keeps the freshest `keep_last` (≥1) |
| Wheel/env churn from worker edits | Batch B1–B4 into one v10 lineage bump; never rebuild per-edit |
| Scope creep into MoE / full-param checkpointing | Explicitly out of this slice; dense `adapter_only` first, MoE roles additive later (G8) |

## 7. Sequencing summary

```
Phase A (control plane, DONE #771)   Phase B (worker SFTTrainer, gated)   Phase C (GPU, gated)
----------------------------------   ----------------------------------   --------------------
A. resume-prepare orchestration      B1 checkpoint-writing callback       C1 GPU checkpoint->resume smoke
   admit_resume -> fresh run ->       B2 resume injection (optimizers=)    C2 promote SupportLevel
   PREPARED record w/ lineage         B3 lift both guards + cadence flag   C3 record evidence
   + reachable CLI + tests            B4 v10 wheel + env re-seal
(cadence flag deferred -> B3)         (built against a real run, on auth)  (on GPU auth)
```
