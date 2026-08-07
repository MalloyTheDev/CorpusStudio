# 7B/seq-4096 checkpoint write -> resume evidence (2026-08-07)

Measured evidence for the `workload_verified` promotion of intermediate-checkpoint **write** and
exact-lineage **resume** (a FEATURE of the `dense_qlora_sft` variant) at production scale. Run through the
**fully-sealed** managed `--subprocess` path (a provenance-sealed worker wheel, from a neutral CWD so the
worker imports the wheel, not the repo).

## Configuration

- Model: Qwen2.5-7B-Instruct (local snapshot `a09a3545...`), QLoRA **nf4** r16, **seq-4096**.
- Winning config: flash SDPA (`torch_sdpa_flash`) + `liger_fused_ce` + `paged_adamw_8bit` +
  allocator `max_split_size:128`, gradient checkpointing.
- Data: WBG 469-row chat corpus (`examples/wbg/data/train.jsonl`), zero-truncation (max 3445 tokens),
  chat-template sha256 `cd8e9439...`.
- Checkpoint cadence 2, `--max-steps 4`.

## Sealed lineage

- Worker wheel `corpus_studio_engine-1.3.0` sha256 **`222b6147...`** (built from clean main HEAD
  `b9cd5d2`, carries #830).
- Sealed env **`backend-corpus-studio-sealed-flp-v3`**, lock **`db797ce1...`**, state `HARDWARE_VERIFIED`
  (its flash-liger-paged GPU probe ran the wheel's code).
- Plan: `environment_binding=managed_lock`, plan_hash **`27ac06df...`**, fit `NATIVE_UNPROVEN`.

## Runs (RunManifests preserved here)

| run | run_id | steps | notes |
|-----|--------|-------|-------|
| write | `run-019fde18-5f61-71d6-84ec-0d46d697254a` | 1,2,3,4 | wrote `step-2` + `step-4` checkpoints |
| resume | `run-019fde1c-d8e5-7c9b-8b4f-b5d19b3d4319` | 3,4 | `--resume-from step-2`, `resumed_from_optimizer_step=2` |

Resumed from checkpoint `run-019fde18-...-ckpt-step-00000002`, manifest hash **`3db56855...`**
(`complete: true`; integrity-verified + proven a compatible resume source BEFORE any restore).

## Measured losses and what they prove

| optimizer_step | write run | resume run | delta |
|---:|---|---|---|
| 1 | 3.192439 | - | - |
| 2 | 2.931450 | - | - |
| 3 | 2.935065 | **2.935065** | `0` (identical) |
| 4 | 2.847071 | 2.847229 | `~1.6e-4` |

- **Step-3 loss is identical.** The step-3 loss is the forward pass on the resumed model weights BEFORE the
  step-3 optimizer update, so its exact match proves the **model weights (including the step-2 update baked
  into the adapter), the data batch, and the RNG** were restored correctly. It does NOT by itself prove
  optimizer-moment restore.
- **Step-4 loss (post-update) agrees to ~1.6e-4.** Step 4 is the first forward AFTER a resumed optimizer
  update (step 3), so its agreement is the actual signal that the **optimizer state continued**. The residual
  ~1.6e-4 is consistent with the documented guarantee - EXACT VERIFIED LINEAGE + **HF-standard, non-bitwise**
  continuation (QLoRA + flash + bf16 carry inherent CUDA nondeterminism at this scale); it is NOT bitwise, and
  is not claimed to be.

## Scope of the interruption claim (honest)

The checkpoint consumed here was produced by a source run that **completed** (steps 1-4), not by a `SIGKILL`
mid-run. The `step-2` checkpoint is written atomically at step 2 with `complete: true` and its manifest hash,
and is fully integrity-verified before any restore - so it is byte-equivalent to the checkpoint an
interrupted run would leave, and the resume (a fresh worker process) is the same operation either way. Using
a completed source additionally yields the **ground-truth** steps 3/4 for the continuation comparison above,
which a killed source could not. A literal kill-then-resume adds only write-atomicity-under-`SIGKILL`
coverage and is noted as an optional follow-up.

## Known resume-hardening follow-ups (from the review of this evidence)

The proof above is sound (real training, verified by the measured continuation), but the review of these
manifests surfaced guarantee-hardening gaps that are the next slice (see `docs/HOST_STATE.md`):

- **Worker-identity binding (H1):** the checkpoint's `worker_wheel_sha256` is null (only `plan_hash` +
  `environment_lock_hash` are bound) and the restore does not verify worker-only identities - so exact
  lineage does not confirm the resuming worker BYTES. Seal + verify it for the managed/sealed tier.
- **Resume lineage:** the resumed `RunManifest.resume_lineage` is null despite `resumed_from_optimizer_step=2`;
  populate the parent run / checkpoint id / checkpoint hash from the verified checkpoint.
- **Post-restore baseline:** `before_sha256` is captured before the HF restore, so the canonical-adapter-change
  gate could be satisfied by the restore alone; capture it after restoration or add a resumed-interval delta.
- **Worker-CWD isolation:** spawn the managed worker neutral-CWD/`-I` so it uses the wheel regardless of the
  launch directory.
