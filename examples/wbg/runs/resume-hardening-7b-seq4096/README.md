# Resume-guarantee hardening - 7B/seq-4096 GPU validation (2026-08-07)

GPU validation of the four resume-guarantee hardening items (#832) on the real 7B/seq-4096 workload,
through the **fully-sealed** managed `--subprocess` path from a **neutral CWD** (the worker is the sealed
wheel, not repo-shadowed code).

## Sealed lineage

- Worker wheel `corpus_studio_engine-1.3.0` sha256 **`1af5a079...`** (built from branch HEAD `2c9b698`,
  carries the hardening), sealed into HARDWARE_VERIFIED env **`backend-corpus-studio-sealed-flp-v5`**
  (lock **`aa9dac62...`**).
- Config: Qwen2.5-7B-Instruct QLoRA nf4 r16, seq-4096, flash SDPA + liger_fused_ce + paged_adamw_8bit +
  max_split_size:128; WBG 469-row chat corpus; managed_lock plan.

## Runs (manifests preserved here)

| run | run_id | result |
|-----|--------|--------|
| write | `run-019fde66-8717-7b19-b166-d57a33d8436d` | wrote step-2/4 checkpoints |
| resume | `run-019fde6a-fc21-70a8-85c6-6997d564b1ef` | `--resume-from step-2`, succeeded |

## What each item's evidence shows

- **F1 - worker-identity binding (WRITE side).** The step-2 `CheckpointManifest.bound.worker_wheel_sha256`
  = `1af5a079ff451dba6a4cfaaba1b352fee01d121e0c75eeaf8db8ebffac6cc628` - the exact sealed wheel the run
  executed (previously null). NOTE: this wheel (`1af5a079`, source `2c9b698`) proves only the write-side
  binding; the restore-side verification gate was added later (`d15b8e8`) and is validated in the re-proof
  section below (wheel `80515175` / env `flp-v7`).
- **F2 - resume lineage.** The resumed `RunManifest.resume_lineage` =
  `{parent_run_id: run-019fde66..., parent_checkpoint_id: ...-ckpt-step-00000002,
  parent_checkpoint_hash: 1d6351e6..., resumed_from_global_step: 2}` (previously null).
- **F3 - post-restore baseline.** The resume **succeeded** with `resumed_from_optimizer_step=2` and steps
  `[3, 4]`: the honesty-core canonical-adapter-change gate now captures its baseline in `on_train_begin`
  (after the HF restore, before the first resumed update) and is fail-closed. Success therefore proves the
  gate was satisfied by the RESUMED interval's real training, not by the checkpoint restore.
- **CWD isolation.** The worker was spawned with `-P` from a neutral CWD, so it imported `corpus_studio`
  from the sealed env's site-packages (the wheel), verified before the run.

All PRODUCT `workload_verified` evidence for the merged code AND the sealed wheel, not a sealed IEEE cell.

## Re-proof with the review-fix wheel (restore-verify + PYTHONPATH isolation)

A code review of the above surfaced three more fixes (PYTHONPATH isolation, verify-worker-identity on
resume, robust lineage). Re-proven on wheel **`80515175...`** (source `d15b8e8`) sealed into env
**`backend-corpus-studio-sealed-flp-v7`** (lock `61d52813...`), same neutral-CWD managed path:

- write `run-019fde83-ff67-7733-a308-a3f1d93e561f` bound `worker_wheel_sha256 = 80515175...`;
- resume (`--resume-from step-2`) **succeeded** with `resumed_from_optimizer_step=2`, steps `[3,4]`,
  `resume_lineage.parent_run_id = run-019fde83...`.

So the **worker-identity gate is active on resume** (the resuming wheel `80515175` matched the
checkpoint's bound wheel, so the resume proceeded - a mismatch fails closed), the worker ran the sealed
wheel under `-P` + a PYTHONPATH-stripped env, and the lineage was recorded from the verified checkpoint.
