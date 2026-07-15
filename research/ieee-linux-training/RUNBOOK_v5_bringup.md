# v5 bring-up runbook (Sections 5-8): wheel -> environments -> matched plans -> 0.5B smokes

This runbook is the exact, ordered, fail-closed procedure for taking the native-Linux RTX 5070 study
from the merged **prospective amendment 0002 (effective matrix 1.2.0)** to two sealed **v5** blue/green
environments and their first-party 0.5B bring-up smokes. It is written so a later, separately
authorized GPU session can execute it without re-deriving any decision.

It stops at the first real optimizer step through `platform-run`. It does **not** cover the 7B
feasibility ladder or the ~500-output full-training arm; those are later phases with their own gates.
The measurement harness and the exact checkpoint/resume execution engine are now implemented in the
engine, but these bring-up smokes remain **checkpoint-free** and short - they do not exercise resume,
and capturing telemetry (`platform-run --telemetry`) during them is optional and additive.

## 0. Authorization gate (do not start without all of these)

This runbook performs **GPU work and network package installation**. Per the standing operating
rules, a human must explicitly authorize the GPU/resource commitment before Section 5. Confirm every
line before the first command:

- [ ] Amendment **0002** is merged to `main`; the effective protocol version is **1.2.0**.
- [ ] The worker source commit (the current `main` HEAD you build the wheel from) has
      `df86db53e294a6e15b724c586f7016a1c9fdac00` as a git **ancestor** - amendment 0002 binds
      `worker_success_admission.required_git_ancestor` (an ancestor rule, not byte-equality), so a
      **descendant** commit is admitted as long as the df86db5 audit-fix floor is in its history.
      `main` also descends from `c1322b5d82854dfc76408d2a550b32366ea7d14d` (#445, the AdamW step fix).
      Prove both and record the exact source commit:
      ```bash
      cd /mnt/training-nvme/repos/CorpusStudio
      git merge-base --is-ancestor df86db53e294a6e15b724c586f7016a1c9fdac00 HEAD && echo "df86db5 ancestor: OK"
      git merge-base --is-ancestor c1322b5d82854dfc76408d2a550b32366ea7d14d HEAD && echo "c1322b5d ancestor: OK"
      git rev-parse HEAD   # record this exact commit into the wheel evidence (per-environment + per-trial)
      ```
      Do NOT require `engine/` bytes to equal `df86db5`: the post-audit engine has advanced (measurement
      harness, checkpoint/resume engine), and the amendment records the **exact** final source commit and
      wheel sha256 in evidence rather than pinning bytes to df86db5. See section 0a.
- [ ] The validator is green from a clean checkout (the effective matrix is byte-frozen and unaffected
      by engine changes):
      `engine/.venv/bin/python research/ieee-linux-training/validate_protocol.py --verify-host-evidence`
      prints `status: valid` with effective-matrix sha256
      `168189145150c0ed13ce70151a065c9490d9e70052ca30569aac709e718f9e12`.
- [ ] A human has authorized building the v5 wheel, creating the v5 environments, and dispatching the
      0.5B smokes.

## 0a. Amendment 0002 reconciliation (why v5 identities remain valid)

The engine changed after `df86db5` (measurement harness, checkpoint/resume engine). This does **not**
invalidate the prospective v5 identities. Classification: **V5_IDENTITIES_REMAIN_VALID**. Evidence:

- Amendment 0002's manifest binds `worker_success_admission.required_git_ancestor =
  df86db53e294a6e15b724c586f7016a1c9fdac00` - an **ancestor** rule. The choice of "ancestor" (not an
  exact-commit pin) explicitly admits descendant worker source, provided the df86db5 audit-fix floor
  stays in history. Its `immutable_binding_additions.worker_source_commit` is
  `required-per-environment-and-trial` (record the exact commit) and
  `exact_source_commit_required_in_wheel_evidence` / `exact_wheel_sha256_required_in_environment_and_trial`
  bind the exact **final** source + wheel - not a specific df86db5 wheel.
- `historical_worker_wheel_reuse_authorized = false`: the v5 wheel is built **fresh** from the final
  commit (Section 5), so no historical wheel is reused.
- No v5 identity has been **instantiated** yet (no v5 environment, wheel, lock, plan, or run exists -
  `preserved_v4_evidence` shows only v4 bring-up attempts). The v5 environment ids are *allocated* by
  0002 but unused, so nothing is reused incorrectly.
- The bring-up smokes are **checkpoint-free** and short; the post-`df86db5` engine additions
  (checkpoint/resume behind a sealed policy the smokes do not set; additive telemetry) leave the sealed
  smoke execution behavior unchanged. Scientific semantics are preserved.
- `validate_protocol.py` binds the exact **wheel** (candidate identities: exactly one shared worker
  wheel, disjoint from `RESERVED_IDENTITIES.v2.json`); the git-**ancestry** of the source commit is a
  procedural gate (the `git merge-base --is-ancestor` check above), exactly as before - the validator
  never enforced git ancestry.

Therefore no new amendment, effective-matrix version, or fresh environment identity is required. The
recommended environment ids remain `backend-corpus-studio-research-math-v5` /
`backend-corpus-studio-research-flash-v5`; every other v5 identity is minted fresh in this runbook.

## 1. Non-negotiable guardrails (apply to every step)

- **One GPU operation at a time.** Never run two GPU workloads (probe, smoke, `nvidia-smi -l`, an
  external process) concurrently.
- **Unload Ollama before every GPU operation** and confirm the GPU is idle
  (`nvidia-smi` shows no resident model, ~0 MiB in use by other processes) before any allocation.
- **Never reuse a reserved identity.** Every new environment id, lock hash, wheel hash, plan id, plan
  hash, execution-configuration id/hash, run id, output path, and evidence root must be
  set-disjoint from `amendments/RESERVED_IDENTITIES.v2.json`. Prove it with the validator's
  `--candidate-identities` mode (Section 7) before any environment mutation or dispatch.
- **Historical evidence is read-only.** Do not delete, move, rename, relabel, retry, reuse, or mutate
  any v1-v4 environment, lock, capability report, execution probe, RunPlan, execution configuration,
  run, adapter, output directory, evidence directory, or `SHA256SUMS`. Do not `git clean`.
- **Never write to `/mnt/windows-c` or `/mnt/windows-f`.** They are read-only historical sources.
- **Large artifacts live under `/mnt/training-nvme`,** never in the source checkout. Only small
  reviewable specs/tooling/summaries belong in `research/`.
- **Do not weaken any contract to obtain a passing run.** Evidence contracts, precision, security,
  provenance, kernel enforcement, artifact integrity, failure taxonomy, and methodology are fixed. A
  cell that fails is recorded with its quantitative failure evidence and taxonomy; it is never omitted,
  imputed, or relabeled.
- **A completed step is not a proven fit.** Predicted fit is never `NATIVE_SAFE`; only a measured run
  is. Bring-up smokes and characterization trials are never reported as paper full-training results.

## 2. Fixed identities allocated by amendment 0002

| Item | Math (blue) | Flash (green) |
|---|---|---|
| Environment id | `backend-corpus-studio-research-math-v5` | `backend-corpus-studio-research-flash-v5` |
| Attention tuple | forced `torch_sdpa_math` | forced `torch_sdpa_flash` |
| Role | verified-safe default | separately sealed flash capability |

Every other v5 identity (wheel hash, lock hashes, plan ids/hashes, execution ids/hashes, run ids,
output paths, evidence roots) is **minted fresh during this runbook** and must be recorded back into
`docs/HOST_STATE.md` under a new "manager-1.3 v5 pair" section (append-only; do not touch the v1-v4
sections). None of these fresh values may collide with `RESERVED_IDENTITIES.v2.json`.

## 3. Section 5 - build the v5 worker wheel twice and hash-compare

Goal: a single, reproducible `corpus_studio_engine` wheel built from the current merged 1.2.0 `main`
HEAD (which descends from `df86db5`, the required-ancestor floor - see section 0a; do NOT require
`engine/` bytes to equal `df86db5`). Build it **twice** in independent clean trees and require identical
SHA-256; a mismatch means the build is not reproducible and blocks the environments.

```bash
cd /mnt/training-nvme/repos/CorpusStudio
WHEEL_COMMIT="$(git rev-parse HEAD)"                 # the merged 1.2.0 main
DEST="/mnt/training-nvme/artifacts/corpusstudio-worker/${WHEEL_COMMIT}"
mkdir -p "$DEST"

# Build A and Build B in separate clean worktrees at the same commit.
for TAG in a b; do
  WT="/mnt/training-nvme/tmp/wheelbuild-${TAG}"
  rm -rf "$WT"
  git worktree add --detach "$WT" "$WHEEL_COMMIT"
  ( cd "$WT/engine" && SOURCE_DATE_EPOCH=1 "$PWD/../../repos/CorpusStudio/engine/.venv/bin/python" -m build --wheel --outdir "$DEST/build-${TAG}" )
  git worktree remove --force "$WT"
done

# Require byte-identical wheels.
sha256sum "$DEST"/build-a/*.whl "$DEST"/build-b/*.whl
```

- If the two SHA-256 values differ, **stop**: investigate non-determinism (timestamps, file ordering)
  before proceeding. Do not seal an environment from a non-reproducible wheel.
- On match, promote one copy to `"$DEST"/corpus_studio_engine-<ver>-py3-none-any.whl`, record its
  SHA-256 and METADATA SHA-256, and confirm both differ from every reserved v1-v4 wheel hash
  (in particular the v4 wheel `f8b03634...12a92`).
- The wheel version stays `1.3.0` (the engine package version); its **identity is the SHA-256**, not
  the version string.

> Note: confirm the exact `python -m build` availability in the engine venv before relying on it; if
> `build` is not installed, use the project's documented packaging entrypoint. The invariant that
> matters is *two independent builds at the same commit produce the same wheel bytes*.

## 4. Section 6 - create the two matched v5 environments

For **each** environment (math first as the safety baseline, then flash), run the manager's
plan -> review -> create -> probe -> lock flow. Never mutate one environment while planning the other.

```bash
ENGINE=/mnt/training-nvme/repos/CorpusStudio/engine/.venv/bin/corpus-studio
MROOT=/mnt/training-nvme/corpusstudio/xdg-data/corpusstudio/environment-manager
WHEEL="$DEST/corpus_studio_engine-<ver>-py3-none-any.whl"

# 6a. Canonical dependency plan (no mutation). Repeat with the flash env id.
$ENGINE env-plan backend-corpus-studio-research-math-v5 \
  --env-id backend-corpus-studio-research-math-v5 \
  --runtime /usr/bin/python3 \
  --accelerator cu128 \
  --worker-wheel "$WHEEL" \
  --manager-root "$MROOT" \
  --out "$DEST/DependencyResolution.math-v5.json"
# -> review printed indexes, size, target path, argv, and the printed resolution hash.

# 6b. Create (network install) only after the plan is reviewed. Confirm with the exact hash.
$ENGINE env-create backend-corpus-studio-research-math-v5 \
  --env-id backend-corpus-studio-research-math-v5 \
  --runtime /usr/bin/python3 \
  --accelerator cu128 \
  --worker-wheel "$WHEEL" \
  --manager-root "$MROOT" \
  --confirm <resolution-hash-from-6a>

# 6c. Health + capability. env-create runs the required complete-tuple probe; confirm the sealed lock.
$ENGINE env-status backend-corpus-studio-research-math-v5 --manager-root "$MROOT"
$ENGINE env-probe  backend-corpus-studio-research-math-v5 --manager-root "$MROOT"
```

Acceptance for each environment before it can back a plan:

- final state `HARDWARE_VERIFIED` with a sealed `EnvironmentLock` (a `DEGRADED`/`INCOMPATIBLE`/`BROKEN`
  result has no lock and requires recreation under a **new** id -- do not retry in place);
- 84 installed packages under `record_count_semantics="all_record_rows_v2"` with
  `record_verified_entries == record_entries == installed_file_count` and zero failed RECORD rows;
- the sealed lock binds the exact v5 wheel SHA-256 from Section 5;
- the math env's capability tuple forces `torch_sdpa_math`; the flash env's forces
  `torch_sdpa_flash`; the two capability-report hashes differ from each other and from the reserved v4
  hashes (`b260040e...` math, `77e1f5fd...` flash);
- `drift_detected=false` immediately after sealing.

Record both lock hashes and both capability-report hashes.

## 5. Section 7 - fresh matched v5 bring-up plans (0.5B)

Produce one math and one flash RunPlan for the Qwen2.5-0.5B bring-up smoke. Keep everything but the
environment/attention tuple identical across the pair so the comparison is clean.

```bash
# Repeat for the flash environment with its own --out path.
$ENGINE platform-plan \
  --base-model <qwen2.5-0.5b-local-snapshot> \
  --dataset <bring-up-smoke.jsonl> \
  --sequence-len 256 \
  --max-steps 12 \
  --backend corpus_studio \
  --optim adamw_torch \
  --environment backend-corpus-studio-research-math-v5 \
  --manager-root "$MROOT" \
  --out /mnt/training-nvme/corpusstudio/runs/ieee-linux-training/phase3-qwen25-05b-matched-v5
```

- `--optim adamw_torch` is the only sealed optimizer; do not request `paged_adamw_8bit`/Liger for the
  bring-up smoke.
- The 12-step count is a bring-up smoke, not a paper cell. Sequence length starts at 256; sequence
  4096 is a later explicit attempted cell.
- Predicted fit will not be `NATIVE_SAFE`; that is expected and correct pre-run.

**Disjointness gate (mandatory before dispatch).** Assemble a candidate-identity JSON for the matched
pair (`stage: "runplan"`, both v5 environment ids, both fresh lock hashes, the one shared wheel hash,
both plan ids/hashes, both execution ids/hashes, the fresh output path, the fresh evidence root) and
prove it reuses nothing reserved:

```bash
engine/.venv/bin/python research/ieee-linux-training/validate_protocol.py \
  --candidate-identities /mnt/training-nvme/tmp/candidate-runplan-v5.json
# must exit 0; any "candidate reuses reserved ..." is a hard stop.
```

Also confirm each plan/execution bundle records the effective-matrix 1.2.0 hash
`168189145...b9c` and that the two plans differ only in the sealed environment, capability, and
attention tuple (plus fresh document identities) -- the same normalized-equality property proved for
the v4 pair.

## 6. Section 8 - dispatch the two 0.5B smokes (one at a time)

Math first (the verified-safe default), then flash. **Unload Ollama and confirm the GPU is idle
before each.** Run them sequentially; never overlap.

```bash
# Math smoke.
$ENGINE platform-run /path/to/math-v5/RunPlan.json \
  --subprocess \
  --timeout 600 \
  --manager-root "$MROOT" \
  --out /mnt/training-nvme/corpusstudio/runs/ieee-linux-training/phase3-qwen25-05b-matched-v5/math
# ... confirm terminal RunManifest, then repeat for flash with its own --out.
```

Expected/accepted outcomes (record whichever occurs; none is relabeled):

- **Success**: at least one real optimizer step completes with exact per-step loss records
  (`logging_steps=1`), the trainable-state and optimizer identity evidence pass, materialized
  post-accumulation gradients are FP32 under the sealed policy, and a terminal `RunManifest.json` is
  written. This is the first real optimizer step through `platform-run`. It is a **bring-up** result,
  still not a 7B or full-training claim.
- **Terminal failure** (e.g. `GRADIENT_FAILURE`, `OPTIMIZER_FAILURE`, OOM, stall/timeout): preserved
  with taxonomy, stage, and quantitative evidence under its own run id and output root. Do not retry
  in place; a corrected attempt needs a fresh wheel/env/plan chain and a new amendment if it changes
  study semantics after results are visible.

This runbook is complete when both smokes have a terminal `RunManifest.json` and their outcomes,
identities, and hashes are appended to `docs/HOST_STATE.md` (new v5 section) and `HANDOFF.md`.

## 7. What Sections 5-8 do and do not prove

Proven on success: the corrected worker (post-#444/#445) can, under the sealed FP32 QLoRA policy and
forced attention path, complete a real optimizer step on Qwen2.5-0.5B through the full
plan -> seal -> run -> manifest lifecycle on this host.

Still **not** proven and explicitly out of scope here: full-sequence 7B training; sequence length 4096;
the ~500-output full-training arm; DeepSpeed/FSDP/CPU/NVMe offload or real offload fit; PCIe/NVMe
throughput or endurance; bare-Linux FlashAttention for the real workload; MoE runtime capability; any
comparative performance claim against WSL2 or native-Windows evidence (which must never be collapsed
into one category or presented as apples-to-apples unless separately proven comparable).
