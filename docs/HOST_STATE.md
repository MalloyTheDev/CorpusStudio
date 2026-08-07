# Host State — Native-Linux RTX 5070 Workstation

**Last verified:** 2026-07-16 (manager-1.3 **v7** math/flash environments created, probed, and
dispatched: both 0.5B smokes succeeded AND the v6 token-throughput observer gap is fixed and validated -
`V7_MATH_AND_FLASH_THROUGHPUT_PASS`; see the v7 section below. The v6 pair
(`V6_MATH_AND_FLASH_BRINGUP_PASS`) remains preserved as history. Earlier v1-v6 environment, plan, and run
evidence remains preserved and non-reusable; legacy environment, GPU, and paths were checked 2026-07-14).

This file records the *verified* runtime facts of the machine CorpusStudio currently runs
on. It supersedes the Windows `C:`/`F:` host descriptions in older docs for **"where you
are."** The prior native-Windows/WDDM (and separately labeled WSL) evidence is **preserved
as history** in [`HANDOFF.md`](../HANDOFF.md) and [`CURRENT_STATE.md`](CURRENT_STATE.md) —
it is not deleted or restated as Linux.

> **Verification level.** Every value below is read directly from the OS, `nvidia-smi`, and
> the Environment Manager registry / lock / health report on this host. What a
> `HARDWARE_VERIFIED` environment does and does **not** prove is spelled out under
> "Verification boundary" — do not read it as a training-workload result.

## Host

| Fact | Value |
|---|---|
| Platform | Linux 6.8.0-134-generic — Ubuntu 24.04.4 LTS, x86-64 |
| Repository (active runtime) | `/mnt/training-nvme/repos/CorpusStudio` |
| Engine control-plane venv | `/mnt/training-nvme/repos/CorpusStudio/engine/.venv` — CPython 3.12.3 (dependency-light core + `[dev]`, torch-free) |
| CLI entrypoint | `engine/.venv/bin/corpus-studio` (equivalently `.venv/bin/python -m corpus_studio.cli`) |
| Linux training filesystem | `/mnt/training-nvme` |
| Windows `C:` drive (mount) | `/mnt/windows-c` — read-write filesystem mount; history-only project policy; e.g. former `C:\CorpusStudio` → `/mnt/windows-c/CorpusStudio` |
| Windows Projects / `F:` drive (mount) | `/mnt/windows-f` — read-write filesystem mount; history-only project policy; e.g. former `F:\CorpusStudio` → `/mnt/windows-f/CorpusStudio` |

The active runtime is the native-Linux NVMe checkout under `/mnt/training-nvme/...`. The old
Windows `C:` and `F:` copies are still visible as read-write `/mnt/windows-c` and `/mnt/windows-f`
filesystem mounts. Read-write is an OS mount fact, not permission to use them as development roots:
they are stale fallbacks that will drift, so **do not work from or write to them.**

## GPU

| Fact | Value |
|---|---|
| Device | NVIDIA GeForce RTX 5070, 12227 MiB (~12 GB) |
| Architecture | Blackwell — compute capability 12.0 (sm_120) |
| Driver | 595.71.05 |
| CUDA (driver-reported) | 13.2 |

## Managed backend environment — `backend-corpus-studio`

The reference training-worker environment (the isolated Layer-3 backend of the three-layer
dependency model — see [`ENVIRONMENT_MANAGER.md`](ENVIRONMENT_MANAGER.md)) is built and
probed on this host.

| Fact | Value |
|---|---|
| Env id | `backend-corpus-studio` |
| Layer | `backend_worker` |
| Root | `/mnt/training-nvme/corpusstudio/xdg-data/corpusstudio/environment-manager/environments/backend-corpus-studio` |
| Manager root | `/mnt/training-nvme/corpusstudio/xdg-data/corpusstudio/environment-manager` |
| Managed interpreter | `<root>/bin/python` — CPython 3.12.3 |
| **State** | **`HARDWARE_VERIFIED`** |
| Drift detected | `false` |
| Recipe ref | `backend-corpus-studio` / `sha256:7fd0c05d…ca94c4` |
| Resolution ref | `resolution-d2c32667f525c17b84d9` / `sha256:d2c32667…55638b` |
| Installation journal | `install-d53b77b4cf9e44c99ab3` |
| Lock ref | `lock-dbc528f0167a2ec0ccfa` |
| Lock digest | `sha256:dbc528f0167a2ec0ccfa42d46ce86c9061be126e02aa557ae5ef5741788a8045` |
| Created / verified | 2026-07-14T03:25:51Z / 2026-07-14T03:29:15Z |
| Owner marker | `.corpusstudio-owner.json` (`corpus-studio-managed-environment-v1`, this manager root) |

**Probe results** (`EnvironmentHealthReport`, `checked_at` 2026-07-14T03:29:28Z):

| Probe | Outcome |
|---|---|
| `reference_backend_imports` | PASS |
| `pip_check` | PASS |
| `reference_backend_functional` | PASS |
| `reference_backend_hardware` | PASS |

**Pinned stack** (`EnvironmentLock` `lock-dbc528f0…`):

| Package | Version |
|---|---|
| torch | 2.11.0+cu128 (build `70d99e99…`) |
| CUDA runtime (wheel) | 12.8 |
| compute capability | 12.0 |
| transformers | 5.13.1 |
| peft | 0.19.1 |
| trl | 1.8.0 |
| bitsandbytes | 0.49.2 |
| accelerate | 1.14.0 |
| datasets | 5.0.0 |

(84 distributions are locked in total; the table lists the training-relevant ones.)

### Readiness-v2 math baseline (preserve)

`backend-corpus-studio-readiness-v2` is a separate exact-pinned managed environment that does not
replace or reinterpret the legacy environment above. It is `HARDWARE_VERIFIED` for one complete
BF16-configured NF4 + double-quant + QLoRA + **math-only SDPA** + AdamW + adapter-reload tuple. Its
sealed manager-1.1 evidence predates the explicit `forward_autocast` field, so it is not retroactively
an observed BF16-activation/autocast claim. Its exact narrow identity remains readable historical
rollback evidence. Manager 1.4 does not grant it a new health/planning claim without a replacement
lock carrying complete all-row RECORD counts. Do not modify, recreate, reseal, or delete it while
developing flash readiness.

Recorded identities for recovery:

| Item | Value |
|---|---|
| Lock hash | `21dd38cbadd11fbf42f8f4de9f87a5c29642b139baefc13008dffe26d0751c13` |
| Recipe digest | `4c0cb365b596cfe2b1371afd5f95130a40e41c7e5b27df833b0c914bd492289c` |
| Probe evidence hash | `5f23457b3ac737b6dbe514c0325f5445b9accac9cc3f642d7137c19ddf868886` |
| Worker wheel SHA-256 | `de747839c300bc4f7bc3288963d6814b204a95402efba3f1787fa6e6462e135f` |
| Baseline record | `/mnt/training-nvme/artifacts/corpusstudio-worker/readiness-math-baseline/math-baseline-record.json` |

### Readiness flash-v1 status

`backend-corpus-studio-readiness-flash-v1` is a separate exact-pinned recipe for a complete forced
`torch_sdpa_flash` QLoRA tuple (`cuda_qlora_sdpa_flash_execution`). It is independent of the math
baseline and must not reuse or mutate readiness-v2. **Linux-only** recipe (native Windows/WDDM fused
flash SDPA is refused on the Windows path; do not claim flash from a Windows math environment).

**Sealed on this host (2026-07-14):** after the bf16-autocast probe fix, the environment was
recreated from commit `f15f1bfeec0b54c4c863b78f03f2b1c3032bd768`. Its preserved manager-1.1
`env-status --refresh` and `env-probe` reports are **`HARDWARE_VERIFIED`** with
`drift_detected=false`. Math readiness-v2 was not mutated. Manager 1.2 preserves those lock/evidence
digests as historical evidence but does not grandfather flash across the new adapter-state equality
requirement; the flash environment needs an audited-wheel replacement before a new manager-1.2 health
claim. Readiness-flash-v1 itself was not recreated during the audit; the later manager-1.2 evidence
uses the separate blue/green research-flash-v2 identity documented below.

| Item | Value |
|---|---|
| Preserved manager-1.1 state | `HARDWARE_VERIFIED` |
| Environment path | `/mnt/training-nvme/corpusstudio/xdg-data/corpusstudio/environment-manager/environments/backend-corpus-studio-readiness-flash-v1` |
| Lock ID | `lock-8a988a716c68beacfa8c` |
| Lock digest | `8a988a716c68beacfa8c8fb46925987ea7c9aca198537340471e1fd08f9c75fe` |
| Recipe digest | `52016adedd5011328efb05e089d54c8edd5c9308e0a38409897cd0f554240fb7` |
| Resolution hash | `941da281bda775a9ca097801900356a99d8b16917a5172b452da1a4d8013b57a` |
| Probe evidence hash | `ad9b5e0c07b4d8d437905d6f0bf888afa2151531f097270b4d40cdb39c7830b8` |
| Capability-report hash | `bb00d68fc76dfdd4bb7b8014e9dadd06ac138b5c114f44a3cecedaa161866215` |
| Worker source commit | `f15f1bfeec0b54c4c863b78f03f2b1c3032bd768` |
| Worker wheel | `.../readiness-flash-v1/f15f1bfeec0b54c4c863b78f03f2b1c3032bd768/corpus_studio_engine-1.3.0-py3-none-any.whl` |
| Worker-wheel SHA-256 | `cb5c05b7d4d8e640d06a4d845ae638930b9e9f3769f937c87365f0e7e445d5f5` |
| Complete probe | `cuda_qlora_sdpa_flash_execution` **PASS** (`torch_sdpa_flash`, forced `FLASH_ATTENTION`, math/mem-efficient off, `forward_autocast=bf16`) |
| Evidence pack | `/mnt/training-nvme/corpusstudio/evidence/backend-corpus-studio-readiness-flash-v1/` (`env-recreate-f15f1bf.json`, status/probe) |

**History:** an earlier authorized create at `082cb15` failed seal (`INCOMPATIBLE`) because float32
Q/K/V under forced flash without bf16 autocast. That failure is superseded by the sealed recreate
above; it is not a positive flash claim for the old wheel.

This sealed flash result is still **environment-probe** evidence only — not full-sequence 7B training,
not Transformers `flash_attention_2`, not external `flash-attn`, and not MoE runtime capability.

### First bounded flash smoke and placement-only diagnostic

The separately authorized production-path smoke used Qwen2.5-0.5B, sequence length 256, exactly three
planned steps, the sealed flash kernel/toggles, and the current flash lock. Its preserved evidence is
under
`/mnt/training-nvme/corpusstudio/evidence/bounded-flash-smoke/20260714T194401Z/`.

| Item | Preserved result |
|---|---|
| Run ID | `run-019f6229-9fda-7067-a20b-80fbf6c1c709` |
| Plan hash | `d9f2763f69df5b7a32b2b2b8fdd2b9f5c965ac8a8848cd30950c4e485c62e41e` |
| Execution-configuration hash | `846d0ac61199b3eaa08c1556ca98481b335da3662ee80f858c4c7f2e8792f687` |
| Production smoke | `failed` / `UNSUPPORTED_CONFIGURATION` at `placement_deviation` because `hf_device_map` was absent |
| Boundary reached | Real model load completed; adapter insertion did not start |
| Optimizer steps / adapter | `0` / none written |
| Post-run environment | `HARDWARE_VERIFIED`, `drift_detected=false` |
| Placement-only diagnostic | All 290 parameters and both registered buffers observed on `cuda:0`; no CPU, disk, meta, or other-GPU state observed |
| Diagnostic classification | `PLACEMENT_MAP_REPRESENTATION_MISMATCH` |

The placement-only diagnostic confirms actual singleton CUDA residency for that one authorized load;
it is not a successful `platform-run`, adapter insertion, backward pass, or optimizer step. No real
optimizer step has yet passed through `platform-run`, and sequence length 4096 remains unverified.
If the flash environment is recreated with a new worker wheel, its new lock hash invalidates this old
RunPlan; generate a new plan against the replacement lock before any later smoke.

### Matched manager-1.2 research environments and bounded smokes

Two blue/green manager-1.2 environments were subsequently sealed from the same worker wheel and
package artifact set. They preserve the older readiness environments rather than mutating them:

| Item | Math | Flash |
|---|---|---|
| Environment | `backend-corpus-studio-research-math-v2` | `backend-corpus-studio-research-flash-v2` |
| Lock hash | `7ffa59ea68a243331cf16f6ab5a16f0c47d3d1e6ae415692d42260cba36decf4` | `256acc9c437897bb02c6ff1cb6d45cf42470612d88e78a4977647b7f27c30416` |
| Required tuple | `cuda_qlora_sdpa_math_execution` | `cuda_qlora_sdpa_flash_execution` |
| State after smoke | `HARDWARE_VERIFIED`, drift `false` | `HARDWARE_VERIFIED`, drift `false` |

Both use worker wheel SHA-256
`eb4cbde415cadda523bb316c11919ba5c8083fccbcecd0d9e04aaa1a65539d3b` from source commit
`a222a82f20dd8a04b7e0994a0deb778c08a0a1f0`. Their matched environment and plan evidence is under
`/mnt/training-nvme/corpusstudio/evidence/backend-corpus-studio-research-matched-v2/` and
`/mnt/training-nvme/corpusstudio/evidence/production-smoke-matched-v2/20260715T034634Z/`.

Fresh sequence-256, three-step RunPlans were generated after package RECORD/tree evidence was bound
into managed capability snapshots. A field-by-field audit found only environment/capability and
attention-kernel/toggle differences; the normalized plans and all rendered examples were identical.
Each plan was dispatched exactly once, math first and flash second:

| Item | Math | Flash |
|---|---|---|
| Run ID | `run-019f640f-a587-7f79-9bf1-2a36c05854fd` | `run-019f6413-c34b-7570-a5b4-ea69caa0579b` |
| Forced kernel observed | `torch_sdpa_math` | `torch_sdpa_flash` |
| Boundary reached | Model and post-adapter placement verified; QLoRA attached; trainer/optimizer created | Same |
| Terminal result | Failed before step 1: incoming autograd hook tensor for one `lora_B` weight was BF16 while the sealed materialized-gradient policy is FP32 | Same |
| Optimizer steps / artifacts / checkpoints | `0` / none / none | `0` / none / none |
| Final GPU state | 10 MiB, no compute process | 10 MiB, no compute process |

The common failure exposed a production-verifier mismatch: readiness checks the materialized leaf
`parameter.grad`, while the worker checked the earlier pre-accumulation hook tensor. The repository
correction uses a post-accumulation hook and remains fail-closed for missing, wrong-dtype, or
wrong-device materialized gradients. That code correction is unit evidence only until a new wheel,
new immutable environments/locks, fresh RunPlans, and separately approved smokes are produced. The
two failed plans and runs are preserved and must not be retried or reused. No real optimizer step has
yet passed through `platform-run`, and sequence length 4096 remains unverified.

### Pre-checkpoint manager-1.2 v3 candidates (preserve; do not admit)

Before the training-success audit checkpoint, a worker wheel from repository commit
`16ef6e95722ec3988ee8826b45333c9356ef76f9`, two manager-1.2 v3 environments, and one normalized
math/flash RunPlan pair were created. The plans were never dispatched: there is no run ID, run output,
model load, adapter insertion, optimizer step, or GPU-workload result for this pair. They remain
read-only reconstruction evidence and must not be deleted, mutated, relabeled, retried, or reused.

| Item | Math | Flash |
|---|---|---|
| Environment | `backend-corpus-studio-research-math-v3` | `backend-corpus-studio-research-flash-v3` |
| Lock hash | `cd86808ce8e96533b6d6d3a0b4c0472e2e6e27ecf8d25bad916a9a08d4e6887d` | `a2b839b160e4676d968cdd006040dde6cce756c30f51a2c92ef2b1442132aa2a` |
| Plan ID | `plan-019f644b-a3c2-7373-abc0-39a0f7d753eb` | `plan-019f644c-511c-7008-a21b-24586c6b4637` |
| Plan hash | `60b390c3e7fa0d0dd6276854be4266c67f29e71b630653ebd1b7a75eeaa2506a` | `cc4856f75f251b8d26cb86e50af7874c21abce1fde2708f183ff9a3ab2a47ed7` |
| Execution hash | `4453d60d23ecd7bcd3811a616a1381b51e68e6d941bfcaa673c9895b79a854c5` | `fe2d99cea56a35a14da71e1352d0b977282d6865a73dbc41861d496ceec6fa53` |

Both environments used `corpus-studio-engine==1.3.0` wheel SHA-256
`6ecc82595af761142b723017a31b980241fe6ef4afebf0a2223f90b8bcef724d` (METADATA SHA-256
`c8eb3e03d457da4495545bc0bb355131a02d3d48f397bc4a9c07fe1cff9704fe`). Evidence is under
`/mnt/training-nvme/corpusstudio/evidence/backend-corpus-studio-research-matched-v3/` and
`/mnt/training-nvme/corpusstudio/evidence/production-smoke-matched-v3/20260715T052743Z/`.

The old investigation correctly found no `record_integrity=unknown` or null RECORD count in the
authoritative v3 plan bundle under manager-1.2 semantics; unknown fields occurred only in preserved
version-only probe journals and were not admitted. The audit checkpoint nevertheless found a stricter
integrity gap: all 84 installed packages in each v3 lock claimed `verified` while
`record_verified_entries < record_entries`. Manager 1.4 tags the new meaning explicitly with
`record_count_semantics="all_record_rows_v2"` and requires positive equality with the installed-file
count. Missing semantics keeps the old documents hash-verifiable, but health refuses them without
rewriting their historical state. Consequently these locks and plans cannot authorize new work. A post-merge
worker wheel, new immutable environment IDs/locks, and completely fresh RunPlans are required before
any separately approved smoke.

### Manager-1.3 v4 pair and preserved math failure (do not reuse)

After the training-success and complete-RECORD hardening merged, two new manager-1.3 environments
were created from one wheel built at repository commit
`e7875629fc6e046dc2a84a53aa941b3d073c18bd`. They remain separate blue/green identities; no prior
environment, lock, plan, run, artifact, or evidence was relabeled or mutated.

| Item | Math | Flash |
|---|---|---|
| Environment | `backend-corpus-studio-research-math-v4` | `backend-corpus-studio-research-flash-v4` |
| Lock hash | `14750ec5932765fe544675aba69d0763931e249d598da8a4d9a44549e85a62a8` | `9f599070fcef83e192d1380ab50683a37cd9034a97194cc523fa58915e47fd30` |
| Bound capability hash | `b260040eb967ab55052320d45805fc7b3056480a1aa2f354791a605157e6e925` | `77e1f5fd57d2b54b7c11d9e4ba14b0656bb96e80e64239298b84d205b1d370a6` |
| Plan ID | `plan-019f650d-cc5d-7028-9763-9e8dfb66f370` | `plan-019f650e-51eb-7fc6-a444-816593a52552` |
| Plan hash | `3bc3f230293c2ccc4eeac0fab63f03f503dfce36e7f10dad49f0feec76163065` | `cb750f36e79d8a119b24a71f95e787faaf8222940524e17497c4544580eef6ce` |
| Execution hash | `4294b8431d1d20076e87b3797185add2fd8c5479db60adb1cc3f1a4c5cd47ea2` | `736ade995bfc6e4fb6d2b0dc6ecd2b717ec8fa53b464144b15299242b662dfe4` |
| Workload dispatch | Once | Not dispatched |

Both environments use `corpus-studio-engine==1.3.0` wheel SHA-256
`f8b03634148c41c2fd44e337c9e562e4a8ce1f0b3f11cd980a7accd0a2a12a92` (METADATA SHA-256
`098220cd2ae18eb38b780cae349a4434ad678f85b9522eaea86fb69752f07dea`). Each lock has 84 installed
packages under `record_count_semantics="all_record_rows_v2"`; every installed package has positive
`record_entries == record_verified_entries == installed_file_count` and zero failed RECORD rows. The
normalized plan comparison found no shared semantic difference after accounting only for the sealed
environment/capability, attention tuple, fresh document identity, and environment-root-bound package
digests.

The sequence-256 math plan was dispatched once as run
`run-019f6518-3927-7d73-b106-15f385b61415`. It verified exact plan/execution/lock identities, forced
`torch_sdpa_math`, model and post-adapter singleton CUDA placement, NF4 preparation, QLoRA insertion,
and a real optimizer at `on_train_begin`. It then failed before optimizer step 1 with taxonomy
`GRADIENT_FAILURE` at stage `backward`: the materialized gradient for
`base_model.model.model.layers.23.mlp.down_proj.lora_B.default.weight` was BF16 while the sealed
gradient dtype was FP32. The terminal fit remained `NATIVE_UNPROVEN`; there were zero loss records,
artifacts, and checkpoints, and the run-scoped output root was never created. Post-run health remained
`HARDWARE_VERIFIED` with drift false, and the GPU returned to 10 MiB with no compute process. The
attempt is preserved under
`/mnt/training-nvme/corpusstudio/evidence/production-smoke-matched-v4/20260715T090840Z/runs/math-20260715T092243Z/`;
its `SHA256SUMS` file hashes all 19 evidence files. The paired flash plan was withheld rather than
consuming a known-common failure path.

The exact pinned TRL source explains the failure: during `SFTTrainer` construction its QLoRA branch
recasts all trainable parameters to BF16 after CorpusStudio registered post-accumulation hooks and
enforced FP32. The post-accumulation verifier therefore reported the actual materialized gradient
honestly. The repository correction restores the sealed master dtype on the same parameter identities
after trainer construction and re-runs complete placement, quantization, and precision verification
before training. That correction currently has CPU/unit evidence only. Because worker behavior
changed, the v4 wheel, environments, locks, and plans must remain preserved and cannot authorize a
retry; any later attempt requires a new wheel, new environment IDs/locks, and completely fresh plans.

**Post-#444 audit correction (2026-07-15, CPU/unit evidence only).** A read-only hardening audit found
the next blocker sitting immediately behind the #444 fix: `verify_optimizer_state_precision` compared
*every* materialized optimizer tensor's device against the sealed `cuda:0`, but torch's default
`adamw_torch` (non-fused, non-capturable) keeps the per-parameter `step` as a 0-dim scalar counter on
CPU by design (`torch.optim.adam._init_group`). That would have failed optimizer step 1 of every real
run with an `OPTIMIZER_FAILURE` placement deviation before any success could be recorded. The verifier
now allows a 0-dim scalar counter on the expected device or CPU while still rejecting a CPU-offloaded
(non-scalar) moment tensor. Separately, the enforced attention-kernel context's cleanup seal
reassertion could raise and replace a real `GRADIENT_FAILURE`/`OPTIMIZER_FAILURE` with an environment
error; it now subordinates a restoration error while a workload failure is already propagating. These
change the worker execution bytes again, so the next environment pair is a fresh **v5**
identity (`backend-corpus-studio-research-math-v5` / `backend-corpus-studio-research-flash-v5`),
built from the corrected commit. Research amendment **0002 -> effective matrix 1.2.0** is now merged
(#448, effective-matrix sha256 `168189145150c0ed13ce70151a065c9490d9e70052ca30569aac709e718f9e12`);
it allocates the v5 identities, binds the audited worker source `df86db5`, and its reserved-identity
set (`RESERVED_IDENTITIES.v2.json`) enumerates every now-historical v4 identity as non-reusable. The
amendment is prospective and does not itself authorize GPU work: building the v5 wheel, creating the
v5 environments, and dispatching the 0.5B smokes remain gated on a separate human GPU/resource
authorization. When that authorization is given, execute the exact ordered procedure in
[`research/ieee-linux-training/RUNBOOK_v5_bringup.md`](../research/ieee-linux-training/RUNBOOK_v5_bringup.md).

### Manager-1.3 v6 pair - first successful 0.5B bring-up (math + flash), 2026-07-16

The v5 bring-up produced the study's first real GPU training (12 QLoRA math steps, loss 5.43 -> 0.39) but
terminally failed at export: TRL's benign `training_args.bin` was rejected by the sealed adapter validator
(`ARTIFACT_FAILURE`), and its telemetry was scientifically incomplete. Two corrections landed on `main`
inside the worker child - **#461** (narrow `training_args.bin` admission, never deserialized) and **#462**
(populate the required paper telemetry) - which change worker execution bytes, so a fresh **v6** lineage was
required. Research amendment **0003 -> effective matrix 1.3.0** (effective-matrix sha256
`e7b95d47aa23a87b4aed0ddac6dabf5fc070dc77e4d7ec710129fb690a7c4587`, `RESERVED_IDENTITIES.v3.json` sha256
`414d23862e7a835f88b0c454c6fb0a930bc3904cca08bac8d793d5de1db10d40`) reserves every v1-v5 identity, allocates
the v6 environment ids, and requires the worker source to descend from `af28be9`. A fresh reproducible v6
wheel `corpus_studio_engine-1.3.0-py3-none-any.whl` sha256
`bdc32196203539cbeb9078ce2317fb41d2a30abe68f7e94bc0fa290a97f414d4` was built twice byte-identically from
source commit `73b756c49da0f03203ebd05dfb5528805b0fd280`.

| Item | Math (blue) | Flash (green) |
|---|---|---|
| Environment id | `backend-corpus-studio-research-math-v6` | `backend-corpus-studio-research-flash-v6` |
| Lock hash | `db8d3dea...a669d825` | `fb104a9b...f243a8d5` |
| Forced kernel | `torch_sdpa_math` (flash+mem-eff disabled) | `torch_sdpa_flash` (math+mem-eff disabled) |
| Plan id / hash | `plan-019f687d...` / `7d4202ce...` | `plan-019f687f...` / `e7fb9f49...` |
| Run id | `run-019f688c-67c0-77cf-82e2-477f52fab76f` | `run-019f6892-3a54-7922-8e10-d138ee7e77ce` |
| Terminal state | `succeeded` | `succeeded` |
| Steps / losses | 12 / 5.4336 -> 0.3937 | 12 / 5.432 -> 0.377 |
| Changed adapter tensors | 336 / 336 | 336 / 336 |
| Adapter safetensors sha256 | `4efe3ec1...59e6d7de` | `845cdeb1...8431f000` |
| Measured fit | `NATIVE_SAFE` (peak ~1.4 GB / 12.34 GB) | `NATIVE_SAFE` |
| GPU power / temp max / energy | 43.5 W / 42 C / 495.3 J | 43.9 W / 42 C / 477.4 J |
| scientifically_complete | `True` | `True` |
| Post-run env state | `HARDWARE_VERIFIED`, drift `false` | `HARDWARE_VERIFIED`, drift `false` |

Both matched arms completed the full plan -> seal -> run -> admit -> manifest lifecycle with forced math and
forced flash respectively; the export/artifact-admission path (the v5 blocker) now succeeds, and telemetry
is `scientifically_complete=True`. Both smokes ran one at a time (Ollama unloaded, GPU idle-confirmed,
supervised subprocess, 600 s silence timeout, 200 ms internal telemetry), peaked 42 C (<< 85 C) with zero
swap growth and no shared-GPU-memory spill, and released the GPU to 10 MiB. Run + telemetry evidence:
`/mnt/training-nvme/corpusstudio/runs/ieee-linux-training/v6-smoke-73b756c/{math,flash}/`; plans:
`.../v6-bringup-73b756c/plans-chat/{math-v6,flash-v6}/`. **Verdict: `V6_MATH_AND_FLASH_BRINGUP_PASS`.**

*Honestly-recorded gap (non-blocking, kernel-independent):* `nonpadding_tokens_per_second` and
`supervised_tokens_per_second` read `0.0` on both runs. Real training occurred (loss fell; 336 tensors
changed; TRL built labels for all 8 rows), so a supervised step necessarily processed tokens: **the
recorded `0.0` must be read as UNAVAILABLE (null), NOT a measured zero.** Root cause: the deployed v6
worker's #462 collate-fn observer never fired, because on the pinned stack (trl 1.8.0 / transformers
5.13.1 / accelerate 1.14.0) `Trainer._get_dataloader` returns an accelerate-prepared `DataLoaderShard`
whose base loader captured `collate_fn` at `accelerator.prepare` time, so reassigning `.collate_fn` on
the returned shard is silently bypassed. The raw v6 records are **preserved unchanged**; the durable
classification `TOKEN_THROUGHPUT_UNAVAILABLE_OBSERVER_MISSED_BATCHES` and the list of v6 metrics thereby
unusable (token throughput and every token-normalized metric, incl. energy-per-token) live in the
sidecar `.../evidence/v6-smoke-73b756c/TOKEN_THROUGHPUT_UNAVAILABLE_OBSERVER_MISSED_BATCHES.md`.
`scientifically_complete=True` holds **only as historical resource-completeness under effective matrix
1.3.0** (tokens/sec was not a required paper field there); it is NOT a throughput or paper-performance
claim. The fix observes `inputs` at `training_step` (the trainer's own consumption boundary) and adds a
throughput-validity gate + separable `scientific_throughput_complete` / `paper_performance_complete`
completeness; because the observer runs in the worker child it changes worker bytes -> the **v7**
lineage. This remains a 0.5B feasibility bring-up, NOT a 7B or full-training result.

### Manager-1.3 v7 pair - token-throughput observer validated (math + flash), 2026-07-16

The v6 token-throughput `0.0` was reclassified UNAVAILABLE (above). PR **#466** (merge `25c901ec`) moved
token accounting to observe `inputs` at `SFTTrainer.training_step` (the trainer's own un-bypassable
consumption boundary) and added a throughput-validity gate + separable
`scientific_resource_complete` / `scientific_throughput_complete` / `paper_performance_complete`
completeness. Because that observer runs in the worker child it changes worker bytes, so research amendment
**0004 -> effective matrix 1.4.0** (effective-matrix sha256
`0ce1fbd425e0401824c3f75f430b72bc4cc51b74e592399cd503a7084c4e593e`, `RESERVED_IDENTITIES.v4.json` sha256
`f0c78fa77ad8f3d93035d58e4cd6b8781d095873e6bdd8c5e41a8ea970c4c27b`) reserves every v1-v6 identity,
allocates the v7 environment ids, and requires the worker source to descend from `25c901ec`. A fresh
reproducible v7 wheel `corpus_studio_engine-1.3.0-py3-none-any.whl` sha256
`090f879b46d52e8c33c96fad8aeb61a41c320d44c20bd189dabfd5be606479b2` was built twice byte-identically from
source commit `21aa81d97ff752709fd4d03791288c1bb76a2339`.

| Item | Math (blue) | Flash (green) |
|---|---|---|
| Environment id | `backend-corpus-studio-research-math-v7` | `backend-corpus-studio-research-flash-v7` |
| Lock hash | `35d1daf1...28907438` | `0409a632...9284dca8` |
| Forced kernel | `torch_sdpa_math` | `torch_sdpa_flash` |
| Plan id / hash | `plan-019f6944...` / `d960cf0e...` | `plan-019f6948...` / `089dda03...` |
| Run id | `run-019f6956-d55c-7684-8c7a-9ebb7bad7a04` | `run-019f6966-1c87-71e4-8c30-6fc5fa085caf` |
| Terminal state | `succeeded` | `succeeded` |
| Steps / losses | 12 / 5.4336 -> 0.3937 | 12 / 5.4319 -> 0.3774 |
| Per-step token counts | positive nonpadding+supervised, obs=1 every step | positive nonpadding+supervised, obs=1 every step |
| Measured throughput (steps 3-12) | ~104 tok/s | ~108 tok/s |
| `scientific_throughput_complete` | `True` (as-dispatched) | `True` (as-dispatched) |
| Changed adapter tensors | 336 / 336 | 336 / 336 |
| Adapter safetensors sha256 | `4efe3ec1...59e6d7de` | `845cdeb1...8431f000` |
| Measured fit | `NATIVE_SAFE` (peak ~1.4 GB / 12.34 GB) | `NATIVE_SAFE` |
| GPU temp max / energy-per-1k-nonpad-tok | 41 C / 486.85 J | 42 C / 487.79 J |
| Post-run env state | `HARDWARE_VERIFIED`, drift `false` | `HARDWARE_VERIFIED`, drift `false` |

**The v6 token-throughput gap is fixed and validated on both arms:** positive nonpadding AND supervised
token counts with `observed_microbatches=1` on every one of the 12 optimizer steps, every measured step's
rate equal to observed tokens / duration, `scientific_throughput_complete=True` as-dispatched. Both ran
one at a time (Ollama unloaded, GPU idle-confirmed, supervised subprocess, 600 s silence timeout, 200 ms
telemetry, 1 Hz temp watchdog), peaked 41-42 C with zero swap growth, and released the GPU to 10 MiB. Run
+ telemetry evidence: `/mnt/training-nvme/corpusstudio/runs/ieee-linux-training/v7-smoke-21aa81d9/{math,flash}/`;
plans: `.../v7-bringup-21aa81d9/plans-chat/{math-v7,flash-v7}/`; sealed evidence:
`.../evidence/v7-smoke-21aa81d9/` (`SHA256SUMS`). **Verdict: `V7_MATH_AND_FLASH_THROUGHPUT_PASS`.**

*Honestly-recorded non-scientific caveat (resolved; follow-up filed):* as-dispatched, both summaries
reported `scientific_resource_complete=false` / `paper_performance_complete=false` with
`missing_required_paper_fields=["identity.repository_commit"]`. Every scientific MEASUREMENT was complete;
the only gap was a provenance identity field: the shipped telemetry reader reads key `source_commit` from
the wheel's `BUILD_PROVENANCE.json`, but the v7 build tooling recorded the authentic commit under
`audited_commit` and omitted `source_commit`. Both summaries were re-derived from the PRESERVED raw records
with `repository_commit` taken from the sealed sidecar's own `audited_commit`
(`21aa81d9...`) -> `scientific_resource_complete` / `scientific_throughput_complete` /
`paper_performance_complete` / `scientifically_complete` all `true`; a field-by-field diff shows ONLY the
provenance field, the completeness flags it drives, and `generated_at` changed (zero measurement changes).
Originals preserved; corrected files are `RunTelemetrySummary.rederived-authentic-commit.json` per run.
Follow-up (does NOT block v7): the worker build-provenance generator must emit `source_commit`; SHA256SUMS
seals the A4 sidecar so it is not mutated retroactively - the fix applies to future wheel builds. This
remains a 0.5B feasibility bring-up, NOT a 7B or full-training result.

### From-scratch pretraining `workload_verified` bring-up (PRODUCT), 2026-08-06

The **pretraining** execution variant is promoted to `workload_verified` (a PRODUCT claim, NOT a sealed
IEEE cell) on the evidence below - from-scratch full-parameter pretraining now runs end to end on this host.

- **Milestone worker wheel:** `corpus_studio_engine-1.3.0-py3-none-any.whl` sha256
  `8818d3ad1b16fd172652adad717cefd4c97d85a4d226ce600e37c05134cca168`, source commit `fa6ce973...` (clean-source
  `build-worker-wheel`, provenance-sealed), at `/mnt/training-nvme/artifacts/corpusstudio-worker/pretraining-fa6ce97/`.
  It carries the pretraining worker (`run_pretraining`, `PretrainingRunner`, packed-corpus + evidence-capture).
- **Managed env:** `backend-corpus-studio-pretraining-v2`, created + probed **`HARDWARE_VERIFIED`** (lock
  `e3852fe8...`), torch 2.11.0+cu128 + the fa6ce97 worker. Creating it surfaced + fixed an env-manager infra gap:
  the PyTorch index omits sha256 for a few pure-python deps, so the env-manager now binds them by a self-computed
  content hash (#808).
- **Sealed bring-up run:** a from-scratch full-parameter **124M GPT-2** (n_embd 768 / 12L / 12H, random init)
  trained on the RTX 5070 at **seq 1024, 20 steps, peak 3.28 GiB / 12** (loss 5.34 -> 0.87, ~7.6 steps/s) through
  the first-party `PretrainingRunner` + the supervisor's independent reload-verify
  (`validate_pretraining_success_evidence`): **supervisor-admitted `PretrainingSuccessEvidence`** - optimizer
  created, one finite loss per step, **all 148/148 trainable tensors changed with an observed materialized
  gradient**, and `model.safetensors` reload-verified to reproduce the trained export state.
- **Honesty scope.** PRODUCT workload_verified, not a sealed IEEE 7B cell. It ran the first-party runner + the
  full supervisor admission gate, invoked directly - the authorized new-variant evidence-gathering; the production
  dispatch gate (`required_runner_lane`) refuses pretraining until it is workload_verified, exactly as here. The
  torch/GPU stack is the sealed HARDWARE_VERIFIED env; the control-plane code is `main` (byte-identical to the
  wheel's fa6ce97 source + the merged #808 fix). This is NOT a 7B, full-corpus, or convergence result - it is a
  feasibility + honesty-evidence bring-up.
- **Managed `platform-run --subprocess` shipping path GPU-GREEN 2026-08-06 (PR #810 / #811 / #812).** The
  original bring-up ran the runner + supervisor DIRECTLY (a bypass); running the real managed subprocess path
  then surfaced 10 shipping-path bugs, all fixed: #810 (7 execution-path defects: a RunManifest terminal-fit
  validator crash on a proven pretraining fit, a dead subprocess argparse lane, silent epoch collapse, sharded
  save, un-run-scoped output, unthreaded corpus_root), #811 (env-manager managed capability snapshot for
  probe-less recipes: `probes=None` -> `"null"`, and a 32 KB probe-log-tail truncation of a big report), and
  #812 (the worker's native/C fd-1 output corrupting the framed stdout protocol - fixed by binding the protocol
  onto a private dup of fd 1 and pointing fd 1 at stderr). A managed `platform-run --subprocess` from-scratch
  run then **SUCCEEDED end to end** on the RTX 5070 (env `backend-corpus-studio-pretraining-hardened-v4`, base
  recipe source-installing the fixed worker): `STATE succeeded`, `final_fit NATIVE_SAFE`, supervisor-admitted
  `PretrainingSuccessEvidence`, 15/15 sealed steps, loss 5.158 -> 4.316, 28/28 tensors changed with observed
  gradients, model reload-verified, run-scoped `model` artifact. So the managed pretraining shipping path is now
  workload_verified through the REAL dispatch, not just the direct bypass. (The base recipe source-installs the
  worker; the provenance-sealed wheel `4bacd4ef...` at `.../pretraining-hardened-c0e9870/` is for a future
  HASH-PINNED deployment via the `-verified` recipe, which `env-create` does not yet provision.)
- **Re-confirmed GREEN with current main 2026-08-07.** The #812 proof predates the S0 shared-optimizer
  refactor + the F3/F5 pretraining-worker audit fixes, which changed the pretraining worker. A managed
  `platform-run --subprocess` re-run on current main (a tiny gpt2 4M random-init, seq 512, 10 steps)
  **SUCCEEDED**: `STATE succeeded`, `final_fit NATIVE_SAFE`, supervisor-admitted `pretraining_success_evidence`
  (loss 5.19 -> 3.46, 52/52 tensors changed with observed gradients, model reload-verified) - so all THREE
  in-process verticals (pretraining, DPO, full-finetune) are now green through the REAL managed dispatch on
  current main. (Hash-pinned wheels + sealed non-editable envs remain the reproducible-deployment last mile.)
- **Wheel source-ahead after the 2026-08-06 audit.** The pre-merge audit fixed two pretraining-worker gaps
  (the sealed optimizer `optim`/betas/eps were not threaded into `TrainingArguments`; an imported tokenizer
  with no `tokenizer.json` silently skipped its required content pin - now fail-closed). These change
  pretraining worker bytes, so the milestone wheels above (`8818d3ad`, `4bacd4ef`) are now SOURCE-AHEAD: a
  fresh provenance-sealed wheel is owed for any hash-pinned deployment. The in-process and base-recipe
  source-install routes (how the workload_verified evidence was gathered) carry the fix immediately.

### Offline DPO (preference) `workload_verified` bring-up (PRODUCT), 2026-08-06

The **preference_dpo** execution variant is promoted to `workload_verified` (a PRODUCT claim, NOT a sealed
IEEE cell) on the evidence below - offline DPO now runs end to end on this host through the first-party
`PreferenceRunner` lane.

- **Sealed bring-up run:** offline DPO of **Qwen3-4B-Instruct-2507** (nf4 QLoRA r16 all-linear, frozen
  reference model) trained on the RTX 5070 at **seq 1024, 15 steps, beta 0.1, peak 5.79 GiB / 12** through
  the first-party `PreferenceRunner` + the supervisor's independent adapter reload-verify
  (`validate_preference_success_evidence`): **supervisor-admitted `PreferenceSuccessEvidence`** - optimizer
  created, one finite loss per step (**0.6931 -> 0.0739**), a monotonic **reward margin 0.0 -> 31.64**
  (chosen reward up, rejected down), `reference_model_frozen`, **all 504/504 trainable LoRA tensors changed
  with an observed materialized gradient**, and `adapter.safetensors` reload-verified to reproduce the
  trained export state (bytes matched the worker's proposal).
- **Worker primitive:** the sealed config is consumed directly; training uses a sequence-chunked log-prob
  that reaches seq 4096 on this 12 GB card where trl / off-the-shelf-liger cap at ~1024 (that seq-4096
  correctness was validated separately, exploratory, at peak 9.49 GiB).
- **Honesty scope.** PRODUCT workload_verified, not a sealed IEEE cell. It ran the first-party runner + the
  full supervisor admission gate, invoked directly - the authorized new-variant evidence-gathering; the
  production dispatch gate (`required_runner_lane`) refuses preference until it is workload_verified, exactly
  as here. The torch/GPU stack is the sealed HARDWARE_VERIFIED env `backend-corpus-studio-pretraining-hardened-v4`;
  the control-plane + worker code is the DPO worker branch (#781). This is a feasibility + honesty-evidence
  bring-up, not a convergence or preference-quality result (the synthetic pairs are trivially separable - the
  large reward margin reflects that, not model quality). The managed `platform-run --subprocess` route (a DPO
  worker wheel + sealed env) is the deployment follow-up, exactly as for pretraining (in-process routes now).
- **Pre-merge audit hardening (2026-08-06).** A deep audit before merging the DPO branch found + fixed 5
  seal-fidelity/integrity gaps (the DPO math, evidence, adapter reload-verify, run-scoping, and fail-closed
  data guards were all sound). The DPO worker fixes were GPU re-validated on BOTH sealed optimizer paths:
  `adamw_torch` admitted at peak **5.79 GiB** and `paged_adamw_8bit` admitted at peak **5.56 GiB** (the
  pre-fix worker silently substituted a full-precision AdamW - dropping both the seal and that headroom).
  Grad clipping now honors the sealed `max_grad_norm`. Both runs were supervisor-adapter-reload-verified.
- **Managed `platform-run --subprocess` shipping path GREEN 2026-08-07.** The DPO managed subprocess path
  (worker spawned from the managed env -> config-consuming DPO worker -> supervisor adapter reload-verify)
  went through the REAL production dispatch after fixing TWO CLI shipping-path bugs: (1) the dataset
  conformance preflight rejected the `preference` format (`unknown dataset_format 'preference'`) - added a
  first-class preference (prompt/chosen/rejected) classifier; (2) the shared SFT `TrainingDataPolicy` was
  built with `dataset_format=preference` and rejected it (its Literal is instruction/chat/trace) - a
  preference plan seals its OWN `PreferenceDataPolicy`, so the SFT policy is no longer built for preference.
  A managed `--subprocess` run on the RTX 5070 (Qwen3-4B nf4 DPO, seq 1024, 12 steps) **SUCCEEDED end to
  end**: `STATE succeeded`, `final_fit NATIVE_SAFE`, supervisor-admitted `preference_success_evidence`
  (loss 0.6931 -> 0.1022, reward margin 0 -> 27.49, adapter reload-verified). A hash-pinned worker wheel +
  sealed env is the reproducible-deployment follow-up.

### Full-parameter SFT `workload_verified` bring-up (PRODUCT), 2026-08-07

The **dense_full_finetune** execution variant is promoted to `workload_verified` (a PRODUCT claim, NOT a
sealed IEEE cell) on the evidence below - full-parameter supervised fine-tuning now runs end to end on this
host through the first-party `FullFinetuneRunner` lane.

- **Sealed bring-up run:** full-parameter SFT of **Qwen2.5-0.5B-Instruct** (bf16, ALL parameters trainable -
  no adapter, no 4-bit) on the RTX 5070 at **seq 512, 12 steps, peak 5.01 GiB / 12** through the first-party
  `FullFinetuneRunner` + the supervisor's independent full-model reload-verify
  (`validate_full_finetune_success_evidence`, reusing the pretraining `_reload_verify_full_model`):
  **supervisor-admitted full-model success evidence** - optimizer created, one finite loss per step
  (**2.28 -> 0.17**), **all 290/290 trainable tensors observed a materialized gradient** (the honesty
  invariant: a real optimizer stepped the complete inventory), **265/290 changed** (the remaining 25 are
  sub-bf16-precision fine-tune updates that do not register a byte change - honest evidence, not a defect),
  and `model.safetensors` reload-verified to reproduce the trained export state.
- **Worker:** `run_full_finetune` reuses the pretraining worker's full-model machinery verbatim (gradient
  hooks, execution tracker, single-file save, success-evidence build); it differs only by `from_pretrained`
  (a real base) + an SFT text dataset. Whole-sequence loss (matches the current first-party SFT trainer).
- **Honesty scope.** PRODUCT workload_verified, not a sealed IEEE cell. It ran the first-party runner + the
  full supervisor admission gate, invoked directly (the authorized new-variant evidence-gathering); the
  production dispatch gate (`required_runner_lane`) refuses full_finetune until it is workload_verified,
  exactly as here. Quality follow-up: an fp32 master-weight / mixed-precision full-finetune precision (so all
  tensors register a change).
- **Managed `platform-run --subprocess` shipping path GREEN on the first try 2026-08-07.** Unlike pretraining
  (which surfaced 10 shipping-path bugs), full-finetune went through the REAL production dispatch (CLI
  `platform-plan --adapter-method full_finetune --export-format merged_safetensors` -> `platform-run
  --subprocess` -> worker spawned from the managed env -> `required_runner_lane` "full_finetune" ->
  supervisor admission -> RunManifest) cleanly, because the pretraining audit's fixes were generalized (the
  shared `build_lane_runner`, the fd-level worker-protocol binding, the RunManifest full-model admission, the
  `_RUNNER_CHOICES` dead-lane fix). The ONE bug: the CLI omitted `--adapter-method`/`--export-format`, so
  full-finetune was workload_verified but UNREACHABLE from the shipping CLI - fixed + reachability-tested. A
  managed `--subprocess` run on the RTX 5070 (env `backend-corpus-studio-pretraining-hardened-v4`, editable
  checkout install) **SUCCEEDED end to end**: `STATE succeeded`, `final_fit NATIVE_SAFE`, supervisor-admitted
  full-model evidence, 12/12 steps, 265/290 tensors changed with 290/290 observed gradients,
  `model.safetensors` reload-verified to the SAME sha256 as the direct bring-up (`a51dcf6f...`) - reproducible
  through the real dispatch, not just the bypass. A hash-pinned worker wheel + a sealed (non-editable) env is
  the reproducible-deployment follow-up.

### End-to-end from-scratch lifecycle chain (pretrain -> SFT -> DPO), 2026-08-07

The four dense training verticals were run as **one connected chain** on this host - a clean random-init
model taken through its whole early life, every stage dispatched through the REAL managed
`platform-run --subprocess` path, and **each stage's saved output pinned as the next stage's base by
content digest** (`source="local_directory"`, not a Hub revision). This is the from-scratch lifecycle
acid test - the demonstration the training vertical was built for - run end to end for the first time.

| stage | task (variant) | base | result (RTX 5070, managed subprocess) |
|---|---|---|---|
| 1 | pretraining (`pretraining`) | random init | succeeded, NATIVE_SAFE, 10 steps, loss **5.204 -> 3.359**, **52/52** full-model tensors observed a gradient |
| 2 | SFT (`dense_full_finetune`) | stage-1 output dir | succeeded, NATIVE_SAFE, 10 steps, loss **5.156 -> 3.699**, **52/52** observed, reload-verified |
| 3 | DPO (`preference_dpo`) | stage-2 output dir | succeeded, NATIVE_SAFE, 10 steps, loss 0.693 -> 0.688, reward margin **0.0 -> 0.105**, **32/32** LoRA tensors observed, adapter reload-verified |

- **The model:** a 4-layer from-scratch **GPT-2** (`n_layer 4, n_head 4, n_embd 256, n_positions 512`)
  with a **from-scratch BPE tokenizer** trained from the tiny corpus (`vocab_size 177`; special tokens
  `<bos>/<eos>/<pad>/<unk>`). Stage 1 random-inits from config; stages 2-3 `from_pretrained` the prior
  stage's saved directory. Content-digest pinning (`stable_directory_sha256`) is what lets one stage's
  output be another stage's sealed input.
- **Honesty scope.** This is a PRODUCT **integration** proof - it shows the stage *handoffs* work through
  the real dispatch (plan -> subprocess worker -> lane -> supervisor reload-verify -> RunManifest) with
  full gradient coverage and reload-verification at each stage. It is **not** a model-quality claim: the
  model and vocab are toy, so the DPO reward margin is deliberately small. Not a sealed IEEE cell.
- **It found real bugs.** Connecting the stages surfaced three integration defects at the handoffs that
  no single-vertical run hit, all fixed on the same branch (PR #824): a from-scratch tokenizer must
  declare an eos token (else the base is unpaddable - now refused at planning), a padless base fails
  closed in the full-finetune worker, and DPO falls back to the raw prompt when the base tokenizer has no
  chat template (a from-scratch pretrain output has none) instead of crashing the run.

### Native 8-bit + 16-bit precision ladder runs end to end, 2026-08-07

The first-party adapter trainer's precision ladder (widened from nf4-only in the #825 worker slice) is now
**selectable and RUNNABLE** for **8-bit (int8/QLoRA)** and **16-bit (bf16/LoRA on an unquantized base)** on
this host - not just nf4. Two new bounded GPU capability probes demonstrate the complete execution tuples
the planner's exact-tuple gate requires, so `platform-plan --quantization {int8,none}` seals a runnable plan
instead of failing closed:

- **Probes (RTX 5070, PASS):** `cuda_bf16_lora_math_execution` proves `bf16/none/lora/math/torch_sdpa_math`
  (16-bit); `cuda_int8_qlora_math_execution` proves `bf16/int8/qlora/math/torch_sdpa_math` (8-bit,
  bitsandbytes LLM.int8()). Each runs a real forward/backward + AdamW step + adapter save/reload and emits
  its exact tuple + proven quantization axis. They run FRESH at plan time (the unmanaged `platform-plan`
  path calls `run_capability_probes`), so the editable install picks them up with no env recreate.
- **End to end (in-process, both succeeded, NATIVE_SAFE):** `platform-plan --quantization none` ->
  `platform-run` trained a 16-bit LoRA adapter; `platform-plan --quantization int8` -> `platform-run`
  trained an 8-bit QLoRA adapter. Both sealed the resolved tuple, passed the objective + execution-config
  admission, and reload-verified.
- **Two integration fixes running it surfaced.** (1) The planner labelled a quantized-base + LoRA plan
  `lora` for any non-nf4 quant; it now reads `qlora` for ANY quantized base (nf4 unchanged), matching the
  int8 probe's tuple. (2) The `qlora` training objective declared only `int4/nf4`, so int8 sealed at
  planning but was refused at execution ("sealed quantization does not match the objective"); it now admits
  `int4/int8/nf4`. That changes the qlora objective hash, but no committed seal/golden breaks and sealed
  IEEE cells reproduce against their PINNED wheel (objective frozen at seal time), so nf4 cells are
  unaffected (they still seal nf4 - the `Literal["nf4"]` contract is unchanged).
- **Honesty scope.** PRODUCT, not a sealed IEEE cell. Validated on the **in-process** platform-run path;
  routing these modes through the **managed `--subprocess`** path additionally needs the two probe names
  added to the training recipe's `capability_probes` + a managed re-probe (a documented follow-up, the same
  managed-shipping-path step the other verticals carry). fp4 + int4 remain declared-but-unprobed (fail
  closed until a probe proves them).

### Intermediate checkpoint WRITING on the adapter SFT lane, 2026-08-07

The exact-lineage checkpoint machinery (`checkpoint_io.py` - seal/verify/admit + the `CheckpointCoordinator`,
CPU-bitwise-proven by the 3-process equivalence test) is now WIRED into the real `SFTTrainer` loop and RUNS
on this host. `platform-plan --checkpoint-cadence N` seals an enabled checkpoint policy on the adapter SFT
lane (`save_strategy="steps"`; the full-parameter lane refuses a cadence, whose worker does not write yet),
and `run_training` drives an on-step-end `CheckpointCoordinator` that seals a checkpoint every N optimizer
steps. HF's own saver stays off (`build_training_kwargs` forces `save_strategy="no"`), so the coordinator
owns writing and the two never double-write; a checkpoint-write fault degrades to a stderr log, never into
the training loop.

- **GPU proof (RTX 5070, in-process, 16-bit LoRA on a from-scratch GPT-2):** `--checkpoint-cadence 2` over 6
  steps wrote `step-2 / step-4 / step-6`; `--checkpoint-cadence 1` over 3 steps wrote `step-1 / step-2 /
  step-3` - the cadence fires exactly. Each checkpoint is a COMPLETE sealed set under the run-scoped
  `.../artifacts/checkpoints/step-*`: `CheckpointManifest.json` + `adapter_state.pt` + `optimizer.pt` +
  `scheduler.pt` + `rng.pt` + `sampler.pt`.
- **Honesty scope.** This is checkpoint WRITING. Consuming a checkpoint to RESUME is the follow-on slice
  below (now also proven). `run_training` is `# pragma: no cover` (proven by a run); the coordinator, the
  threading, and the policy sealing are unit-tested.

### Exact-lineage checkpoint RESUME on the adapter SFT lane, 2026-08-07

Consuming a sealed checkpoint to CONTINUE a run now works end to end on this host (the mirror of writing,
above). `platform-run <plan> --resume-from <checkpoint-dir>` fully integrity-verifies the checkpoint AND
proves it a compatible resume source for the plan (exact-lineage or nothing) BEFORE anything is restored,
then the first-party trainer materializes an HF `resume_from_checkpoint` layout from the sealed files
(`materialize_hf_checkpoint`: optimizer / scheduler / RNG->HF / adapter safetensors+config / TrainerState)
and lets SFTTrainer's own proven resume (optimizer / scheduler / RNG / data-cursor skip / step count)
continue. The sealed format is the trust anchor; HF is the restore engine.

- **GPU proof (RTX 5070, in-process, 16-bit LoRA on a from-scratch GPT-2):** a write run (cadence 1, 4
  steps) sealed `step-1..4`; `--resume-from step-2` on the SAME plan **succeeded, NATIVE_SAFE**, with the
  HF global_step continuing from 2 - the resumed run trained exactly the two remaining steps and its sealed
  evidence records `resumed_from_optimizer_step=2` + step_losses for the ABSOLUTE steps `[3, 4]`. The write
  run's RunManifest recorded all 4 checkpoints (a resume discovers a parent from the record, not by
  scanning disk).
- **The honesty core is resume-aware.** Running it surfaced THREE step-sequence checks that assumed a run
  starts at step 1 - the tracker's `on_step_end` sequence check, the tracker's `finalize` coverage check,
  and the `TrainingExecutionEvidence` contract validator - all now count from `resumed_from+1`, so a
  resumed run's evidence is honest rather than falsely rejected. The guarantee is EXACT VERIFIED LINEAGE
  (our seal) + HF-standard numerical continuation (not bitwise; bitwise holds only on the hand-rolled
  reference). SupportLevel promotion + a pinned wheel + a full 7B/seq-4096 resume remain the sealed-deploy
  follow-up.

## Verification boundary — what `HARDWARE_VERIFIED` does and does NOT prove

`HARDWARE_VERIFIED` is the **Environment Manager** evidence level, not a training-run result.
Per [`ENVIRONMENT_MANAGER.md`](ENVIRONMENT_MANAGER.md), the legacy passing
`reference_backend_hardware` probe proves its managed interpreter can, on this GPU: allocate
CUDA memory, read compute capability, produce a BF16 signal, construct a bitsandbytes 4-bit
layer, run a **minimal** GPU forward/backward, and execute the safe **math** SDPA attention
path. The readiness-v2 and readiness-flash-v1 locks add their own distinct complete tiny QLoRA tuple
evidence; they do not broaden the legacy lock or one another.

That is **real native-Linux GPU evidence** — the old "until the Linux NVMe is installed in the
final RTX 5070 machine, do not claim native-Linux" precondition is now satisfied *for the
environment probe*. It is **not** proof of any of the following, which remain unverified and
must not be claimed from this state alone:

- **Full-sequence 7B training success as a SEALED result** — the probe is a minimal kernel check,
  not a real workload. (The `~10.8 GB @ seq 1024` / `~13.8 GB @ 2048` VRAM ceiling on record was
  measured on native-Windows/WDDM.) NOTE (2026-07-19): an **exploratory/product** run (NOT a sealed
  IEEE cell) has since trained 7B QLoRA at seq 4096 on this host and measured native-Linux workload
  VRAM (peak ~11.77 GB @ seq 4096; envelope math<=2048 / flash<=3072 / flash+liger+paged=4096 - see
  `docs/CURRENT_STATE.md`, `examples/wbg/README.md`). NOTE (2026-07-30): the **product** lineage reproduced this end to end on a freshly created `backend-corpus-studio-flash-liger-paged-v9-product` environment (`HARDWARE_VERIFIED`) built from the sealed v9 **product** wheel (sha `45bdd989`, source `7ae4ea6`) - the after-r8 WBG run measured **`NATIVE_TIGHT`** (peak ~11.4 GB, zero spill) and passed its held-out completeness eval. The full recipe, training metrics, and eval numbers are canonical in `examples/wbg/README.md` (run `run-019fb083`, evidence under `runs/wbg-after-seq4096-v9product/`). Both are exploratory evidence, not a sealed
  research result, and does not retire the paper's immutable ladder.
- **DeepSpeed / FSDP / CPU / NVMe offload** — no offload backend is implemented; only the dense
  `backend-corpus-studio` reference exists.
- **Real offload fit, PCIe/NVMe throughput, sustained-write endurance** — the NVMe has not been
  benchmarked (`platform-storage` is non-destructive and reads no SMART data).
- **Bare-Linux flash BEYOND the bounded 0.5B tuple** — the manager-1.3 **v6** green run executed
  **12 real optimizer steps** of the 0.5B, sequence-256, QLoRA tuple under forced `torch_sdpa_flash`
  on native Linux (adapter admitted, predicted-fit `NATIVE_SAFE`; see the v6 section above), so
  forced-SDPA-flash **at that exact bounded scale is now VERIFIED** — this supersedes the older
  "stopped before step 1" wording. It does **NOT** extend to any of: Transformers
  `flash_attention_2`, an external `flash-attn` package, or any Windows/WDDM flash path — those stay
  unverified/refused-on-Windows. (Sequence 4096, the full WBG corpus, and 7B flash SDPA DID run
  together in the 2026-07-19 **exploratory/product** run above; that is not a SEALED research result.)
- **MoE runtime capability** — static inspection only (Phase 8); no MoE execution.

"Installed ≠ supported" and "a completed step ≠ proven fit" both still hold: a passing
environment hardware probe is the *precondition* for the GPU-workload bring-up steps in
[`HANDOFF.md`](../HANDOFF.md) §7, not their completion.

## Re-verifying this state

From the engine control-plane venv:

```bash
cd /mnt/training-nvme/repos/CorpusStudio/engine
.venv/bin/corpus-studio env-status backend-corpus-studio --refresh --json
.venv/bin/corpus-studio env-probe  backend-corpus-studio --json
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
```

A changed recipe, runtime, installed package, source, or CUDA / compute-capability signal
flips the environment out of `HARDWARE_VERIFIED` into a `DRIFTED` / `DEGRADED` / `BROKEN`
state; re-probe before trusting this file.
