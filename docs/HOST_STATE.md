# Host State — Native-Linux RTX 5070 Workstation

**Last verified:** 2026-07-16 (manager-1.3 **v6** math/flash environments created, probed, and
dispatched: both 0.5B smokes succeeded - `V6_MATH_AND_FLASH_BRINGUP_PASS`; see the v6 section below.
Earlier v1-v5 environment, plan, and run evidence remains preserved and non-reusable; legacy
environment, GPU, and paths were checked 2026-07-14).

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
rollback evidence. Manager 1.3 does not grant it a new health/planning claim without a replacement
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
`record_verified_entries < record_entries`. Manager 1.3 tags the new meaning explicitly with
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
changed; TRL built labels for all 8 rows), so this is a runner-side token-observer instrumentation gap in
the deployed v6 worker (the #462 collate-fn observer did not capture batch token counts under trl 1.8.0 /
transformers 5.13.1), not a workload failure. tokens/sec is not a required paper field, so
`scientifically_complete=True` holds; the field is reported as measured (0.0), no gate was weakened. Fixing
it changes worker bytes -> a future **v7** lineage. This remains a 0.5B feasibility bring-up, NOT a 7B or
full-training result.

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

- **Full-sequence 7B training success** — the probe is a minimal kernel check, not a real
  workload. (The `~10.8 GB @ seq 1024` / `~13.8 GB @ 2048` VRAM ceiling on record was measured
  on native-Windows/WDDM; native-Linux *workload* VRAM behavior is not yet measured.)
- **DeepSpeed / FSDP / CPU / NVMe offload** — no offload backend is implemented; only the dense
  `backend-corpus-studio` reference exists.
- **Real offload fit, PCIe/NVMe throughput, sustained-write endurance** — the NVMe has not been
  benchmarked (`platform-storage` is non-destructive and reads no SMART data).
- **Bare-Linux flash for a real optimizer step or sequence 4096** — matched managed environments
  verified separate tiny math and forced-`torch_sdpa_flash` tuples. The manager-1.2 real 0.5B
  attempts stopped before step 1 on the earlier verifier mismatch; the manager-1.3 v4 math attempt
  stopped before step 1 on TRL's constructor-time adapter recast, and its paired flash plan was not
  dispatched. Neither
  environment is the same identity as Transformers `flash_attention_2` or an external `flash-attn`
  package, and neither is full-sequence 7B proof.
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
