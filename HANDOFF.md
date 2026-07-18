# CorpusStudio — Session Handoff

**Last updated:** 2026-07-16 (**v7 0.5B GPU bring-up PASSED - `V7_MATH_AND_FLASH_THROUGHPUT_PASS`**: the
v6 token-throughput observer gap is fixed (PR #466, merge `25c901ec`; amendment 0004 -> matrix 1.4.0;
reproducible v7 wheel `090f879b...` from source `21aa81d9`; matched `research-{math,flash}-v7`
environments) and **validated on both arms** - positive non-padding AND supervised counts with
`observed_microbatches=1` on every measured step, rates equal to observed tokens / duration,
`scientific_throughput_complete=True` as-dispatched; runs `run-019f6956...` (math, `torch_sdpa_math`) /
`run-019f6966...` (flash, `torch_sdpa_flash`), 12 steps each, 336/336 adapter tensors, adapter admitted,
measured `NATIVE_SAFE`, under `.../runs/ieee-linux-training/v7-smoke-21aa81d9/`; sealed evidence
`.../evidence/v7-smoke-21aa81d9/`. One non-scientific caveat (resolved, follow-up filed): the v7 build
tooling wrote the source commit under `audited_commit` not the telemetry reader's canonical
`source_commit`, so the auto summary lacked `identity.repository_commit`
(`scientific_resource_complete=false`); re-deriving from the preserved raw records with the authentic
sealed commit yields `paper_performance_complete=true` with zero measurement change. Prior milestone
below: **`V6_MATH_AND_FLASH_BRINGUP_PASS`**.) After
the v5 bring-up hit its first real GPU training then failed at export, the worker-child corrections #461
(narrow `training_args.bin` admission) and #462 (complete paper telemetry) forced a fresh **v6** lineage:
research amendment **0003 -> effective matrix 1.3.0** (reserves all v1-v5 identities), a reproducible v6
wheel `bdc32196...` built twice byte-identically from source `73b756c`, matched
`backend-corpus-studio-research-{math,flash}-v6` environments (both `HARDWARE_VERIFIED`, drift false;
forced `torch_sdpa_math` / `torch_sdpa_flash`), fresh matched chat RunPlans, all pre-dispatch fixture /
artifact / telemetry / disjointness gates green, and **both 0.5B smokes SUCCEEDED** (12 steps each, loss
~5.43 -> ~0.38, 336/336 adapter tensors changed, adapter admitted, measured `NATIVE_SAFE`,
`scientifically_complete=True`, GPU released, post-run `HARDWARE_VERIFIED`). Runs
`run-019f688c...` (math) / `run-019f6892...` (flash) under
`/mnt/training-nvme/corpusstudio/runs/ieee-linux-training/v6-smoke-73b756c/`. Honestly-recorded
non-blocking gap: `nonpadding/supervised_tokens_per_second` read `0.0` - now reclassified as
**UNAVAILABLE (null), not a measured zero** (`TOKEN_THROUGHPUT_UNAVAILABLE_OBSERVER_MISSED_BATCHES`;
sidecar under `.../evidence/v6-smoke-73b756c/`). Root cause: the #462 collate-fn observer never fired
because the accelerate-prepared `DataLoaderShard` ignores a `.collate_fn` reassignment on the pinned
stack. **Fix now landed on the source side** (branch `fix/token-throughput-accounting`): observe
`inputs` at `training_step`, emit raw per-step `nonpadding_tokens` / `supervised_tokens` /
`observed_microbatches`, and gate `scientific_throughput_complete` / `paper_performance_complete`
separately from resource completeness; proven on the real pinned stack (`INTEGRATION_PASS`, counts
40/39/42 per step, collate-wrap fired 0). Because the observer runs in the worker child it changes
worker bytes -> the **v7** lineage (amendment 0004, math/flash-v7). **That v7 lineage is now COMPLETE
and PASSED** (`V7_MATH_AND_FLASH_THROUGHPUT_PASS`, see the lead paragraph above). Still a 0.5B
feasibility bring-up, NOT a 7B or full-training claim. Details in
[`docs/HOST_STATE.md`](docs/HOST_STATE.md) (v6 + v7 sections). Earlier (2026-07-15,
pre-GPU finalization): the exact checkpoint/resume **execution engine**
#454 - torch worker that writes + restores sealed checkpoints, proven by a real-torch fresh-process
**bitwise-equivalence** test; amendment-0002 reconciliation #455 - classification
**V5_IDENTITIES_REMAIN_VALID**, the `df86db5` admission rule is an ANCESTOR rule so a descendant worker
source is admitted, only the runbook's byte-equality preconditions were stale; pre-live nvidia-smi
parser validation #456; corpus-studio skill refresh #457 - all merged. Earlier this day: post-#444
audit fixes #445-#447, amendment 0002 -> matrix 1.2.0 #448 + v5 runbook #449, Section 11 measurement
harness #450, #440 checkpoint/resume design #452, corpus-studio skill #451. Next gates (corpus freeze,
v5 0.5B GPU bring-up, 7B ladder, full runs) still need human GPU/data authorization. Earlier:
manager-1.3 v4 math failure + sealed-precision correction - see
[`docs/HOST_STATE.md`](docs/HOST_STATE.md); previous snapshot 2026-07-14). This is a snapshot for the
next agent session (Claude Code or Codex).
For the authoritative *feature* state see [`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md); for the
forward plan see [`docs/ROADMAP.md`](docs/ROADMAP.md) + [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

---

## 0. READ FIRST — where you are

- **CorpusStudio now runs on a native-Linux RTX 5070 host** (Ubuntu 24.04). Open your session at
  **`/mnt/training-nvme/repos/CorpusStudio`** on the Linux training NVMe. The verified host facts —
  paths, GPU, and the `HARDWARE_VERIFIED` managed environment — are in
  [`docs/HOST_STATE.md`](docs/HOST_STATE.md); read it first for anything hardware-adjacent.
- Engine virtualenv: **`/mnt/training-nvme/repos/CorpusStudio/engine/.venv`** (CPython 3.12.3,
  dependency-light core + `[dev]`, torch-free).
- The GPU **training** runtime is managed by the Environment Manager. The legacy
  **`backend-corpus-studio`**, readiness environments, and blue/green manager-1.2 research
  **`backend-corpus-studio-research-math-v2`** / **`backend-corpus-studio-research-flash-v2`** and
  the later preserved v3/v4 identities have separate evidence for their exact
  environment-level tuples. They supersede the old standalone `cs-train-venv`; none is a completed
  real-workload claim. See
  [`docs/HOST_STATE.md`](docs/HOST_STATE.md) and
  [`docs/ENVIRONMENT_MANAGER.md`](docs/ENVIRONMENT_MANAGER.md).
- **History (mounted, do not work from):** this repo previously lived on Windows `F:` (USB) then `C:`
  (migrated 2026-07-13). Those filesystems are mounted read-write — `C:\…` → `/mnt/windows-c`,
  `F:\…` → `/mnt/windows-f` (e.g. old `C:\CorpusStudio` → `/mnt/windows-c/CorpusStudio`) — but project
  policy is history-only: do not develop from or write to them. They are stale fallbacks that will
  drift; the active runtime is the `/mnt/training-nvme` checkout.
- WorldBibleGenerator (the "WBG" reference project) still lives on the Windows mounts:
  `/mnt/windows-c/WorldBibleGenerator` (source + datasets + trained adapters) and the large merged/GGUF
  models + pretrain data on `/mnt/windows-f/WorldBibleGenerator` (reproducible from adapters). WBG is a
  separate git repo with its own uncommitted training-path edits; it is not part of this repo's build.

## 1. What CorpusStudio is

A **local-first AI dataset→model→evaluation lifecycle platform**. Three pieces:
1. **Engine** (`engine/corpus_studio/`, Python) — a **dependency-light** core (no torch at import) for
   dataset authoring/validation/quality/gates/splits/eval, PLUS an opt-in `[train]` extra with a
   first-party TRL/PEFT QLoRA trainer.
2. **Platform** (`engine/corpus_studio/platform/`) — a **contract-first, torch-free** run lifecycle:
   profile → plan (hash-sealed RunPlan) → predict-fit → run (supervised, in-proc or kill-able
   subprocess) → measure-fit (watchdog) → artifacts. The historical run-lifecycle evidence is from
   native Windows/WDDM (plus separately labeled WSL probes); the native-Linux host now adds a
   `HARDWARE_VERIFIED` managed environment (env-manager GPU probe — see
   [`docs/HOST_STATE.md`](docs/HOST_STATE.md)), still not bare-Linux full-workload or real-offload proof.
3. **UI** — Tauri 2 + React (`apps/web`) is the UI head; the WPF/Avalonia desktop was **removed**
   (#545). The UI is a **client** over the engine CLI; it never owns training.

## 2. Current git / PR state

`main` is the source of truth. The current foundational history is:
- **#404** configurable checkpoint retention · **#405** StorageProfile + the dependency-architecture
  correction · **#406** Environment Manager substrate (Phase 2 slice 1) · **#407** storage USB/WSL
  runtime-role risks + storage-vs-not failure diagnostic · **#408** HANDOFF/AGENTS · **#409**
  CURRENT_STATE/CLI_REFERENCE reconciliation · **#410** the follow-up docs refresh · **#411** the
  managed `backend-corpus-studio` environment creation/lock/probe/drift/recreate lifecycle; **#412**
  adds the Phase 3 model/tokenizer descriptor and static inspection foundation; **#413** adds the
  Phase 4 TrainingObjective registry and compatibility checker; **#414** adds the Phase 5
  parameter-accounting foundation; **#415** adds the Phase 6 physical `RunPlan` foundation and was
  squash-merged as `f142b6b7517b0755fbc8ba04fdc1605366e30979`. Its local gate was 1,374 Python
  tests passed / 6 skipped / 88.32% Windows coverage, full Ruff/MyPy, deterministic 26-root
  schema/TypeScript generation, web
  production build, 815 desktop tests, and clean WPF/Avalonia Release builds. After any
  `platform/` contract change,
  regenerate the committed schemas (see §4), update the count in
  `tests/test_platform_contracts.py`, and regenerate the TypeScript types.
- **#416 merged the Phase 7 generalized `TraceRecord` foundation** as
  `f3cb18eeb2a52acf7ba7a34ae5c342da65065e2a`: one `TraceRecord` root,
  hash/policy/source/review evidence, legacy migration, explicit reviewed successors, external
  provider-authority and trainer gates, generated clients, and a desktop-selectable trace draft schema.
  See `docs/TRACE_RECORDS.md`.
- **#417 merged the Phase 8 static MoE model-inspection foundation** as
  `3262de48d106f39e8e28968c080c1461d9a68942`: bounded, hash-pinned parsing for exact
  Mixtral, Qwen2-MoE, DeepSeek V2, and DeepSeek V3 config mappings; component-scoped routed/shared
  expert groups; derived expert-instance counts; pre-Phase-8 descriptor migration; generated clients;
  and explicit static-only/non-runtime evidence. Its local gate was Ruff clean, MyPy clean (122 files),
  1,434 Python tests passed /
  6 skipped / 88.47% Windows coverage, deterministic 27-root schema/TypeScript generation, web
  production build, 815 desktop tests, and zero-warning WPF/Avalonia Release builds. See
  `docs/MOE_ARCHITECTURE.md`.
- **#418 merged the Phase 9A backend-worker protocol/isolation foundation** as `bca5246`:
  protocol 2.0 worker-first
  backend/environment identity, typed envelope/state-machine validation, managed recipe/lock/backend
  binding, pre-run seal checks at both public execution entry points, and shared bounded process-tree
  termination with a real descendant fake-worker test. It adds no new training backend or hardware
  capability. The final local gate is Ruff clean, MyPy clean (124 files), 1,450 Python tests passed /
  6 skipped / 88.38% Windows coverage, deterministic 27-root schema/TypeScript generation, web
  production build, 815 desktop tests, and zero-warning WPF/Avalonia Release builds. See
  `docs/BACKEND_WORKER_PROTOCOL.md`.
- **#419 merged the Phase 9B effective-execution-truth slice** as
  `6139b9e78ad383af95edbd2b70e34fd7808b2133`:
  a separately hash-sealed `ResolvedExecutionConfiguration`, immutable dataset/model/tokenizer refs,
  per-state precision, exact attention API/kernel/toggles, an explicit device map, complete semantic
  trainer defaults, and exact optimizer/loss/trainer-surface admission. Readiness now requires one
  passing complete execution tuple instead of unioning independent probes. Runner lanes are bound
  fail-closed (echo cannot consume training), the worker echoes the hash, datasets are consumed from
  stabilized bytes, local model/tokenizer roots are checked after load, and post-adapter
  precision/placement is observed. Runner identity now derives from the pinned backend, every plan and
  execution gets a collision-resistant UUIDv7 identity, outputs are run-scoped, and success requires
  optimizer-step plus recognized adapter-weight evidence at the exact derived path. The shipping
  desktop no longer exposes the unsealed direct trainer; its external-trainer path remains separate.
  The Tauri/React live planner requires an immutable model revision and displays the nested
  hash/kernel/device map. This is contract and CPU/fake-test evidence, not a new hardware-verification
  claim. The final local gate is Ruff clean, MyPy clean (125 files), 1,497 Python tests passed / 6
  skipped / 88.24% Windows coverage, deterministic 28-root
  schema/TypeScript generation, web typecheck + production build, Tauri `cargo fmt --check` +
  `cargo check`, 812 desktop tests, and successful WPF/Avalonia Release builds (with 22 existing
  Avalonia `Watermark` deprecation warnings). The PR and the four post-merge `main` workflows (Engine,
  web, Desktop, and CodeQL) all passed. See
  `docs/EFFECTIVE_EXECUTION_CONFIGURATION.md`.
- **#429 and #430 merged flash-readiness evidence sealing and semantic singleton placement** as
  `c53efe52` and `bbd14146`. The Linux-only flash environment passed its tiny forced
  `SDPBackend.FLASH_ATTENTION` BF16/NF4/QLoRA tuple. The first separately authorized real 0.5B,
  three-step smoke loaded the model but failed before adapter insertion because Transformers exposed
  no `hf_device_map`; it completed zero optimizer steps and wrote no adapter. A separate placement-only
  diagnostic then found all 290 parameters and both buffers resident on `cuda:0`. That diagnostic is
  evidence about that load only, not a completed `platform-run`, optimizer step, or sequence-4096
  result. Preserved evidence is indexed in [`docs/HOST_STATE.md`](docs/HOST_STATE.md).
- **#434 through #437 closed the checkpoint/liveness and managed-integrity prerequisites:** sealed
  production execution is checkpoint-free, manager operations use bounded inter-process leases and
  immutable blue/green environment IDs, PyTorch prerequisite artifacts are hash-bound, and managed
  planning retains verified package artifact/RECORD/installed-tree evidence. The matched research
  environments use the same worker wheel and package artifacts with separate math/flash locks.
- **#441 through #443 merged the success-evidence, manager-1.3, and host-build prerequisites:** a
  successful training run now requires canonical adapter updates, honest materialized-gradient
  coverage, exact per-step loss evidence, a real optimizer, trained-to-saved artifact identity, and
  complete output/integrity gates before proven fit. Manager-1.3 requires complete positive RECORD
  counts for every installed package, and the desktop gate is compatible with the pinned .NET 8 host.
- **The first matched manager-1.2 0.5B attempts are preserved failures, not retries or paper data.**
  Fresh normalized-equal plans ran once each (math then flash). Both verified the exact execution
  hash, intended kernel/toggles, model and post-adapter CUDA placement, QLoRA insertion, and trainer
  creation, then failed before optimizer step 1 because a pre-accumulation BF16 autograd tensor was
  treated as the sealed FP32 materialized gradient. No artifact or checkpoint was written; both
  environments remained `HARDWARE_VERIFIED`, drift false, and VRAM returned to 10 MiB. The repository
  verifier now uses post-accumulation leaf-gradient evidence, but that code has only unit evidence:
  build a new wheel, create new environment IDs/locks, generate new plans, and obtain separate smoke
  approval before claiming the correction on hardware. See [`docs/HOST_STATE.md`](docs/HOST_STATE.md).
- **A first-party success now requires evidence stronger than "the loop returned":** canonical hashes
  of the complete trainable adapter state must differ before/after; at least one materialized adapter
  gradient and honest observed/eligible coverage must be recorded; `on_train_begin` must expose the
  real optimizer; and every completed optimizer step must have exactly one finite loss under sealed
  `logging_strategy="steps"`, `logging_steps=1`, and `logging_nan_inf_filter=false`. Final trainable
  tensors must remain finite. A second canonical identity binds the exact PEFT export keys, shapes,
  dtypes, and bytes to `adapter_model.safetensors`; the complete in-memory PEFT config is likewise
  bound to `adapter_config.json`. Link-like output components, alternate/nested weight payloads,
  incomplete Safetensors ranges, and config or weight mutation fail closed. Weight and config hashes,
  durable artifact/run records, and raw measured-peak evidence must all pass before terminal success
  or a measured native fit is exposed. Gradient, loss, optimizer, update, artifact, environment, and
  numerical failures retain their own taxonomy/stage.
- **Manager 1.3 makes complete RECORD counts authoritative:** new evidence carries explicit
  `record_count_semantics="all_record_rows_v2"` and requires positive `record_entries`,
  `record_verified_entries == record_entries`, and the same positive installed-file count, with no
  failed row. Unhashed RECORD/pyc rows count only after their paths and exact bytes are bound by the
  installed-tree digest. Missing semantics retains the older hash-bearing-row meaning so manager-1.2
  locks and plans remain hash-verifiable for reconstruction, but health returns a non-mutating
  admission refusal and those identities cannot authorize planning or execution.
- **The manager-1.2 v3 pair is preserved, unexecuted evidence:** the wheel from repository commit
  `16ef6e95722ec3988ee8826b45333c9356ef76f9`, research-math-v3/research-flash-v3 environments, and
  normalized RunPlan candidates were all created before this audit checkpoint. Neither plan was
  dispatched and no model load or GPU workload occurred for them. All 84 packages in each lock used
  the older partial RECORD-count meaning of `verified`, so the environments and plans must not be
  mutated, relabeled, or reused. Exact identities are in [`docs/HOST_STATE.md`](docs/HOST_STATE.md).
- **The manager-1.3 v4 math attempt is a preserved failure; its flash plan was not dispatched.** The
  v4 pair used one wheel from `e7875629fc6e046dc2a84a53aa941b3d073c18bd`, complete RECORD evidence,
  fresh locks, and normalized-equal sequence-256 plans. Math ran once as
  `run-019f6518-3927-7d73-b106-15f385b61415`; it verified math attention, placement, QLoRA, and a real
  optimizer, then failed before step 1 with `GRADIENT_FAILURE` at `backward`. Pinned TRL had recast
  the sealed FP32 trainable parameters to BF16 during `SFTTrainer` construction, and the corrected
  post-accumulation hook accurately observed the BF16 materialized gradient. The run remained
  `NATIVE_UNPROVEN`, wrote no artifact/checkpoint/output, retained drift=false, and released VRAM. The
  worker now restores the sealed master dtype on the same identities after trainer construction and
  re-verifies placement/quantization/precision before training; this latest correction has unit
  evidence only. Preserve all v4 identities and evidence. A later attempt requires another wheel,
  new environment IDs/locks, fresh plans, and separate authorization.
- **The first-party checkpoint boundary is now fail-closed:** new plans seal
  `save_strategy="no"` with no cadence or retention, both sealed and explicitly unsealed trainer paths
  refuse legacy step-checkpoint execution before loading data or weights, and the runner rejects any
  unexpected checkpoint result. Final adapter output remains run-scoped. The control-plane design +
  verifier for exact resume ([#440](https://github.com/MalloyTheDev/CorpusStudio/issues/440), #452)
  now exists - a hash-sealed `CheckpointManifest` with an atomic complete marker + per-file byte
  integrity, `checkpoint.py` fail-closed verification (missing/malformed/incomplete/hash_mismatch/
  external_change), `verify_resumable_into` (incompatible unless every plan-derivable bound identity
  matches), `admit_resume` + `RunManifest.resume_lineage`, and `corpus-studio checkpoint-verify`. See
  [`docs/CHECKPOINT_RESUME.md`](docs/CHECKPOINT_RESUME.md). It does NOT enable automatic resume:
  intermediate checkpoints stay disabled and runs expected to exceed 30 minutes remain blocked until a
  separately reviewed trainer change consumes a `CheckpointResumeRequest`.
- **Managed-environment concurrency is fail-closed:** bounded manager/per-environment inter-process
  locks serialize create/recreate/remove, evidence-producing health and planning operations take a
  consistent environment lease, and `platform-run` holds that lease through worker termination.
  Sealed environment IDs cannot be recreated or silently reused after removal; use a new blue/green
  ID so the previous lock remains an unambiguous rollback and evidence identity. Same-ID
  `env-recreate` remains available only for an unsealed failed attempt.
- **Post-#444 audit + readiness program (2026-07-15), all merged:** #445 accepts torch's CPU-resident
  0-dim AdamW step counter and stops the attention-cleanup masking a real GRADIENT/OPTIMIZER failure;
  #446 fixes the Google/Gemini provenance under-classification (issue #422); #447 pins codecov-action
  to an immutable SHA (issue #426). #448 landed prospective research **amendment 0002 -> effective
  matrix 1.2.0** (v5 blue/green identities, worker source bound to `df86db5`, `RESERVED_IDENTITIES.v2`
  reserving all v4); #449 the [`v5 bring-up runbook`](research/ieee-linux-training/RUNBOOK_v5_bringup.md)
  (Sections 5-8). #450 added the **Section 11 measurement harness** (`platform/telemetry.py`: raw
  `TelemetrySample` + derived `RunTelemetrySummary`, wired in-path, `platform-run --telemetry` +
  `telemetry-summarize`; [`docs/MEASUREMENT_HARNESS.md`](docs/MEASUREMENT_HARNESS.md)). #452 added the
  #440 checkpoint/resume design (above). #451 committed the project-scoped `corpus-studio` Claude skill
  (`.claude/skills/corpus-studio/`). **Next gates (all require separate human authorization): freeze
  the ~500-output corpus; then GPU v5 bring-up (build wheel x2 -> create v5 envs -> matched 0.5B
  smokes) per the runbook; then the 7B sequence ladder; then full 500-output runs.** Unload Ollama
  before any GPU op; one GPU op at a time; never conflate WSL and native Linux.

## 3. The architecture North Star + binding directives

**The bigger picture (the end state).** CorpusStudio is becoming a complete **local-first,
hardware-aware AI engineering platform** covering the whole lifecycle: raw sources → dataset
construction (multiple training objectives, mixtures, reasoning/tool **traces**) → model + tokenizer
management (Model & Tokenizer) → hardware- and storage-aware **run planning** → training through
**swappable, isolated backends** → live-telemetry supervision → checkpoint + **artifact lineage** →
evaluation → deployment prep → reproducible experimentation. The **control plane stays lightweight and
torch-free**; heavy frameworks live in **isolated worker environments** behind the versioned
`WorkerMessage` protocol; the **UI is always a client**, never the owner of training behavior. A
concrete research North Star driving the contract design: **resource-elastic MoE** — training a
~30B-logical / 2–4B-active / 50–200M-resident model on consumer hardware
(`N_resident << N_active << N_logical`), which is exactly why the foundational contracts must be
MoE-safe *now*. The WBG-7B / RTX-5070 work is the **reference stress-test** that surfaces the generic
requirements — not the product scope. The eventual shell replaces the desktop prototype directly
with Tauri 2 + React (#545), over the stable language-neutral contracts, with a Rust authoritative core (#522).

The big epic (memory `platform-architecture-epic`). Non-negotiables the user has set:

- **3-layer dependency model** ([`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) §2):
  "dependency-light" describes the **CONTROL PLANE only**, not the whole product. Layers = control
  plane (always installable, no CUDA) / capability profiles (opt-in, into the core) / **isolated
  per-backend worker environments** (heavy frameworks pin conflicting builds — never one `[everything]`
  env). The `[train]` extra is the reference `backend-corpus-studio` env.
- **★ MoE-safe foundational contracts** ([`docs/MOE_ARCHITECTURE.md`](docs/MOE_ARCHITECTURE.md)): **no
  new foundational contract may assume dense execution.** Build `ModelDescriptor` / `TrainingObjective`
  / `RunPlan`-expansion / checkpoint / telemetry with the multi-count parameter accounting
  (`N_logical`/`N_active`/`N_resident`/`N_touched`/`N_updated`/`N_exposed`), the
  semantic-router-vs-physical-scheduler split, stable expert identity, and sharded transactional
  checkpoints **from day one** — retrofitting sparse later is a disruptive redesign.
- **Revised foundational order** (do these in order): 1 StorageProfile ✅ → 2 Env Manager reference
  lifecycle ✅ → 3 `ModelDescriptor` + `TokenizerDescriptor` foundation ✅ → 4 `TrainingObjective`
  registry ✅ → 5 parameter accounting ✅ → 6 `RunPlan` expansion ✅ → 7 `TraceRecord` ✅ → 8 static
  MoE inspection ✅ (PR #417) → 9A worker protocol/isolation ✅ → 9B effective execution truth ✅ → 9C dense
  backends → 10 existing-model MoE FT → 11 full MoE → 12 resource-elastic expert runtime.

## 4. How to build + verify (the gate)

From `/mnt/training-nvme/repos/CorpusStudio/engine` (bash):
```
.venv/bin/python -m ruff check corpus_studio tests
.venv/bin/python -m mypy corpus_studio
.venv/bin/python -m pytest -q --no-header --basetemp=.pytest_tmp
```
CI (GitHub Actions, **Linux + Python 3.11**) runs `pytest --cov=corpus_studio --cov-report=term-missing`
with **`--cov-fail-under=88`**, plus the web build (`web.yml`, incl. schema-to-TypeScript
regeneration drift checks) and Python CodeQL.

- **★ COVERAGE:** this host is native Linux, so `storage_profiler`'s Linux-only detection now executes
  locally — the old ~0.3% Windows-run under-measurement no longer applies and local coverage tracks CI.
  The CI floor is `--cov-fail-under=88`; Windows-only detection funcs stay marked `# pragma: no cover`.
- **After changing any `platform/` contract**, regenerate the committed JSON Schemas and update the
  count test:
  ```
  .venv/bin/python -c "from corpus_studio.platform.schema_export import export_json_schemas; export_json_schemas('../docs/contracts')"
  ```
  then fix the two counts in `tests/test_platform_contracts.py` (`len(ROOT_CONTRACTS)` + the export
  test). This branch exports **28 root contracts**.

## 5. Workflow / process conventions (the user cares about these)

- **Branch first**: `git checkout -b feat/<slice>` before editing. One coherent slice per PR.
- **One CI-green PR per slice.** Do not assume merging is forbidden: the user explicitly authorized
  completing and merging continuation PRs on 2026-07-13. Still require green checks and verify the
  exact live PR before merging.
- **Verify in the main loop — do NOT spawn multi-agent Workflow fan-outs** (they burn the user's usage
  limit) unless the user *explicitly* asks for N agents. (Memory: `no-audit-workflows`.)
- **ASCII only in CLI-facing strings** (Windows console UTF-8 — no em-dashes `—`, use `-`). This repo
  has hit Windows-console UTF-8 crashes before.
- Momentum: pick the highest-priority slice and **build it** (branch → CI-green PR → report). Only
  pause for destructive/irreversible actions. (Memory: `execute-dont-ask`.)
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` (Claude sessions).

## 6. Invariants — DO NOT weaken

- License **fail-closed**; provenance gate; **"a completed step ≠ proven fit"**; **no silent target
  truncation**; **"installed ≠ supported"**; **no-shell argv** execution (installers/trainer launches
  are `argv` lists, never shell strings); single-writer `examples.jsonl` (the engine's `examples-append` /
  `storage/examples_writer.py` is the sole sanctioned writer); provider policy enforced **in the engine**, not just the UI.
- **Predicted fit is never `NATIVE_SAFE`** — only a *measured* run earns it (the calibrator/watchdog).
- **Blackwell / sm_120**: the **math** attention path is the verified-safe default — the fused flash-SDPA
  kernel **deadlocks on the first backward under native-Windows WDDM** (high GPU util, low power). WSL
  flash was measured separately. On the native-Linux host, readiness-v2 verified the environment-level
  **math** tuple and the research flash environment separately verified the tiny forced **flash**
  tuple. The manager-1.2 matched real 0.5B attempts stopped before step 1 on their earlier verifier
  mismatch. The later manager-1.3 v4 math attempt also stopped before step 1 when the corrected
  materialized-gradient verifier exposed TRL's constructor-time BF16 adapter recast; its paired flash
  plan was not dispatched. Bare-Linux flash for a real optimizer step and sequence 4096 are therefore
  still not claimed. Unsloth is
  honestly **refused on native-Windows/WDDM Blackwell** (it declares no math path) and routed to
  `backend-corpus-studio`. Other hosts still need their own exact functional evidence.
- **Dependency-light boundary**: `import corpus_studio.platform` pulls **no torch**; all heavy imports
  are lazy inside `run()`/functions. There is a test that asserts this.
- **Storage guardrail (new, #405/#407)**: USB/`/mnt`-WSL/cloud-sync/in-repo/near-full paths are
  flagged per role — the reason this migration off F: happened.

## 7. Immediate next actions (ranked)

1. **Finish and merge the focused post-trainer precision correction before any new worker wheel.** Do
   not reuse the preserved manager-1.2 v2/v3 or manager-1.3 v4 environments, plans, failed runs, or
   artifacts. After the merge, a new wheel and new blue/green environment IDs/locks are required
   because worker behavior changed. **A follow-up audit correction (2026-07-15) changed the worker
   again:** the sealed evidence verifier `verify_optimizer_state_precision` rejected torch's own
   CPU-resident 0-dim AdamW `step` counter (a placement false-failure that would have blocked
   optimizer step 1 of every real `adamw_torch` run - the next blocker after the #444 BF16 recast
   fix), and the enforced attention-kernel cleanup could rewrite a real GRADIENT/OPTIMIZER failure as
   an environment error. Both are corrected with CPU/unit evidence only. Because worker execution
   bytes changed once more, the next wheel and next environment IDs/locks must be built from this
   corrected commit; the environment pair is a fresh **v5** identity (see the manager-1.3 v4 section of
   `docs/HOST_STATE.md`). Research amendment **0002 -> effective matrix 1.2.0 is now merged** (#448,
   effective sha256 `168189145...b9c`): it allocates the v5 identities, binds the audited worker source
   `df86db5`, and reserves every now-historical v4 identity (`RESERVED_IDENTITIES.v2.json`). The
   amendment is prospective; the wheel/env/smoke steps are gated only on a separate GPU authorization.
   The exact ordered procedure is [`research/ieee-linux-training/RUNBOOK_v5_bringup.md`](research/ieee-linux-training/RUNBOOK_v5_bringup.md).
2. **Continue hardware-independent work alongside the measured workload bring-up:** bounded event
   journaling, prepared-dataset/transactional row-store groundwork, TraceRecord identity/governance,
   and remaining desktop/web contracts. Keep non-trivial placement/offload execution unimplemented
   until an isolated backend proves it.
3. **Then continue down the revised order (§3):** dense backends → existing-model MoE FT
   → full MoE → resource-elastic expert runtime.
4. **GPU workload bring-up remains approval-gated:** only after the corrected wheel, new environments,
   fresh normalized-equal plans, and a field-by-field plan diff may one corrected 0.5B math smoke and
   one corrected 0.5B flash smoke be separately approved. The 500-output corpus and 7B workload are not
   ready and must not be loaded or trained. NVMe/offload work remains later and separately evidenced.

## 8. Hardware + environments

- **GPU**: RTX 5070, 12 GB (12227 MiB), Blackwell (sm_120, cc 12.0), driver 595.71.05, now on **native
  Linux** (Ubuntu 24.04). The **historical** WDDM-measured VRAM ceiling for 7B 4-bit QLoRA was **~10.8 GB
  @ seq 1024, ~13.8 GB @ 2048** (above ~1280 the WDDM path spilled to system RAM and crawled);
  native-Linux *workload* VRAM behavior and every real offload fit are not yet measured.
- **Hard verification boundary (host now assembled):** the native-Linux host and its `HARDWARE_VERIFIED`
  `backend-corpus-studio` env are real GPU evidence for the env-manager probe, but that is not a workload
  result - still do not claim full-sequence 7B success, DeepSpeed NVMe offload, Linux FSDP, a
  bare-Linux flash optimizer step, PCIe 4.0 NVMe throughput, sustained NVMe writes, real offload
  fit, or MoE runtime capability. Contract representation, fake workers, CI, and a passing env hardware
  probe are not workload proof. See [`docs/HOST_STATE.md`](docs/HOST_STATE.md).
- **Engine venv (control plane)**: `/mnt/training-nvme/repos/CorpusStudio/engine/.venv` (CPython 3.12.3,
  torch-free). Recreate with `python3.12 -m venv .venv && .venv/bin/python -m pip install -e .[dev]`.
- **GPU training runtime**: the managed **`backend-corpus-studio`** env (torch 2.11.0+cu128 + `[train]`),
  built and `HARDWARE_VERIFIED` - it replaces the old standalone `cs-train-venv`. Recreate/probe it via
  the Environment Manager (`env-plan` / `env-create` / `env-probe`), not by hand.
- **Optional extras**: `[train]` (torch/transformers/peft/trl/bitsandbytes), `[parquet]` (pyarrow),
  `[tokenizer]` (tiktoken), `[model-tokenizer]` (tokenizers).

## 9. Repository map

- **Platform contracts**: `engine/corpus_studio/platform/contracts.py` (28 root contracts) + `enums.py`
  + `common.py`; `schema_export.py` → `docs/contracts/*.schema.json` (language-neutral, consumed by
  `apps/web`).
- **Lifecycle**: `platform/{profiler, probes, planner, calibrator, supervisor, runners, watchdog,
  subprocess_supervisor, worker, backends}.py`; Phase 9B hashing/immutable-input helpers live in
  `platform/execution_config.py`.
- **Environment Manager** (Phase 2): `platform/environments.py` (recipes + concrete resolver) and
  `platform/environment_manager.py` (runtime discovery, creation, journals, lock/probe/drift, owned
  removal/recreation, and RunPlan compatibility). Readiness recipes bind an exact worker wheel and a
  complete QLoRA probe tuple in the plan (math `cuda_qlora_math_execution` or forced-flash
  `cuda_qlora_sdpa_flash_execution`), capture sanitized pip/RECORD plus complete record-owned
  installed-file-tree evidence, compare the installed worker payload with the reviewed wheel,
  and seal the lock only after required probes plus stable pre/post inventories. Manager 1.3 keeps the
  sealed 1.1 math files and hashes readable as historical rollback evidence while refusing a new
  health/planning claim without complete-count replacement. Math readiness-v2 is the preserved
  historical baseline; the manager-1.1 flash readiness-v1 lock remains preserved historical evidence for
  its tiny forced-flash tuple, but manager 1.2 does not grandfather its missing adapter-state equality
  observation. It requires replacement before a new health claim. Blue/green research-math-v2 and
  research-flash-v2 preserve the manager-1.2 identities; their matched 0.5B attempts reached adapter
  insertion but failed before step 1 on the earlier verifier mismatch. Manager-1.2 v3 is preserved
  but inadmissible under complete RECORD semantics. Manager-1.3 v4 has complete package evidence, but
  its one math attempt exposed the post-constructor precision issue and its flash plan was withheld;
  both v4 identities now remain historical because the worker correction changes their execution
  bytes. The legacy
  `backend-corpus-studio` environment also remains available.
- **Model/Tokenizer foundation** (Phase 3): `platform/model_inspector.py` + the `ModelDescriptor` /
  `TokenizerDescriptor` roots. `model-inspect` inventories local snapshots without network access,
  heavy imports, link traversal, or custom-code execution; see `docs/MODEL_TOKENIZER_CONTRACTS.md`.
- **TrainingObjective foundation** (Phase 4): `platform/objectives.py` + the `TrainingObjective` /
  `ObjectiveCompatibilityReport` roots. The sealed 29-entry registry is backend-independent;
  `training-objectives` and `training-objective-check` expose definitions and conservative evidence
  axes. It does not add RunPlan wiring or trainer implementations; see `docs/TRAINING_SYSTEMS_ARCHITECTURE.md`.
- **Parameter-accounting foundation** (Phase 5): `platform/parameter_accounting.py` + the
  `ParameterAccountingReport` root. `model-inspect --parameter-accounting` and `parameter-account`
  produce/reconcile sealed static and typed runtime evidence; stored elements/allocator bytes are not
  promoted into stronger coordinate claims. Existing workers still need measured coordinate
  instrumentation; see `docs/PARAMETER_ACCOUNTING.md`.
- **Physical RunPlan foundation** (Phase 6): `platform/{contracts,planner,backends,probes}.py` make
  resources, state placement, offload rules, ranks/groups, storage/report pins, and capability proof
  explicit. `platform-run` and the worker reject hash tampering; current built-in runners support only
  the singleton CPU/GPU path and refuse non-trivial execution; see
  `docs/RUN_PLAN_PHYSICAL_EXECUTION.md`.
- **TraceRecord foundation** (Phase 7): `platform/trace_records.py` + `training/{traces,
  trace_generation,trainer}.py`; the contract preserves source/context/segments/producer/validation/
  review, while `trace-migrate`, `trace-generate`, `trace-review`, and `trace-validate` implement the
  pending → reviewed workflow. Stored provider-policy snapshots are evidence only: review/training
  re-resolve the external project override and block unknown/frontier/drifted authority. See
  `docs/TRACE_RECORDS.md`.
- **Static MoE inspection** (Phase 8): `platform/moe_inspector.py` + the `ModelTopology` evidence
  surface in `contracts.py`. `model-inspect` recognizes only exact allowlisted family mappings, pins
  the verified config digest/paths, reports expert-instance structure, and leaves all runtime,
  backend, placement, residency, fit, and hardware claims unverified. See
  `docs/MOE_ARCHITECTURE.md`.
- **Worker protocol/isolation** (Phase 9A): `platform/worker_protocol.py`,
  `subprocess_supervisor.py`, `process_control.py`, `worker.py`, and `backends.py`; exact backend and
  environment identity, strict message ordering/lineage, pre-run seal checks, and process-tree cleanup.
  See `docs/BACKEND_WORKER_PROTOCOL.md`.
- **Storage**: `platform/storage_profiler.py` (topology + per-role safe-spill guardrail + failure
  diagnostic).
- **Trainer**: `training/trainer.py` (+ `unsloth_trainer.py`). CLI: `engine/corpus_studio/cli.py`
  (including `platform-*`, the full `env-*` lifecycle, and `train-*`).
- **Docs**: source of truth `docs/CURRENT_STATE.md`; plan `docs/IMPLEMENTATION_PLAN.md`; MoE
  `docs/MOE_ARCHITECTURE.md`; storage `docs/HARDWARE_STORAGE_PROFILE.md`; env
  `docs/ENVIRONMENT_MANAGER.md`; objectives `docs/TRAINING_SYSTEMS_ARCHITECTURE.md`; parameter evidence
  `docs/PARAMETER_ACCOUNTING.md`; platform run `docs/PLATFORM_RUN.md`; worker boundary
  `docs/BACKEND_WORKER_PROTOCOL.md`.
- **Claude memory** (persists across Claude sessions): on this Linux host,
  `~/.claude/projects/-mnt-training-nvme-repos-CorpusStudio/memory/` — start at `MEMORY.md` (index);
  `platform-architecture-epic.md` is the big one. (Pre-migration Windows sessions used
  `C:\Users\Brend\.claude\projects\F--CorpusStudio\memory\`.)

## 10. Notes for Codex specifically

- Languages: **Python** (engine — the primary surface) and **TypeScript/React** (`apps/web`, Tauri 2).
  (The C#/.NET WPF/Avalonia desktop was removed - #545.)
- Use the venv at `engine/.venv`; run the gate in §4. Respect the invariants in §6 (especially
  no-shell argv, fail-closed policy, and the Blackwell math-attention rule).
- The engine core must stay import-torch-free; put heavy deps behind lazy imports + the `[train]` extra.
- Language-neutral contracts are the boundary — change the pydantic models in `platform/contracts.py`,
  then regenerate `docs/contracts/*.schema.json` (§4); the TS types in `apps/web/src/contracts/` are
  generated from those schemas.
