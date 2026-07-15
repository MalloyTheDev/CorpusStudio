# CorpusStudio — Session Handoff

**Last updated:** 2026-07-14 (native-Linux flash readiness and first bounded smoke - see
[`docs/HOST_STATE.md`](docs/HOST_STATE.md); previous snapshot 2026-07-13). This is a snapshot for the
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
  **`backend-corpus-studio`**, math rollback **`backend-corpus-studio-readiness-v2`**, and forced-flash
  **`backend-corpus-studio-readiness-flash-v1`** environments have separate preserved
  `HARDWARE_VERIFIED` evidence for their exact environment-level tuples. Manager 1.2 preserves the
  math rollback but requires replacement of the manager-1.1 flash instance before a new health claim.
  They supersede the old standalone `cs-train-venv`; none is a real-workload success claim. See
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
3. **UI heads** — WPF (shipping), Avalonia (cross-platform interim), Tauri 2 + React (`apps/web`,
   contract-first future head). The UI is a **client** over the engine CLI; it never owns training.

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
  `docs/MOE_MODEL_INSPECTION.md`.
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
- **The first-party checkpoint boundary is now fail-closed:** new plans seal
  `save_strategy="no"` with no cadence or retention, both sealed and explicitly unsealed trainer paths
  refuse legacy step-checkpoint execution before loading data or weights, and the runner rejects any
  unexpected checkpoint result. Final adapter output remains run-scoped. Exact sealed resume is still
  unimplemented, so short benchmark trials are checkpoint-free and runs expected to exceed 30 minutes
  remain blocked pending a separate resume-lineage design.
- **Managed-environment concurrency is fail-closed:** bounded manager/per-environment inter-process
  locks serialize create/recreate/remove, evidence-producing health and planning operations take a
  consistent environment lease, and `platform-run` holds that lease through worker termination.
  Sealed environment IDs cannot be recreated or silently reused after removal; use a new blue/green
  ID so the previous lock remains an unambiguous rollback and evidence identity. Same-ID
  `env-recreate` remains available only for an unsealed failed attempt.

## 3. The architecture North Star + binding directives

**The bigger picture (the end state).** CorpusStudio is becoming a complete **local-first,
hardware-aware AI engineering platform** covering the whole lifecycle: raw sources → dataset
construction (multiple training objectives, mixtures, reasoning/tool **traces**) → model + tokenizer
management (a Model/Tokenizer Lab) → hardware- and storage-aware **run planning** → training through
**swappable, isolated backends** → live-telemetry supervision → checkpoint + **artifact lineage** →
evaluation → deployment prep → reproducible experimentation. The **control plane stays lightweight and
torch-free**; heavy frameworks live in **isolated worker environments** behind the versioned
`WorkerMessage` protocol; the **UI is always a client**, never the owner of training behavior. A
concrete research North Star driving the contract design: **resource-elastic MoE** — training a
~30B-logical / 2–4B-active / 50–200M-resident model on consumer hardware
(`N_resident << N_active << N_logical`), which is exactly why the foundational contracts must be
MoE-safe *now*. The WBG-7B / RTX-5070 work is the **reference stress-test** that surfaces the generic
requirements — not the product scope. The eventual shell strangles WPF → Avalonia (interim) → Tauri 2 +
React, over the stable language-neutral contracts, with a progressively Rust-ified core.

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
with **`--cov-fail-under=88`**, plus `avalonia-linux-build`, `desktop-tests`, web typecheck/build,
schema-to-TypeScript regeneration drift checks, and C# + Python CodeQL.

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
  are `argv` lists, never shell strings); single-writer `examples.jsonl` (desktop is the only writer;
  the engine refuses to write it); provider policy enforced **in the engine**, not just the UI.
- **Predicted fit is never `NATIVE_SAFE`** — only a *measured* run earns it (the calibrator/watchdog).
- **Blackwell / sm_120**: the **math** attention path is the verified-safe default — the fused flash-SDPA
  kernel **deadlocks on the first backward under native-Windows WDDM** (high GPU util, low power). WSL
  flash was measured separately. On the native-Linux host, readiness-v2 verified the environment-level
  **math** tuple and readiness-flash-v1 separately verified the tiny forced **flash** tuple. The first
  real 0.5B smoke stopped at placement verification before adapter insertion, so bare-Linux flash for
  a real optimizer step or the sequence-4096 workload is still not claimed. Unsloth is
  honestly **refused on native-Windows/WDDM Blackwell** (it declares no math path) and routed to
  `backend-corpus-studio`. Other hosts still need their own exact functional evidence.
- **Dependency-light boundary**: `import corpus_studio.platform` pulls **no torch**; all heavy imports
  are lazy inside `run()`/functions. There is a test that asserts this.
- **Storage guardrail (new, #405/#407)**: USB/`/mnt`-WSL/cloud-sync/in-repo/near-full paths are
  flagged per role — the reason this migration off F: happened.

## 7. Immediate next actions (ranked)

1. **Remaining environment concurrency/transaction hardening:** installed environment inventories now
   verify every contained RECORD entry, reject unrecorded site-package files and symlinks, then import
   torch only in a second process after the parent validates that evidence. Readiness workers also
   compare installed payload files with the reviewed wheel's RECORD/METADATA/pip identity. The
   remaining slice is to share
   safe inventory primitives across artifacts/checkpoints, add manager/per-environment inter-process
   locks, and make recreation blue/green. Build and hash an exact worker wheel only from an approved
   audited commit.
2. **Continue hardware-independent work alongside the measured workload bring-up:** bounded event
   journaling, prepared-dataset/transactional row-store groundwork, TraceRecord identity/governance,
   and remaining desktop/web contracts. Keep non-trivial placement/offload execution unimplemented
   until an isolated backend proves it.
3. **Then continue down the revised order (§3):** dense backends → existing-model MoE FT
   → full MoE → resource-elastic expert runtime.
4. **GPU workload bring-up (host now assembled; env `HARDWARE_VERIFIED`):** the Linux NVMe is
   installed, the NVIDIA/CUDA stack lives in the managed env, and `backend-corpus-studio` is built and
   probed ([`docs/HOST_STATE.md`](docs/HOST_STATE.md)). The env hardware probe is a *minimal* GPU check,
   so the remaining hardware work is still open: run a real GPU smoke, benchmark the NVMe
   non-destructively, then step the 7B workload from sequence 1024 upward before CPU and finally NVMe
   offload - none of which the env probe proves.

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
  and seal the lock only after required probes plus stable pre/post inventories. Manager 1.2 preserves
  the sealed 1.1 math rollback identity while requiring stronger evidence for new creations. Math
  readiness-v2 is the preserved
  baseline/rollback; the manager-1.1 flash readiness-v1 lock remains preserved historical evidence for
  its tiny forced-flash tuple, but manager 1.2 does not grandfather its missing adapter-state equality
  observation. It requires replacement before a new health claim. Its first real 0.5B smoke did not
  reach adapter insertion or an optimizer step. The legacy
  `backend-corpus-studio` environment also remains available.
- **Model/Tokenizer foundation** (Phase 3): `platform/model_inspector.py` + the `ModelDescriptor` /
  `TokenizerDescriptor` roots. `model-inspect` inventories local snapshots without network access,
  heavy imports, link traversal, or custom-code execution; see `docs/MODEL_TOKENIZER_CONTRACTS.md`.
- **TrainingObjective foundation** (Phase 4): `platform/objectives.py` + the `TrainingObjective` /
  `ObjectiveCompatibilityReport` roots. The sealed 29-entry registry is backend-independent;
  `training-objectives` and `training-objective-check` expose definitions and conservative evidence
  axes. It does not add RunPlan wiring or trainer implementations; see `docs/TRAINING_OBJECTIVES.md`.
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
  `docs/MOE_MODEL_INSPECTION.md`.
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
  `docs/ENVIRONMENT_MANAGER.md`; objectives `docs/TRAINING_OBJECTIVES.md`; parameter evidence
  `docs/PARAMETER_ACCOUNTING.md`; platform run `docs/PLATFORM_RUN.md`; worker boundary
  `docs/BACKEND_WORKER_PROTOCOL.md`.
- **Claude memory** (persists across Claude sessions): on this Linux host,
  `~/.claude/projects/-mnt-training-nvme-repos-CorpusStudio/memory/` — start at `MEMORY.md` (index);
  `platform-architecture-epic.md` is the big one. (Pre-migration Windows sessions used
  `C:\Users\Brend\.claude\projects\F--CorpusStudio\memory\`.)

## 10. Notes for Codex specifically

- Languages: **Python** (engine — the primary surface), **C#/.NET** (`apps/desktop` WPF +
  `apps/desktop/CorpusStudio.Avalonia`), **TypeScript/React** (`apps/web`, Tauri 2).
- Use the venv at `engine/.venv`; run the gate in §4. Respect the invariants in §6 (especially
  no-shell argv, fail-closed policy, and the Blackwell math-attention rule).
- The engine core must stay import-torch-free; put heavy deps behind lazy imports + the `[train]` extra.
- Language-neutral contracts are the boundary — change the pydantic models in `platform/contracts.py`,
  then regenerate `docs/contracts/*.schema.json` (§4); the TS types in `apps/web/src/contracts/` are
  generated from those schemas.
