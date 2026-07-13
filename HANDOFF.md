# CorpusStudio — Session Handoff

**Last updated:** 2026-07-13. This is a snapshot for the next agent session (Claude Code or Codex).
For the authoritative *feature* state see [`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md); for the
forward plan see [`docs/ROADMAP.md`](docs/ROADMAP.md) + [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

---

## 0. READ FIRST — where you are

- **CorpusStudio now lives at `C:\CorpusStudio`.** It was migrated off the **F: USB external drive**
  onto internal C: on 2026-07-13 (F: is USB-connected — poor for an active runtime). **Open your
  session at `C:\CorpusStudio`.**
- `F:\CorpusStudio` is the **old copy — a fallback, do not use it** for new work (it will drift).
- Engine virtualenv: **`C:\CorpusStudio\engine\.venv`** (Python 3.12.10, dependency-light core + `[dev]`).
- The GPU **training** venv was NOT recreated on C: (was `F:\cs-train-venv`, torch cu128). Rebuild it
  when you next actually train on the GPU.
- WorldBibleGenerator (the "WBG" reference project): **`C:\WorldBibleGenerator`** holds the *important*
  stuff (source + datasets + trained adapters, ~2.4 GB). The large merged/GGUF models + pretrain data
  (~260 GB) were intentionally left on `F:\WorldBibleGenerator` (reproducible from adapters).
- **WBG WSL training now points to C:** — the launch scripts / configs / README under
  `C:\WorldBibleGenerator\training` were repointed off `/mnt/f` → `/mnt/c` (the venv stays on the Linux
  FS `~/wbg-venv`, which is correct). The base model loads from the HF cache, so the data→adapter path
  is self-contained on C:. (Those edits are uncommitted in the WBG git repo.)

## 1. What CorpusStudio is

A **local-first AI dataset→model→evaluation lifecycle platform**. Three pieces:
1. **Engine** (`engine/corpus_studio/`, Python) — a **dependency-light** core (no torch at import) for
   dataset authoring/validation/quality/gates/splits/eval, PLUS an opt-in `[train]` extra with a
   first-party TRL/PEFT QLoRA trainer.
2. **Platform** (`engine/corpus_studio/platform/`) — a **contract-first, torch-free** run lifecycle:
   profile → plan (hash-sealed RunPlan) → predict-fit → run (supervised, in-proc or kill-able
   subprocess) → measure-fit (watchdog) → artifacts. Verified end-to-end on a real RTX 5070.
3. **UI heads** — WPF (shipping), Avalonia (cross-platform interim), Tauri 2 + React (`apps/web`,
   contract-first future head). The UI is a **client** over the engine CLI; it never owns training.

## 2. Current git / PR state

`main` is the source of truth; everything through **#414 was merged when the Phase 6 branch began**:
- **#404** configurable checkpoint retention · **#405** StorageProfile + the dependency-architecture
  correction · **#406** Environment Manager substrate (Phase 2 slice 1) · **#407** storage USB/WSL
  runtime-role risks + storage-vs-not failure diagnostic · **#408** HANDOFF/AGENTS · **#409**
  CURRENT_STATE/CLI_REFERENCE reconciliation · **#410** the follow-up docs refresh · **#411** the
  managed `backend-corpus-studio` environment creation/lock/probe/drift/recreate lifecycle; **#412**
  adds the Phase 3 model/tokenizer descriptor and static inspection foundation; **#413** adds the
  Phase 4 TrainingObjective registry and compatibility checker; **#414** adds the Phase 5
  parameter-accounting foundation. GitHub was checked live at Phase 6 publication: no competing PRs;
  only unrelated UI-theme issues #187 and #201 were open. The `feat/run-plan-placement` branch
  delivers the Phase 6 physical `RunPlan` foundation; verify its live PR/merge state rather than
  inferring it from this snapshot. Its local gate was 1,374 Python tests passed / 6 skipped / 88.32%
  Windows coverage, full Ruff/MyPy, deterministic 26-root schema/TypeScript generation, web
  production build, 815 desktop tests, and clean WPF/Avalonia Release builds. After any
  `platform/` contract change,
  regenerate the committed schemas (see §4), update the count in
  `tests/test_platform_contracts.py`, and regenerate the TypeScript types.

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
  registry ✅ → 5 parameter accounting ✅ → 6 `RunPlan` expansion ✅ → 7 `TraceRecord` → 8 MoE
  inspection → 9 dense backends → 10 existing-model MoE FT → 11 full MoE → 12 resource-elastic
  expert runtime.

## 4. How to build + verify (the gate)

From `C:\CorpusStudio\engine` (PowerShell):
```
.\.venv\Scripts\python.exe -m ruff check corpus_studio tests
.\.venv\Scripts\python.exe -m mypy corpus_studio
.\.venv\Scripts\python.exe -m pytest -q --no-header --basetemp=.pytest_tmp
```
CI (GitHub Actions, **Linux + Python 3.11**) runs `pytest --cov=corpus_studio --cov-report=term-missing`
with **`--cov-fail-under=88`**, plus `avalonia-linux-build`, `desktop-tests`, web typecheck/build,
schema-to-TypeScript regeneration drift checks, and C# + Python CodeQL.

- **★ COVERAGE GOTCHA:** running the suite on **Windows under-measures** `storage_profiler`'s
  Linux-only detection by ~0.3% (those lines only execute on the Linux CI runner). **Target ≥ 88.2%
  locally** to have CI margin. Windows-only detection funcs are marked `# pragma: no cover`.
- **After changing any `platform/` contract**, regenerate the committed JSON Schemas and update the
  count test:
  ```
  .\.venv\Scripts\python.exe -c "from corpus_studio.platform.schema_export import export_json_schemas; export_json_schemas('../docs/contracts')"
  ```
  then fix the two counts in `tests/test_platform_contracts.py` (`len(ROOT_CONTRACTS)` + the export
  test). This branch exports **26 root contracts**.

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
- **Blackwell / sm_120**: force the **math** attention path — the fused flash-SDPA kernel **deadlocks
  on the first backward under native-Windows WDDM** (high GPU util, low power). WSL/Linux flash is safe.
  Unsloth is honestly **refused on Blackwell** (it declares no math path) and routed to
  `backend-corpus-studio`.
- **Dependency-light boundary**: `import corpus_studio.platform` pulls **no torch**; all heavy imports
  are lazy inside `run()`/functions. There is a test that asserts this.
- **Storage guardrail (new, #405/#407)**: USB/`/mnt`-WSL/cloud-sync/in-repo/near-full paths are
  flagged per role — the reason this migration off F: happened.

## 7. Immediate next actions (ranked)

1. **Phase 7 — generalized `TraceRecord` + Trace Studio foundation.** Reconcile the existing
   lightweight `training/traces.py` shape with a versioned language-neutral contract; preserve
   provenance, tool/reasoning boundaries, answer-leak protections, and dataset lineage without
   treating generated reasoning as ground truth.
2. **Environment Manager hardware verification (explicit/networked).** After reviewing `env-plan`,
   build and probe a real `backend-corpus-studio` CUDA environment on the RTX 5070. The code and CI
   harness do not claim this newly downloaded environment exists or is hardware-verified.
3. **Phase 8+** down the revised order (§3): MoE inspection → dense backends → existing-model MoE FT
   → full MoE → resource-elastic expert runtime.
4. When a native-Linux NVMe box is ready: the **untruncated seq-4096 WBG-7B re-train** for paper numbers
   (the WBG WSL training now points at C:, not the F: USB drive — see §0).

## 8. Hardware + environments

- **GPU**: RTX 5070, 12 GB, Blackwell (sm_120, cc 12.0), Windows/WDDM. Measured VRAM ceiling for 7B
  4-bit QLoRA: **~10.8 GB @ seq 1024, ~13.8 GB @ 2048**; above ~1280 the WDDM path spills to system RAM
  and crawls (native Linux OOMs instead). This is why long-seq training wants native Linux + offload.
- **Engine venv**: `C:\CorpusStudio\engine\.venv` (3.12.10). Recreate with
  `py -3.12 -m venv .venv; .\.venv\Scripts\python -m pip install -e .[dev]`.
- **GPU train venv**: rebuild `C:\cs-train-venv` when training (was torch 2.11.0+cu128 + `[train]`).
- **Optional extras**: `[train]` (torch/transformers/peft/trl/bitsandbytes), `[parquet]` (pyarrow),
  `[tokenizer]` (tiktoken), `[model-tokenizer]` (tokenizers).

## 9. Repository map

- **Platform contracts**: `engine/corpus_studio/platform/contracts.py` (26 root contracts) + `enums.py`
  + `common.py`; `schema_export.py` → `docs/contracts/*.schema.json` (language-neutral, consumed by
  `apps/web`).
- **Lifecycle**: `platform/{profiler, probes, planner, calibrator, supervisor, runners, watchdog,
  subprocess_supervisor, worker, backends}.py`.
- **Environment Manager** (Phase 2): `platform/environments.py` (recipes + concrete resolver) and
  `platform/environment_manager.py` (runtime discovery, creation, journals, lock/probe/drift, owned
  removal/recreation, and RunPlan compatibility).
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
- **Storage**: `platform/storage_profiler.py` (topology + per-role safe-spill guardrail + failure
  diagnostic).
- **Trainer**: `training/trainer.py` (+ `unsloth_trainer.py`). CLI: `engine/corpus_studio/cli.py`
  (including `platform-*`, the full `env-*` lifecycle, and `train-*`).
- **Docs**: source of truth `docs/CURRENT_STATE.md`; plan `docs/IMPLEMENTATION_PLAN.md`; MoE
  `docs/MOE_ARCHITECTURE.md`; storage `docs/HARDWARE_STORAGE_PROFILE.md`; env
  `docs/ENVIRONMENT_MANAGER.md`; objectives `docs/TRAINING_OBJECTIVES.md`; parameter evidence
  `docs/PARAMETER_ACCOUNTING.md`; platform run `docs/PLATFORM_RUN.md`.
- **Claude memory** (persists across Claude sessions): `C:\Users\Brend\.claude\projects\F--CorpusStudio\memory\`
  — start at `MEMORY.md` (index); `platform-architecture-epic.md` is the big one;
  `storage-migration-c-drive.md` records this move.

## 10. Notes for Codex specifically

- Languages: **Python** (engine — the primary surface), **C#/.NET** (`apps/desktop` WPF +
  `apps/desktop/CorpusStudio.Avalonia`), **TypeScript/React** (`apps/web`, Tauri 2).
- Use the venv at `engine/.venv`; run the gate in §4. Respect the invariants in §6 (especially
  no-shell argv, fail-closed policy, and the Blackwell math-attention rule).
- The engine core must stay import-torch-free; put heavy deps behind lazy imports + the `[train]` extra.
- Language-neutral contracts are the boundary — change the pydantic models in `platform/contracts.py`,
  then regenerate `docs/contracts/*.schema.json` (§4); the TS types in `apps/web/src/contracts/` are
  generated from those schemas.
