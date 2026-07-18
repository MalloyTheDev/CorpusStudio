# AGENTS.md — CorpusStudio

Instructions for AI coding agents (Codex, Claude, etc.) working in this repo.
**Full session state + roadmap: read [`HANDOFF.md`](HANDOFF.md) first.**

## Where you are
- Work from **`/mnt/training-nvme/repos/CorpusStudio`** - the native-Linux RTX 5070 host (Ubuntu 24.04).
  Verified host facts (paths, GPU, managed environment): [`docs/HOST_STATE.md`](docs/HOST_STATE.md).
- Engine venv: `/mnt/training-nvme/repos/CorpusStudio/engine/.venv` (CPython 3.12.3, torch-free core + `[dev]`).
- History (mounted, do not work from): this repo previously lived on Windows `F:` (USB) then `C:`
  (migrated 2026-07-13). The filesystems are mounted read-write at `/mnt/windows-f` and
  `/mnt/windows-c` (e.g. old `C:\CorpusStudio` -> `/mnt/windows-c/CorpusStudio`), but project policy is
  history-only: do not develop from or write to those stale fallbacks because they will drift.

## What this is
CorpusStudio is a **local-first, end-to-end AI development ecosystem and IDE** covering the complete model
lifecycle, organized into **seven co-equal product areas**: **Data Studio**, **Training Studio**,
**Evaluation Studio**, **Behavior Lab**, **Model & Release Studio**, **Environment & Hardware**, and
**Evidence & Experiments** (canonical map: [`docs/PRODUCT_AREAS.md`](docs/PRODUCT_AREAS.md)). It is **not** a
"research platform" or "training platform" - those are individual capabilities. Its surface spans: data
ingestion / import / conversion / cleaning / dedup / validation / versioning → schema support (pretraining,
instruction, chat, preference, evaluation) → inspection, quality, provenance, licensing → model + tokenizer
selection → fine-tuning / (future) pretraining (config, env setup, checkpoint/resume) → evaluation &
comparison → behavior analysis & modification → adapter / model export → release → reproducible evidence -
all **hardware-aware**. Behavior Lab is a first-class area (implementation gated). Control plane stays
lightweight; heavy frameworks live in isolated worker envs; the UI is a client. See
[`docs/PRODUCT_SPEC.md`](docs/PRODUCT_SPEC.md).

The **native-Linux 7B research paper** (`research/ieee-linux-training/`, `docs/paper/`) is a **separate
project that uses** CorpusStudio to verify the training engine can train a 7B model at sequence length
4096 on this host. Its experiment matrices, amendments, reserved identities, paper-performance gates, and
sealed-research evidence rules are an **opt-in overlay**. **The IEEE 7B paper must not define
CorpusStudio's product identity, defaults, navigation, or ordinary user workflow** — though CorpusStudio
may still contain opt-in research and interpretability tools (e.g. a future Behavior Lab). Resource-elastic
MoE likewise must not define the product's identity, navigation, defaults, or ordinary workflow — **but
foundational contracts must stay dense-safe and MoE-compatible: no new foundational contract may assume
dense execution** (`docs/IMPLEMENTATION_PLAN.md`, `docs/MOE_ARCHITECTURE.md`). The standard / verified /
sealed-research boundary is [`docs/PRODUCT_VS_RESEARCH.md`](docs/PRODUCT_VS_RESEARCH.md).
Three surfaces:
- **Engine** (`engine/corpus_studio/`, Python) — a **dependency-light** core (no torch at import) +
  an opt-in `[train]` QLoRA trainer extra.
- **Platform** (`engine/corpus_studio/platform/`) — a **contract-first, torch-free** run lifecycle
  (profile → plan → predict-fit → run → measure-fit → artifacts) + the Environment Manager (sealed
  reference-backend creation, locks, probes, drift, and safe recreation) and the storage safe-spill
  profiler + static, offline model/tokenizer descriptors and inspection + the sealed,
  backend-independent TrainingObjective registry and compatibility evidence checker + the versioned,
  provenance/review-safe TraceRecord workflow + hash-pinned, allowlisted static MoE topology evidence
  (never runtime capability proof) + the identity-bound worker protocol 2.0 and fake-worker
  conformance/process-tree boundary (not a real new backend) + the hash-sealed
  `ResolvedExecutionConfiguration` consumed directly by the first-party worker (no post-seal semantic
  overrides, implicit placement, or silent trainer-field filtering).
- **UI** — WPF + Avalonia (`apps/desktop`, C#) and Tauri 2 + React (`apps/web`, TS). UI is a client
  over the engine CLI; it never owns training behavior.

## Build & verify (the gate)
From `engine/` with the venv:
```
.venv/bin/python -m ruff check corpus_studio tests
.venv/bin/python -m mypy corpus_studio
.venv/bin/python -m pytest -q --no-header --basetemp=.pytest_tmp
```
CI runs on **Linux / Python 3.11** with `pytest --cov=corpus_studio --cov-fail-under=88` + C#/web jobs.
- **Coverage**: this host is native Linux, so `storage_profiler`'s Linux-only detection now runs
  locally - the old ~0.3% Windows-run under-measurement no longer applies; the CI floor is 88%.
- **After editing any `platform/` contract**, regenerate schemas:
  `python -c "from corpus_studio.platform.schema_export import export_json_schemas; export_json_schemas('../docs/contracts')"`
  and update the two counts in `tests/test_platform_contracts.py`.

## Rules — do not break
- **Dependency-light boundary**: `import corpus_studio.platform` and the engine core must pull **no
  torch**; all heavy deps are lazy-imported and live behind the `[train]` extra.
- **No-shell execution**: installers and trainer launches are `argv` lists, never shell strings.
- **Honesty invariants**: license fail-closed; provenance gate; "a completed step ≠ proven fit";
  "installed ≠ supported"; no silent target truncation; predicted fit is never `NATIVE_SAFE` (only a
  measured run is); single-writer `examples.jsonl`.
- **Blackwell / sm_120**: the **math** attention path is the verified-safe default (fused flash-SDPA
  deadlocks on native Windows WDDM; on this native-Linux host the env hardware probe verified the math
  path, and bare-Linux flash-for-the-workload is not yet claimed). Unsloth is refused on Windows/WDDM.
- **ASCII in CLI-facing strings** (Windows console UTF-8 — no `—`, use `-`).
- **Contracts are the boundary**: change `platform/contracts.py` (pydantic) → regenerate
  `docs/contracts/*.schema.json` → the TS types in `apps/web/src/contracts/` derive from those.
- **One training authority**: shipping clients use `platform-plan` → `platform-run`; they never invoke
  the development-only `train-run` compatibility command. Every execution gets a fresh run ID and a
  run-scoped output directory; runner identity derives from the pinned backend manifest.
- **Hardware claims stay evidence-bound**: the native-Linux RTX 5070 host is now assembled and the
  managed `backend-corpus-studio` environment is `HARDWARE_VERIFIED` (env-manager CUDA alloc + 4-bit
  construction + minimal GPU fwd/bwd + math SDPA - see [`docs/HOST_STATE.md`](docs/HOST_STATE.md)). That
  environment-probe level is NOT a workload result: still do not claim full-sequence 7B success,
  DeepSpeed/FSDP/CPU/NVMe offload, real offload fit, PCIe/NVMe throughput or endurance, bare-Linux
  FlashAttention for the real workload, or MoE runtime capability. Contracts, fake workers, CI, and a
  passing env hardware probe are not proof the 7B workload trains.

## Process
- Branch first (`git checkout -b feat/<slice>`), one coherent CI-green PR per slice.
- Do NOT spawn multi-agent fan-outs by default (cost); verify inline.
- Source of truth for features: `docs/CURRENT_STATE.md`. Plan: `docs/IMPLEMENTATION_PLAN.md`.
  Product vs research boundary: `docs/PRODUCT_VS_RESEARCH.md`. MoE must not define product identity,
  navigation, defaults, or ordinary workflow, but **no new foundational contract may assume dense
  execution**: `ModelDescriptor`, `TrainingObjective`, `RunPlan`, `ArtifactManifest`, checkpoint,
  telemetry, and evaluation stay dense-safe / MoE-compatible (`docs/IMPLEMENTATION_PLAN.md`,
  `docs/MOE_ARCHITECTURE.md`).
