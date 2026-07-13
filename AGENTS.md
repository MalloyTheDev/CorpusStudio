# AGENTS.md — CorpusStudio

Instructions for AI coding agents (Codex, Claude, etc.) working in this repo.
**Full session state + roadmap: read [`HANDOFF.md`](HANDOFF.md) first.**

## Where you are
- Work from **`C:\CorpusStudio`** (migrated off the F: USB drive on 2026-07-13; F: is a stale fallback).
- Engine venv: `C:\CorpusStudio\engine\.venv` (Python 3.12.10).

## What this is
A **local-first, hardware-aware AI engineering platform** covering the whole lifecycle — sources →
dataset construction (objectives / mixtures / reasoning traces) → model + tokenizer management →
storage/hardware-aware run planning → training through swappable **isolated backends** → telemetry +
checkpoint/artifact lineage → evaluation → deployment prep. Control plane stays lightweight; heavy
frameworks live in isolated worker envs; the UI is a client. Research North Star: **resource-elastic
MoE** on consumer hardware (`N_resident << N_active << N_logical`) — so foundational contracts must be
**MoE-safe now**. See [`HANDOFF.md`](HANDOFF.md) §3 and [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).
Three surfaces:
- **Engine** (`engine/corpus_studio/`, Python) — a **dependency-light** core (no torch at import) +
  an opt-in `[train]` QLoRA trainer extra.
- **Platform** (`engine/corpus_studio/platform/`) — a **contract-first, torch-free** run lifecycle
  (profile → plan → predict-fit → run → measure-fit → artifacts) + the Environment Manager (sealed
  reference-backend creation, locks, probes, drift, and safe recreation) and the storage safe-spill
  profiler + static, offline model/tokenizer descriptors and inspection + the sealed,
  backend-independent TrainingObjective registry and compatibility evidence checker.
- **UI** — WPF + Avalonia (`apps/desktop`, C#) and Tauri 2 + React (`apps/web`, TS). UI is a client
  over the engine CLI; it never owns training behavior.

## Build & verify (the gate)
From `engine/` with the venv:
```
.\.venv\Scripts\python.exe -m ruff check corpus_studio tests
.\.venv\Scripts\python.exe -m mypy corpus_studio
.\.venv\Scripts\python.exe -m pytest -q --no-header --basetemp=.pytest_tmp
```
CI runs on **Linux / Python 3.11** with `pytest --cov=corpus_studio --cov-fail-under=88` + C#/web jobs.
- **Coverage gotcha**: a Windows run under-measures the Linux-only storage detection ~0.3%; target
  ≥ 88.2% locally.
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
- **Blackwell / sm_120**: force the **math** attention path (fused flash-SDPA deadlocks on native
  Windows WDDM). Unsloth is refused there.
- **ASCII in CLI-facing strings** (Windows console UTF-8 — no `—`, use `-`).
- **Contracts are the boundary**: change `platform/contracts.py` (pydantic) → regenerate
  `docs/contracts/*.schema.json` → the TS types in `apps/web/src/contracts/` derive from those.

## Process
- Branch first (`git checkout -b feat/<slice>`), one coherent CI-green PR per slice.
- Do NOT spawn multi-agent fan-outs by default (cost); verify inline.
- Source of truth for features: `docs/CURRENT_STATE.md`. Plan: `docs/IMPLEMENTATION_PLAN.md`.
  MoE constraint (foundational contracts must be MoE-safe): `docs/MOE_ARCHITECTURE.md`.
