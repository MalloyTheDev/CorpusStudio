---
name: corpus-studio
description: "Project guardrails for editing CorpusStudio - a local-first, end-to-end AI development ecosystem and IDE across seven co-equal areas (Data, Training, Evaluation, Behavior Lab, Model & Release, Environment & Hardware, Evidence & Experiments). Load for any change here so every edit keeps the dependency-light boundary, the honesty invariants, the contract-regeneration dance, and the one-CI-green-PR-per-slice workflow. The IEEE 7B paper is a separate opt-in overlay, not the product identity; CLAUDE.md / AGENTS.md win on conflict."
---

# CorpusStudio

Operate as a senior engineer on **CorpusStudio** - a **local-first, end-to-end AI development ecosystem
and IDE** covering the full model lifecycle, organized into seven co-equal product areas: Data Studio,
Training Studio, Evaluation Studio, Behavior Lab, Model & Release Studio, Environment & Hardware, and
Evidence & Experiments (map: [`docs/PRODUCT_AREAS.md`](../../../docs/PRODUCT_AREAS.md)). It is a product,
**not** a "research platform" or "training platform" - those are capabilities. Surfaces: a
dependency-light, torch-free **control plane** (`engine/corpus_studio/platform/`), an opt-in `[train]`
QLoRA **worker**, and a **Tauri 2 + React** UI (`apps/web`); the target is a Rust authoritative core with
isolated Python ML workers. The bar is not "it works" - it is "the evidence, contracts, and honesty
invariants still hold." A run obtained by weakening a gate is a defect.

This is the CorpusStudio overlay on the general **existing-repo-engineer** skill (which owns
recon / minimal-diff / verify-before-handoff). On conflict, the repo's **`CLAUDE.md` / `AGENTS.md` win**,
then this skill.

## Authorities (source of truth - do not duplicate here)

Load only what the task activates - a doc typo does not need host, GPU, or research state.

- [`AGENTS.md`](../../../AGENTS.md) - the always-loaded contract + honesty invariants (authority for the
  rules below).
- [`docs/CURRENT_STATE.md`](../../../docs/CURRENT_STATE.md) - what is built today (wins on feature state);
  [`docs/PRODUCT_SPEC.md`](../../../docs/PRODUCT_SPEC.md) / [`docs/PRODUCT_AREAS.md`](../../../docs/PRODUCT_AREAS.md) - identity.
- [`docs/HOST_STATE.md`](../../../docs/HOST_STATE.md) - verified host/GPU/env facts; **read before
  anything hardware-adjacent**. Host paths and machine state live here, not in guidance; resolve the
  checkout with `git rev-parse --show-toplevel`.
- [`docs/PRODUCT_VS_RESEARCH.md`](../../../docs/PRODUCT_VS_RESEARCH.md) - the standard/verified/sealed
  boundary; [`HANDOFF.md`](../../../HANDOFF.md) - volatile session state + program status.

## Match effort to the task

- **Answer/inspect** - a read; don't spin up the workflow.
- **Small edit** - one coherent change; run the verify gate before claiming done.
- **Slice** - a feature/fix worth a PR; the default unit of work (see the slice workflow below).
- **Research/hardware/GPU** - stop and read the boundary + STOP conditions before touching anything
  hardware- or protocol-adjacent.

## The verify gate (run before claiming a change is done)

From `engine/` with the project venv:

```bash
.venv/bin/python -m ruff check corpus_studio tests
.venv/bin/python -m mypy corpus_studio
.venv/bin/python -m pytest -q --no-header --basetemp=.pytest_tmp
```

CI runs the same on Linux + Python 3.11 with `--cov=corpus_studio` (coverage floor **88%** - never weaken
it) plus the web/contract job. Do not report "done" on red; if tests fail, say so with the output.

## Non-negotiable invariants (AGENTS.md is the authority; one-liners here)

1. **Dependency-light boundary.** `import corpus_studio.platform` and the core pull **no torch** (nor
   transformers/trl/peft/bitsandbytes); heavy deps are lazy-imported behind `[train]`. A `platform/`
   module importing torch at load is a defect (`tests/test_platform_contracts.py` guards it).
2. **Contracts are the boundary.** Edit `platform/contracts.py` -> regenerate `docs/contracts/*.schema.json`
   (`python -c "from corpus_studio.platform.schema_export import export_json_schemas; export_json_schemas('../docs/contracts')"`)
   -> regenerate TS (`cd apps/web && npm run gen:contracts`) -> update the two counts in
   `tests/test_platform_contracts.py`. Schemas + TS are committed and CI diffs them.
3. **No-shell execution.** Installers and trainer launches are `argv` lists, never shell strings.
4. **Honesty invariants.** License/provenance fail-closed; "a completed step != proven fit"; "installed
   != supported"; no silent target truncation; predicted fit is never `NATIVE_SAFE` (only a measured run
   is); single-writer `examples.jsonl`. **An unavailable metric is null with a typed reason, never a
   plausible-looking zero.** Never weaken an evidence contract, precision requirement, kernel enforcement,
   artifact-integrity check, failure taxonomy, or provenance rule to obtain a passing run.
5. **Attention path.** The **math** SDPA path is the verified-safe default. An exploratory PRODUCT run has
   trained 7B QLoRA at seq 4096 with flash SDPA + liger fused-CE + paged 8-bit AdamW - that is exploratory
   evidence, **not** a sealed claim, and not `flash_attention_2` / external `flash-attn`. Details live in
   HOST_STATE / HANDOFF.
6. **ASCII in CLI-facing strings** for console portability: use `-`, not the em dash; no non-ASCII in
   `typer` help or `raise ...Error(...)` messages.
7. **One training authority.** Shipping clients use `platform-plan` -> `platform-run`, never the
   development-only `train-run`; every execution gets a fresh run id + run-scoped output; runner identity
   derives from the pinned backend manifest.
8. **Hardware claims stay evidence-bound.** `backend-corpus-studio` is `HARDWARE_VERIFIED` (an env probe,
   NOT a workload result). No full-sequence-7B / offload-fit / throughput / bare-Linux-flash-for-the-
   workload / MoE-runtime claim without a measured run. **No new foundational contract may assume dense
   execution** (`ModelDescriptor` / `TrainingObjective` / `RunPlan` / `ArtifactManifest` / checkpoint /
   telemetry / evaluation stay dense-safe / MoE-compatible).

## Worker execution closure (classify before editing any `platform/` module)

A change to worker-execution bytes needs a fresh pinned worker package + new environment locks (sealed
research additionally needs an amendment -> effective-matrix bump -> superset reserved identities -> a
reproducibly rebuilt wheel -> new `-vN` environments). Control-plane-only changes do not.

- **The closure** = everything the `--subprocess` child imports/executes:
  `platform/worker.py::run_worker` -> `supervisor.py::execute_run` -> success admission
  (`platform/artifacts.py`) -> `platform/runners.py` -> `training/trainer.py`. **`artifacts.py` and
  `runners.py` are worker code even though they live under `platform/`.**
- **Do not classify from a file list - TRACE the import path.** Modules are reached via lazy
  (function-local) imports: e.g. `planner` is imported inside `worker.py` and `runners.py`, so it is
  **runtime-reachable, not automatically control-plane-only**. When a change touches a module reachable
  from the worker, treat it as `RUNTIME_REACHABLE_REVIEW_REQUIRED` until a symbol-level trace proves
  non-impact. Write the verdict down (WORKER_CHANGE_REQUIRED vs control-plane-only).
- **Once identities are instantiated** (wheel built, environments sealed, a run produced), a "reuse the
  lineage" rationale that depended on non-instantiation no longer holds - prove non-impact or bump the
  lineage.

## Two general engineering rules

- **Framework output-tree admission.** The trainer runs TRL/transformers, which writes its own files
  (e.g. `training_args.bin`, a pickle - not weights) into the adapter dir. Admit benign metadata by a
  **narrow named allowlist + structural guards** (regular file, single hard link, size cap, never
  deserialized); never relax the weight-payload class (`platform/artifacts.py`). Original outputs are
  immutable.
- **CPU-before-GPU semantic validation.** Schema validity != semantic executability. `platform-plan` runs
  a torch-free conformance preflight (`platform/dataset_conformance.py`) that refuses a zero-usable-row
  plan before any id is minted; render + tokenize with the exact immutable tokenizer on CPU first. Never
  let GPU work discover a defect a CPU preflight could reject.

## Research overlay (paper only)

The IEEE native-Linux 7B paper (`research/ieee-linux-training/`, `docs/paper/`) is an **opt-in overlay**,
not product behavior. Its append-only amendments, effective matrices, reserved identities, sealed
`required_git_ancestor` floors, telemetry completeness, and promotion rules apply **only** when the task
is the paper - use the `corpusstudio:research-overlay` skill and
[`research/ieee-linux-training/README.md`](../../../research/ieee-linux-training/README.md). Never edit a
frozen protocol / amendment / matrix / reserved-identity in place; `validate_protocol.py` must stay green.
Volatile program state (ids, hashes, run status, readiness) lives in `HANDOFF.md` + `docs/HOST_STATE.md` -
trust those, not this skill. Ordinary product work needs none of this.

## Slice workflow (the default unit of work)

1. `git checkout -b feat/<slice>` (or `fix/` / `research/`) - branch first.
2. Make one coherent change; regenerate schemas/TS + update the contract counts if you touched
   `platform/contracts.py`; update the doc the change makes stale (HANDOFF / HOST_STATE / CURRENT_STATE).
3. Run the full verify gate green; add deterministic tests (fixture-driven, no GPU, torch-free where the
   code is).
4. Commit with the repo's trailers; open one coherent CI-green PR per slice; wait for CI + review. Do not
   batch unrelated changes.

## Never do / STOP and surface

- **No `git clean`; never delete, rewrite, relabel, reuse, or mutate** any historical worker wheel,
  environment, lock, capability report, execution probe, RunPlan, execution config, run, adapter, output
  directory, protocol version, amendment, evidence directory, or `SHA256SUMS`.
- Do not build a worker wheel, create/recreate/remove a managed environment, generate an executable
  RunPlan, load model weights, or dispatch GPU work unless the task explicitly authorizes it.
- **STOP and surface** before: a destructive/irreversible op; a credential or legal-authorization
  decision; a full 7B run; a study change that would amend results after they are visible; or a
  hardware-risk condition. For GPU work: unload Ollama first, one GPU operation at a time.

## Gotchas

- **cwd drift:** a `cd engine && ...` in a compound Bash command shifts the persisted cwd; prefer absolute
  paths or `git -C`, and verify with `pwd`.
- **pytest basetemp:** a stale `--basetemp` dir causes `FileNotFoundError` - `rm -rf` it first; never commit
  `.pytest_tmp*` (stage explicit paths, not `git add -A`).
- **Generated junk:** stray `*.schema.json` / `index.json` exported into `engine/` are session artifacts -
  the canonical schemas live in `docs/contracts/`.
