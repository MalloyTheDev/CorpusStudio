---
name: corpus-studio
description: "Project-specific guardrails for editing the CorpusStudio repo (the local-first, hardware-aware AI training platform at /mnt/training-nvme/repos/CorpusStudio: torch-free control plane + isolated [train] QLoRA worker + WPF/Avalonia/Tauri UI + the IEEE native-Linux research protocol). Use for ANY change here - engine/platform contracts, the trainer, the Environment Manager, the CLI, the research amendments, or docs - so every edit keeps the dependency-light boundary, the honesty invariants, the contract-regeneration dance, the append-only research protocol, and the one-coherent-CI-green-PR-per-slice workflow. Pair it with the general existing-repo-engineer skill (that one owns generic repo mechanics; this one owns CorpusStudio's non-negotiable invariants and current program state). Keep trivial questions lightweight; load this whenever files or repository behavior in CorpusStudio actually matter."
---

# CorpusStudio

Operate as a senior engineer on **CorpusStudio** - a local-first, hardware-aware AI training platform:
a dependency-light, torch-free **control plane** (`engine/corpus_studio/platform/`), an opt-in `[train]`
QLoRA **worker**, WPF/Avalonia/Tauri **UI** clients, and a preregistered **IEEE native-Linux research
protocol** (`research/ieee-linux-training/`). The bar here is not "it works" - it is "the evidence,
contracts, and honesty invariants still hold." A passing run obtained by weakening a gate is a defect.

This skill is the CorpusStudio-specific overlay. It assumes the general **existing-repo-engineer**
skill for repo mechanics (recon, minimal-diff, verify-before-handoff) and adds the invariants a generic
skill cannot know. When they conflict, **this skill and the repo's own `CLAUDE.md` / `AGENTS.md` win.**

## Match effort to the task

- **Answer/inspect** - a question or a read. Don't spin up the workflow; just read the right files.
- **Small edit** - one coherent change. Follow the verify gate before you claim done.
- **Slice** - a feature/fix worth a PR. Branch, keep it one coherent CI-green PR, regenerate what the
  change touches, merge after CI. This is the default unit of work here.
- **Research/hardware/GPU** - stop and read the boundary sections below before touching anything
  hardware-adjacent or protocol-adjacent. These have hard stop conditions.

## Read-first, always

Before non-trivial work, ground yourself in the repo's own source of truth (do not duplicate it here):

- [`CLAUDE.md`](../../../CLAUDE.md) + [`AGENTS.md`](../../../AGENTS.md) - the agent contract (imported).
- [`docs/HOST_STATE.md`](../../../docs/HOST_STATE.md) - verified host facts; read before anything
  hardware-adjacent. Where older docs show Windows `C:`/`F:` paths, HOST_STATE supersedes them.
- [`HANDOFF.md`](../../../HANDOFF.md) - session state + roadmap.
- [`docs/CURRENT_STATE.md`](../../../docs/CURRENT_STATE.md) - authoritative feature state.
- [`docs/IMPLEMENTATION_PLAN.md`](../../../docs/IMPLEMENTATION_PLAN.md) / [`docs/MOE_ARCHITECTURE.md`](../../../docs/MOE_ARCHITECTURE.md) - forward plan; foundational contracts must be MoE-safe now.

## The verify gate (run before you claim a change is done)

From `engine/` with the venv (`engine/.venv`, CPython 3.12.3, torch-free core + `[dev]`):

```bash
.venv/bin/python -m ruff check corpus_studio tests
.venv/bin/python -m mypy corpus_studio
.venv/bin/python -m pytest -q --no-header --basetemp=.pytest_tmp
```

- CI runs on **Linux / Python 3.11** with `pytest --cov=corpus_studio --cov-fail-under=88` plus C#/web
  jobs. The coverage floor is **88%** and this host is native Linux, so local coverage ~= CI coverage
  (no Windows under-measurement). Keep a margin above 88, not 88.0x exactly.
- Do not report "done" on red. If tests fail, say so with the output.

## Do not break - the non-negotiable invariants

1. **Dependency-light boundary.** `import corpus_studio.platform` and the engine core must pull **no
   torch** (nor transformers/trl/peft/bitsandbytes). All heavy deps are lazy-imported and live behind
   the `[train]` extra. `tests/test_platform_contracts.py::test_platform_import_is_torch_free` guards
   this; a new `platform/` module that imports torch at module load is a defect.
2. **Contracts are the boundary.** Change `platform/contracts.py` (pydantic) -> **regenerate**
   `docs/contracts/*.schema.json`:
   `python -c "from corpus_studio.platform.schema_export import export_json_schemas; export_json_schemas('../docs/contracts')"`
   -> regenerate the TS types (`cd apps/web && npm run gen:contracts`) -> **update the two counts** in
   `tests/test_platform_contracts.py` (root-contract count + the schema-writer count) when you add/remove
   a `ROOT_CONTRACTS` entry. Schemas and TS are committed and CI diffs them; a drifted schema fails CI.
3. **No-shell execution.** Installers and trainer launches are `argv` lists, never shell strings.
4. **Honesty invariants.** License fail-closed; provenance gate blocks frontier teachers; "a completed
   step != proven fit"; "installed != supported"; no silent target truncation; predicted fit is never
   `NATIVE_SAFE` (only a measured run is); single-writer `examples.jsonl`. Never weaken an evidence
   contract, precision requirement, kernel enforcement, artifact integrity, failure taxonomy, or
   provenance rule to obtain a passing run.
5. **Blackwell / sm_120.** The **math** SDPA attention path is the verified-safe default (fused
   flash-SDPA deadlocks on native Windows/WDDM). Unsloth is refused on Windows/WDDM. Bare-Linux flash
   for the real workload is not yet claimed.
6. **ASCII in CLI-facing strings** (Windows console UTF-8): use `-`, not the em dash; no non-ASCII in
   `typer` help, `raise ...Error(...)` messages, or anything printed to a console.
7. **One training authority.** Shipping clients use `platform-plan` -> `platform-run`; never the
   development-only `train-run`. Every execution gets a fresh run ID and a run-scoped output directory;
   runner identity derives from the pinned backend manifest.
8. **Hardware claims stay evidence-bound.** The managed `backend-corpus-studio` environment is
   `HARDWARE_VERIFIED` (env-manager GPU probe) - that is NOT a workload result. Do not claim
   full-sequence 7B success, DeepSpeed/FSDP/CPU/NVMe offload, real offload fit, PCIe/NVMe throughput,
   bare-Linux FlashAttention for the workload, or MoE runtime capability without a measured run.

## Research protocol (research/ieee-linux-training/) - append-only

- The protocol is **append-only after any visible result**. Never edit `PROTOCOL.md`, the base
  `EXPERIMENT_MATRIX.yaml`, a frozen amendment, a frozen effective-matrix JSON, or a `RESERVED_IDENTITIES.v*`
  in place. A change is a **new dated amendment** -> new effective-matrix version -> superset
  `RESERVED_IDENTITIES.vN+1` (append-only over the prior).
- `validate_protocol.py` must stay green (`--verify-host-evidence` too), effective-matrix reconstruction
  must be **byte-deterministic**, and a new amendment must hash-bind the prior one (supersession) and be
  set-disjoint from the reserved identities. Current effective version and the v5 bring-up procedure are
  in [`research/ieee-linux-training/README.md`](../../../research/ieee-linux-training/README.md) and
  `RUNBOOK_v5_bringup.md`.
- Metric definitions are authoritative in `research/ieee-linux-training/METRICS.md` (warm-up steps 1-2,
  measured 3-12; trapezoidal energy; n=3 t=4.3026527299). Any measurement code must implement them
  exactly, not approximately.

## Never do (project policy)

- **No multi-agent workflow fan-outs** for CorpusStudio work (cost) - verify inline, one Claude process.
- **No `git clean`.** Do not delete, rewrite, relabel, reuse, or mutate ANY historical worker wheel,
  environment, lock, capability report, execution probe, RunPlan, execution config, run, adapter, output
  directory, protocol version, amendment, evidence directory, or `SHA256SUMS`.
- **No writes to `/mnt/windows-c` or `/mnt/windows-f`** - read-only historical mounts by policy, even
  though the filesystem permits writes. Develop only from `/mnt/training-nvme/repos/CorpusStudio`.
- Do not build a worker wheel, create/recreate/remove a managed environment, generate an executable
  RunPlan, load model weights, or dispatch a GPU workload unless the task explicitly authorizes it.
- **STOP and surface** before: a destructive/irreversible op; a credential or legal-authorization
  decision; a full 7B run (substantial time); a scientific change that would amend the study after
  results are visible; or a hardware-risk condition. For GPU work: **unload Ollama first, one GPU
  operation at a time.**

## Slice workflow (the default unit of work)

1. `git checkout -b feat/<slice>` (or `fix/`, `research/`) - branch first; `main` is protected.
2. Make one coherent change. Regenerate schemas/TS + update contract counts if you touched
   `platform/contracts.py`. Update the doc that the change makes stale (HANDOFF/HOST_STATE/CURRENT_STATE).
3. Run the full verify gate green (ruff + mypy + pytest at >=88% coverage). Add deterministic tests -
   fixture-driven, no GPU, torch-free where the code is torch-free.
4. Commit with the repo's trailers; open one PR; **wait for CI**; merge after green (admin-merge is used
   here for CI-green PRs when review is the only blocker, per standing authorization). Keep one coherent
   CI-green PR per slice - do not batch unrelated changes.

## Environment gotchas (learned the hard way)

- **cwd drift:** a `cd engine && ...` in a compound Bash command shifts the persisted working directory
  and scopes later `git`/greps to `engine/`. Prefer absolute paths or `git -C`; verify with `pwd`.
- **CRLF working tree:** the working tree is all CRLF, committed is LF. `git diff --check` is noisy;
  preserve per-file line endings, never bulk-normalize. Write new files with LF.
- **pytest basetemp:** a stale custom `--basetemp` dir causes `FileNotFoundError`; `rm -rf` it before a
  run, and never commit `.pytest_tmp*` artifacts (stage explicit paths, not `git add -A`).
- **Do not commit generated junk:** stray `*.schema.json`/`index.json` exported into `engine/` are
  session artifacts - clean them; the canonical schemas live in `docs/contracts/`.

## Where the current program stands (update this as it moves)

The active thread is bringing CorpusStudio to full 7B training/research readiness. Amendment 0002 ->
effective matrix 1.2.0 is merged (v5 identities allocated, v4 reserved). The Section-11 measurement
harness is in review. Next non-GPU slices: exact resume + checkpoint lineage (#440), then the 500-output
corpus freeze - all gated before any GPU/7B/corpus work, which needs separate human authorization. The
live detail always lives in `HANDOFF.md` + `docs/HOST_STATE.md`; trust those over this paragraph.
