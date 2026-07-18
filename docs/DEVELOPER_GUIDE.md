# Developer Guide

A hands-on guide to working in CorpusStudio's **Python engine + platform** — the
dependency-light control plane that all UI clients drive. The WPF/Avalonia desktop that
this guide used to document has been **removed** (#545); the target UI is the **Tauri 2 +
React** frontend (`apps/web`), a contract-first client whose Studio-screen port is in
progress. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the layered picture and
[`CLI_REFERENCE.md`](CLI_REFERENCE.md) for every command.

## The pieces

- **`engine/corpus_studio/`** — the Python engine: a **dependency-light** core (no torch at
  import) that owns dataset logic (schemas, validation, importers, quality, splitting,
  exporters, versioning, evaluation reporting, provider policy) and exposes it as a CLI.
- **`engine/corpus_studio/platform/`** — the **contract-first, torch-free** run lifecycle
  (profile → plan → predict-fit → run → measure-fit → artifacts), the Environment Manager,
  and the sealed worker protocol. Heavy ML deps are lazy and behind the `[train]` extra.
- **`apps/web/`** — the Tauri 2 + React client. It is a *client* over the engine CLI (it
  shells out over argv, never shell strings); it never owns training behavior.

Target architecture: a **Rust authoritative core** + isolated, untrusted Python ML workers
("Rust owns truth; Python computes ML and returns evidence") — tracking epic #522, staged and
incremental (no big-bang rewrite).

## Requirements

- **Python 3.12** for local dev (CI runs 3.11); the engine venv is `engine/.venv` (CPython
  3.12.3, torch-free core + `[dev]`).
- A **native-Linux** host is the current target (Ubuntu 24.04). See
  [`HOST_STATE.md`](HOST_STATE.md) for verified host facts before anything hardware-adjacent.
- Node.js/npm and the Rust/Tauri prerequisites for building `apps/web`.

## Native-Linux quick start

Work from the active checkout, not the historical Windows mounts:

```bash
cd /mnt/training-nvme/repos/CorpusStudio
cp .env.example .env
cd engine
python3 -m venv .venv            # first time only
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q --no-header --basetemp=.pytest_tmp
```

Heavy training deps are opt-in (`pip install -e '.[train]'`) and never imported by the core.
Optional accuracy extras: `[tokenizer]`, `[model-tokenizer]`, `[parquet]`. The web client:
`cd apps/web && npm ci && npm run build`.

## The verify gate (run before you claim a change done)

From `engine/` with the venv:

```bash
.venv/bin/python -m ruff check corpus_studio tests
.venv/bin/python -m mypy corpus_studio
.venv/bin/python -m pytest -q --no-header --basetemp=.pytest_tmp --cov=corpus_studio --cov-fail-under=88
```

CI runs on **Linux / Python 3.11** with the same coverage floor (**88%**), plus the web build
(`.github/workflows/web.yml`) and CodeQL (Python). Do not report "done" on red.

## Contracts are the boundary

Change `platform/contracts.py` (pydantic) → regenerate the JSON schemas
(`python -c "from corpus_studio.platform.schema_export import export_json_schemas; export_json_schemas('../docs/contracts')"`)
→ regenerate the TS types (`cd apps/web && npm run gen:contracts`) → update the two counts in
`tests/test_platform_contracts.py`. Schemas and TS are committed and CI diffs them; a drifted
schema fails CI.

## Recipe: add a feature

1. **Engine first.** Add the dataset/platform logic to the Python engine + deterministic tests;
   expose a CLI command. Keep it dependency-light and honest (report what a number measures;
   never overclaim). Run the verify gate.
2. **Contract, if the shape crosses the boundary.** If a UI client consumes it, add/extend the
   pydantic contract and regenerate schemas + TS (above).
3. **UI (client).** The Tauri/React client shells out to the new command over argv and renders
   the JSON; it never re-implements engine logic.

## Testing philosophy

The engine has a large pytest suite (2,000+ tests): deterministic, fixture-driven, no GPU,
torch-free where the code is torch-free. Tests inject fakes for network/GPU/worker boundaries
so a run never contacts a real endpoint or loads weights. Opt-in local Ollama integration tests
(`engine/tests/test_ollama_integration.py`) require `CORPUS_STUDIO_OLLAMA_INTEGRATION=1` and a
running backend, and self-skip when unavailable (with `CORPUS_STUDIO_OLLAMA_MODEL` /
`CORPUS_STUDIO_OLLAMA_BASE_URL` overrides).

## Honesty invariants (don't weaken these)

License fail-closed; provenance gate; "a completed step ≠ proven fit" (predicted fit is never
`NATIVE_SAFE`); "installed ≠ supported"; no silent target truncation; single-writer
`examples.jsonl` (the engine's `examples-append` / `storage/examples_writer.py` is the sole
sanctioned writer). A suite/gate PASS is a *structure/threshold* verdict, not proof of quality;
keyword-overlap is a lexical proxy; provider policy keeps cloud models evaluator-only; PII
redaction masks known patterns and is *not* de-identification. Preserve the verdict semantics
and the "what this measures" wording when you touch these.

## Useful engine commands

On the native-Linux host (venv `engine/.venv/bin/python`):

```bash
cd /mnt/training-nvme/repos/CorpusStudio
engine/.venv/bin/python -m corpus_studio.cli schemas
engine/.venv/bin/python -m corpus_studio.cli validate examples/datasets/instruction/train.jsonl instruction
engine/.venv/bin/python -m corpus_studio.cli new-project demo "Demo Dataset" instruction
engine/.venv/bin/python -m corpus_studio.cli examples-append data/projects/demo --from rows.jsonl
engine/.venv/bin/python -m corpus_studio.cli quality examples/datasets/instruction/train.jsonl
engine/.venv/bin/python -m corpus_studio.cli platform-plan --help
```

See [`CLI_REFERENCE.md`](CLI_REFERENCE.md) for the full command set. First-party training is a
separate **opt-in** feature — build the managed `[train]` worker and use `platform-plan` /
`platform-run`; see [`TRAINING.md`](TRAINING.md).
