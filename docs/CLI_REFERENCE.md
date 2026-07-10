# CLI Reference

The Corpus Studio engine is a dependency-light Python CLI. The desktop app shells out
to these same commands, so anything the app does you can also script.

**Invocation.** Installed (`pip install -e engine`): `corpus-studio <command> [args]`.
From source: `python -m corpus_studio.cli <command> [args]`. Every command supports
`--help` for its full option list; many emit JSON to **stdout** (parse it) and human
notes/progress to **stderr**.

**Conventions.** `--project-dir` / `<project_dir>` points at a dataset project folder
(the one holding `examples.jsonl`). Schemas are the built-in ids from `schemas`. Reports
are JSON-first. Live-backend commands (eval/suite/benchmark/ai-assist/backend-health)
make real calls to a running local model backend.

---

## Schemas & projects

| Command | What it does |
|---|---|
| `schemas` | List the built-in dataset schemas (id, fields). |
| `validate <file.jsonl> <schema>` | Validate a JSONL file against a built-in schema. |
| `new-project <id> <name> <schema>` | Create a local dataset project folder. |
| `project-list [--root <dir>]` | List local dataset projects (via the optional SQLite index). |
| `project-index-rebuild [--root <dir>]` | Rebuild the optional SQLite project index from `project.json` files. |

## Quality & debt

| Command | What it does |
|---|---|
| `quality <examples.jsonl>` | Build a basic quality report (empties, duplicates, low-information, first-pass synthetic patterns). |
| `dataset-debt <examples.jsonl> [--json]` | Summarize outstanding quality debt as a prioritized, graded (A–F) ledger. |

## Import (→ staging JSONL)

Corpus Studio is JSONL-canonical; tabular / Hub sources convert to a **staging JSONL**
that flows through the same preview → quarantine → commit path.

| Command | What it does |
|---|---|
| `import-preview <file.jsonl> <schema>` | Preview a JSONL import and report accepted / rejected rows. |
| `import-convert <file> <output.jsonl>` | Convert a CSV / TSV / **Parquet** file to a staging JSONL (Parquet needs the `[parquet]` extra). |
| `hf-inspect <dataset-id>` | Inspect a public Hugging Face dataset: configs/splits, columns, license. |
| `hf-import <dataset-id> …` | Import rows from a **public** Hugging Face dataset into a staging JSONL. |

## Splits

| Command | What it does |
|---|---|
| `split <file.jsonl> <schema> …` | Validate + split JSONL into train / validation / test (deterministic seed; leakage-checked). |

## Evaluation & suites

All make **live backend calls**. The automatic score is keyword-overlap recall unless
`--judge-model` opts into an evaluator-model score (see [EVALUATION_LAB.md](EVALUATION_LAB.md)).

| Command | What it does |
|---|---|
| `eval-run <file.jsonl> <schema> --model … [--backend … --base-url … --judge-model … --judge-backend … --judge-base-url … --limit … --progress]` | Run one Evaluation Lab pass; `--progress` streams `[k/N] evaluated` to stderr. |
| `suite-init <name> [--project-dir …]` | Scaffold an example evaluation suite at `evaluation_suites/<name>.json`. |
| `suite-list [--project-dir …] [--json]` | List the registered evaluation suites. |
| `suite-run <name-or-path> [--project-dir …] [--strict] [--json]` | Run a suite (each case = a live eval + gate); rolls up **per metric**. `--strict` exits 2 on a block. |
| `suite-history <name> [--project-dir …]` | Show a suite's run history (oldest → newest) for pass/warn/block trending. |
| `benchmark <file.jsonl> <schema> --model … (repeatable)` | Benchmark one dataset across several models and rank them. |
| `backend-health --backend … --model … [--base-url …]` | Check whether a configured model backend is reachable + the model is available. |
| `model-list --backend … [--base-url …]` | List models available from a configured local backend. |

## AI Assist

| Command | What it does |
|---|---|
| `ai-assist <draft.jsonl> <schema> --action … --model … [--backend … --base-url …]` | Run the AI Assist Lab on draft rows; returns **review-only** suggestions (never auto-saved). |

## Training

The engine generates + inspects configs and records runs; it **never runs the trainer**
(the desktop launches your installed trainer). See [TRAINING.md](TRAINING.md).

| Command | What it does |
|---|---|
| `training-config <input.jsonl> <schema> --output-path … --base-model … [--target … --seed … --sequence-len … …]` | Generate an inspectable training config + token budget + VRAM estimate + **pre-flight** verdict. |
| `training-compat <schema> <format> <target>` | Report training-config compatibility warnings without generating a config. |
| `training-checkpoints <output_dir> [--target …]` | List checkpoints in an output dir and build a resume command. |
| `run-provenance <project_dir> <config_path>` | Build a run's reproducibility manifest (dataset fingerprint + rows, config SHA-256, engine/platform). |
| `training-run-list <project_dir>` | List durable training run records (newest first; reconciles dead `running` records). |
| `training-run-update <project_dir> --run-id … [--status … --after-eval-path … --after-eval-model …]` | Headless status / eval-link update with transition validation. |
| `training-run-gate <project_dir> --run-id …` | Regression-gate a run using its linked before/after eval reports. |
| `training-eval-plan <project_dir> --run-id … [--json]` | Close the train→eval loop: print the ordered serve → eval → link → gate steps for a finished run. |

## Model artifacts

| Command | What it does |
|---|---|
| `artifact-register <project_dir> --run-id … --path …` | Register (idempotently) a model artifact a run produced (referenced, never moved). |
| `artifact-list <project_dir>` | List artifacts (newest first) with computed path integrity. |
| `artifact-update <project_dir> --artifact-id … --status kept\|rejected` | Update an artifact's keep/reject status (keep is promote-gated). |
| `artifact-card <project_dir> --artifact-id …` | Render a weight card (live projection; nothing stored). |
| `artifact-gate <project_dir> --artifact-id …` | Promote-gate an artifact (integrity + source-run regression) and save it. |

## Gates & provider policy

| Command | What it does |
|---|---|
| `gate-run <examples.jsonl> <schema>` | Run gates (schema/quality/leakage/PII/eval) → serializable pass/warn/block report. |
| `chat-gate <examples.jsonl>` | Gate a chat dataset's conversation structure (advisory; verdict in the report). |
| `provenance-gate <examples.jsonl> [--strict] [--allow-teacher …]` | Per-row provenance: read each row's `meta.teacher` and quarantine rows generated by a restricted provider (e.g. Anthropic/OpenAI) that can't be trained on. Licensing counterpart to `provider-policy`; trusts the declared teacher. |
| `gate-thresholds <project_dir>` | Show the effective gate thresholds for a project. |
| `gate-thresholds-set <project_dir> --values-json …` | Validate + write a project's `gate_thresholds.json`. |
| `provider-policy [--project-dir …]` | Show effective provider role policies (with project overrides). |
| `provider-approve <provider> <model> [--revoke] …` | Approve (or revoke) **trainable generation** for a specific local model/route. |

## Export, preference & dataset card

| Command | What it does |
|---|---|
| `export <input.jsonl> <output> <schema> [--format jsonl\|csv\|tsv\|parquet] [--dedupe --drop-low-information --redact-pii --check-provenance]` | Validate + export (optionally clean / mask PII / enforce provenance). `--check-provenance` BLOCKs the export (exit 2) if any row's `meta.teacher` is a restricted provider (`--provenance-strict` also blocks unknown; `--allow-teacher` clears one). CSV/TSV = flat schemas; JSONL/Parquet = all schemas. |
| `preference-export <input.jsonl> <output> [--format dpo\|kto\|reward]` | Export preference rows into a trainer-ready format. |
| `dataset-card <project_dir> <schema>` | Build an inspectable dataset card from a project's existing artifacts. |

## Dataset versions (row-store)

| Command | What it does |
|---|---|
| `dataset-version-create <project_dir> …` | Capture a version: fingerprint + row count of `examples.jsonl` with pinned lineage. |
| `dataset-version-list <project_dir>` | List versions (newest first) with live integrity. |
| `dataset-version-show <project_dir> --version-id …` | Render a version card (live projection). |
| `dataset-version-diff <project_dir> --base-version-id … --other-version-id …` | Diff two versions by their stored row manifests (read-only). |
| `dataset-version-restore <project_dir> --version-id … --output …` | Reconstruct a version's exact rows from the row store (verified against the recorded fingerprint). |
| `dataset-version-gc <project_dir>` | Prune row-store rows no version references (fail-closed; never touches referenced rows). |

## Arena

| Command | What it does |
|---|---|
| `arena-run <prompts> --model … (repeatable) [--judge-model …]` | Run a prompt suite across several models and capture responses side by side. |

---

*Generated from the engine CLI (`corpus-studio --help`, 50 commands). Run any command with
`--help` for its complete, authoritative option list — that is always the source of truth.*
