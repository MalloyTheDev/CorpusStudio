# Corpus Studio

**Corpus Studio** is a local-first dataset creation studio for AI builders.

It is designed to be a one-stop shop for authoring, importing, cleaning, validating, splitting, versioning, and exporting model-ready datasets across multiple schemas:

- raw pretraining corpora
- instruction-tuning datasets
- chat/message datasets
- preference/DPO datasets
- code datasets
- image-caption datasets
- classification datasets
- retrieval/embedding datasets
- evaluation datasets

Corpus Studio is not just a JSONL editor. It is a writing-first dataset IDE.
The long-term direction is a full dataset-to-model workflow: create datasets,
validate them, test them against models, export them, and eventually generate
training configs and launch local adapter jobs.

## Current Status

Corpus Studio now has a working local dataset loop and first-pass lab surfaces:

- create a dataset project from the WPF desktop app
- choose a built-in schema
- author a JSON example
- validate it through the Python engine with selectable issue navigation
- save it to the local project
- preview and import JSONL rows with failed-row reporting
- review quarantined import rows and retry them in the editor
- run quality checks for empty rows, duplicates, and low-information examples
- record project-level quality history
- triage synthetic-pattern quality issues into a prepared AI Assist rewrite flow
- generate train/validation/test split files with saved ratios, seed, and tiny-split warnings
- export validated JSONL
- run Evaluation Lab samples against configured Ollama or OpenAI-compatible
  local endpoints
- check backend health before model-backed runs
- refresh local model pickers from running backends
- reload saved evaluation reports
- compare two saved evaluation reports for score, failure, weak-tag, and
  row-level deltas
- rerun saved evaluation configurations as regression checks using stored
  backend, model, limit, score threshold, timeout, and schema settings
- inspect Evaluation summaries by tag, failure reason, and score band
- review failed evaluation examples, load failed rows back into Writing Studio
  for explicit edits, add manual scores/notes, and prepare AI Assist rewrite
  triage
- run review-first AI Assist passes without automatically accepting generated
  rows
- manage an AI Assist review queue with filters, search, sorting, saved views,
  bulk triage, and undo
- review preference-pair contrast and prepare preference judge passes
- export inspectable Training Lab config files without launching trainers

The next phase is hardening these surfaces into a dependable v0.2/v0.3
workflow. Corpus Studio still does not launch local training jobs, manage CUDA
or PyTorch, run checkpoints, or publish datasets automatically.

## License

MIT. See [`LICENSE`](LICENSE).

## Product principle

Every dataset example should be:

- valid
- inspectable
- traceable
- exportable
- versioned

## Repository Layout

```text
CorpusStudio
├── apps/
│   └── desktop/             # C# WPF desktop app
├── engine/                  # Python dataset engine
├── schemas/                 # Built-in schema definitions
├── docs/                    # Product, architecture, roadmap, workflows
├── examples/                # Example dataset rows
├── scripts/                 # Developer scripts
├── data/                    # Local project data, ignored by git
└── exports/                 # Exported datasets, ignored by git
```

## Desktop preview

![Corpus Studio desktop v0.1](docs/screenshots/desktop-v0.1.png)

## Core Local Loop

Build a local desktop app that supports:

1. project creation
2. built-in schema templates
3. raw text, instruction, chat, and preference datasets
4. example authoring
5. schema validation
6. quality checks
7. train/validation/test split generation
8. JSONL export

## Development notes

The recommended stack is:

- C# WPF / WinUI-style desktop front-end
- Python dataset engine
- file-backed project state today
- SQLite later for indexing and larger project state
- JSONL as the first export target
- Pydantic for schema validation
- Polars / DuckDB later for large datasets when needed

See [`docs/PRODUCT_SPEC.md`](docs/PRODUCT_SPEC.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

For hands-on setup, see [`docs/DEVELOPMENT_SETUP.md`](docs/DEVELOPMENT_SETUP.md).
For copyable row formats, see [`docs/SCHEMA_EXAMPLES.md`](docs/SCHEMA_EXAMPLES.md).
For the staged labs, see [`docs/EVALUATION_LAB.md`](docs/EVALUATION_LAB.md),
[`docs/AI_ASSIST_LAB.md`](docs/AI_ASSIST_LAB.md), and
[`docs/TRAINING_LAB.md`](docs/TRAINING_LAB.md).
