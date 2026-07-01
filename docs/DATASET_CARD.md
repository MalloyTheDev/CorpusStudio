# Dataset Cards

A dataset card is a single, inspectable summary of a Corpus Studio project. It
aggregates artifacts the studio already produces — nothing is recomputed by a
model and nothing is published automatically.

## What a card contains

- **Overview**: project id, schema (name, id, version), example count, and the
  project's created/updated timestamps plus the card generation time.
- **Schema fields**: each field's name, type, required flag, and description.
- **Quality summary**: example count, empty rows, exact and near duplicates,
  low-information rows, and synthetic-pattern issue count from the basic quality
  report.
- **Splits**: train/validation/test row counts when splits have been generated.
- **Evaluation**: the most recent saved evaluation report's model, examples
  tested, average score, failed examples, and weak tags.
- **Notes & warnings**: missing splits, missing evaluation runs, and quality
  issues that a reviewer should resolve before exporting or training.

## How it is produced

The engine builds the card by reading existing files only:

- `project.json` and `examples.jsonl` inside the project directory
- `splits/{train,validation,test}.jsonl` under the project's export directory
- the newest `evaluation/*_evaluation_report.json` under the export directory

### CLI

```bash
corpus-studio dataset-card <project_dir> \
  --output-path exports/<project_id>/dataset_card.md \
  --export-dir exports/<project_id>
```

The command prints a JSON payload with the rendered `markdown`, the aggregated
`card` object, the resolved `output_path`, and any `warnings`. When
`--export-dir` is omitted it defaults to `CORPUS_STUDIO_EXPORT_DIR` (or
`exports/`) plus the project id. When `--schema` is omitted it uses the schema
recorded in `project.json`.

### Desktop

The Export section of the desktop app has a **Generate Dataset Card** button. It
writes `exports/<project_id>/dataset_card.md` and shows the rendered card and any
warnings in place.

## Out of scope

- No dataset upload or publishing.
- No model-generated prose in the card body.
- No mutation of project files while building the card.
