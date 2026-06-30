# Corpus Studio Product Specification

## Product identity

Corpus Studio is a local-first dataset creation studio for AI builders.

It combines:

- a writing application
- a schema-driven dataset editor
- a validation engine
- a cleaning lab
- a quality dashboard
- a split/export manager

The app exists to make training-data creation less fragile, less manual, and less scattered.

## Target users

Primary users:

- independent AI toolmakers
- model fine-tuners
- game developers building AI-assisted tools
- researchers preparing local datasets
- small teams building domain-specific assistants
- people creating code, image-caption, chat, or preference datasets

## Core problem

Dataset creation is usually fragmented:

```text
notes -> scripts -> spreadsheets -> JSONL -> training configs -> fixes by hand
```

This creates broken rows, inconsistent schemas, duplicate examples, data leakage, poor provenance, and painful exports.

## Product promise

Corpus Studio lets a user move from idea to model-ready dataset inside one local application.

## Non-goals for v0.1

- training models directly
- cloud collaboration
- automatic scraping
- bulk synthetic generation
- production-grade PII detection
- Hugging Face publishing
- PDF OCR
- advanced multimodal annotation

## v0.1 supported dataset types

1. raw text
2. instruction
3. chat/messages
4. preference pairs

These are available in the desktop project creation flow and in the Python engine CLI.

## Future dataset types

5. code
6. image-caption
7. classification
8. retrieval/embedding
9. evaluation

## Core workflow

```text
Create project
-> choose schema
-> author examples
-> validate
-> export
```

## v0.1 working loop

The current app proves the smallest local dataset-authoring loop:

1. Launch the WPF desktop app.
2. Create a local project under `data/projects`.
3. Choose a built-in schema from the engine.
4. Edit the generated JSON example in Writing Studio.
5. Validate the draft through the Python engine.
6. Save the example to the active project's `examples.jsonl`.
7. Inspect saved example details from the Examples tab.
8. Reopen an existing project from the project list.
9. Export validated JSONL to `exports/<project_id>/export.jsonl`.
10. Inspect local repository, engine, Python, project, and export paths from Settings.

The Python engine also exposes schema listing, validation, project creation, quality reporting, splitting, and export commands for developer workflows.

## v0.1 constraints

- Project data is file-backed JSON and JSONL.
- The desktop app writes one JSON object per saved example.
- Validation currently enforces JSON validity plus required non-empty fields.
- The Quality and Splits desktop tabs are placeholders for future richer workflows.
- SQLite remains planned for durable project state beyond the v0.1 file-backed loop.

## Product principles

1. Local-first by default.
2. User owns their data.
3. Dataset examples are first-class objects.
4. Schemas drive the editor.
5. Validation must be explicit.
6. Cleaning should be reversible or auditable.
7. Export formats must be deterministic.
8. Evaluation datasets are as important as training datasets.

## v0.2 priority direction

The next product slice should turn saved examples into a stronger review loop:

1. show saved example details in the Examples tab
2. expose quality checks in the desktop app
3. add split generation to the desktop app
4. make validation stricter for schema field types
5. add import preview and failed-row reporting
