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
-> author/import examples
-> validate
-> inspect quality
-> split
-> export
```

## Product principles

1. Local-first by default.
2. User owns their data.
3. Dataset examples are first-class objects.
4. Schemas drive the editor.
5. Validation must be explicit.
6. Cleaning should be reversible or auditable.
7. Export formats must be deterministic.
8. Evaluation datasets are as important as training datasets.
