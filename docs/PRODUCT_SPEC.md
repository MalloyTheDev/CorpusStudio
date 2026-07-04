# Corpus Studio Product Specification

This spec captures the **product identity** — who Corpus Studio is for, the problem
it solves, and the principles it holds to. For the authoritative list of **what is
built today** see [`CURRENT_STATE.md`](CURRENT_STATE.md); for the staged milestone
history see [`ROADMAP.md`](ROADMAP.md).

## Product identity

Corpus Studio is a local-first dataset creation studio for AI builders.

It combines:

- a writing application
- a schema-driven dataset editor
- a validation engine
- a cleaning lab
- a quality dashboard + graded debt ledger
- a split/export manager
- an Evaluation Lab (local models, optional LLM judge) + Model Arena
- a review-first AI Assist Lab
- a Training Lab (config export + in-app local launcher)
- dataset version history + a model-artifact registry

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

Long term, Corpus Studio should support the full dataset-to-model path:

```text
create dataset
-> validate and split
-> test with models
-> improve weak examples
-> export clean dataset
-> generate training config
-> launch local adapter training
-> compare checkpoints with evaluation runs
```

## Current Non-Goals

- training models directly (the app orchestrates the user's installed trainer)
- cloud collaboration
- automatic scraping
- bulk synthetic generation
- production-grade PII detection
- Hugging Face publishing / Hub import-export
- PDF OCR
- advanced multimodal annotation
- automatic acceptance of AI-generated dataset rows
- production-grade synthetic-pattern analysis

## Supported dataset types

Available today in both the desktop project-creation flow and the Python engine CLI:

1. raw text
2. instruction
3. chat/messages
4. preference pairs

Future dataset types: code, image-caption, classification, retrieval/embedding, evaluation.

## Core workflow

```text
Create project
-> choose schema
-> author examples
-> validate
-> export
```

The full shipped loop extends this through split/quality/debt, Evaluation Lab and Model
Arena, review-first AI Assist, training-config export and in-app launch, and dataset
version history — see [`CURRENT_STATE.md`](CURRENT_STATE.md) for the step-by-step feature
list and [`WORKFLOWS.md`](WORKFLOWS.md) for the end-to-end walkthrough.

## Product principles

1. Local-first by default.
2. User owns their data.
3. Dataset examples are first-class objects.
4. Schemas drive the editor.
5. Validation must be explicit.
6. Cleaning should be reversible or auditable.
7. Export formats must be deterministic.
8. Evaluation datasets are as important as training datasets.
9. Training stays downstream of dataset validation and evaluation, and orchestrates the
   user's installed tools — the app never embeds a training framework.

## Current state & roadmap

This spec intentionally does **not** duplicate the running feature list or milestone
board (they drift when copied). Instead:

- **What works today** (authoritative): [`CURRENT_STATE.md`](CURRENT_STATE.md).
- **Staged milestones v0.1 → v1.3** and their exit criteria: [`ROADMAP.md`](ROADMAP.md).
- **Initial build board** (historical, through v0.4): [`../TASKS.md`](../TASKS.md).
- Deep-dive references live under [`docs/`](.) (Evaluation Lab, AI Assist Lab, Gates,
  Provider Policy, Versioning, Training, Workspace System, and more).
