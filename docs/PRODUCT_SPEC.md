# Corpus Studio Product Specification

This spec captures the **product identity** — who Corpus Studio is for, the problem
it solves, and the principles it holds to. For the authoritative list of **what is
built today** see [`CURRENT_STATE.md`](CURRENT_STATE.md); for the staged milestone
history see [`ROADMAP.md`](ROADMAP.md).

## Product identity

Corpus Studio is a **local-first, end-to-end AI development ecosystem and IDE** for AI builders - a
dataset-to-model workspace covering the complete model lifecycle: data ingestion -> dataset engineering ->
validation & provenance -> model & tokenizer selection -> training -> checkpointing -> evaluation ->
behavior analysis & modification -> packaging -> export -> release -> reproducible evidence. It is **not** a
research platform, a training platform, an experiment runner, a dataset tool, or a fine-tuning app - those
are individual capabilities.

Its surface is organized into **seven co-equal product areas**: Data Studio, Training Studio, Evaluation
Studio, Behavior Lab, Model & Release Studio, Environment & Hardware, and Evidence & Experiments. Research
protocols, telemetry, and the IEEE paper are a **supporting track** (the Evidence & Experiments area) that
validates the product and documents discoveries - they do not define it. See
[`PRODUCT_AREAS.md`](PRODUCT_AREAS.md) for the canonical product map.

Today it combines:

- a writing application
- a schema-driven dataset editor
- a validation engine
- a cleaning lab
- a quality dashboard + graded debt ledger
- a split/export manager
- an Evaluation Studio (local models, optional LLM judge) + Model Arena
- a review-first AI Assist
- a Training Studio (external config/launcher + sealed first-party platform client)
- dataset version history + a model-artifact registry

The app exists to make the whole path from raw data to a released model - and the evidence that it
works - less fragile, less manual, and less scattered.

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
-> launch local training (pretraining, fine-tuning, or post-training)
-> compare checkpoints with evaluation runs
```

The training surface expands from adapter fine-tuning into a complete, pluggable **model-development
system** — from-scratch and continued pretraining, full-parameter and adapter/PEFT fine-tuning,
preference/RL post-training, distillation, dense and MoE architectures, and single-device or distributed
execution across multiple framework/orchestrator adapters. It stays **opt-in and evidence-gated**:
CorpusStudio composes a `TrainingPlan` from independent capability registries and dispatches a sealed plan
to an isolated worker (or exports a config for an external trainer); it never bundles a training framework
into the dependency-light control plane, and no capability is a default until it is workload-verified. See
[`TRAINING_SYSTEMS_ARCHITECTURE.md`](TRAINING_SYSTEMS_ARCHITECTURE.md).

## Current Non-Goals

- bundling CUDA/PyTorch or a training framework into the dependency-light control plane
  (training is **opt-in**: a sealed plan dispatched to the `[train]` worker, or your
  own installed external trainer)
- cloud collaboration
- automatic scraping
- bulk synthetic generation
- production-grade PII detection
- Hugging Face **publishing / Hub export** (upload). Read-only Hub *import* of
  public datasets is supported — see [`CURRENT_STATE.md`](CURRENT_STATE.md).
- PDF OCR
- advanced multimodal annotation
- automatic acceptance of AI-generated dataset rows
- production-grade synthetic-pattern analysis

## Supported dataset types

Available today in both the desktop project-creation flow and the Python engine CLI (nine built-in dataset schemas; a tenth built-in, `trace`, is a reasoning-trace draft-authoring schema):

1. raw text
2. instruction
3. chat/messages
4. preference pairs
5. code
6. image-caption
7. classification
8. retrieval / embedding
9. evaluation

## Core workflow

```text
Create project
-> choose schema
-> author examples
-> validate
-> export
```

The full shipped loop extends this through split/quality/debt, Evaluation Studio and Model
Arena, review-first AI Assist, external training-config launch and sealed first-party platform
dispatch, and dataset
version history — see [`CURRENT_STATE.md`](CURRENT_STATE.md) for the step-by-step feature
list and [`WORKFLOWS.md`](WORKFLOWS.md) for dataset-task walkthroughs.

## Product principles

1. Local-first by default.
2. User owns their data.
3. Dataset examples are first-class objects.
4. Schemas drive the editor.
5. Validation must be explicit.
6. Cleaning should be reversible or auditable.
7. Export formats must be deterministic.
8. Evaluation datasets are as important as training datasets.
9. Training stays downstream of dataset validation and evaluation; first-party work runs only from a
   sealed plan through the opt-in `[train]` worker, while external tools use separately reviewed argv.
   The dependency-light control plane never bundles a training framework.

## Current state & roadmap

This spec intentionally does **not** duplicate the running feature list or milestone
board (they drift when copied). Instead:

- **What works today** (authoritative): [`CURRENT_STATE.md`](CURRENT_STATE.md).
- **Staged milestones v0.1 → v1.3** and their exit criteria: [`ROADMAP.md`](ROADMAP.md).
- **Initial build board** (historical, through v0.4): [`../TASKS.md`](../TASKS.md).
- Deep-dive references live under [`docs/`](.) (Evaluation Studio, AI Assist, Gates,
  Provider Policy, Versioning, Training, Workspace System, and more).

---

## Local-First Design

_Consolidated from the former `docs/LOCAL_FIRST.md`._

Corpus Studio should work without cloud services.

### Local-first means

- user data stays on the machine by default
- no hidden upload behavior
- no account required for core features
- exports are normal files
- project data is inspectable

### Optional integrations

Current local integrations include:

- local LLM providers

Future optional integrations may include:

- Hugging Face publishing
- cloud storage
- team collaboration

These should be optional.

Local model calls are explicit user actions. Corpus Studio should never upload
datasets, call hosted providers, or start training jobs without clear user
configuration and action.
