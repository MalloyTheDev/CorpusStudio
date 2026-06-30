# Corpus Studio Roadmap

## v0.1 — Dataset Writing Studio

Goal: prove the local dataset authoring loop.

Features:

- local project creation
- built-in schemas:
  - raw text
  - instruction
  - chat/messages
  - preference
- example editor
- basic validation panel
- JSONL export
- train/validation/test split
- SQLite project storage
- basic quality dashboard

Exit criteria:

- user can create a dataset project
- user can add examples
- examples are validated against schema
- user can export valid JSONL
- user can split dataset into train/validation/test

## v0.2 — Cleaning and Quality Lab

Features:

- duplicate detection
- near-duplicate detection
- empty-field detection
- token length estimation
- quality report
- bad-row quarantine
- reversible cleaning operations

## v0.3 — Code Dataset Studio

Features:

- code schema
- syntax validation
- bug-fix pair editor
- code explanation editor
- test-case field support
- language metadata

## v0.4 — Image-Caption Dataset Studio

Features:

- image folder import
- image preview
- caption editor
- tag editor
- resolution metadata
- transparent-background metadata
- pixel-art workflow support

## v0.5 — Import and Transform System

Features:

- CSV import
- JSONL import
- Markdown import
- source folder import
- schema mapping UI
- import preview
- failed-row report

## v0.6 — Export Ecosystem

Features:

- Alpaca export
- ShareGPT export
- DPO export
- ChatML-like export
- Hugging Face folder export
- generated README and dataset card

## v0.7 — Local LLM Assist

Features:

- draft example generation
- output rewriting
- tag suggestions
- weak-example detection
- chosen/rejected quality feedback
- local model provider abstraction

## v0.8 — Evaluation Dataset Studio

Features:

- eval case editor
- expected-answer fields
- rubric fields
- category balancing
- regression test set support

## v0.9 — Training Config Generator

Features:

- Axolotl config templates
- TRL config templates
- Unsloth config templates
- llama.cpp preparation notes
- dataset format compatibility checks

## v1.0 — Full Local Dataset Studio

Features:

- stable schema engine
- stable project database
- stable export center
- version history
- full documentation
- examples for all built-in schemas
