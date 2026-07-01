# Local-First Design

Corpus Studio should work without cloud services.

## Local-first means

- user data stays on the machine by default
- no hidden upload behavior
- no account required for core features
- exports are normal files
- project data is inspectable

## Optional integrations

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
