# Handoff Prompt for Claude/Codex

You are working inside the Corpus Studio repository.

Corpus Studio is a local-first dataset creation studio for AI builders. It is MIT-licensed.

Your job is to help implement the v0.1 skeleton into a working application while preserving the architecture.

## Product goal

Build a one-stop shop for creating AI training datasets across multiple schemas.

The app should eventually support:

- raw pretraining corpora
- instruction tuning
- chat/message datasets
- preference/DPO datasets
- code datasets
- image-caption datasets
- classification datasets
- retrieval/embedding datasets
- evaluation datasets

## v0.1 scope

Implement only:

- raw text
- instruction
- chat/messages
- preference

Core v0.1 flow:

```text
Create project
-> choose schema
-> author/import examples
-> validate
-> inspect quality
-> split train/validation/test
-> export JSONL
```

## Architecture constraints

- Keep the app local-first.
- Keep schemas first-class.
- Use the Python engine for dataset logic.
- Use the C# desktop app for UX.
- Do not hardcode every dataset type directly into UI behavior.
- Do not silently mutate user data.
- Exports must be deterministic.
- Validation errors must be specific and actionable.

## First implementation targets

1. Make the Python engine tests pass.
2. Add a project creation CLI.
3. Add split command.
4. Add export command with schema validation.
5. Create the desktop solution.
6. Wire the desktop Validate button to the Python engine.
7. Add a New Project dialog.
8. Add schema picker using built-in schema JSON files.
9. Add a simple example list/editor.
10. Add JSONL export.

Read these files first:

- README.md
- docs/PRODUCT_SPEC.md
- docs/ARCHITECTURE.md
- docs/SCHEMA_SYSTEM.md
- docs/ROADMAP.md
- docs/QUALITY_GATES.md
