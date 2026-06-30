# Architecture

Corpus Studio is split into two major layers:

1. Desktop application
2. Dataset engine

The desktop app owns user interaction.
The Python engine owns dataset logic.

## System diagram

```text
Desktop UI
  |
  | calls local engine service / CLI / IPC
  v
Dataset Engine
  |
  +-- schemas
  +-- validators
  +-- importers
  +-- cleaners
  +-- splitters
  +-- exporters
  +-- quality analyzers
  +-- storage adapters
```

## Desktop layer

The desktop layer is responsible for:

- project dashboard
- schema selection
- editing examples
- showing validation results
- showing quality reports
- triggering imports/exports
- showing split/export status

## Engine layer

The engine is responsible for:

- loading dataset projects
- validating examples
- converting between schemas
- cleaning rows
- estimating tokens
- detecting duplicates
- splitting datasets
- exporting files
- producing quality reports

## Storage model

v0.1 storage should be simple:

```text
data/
└── projects/
    └── my_dataset/
        ├── project.json
        ├── examples.jsonl
        ├── drafts.jsonl
        └── exports/
```

SQLite can be added early for indexing and UI speed.

## IPC options

Possible app-to-engine communication:

1. run Python engine as a CLI
2. run Python engine as a local HTTP service
3. embed Python later
4. use file-based project exchange for v0.1

Recommended v0.1 path:

```text
Desktop writes project files -> Python CLI validates/exports -> Desktop reads results
```

This is simple, debuggable, and avoids premature complexity.

## Design rule

The UI should not hardcode dataset behavior. The schema system should define:

- fields
- required fields
- editor hints
- validation rules
- export mappings
