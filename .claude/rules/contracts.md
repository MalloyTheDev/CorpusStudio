---
paths:
  - "engine/corpus_studio/platform/contracts.py"
  - "engine/corpus_studio/platform/enums.py"
  - "docs/contracts/**"
---

# Contracts are the boundary

Editing `platform/contracts.py` (pydantic) means regenerating the derived surfaces in the same change,
or CI fails on the drift diff:

1. Regenerate the JSON Schemas:
   `python -c "from corpus_studio.platform.schema_export import export_json_schemas; export_json_schemas('../docs/contracts')"`
2. Regenerate the TypeScript types: `cd apps/web && npm run gen:contracts`.
3. If you added or removed a `ROOT_CONTRACTS` entry, update the two counts in
   `tests/test_platform_contracts.py` (the root-contract count and the schema-writer count).

`docs/contracts/*.schema.json` and the TS types are committed; CI diffs the regenerated output, and a
trigger-independent backstop test byte-compares them. No new foundational contract may assume dense
execution - keep `ModelDescriptor` / `TrainingObjective` / `RunPlan` / `ArtifactManifest` / checkpoint /
telemetry / evaluation dense-safe and MoE-compatible.
