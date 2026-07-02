# Gates

A **gate** decides whether a dataset, split, export, or evaluation report may
move forward. Gates reuse the existing validation, quality, leakage, PII, and
evaluation logic — they only aggregate results against thresholds and produce a
serializable, project-local report. No gate adds new detection logic.

## Model

- **Scopes**: `dataset`, `row`, `import`, `export`, `split`,
  `evaluation_report`, `training_run`, `model_artifact`, `chat_suite`.
- **Status**: `pass`, `warn`, `block`.
- **`GateResult`** fields: `gate_id`, `name`, `scope`, `status`, `observed`,
  `expected`, `affected`, `message`, `repair`.
- **`GateReport`**: `scope`, `target`, `generated_at`, `overall_status`
  (= worst result), pass/warn/block counts, and `results`. Serializes to JSON
  and reloads via Pydantic.

## Initial gates (wired to existing logic)

| Gate | Reuses | Default behavior |
|---|---|---|
| **schema** | `validate_jsonl_row` | **block** if any row fails validation. |
| **quality** | `build_basic_quality_report` | **block** on exact duplicates; **warn** on near-duplicates, low-information, or synthetic-pattern issues. |
| **leakage** | `detect_split_leakage` | **block** if any row is shared across train/validation/test. |
| **pii** | quality report `pii_findings` | **block** on high-severity (keys/tokens/JWT); **warn** on medium (email/SSN). |
| **eval_score** | `EvaluationReport` | **block** below the average-score or pass-rate threshold. |

The **export gate** is a composite: it **blocks** on empty input, schema, or PII
failure, and **warns** on quality issues (duplicates/low-information) because the
export command has a dedicated cleaning pass. An `input_present` gate ensures an
empty dataset can never pass silently (warn for `dataset` scope, block for
`export`). Thresholds ship as sensible defaults in `GateThresholds` and are
designed for future per-project configuration.

## Running gates

```
python -m corpus_studio.cli gate-run dataset.jsonl instruction --scope dataset \
  --project-dir path/to/project
```

Writes `gate_reports/<scope>-<target>.json` under the project (the target is in
the filename so gating different files in one scope does not clobber earlier
reports) and echoes the report.
`--scope export` runs the export gate. Split and evaluation gates are available
through the engine API (`run_split_gate`, `run_evaluation_gate`).

## Future work

A **regression gate** (trained vs base model) depends on the before/after run
registry (v0.8) and is intentionally not implemented in v0.6. Per-project
threshold configuration and desktop surfacing of gate reports are follow-ups.
