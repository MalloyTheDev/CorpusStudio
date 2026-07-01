# Data Model

## Project

```json
{
  "id": "coding_tutor_v1",
  "name": "Coding Tutor v1",
  "schema_id": "instruction",
  "created_at": "2026-06-30T00:00:00Z",
  "updated_at": "2026-06-30T00:00:00Z",
  "split_settings": {
    "train_ratio": 0.9,
    "validation_ratio": 0.05,
    "seed": 42
  },
  "lab_settings": {
    "evaluation": {
      "backend": "ollama",
      "model": "qwen2.5-coder:7b",
      "base_url": "http://localhost:11434",
      "timeout_seconds": 120
    },
    "ai_assist": {
      "backend": "ollama",
      "model": "qwen2.5-coder:7b",
      "base_url": "http://localhost:11434",
      "timeout_seconds": 120
    }
  }
}
```

## Dataset example

```json
{
  "id": "example_000001",
  "schema_id": "instruction",
  "fields": {
    "instruction": "Explain what a variable is.",
    "input": "",
    "output": "A variable is a named storage location."
  },
  "metadata": {
    "tags": ["programming", "beginner"],
    "source": "user_written",
    "license": "owned",
    "quality_score": null
  }
}
```

## Validation result

```json
{
  "example_id": "example_000001",
  "valid": true,
  "errors": [],
  "warnings": []
}
```

## Quality report

```json
{
  "dataset_id": "coding_tutor_v1",
  "example_count": 1000,
  "valid_count": 985,
  "invalid_count": 15,
  "duplicate_rate": 0.02,
  "average_tokens": 128
}
```

## Evaluation report run settings

Saved Evaluation reports include repeatable run settings so the desktop app can
rerun the same local configuration as a regression check. Reports also include
derived summaries by tag, failure reason, and score band so users can scan weak
areas without reading every row first.

```json
{
  "tag_summary": [
    {
      "tag": "recursion",
      "examples": 8,
      "failed_examples": 3,
      "average_score": 71.25
    }
  ],
  "failure_reason_summary": [
    {
      "reason": "score_below_threshold",
      "failed_examples": 5
    }
  ],
  "score_band_summary": [
    {
      "band": "70-84",
      "examples": 12,
      "failed_examples": 2,
      "average_score": 78.4
    }
  ],
  "run_settings": {
    "dataset_path": "data/projects/coding_tutor_v1/examples.jsonl",
    "schema_id": "instruction",
    "backend": "ollama",
    "base_url": "http://localhost:11434",
    "model": "qwen2.5-coder:7b",
    "limit": 10,
    "score_threshold": 70.0,
    "timeout_seconds": 120
  }
}
```

## Project-local artifacts

Project folders can also contain:

- `examples.jsonl`
- `drafts.jsonl`
- `quality_history.jsonl`
- `import_quarantine/`
- `ai_assist_reviews.jsonl`
- `ai_assist_queue_views.json`
- `ai_assist_rewrite_batches.json`
- `reviewed_fixes.json`
- `evaluation_failure_filters.json`

`ai_assist_rewrite_batches.json` stores prepared synthetic batch rewrite
handoffs only. Each item keeps the affected row numbers, issue count, source
draft, and AI Assist instruction so a user can resume the rewrite after app
restart. It is not accepted training data.

`reviewed_fixes.json` tracks failed evaluation rows that a user opened for
editing. Each record keeps the example id, saved row number, originating report,
original score, and a version number that increments on repeat edits of the same
example. After the next evaluation run the record is reconciled to `resolved` or
`still-failing`, giving an inspectable audit trail of which failures were
addressed. It is workflow state, not accepted training data.

`evaluation_failure_filters.json` stores named Evaluation drilldowns. Each saved
filter keeps a status, tag, failure-reason, and score-band selection so a
reviewer can reapply the same failure slice across runs. It is workflow state,
not accepted training data.

Export folders can contain:

- `export.jsonl`
- `splits/train.jsonl`
- `splits/validation.jsonl`
- `splits/test.jsonl`
- `evaluation/*.json`
- `preference_review/*.json`
- generated training config files
