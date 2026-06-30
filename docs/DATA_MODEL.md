# Data Model

## Project

```json
{
  "id": "coding_tutor_v1",
  "name": "Coding Tutor v1",
  "schema_id": "instruction",
  "created_at": "2026-06-30T00:00:00Z",
  "updated_at": "2026-06-30T00:00:00Z"
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
