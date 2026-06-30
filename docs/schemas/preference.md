# Preference Schema

Used for DPO, ORPO, reward modeling, and ranking datasets.

## Minimal JSONL

```json
{"prompt":"Explain recursion.","chosen":"A clear answer.","rejected":"A weak answer."}
```

## Required fields

- prompt
- chosen
- rejected

## Optional fields

- reason
- quality_dimensions
- tags
