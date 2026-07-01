# Export Formats

Corpus Studio should support deterministic exports.

## v0.1 exports

### Generic JSONL

One JSON object per line.

```json
{"instruction":"Explain variables.","input":"","output":"A variable stores a value."}
```

### Raw text JSONL

```json
{"text":"A compiler translates source code into machine instructions."}
```

### Chat messages JSONL

```json
{"messages":[{"role":"user","content":"What is recursion?"},{"role":"assistant","content":"Recursion is when a function calls itself."}]}
```

### Preference JSONL

```json
{"prompt":"Explain recursion simply.","chosen":"Recursion is when a function calls itself.","rejected":"Recursion is when code does things again."}
```

## Planned dataset exports

- CSV
- Parquet
- Alpaca
- ShareGPT
- ChatML-like
- DPO
- Hugging Face dataset folder
- dataset card
- richer target-specific training config templates

Training Lab now has a first-pass config export path, but dataset format
exports and full target-specific compatibility checks remain staged work.

## Export rule

Exports must never silently drop fields. If a target format cannot support a field, the exporter must warn the user.
