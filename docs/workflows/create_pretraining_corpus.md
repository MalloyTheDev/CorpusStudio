# Workflow: Create a Pretraining Corpus

1. Create a raw text project.
2. Import or paste JSONL raw-text rows.
3. Validate required text fields.
4. Remove empty rows.
5. Deduplicate and review low-information rows.
6. Add source/license metadata.
7. Split train/validation/test when useful.
8. Export raw text JSONL.

TXT, Markdown, code-folder, and richer corpus chunking imports remain planned.

## Example row

```json
{"text":"A game loop processes input, updates simulation, and renders frames."}
```
