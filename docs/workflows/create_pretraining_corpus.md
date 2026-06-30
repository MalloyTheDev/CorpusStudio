# Workflow: Create a Pretraining Corpus

1. Create a raw text project.
2. Import TXT, Markdown, or JSONL sources.
3. Normalize text.
4. Chunk long documents.
5. Remove empty chunks.
6. Deduplicate.
7. Add source/license metadata.
8. Export raw text JSONL.

## Example row

```json
{"text":"A game loop processes input, updates simulation, and renders frames."}
```
