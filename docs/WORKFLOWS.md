# Dataset Workflows

Short, hands-on walkthroughs for common dataset tasks. Consolidated from the
former `docs/workflows/` files.


---

## Workflow: Create an Instruction Dataset

1. Create a new project.
2. Select the instruction schema.
3. Write or import examples.
4. Validate required fields.
5. Add tags and metadata.
6. Run quality checks.
7. Split train/validation/test.
8. Optionally run an Evaluation Lab sample against a local model.
9. Prepare weak examples for AI Assist review when needed.
10. Export JSONL.

### Example row

```json
{"instruction":"Explain a for loop.","input":"","output":"A for loop repeats code for each item in a sequence."}
```


---

## Workflow: Create a Preference Dataset

1. Create a preference project.
2. Write the prompt.
3. Write the chosen response.
4. Write the rejected response.
5. Explain why chosen is better.
6. Validate the pair.
7. Use Preference Review to inspect chosen/rejected contrast.
8. Optionally prepare an AI Assist preference-strength judge pass.
9. Export preference JSONL or a preference ranking artifact.

### Example row

```json
{"prompt":"Explain recursion simply.","chosen":"Recursion is when a function calls itself.","rejected":"Recursion is a programming thing where stuff happens again."}
```


---

## Workflow: Create a Pretraining Corpus

1. Create a raw text project.
2. Import or paste JSONL raw-text rows.
3. Validate required text fields.
4. Remove empty rows.
5. Deduplicate and review low-information rows.
6. Add source/license metadata.
7. Split train/validation/test when useful.
8. Export raw text JSONL.

TXT, Markdown, code-folder, and richer corpus chunking imports remain planned.

### Example row

```json
{"text":"A game loop processes input, updates simulation, and renders frames."}
```


---

## Workflow: Export JSONL

1. Select dataset project.
2. Choose export format.
3. Choose split strategy.
4. Run validation.
5. Review warnings.
6. Export.
7. Optionally generate a Training Lab config from the clean export or split.
8. Open export folder.

### Export guarantee

The exporter should fail clearly if required fields are missing.
