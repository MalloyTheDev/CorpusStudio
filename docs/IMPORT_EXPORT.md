# Import & Export

How datasets get into and out of Corpus Studio. Consolidated from the former
IMPORT_SYSTEM and EXPORT_FORMATS docs.


---

## Import System

The import system turns existing files into dataset examples.

### Supported current imports

- JSONL through the desktop app and Python engine preview command
- **Hugging Face Hub datasets** (read-only, public) via the engine
  `hf-inspect` / `hf-import` commands — see below

### Future imports

- JSON
- TXT
- Markdown
- CSV
- Parquet
- code folders
- image folders
- Git repositories
- PDFs with OCR

### Import workflow

```text
Select source
-> preview files
-> choose target schema
-> validate preview
-> import
-> quarantine failed rows
```

The current desktop import flow accepts a JSONL file, previews every non-empty
row against the active project schema, reports accepted and rejected row counts,
and shows failed row numbers with validation errors. When a file contains both
valid and rejected rows, the user can explicitly import the valid rows and save
the rejected rows to `import_quarantine` for repair. The desktop app can review
quarantined rows and retry a selected raw row by loading it back into Writing
Studio.

### Import principles

- Never destroy original files.
- Always show a preview.
- Always report failed rows.
- Do not import partially valid files without explicit user confirmation.
- Keep rejected rows recoverable in a quarantine report.
- Keep source metadata when possible.
- Do not assume imported data is licensed for training.

### Hugging Face Hub import (read-only)

Import rows from a **public** Hugging Face dataset without pulling in the
`datasets` / `huggingface_hub` libraries — the engine calls the public
**datasets-server** JSON API with the standard library only. The desktop surfaces
this as an **Import from Hugging Face** dialog (Studio sidebar) that inspects the
dataset, shows its license, lets you map columns to the project schema, and runs
the staged rows through the normal import preview; the CLI below is the same path.

- `hf-inspect <dataset_id>` lists the dataset's configs/splits, sample columns,
  and **license**, so you can decide whether the data may be used for training
  and how its columns map to a Corpus Studio schema.
- `hf-import <dataset_id> --schema <id> --out <staging.jsonl> [--config --split
  --limit --map field=column …]` fetches rows, maps columns to schema fields
  (exact-name matches auto-detected; `--map` overrides), and writes a **staging
  JSONL** file. It reports the mapping, any unmapped fields / unused columns, and
  the license with the caveat above.

This stays inside the project's boundaries:

- **Read-only and public-only** — no auth, no upload, no publishing. The engine
  makes a Hub network call only when you explicitly run these commands.
- **The engine never writes `examples.jsonl`.** `hf-import` writes a *staging*
  file (and refuses to target `examples.jsonl`); that file flows through the
  normal import-preview → validate → quarantine path, where the **desktop** is
  the single writer that appends accepted rows.
- **Gated / private datasets are refused** (they need auth — out of scope here).

Pushing datasets *to* the Hub (export/publish) is deliberately **not** supported —
see the non-goals in `PRODUCT_SPEC.md`.


---

## Export Formats

Corpus Studio should support deterministic exports.

### v0.1 exports

#### Generic JSONL

One JSON object per line.

```json
{"instruction":"Explain variables.","input":"","output":"A variable stores a value."}
```

#### Raw text JSONL

```json
{"text":"A compiler translates source code into machine instructions."}
```

#### Chat messages JSONL

```json
{"messages":[{"role":"user","content":"What is recursion?"},{"role":"assistant","content":"Recursion is when a function calls itself."}]}
```

#### Preference JSONL

```json
{"prompt":"Explain recursion simply.","chosen":"Recursion is when a function calls itself.","rejected":"Recursion is when code does things again."}
```

### Planned dataset exports

Shipped: preference exports (DPO/KTO/reward), the dataset card, and training
config templates for the major targets (Axolotl / TRL / Unsloth / HF /
LLaMA-Factory). Still planned:

- CSV
- Parquet
- Alpaca
- ShareGPT
- ChatML-like
- Hugging Face dataset folder

Full target-specific compatibility checks beyond the current config warnings
remain staged work.

### Export rule

Exports must never silently drop fields. If a target format cannot support a field, the exporter must warn the user.
