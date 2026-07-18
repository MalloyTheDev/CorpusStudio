# Import & Export

How datasets get into and out of Corpus Studio. Consolidated from the former
IMPORT_SYSTEM and EXPORT_FORMATS docs.


---

## Import System

The import system turns existing files into dataset examples.

### Supported current imports

- JSONL through the desktop app and Python engine preview command
- **CSV / TSV** through the desktop **Import Dataset** picker and the engine
  `import-convert` command — see below
- **Parquet** via `import-convert <file.parquet> <output>` (optional `[parquet]`
  extra) — see below
- **Hugging Face Hub datasets** (read-only, public) via the engine
  `hf-inspect` / `hf-import` commands — see below

### Future imports

- JSON
- TXT
- Markdown
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

The current desktop import flow accepts a JSONL, CSV, or TSV file, previews every
non-empty row against the active project schema, reports accepted and rejected row
counts, and shows failed row numbers with validation errors. When a file contains
both valid and rejected rows, the user can explicitly import the valid rows and save
the rejected rows to `import_quarantine` for repair. The desktop app can review
quarantined rows and retry a selected raw row by loading it back into Writing
Studio.

### CSV / TSV import

Corpus Studio stays **JSONL-canonical** — it authors, validates, gates, and exports
JSONL. Tabular files are supported at the import boundary by converting them to a
staging JSONL that then flows through the *exact same* preview → quarantine → commit
path as any JSONL (or Hugging Face) import, so there is one validation/commit path,
not a second tabular one. The desktop **Import Dataset** picker accepts `.csv`,
`.tsv`, and `.jsonl`; a CSV/TSV pick is converted to a temp staging JSONL first (the
engine `import-convert <path> <output>` command), then previewed and imported.

The header row defines the field keys and the delimiter is chosen by extension
(`.tsv`/`.tab` → tab, else comma; a UTF-8 BOM from Excel/Windows exports is
tolerated). **Honesty boundary:** a CSV cell has no type, so every value is imported
as text. A row whose schema field expects a number/list/object will therefore fail
the normal import-preview validation and land in **quarantine** for repair — exactly
like a malformed JSONL row. The converter never coerces types or drops columns; the
original file is never modified. (Excel remains future work; CSV/TSV use the standard
library only.)

### Parquet import

Parquet is a typed, columnar format, so unlike CSV a Parquet integer/list/struct
column keeps its type into the staging JSONL and validates against the schema
directly (no forced-string quarantine). `import-convert <file.parquet> <output>` is
routed by the `.parquet` extension, streams the file per record-batch (bounded
memory), and takes the columns from the Parquet schema. Because Parquet needs the
large `pyarrow` dependency, it is an **optional extra** — the engine stays
dependency-light by default and, without it installed, every Parquet path fails fast
with `pip install corpus-studio-engine[parquet]` instead of an opaque error. Binary
columns are best-effort decoded as text (image/audio payloads are out of scope for this
JSONL/Parquet import path).

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
- **`hf-import` writes a *staging* file** (and refuses to target `examples.jsonl`);
  that file flows through the normal import-preview → validate → quarantine path.
  Accepted rows are appended to `examples.jsonl` by the engine's sanctioned
  single-writer, `examples-append` (atomic, locked, schema-validated).
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

#### CSV / TSV (flat schemas)

JSONL is the **canonical, model-ready** export — it works for every schema and is
what trainers consume. CSV/TSV is an opt-in convenience for **flat** schemas
(classification, raw text, simple instruction, evaluation, retrieval, preference,
…): open the cleaned set in Excel/pandas or share it. Select the format in the
desktop Export panel, or pass `export … --format csv` / `--format tsv` on the CLI.

The header row is the schema's fields (then any extra row keys, so nothing a row
carries is dropped); a scalar list field like `tags` is written as a `"; "`-joined
cell. CSV/TSV goes through the *same* validate → PII/secret export-gate → optional
clean/redact pipeline as JSONL — only the writer differs. A schema with a chat
`messages` field, a nested `object`, or a list of objects is **refused** (exit 2)
with a clear "export as JSONL instead" message rather than lossy-flattened, honoring
the export rule below. (Excel remains future work; CSV/TSV use the standard library
only.)

```text
text,label,tags
Great value for the price.,positive,sentiment; review
```

#### Parquet (all schemas)

Parquet is columnar and represents nested types natively, so — unlike CSV/TSV — it
supports **every** schema, chat and nested objects included (a `messages` array
becomes a `list<struct>` column, not a refused field). It is an analytics-friendly
deliverable (pandas / DuckDB / Spark) while JSONL stays the canonical, diffable,
trainer-ready format. Pass `export … --format parquet` (or select it in the desktop
Export panel); it shares the same validate → PII/secret export-gate → optional
clean/redact pipeline — only the writer differs. Parquet needs the optional
`[parquet]` extra; without `pyarrow` installed the export fails fast (before any
output is written) with `pip install corpus-studio-engine[parquet]`.

### Planned dataset exports

Shipped: preference exports (DPO/KTO/reward), CSV/TSV (flat schemas), **Parquet (all
schemas)**, the dataset card, and training config templates for the major targets
(Axolotl / TRL / Unsloth / HF / LLaMA-Factory). Still planned:

- Alpaca
- ShareGPT
- ChatML-like
- Hugging Face dataset folder

Full target-specific compatibility checks beyond the current config warnings
remain staged work.

### Export rule

Exports must never silently drop fields. If a target format cannot support a field, the exporter must warn the user.

### Provenance enforcement (opt-in)

`export … --check-provenance` runs the per-row provenance gate before writing the
deliverable and **refuses the export (exit 2)** if any row's declared `meta.teacher`
is a provider whose terms forbid training on its outputs (e.g. Anthropic/OpenAI) —
the licensing counterpart to the PII export gate. Rows with unknown provenance
*warn* by default; `--provenance-strict` blocks them too. A teacher you know is
trainable-clean is cleared with `--allow-teacher <name>` or a
`provenance_allowlist.json` sitting next to the input. Honest scope: it trusts each
row's **declared** teacher — a mislabeled/omitted teacher is not caught by content,
and unknown provenance is quarantine-until-verified, never assumed safe.
