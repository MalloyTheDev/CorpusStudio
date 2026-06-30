# Import System

The import system turns existing files into dataset examples.

## Supported current imports

- JSONL through the desktop app and Python engine preview command

## Future imports

- JSON
- TXT
- Markdown
- CSV
- Parquet
- code folders
- image folders
- Git repositories
- Hugging Face datasets
- PDFs with OCR

## Import workflow

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
and shows failed row numbers with validation errors. Rows are imported only when
the preview has zero rejected rows.

## Import principles

- Never destroy original files.
- Always show a preview.
- Always report failed rows.
- Do not import partially valid files without an explicit recovery workflow.
- Keep source metadata when possible.
- Do not assume imported data is licensed for training.
