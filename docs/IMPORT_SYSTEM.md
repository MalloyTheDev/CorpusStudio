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
and shows failed row numbers with validation errors. When a file contains both
valid and rejected rows, the user can explicitly import the valid rows and save
the rejected rows to `import_quarantine` for repair. The desktop app can review
quarantined rows and retry a selected raw row by loading it back into Writing
Studio.

## Import principles

- Never destroy original files.
- Always show a preview.
- Always report failed rows.
- Do not import partially valid files without explicit user confirmation.
- Keep rejected rows recoverable in a quarantine report.
- Keep source metadata when possible.
- Do not assume imported data is licensed for training.
