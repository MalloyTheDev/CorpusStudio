# Import System

The import system turns existing files into dataset examples.

## Supported v0.1 imports

- JSONL
- JSON
- TXT
- Markdown

## Future imports

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
-> map fields
-> validate preview
-> import
-> quarantine failed rows
```

## Import principles

- Never destroy original files.
- Always show a preview.
- Always report failed rows.
- Keep source metadata when possible.
- Do not assume imported data is licensed for training.
