from pathlib import Path
from typing import Optional

import typer

from corpus_studio.exporters.jsonl_exporter import export_jsonl
from corpus_studio.validators.basic_validator import validate_jsonl_file

app = typer.Typer(help="Corpus Studio dataset engine CLI.")


@app.command()
def validate(path: Path, schema: str):
    """Validate a JSONL file against a built-in schema."""
    result = validate_jsonl_file(path, schema)
    typer.echo(result.model_dump_json(indent=2))


@app.command()
def export(input_path: Path, output_path: Path, schema: Optional[str] = None):
    """Export a JSONL file. Placeholder for future format conversion."""
    export_jsonl(input_path, output_path)
    typer.echo(f"Exported {input_path} -> {output_path}")


if __name__ == "__main__":
    app()
