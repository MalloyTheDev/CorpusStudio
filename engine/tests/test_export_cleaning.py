import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.exporters.cleaning import clean_rows

runner = CliRunner()


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_clean_rows_drops_exact_and_normalized_duplicates():
    rows = [
        {"instruction": "Explain x.", "output": "A value."},
        {"instruction": "Explain x.", "output": "A value."},  # exact dup
        {"instruction": "explain X.", "output": "a value."},  # normalized dup
        {"instruction": "Explain y.", "output": "Another value."},
    ]
    kept, result = clean_rows(rows, dedupe=True)
    assert result.kept_rows == 2
    assert result.removed_exact_duplicates == 1
    assert result.removed_normalized_duplicates == 1
    assert len(kept) == 2


def test_clean_rows_drops_low_information():
    rows = [
        {"instruction": "Explain recursion in detail.", "output": "It is a function calling itself repeatedly."},
        {"instruction": "Hi", "output": "Ok"},  # low information
    ]
    kept, result = clean_rows(rows, drop_low_information=True)
    assert result.removed_low_information == 1
    assert result.kept_rows == 1


def test_clean_rows_no_flags_keeps_everything():
    rows = [{"instruction": "a", "output": "b"}, {"instruction": "a", "output": "b"}]
    kept, result = clean_rows(rows)
    assert result.kept_rows == 2
    assert result.removed_rows == 0


def test_cli_export_without_cleaning_warns_about_duplicates(tmp_path: Path):
    input_path = tmp_path / "rows.jsonl"
    row = {"instruction": "Explain variables.", "output": "A variable stores a value."}
    _write(input_path, [row, row])
    output_path = tmp_path / "export.jsonl"

    result = runner.invoke(app, ["export", str(input_path), str(output_path), "instruction"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["cleaned"] is False
    assert any("duplicate" in warning.lower() for warning in payload["warnings"])
    # Verbatim copy keeps both rows.
    lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2


def test_cli_export_with_dedupe_removes_duplicates_and_writes_manifest(tmp_path: Path):
    input_path = tmp_path / "rows.jsonl"
    row = {"instruction": "Explain variables.", "output": "A variable stores a value."}
    _write(input_path, [row, row, {"instruction": "Explain loops.", "output": "They repeat work."}])
    output_path = tmp_path / "export.jsonl"

    result = runner.invoke(
        app, ["export", str(input_path), str(output_path), "instruction", "--dedupe"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["cleaned"] is True
    assert payload["output_rows"] == 2
    assert payload["removed_exact_duplicates"] == 1

    manifest_path = Path(payload["manifest_path"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["removed_rows"] == 1
    assert manifest["removed"][0]["reason"] == "exact_duplicate"

    lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2


def test_cli_export_still_rejects_invalid_rows(tmp_path: Path):
    input_path = tmp_path / "invalid.jsonl"
    _write(input_path, [{"instruction": "no output field"}])
    output_path = tmp_path / "export.jsonl"

    result = runner.invoke(app, ["export", str(input_path), str(output_path), "instruction"])

    assert result.exit_code == 1
    assert "Missing required field: output" in result.output
    assert not output_path.exists()


_SECRET_ROW = {
    "instruction": "Store this deploy key.",
    "output": "-----BEGIN RSA PRIVATE KEY-----\nMIIBOwIBAAJBAKj34GkxFhD\n-----END RSA PRIVATE KEY-----",
}


def test_cli_export_blocks_on_pii_secret_and_writes_nothing(tmp_path: Path):
    # A private key must stop the export (the documented "export blocks on PII"
    # guarantee) — and no deliverable file may be written.
    input_path = tmp_path / "secret.jsonl"
    _write(input_path, [_SECRET_ROW])
    output_path = tmp_path / "export.jsonl"

    result = runner.invoke(app, ["export", str(input_path), str(output_path), "instruction"])

    assert result.exit_code == 2
    assert "blocked" in result.output.lower()
    assert not output_path.exists()


def test_cli_export_blocks_on_pii_even_with_dedupe(tmp_path: Path):
    # The gate runs BEFORE the cleaning pass, so --dedupe cannot smuggle a secret out.
    input_path = tmp_path / "secret.jsonl"
    _write(input_path, [_SECRET_ROW, _SECRET_ROW])
    output_path = tmp_path / "export.jsonl"

    result = runner.invoke(
        app, ["export", str(input_path), str(output_path), "instruction", "--dedupe"]
    )

    assert result.exit_code == 2
    assert not output_path.exists()
    assert not output_path.with_name(output_path.name + ".cleaning_manifest.json").exists()


def test_cli_export_clean_dataset_is_unaffected(tmp_path: Path):
    # A dataset with no PII/secrets exports normally — the gate must not false-block.
    input_path = tmp_path / "clean.jsonl"
    _write(input_path, [{"instruction": "Explain variables.", "output": "A variable stores a value."}])
    output_path = tmp_path / "export.jsonl"

    result = runner.invoke(app, ["export", str(input_path), str(output_path), "instruction"])

    assert result.exit_code == 0, result.output
    assert output_path.exists()
