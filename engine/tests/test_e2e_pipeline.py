"""End-to-end pipeline smoke test: the real import -> quality -> export flow the desktop drives,
exercised through the engine CLI (CliRunner).

This is intentionally an ENGINE-level e2e, not a desktop-process one — the desktop CI has no
Python/engine — but it chains the actual commands, so a break in the data flow between stages
(a preview that miscounts, a quality report that misses a dup, an export that doesn't clean) is
caught end to end. A desktop-process e2e (PythonEngineService shelling out to the engine) is a
deliberate deferral: it needs Python + the engine installed in the desktop CI runner.
"""

import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app

runner = CliRunner()

_ROW = {"instruction": "Explain recursion.", "output": "A function calls itself."}
_DUP = {"instruction": "Explain recursion.", "output": "A function calls itself."}  # exact dup of _ROW
_OTHER = {"instruction": "Explain a loop.", "output": "It repeats work until a condition holds."}


def test_e2e_import_preview_then_quality_then_export(tmp_path: Path):
    # 1) A raw import file: three valid rows (one an exact duplicate) plus a malformed line.
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(json.dumps(row) for row in (_ROW, _DUP, _OTHER)) + "\n" + "{ not valid json\n",
        encoding="utf-8",
    )

    # import-preview separates accepted rows from the malformed one. (The desktop then commits the
    # accepted rows; the engine never writes examples.jsonl itself.)
    preview = runner.invoke(app, ["import-preview", str(raw), "instruction"])
    assert preview.exit_code == 0, preview.output
    preview_payload = json.loads(preview.output)
    assert preview_payload["accepted_rows"] == 3
    assert preview_payload["rejected_rows"] == 1

    # 2) Simulate the commit: the accepted (valid) rows become the dataset the next stages see.
    committed = tmp_path / "examples.jsonl"
    committed.write_text(
        "\n".join(json.dumps(row) for row in (_ROW, _DUP, _OTHER)) + "\n", encoding="utf-8"
    )

    # quality finds the exact duplicate among the committed rows.
    quality = runner.invoke(app, ["quality", str(committed)])
    assert quality.exit_code == 0, quality.output
    quality_payload = json.loads(quality.output)
    assert quality_payload["example_count"] == 3
    assert quality_payload["duplicate_exact_count"] == 1

    # 3) export --dedupe drops the duplicate, writes the deliverable + a cleaning manifest, and
    # clears the export gate (no PII/secrets).
    out = tmp_path / "export.jsonl"
    export = runner.invoke(app, ["export", str(committed), str(out), "instruction", "--dedupe"])
    assert export.exit_code == 0, export.output
    export_payload = json.loads(export.output)
    assert export_payload["cleaned"] is True
    assert export_payload["output_rows"] == 2  # the duplicate is gone
    assert export_payload["removed_exact_duplicates"] == 1
    assert out.exists()
    kept = [line for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(kept) == 2
