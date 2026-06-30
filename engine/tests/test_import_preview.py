import json
from pathlib import Path

from corpus_studio.importers.jsonl_preview import preview_jsonl_import


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_import_preview_reports_accepted_and_failed_rows(tmp_path: Path):
    path = tmp_path / "mixed.jsonl"
    write_lines(
        path,
        [
            json.dumps({"instruction": "Explain variables.", "output": "A variable stores a value."}),
            json.dumps({"instruction": "Missing output."}),
            "{not json}",
            json.dumps(["not", "object"]),
        ],
    )

    report = preview_jsonl_import(path, "instruction")

    assert not report.valid
    assert report.total_rows == 4
    assert report.accepted_rows == 1
    assert report.rejected_rows == 3
    assert [row.row_number for row in report.failed_rows] == [2, 3, 4]
    assert report.failed_rows[0].errors[0].message == "Missing required field: output"
    assert report.failed_rows[1].errors[0].message.startswith("Invalid JSON:")
    assert report.failed_rows[2].errors[0].message == "Row must be a JSON object."


def test_import_preview_accepts_all_valid_rows(tmp_path: Path):
    path = tmp_path / "valid.jsonl"
    write_lines(
        path,
        [
            json.dumps({"instruction": "Explain item 1.", "output": "Item 1."}),
            json.dumps({"instruction": "Explain item 2.", "output": "Item 2.", "tags": ["ok"]}),
        ],
    )

    report = preview_jsonl_import(path, "instruction")

    assert report.valid
    assert report.total_rows == 2
    assert report.accepted_rows == 2
    assert report.rejected_rows == 0
    assert report.failed_rows == []
