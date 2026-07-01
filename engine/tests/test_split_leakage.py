import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.splitters.leakage import detect_split_leakage

runner = CliRunner()


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_exact_duplicate_across_splits_is_leakage():
    train = [{"text": "shared row"}]
    test = [{"text": "shared row"}]
    report = detect_split_leakage(train, [], test)
    assert report.leaked_group_count == 1
    assert report.rows_shared_across_splits == 2
    leak = report.leaks[0]
    assert leak.exact is True
    assert leak.splits == ["test", "train"]


def test_near_duplicate_across_splits_is_leakage():
    # Same tokens, only case/punctuation differ — normalized (not exact) leak.
    train = [{"text": "Hello world"}]
    validation = [{"text": "hello, world!"}]
    report = detect_split_leakage(train, validation, [])
    assert report.leaked_group_count == 1
    assert report.leaks[0].exact is False
    assert report.leaks[0].splits == ["train", "validation"]


def test_distinct_rows_have_no_leakage():
    train = [{"text": "alpha one"}]
    validation = [{"text": "beta two"}]
    test = [{"text": "gamma three"}]
    report = detect_split_leakage(train, validation, test)
    assert report.leaked_group_count == 0
    assert report.rows_shared_across_splits == 0


def test_duplicate_within_one_split_is_not_leakage():
    train = [{"text": "same"}, {"text": "same"}]
    report = detect_split_leakage(train, [], [])
    assert report.leaked_group_count == 0


def test_cli_split_reports_leakage_for_duplicated_rows(tmp_path: Path):
    input_path = tmp_path / "rows.jsonl"
    row = {"instruction": "Explain variables.", "output": "A variable stores a value."}
    _write_rows(input_path, [row] * 10)  # all identical → must span all splits
    output_dir = tmp_path / "splits"

    result = runner.invoke(app, ["split", str(input_path), str(output_dir), "instruction"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["rows_shared_across_splits"] == 10
    assert payload["leakage"]["leaked_group_count"] == 1
    assert any("leakage" in warning.lower() for warning in payload["warnings"])


def test_cli_split_reports_no_leakage_for_distinct_rows(tmp_path: Path):
    input_path = tmp_path / "rows.jsonl"
    rows = [
        {"instruction": f"Explain topic {index}.", "output": f"Answer number {index} here."}
        for index in range(10)
    ]
    _write_rows(input_path, rows)
    output_dir = tmp_path / "splits"

    result = runner.invoke(app, ["split", str(input_path), str(output_dir), "instruction"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["rows_shared_across_splits"] == 0
    assert payload["leakage"]["leaked_group_count"] == 0
    assert not any("leakage" in warning.lower() for warning in payload["warnings"])
