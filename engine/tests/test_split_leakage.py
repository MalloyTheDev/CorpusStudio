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
    # One copy crossed the boundary (2 copies, home split holds 1): total - max = 1.
    assert report.rows_shared_across_splits == 1
    leak = report.leaks[0]
    assert leak.exact is True
    assert leak.splits == ["test", "train"]
    assert leak.row_count == 2  # the group still reports both copies


def test_leakage_count_excludes_within_split_duplicates():
    # 3 copies in train + 1 in validation + 1 in test: 5 total, home split (train) holds 3,
    # so only 2 copies actually leaked across a boundary.
    train = [{"text": "dup"}, {"text": "dup"}, {"text": "dup"}]
    validation = [{"text": "dup"}]
    test = [{"text": "dup"}]
    report = detect_split_leakage(train, validation, test)
    assert report.leaked_group_count == 1
    assert report.rows_shared_across_splits == 2
    assert report.leaks[0].row_count == 5


def test_near_duplicate_across_splits_is_leakage():
    # Same tokens, only case/punctuation differ — normalized (not exact) leak.
    train = [{"text": "Hello world"}]
    validation = [{"text": "hello, world!"}]
    report = detect_split_leakage(train, validation, [])
    assert report.leaked_group_count == 1
    assert report.leaks[0].exact is False
    assert report.leaks[0].splits == ["train", "validation"]


def test_leak_sample_is_readable_original_not_normalized_signature():
    # The sample must show the real row text (original case, JSON), not the lowercased,
    # field-separator-joined normalized signature used for matching.
    train = [{"instruction": "Explain Recursion", "output": "A function calls itself"}]
    test = [{"instruction": "explain recursion", "output": "a function calls itself"}]  # near-dup
    report = detect_split_leakage(train, [], test)
    assert report.leaked_group_count == 1
    sample = report.leaks[0].sample
    assert "Explain Recursion" in sample   # original casing preserved (not normalized)
    assert "\x1f" not in sample            # not the field-separator signature
    assert sample.startswith("{")          # readable JSON rendering of the original row


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
    # 10 identical rows land mostly in one split; the copies beyond it are the leak count.
    assert payload["rows_shared_across_splits"] >= 1
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
