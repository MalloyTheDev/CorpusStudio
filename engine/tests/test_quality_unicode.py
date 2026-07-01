"""Regression tests: quality signals must work for non-Latin scripts.

Before the Unicode tokenizer fix, near-duplicate, low-information, and
synthetic-pattern detection were ASCII-only (`[a-z0-9_]+`), so CJK / Cyrillic /
accented-Latin datasets produced empty normalized signatures and silently
false-clean quality reports.
"""

import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.quality.basic_quality import build_basic_quality_report

runner = CliRunner()


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_identical_cjk_rows_are_normalized_duplicates():
    report = build_basic_quality_report([{"text": "これはテストです"}, {"text": "これはテストです"}])
    assert report.duplicate_normalized_count >= 1


def test_cjk_whitespace_variant_is_near_duplicate():
    # Same characters, only spacing differs — a normalized (not exact) duplicate.
    report = build_basic_quality_report([{"text": "日本語 テスト"}, {"text": "日本語テスト"}])
    assert report.duplicate_exact_count == 0
    assert report.duplicate_normalized_count >= 1


def test_accented_latin_case_fold_dedup():
    # "café" vs "CAFÉ": case-fold + NFKC should collapse to one normalized form.
    report = build_basic_quality_report([{"text": "café"}, {"text": "CAFÉ"}])
    assert report.duplicate_exact_count == 0
    assert report.duplicate_normalized_count >= 1


def test_cyrillic_case_and_punctuation_variant_dedup():
    report = build_basic_quality_report([{"text": "Привет мир"}, {"text": "привет, мир"}])
    assert report.duplicate_exact_count == 0
    assert report.duplicate_normalized_count >= 1


def test_short_cjk_row_is_low_information():
    report = build_basic_quality_report([{"text": "短い"}])
    assert report.low_information_count == 1


def test_long_cjk_sentence_is_not_low_information():
    long_sentence = "これは非常に長い日本語の文章でありテストのために十分な情報を含んでいます"
    report = build_basic_quality_report([{"text": long_sentence}])
    assert report.example_count == 1
    assert report.low_information_count == 0


def test_quality_cli_handles_non_ascii_without_crashing(tmp_path: Path):
    input_path = tmp_path / "unicode_rows.jsonl"
    _write_rows(
        input_path,
        [
            {"instruction": "説明してください", "output": "これは変数を格納します ✓"},
            {"instruction": "説明してください", "output": "これは変数を格納します ✓"},
        ],
    )
    result = runner.invoke(app, ["quality", str(input_path)])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["example_count"] == 2
    assert report["duplicate_exact_count"] == 1
    assert report["duplicate_normalized_count"] == 1
