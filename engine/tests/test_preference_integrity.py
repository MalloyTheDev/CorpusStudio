import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.exporters.preference_exporter import (
    analyze_preference_pairs,
    drop_degenerate_pairs,
)

runner = CliRunner()


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_identical_pair_is_degenerate():
    rows = [{"prompt": "p", "chosen": "same answer", "rejected": "same answer"}]
    issues = analyze_preference_pairs(rows)
    assert issues.identical == 1
    assert issues.degenerate == 1


def test_normalized_identical_pair_is_degenerate():
    # Differ only by case/punctuation — normalized signatures collapse to equal.
    rows = [{"prompt": "p", "chosen": "The Answer.", "rejected": "the answer"}]
    issues = analyze_preference_pairs(rows)
    assert issues.identical == 1


def test_empty_side_is_degenerate():
    rows = [{"prompt": "p", "chosen": "real", "rejected": "   "}]
    issues = analyze_preference_pairs(rows)
    assert issues.empty_rejected == 1
    assert issues.degenerate == 1


def test_low_contrast_pair_is_flagged_but_not_degenerate():
    rows = [{"prompt": "p", "chosen": "a b c d e f g h i j", "rejected": "a b c d e f g h i j k"}]
    issues = analyze_preference_pairs(rows)
    assert issues.low_contrast == 1
    assert issues.identical == 0
    assert issues.degenerate == 0


def test_healthy_pair_has_no_issues():
    rows = [{"prompt": "p", "chosen": "cats are mammals", "rejected": "cats are insects"}]
    issues = analyze_preference_pairs(rows)
    assert issues.degenerate == 0
    assert issues.low_contrast == 0


def test_drop_degenerate_pairs_removes_only_unusable():
    rows = [
        {"prompt": "p1", "chosen": "good", "rejected": "bad"},
        {"prompt": "p2", "chosen": "same", "rejected": "same"},
        {"prompt": "p3", "chosen": "real", "rejected": ""},
    ]
    kept = drop_degenerate_pairs(rows)
    assert [row["prompt"] for row in kept] == ["p1"]


def test_cli_reports_degenerate_pairs_without_dropping(tmp_path: Path):
    src = tmp_path / "pref.jsonl"
    _write(
        src,
        [
            {"prompt": "p1", "chosen": "good", "rejected": "bad"},
            {"prompt": "p2", "chosen": "same", "rejected": "same"},
        ],
    )
    out = tmp_path / "dpo.jsonl"
    result = runner.invoke(
        app, ["preference-export", str(src), "--output-path", str(out), "--format", "dpo"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["output_rows"] == 2  # nothing dropped by default
    assert payload["dropped_degenerate"] == 0
    assert payload["pair_issues"]["identical"] == 1
    assert any("identical" in warning.lower() for warning in payload["warnings"])


def test_cli_drop_degenerate_excludes_bad_pairs(tmp_path: Path):
    src = tmp_path / "pref.jsonl"
    _write(
        src,
        [
            {"prompt": "p1", "chosen": "good", "rejected": "bad"},
            {"prompt": "p2", "chosen": "same", "rejected": "same"},
        ],
    )
    out = tmp_path / "dpo.jsonl"
    result = runner.invoke(
        app,
        [
            "preference-export",
            str(src),
            "--output-path",
            str(out),
            "--format",
            "dpo",
            "--drop-degenerate",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["output_rows"] == 1
    assert payload["dropped_degenerate"] == 1
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines == [{"prompt": "p1", "chosen": "good", "rejected": "bad"}]
