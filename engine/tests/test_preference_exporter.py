import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.exporters.preference_exporter import (
    export_preference,
    to_dpo,
    to_kto,
    to_reward,
)

runner = CliRunner()

ROWS = [
    {
        "prompt": "Explain recursion.",
        "chosen": "It calls itself.",
        "rejected": "It loops.",
        "reason": "clearer",
    }
]


def test_to_dpo_keeps_prompt_chosen_rejected():
    assert to_dpo(ROWS) == [
        {"prompt": "Explain recursion.", "chosen": "It calls itself.", "rejected": "It loops."}
    ]


def test_to_kto_expands_each_pair_into_labeled_rows():
    assert to_kto(ROWS) == [
        {"prompt": "Explain recursion.", "completion": "It calls itself.", "label": True},
        {"prompt": "Explain recursion.", "completion": "It loops.", "label": False},
    ]


def test_to_reward_builds_conversational_pairs():
    out = to_reward(ROWS)
    assert len(out) == 1
    assert out[0]["chosen"][0] == {"role": "user", "content": "Explain recursion."}
    assert out[0]["chosen"][-1] == {"role": "assistant", "content": "It calls itself."}
    assert out[0]["rejected"][-1] == {"role": "assistant", "content": "It loops."}


def test_export_preference_rejects_unknown_format():
    with pytest.raises(ValueError):
        export_preference(ROWS, "sharegpt")


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_cli_preference_export_writes_dpo(tmp_path: Path):
    src = tmp_path / "pref.jsonl"
    _write(src, ROWS)
    out = tmp_path / "dpo.jsonl"
    result = runner.invoke(
        app, ["preference-export", str(src), "--output-path", str(out), "--format", "dpo"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["output_rows"] == 1
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines == [
        {"prompt": "Explain recursion.", "chosen": "It calls itself.", "rejected": "It loops."}
    ]


def test_cli_preference_export_kto_doubles_rows(tmp_path: Path):
    src = tmp_path / "pref.jsonl"
    _write(src, ROWS)
    out = tmp_path / "kto.jsonl"
    result = runner.invoke(
        app, ["preference-export", str(src), "--output-path", str(out), "--format", "kto"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["output_rows"] == 2


def test_cli_preference_export_rejects_invalid_input(tmp_path: Path):
    src = tmp_path / "bad.jsonl"
    _write(src, [{"prompt": "p", "chosen": "c"}])  # missing required 'rejected'
    out = tmp_path / "out.jsonl"
    result = runner.invoke(
        app, ["preference-export", str(src), "--output-path", str(out), "--format", "dpo"]
    )
    assert result.exit_code == 1
    assert not out.exists()
