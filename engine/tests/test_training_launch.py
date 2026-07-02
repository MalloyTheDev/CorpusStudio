import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.training.launch import (
    RESUME_CHECKPOINT_PLACEHOLDER,
    build_launch_plan,
    find_checkpoints,
    latest_checkpoint,
)

runner = CliRunner()


def test_axolotl_command_and_resume_flag():
    plan = build_launch_plan("axolotl_yaml", "out/config.yaml")
    assert plan.command == 'accelerate launch -m axolotl.cli.train "out/config.yaml"'
    assert plan.resume_supported is True
    assert RESUME_CHECKPOINT_PLACEHOLDER in plan.resume_command
    assert "axolotl" in plan.dependencies


def test_axolotl_resume_with_checkpoint():
    plan = build_launch_plan("axolotl_yaml", "c.yaml", resume_checkpoint="out/checkpoint-200")
    assert '--resume_from_checkpoint="out/checkpoint-200"' in plan.resume_command


def test_python_target_command_and_config_driven_resume():
    plan = build_launch_plan("trl_config", "out/config.py")
    assert plan.command == 'python "out/config.py"'
    assert plan.resume_supported is False
    assert plan.resume_command == plan.command
    assert any("config-driven" in note for note in plan.notes)


def test_llama_factory_command():
    plan = build_launch_plan("llama_factory", "out/config.yaml")
    assert plan.command == 'llamafactory-cli train "out/config.yaml"'


def test_unknown_target_raises():
    with pytest.raises(ValueError):
        build_launch_plan("nope", "c.yaml")


def test_find_checkpoints_sorted_by_step(tmp_path: Path):
    for name in ["checkpoint-50", "checkpoint-200", "checkpoint-100", "not-a-checkpoint"]:
        (tmp_path / name).mkdir()
    assert find_checkpoints(tmp_path) == ["checkpoint-50", "checkpoint-100", "checkpoint-200"]
    assert latest_checkpoint(tmp_path) == "checkpoint-200"


def test_find_checkpoints_missing_dir(tmp_path: Path):
    assert find_checkpoints(tmp_path / "nope") == []
    assert latest_checkpoint(tmp_path / "nope") is None


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_cli_training_config_includes_launch(tmp_path: Path):
    dataset = tmp_path / "train.jsonl"
    _write(dataset, [{"instruction": "x", "output": "y"}])
    out = tmp_path / "config.yaml"
    result = runner.invoke(
        app,
        [
            "training-config",
            str(dataset),
            "instruction",
            "--output-path",
            str(out),
            "--base-model",
            "some-model",
            "--target",
            "axolotl",
        ],
    )
    assert result.exit_code == 0, result.output
    launch = json.loads(result.output)["launch"]
    assert launch["target"] == "axolotl_yaml"
    assert "accelerate launch" in launch["command"]
    assert str(out) in launch["command"]


def test_cli_training_checkpoints_lists_and_builds_resume(tmp_path: Path):
    output_dir = tmp_path / "run"
    (output_dir / "checkpoint-10").mkdir(parents=True)
    (output_dir / "checkpoint-40").mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("x: 1\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "training-checkpoints",
            str(output_dir),
            "--target",
            "axolotl",
            "--config-path",
            str(config),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["checkpoints"] == ["checkpoint-10", "checkpoint-40"]
    assert payload["latest_checkpoint"] == "checkpoint-40"
    assert "checkpoint-40" in payload["resume_command"]
