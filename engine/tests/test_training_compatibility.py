import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.training.compatibility import training_compatibility_warnings


runner = CliRunner()


def test_training_compat_command_clean_for_instruction_sft():
    result = runner.invoke(app, ["training-compat", "--schema", "instruction", "--target", "axolotl_yaml"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["compatible"] is True
    assert payload["warnings"] == []


def test_training_compat_command_flags_preference_mismatch():
    result = runner.invoke(app, ["training-compat", "--schema", "preference", "--target", "trl_config"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["compatible"] is False
    assert payload["warnings"]


def test_training_compat_command_rejects_unknown_schema():
    result = runner.invoke(app, ["training-compat", "--schema", "does_not_exist"])
    assert result.exit_code == 1


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_matching_schema_format_and_target_has_no_warnings():
    warnings = training_compatibility_warnings(
        schema_id="instruction",
        dataset_format="instruction",
        target="axolotl_yaml",
    )

    assert warnings == []


def test_first_party_target_clean_for_instruction_sft():
    # The first-party corpus_studio trainer is TRL SFT — an instruction/SFT dataset is compatible.
    warnings = training_compatibility_warnings(
        schema_id="instruction",
        dataset_format="instruction",
        target="corpus_studio",
    )

    assert warnings == []


def test_first_party_target_warns_on_preference_data():
    # It has no DPO/preference path (SFT only) — a preference dataset should be flagged.
    warnings = training_compatibility_warnings(
        schema_id="preference",
        dataset_format="preference",
        target="corpus_studio",
    )

    assert any("no built-in preference" in warning for warning in warnings)


def test_preference_schema_warns_about_dpo_pipeline():
    warnings = training_compatibility_warnings(
        schema_id="preference",
        dataset_format="preference",
        target="axolotl_yaml",
    )

    assert any("DPO" in warning for warning in warnings)


def test_preference_schema_warns_when_target_lacks_preference_path():
    warnings = training_compatibility_warnings(
        schema_id="preference",
        dataset_format="preference",
        target="huggingface_trainer",
    )

    assert any("no built-in preference" in warning for warning in warnings)


def test_non_causal_schema_warns_about_wrong_trainer():
    warnings = training_compatibility_warnings(
        schema_id="image_caption",
        dataset_format="image_caption",
        target="axolotl_yaml",
    )

    assert any("different trainer" in warning for warning in warnings)


def test_unusual_format_label_is_flagged():
    warnings = training_compatibility_warnings(
        schema_id="instruction",
        dataset_format="sharegpt",
        target="axolotl_yaml",
    )

    assert any("unusual for the instruction schema" in warning for warning in warnings)


def test_training_config_command_surfaces_compatibility_warnings(tmp_path: Path):
    input_path = tmp_path / "preference.jsonl"
    output_path = tmp_path / "config.yaml"
    write_rows(
        input_path,
        [
            {
                "prompt": f"Question {index}?",
                "chosen": f"Good answer {index} with detail.",
                "rejected": f"Weak answer {index}.",
            }
            for index in range(3)
        ],
    )

    result = runner.invoke(
        app,
        [
            "training-config",
            str(input_path),
            "preference",
            "--output-path",
            str(output_path),
            "--base-model",
            "Qwen/Qwen2.5-7B",
            "--target",
            "axolotl",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert any("DPO" in warning for warning in payload["warnings"])
    assert payload["compatibility_warnings"]
