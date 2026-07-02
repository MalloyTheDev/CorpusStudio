import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.training.estimators import (
    build_vram_estimate,
    parse_parameter_count,
    recommend_lora,
)

runner = CliRunner()


# --- parameter count parsing -------------------------------------------------

def test_parses_common_model_names():
    assert parse_parameter_count("Qwen/Qwen2.5-Coder-7B-Instruct") == 7.0
    assert parse_parameter_count("llama-3-8b") == 8.0
    assert parse_parameter_count("tiny-0.5B-chat") == 0.5
    assert parse_parameter_count("meta-llama/Llama-2-13b-hf") == 13.0


def test_parses_moe_total_params():
    assert parse_parameter_count("mixtral-8x7b-instruct") == 56.0


def test_version_fragments_do_not_win():
    # "Qwen2.5" must not parse as 2.5B when a real size suffix exists.
    assert parse_parameter_count("Qwen2.5-Coder-7B") == 7.0


def test_unparseable_name_returns_none():
    assert parse_parameter_count("my-custom-model") is None


# --- VRAM estimate -----------------------------------------------------------

def test_7b_fp16_weights_are_14gb():
    estimate = build_vram_estimate("some-7B-model")
    assert estimate.parameter_count_billions == 7.0
    assert estimate.weights_gb_fp16 == 14.0
    assert estimate.weights_gb_int8 == 7.0
    assert estimate.weights_gb_int4 == 3.5
    # Totals include LoRA + activation + runtime overhead, so exceed weights.
    assert estimate.total_gb_fp16 > estimate.weights_gb_fp16
    assert estimate.total_gb_int4 < estimate.total_gb_fp16
    assert estimate.assumptions  # assumptions are always listed


def test_activation_scales_with_sequence_and_batch():
    small = build_vram_estimate("7B", sequence_len=2048, micro_batch_size=1)
    large = build_vram_estimate("7B", sequence_len=8192, micro_batch_size=4)
    assert large.activation_overhead_gb > small.activation_overhead_gb


def test_lora_overhead_scales_with_rank():
    low = build_vram_estimate("7B", lora_r=8)
    high = build_vram_estimate("7B", lora_r=64)
    assert high.lora_overhead_gb > low.lora_overhead_gb


def test_unknown_model_is_honest():
    estimate = build_vram_estimate("my-custom-model")
    assert estimate.parameter_count_billions is None
    assert estimate.total_gb_fp16 is None
    assert "Could not parse" in estimate.note


# --- LoRA recommendation -----------------------------------------------------

def test_recommends_r_by_model_size():
    assert recommend_lora(1.0, 8, 16).recommended_r == 8
    assert recommend_lora(7.0, 16, 32).recommended_r == 16
    assert recommend_lora(70.0, 64, 128).recommended_r == 64


def test_conventional_choice_has_no_warnings():
    assert recommend_lora(7.0, 16, 32).warnings == []


def test_unusually_high_r_warns():
    warnings = recommend_lora(7.0, 128, 256).warnings
    assert any("unusually high" in warning for warning in warnings)


def test_alpha_convention_deviation_warns():
    warnings = recommend_lora(7.0, 16, 64).warnings
    assert any("alpha=2*r" in warning for warning in warnings)


def test_unknown_size_still_recommends_default():
    assert recommend_lora(None, 16, 32).recommended_r == 16


# --- CLI wiring ----------------------------------------------------------------

def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_cli_training_config_emits_vram_and_lora(tmp_path: Path):
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
            "Qwen/Qwen2.5-Coder-7B-Instruct",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["vram_estimate"]["parameter_count_billions"] == 7.0
    assert payload["vram_estimate"]["weights_gb_fp16"] == 14.0
    assert payload["lora_recommendation"]["recommended_r"] == 16


def test_cli_unknown_model_size_warns(tmp_path: Path):
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
            "my-custom-model",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["vram_estimate"]["parameter_count_billions"] is None
    assert any("No VRAM estimate" in warning for warning in payload["warnings"])
