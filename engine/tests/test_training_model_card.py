import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.training.model_card import (
    build_model_card,
    read_adapter_config,
    write_model_card,
)

runner = CliRunner()


def _write_adapter(adapter_dir: Path, config: dict) -> Path:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    return adapter_dir


def test_read_adapter_config_reads_and_tolerates_absence(tmp_path: Path):
    assert read_adapter_config(tmp_path) is None  # no file
    _write_adapter(tmp_path, {"base_model_name_or_path": "m", "r": 8})
    assert read_adapter_config(tmp_path)["r"] == 8
    (tmp_path / "adapter_config.json").write_text("{not json", encoding="utf-8")
    assert read_adapter_config(tmp_path) is None  # unreadable → None, not a crash


def test_card_includes_base_model_lora_params_and_license_note(tmp_path: Path):
    adapter = _write_adapter(
        tmp_path / "run",
        {
            "base_model_name_or_path": "Qwen/Qwen2.5-7B",
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
        },
    )
    card = build_model_card(adapter)

    assert "Qwen/Qwen2.5-7B" in card
    assert "r: 16, alpha: 32" in card
    assert "q_proj, v_proj" in card
    # The base model's license governing the result is the key honesty point.
    assert "license" in card.lower()
    assert "not an evaluation of the model's quality" in card.lower() or "not an evaluation" in card.lower()


def test_cpu_toy_card_loudly_warns_it_is_not_a_usable_model(tmp_path: Path):
    adapter = _write_adapter(tmp_path / "toy", {"base_model_name_or_path": "tiny", "r": 8})
    card = build_model_card(adapter, train_result={"cpu_toy": True, "steps": 3})

    assert "not a usable model" in card.lower()
    assert "smoke test" in card.lower()


def test_base_model_override_and_folded_in_config_and_provenance(tmp_path: Path):
    adapter = _write_adapter(tmp_path / "run", {"base_model_name_or_path": "recorded-base", "r": 16})
    card = build_model_card(
        adapter,
        base_model="override-base",
        training_config={"format": "chat", "sequence_len": 4096, "learning_rate": 0.0002, "seed": 42},
        train_result={"steps": 100, "final_loss": 0.1234, "cpu_toy": False},
        provenance={
            "dataset_fingerprint": "abcd1234",
            "dataset_row_count": 500,
            "config_sha256": "deadbeef",
            "engine_version": "1.3.0",
            "platform": "Windows",
        },
        generated_at="2026-07-10T00:00:00Z",
    )

    assert "override-base" in card
    assert "recorded-base" not in card  # the override wins
    assert "Sequence length: 4096" in card
    assert "final train loss: 0.1234" in card
    assert "abcd1234" in card and "500 rows" in card
    assert "1.3.0" in card and "Windows" in card
    assert "4-bit QLoRA" in card  # not the toy mode


def test_write_model_card_writes_the_file(tmp_path: Path):
    written = write_model_card(tmp_path, "# hi\n")
    assert written == tmp_path / "MODEL_CARD.md"
    assert written.read_text(encoding="utf-8") == "# hi\n"


def test_cli_model_card_prints_and_writes(tmp_path: Path):
    adapter = _write_adapter(tmp_path / "run", {"base_model_name_or_path": "Qwen/Qwen2.5-7B", "r": 16, "lora_alpha": 32})

    # stdout form
    result = runner.invoke(app, ["model-card", str(adapter)])
    assert result.exit_code == 0, result.output
    assert "Model card" in result.output
    assert "Qwen/Qwen2.5-7B" in result.output

    # --output form writes a file
    out = tmp_path / "CARD.md"
    result = runner.invoke(app, ["model-card", str(adapter), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "Qwen/Qwen2.5-7B" in out.read_text(encoding="utf-8")


def test_cli_model_card_folds_in_a_training_config(tmp_path: Path):
    adapter = _write_adapter(tmp_path / "run", {"base_model_name_or_path": "m", "r": 8})
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"format": "instruction", "sequence_len": 2048, "learning_rate": 0.0002, "seed": 7}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["model-card", str(adapter), "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "Sequence length: 2048" in result.output
    assert "Format: instruction" in result.output
