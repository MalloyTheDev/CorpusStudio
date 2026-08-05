"""The model architecture builder (S3b front-end): a torch-free `create-model` that turns a preset /
target parameter count / explicit dims into a validated architecture config for from-scratch pretraining.
The parameter formula is validated against a reference model (GPT-2 small ~124M)."""

import json

import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.platform.architecture_builder import (
    ArchitectureBuilderError,
    build_architecture,
    estimate_parameters,
)

_runner = CliRunner()


def test_gpt2_small_formula_matches_the_reference_124m():
    # GPT-2 small: vocab 50257, hidden 768, 12 layers, 12 heads, 4x MLP, context 1024, tied -> ~124.4M.
    params = estimate_parameters(
        "gpt2",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=12,
        intermediate_size=3072,
        vocab_size=50257,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
    )
    assert 123_000_000 <= params <= 126_000_000


def test_untying_the_output_head_adds_the_embedding_matrix():
    kw = dict(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=12,
        intermediate_size=3072,
        vocab_size=50257,
        max_position_embeddings=1024,
    )
    tied = estimate_parameters("gpt2", tie_word_embeddings=True, **kw)
    untied = estimate_parameters("gpt2", tie_word_embeddings=False, **kw)
    assert untied - tied == 50257 * 768


def test_preset_small_builds_a_valid_llama_config():
    built = build_architecture("llama", preset="small", vocab_size=32000)
    assert built.hidden_size == 768 and built.num_hidden_layers == 12
    assert built.config["model_type"] == "llama"
    assert built.config["hidden_size"] == 768 and built.config["num_attention_heads"] == 12
    assert built.estimated_parameters > 0


def test_target_parameters_solves_close_to_the_ask():
    built = build_architecture("llama", target_parameters=125_000_000, vocab_size=32000)
    # the solver lands in a reasonable band of the target and reports the ACTUAL estimate (never exact)
    assert 90_000_000 <= built.estimated_parameters <= 170_000_000
    assert built.config["hidden_size"] % built.config["num_attention_heads"] == 0


def test_explicit_dims_build_and_report_params():
    built = build_architecture(
        "gpt2",
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        vocab_size=16000,
        max_position_embeddings=2048,
    )
    assert built.hidden_size == 512 and built.num_hidden_layers == 8
    assert built.config["n_embd"] == 512 and built.config["n_layer"] == 8


def test_exactly_one_mode_required():
    with pytest.raises(ArchitectureBuilderError, match="exactly one"):
        build_architecture("llama", preset="small", target_parameters=100_000_000)
    with pytest.raises(ArchitectureBuilderError, match="exactly one"):
        build_architecture("llama")


def test_unknown_family_and_preset_fail_closed():
    with pytest.raises(ArchitectureBuilderError, match="unsupported family"):
        build_architecture("mamba", preset="small")  # type: ignore[arg-type]
    with pytest.raises(ArchitectureBuilderError, match="unknown preset"):
        build_architecture("llama", preset="huge")


def test_hidden_must_be_divisible_by_heads():
    with pytest.raises(ArchitectureBuilderError, match="divisible"):
        build_architecture("llama", hidden_size=768, num_attention_heads=7)


# ---- the create-model CLI (the "create a fresh model" surface) --------------------------------------


def test_create_model_cli_preset_writes_a_config(tmp_path):
    out = tmp_path / "config.json"
    result = _runner.invoke(
        app,
        [
            "create-model", "--family", "llama", "--preset", "small",
            "--vocab-size", "32000", "--out", str(out), "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["family"] == "llama" and data["config"]["model_type"] == "llama"
    assert data["estimated_parameters"] > 0
    assert json.loads(out.read_text(encoding="utf-8"))["hidden_size"] == 768


def test_create_model_cli_solves_a_param_target():
    result = _runner.invoke(
        app,
        [
            "create-model", "--family", "gpt2", "--params", "125M",
            "--vocab-size", "50257", "--context-length", "1024", "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert 120_000_000 <= json.loads(result.stdout)["estimated_parameters"] <= 130_000_000


def test_create_model_cli_rejects_two_modes():
    result = _runner.invoke(app, ["create-model", "--preset", "small", "--params", "125M"])
    assert result.exit_code == 2
