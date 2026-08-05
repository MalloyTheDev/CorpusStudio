"""The model creator (S3b front-end): torch-free `create-model` with two HONEST modes - BASE ON a known
family, or COMPOSE your own architecture from building blocks. The parameter formula is validated against
reference models (GPT-2 small ~124M, Llama-7B ~6.7B), and a composed design that no reference
implementation builds is honestly flagged as needing the custom-block path."""

import json

import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.platform.architecture_builder import (
    _FAMILIES,
    ArchitectureBuilderError,
    KNOWN_FAMILIES,
    build_composed,
    build_from_family,
    estimate_parameters,
)

_runner = CliRunner()


# ---- the parameter formula (generic, validated against reference models) ---------------------------


def test_gpt2_small_formula_matches_the_reference_124m():
    params = estimate_parameters(
        _FAMILIES["gpt2"].traits,
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


def test_llama_7b_formula_is_in_range():
    params = estimate_parameters(
        _FAMILIES["llama"].traits,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
        vocab_size=32000,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    assert 6_500_000_000 <= params <= 7_000_000_000


def test_untying_adds_the_embedding_matrix():
    kw = dict(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=12,
        intermediate_size=3072,
        vocab_size=50257,
        max_position_embeddings=1024,
    )
    tied = estimate_parameters(_FAMILIES["gpt2"].traits, tie_word_embeddings=True, **kw)
    untied = estimate_parameters(_FAMILIES["gpt2"].traits, tie_word_embeddings=False, **kw)
    assert untied - tied == 50257 * 768


def test_grouped_query_attention_reduces_params():
    kw = dict(
        hidden_size=1024,
        num_hidden_layers=8,
        num_attention_heads=16,
        intermediate_size=2816,
        vocab_size=32000,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    mha = estimate_parameters(_FAMILIES["llama"].traits, num_key_value_heads=16, **kw)
    gqa = estimate_parameters(_FAMILIES["llama"].traits, num_key_value_heads=4, **kw)
    assert gqa < mha


# ---- base on a family (honestly borrowed) ----------------------------------------------------------


def test_build_from_family_is_honestly_provenanced():
    built = build_from_family("llama", preset="small", vocab_size=32000)
    assert built.design_source == "family:llama"
    assert built.realizing_family == "llama" and built.needs_custom_code is False
    assert built.config["model_type"] == "llama" and built.hidden_size == 768


def test_build_from_family_solves_a_param_target():
    built = build_from_family(
        "gpt2", target_parameters=125_000_000, vocab_size=50257, max_position_embeddings=1024
    )
    assert 120_000_000 <= built.estimated_parameters <= 130_000_000


def test_unknown_family_fails_closed():
    with pytest.raises(ArchitectureBuilderError, match="unknown family"):
        build_from_family("mamba", preset="small")


# ---- compose your own design -----------------------------------------------------------------------


def test_composed_design_maps_to_a_real_implementation():
    # GQA + RoPE + SwiGLU + RMSNorm (no bias) IS the Llama block implementation: your DESIGN, its blocks.
    built = build_composed(
        name="MyModel",
        positions="rope",
        gated_mlp=True,
        norm="rmsnorm",
        attention_bias=False,
        mlp_bias=False,
        preset="small",
    )
    assert built.design_source == "composed" and built.name == "MyModel"
    assert built.realizing_family == "llama" and built.needs_custom_code is False
    assert built.config["corpus_studio_name"] == "MyModel"


def test_a_novel_composition_is_flagged_needs_custom_code():
    # Learned positions + SwiGLU + LayerNorm is a combination no reference implementation builds.
    built = build_composed(
        name="Novel", positions="learned", gated_mlp=True, norm="layernorm", preset="small"
    )
    assert built.needs_custom_code is True and built.realizing_family is None
    assert built.config["model_type"] == "custom_decoder"


def test_composed_attention_bias_maps_to_qwen2():
    built = build_composed(
        name="Q", positions="rope", gated_mlp=True, norm="rmsnorm", attention_bias=True, preset="small"
    )
    assert built.realizing_family == "qwen2"


def test_exactly_one_size_mode_required():
    with pytest.raises(ArchitectureBuilderError, match="exactly one"):
        build_composed(name="x", preset="small", target_parameters=100_000_000)
    with pytest.raises(ArchitectureBuilderError, match="exactly one"):
        build_composed(name="x")


# ---- the create-model CLI --------------------------------------------------------------------------


def test_cli_from_family_is_labeled_honestly(tmp_path):
    out = tmp_path / "config.json"
    result = _runner.invoke(
        app,
        [
            "create-model", "--from-family", "llama", "--preset", "small",
            "--vocab-size", "32000", "--out", str(out), "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["design_source"] == "family:llama" and data["needs_custom_code"] is False
    assert json.loads(out.read_text(encoding="utf-8"))["hidden_size"] == 768


def test_cli_compose_your_own():
    result = _runner.invoke(
        app,
        [
            "create-model", "--compose", "--name", "MyModel",
            "--positions", "rope", "--mlp", "gated", "--norm", "rmsnorm", "--preset", "small", "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["name"] == "MyModel" and data["design_source"] == "composed"
    assert data["realizing_family"] == "llama"


def test_cli_compose_novel_flags_custom_code():
    result = _runner.invoke(
        app,
        [
            "create-model", "--compose", "--name", "Novel",
            "--positions", "learned", "--mlp", "gated", "--norm", "layernorm", "--preset", "small", "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["needs_custom_code"] is True


def test_cli_requires_exactly_one_mode():
    assert _runner.invoke(app, ["create-model", "--preset", "small"]).exit_code == 2
    assert (
        _runner.invoke(
            app, ["create-model", "--from-family", "llama", "--compose", "--name", "x"]
        ).exit_code
        == 2
    )


def test_cli_compose_requires_a_name():
    result = _runner.invoke(app, ["create-model", "--compose", "--preset", "small"])
    assert result.exit_code == 2
    assert "requires --name" in result.output


def test_cli_rejects_non_finite_params():
    for bad in ("inf", "1e999", "nan"):
        assert (
            _runner.invoke(app, ["create-model", "--from-family", "llama", "--params", bad]).exit_code
            == 2
        )


def test_cli_reports_an_unwritable_out_path(tmp_path):
    bad = tmp_path / "missing-dir" / "config.json"
    result = _runner.invoke(
        app, ["create-model", "--from-family", "llama", "--preset", "small", "--out", str(bad)]
    )
    assert result.exit_code == 2
    assert "cannot write --out" in result.output


# ---- audit fixes: family identity + the expanded set -----------------------------------------------


def test_every_family_uses_its_own_model_type_and_class():
    # AUDIT fix: a family must carry its OWN transformers model_type - NOT the one its block signature
    # happens to share. Mistral/Gemma share Llama's blocks but are NOT llama.
    assert set(KNOWN_FAMILIES) >= {"mistral", "gemma", "qwen3", "phi", "starcoder2", "stablelm"}
    for fam in KNOWN_FAMILIES:
        assert build_from_family(fam, preset="tiny").config["model_type"] == fam
    assert build_from_family("mistral", preset="small").config["architectures"] == ["MistralForCausalLM"]
    # AUDIT fix: multi-word model types must not be mangled by capitalize()
    assert build_from_family("gpt_neox", preset="small").config["architectures"] == ["GPTNeoXForCausalLM"]


def test_composing_gated_layernorm_realizes_on_stablelm():
    built = build_composed(name="S", positions="rope", gated_mlp=True, norm="layernorm", preset="small")
    assert built.realizing_family == "stablelm" and built.needs_custom_code is False


# ---- audit fixes: fail-closed on invalid / unsupported requests ------------------------------------


def test_zero_kv_heads_fails_closed_not_crash():
    # AUDIT (Codex): --num-kv-heads 0 must be a typed error, not a ZeroDivisionError. Covers all size modes.
    with pytest.raises(ArchitectureBuilderError, match="num_key_value_heads must be positive"):
        build_from_family("llama", preset="small", num_key_value_heads=0)
    with pytest.raises(ArchitectureBuilderError, match="num_key_value_heads must be positive"):
        build_from_family("llama", target_parameters=125_000_000, num_key_value_heads=0)


def test_impossible_kv_head_target_fails_closed_not_assert():
    # AUDIT (Codex): a KV-head count no searched width can satisfy must raise a typed error, not an
    # uncaught AssertionError from the solve loop.
    with pytest.raises(ArchitectureBuilderError, match="no architecture in the search range"):
        build_from_family("llama", target_parameters=125_000_000, num_key_value_heads=129)


def test_gqa_on_a_non_gqa_block_is_refused():
    # AUDIT (Codex): GPT-2 / GPT-NeoX cannot express distinct KV heads; a GQA request there would make the
    # estimate diverge from the built model. Fail closed rather than silently drop it.
    with pytest.raises(ArchitectureBuilderError, match="grouped-query attention"):
        build_from_family("gpt2", preset="small", num_key_value_heads=4)
    with pytest.raises(ArchitectureBuilderError, match="grouped-query attention"):
        build_composed(
            name="G", positions="learned", gated_mlp=False, norm="layernorm",
            attention_bias=True, mlp_bias=True, num_key_value_heads=4, preset="small",
        )


def test_unknown_activation_is_refused():
    # AUDIT (Codex): an activation no ACT2FN entry provides would only fail far away at worker
    # instantiation; refuse it at compose time.
    with pytest.raises(ArchitectureBuilderError, match="unknown activation"):
        build_composed(name="A", activation="banana", preset="small")


def test_custom_decoder_config_carries_a_durable_marker():
    built = build_composed(
        name="Novel", positions="learned", gated_mlp=True, norm="layernorm", preset="small"
    )
    assert built.config["corpus_studio_needs_custom_code"] is True


# ---- audit round 2: no silently-ignored controls (wiring) ------------------------------------------


def test_family_mode_refuses_compose_only_block_flags():
    # AUDIT: block-shape flags are meaningless for a fixed family; silently ignoring them is a broken
    # control (memory: "a control that is tested but UNREACHABLE is not a control"). Refuse fail-closed.
    result = _runner.invoke(
        app, ["create-model", "--from-family", "llama", "--norm", "layernorm", "--preset", "small"]
    )
    assert result.exit_code == 2
    assert "--norm" in result.output and "--compose" in result.output


def test_composed_standard_mlp_is_not_undersized_via_cli():
    # AUDIT: the CLI must not pin an intermediate ratio that starves a standard MLP - a standard FFN is 4x,
    # not the gated 8/3. Regression: a hardcoded 2.6667 CLI default once bypassed the shape default.
    result = _runner.invoke(
        app,
        [
            "create-model", "--compose", "--name", "S", "--mlp", "standard",
            "--norm", "layernorm", "--positions", "learned", "--preset", "small", "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["intermediate_size"] == 3072  # 768 * 4, not 768 * 2.6667


def test_explicit_dims_with_a_preset_are_refused_not_ignored():
    # AUDIT: num_layers / num_heads / intermediate_size refine explicit hidden_size sizing only; combining
    # them with a preset once silently ignored them.
    with pytest.raises(ArchitectureBuilderError, match="explicit hidden_size sizing"):
        build_from_family("llama", preset="small", num_hidden_layers=8)
    result = _runner.invoke(
        app, ["create-model", "--from-family", "llama", "--preset", "small", "--num-layers", "8"]
    )
    assert result.exit_code == 2


def test_zero_attention_heads_fails_closed():
    with pytest.raises(ArchitectureBuilderError, match="num_attention_heads must be positive"):
        build_composed(name="x", hidden_size=256, num_attention_heads=0)
