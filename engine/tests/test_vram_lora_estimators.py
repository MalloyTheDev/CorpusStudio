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


def test_local_model_path_ignores_the_git_revision_component():
    # A local snapshot path ends in a git-revision hex commit whose digits (incl. b/B) can read as a
    # '<n>b' parameter token. The model NAME must win, not the revision. Regression: the 7B ladder plan
    # bound .../Qwen2.5-7B-Instruct/a09a35458c702b33... and the fit calibrator read 702B -> a false
    # 836 GB hard-OOM (the real 7B QLoRA fits in 12 GB).
    assert (
        parse_parameter_count(
            "/mnt/training-nvme/models/Qwen2.5-7B-Instruct/a09a35458c702b33eeacc393d103063234e8bc28"
        )
        == 7.0
    )
    assert (
        parse_parameter_count("/models/Qwen2.5-0.5B-Instruct/7ae557604adf67be50417f59c2c2f167def9a775")
        == 0.5
    )
    # a Hub id (no revision component) is unchanged
    assert parse_parameter_count("Qwen/Qwen2.5-7B-Instruct") == 7.0


def test_unparseable_name_returns_none():
    assert parse_parameter_count("my-custom-model") is None


def test_moe_active_expert_suffix_uses_total_not_active():
    # 'A##B' active-expert suffix must not win over the real total size.
    assert parse_parameter_count("Qwen3-30B-A3B") == 30.0
    assert parse_parameter_count("Qwen2-57B-A14B") == 57.0


def test_bloom_style_trailing_digit_parses():
    assert parse_parameter_count("bigscience/bloom-7b1") == 7.1


def test_quantization_suffix_not_read_as_size():
    # '8bit' is quantization, not an 8B size.
    assert parse_parameter_count("some-model-8bit") is None


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


def test_vram_estimate_matches_the_native_linux_7b_ladder():
    # RECALIBRATED to the native-Linux RTX 5070 7B QLoRA ladder (MATH path, torch peak_reserved):
    # seq512 -> 10.44 GB, seq1024 -> 10.60 GB measured NATIVE_SAFE; seq2048 -> OOM (true demand ~12.5 GB
    # vs the 12.34 GB card). The estimate is CONSERVATIVE (>= the measured points, never under-predicts)
    # and predicts the OOM boundary at 2048. The old 11.4/2.2 coefficients were a Windows/WDDM sweep that
    # overstated the native slope AND under-predicted the seq-512 fit (9.3 vs the measured 10.44).
    def m(seq):
        return build_vram_estimate("Qwen/Qwen2.5-7B-Instruct", sequence_len=seq, math_attention=True).total_gb_int4

    assert 10.3 <= m(512) <= 11.5, m(512)  # >= measured 10.44 (no under-prediction), still conservative
    assert 10.6 <= m(1024) <= 11.8, m(1024)  # >= measured 10.60
    assert m(2048) > 12.34, m(2048)  # predicts the measured OOM (over the 12.34 GB card)
    # Flash/mem-efficient attention avoids the fp32 attention-score matrix and REACHES seq2048...
    assert build_vram_estimate("Qwen/Qwen2.5-7B-Instruct", sequence_len=2048).total_gb_int4 <= 12.34
    # ...but keeps the kernel-INDEPENDENT vocab-logits transient, so flash alone does NOT reach seq4096
    # (that needs a fused/chunked cross-entropy - a separate worker change).
    assert build_vram_estimate("Qwen/Qwen2.5-7B-Instruct", sequence_len=4096).total_gb_int4 > 12.34
    # Quantized paths carry the bitsandbytes runtime overhead that fp16 does not.
    assert any("bitsandbytes" in a for a in build_vram_estimate("7B").assumptions)


def test_math_path_estimate_exceeds_flash():
    # Blackwell/sm_120 is forced onto math/eager attention, which materializes the fp32 attention-score
    # matrix that flash/mem-efficient attention avoids -> math is heavier at a given seq. Both paths share
    # the kernel-independent vocab-logits transient, so the gap is the attention term (not a 5x factor).
    flash = build_vram_estimate("Qwen/Qwen2.5-7B-Instruct", sequence_len=2048)
    math = build_vram_estimate("Qwen/Qwen2.5-7B-Instruct", sequence_len=2048, math_attention=True)
    assert math.total_gb_int4 > flash.total_gb_int4  # math is heavier (it adds the attention scores)
    assert any("math/eager" in a for a in math.assumptions)
    assert any("flash" in a for a in flash.assumptions)


def test_activation_scales_linearly_with_sequence_not_squared():
    # Gradient checkpointing makes the peak LINEAR in seq_len (the memory sweep proved it — not seq²):
    # equal-width seq steps give ~equal memory steps.
    a = build_vram_estimate("7B", sequence_len=1024, math_attention=True).total_gb_int4
    b = build_vram_estimate("7B", sequence_len=2048, math_attention=True).total_gb_int4
    c = build_vram_estimate("7B", sequence_len=3072, math_attention=True).total_gb_int4
    assert abs((b - a) - (c - b)) < 0.3  # 1024→2048 step ≈ 2048→3072 step ⇒ linear


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
