"""First-party trainer — the PURE helpers (config load, formatting, arg mapping, run-plan resolution).

These carry no heavy imports, so they run in CI without torch/TRL. The actual `run_training` is
verified separately via the CPU toy path (installing the CPU subset of the [train] extra).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpus_studio.training.environment import TrainingRuntimeReport
from corpus_studio.training.trainer import (
    TINY_TOY_MODEL,
    TrainerError,
    TrainRunConfig,
    build_lora_kwargs,
    build_training_kwargs,
    format_example_text,
    load_run_config_from_file,
    resolve_attention_implementation,
    resolve_run_plan,
    _list_checkpoints,
)


def _config(tmp_path: Path, **overrides) -> Path:
    data = {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "dataset_path": "train.jsonl",
        "format": "chat",
        "sequence_len": 4096,
        "lora_r": 16,
        "lora_alpha": 32,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 0.0002,
        "seed": 42,
    }
    data.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---- config load -------------------------------------------------------------


def test_load_config_maps_all_fields(tmp_path):
    cfg = load_run_config_from_file(_config(tmp_path))
    assert cfg.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.dataset_path == "train.jsonl"
    assert cfg.dataset_format == "chat"
    assert cfg.lora_r == 16 and cfg.lora_alpha == 32
    assert cfg.seed == 42
    assert cfg.cpu_toy is False


def test_cpu_toy_forces_tiny_model_short_seq_and_steps(tmp_path):
    cfg = load_run_config_from_file(_config(tmp_path), cpu_toy=True)
    assert cfg.base_model == TINY_TOY_MODEL  # not the 7B
    assert cfg.sequence_len <= 128
    assert cfg.max_steps == 3
    assert cfg.cpu_toy is True


def test_overrides_win(tmp_path):
    cfg = load_run_config_from_file(
        _config(tmp_path), base_model="my/model", dataset_path="split/train.jsonl", max_steps=10
    )
    assert cfg.base_model == "my/model"
    assert cfg.dataset_path == "split/train.jsonl"
    assert cfg.max_steps == 10


def test_missing_base_or_dataset_raises(tmp_path):
    with pytest.raises(TrainerError):
        load_run_config_from_file(_config(tmp_path, base_model="", dataset_path=""))


def test_load_config_accepts_yaml_so_a_named_yaml_does_not_die(tmp_path):
    # train-run parses JSON, but a config named *.yaml (or a hand-written YAML) must still load —
    # the WBG run pointed train-run at wbg7b_corpus.yaml. JSON is a YAML subset, but real YAML too.
    # PyYAML is only needed for the YAML fallback and ships with the [train] extra (transformers/
    # datasets) — where train-run actually runs — so skip when it's absent (the dependency-light gate).
    pytest.importorskip("yaml")
    config = tmp_path / "wbg7b_corpus.yaml"
    config.write_text(
        "base_model: Qwen/Qwen2.5-7B\ndataset_path: train.jsonl\nformat: chat\nsequence_len: 4096\nlora_r: 16\n",
        encoding="utf-8",
    )
    cfg = load_run_config_from_file(config)
    assert cfg.base_model == "Qwen/Qwen2.5-7B"
    assert cfg.dataset_path == "train.jsonl"
    assert cfg.dataset_format == "chat"
    assert cfg.sequence_len == 4096 and cfg.lora_r == 16


def test_load_config_non_mapping_raises_trainer_error(tmp_path):
    # A file that is neither a JSON object nor a YAML mapping is a clean TrainerError (→ CLI exit 2),
    # not a cryptic parser traceback.
    config = tmp_path / "bad.yaml"
    config.write_text("just a plain string, not a config mapping\n", encoding="utf-8")
    with pytest.raises(TrainerError):
        load_run_config_from_file(config)


def test_load_config_reads_attn_implementation(tmp_path):
    # From the config file, and an explicit override wins.
    cfg = load_run_config_from_file(_config(tmp_path, attn_implementation="eager"))
    assert cfg.attn_implementation == "eager"
    override = load_run_config_from_file(_config(tmp_path), attn_implementation="sdpa")
    assert override.attn_implementation == "sdpa"
    assert load_run_config_from_file(_config(tmp_path)).attn_implementation is None


# ---- memory / spill-avoidance levers -----------------------------------------


def test_build_kwargs_sets_optim_and_liger():
    cfg = TrainRunConfig(base_model="m", dataset_path="d", optim="paged_adamw_8bit", use_liger=True)
    kwargs = build_training_kwargs(cfg)
    assert kwargs["optim"] == "paged_adamw_8bit"
    assert kwargs["use_liger_kernel"] is True


def test_build_kwargs_default_optim_and_no_liger():
    kwargs = build_training_kwargs(TrainRunConfig(base_model="m", dataset_path="d"))
    assert kwargs["optim"] == "adamw_torch"
    assert "use_liger_kernel" not in kwargs  # off by default — never requested unless opted in


def test_cpu_toy_forces_plain_optimizer_and_no_liger():
    # The paged optimizer (bitsandbytes) and Liger (Triton) are CUDA-only; the CPU toy must never
    # request them or it would crash on a GPU-less machine, defeating the smoke test.
    cfg = TrainRunConfig(
        base_model="m", dataset_path="d", cpu_toy=True, optim="paged_adamw_8bit", use_liger=True
    )
    kwargs = build_training_kwargs(cfg)
    assert kwargs["optim"] == "adamw_torch"
    assert "use_liger_kernel" not in kwargs
    assert kwargs["use_cpu"] is True


def test_load_config_reads_optim_and_liger(tmp_path):
    cfg = load_run_config_from_file(_config(tmp_path, optim="paged_adamw_8bit", use_liger=True))
    assert cfg.optim == "paged_adamw_8bit"
    assert cfg.use_liger is True
    # An explicit override wins over the config file.
    override = load_run_config_from_file(_config(tmp_path), optim="adamw_8bit", use_liger=True)
    assert override.optim == "adamw_8bit" and override.use_liger is True
    # Defaults when absent — the levers are opt-in.
    base = load_run_config_from_file(_config(tmp_path))
    assert base.optim == "adamw_torch" and base.use_liger is False


def test_resolve_attention_native_windows_blackwell_disables_flash_sdpa():
    # NATIVE WINDOWS + Blackwell (sm_120 → capability major 12): the fused FLASH SDPA kernel deadlocks
    # on the first backward under the Windows WDDM driver (verified on a real 5070; mem-efficient + math
    # are fine), so keep default SDPA but signal the caller to disable just the flash backend.
    assert resolve_attention_implementation(None, 12, native_windows=True) == (None, True)
    assert resolve_attention_implementation(None, 13, native_windows=True) == (None, True)


def test_resolve_attention_wsl_or_linux_blackwell_keeps_flash_enabled():
    # The deadlock is a Windows WDDM property, NOT an sm_120 kernel bug: on WSL / bare Linux the SAME
    # flash kernel runs fine (verified on a real 5070 under WSL2), so flash must stay ENABLED there —
    # the whole reason to run training under WSL. native_windows=False (WSL Python reports sys.platform
    # 'linux') → no SDP toggling on Blackwell.
    assert resolve_attention_implementation(None, 12, native_windows=False) == (None, False)
    assert resolve_attention_implementation(None, 13, native_windows=False) == (None, False)
    assert resolve_attention_implementation(None, 12) == (None, False)  # default (unknown host) = safe


def test_resolve_attention_older_arch_is_unchanged():
    # Pre-Blackwell arch: no toggling regardless of OS (the deadlock is sm_120-specific).
    assert resolve_attention_implementation(None, 9, native_windows=True) == (None, False)   # Ada/Hopper
    assert resolve_attention_implementation(None, 8, native_windows=True) == (None, False)
    assert resolve_attention_implementation(None, None, native_windows=True) == (None, False)  # no GPU


def test_resolve_attention_explicit_choice_always_wins():
    # An explicit attn_implementation is honored verbatim and never toggles the SDP backends, even on
    # native-Windows Blackwell.
    assert resolve_attention_implementation("eager", 12, native_windows=True) == ("eager", False)
    assert resolve_attention_implementation("flash_attention_2", 8, native_windows=True) == (
        "flash_attention_2",
        False,
    )


# ---- formatting --------------------------------------------------------------


def test_format_instruction():
    text = format_example_text({"instruction": "Explain X.", "output": "It is Y."}, "instruction")
    assert "Explain X." in text and "It is Y." in text and "### Response:" in text


def test_format_chat_without_tokenizer_joins_roles():
    row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    text = format_example_text(row, "chat")
    assert "user: hi" in text and "assistant: hello" in text


def test_format_empty_row_is_dropped():
    assert format_example_text({"instruction": "", "output": ""}, "instruction") == ""
    assert format_example_text({"messages": []}, "chat") == ""


# ---- arg mapping -------------------------------------------------------------


def test_lora_kwargs_use_all_linear():
    kw = build_lora_kwargs(_cfg(lora_r=16, lora_alpha=32))
    assert kw["r"] == 16 and kw["lora_alpha"] == 32
    assert kw["target_modules"] == "all-linear"
    assert kw["task_type"] == "CAUSAL_LM"


def test_training_kwargs_capped_steps_vs_epochs(tmp_path):
    with_steps = build_training_kwargs(load_run_config_from_file(_config(tmp_path), max_steps=5))
    assert with_steps["max_steps"] == 5 and "num_train_epochs" not in with_steps
    without = build_training_kwargs(load_run_config_from_file(_config(tmp_path)))
    assert without["num_train_epochs"] == 1 and "max_steps" not in without
    assert without["report_to"] == [] and without["dataset_text_field"] == "text"
    assert without["disable_tqdm"] is True
    assert "use_cpu" not in without  # only the toy forces CPU


def test_cpu_toy_kwargs_force_cpu(tmp_path):
    kw = build_training_kwargs(load_run_config_from_file(_config(tmp_path), cpu_toy=True))
    assert kw["use_cpu"] is True and kw["bf16"] is False and kw["fp16"] is False


# ---- run-plan resolution -----------------------------------------------------


def _report(ready: bool, cpu_toy_ready: bool) -> TrainingRuntimeReport:
    return TrainingRuntimeReport(ready=ready, cpu_toy_ready=cpu_toy_ready)


def test_cpu_toy_plan_requires_cpu_toy_ready():
    cfg = _cfg(cpu_toy=True)
    plan = resolve_run_plan(cfg, _report(ready=False, cpu_toy_ready=True))
    assert plan == {"device": "cpu", "quantize": False}
    with pytest.raises(TrainerError):
        resolve_run_plan(cfg, _report(ready=False, cpu_toy_ready=False))


def test_real_plan_requires_full_ready():
    cfg = _cfg(cpu_toy=False)
    plan = resolve_run_plan(cfg, _report(ready=True, cpu_toy_ready=True))
    assert plan == {"device": "cuda", "quantize": True}
    with pytest.raises(TrainerError):
        resolve_run_plan(cfg, _report(ready=False, cpu_toy_ready=True))


def test_list_checkpoints(tmp_path):
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-2").mkdir()
    (tmp_path / "not-a-checkpoint").mkdir()
    found = _list_checkpoints(tmp_path)
    assert len(found) == 2 and all("checkpoint-" in c for c in found)


# ---- helpers -----------------------------------------------------------------


def _cfg(**overrides):
    from corpus_studio.training.trainer import TrainRunConfig

    base = {"base_model": "m", "dataset_path": "d.jsonl"}
    base.update(overrides)
    return TrainRunConfig(**base)
