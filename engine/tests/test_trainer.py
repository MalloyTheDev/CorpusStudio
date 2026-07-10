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
    build_lora_kwargs,
    build_training_kwargs,
    format_example_text,
    load_run_config_from_file,
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
