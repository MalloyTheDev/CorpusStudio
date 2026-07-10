"""Adapter merge — the PURE logic (strategy plan, base-model resolution, serving instructions).

No heavy imports, so it runs in CI without torch/peft. The actual merge (`_merge_on_device`) is
verified on real hardware; the OOM→CPU→adapter-only fallback is exercised by the plan/try structure.
"""

from __future__ import annotations

import json

import pytest

from corpus_studio.training import gpu_probe
from corpus_studio.training.gpu_probe import GpuMemory
from corpus_studio.training.merge import (
    MergeError,
    base_model_from_adapter,
    gpu_merge_fits,
    resolve_merge_plan,
    serving_instructions,
)


def test_auto_plan_prefers_gpu_then_cpu_then_adapter_only():
    assert resolve_merge_plan("auto", gpu_available=True) == ["gpu", "cpu", "adapter-only"]
    # No GPU → skip straight to CPU, then the always-available adapter-only fallback.
    assert resolve_merge_plan("auto", gpu_available=False) == ["cpu", "adapter-only"]


def test_auto_skips_gpu_when_the_merge_wont_fit_vram():
    # A fp16 merge too big for VRAM must NOT be tried on GPU first: on Windows the GPU attempt spills to
    # system RAM and crawls instead of cleanly OOM'ing (so the fallback never trips). Drop to CPU first.
    assert resolve_merge_plan("auto", gpu_available=True, gpu_fits=False) == ["cpu", "adapter-only"]
    assert resolve_merge_plan("auto", gpu_available=True, gpu_fits=True) == ["gpu", "cpu", "adapter-only"]


def test_gpu_merge_fits_small_model_fits_big_card(monkeypatch):
    monkeypatch.setattr(gpu_probe, "probe_gpu_memory", lambda: GpuMemory(total_gb=24.0, free_gb=22.0))
    assert gpu_merge_fits("some-1.5B-model")[0] is True  # 1.5B fp16 ≈ 3 GB + 2 = 5 GB < 22


def test_gpu_merge_fits_7b_does_not_fit_12gb_card(monkeypatch):
    monkeypatch.setattr(gpu_probe, "probe_gpu_memory", lambda: GpuMemory(total_gb=12.0, free_gb=11.6))
    fits, reason = gpu_merge_fits("Qwen/Qwen2.5-7B-Instruct")  # 14 + 2 = 16 GB > 11.6 free
    assert fits is False
    assert "16 GB" in reason and "11.6" in reason


def test_gpu_merge_fits_unknown_attempts_the_gpu(monkeypatch):
    # No nvidia-smi → can't tell → still attempt the GPU merge (prior behavior).
    monkeypatch.setattr(gpu_probe, "probe_gpu_memory", lambda: None)
    assert gpu_merge_fits("Qwen/Qwen2.5-7B")[0] is True
    # Unparseable model size → also attempt (can't estimate).
    monkeypatch.setattr(gpu_probe, "probe_gpu_memory", lambda: GpuMemory(total_gb=12.0, free_gb=11.6))
    assert gpu_merge_fits("my-custom-model")[0] is True


def test_explicit_strategy_is_tried_alone():
    assert resolve_merge_plan("gpu", gpu_available=True) == ["gpu"]
    assert resolve_merge_plan("cpu", gpu_available=True) == ["cpu"]
    assert resolve_merge_plan("adapter-only", gpu_available=True) == ["adapter-only"]


def test_unknown_strategy_raises():
    with pytest.raises(MergeError):
        resolve_merge_plan("magic", gpu_available=True)


def test_base_model_read_from_adapter_config(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct"}), encoding="utf-8"
    )
    assert base_model_from_adapter(tmp_path) == "Qwen/Qwen2.5-7B-Instruct"


def test_base_model_override_wins(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "a"}), encoding="utf-8")
    assert base_model_from_adapter(tmp_path, override="b") == "b"


def test_missing_or_empty_base_raises(tmp_path):
    with pytest.raises(MergeError):
        base_model_from_adapter(tmp_path)  # no adapter_config.json
    (tmp_path / "adapter_config.json").write_text(json.dumps({"x": 1}), encoding="utf-8")
    with pytest.raises(MergeError):
        base_model_from_adapter(tmp_path)  # no base_model_name_or_path


def test_serving_instructions_reference_base_and_adapter():
    text = serving_instructions("Qwen/Qwen2.5-7B", "/out/adapter")
    assert "Qwen/Qwen2.5-7B" in text and "/out/adapter" in text and "PeftModel" in text
