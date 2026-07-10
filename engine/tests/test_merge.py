"""Adapter merge — the PURE logic (strategy plan, base-model resolution, serving instructions).

No heavy imports, so it runs in CI without torch/peft. The actual merge (`_merge_on_device`) is
verified on real hardware; the OOM→CPU→adapter-only fallback is exercised by the plan/try structure.
"""

from __future__ import annotations

import json

import pytest

from corpus_studio.training.merge import (
    MergeError,
    base_model_from_adapter,
    resolve_merge_plan,
    serving_instructions,
)


def test_auto_plan_prefers_gpu_then_cpu_then_adapter_only():
    assert resolve_merge_plan("auto", gpu_available=True) == ["gpu", "cpu", "adapter-only"]
    # No GPU → skip straight to CPU, then the always-available adapter-only fallback.
    assert resolve_merge_plan("auto", gpu_available=False) == ["cpu", "adapter-only"]


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
