"""First-party training-runtime preflight (the opt-in [train] extra).

The heavy deps (torch/transformers/bitsandbytes/…) are NOT installed in CI, so every scenario is
simulated by monkeypatching the version-lookup + GPU-probe seams — the same pattern the tokenizer
estimator uses. This proves the readiness logic without pulling multi-GB packages.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.training import environment as env

runner = CliRunner()

_FULL = ("torch", "transformers", "peft", "trl", "accelerate", "datasets", "bitsandbytes")


def _patch(monkeypatch, present: dict[str, str], gpu: env.GpuInfo) -> None:
    monkeypatch.setattr(env, "_installed_version", lambda pkg: present.get(pkg))
    monkeypatch.setattr(env, "_probe_gpu", lambda torch_present: gpu)


def test_full_stack_with_gpu_is_ready(monkeypatch):
    _patch(
        monkeypatch,
        {p: "1.0" for p in _FULL},
        env.GpuInfo(available=True, device_count=1, name="RTX 4070", total_memory_gb=12.0),
    )
    r = env.probe_training_runtime()
    assert r.ready is True
    assert r.cpu_toy_ready is True
    assert r.bitsandbytes_ok is True
    assert r.missing == []
    # 12 GB → a tight-VRAM note fires (7B QLoRA merge may OOM).
    assert any("VRAM" in n for n in r.notes)


def test_torch_without_cuda_is_cpu_toy_only(monkeypatch):
    _patch(monkeypatch, {p: "1.0" for p in _FULL}, env.GpuInfo(available=False))
    r = env.probe_training_runtime()
    assert r.ready is False  # no GPU
    assert r.cpu_toy_ready is True
    assert r.bitsandbytes_ok is False  # bnb present but no CUDA
    assert any("CUDA build" in n for n in r.notes)


def test_nothing_installed_is_not_ready(monkeypatch):
    _patch(monkeypatch, {}, env.GpuInfo())
    r = env.probe_training_runtime()
    assert r.ready is False
    assert r.cpu_toy_ready is False
    assert sorted(r.missing) == sorted(_FULL)
    assert r.install_hint.startswith("pip install")
    assert any(r.install_hint in n for n in r.notes)


def test_cpu_toy_set_without_bitsandbytes_or_gpu(monkeypatch):
    # The CPU toy path needs accelerate (Trainer dep) but NOT bitsandbytes/GPU.
    present = {
        "torch": "2.3", "transformers": "4.44", "peft": "0.11",
        "trl": "0.9", "datasets": "2.19", "accelerate": "0.30",
    }
    _patch(monkeypatch, present, env.GpuInfo(available=False))
    r = env.probe_training_runtime()
    assert r.cpu_toy_ready is True  # the CPU toy path works
    assert r.ready is False
    assert "bitsandbytes" in r.missing
    assert any("bitsandbytes" in n for n in r.notes)


def test_low_vram_gpu_warns_about_merge(monkeypatch):
    _patch(
        monkeypatch,
        {p: "1.0" for p in _FULL},
        env.GpuInfo(available=True, device_count=1, name="RTX 3060", total_memory_gb=8.0),
    )
    r = env.probe_training_runtime()
    assert any("OOM" in n or "tight" in n for n in r.notes)


def test_blackwell_gpu_notes_the_math_sdpa_fallback(monkeypatch):
    # sm_120 → the trainer forces math SDPA (the fused flash/mem-efficient kernels deadlock there).
    _patch(
        monkeypatch,
        {p: "1.0" for p in _FULL},
        env.GpuInfo(available=True, device_count=1, name="RTX 5070", total_memory_gb=12.0, compute_capability="12.0"),
    )
    r = env.probe_training_runtime()
    assert any("Blackwell" in note and "math SDPA" in note for note in r.notes)


def test_non_blackwell_gpu_has_no_sdpa_note(monkeypatch):
    _patch(
        monkeypatch,
        {p: "1.0" for p in _FULL},
        env.GpuInfo(available=True, device_count=1, name="RTX 4090", total_memory_gb=24.0, compute_capability="8.9"),
    )
    r = env.probe_training_runtime()
    assert not any("Blackwell" in note for note in r.notes)


def test_render_text_has_verdict_and_packages(monkeypatch):
    _patch(monkeypatch, {p: "1.0" for p in _FULL}, env.GpuInfo(available=True, total_memory_gb=24.0))
    text = env.render_training_runtime_text(env.probe_training_runtime())
    assert "VERDICT:" in text
    assert "torch" in text and "bitsandbytes" in text


# ---- CLI ---------------------------------------------------------------------


def test_cli_train_check_text(monkeypatch):
    _patch(monkeypatch, {}, env.GpuInfo())
    result = runner.invoke(app, ["train-check"])
    assert result.exit_code == 0, result.output
    assert "VERDICT: NOT READY" in result.output


def test_cli_train_check_json(monkeypatch):
    _patch(monkeypatch, {p: "1.0" for p in _FULL}, env.GpuInfo(available=True, total_memory_gb=24.0))
    result = runner.invoke(app, ["train-check", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ready"] is True
    assert payload["gpu"]["total_memory_gb"] == 24.0
