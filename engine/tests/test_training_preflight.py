"""Training pre-flight: cheap fail-fast checks before a long launch."""

from pathlib import Path

from corpus_studio.training import gpu_probe
from corpus_studio.training.gpu_probe import GpuMemory
from corpus_studio.training.preflight import (
    BLOCK,
    PASS,
    WARN,
    run_training_preflight,
)


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text("base_model: x\n", encoding="utf-8")
    return path


def _data(tmp_path: Path, name: str = "train.jsonl") -> Path:
    path = tmp_path / name
    path.write_text('{"instruction":"a","output":"b"}\n', encoding="utf-8")
    return path


def _run(tmp_path, **overrides):
    kwargs = dict(
        config_path=_config(tmp_path),
        launch_argv=["python", str(tmp_path / "config.yaml")],  # python is on PATH
        dependencies=["trl", "transformers", "torch"],
        data_paths=[_data(tmp_path)],
        dataset_row_count=100,
        examples_over_sequence_len=0,
        sequence_len=2048,
    )
    kwargs.update(overrides)
    return run_training_preflight(**kwargs)


def test_all_good_passes_and_can_launch(tmp_path: Path):
    report = _run(tmp_path)
    assert report.status == PASS
    assert report.can_launch is True
    assert report.trainer_found is True


def test_missing_trainer_warns_but_does_not_block(tmp_path: Path):
    report = _run(tmp_path, launch_argv=["definitely-not-a-real-trainer-xyz", "cfg"])
    assert report.status == WARN
    assert report.can_launch is True  # warn, not block — the desktop may spawn in another env
    assert report.trainer_found is False
    trainer_check = next(c for c in report.checks if c.name == "trainer_available")
    assert trainer_check.status == WARN
    assert "transformers" in trainer_check.message  # names the target's deps


def test_missing_data_file_blocks(tmp_path: Path):
    report = _run(tmp_path, data_paths=[tmp_path / "does_not_exist.jsonl"])
    assert report.status == BLOCK
    assert report.can_launch is False


def test_empty_dataset_blocks(tmp_path: Path):
    report = _run(tmp_path, dataset_row_count=0)
    assert report.status == BLOCK
    assert report.can_launch is False


def test_tiny_dataset_warns(tmp_path: Path):
    report = _run(tmp_path, dataset_row_count=3)
    assert report.status == WARN
    assert report.can_launch is True


def test_truncation_warns(tmp_path: Path):
    report = _run(tmp_path, examples_over_sequence_len=5)
    assert report.status == WARN
    seq_check = next(c for c in report.checks if c.name == "sequence_length")
    assert seq_check.status == WARN
    assert "truncated" in seq_check.message


def test_missing_config_blocks(tmp_path: Path):
    report = _run(tmp_path, config_path=tmp_path / "nope.yaml")
    assert report.status == BLOCK
    assert report.can_launch is False


# --- OOM / GPU-memory realism (nvidia-smi probe) -----------------------------


def _gpu(monkeypatch, memory):
    monkeypatch.setattr(gpu_probe, "probe_gpu_memory", lambda: memory)


def test_no_gpu_detected_skips_the_oom_check(tmp_path: Path, monkeypatch):
    _gpu(monkeypatch, None)  # nvidia-smi not available
    report = _run(tmp_path, vram_min_gb=48.0)
    assert not any(check.name == "gpu_memory" for check in report.checks)


def test_oom_warns_when_the_estimate_exceeds_free_vram(tmp_path: Path, monkeypatch):
    _gpu(monkeypatch, GpuMemory(total_gb=24.0, free_gb=23.0))
    report = _run(tmp_path, vram_min_gb=40.0)  # 40 GB needed, 23 free
    gpu_check = next(c for c in report.checks if c.name == "gpu_memory")
    assert gpu_check.status == WARN
    assert "Likely OOM" in gpu_check.message
    assert report.can_launch is True  # OOM is a warning, not a hard block


def test_oom_passes_when_it_fits(tmp_path: Path, monkeypatch):
    _gpu(monkeypatch, GpuMemory(total_gb=24.0, free_gb=22.0))
    report = _run(tmp_path, vram_min_gb=8.0)  # 8 GB needed, 22 free
    gpu_check = next(c for c in report.checks if c.name == "gpu_memory")
    assert gpu_check.status == PASS
