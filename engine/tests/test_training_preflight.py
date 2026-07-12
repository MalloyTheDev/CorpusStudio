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


def test_first_party_trainer_passes_without_a_path_check(tmp_path: Path):
    # For the corpus_studio target the "trainer" is Corpus Studio itself (run via -m corpus_studio.cli),
    # not an executable on the global PATH — the check should PASS (and point at train-check), not warn.
    report = _run(
        tmp_path,
        launch_argv=["corpus-studio", "train-run", "config.json"],
        dependencies=["corpus-studio-engine[train]"],
    )
    check = next(c for c in report.checks if c.name == "trainer_available")
    assert check.status == PASS
    assert "train-check" in check.message
    assert report.trainer_found is True


def test_external_trainer_not_on_path_still_warns(tmp_path: Path):
    report = _run(
        tmp_path,
        launch_argv=["definitely-not-a-real-trainer-xyz-123", "c.yaml"],
        dependencies=["axolotl"],
    )
    check = next(c for c in report.checks if c.name == "trainer_available")
    assert check.status == WARN


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


# --- nvidia-smi CSV parsing (incl. the GPU name, so a torch-free host names the device) ------------


def test_probe_gpu_memory_parses_the_gpu_name(monkeypatch):
    import subprocess as sp

    monkeypatch.setattr("shutil.which", lambda _name: "nvidia-smi")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: sp.CompletedProcess(
            a, 0, stdout="11900, 11100, 12.0, NVIDIA GeForce RTX 5070\n", stderr=""
        ),
    )
    mem = gpu_probe.probe_gpu_memory()
    assert mem is not None
    assert mem.name == "NVIDIA GeForce RTX 5070"
    assert mem.compute_capability == "12.0"
    assert mem.total_gb == round(11900 / 1024, 1)


def test_probe_gpu_memory_tolerates_older_driver_without_name(monkeypatch):
    # An older nvidia-smi emits only memory columns; name/compute_cap default to empty, never crash.
    import subprocess as sp

    monkeypatch.setattr("shutil.which", lambda _name: "nvidia-smi")
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: sp.CompletedProcess(a, 0, stdout="11900, 11100\n", stderr="")
    )
    mem = gpu_probe.probe_gpu_memory()
    assert mem is not None
    assert mem.name == ""
    assert mem.compute_capability == ""


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
    assert "Won't fit" in gpu_check.message
    assert "SPILLS" in gpu_check.message  # Windows WDDM spill, not a clean OOM/deadlock
    assert report.can_launch is True  # a warning, not a hard block


def test_oom_passes_when_it_fits(tmp_path: Path, monkeypatch):
    _gpu(monkeypatch, GpuMemory(total_gb=24.0, free_gb=22.0))
    report = _run(tmp_path, vram_min_gb=8.0)  # 8 GB needed, 22 free
    gpu_check = next(c for c in report.checks if c.name == "gpu_memory")
    assert gpu_check.status == PASS


def test_tight_vram_warns_even_when_it_technically_fits(tmp_path: Path, monkeypatch):
    # A run that only "just fits" can VRAM-pressure DEADLOCK (not cleanly OOM): a real 7B/4-bit/seq-4096
    # run peaked ~11.9 GB and hung a 12 GB card at 58 MiB free. Estimate 11.5 GB with 12 GB free is
    # within the safety margin → WARN, not the old green PASS that green-lit the deadlock.
    _gpu(monkeypatch, GpuMemory(total_gb=12.0, free_gb=12.0))
    report = _run(tmp_path, vram_min_gb=11.5)
    gpu_check = next(c for c in report.checks if c.name == "gpu_memory")
    assert gpu_check.status == WARN
    assert "Tight VRAM" in gpu_check.message
    assert report.can_launch is True  # a warning, not a hard block


def test_capability_major_parsing():
    from corpus_studio.training.gpu_probe import _capability_major

    assert _capability_major("12.0") == 12  # Blackwell / sm_120
    assert _capability_major("8.9") == 8  # Ada
    assert _capability_major("") == 0  # unknown / older nvidia-smi
    assert _capability_major("garbage") == 0


def test_native_windows_blackwell_gpu_checks_against_the_higher_math_estimate(tmp_path: Path, monkeypatch):
    # On NATIVE WINDOWS + Blackwell (sm_120) the trainer is forced onto the math attention path (WDDM
    # flash deadlock), which uses MORE VRAM (seq² scores). The pre-flight must check the higher math
    # estimate there — so a config that "fits" on the flash estimate can still warn on a 12 GB card.
    monkeypatch.setattr("corpus_studio.training.preflight.sys.platform", "win32")
    _gpu(monkeypatch, GpuMemory(total_gb=12.0, free_gb=11.6, compute_capability="12.0"))
    report = _run(tmp_path, vram_min_gb=10.7, vram_min_gb_math=12.1)  # flash ~fits, math over the ceiling
    gpu_check = next(c for c in report.checks if c.name == "gpu_memory")
    assert gpu_check.status == WARN
    assert "math attention" in gpu_check.message.lower()


def test_wsl_blackwell_gpu_uses_the_flash_estimate(tmp_path: Path, monkeypatch):
    # On WSL/Linux + Blackwell flash works, so the pre-flight must use the FLASH estimate even on
    # sm_120 — NOT the higher math estimate native Windows would. A config that comfortably fits on
    # flash must PASS here (no over-warning), and the message must not claim math attention is forced.
    monkeypatch.setattr("corpus_studio.training.preflight.sys.platform", "linux")
    _gpu(monkeypatch, GpuMemory(total_gb=12.0, free_gb=11.6, compute_capability="12.0"))
    report = _run(tmp_path, vram_min_gb=8.0, vram_min_gb_math=12.1)  # flash fits with headroom; math would warn
    gpu_check = next(c for c in report.checks if c.name == "gpu_memory")
    assert gpu_check.status == PASS
    assert "math attention" not in gpu_check.message.lower()  # the WSL host is NOT on the math path


def test_non_blackwell_gpu_uses_the_flash_estimate(tmp_path: Path, monkeypatch):
    # Where flash attention works, the flash estimate drives the check even if a math estimate is given.
    _gpu(monkeypatch, GpuMemory(total_gb=24.0, free_gb=22.0, compute_capability="8.9"))
    report = _run(tmp_path, vram_min_gb=8.0, vram_min_gb_math=20.0)
    gpu_check = next(c for c in report.checks if c.name == "gpu_memory")
    assert gpu_check.status == PASS  # 8 GB flash estimate fits 22 free; the math estimate is ignored
    assert "math attention" not in gpu_check.message.lower()
