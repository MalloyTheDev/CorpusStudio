"""GPU health — the wedged-GPU detector born from real WSL2 testing: a crashed CUDA process poisons
the GPU-PV state so every subsequent run fails 'device not ready' until a reset. These lock the pure
classifier + the OS-specific reset remediation + the spill guidance."""

from corpus_studio.platform.enums import FitClass, OperatingSystem
from corpus_studio.platform.gpu_health import (
    classify_gpu_health,
    probe_gpu_responsive,
    spill_remediation,
    wedged_gpu_remediation,
)


def test_classify_healthy_when_probe_ran():
    assert classify_gpu_health(None) == "healthy"


def test_classify_wedged_fingerprints():
    for err in (
        "CUDA error: device not ready",
        "cudaErrorNotReady",
        "an illegal memory access was encountered",
        "CUDA error: unspecified launch failure",
        "misaligned address",
    ):
        assert classify_gpu_health(err) == "wedged", err


def test_classify_absent_is_not_a_wedge():
    # No GPU / no torch is 'absent' — nothing to reset — NOT a wedge (would mislead the operator).
    for err in ("no CUDA GPU available", "torch not importable: No module named 'torch'",
                "CUDA driver version is insufficient"):
        assert classify_gpu_health(err) == "absent", err


def test_classify_unknown_bucket():
    assert classify_gpu_health("some unrecognized runtime error") == "unknown"


def test_wedged_remediation_is_os_specific():
    wsl = wedged_gpu_remediation(OperatingSystem.wsl, wsl_distro="Ubuntu")
    assert "wsl --terminate Ubuntu" in wsl
    win = wedged_gpu_remediation(OperatingSystem.windows)
    assert "wsl --terminate" not in win and "driver" in win.lower()
    lin = wedged_gpu_remediation(OperatingSystem.linux)
    assert "nvidia-smi" in lin
    assert "Ubuntu" in wedged_gpu_remediation(OperatingSystem.wsl)  # sensible distro default


def test_spill_remediation_only_for_spill_classes():
    assert spill_remediation(FitClass.NATIVE_SAFE, OperatingSystem.wsl) == ""
    assert spill_remediation(FitClass.NATIVE_TIGHT, OperatingSystem.windows) == ""
    spill = spill_remediation(FitClass.ACCIDENTAL_WDDM_SPILL, OperatingSystem.windows)
    assert "train" in spill.lower() and "slower" in spill.lower()  # "will train, but crawl"
    # bare Linux hard-OOMs where WDDM spills — the guidance must say so
    assert "OOM" in spill_remediation(FitClass.ACCIDENTAL_WDDM_SPILL, OperatingSystem.linux)
    assert "OOM" not in spill_remediation(FitClass.THRASHING, OperatingSystem.windows)


def test_probe_gpu_responsive_degrades_cleanly_without_torch():
    # In the torch-free engine gate the probe must return an 'absent' signal, never a false 'wedged'
    # (which would tell CI to reset a GPU that isn't there) or a false 'healthy'.
    assert classify_gpu_health(probe_gpu_responsive()) in ("absent", "healthy")
