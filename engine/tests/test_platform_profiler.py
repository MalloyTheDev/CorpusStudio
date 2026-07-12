"""The platform profiler's GPU detection. Regression for a bug found by running platform-probe on a
REAL torch-free RTX 5070 host: the profile reported GPU name "unknown" because the name came only
from torch's GpuInfo (absent), while nvidia-smi — which the profiler already calls for memory — was
never asked for the name. A torch-free pre-install hardware profile must still name the device."""

import types

from corpus_studio.platform import profiler
from corpus_studio.training.environment import GpuInfo
from corpus_studio.training.gpu_probe import GpuMemory


def _report(gpu: GpuInfo) -> types.SimpleNamespace:
    return types.SimpleNamespace(gpu=gpu)


def test_gpus_names_the_device_from_nvidia_smi_when_torch_is_absent(monkeypatch):
    # Torch-free host: GpuInfo is empty (available=False, name=""), but nvidia-smi supplies name+cc.
    monkeypatch.setattr(
        profiler, "probe_training_runtime", lambda: _report(GpuInfo(available=False, name=""))
    )
    monkeypatch.setattr(
        profiler,
        "probe_gpu_memory",
        lambda: GpuMemory(
            total_gb=11.9, free_gb=11.1, compute_capability="12.0", name="NVIDIA GeForce RTX 5070"
        ),
    )
    gpus = profiler._gpus()
    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA GeForce RTX 5070"  # NOT "unknown"
    assert gpus[0].compute_capability_major == 12
    assert gpus[0].vram_total_bytes == int(11.9 * 1e9)


def test_gpus_prefers_the_torch_name_when_present(monkeypatch):
    # When torch IS available, its GpuInfo name wins over nvidia-smi's.
    monkeypatch.setattr(
        profiler,
        "probe_training_runtime",
        lambda: _report(
            GpuInfo(available=True, name="Torch Name", compute_capability="8.0", total_memory_gb=24.0)
        ),
    )
    monkeypatch.setattr(
        profiler,
        "probe_gpu_memory",
        lambda: GpuMemory(total_gb=24.0, free_gb=20.0, compute_capability="8.0", name="smi name"),
    )
    assert profiler._gpus()[0].name == "Torch Name"


def test_gpus_empty_when_no_accelerator_at_all(monkeypatch):
    monkeypatch.setattr(
        profiler, "probe_training_runtime", lambda: _report(GpuInfo(available=False, name=""))
    )
    monkeypatch.setattr(profiler, "probe_gpu_memory", lambda: None)
    assert profiler._gpus() == []
