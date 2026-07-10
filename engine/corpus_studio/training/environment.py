"""First-party training-runtime detection (the opt-in ``[train]`` extra).

The engine is dependency-light: data prep, gates, and config export need NO training deps. When the
``[train]`` extra is installed, CorpusStudio can *run* the QLoRA itself. This module reports what is
available **without importing anything heavy at module load** — dependency *presence + version* is
read from installed-package metadata (no import), and the GPU probe imports ``torch`` only when it is
present and only inside a guarded call. So importing this module (and the whole engine) stays cheap
and never fails when torch/transformers/etc. are absent.

Two readiness levels are distinguished:

* ``cpu_toy_ready`` — the tiny CPU smoke path can run (torch + transformers + trl + peft + datasets).
  No GPU, no bitsandbytes. Lets the training *plumbing* be exercised without a GPU.
* ``ready`` — a real 4-bit QLoRA GPU run is possible (the CPU set **plus** accelerate + bitsandbytes
  **plus** an available CUDA GPU).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from pydantic import BaseModel, Field

INSTALL_HINT = "pip install corpus-studio-engine[train]"

# Distribution names probed for the report, in display order.
_TRAIN_PACKAGES: tuple[str, ...] = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "datasets",
    "bitsandbytes",
)
# The subset needed for the CPU toy path (no quantization, no GPU). accelerate is required because
# the HuggingFace Trainer/TRL SFTTrainer depend on it even on CPU.
_CPU_TOY_PACKAGES: tuple[str, ...] = ("torch", "transformers", "peft", "trl", "datasets", "accelerate")
# A 7B in fp16 ≈ 14 GB, so the final fp16 MERGE won't fit a card below ~16 GB (4-bit QLoRA training
# itself needs far less). Below this we warn that the merge may OOM and the fallback kicks in.
_MERGE_TIGHT_VRAM_GB = 16.0


class GpuInfo(BaseModel):
    available: bool = False
    device_count: int = 0
    name: str = ""
    total_memory_gb: float = 0.0
    compute_capability: str = ""  # e.g. "12.0" for Blackwell / sm_120


class TrainingRuntimeReport(BaseModel):
    """What the local machine can do for first-party training. Counts/flags only — no secrets."""

    installed: dict[str, str | None] = Field(default_factory=dict)  # package -> version or None
    missing: list[str] = Field(default_factory=list)
    gpu: GpuInfo = Field(default_factory=GpuInfo)
    bitsandbytes_ok: bool = False  # 4-bit QLoRA available (bitsandbytes present AND a CUDA GPU)
    cpu_toy_ready: bool = False  # the tiny CPU smoke path can run
    ready: bool = False  # a real GPU QLoRA run is possible
    notes: list[str] = Field(default_factory=list)
    install_hint: str = INSTALL_HINT


def _installed_version(package: str) -> str | None:
    """Installed version from package metadata, or None — without importing the package."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - a broken metadata entry must not crash the preflight.
        return None


def _capability_major(compute_capability: str) -> int:
    """Major compute capability from a ``"12.0"``-style string, or 0 when unknown."""
    try:
        return int(compute_capability.split(".")[0])
    except (ValueError, AttributeError, IndexError):
        return 0


def _probe_gpu(torch_present: bool) -> GpuInfo:
    """Best-effort CUDA probe. Imports torch only when present, fully guarded (a torch/driver
    mismatch must degrade to 'no GPU', never crash)."""
    if not torch_present:
        return GpuInfo()
    try:
        import torch  # noqa: PLC0415 - intentionally lazy: torch is heavy and optional.

        if not torch.cuda.is_available():
            return GpuInfo(available=False)
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        try:
            major, minor = torch.cuda.get_device_capability(index)
            capability = f"{major}.{minor}"
        except Exception:  # noqa: BLE001 - capability is advisory; never fail the probe on it.
            capability = ""
        return GpuInfo(
            available=True,
            device_count=torch.cuda.device_count(),
            name=str(props.name),
            total_memory_gb=round(props.total_memory / 1e9, 1),
            compute_capability=capability,
        )
    except Exception:  # noqa: BLE001 - any probe failure means 'treat as no usable GPU'.
        return GpuInfo()


def probe_training_runtime() -> TrainingRuntimeReport:
    """Detect the local training runtime. Pure w.r.t. the filesystem/project; reads only the Python
    environment. Safe to call even with none of the ``[train]`` deps installed."""
    installed = {pkg: _installed_version(pkg) for pkg in _TRAIN_PACKAGES}
    missing = [pkg for pkg, ver in installed.items() if ver is None]

    gpu = _probe_gpu(torch_present=installed.get("torch") is not None)
    bitsandbytes_ok = installed.get("bitsandbytes") is not None and gpu.available
    cpu_toy_ready = all(installed.get(pkg) is not None for pkg in _CPU_TOY_PACKAGES)
    ready = (
        not missing  # the whole [train] set is present
        and gpu.available
        and bitsandbytes_ok
    )

    notes: list[str] = []
    if missing:
        notes.append(f"Missing {len(missing)} training package(s): {', '.join(missing)}. {INSTALL_HINT}")
    if installed.get("torch") is not None and not gpu.available:
        notes.append(
            "torch is installed but no CUDA GPU is available — a default pip installs the CPU build. "
            "Install the CUDA build (https://pytorch.org) for GPU training; the CPU toy path still works."
        )
    if installed.get("bitsandbytes") is None:
        notes.append("bitsandbytes not installed — 4-bit QLoRA is unavailable (it is CUDA-only).")
    if gpu.available and 0 < gpu.total_memory_gb < _MERGE_TIGHT_VRAM_GB:
        notes.append(
            f"GPU has {gpu.total_memory_gb} GB VRAM — a 7B fp16 merge (~14 GB) is tight and may OOM; "
            "the fallback is a CPU-offload merge or adapter-only serving."
        )
    if gpu.available and _capability_major(gpu.compute_capability) >= 12:
        notes.append(
            f"Blackwell GPU (sm_{gpu.compute_capability.replace('.', '')}): the trainer forces the math SDPA "
            "attention path — the fused flash/mem-efficient kernels deadlock on the first backward on this "
            "arch. Math attention uses more VRAM than flash, so a long sequence_len is tighter here."
        )
    if ready:
        notes.append("Ready: a 4-bit QLoRA GPU run is possible.")
    elif cpu_toy_ready:
        notes.append("The CPU toy-training path is available (real GPU training still needs a CUDA GPU + bitsandbytes).")

    return TrainingRuntimeReport(
        installed=installed,
        missing=missing,
        gpu=gpu,
        bitsandbytes_ok=bitsandbytes_ok,
        cpu_toy_ready=cpu_toy_ready,
        ready=ready,
        notes=notes,
    )


def render_training_runtime_text(report: TrainingRuntimeReport) -> str:
    """Human-readable preflight (for the CLI / a screenshot). The JSON report is the machine form."""
    lines = ["Training runtime check"]
    for pkg in _TRAIN_PACKAGES:
        ver = report.installed.get(pkg)
        lines.append(f"  {'OK ' if ver else '-- '} {pkg:<14} {ver or 'not installed'}")
    if report.gpu.available:
        lines.append(f"  GPU: {report.gpu.name} ({report.gpu.total_memory_gb} GB, {report.gpu.device_count} device(s))")
    else:
        lines.append("  GPU: none detected")
    verdict = "READY (GPU QLoRA)" if report.ready else ("CPU-TOY ONLY" if report.cpu_toy_ready else "NOT READY")
    lines.append(f"VERDICT: {verdict}")
    lines += [f"  • {note}" for note in report.notes]
    return "\n".join(lines)
