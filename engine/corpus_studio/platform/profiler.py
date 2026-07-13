"""Build an :class:`EnvironmentProfile` contract from cheap host probes.

This is the environment MANAGER's read side: it characterizes the current host + software environment
and produces the hashable :class:`~corpus_studio.platform.contracts.EnvironmentProfile` the planner and
worker protocol reference. It REUSES the existing dependency-light probes
(``training.environment.probe_training_runtime`` for GPU + package versions, ``training.gpu_probe``
for free VRAM) — it does not import torch and it does not duplicate that logic. Env *creation*/locking
is a later slice; this only reads the current process's environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

from ..training.environment import probe_training_runtime
from ..training.gpu_probe import probe_gpu_memory
from .common import PackageLock
from .contracts import (
    AcceleratorRuntime,
    EnvCpu,
    EnvHost,
    EnvironmentProfile,
    EnvRam,
    GpuDevice,
)
from .enums import DeviceKind, MemoryResidencyModel, OperatingSystem, PrecisionMode
from .host_platform import detect_operating_system

# The dependency stack the planner + backends care about. Distribution (PyPI) names — read from
# installed-package metadata, never imported. Extends training.environment._TRAIN_PACKAGES with the
# accelerator/kernel stack a full profile needs.
PROFILE_PACKAGES: tuple[str, ...] = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "bitsandbytes",
    "triton",
    "liger-kernel",
    "unsloth",
    "flash-attn",
    "deepspeed",
    "torchao",
    "accelerate",
    "datasets",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _installed_version(package: str) -> str | None:
    """Installed version from metadata, or None — without importing the package."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - a broken metadata entry must not crash the profile.
        return None


def _capability_major(compute_capability: str) -> int | None:
    """Major compute capability from a ``"12.0"``-style string, or None when unknown."""
    try:
        return int(compute_capability.split(".")[0])
    except (ValueError, AttributeError, IndexError):
        return None


def _operating_system() -> tuple[OperatingSystem, MemoryResidencyModel]:
    """Map the host OS to the contract enum + the memory-residency model that decides spill-vs-OOM.
    Delegates to the shared detector so WSL is recognized as its own platform with WDDM residency.
    Its separately measured flash evidence is not generalized to bare Linux."""
    return detect_operating_system()


def _system_ram_bytes() -> tuple[int | None, int | None]:
    """Best-effort (total, available) system RAM in bytes. Tries psutil (optional), then POSIX
    ``sysconf``, then the Windows ``GlobalMemoryStatusEx`` ctypes call. Returns ``(None, None)`` when
    none resolve — an honest 'unknown', never a crash."""
    try:
        import psutil  # noqa: PLC0415 - optional; absent in the dependency-light core.

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:  # noqa: BLE001
        pass
    try:  # POSIX
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")  # type: ignore[attr-defined]
        return int(total), None
    except (ValueError, AttributeError, OSError):
        pass
    try:  # Windows
        import ctypes  # noqa: PLC0415

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
    except Exception:  # noqa: BLE001
        pass
    return None, None


def _supported_dtypes(cc_major: int | None) -> list[PrecisionMode]:
    """The precision modes a CUDA GPU of this compute-capability major supports natively."""
    if cc_major is None:
        return []
    dtypes = [PrecisionMode.fp32, PrecisionMode.fp16]
    if cc_major >= 8:  # Ampere+: tf32 + bf16
        dtypes += [PrecisionMode.tf32, PrecisionMode.bf16]
    if cc_major >= 9:  # Hopper+: fp8
        dtypes.append(PrecisionMode.fp8)
    return dtypes


def _gpus(gb_to_bytes: float = 1e9) -> list[GpuDevice]:
    """The first detected accelerator, from the existing GpuInfo + nvidia-smi free-memory probes.
    Empty when no CUDA GPU is detected. (Multi-GPU enumeration is a later slice.)"""
    report = probe_training_runtime()
    mem = probe_gpu_memory()
    info = report.gpu
    if not info.available and mem is None:
        return []
    cc = info.compute_capability or (mem.compute_capability if mem else "") or ""
    cc_major = _capability_major(cc)
    total_gb = info.total_memory_gb or (mem.total_gb if mem else 0.0)
    free_bytes = int(mem.free_gb * gb_to_bytes) if mem is not None else None
    # Name from torch's GpuInfo when present, else nvidia-smi's name (so a torch-free Blackwell host
    # still reports "RTX 5070", not "unknown" — the whole point of a pre-install hardware profile).
    name = info.name or (mem.name if mem else "") or "unknown"
    return [
        GpuDevice(
            index=0,
            kind=DeviceKind.cuda,  # nvidia-smi / torch.cuda implies a CUDA device
            name=name,
            vram_total_bytes=int(total_gb * gb_to_bytes) if total_gb else None,
            vram_free_bytes=free_bytes,
            compute_capability=cc or None,
            compute_capability_major=cc_major,
            supported_dtypes=_supported_dtypes(cc_major),
        )
    ]


def _package_locks() -> list[PackageLock]:
    return [PackageLock(name=pkg, version=_installed_version(pkg)) for pkg in PROFILE_PACKAGES]


def _environment_signature(
    os_enum: OperatingSystem,
    residency: MemoryResidencyModel,
    python_version: str,
    cpu_model: str,
    gpus: list[GpuDevice],
    packages: list[PackageLock],
) -> str:
    """Deterministic sha256 over the STABLE host + package fields. Excludes volatile values (free
    memory, timestamps, available RAM) so the same machine + env hashes identically across runs."""
    payload = {
        "os": os_enum.value,
        "residency": residency.value,
        "python": python_version,
        "cpu": cpu_model,
        "gpus": sorted(
            [
                {
                    "name": g.name,
                    "cc": g.compute_capability or "",
                    "vram_total": g.vram_total_bytes or 0,
                }
                for g in gpus
            ],
            key=lambda d: (d["name"], d["cc"]),
        ),
        "packages": sorted([(p.name, p.version or "") for p in packages]),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_environment_profile() -> EnvironmentProfile:
    """Characterize the current host + software environment as a hashable ``EnvironmentProfile``.

    Pure w.r.t. the project filesystem and dependency-light: it imports no torch (the GPU details come
    from the existing torch-lazy / nvidia-smi probes). The ``environment_signature`` is stable across
    runs on the same machine/env; ``captured_at`` and free-memory fields are recorded but excluded
    from the signature.
    """
    os_enum, residency = _operating_system()
    python_version = platform.python_version()
    cpu_model = platform.processor() or platform.machine() or ""
    total_ram, avail_ram = _system_ram_bytes()
    gpus = _gpus()
    packages = _package_locks()

    nvidia_smi_present = any(g.kind == DeviceKind.cuda for g in gpus)
    signature = _environment_signature(
        os_enum, residency, python_version, cpu_model, gpus, packages
    )
    return EnvironmentProfile(
        environment_signature=signature,
        captured_at=_now_iso(),
        host=EnvHost(
            os=os_enum,
            os_detail=platform.platform(),
            memory_residency_model=residency,
            python_version=python_version,
        ),
        cpu=EnvCpu(model=cpu_model, logical_cores=os.cpu_count()),
        ram=EnvRam(total_bytes=total_ram, available_bytes=avail_ram),
        gpus=gpus,
        accelerator_runtime=AcceleratorRuntime(
            kind=DeviceKind.cuda if gpus else None,
            nvidia_smi_available=nvidia_smi_present,
        ),
        packages=packages,
    )
