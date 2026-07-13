"""Runtime host-platform detection — the single source of truth for OS, WSL, the GPU memory
residency model, and the Blackwell flash-SDPA deadlock. Pure stdlib; no torch, no heavy imports
(``import corpus_studio.platform`` stays torch-free).

WSL is a DISTINCT platform, not "Windows" and not bare "Linux" (see :class:`OperatingSystem.wsl`).
The fused FLASH SDPA backward that DEADLOCKS on native Windows (WDDM) has separate passing evidence
under WSL2 on an RTX 5070. WSL GPU memory still spills to shared system RAM through the host WDDM
driver (``wddm`` residency). The known-deadlock guard therefore fires on native Windows only; every
other host still needs its own capability result. Bare-Linux RTX 5070 behavior remains unverified.
"""

from __future__ import annotations

import os
import platform

from corpus_studio.platform.enums import MemoryResidencyModel, OperatingSystem

# Blackwell (RTX 50-series) is sm_120 → compute-capability major 12: the arch whose fused FLASH SDPA
# backward deadlocks under the Windows WDDM driver model.
_BLACKWELL_CAPABILITY_MAJOR = 12


def is_wsl() -> bool:
    """True when running under WSL (a Linux kernel hosted by Windows). Detection must never raise."""
    if platform.system() != "Linux":
        return False
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        release = platform.uname().release.lower()
        if "microsoft" in release or "wsl" in release:
            return True
    except Exception:  # noqa: BLE001 - a probe fault must not crash environment detection.
        pass
    try:
        with open("/proc/version", encoding="utf-8", errors="ignore") as handle:
            body = handle.read().lower()
        return "microsoft" in body or "wsl" in body
    except OSError:
        return False


def detect_operating_system() -> tuple[OperatingSystem, MemoryResidencyModel]:
    """Map the running host to ``(OperatingSystem, MemoryResidencyModel)``. WSL is its own platform:
    ``os=wsl`` with ``wddm`` residency (spills via the host, like Windows). Capability remains a
    separate probe result, not an implication of this OS label."""
    system = platform.system()
    if system == "Windows":
        return OperatingSystem.windows, MemoryResidencyModel.wddm
    if system == "Linux":
        if is_wsl():
            return OperatingSystem.wsl, MemoryResidencyModel.wddm
        return OperatingSystem.linux, MemoryResidencyModel.linux_dedicated
    if system == "Darwin":
        return OperatingSystem.macos, MemoryResidencyModel.unified_memory
    return OperatingSystem.unknown, MemoryResidencyModel.unknown


def is_native_windows(os_value: OperatingSystem | None = None) -> bool:
    """Native Windows (WDDM) — explicitly NOT WSL. Auto-detects the running host when ``os_value`` is
    None. This is the exact condition under which the fused flash SDPA kernel must be disabled."""
    if os_value is None:
        os_value, _ = detect_operating_system()
    return os_value == OperatingSystem.windows


def flash_sdpa_deadlocks(os_value: OperatingSystem | None, cc_major: int | None) -> bool:
    """The fused FLASH SDPA backward deadlocks ONLY on native Windows (WDDM) + Blackwell (sm_120,
    ``cc_major >= 12``). Other platforms are not special-cased by this KNOWN-HAZARD predicate; that is
    not a positive capability claim, and their flash probe must still pass. ``os_value=None`` is
    treated as 'not native Windows' because the WDDM condition is not established."""
    return (
        os_value == OperatingSystem.windows
        and cc_major is not None
        and cc_major >= _BLACKWELL_CAPABILITY_MAJOR
    )
