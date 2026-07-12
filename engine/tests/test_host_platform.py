"""The shared host-platform detector: WSL is a DISTINCT platform (flash-safe like Linux, wddm spill
like Windows), and the fused flash SDPA deadlock is native-Windows-Blackwell ONLY. Pure/deterministic
— every host-sensitive path is monkeypatched, so it runs identically on the Windows dev venv and the
Linux CI runner."""

import types

from corpus_studio.platform.enums import MemoryResidencyModel, OperatingSystem
from corpus_studio.platform.host_platform import (
    detect_operating_system,
    flash_sdpa_deadlocks,
    is_native_windows,
    is_wsl,
)


def _uname(release: str):
    return types.SimpleNamespace(release=release)


def test_flash_sdpa_deadlocks_is_native_windows_blackwell_only():
    # The deadlock fires ONLY on native Windows + Blackwell (sm_120, cc_major>=12).
    assert flash_sdpa_deadlocks(OperatingSystem.windows, 12) is True
    assert flash_sdpa_deadlocks(OperatingSystem.windows, 13) is True
    # WSL / bare Linux on Blackwell → flash is SAFE (verified on a real 5070 under WSL2).
    assert flash_sdpa_deadlocks(OperatingSystem.wsl, 12) is False
    assert flash_sdpa_deadlocks(OperatingSystem.linux, 12) is False
    # Older arch on Windows, or no GPU info → safe.
    assert flash_sdpa_deadlocks(OperatingSystem.windows, 9) is False
    assert flash_sdpa_deadlocks(OperatingSystem.windows, None) is False
    # Unknown host → never disable a kernel we can't prove is on the WDDM path.
    assert flash_sdpa_deadlocks(None, 12) is False


def test_is_native_windows_distinguishes_wsl():
    assert is_native_windows(OperatingSystem.windows) is True
    assert is_native_windows(OperatingSystem.wsl) is False  # the crux: WSL is NOT native Windows
    assert is_native_windows(OperatingSystem.linux) is False


def test_detect_windows(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Windows")
    assert detect_operating_system() == (OperatingSystem.windows, MemoryResidencyModel.wddm)
    assert is_wsl() is False


def test_detect_macos(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    assert detect_operating_system() == (OperatingSystem.macos, MemoryResidencyModel.unified_memory)


def test_detect_wsl_via_env_marker(monkeypatch):
    # WSL sets WSL_DISTRO_NAME. It is its OWN platform: os=wsl, residency=wddm (spills like Windows).
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    assert is_wsl() is True
    assert detect_operating_system() == (OperatingSystem.wsl, MemoryResidencyModel.wddm)


def test_detect_wsl_via_kernel_release(monkeypatch):
    # The WSL2 kernel release carries "microsoft"/"WSL"; no env marker needed.
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    monkeypatch.setattr("platform.uname", lambda: _uname("5.15.153.1-microsoft-standard-WSL2"))
    assert is_wsl() is True
    assert detect_operating_system()[0] == OperatingSystem.wsl


def test_detect_bare_linux_is_not_wsl(monkeypatch):
    # A generic Linux kernel with no WSL markers → bare Linux (linux_dedicated: hard-OOM, not spill).
    # /proc/version resolves non-WSL on the CI ubuntu runner and errors on the Windows venv — both → False.
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    monkeypatch.setattr("platform.uname", lambda: _uname("5.15.0-91-generic"))
    assert is_wsl() is False
    assert detect_operating_system() == (OperatingSystem.linux, MemoryResidencyModel.linux_dedicated)
