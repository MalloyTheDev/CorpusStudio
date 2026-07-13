"""GPU health — turn a cryptic CUDA failure into an actionable diagnosis.

Born from real-hardware testing: heavy iterative 7B runs on WSL2 can WEDGE the GPU-PV state so that
EVERY subsequent process fails with ``cudaErrorNotReady`` ("device not ready") — confusingly, because
the config is fine; the GPU just needs a RESET (``wsl --terminate <distro>``). Without this, the
platform emits a cascade of identical cryptic errors and the operator can't tell "my config is too
big" from "my GPU is wedged". This module recognises the wedged-GPU fingerprint and produces the
OS-specific reset remediation, plus actionable guidance for a memory spill.

Pure stdlib for the classifiers; :func:`probe_gpu_responsive` lazily runs one tiny CUDA op and never
raises — importing this module pulls no torch.
"""

from __future__ import annotations

from corpus_studio.platform.enums import FitClass, OperatingSystem

# Error-string fingerprints of a GPU/driver in a bad TRANSIENT state that a RESET fixes — NOT a config
# bug. `device not ready` (cudaErrorNotReady) is the WSL2 GPU-PV wedge after a crashed process; the
# others are the sibling "the context is poisoned, reset it" errors.
_WEDGED_MARKERS = (
    "device not ready",
    "cudaerrornotready",
    "an illegal memory access",
    "device-side assert",
    "misaligned address",
    "unspecified launch failure",
    "ecc error",
    "uncorrectable",
)
_ABSENT_MARKERS = ("no cuda", "no gpu", "not available", "not importable", "driver version")


def classify_gpu_health(probe_error: str | None) -> str:
    """PURE. Map a responsiveness probe result to a health verdict:

    * ``"healthy"``   — the probe ran (``probe_error is None``);
    * ``"wedged"``    — a poisoned/transient GPU state a RESET fixes (the WSL2 GPU-PV wedge);
    * ``"absent"``    — no usable CUDA GPU / torch (not a wedge — nothing to reset);
    * ``"unknown"``   — an error we can't confidently bucket.
    """
    if probe_error is None:
        return "healthy"
    err = probe_error.lower()
    if any(m in err for m in _WEDGED_MARKERS):
        return "wedged"
    if any(m in err for m in _ABSENT_MARKERS):
        return "absent"
    return "unknown"


def wedged_gpu_remediation(os_value: OperatingSystem, wsl_distro: str | None = None) -> str:
    """PURE. The OS-specific 'reset your GPU' instruction for a wedged device."""
    if os_value == OperatingSystem.wsl:
        distro = wsl_distro or "Ubuntu"
        return (
            "The GPU appears WEDGED — a prior crashed CUDA process left the WSL GPU-PV layer in a bad "
            f"state, so runs fail with 'device not ready' regardless of config. Reset it from Windows "
            f"PowerShell: `wsl --terminate {distro}` (or `wsl --shutdown`), then re-run. Your files "
            "persist across the reset."
        )
    if os_value == OperatingSystem.windows:
        return (
            "The GPU appears WEDGED. Close other GPU processes; if it persists, reset the display "
            "driver (Win+Ctrl+Shift+B) or restart the machine."
        )
    return (
        "The GPU appears WEDGED/unresponsive. Check `nvidia-smi`, clear any stuck processes, then reset "
        "the driver (`sudo nvidia-smi --gpu-reset`) or reboot."
    )


# Fit classes that mean "spilled off-device to shared RAM" (trains, but crawls).
_SPILL_CLASSES = frozenset(
    {
        FitClass.ACCIDENTAL_WDDM_SPILL,
        FitClass.ACCIDENTAL_UNIFIED_MEMORY_PAGING,
        FitClass.THRASHING,
    }
)


def spill_remediation(fit_class: FitClass, os_value: OperatingSystem) -> str:
    """PURE. Actionable guidance for a spill fit — what it costs + how to avoid it. Empty for a
    non-spill fit. Verified on a real 5070: a true-seq2048 7B QLoRA spilled to shared RAM and trained
    at ~145 s/step (~100x slower) instead of crashing."""
    if fit_class not in _SPILL_CLASSES:
        return ""
    guidance = (
        "This spills GPU memory to shared system RAM — it WILL train, but crawl (~10-100x slower; looks "
        "frozen but is progressing). To run at full speed: reduce sequence_len or micro_batch_size, use "
        "a smaller base model / LoRA rank, or add VRAM."
    )
    if os_value == OperatingSystem.linux:
        guidance += (
            " On bare Linux the dedicated-memory model predicts a hard OOM instead of WDDM spill; "
            "that prediction is not a measured fit result."
        )
    return guidance


def probe_gpu_responsive() -> str | None:
    """Lazily run ONE tiny CUDA op (alloc + matmul + sync) to detect a wedged GPU. Returns ``None`` when
    the GPU responds, else the error string (feed to :func:`classify_gpu_health`). NEVER raises —
    torch-free until called. This is the fast, cheap gate to run BEFORE a real training dispatch so a
    wedged GPU is diagnosed up front, not mid-run as a cryptic crash."""
    try:
        import torch  # noqa: PLC0415 - lazy; the health probe only runs when explicitly invoked.
    except Exception as exc:  # noqa: BLE001
        return f"torch not importable: {exc}"
    try:
        if not torch.cuda.is_available():
            return "no CUDA GPU available"
        probe = torch.ones((16, 16), device="cuda")
        result = (probe @ probe).sum()
        torch.cuda.synchronize()
        float(result)  # force the value back to the host — surfaces an async error here, not later
        return None
    except Exception as exc:  # noqa: BLE001 - ANY failure is the signal; classify_gpu_health buckets it.
        return str(exc)
