"""Best-effort GPU memory probe (dependency-light, no ML frameworks).

The VRAM estimate elsewhere is pure arithmetic from the model name — it never
looks at the actual machine. OOM is the single most common way a real training
run dies, so this optionally reads the GPU's total/free memory from ``nvidia-smi``
(which is on ``PATH`` wherever NVIDIA drivers are installed — i.e. anywhere a
CUDA trainer can run) and lets the pre-flight warn that a config likely won't fit
*before* the user waits.

It is best-effort and never a hard dependency: no ``nvidia-smi`` (CPU-only box,
non-NVIDIA GPU, generating the config on a laptop to train elsewhere) simply
returns ``None`` and the pre-flight skips the OOM check. It imports no ML
framework and spawns only the read-only query.
"""

import shutil
import subprocess

from pydantic import BaseModel


class GpuMemory(BaseModel):
    """The first CUDA GPU's memory, in GB (as reported by ``nvidia-smi``)."""

    total_gb: float
    free_gb: float
    compute_capability: str = ""  # e.g. "12.0" for Blackwell / sm_120 (empty when unreadable)
    name: str = ""  # e.g. "NVIDIA GeForce RTX 5070" — so the profile names the GPU without torch


def _capability_major(compute_capability: str) -> int:
    """Major compute capability from a ``"12.0"``-style string, or 0 when unknown."""
    try:
        return int(compute_capability.split(".")[0])
    except (ValueError, AttributeError, IndexError):
        return 0


def probe_gpu_memory() -> GpuMemory | None:
    """Return the first GPU's total/free memory via ``nvidia-smi``, or ``None`` if
    it can't be read (not installed, no GPU, or the query fails). Never raises."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=memory.total,memory.free,compute_cap,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None

    first_line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 2:
        return None
    try:
        total_mb = float(parts[0])
        free_mb = float(parts[1])
    except ValueError:
        return None

    # compute_cap + name are trailing fields — tolerate their absence on older nvidia-smi (the name
    # is never comma-bearing, so CSV splitting stays safe).
    capability = parts[2] if len(parts) >= 3 else ""
    name = parts[3] if len(parts) >= 4 else ""

    return GpuMemory(
        total_gb=round(total_mb / 1024, 1),
        free_gb=round(free_mb / 1024, 1),
        compute_capability=capability,
        name=name,
    )
