"""Training pre-flight: cheap, fast checks run BEFORE a (long) training launch.

The point is to fail in seconds, not hours. A trainer run can take a very long
time and use significant GPU; a missing data file, an empty dataset, or a trainer
that isn't on PATH shouldn't be discovered only after the user has waited. This
runs a handful of deterministic, dependency-light checks and returns a structured
verdict the desktop can surface (and gate the Launch button on).

Honesty boundary: this is a *pre-flight*, not a guarantee. It catches common,
cheaply-detectable problems (no data, empty dataset, trainer not installed,
truncation) — it does NOT run the trainer, inspect the GPU, or validate every
config key, so a green pre-flight is "nothing obviously wrong", not "this run will
succeed". Blocking checks are only for things that make a launch certainly fail
(missing config/data, empty dataset); everything else warns.
"""

import shutil
import sys
from pathlib import Path

from pydantic import BaseModel

from corpus_studio.training import gpu_probe

PASS = "pass"
WARN = "warn"
BLOCK = "block"

# Below this, a run is almost certainly not worth the compute — warn loudly.
_MIN_USEFUL_ROWS = 10

# Keep at least this much VRAM free ABOVE the (rough) estimate. A run that only "just fits" can
# VRAM-pressure DEADLOCK rather than cleanly OOM — a real Qwen2.5-7B / 4-bit / seq-4096 run peaked
# ~11.9 GB and hung on a 12 GB card at 58 MiB free. Warn in that band instead of green-lighting it.
_VRAM_SAFETY_MARGIN_GB = 1.5


class PreflightCheck(BaseModel):
    """One pre-flight check outcome."""

    name: str
    status: str  # pass | warn | block
    message: str


class TrainingPreflightReport(BaseModel):
    """Aggregate pre-flight verdict for a training launch."""

    status: str = PASS  # worst of the checks
    can_launch: bool = True  # False when any check blocks
    trainer_command: str = ""
    trainer_found: bool = False
    checks: list[PreflightCheck] = []


def _worst(statuses: list[str]) -> str:
    if BLOCK in statuses:
        return BLOCK
    if WARN in statuses:
        return WARN
    return PASS


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def run_training_preflight(
    config_path: Path,
    launch_argv: list[str],
    dependencies: list[str],
    data_paths: list[Path],
    dataset_row_count: int,
    examples_over_sequence_len: int,
    sequence_len: int,
    vram_min_gb: float | None = None,
    vram_min_gb_math: float | None = None,
) -> TrainingPreflightReport:
    """Run the pre-flight checks and return a structured verdict.

    ``launch_argv[0]`` is the trainer executable to resolve on PATH; ``data_paths``
    are the dataset/split files the run reads; the sequence-length inputs come from
    the token budget already computed for the config. ``vram_min_gb`` is the most
    memory-efficient (4-bit, flash-attention) VRAM estimate — when a GPU is detected
    it drives the OOM check. ``vram_min_gb_math`` is the higher math-attention-path
    estimate; on a **native-Windows** Blackwell GPU (sm_120, forced onto the math path
    because the fused flash kernel deadlocks under WDDM) it is used instead. The efficient estimate
    on other platforms is still a prediction; only a capability-probed run proves the attention path.
    """
    checks: list[PreflightCheck] = []

    # 1. Trainer on PATH. A missing trainer makes the launch fail immediately, but the
    #    desktop may spawn it in a differently-activated env, so warn (don't hard-block).
    trainer = launch_argv[0] if launch_argv else ""
    if trainer == "corpus-studio":
        # First-party trainer: Corpus Studio itself, run via `<python> -m corpus_studio.cli`, not a
        # separate executable on the GLOBAL PATH — so a PATH check is meaningless (and misleadingly
        # warns for a venv install). Whether the [train] RUNTIME is present is train-check's job.
        trainer_found = True
        checks.append(
            PreflightCheck(
                name="trainer_available",
                status=PASS,
                message="First-party trainer (corpus_studio) — run `train-check` to confirm the training runtime.",
            )
        )
    else:
        trainer_found = bool(trainer) and shutil.which(trainer) is not None
        if trainer_found:
            checks.append(
                PreflightCheck(name="trainer_available", status=PASS, message=f"Trainer '{trainer}' found on PATH.")
            )
        else:
            deps = ", ".join(dependencies) if dependencies else "the target's tools"
            checks.append(
                PreflightCheck(
                    name="trainer_available",
                    status=WARN,
                    message=(
                        f"Trainer command '{trainer}' was not found on PATH. Install your trainer "
                        f"(requires: {deps}) or check PATH before launching."
                    ),
                )
            )

    # 2. Config file present + non-empty (definitely fatal if missing).
    if _nonempty_file(config_path):
        checks.append(PreflightCheck(name="config_file", status=PASS, message="Config file is present."))
    else:
        checks.append(
            PreflightCheck(
                name="config_file",
                status=BLOCK,
                message=f"Config file is missing or empty: {config_path}. Generate the training config first.",
            )
        )

    # 3. Training data files present + non-empty (fatal if missing).
    missing = [str(path) for path in data_paths if not _nonempty_file(path)]
    if not missing:
        checks.append(
            PreflightCheck(name="training_data", status=PASS, message="All referenced data files are present.")
        )
    else:
        checks.append(
            PreflightCheck(
                name="training_data",
                status=BLOCK,
                message=(
                    "Missing or empty data file(s): " + ", ".join(missing) + ". Generate splits / re-export first."
                ),
            )
        )

    # 4. Dataset size sanity.
    if dataset_row_count <= 0:
        checks.append(
            PreflightCheck(name="dataset_size", status=BLOCK, message="The dataset has no rows to train on.")
        )
    elif dataset_row_count < _MIN_USEFUL_ROWS:
        checks.append(
            PreflightCheck(
                name="dataset_size",
                status=WARN,
                message=(
                    f"Only {dataset_row_count} example(s); training on very few rows is unlikely to "
                    "produce a useful model."
                ),
            )
        )
    else:
        checks.append(
            PreflightCheck(name="dataset_size", status=PASS, message=f"{dataset_row_count} example(s).")
        )

    # 5. Truncation against sequence_len (warn — the trainer truncates, it won't fail).
    if examples_over_sequence_len > 0:
        checks.append(
            PreflightCheck(
                name="sequence_length",
                status=WARN,
                message=(
                    f"{examples_over_sequence_len} example(s) exceed sequence_len={sequence_len} and will be "
                    "truncated. Raise sequence_len or shorten/split those rows to avoid dropping content."
                ),
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="sequence_length",
                status=PASS,
                message=f"No examples exceed sequence_len={sequence_len}.",
            )
        )

    # 6. OOM realism (only when a GPU is actually detected — otherwise the VRAM
    #    estimate is arithmetic-only and this is skipped rather than guessing).
    gpu = gpu_probe.probe_gpu_memory()
    if gpu is not None and vram_min_gb is not None:
        # The fused flash SDPA kernel deadlocks on Blackwell (sm_120) ONLY under native Windows (WDDM),
        # forcing the higher-VRAM math path THERE. Outside native Windows this legacy preflight keeps
        # the efficient estimate, but it is not proof of a bare-Linux attention path or fit; Platform
        # capability probes and measured fit remain authoritative.
        is_blackwell = gpu_probe._capability_major(gpu.compute_capability) >= 12
        native_windows = sys.platform == "win32"
        if is_blackwell and native_windows and vram_min_gb_math is not None:
            effective_min_gb = vram_min_gb_math
            path = " (math attention forced on this native-Windows Blackwell GPU)"
        else:
            effective_min_gb = vram_min_gb
            path = ""
        if effective_min_gb > gpu.free_gb:
            checks.append(
                PreflightCheck(
                    name="gpu_memory",
                    status=WARN,
                    message=(
                        f"Won't fit fast: the 4-bit training peak is ~{effective_min_gb:.1f} GB{path} but the GPU has "
                        f"~{gpu.free_gb:.1f} GB free (of ~{gpu.total_gb:.1f} GB). On Windows/WSL the driver silently SPILLS "
                        "the overflow to system RAM and thrashes over PCIe — steps run 10-25x slower (looks frozen but is "
                        "crawling). A dedicated-memory Linux host is expected to OOM instead of WDDM-spill, but this "
                        "specific fit remains unverified. Lower sequence_len / micro-batch, or use a smaller base model."
                    ),
                )
            )
        elif effective_min_gb > gpu.free_gb - _VRAM_SAFETY_MARGIN_GB:
            checks.append(
                PreflightCheck(
                    name="gpu_memory",
                    status=WARN,
                    message=(
                        f"Tight VRAM: the 4-bit training peak ~{effective_min_gb:.1f} GB{path} leaves under "
                        f"~{_VRAM_SAFETY_MARGIN_GB:.0f} GB of ~{gpu.free_gb:.1f} GB free — close to the edge the run can start "
                        "SPILLING to system RAM (Windows) and crawl at 10-25x slower. The peak is ~linear in sequence_len; "
                        "lower sequence_len or micro-batch for headroom."
                    ),
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    name="gpu_memory",
                    status=PASS,
                    message=(
                        f"GPU has ~{gpu.free_gb:.1f} GB free (of ~{gpu.total_gb:.1f} GB); the 4-bit estimate "
                        f"~{effective_min_gb:.1f} GB{path} fits with headroom."
                    ),
                )
            )

    status = _worst([check.status for check in checks])
    return TrainingPreflightReport(
        status=status,
        can_launch=status != BLOCK,
        trainer_command=trainer,
        trainer_found=trainer_found,
        checks=checks,
    )
