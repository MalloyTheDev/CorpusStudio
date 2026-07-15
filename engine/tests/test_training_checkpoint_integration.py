"""Real-torch CPU integration proof for exact checkpoint + resume (#440).

Skipped wherever torch is absent (the torch-free CI gate), so it never runs in the dependency-light
lane. Where torch IS present it drives ``_checkpoint_reference.py`` as three SEPARATE processes -
uninterrupted N steps; K steps + checkpoint; a fresh-process resume of the remaining N-K steps - and
proves the strongest defensible equivalence level for this configuration.

Equivalence level proved here: under a controlled deterministic CPU configuration (fixed seed, single
intra-op thread, ``torch.use_deterministic_algorithms(True)``, and full RNG-stream restore) the
resumed run reproduces the uninterrupted run BITWISE - identical final parameters and identical
per-step losses for the shared step numbers. This is a demonstration of exact state restoration
(optimizer + scheduler + every RNG stream + sampler cursor + step position); it does NOT claim bitwise
equivalence on GPU, where non-deterministic reductions can perturb the low bits even with correct
state restoration.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REFERENCE = Path(__file__).parent / "_checkpoint_reference.py"


def _run(mode: str, workdir: Path, k: int, n: int) -> None:
    result = subprocess.run(
        [sys.executable, str(REFERENCE), mode, "--workdir", str(workdir), "--k", str(k), "--n", str(n)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{mode} failed:\n{result.stdout}\n{result.stderr}"


def test_resume_reproduces_uninterrupted_training_bitwise(tmp_path: Path) -> None:
    k, n = 4, 10
    work = tmp_path / "eq"
    # Three independent processes: the resume is a genuinely fresh interpreter.
    _run("uninterrupted", work, k, n)
    _run("firsthalf", work, k, n)
    _run("resume", work, k, n)

    uninterrupted = torch.load(work / "uninterrupted.pt")
    resumed = torch.load(work / "resumed.pt")
    assert set(uninterrupted) == set(resumed) and uninterrupted

    # Strongest defensible level for this CPU config: bitwise-identical final parameters.
    for name in uninterrupted:
        assert torch.equal(uninterrupted[name], resumed[name]), f"param {name} diverged on resume"

    full = {row["optimizer_step"]: row["loss"] for row in json.loads((work / "losses_full.json").read_text())}
    tail = json.loads((work / "losses_tail.json").read_text())
    meta = json.loads((work / "resume_meta.json").read_text())

    # The first resumed optimizer step continues from exactly K+1, and the sampler cursor restored to K.
    assert meta["start_step"] == k + 1
    assert meta["cursor"] == {"cursor": k}
    assert [row["optimizer_step"] for row in tail] == list(range(k + 1, n + 1))
    # Every resumed step reproduces the uninterrupted loss for that step number, bitwise.
    for row in tail:
        assert row["loss"] == full[row["optimizer_step"]], f"loss diverged at step {row['optimizer_step']}"


def test_firsthalf_checkpoint_is_independently_verifiable(tmp_path: Path) -> None:
    from corpus_studio.platform.checkpoint import verify_checkpoint_integrity

    work = tmp_path / "half"
    _run("firsthalf", work, 3, 8)
    manifest = verify_checkpoint_integrity(work / "checkpoint")
    assert manifest.complete and manifest.state.global_optimizer_step == 3
    assert manifest.state.scheduler_captured and manifest.state.rng_captured
