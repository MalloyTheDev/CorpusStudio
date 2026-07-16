"""Generated-output containment (finding F5).

Two guarantees: (1) `platform-plan` seals an ABSOLUTE output root so a plan's write location is
CWD-independent (a relative root resolves against the process CWD at run time and lands the run tree +
adapter in the checkout when dispatched from the repo root); (2) the repository `.gitignore` anchors the
historical relative-default landing spots (`/output/`, `/engine/output/`) so an accidental in-checkout
write can never enter a tracked area - WITHOUT hiding the legitimately tracked example adapter.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from corpus_studio.cli import _absolute_output_root

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---- absolutization ------------------------------------------------------------------------------


def test_relative_default_output_root_is_absolutized() -> None:
    result = _absolute_output_root("output")
    assert os.path.isabs(result)
    assert result.endswith(os.sep + "output")


def test_relative_subpath_is_absolutized() -> None:
    result = _absolute_output_root(os.path.join("some", "rel"))
    assert os.path.isabs(result)


def test_absolute_output_root_passes_through_normalized(tmp_path: Path) -> None:
    absolute = str(tmp_path / "planned-output")
    assert _absolute_output_root(absolute) == absolute


def test_user_home_is_expanded() -> None:
    result = _absolute_output_root(os.path.join("~", "cs-runs"))
    assert os.path.isabs(result)
    assert "~" not in result


# ---- .gitignore containment behavior (real repo rules) -------------------------------------------


def _check_ignored(relpath: str) -> bool:
    """True iff git would ignore ``relpath`` under the repository's own .gitignore rules."""
    completed = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "check-ignore", "-q", relpath],
        check=False,
        capture_output=True,
        text=True,
    )
    # git check-ignore: exit 0 = ignored, 1 = not ignored, 128 = error/not a repo.
    if completed.returncode == 128:
        pytest.skip("not a git repository (git check-ignore unavailable)")
    return completed.returncode == 0


def test_root_output_tree_is_ignored() -> None:
    assert _check_ignored("output/runs/run-x/artifacts/adapter/adapter_model.safetensors")
    assert _check_ignored("engine/output/runs/run-x/artifacts/adapter/adapter_model.safetensors")


def test_tracked_example_adapter_is_not_ignored() -> None:
    # A legitimately tracked artifact must NOT be swept up by the containment rules (no broad ignore).
    tracked = "examples/wbg/adapter-seq1536-baseline/adapter_model.safetensors"
    assert (_REPO_ROOT / tracked).exists(), "expected tracked example adapter is missing"
    assert not _check_ignored(tracked)


def test_nested_non_root_output_dir_is_not_ignored() -> None:
    # Anchored rules only bind the repo root; a legitimately-placed nested 'output' is not hidden.
    assert not _check_ignored("examples/some_project/output/keepme.txt")
