"""Generated-output containment (finding F5).

Two guarantees: (1) `platform-plan` seals an output root that is canonical, absolute, CWD-independent,
and OUTSIDE the source checkout - the default lives under the CorpusStudio application-data directory,
and any root that resolves into the repository (including via `..` traversal or a symlink) is refused;
(2) the repository `.gitignore` still anchors the historical relative-default landing spots (`/output/`,
`/engine/output/`) as historical defense-in-depth only, without hiding the tracked example adapter.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from corpus_studio.cli import (
    OutputContainmentError,
    _default_output_root,
    _resolve_sealed_output_root,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _outside_repo(path: str) -> bool:
    resolved = Path(path).resolve()
    return resolved != _REPO_ROOT and _REPO_ROOT not in resolved.parents


@pytest.fixture
def outside_repo_dir():
    """A real directory guaranteed to be OUTSIDE the repo checkout.

    Uses the system temp dir, NOT pytest's ``tmp_path`` - the local verify gate runs pytest with
    ``--basetemp`` inside ``engine/``, so ``tmp_path`` is in-repo and would (correctly) be refused by the
    containment guard. External roots must come from a genuinely outside-repo location."""

    directory = Path(tempfile.mkdtemp(prefix="cs-out-"))
    assert _outside_repo(str(directory)), "system temp dir unexpectedly inside the repo"
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# ---- default output root is outside the checkout, from any CWD -----------------------------------


def test_default_output_root_from_repository_root_is_outside_repo(
    outside_repo_dir, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(outside_repo_dir))
    monkeypatch.setenv("LOCALAPPDATA", str(outside_repo_dir))
    monkeypatch.chdir(_REPO_ROOT)
    root = _resolve_sealed_output_root(None)
    # The F5 central assertion: the default is NOT <repo>/output; it is an absolute, outside-repo path.
    assert os.path.isabs(root)
    assert _outside_repo(root)


def test_default_output_root_from_engine_dir_is_outside_repo(
    outside_repo_dir, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(outside_repo_dir))
    monkeypatch.setenv("LOCALAPPDATA", str(outside_repo_dir))
    monkeypatch.chdir(_REPO_ROOT / "engine")
    assert _outside_repo(_resolve_sealed_output_root(None))


def test_default_output_root_from_unrelated_cwd_is_outside_repo(
    outside_repo_dir, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(outside_repo_dir))
    monkeypatch.setenv("LOCALAPPDATA", str(outside_repo_dir))
    monkeypatch.chdir(outside_repo_dir)  # CWD-independence
    assert _outside_repo(_resolve_sealed_output_root(None))


# ---- in-repo roots are refused (repo, traversal, symlink, .git) ----------------------------------


def test_explicit_repository_path_is_refused() -> None:
    with pytest.raises(OutputContainmentError):
        _resolve_sealed_output_root(str(_REPO_ROOT / "engine" / "output"))


def test_dotdot_traversal_into_repository_is_refused() -> None:
    # <repo>/engine/../docs re-enters the repo as <repo>/docs.
    with pytest.raises(OutputContainmentError):
        _resolve_sealed_output_root(str(_REPO_ROOT / "engine" / ".." / "docs"))


def test_symlink_into_repository_is_refused(tmp_path) -> None:
    link = tmp_path / "into-repo"
    link.symlink_to(_REPO_ROOT / "engine")  # realpath resolves the link into the repo
    with pytest.raises(OutputContainmentError):
        _resolve_sealed_output_root(str(link))


def test_dot_git_path_is_refused() -> None:
    with pytest.raises(OutputContainmentError):
        _resolve_sealed_output_root(str(_REPO_ROOT / ".git" / "cs-out"))


# ---- external + application-data roots are accepted ----------------------------------------------


def test_valid_external_absolute_root_is_accepted(outside_repo_dir) -> None:
    external = outside_repo_dir / "planned-output"
    resolved = _resolve_sealed_output_root(str(external))
    assert resolved == os.path.realpath(external)
    assert _outside_repo(resolved)


def test_application_data_default_is_accepted_and_outside_repo(
    outside_repo_dir, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(outside_repo_dir))
    monkeypatch.setenv("LOCALAPPDATA", str(outside_repo_dir))
    root = _resolve_sealed_output_root(None)
    if os.name == "nt":
        expected = (outside_repo_dir / "CorpusStudio" / "runs").resolve()
    else:
        expected = (outside_repo_dir / "corpusstudio" / "runs").resolve()
    assert Path(root) == expected
    assert Path(_default_output_root()).resolve() == expected
    assert _outside_repo(root)


def test_v7_style_external_root_is_accepted() -> None:
    # Preserve the v7 absolute-root semantics: a plan output root under the runs area is untouched.
    v7 = "/mnt/training-nvme/corpusstudio/runs/v7-plan/abc"
    resolved = _resolve_sealed_output_root(v7)
    assert resolved == os.path.realpath(v7)
    assert _outside_repo(resolved)


# ---- .gitignore historical defense-in-depth (unchanged) ------------------------------------------


def _check_ignored(relpath: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "check-ignore", "-q", relpath],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 128:
        pytest.skip("not a git repository (git check-ignore unavailable)")
    return completed.returncode == 0


def test_root_output_tree_is_ignored() -> None:
    assert _check_ignored("output/runs/run-x/artifacts/adapter/adapter_model.safetensors")
    assert _check_ignored("engine/output/runs/run-x/artifacts/adapter/adapter_model.safetensors")


def test_tracked_example_adapter_is_not_ignored() -> None:
    tracked = "examples/wbg/adapter-seq1536-baseline/adapter_model.safetensors"
    assert (_REPO_ROOT / tracked).exists(), "expected tracked example adapter is missing"
    assert not _check_ignored(tracked)


def test_nested_non_root_output_dir_is_not_ignored() -> None:
    assert not _check_ignored("examples/some_project/output/keepme.txt")
