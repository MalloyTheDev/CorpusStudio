"""Canonical worker-wheel build-provenance contract: writer, strict reader, and admission gate.

Guards the v7 defect (a sidecar that recorded the commit under ``audited_commit`` and omitted
``source_commit``, leaving ``identity.repository_commit`` null) from recurring: the reader consumes ONLY
a canonical ``source_commit`` and never a prohibited alias, and a wheel without one is refused for
scientific admission.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from corpus_studio.platform import build_provenance as bp
from corpus_studio.platform.contracts import HashRef, WorkerArtifactIdentity
from corpus_studio.platform.telemetry import worker_identity_overlay

_GOOD = "21aa81d97ff752709fd4d03791288c1bb76a2339"  # exact 40-char lowercase hex


def _artifact(wheel: Path) -> WorkerArtifactIdentity:
    return WorkerArtifactIdentity(
        distribution_name="corpus-studio-engine",
        normalized_name="corpus_studio_engine",
        version="1.3.0",
        filename=wheel.name,
        path=str(wheel),
        size_bytes=max(1, wheel.stat().st_size),
        content_hash=HashRef(value="a" * 64),
    )


def _write(dirpath: Path, payload: dict) -> Path:
    p = dirpath / bp.PROVENANCE_FILENAME
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ---- canonical format ----------------------------------------------------------------------------


@pytest.mark.parametrize("value", [_GOOD, "0" * 40, "abcdef0123456789" * 2 + "abcdef0f"])
def test_is_canonical_accepts_exact_lowercase_40_hex(value: str) -> None:
    assert bp.is_canonical_source_commit(value)


@pytest.mark.parametrize(
    "value",
    [
        "21AA81D97FF752709FD4D03791288C1BB76A2339",  # uppercase
        "21aa81d9",  # abbreviated
        _GOOD + "0",  # 41 chars
        _GOOD[:-1],  # 39 chars
        "21aa81g97ff752709fd4d03791288c1bb76a2339",  # non-hex 'g'
        "",
        None,
        1234,
    ],
)
def test_is_canonical_rejects_noncanonical(value: object) -> None:
    assert not bp.is_canonical_source_commit(value)


# ---- strict reader (best-effort, never raises) ---------------------------------------------------


def test_reader_returns_canonical_source_commit(tmp_path: Path) -> None:
    wheel = tmp_path / "corpus_studio_engine-1.3.0-py3-none-any.whl"
    wheel.write_bytes(b"x")
    _write(tmp_path, {"source_commit": _GOOD, "built_at": "2026-07-16T00:00:00Z"})
    assert bp.read_source_commit(str(wheel)) == _GOOD


def test_reader_rejects_audited_commit_only_no_fallback(tmp_path: Path) -> None:
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"x")
    # Exactly the v7 sidecar shape: authentic commit under audited_commit, no source_commit.
    _write(tmp_path, {"audited_commit": _GOOD})
    assert bp.read_source_commit(str(wheel)) is None


@pytest.mark.parametrize(
    "payload",
    [
        {"source_commit": _GOOD.upper()},  # uppercase
        {"source_commit": "21aa81d9"},  # abbreviated
        {"source_commit": ""},  # empty
        {"source_commit": 12345},  # non-string
        {"not_source": _GOOD},  # missing key
        [_GOOD],  # not an object
    ],
)
def test_reader_returns_none_for_noncanonical_or_missing(tmp_path: Path, payload: object) -> None:
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"x")
    (tmp_path / bp.PROVENANCE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    assert bp.read_source_commit(str(wheel)) is None


def test_reader_returns_none_for_malformed_json(tmp_path: Path) -> None:
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"x")
    (tmp_path / bp.PROVENANCE_FILENAME).write_text("{ not json", encoding="utf-8")
    assert bp.read_source_commit(str(wheel)) is None


def test_reader_returns_none_for_missing_file_or_no_path(tmp_path: Path) -> None:
    assert bp.read_source_commit(None) is None
    assert bp.read_source_commit(str(tmp_path / "w.whl")) is None  # no sidecar next to it


# ---- writer --------------------------------------------------------------------------------------


def test_writer_emits_canonical_document(tmp_path: Path) -> None:
    doc = bp.build_provenance_document(source_commit=_GOOD, extra={"audited_commit": _GOOD})
    path = bp.write_build_provenance(tmp_path, doc)
    assert path.name == bp.PROVENANCE_FILENAME
    read_back = json.loads(path.read_text(encoding="utf-8"))
    assert read_back["source_commit"] == _GOOD
    # The written sidecar is immediately readable through the strict reader (round-trip).
    (tmp_path / "w.whl").write_bytes(b"x")
    assert bp.read_source_commit(str(tmp_path / "w.whl")) == _GOOD


def test_writer_and_document_reject_noncanonical_commit(tmp_path: Path) -> None:
    with pytest.raises(bp.BuildProvenanceError):
        bp.build_provenance_document(source_commit="21aa81d9")
    with pytest.raises(bp.BuildProvenanceError):
        bp.write_build_provenance(tmp_path, {"source_commit": "nope"})


def test_document_extra_cannot_override_source_commit() -> None:
    with pytest.raises(bp.BuildProvenanceError):
        bp.build_provenance_document(source_commit=_GOOD, extra={"source_commit": "0" * 40})


# ---- admission gate ------------------------------------------------------------------------------


def test_admission_gate_returns_commit_for_canonical_sidecar(tmp_path: Path) -> None:
    (tmp_path / "w.whl").write_bytes(b"x")
    _write(tmp_path, {"source_commit": _GOOD})
    assert bp.validate_wheel_provenance_for_scientific_admission(str(tmp_path / "w.whl")) == _GOOD


def test_admission_gate_refuses_audited_commit_only_with_alias_hint(tmp_path: Path) -> None:
    (tmp_path / "w.whl").write_bytes(b"x")
    _write(tmp_path, {"audited_commit": _GOOD})
    with pytest.raises(bp.BuildProvenanceError, match="audited_commit"):
        bp.validate_wheel_provenance_for_scientific_admission(str(tmp_path / "w.whl"))


def test_admission_gate_refuses_missing_sidecar(tmp_path: Path) -> None:
    (tmp_path / "w.whl").write_bytes(b"x")
    with pytest.raises(bp.BuildProvenanceError, match="no BUILD_PROVENANCE.json"):
        bp.validate_wheel_provenance_for_scientific_admission(str(tmp_path / "w.whl"))


# ---- repository validation (real tmp git repo) ---------------------------------------------------


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_repo(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@e.st")
    _git(root, "config", "user.name", "t")
    (root / "a.txt").write_text("a", encoding="utf-8")
    _git(root, "add", "a.txt")
    _git(root, "commit", "-q", "-m", "first")
    return _git(root, "rev-parse", "HEAD")


def test_repo_validation_accepts_existing_clean_commit(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    head = _init_repo(root)
    bp.validate_source_commit_against_repo(head, root)  # clean worktree, commit exists -> no raise


def test_repo_validation_refuses_dirty_worktree(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    head = _init_repo(root)
    (root / "a.txt").write_text("changed", encoding="utf-8")  # tracked modification
    with pytest.raises(bp.BuildProvenanceError, match="uncommitted changes"):
        bp.validate_source_commit_against_repo(head, root)


def test_repo_validation_refuses_unknown_commit(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repo(root)
    with pytest.raises(bp.BuildProvenanceError, match="does not exist"):
        bp.validate_source_commit_against_repo("0" * 40, root)


def test_repo_validation_enforces_required_ancestor(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    first = _init_repo(root)
    (root / "b.txt").write_text("b", encoding="utf-8")
    _git(root, "add", "b.txt")
    _git(root, "commit", "-q", "-m", "second")
    second = _git(root, "rev-parse", "HEAD")
    # first is an ancestor of second -> ok; second is NOT an ancestor of first -> refused.
    bp.validate_source_commit_against_repo(first, root, must_be_ancestor_of=second)
    with pytest.raises(bp.BuildProvenanceError, match="not an ancestor"):
        bp.validate_source_commit_against_repo(second, root, must_be_ancestor_of=first)


# ---- end-to-end: telemetry identity populated WITHOUT an overlay ----------------------------------


def test_worker_identity_overlay_populates_repository_commit_from_canonical_sidecar(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "corpus_studio_engine-1.3.0-py3-none-any.whl"
    wheel.write_bytes(b"x")
    _write(tmp_path, {"source_commit": _GOOD})
    overlay = worker_identity_overlay(_artifact(wheel))
    # The auto overlay (no hand-supplied identity) now carries the commit - no post-hoc injection needed.
    assert overlay.repository_commit == _GOOD
    assert overlay.worker_wheel_sha256 == "a" * 64


def test_worker_identity_overlay_is_null_for_audited_commit_only_sidecar(tmp_path: Path) -> None:
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"x")
    _write(tmp_path, {"audited_commit": _GOOD})  # the v7 shape
    overlay = worker_identity_overlay(_artifact(wheel))
    assert overlay.repository_commit is None  # honestly null, never fabricated from the alias
