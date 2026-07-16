"""Canonical worker-wheel build-provenance contract: writer, strict reader, and admission gate.

Guards the v7 defect (a sidecar that recorded the commit under ``audited_commit`` and omitted
``source_commit``, leaving ``identity.repository_commit`` null) from recurring: the reader consumes ONLY
a canonical ``source_commit`` and never a prohibited alias, and a wheel without one is refused for
scientific admission.
"""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from corpus_studio.platform import build_provenance as bp
from corpus_studio.platform.contracts import HashRef, WorkerArtifactIdentity
from corpus_studio.platform.telemetry import worker_identity_overlay

_GOOD = "21aa81d97ff752709fd4d03791288c1bb76a2339"  # exact 40-char lowercase hex
_DIST_INFO = "corpus_studio_engine-1.3.0.dist-info/"


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


def _minimal_wheel(path: Path, *, members: dict[str, bytes] | None = None) -> Path:
    """Build a minimal but structurally valid wheel (dist-info/METADATA + a payload + a sealed RECORD)."""

    base: dict[str, bytes] = {
        f"{_DIST_INFO}METADATA": b"Metadata-Version: 2.1\nName: corpus-studio-engine\nVersion: 1.3.0\n",
        "corpus_studio/__init__.py": b"# worker\n",
    }
    if members:
        base.update(members)
    record_name = f"{_DIST_INFO}RECORD"
    lines = []
    for name, data in base.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        lines.append(f"{name},sha256={digest},{len(data)}")
    lines.append(f"{record_name},,")
    base[record_name] = ("\n".join(lines) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in base.items():
            archive.writestr(name, data)
    return path


def _stamped_wheel(path: Path, *, commit: str = _GOOD, external_copy: bool = False) -> Path:
    """A minimal wheel with canonical provenance embedded (and sealed in RECORD)."""

    _minimal_wheel(path)
    bp.stamp_wheel_with_provenance(
        path, bp.build_provenance_document(source_commit=commit), external_copy=external_copy
    )
    return path


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


# ---- wheel-embedded provenance: stamp, RECORD seal, read ----------------------------------------


def test_stamp_embeds_provenance_sealed_in_record_and_changes_sha(tmp_path: Path) -> None:
    wheel = _minimal_wheel(tmp_path / "corpus_studio_engine-1.3.0-py3-none-any.whl")
    before = hashlib.sha256(wheel.read_bytes()).hexdigest()
    stamp = bp.stamp_wheel_with_provenance(
        wheel, bp.build_provenance_document(source_commit=_GOOD), external_copy=True
    )
    after = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert stamp.wheel_sha256 == after != before  # embedding changed the sealed wheel identity
    assert stamp.source_commit == _GOOD
    assert stamp.embedded_arcname == f"{_DIST_INFO}{bp.PROVENANCE_FILENAME}"
    assert bp.read_embedded_source_commit(str(wheel)) == _GOOD
    # RECORD seals it: integrity check passes, and the external copy matches the embedded bytes.
    bp.verify_embedded_provenance_record_integrity(wheel)
    assert stamp.external_path is not None and stamp.external_path.exists()


def test_stamp_is_deterministic_for_same_input_and_commit(tmp_path: Path) -> None:
    a = _stamped_wheel(tmp_path / "a.whl")
    b = _stamped_wheel(tmp_path / "b.whl")
    assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()


def test_stamp_refuses_noncanonical_document(tmp_path: Path) -> None:
    wheel = _minimal_wheel(tmp_path / "w.whl")
    with pytest.raises(bp.BuildProvenanceError):
        bp.stamp_wheel_with_provenance(wheel, {"source_commit": "21aa81d9"})


def test_record_integrity_detects_swapped_embedded_provenance(tmp_path: Path) -> None:
    # A tampered embedded provenance whose bytes no longer match the RECORD entry is detected.
    wheel = _stamped_wheel(tmp_path / "w.whl")
    with zipfile.ZipFile(wheel) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    prov = f"{_DIST_INFO}{bp.PROVENANCE_FILENAME}"
    members[prov] = json.dumps({"source_commit": "0" * 40}).encode("utf-8")  # RECORD not updated
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    with pytest.raises(bp.BuildProvenanceError, match="disagrees with RECORD"):
        bp.verify_embedded_provenance_record_integrity(wheel)


# ---- admission gate (embedded provenance) --------------------------------------------------------


def test_admission_gate_returns_commit_for_embedded_provenance(tmp_path: Path) -> None:
    wheel = _stamped_wheel(tmp_path / "w.whl", external_copy=True)
    assert bp.validate_wheel_provenance_for_scientific_admission(str(wheel)) == _GOOD


def test_admission_gate_refuses_external_only_v7_shape(tmp_path: Path) -> None:
    # The v7 defect shape: an external sidecar and NO embedded provenance is not admissible.
    wheel = _minimal_wheel(tmp_path / "w.whl")
    _write(tmp_path, {"source_commit": _GOOD})
    with pytest.raises(bp.BuildProvenanceError, match="no embedded"):
        bp.validate_wheel_provenance_for_scientific_admission(str(wheel))


def test_admission_gate_refuses_audited_commit_only_with_alias_hint(tmp_path: Path) -> None:
    wheel = _minimal_wheel(tmp_path / "w.whl")
    # Embed a provenance doc carrying ONLY the prohibited alias (no canonical source_commit).
    _embed_raw(wheel, {"audited_commit": _GOOD})
    with pytest.raises(bp.BuildProvenanceError, match="audited_commit"):
        bp.validate_wheel_provenance_for_scientific_admission(str(wheel))


def test_admission_gate_refuses_missing_provenance(tmp_path: Path) -> None:
    wheel = _minimal_wheel(tmp_path / "w.whl")
    with pytest.raises(bp.BuildProvenanceError, match="no embedded"):
        bp.validate_wheel_provenance_for_scientific_admission(str(wheel))


def test_admission_gate_refuses_external_mismatch(tmp_path: Path) -> None:
    wheel = _stamped_wheel(tmp_path / "w.whl", external_copy=True)
    (wheel.parent / bp.PROVENANCE_FILENAME).write_text('{"source_commit":"0000"}', encoding="utf-8")
    with pytest.raises(bp.BuildProvenanceError, match="disagrees with the embedded"):
        bp.validate_wheel_provenance_for_scientific_admission(str(wheel))


def _embed_raw(wheel: Path, document: dict) -> None:
    """Embed an ARBITRARY (possibly non-canonical) provenance doc + RECORD line, for refusal tests."""

    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with zipfile.ZipFile(wheel) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    record_name = f"{_DIST_INFO}RECORD"
    prov = f"{_DIST_INFO}{bp.PROVENANCE_FILENAME}"
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
    lines = [ln for ln in members[record_name].decode().splitlines() if ln and not ln.startswith(record_name)]
    lines.append(f"{prov},sha256={digest},{len(payload)}")
    lines.append(f"{record_name},,")
    members[prov] = payload
    members[record_name] = ("\n".join(lines) + "\n").encode("utf-8")
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


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


def test_repo_validation_enforces_required_ancestor_direction(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    first = _init_repo(root)
    (root / "b.txt").write_text("b", encoding="utf-8")
    _git(root, "add", "b.txt")
    _git(root, "commit", "-q", "-m", "second")
    second = _git(root, "rev-parse", "HEAD")
    # Corrected direction: required_git_ancestor MUST be an ancestor of source_commit (source descends
    # from the floor). floor=first, source=second -> ok (second descends from first).
    bp.validate_source_commit_against_repo(second, root, required_git_ancestor=first)
    # floor=second, source=first -> refused (first does NOT descend from second).
    with pytest.raises(bp.BuildProvenanceError, match="does not descend from"):
        bp.validate_source_commit_against_repo(first, root, required_git_ancestor=second)


def test_repo_validation_refuses_unknown_required_ancestor(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    head = _init_repo(root)
    with pytest.raises(bp.BuildProvenanceError, match="required_git_ancestor .* does not exist"):
        bp.validate_source_commit_against_repo(head, root, required_git_ancestor="0" * 40)


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


def test_worker_identity_overlay_reads_embedded_provenance_without_external_sidecar(
    tmp_path: Path,
) -> None:
    # A self-describing wheel (embedded provenance, NO external sidecar) still populates the identity:
    # telemetry obtains repository_commit automatically, without a post-hoc overlay or a loose file.
    wheel = _stamped_wheel(
        tmp_path / "corpus_studio_engine-1.3.0-py3-none-any.whl", external_copy=False
    )
    assert not (wheel.parent / bp.PROVENANCE_FILENAME).exists()
    overlay = worker_identity_overlay(_artifact(wheel))
    assert overlay.repository_commit == _GOOD


# ---- build-worker-wheel CLI (the authoritative first-party build command) -------------------------


def test_build_worker_wheel_cli_stamps_and_admits(tmp_path: Path, monkeypatch) -> None:
    # End-to-end through the ACTUAL build command (source-wheel path, avoiding a heavy real compile):
    # resolve HEAD -> validate clean worktree + required-ancestor descent -> stamp embedded provenance
    # -> re-verify scientific admission -> the stamped wheel is self-describing for telemetry.
    from typer.testing import CliRunner

    import corpus_studio.cli as cli

    root = tmp_path / "repo"
    head = _init_repo(root)
    monkeypatch.setattr(cli, "repository_root", lambda: root)

    source = _minimal_wheel(tmp_path / "corpus_studio_engine-1.3.0-py3-none-any.whl")
    dest = tmp_path / "out"
    result = CliRunner().invoke(
        cli.app,
        [
            "build-worker-wheel",
            "--dest-dir",
            str(dest),
            "--source-wheel",
            str(source),
            "--required-ancestor",
            head,  # head is an ancestor of itself -> descent holds
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["source_commit"] == head
    assert payload["admission_source_commit"] == head
    assert payload["required_git_ancestor"] == head
    stamped = dest / source.name
    assert bp.read_embedded_source_commit(str(stamped)) == head
    assert bp.validate_wheel_provenance_for_scientific_admission(str(stamped)) == head
    overlay = worker_identity_overlay(_artifact(stamped))
    assert overlay.repository_commit == head


def test_build_worker_wheel_cli_refuses_dirty_worktree(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    import corpus_studio.cli as cli

    root = tmp_path / "repo"
    _init_repo(root)
    (root / "a.txt").write_text("dirtied", encoding="utf-8")  # uncommitted tracked change
    monkeypatch.setattr(cli, "repository_root", lambda: root)

    source = _minimal_wheel(tmp_path / "corpus_studio_engine-1.3.0-py3-none-any.whl")
    result = CliRunner().invoke(
        cli.app,
        ["build-worker-wheel", "--dest-dir", str(tmp_path / "out"), "--source-wheel", str(source)],
    )
    assert result.exit_code == 2
    assert "uncommitted changes" in result.output
