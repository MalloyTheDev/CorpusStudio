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
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from corpus_studio.platform import build_provenance as bp
from corpus_studio.platform.contracts import HashRef, WorkerArtifactIdentity
from corpus_studio.platform.telemetry import worker_identity_overlay

_GOOD = "21aa81d97ff752709fd4d03791288c1bb76a2339"  # exact 40-char lowercase hex
_FLOOR = "1234567890abcdef1234567890abcdef12345678"  # canonical required_git_ancestor for fixtures
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
    """A minimal wheel with canonical provenance embedded (source_commit + floor, sealed in RECORD)."""

    _minimal_wheel(path)
    bp.stamp_wheel_with_provenance(
        path,
        bp.build_provenance_document(
            source_commit=commit, extra={"required_git_ancestor": _FLOOR}
        ),
        external_copy=external_copy,
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


_MISSING_FLOOR = object()


@pytest.mark.parametrize(
    "floor",
    [
        _MISSING_FLOOR,  # source-commit-only wheel: no required_git_ancestor key at all
        None,  # null floor
        1234,  # non-string floor
        "21aa81d9",  # abbreviated
        _GOOD.upper(),  # uppercase
        "z" * 40,  # malformed (non-hex)
        "",  # empty
        _GOOD[:-1],  # 39 chars
    ],
)
def test_admission_requires_canonical_embedded_floor(tmp_path: Path, floor: object) -> None:
    # Admission ALWAYS requires a canonical embedded required_git_ancestor - even with no expected floor
    # and no repo. A source-commit-only or malformed-floor wheel is refused (this is exactly the shape
    # the Environment Manager gate sees, since it supplies neither optional argument).
    wheel = _minimal_wheel(tmp_path / "w.whl")
    extra = None if floor is _MISSING_FLOOR else {"required_git_ancestor": floor}
    bp.stamp_wheel_with_provenance(
        wheel, bp.build_provenance_document(source_commit=_GOOD, extra=extra), external_copy=False
    )
    with pytest.raises(bp.BuildProvenanceError, match="required_git_ancestor is"):
        bp.validate_wheel_provenance_for_scientific_admission(str(wheel))


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


# ---- admission floor value-comparison hook (unfed by the Environment Manager today) --------------


def test_admission_expected_floor_match_and_mismatch(tmp_path: Path) -> None:
    floor = "1234567890abcdef1234567890abcdef12345678"
    wheel = _minimal_wheel(tmp_path / "w.whl")
    bp.stamp_wheel_with_provenance(
        wheel,
        bp.build_provenance_document(source_commit=_GOOD, extra={"required_git_ancestor": floor}),
        external_copy=False,
    )
    # A matching reviewed floor passes; a different reviewed floor is refused (value comparison only,
    # no repository needed). The Environment Manager passes no expected floor, so this is a hook for a
    # future plumbed floor, not something admission enforces today.
    assert (
        bp.validate_wheel_provenance_for_scientific_admission(
            str(wheel), expected_required_git_ancestor=floor
        )
        == _GOOD
    )
    with pytest.raises(bp.BuildProvenanceError, match="does not match the reviewed floor"):
        bp.validate_wheel_provenance_for_scientific_admission(
            str(wheel), expected_required_git_ancestor="0" * 40
        )


# ---- build-worker-wheel CLI: the AUTHORITATIVE scientific builder (real source builds) ------------
#
# The production-path proof invokes the REAL source build (`python -m build`), never a stamp-an-
# arbitrary-wheel shortcut (that public path was removed). Refusal tests short-circuit before the build.


def _minimal_source_repo(root: Path) -> str:
    """A minimal but buildable, clean git source tree; returns HEAD (the build source commit)."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=64"]\nbuild-backend = "setuptools.build_meta"\n\n'
        '[project]\nname = "corpus-studio-engine"\nversion = "0.0.1"\n\n'
        '[tool.setuptools.packages.find]\ninclude = ["corpus_studio*"]\n',
        encoding="utf-8",
    )
    pkg = root / "corpus_studio"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("# worker\n", encoding="utf-8")
    # Ignore build byproducts so a second build keeps the worktree clean for the reproducibility test.
    (root / ".gitignore").write_text("build/\ndist/\n*.egg-info/\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@e.st")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "src")
    return _git(root, "rev-parse", "HEAD")


def _invoke_build(cli_module, root: Path, dest: Path, *args: str):
    from typer.testing import CliRunner

    return CliRunner().invoke(
        cli_module.app, ["build-worker-wheel", "--dest-dir", str(dest), *args]
    )


def test_build_cli_refuses_dirty_tracked_file(tmp_path: Path, monkeypatch) -> None:
    import corpus_studio.cli as cli

    root = tmp_path / "repo"
    head = _minimal_source_repo(root)
    (root / "corpus_studio" / "__init__.py").write_text("# changed\n", encoding="utf-8")  # tracked mod
    monkeypatch.setattr(cli, "repository_root", lambda: root)
    result = _invoke_build(cli, root, tmp_path / "out", "--required-ancestor", head)
    assert result.exit_code == 2
    assert "uncommitted changes" in result.output


def test_build_cli_refuses_untracked_file(tmp_path: Path, monkeypatch) -> None:
    import corpus_studio.cli as cli

    root = tmp_path / "repo"
    head = _minimal_source_repo(root)
    (root / "stray.txt").write_text("untracked", encoding="utf-8")  # untracked, not git-ignored
    monkeypatch.setattr(cli, "repository_root", lambda: root)
    result = _invoke_build(cli, root, tmp_path / "out", "--required-ancestor", head)
    assert result.exit_code == 2
    assert "uncommitted changes" in result.output


def test_build_cli_requires_required_ancestor(tmp_path: Path, monkeypatch) -> None:
    import corpus_studio.cli as cli

    root = tmp_path / "repo"
    _minimal_source_repo(root)
    monkeypatch.setattr(cli, "repository_root", lambda: root)
    result = _invoke_build(cli, root, tmp_path / "out")  # no --required-ancestor
    assert result.exit_code != 0  # typer refuses the missing mandatory research floor


def test_build_cli_refuses_reversed_ancestry(tmp_path: Path, monkeypatch) -> None:
    import corpus_studio.cli as cli

    root = tmp_path / "repo"
    c1 = _minimal_source_repo(root)
    (root / "corpus_studio" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "c2")
    c2 = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-q", c1)  # detached HEAD at c1 -> source_commit = c1
    monkeypatch.setattr(cli, "repository_root", lambda: root)
    # floor = c2 (a DESCENDANT of the source c1) -> c1 does not descend from c2 -> refused.
    result = _invoke_build(cli, root, tmp_path / "out", "--required-ancestor", c2)
    assert result.exit_code == 2
    assert "does not descend from" in result.output


def test_build_cli_refuses_unrelated_branch_floor(tmp_path: Path, monkeypatch) -> None:
    import corpus_studio.cli as cli

    root = tmp_path / "repo"
    c1 = _minimal_source_repo(root)
    default_branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")  # 'master' or 'main'
    # An unrelated branch off c1 whose tip is not in the default branch's history.
    _git(root, "checkout", "-q", "-b", "side", c1)
    (root / "corpus_studio" / "side.py").write_text("y = 2\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "side")
    side = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-q", default_branch)
    # advance the default branch so HEAD != c1 and 'side' is on an unrelated line of history
    (root / "corpus_studio" / "mainmod.py").write_text("z = 3\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "c2")
    monkeypatch.setattr(cli, "repository_root", lambda: root)
    result = _invoke_build(cli, root, tmp_path / "out", "--required-ancestor", side)
    assert result.exit_code == 2
    assert "does not descend from" in result.output


def test_build_cli_has_no_arbitrary_wheel_relabel_path(tmp_path: Path, monkeypatch) -> None:
    import corpus_studio.cli as cli

    root = tmp_path / "repo"
    head = _minimal_source_repo(root)
    monkeypatch.setattr(cli, "repository_root", lambda: root)
    prebuilt = _minimal_wheel(tmp_path / "corpus_studio_engine-9.9.9-py3-none-any.whl")
    # The removed public --source-wheel path is gone: arbitrary prebuilt bytes cannot be relabeled HEAD.
    result = _invoke_build(
        cli, root, tmp_path / "out", "--required-ancestor", head, "--source-wheel", str(prebuilt)
    )
    assert result.exit_code != 0
    assert "source-wheel" in result.output.lower() or "no such option" in result.output.lower()


def test_build_cli_real_source_build_embeds_provenance_and_admits(tmp_path: Path, monkeypatch) -> None:
    import corpus_studio.cli as cli

    root = tmp_path / "repo"
    head = _minimal_source_repo(root)
    monkeypatch.setattr(cli, "repository_root", lambda: root)
    dest = tmp_path / "out"
    result = _invoke_build(cli, root, dest, "--required-ancestor", head)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["source_commit"] == head
    assert payload["required_git_ancestor"] == head
    wheel = Path(payload["wheel"])
    assert wheel.exists() and wheel.suffix == ".whl"
    # Real build -> embedded provenance -> passes Environment Manager admission -> telemetry reads it.
    assert bp.read_embedded_source_commit(str(wheel)) == head
    assert bp.validate_wheel_provenance_for_scientific_admission(str(wheel)) == head
    overlay = worker_identity_overlay(_artifact(wheel))
    assert overlay.repository_commit == head


def test_build_cli_two_builds_same_commit_are_byte_identical(tmp_path: Path, monkeypatch) -> None:
    import corpus_studio.cli as cli

    root1 = tmp_path / "repo1"
    head = _minimal_source_repo(root1)
    root2 = tmp_path / "repo2"
    shutil.copytree(root1, root2)  # identical source + .git (same commit) -> same SOURCE_DATE_EPOCH

    monkeypatch.setattr(cli, "repository_root", lambda: root1)
    r1 = _invoke_build(cli, root1, tmp_path / "out1", "--required-ancestor", head)
    assert r1.exit_code == 0, r1.output
    monkeypatch.setattr(cli, "repository_root", lambda: root2)
    r2 = _invoke_build(cli, root2, tmp_path / "out2", "--required-ancestor", head)
    assert r2.exit_code == 0, r2.output

    w1 = Path(json.loads(r1.stdout)["wheel"])
    w2 = Path(json.loads(r2.stdout)["wheel"])
    assert (
        hashlib.sha256(w1.read_bytes()).hexdigest()
        == hashlib.sha256(w2.read_bytes()).hexdigest()
    )
