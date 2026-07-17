"""Canonical worker-wheel build provenance: the ``BUILD_PROVENANCE.json`` sidecar written next to a
sealed worker wheel in the artifact store.

Why this module exists (the v7 lesson): the telemetry identity reader requires the exact key
``source_commit`` to populate ``identity.repository_commit`` for a run's scientific summary. The v7 wheel
was built by ad-hoc tooling that recorded the authentic commit under ``audited_commit`` and omitted
``source_commit``, so every auto-derived v7 summary was ``scientific_resource_complete=false`` even though
all measurement was complete, and the commit had to be supplied post-hoc through an identity overlay. This
module makes the contract canonical and machine-checked so FUTURE wheels populate the identity WITHOUT an
overlay and a wheel with malformed/absent provenance fails before it is admitted for a scientific
environment.

Torch-free by construction (stdlib + git subprocess only); safe to import from ``corpus_studio.platform``.

Canonical contract: ``BUILD_PROVENANCE.json`` MUST contain a string ``source_commit`` that is an exact
40-character lowercase hex Git SHA. ``audited_commit`` is NOT an accepted alias for ``source_commit`` -
it may exist as a separate, explicitly-defined, reader-ignored companion field, but it never substitutes
for a canonical ``source_commit`` (no silent fallback across ambiguous keys).
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess  # noqa: S404 - fixed-argv git only; never a shell string.
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROVENANCE_FILENAME = "BUILD_PROVENANCE.json"

# Exact full lowercase hex Git object name. Abbreviated or uppercase SHAs are rejected: an ambiguous or
# non-canonical commit id must never masquerade as authoritative scientific provenance.
_SOURCE_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

# The reader consumes ONLY this key. Listed here so the prohibition is explicit and testable.
CANONICAL_SOURCE_COMMIT_KEY = "source_commit"
PROHIBITED_SOURCE_COMMIT_ALIASES = ("audited_commit",)


class BuildProvenanceError(Exception):
    """A worker wheel's build provenance is missing, malformed, or disagrees with the repository.

    Raised at build/admission time (never by the best-effort telemetry reader, which returns ``None``).
    """


def is_canonical_source_commit(value: Any) -> bool:
    """True iff ``value`` is an exact 40-character lowercase hex Git commit id."""

    return isinstance(value, str) and bool(_SOURCE_COMMIT_RE.fullmatch(value))


def provenance_path_for_wheel(wheel_path: str | Path | None) -> Path | None:
    """The sidecar path next to a sealed wheel, or ``None`` when no wheel path is known."""

    if not wheel_path:
        return None
    return Path(wheel_path).parent / PROVENANCE_FILENAME


def read_source_commit(wheel_path: str | Path | None) -> str | None:
    """Best-effort strict read of the canonical ``source_commit`` for a worker wheel.

    Prefers the provenance EMBEDDED in the wheel (``*.dist-info/BUILD_PROVENANCE.json``, sealed by the
    wheel RECORD and therefore by the wheel sha256 identity); falls back to the external
    ``BUILD_PROVENANCE.json`` sidecar next to the wheel (the historical v7 shape). Returns the commit
    ONLY when a canonical ``source_commit`` (exact 40-char lowercase hex) is found. Returns ``None`` -
    never raises, never fabricates, never falls back to a prohibited alias - when neither source carries
    one. This is the exact contract the telemetry identity reader relies on, so telemetry obtains
    ``repository_commit`` automatically from a self-describing wheel without any post-hoc overlay.
    """

    embedded = read_embedded_source_commit(wheel_path)
    if embedded is not None:
        return embedded
    provenance = provenance_path_for_wheel(wheel_path)
    if provenance is None:
        return None
    try:
        data = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    commit = data.get(CANONICAL_SOURCE_COMMIT_KEY)
    return commit if is_canonical_source_commit(commit) else None


def build_provenance_document(
    *,
    source_commit: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a canonical provenance document, validating ``source_commit`` up front.

    ``extra`` may carry additional descriptive fields (e.g. ``built_at``, ``engine_tree_git_sha``,
    ``required_git_ancestor``, ``audited_commit`` as an explicit companion). ``extra`` may NOT override
    or shadow the canonical ``source_commit`` key.
    """

    if not is_canonical_source_commit(source_commit):
        raise BuildProvenanceError(
            f"source_commit must be an exact 40-character lowercase hex git sha, got {source_commit!r}"
        )
    document: dict[str, Any] = {CANONICAL_SOURCE_COMMIT_KEY: source_commit}
    if extra:
        if CANONICAL_SOURCE_COMMIT_KEY in extra:
            raise BuildProvenanceError(
                "extra provenance fields may not override the canonical source_commit"
            )
        # An embedded research floor is admission-critical: validate it here so no first-party path
        # (CLI or library caller) can stamp a wheel whose embedded required_git_ancestor is non-canonical
        # and only get caught later at admission.
        extra_floor = extra.get("required_git_ancestor")
        if extra_floor is not None and not is_canonical_source_commit(extra_floor):
            raise BuildProvenanceError(
                "extra required_git_ancestor must be an exact 40-character lowercase hex git sha, got "
                f"{extra_floor!r}"
            )
        document.update(extra)
    return document


def write_build_provenance(dest_dir: str | Path, document: dict[str, Any]) -> Path:
    """Write a validated canonical ``BUILD_PROVENANCE.json`` into ``dest_dir`` and return its path.

    Refuses to write a document whose ``source_commit`` is absent or non-canonical, so a malformed
    sidecar can never be produced by the first-party path.
    """

    if not is_canonical_source_commit(document.get(CANONICAL_SOURCE_COMMIT_KEY)):
        raise BuildProvenanceError(
            "refusing to write BUILD_PROVENANCE.json without a canonical source_commit"
        )
    destination = Path(dest_dir) / PROVENANCE_FILENAME
    destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed git binary, no shell, repository-owned args.
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def validate_source_commit_against_repo(
    source_commit: str,
    repo_root: str | Path,
    *,
    require_clean_worktree: bool = True,
    required_git_ancestor: str | None = None,
) -> None:
    """Validate a canonical ``source_commit`` against a Git repository, raising on any failure.

    Checks (all required unless parameterized off):
    - ``source_commit`` is canonical (exact 40-char lowercase hex);
    - the commit object exists in ``repo_root``;
    - ``require_clean_worktree`` (default): the worktree has no tracked modifications (a dirty tree is
      ambiguous provenance and is refused). A detached, clean, reproducible build worktree passes;
    - ``required_git_ancestor`` (when given): the required-descent rule, in the direction the research
      protocol uses - ``required_git_ancestor`` MUST be an ancestor of (or equal to) ``source_commit``.
      That is, the built source DESCENDS FROM the required floor commit (e.g. the merged
      build-provenance-fix commit or a prior worker-source floor); the floor is not required to descend
      from the source. This is the opposite direction from a naive ``source is ancestor of X`` check.
    """

    if not is_canonical_source_commit(source_commit):
        raise BuildProvenanceError(
            f"source_commit must be an exact 40-character lowercase hex git sha, got {source_commit!r}"
        )
    root = Path(repo_root)
    exists = _git(root, "cat-file", "-e", f"{source_commit}^{{commit}}")
    if exists.returncode != 0:
        raise BuildProvenanceError(
            f"source_commit {source_commit} does not exist as a commit in {root}"
        )
    if require_clean_worktree:
        status = _git(root, "status", "--porcelain")
        if status.returncode != 0:
            raise BuildProvenanceError(f"unable to read worktree status for {root}")
        if status.stdout.strip():
            raise BuildProvenanceError(
                "refusing ambiguous provenance: the build worktree has uncommitted changes"
            )
    if required_git_ancestor is not None:
        if not is_canonical_source_commit(required_git_ancestor):
            raise BuildProvenanceError(
                "required_git_ancestor must be an exact 40-character lowercase hex git sha, got "
                f"{required_git_ancestor!r}"
            )
        # required_git_ancestor -> ancestor of source_commit: the floor must be reachable from source.
        exists_floor = _git(root, "cat-file", "-e", f"{required_git_ancestor}^{{commit}}")
        if exists_floor.returncode != 0:
            raise BuildProvenanceError(
                f"required_git_ancestor {required_git_ancestor} does not exist as a commit in {root}"
            )
        descends = _git(
            root, "merge-base", "--is-ancestor", required_git_ancestor, source_commit
        )
        if descends.returncode != 0:
            raise BuildProvenanceError(
                f"source_commit {source_commit} does not descend from required_git_ancestor "
                f"{required_git_ancestor}"
            )


def validate_wheel_provenance_for_scientific_admission(
    wheel_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    require_clean_worktree: bool = True,
    required_git_ancestor: str | None = None,
    expected_required_git_ancestor: str | None = None,
    expected_source_commit: str | None = None,
) -> str:
    """Gate a worker wheel for admission into a scientific environment; return its ``source_commit``.

    This is the artifact-self-contained check the Environment Manager runs at admission (before any
    environment directory, installation, lock, capability probe, or GPU operation), so it does NOT
    require the source repository to be present on the host. It enforces, in order:

    1. The wheel EMBEDS ``*.dist-info/BUILD_PROVENANCE.json`` carrying a canonical ``source_commit``
       (exact 40-char lowercase hex). A wheel with only an external sidecar, only a prohibited alias
       such as ``audited_commit``, or no provenance at all - the v7 defect shape - is refused here,
       never patched post-hoc.
    2. The embedded provenance ALSO carries a canonical ``required_git_ancestor`` (exact 40-char
       lowercase hex). This is ALWAYS required - even when no ``expected_required_git_ancestor`` and no
       ``repo_root`` are supplied - so a ``source_commit``-only wheel (a floor absent, null, non-string,
       abbreviated, uppercase, or otherwise malformed) is refused here. This is what makes admission's
       "canonical embedded floor" guarantee true rather than conditional.
    3. The embedded provenance is SEALED by the wheel RECORD (its listed sha256 matches the bytes), so
       it is a first-class wheel member covered by the wheel ``content_hash`` identity, not a stray
       file dropped into the zip.
    4. If an external ``BUILD_PROVENANCE.json`` copy sits next to the wheel, it must byte-match the
       embedded copy (no divergent provenance stories).

    IMPORTANT honesty boundary - what admission does and does NOT prove about the protocol floor:
    admission verifies the wheel's own EMBEDDED SELF-ASSERTION (integrity, presence, canonical
    ``source_commit`` AND canonical ``required_git_ancestor``, RECORD-seal, external-match). It does NOT,
    on its own, verify that the embedded floor is the CORRECT reviewed protocol floor, nor that
    ``source_commit`` actually descends from it - those are facts that need more than the wheel. Three
    optional inputs let a caller add real checks:
    - ``repo_root`` (build-time, repo present): re-validate ``source_commit`` against the repository -
      existence, clean worktree, and ``required_git_ancestor`` descent.
    - ``expected_required_git_ancestor``: a reviewed floor the CALLER already trusts (e.g. from a plan).
      When given, the embedded floor must equal it exactly (a value match, no repo needed).
    - ``expected_source_commit``: a reviewed source commit the CALLER already trusts. When given, the
      embedded ``source_commit`` must equal it exactly - the same value-match discipline as the floor,
      applied to the field that becomes the recorded scientific provenance.
    The Environment Manager admission gate now passes ``expected_required_git_ancestor``: the exact
    reviewed per-lineage floor carried on the sealed ``DependencyResolution`` (supplied to
    ``env-plan --required-git-ancestor`` and bound by the confirmation hash). So admission proves EXACT
    EQUALITY between the confirmed plan floor and the wheel's embedded floor. The Environment Manager
    still does NOT pass ``repo_root`` (no source repository reaches a scientific host), so admission does
    NOT independently prove Git ANCESTRY - it does not prove ``source_commit`` descends from the floor,
    only that the embedded floor equals the confirmed plan floor. Descent is proven where the repo is
    present: at BUILD time by the authoritative clean-source ``build-worker-wheel`` (``--required-ancestor``
    against the real repo) and later by the prospective research-protocol validator. Equality (what this
    gate proves) and ancestry (what build/validator prove) are SEPARATE claims, not restatements.
    """

    data = read_embedded_provenance(wheel_path)
    if data is None:
        raise BuildProvenanceError(
            f"worker wheel {wheel_path} has no embedded {PROVENANCE_FILENAME} "
            "(an external-only or absent sidecar is not admissible)"
        )
    commit = data.get(CANONICAL_SOURCE_COMMIT_KEY)
    if not is_canonical_source_commit(commit):
        detail = "absent" if commit is None else f"non-canonical ({commit!r})"
        present_aliases = [a for a in PROHIBITED_SOURCE_COMMIT_ALIASES if a in data]
        hint = (
            f" (found prohibited alias {present_aliases!r} which is NOT accepted as source_commit)"
            if present_aliases
            else ""
        )
        raise BuildProvenanceError(
            f"embedded {PROVENANCE_FILENAME} source_commit is {detail}{hint}"
        )
    assert isinstance(commit, str)  # narrowed by is_canonical_source_commit
    # The embedded research floor is ALWAYS required and must be canonical - independent of whether an
    # expected floor or a repository is supplied. A source-commit-only wheel is refused here.
    embedded_floor = data.get("required_git_ancestor")
    if not is_canonical_source_commit(embedded_floor):
        detail = "absent" if embedded_floor is None else f"non-canonical ({embedded_floor!r})"
        raise BuildProvenanceError(
            f"embedded {PROVENANCE_FILENAME} required_git_ancestor is {detail}"
        )
    assert isinstance(embedded_floor, str)  # narrowed by is_canonical_source_commit
    verify_embedded_provenance_record_integrity(wheel_path)
    _assert_external_matches_embedded(wheel_path)
    if (
        expected_required_git_ancestor is not None
        and embedded_floor != expected_required_git_ancestor
    ):
        raise BuildProvenanceError(
            "embedded required_git_ancestor "
            f"{embedded_floor!r} does not match the reviewed floor "
            f"{expected_required_git_ancestor!r}"
        )
    # A reviewed source_commit the CALLER already trusts (e.g. one recorded in the sealed plan or the
    # research protocol). When given, the wheel's embedded source_commit must equal it exactly - the
    # same value-match discipline the floor gets, so the field that becomes the recorded scientific
    # provenance is not merely format-checked.
    if expected_source_commit is not None and commit != expected_source_commit:
        raise BuildProvenanceError(
            f"embedded source_commit {commit!r} does not match the reviewed source commit "
            f"{expected_source_commit!r}"
        )
    if repo_root is not None:
        validate_source_commit_against_repo(
            commit,
            repo_root,
            require_clean_worktree=require_clean_worktree,
            required_git_ancestor=required_git_ancestor,
        )
    return commit


# --------------------------------------------------------------------------------------------------
# Wheel-embedded provenance: write, read, and RECORD-seal integrity.
#
# The wheel is a zip; ``*.dist-info/BUILD_PROVENANCE.json`` is embedded as a first-class member listed
# in ``*.dist-info/RECORD``. Because RECORD lists it (with its sha256) and the whole wheel's bytes are
# the sealed ``worker_wheel_sha256`` identity that already flows through the artifact, environment,
# lock, plan, execution, and telemetry contracts, embedding needs NO new sealed-identity contract - the
# provenance rides inside the identity that already exists. An optional external copy is defense in
# depth only and must match the embedded copy byte-for-byte.
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WheelProvenanceStamp:
    """The result of embedding canonical provenance into a worker wheel."""

    source_commit: str
    wheel_sha256: str
    embedded_arcname: str
    external_path: Path | None


def _record_field(data: bytes) -> str:
    """The RECORD hash field for ``data``: ``sha256=<urlsafe-base64-no-padding>``."""

    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")
    return f"sha256={digest}"


def _record_arcname(names: list[str]) -> str:
    for name in names:
        if name.endswith(".dist-info/RECORD"):
            return name
    raise BuildProvenanceError("wheel has no .dist-info/RECORD member")


def _embedded_arcname(names: list[str]) -> str | None:
    return next(
        (name for name in names if name.endswith(f".dist-info/{PROVENANCE_FILENAME}")), None
    )


def _provenance_payload(document: dict[str, Any]) -> bytes:
    """The exact canonical bytes for a provenance document (embedded and external copies share these)."""

    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def stamp_wheel_with_provenance(
    wheel_path: str | Path,
    document: dict[str, Any],
    *,
    external_copy: bool = True,
) -> WheelProvenanceStamp:
    """Embed a validated canonical ``BUILD_PROVENANCE.json`` into ``wheel_path`` and re-seal RECORD.

    Rewrites the wheel in place (via a temp file + atomic replace) with the provenance added under the
    wheel's ``.dist-info/`` and its line inserted into RECORD (so the embedded copy is sealed and the
    wheel sha256 - the worker identity - covers it). Refuses a document without a canonical
    ``source_commit``. When ``external_copy`` is set, also writes an identical external sidecar next to
    the wheel (defense in depth; admission requires it to match the embedded copy). Returns the new
    wheel sha256 and the embedded arcname.
    """

    commit = document.get(CANONICAL_SOURCE_COMMIT_KEY)
    if not is_canonical_source_commit(commit):
        raise BuildProvenanceError(
            "refusing to stamp a wheel without a canonical source_commit"
        )
    assert isinstance(commit, str)
    payload = _provenance_payload(document)
    wheel = Path(wheel_path)

    with zipfile.ZipFile(wheel) as archive:
        infos = list(archive.infolist())
        names = [info.filename for info in infos]
        record_name = _record_arcname(names)
        dist_info = record_name[: -len("RECORD")]
        prov_name = f"{dist_info}{PROVENANCE_FILENAME}"
        member_bytes = {info.filename: archive.read(info.filename) for info in infos}

    # Rebuild RECORD: keep every line except any prior provenance line and RECORD's own line, append the
    # provenance line, then RECORD's hashless self-line last (per the wheel spec).
    record_text = member_bytes[record_name].decode("utf-8")
    kept: list[str] = []
    for line in record_text.splitlines():
        if not line.strip():
            continue
        first = line.split(",", 1)[0]
        if first == prov_name or first == record_name:
            continue
        kept.append(line)
    kept.append(f"{prov_name},{_record_field(payload)},{len(payload)}")
    kept.append(f"{record_name},,")
    new_record = ("\n".join(kept) + "\n").encode("utf-8")

    fixed_dt = (1980, 1, 1, 0, 0, 0)  # deterministic mtime for stamped members (reproducible stamp)
    tmp = wheel.with_name(wheel.name + ".provenance.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for info in infos:
            if info.filename in (record_name, prov_name):
                continue  # (re)written below with fresh, deterministic metadata
            out.writestr(info, member_bytes[info.filename])
        prov_info = zipfile.ZipInfo(prov_name, date_time=fixed_dt)
        prov_info.compress_type = zipfile.ZIP_DEFLATED
        prov_info.external_attr = 0o644 << 16
        out.writestr(prov_info, payload)
        record_info = zipfile.ZipInfo(record_name, date_time=fixed_dt)
        record_info.compress_type = zipfile.ZIP_DEFLATED
        record_info.external_attr = 0o644 << 16
        out.writestr(record_info, new_record)
    tmp.replace(wheel)

    wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    external_path: Path | None = None
    if external_copy:
        external_path = wheel.parent / PROVENANCE_FILENAME
        external_path.write_bytes(payload)
    return WheelProvenanceStamp(
        source_commit=commit,
        wheel_sha256=wheel_sha256,
        embedded_arcname=prov_name,
        external_path=external_path,
    )


def read_embedded_provenance(wheel_path: str | Path | None) -> dict[str, Any] | None:
    """Best-effort read of the embedded ``*.dist-info/BUILD_PROVENANCE.json`` as a dict, or ``None``.

    Never raises: a missing wheel, a non-zip file, an absent member, or malformed JSON all yield
    ``None`` so telemetry stays best-effort.
    """

    if not wheel_path:
        return None
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            arcname = _embedded_arcname(archive.namelist())
            if arcname is None:
                return None
            data = json.loads(archive.read(arcname).decode("utf-8"))
    except (OSError, ValueError, zipfile.BadZipFile):
        return None
    return data if isinstance(data, dict) else None


def read_embedded_source_commit(wheel_path: str | Path | None) -> str | None:
    """The canonical embedded ``source_commit`` for a wheel, or ``None`` (best-effort, never raises)."""

    data = read_embedded_provenance(wheel_path)
    if data is None:
        return None
    commit = data.get(CANONICAL_SOURCE_COMMIT_KEY)
    return commit if is_canonical_source_commit(commit) else None


def verify_embedded_provenance_record_integrity(wheel_path: str | Path) -> None:
    """Raise unless the embedded provenance is listed in RECORD with a matching sha256.

    Proves the embedded ``BUILD_PROVENANCE.json`` is a first-class, RECORD-sealed wheel member (covered
    by the wheel sha256 identity), not a stray file that a repackager could swap without detection.
    """

    try:
        with zipfile.ZipFile(wheel_path) as archive:
            names = archive.namelist()
            arcname = _embedded_arcname(names)
            if arcname is None:
                raise BuildProvenanceError(
                    f"wheel {wheel_path} has no embedded {PROVENANCE_FILENAME}"
                )
            record_name = _record_arcname(names)
            payload = archive.read(arcname)
            record_text = archive.read(record_name).decode("utf-8")
    except (OSError, zipfile.BadZipFile) as exc:
        raise BuildProvenanceError(f"unreadable wheel {wheel_path}: {exc}") from exc
    expected = _record_field(payload)
    for line in record_text.splitlines():
        fields = line.split(",")
        if fields and fields[0] == arcname:
            if len(fields) >= 2 and fields[1] == expected:
                return
            raise BuildProvenanceError(
                f"embedded {PROVENANCE_FILENAME} hash disagrees with RECORD for {wheel_path}"
            )
    raise BuildProvenanceError(
        f"embedded {PROVENANCE_FILENAME} is not listed in RECORD for {wheel_path}"
    )


def _assert_external_matches_embedded(wheel_path: str | Path) -> None:
    """If an external sidecar exists next to the wheel, require it to match the embedded copy byte for
    byte. A wheel with no external copy is fine (embedded is authoritative)."""

    external = provenance_path_for_wheel(wheel_path)
    if external is None or not external.exists():
        return
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            arcname = _embedded_arcname(archive.namelist())
            embedded = archive.read(arcname) if arcname is not None else b""
    except (OSError, zipfile.BadZipFile) as exc:
        raise BuildProvenanceError(f"unreadable wheel {wheel_path}: {exc}") from exc
    if external.read_bytes() != embedded:
        raise BuildProvenanceError(
            f"external {PROVENANCE_FILENAME} next to {wheel_path} disagrees with the embedded copy"
        )
