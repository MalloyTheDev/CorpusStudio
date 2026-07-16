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

import json
import re
import subprocess  # noqa: S404 - fixed-argv git only; never a shell string.
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
    """Best-effort strict read of the canonical ``source_commit`` from the wheel's sidecar.

    Returns the commit ONLY when the sidecar exists, is valid JSON, and carries a canonical
    ``source_commit`` (exact 40-char lowercase hex). Returns ``None`` - never raises, never fabricates,
    never falls back to a prohibited alias - when the sidecar is absent/unreadable, the key is missing,
    or the value is non-canonical (abbreviated, uppercase, empty, or non-string). This is the exact
    contract the telemetry identity reader relies on.
    """

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
    must_be_ancestor_of: str | None = None,
) -> None:
    """Validate a canonical ``source_commit`` against a Git repository, raising on any failure.

    Checks (all required unless parameterized off):
    - ``source_commit`` is canonical (exact 40-char lowercase hex);
    - the commit object exists in ``repo_root``;
    - ``require_clean_worktree`` (default): the worktree has no tracked modifications (a dirty tree is
      ambiguous provenance and is refused). A detached, clean, reproducible build worktree passes;
    - ``must_be_ancestor_of`` (when given): ``source_commit`` is an ancestor of, or equal to, that
      commit (the required-descent rule).
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
    if must_be_ancestor_of is not None:
        ancestor = _git(root, "merge-base", "--is-ancestor", source_commit, must_be_ancestor_of)
        if ancestor.returncode != 0:
            raise BuildProvenanceError(
                f"source_commit {source_commit} is not an ancestor of {must_be_ancestor_of}"
            )


def validate_wheel_provenance_for_scientific_admission(
    wheel_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    require_clean_worktree: bool = True,
    must_be_ancestor_of: str | None = None,
) -> str:
    """Gate a worker wheel for admission into a scientific environment; return its ``source_commit``.

    Raises ``BuildProvenanceError`` if the sidecar is absent, malformed, or lacks a canonical
    ``source_commit`` (so a wheel that would leave ``identity.repository_commit`` null - the v7 defect -
    is refused before admission, never patched post-hoc). When ``repo_root`` is given, the commit is
    additionally validated against the repository (existence, clean worktree, required ancestry).
    """

    provenance = provenance_path_for_wheel(wheel_path)
    if provenance is None or not provenance.exists():
        raise BuildProvenanceError(
            f"no {PROVENANCE_FILENAME} sidecar next to wheel {wheel_path}"
        )
    try:
        data = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BuildProvenanceError(f"unreadable {PROVENANCE_FILENAME}: {exc}") from exc
    if not isinstance(data, dict):
        raise BuildProvenanceError(f"{PROVENANCE_FILENAME} is not a JSON object")
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
            f"{PROVENANCE_FILENAME} source_commit is {detail}{hint}"
        )
    assert isinstance(commit, str)  # narrowed by is_canonical_source_commit
    if repo_root is not None:
        validate_source_commit_against_repo(
            commit,
            repo_root,
            require_clean_worktree=require_clean_worktree,
            must_be_ancestor_of=must_be_ancestor_of,
        )
    return commit
