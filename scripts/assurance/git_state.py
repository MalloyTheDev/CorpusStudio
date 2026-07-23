"""Read-only Git plumbing for the assurance change-set kernel (Phase 1).

Every git call is a fixed ``argv`` list run with ``git -C <root>`` - never a shell string. These
calls READ repository state: none mutates the object database, the refs, the committed tree, or the
working tree, so the committed content the repository represents - and the computed change set - are
never altered. (One content-neutral exception is honest to state: a read such as ``git diff`` may
refresh the index's cached stat metadata - mtime/size - to reflect the current working tree; this
changes no tracked bytes, no tree, and no change-set result. ``GIT_OPTIONAL_LOCKS=0`` is set to
avoid the optional-lock, status-style refreshes; a plumbing ``diff`` still refreshes the stat-cache,
which is why the promise here is "never mutates committed state", not "never touches .git/index".)
All name-bearing output is requested in the NUL-delimited ``-z`` form so paths are never quoted or
ambiguous; a path that is not valid UTF-8 fails closed (assurance records require UTF-8 POSIX paths).

The base source view is derived from the git object store (``ls-tree`` / ``cat-file``), so a
base tree can be inspected WITHOUT checking it out.
"""

from __future__ import annotations

import os
import subprocess  # noqa: S404 - fixed-argv git only; never a shell string.
from dataclasses import dataclass
from pathlib import Path


class AssuranceError(Exception):
    """Base class for every fail-closed assurance-kernel refusal (maps to CLI exit code 2)."""


class GitStateError(AssuranceError):
    """A git operation failed or returned something the kernel refuses to interpret."""


class NotAGitRepo(GitStateError):
    """The start directory is not inside a git working tree."""


class MissingBaseRef(GitStateError):
    """The requested ``--base`` ref does not resolve to a commit."""


class NoMergeBase(GitStateError):
    """HEAD and the base commit have no common ancestor (unrelated histories / unborn HEAD)."""


class ShallowHistoryLimitation(GitStateError):
    """A merge base could not be computed because the clone's history is shallow (truncated)."""


class UnsupportedPathEncoding(GitStateError):
    """A path (or symlink target) is not valid UTF-8; assurance records require UTF-8."""


@dataclass(frozen=True)
class GitContext:
    """The resolved identity of the working tree the kernel was pointed at."""

    root: Path  # absolute repository top-level (of this worktree)
    git_dir: Path  # absolute git dir (a linked worktree has its own)
    head_oid: str  # 40-hex commit at HEAD, or "" for an unborn branch
    is_shallow: bool


@dataclass(frozen=True)
class TreeEntry:
    """One entry of a git tree: a blob (regular/symlink) or a commit (gitlink)."""

    mode: str  # 6-digit git mode, e.g. "100644", "120000", "160000"
    type: str  # "blob" | "commit" | "tree"
    oid: str  # 40-hex object id
    path: str  # repository-relative POSIX path


# A generous per-call bound so a wedged git (giant blob, stalled filesystem, a credential prompt)
# can never hang the loop unbounded - every subprocess in the kernel has a timeout.
_GIT_TIMEOUT_S = 120


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    """Run ``git -C <root> <args...>`` capturing bytes. Raises ``GitStateError`` on failure.

    ``GIT_OPTIONAL_LOCKS=0`` avoids taking the optional index lock for status-style refreshes (see
    the module docstring for the honest scope of "read-only"). A missing/unrunnable ``git`` binary
    fails CLOSED as ``GitStateError`` rather than escaping as an uncaught ``OSError``.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, no untrusted executable.
            ["git", "-C", str(root), *args],
            capture_output=True,
            check=False,
            timeout=_GIT_TIMEOUT_S,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except subprocess.TimeoutExpired as exc:  # a wedged git (stalled FS, credential prompt) fails closed
        raise GitStateError(f"git {' '.join(args[:2])} timed out after {_GIT_TIMEOUT_S}s") from exc
    except OSError as exc:  # e.g. git not installed / not on PATH
        raise GitStateError(f"cannot run git {' '.join(args[:2])}: {exc}") from exc
    if check and proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise GitStateError(f"git {' '.join(args[:2])} failed (exit {proc.returncode}): {stderr}")
    return proc


def _decode_utf8(raw: bytes, *, what: str) -> str:
    """Decode git output that must be UTF-8 (paths, targets); fail closed otherwise."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedPathEncoding(
            f"{what} is not valid UTF-8 ({raw!r}); assurance records require UTF-8 POSIX paths"
        ) from exc


def discover_git_context(start: Path) -> GitContext:
    """Resolve the working tree at (or above) ``start``. Fails closed if it is not a repo."""
    top = _git(start, "rev-parse", "--show-toplevel", check=False)
    if top.returncode != 0:
        raise NotAGitRepo(f"{start} is not inside a git working tree")
    root = Path(_decode_utf8(top.stdout.strip(), what="repository root"))
    git_dir = Path(_git(root, "rev-parse", "--absolute-git-dir").stdout.decode("utf-8").strip())
    is_shallow = (
        _git(root, "rev-parse", "--is-shallow-repository").stdout.decode("utf-8").strip() == "true"
    )
    head = _git(root, "rev-parse", "--verify", "--quiet", "HEAD", check=False)
    head_oid = head.stdout.decode("utf-8").strip() if head.returncode == 0 else ""
    return GitContext(root=root, git_dir=git_dir, head_oid=head_oid, is_shallow=is_shallow)


def resolve_commit(root: Path, ref: str) -> str:
    """Resolve ``ref`` to a 40-hex commit oid; fail closed if it does not name a commit."""
    proc = _git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
    oid = proc.stdout.decode("utf-8").strip()
    if proc.returncode != 0 or not oid:
        raise MissingBaseRef(f"base ref does not resolve to a commit: {ref!r}")
    return oid


def merge_base(ctx: GitContext, base_commit: str) -> str:
    """Return the merge-base of HEAD and ``base_commit`` (read-only). Fails closed on absence.

    A shallow clone whose history is truncated is reported distinctly from genuinely unrelated
    histories, because the two demand different operator responses.
    """
    if not ctx.head_oid:
        raise NoMergeBase("HEAD is unborn (no commits); cannot compute a merge base")
    proc = _git(ctx.root, "merge-base", ctx.head_oid, base_commit, check=False)
    oid = proc.stdout.decode("utf-8").strip()
    if proc.returncode != 0 or not oid:
        if ctx.is_shallow:
            raise ShallowHistoryLimitation(
                f"no merge base between HEAD and {base_commit} in a shallow clone "
                "(history is truncated; unshallow to compare)"
            )
        raise NoMergeBase(f"no merge base between HEAD ({ctx.head_oid}) and {base_commit}")
    return oid


def read_committed_file(ctx: GitContext, commit: str, relpath: str) -> bytes | None:
    """The bytes of a repo-relative file AS COMMITTED at ``commit`` (read-only), or None if the path did
    not exist at that commit. Fails closed (GitStateError) on a git error other than an absent path - so
    a trusted-base file (e.g. an earlier policy) can be read without touching the working tree."""
    proc = _git(ctx.root, "show", f"{commit}:{relpath}", check=False)
    if proc.returncode != 0:
        return None  # the path did not exist at that commit
    return proc.stdout


def changed_tracked_paths(root: Path, base_commit: str) -> list[str]:
    """Repository-relative paths of TRACKED files differing between ``base_commit`` and the tree.

    Renames are deliberately disabled (``--no-renames``) so a rename appears canonically as a
    delete of the old path plus an add of the new path.
    """
    proc = _git(root, "diff", "--raw", "-z", "--no-renames", base_commit, "--")
    return _parse_diff_raw_z(proc.stdout)


def _parse_diff_raw_z(raw: bytes) -> list[str]:
    """Parse ``git diff --raw -z`` output into the changed path list.

    Each entry (with ``--no-renames``) is two NUL-terminated tokens: a metadata field beginning
    with ``:`` (``:<srcmode> <dstmode> <srcsha> <dstsha> <status>``) followed by the path.
    """
    tokens = raw.split(b"\x00")
    if tokens and tokens[-1] == b"":
        tokens.pop()
    paths: list[str] = []
    index = 0
    while index < len(tokens):
        meta = tokens[index]
        if not meta.startswith(b":"):
            raise GitStateError(f"unexpected git diff --raw token: {meta!r}")
        if index + 1 >= len(tokens):
            raise GitStateError("git diff --raw ended without a path for the final entry")
        paths.append(_decode_utf8(tokens[index + 1], what="changed path"))
        index += 2
    return paths


def untracked_paths(root: Path) -> list[str]:
    """Repository-relative paths of untracked, NON-ignored files (``--exclude-standard``)."""
    proc = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    return [_decode_utf8(token, what="untracked path") for token in proc.stdout.split(b"\x00") if token]


def tree_entry(root: Path, tree_commit: str, path: str) -> TreeEntry | None:
    """Return the tree entry for ``path`` in ``tree_commit``, or ``None`` if absent there."""
    proc = _git(root, "ls-tree", "-z", tree_commit, "--", path)
    body = proc.stdout.strip(b"\x00")
    if not body:
        return None
    entry = proc.stdout.split(b"\x00")[0]
    meta, _sep, raw_path = entry.partition(b"\t")
    mode, obj_type, oid = meta.decode("utf-8").split()
    return TreeEntry(mode=mode, type=obj_type, oid=oid, path=_decode_utf8(raw_path, what="tree path"))


def blob_bytes(root: Path, oid: str) -> bytes:
    """Return the raw bytes of a blob object (regular content, or a symlink's target string)."""
    return _git(root, "cat-file", "-p", oid).stdout


def current_branch(ctx: GitContext) -> str:
    """The abbreviated ref of HEAD (``'main'`` / ``'feat/...'``), or ``''`` for a detached/unborn HEAD."""
    if not ctx.head_oid:
        return ""
    proc = _git(ctx.root, "rev-parse", "--abbrev-ref", "HEAD", check=False)
    if proc.returncode != 0:
        return ""
    name = _decode_utf8(proc.stdout, what="branch name").strip()  # a non-UTF8 ref fails closed, not exit 1
    return "" if name == "HEAD" else name


def recent_commits(ctx: GitContext, limit: int) -> list[dict[str, str]]:
    """Up to ``limit`` most-recent ``{oid, subject}`` in git-log recency order.

    An unborn HEAD (no commits) is a real, non-erroneous state -> ``[]`` (fail-OPEN); a git *failure*
    still fails closed via ``_git`` (``check=True``). ``%s`` is the single-line subject, so splitting on
    newlines is unambiguous.
    """
    if limit <= 0 or not ctx.head_oid:
        return []
    proc = _git(ctx.root, "log", "--no-color", f"--max-count={limit}", "--format=%H %s")
    commits: list[dict[str, str]] = []
    for line in _decode_utf8(proc.stdout, what="git log output").splitlines():
        oid, _sep, subject = line.partition(" ")
        commits.append({"oid": oid, "subject": subject})
    return commits
