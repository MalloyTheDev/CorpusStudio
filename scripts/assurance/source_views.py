"""Source views and file-state modeling for the assurance change-set kernel (Phase 1).

A *source view* answers one question: "what is the state of repository-relative path ``P`` in
this view?" Two views exist in Phase 1:

  * :class:`WorkspaceSourceView` - the exact local WORKING-TREE bytes (symlinks are NOT followed;
    a symlink is recorded by the digest of its target string, a gitlink/submodule directory and
    any non-regular special file fail closed).
  * :class:`GitTreeSourceView` - the state recorded in a committed git tree, read from the object
    store (no checkout).

Both produce the same :class:`FileState` value object, so a change is simply ``base_state !=
candidate_state`` regardless of which side is a tree and which is the working copy. This symmetry
is what lets later phases reuse these views for tree-vs-tree scopes (index / head / merge
candidate) unchanged.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from assurance.canonical_json import sha256_of_bytes
from assurance.git_state import (
    AssuranceError,
    UnsupportedPathEncoding,
    blob_bytes,
    tree_entry,
)

_MODE_REGULAR = frozenset({"100644", "100755"})
_MODE_SYMLINK = "120000"
_MODE_GITLINK = "160000"


class SourceViewError(AssuranceError):
    """A path's state cannot be determined for a source view."""


class UnsupportedSpecialFile(SourceViewError):
    """A path is a special file (fifo, socket, device, or a live submodule dir) the kernel refuses.

    The kernel refuses rather than guessing so a non-regular file is never silently hashed as if
    it were ordinary content.
    """


@dataclass(frozen=True)
class FileState:
    """The content-identity of one path within a source view (never a rename heuristic).

    ``content_digest`` is the ``sha256:`` digest of the raw file bytes for a regular file, or of
    the raw target-string bytes for a symlink. ``commit_oid`` is set only for a gitlink.
    """

    kind: str  # "regular" | "symlink" | "gitlink"
    mode: str  # git-style 6-digit mode
    content_digest: str | None = None
    commit_oid: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Serialize to the deterministic per-side record dict (omitting absent optional fields)."""
        record: dict[str, Any] = {"kind": self.kind, "mode": self.mode}
        if self.content_digest is not None:
            record["content_digest"] = self.content_digest
        if self.commit_oid is not None:
            record["commit_oid"] = self.commit_oid
        return record


class GitTreeSourceView:
    """The state of a path as recorded in a committed git tree (read via the object store)."""

    def __init__(self, root: Path, tree_commit: str) -> None:
        self.root = root
        self.tree_commit = tree_commit

    def state(self, path: str) -> FileState | None:
        entry = tree_entry(self.root, self.tree_commit, path)
        if entry is None:
            return None
        if entry.type == "commit" or entry.mode == _MODE_GITLINK:
            return FileState(kind="gitlink", mode=entry.mode, commit_oid=entry.oid)
        if entry.mode == _MODE_SYMLINK:
            target = blob_bytes(self.root, entry.oid)
            return FileState(kind="symlink", mode=entry.mode, content_digest=sha256_of_bytes(target))
        if entry.mode in _MODE_REGULAR:
            data = blob_bytes(self.root, entry.oid)
            return FileState(kind="regular", mode=entry.mode, content_digest=sha256_of_bytes(data))
        if entry.type == "tree":
            raise UnsupportedSpecialFile(
                f"{path} is a directory in {self.tree_commit[:12]}, not a single file"
            )
        raise UnsupportedSpecialFile(f"unsupported git mode {entry.mode} for {path}")


class WorkspaceSourceView:
    """The state of a path in the local working tree - exact on-disk bytes; symlinks not followed."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def state(self, path: str) -> FileState | None:
        full = self.root / path
        try:
            info = os.lstat(full)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise SourceViewError(f"cannot stat {path}: {exc}") from exc
        mode = info.st_mode
        if stat.S_ISLNK(mode):
            target = os.readlink(full)
            return FileState(
                kind="symlink",
                mode=_MODE_SYMLINK,
                content_digest=sha256_of_bytes(_encode_target(target, path)),
            )
        if stat.S_ISREG(mode):
            git_mode = "100755" if (mode & 0o111) else "100644"
            data = full.read_bytes()
            return FileState(kind="regular", mode=git_mode, content_digest=sha256_of_bytes(data))
        if stat.S_ISDIR(mode):
            # A tracked path that is a directory in the working tree is a live submodule/gitlink
            # checkout; the kernel models gitlinks from TREES (both sides), not from a live subrepo.
            raise UnsupportedSpecialFile(
                f"{path} is a directory in the working tree (live submodule not supported in this scope)"
            )
        raise UnsupportedSpecialFile(
            f"{path} is an unsupported special file (st_mode type {stat.S_IFMT(mode):#o})"
        )


def _encode_target(target: str, path: str) -> bytes:
    """Encode a symlink target string to UTF-8 bytes, matching the git blob; fail closed otherwise."""
    try:
        return target.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise UnsupportedPathEncoding(f"symlink target of {path} is not valid UTF-8") from exc
