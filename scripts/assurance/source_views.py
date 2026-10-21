"""Source views and file-state modeling for the assurance change-set kernel (Phase 1).

A *source view* answers one question: "what is the state of repository-relative path ``P`` in
this view?" Two views exist in Phase 1:

  * :class:`WorkspaceSourceView` - the exact local WORKING-TREE bytes (symlinks are NOT followed;
    a symlink is recorded by the digest of its target string, a gitlink/submodule directory and
    any non-regular special file fail closed). The bytes come from the filesystem; the git-visible
    mode and kind come from the filesystem only where git honours it (``core.filemode`` /
    ``core.symlinks``) and from the INDEX otherwise, so the record states what git states rather
    than what the local operating system happens to be able to store.
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
    config_bool,
    index_mode,
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
    """The state of a path as recorded in a committed git TREE-ISH (read via the object store).

    ``tree_ish`` may be a commit OID (the ``head`` / base-tip scopes) or a bare tree OID (the
    ``merge_candidate`` scope's synthesized merge tree); ``git ls-tree`` / ``cat-file`` accept both,
    so the same view serves every tree-vs-tree scope with no checkout.
    """

    def __init__(self, root: Path, tree_ish: str) -> None:
        self.root = root
        self.tree_ish = tree_ish

    def state(self, path: str) -> FileState | None:
        entry = tree_entry(self.root, self.tree_ish, path)
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
                f"{path} is a directory in {self.tree_ish[:12]}, not a single file"
            )
        raise UnsupportedSpecialFile(f"unsupported git mode {entry.mode} for {path}")


class WorkspaceSourceView:
    """The state of a path in the local working tree - exact on-disk bytes; symlinks not followed.

    The BYTES are always the local file's own bytes. The git-visible kind and mode are not always
    derivable from them: git honours ``st_mode``'s owner-exec bit only where ``core.filemode`` is
    true, and materialises a tracked symlink as a placeholder REGULAR file (whose bytes are the
    target string) wherever ``core.symlinks`` is false. On such a checkout - the Git-for-Windows
    default, and any FAT/exFAT or no-symlink-permission checkout on Linux - ``os.lstat`` cannot see
    what git sees, so the INDEX supplies the authoritative mode instead.

    Deriving them from ``st_mode`` alone there states facts that are FALSE and seals them into
    ``record_digest`` / ``fingerprint``: every worktree file reads 100644, so a content-only edit of
    a committed 100755 script claims to de-executable it, and a symlink RETARGET reads as
    ``symlink -> regular``, which is a false positive for any check gating on symlink churn.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._config: dict[str, bool] = {}

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
            try:
                target = os.readlink(full)
            except OSError as exc:
                raise SourceViewError(f"cannot read {path}: {exc}") from exc
            return FileState(
                kind="symlink",
                mode=_MODE_SYMLINK,
                content_digest=sha256_of_bytes(_encode_target(target, path)),
            )
        if stat.S_ISREG(mode):
            try:
                data = full.read_bytes()
            except OSError as exc:
                # A file we can stat but not read (permissions, races) fails CLOSED, not with an
                # uncaught traceback - the CLI maps AssuranceError to exit 2. The bytes are read
                # BEFORE any git lookup so that refusal never depends on git.
                raise SourceViewError(f"cannot read {path}: {exc}") from exc
            return self._regular_file_state(path, mode, data)
        if stat.S_ISDIR(mode):
            # A tracked path that is a directory in the working tree is a live submodule/gitlink
            # checkout; the kernel models gitlinks from TREES (both sides), not from a live subrepo.
            raise UnsupportedSpecialFile(
                f"{path} is a directory in the working tree (live submodule not supported in this scope)"
            )
        raise UnsupportedSpecialFile(
            f"{path} is an unsupported special file (st_mode type {stat.S_IFMT(mode):#o})"
        )

    def _regular_file_state(self, path: str, st_mode: int, data: bytes) -> FileState:
        """Model a regular working-tree file the way GIT's own worktree diff models it.

        git substitutes the index entry's mode wherever the filesystem cannot carry its view (its
        ``ce_mode_from_stat``): with ``core.filemode`` false the RECORDED mode wins over ``st_mode``,
        and with ``core.symlinks`` false an index entry of 120000 means this regular file is a
        symlink PLACEHOLDER. Mirroring that is what makes the record state what git states on the
        same checkout, instead of what one operating system happens to be able to store.
        """
        trust_filemode = self._config_bool("core.filemode")
        has_symlinks = self._config_bool("core.symlinks")
        # The index is consulted ONLY where the filesystem is not the authority, so an ordinary POSIX
        # checkout (both knobs true) keeps the pure st_mode derivation and spawns no extra git process.
        staged_mode = None if (trust_filemode and has_symlinks) else index_mode(self.root, path)
        if staged_mode == _MODE_SYMLINK and not has_symlinks:
            # A tracked symlink checked out as a placeholder file: its bytes ARE the target string,
            # so the digest already is the symlink digest; only kind + mode would have been wrong.
            return FileState(
                kind="symlink", mode=_MODE_SYMLINK, content_digest=sha256_of_bytes(data)
            )
        if trust_filemode:
            # git canonicalizes the executable bit on the OWNER bit only (mode & 0o100), so a
            # group/other-exec-only change (e.g. 0o755 -> 0o655) IS a git mode change and must not
            # be masked by the other exec bits.
            git_mode = "100755" if (st_mode & 0o100) else "100644"
        elif staged_mode is None:
            # Untracked, and the exec bit is not trusted: git's own create_ce_mode(0666) is 100644.
            git_mode = "100644"
        elif staged_mode in _MODE_REGULAR:
            git_mode = staged_mode
        elif staged_mode in (_MODE_SYMLINK, _MODE_GITLINK):
            # git substitutes the index mode only for a REGULAR index entry; where the entry is a
            # symlink (with symlinks supported) or a gitlink and the working tree holds a regular
            # file, git itself falls back to 100644 - an honest type change, not a refusal.
            git_mode = "100644"
        else:
            # The authoritative mode could not be determined: refuse rather than seal a guess that
            # record_digest and fingerprint would make permanent.
            raise SourceViewError(
                f"cannot determine the git mode of {path}: core.filemode is false and the index "
                f"records the unrecognized mode {staged_mode!r} for a regular working-tree file"
            )
        return FileState(kind="regular", mode=git_mode, content_digest=sha256_of_bytes(data))

    def _config_bool(self, name: str) -> bool:
        """Read one boolean git-config knob once per view (a view is rebuilt per snapshot pass).

        The default matches git's own compiled-in default for an absent key, so an unset knob is
        never interpreted differently here than git interprets it.
        """
        if name not in self._config:
            self._config[name] = config_bool(self.root, name, default=True)
        return self._config[name]


def _encode_target(target: str, path: str) -> bytes:
    """Encode a symlink target string to UTF-8 bytes, matching the git blob; fail closed otherwise."""
    try:
        return target.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise UnsupportedPathEncoding(f"symlink target of {path} is not valid UTF-8") from exc
