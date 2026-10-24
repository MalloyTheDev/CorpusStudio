"""Phase-1 fixtures for the CorpusStudio assurance change-set kernel (scripts/assurance).

The kernel lives under ``scripts/`` (stdlib-only, outside the ``corpus_studio`` package), so these
tests exercise it two ways: (1) as the real CLI contract - ``python scripts/cs_assure.py`` run in a
throwaway git repo, asserting exit codes + the sealed JSON record; and (2) via direct import for the
few library-only behaviours (the scope guard, the two-pass stability guard, canonical JSON, and
record verification). Each test builds its own temporary git repository so nothing depends on the
host repo's state.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CS_ASSURE = SCRIPTS_DIR / "cs_assure.py"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from assurance.canonical_json import CanonicalJsonError, canonical_dumps, sha256_of_bytes  # noqa: E402
from assurance.fingerprint import compute_fingerprint  # noqa: E402
from assurance.records import (  # noqa: E402
    ChangeSetUnstable,
    ScopeNotImplemented,
    build_change_set_record,
    seal_record,
    verify_record,
)
from assurance.source_views import (  # noqa: E402
    GitTreeSourceView,
    SourceViewError,
    UnsupportedSpecialFile,
    WorkspaceSourceView,
)


# --------------------------------------------------------------------------- helpers


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=check
    )


def _git_stdin(root: Path, *args: str, stdin: str) -> str:
    """Run git with STDIN written as EXACT bytes, returning stdout.

    Not `_git(..., input=...)`: subprocess's text mode translates a newline to the platform line
    ending on write, so on Windows `update-index --index-info` would read the CR as part of the
    path. The fixtures that feed git stdin care about the exact bytes, so they encode them here.
    """
    proc = subprocess.run(
        ["git", "-C", str(root), *args], input=stdin.encode("utf-8"), capture_output=True, check=True
    )
    return proc.stdout.decode("utf-8")


def init_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "assurance@example.com")
    _git(root, "config", "user.name", "assurance-test")
    # No background git maintenance in a test repo. Auto-gc DETACHES by default, so it can rewrite
    # .git/objects while a test is mid-assertion: test_changeset_never_mutates_committed_state
    # snapshots the object store before and after a read-only changeset and flaked exactly that way
    # in CI, where a git temporary name present in the first snapshot had become a pack-*.rev
    # revindex by the second. Disabling maintenance removes the race instead of teaching the
    # assertion to ignore git's housekeeping artifacts, which keeps it strict about what it covers.
    _git(root, "config", "gc.auto", "0")
    _git(root, "config", "maintenance.auto", "false")
    return root


def commit_all(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)
    return _git(root, "rev-parse", "HEAD").stdout.strip()


def make_base_repo(tmp_path: Path) -> tuple[Path, str]:
    """A repo on ``main`` with a handful of tracked files + a symlink, one 'base' commit.

    Fixture content is written as BYTES deliberately. ``Path.write_text`` translates a newline to
    the platform line ending, so on Windows the on-disk bytes are CRLF while the committed blob
    (normalized by ``core.autocrlf``) stays LF - and the workspace view correctly digests the REAL
    on-disk bytes, so an LF expectation would then mismatch for a reason that has nothing to do
    with the behaviour under test. Bytes make the intended content explicit and identical on both
    platforms.
    """
    root = tmp_path / "repo"
    init_repo(root)
    (root / "mod.txt").write_bytes(b"orig\n")
    (root / "del.txt").write_bytes(b"gone\n")
    (root / "keep.txt").write_bytes(b"hello\n")
    (root / "a.txt").write_bytes(b"aaa\n")
    os.symlink("keep.txt", root / "link")
    base = commit_all(root, "base")
    return root, base


def run_cli(
    start_dir: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CS_ASSURE), *args],
        cwd=str(start_dir),
        capture_output=True,
        text=True,
        env=env,
    )


def changeset(start_dir: Path, base: str = "main") -> tuple[subprocess.CompletedProcess[str], dict]:
    proc = run_cli(start_dir, "changeset", "--scope", "workspace", "--base", base)
    record = json.loads(proc.stdout) if proc.returncode == 0 and proc.stdout.strip() else {}
    return proc, record


def changed(record: dict) -> dict[str, dict]:
    return {cp["path"]: cp for cp in record["payload"]["changed_paths"]}


def sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _stage_exec_bit(root: Path, relpath: str, *, executable: bool) -> None:
    """Record the exec bit in the INDEX - where git reads it when ``core.filemode`` is false.

    ``os.chmod`` alone cannot state an exec-bit change on every supported platform: a Windows
    ``st_mode`` carries no owner-exec bit at all and ``core.filemode`` is false there, so git takes
    the mode from the index and a chmod-only fixture asserts nothing (which is exactly what hid the
    wrong-verdict bug these tests now cover). The fixtures therefore state the intended mode in BOTH
    authorities - chmod for a ``core.filemode=true`` checkout, this for the index - so git itself
    reports the same mode on Linux and on Windows, and the assertions are the same on both.
    """
    _git(root, "update-index", "--chmod=+x" if executable else "--chmod=-x", relpath)


def _diff_raw(root: Path, *args: str) -> str:
    """GIT's own verdict on the change, so a test asserts the record matches git, not one OS."""
    return _git(root, "diff", "--raw", "--no-renames", *args).stdout


# --------------------------------------------------------------------------- change-set shapes


def test_clean_repository_is_empty_change_set(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    proc, record = changeset(root)
    assert proc.returncode == 0, proc.stderr
    assert record["payload"]["changed_path_count"] == 0
    assert record["payload"]["changed_paths"] == []


def test_unstaged_modification(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / "mod.txt").write_bytes(b"changed\n")
    _proc, record = changeset(root)
    entry = changed(record)["mod.txt"]
    assert entry["base"]["kind"] == "regular"
    assert entry["base"]["content_digest"] == sha(b"orig\n")
    assert entry["candidate"]["content_digest"] == sha(b"changed\n")
    assert record["payload"]["changed_path_count"] == 1


def test_staged_modification(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / "mod.txt").write_bytes(b"staged\n")
    _git(root, "add", "mod.txt")
    _proc, record = changeset(root)
    assert changed(record)["mod.txt"]["candidate"]["content_digest"] == sha(b"staged\n")


def test_staged_and_unstaged_divergence_records_net_worktree(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / "mod.txt").write_bytes(b"stagedA\n")
    _git(root, "add", "mod.txt")
    (root / "mod.txt").write_bytes(b"worktreeB\n")
    _proc, record = changeset(root)
    entry = changed(record)["mod.txt"]
    # workspace scope == exact local working-tree bytes: the net (B), not the staged intermediate.
    assert entry["candidate"]["content_digest"] == sha(b"worktreeB\n")
    assert entry["candidate"]["content_digest"] != sha(b"stagedA\n")
    assert entry["base"]["content_digest"] == sha(b"orig\n")


def test_addition_untracked(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / "added.txt").write_bytes(b"new\n")
    _proc, record = changeset(root)
    entry = changed(record)["added.txt"]
    assert entry["base"] is None
    assert entry["candidate"]["kind"] == "regular"
    assert entry["candidate"]["content_digest"] == sha(b"new\n")


def test_deletion(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / "del.txt").unlink()
    _proc, record = changeset(root)
    entry = changed(record)["del.txt"]
    assert entry["base"]["kind"] == "regular"
    assert entry["candidate"] is None


def test_executable_mode_change(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    # State the new mode in both authorities (see _stage_exec_bit): where core.filemode is true the
    # chmod is what git honours, where it is false (the Git-for-Windows default) the index is - and
    # they agree, so git reports 100644 -> 100755 on either platform and so must the record.
    os.chmod(root / "keep.txt", 0o755)
    _stage_exec_bit(root, "keep.txt", executable=True)
    assert ":100644 100755" in _diff_raw(root, "HEAD"), "fixture must be a mode change to git itself"
    _proc, record = changeset(root)
    entry = changed(record)["keep.txt"]
    assert entry["base"]["mode"] == "100644"
    assert entry["candidate"]["mode"] == "100755"
    # content is unchanged - only the mode differs, and that alone counts as a change.
    assert entry["base"]["content_digest"] == entry["candidate"]["content_digest"] == sha(b"hello\n")


def test_exec_bit_uses_owner_bit_not_any_exec_bit(tmp_path: Path) -> None:
    # git canonicalizes the exec bit on the OWNER bit only. A file committed 100755 whose worktree
    # mode keeps group/other exec but drops OWNER exec (0o655) is a real git mode change
    # (100755 -> 100644) and must NOT be silently dropped by treating any exec bit as executable.
    # The index flag states that same drop for a checkout where core.filemode is false, so the one
    # assertion below is the real contract on both platforms instead of a posix-only one.
    root, _ = make_base_repo(tmp_path)
    exe = root / "run.sh"
    exe.write_bytes(b"#!/bin/sh\n")
    os.chmod(exe, 0o755)
    _git(root, "add", "run.sh")
    _stage_exec_bit(root, "run.sh", executable=True)
    commit_all(root, "add executable")  # committed as 100755 on main == HEAD
    assert _git(root, "ls-files", "--stage", "--", "run.sh").stdout.startswith("100755")
    os.chmod(exe, 0o655)  # drop OWNER exec, keep group/other exec
    _stage_exec_bit(root, "run.sh", executable=False)
    # Sanity: git itself reports the de-executable (owner-exec removed).
    assert ":100755 100644" in _diff_raw(root, "HEAD")
    _proc, record = changeset(root)
    entry = changed(record)["run.sh"]
    assert entry["base"]["mode"] == "100755"
    assert entry["candidate"]["mode"] == "100644"  # NOT silently equal to 100755


def test_content_edit_of_an_executable_keeps_its_mode(tmp_path: Path) -> None:
    """A content-only edit of a committed 100755 script must NOT claim to de-executable it.

    Where core.filemode is false (the Git-for-Windows default) os.lstat carries no owner-exec bit at
    all, so a mode derived from it reads 100644 for EVERY worktree file: the change set would state
    that editing a script also dropped its exec bit, and would seal that fabricated fact into
    record_digest + fingerprint. git reports the mode unchanged (:100755 100755). The knob is pinned
    in the fixture so this asserts the same contract on both platforms.
    """
    root = init_repo(tmp_path / "repo")
    _git(root, "config", "core.filemode", "false")
    script = root / "run.sh"
    script.write_bytes(b"#!/bin/sh\necho hi\n")
    _git(root, "add", "run.sh")
    _stage_exec_bit(root, "run.sh", executable=True)
    _git(root, "commit", "-q", "-m", "add executable")
    script.write_bytes(b"#!/bin/sh\necho changed\n")  # content only; nobody touched the mode
    assert ":100755 100755" in _diff_raw(root, "HEAD")  # git: content modified, mode unchanged
    _proc, record = changeset(root)
    entry = changed(record)["run.sh"]
    assert entry["base"]["mode"] == entry["candidate"]["mode"] == "100755"
    assert entry["candidate"]["content_digest"] == sha(b"#!/bin/sh\necho changed\n")


def test_trusted_filemode_keeps_the_index_out_of_the_mode(tmp_path: Path) -> None:
    """Where core.filemode is true, the owner-exec bit of st_mode is the authority and the index is
    not consulted at all - the posix derivation is exactly what it was before the index fallback.

    Pinned true so the expectation is identical on both platforms: neither a posix 0o644 file nor a
    Windows st_mode carries the owner-exec bit, so an index entry of 100755 must NOT move the mode.
    """
    root = init_repo(tmp_path / "repo")
    _git(root, "config", "core.filemode", "true")
    script = root / "run.sh"
    script.write_bytes(b"#!/bin/sh\n")
    os.chmod(script, 0o644)
    _git(root, "add", "run.sh")
    _stage_exec_bit(root, "run.sh", executable=True)  # the index now says 100755 ...
    state = WorkspaceSourceView(root).state("run.sh")
    assert state is not None
    assert state.kind == "regular"
    assert state.mode == "100644"  # ... and st_mode, the authority here, says otherwise


def test_symlink_checked_out_as_a_placeholder_file_is_still_a_symlink(tmp_path: Path) -> None:
    """With core.symlinks false, git materialises a tracked symlink as a REGULAR file whose bytes
    are the target string. Deriving the kind from os.lstat then reports a RETARGET as
    "symlink -> regular" - a false positive for any check gating on added or removed symlinks. git
    reports :120000 120000 (a retarget) and so must the record; the digest is already right, since
    the bytes of the placeholder ARE the target string.
    """
    root = init_repo(tmp_path / "repo")
    _git(root, "config", "core.symlinks", "false")
    (root / "keep.txt").write_bytes(b"hello\n")
    _git(root, "add", "keep.txt")
    target_oid = _git_stdin(root, "hash-object", "-w", "--stdin", stdin="keep.txt").strip()
    _git(root, "update-index", "--add", "--cacheinfo", f"120000,{target_oid},link")
    _git(root, "commit", "-q", "-m", "base")  # commit the staged index; `add -A` is not involved
    _git(root, "checkout", "-q", "-f", "--", "link")
    placeholder = root / "link"
    assert placeholder.is_file() and not placeholder.is_symlink()  # the point of the fixture
    assert placeholder.read_bytes() == b"keep.txt"
    placeholder.write_bytes(b"mod.txt")  # a RETARGET, not a type change
    assert ":120000 120000" in _diff_raw(root, "HEAD")
    _proc, record = changeset(root)
    entry = changed(record)["link"]
    assert entry["base"]["kind"] == entry["candidate"]["kind"] == "symlink"
    assert entry["base"]["mode"] == entry["candidate"]["mode"] == "120000"
    assert entry["base"]["content_digest"] == sha(b"keep.txt")
    assert entry["candidate"]["content_digest"] == sha(b"mod.txt")


def test_unmerged_index_entry_fails_closed(tmp_path: Path) -> None:
    """A conflicted path has index stages 1-3 and no stage 0, so wherever the index is the authority
    for the mode there is no single recorded mode to attribute to it. Those modes feed record_digest
    and fingerprint, so the kernel refuses instead of picking one of the three stages.

    (Where core.filemode and core.symlinks are both true the filesystem IS the authority and no
    index lookup happens, so the refusal is scoped to the checkouts that need the index - which is
    why the fixture pins the knob rather than depending on the default of the host.)
    """
    from assurance.git_state import UnmergedIndexPath

    root = init_repo(tmp_path / "repo")
    _git(root, "config", "core.filemode", "false")
    conflicted = root / "c.txt"
    conflicted.write_bytes(b"base\n")
    commit_all(root, "base")
    _git(root, "checkout", "-q", "-b", "other")
    conflicted.write_bytes(b"theirs\n")
    commit_all(root, "theirs")
    _git(root, "checkout", "-q", "main")
    conflicted.write_bytes(b"ours\n")
    commit_all(root, "ours")
    _git(root, "merge", "other", check=False)  # a real conflict: stages 1-3, no stage 0
    assert "UU c.txt" in _git(root, "status", "--porcelain").stdout
    with pytest.raises(UnmergedIndexPath):
        WorkspaceSourceView(root).state("c.txt")


@pytest.mark.skipif(os.name != "posix", reason="requires posix chmod semantics")
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root bypasses file permissions"
)
def test_workspace_view_fails_closed_on_unreadable_regular_file(tmp_path: Path) -> None:
    # A regular file we can stat but not READ must fail closed as SourceViewError (an AssuranceError
    # the CLI maps to exit 2), never escape as an unguarded OSError -> traceback -> exit 1.
    victim = tmp_path / "secret.txt"
    victim.write_bytes(b"nope\n")
    os.chmod(victim, 0o000)
    try:
        with pytest.raises(SourceViewError):
            WorkspaceSourceView(tmp_path).state("secret.txt")
    finally:
        os.chmod(victim, 0o644)  # restore so tmp cleanup can remove it


def test_git_launch_failure_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A missing/unrunnable `git` binary must fail CLOSED as GitStateError, never escape as an
    # uncaught OSError. exit 1 is also indistinguishable from a doclint --strict staleness signal.
    from assurance import git_state

    def _no_git(*_a: object, **_k: object) -> None:
        raise FileNotFoundError(2, "No such file or directory: 'git'")

    monkeypatch.setattr(git_state.subprocess, "run", _no_git)
    with pytest.raises(git_state.GitStateError):
        git_state._git(tmp_path, "rev-parse", "HEAD")


def test_git_timeout_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A wedged git (stalled FS / credential prompt) must fail CLOSED, never hang the loop unbounded.
    from assurance import git_state

    def _hang(*_a: object, **_k: object) -> None:
        raise git_state.subprocess.TimeoutExpired(cmd="git", timeout=git_state._GIT_TIMEOUT_S)

    monkeypatch.setattr(git_state.subprocess, "run", _hang)
    with pytest.raises(git_state.GitStateError, match="timed out"):
        git_state._git(tmp_path, "rev-parse", "HEAD")


def test_changeset_never_mutates_committed_state(tmp_path: Path) -> None:
    # `changeset` READS repository state: even when a stale stat-cache makes git refresh .git/index
    # (a content-neutral write we honestly do NOT claim to avoid), it must never alter committed
    # state - HEAD, refs, or the object store - and the computed change set must be deterministic.
    root, _ = make_base_repo(tmp_path)
    objects_dir = root / ".git" / "objects"

    def _committed_state() -> tuple[str, list[str]]:
        head = _git(root, "rev-parse", "HEAD").stdout.strip()
        objects = sorted(
            str(p.relative_to(objects_dir)) for p in objects_dir.rglob("*") if p.is_file()
        )
        return head, objects

    # Stale every tracked file's cached stat info so git WANTS to refresh the index on `diff`.
    for name in ("mod.txt", "keep.txt", "a.txt"):
        os.utime(root / name, (1_000_000_000, 1_000_000_000))
    before = _committed_state()
    proc1, rec1 = changeset(root)
    proc2, rec2 = changeset(root)
    assert proc1.returncode == 0 and proc2.returncode == 0, (proc1.stderr, proc2.stderr)
    # Committed state (HEAD + object store) is byte-identical after the reads ...
    assert _committed_state() == before
    # ... and the change set is deterministic despite any stat-cache churn between the two runs.
    assert rec1["payload"]["changed_paths"] == rec2["payload"]["changed_paths"]


def test_symlink_target_change_is_not_followed(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / "link").unlink()
    os.symlink("mod.txt", root / "link")
    _proc, record = changeset(root)
    entry = changed(record)["link"]
    assert entry["base"]["kind"] == entry["candidate"]["kind"] == "symlink"
    # the digest is over the target STRING (not the pointed-to file), proving non-follow.
    assert entry["base"]["content_digest"] == sha(b"keep.txt")
    assert entry["candidate"]["content_digest"] == sha(b"mod.txt")


def test_ignored_file_is_excluded(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / ".gitignore").write_bytes(b"ignore.me\n")
    commit_all(root, "add gitignore")
    (root / "ignore.me").write_bytes(b"secret\n")
    _proc, record = changeset(root)
    assert "ignore.me" not in changed(record)
    assert record["payload"]["changed_path_count"] == 0


def test_rename_is_delete_plus_add(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    _git(root, "mv", "a.txt", "b.txt")
    _proc, record = changeset(root)
    paths = changed(record)
    assert "a.txt" in paths and "b.txt" in paths
    assert paths["a.txt"]["candidate"] is None  # old path deleted
    assert paths["b.txt"]["base"] is None  # new path added
    # the canonical model has no 'rename' concept anywhere in the record.
    assert "rename" not in json.dumps(record)


def test_committed_branch_change_is_captured(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    _git(root, "checkout", "-q", "-b", "feature")
    (root / "mod.txt").write_bytes(b"feat\n")
    commit_all(root, "feature change")
    # worktree is clean, but the change set vs merge-base(feature, main) still shows it.
    proc, record = changeset(root, base="main")
    assert proc.returncode == 0, proc.stderr
    assert changed(record)["mod.txt"]["candidate"]["content_digest"] == sha(b"feat\n")


def test_base_side_gitlink_is_modeled(tmp_path: Path) -> None:
    # Gitlinks are modeled from TREES. Build one via cacheinfo (no live submodule checkout needed).
    root, base = make_base_repo(tmp_path)
    _git(root, "update-index", "--add", "--cacheinfo", f"160000,{base},subm")
    # commit the staged index directly - `git add -A` would stage a deletion of the
    # working-tree-less gitlink and drop it before the commit.
    _git(root, "commit", "-q", "-m", "add gitlink")
    gitlink_commit = _git(root, "rev-parse", "HEAD").stdout.strip()
    state = GitTreeSourceView(root, gitlink_commit).state("subm")
    assert state is not None
    assert state.kind == "gitlink"
    assert state.mode == "160000"
    assert state.commit_oid == base
    assert state.content_digest is None


# --------------------------------------------------------------------------- fingerprint semantics


def test_fingerprint_is_stable_for_same_content(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / "mod.txt").write_bytes(b"changed\n")
    _p1, r1 = changeset(root)
    _p2, r2 = changeset(root)
    assert r1["payload"]["fingerprint"] == r2["payload"]["fingerprint"]


def test_fingerprint_invalidates_on_change(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / "mod.txt").write_bytes(b"first\n")
    _p1, r1 = changeset(root)
    (root / "mod.txt").write_bytes(b"second\n")
    _p2, r2 = changeset(root)
    assert r1["payload"]["fingerprint"] != r2["payload"]["fingerprint"]


def test_deterministic_json_output(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / "mod.txt").write_bytes(b"changed\n")
    p1 = run_cli(root, "changeset", "--scope", "workspace", "--base", "main")
    p2 = run_cli(root, "changeset", "--scope", "workspace", "--base", "main")
    assert p1.returncode == 0 and p2.returncode == 0
    assert p1.stdout == p2.stdout  # byte-identical, including the record_digest


def test_record_verifies_and_fingerprint_recomputes(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / "mod.txt").write_bytes(b"changed\n")
    _proc, record = changeset(root)
    assert verify_record(record) is True
    payload = record["payload"]
    assert compute_fingerprint(payload["scope"], payload["base_oid"], payload["changed_paths"]) == (
        payload["fingerprint"]
    )
    # tampering with any digested field breaks integrity.
    tampered = json.loads(json.dumps(record))
    tampered["payload"]["changed_path_count"] = 999
    assert verify_record(tampered) is False


# --------------------------------------------------------------------------- fail-closed refusals


def test_missing_base_ref_fails_closed(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    proc, _ = changeset(root, base="no-such-ref")
    assert proc.returncode == 2
    assert "MissingBaseRef" in proc.stderr


def test_no_merge_base_fails_closed(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "repo")
    (root / "x.txt").write_bytes(b"1\n")
    commit_all(root, "c1")
    _git(root, "checkout", "-q", "--orphan", "island")
    _git(root, "rm", "-rfq", ".", check=False)
    (root / "y.txt").write_bytes(b"island\n")
    commit_all(root, "orphan root")
    proc, _ = changeset(root, base="main")  # HEAD (island) is unrelated to main
    assert proc.returncode == 2
    assert "merge base" in proc.stderr.lower()


def test_not_a_git_repo_fails_closed(tmp_path: Path) -> None:
    not_repo = tmp_path / "plain"
    not_repo.mkdir()
    # Hermetic even when pytest's basetemp lives inside this repo: a ceiling stops git's upward
    # discovery at tmp_path, so the outer CorpusStudio repo is never found.
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    proc = run_cli(not_repo, "changeset", "--scope", "workspace", "--base", "main", env=env)
    assert proc.returncode == 2
    assert "NotAGitRepo" in proc.stderr


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires os.mkfifo")
def test_workspace_view_refuses_special_file(tmp_path: Path) -> None:
    # git does not enumerate a fifo as an untracked file, so the guard is exercised at the
    # source-view layer directly: a non-regular/non-symlink path fails closed rather than be hashed.
    root = init_repo(tmp_path / "repo")
    os.mkfifo(root / "pipe")  # type: ignore[attr-defined]
    with pytest.raises(UnsupportedSpecialFile):
        WorkspaceSourceView(root).state("pipe")


@pytest.mark.skipif(os.name != "posix", reason="requires posix byte-encoded paths")
def test_non_utf8_path_fails_closed(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    bad = os.fsencode(str(root)) + b"/bad\xff.txt"
    with open(bad, "wb") as handle:
        handle.write(b"x")
    proc, _ = changeset(root)
    assert proc.returncode == 2
    assert "UTF-8" in proc.stderr


def test_scope_not_implemented_via_library(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    with pytest.raises(ScopeNotImplemented):
        build_change_set_record(start_dir=root, scope="index")


def test_scope_rejected_by_cli(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    proc = run_cli(root, "changeset", "--scope", "index", "--base", "main")
    assert proc.returncode == 2  # argparse choices refuse it


def test_worktree_mutation_during_collection_is_unstable(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / "mod.txt").write_bytes(b"changed\n")

    def mutate() -> None:  # runs between the two snapshot passes
        (root / "surprise.txt").write_bytes(b"appeared\n")

    with pytest.raises(ChangeSetUnstable):
        build_change_set_record(start_dir=root, _between_passes=mutate)


# --------------------------------------------------------------------------- worktrees & shallow


def test_linked_worktree(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    wt = tmp_path / "wt"
    _git(root, "worktree", "add", "-q", "-b", "wt", str(wt))
    (wt / "mod.txt").write_bytes(b"from-worktree\n")
    proc, record = changeset(wt, base="main")
    assert proc.returncode == 0, proc.stderr
    assert changed(record)["mod.txt"]["candidate"]["content_digest"] == sha(b"from-worktree\n")


def test_shallow_clone_records_shallow_and_limits_merge_base(tmp_path: Path) -> None:
    source = init_repo(tmp_path / "src")
    (source / "x.txt").write_bytes(b"1\n")
    commit_all(source, "c1")
    _git(source, "branch", "base_old")  # base_old := c1
    (source / "x.txt").write_bytes(b"2\n")
    commit_all(source, "c2")
    (source / "x.txt").write_bytes(b"3\n")
    commit_all(source, "c3")

    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "--depth", "1", f"file://{source}", str(shallow)],
        capture_output=True,
        text=True,
        check=True,
    )
    # `git clone --depth 1` implies --single-branch, so base_old was not fetched; fetch it shallowly.
    _git(shallow, "fetch", "--depth", "1", "origin", "base_old:base_old")

    # (a) the kernel works in a shallow clone and records shallowness.
    ok, record = changeset(shallow, base="main")
    assert ok.returncode == 0, ok.stderr
    assert record["provenance"]["is_shallow"] is True

    # (b) a merge base that is severed by the shallow boundary fails closed as a shallow limitation.
    proc, _ = changeset(shallow, base="base_old")
    assert proc.returncode == 2
    assert "shallow" in proc.stderr.lower()


# --------------------------------------------------------------------------- canonical json + seal


def test_canonical_json_is_sorted_and_compact() -> None:
    assert canonical_dumps({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_canonical_json_preserves_unicode() -> None:
    assert canonical_dumps({"p": "café"}) == '{"p":"café"}'  # not \\u-escaped


def test_canonical_json_rejects_floats() -> None:
    with pytest.raises(CanonicalJsonError):
        canonical_dumps({"x": 1.5})


def test_digest_has_algorithm_prefix() -> None:
    empty = sha256_of_bytes(b"")
    assert empty.startswith("sha256:")
    assert empty == "sha256:" + hashlib.sha256(b"").hexdigest()


def test_seal_and_verify_roundtrip() -> None:
    record = seal_record("change_set", 1, {"a": 1}, {"b": 2})
    assert record["record_digest"].startswith("sha256:")
    assert verify_record(record) is True
    record["payload"]["a"] = 99
    assert verify_record(record) is False


def test_verify_rejects_record_without_digest() -> None:
    assert verify_record({"record_type": "change_set", "payload": {}}) is False


def test_read_committed_file_distinguishes_absent_from_error(tmp_path: Path) -> None:
    from assurance.git_state import GitStateError, discover_git_context, read_committed_file
    repo = init_repo(tmp_path / "repo")
    (repo / "policy.txt").write_bytes(b"v1\n")
    commit_all(repo, "seed")
    ctx = discover_git_context(repo)
    assert read_committed_file(ctx, "HEAD", "policy.txt") == b"v1\n"      # a committed file reads back
    assert read_committed_file(ctx, "HEAD", "nope.txt") is None          # an ABSENT path -> None, not an error
    with pytest.raises(GitStateError):                                   # an INVALID ref fails CLOSED, not 'absent'
        read_committed_file(ctx, "notacommitref", "policy.txt")
