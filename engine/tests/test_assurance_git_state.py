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


def init_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "assurance@example.com")
    _git(root, "config", "user.name", "assurance-test")
    return root


def commit_all(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)
    return _git(root, "rev-parse", "HEAD").stdout.strip()


def make_base_repo(tmp_path: Path) -> tuple[Path, str]:
    """A repo on ``main`` with a handful of tracked files + a symlink, one 'base' commit."""
    root = tmp_path / "repo"
    init_repo(root)
    (root / "mod.txt").write_text("orig\n")
    (root / "del.txt").write_text("gone\n")
    (root / "keep.txt").write_text("hello\n")
    (root / "a.txt").write_text("aaa\n")
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


# --------------------------------------------------------------------------- change-set shapes


def test_clean_repository_is_empty_change_set(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    proc, record = changeset(root)
    assert proc.returncode == 0, proc.stderr
    assert record["payload"]["changed_path_count"] == 0
    assert record["payload"]["changed_paths"] == []


def test_unstaged_modification(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / "mod.txt").write_text("changed\n")
    _proc, record = changeset(root)
    entry = changed(record)["mod.txt"]
    assert entry["base"]["kind"] == "regular"
    assert entry["base"]["content_digest"] == sha(b"orig\n")
    assert entry["candidate"]["content_digest"] == sha(b"changed\n")
    assert record["payload"]["changed_path_count"] == 1


def test_staged_modification(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / "mod.txt").write_text("staged\n")
    _git(root, "add", "mod.txt")
    _proc, record = changeset(root)
    assert changed(record)["mod.txt"]["candidate"]["content_digest"] == sha(b"staged\n")


def test_staged_and_unstaged_divergence_records_net_worktree(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / "mod.txt").write_text("stagedA\n")
    _git(root, "add", "mod.txt")
    (root / "mod.txt").write_text("worktreeB\n")
    _proc, record = changeset(root)
    entry = changed(record)["mod.txt"]
    # workspace scope == exact local working-tree bytes: the net (B), not the staged intermediate.
    assert entry["candidate"]["content_digest"] == sha(b"worktreeB\n")
    assert entry["candidate"]["content_digest"] != sha(b"stagedA\n")
    assert entry["base"]["content_digest"] == sha(b"orig\n")


def test_addition_untracked(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / "added.txt").write_text("new\n")
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
    os.chmod(root / "keep.txt", 0o755)
    _proc, record = changeset(root)
    entry = changed(record)["keep.txt"]
    assert entry["base"]["mode"] == "100644"
    assert entry["candidate"]["mode"] == "100755"
    # content is unchanged - only the mode differs, and that alone counts as a change.
    assert entry["base"]["content_digest"] == entry["candidate"]["content_digest"] == sha(b"hello\n")


def test_exec_bit_uses_owner_bit_not_any_exec_bit(tmp_path: Path) -> None:
    # git canonicalizes the exec bit on the OWNER bit only. A file committed 100755 then chmod'd to
    # a mode with group/other exec but NO owner exec (0o655) is a real git mode change
    # (100755 -> 100644) and must NOT be silently dropped by treating any exec bit as executable.
    root, _ = make_base_repo(tmp_path)
    exe = root / "run.sh"
    exe.write_text("#!/bin/sh\n")
    os.chmod(exe, 0o755)
    commit_all(root, "add executable")  # committed as 100755 on main == HEAD
    os.chmod(exe, 0o655)  # drop OWNER exec, keep group/other exec
    # Sanity: git itself reports this as a modification (owner-exec removed).
    assert "run.sh" in _git(root, "status", "--porcelain").stdout
    _proc, record = changeset(root)
    entry = changed(record)["run.sh"]
    assert entry["base"]["mode"] == "100755"
    assert entry["candidate"]["mode"] == "100644"  # NOT silently equal to 100755


@pytest.mark.skipif(os.name != "posix", reason="requires posix chmod semantics")
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root bypasses file permissions"
)
def test_workspace_view_fails_closed_on_unreadable_regular_file(tmp_path: Path) -> None:
    # A regular file we can stat but not READ must fail closed as SourceViewError (an AssuranceError
    # the CLI maps to exit 2), never escape as an unguarded OSError -> traceback -> exit 1.
    victim = tmp_path / "secret.txt"
    victim.write_text("nope\n")
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
    (root / ".gitignore").write_text("ignore.me\n")
    commit_all(root, "add gitignore")
    (root / "ignore.me").write_text("secret\n")
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
    (root / "mod.txt").write_text("feat\n")
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
    (root / "mod.txt").write_text("changed\n")
    _p1, r1 = changeset(root)
    _p2, r2 = changeset(root)
    assert r1["payload"]["fingerprint"] == r2["payload"]["fingerprint"]


def test_fingerprint_invalidates_on_change(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / "mod.txt").write_text("first\n")
    _p1, r1 = changeset(root)
    (root / "mod.txt").write_text("second\n")
    _p2, r2 = changeset(root)
    assert r1["payload"]["fingerprint"] != r2["payload"]["fingerprint"]


def test_deterministic_json_output(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / "mod.txt").write_text("changed\n")
    p1 = run_cli(root, "changeset", "--scope", "workspace", "--base", "main")
    p2 = run_cli(root, "changeset", "--scope", "workspace", "--base", "main")
    assert p1.returncode == 0 and p2.returncode == 0
    assert p1.stdout == p2.stdout  # byte-identical, including the record_digest


def test_record_verifies_and_fingerprint_recomputes(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    (root / "mod.txt").write_text("changed\n")
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
    (root / "x.txt").write_text("1\n")
    commit_all(root, "c1")
    _git(root, "checkout", "-q", "--orphan", "island")
    _git(root, "rm", "-rfq", ".", check=False)
    (root / "y.txt").write_text("island\n")
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
    (root / "mod.txt").write_text("changed\n")

    def mutate() -> None:  # runs between the two snapshot passes
        (root / "surprise.txt").write_text("appeared\n")

    with pytest.raises(ChangeSetUnstable):
        build_change_set_record(start_dir=root, _between_passes=mutate)


# --------------------------------------------------------------------------- worktrees & shallow


def test_linked_worktree(tmp_path: Path) -> None:
    root, _ = make_base_repo(tmp_path)
    wt = tmp_path / "wt"
    _git(root, "worktree", "add", "-q", "-b", "wt", str(wt))
    (wt / "mod.txt").write_text("from-worktree\n")
    proc, record = changeset(wt, base="main")
    assert proc.returncode == 0, proc.stderr
    assert changed(record)["mod.txt"]["candidate"]["content_digest"] == sha(b"from-worktree\n")


def test_shallow_clone_records_shallow_and_limits_merge_base(tmp_path: Path) -> None:
    source = init_repo(tmp_path / "src")
    (source / "x.txt").write_text("1\n")
    commit_all(source, "c1")
    _git(source, "branch", "base_old")  # base_old := c1
    (source / "x.txt").write_text("2\n")
    commit_all(source, "c2")
    (source / "x.txt").write_text("3\n")
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
