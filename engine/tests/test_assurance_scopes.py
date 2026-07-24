"""Head / merge-candidate assurance scopes (re-review #15).

The Phase-1 kernel modeled only the ``workspace`` scope (base = merge-base tree, candidate = the live
working tree). A production integration gate must instead bind its impact assessment to the EXACT
commit it validated, not the local dirty tree - so the change-set kernel now also models:

  * ``head``            - the committed HEAD tree vs merge-base(HEAD, --base): the branch's own diff,
                          independent of uncommitted working-tree edits;
  * ``merge_candidate`` - the tree a 3-way merge of HEAD into the --base TIP would produce vs that tip:
                          what merging would ADD to the base as it stands now (base-movement aware),
                          failing closed on a conflicted merge.

These exercise the CLI contract (throwaway git repos) plus the library guards (unborn HEAD, conflict,
verify's workspace-only restriction). Each test builds its own temporary repo.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CS_ASSURE = SCRIPTS_DIR / "cs_assure.py"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from assurance import git_state  # noqa: E402
from assurance.git_state import (  # noqa: E402
    GitStateError,
    MergeCandidateConflicted,
    MergeTreeUnsupported,
    write_merge_tree,
)
from assurance.records import (  # noqa: E402
    ScopeUnavailable,
    build_change_set_record,
    verify_record,
)
from assurance.verification import GateError, build_verification_record  # noqa: E402


# --------------------------------------------------------------------------- helpers


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=check)


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


def run_cli(start_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CS_ASSURE), *args], cwd=str(start_dir), capture_output=True, text=True
    )


def cli_changeset(start_dir: Path, scope: str, base: str = "main") -> tuple[subprocess.CompletedProcess[str], dict]:
    proc = run_cli(start_dir, "changeset", "--scope", scope, "--base", base)
    record = json.loads(proc.stdout) if proc.returncode == 0 and proc.stdout.strip() else {}
    return proc, record


def changed(record: dict) -> dict[str, dict]:
    return {cp["path"]: cp for cp in record["payload"]["changed_paths"]}


def sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def base_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    init_repo(root)
    (root / "a.txt").write_text("aaa\n")
    (root / "b.txt").write_text("bbb\n")
    commit_all(root, "base")
    return root


# --------------------------------------------------------------------------- head scope


def test_head_scope_reflects_the_committed_head_not_the_working_tree(tmp_path: Path) -> None:
    root = base_repo(tmp_path)
    _git(root, "checkout", "-q", "-b", "feat")
    (root / "a.txt").write_text("committed\n")
    commit_all(root, "change a on the branch")
    (root / "b.txt").write_text("uncommitted\n")  # a dirty working-tree edit NOT in HEAD

    _proc, head = cli_changeset(root, "head")
    head_paths = changed(head)
    assert set(head_paths) == {"a.txt"}  # only the COMMITTED change; the dirty b.txt is invisible to head
    assert head_paths["a.txt"]["candidate"]["content_digest"] == sha(b"committed\n")

    _proc, ws = cli_changeset(root, "workspace")
    assert set(changed(ws)) == {"a.txt", "b.txt"}  # workspace sees the dirty edit too


def test_head_scope_pins_the_candidate_oid_to_head(tmp_path: Path) -> None:
    root = base_repo(tmp_path)
    _git(root, "checkout", "-q", "-b", "feat")
    (root / "a.txt").write_text("x\n")
    head_oid = commit_all(root, "c")
    proc, rec = cli_changeset(root, "head")
    assert proc.returncode == 0, proc.stderr
    assert rec["payload"]["candidate_view"] == "head" and rec["payload"]["base_view"] == "merge_base"
    assert rec["payload"]["candidate_oid"] == head_oid
    assert rec["provenance"]["candidate_oid"] == head_oid
    assert verify_record(rec)


def test_workspace_records_never_carry_a_candidate_oid(tmp_path: Path) -> None:
    # Back-compat guarantee: only the tree scopes pin candidate_oid; the workspace record must stay
    # byte-shape-identical to the pre-#15 kernel so existing consumers never see schema drift.
    root = base_repo(tmp_path)
    (root / "a.txt").write_text("dirty\n")
    _proc, rec = cli_changeset(root, "workspace")
    assert rec["payload"]["changed_path_count"] == 1
    assert "candidate_oid" not in rec["payload"]
    assert "candidate_oid" not in rec["provenance"]


def test_head_scope_is_stable_across_runs(tmp_path: Path) -> None:
    root = base_repo(tmp_path)
    _git(root, "checkout", "-q", "-b", "feat")
    (root / "a.txt").write_text("x\n")
    commit_all(root, "c")
    _p1, r1 = cli_changeset(root, "head")
    _p2, r2 = cli_changeset(root, "head")
    assert r1["payload"]["fingerprint"] == r2["payload"]["fingerprint"]


# --------------------------------------------------------------------------- merge_candidate scope


def test_merge_candidate_is_relative_to_the_current_base_tip(tmp_path: Path) -> None:
    root = base_repo(tmp_path)
    _git(root, "checkout", "-q", "-b", "feat")
    (root / "a.txt").write_text("feat-change\n")
    commit_all(root, "feat changes a")
    _git(root, "checkout", "-q", "main")
    (root / "b.txt").write_text("base-moved\n")  # base advances on a DIFFERENT file
    main_tip = commit_all(root, "base moves b")
    _git(root, "checkout", "-q", "feat")

    proc, rec = cli_changeset(root, "merge_candidate")
    assert proc.returncode == 0, proc.stderr
    # merging feat into the moved main introduces feat's a.txt change on top of the current tip;
    # b.txt is already in the base, so it is NOT part of what the merge adds.
    assert set(changed(rec)) == {"a.txt"}
    assert rec["payload"]["base_view"] == "base_tip" and rec["payload"]["base_oid"] == main_tip
    assert rec["payload"]["candidate_view"] == "merge_candidate"
    assert verify_record(rec)


def test_merge_candidate_fails_closed_on_a_conflict_cli(tmp_path: Path) -> None:
    root = base_repo(tmp_path)
    _git(root, "checkout", "-q", "-b", "feat")
    (root / "a.txt").write_text("feat-line\n")
    commit_all(root, "feat edits a")
    _git(root, "checkout", "-q", "main")
    (root / "a.txt").write_text("main-line\n")  # same file, conflicting edit
    commit_all(root, "main edits a")
    _git(root, "checkout", "-q", "feat")

    proc, _ = cli_changeset(root, "merge_candidate")
    assert proc.returncode == 2  # fail-closed refusal, no record on stdout
    assert "conflict" in proc.stderr.lower()


def test_merge_candidate_conflict_via_library(tmp_path: Path) -> None:
    root = base_repo(tmp_path)
    _git(root, "checkout", "-q", "-b", "feat")
    (root / "a.txt").write_text("feat\n")
    commit_all(root, "feat")
    _git(root, "checkout", "-q", "main")
    (root / "a.txt").write_text("main\n")
    commit_all(root, "main")
    _git(root, "checkout", "-q", "feat")
    with pytest.raises(MergeCandidateConflicted):
        build_change_set_record(start_dir=root, scope="merge_candidate", base_ref="main")


def _fake_git_returning(returncode: int, stdout: bytes = b"", stderr: bytes = b""):
    def fake(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(list(args), returncode, stdout=stdout, stderr=stderr)
    return fake


def test_write_merge_tree_reports_an_old_git_as_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    # A git that lacks --write-tree names the flag / prints "unknown option" -> MergeTreeUnsupported.
    monkeypatch.setattr(git_state, "_git",
                        _fake_git_returning(129, stderr=b"error: unknown option `write-tree'\nusage: git ..."))
    with pytest.raises(MergeTreeUnsupported):
        write_merge_tree(Path("/nonexistent"), "base", "head")


def test_write_merge_tree_does_not_mislabel_a_generic_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # A real invocation error (bad object) must NOT be reported as version incompatibility - it fails
    # closed as a plain GitStateError, honestly labeled (the tightened classification, Sourcery #692).
    monkeypatch.setattr(git_state, "_git",
                        _fake_git_returning(128, stderr=b"fatal: not a valid object name 'base'"))
    with pytest.raises(GitStateError) as excinfo:
        write_merge_tree(Path("/nonexistent"), "base", "head")
    assert not isinstance(excinfo.value, (MergeTreeUnsupported, MergeCandidateConflicted))


# --------------------------------------------------------------------------- guards


def test_tree_scopes_are_unavailable_on_an_unborn_head(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "empty")  # no commits at all -> HEAD is unborn
    for scope in ("head", "merge_candidate"):
        with pytest.raises(ScopeUnavailable):
            build_change_set_record(start_dir=root, scope=scope, base_ref="HEAD")


def test_verify_refuses_a_non_workspace_scope(tmp_path: Path) -> None:
    # verify runs the gate against the working tree; a head/merge_candidate record would be mislabeled.
    root = base_repo(tmp_path)
    with pytest.raises(GateError, match="workspace"):
        build_verification_record(start_dir=root, scope="head", base_ref="main")


def test_cli_accepts_head_and_merge_candidate_but_still_rejects_index(tmp_path: Path) -> None:
    root = base_repo(tmp_path)
    assert run_cli(root, "changeset", "--scope", "head").returncode == 0
    assert run_cli(root, "changeset", "--scope", "merge_candidate").returncode == 0
    # 'index' is still an unimplemented scope: argparse refuses it (exit 2, an "invalid choice").
    index_cs = run_cli(root, "changeset", "--scope", "index")
    assert index_cs.returncode == 2 and "invalid choice" in index_cs.stderr
    # impact/status accept the new scopes at the parse layer: a 'head' scope is NOT an argparse refusal
    # (it may still exit 2 later on a missing policy fixture, but never with "invalid choice").
    for cmd in ("impact", "status"):
        assert "invalid choice" not in run_cli(root, cmd, "--scope", "head").stderr, cmd
        assert "invalid choice" in run_cli(root, cmd, "--scope", "index").stderr, cmd
    # verify stays workspace-only at the CLI layer (choices refuse head), plus the library guard.
    verify_head = run_cli(root, "verify", "--scope", "head")
    assert verify_head.returncode == 2 and "invalid choice" in verify_head.stderr
