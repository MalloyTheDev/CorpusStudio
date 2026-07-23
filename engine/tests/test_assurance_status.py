"""Tests for the project-progression intake (Phase-4: scripts/assurance/{status,github_issues}.py).

Proves the FAIL-SOFT gh sensor (every failure -> available:false + a stable reason, snapshot still
seals at exit 0), the area-tag parser, the deterministic composition of the sensors, the honest
deterministic/measurement split (no top-level applicability_key), bounding-with-counted-drops, the
cross-consistency of the embedded impact key with `cs_assure impact`, and that git/kernel errors stay
fail-CLOSED (exit 2). All gh is faked offline - never a real network call.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CS_ASSURE = SCRIPTS_DIR / "cs_assure.py"
REAL_POLICY_TEXT = (SCRIPTS_DIR / "assurance" / "policy" / "obligations.json").read_text("utf-8")
REAL_REGISTRY_TEXT = (SCRIPTS_DIR / "assurance" / "policy" / "context_sources.json").read_text("utf-8")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from assurance import github_issues, status  # noqa: E402
from assurance.canonical_json import canonical_dumps  # noqa: E402
from assurance.records import verify_record  # noqa: E402


# --------------------------------------------------------------------------- helpers


class _Completed:
    def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _issue(number: int, title: str, updated: str = "2026-01-01T00:00:00Z") -> dict:
    return {"number": number, "title": title, "labels": [], "updatedAt": updated, "state": "OPEN"}


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts" / "assurance" / "policy").mkdir(parents=True)
    for a in (("init", "-q", "-b", "main"), ("config", "user.email", "a@b.c"),
              ("config", "user.name", "t")):
        subprocess.run(["git", "-C", str(repo), *a], check=True)
    (repo / "scripts" / "assurance" / "policy" / "obligations.json").write_text(REAL_POLICY_TEXT)
    (repo / "scripts" / "assurance" / "policy" / "context_sources.json").write_text(REAL_REGISTRY_TEXT)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "seed"], check=True)
    return repo


def _stub_gh(bindir: Path, issues_json: str = "[]", exit_code: int = 0, stderr: str = "") -> Path:
    bindir.mkdir(parents=True, exist_ok=True)
    gh = bindir / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\nimport sys\n"
        f"sys.stderr.write({stderr!r})\nsys.stdout.write({issues_json!r})\nsys.exit({exit_code})\n"
    )
    gh.chmod(0o755)
    return bindir


def run_cli(start_dir: Path, *args: str, gh_bin: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if gh_bin is not None:
        env["PATH"] = f"{gh_bin}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        [sys.executable, str(CS_ASSURE), *args], cwd=str(start_dir), capture_output=True, text=True, env=env
    )


# --------------------------------------------------------------------------- fail-soft gh taxonomy


def _raise(exc: BaseException):
    def _run(*_a: object, **_k: object):
        raise exc
    return _run


@pytest.mark.parametrize(
    "side_effect, reason",
    [
        (_raise(FileNotFoundError()), "gh-missing"),
        (_raise(subprocess.TimeoutExpired(cmd="gh", timeout=15)), "timeout"),
        (_raise(OSError("boom")), "gh-error"),
        (lambda *a, **k: _Completed(1, b"", b"HTTP 401: Bad credentials"), "unauthenticated"),
        (lambda *a, **k: _Completed(1, b"", b"error connecting to github.com"), "network"),
        (lambda *a, **k: _Completed(0, b"not json at all", b""), "parse-error"),
        (lambda *a, **k: _Completed(0, b'{"x": 1}', b""), "parse-error"),
        (lambda *a, **k: _Completed(0, b"[null]", b""), "parse-error"),      # non-object element
        (lambda *a, **k: _Completed(0, b"[5, 6]", b""), "parse-error"),      # non-object elements
        (lambda *a, **k: _Completed(1, b"", b"HTTP 403: rate limit exceeded"), "gh-error"),
    ],
)
def test_gather_issues_fail_soft(monkeypatch, tmp_path: Path, side_effect, reason: str) -> None:
    # Every failure - including a weird-but-valid-JSON payload - returns a sentinel, never raises.
    monkeypatch.setattr(github_issues.subprocess, "run", side_effect)
    result = github_issues.gather_issues(tmp_path, limit_recent=5)
    assert result["available"] is False and result["reason"] == reason
    assert "detail" in result and result["source"] == github_issues.ISSUES_SOURCE


def test_summarize_issues_flags_total_open_as_lower_bound_at_cap() -> None:
    assert github_issues._summarize_issues([_issue(1, "[web] a")], 10)["total_open_is_lower_bound"] is False
    capped = github_issues._summarize_issues([_issue(i, "x") for i in range(github_issues.GH_FETCH_LIMIT)], 5)
    assert capped["total_open"] == github_issues.GH_FETCH_LIMIT
    assert capped["total_open_is_lower_bound"] is True  # the drop is flagged, not hidden


def test_parse_area_bounds_the_tag() -> None:
    # A pathological long tag must not re-inject content past the title's own cap.
    assert github_issues.parse_area("[" + "a" * 200 + "] x") == "a" * 64


@pytest.mark.parametrize(
    "title, area",
    [("[web] wire aria", "web"), ("[rust-core][phase5] x", "rust-core"),
     ("no bracket here", "untagged"), ("[] empty", "untagged"), ("  [Security] X", "security")],
)
def test_parse_area(title: str, area: str) -> None:
    assert github_issues.parse_area(title) == area


def test_gather_issues_survives_weird_field_types(monkeypatch, tmp_path: Path) -> None:
    # D8: a valid array of objects with weird field types (non-string title, non-int number) must NOT
    # raise - _summarize_issues coerces defensively (the fail-soft sensor never raises).
    payload = json.dumps([{"title": 123, "number": "x", "updatedAt": None}]).encode("utf-8")
    monkeypatch.setattr(github_issues.subprocess, "run", lambda *a, **k: _Completed(0, payload))
    result = github_issues.gather_issues(tmp_path, limit_recent=5)
    assert result["available"] is True and result["total_open"] == 1
    assert result["recent"][0]["area"] == "untagged" and result["recent"][0]["number"] == 0


def test_current_branch_non_utf8_fails_closed(monkeypatch, tmp_path: Path) -> None:
    # D7: a git-permitted non-UTF8 branch name must fail CLOSED (exit 2), not crash `status` with a
    # traceback + exit 1. current_branch now routes through the fail-closed _decode_utf8 helper.
    from assurance import git_state

    ctx = git_state.GitContext(root=tmp_path, git_dir=tmp_path, head_oid="deadbeef", is_shallow=False)

    class _NonUtf8:
        returncode = 0
        stdout = b"br-\xff-x"

    monkeypatch.setattr(git_state, "_git", lambda *a, **k: _NonUtf8())
    with pytest.raises(git_state.UnsupportedPathEncoding):
        git_state.current_branch(ctx)


# --------------------------------------------------------------------------- snapshot composition


# NB: patch `status.gather_issues` (narrow) - NOT github_issues.subprocess.run, which is the shared
# subprocess module and would also fake git_state's git calls.
def _avail(*issues: dict, limit: int = 10) -> dict:
    return github_issues._summarize_issues(list(issues), limit)


def _unavail(reason: str = "gh-missing") -> dict:
    return {"available": False, "reason": reason, "detail": "stub", "source": github_issues.ISSUES_SOURCE}


def test_summarize_issues_bounds_and_counts_the_drop() -> None:
    five = [_issue(1, "[web] a"), _issue(2, "[web] b"), _issue(3, "[train] c"),
            _issue(4, "[docs] d"), _issue(5, "e")]  # 5 issues, 4 tags (+ untagged)
    summary = github_issues._summarize_issues(five, 2)
    assert summary["total_open"] == 5 and len(summary["recent"]) == 2
    assert summary["recent_omitted_count"] == 3 and sum(summary["by_area"].values()) == 5
    numbers = [r["number"] for r in summary["recent"]]
    assert numbers == sorted(numbers, reverse=True)  # emitted number-desc


def test_status_seals_when_issues_unavailable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(status, "gather_issues", lambda *a, **k: _unavail())
    record = status.build_status_record(start_dir=_repo(tmp_path), base_ref="HEAD")
    assert record["record_type"] == "project_status"
    assert record["payload"]["issues"]["available"] is False
    assert "applicability_key" not in record["payload"]  # honest: a mixed record makes no such claim
    assert record["provenance"]["is_measurement"] is True
    assert verify_record(record)


def test_status_deterministic_block_is_byte_stable(monkeypatch, tmp_path: Path) -> None:
    fixed = _avail(_issue(5, "[web] a"))
    monkeypatch.setattr(status, "gather_issues", lambda *a, **k: fixed)
    repo = _repo(tmp_path)
    r1 = status.build_status_record(start_dir=repo, base_ref="HEAD")
    r2 = status.build_status_record(start_dir=repo, base_ref="HEAD")
    assert canonical_dumps(r1) == canonical_dumps(r2)  # no wall-clock -> byte-identical on a fixed tree
    assert verify_record(r1)


def test_status_composition_surfaces_change_in_both_change_set_and_impact(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(status, "gather_issues", lambda *a, **k: _avail())
    repo = _repo(tmp_path)
    (repo / "engine" / "corpus_studio" / "platform").mkdir(parents=True)
    (repo / "engine" / "corpus_studio" / "platform" / "worker.py").write_text("x\n")
    p = status.build_status_record(start_dir=repo, base_ref="HEAD")["payload"]
    assert p["change_set"]["changed_path_count"] >= 1 and p["change_set"]["fingerprint"].startswith("sha256:")
    fired = {o["id"]: o["severity"] for o in p["impact"]["fired_obligations"]}
    assert fired.get("worker-closure") == "blocking"
    assert p["impact"]["by_severity"].get("blocking", 0) >= 1


def test_status_bounds_recent_commits(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(status, "gather_issues", lambda *a, **k: _avail())
    repo = _repo(tmp_path)
    for i in range(3):
        (repo / f"f{i}.txt").write_text(f"{i}\n")
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", f"commit {i}"], check=True)
    p = status.build_status_record(start_dir=repo, base_ref="HEAD", limit_commits=2)["payload"]
    assert p["recent_commit_count"] == 2 and p["recent_commit_limit"] == 2
    assert p["recent_commits"][0]["subject"] == "commit 2"  # newest first


# --------------------------------------------------------------------------- CLI (exit contract)


def test_cli_status_exit_0_even_when_gh_fails(tmp_path: Path) -> None:
    gh_bin = _stub_gh(tmp_path / "bin", issues_json="", exit_code=1, stderr="HTTP 401: Bad credentials")
    proc = run_cli(_repo(tmp_path), "status", "--base", "HEAD", gh_bin=gh_bin)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)["payload"]
    assert payload["issues"]["available"] is False and payload["issues"]["reason"] == "unauthenticated"


def test_cli_status_exit_0_with_issues(tmp_path: Path) -> None:
    gh_bin = _stub_gh(tmp_path / "bin", issues_json=json.dumps([_issue(9, "[web] x")]))
    proc = run_cli(_repo(tmp_path), "status", "--base", "HEAD", gh_bin=gh_bin)
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)["payload"]
    assert payload["issues"]["available"] is True and payload["issues"]["total_open"] == 1


def test_cli_status_fail_closed_exit_2(tmp_path: Path) -> None:
    # A non-repo start dir is a fail-CLOSED git error (exit 2) - a gh failure must never mask it.
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    not_repo = tmp_path / "empty"
    not_repo.mkdir()
    proc = subprocess.run(
        [sys.executable, str(CS_ASSURE), "status", "--base", "HEAD"],
        cwd=str(not_repo), capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 2 and not proc.stdout.strip()


def test_status_impact_key_cross_consistent_with_impact_command(tmp_path: Path) -> None:
    gh_bin = _stub_gh(tmp_path / "bin", issues_json="[]")
    repo = _repo(tmp_path)
    (repo / "engine" / "corpus_studio" / "platform").mkdir(parents=True)
    (repo / "engine" / "corpus_studio" / "platform" / "worker.py").write_text("x\n")
    status_rec = json.loads(run_cli(repo, "status", "--base", "HEAD", gh_bin=gh_bin).stdout)
    impact_rec = json.loads(run_cli(repo, "impact", "--base", "HEAD").stdout)
    assert (status_rec["payload"]["impact"]["impact_applicability_key"]
            == impact_rec["payload"]["applicability_key"])
