"""Robustness invariants for the assurance loop (holistic-audit + expansion-design hardening).

The non-negotiable: EVERY refusal fails CLOSED (exit 2), never a bare traceback aliased onto the
exit-1 "not-clean" (doclint-stale / red-gate) rung. A candidate-controlled deeply-nested JSON
policy/registry/gate makes `json.loads` raise RecursionError - which is NOT a ValueError, so it used
to escape every loader's narrow except AND main()'s typed catch. These tests pin the fix + the
main() catch-all backstop.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CS_ASSURE = SCRIPTS_DIR / "cs_assure.py"
POLICY_DIR = SCRIPTS_DIR / "assurance" / "policy"
REAL_OBLIGATIONS = (POLICY_DIR / "obligations.json").read_text("utf-8")
REAL_REGISTRY = (POLICY_DIR / "context_sources.json").read_text("utf-8")
REAL_GATE = (POLICY_DIR / "gate.json").read_text("utf-8")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# 20k-deep nested JSON: json.loads raises RecursionError (confirmed >~ the default recursion limit).
_DEEP_JSON = "[" * 20000 + "]" * 20000


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts" / "assurance" / "policy").mkdir(parents=True)
    for a in (("init", "-q", "-b", "main"), ("config", "user.email", "a@b.c"),
              ("config", "user.name", "t")):
        _git(repo, *a)
    (repo / "scripts" / "assurance" / "policy" / "obligations.json").write_text(REAL_OBLIGATIONS)
    (repo / "scripts" / "assurance" / "policy" / "context_sources.json").write_text(REAL_REGISTRY)
    (repo / "scripts" / "assurance" / "policy" / "gate.json").write_text(REAL_GATE)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed")
    return repo


def _run(start_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CS_ASSURE), *args], cwd=str(start_dir), capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "subcommand, deep_file",
    [
        (["impact", "--base", "HEAD"], "obligations.json"),
        (["doclint"], "context_sources.json"),
        (["verify", "--base", "HEAD"], "gate.json"),
        (["status", "--base", "HEAD"], "obligations.json"),
    ],
)
def test_loaders_fail_closed_on_deep_json(tmp_path: Path, subcommand: list[str], deep_file: str) -> None:
    # A hostile deeply-nested policy/registry/gate must fail CLOSED (exit 2) with a structured refusal,
    # NEVER a bare traceback + exit 1 (which aliases onto the doclint-stale / red-gate rung).
    repo = _repo(tmp_path)
    (repo / "scripts" / "assurance" / "policy" / deep_file).write_text(_DEEP_JSON)
    proc = _run(repo, *subcommand)
    assert proc.returncode == 2, (proc.returncode, proc.stderr)
    assert proc.stdout.strip() == ""  # no partial record on stdout
    assert proc.stderr.startswith("cs_assure:")  # a structured refusal, not a bare traceback
    assert "Traceback" not in proc.stderr


def test_main_backstop_catches_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # The main() catch-all backstop maps ANY leaked non-typed error to fail-closed exit 2 (never a
    # bare traceback), while still letting KeyboardInterrupt / SystemExit propagate.
    import cs_assure

    def _boom(_args: object) -> int:
        raise RuntimeError("leaked past the typed handlers")

    monkeypatch.setattr(cs_assure, "_cmd_changeset", _boom)
    assert cs_assure.main(["changeset", "--base", "HEAD", "--start-dir", str(REPO_ROOT)]) == 2
