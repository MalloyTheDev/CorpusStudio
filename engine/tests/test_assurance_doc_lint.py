"""Tests for the context-source registry + deterministic doc-lint (assurance Phase 5, slice 1).

The rule behaviour is proven with SYNTHETIC lines so the tests stay green after the prose slices
fix the real docs; the real-repo assertions deliberately avoid hard-coding finding counts (those
shrink to zero as the docs are cleaned) and check structure + determinism instead.
"""

from __future__ import annotations

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

from assurance.doc_lint import (  # noqa: E402
    DocLintError,
    DocSource,
    lint_repo,
    load_registry,
    parse_registry,
    scan_source,
)


def _src(mode: str = "CURRENT", *, stable_guidance: bool = False, authority: str = "canonical") -> DocSource:
    return DocSource(
        path="x.md",
        mode=mode,
        authority=authority,
        always_loaded=False,
        stable_guidance=stable_guidance,
        superseded_by=None,
        note="",
    )


def _rules(source: DocSource, lines: list[str]) -> set[str]:
    return {f.rule for f in scan_source(source, lines)}


def run_cli(start_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CS_ASSURE), *args],
        cwd=str(start_dir),
        capture_output=True,
        text=True,
    )


# --------------------------------------------------------------------------- removed-UI rule


def test_removed_ui_flagged_in_current_doc() -> None:
    assert "removed-ui-present-tense" in _rules(_src("CURRENT"), ["the WPF desktop tab is here"])


def test_removed_ui_marker_on_same_line_suppresses() -> None:
    assert "removed-ui-present-tense" not in _rules(
        _src("CURRENT"), ["the WPF/Avalonia desktop was removed (#545)"]
    )


def test_removed_ui_marker_on_adjacent_line_suppresses() -> None:
    lines = ["The WPF/Avalonia desktop that", "was removed in #545 is gone"]
    assert "removed-ui-present-tense" not in _rules(_src("CURRENT"), lines)


def test_removed_ui_not_flagged_in_historical_doc() -> None:
    assert "removed-ui-present-tense" not in _rules(_src("HISTORICAL"), ["the WPF desktop tab"])


def test_nocturne_is_not_treated_as_removed_ui() -> None:
    # Nocturne is the framework-agnostic design system that carries forward to Tauri/React.
    assert _rules(_src("CURRENT"), ["the Nocturne design system and its tokens"]) == set()


# --------------------------------------------------------------------------- host-path + volatile rules


def test_absolute_host_path_flagged_only_in_stable_guidance() -> None:
    line = ["develop from /mnt/training-nvme/repos/CorpusStudio on the RTX 5070"]
    assert "absolute-host-path" in _rules(_src("CURRENT", stable_guidance=True), line)
    assert "absolute-host-path" not in _rules(_src("CURRENT", stable_guidance=False), line)


def test_volatile_identity_flagged_only_in_stable_guidance() -> None:
    line = ["fresh wheel 090f879b from source_commit 21aa81d9, matrix 1.4.0"]
    assert "volatile-identity-in-stable" in _rules(_src("CURRENT", stable_guidance=True), line)
    assert "volatile-identity-in-stable" not in _rules(_src("VOLATILE_CURRENT"), line)


# --------------------------------------------------------------------------- github-setting + count rules


def test_github_setting_asserted_as_fact_is_flagged() -> None:
    found = scan_source(_src("CURRENT"), ["merge after green CI; admin-merge per standing authorization"])
    assert any(f.classification == "AUTHENTICATED_GITHUB_SETTING_REQUIRED" for f in found)


def test_known_count_drift_scoped_to_durable_current_docs() -> None:
    line = ["this branch exports 28 root contracts"]
    assert "known-count-drift" in _rules(_src("CURRENT"), line)
    assert "known-count-drift" in _rules(_src("MIXED_CURRENT_AND_HISTORY"), line)
    # a dated log or a frozen record legitimately preserves the count it had when written.
    assert "known-count-drift" not in _rules(_src("VOLATILE_CURRENT"), line)
    assert "known-count-drift" not in _rules(_src("FROZEN_EVIDENCE"), line)


# --------------------------------------------------------------------------- registry validation


def test_parse_registry_rejects_invalid_mode() -> None:
    with pytest.raises(DocLintError):
        parse_registry({"sources": [{"path": "x.md", "mode": "BOGUS", "authority": "canonical"}]})


def test_parse_registry_rejects_invalid_authority() -> None:
    with pytest.raises(DocLintError):
        parse_registry({"sources": [{"path": "x.md", "mode": "CURRENT", "authority": "nope"}]})


def test_parse_registry_rejects_empty_sources() -> None:
    with pytest.raises(DocLintError):
        parse_registry({"sources": []})


def test_real_registry_loads_and_covers_always_loaded_contract() -> None:
    sources = load_registry(REPO_ROOT)
    by_path = {s.path: s for s in sources}
    assert "CLAUDE.md" in by_path and by_path["CLAUDE.md"].always_loaded
    assert "AGENTS.md" in by_path and by_path["AGENTS.md"].stable_guidance
    # the sealed protocol is registered FROZEN so the staleness rules never touch it.
    assert by_path["research/ieee-linux-training/PROTOCOL.md"].mode == "FROZEN_EVIDENCE"


def test_lint_repo_flags_a_registered_but_missing_file(tmp_path: Path) -> None:
    findings = lint_repo(tmp_path, [_src("CURRENT")])  # x.md does not exist under tmp_path
    assert [f.rule for f in findings] == ["registry-missing-file"]


# --------------------------------------------------------------------------- CLI


def test_cli_doclint_json_is_wellformed_and_deterministic() -> None:
    first = run_cli(REPO_ROOT, "doclint", "--format", "json")
    assert first.returncode == 0, first.stderr
    payload = json.loads(first.stdout)
    assert payload["finding_count"] == len(payload["findings"])
    assert payload["registry_source_count"] > 0
    for finding in payload["findings"]:
        assert {"rule", "severity", "classification", "path", "line", "message"} <= set(finding)
    second = run_cli(REPO_ROOT, "doclint", "--format", "json")
    assert first.stdout == second.stdout  # deterministic


def test_cli_doclint_strict_exit_matches_finding_count() -> None:
    payload = json.loads(run_cli(REPO_ROOT, "doclint", "--format", "json").stdout)
    strict = run_cli(REPO_ROOT, "doclint", "--strict")
    assert strict.returncode == (1 if payload["finding_count"] else 0)


# --------------------------------------------------------------------------- hardening (adversarial)


def test_removed_ui_bare_was_does_not_suppress() -> None:
    # "was"/"were" are too common to be historical markers: a genuine present-tense WPF claim next
    # to an unrelated "was" must STILL be flagged (previously it was silently suppressed).
    lines = ["The build was slow yesterday.", "The WPF Training Studio renders the tabs."]
    assert "removed-ui-present-tense" in _rules(_src("CURRENT"), lines)


def test_removed_ui_real_marker_still_suppresses() -> None:
    # A genuine historical marker on an adjacent line still suppresses (no regression from the fix).
    lines = ["The WPF Training Studio", "was removed in #545."]
    assert "removed-ui-present-tense" not in _rules(_src("CURRENT"), lines)


def test_volatile_labeled_bare_hash_flagged_in_stable_guidance() -> None:
    src = _src("CURRENT", stable_guidance=True)
    assert "volatile-identity-in-stable" in _rules(src, ["the wheel was built from source fedd7d5"])
    assert "volatile-identity-in-stable" in _rules(src, ["floor 45bdd989 for this lineage"])


def _tmp_repo(tmp_path: Path, doc_text: str, *, mode: str = "CURRENT") -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts" / "assurance" / "policy").mkdir(parents=True)
    for args in (("init", "-q", "-b", "main"), ("config", "user.email", "a@b.c"),
                 ("config", "user.name", "t")):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    (repo / "doc.md").write_text(doc_text)
    (repo / "scripts" / "assurance" / "policy" / "context_sources.json").write_text(
        json.dumps({"sources": [{"path": "doc.md", "mode": mode, "authority": "canonical"}]})
    )
    return repo


def test_cli_doclint_strict_exit_1_on_planted_staleness(tmp_path: Path) -> None:
    # Deterministic exercise of the --strict GATE branch: the real-tree strict test only sees exit 0
    # once the docs are clean, so a planted stale doc proves exit 1 (and a clean doc proves exit 0).
    repo = _tmp_repo(tmp_path, "The WPF desktop launches the trainer.\n")
    detect = run_cli(repo, "doclint", "--format", "json")
    assert detect.returncode == 0 and json.loads(detect.stdout)["finding_count"] >= 1
    assert run_cli(repo, "doclint", "--strict").returncode == 1
    (repo / "doc.md").write_text("The Tauri 2 + React client shows the trainer.\n")
    assert run_cli(repo, "doclint", "--strict").returncode == 0


def test_parse_registry_rejects_duplicate_path() -> None:
    raw = {"sources": [
        {"path": "a.md", "mode": "CURRENT", "authority": "canonical"},
        {"path": "a.md", "mode": "HISTORICAL", "authority": "derived"},
    ]}
    with pytest.raises(DocLintError, match="more than once"):
        parse_registry(raw)


def test_parse_registry_rejects_out_of_tree_path() -> None:
    for bad in ("/etc/passwd", "../secret.md", "a/../../b.md", "a\\b.md"):
        with pytest.raises(DocLintError, match="repo-relative"):
            parse_registry({"sources": [{"path": bad, "mode": "CURRENT", "authority": "canonical"}]})


def test_load_registry_fails_closed_when_registry_is_a_directory(tmp_path: Path) -> None:
    # A registry path that is a directory (IsADirectoryError) must fail CLOSED as DocLintError, not
    # escape as an uncaught OSError -> exit-1 traceback (which collides with the --strict signal).
    (tmp_path / "scripts" / "assurance" / "policy" / "context_sources.json").mkdir(parents=True)
    with pytest.raises(DocLintError):
        load_registry(tmp_path)


def test_lint_repo_flags_registered_directory_instead_of_silently_skipping(tmp_path: Path) -> None:
    (tmp_path / "adir").mkdir()
    source = DocSource(path="adir", mode="CURRENT", authority="canonical", always_loaded=False,
                       stable_guidance=False, superseded_by=None, note="")
    assert [f.rule for f in lint_repo(tmp_path, [source])] == ["registry-not-a-file"]
