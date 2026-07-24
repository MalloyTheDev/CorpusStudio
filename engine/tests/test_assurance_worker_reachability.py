"""Two-sided static worker-reachability analysis (re-review #16).

The ``worker-closure`` obligation flagged only a DECLARED 7-path list; a change to a module the worker
transitively imports but that is not on the list fired no obligation. These tests pin the pure import-graph
analyzer (a dict reader, no git), the git-backed two-sided builder (a throwaway repo), and the guards.
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

from assurance.records import verify_record  # noqa: E402
from assurance.worker_reachability import (  # noqa: E402
    DEFAULT_WORKER_ROOTS,
    WorkerReachabilityError,
    analyze_two_sided,
    build_worker_reachability_record,
    module_to_path,
    path_to_module,
    reachable_from,
)

PKG = "engine/corpus_studio"


def _reader(files: dict[str, bytes]):
    return lambda path: files.get(path)


def _mod(rel: str, body: str) -> tuple[str, bytes]:
    return f"{PKG}/{rel}", body.encode("utf-8")


# --------------------------------------------------------------------------- module <-> path


def test_path_and_module_round_trip() -> None:
    assert path_to_module(f"{PKG}/platform/worker.py") == ("corpus_studio.platform.worker", False)
    assert path_to_module(f"{PKG}/platform/__init__.py") == ("corpus_studio.platform", True)
    assert path_to_module("engine/README.md") is None  # not a .py under the package root
    read = _reader({f"{PKG}/a.py": b"", f"{PKG}/pkg/__init__.py": b""})
    assert module_to_path("corpus_studio.a", read) == (f"{PKG}/a.py", False)
    assert module_to_path("corpus_studio.pkg", read) == (f"{PKG}/pkg/__init__.py", True)
    assert module_to_path("corpus_studio.missing", read) is None


# --------------------------------------------------------------------------- pure reachability


def test_follows_transitive_intra_repo_imports() -> None:
    files = dict([
        _mod("root.py", "from corpus_studio import a\nimport corpus_studio.b"),
        _mod("a.py", "from corpus_studio.c import thing"),
        _mod("b.py", "x = 1"),
        _mod("c.py", "y = 2"),
        _mod("__init__.py", ""),
    ])
    r = reachable_from((f"{PKG}/root.py",), _reader(files))
    assert set(r.reachable) == {f"{PKG}/root.py", f"{PKG}/a.py", f"{PKG}/b.py", f"{PKG}/c.py",
                                f"{PKG}/__init__.py"}
    assert r.unresolved_dynamic == () and r.unreadable == ()


def test_relative_imports_resolve() -> None:
    files = dict([
        _mod("pkg/__init__.py", ""),
        _mod("pkg/root.py", "from . import sib\nfrom .deep import leaf\nfrom ..top import mod"),
        _mod("pkg/sib.py", ""),
        _mod("pkg/deep.py", "leaf = 1"),
        _mod("top.py", "mod = 1"),
    ])
    r = reachable_from((f"{PKG}/pkg/root.py",), _reader(files))
    assert {f"{PKG}/pkg/sib.py", f"{PKG}/pkg/deep.py", f"{PKG}/top.py"} <= set(r.reachable)


def test_external_and_stdlib_imports_are_ignored() -> None:
    files = dict([_mod("root.py", "import os\nimport torch\nfrom numpy import array\nx = 1")])
    r = reachable_from((f"{PKG}/root.py",), _reader(files))
    assert set(r.reachable) == {f"{PKG}/root.py"}  # nothing intra-repo followed


def test_dynamic_imports_literal_resolved_nonliteral_recorded() -> None:
    files = dict([
        _mod("root.py",
             "import importlib\n"
             "importlib.import_module('corpus_studio.dyn')\n"
             "name = 'x'\n"
             "importlib.import_module(name)\n"
             "__import__(name)\n"),
        _mod("dyn.py", "z = 1"),
    ])
    r = reachable_from((f"{PKG}/root.py",), _reader(files))
    assert f"{PKG}/dyn.py" in r.reachable  # the string-literal target is followed
    kinds = {d["kind"] for d in r.unresolved_dynamic}
    assert kinds == {"dynamic_import"} and len(r.unresolved_dynamic) >= 1  # non-literal targets recorded


def test_unparseable_module_is_recorded_not_dropped() -> None:
    files = dict([_mod("root.py", "def broken(:\n")])  # a syntax error
    r = reachable_from((f"{PKG}/root.py",), _reader(files))
    assert r.reachable == (f"{PKG}/root.py",) and len(r.unreadable) == 1
    assert "unparseable" in r.unreadable[0]["detail"]


def test_star_import_does_not_crash_and_follows_the_module() -> None:
    files = dict([_mod("root.py", "from corpus_studio.helpers import *"), _mod("helpers.py", "a = 1")])
    r = reachable_from((f"{PKG}/root.py",), _reader(files))
    assert f"{PKG}/helpers.py" in r.reachable


# --------------------------------------------------------------------------- two-sided delta


def test_two_sided_delta_undeclared_and_distribution_impact() -> None:
    roots = (f"{PKG}/root.py",)
    base = _reader(dict([_mod("root.py", "from corpus_studio import keep"), _mod("keep.py", "a = 1")]))
    cand = _reader(dict([
        _mod("root.py", "from corpus_studio import keep\nfrom corpus_studio import fresh"),
        _mod("keep.py", "a = 2"),        # modified
        _mod("fresh.py", "b = 1"),       # newly imported + new file
    ]))
    changed = (f"{PKG}/keep.py", f"{PKG}/fresh.py", "docs/unrelated.md")
    a = analyze_two_sided(roots, base, cand, changed)
    assert a.added_reachable == (f"{PKG}/fresh.py",)     # reachable on candidate, not base
    assert a.removed_reachable == ()
    # undeclared = reachable union minus the roots (the modules the declared list would miss)
    assert set(a.undeclared_reachable) == {f"{PKG}/keep.py", f"{PKG}/fresh.py"}
    # distribution impact = changed files that ARE in the reachable closure (the unrelated doc is excluded)
    assert set(a.distribution_impacting_paths) == {f"{PKG}/keep.py", f"{PKG}/fresh.py"}


# --------------------------------------------------------------------------- git-backed builder + CLI


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / PKG / "w").mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "a@b.c")
    _git(root, "config", "user.name", "t")
    (root / PKG / "__init__.py").write_text("")
    (root / PKG / "w" / "__init__.py").write_text("")
    (root / PKG / "w" / "root.py").write_text("from corpus_studio.w import helper\n")
    (root / PKG / "w" / "helper.py").write_text("x = 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return root


def test_builder_reports_two_sided_reachability_over_a_real_repo(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    roots = (f"{PKG}/w/root.py",)
    # candidate (workspace) edits: modify helper + import a NEW untracked module.
    (root / PKG / "w" / "helper.py").write_text("x = 2\n")
    (root / PKG / "w" / "root.py").write_text("from corpus_studio.w import helper\nfrom corpus_studio.w import added\n")
    (root / PKG / "w" / "added.py").write_text("y = 1\n")

    rec = build_worker_reachability_record(start_dir=root, base_ref="main", roots=roots)
    assert verify_record(rec)
    p = rec["payload"]
    assert p["added_reachable"] == [f"{PKG}/w/added.py"]  # the new import target
    assert f"{PKG}/w/helper.py" in p["undeclared_reachable"]  # reachable but not a declared root
    # distribution impact = the changed files inside the reachable closure
    assert set(p["distribution_impacting_paths"]) == {f"{PKG}/w/helper.py", f"{PKG}/w/root.py",
                                                      f"{PKG}/w/added.py"}


def test_cli_runs_and_rejects_a_bad_scope(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    proc = subprocess.run([sys.executable, str(CS_ASSURE), "worker-reachability", "--scope", "head"],
                          cwd=str(root), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["record_type"] == "worker_static_reachability"
    bad = subprocess.run([sys.executable, str(CS_ASSURE), "worker-reachability", "--scope", "index"],
                         cwd=str(root), capture_output=True, text=True)
    assert bad.returncode == 2 and "invalid choice" in bad.stderr


def test_builder_fails_closed_on_an_unsupported_scope(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    with pytest.raises(WorkerReachabilityError):
        build_worker_reachability_record(start_dir=root, scope="merge_candidate")


# --------------------------------------------------------------------------- policy sync


def test_default_roots_match_the_worker_closure_policy_globs() -> None:
    # The reachability roots MUST stay in sync with the declared worker-closure globs, so the analysis
    # extends exactly the list the obligation is about. A drift here is a real (caught) inconsistency.
    policy = json.loads((SCRIPTS_DIR / "assurance" / "policy" / "obligations.json").read_text())
    globs = next(o["globs"] for o in policy["obligations"] if o["id"] == "worker-closure")
    assert sorted(DEFAULT_WORKER_ROOTS) == sorted(globs)
