"""Tests for the docs-freshness coupling check (scripts/loop/docs.py).

Pins gap detection (code changed without its doc), the no-gap cases (doc also changed / unrelated
change), the match semantics (dir prefix / exact file / glob), the observation mapping, and that a gap
becomes a valid correction task.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from loop.controller import Observation  # noqa: E402
from loop.docs import (  # noqa: E402
    DEFAULT_COUPLINGS,
    DocCoupling,
    doc_correction_tasks,
    docs_observation,
    stale_docs,
)
from loop.tasks import parse_tasks  # noqa: E402

_C = (DocCoupling(name="loopdoc", code_globs=["scripts/loop/"], doc_paths=["docs/AUTONOMOUS_LOOP.md"],
                  reason="loop change needs its doc"),)


def test_code_changed_without_doc_is_a_gap() -> None:
    gaps = stale_docs(["scripts/loop/controller.py"], _C)
    assert [g.coupling for g in gaps] == ["loopdoc"]
    assert gaps[0].changed_code == ["scripts/loop/controller.py"]


def test_no_gap_when_the_doc_also_changed() -> None:
    assert stale_docs(["scripts/loop/controller.py", "docs/AUTONOMOUS_LOOP.md"], _C) == []


def test_no_gap_for_unrelated_changes() -> None:
    assert stale_docs(["engine/corpus_studio/foo.py", "README.md"], _C) == []


def test_match_semantics_dir_file_and_glob() -> None:
    dir_c = (DocCoupling("d", ["engine/corpus_studio/platform/"], ["docs/contracts/"], "x"),)
    assert stale_docs(["engine/corpus_studio/platform/enums.py"], dir_c)          # dir prefix
    assert not stale_docs(["engine/corpus_studio/evaluation/x.py"], dir_c)         # outside the dir
    glob_c = (DocCoupling("g", ["engine/**/cli.py"], ["docs/CLI_REFERENCE.md"], "x"),)
    assert stale_docs(["engine/corpus_studio/cli.py"], glob_c)                     # glob match


def test_docs_observation_mapping() -> None:
    assert docs_observation([])[0] is Observation.SUCCESS
    obs, reason = docs_observation(stale_docs(["scripts/loop/x.py"], _C))
    assert obs is Observation.CONTRACT_DRIFT and "loopdoc" in reason


def test_gap_becomes_a_valid_correction_task() -> None:
    gaps = stale_docs(["scripts/loop/x.py"], _C)
    tasks = parse_tasks(doc_correction_tasks(gaps))  # must validate as a real task graph
    assert tasks[0].id == "docs-loopdoc" and tasks[0].allowed_paths == ["docs/AUTONOMOUS_LOOP.md"]


def test_default_couplings_flag_a_loop_change_without_its_doc() -> None:
    # The real default set: touching the loop without its doc is a gap; touching both is clean.
    assert any(g.coupling == "autonomous-loop"
               for g in stale_docs(["scripts/loop/router.py"], DEFAULT_COUPLINGS))
    assert not any(g.coupling == "autonomous-loop" for g in stale_docs(
        ["scripts/loop/router.py", "docs/AUTONOMOUS_LOOP.md"], DEFAULT_COUPLINGS))
