"""#13: the loop CONTROLLER package (scripts/loop/) must import ONLY the standard library + its own
``loop.*`` siblings - never corpus_studio, torch, the assurance library, or any third-party package. That
boundary is what lets the controller run under any bare ``python3`` and stay a pure, deterministic core;
this test enforces it (so the dedicated loop CI job and the engine suite both catch a violation).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOOP_DIR = REPO_ROOT / "scripts" / "loop"


def _imported_top_level_names(py: Path) -> set[str]:
    """The top-level module name of EVERY import in the file - deliberately via ``ast.walk`` so a nested
    (function-local / conditional) import is caught too: a ``def f(): import torch`` violates the
    stdlib-only boundary just as much as a module-level one."""
    tops: set[str] = set()
    for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            tops |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            tops.add(node.module.split(".")[0])
    return tops


def test_loop_controller_imports_only_stdlib_and_loop() -> None:
    # stdlib_module_names is the authoritative set of top-level stdlib modules (3.10+); `loop` is the
    # controller's own package, `__future__` is a language pseudo-module. rglob covers any future
    # subpackage (e.g. scripts/loop/util/), not just the top-level modules.
    allowed = set(sys.stdlib_module_names) | {"loop", "__future__"}
    offenders: dict[str, list[str]] = {}
    for py in sorted(LOOP_DIR.rglob("*.py")):
        bad = _imported_top_level_names(py) - allowed
        if bad:
            offenders[py.relative_to(LOOP_DIR).as_posix()] = sorted(bad)
    assert not offenders, (
        f"scripts/loop must import only the standard library + loop.*; found non-stdlib imports: {offenders}"
    )
