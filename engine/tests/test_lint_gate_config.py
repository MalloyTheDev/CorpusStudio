"""The lint gate must declare WHICH rules it enforces, rather than inheriting the tool's defaults.

`[tool.ruff]` previously set only `line-length`, so the gate meant "whatever this ruff version turns on
by default" - which is not something the repository controls. Measured on an unchanged tree: 0.15.21
defaults to E4/E7/E9/F and is green, while 0.16.0 enables roughly twenty further rule families and
reports 928 findings. A routine dependency bump would therefore have silently redefined what CI enforces.

The noisy direction is the safe one - a bump that turns the tree red gets noticed. The dangerous direction
is a future ruff that NARROWS its defaults: the gate would quietly stop enforcing rules while CI kept
reporting green, which is the failure mode this repository cares about most.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
# What ruff enforced by default at the version this was pinned against; freezing it changed nothing.
HISTORICAL_DEFAULT = {"E4", "E7", "E9", "F"}


def _ruff_lint_config() -> dict:
    return tomllib.loads(PYPROJECT.read_text("utf-8")).get("tool", {}).get("ruff", {}).get("lint", {})


def test_the_ruff_rule_set_is_pinned_not_inherited() -> None:
    select = _ruff_lint_config().get("select")
    assert select, (
        "[tool.ruff.lint] select must be set explicitly - a rule set inherited from the tool's defaults "
        "is not a gate the repository controls, and changes meaning on every upgrade"
    )


def test_the_pinned_rule_set_never_silently_narrows() -> None:
    select = set(_ruff_lint_config().get("select") or [])
    missing = HISTORICAL_DEFAULT - select
    assert not missing, (
        f"the pinned rule set dropped {sorted(missing)}, which ruff enforced by default when the pin was "
        "introduced. Narrowing the gate is a deliberate decision needing its own justification - it must "
        "not happen as a silent edit, because the result still reports green while checking less"
    )
