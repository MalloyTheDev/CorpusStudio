"""Tests for CI / PR continuation (scripts/loop/integrate.py, controller slice 7 - completes L7).

Pins CI-check parsing (failure dominates, pending is not done, none = nothing to satisfy), the CI-status
-> Observation mapping (incl. name-based classification), the merge-authorization gate (product merges,
self-modify / sealed-research / dangerous escalate), and fail-closed CI observation via an injected gh.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from loop.controller import Observation  # noqa: E402
from loop.integrate import (  # noqa: E402
    IntegrateError,
    ci_observation,
    merge_gate,
    observe_ci,
    parse_ci_checks,
)


def _checks(*pairs: tuple[str, str]) -> list[dict]:
    return [{"name": n, "bucket": b} for n, b in pairs]


# --------------------------------------------------------------------------- parsing


def test_parse_failure_dominates_then_pending_then_pass() -> None:
    assert parse_ci_checks(_checks(("a", "pass"), ("b", "fail"), ("c", "pending"))).state == "failing"
    assert parse_ci_checks(_checks(("a", "pass"), ("c", "pending"))).state == "pending"
    assert parse_ci_checks(_checks(("a", "pass"), ("b", "pass"))).state == "passing"
    assert parse_ci_checks([]).state == "none"


def test_parse_collects_failing_names_and_counts() -> None:
    status = parse_ci_checks(_checks(("ruff", "pass"), ("pytest", "fail"), ("mypy", "fail")))
    assert status.failing_checks == ["pytest", "mypy"] and status.total == 3 and status.passed == 1


def test_parse_fails_closed_on_non_list() -> None:
    with pytest.raises(IntegrateError):
        parse_ci_checks({"not": "a list"})


# --------------------------------------------------------------------------- status -> observation


def test_ci_observation_pending_is_progress_not_done() -> None:
    obs, _ = ci_observation(parse_ci_checks(_checks(("a", "pass"), ("b", "pending"))))
    assert obs is Observation.PROGRESS


def test_ci_observation_green_and_none_are_success() -> None:
    assert ci_observation(parse_ci_checks(_checks(("a", "pass"))))[0] is Observation.SUCCESS
    assert ci_observation(parse_ci_checks([]))[0] is Observation.SUCCESS


def test_ci_observation_classifies_failure_by_check_name() -> None:
    cases = {
        "ruff": Observation.SYNTAX_FAILURE,
        "mypy": Observation.TYPE_FAILURE,
        "pytest": Observation.TEST_REGRESSION,
        "web-build": Observation.DEPENDENCY_FAILURE,
        "something-else": Observation.TEST_REGRESSION,  # generic CI failure
    }
    for name, expected in cases.items():
        obs, _ = ci_observation(parse_ci_checks(_checks(("ok", "pass"), (name, "fail"))))
        assert obs is expected, name


# --------------------------------------------------------------------------- merge authorization


def test_merge_gate_allows_a_product_change() -> None:
    gate = merge_gate(["contracts", "evaluation-honesty"])
    assert gate.authorized and gate.observation is Observation.SUCCESS


def test_merge_gate_escalates_self_modify_and_sealed_research() -> None:
    for ob in ("assurance-self-modify", "sealed-research"):
        gate = merge_gate([ob, "contracts"])
        assert not gate.authorized and gate.observation is Observation.AUTHORIZATION_REQUIRED


def test_merge_gate_escalates_a_dangerous_action_even_with_no_obligation() -> None:
    gate = merge_gate([], dangerous=True)
    assert not gate.authorized and gate.observation is Observation.AUTHORIZATION_REQUIRED


# --------------------------------------------------------------------------- observe_ci runner


def test_observe_ci_parses_injected_gh_output() -> None:
    def gh(*_argv: str) -> tuple[int, str, str]:
        return (0, json.dumps(_checks(("ruff", "pass"), ("pytest", "fail"))), "")
    status = observe_ci(gh, "42")
    assert status.state == "failing" and status.failing_checks == ["pytest"]


def test_observe_ci_fails_closed_on_bad_output() -> None:
    with pytest.raises(IntegrateError):
        observe_ci(lambda *_a: (1, "not json", "boom"), "42")
