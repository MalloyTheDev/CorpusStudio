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
    head_bound_merge,
    merge_gate,
    observe_ci,
    parse_ci_checks,
)


def _checks(*pairs: tuple[str, str]) -> list[dict]:
    return [{"name": n, "bucket": b} for n, b in pairs]


def _gh_view(head: str = "sha0", *pairs: tuple[str, str]):
    """A fake gh whose `pr view` returns one snapshot: a head SHA + its statusCheckRollup."""
    payload = json.dumps({"headRefOid": head, "statusCheckRollup": list(_checks(*pairs))})
    return lambda *_argv: (0, payload, "")


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


def test_ci_observation_green_is_success_but_no_checks_is_progress() -> None:
    assert ci_observation(parse_ci_checks(_checks(("a", "pass"))))[0] is Observation.SUCCESS
    # No checks reported yet must NOT read as done (would merge before CI validates the diff).
    assert ci_observation(parse_ci_checks([]))[0] is Observation.PROGRESS


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


def test_merge_gate_escalates_every_human_gated_obligation() -> None:
    # worker-closure needs the human-gated worker workflow (fresh wheel/env); loop-controller-self-modify
    # must NOT be admitted by the loop's own merge gate (rule #666) - both escalate, like self-modify/sealed.
    for ob in ("assurance-self-modify", "sealed-research", "worker-closure", "loop-controller-self-modify"):
        gate = merge_gate([ob, "contracts"])
        assert not gate.authorized and gate.observation is Observation.AUTHORIZATION_REQUIRED, ob


def test_merge_gate_fails_closed_on_an_unknown_blocking_obligation() -> None:
    # RISK FROM POLICY: a NEW blocking obligation the gate has never heard of must escalate by default,
    # not silently auto-merge. A non-blocking (advisory/info) unknown obligation does not gate.
    assert not merge_gate([{"id": "brand-new-rule", "severity": "blocking"}]).authorized
    assert merge_gate([{"id": "some-advisory", "severity": "advisory"}]).authorized
    # contracts / evaluation-honesty are blocking but candidate-satisfiable when CI is green -> auto-merge.
    assert merge_gate([{"id": "contracts", "severity": "blocking"}]).authorized


def test_merge_gate_fails_closed_on_unknown_or_missing_severity() -> None:
    # An obligation with an UNKNOWN / missing severity that is not on the auto-mergeable allowlist must
    # escalate - we cannot confirm it is safe, so we do not auto-merge it (fail-closed, not fail-open).
    assert not merge_gate([{"id": "mystery"}]).authorized          # no severity field
    assert not merge_gate([{"id": "mystery", "severity": ""}]).authorized
    assert not merge_gate(["a-bare-string-obligation"]).authorized  # advertised bare form, no severity


def test_merge_gate_fails_closed_on_a_malformed_obligation_entry() -> None:
    # An unparseable entry (not a str, not a dict-with-str-id) means we cannot assess risk -> escalate.
    assert not merge_gate([123]).authorized
    assert not merge_gate([{"severity": "blocking"}]).authorized  # dict without a string id


def test_merge_gate_dangerous_is_a_one_way_override() -> None:
    gate = merge_gate([], dangerous=True)
    assert not gate.authorized and gate.observation is Observation.AUTHORIZATION_REQUIRED


# --------------------------------------------------------------------------- observe_ci runner


def test_observe_ci_parses_one_snapshot_with_head_sha() -> None:
    snap = observe_ci(_gh_view("deadbeef", ("ruff", "pass"), ("pytest", "fail")), "42")
    assert snap.status.state == "failing" and snap.status.failing_checks == ["pytest"]
    assert snap.head_sha == "deadbeef"  # bound to the head the checks ran against


def test_observe_ci_fails_closed_on_a_gh_error() -> None:
    # A non-zero gh exit is a real error (auth/network/not-found) - fail closed, never read as "no checks".
    with pytest.raises(IntegrateError):
        observe_ci(lambda *_a: (1, "", "not found"), "42")
    with pytest.raises(IntegrateError):
        observe_ci(lambda *_a: (0, "not json", ""), "42")


def test_observe_ci_treats_a_null_rollup_as_no_checks_not_a_crash() -> None:
    # A PR with no checks configured reports statusCheckRollup: null; that is 'no checks' (state none ->
    # keep observing), NOT a parse error - it must never crash the loop.
    snap = observe_ci(lambda *_a: (0, json.dumps({"headRefOid": "abc", "statusCheckRollup": None}), ""), "42")
    assert snap.status.state == "none" and snap.head_sha == "abc"


def test_parse_handles_real_status_check_rollup_shapes() -> None:
    # Real `gh pr view --json statusCheckRollup`: CheckRun {name,status,conclusion} with UPPERCASE values,
    # and StatusContext {context,state}. A green CheckRun + a failing StatusContext -> failing, named.
    rollup = [
        {"__typename": "CheckRun", "name": "pytest", "status": "COMPLETED", "conclusion": "SUCCESS"},
        {"__typename": "StatusContext", "context": "ci/deploy", "state": "FAILURE"},
    ]
    status = parse_ci_checks(rollup)
    assert status.state == "failing" and status.failing_checks == ["ci/deploy"]
    assert status.passing_checks == ["pytest"]


def test_ci_observation_holds_when_a_required_check_has_not_reported() -> None:
    # An all-green rollup that is still MISSING a required check must not read as done (partial-green race).
    green_subset = parse_ci_checks(_checks(("lint", "pass")))
    assert ci_observation(green_subset, required=frozenset({"pytest"}))[0] is Observation.PROGRESS
    full = parse_ci_checks(_checks(("lint", "pass"), ("pytest", "pass")))
    assert ci_observation(full, required=frozenset({"pytest"}))[0] is Observation.SUCCESS


# --------------------------------------------------------------------------- head-bound merge (race-safe)


def test_head_bound_merge_binds_to_the_observed_head() -> None:
    seen: list[tuple[str, ...]] = []

    def gh(*argv: str) -> tuple[int, str, str]:
        seen.append(argv)
        return (0, "merged", "")
    obs, _ = head_bound_merge(gh, "42", "abc123")
    assert obs is Observation.SUCCESS
    assert "--match-head-commit" in seen[0] and "abc123" in seen[0]  # the merge is pinned to that head


def test_head_bound_merge_holds_when_the_head_moved() -> None:
    obs, reason = head_bound_merge(
        lambda *_a: (1, "", "Head branch was modified; it is not the most recent commit"), "42", "abc123")
    assert obs is Observation.HOLD and "moved" in reason  # a new commit since CI -> re-observe, don't merge


def test_head_bound_merge_holds_without_a_head_rather_than_merging_blind() -> None:
    obs, _ = head_bound_merge(lambda *_a: (0, "merged", ""), "42", None)
    assert obs is Observation.HOLD  # no binding -> never merge blind


def test_head_bound_merge_reports_a_real_merge_failure() -> None:
    obs, _ = head_bound_merge(lambda *_a: (1, "", "merge conflict"), "42", "abc123")
    assert obs is Observation.TEST_REGRESSION
