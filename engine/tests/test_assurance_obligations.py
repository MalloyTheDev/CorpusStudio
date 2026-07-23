"""Tests for the assurance impact->obligations engine (Phase 4: scripts/assurance/obligations.py).

Proves the deterministic policy loader (fail-closed), the boundary-correct glob matcher, the
change-set->obligation mapping, the sealed ImpactAssessment record, the observation-only CLI, and -
critically - that the shipped policy stays in lockstep with the .claude/rules/*.md globs (so the
machine change->obligation map and the read-time rules can never drift).
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CS_ASSURE = SCRIPTS_DIR / "cs_assure.py"
REAL_POLICY_TEXT = (SCRIPTS_DIR / "assurance" / "policy" / "obligations.json").read_text("utf-8")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from assurance.obligations import (  # noqa: E402
    LoadedPolicy,
    Obligation,
    PolicyError,
    build_impact_assessment,
    glob_matches,
    load_effective_policy,
    load_policy,
    match_obligations,
    parse_policy,
    union_policy,
)
from assurance.records import verify_record  # noqa: E402


# --------------------------------------------------------------------------- helpers


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)


def _repo_with_policy(tmp_path: Path, policy_text: str = REAL_POLICY_TEXT) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts" / "assurance" / "policy").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "a@b.c")
    _git(repo, "config", "user.name", "t")
    (repo / "scripts" / "assurance" / "policy" / "obligations.json").write_text(policy_text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed")
    return repo


def _plant(repo: Path, relpath: str, text: str = "x\n") -> None:
    p = repo / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def run_cli(start_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CS_ASSURE), *args], cwd=str(start_dir), capture_output=True, text=True
    )


def impact(repo: Path) -> tuple[subprocess.CompletedProcess[str], dict]:
    proc = run_cli(repo, "impact", "--base", "HEAD")
    record = json.loads(proc.stdout) if proc.returncode == 0 and proc.stdout.strip() else {}
    return proc, record


def _ob(id_: str, globs: list[str], severity: str = "blocking") -> Obligation:
    return Obligation(id=id_, globs=tuple(globs), obligation="o", authority="a", severity=severity, source="s")


def _loaded(obs: list[Obligation]) -> LoadedPolicy:
    return LoadedPolicy(obligations=tuple(obs), digest="sha256:d", schema_version=1, relpath="p",
                        obligation_count=len(obs))


# --------------------------------------------------------------------------- trusted-base policy union (PR2)


def test_union_policy_cannot_weaken_the_trusted_base() -> None:
    base = _loaded([_ob("a", ["engine/**", "docs/**"], "blocking"), _ob("b", ["scripts/**"])])
    # candidate WEAKENS 'a' (drops docs/**, lowers severity), REMOVES 'b', ADDS 'c'
    cand = _loaded([_ob("a", ["engine/**"], "advisory"), _ob("c", ["new/**"], "info")])
    by = {o.id: o for o in union_policy(cand, base).obligations}
    assert set(by) == {"a", "b", "c"}                      # base-only 'b' preserved; candidate-only 'c' added
    assert set(by["a"].globs) == {"engine/**", "docs/**"}  # the dropped base glob is restored (no shrink)
    assert by["a"].severity == "blocking"                  # the lowered severity is restored (no weaken)
    assert by["b"].globs == ("scripts/**",)                # the removed base obligation is fully preserved


def test_load_effective_policy_falls_back_honestly_when_base_unreachable(tmp_path: Path) -> None:
    from assurance.git_state import discover_git_context
    ctx = discover_git_context(_repo_with_policy(tmp_path))
    policy, available = load_effective_policy(ctx, "no-such-ref-xyz")
    assert not available and policy.obligations  # candidate-only + honest flag, no crash


def test_a_weakened_candidate_policy_still_fires_the_base_obligation(tmp_path: Path) -> None:
    # THE POINT of PR2: a change that REMOVES a glob from the policy cannot escape firing it - the impact
    # engine judges against candidate UNION trusted-merge-base, so the removed coverage is restored.
    repo = _repo_with_policy(tmp_path, REAL_POLICY_TEXT)  # main has the real policy (contracts -> docs/contracts/**)
    _git(repo, "checkout", "-q", "-b", "feat")
    weakened = REAL_POLICY_TEXT.replace(', "docs/contracts/**"', "")  # drop that glob from 'contracts'
    (repo / "scripts" / "assurance" / "policy" / "obligations.json").write_text(weakened)
    _plant(repo, "docs/contracts/RunPlan.schema.json")   # a path only the REMOVED glob would catch
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "weaken the policy + touch a contracts doc")
    proc = run_cli(repo, "impact", "--base", "main")
    record = json.loads(proc.stdout)
    fired = {f["id"] for f in record["payload"]["fired_obligations"]}
    assert record["payload"]["base_policy_available"] is True
    assert "contracts" in fired  # restored by the union - a candidate-only policy would NOT have fired it


def _valid_ob(**over: object) -> dict:
    ob = {"id": "x", "globs": ["a/b.py"], "obligation": "o", "authority": "au",
          "severity": "blocking", "source": "s"}
    ob.update(over)
    return ob


def _policy(obligations: list[dict], **top: object) -> dict:
    doc = {"schema_version": 1, "description": "d",
           "severities": ["info", "advisory", "blocking"], "obligations": obligations}
    doc.update(top)
    return doc


# --------------------------------------------------------------------------- glob matcher


def test_glob_matches_literal_is_exact() -> None:
    assert glob_matches("engine/corpus_studio/cli.py", "engine/corpus_studio/cli.py")
    assert not glob_matches("engine/corpus_studio/cli.py", "engine/corpus_studio/cli_utils.py")
    assert not glob_matches("engine/corpus_studio/cli.py", "engine/corpus_studio/sub/cli.py")


def test_glob_matches_dir_star_star_is_boundary_correct() -> None:
    g = "docs/paper/**"
    assert glob_matches(g, "docs/paper")  # the dir itself
    assert glob_matches(g, "docs/paper/OUTLINE.md")  # a child
    assert glob_matches(g, "docs/paper/sec/1.md")  # a descendant
    assert not glob_matches(g, "docs/paper2/x.md")  # sibling sharing the prefix
    assert not glob_matches(g, "docs")  # the parent


def test_glob_matches_is_case_sensitive() -> None:
    assert not glob_matches("engine/corpus_studio/cli.py", "Engine/corpus_studio/cli.py")


# --------------------------------------------------------------------------- policy load (fail-closed)


@pytest.mark.parametrize(
    "doc, needle",
    [
        (_policy([_valid_ob()], extra_key=1), "unknown top-level"),
        (_policy([_valid_ob()], schema_version=2), "schema_version"),
        (_policy([]), "non-empty 'obligations'"),
        (_policy([_valid_ob(severity="critical")]), "invalid severity"),
        (_policy([_valid_ob(globs=[])]), "empty/non-list"),
        (_policy([_valid_ob(id="")]), "empty/non-string id"),
        (_policy([_valid_ob(id="d"), _valid_ob(id="d")]), "more than once"),
        (_policy([_valid_ob(obligation="")]), "empty/non-string"),
        (_policy([{"id": "x", "globs": ["a"]}]), "key mismatch"),
        (_policy([_valid_ob(globs=["/abs.py"])]), "repo-relative"),
        (_policy([_valid_ob(globs=["a/../b.py"])]), "'.' / '..'"),
        (_policy([_valid_ob(globs=["a\\b.py"])]), "backslash"),
        (_policy([_valid_ob(globs=["**"])]), "catch-all"),
        (_policy([_valid_ob(globs=["a/**/b.py"])]), "trailing '/**'"),
    ],
)
def test_parse_policy_fails_closed(doc: dict, needle: str) -> None:
    with pytest.raises(PolicyError, match=re.escape(needle)):
        parse_policy(doc)


def test_load_policy_fails_closed_on_missing_and_bad_json(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match="could not be read"):
        load_policy(tmp_path)  # no policy file at all
    reg = tmp_path / "scripts" / "assurance" / "policy" / "obligations.json"
    reg.parent.mkdir(parents=True)
    reg.write_text("{not json")
    with pytest.raises(PolicyError, match="not valid UTF-8 JSON"):
        load_policy(tmp_path)


def test_load_policy_happy_is_sorted_and_digest_stable(tmp_path: Path) -> None:
    reg = tmp_path / "scripts" / "assurance" / "policy" / "obligations.json"
    reg.parent.mkdir(parents=True)
    reg.write_text(REAL_POLICY_TEXT)
    p1 = load_policy(tmp_path)
    p2 = load_policy(tmp_path)
    assert [o.id for o in p1.obligations] == sorted(o.id for o in p1.obligations)
    assert p1.digest == p2.digest and p1.digest.startswith("sha256:")
    assert p1.obligation_count == len(p1.obligations) == 6


# --------------------------------------------------------------------------- match_obligations


def test_match_obligations_overlap_and_unmatched() -> None:
    obs = [_ob("a", ["x/**"]), _ob("b", ["x/y.py"]), _ob("c", ["z.py"])]
    fired, unmatched = match_obligations(["x/y.py", "w.txt"], obs)
    assert [f["id"] for f in fired] == ["a", "b"]  # x/y.py fires both; sorted by id
    assert unmatched == 1  # w.txt fired nothing
    a = next(f for f in fired if f["id"] == "a")
    assert a["triggers"] == [{"path": "x/y.py", "globs": ["x/**"]}]
    assert a["trigger_path_count"] == 1


def test_match_obligations_deletion_of_literal_still_fires() -> None:
    # A change set carries a deleted path as a plain path string; a literal glob still matches it.
    fired, _ = match_obligations(["z.py"], [_ob("c", ["z.py"])])
    assert [f["id"] for f in fired] == ["c"]


# --------------------------------------------------------------------------- impact assessment (integration)


def test_impact_clean_tree_fires_nothing(tmp_path: Path) -> None:
    repo = _repo_with_policy(tmp_path)
    record = build_impact_assessment(start_dir=repo, base_ref="HEAD")
    assert record["record_type"] == "impact_assessment"
    assert record["payload"]["obligation_count"] == 0
    assert record["payload"]["fired_obligations"] == []
    assert verify_record(record)


def test_impact_fires_worker_closure_on_planted_change(tmp_path: Path) -> None:
    repo = _repo_with_policy(tmp_path)
    _plant(repo, "engine/corpus_studio/platform/worker.py")
    record = build_impact_assessment(start_dir=repo, base_ref="HEAD")
    fired = {f["id"] for f in record["payload"]["fired_obligations"]}
    assert "worker-closure" in fired
    wc = next(f for f in record["payload"]["fired_obligations"] if f["id"] == "worker-closure")
    assert wc["severity"] == "blocking"
    assert wc["triggers"][0]["path"] == "engine/corpus_studio/platform/worker.py"


def test_impact_self_modify_covers_plugin_and_tests(tmp_path: Path) -> None:
    # D3: the plugin config + the assurance tests ARE the judge - editing them must fire
    # assurance-self-modify (was: zero signal). planner.py must fire worker-closure (D4).
    repo = _repo_with_policy(tmp_path)
    for rel in (".claude/rules/worker-closure.md", "engine/tests/test_assurance_obligations.py",
                "engine/tests/test_plugin_hooks.py"):
        _plant(repo, rel)
    _plant(repo, "engine/corpus_studio/platform/planner.py")
    fired = {f["id"] for f in build_impact_assessment(start_dir=repo, base_ref="HEAD")["payload"]["fired_obligations"]}
    assert "assurance-self-modify" in fired  # .claude/** + test_assurance_*.py + test_plugin_hooks.py
    assert "worker-closure" in fired  # planner.py is now a declared worker-reachable path


def test_impact_dir_glob_and_unmatched(tmp_path: Path) -> None:
    repo = _repo_with_policy(tmp_path)
    _plant(repo, "docs/contracts/NEW.md")  # fires contracts (docs/contracts/**)
    _plant(repo, "unrelated_top_level.txt")  # fires nothing
    record = build_impact_assessment(start_dir=repo, base_ref="HEAD")
    fired = {f["id"] for f in record["payload"]["fired_obligations"]}
    assert "contracts" in fired
    assert record["payload"]["unmatched_path_count"] == 1


def test_impact_is_deterministic_and_seals(tmp_path: Path) -> None:
    repo = _repo_with_policy(tmp_path)
    _plant(repo, "research/ieee-linux-training/AMEND.md")
    r1 = build_impact_assessment(start_dir=repo, base_ref="HEAD")
    r2 = build_impact_assessment(start_dir=repo, base_ref="HEAD")
    assert r1["record_digest"] == r2["record_digest"]
    assert r1["payload"]["applicability_key"] == r2["payload"]["applicability_key"]
    assert verify_record(r1)
    assert {f["id"] for f in r1["payload"]["fired_obligations"]} == {"sealed-research"}


# --------------------------------------------------------------------------- CLI


def test_cli_impact_exit_0_even_when_obligations_fire(tmp_path: Path) -> None:
    repo = _repo_with_policy(tmp_path)
    _plant(repo, "engine/corpus_studio/platform/worker.py")
    proc, record = impact(repo)
    assert proc.returncode == 0, proc.stderr  # observation-only: firing is NOT a failure
    assert record["payload"]["obligation_count"] == 1


def test_cli_impact_fails_closed_on_malformed_policy(tmp_path: Path) -> None:
    repo = _repo_with_policy(tmp_path, policy_text='{"schema_version": 1, "obligations": []}')
    proc = run_cli(repo, "impact", "--base", "HEAD")
    assert proc.returncode == 2
    assert "PolicyError" in proc.stderr and not proc.stdout.strip()


def test_cli_impact_rejects_unimplemented_scope(tmp_path: Path) -> None:
    repo = _repo_with_policy(tmp_path)
    assert run_cli(repo, "impact", "--scope", "index").returncode == 2  # argparse choices refuse it


# --------------------------------------------------------------------------- consistency (no drift)


def _rule_paths(text: str) -> list[str]:
    out: list[str] = []
    in_paths = False
    for line in text.splitlines():
        if line.strip() == "---" and in_paths:
            break
        if line.strip().startswith("paths:"):
            in_paths = True
            continue
        if in_paths:
            m = re.match(r'\s*-\s*"([^"]+)"', line)
            if m:
                out.append(m.group(1))
            elif line.strip() and not line.startswith(" "):
                break
    return out


def test_impact_policy_matches_rules_exactly() -> None:
    # Every .claude/rules/<id>.md must have an obligation with id == stem and globs == its paths:.
    policy = load_policy(REPO_ROOT)
    by_id = {o.id: o for o in policy.obligations}
    rules_dir = REPO_ROOT / ".claude" / "rules"
    rule_stems = set()
    for rule in sorted(rules_dir.glob("*.md")):
        stem = rule.stem
        rule_stems.add(stem)
        paths = _rule_paths(rule.read_text("utf-8"))
        assert paths, f"rule {stem} has no paths: front-matter to key off"
        assert stem in by_id, f"rule {stem} has no matching obligation in the policy"
        assert set(by_id[stem].globs) == set(paths), (
            f"{stem}: policy globs {sorted(by_id[stem].globs)} != rule paths {sorted(paths)}"
        )
    # And every shipped obligation is rule-anchored (this slice ships Tier A only; Tier B lands with
    # its own rule file). This assertion is the guard that keeps the shipped policy honest.
    for o in policy.obligations:
        assert o.id in rule_stems, f"obligation {o.id!r} has no .claude/rules/{o.id}.md anchor"


def _load_hook() -> ModuleType:
    hook = REPO_ROOT / ".claude" / "hooks" / "advisory_classify.py"
    spec = importlib.util.spec_from_file_location("advisory_classify_under_test", hook)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _representative_path(glob: str) -> str:
    """A concrete example path that lives under a policy glob (``**`` / ``*`` -> a literal segment)."""
    return glob.replace("**/", "x/").replace("**", "x").replace("*", "x")


def test_impact_policy_covers_advisory_hook_fragments() -> None:
    # SOUNDNESS (leg 1): every write-time advisory-hook fragment is covered by a policy glob - the hook
    # never nudges on something the policy does not flag.
    module = _load_hook()
    all_globs = [g for o in load_policy(REPO_ROOT).obligations for g in o.globs]
    for fragment, _reminder in module._SENSITIVE:
        assert any(fragment in glob for glob in all_globs), (
            f"advisory fragment {fragment!r} is not covered by any policy glob"
        )


def test_advisory_hook_covers_every_policy_glob() -> None:
    # COMPLETENESS (leg 2 - the third leg of the seam): every path the policy flags ALSO produces a
    # write-time nudge, exercised through the hook's REAL classifier (reminder_for), not a proxy. This
    # is the assertion that catches a policy glob added without a matching hook fragment (the gap R3
    # closed for .claude/**, the assurance tests, docs/contracts/**, and .github/workflows/assurance.yml).
    module = _load_hook()
    for obligation in load_policy(REPO_ROOT).obligations:
        for glob in obligation.globs:
            example = _representative_path(glob)
            assert module.reminder_for(example) is not None, (
                f"obligation {obligation.id!r} glob {glob!r} (e.g. {example!r}) has no advisory-hook "
                "coverage; add a fragment to advisory_classify._SENSITIVE or narrow the policy glob"
            )
