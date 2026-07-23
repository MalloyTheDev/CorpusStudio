"""Project-progression intake for the assurance loop (Phase-4 intake): the ``cs_assure status`` snapshot.

Composes the already-built sensors into ONE sealed ``project_status`` record so the loop can answer
"where are we": the current change set (summary), the obligations it fires, doc staleness, the current
branch + recent commits, and a fail-SOFT open-issue summary - plus fixed pointers to the authority docs
and the seven product areas. It fires each sensor ONCE over a single change set (it calls the loaders +
matcher directly, not build_impact_assessment/build_verification_record, which would each rebuild the
change set).

Division of labour (the seam): cs_assure gathers FACTS; the ``corpus-progress`` skill + Claude make the
JUDGMENT. So the tool lists authority-doc POINTERS (it never opens/parses/summarizes their content), and
it emits NO "what's next" / per-area verdict - that would be a judgment masquerading as a fact.

Honesty: the record MIXES deterministic facts (change-set / impact / doclint - pure functions of the
tree + policy bytes) with MEASUREMENTS (branch, recent commits, issues - time-varying, and issues are
external). It is therefore an EVIDENCE snapshot, not a pure applicability function: it carries no
top-level ``applicability_key`` and no wall-clock (so two snapshots at the same HEAD/tree/remote seal
byte-identically, and ``record_digest`` moves iff an observed fact moved). The only applicability key is
the narrowly-scoped ``impact.impact_applicability_key``, honestly re-derivable from its deterministic
inputs. A gh failure never changes the exit code; git/kernel/registry/policy errors stay fail-closed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from assurance import KERNEL_VERSION
from assurance.canonical_json import sha256_digest
from assurance.doc_lint import REGISTRY_RELPATH, lint_repo, load_registry
from assurance.git_state import current_branch, discover_git_context, recent_commits
from assurance.github_issues import ISSUES_SOURCE, gather_issues
from assurance.obligations import DEFAULT_POLICY_RELPATH, load_effective_policy, match_obligations
from assurance.records import RECORD_SCHEMA_VERSION, build_change_set_record, seal_record

STATUS_RECORD_TYPE = "project_status"
STATUS_SCHEMA_VERSION = 1
DEFAULT_ISSUE_LIMIT = 10
DEFAULT_COMMIT_LIMIT = 10

# The canonical seven co-equal product areas (docs/PRODUCT_AREAS.md is the authority). Echoed verbatim
# so the snapshot names the frame; the tool assigns NO per-area status (that is the skill's judgment).
PRODUCT_AREAS = (
    "Data Studio",
    "Training Studio",
    "Evaluation Studio",
    "Behavior Lab",
    "Model & Release Studio",
    "Environment & Hardware",
    "Evidence & Experiments",
)

# Fixed, ordered allowlist of authority docs to surface as POINTERS. ``why`` is a fixed role label, NOT
# parsed from the doc's content; the tool never opens these files.
AUTHORITY_POINTERS = (
    ("docs/CURRENT_STATE.md", "feature-state authority; wins on conflict"),
    ("docs/ROADMAP.md", "forward-plan milestones + next"),
    ("docs/IMPLEMENTATION_PLAN.md", "forward-plan execution frontier"),
    ("docs/PRODUCT_AREAS.md", "canonical seven-area identity map"),
    ("docs/PRODUCT_VS_RESEARCH.md", "standard/verified/sealed-research boundary"),
    ("HANDOFF.md", "session state + volatile run status"),
    ("docs/HOST_STATE.md", "host/GPU/env facts; read before anything hardware-adjacent"),
    ("CLAUDE.md", "always-loaded router contract"),
    ("AGENTS.md", "always-loaded honesty invariants"),
)


def _summarize_change_set(cs_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "changed_path_count": cs_payload["changed_path_count"],
        "fingerprint": cs_payload["fingerprint"],
    }


def _summarize_impact(
    fired: list[dict[str, Any]], unmatched_count: int, applicability_key: str, base_policy_available: bool
) -> dict[str, Any]:
    by_severity: dict[str, int] = {}
    for obligation in fired:
        by_severity[obligation["severity"]] = by_severity.get(obligation["severity"], 0) + 1
    return {
        "obligation_count": len(fired),
        "unmatched_path_count": unmatched_count,
        "impact_applicability_key": applicability_key,
        "base_policy_available": base_policy_available,
        "fired_obligations": [{"id": o["id"], "severity": o["severity"]} for o in fired],  # id-sorted
        "by_severity": by_severity,
    }


def _summarize_doclint(sources: list[Any], findings: list[Any]) -> dict[str, Any]:
    by_rule: dict[str, int] = {}
    for finding in findings:
        by_rule[finding.rule] = by_rule.get(finding.rule, 0) + 1
    return {
        "registry_source_count": len(sources),
        "finding_count": len(findings),
        "by_rule": by_rule,
    }


def _authority_pointers(sources: list[Any]) -> list[dict[str, Any]]:
    """Emit the fixed authority allowlist, enriched from the loaded registry (or null if unregistered)."""
    by_path = {source.path: source for source in sources}
    pointers: list[dict[str, Any]] = []
    for path, why in AUTHORITY_POINTERS:
        source = by_path.get(path)
        pointers.append(
            {
                "path": path,
                "authority": source.authority if source else None,
                "mode": source.mode if source else None,
                "superseded_by": source.superseded_by if source else None,
                "why": why,
            }
        )
    return pointers


def build_status_record(
    *,
    start_dir: Path,
    scope: str = "workspace",
    base_ref: str = "main",
    policy_relpath: str = DEFAULT_POLICY_RELPATH,
    limit_issues: int = DEFAULT_ISSUE_LIMIT,
    limit_commits: int = DEFAULT_COMMIT_LIMIT,
) -> dict[str, Any]:
    """Gather + seal a ``project_status`` snapshot. Exit-2 fail-closed on git/kernel errors; gh is soft."""
    ctx = discover_git_context(start_dir)
    change_set = build_change_set_record(start_dir=start_dir, scope=scope, base_ref=base_ref)
    cs_payload = change_set["payload"]
    cs_provenance = change_set["provenance"]
    changed_paths = [cp["path"] for cp in cs_payload["changed_paths"]]

    # Effective policy = candidate UNION trusted-base (a candidate cannot weaken the policy it is judged
    # against); consistent with the impact command so the two records' applicability keys still match.
    policy, base_policy_available = load_effective_policy(ctx, base_ref, policy_relpath)
    fired, unmatched_count = match_obligations(changed_paths, list(policy.obligations))
    impact_key = sha256_digest(
        {"change_set_fingerprint": cs_payload["fingerprint"], "policy_digest": policy.digest}
    )

    sources = load_registry(ctx.root)
    findings = lint_repo(ctx.root, sources)

    issues = gather_issues(ctx.root, limit_recent=limit_issues)
    commits = recent_commits(ctx, limit_commits)

    payload = {
        "scope": scope,
        "base_oid": cs_payload["base_oid"],
        "change_set": _summarize_change_set(cs_payload),
        "impact": _summarize_impact(fired, unmatched_count, impact_key, base_policy_available),
        "doclint": _summarize_doclint(sources, findings),
        "branch": {
            "current_branch": current_branch(ctx),
            "head_oid": ctx.head_oid,
            "is_shallow": ctx.is_shallow,
        },
        "recent_commits": commits,
        "recent_commit_count": len(commits),
        "recent_commit_limit": limit_commits,
        "issues": issues,
        "product_areas": list(PRODUCT_AREAS),
        "authority_pointers": _authority_pointers(sources),
    }
    provenance = {
        "tool": "cs_assure",
        "tool_version": KERNEL_VERSION,
        "record_kernel_schema_version": RECORD_SCHEMA_VERSION,
        "subcommand": "status",
        "base_ref": cs_provenance["base_ref"],
        "base_oid": cs_provenance["base_oid"],
        "head_oid": cs_provenance["head_oid"],
        "is_shallow": cs_provenance["is_shallow"],
        "change_set_digest": change_set["record_digest"],
        "policy_path": policy.relpath,
        "policy_schema_version": policy.schema_version,
        "policy_digest": policy.digest,
        "policy_obligation_count": policy.obligation_count,
        "registry_relpath": REGISTRY_RELPATH,
        "registry_source_count": len(sources),
        "issues_source": ISSUES_SOURCE,
        "issues_available": issues["available"],
        "limit_issues": limit_issues,
        "limit_commits": limit_commits,
        "is_measurement": True,
    }
    return seal_record(STATUS_RECORD_TYPE, STATUS_SCHEMA_VERSION, payload, provenance)
