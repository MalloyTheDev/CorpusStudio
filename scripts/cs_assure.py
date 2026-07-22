#!/usr/bin/env python3
"""cs_assure - CorpusStudio Assurance Loop CLI (Phase 1: change-set kernel only).

Usage::

    python3 scripts/cs_assure.py changeset --scope workspace --base main --format json

Subcommands:
    changeset  snapshot the selected Git state and print a deterministic, rename-free,
               content-addressed ChangeSetRecord vs merge-base(HEAD, --base) (Phase 1 kernel).
    doclint    lint the documentation context plane for staleness against the context-source
               registry (detect-only; edits nothing) - the doc-trust sub-loop's sensor.
    impact     map the change set onto the obligations policy and print a sealed ImpactAssessment
               ("given what changed, what must I now do?"). OBSERVATION-ONLY: it reports fired
               obligations; it never enforces, gates, or blocks (no --strict, no exit 1).
    verify     RUN the declared workspace gate (ruff/mypy/pytest) and print a sealed
               WorkspaceVerification record binding the real exit codes + the fired obligations to
               the change set. Green == WORKSPACE_GATE only - never fit / commit / PR / release /
               sealed / CI. Exit 1 if the gate is red.

Exit-code contract:
    0  success (an empty change set on a clean tree, a doclint run, an impact assessment - even one
       that fires obligations - or a GREEN verify gate is 0),
    1  a not-clean result of a checking command: doclint --strict found staleness, OR verify's gate
       is red (a step's real exit code did not match its expected code). impact never uses exit 1.
    2  a fail-closed refusal (not a repo, missing base ref, no merge base, shallow-history
       limitation, unsupported special file, non-UTF-8 path, tree moved mid-collection, a malformed
       context-source registry / obligations policy / gate spec, or a gate step that could not be
       launched at all).

Verify RUNS external gate commands (declared argv, no shell); every other subcommand is read-only.
cs_assure itself never mutates the repository's committed state - the object store, refs, the
committed tree, or the working tree (a read may refresh the content-neutral index stat-cache; see
``assurance/git_state.py``). Later phases add gate / evidence subcommands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the sibling ``assurance`` package importable whether run as a script, via -m, or from a
# subprocess. When run as ``python scripts/cs_assure.py`` this dir is already sys.path[0]; the
# explicit insert makes the other invocation paths robust too.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from assurance import KERNEL_VERSION  # noqa: E402 - must follow the sys.path bootstrap above.
from assurance.canonical_json import CanonicalJsonError  # noqa: E402
from assurance.git_state import AssuranceError  # noqa: E402
from assurance.records import build_change_set_record  # noqa: E402

EXIT_OK = 0
EXIT_LINT_FINDINGS = 1  # doclint --strict: staleness present
EXIT_GATE_RED = 1  # verify: the workspace gate is red (same "not-clean" rung as EXIT_LINT_FINDINGS)
EXIT_FAIL_CLOSED = 2


def _cmd_changeset(args: argparse.Namespace) -> int:
    start_dir = Path(args.start_dir) if args.start_dir else Path.cwd()
    record = build_change_set_record(start_dir=start_dir, scope=args.scope, base_ref=args.base)
    # Pretty, key-sorted JSON for human/tooling readability. The embedded record_digest is
    # computed over the CANONICAL (compact) form, so re-verification always re-canonicalizes the
    # parsed object rather than re-hashing these display bytes.
    sys.stdout.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return EXIT_OK


def _cmd_doclint(args: argparse.Namespace) -> int:
    from assurance.doc_lint import load_registry, lint_repo  # noqa: PLC0415
    from assurance.git_state import discover_git_context  # noqa: PLC0415

    start_dir = Path(args.start_dir) if args.start_dir else Path.cwd()
    ctx = discover_git_context(start_dir)
    sources = load_registry(ctx.root)
    findings = lint_repo(ctx.root, sources)
    if args.format == "json":
        payload = {
            "tool": "cs_assure doclint",
            "registry_source_count": len(sources),
            "finding_count": len(findings),
            "findings": [f.to_record() for f in findings],
        }
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _print_doclint_summary(len(sources), findings)
    # Detect-only by default (exit 0). --strict turns it into a gate (exit 1 on any finding).
    if args.strict and findings:
        return EXIT_LINT_FINDINGS
    return EXIT_OK


def _cmd_impact(args: argparse.Namespace) -> int:
    from assurance.obligations import build_impact_assessment  # noqa: PLC0415

    start_dir = Path(args.start_dir) if args.start_dir else Path.cwd()
    record = build_impact_assessment(
        start_dir=start_dir, scope=args.scope, base_ref=args.base, policy_relpath=args.policy
    )
    # Observation-only: exit 0 even when obligations fire. Pretty display bytes; the sealed
    # record_digest is over the canonical (compact) form, so re-verification re-canonicalizes.
    sys.stdout.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return EXIT_OK


def _cmd_verify(args: argparse.Namespace) -> int:
    from assurance.verification import build_verification_record  # noqa: PLC0415

    start_dir = Path(args.start_dir) if args.start_dir else Path.cwd()
    record = build_verification_record(
        start_dir=start_dir,
        scope=args.scope,
        base_ref=args.base,
        gate_relpath=args.gate,
        policy_relpath=args.policy,
    )
    # The record is emitted whether the gate is green or red (it is evidence either way). A red gate
    # is a not-clean result (exit 1), NOT a fail-closed refusal (exit 2, which means the gate could
    # not be evaluated). Green is WORKSPACE_GATE only - the record never claims more.
    sys.stdout.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return EXIT_OK if record["payload"]["gate_passed"] else EXIT_GATE_RED


def _print_doclint_summary(source_count: int, findings: list) -> None:
    out = sys.stdout
    out.write(f"cs_assure doclint - {source_count} registered sources, {len(findings)} findings\n")
    if not findings:
        out.write("  no staleness findings\n")
        return
    by_rule: dict[str, int] = {}
    by_file: dict[str, int] = {}
    for finding in findings:
        by_rule[finding.rule] = by_rule.get(finding.rule, 0) + 1
        by_file[finding.path] = by_file.get(finding.path, 0) + 1
    out.write("  by rule:\n")
    for rule in sorted(by_rule):
        out.write(f"    {rule}: {by_rule[rule]}\n")
    out.write("  by file:\n")
    for path in sorted(by_file):
        out.write(f"    {path}: {by_file[path]}\n")
    out.write("  findings (path:line [rule] excerpt):\n")
    for finding in findings:
        out.write(f"    {finding.path}:{finding.line} [{finding.rule}] {finding.excerpt}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cs_assure", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"cs_assure {KERNEL_VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    changeset = subparsers.add_parser(
        "changeset", help="compute a sealed change-set record for a selected Git state"
    )
    changeset.add_argument(
        "--scope",
        default="workspace",
        choices=["workspace"],
        help="which Git state to snapshot (Phase 1 implements the workspace scope only)",
    )
    changeset.add_argument(
        "--base",
        default="main",
        help="base ref; the change set is computed against merge-base(HEAD, base)",
    )
    changeset.add_argument(
        "--format",
        default="json",
        choices=["json"],
        help="output format (Phase 1: json)",
    )
    changeset.add_argument(
        "--start-dir",
        default=None,
        help="directory to resolve the repository from (default: current directory)",
    )
    changeset.set_defaults(func=_cmd_changeset)

    doclint = subparsers.add_parser(
        "doclint",
        help="lint the documentation context plane for staleness (detect-only; edits nothing)",
    )
    doclint.add_argument(
        "--format",
        default="summary",
        choices=["summary", "json"],
        help="summary = human counts + findings; json = full deterministic list",
    )
    doclint.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if any findings are reported (for CI gating); default is detect-only exit 0",
    )
    doclint.add_argument(
        "--start-dir",
        default=None,
        help="directory to resolve the repository from (default: current directory)",
    )
    doclint.set_defaults(func=_cmd_doclint)

    impact = subparsers.add_parser(
        "impact",
        help="map the change set onto the obligations policy (observation-only; never gates)",
    )
    impact.add_argument(
        "--scope",
        default="workspace",
        choices=["workspace"],
        help="which Git state to assess (inherits the kernel's workspace-only scope)",
    )
    impact.add_argument(
        "--base",
        default="main",
        help="base ref; the change set is computed against merge-base(HEAD, base)",
    )
    impact.add_argument(
        "--policy",
        default="scripts/assurance/policy/obligations.json",
        help="repo-relative path to the obligations policy bundle (read from the working tree)",
    )
    impact.add_argument(
        "--format",
        default="json",
        choices=["json"],
        help="output format (json)",
    )
    impact.add_argument(
        "--start-dir",
        default=None,
        help="directory to resolve the repository from (default: current directory)",
    )
    impact.set_defaults(func=_cmd_impact)

    verify = subparsers.add_parser(
        "verify",
        help="run the declared workspace gate and seal a WorkspaceVerification (exit 1 if red)",
    )
    verify.add_argument(
        "--scope",
        default="workspace",
        choices=["workspace"],
        help="which Git state to bind the verification to (workspace only)",
    )
    verify.add_argument(
        "--base",
        default="main",
        help="base ref; the bound change set is computed against merge-base(HEAD, base)",
    )
    verify.add_argument(
        "--gate",
        default="scripts/assurance/policy/gate.json",
        help="repo-relative path to the gate spec (no-shell argv steps) to run",
    )
    verify.add_argument(
        "--policy",
        default="scripts/assurance/policy/obligations.json",
        help="repo-relative obligations policy, to list what fired on the same change set",
    )
    verify.add_argument(
        "--format",
        default="json",
        choices=["json"],
        help="output format (json)",
    )
    verify.add_argument(
        "--start-dir",
        default=None,
        help="directory to resolve the repository from (default: current directory)",
    )
    verify.set_defaults(func=_cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (AssuranceError, CanonicalJsonError) as exc:
        # Fail closed: a structured refusal on stderr, exit 2, and NO partial record on stdout.
        sys.stderr.write(f"cs_assure: {type(exc).__name__}: {exc}\n")
        return EXIT_FAIL_CLOSED


if __name__ == "__main__":
    raise SystemExit(main())
