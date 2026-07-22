#!/usr/bin/env python3
"""cs_assure - CorpusStudio Assurance Loop CLI (Phase 1: change-set kernel only).

Usage::

    python3 scripts/cs_assure.py changeset --scope workspace --base main --format json

Phase 1 exposes exactly one subcommand, ``changeset``: it snapshots the selected Git state,
computes a deterministic, rename-free, content-addressed change set against
``merge-base(HEAD, --base)``, and prints a sealed ChangeSetRecord.

Exit-code contract:
    0  a record was produced (an empty change set on a clean tree is still success),
    2  a fail-closed refusal (not a repo, missing base ref, no merge base, shallow-history
       limitation, unsupported special file, non-UTF-8 path, or a tree that moved mid-collection).

Later phases add impact / verification / gate / evidence subcommands; this bootstrap kernel
intentionally does not, and it never mutates the repository.
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
EXIT_FAIL_CLOSED = 2


def _cmd_changeset(args: argparse.Namespace) -> int:
    start_dir = Path(args.start_dir) if args.start_dir else Path.cwd()
    record = build_change_set_record(start_dir=start_dir, scope=args.scope, base_ref=args.base)
    # Pretty, key-sorted JSON for human/tooling readability. The embedded record_digest is
    # computed over the CANONICAL (compact) form, so re-verification always re-canonicalizes the
    # parsed object rather than re-hashing these display bytes.
    sys.stdout.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return EXIT_OK


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
