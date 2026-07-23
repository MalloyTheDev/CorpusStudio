"""Docs-freshness: keep docs from going stale when code changes (a loop sub-check).

The doc-trust sensor (``cs_assure doclint``) detects stale doc PROSE. This complements it with the other
half - a CODE<->DOC COUPLING check: when a change touches a coupled code area but does NOT touch its
documentation, the doc is drifting even if its current prose reads clean. Wired into the loop's OBSERVE
step this becomes enforcement: the loop will not advance to INTEGRATE while a coupled doc is stale, and
each gap can be turned into a correction task (update the doc) exactly like a review finding.

Couplings are DATA (so this module cannot itself go stale on a hardcoded rule); a small verified default
set ships here and callers may pass their own. stdlib-only; pure.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import PurePosixPath

from loop.controller import Observation


@dataclass(frozen=True)
class DocCoupling:
    """When a changed path matches any ``code_globs``, at least one ``doc_paths`` entry should also have
    changed - otherwise the docs are drifting from the code."""

    name: str
    code_globs: list[str]
    doc_paths: list[str]
    reason: str


@dataclass(frozen=True)
class DocGap:
    """A coupling whose code changed but whose docs did not."""

    coupling: str
    changed_code: list[str]
    expected_docs: list[str]
    reason: str


# Verified default couplings (both sides exist in the tree). Extend as new code<->doc pairs appear.
DEFAULT_COUPLINGS: tuple[DocCoupling, ...] = (
    DocCoupling(
        name="platform-contracts",
        code_globs=["engine/corpus_studio/platform/contracts.py", "engine/corpus_studio/platform/enums.py"],
        doc_paths=["docs/contracts/"],
        reason="a contract change needs regenerated schemas (docs/contracts) + the count assertions",
    ),
    DocCoupling(
        name="cli",
        code_globs=["engine/corpus_studio/cli.py"],
        doc_paths=["docs/CLI_REFERENCE.md"],
        reason="a CLI change needs docs/CLI_REFERENCE.md updated",
    ),
    DocCoupling(
        name="autonomous-loop",
        code_globs=["scripts/loop/"],
        doc_paths=["docs/AUTONOMOUS_LOOP.md"],
        reason="a loop change needs docs/AUTONOMOUS_LOOP.md kept current",
    ),
)


def _covers(pattern: str, path: str) -> bool:
    pat = PurePosixPath(pattern.rstrip("/"))
    p = PurePosixPath(path)
    return p == pat or pat in p.parents


def _matches(path: str, pattern: str) -> bool:
    norm = path.replace("\\", "/")
    if "*" in pattern or "?" in pattern:
        return fnmatch.fnmatch(norm, pattern)
    return _covers(pattern, norm)


def stale_docs(changed_paths: list[str],
               couplings: tuple[DocCoupling, ...] = DEFAULT_COUPLINGS) -> list[DocGap]:
    """The couplings whose code changed but whose docs did NOT (pure). Empty = docs are in sync."""
    changed = [str(p) for p in changed_paths]
    gaps: list[DocGap] = []
    for coupling in couplings:
        touched_code = [p for p in changed if any(_matches(p, g) for g in coupling.code_globs)]
        if not touched_code:
            continue
        touched_doc = any(_matches(p, d) for p in changed for d in coupling.doc_paths)
        if not touched_doc:
            gaps.append(DocGap(coupling.name, touched_code, list(coupling.doc_paths), coupling.reason))
    return gaps


def docs_observation(gaps: list[DocGap]) -> tuple[Observation, str]:
    """No gaps -> SUCCESS; any gap -> CONTRACT_DRIFT (the docs are out of sync with the changed code)."""
    if not gaps:
        return Observation.SUCCESS, "docs in sync with the changed code"
    names = sorted(g.coupling for g in gaps)
    return Observation.CONTRACT_DRIFT, f"stale doc coupling(s): {names} - code changed but docs did not"


def doc_correction_tasks(gaps: list[DocGap]) -> list[dict[str, object]]:
    """Turn each gap into a correction task scoped to the doc(s) that must be updated - feedable into the
    task graph exactly like a review finding."""
    return [
        {
            "id": f"docs-{gap.coupling}",
            "description": f"Update docs: {gap.reason}",
            "owner": "self",
            "allowed_paths": list(gap.expected_docs),
            "depends_on": [],
            "success_criteria": [gap.reason],
            "status": "PENDING",
        }
        for gap in gaps
    ]
