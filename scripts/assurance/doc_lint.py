"""Deterministic documentation-staleness lint for the CorpusStudio context plane (Phase 5, slice 1).

DETECT-ONLY. This reads the docs named in the context-source registry
(``scripts/assurance/policy/context_sources.json``) and REPORTS staleness; it edits no prose. It is
the engine of the loop's doc-trust sub-loop and the machine-policy layer's context sensor - it turns
the 2026-07-22 study's stale-doc findings into a re-runnable check so drift stays caught instead of
needing a manual audit each time. Every prose-fix slice after this one becomes "make the lint pass".

The rules are MODE-AWARE via the registry, so they do not fight a doc's legitimate purpose:
  * a WPF / Avalonia mention is flagged in a CURRENT/MIXED (or stable-guidance) doc, but NOT in a
    HISTORICAL / SUPERSEDED / FROZEN_EVIDENCE doc, and NOT on a line that already carries a
    historical marker (removed / former / #545 / ...);
  * an absolute host path (``/mnt/training-nvme``, ``RTX 5070``, ``engine/.venv``, ...) is flagged
    only in stable, host-portable guidance - it is CORRECT in ``docs/HOST_STATE.md``;
  * a pinned wheel/commit/run identity is flagged only in stable guidance - it is the whole point of
    a ``FROZEN_EVIDENCE`` research record and belongs in the ``VOLATILE_CURRENT`` handoff.

This registry MIRRORS the doc authorities; the lint never rewrites a doc and never overrides an
authority (``CURRENT_STATE.md`` still wins on feature state).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from assurance.git_state import AssuranceError

REGISTRY_RELPATH = "scripts/assurance/policy/context_sources.json"

VALID_MODES = (
    "CURRENT",
    "VOLATILE_CURRENT",
    "MIXED_CURRENT_AND_HISTORY",
    "HISTORICAL",
    "SUPERSEDED",
    "FROZEN_EVIDENCE",
)
VALID_AUTHORITIES = ("canonical", "advisory", "derived", "evidence")

# Modes whose prose legitimately discusses removed/historical detail or pins identities.
_HISTORICAL_MODES = frozenset({"HISTORICAL", "SUPERSEDED", "FROZEN_EVIDENCE"})
# Modes that should read as clean-current (subject to the removed-UI rule).
_CURRENTISH_MODES = frozenset({"CURRENT", "MIXED_CURRENT_AND_HISTORY"})

# A per-line marker that a removed-UI / historical mention is intentional. Bare "was"/"were" are
# deliberately NOT markers: they are too common, and a genuine present-tense claim adjacent to an
# unrelated "was"/"were" (within the +/-1-line window) would be silently suppressed. A real
# historical passage names the removal (removed / former / decommissioned / #545 / ...).
_HISTORICAL_MARKER = re.compile(
    r"remov|former|legacy|historic|decommission|retire|supersed|no longer|replaced|prototype"
    r"|deleted|#545|#546",
    re.IGNORECASE,
)
# WPF and Avalonia are the REMOVED desktop frameworks (#545). Nocturne is deliberately excluded:
# it is the framework-agnostic design-token system that carries forward into the Tauri/React client.
_UI_PATTERN = re.compile(r"\b(WPF|Avalonia)\b", re.IGNORECASE)
_ABS_PATH_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"/mnt/training-nvme"), "absolute checkout/host path"),
    (re.compile(r"/mnt/windows-[cf]"), "historical Windows mount path"),
    (re.compile(r"\bRTX\s*5070\b"), "specific GPU model"),
    (re.compile(r"\bUbuntu\s*24\.04\b"), "specific OS version"),
    (re.compile(r"engine/\.venv"), "specific venv path"),
    (re.compile(r"\b(?:CPython|Python)\s*3\.\d+\.\d+\b"), "exact interpreter patch version"),
)
# High-precision volatile-identity line signature (avoids flagging ordinary prose).
_VOLATILE_LINE = re.compile(
    r"wheel\s+[0-9a-f]{7,}"
    r"|sha256[:=\s]+[0-9a-f]{8,}"
    r"|source[_ ]commit"
    r"|\b(?:source|commit|floor|ancestor|built from)\s+[0-9a-f]{7,40}\b"
    r"|merge\s+(?:commit\s+)?[0-9a-f]{7,}"
    r"|\brun-[0-9a-f]{4,}"
    r"|effective[- ]?matrix\s+1\.\d+\.\d+"
    r"|\bmatrix\s+1\.\d+\.\d+"
    r"|RESERVED_IDENTITIES\.v\d+"
    r"|amendment\s+000\d"
    r"|\bPR\s*#\d{3,}"
    r"|\b\d\.\d{2}\s*(?:->|→)\s*\d\.\d{2}\b",
    re.IGNORECASE,
)
_GITHUB_SETTING = re.compile(
    r"main is protected|branch protection|branch-protection|required status check|required check"
    r"|admin[- ]merge|standing authorization|merge queue|merge-queue",
    re.IGNORECASE,
)
# The stated root-contract count in prose. The actual count is DERIVED from the committed schema
# index (count_root_contracts) - never hardcoded here - so this rule tracks the real number as it
# changes instead of going stale itself. The digits must DIRECTLY quantify "root contracts" (adjacent,
# not merely within N chars) so a nearby date ("2026-06-28 ... root contracts") or an unrelated count
# ("28 enums and 31 root contracts") is not mis-captured as the contract count (false-positive drift).
_ROOT_CONTRACT_COUNT = re.compile(r"\b(\d{1,3})\s+root contracts\b", re.IGNORECASE)

_MAX_EXCERPT = 160
# A registered doc is guidance prose/JSON; the largest today is ~44 KiB. Bound the read so a registry
# entry that (by accident or malice) points at a huge or binary file cannot exhaust memory. A file
# over the bound is surfaced as drift, never silently truncated-and-linted (that would skip content).
_MAX_DOC_BYTES = 4 * 1024 * 1024


def _sanitize_excerpt(text: str) -> str:
    """Replace C0/C1 control characters (NUL, ANSI/terminal-escape and other non-printing bytes) with
    U+FFFD, keeping tab. The excerpt is a display of UNTRUSTED doc content; a hostile registered line
    must not be able to inject terminal control sequences into the human-readable report."""
    return "".join(
        ch if ch == "\t" or (ch >= "\x20" and not ("\x7f" <= ch <= "\x9f")) else "\ufffd"
        for ch in text
    )


class DocLintError(AssuranceError):
    """The context-source registry is malformed or missing (fail-closed, CLI exit 2)."""


@dataclass(frozen=True)
class DocSource:
    """One registered documentation source and its authority classification."""

    path: str
    mode: str
    authority: str
    always_loaded: bool
    stable_guidance: bool
    superseded_by: str | None
    note: str


@dataclass(frozen=True)
class Finding:
    """One deterministic staleness finding (advisory; the lint never edits the doc)."""

    rule: str
    severity: str  # "low" | "med" | "high"
    classification: str
    path: str
    line: int
    excerpt: str
    superseding_authority: str
    message: str

    def to_record(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "classification": self.classification,
            "path": self.path,
            "line": self.line,
            "excerpt": self.excerpt,
            "superseding_authority": self.superseding_authority,
            "message": self.message,
        }


def parse_registry(raw: dict[str, Any]) -> list[DocSource]:
    """Validate the registry document and return its sources (fails closed on any bad entry)."""
    if not isinstance(raw, dict):
        raise DocLintError("context-source registry root is not an object")
    entries = raw.get("sources")
    if not isinstance(entries, list) or not entries:
        raise DocLintError("context-source registry has no 'sources' list")
    sources: list[DocSource] = []
    seen_paths: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise DocLintError(f"source[{index}] is not an object")
        path = entry.get("path")
        mode = entry.get("mode")
        authority = entry.get("authority")
        if not isinstance(path, str) or not path:
            raise DocLintError(f"source[{index}] has no 'path'")
        # A registered path must be repo-relative and stay inside the tree: an absolute path, a
        # ".." traversal, or a backslash would make the sensor read (and vouch for) files outside
        # the repo. Fail closed rather than silently scan out-of-tree content.
        if PurePosixPath(path).is_absolute() or "\\" in path or ".." in PurePosixPath(path).parts:
            raise DocLintError(f"source {path!r} must be a repo-relative path (no absolute / '..' / '\\')")
        if path in seen_paths:
            raise DocLintError(f"source {path!r} is registered more than once (duplicate entry)")
        seen_paths.add(path)
        if mode not in VALID_MODES:
            raise DocLintError(f"source {path!r}: invalid mode {mode!r} (allowed: {VALID_MODES})")
        if authority not in VALID_AUTHORITIES:
            raise DocLintError(
                f"source {path!r}: invalid authority {authority!r} (allowed: {VALID_AUTHORITIES})"
            )
        sources.append(
            DocSource(
                path=path,
                mode=mode,
                authority=authority,
                always_loaded=bool(entry.get("always_loaded", False)),
                stable_guidance=bool(entry.get("stable_guidance", False)),
                superseded_by=entry.get("superseded_by"),
                note=str(entry.get("note", "")),
            )
        )
    return sources


def load_registry(repo_root: Path) -> list[DocSource]:
    """Read and validate the on-disk registry at ``REGISTRY_RELPATH`` under ``repo_root``."""
    registry_path = repo_root / REGISTRY_RELPATH
    try:
        raw = json.loads(registry_path.read_text("utf-8"))
    except FileNotFoundError as exc:
        raise DocLintError(f"context-source registry not found at {REGISTRY_RELPATH}") from exc
    except json.JSONDecodeError as exc:
        raise DocLintError(f"context-source registry is not valid JSON: {exc}") from exc
    except UnicodeDecodeError as exc:
        # Invalid UTF-8 bytes: a ValueError but NOT a JSONDecodeError - catch it explicitly (as the
        # sibling loaders do) so it fails CLOSED as a typed DocLintError, not an untyped leak.
        raise DocLintError(f"context-source registry is not valid UTF-8 ({REGISTRY_RELPATH}): {exc}") from exc
    except RecursionError as exc:
        # Deeply-nested JSON: NOT a JSONDecodeError/ValueError - fail CLOSED, not exit-1 traceback.
        raise DocLintError(f"context-source registry is nested too deeply to parse: {exc}") from exc
    except OSError as exc:
        # A registry path that is a directory / unreadable / otherwise un-openable fails CLOSED as a
        # structured refusal (exit 2), not an uncaught traceback (which would surface as exit 1 and
        # collide with the doclint --strict staleness signal).
        raise DocLintError(f"context-source registry could not be read ({REGISTRY_RELPATH}): {exc}") from exc
    return parse_registry(raw)


def _finding(rule: str, severity: str, classification: str, source: DocSource, line_no: int,
             line: str, authority: str, message: str) -> Finding:
    return Finding(
        rule=rule,
        severity=severity,
        classification=classification,
        path=source.path,
        line=line_no,
        excerpt=_sanitize_excerpt(line.strip())[:_MAX_EXCERPT],
        superseding_authority=authority,
        message=message,
    )


_CONTRACT_INDEX_RELPATH = "docs/contracts/index.json"


def count_root_contracts(repo_root: Path) -> int | None:
    """Derive the current root-contract count from the committed schema index (stdlib-only; imports
    nothing from corpus_studio). Returns None when the index is absent or malformed so the count-drift
    rule degrades to a silent no-op rather than emitting a finding off a bad derive - the AUTHORITATIVE
    check is the pytest ``len(ROOT_CONTRACTS)`` assertion; this prose rule is an advisory tripwire."""
    index_path = repo_root / _CONTRACT_INDEX_RELPATH
    try:
        data = json.loads(index_path.read_text("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    contracts = data.get("contracts")
    if not isinstance(contracts, list):
        return None
    return len(contracts)


def scan_source(source: DocSource, lines: list[str], contract_count: int | None = None) -> list[Finding]:
    """Apply the mode-aware rules to one source's lines. Pure; no filesystem access. ``contract_count``
    is the DERIVED root-contract count (from :func:`count_root_contracts`); when None the count-drift
    rule is a no-op."""
    findings: list[Finding] = []
    ui_in_scope = source.mode in _CURRENTISH_MODES or source.stable_guidance
    historical = source.mode in _HISTORICAL_MODES
    for line_no, line in enumerate(lines, start=1):
        # R1 - removed WPF/Avalonia UI presented without a historical marker. The marker is sought
        # across a small window (previous + current + next line) so a sentence like "The WPF/Avalonia
        # desktop that\nwas removed" is not flagged just because the marker fell on the next line.
        window = " ".join(lines[max(0, line_no - 2):line_no + 1])
        if ui_in_scope and _UI_PATTERN.search(line) and not _HISTORICAL_MARKER.search(window):
            findings.append(_finding(
                "removed-ui-present-tense", "med", "STALE_REMOVED_UI", source, line_no, line,
                "AGENTS.md (UI is Tauri 2 + React; WPF/Avalonia removed #545)",
                "WPF/Avalonia/Nocturne referenced without a historical marker; reword to Tauri 2 + "
                "React or mark the line historical",
            ))
        # R2 - absolute host facts baked into host-portable guidance.
        if source.stable_guidance:
            for pattern, what in _ABS_PATH_PATTERNS:
                if pattern.search(line):
                    findings.append(_finding(
                        "absolute-host-path", "low", "ABSOLUTE_HOST_PATH", source, line_no, line,
                        "docs/HOST_STATE.md + `git rev-parse --show-toplevel`",
                        f"{what} hardcoded in always-loaded guidance; host facts belong in "
                        "HOST_STATE.md",
                    ))
                    break
            # R3 - volatile identities (PR/commit/wheel/run/matrix) embedded in stable guidance.
            if _VOLATILE_LINE.search(line):
                findings.append(_finding(
                    "volatile-identity-in-stable", "low", "VOLATILE_STATE_IN_STABLE_DOC", source,
                    line_no, line, "HANDOFF.md / docs/HOST_STATE.md",
                    "volatile identity (PR/commit/wheel/run/matrix) in stable guidance; move to "
                    "HANDOFF/HOST_STATE",
                ))
        # R4 - GitHub settings asserted as durable repository facts.
        if (source.stable_guidance or source.authority == "canonical") and not historical:
            if _GITHUB_SETTING.search(line):
                findings.append(_finding(
                    "github-setting-as-fact", "low", "AUTHENTICATED_GITHUB_SETTING_REQUIRED",
                    source, line_no, line, "live GitHub settings (not present in the tree)",
                    "branch-protection/required-check/merge-authorization asserted as fact; these "
                    "are authenticated GitHub settings, not repository facts",
                ))
        # R5 - root-contract count drift, checked against the DERIVED count (never a hardcoded number,
        # so the rule cannot itself go stale). Scoped to durable current guidance: a dated log
        # (VOLATILE_CURRENT) or a frozen record legitimately preserves the count it had when written.
        if source.mode in _CURRENTISH_MODES and contract_count is not None:
            count_match = _ROOT_CONTRACT_COUNT.search(line)
            if count_match is not None and int(count_match.group(1)) != contract_count:
                findings.append(_finding(
                    "root-contract-count-drift", "low", "COUNT_DRIFT", source, line_no, line,
                    "docs/contracts/index.json + tests/test_platform_contracts.py",
                    f"stated '{count_match.group(1)} root contracts' does not match the "
                    f"{contract_count} exported root contracts",
                ))
    return findings


def _unreadable_finding(source: DocSource, exc: Exception) -> Finding:
    # A present-but-unreadable doc must not be silently dropped (the doc-trust sensor would then read
    # as "clean" while a registered doc went unchecked). Surface it as drift.
    return Finding(
        rule="registry-unreadable-file",
        severity="low",
        classification="REGISTRY_STALE",
        path=source.path,
        line=0,
        excerpt="",
        superseding_authority="the repository tree",
        message=f"registered doc could not be read ({type(exc).__name__}); "
                "the sensor must not silently skip a registered doc",
    )


def lint_repo(repo_root: Path, sources: list[DocSource]) -> list[Finding]:
    """Run the lint over every registered source. A registered path that is missing is itself a
    finding (registry drift). Findings are returned sorted for deterministic output."""
    findings: list[Finding] = []
    contract_count = count_root_contracts(repo_root)
    for source in sources:
        target = repo_root / source.path
        if not target.exists():
            findings.append(Finding(
                rule="registry-missing-file",
                severity="low",
                classification="REGISTRY_STALE",
                path=source.path,
                line=0,
                excerpt="",
                superseding_authority="the repository tree",
                message="registered doc does not exist (registry drift); update the registry",
            ))
            continue
        if not target.is_file():
            # is_file() follows the symlink and is False for a directory / device / FIFO / socket, so
            # this rejects a registered path repointed at /dev/zero or a blocking FIFO BEFORE open() -
            # a bounded read alone would still block on a FIFO. (A broken symlink was caught by exists.)
            findings.append(Finding(
                rule="registry-not-a-file",
                severity="low",
                classification="REGISTRY_STALE",
                path=source.path,
                line=0,
                excerpt="",
                superseding_authority="the repository tree",
                message="registered path is not a regular file (directory / device / FIFO / symlink "
                        "to one); fix the registry",
            ))
            continue
        try:
            with target.open("rb") as handle:
                raw_bytes = handle.read(_MAX_DOC_BYTES + 1)
        except OSError as exc:
            findings.append(_unreadable_finding(source, exc))
            continue
        if len(raw_bytes) > _MAX_DOC_BYTES:
            findings.append(Finding(
                rule="registry-oversized-file",
                severity="low",
                classification="REGISTRY_STALE",
                path=source.path,
                line=0,
                excerpt="",
                superseding_authority="the repository tree",
                message=f"registered doc exceeds the {_MAX_DOC_BYTES // (1024 * 1024)} MiB lint bound; "
                        "the sensor must not read an unbounded file into memory",
            ))
            continue
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            findings.append(_unreadable_finding(source, exc))
            continue
        findings.extend(scan_source(source, text.splitlines(), contract_count))
    findings.sort(key=lambda f: (f.path, f.line, f.rule))
    return findings
