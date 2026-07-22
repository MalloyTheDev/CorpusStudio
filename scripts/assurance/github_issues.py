"""Fail-SOFT GitHub-issue sensor for `cs_assure status` (the loop's only external, non-offline input).

Unlike the rest of the kernel, which fails CLOSED (a refusal is exit 2), missing issues are honest
absence, not a refusal: ``gh`` may be uninstalled, unauthenticated, offline, rate-limited, or slow, and
none of that invalidates the local snapshot. So this module NEVER raises ``AssuranceError``; every
failure becomes an ``{"available": False, "reason": ..., "detail": ...}`` sentinel and the caller still
seals the rest of the snapshot at exit 0. (Contrast: a git/kernel error stays fail-closed -> exit 2.)

It shells out to ``gh`` as a fixed ``argv`` list (no shell), read-only (``gh issue list`` only - never
create/edit/close/label), with a hard timeout so a live network call can never hang the loop.
Stdlib-only; no ``gh`` python library.
"""

from __future__ import annotations

import json
import os
import re
import subprocess  # noqa: S404 - fixed-argv gh only; never a shell string.
from pathlib import Path
from typing import Any

# Read-only issue list. ``--limit`` is a ceiling well above the real open count so gh's default of 30
# never SILENTLY truncates total_open / by_area; ``--json`` takes ONE comma-joined field argument.
# If the true open count ever exceeds the cap, the summary flags total_open as a lower bound (below) -
# the drop is COUNTED, never hidden.
GH_FETCH_LIMIT = 500
GH_ISSUE_ARGV = [
    "gh", "issue", "list",
    "--state", "open",
    "--limit", str(GH_FETCH_LIMIT),
    "--json", "number,title,labels,updatedAt,state",
]
GH_TIMEOUT_S = 15
ISSUES_SOURCE = "gh issue list --state open"

# The area tag is the FIRST bracketed prefix of the title, lower-cased. It is a raw parsed tag, NOT one
# of the seven authoritative product areas - mapping to those is the corpus-progress skill's judgment.
_AREA_RE = re.compile(r"^\s*\[([^\]]+)\]")


def parse_area(title: str) -> str:
    """Return the lower-cased first ``[bracket]`` tag of an issue title (<=64 chars), or ``'untagged'``."""
    match = _AREA_RE.match(title or "")
    if not (match and match.group(1).strip()):
        return "untagged"
    # Bound the tag so a pathological title cannot re-inject long content past the title's own cap.
    return match.group(1).strip().lower()[:64]


def _unavailable(reason: str, detail: str) -> dict[str, Any]:
    return {"available": False, "reason": reason, "detail": detail[:200], "source": ISSUES_SOURCE}


def _classify_failure(returncode: int, stderr_bytes: bytes) -> dict[str, Any]:
    """Map a nonzero ``gh`` exit to a stable ``reason`` enum by stderr signature (generic fallback)."""
    stderr = stderr_bytes.decode("utf-8", "replace")
    low = stderr.lower()
    if "401" in stderr or "gh auth login" in low or "authentication" in low:
        reason = "unauthenticated"
    elif ("error connecting" in low or "check your internet" in low
          or "dial tcp" in low or "no such host" in low):
        reason = "network"
    else:
        # rate-limit (HTTP 403) / no-repo-context / any other nonzero exit; discriminator kept in detail.
        reason = "gh-error"
    return _unavailable(reason, f"gh exited {returncode}: {stderr.strip()[:180]}")


def _summarize_issues(raw: list[dict[str, Any]], limit_recent: int) -> dict[str, Any]:
    """Bound the issue list to a summary. ``total_open``/``by_area`` cover ALL; only ``recent`` is capped."""
    by_area: dict[str, int] = {}
    for issue in raw:
        by_area[parse_area(issue.get("title", ""))] = by_area.get(parse_area(issue.get("title", "")), 0) + 1
    # Select the N most-recently-updated (updatedAt is RFC3339-Z: lexicographic == chronological;
    # number desc breaks ties for a deterministic total order), then EMIT ordered by number desc.
    ranked = sorted(raw, key=lambda it: (str(it.get("updatedAt", "")), it.get("number", 0)), reverse=True)
    chosen = ranked[: max(0, limit_recent)]
    recent = sorted(
        (
            {
                "number": issue.get("number", 0),
                "area": parse_area(issue.get("title", "")),
                "title": (issue.get("title") or "")[:120],
                "updatedAt": str(issue.get("updatedAt", "")),
            }
            for issue in chosen
        ),
        key=lambda r: r["number"],
        reverse=True,
    )
    return {
        "available": True,
        "source": ISSUES_SOURCE,
        "total_open": len(raw),
        # If the fetch hit the cap, total_open / by_area may undercount - flag it, never silently truncate.
        "total_open_is_lower_bound": len(raw) >= GH_FETCH_LIMIT,
        "by_area": by_area,
        "recent_limit": limit_recent,
        "recent": recent,
        "recent_omitted_count": max(0, len(raw) - len(recent)),
    }


def gather_issues(
    root: Path,
    *,
    limit_recent: int,
    timeout_s: int = GH_TIMEOUT_S,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    """Return a bounded open-issue summary, or a fail-soft ``available:false`` sentinel. Never raises."""
    argv = argv if argv is not None else GH_ISSUE_ARGV
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, read-only gh.
            argv,
            cwd=str(root),
            capture_output=True,
            check=False,
            timeout=timeout_s,
            env={**os.environ, "GH_PROMPT_DISABLED": "1", "NO_COLOR": "1"},
        )
    except FileNotFoundError:
        return _unavailable("gh-missing", "gh binary not found on PATH")
    except subprocess.TimeoutExpired:
        return _unavailable("timeout", f"gh exceeded {timeout_s}s")
    except OSError as exc:
        return _unavailable("gh-error", f"spawn failed: {exc}")
    if proc.returncode != 0:
        return _classify_failure(proc.returncode, proc.stderr)
    try:
        raw = json.loads(proc.stdout.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        return _unavailable("parse-error", f"could not parse gh JSON: {exc}")
    if not isinstance(raw, list):
        return _unavailable("parse-error", "gh JSON was not an array")
    if not all(isinstance(item, dict) for item in raw):
        # Non-object elements ([null], [5], ...) would make _summarize_issues raise OUTSIDE the caller's
        # try; catch it here so the "never raises" fail-soft contract holds even on a weird gh payload.
        return _unavailable("parse-error", "gh JSON array had non-object elements")
    return _summarize_issues(raw, limit_recent)
