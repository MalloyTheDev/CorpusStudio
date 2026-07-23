#!/usr/bin/env python3
"""Stop hook: a state-aware finalize reminder for the CorpusStudio corpus-slice loop.

Defense in depth, NOT a gate. It does NOTHING for ordinary sessions: it only blocks the stop when the
corpus-slice loop has explicitly written phase="FINALIZE_REQUESTED" (and no stop_reason yet) to its
loop state (a fixed per-repo git path), nudging Claude to produce the completion record + verify gate
first. Every other case - no state, any other phase, a stop already resolved, a re-entrant block
(stop_hook_active), or
any error - ALLOWS the stop. Fail-safe by construction: a bug or a missing state never traps a session.

Contract (verified against current Claude Code hook docs): read the event JSON on stdin; exit 0 with no
output to allow the stop; print {"decision":"block","reason":"..."} + exit 0 to ask Claude to continue.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _allow() -> None:
    # Exit 0 with no decision -> Claude stops normally.
    sys.exit(0)


STATE_RELPATH = "corpusstudio-assurance/current-slice.json"


def _load_state(cwd: str) -> dict[str, Any] | None:
    """Read the corpus-slice loop state from its worktree-safe git path, or None if absent.

    The path is FIXED (not session-scoped): Claude cannot read its own harness ``session_id`` to
    write a session-scoped file, so a session-keyed path could never actually be produced. A fixed
    per-repo path is writable by the loop and readable here; cross-session risk is negligible (the
    block is a soft nudge, ``stop_hook_active`` clears it after a re-entry, and the loop clears the
    state on finalize).
    """
    rel = STATE_RELPATH
    try:
        proc = subprocess.run(  # noqa: S603,S607 - fixed argv, no shell
            ["git", "-C", cwd, "rev-parse", "--git-path", rel],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    path = Path(proc.stdout.strip())
    if not path.is_absolute():
        path = Path(cwd) / path
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (ValueError, OSError):
        _allow()
    if not isinstance(payload, dict) or payload.get("stop_hook_active") is True:
        _allow()  # re-entrant / block-cap guard, or malformed input
    cwd = str(payload.get("cwd") or ".").strip() or "."
    state = _load_state(cwd)
    # Block ONLY when the loop has explicitly asked to finalize and has not yet recorded a
    # stop_reason. Resolution is tested by PRESENCE (`is not None`), not truthiness, so a present but
    # falsy stop_reason cannot re-trap the session.
    if not state or state.get("phase") != "FINALIZE_REQUESTED" or state.get("stop_reason") is not None:
        _allow()
    reason = (
        "corpus-slice is FINALIZE_REQUESTED: run the verify gate green, confirm the change set with "
        "`cs_assure changeset`, and produce the completion record before stopping. Set stop_reason in "
        f"{STATE_RELPATH} (or change phase) to stop normally."
    )
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(0)


if __name__ == "__main__":
    main()
