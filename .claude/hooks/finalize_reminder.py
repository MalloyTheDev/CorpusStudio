#!/usr/bin/env python3
"""Stop hook: a state-aware finalize reminder for the CorpusStudio corpus-slice loop.

Defense in depth, NOT a gate. It does NOTHING for ordinary sessions: it only blocks the stop when the
corpus-slice loop has explicitly written phase="FINALIZE_REQUESTED" (and no stop_reason yet) to its
session state, nudging Claude to produce the completion record + verify gate first. Every other case -
no session state, any other phase, a stop already resolved, a re-entrant block (stop_hook_active), or
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


def _load_state(cwd: str, session_id: str) -> dict[str, Any] | None:
    """Read the corpus-slice session state from its worktree-safe git path, or None if absent."""
    rel = f"corpusstudio-assurance/sessions/{session_id}/slice.json"
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
    session_id = str(payload.get("session_id") or "").strip()
    cwd = str(payload.get("cwd") or ".").strip() or "."
    if not session_id:
        _allow()
    state = _load_state(cwd, session_id)
    if not state or state.get("phase") != "FINALIZE_REQUESTED" or state.get("stop_reason"):
        _allow()
    reason = (
        "corpus-slice is FINALIZE_REQUESTED: run the verify gate green, confirm the change set with "
        "`cs_assure changeset`, and produce the completion record before stopping. Set stop_reason in "
        f"corpusstudio-assurance/sessions/{session_id}/slice.json (or change phase) to stop normally."
    )
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(0)


if __name__ == "__main__":
    main()
