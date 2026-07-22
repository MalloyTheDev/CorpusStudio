#!/usr/bin/env python3
"""PostToolUse (Edit|Write) hook: advisory classification for CorpusStudio. OPT-IN (see README).

Non-blocking and fail-safe: after an edit, if the edited path is in a sensitive area (contract source,
worker execution closure, sealed research, the assurance system itself, or the evaluation path) it
prints a one-line reminder to stderr. It NEVER blocks and NEVER fails the tool - ordinary paths pass
silently, and any error still exits 0. It complements the path-scoped `.claude/rules/*` (which fire on
read); this fires on write. It runs on every Edit/Write, so it ships disabled by default - enable it in
`.claude/settings.json` only if you want the write-time reminder (see .claude/hooks/README.md).
"""

from __future__ import annotations

import json
import sys

# (path fragment, reminder). Matched against the edited file path.
_SENSITIVE: tuple[tuple[str, str], ...] = (
    ("platform/contracts.py", "contracts changed -> regenerate schemas + TS and update the 3 contract-count assertions (test_platform_contracts)."),
    ("platform/worker.py", "worker-closure file -> a worker-byte change needs a fresh package + env locks; trace the import path."),
    ("platform/runners.py", "worker-closure file -> worker code under platform/; classify WORKER_CHANGE_REQUIRED vs control-plane-only."),
    ("platform/artifacts.py", "worker-closure file -> worker code under platform/; success admission runs in the child."),
    ("platform/supervisor.py", "worker-closure-adjacent -> trace whether this touches worker-execution bytes."),
    ("training/trainer.py", "worker-closure file -> a change here changes worker-execution bytes."),
    ("research/ieee-linux-training/", "SEALED RESEARCH -> append-only + auth-gated; never edit a frozen protocol/amendment/matrix in place."),
    ("docs/paper/", "research paper -> frozen snapshot evidence; do not rewrite for product advances."),
    ("scripts/assurance/", "ASSURANCE SELF-MODIFY (BOOTSTRAP_SELF_MODIFIED) -> provisional; needs trusted-base CI + independent review."),
    ("scripts/cs_assure.py", "ASSURANCE SELF-MODIFY (BOOTSTRAP_SELF_MODIFIED) -> provisional; needs trusted-base CI + independent review."),
    ("corpus_studio/evaluation/", "evaluation path -> an unavailable metric is null-with-reason, never a fabricated 0."),
)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (ValueError, OSError):
        sys.exit(0)
    # An explicit "tool_input": null must not crash (None.get -> AttributeError); default to {}.
    tool_input = payload.get("tool_input") if isinstance(payload, dict) else None
    path = str((tool_input if isinstance(tool_input, dict) else {}).get("file_path") or "")
    norm = path.replace("\\", "/")
    # Segment-boundary match: prepend "/" to both sides so a fragment matches only at a path
    # boundary ("training/trainer.py" must NOT match "myproj_training/trainer.py").
    haystack = "/" + norm
    for fragment, reminder in _SENSITIVE:
        if ("/" + fragment) in haystack:
            name = norm.rsplit("/", 1)[-1] or norm
            # Strip control chars: file_path is caller-influenced; never inject ANSI/newlines here.
            safe = "".join(ch if ch.isprintable() and ch not in "\r\n\t" else "?" for ch in name)
            sys.stderr.write(f"[corpus-assure] {safe}: {reminder}\n")
            break
    sys.exit(0)


if __name__ == "__main__":
    main()
