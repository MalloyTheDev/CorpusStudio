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
    ("platform/enums.py", "contracts trigger -> a contract enum change needs the schema/TS regen + the 3 count assertions (test_platform_contracts)."),
    ("docs/contracts/", "generated schema -> do not hand-edit; regenerate via schema_export and update the 3 count assertions (test_platform_contracts)."),
    ("platform/execution_config.py", "worker-closure file -> a worker-byte change needs a fresh package + env locks; trace the import path."),
    ("platform/planner.py", "worker-RUNTIME-reachable (lazy imports from worker.py/runners.py) -> classify RUNTIME_REACHABLE_REVIEW_REQUIRED, not control-plane by default."),
    ("platform/worker.py", "worker-closure file -> a worker-byte change needs a fresh package + env locks; trace the import path."),
    ("platform/runners.py", "worker-closure file -> worker code under platform/; classify WORKER_CHANGE_REQUIRED vs control-plane-only."),
    ("platform/artifacts.py", "worker-closure file -> worker code under platform/; success admission runs in the child."),
    ("platform/supervisor.py", "worker-closure-adjacent -> trace whether this touches worker-execution bytes."),
    ("training/trainer.py", "worker-closure file -> a change here changes worker-execution bytes."),
    ("research/ieee-linux-training/", "SEALED RESEARCH -> append-only + auth-gated; never edit a frozen protocol/amendment/matrix in place."),
    ("docs/paper/", "research paper -> frozen snapshot evidence; do not rewrite for product advances."),
    ("scripts/assurance/", "ASSURANCE SELF-MODIFY (BOOTSTRAP_SELF_MODIFIED) -> provisional; needs trusted-base CI + independent review."),
    ("scripts/cs_assure.py", "ASSURANCE SELF-MODIFY (BOOTSTRAP_SELF_MODIFIED) -> provisional; needs trusted-base CI + independent review."),
    (".claude/", "ASSURANCE SELF-MODIFY (BOOTSTRAP_SELF_MODIFIED) -> the plugin/policy/rules/hooks are candidate-controlled; needs independent review."),
    ("engine/tests/test_assurance_", "ASSURANCE SELF-MODIFY (BOOTSTRAP_SELF_MODIFIED) -> a change here alters the assurance system's own proofs; needs independent review."),
    ("engine/tests/test_plugin_hooks", "ASSURANCE SELF-MODIFY (BOOTSTRAP_SELF_MODIFIED) -> a change here alters the plugin-hook proofs; needs independent review."),
    (".github/workflows/", "ASSURANCE SELF-MODIFY (BOOTSTRAP_SELF_MODIFIED) -> a CI workflow ENFORCES a gate; weakening it (drop the pytest job / --strict / lower the coverage floor / make a check non-required) defangs the judge, possibly reading 'green' by removing the failing check. Independent review; the loop must not auto-merge it."),
    ("scripts/loop/", "LOOP CONTROLLER SELF-MODIFY -> the loop decides retry/stop/escalate/merge/finalize; the loop cannot verify a change to itself. Trusted-base tests + independent review; no autonomous merge."),
    ("scripts/cs_loop.py", "LOOP CONTROLLER SELF-MODIFY -> the loop CLI is candidate-controlled; trusted-base tests + independent review; no autonomous merge."),
    ("engine/tests/test_loop_", "LOOP CONTROLLER SELF-MODIFY -> a change here alters the loop controller's own proofs; independent review."),
    ("engine/tests/test_cs_loop", "LOOP CONTROLLER SELF-MODIFY -> a change here alters the loop CLI's proofs; independent review."),
    ("docs/AUTONOMOUS_LOOP.md", "LOOP CONTROLLER SELF-MODIFY -> the loop's own contract doc; keep it honest (no overclaim) and reviewed."),
    ("corpus_studio/evaluation/", "evaluation path -> an unavailable metric is null-with-reason, never a fabricated 0."),
)


def reminder_for(file_path: str) -> str | None:
    """Return the advisory reminder for an edited path, or None if the path is not sensitive. Pure and
    the SINGLE source of the match semantics (segment-boundary substring) - shared by main() and the
    3-way policy<->rules<->hook conformance test, so the two can never drift apart."""
    norm = file_path.replace("\\", "/")
    # Segment-boundary match: prepend "/" to both sides so a fragment matches only at a path
    # boundary ("training/trainer.py" must NOT match "myproj_training/trainer.py").
    haystack = "/" + norm
    for fragment, reminder in _SENSITIVE:
        if ("/" + fragment) in haystack:
            return reminder
    return None


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (ValueError, OSError):
        sys.exit(0)
    # An explicit "tool_input": null must not crash (None.get -> AttributeError); default to {}.
    tool_input = payload.get("tool_input") if isinstance(payload, dict) else None
    path = str((tool_input if isinstance(tool_input, dict) else {}).get("file_path") or "")
    reminder = reminder_for(path)
    if reminder is not None:
        name = path.replace("\\", "/").rsplit("/", 1)[-1] or path
        # Strip control chars: file_path is caller-influenced; never inject ANSI/newlines here.
        safe = "".join(ch if ch.isprintable() and ch not in "\r\n\t" else "?" for ch in name)
        sys.stderr.write(f"[corpus-assure] {safe}: {reminder}\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
