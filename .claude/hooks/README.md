# CorpusStudio assurance hooks (defense in depth)

Advisory enforcement for the loop-engineering plugin. Hooks are **not** the primary gate - the primary
finalization is the explicit corpus-slice completion record + the verify gate. These hooks are
best-effort, non-blocking-by-default, and fail-safe (a missing script or an unset `CLAUDE_PROJECT_DIR`
makes them inert; they never trap a session).

## Enabled by default: `finalize_reminder.py` (Stop hook)

Wired in `.claude/settings.json`. It is a **no-op for ordinary sessions**. It only acts when the
corpus-slice loop has written `phase="FINALIZE_REQUESTED"` (and no `stop_reason`) to its session state at
`git rev-parse --git-path corpusstudio-assurance/sessions/<session-id>/slice.json` - then it asks Claude
to run the verify gate and produce the completion record before stopping. It allows normal stopping in
every other case (any other phase, an already-resolved stop, a malformed input, a missing state), and it
honours the re-entrant `stop_hook_active` block-cap guard so it can never cause an infinite continue loop.

## Opt-in: `advisory_classify.py` (PostToolUse Edit|Write)

Not wired by default - it would run on every edit. It prints a one-line stderr reminder when an edit
touches a sensitive area (contracts, worker closure, sealed research, the assurance system itself, the
evaluation path); it never blocks. To enable it, add this to the `hooks` object in `.claude/settings.json`:

```json
"PostToolUse": [
  {
    "matcher": "Edit|Write",
    "hooks": [
      {
        "type": "command",
        "command": "[ -f \"$CLAUDE_PROJECT_DIR/.claude/hooks/advisory_classify.py\" ] && exec python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/advisory_classify.py\" || exit 0"
      }
    ]
  }
]
```

## Disabling

Remove the relevant entry from `.claude/settings.json`. Both scripts are stdlib-only Python.
