---
paths:
  - "scripts/assurance/**"
  - "scripts/cs_assure.py"
  - ".claude/**"
  - "engine/tests/test_assurance_*.py"
  - "engine/tests/test_plugin_hooks.py"
  - ".github/workflows/**"
---

# Changing the assurance system is self-modification

A change under `scripts/assurance/`, to `scripts/cs_assure.py`, or to ANY CI workflow that ENFORCES a
gate (`.github/workflows/**` - the assurance job, the engine test/lint/type gate, the web schema-diff
gate, the security/CodeQL scans) modifies the very machinery that judges changes - it is
`BOOTSTRAP_SELF_MODIFIED` / candidate-controlled. The CI workflows are in scope precisely because a change
that quietly drops `--strict`, deletes the pytest job, lowers the coverage floor, or makes a required
check non-required would defang the gate with no independent-review trigger - and such a change can even
read as "green" because it removes the very check that would have failed. The loop's merge gate must never
auto-merge a CI-policy change; it needs an independent human.

- **Candidate-local results are provisional.** The assurance system cannot vouch for changes to itself.
  Such a change must be admitted by the repo's pre-existing gate, CI, and INDEPENDENT human review -
  never by the assurance system's own output.
- **Candidate self-approval is prohibited.** Do not self-merge your own assurance PR (even if you have
  admin rights); an independent reviewer merges it.
- **Keep the kernel stdlib-only.** `scripts/assurance/` imports nothing from `corpus_studio` and pulls
  no torch; it must run under any `python3`. Preserve its contracts: the narrow canonical-JSON profile,
  the `sha256:`-prefixed digests, the record envelope + fingerprint split, and fail-closed behaviour
  (exit 2 on refusal). Add deterministic tests under `engine/tests/`.
