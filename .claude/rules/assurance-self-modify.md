---
paths:
  - "scripts/assurance/**"
  - "scripts/cs_assure.py"
---

# Changing the assurance system is self-modification

A change under `scripts/assurance/` or to `scripts/cs_assure.py` modifies the very tool that judges
changes - it is `BOOTSTRAP_SELF_MODIFIED` / candidate-controlled.

- **Candidate-local results are provisional.** The assurance system cannot vouch for changes to itself.
  Such a change must be admitted by the repo's pre-existing gate, CI, and INDEPENDENT human review -
  never by the assurance system's own output.
- **Candidate self-approval is prohibited.** Do not self-merge your own assurance PR (even if you have
  admin rights); an independent reviewer merges it.
- **Keep the kernel stdlib-only.** `scripts/assurance/` imports nothing from `corpus_studio` and pulls
  no torch; it must run under any `python3`. Preserve its contracts: the narrow canonical-JSON profile,
  the `sha256:`-prefixed digests, the record envelope + fingerprint split, and fail-closed behaviour
  (exit 2 on refusal). Add deterministic tests under `engine/tests/`.
