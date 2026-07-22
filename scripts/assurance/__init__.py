"""CorpusStudio Assurance kernel - Phase 1 (change-set + canonical records).

This package is the deterministic core of the CorpusStudio Assurance Loop. It answers, for a
selected Git state, the narrow first question of the loop: *what changed* - as a byte-exact,
rename-free, content-addressed change set.

Phase 1 implements ONLY:
  * canonical JSON (a narrow deterministic profile - see ``canonical_json``),
  * two source views (the local working tree, and a committed merge-base tree read from the
    git object store without any checkout),
  * a state-based change-set fingerprint (the applicability key),
  * the sealed ``ChangeSetRecord`` envelope.

Deliberate boundaries (do not add these here until their phase):
  * stdlib ONLY - this package imports nothing from ``corpus_studio`` and must run under any
    ``python3``; it must never pull the training stack or any heavy dependency.
  * it does NOT classify impact, analyse the worker graph, route product areas, load policy,
    run verification, evaluate gates, emit evidence, or install any hooks.

The kernel proves deterministic change-set representation and fingerprint behaviour. It does
NOT prove its own correctness, trust, or the applicability of any downstream conclusion.
"""

from __future__ import annotations

# The assurance kernel's own version. Recorded in change-set provenance (byte-integrity), but
# intentionally kept OUT of the state-based fingerprint so a tool-version bump never changes a
# change set's applicability identity.
KERNEL_VERSION = "0.1.0"

__all__ = ["KERNEL_VERSION"]
