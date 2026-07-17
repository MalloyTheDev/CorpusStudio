# Amendment 0006 - protocol-validator hardening (no scientific change)

- **Amendment id:** `cs-ieee-linux-training-amendment-0006`
- **Study:** `cs-ieee-linux-training-v1`
- **Base protocol version:** 1.0.0
- **Effective protocol version:** 1.6.0 (supersedes 1.5.0)
- **Status:** validator-tooling hardening (no scientific / spec change)
- **Authored:** 2026-07-17
- **Analysis role:** primary

This amendment is **append-only**. It does not edit `PROTOCOL.md`, `EXPERIMENT_MATRIX.yaml`, amendments
0001-0005, effective matrices 1.1.0-1.5.0, or `RESERVED_IDENTITIES.v1`..`.v5` in place. It adds effective
matrix **1.6.0**, reserved-identity registry **v6** (a byte-superset of v5 with NO new identities), and
this narrative + manifest. Amendment 0005 is bound by exact raw-file hash in the manifest `supersedes`
block, so the amendment chain stays ordered and 0005 stays provably byte-frozen.

## Why this amendment exists

A pre-training audit of the research-protocol validator surfaced four hardening gaps. Because
`validate_protocol.py` is hash-sealed inside amendment 0005 (and self-hash-enforced at validation),
changing it requires a superseding amendment; 0006 re-seals the hardened validator. **There is no
scientific change:** the Qwen2.5-0.5B / seq-256 bring-up tuple, the primary/secondary matrix cells, the
v8 lineage identities (`math-v8`/`flash-v8`), the exact per-lineage floor and reviewed worker-source-commit
bindings, the 7B feasibility ladder and its grounded flash-eligibility mapping, and every amendment-0005
field are unchanged except the effective protocol version stamp (1.5.0 -> 1.6.0) and this amendment's own
metadata (id, supersession, prior-amendment chain, reserved-registry pointer).

## What 1.6.0 hardens (validator only)

1. **Live-enum grounding.** `_validate_math_terminal_flash_eligibility` binds the declared
   `known_failure_taxonomy` / `known_stage_markers` snapshots to the LIVE
   `corpus_studio.platform.enums.FailureTaxonomy` / `StageMarker` (exact, ordered) inside the validator
   itself, fail-closed - not only in a CI test. A drifted or fabricated snapshot is refused.
2. **Preserved-evidence reservation.** Every completed-run identity the effective matrix documents as
   history (any `preserved_*_evidence` block: run ids, wheel sha-256s, environment ids, plan ids,
   artifact ids) must be a member of the reserved-identity registry, so the newest lineage's real run
   identities cannot be recorded as history yet left reusable at plan time.
3. **Case-insensitive disjointness.** Reserved-identity disjointness for id classes is now
   case-insensitive, so a case-variant of a reserved id (e.g. an uppercased-hex UUID) cannot slip past.
4. **Stronger worker-execution reason guard.** When a lineage classification denies a worker-execution
   change, its reason code is checked against a broader denylist of execution-implying phrasings, not
   only the exact `worker-execution` token.

Plus a robustness fix: `_validate_reserved` type-checks reserved values before the sort, so a malformed
reserved file fails as a clean protocol error rather than an unhandled exception.

## What 1.6.0 does NOT change

The effective matrix 1.6.0 is byte-identical to 1.5.0 except the version stamp and this amendment's own
metadata. No environment, plan, run, adapter, wheel, or model identity changes.
`RESERVED_IDENTITIES.v6` reserves exactly the v5 set (append-only superset, zero additions).

## Note on the superseded prospective v8 wheel

The prospective v8 wheel built earlier (source `fedd7d5`, sha-256 `1762f1e8...`) was superseded by the
builder/admission hardening and will be rebuilt from the post-hardening main. It was never sealed into
any environment, so it is a build artifact - not an instantiated lineage identity - and is not reserved
here; the rebuilt v8 wheel becomes a reserved identity only when it is sealed into `math-v8`/`flash-v8`.

## Preregistration boundary (unchanged)

No 7B model is loaded, no wheel is built, no environment is created, and no rung is executed by this
amendment.
