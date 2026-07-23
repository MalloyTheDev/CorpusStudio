---
paths:
  - "research/ieee-linux-training/**"
  - "docs/paper/**"
---

# Sealed research is append-only and auth-gated

This is the IEEE native-Linux 7B paper overlay - an opt-in evidence program, not product behavior. It
never defines product identity, defaults, or ordinary workflow.

- **Append-only + immutable.** Never edit a frozen protocol, amendment, effective matrix, or reserved-
  identity set in place. A change is a new dated amendment -> effective-matrix bump -> superset reserved
  identities. `validate_protocol.py` must stay green (its effective-matrix reconstruction is
  byte-deterministic and a new amendment hash-binds the prior one).
- **Auth-gated.** Building a worker wheel, creating or sealing an environment, dispatching a GPU run, or
  authoring a new amendment each needs explicit human authorization. Do not infer authorization from
  exploratory/product evidence, and run one GPU operation at a time (unload Ollama first).
- **Boundary integrity - fail closed, do not reconcile.** Never reword sealed evidence to "exploratory"
  or the reverse. If the committed spec and the on-disk evidence disagree - for example an instantiated
  identity (env/wheel/run) that is not reserved in the committed registry - STOP and escalate to the
  research authority; do not silently reconcile.
- Volatile program state (ids, hashes, run status, readiness) lives in `HANDOFF.md` +
  `docs/HOST_STATE.md` - not in the sealed spec or here.
