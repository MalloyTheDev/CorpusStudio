---
name: corpus-research
description: Manual workflow for the CorpusStudio IEEE 7B research overlay - the append-only amendment/identity discipline, the GPU/environment/wheel hard stops, and the sealed-evidence and forbidden-claim rules. Invoke it ONLY when the task is the paper (research/ieee-linux-training, docs/paper); ordinary product work must not load it.
disable-model-invocation: true
---

# corpus-research - sealed research overlay

Load this ONLY when the task is the native-Linux 7B paper overlay. It is opt-in evidence machinery, not
product behavior; it never defines product identity, defaults, navigation, or ordinary workflow.

## Hard stops (auth-gated - each needs explicit human authorization)

- Building a worker wheel; creating / sealing / removing a managed environment; generating an executable
  RunPlan; loading model weights; dispatching ANY GPU run. Do not infer authorization from
  exploratory / product evidence. For GPU work: unload Ollama first, one GPU operation at a time, no
  auto-retry.
- STOP and surface before a full 7B run, or any change that would amend the study after results are
  visible.

## Append-only + immutable

- Never edit a frozen protocol, amendment, effective matrix, or reserved-identity set in place. A change
  is a new dated amendment -> effective-matrix bump -> superset reserved identities (append-only over
  the prior). `validate_protocol.py` must stay green: effective-matrix reconstruction is
  byte-deterministic, a new amendment hash-binds the prior one, and it is set-disjoint from the reserved
  identities.
- Metric definitions are authoritative in `research/ieee-linux-training/METRICS.md` - implement them
  exactly, not approximately.

## Identity + boundary integrity

- A worker-execution-closure change forces a fresh worker package + new environment locks (see the
  worker-closure rule) BEFORE any dispatch. Once identities are instantiated (wheel built, environments
  sealed, a run produced), prove non-impact by call-graph or bump the lineage.
- **Fail closed, do not reconcile.** Never reword sealed evidence to "exploratory" or the reverse. If
  the committed spec and the on-disk evidence disagree - an instantiated identity that is not reserved
  in the committed registry - STOP and escalate to the research authority.
- Volatile ids / hashes / run status / readiness live in `HANDOFF.md` + `docs/HOST_STATE.md`, never in
  the sealed spec or a stable doc.

## Forbidden claims

Do not claim sealed full-sequence 7B success, real offload fit (DeepSpeed / FSDP / CPU / NVMe),
throughput or endurance, bare-Linux FlashAttention for the real workload, or MoE runtime capability
without a measured run. Contracts, fake workers, CI, and a passing environment probe are not proof of a
sealed claim; an exploratory product run does not amend the sealed ladder.
