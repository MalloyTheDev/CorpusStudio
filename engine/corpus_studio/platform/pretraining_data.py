"""Pretraining data helpers (#487): the seeded, reproducible global shard order.

Control-plane and torch-free - the pure planning half of the pretraining data path (parallel to the
SFT-only ``TrainingDataPolicy``). The runtime per-rank streaming cursor + bitwise streaming resume is a
separate (worker) slice; this module only computes the deterministic ORDER a single-rank stream follows,
which the resume gate reproduces from ``data_seed``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from corpus_studio.platform.contracts import PretrainingDataPolicy


@dataclass(frozen=True)
class PretrainingDataPlan:
    """Control-plane accounting for a :class:`PretrainingDataPolicy` (#487): the corpus token total, the
    per-source token split + the REALIZED mixture (the actual per-source proportions of the corpus,
    which a declared ``mixture_weights`` may not match), and how the token budget lands against the
    corpus (the number of epochs it implies, and whether it repeats the corpus). The budget is an
    explicit stop condition, never a silent truncation; this reports how it lands so a run is sized
    honestly."""

    total_shard_tokens: int
    per_source_tokens: dict[str, int]
    realized_mixture: dict[str, float]
    token_budget: int | None
    epochs_for_budget: float | None
    repeats_corpus: bool


def plan_pretraining_data(policy: PretrainingDataPolicy) -> PretrainingDataPlan:
    """Account for a pretraining data policy WITHOUT running it: the total + per-source token counts,
    the realized corpus mixture, and how ``token_budget`` lands (the epochs it implies; whether it
    repeats the corpus). Pure + control-plane."""
    per_source: dict[str, int] = {}
    for shard in policy.shards:
        per_source[shard.source] = per_source.get(shard.source, 0) + shard.token_count
    total = sum(per_source.values())
    realized = {source: tokens / total for source, tokens in per_source.items()} if total else {}
    budget = policy.token_budget
    epochs_for_budget = (budget / total) if (budget is not None and total) else None
    return PretrainingDataPlan(
        total_shard_tokens=total,
        per_source_tokens=per_source,
        realized_mixture=realized,
        token_budget=budget,
        epochs_for_budget=epochs_for_budget,
        repeats_corpus=bool(budget is not None and total and budget > total),
    )


def deterministic_shard_order(policy: PretrainingDataPolicy) -> tuple[str, ...]:
    """A seeded, reproducible global shard order (the #487 determinism gate).

    The SAME ``data_seed`` always yields the SAME order - so a single-rank streaming resume can
    reproduce the exact shard sequence - and a different seed yields a different order. The order is
    version-independent: a seeded SHA-256 key sort, NOT ``random.shuffle`` (whose algorithm is not
    guaranteed stable across Python versions). Shard ids are unique (validated on the policy), so the
    key sort is a total, stable permutation.
    """

    def _key(shard_id: str) -> str:
        return hashlib.sha256(f"{policy.data_seed}:{shard_id}".encode()).hexdigest()

    return tuple(sorted((shard.shard_id for shard in policy.shards), key=_key))
