"""Pretraining data helpers (#487): the seeded, reproducible global shard order.

Control-plane and torch-free - the pure planning half of the pretraining data path (parallel to the
SFT-only ``TrainingDataPolicy``). The runtime per-rank streaming cursor + bitwise streaming resume is a
separate (worker) slice; this module only computes the deterministic ORDER a single-rank stream follows,
which the resume gate reproduces from ``data_seed``.
"""

from __future__ import annotations

import hashlib

from corpus_studio.platform.contracts import PretrainingDataPolicy


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
