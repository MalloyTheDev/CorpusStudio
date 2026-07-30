"""Pretraining data policy + the seeded deterministic shard order (#487).

The policy is additive + dense/MoE-safe (parallel to the SFT-only TrainingDataPolicy); it fails closed
without a stop condition and on an orphan / non-positive mixture weight. The global order is
reproducible from ``data_seed`` (the single-rank streaming-resume determinism gate).
"""

import pytest

from corpus_studio.platform.contracts import PretrainingDataPolicy, PretrainingShard
from corpus_studio.platform.pretraining_data import deterministic_shard_order


def _shard(i: int, source: str = "web") -> PretrainingShard:
    return PretrainingShard(
        shard_id=f"s{i}",
        location=f"/corpus/s{i}.jsonl",
        source=source,
        row_count=100,
        token_count=10_000,
        content_sha256=f"{i:064x}",
    )


def _policy(**overrides) -> PretrainingDataPolicy:
    kwargs: dict = dict(
        shards=tuple(_shard(i) for i in range(6)),
        data_seed=42,
        global_batch_size=8,
        token_budget=1_000_000,
    )
    kwargs.update(overrides)
    return PretrainingDataPolicy(**kwargs)


def test_policy_needs_a_stop_condition_token_budget_or_epochs():
    # A pretraining run must never stream without a bound (no silent non-termination / truncation).
    with pytest.raises(ValueError, match="stop condition"):
        PretrainingDataPolicy(
            shards=tuple(_shard(i) for i in range(3)), data_seed=1, global_batch_size=8
        )
    assert _policy(token_budget=5_000, epochs=None).token_budget == 5_000
    assert _policy(token_budget=None, epochs=2).epochs == 2


def test_policy_rejects_duplicate_shards_and_bad_mixture_weights():
    with pytest.raises(ValueError, match="unique"):
        PretrainingDataPolicy(
            shards=(_shard(0), _shard(0)), data_seed=1, global_batch_size=8, token_budget=10
        )
    with pytest.raises(ValueError, match="no shard"):
        _policy(mixture_weights={"books": 1.0})  # 'books' names no shard (every shard is 'web')
    with pytest.raises(ValueError, match="positive"):
        _policy(mixture_weights={"web": 0.0})


def test_valid_mixture_over_real_sources_is_accepted():
    policy = PretrainingDataPolicy(
        shards=(_shard(0, "web"), _shard(1, "books")),
        data_seed=1,
        global_batch_size=4,
        token_budget=100,
        mixture_weights={"web": 0.7, "books": 0.3},
    )
    assert policy.mixture_weights == {"web": 0.7, "books": 0.3}


def test_deterministic_order_is_a_reproducible_total_permutation():
    policy = _policy()
    order = deterministic_shard_order(policy)
    assert sorted(order) == sorted(s.shard_id for s in policy.shards)  # a total permutation
    assert deterministic_shard_order(policy) == order  # same data_seed -> identical order


def test_deterministic_order_changes_with_the_seed():
    assert deterministic_shard_order(_policy(data_seed=1)) != deterministic_shard_order(
        _policy(data_seed=2)
    )
