"""Sequence packing for from-scratch pretraining: documents concatenate with an EOS separator and split
into fixed-length blocks, and EVERY token is accounted for (the no-silent-truncation invariant) - a
trailing remainder is dropped or padded, never hidden."""

import pytest

from corpus_studio.training.corpus_packing import (
    finalize_remainder,
    pack_chunk,
    pack_documents,
)


def test_packs_documents_into_fixed_blocks():
    packed = pack_documents([[1, 2, 3], [4, 5]], sequence_len=3, eos_id=0)
    # stream = [1,2,3,0,4,5,0] (len 7) -> two full blocks of 3, a remainder of 1 dropped.
    assert packed.blocks == [[1, 2, 3], [0, 4, 5]]
    assert all(len(b) == 3 for b in packed.blocks)
    assert packed.coverage.num_documents == 2


def test_every_token_is_accounted_for():
    cov = pack_documents([[1, 2, 3], [4, 5]], sequence_len=3, eos_id=0).coverage
    assert cov.total_tokens == 7  # 3 + 1(eos) + 2 + 1(eos)
    assert cov.packed_tokens + cov.dropped_tokens == cov.total_tokens
    assert cov.dropped_tokens == 1 and cov.padded_tokens == 0
    assert cov.coverage_ratio == pytest.approx(6 / 7)


def test_drop_remainder_false_pads_explicitly():
    packed = pack_documents([[1, 2, 3], [4, 5]], sequence_len=3, eos_id=0, drop_remainder=False)
    assert packed.blocks[-1] == [0, 0, 0]  # remainder [0] + two EOS pads
    assert packed.coverage.padded_tokens == 2
    assert packed.coverage.dropped_tokens == 0
    assert packed.coverage.coverage_ratio == 1.0
    assert packed.coverage.num_blocks == 3


def test_a_document_longer_than_a_block_spans_blocks():
    packed = pack_documents([[1, 2, 3, 4, 5]], sequence_len=2, eos_id=9)
    # stream = [1,2,3,4,5,9] (len 6) -> three blocks of 2, nothing dropped.
    assert packed.blocks == [[1, 2], [3, 4], [5, 9]]
    assert packed.coverage.dropped_tokens == 0


def test_no_full_block_drops_everything_but_reports_it():
    packed = pack_documents([[1, 2]], sequence_len=8, eos_id=0)
    # stream = [1,2,0] (len 3) < one block -> zero blocks, all 3 tokens reported dropped.
    assert packed.blocks == []
    assert packed.coverage.dropped_tokens == 3
    assert packed.coverage.coverage_ratio == 0.0


def test_empty_corpus():
    cov = pack_documents([], sequence_len=4, eos_id=0).coverage
    assert cov.total_tokens == 0 and cov.coverage_ratio == 0.0


def test_empty_documents_are_skipped_not_given_a_spurious_eos():
    # AUDIT (Sourcery): an empty document has no tokens, so it must not emit an EOS-only segment.
    packed = pack_documents([[], [1, 2, 3], []], sequence_len=4, eos_id=0)
    assert packed.blocks == [[1, 2, 3, 0]]  # only the one real document + its EOS
    assert packed.coverage.num_documents == 1
    assert packed.coverage.total_tokens == 4


@pytest.mark.parametrize("kwargs", [{"sequence_len": 0, "eos_id": 0}, {"sequence_len": 4, "eos_id": -1}])
def test_invalid_arguments_fail_closed(kwargs):
    with pytest.raises(ValueError):
        pack_documents([[1, 2]], **kwargs)


# ---- streaming composition (no boundary loss, memory-bounded) --------------------------------------


def test_pack_chunk_carries_residual_across_shards():
    # AUDIT (Codex): the residual from one shard threads into the next, so the boundary token is not lost
    # and the whole corpus need not be materialized.
    c1 = pack_chunk([[1, 2]], sequence_len=2, eos_id=0)  # stream [1,2,0] -> block [1,2], remainder [0]
    assert c1.blocks == [[1, 2]] and c1.remainder == [0]
    c2 = pack_chunk([[3, 4]], sequence_len=2, eos_id=0, carry_in=c1.remainder)  # [0,3,4,0]
    assert c2.blocks == [[0, 3], [4, 0]] and c2.remainder == []
    # streaming equals the whole-corpus pack (no boundary loss)
    whole = pack_documents([[1, 2], [3, 4]], sequence_len=2, eos_id=0)
    assert c1.blocks + c2.blocks == whole.blocks


def test_finalize_remainder_drops_or_pads():
    assert finalize_remainder([9], sequence_len=4, eos_id=0).dropped_tokens == 1
    padded = finalize_remainder([9], sequence_len=4, eos_id=0, pad=True)
    assert padded.blocks == [[9, 0, 0, 0]] and padded.padded_tokens == 3


def test_finalize_rejects_an_oversized_remainder():
    with pytest.raises(ValueError):  # a full block is not a "remainder" - pack it first
        finalize_remainder([1, 2, 3, 4], sequence_len=4, eos_id=0)
