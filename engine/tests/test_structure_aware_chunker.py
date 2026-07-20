"""Tests for the structure-aware chat chunker (roadmap item 1, no silent truncation).

Deterministic + torch-free: token length is a stub word count, so cut points are exact.
"""
from __future__ import annotations

import pytest

from corpus_studio.splitters.structure_aware_chunker import (
    ChunkingRefusal,
    chunk_chat_dataset,
    chunk_chat_row,
)
from corpus_studio.training.trainer import ExampleTokenSpan, compute_token_coverage

# token length = whitespace word count -> deterministic, no tokenizer.
WORDS = lambda text: len(text.split())  # noqa: E731


def _multi_turn_row():
    # system(2) + 3 exchanges of user(3)+assistant(3)=6 each -> total 20 tokens.
    return {
        "messages": [
            {"role": "system", "content": "sys ctx"},
            {"role": "user", "content": "u1 aaa bbb"},
            {"role": "assistant", "content": "a1 ccc ddd"},
            {"role": "user", "content": "u2 eee fff"},
            {"role": "assistant", "content": "a2 ggg hhh"},
            {"role": "user", "content": "u3 iii jjj"},
            {"role": "assistant", "content": "a3 kkk lll"},
        ]
    }


def test_passthrough_when_the_row_already_fits():
    row = _multi_turn_row()
    chunks = chunk_chat_row(row["messages"], seq_len=1000, token_len=WORDS)
    assert len(chunks) == 1
    assert chunks[0].provenance.strategy == "passthrough"
    assert chunks[0].messages == row["messages"]


def test_multi_turn_splits_at_turn_boundaries_no_assistant_severed():
    row = _multi_turn_row()
    chunks = chunk_chat_row(row["messages"], seq_len=10, token_len=WORDS, source_row_index=7)

    # seq_len 10, preamble 2, each exchange 6 -> one exchange per chunk.
    assert len(chunks) == 3
    for c in chunks:
        assert c.provenance.strategy == "chunk"
        assert sum(WORDS(m["content"]) for m in c.messages) <= 10  # every chunk fits
        assert c.messages[0]["role"] == "system"  # preamble carried
        assert sum(1 for m in c.messages if m["role"] == "assistant") == 1  # >=1 supervised turn, whole
        assert c.provenance.source_row_index == 7
    # the system preamble is duplicated on chunks after the first, and that is flagged
    assert chunks[0].provenance.duplicated_prompt_context is False
    assert all(c.provenance.duplicated_prompt_context for c in chunks[1:])
    # every original assistant turn appears exactly once across the chunks (no drop, no duplication)
    emitted_assistants = [m["content"] for c in chunks for m in c.messages if m["role"] == "assistant"]
    assert emitted_assistants == ["a1 ccc ddd", "a2 ggg hhh", "a3 kkk lll"]


def test_single_over_length_turn_is_refused_fail_closed():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u a b"},
        {"role": "assistant", "content": "resp w x y z"},  # this one turn alone is too big
    ]
    with pytest.raises(ChunkingRefusal, match="cannot chunk without severing"):
        chunk_chat_row(messages, seq_len=3, token_len=WORDS, source_row_index=2)


def test_all_system_over_length_row_is_refused_not_silently_dropped():
    # A degenerate over-length row whose ONLY messages are system (no user/assistant turn) has an empty
    # body: the emit loop would return [] and vanish the record. It must fail CLOSED instead.
    messages = [
        {"role": "system", "content": "a b c d e f"},
        {"role": "system", "content": "g h i j k l"},
    ]
    with pytest.raises(ChunkingRefusal, match="system preamble alone"):
        chunk_chat_row(messages, seq_len=3, token_len=WORDS, source_row_index=5)
    # at the dataset level it is surfaced as a refusal (ok False), never a silent drop
    plan = chunk_chat_dataset([{"messages": messages}], seq_len=3, token_len=WORDS)
    assert plan.ok is False
    assert len(plan.refusals) == 1 and len(plan.rows) == 0


def test_emitted_chunks_satisfy_the_coverage_ledger_supervision_intact():
    # The chunker's output must make the worker preflight gate (compute_token_coverage) pass.
    row = _multi_turn_row()
    seq_len = 10
    chunks = chunk_chat_row(row["messages"], seq_len=seq_len, token_len=WORDS)
    spans = []
    for c in chunks:
        total = sum(WORDS(m["content"]) for m in c.messages)
        supervised = sum(WORDS(m["content"]) for m in c.messages if m["role"] == "assistant")
        spans.append(ExampleTokenSpan(
            total_tokens=total, supervised_tokens=supervised,
            dropped_supervised_tokens=max(0, total - seq_len),  # 0 - each chunk fits
        ))
    ledger = compute_token_coverage(spans, seq_len)
    assert ledger.supervision_intact is True
    assert ledger.is_lossless is True  # pure chunk: nothing dropped from the emitted rows
    # supervised tokens are conserved: sum across chunks == the original supervised total (9)
    original_supervised = sum(
        WORDS(m["content"]) for m in row["messages"] if m["role"] == "assistant")
    assert ledger.supervised_tokens_total == original_supervised == 9


def test_dataset_level_is_fail_closed_never_silently_dropping_a_bad_record():
    good = _multi_turn_row()
    bad = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "way too long"},
            {"role": "assistant", "content": "r " * 20},  # single turn far over seq_len
        ]
    }
    plan = chunk_chat_dataset([good, bad], seq_len=10, token_len=WORDS)
    assert plan.ok is False  # a refusal exists
    assert len(plan.refusals) == 1
    assert plan.source_row_count == 2
    # the good row's chunks are emitted; the bad row is REFUSED (surfaced), not silently dropped
    assert len(plan.rows) == 3
    assert all(c.provenance.source_row_index == 0 for c in plan.rows)
