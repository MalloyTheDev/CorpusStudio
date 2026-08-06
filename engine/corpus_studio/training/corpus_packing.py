"""Sequence packing for from-scratch pretraining (torch-free, pure) - the worker-side complement to the
control-plane ``platform.pretraining_data`` (which plans shard order + token budget).

Pretraining trains on PACKED sequences: tokenized documents concatenated with an EOS separator and split
into fixed ``sequence_len`` blocks, no padding - a different data path than SFT's one-example-per-row.

Two layers, so the loader can be memory-bounded AND correct across shards:

* :func:`pack_chunk` - STREAMING-COMPOSABLE. Packs the full blocks it can from one chunk (a shard) plus a
  ``carry_in`` residual, and EXPOSES the trailing ``remainder`` to thread into the next call. The loader
  holds only one chunk + a sub-block residual in memory, and no tokens are lost at a shard boundary.
* :func:`finalize_remainder` - the FINAL leftover after the last chunk: dropped, or EOS-padded into one
  last block. Padded positions are the last ``padded_tokens`` of that block and a loader MUST mask them in
  the labels, or the model is trained to predict synthetic EOS.

:func:`pack_documents` is a whole-corpus convenience over the two (materializes; fine for small corpora +
tests). Honesty invariant (no silent truncation): every token is accounted for - a remainder is dropped or
padded, never hidden; empty documents carry no tokens and are skipped (not a truncation).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class PackingCoverage:
    """How the token stream was accounted for - so a run can prove nothing was silently dropped."""

    total_tokens: int  # every token consumed (document tokens + one EOS separator per non-empty document)
    packed_tokens: int  # real stream tokens that landed in a block
    dropped_tokens: int  # a trailing remainder dropped because it did not fill a block
    padded_tokens: int  # EOS padding added to complete a final block (a loader MUST mask these)
    num_blocks: int
    num_documents: int

    @property
    def coverage_ratio(self) -> float:
        """Fraction of real stream tokens that made it into a block (1.0 => nothing dropped)."""
        return self.packed_tokens / self.total_tokens if self.total_tokens else 0.0


@dataclass(frozen=True)
class PackedChunk:
    """Blocks packed from one chunk, plus the trailing ``remainder`` to carry into the next call. Coverage
    counts this chunk only; a streaming loader accumulates across chunks then finalizes the last remainder."""

    blocks: list[list[int]]
    remainder: list[int]
    sequence_len: int
    tokens_consumed: int  # real tokens added THIS chunk (documents + EOS separators), excluding carry_in
    num_documents: int


@dataclass(frozen=True)
class FinalizedRemainder:
    """The last remainder resolved after every chunk: dropped or EOS-padded into one final block."""

    blocks: list[list[int]]
    dropped_tokens: int
    padded_tokens: int


@dataclass(frozen=True)
class PackedCorpus:
    """A whole-corpus pack (blocks + honest coverage) - the convenience result of :func:`pack_documents`."""

    blocks: list[list[int]]
    sequence_len: int
    coverage: PackingCoverage


def _validate(sequence_len: int, eos_id: int) -> None:
    if sequence_len < 1:
        raise ValueError("sequence_len must be positive")
    if eos_id < 0:
        raise ValueError("eos_id must be non-negative")


def pack_chunk(
    documents: Iterable[list[int]],
    *,
    sequence_len: int,
    eos_id: int,
    carry_in: list[int] | None = None,
) -> PackedChunk:
    """Pack one chunk (e.g. a shard): prepend ``carry_in`` (the previous chunk's residual), concatenate the
    EOS-separated documents, emit every full ``sequence_len`` block, and EXPOSE the leftover ``remainder``
    (< one block) to thread into the next call. Empty documents are skipped. Memory-bounded: holds one
    chunk + a sub-block residual, never the whole corpus."""
    _validate(sequence_len, eos_id)
    stream: list[int] = list(carry_in) if carry_in else []
    tokens_consumed = 0
    num_documents = 0
    for document in documents:
        if not document:
            continue  # an empty document has no tokens; do not emit a spurious EOS-only segment
        num_documents += 1
        stream.extend(document)
        stream.append(eos_id)  # EOS separator bounds cross-document continuation
        tokens_consumed += len(document) + 1

    full_blocks = len(stream) // sequence_len
    blocks = [stream[i * sequence_len : (i + 1) * sequence_len] for i in range(full_blocks)]
    remainder = stream[full_blocks * sequence_len :]
    return PackedChunk(blocks, remainder, sequence_len, tokens_consumed, num_documents)


def finalize_remainder(
    remainder: list[int], *, sequence_len: int, eos_id: int, pad: bool = False
) -> FinalizedRemainder:
    """Resolve the FINAL leftover after all chunks. Default drops it (reports the count). ``pad=True`` EOS-
    pads it into one last block - the last ``padded_tokens`` positions are synthetic and a loader MUST mask
    them in the labels so the model is not trained to predict EOS."""
    _validate(sequence_len, eos_id)
    if not remainder:
        return FinalizedRemainder([], 0, 0)
    if len(remainder) >= sequence_len:
        raise ValueError("remainder must be smaller than one block; pack it before finalizing")
    if not pad:
        return FinalizedRemainder([], len(remainder), 0)
    padded = sequence_len - len(remainder)
    return FinalizedRemainder([remainder + [eos_id] * padded], 0, padded)


def pack_documents(
    documents: Iterable[list[int]],
    *,
    sequence_len: int,
    eos_id: int,
    drop_remainder: bool = True,
) -> PackedCorpus:
    """Whole-corpus convenience: pack every document then resolve the final remainder. Materializes the
    corpus, so a large streamed corpus should use :func:`pack_chunk` + :func:`finalize_remainder` instead."""
    chunk = pack_chunk(documents, sequence_len=sequence_len, eos_id=eos_id)
    tail = finalize_remainder(
        chunk.remainder, sequence_len=sequence_len, eos_id=eos_id, pad=not drop_remainder
    )
    blocks = chunk.blocks + tail.blocks
    coverage = PackingCoverage(
        total_tokens=chunk.tokens_consumed,
        packed_tokens=chunk.tokens_consumed - tail.dropped_tokens,
        dropped_tokens=tail.dropped_tokens,
        padded_tokens=tail.padded_tokens,
        num_blocks=len(blocks),
        num_documents=chunk.num_documents,
    )
    return PackedCorpus(blocks=blocks, sequence_len=sequence_len, coverage=coverage)
