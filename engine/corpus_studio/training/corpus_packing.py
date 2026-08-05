"""Sequence packing for from-scratch pretraining (torch-free, pure) - the worker-side complement to the
control-plane ``platform.pretraining_data`` (which plans shard order + token budget).

Pretraining trains on PACKED sequences: tokenized documents concatenated with an EOS separator and split
into fixed ``sequence_len`` blocks, no padding - which is why a from-scratch run needs a different data
path than SFT's one-example-per-row. This module is the packing primitive + its coverage accounting.

Honesty invariant (no silent truncation): EVERY token is accounted for. A trailing remainder shorter than
one block is REPORTED - dropped (``drop_remainder=True``) or explicitly padded (``drop_remainder=False``)
- never hidden. The worker's data loader (a later slice) streams shards through this; here it is a pure
function so the packing math is unit-tested without torch or a tokenizer.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class PackingCoverage:
    """How the token stream was accounted for - so a run can prove nothing was silently dropped."""

    total_tokens: int  # every token in the stream (document tokens + one EOS separator per document)
    packed_tokens: int  # tokens that landed in a full block (from the real stream)
    dropped_tokens: int  # a trailing remainder dropped because it did not fill a block
    padded_tokens: int  # EOS padding added to complete a final block (drop_remainder=False)
    num_blocks: int
    num_documents: int

    @property
    def coverage_ratio(self) -> float:
        """Fraction of real stream tokens that made it into a block (1.0 => nothing dropped)."""
        return self.packed_tokens / self.total_tokens if self.total_tokens else 0.0


@dataclass(frozen=True)
class PackedCorpus:
    """The packed blocks (each exactly ``sequence_len`` tokens) + honest coverage accounting."""

    blocks: list[list[int]]
    sequence_len: int
    coverage: PackingCoverage


def pack_documents(
    documents: Iterable[list[int]],
    *,
    sequence_len: int,
    eos_id: int,
    drop_remainder: bool = True,
) -> PackedCorpus:
    """Concatenate tokenized ``documents`` (each an EOS-separated segment) and split into fixed
    ``sequence_len`` blocks. A document longer than a block simply spans multiple blocks - packing works
    on the flat stream. The final short remainder is dropped or EOS-padded per ``drop_remainder``, and
    either way it is counted in the returned :class:`PackingCoverage` (never silently discarded)."""
    if sequence_len < 1:
        raise ValueError("sequence_len must be positive")
    if eos_id < 0:
        raise ValueError("eos_id must be non-negative")

    stream: list[int] = []
    num_documents = 0
    for document in documents:
        num_documents += 1
        stream.extend(document)
        stream.append(eos_id)  # EOS separator bounds cross-document continuation

    total = len(stream)
    full_blocks = total // sequence_len
    blocks = [stream[i * sequence_len : (i + 1) * sequence_len] for i in range(full_blocks)]

    remainder = total - full_blocks * sequence_len
    dropped = 0
    padded = 0
    if remainder:
        if drop_remainder:
            dropped = remainder
        else:
            padded = sequence_len - remainder
            blocks.append(stream[full_blocks * sequence_len :] + [eos_id] * padded)

    coverage = PackingCoverage(
        total_tokens=total,
        packed_tokens=total - dropped,
        dropped_tokens=dropped,
        padded_tokens=padded,
        num_blocks=len(blocks),
        num_documents=num_documents,
    )
    return PackedCorpus(blocks=blocks, sequence_len=sequence_len, coverage=coverage)
