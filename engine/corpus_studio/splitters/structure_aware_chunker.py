"""Structure-aware chunker for over-length CHAT training records (roadmap item 1: no silent truncation).

Control-plane and **torch-free**. Splits an over-length multi-turn chat record at TURN boundaries so
every supervised (assistant) turn stays WHOLE in exactly one ``<= seq_len`` chunk - the honest
alternative to a silent right-truncation that severs the completion the model is supposed to learn.

Design guarantees:
- **No silent drop.** Every original assistant turn appears in exactly one emitted chunk. Only
  non-supervised prompt context (the leading system preamble) is duplicated across chunks, and that is
  recorded in the provenance (``duplicated_prompt_context``) - never counted as supervised.
- **Fail closed.** A single turn (a user prompt + its assistant response) that alone exceeds ``seq_len``
  cannot be split at a boundary without severing a supervised turn, so the record is REFUSED with a
  clear ASCII message (raise ``seq_len``, shorten the turn, or opt into an exact intra-turn window later)
  rather than cut.
- **Dense-safe / MoE-safe.** Token counting is an INJECTED callable, so the module reasons only over
  message structure + token counts and never over model execution or dense-vs-expert assumptions. Tests
  and the control plane pass a stub/estimator; an exact re-check can pass the real tokenizer.

The emitted chunks are the SAME chat schema, so they re-enter ``platform-plan`` normally and satisfy the
worker preflight token-coverage gate (feeding the chunks back through ``compute_token_coverage`` yields
``supervision_intact``).
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from pydantic import BaseModel, Field

from ..platform.dataset_conformance import TRAINABLE_CHAT_ROLES

# text -> token count. Injected so the module stays torch-free (no tokenizer import here).
TokenLen = Callable[[str], int]


class ChunkingRefusal(ValueError):
    """A record cannot be chunked without severing a supervised turn - fail closed, never silently cut."""


class ChunkProvenance(BaseModel):
    """Where an emitted chunk came from, so retained-but-duplicated context is auditable and never
    mistaken for supervised signal. Plain BaseModel (NOT a platform ContractModel) - no schema surface."""

    source_row_index: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    strategy: str  # "passthrough" | "chunk"
    # half-open [start, end) range of BODY messages (after the leading system preamble) in this chunk
    body_message_start: int = Field(ge=0)
    body_message_end: int = Field(ge=0)
    duplicated_prompt_context: bool


class ChunkedRow(BaseModel):
    """One emitted chat row (same schema) + its provenance."""

    messages: list[dict] = Field(default_factory=list)
    provenance: ChunkProvenance


def _content(message: Mapping[str, Any]) -> str:
    value = message.get("content", "")
    return value if isinstance(value, str) else str(value)


def _is_assistant(message: Mapping[str, Any]) -> bool:
    return message.get("role") in TRAINABLE_CHAT_ROLES


def chunk_chat_row(
    messages: Sequence[Mapping[str, Any]],
    *,
    seq_len: int,
    token_len: TokenLen,
    source_row_index: int = 0,
) -> list[ChunkedRow]:
    """Chunk one chat row's ``messages`` into ``<= seq_len`` rows at turn boundaries.

    Passthrough when the whole row already fits. Raises :class:`ChunkingRefusal` when a single turn plus
    the carried system preamble exceeds ``seq_len``.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    messages = list(messages)
    lengths = [token_len(_content(m)) for m in messages]
    total = sum(lengths)

    # Leading system messages are the carried preamble (context, not supervised).
    preamble_end = 0
    while preamble_end < len(messages) and messages[preamble_end].get("role") == "system":
        preamble_end += 1
    preamble = messages[:preamble_end]
    preamble_tokens = sum(lengths[:preamble_end])
    body = messages[preamble_end:]
    body_lengths = lengths[preamble_end:]

    if total <= seq_len:
        return [ChunkedRow(
            messages=[dict(m) for m in messages],
            provenance=ChunkProvenance(
                source_row_index=source_row_index, chunk_index=0, strategy="passthrough",
                body_message_start=0, body_message_end=len(body), duplicated_prompt_context=False,
            ),
        )]

    # Group the body into exchanges: each starts at a user turn (or the first body message) and runs up
    # to just before the next user turn, so a user prompt and its assistant response(s) stay together.
    exchanges: list[tuple[int, int]] = []
    i = 0
    while i < len(body):
        j = i + 1
        while j < len(body) and body[j].get("role") != "user":
            j += 1
        exchanges.append((i, j))
        i = j

    def span_tokens(start: int, end: int) -> int:
        return sum(body_lengths[start:end])

    # Greedily pack whole exchanges (with the preamble) into chunks.
    chunks: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = None
    cur_tokens = 0
    for start, end in exchanges:
        et = span_tokens(start, end)
        if preamble_tokens + et > seq_len:
            raise ChunkingRefusal(
                f"record {source_row_index}: a single turn (body messages {start}-{end - 1}) needs "
                f"{preamble_tokens + et} tokens with the system preamble > seq_len {seq_len}; cannot "
                f"chunk without severing a supervised turn - raise seq_len or shorten the turn"
            )
        if cur is None:
            cur, cur_tokens = (start, end), et
        elif preamble_tokens + cur_tokens + et <= seq_len:
            cur, cur_tokens = (cur[0], end), cur_tokens + et
        else:
            chunks.append(cur)
            cur, cur_tokens = (start, end), et
    if cur is not None:
        chunks.append(cur)

    out: list[ChunkedRow] = []
    for k, (start, end) in enumerate(chunks):
        out.append(ChunkedRow(
            messages=[dict(m) for m in preamble] + [dict(m) for m in body[start:end]],
            provenance=ChunkProvenance(
                source_row_index=source_row_index, chunk_index=k, strategy="chunk",
                body_message_start=start, body_message_end=end,
                duplicated_prompt_context=(k > 0 and preamble_end > 0),
            ),
        ))
    return out


class ChunkPlan(BaseModel):
    """Fail-closed result of chunking a dataset: the emitted rows AND every refused record (never a
    silent drop). ``ok`` is True only when no record was refused."""

    rows: list[ChunkedRow] = Field(default_factory=list)
    refusals: list[str] = Field(default_factory=list)
    source_row_count: int = Field(ge=0)

    @property
    def ok(self) -> bool:
        return not self.refusals


def chunk_chat_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    seq_len: int,
    token_len: TokenLen,
) -> ChunkPlan:
    """Chunk every chat row. A record that cannot be chunked is collected as a refusal (fail closed) -
    the plan is never partially applied silently."""
    emitted: list[ChunkedRow] = []
    refusals: list[str] = []
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list):
            refusals.append(f"record {index}: not a chat row (no 'messages' list)")
            continue
        try:
            emitted.extend(chunk_chat_row(
                messages, seq_len=seq_len, token_len=token_len, source_row_index=index))
        except ChunkingRefusal as refusal:
            refusals.append(str(refusal))
    return ChunkPlan(rows=emitted, refusals=refusals, source_row_count=len(rows))
