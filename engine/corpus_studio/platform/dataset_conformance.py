"""Structural dataset-format conformance preflight (torch-free, control-plane).

A RunPlan seals a ``dataset_format``, but a syntactically valid plan is not necessarily a semantically
executable one. The first-party formatter (``training.trainer.format_example_text``) silently returns
"" for a row it cannot render, and the caller drops those rows; a plan whose sealed format does not
match the dataset's structure therefore renders ZERO usable rows and fails only AFTER model allocation
on the GPU (``UNSUPPORTED_CONFIGURATION`` / "The dataset produced no usable training rows.").

This module lets the planner refuse such a plan at planning time, on the CPU, before any plan id is
minted. It mirrors the formatter's per-format structural contract WITHOUT importing torch or the worker:

* ``instruction`` (Alpaca): a row is compatible when ``instruction`` or ``output`` has text - exactly
  the condition under which ``format_example_text`` renders a non-empty string.
* ``chat``: stricter than the formatter's minimal "non-empty messages list" so a structurally useless
  row is caught early - a compatible row has a non-empty ``messages`` list whose every message is an
  object with a recognized role and non-empty content, including at least one trainable assistant turn.
* ``trace``: a trace-record row, or a chat row with an assistant turn (mirrors ``traces.trace_from_row``).

It never reinterprets one format as another, never auto-switches the format, and never rewrites dataset
bytes. The dataset is read once, read-only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

SUPPORTED_DATASET_FORMATS: tuple[str, ...] = ("instruction", "chat", "trace")
RECOGNIZED_CHAT_ROLES = frozenset({"system", "user", "assistant", "tool"})
TRAINABLE_CHAT_ROLES = frozenset({"assistant"})
_MAX_REPRESENTATIVE_REJECTIONS = 5


class DatasetConformanceError(ValueError):
    """The dataset could not be read, or its selected format is unknown."""


@dataclass(frozen=True)
class RowRejection:
    """One structurally incompatible row: its 0-based index and a concrete reason."""

    index: int
    reason: str


@dataclass(frozen=True)
class DatasetFormatConformance:
    """The structural verdict for a dataset against a selected ``dataset_format``."""

    dataset_format: str
    total_rows: int
    compatible_rows: int
    rejected_rows: int
    representative_rejections: tuple[RowRejection, ...]

    @property
    def is_conformant(self) -> bool:
        """At least one row can be rendered into a usable training example."""
        return self.compatible_rows > 0

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable report (no new root contract; a plain, stable mapping)."""
        return {
            "dataset_format": self.dataset_format,
            "total_rows": self.total_rows,
            "compatible_rows": self.compatible_rows,
            "rejected_rows": self.rejected_rows,
            "representative_rejections": [
                {"index": rejection.index, "reason": rejection.reason}
                for rejection in self.representative_rejections
            ],
        }

    def describe_refusal(self, dataset_path: str) -> str:
        """The exact, ASCII, single-line refusal message identifying the mismatch."""
        reasons = "; ".join(
            f"row {rejection.index}: {rejection.reason}"
            for rejection in self.representative_rejections
        )
        return (
            f"dataset '{dataset_path}' is structurally incompatible with the requested dataset_format "
            f"'{self.dataset_format}': of {self.total_rows} row(s), {self.compatible_rows} are "
            f"structurally compatible and {self.rejected_rows} were rejected. Planning is refused "
            f"because no row renders into a usable training example. Representative rejections: "
            f"{reasons or '(none)'}. The dataset bytes are unchanged and the format was NOT "
            f"auto-switched - select the dataset_format that matches the data (for example 'chat' for "
            f"rows carrying a 'messages' list) or supply a compatible dataset."
        )


def load_jsonl_rows(path: str | Path) -> list[Any]:
    """Read a JSONL dataset into a list of parsed rows (torch-free, read-only).

    Raises ``DatasetConformanceError`` on an unreadable file or a malformed JSON line."""
    rows: list[Any] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                text = raw.strip()
                if not text:
                    continue
                try:
                    rows.append(json.loads(text))
                except json.JSONDecodeError as exc:
                    raise DatasetConformanceError(
                        f"dataset line {line_number} is not valid JSON: {exc}"
                    ) from exc
    except OSError as exc:
        raise DatasetConformanceError(f"cannot read dataset '{path}': {exc}") from exc
    return rows


def _nonempty_content(content: Any) -> bool:
    """True when a message carries non-empty textual content (string or structured content parts)."""
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, Mapping):
        return _nonempty_content(content.get("text", ""))
    if isinstance(content, Sequence):
        return any(_nonempty_content(part) for part in content)
    return False


def _classify_instruction(row: Mapping[str, Any]) -> str | None:
    instruction = str(row.get("instruction", "")).strip()
    output = str(row.get("output", "")).strip()
    if not instruction and not output:
        return "neither 'instruction' nor 'output' has text (Alpaca instruction format)"
    return None


def _classify_chat(row: Mapping[str, Any]) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return "no non-empty 'messages' list"
    has_trainable_turn = False
    for position, message in enumerate(messages):
        if not isinstance(message, Mapping):
            return f"message {position} is not an object"
        role = message.get("role")
        if role not in RECOGNIZED_CHAT_ROLES:
            return f"message {position} has unrecognized role {role!r}"
        if not _nonempty_content(message.get("content")):
            return f"message {position} (role {role!r}) has empty content"
        if role in TRAINABLE_CHAT_ROLES:
            has_trainable_turn = True
    if not has_trainable_turn:
        return "no trainable assistant turn"
    return None


def _classify_trace(row: Mapping[str, Any]) -> str | None:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return _classify_chat(row)
    has_prompt = any(str(row.get(key, "")).strip() for key in ("prompt", "question", "instruction"))
    has_answer = any(
        str(row.get(key, "")).strip() for key in ("answer", "output", "response", "final")
    )
    if has_prompt and has_answer:
        return None
    return (
        "no trace structure (needs a prompt/question and an answer/output, or a chat 'messages' list)"
    )


_CLASSIFIERS: dict[str, Callable[[Mapping[str, Any]], str | None]] = {
    "instruction": _classify_instruction,
    "chat": _classify_chat,
    "trace": _classify_trace,
}


def assess_dataset_format_conformance(
    rows: Iterable[Any], dataset_format: str
) -> DatasetFormatConformance:
    """Structurally classify each row against ``dataset_format`` and count compatibility.

    Never reinterprets the format and never mutates rows. Raises ``DatasetConformanceError`` for an
    unsupported ``dataset_format`` (a plan cannot be sealed for a format the worker cannot render)."""
    classifier = _CLASSIFIERS.get(dataset_format)
    if classifier is None:
        raise DatasetConformanceError(
            f"unknown dataset_format '{dataset_format}'; supported formats: "
            f"{', '.join(SUPPORTED_DATASET_FORMATS)}"
        )
    total = 0
    compatible = 0
    rejections: list[RowRejection] = []
    for index, row in enumerate(rows):
        total += 1
        if not isinstance(row, Mapping):
            reason: str | None = "row is not a JSON object"
        else:
            reason = classifier(row)
        if reason is None:
            compatible += 1
        elif len(rejections) < _MAX_REPRESENTATIVE_REJECTIONS:
            rejections.append(RowRejection(index=index, reason=reason))
    return DatasetFormatConformance(
        dataset_format=dataset_format,
        total_rows=total,
        compatible_rows=compatible,
        rejected_rows=total - compatible,
        representative_rejections=tuple(rejections),
    )


def assess_dataset_file_conformance(
    dataset_path: str | Path, dataset_format: str
) -> DatasetFormatConformance:
    """Load a JSONL dataset file and assess structural conformance against ``dataset_format``."""
    return assess_dataset_format_conformance(load_jsonl_rows(dataset_path), dataset_format)
