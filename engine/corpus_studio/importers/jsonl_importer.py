import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# orjson is a declared-but-OPTIONAL accelerator (see pyproject.toml). When it is
# importable we use it on the hot per-line path (several times faster on large
# datasets); when it is absent the engine falls back to the stdlib so it stays
# dependency-light and runnable anywhere. Both raise a ValueError subclass on a
# malformed line (stdlib: json.JSONDecodeError; orjson: orjson.JSONDecodeError,
# which is itself a subclass of json.JSONDecodeError), so every caller's
# (OSError, ValueError) handler keeps working unchanged either way.
try:  # pragma: no cover - the chosen branch depends on the environment
    import orjson

    def _loads(line: str) -> Any:
        return orjson.loads(line)

except ImportError:  # pragma: no cover - stdlib fallback path

    def _loads(line: str) -> Any:
        return json.loads(line)


@dataclass(frozen=True)
class JsonlLine:
    """One non-blank physical line of a JSONL file, parsed leniently.

    Exactly one of the parse outcomes is meaningful: when ``error`` is ``None``
    the line decoded and ``value`` holds the JSON value (any type); when
    ``error`` is set it is a human-readable ``"Invalid JSON: ..."`` reason and
    ``value`` is ``None``.

    Object-shape validation (a dataset row must be a JSON *object*) is left to
    the caller on purpose: strict readers raise on a non-object line while the
    tolerant import/validation readers record it — but both decisions are made
    off this one shared, identically-configured parse, so encoding, blank-line
    skipping, and malformed-line wording can never drift between them.
    """

    line_number: int
    raw: str
    value: Any
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


def iter_jsonl(path: Path) -> Iterator[JsonlLine]:
    """Stream a JSONL file leniently, one :class:`JsonlLine` per non-blank line.

    This is the single shared reader primitive. It defines — once — that the
    file is read as ``utf-8-sig`` (tolerating a leading BOM from Excel/Windows
    exports), that blank/whitespace-only lines are skipped, and that a malformed
    line is reported (never raised) so tolerant callers can collect every bad
    line in one pass. It intentionally does NOT reject non-object rows; that is
    the caller's policy.
    """
    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                value = _loads(line)
            except ValueError as exc:
                # Covers stdlib json.JSONDecodeError and orjson.JSONDecodeError
                # (both ValueError subclasses). A non-JSON-error such as a
                # RecursionError from pathologically nested JSON deliberately
                # propagates, matching the previous behavior.
                yield JsonlLine(line_number, line, None, f"Invalid JSON: {exc}")
            else:
                yield JsonlLine(line_number, line, value, None)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Strict streaming reader: yield each row as a ``dict``, raising ``ValueError``
    on the first malformed or non-object line.

    Consumers that must not proceed on bad data (training / evaluation / gates /
    fingerprinting) rely on this fail-fast contract. Import preview and file
    validation use :func:`iter_jsonl` directly so they can tolerate and report
    bad lines instead.
    """
    for parsed in iter_jsonl(path):
        if parsed.error is not None:
            raise ValueError(f"line {parsed.line_number}: {parsed.error}")
        if not isinstance(parsed.value, dict):
            # A JSONL dataset row must be a JSON object. A non-object line
            # (list/scalar/null) is malformed input — raise a ValueError so
            # callers' (OSError, ValueError) handlers surface it cleanly instead
            # of a downstream AttributeError from row.values()/row.items().
            raise ValueError(
                f"line {parsed.line_number}: expected a JSON object, "
                f"got {type(parsed.value).__name__}."
            )
        yield parsed.value


def read_jsonl_bytes(content: bytes) -> Iterator[dict[str, Any]]:
    """Strictly parse already-stabilized JSONL bytes without reopening a mutable path."""

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"dataset is not valid UTF-8: {exc}") from exc
    for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
        if not line.strip():
            continue
        try:
            value = _loads(line)
        except ValueError as exc:
            raise ValueError(f"line {line_number}: Invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"line {line_number}: expected a JSON object, got {type(value).__name__}."
            )
        yield value
