"""Optional Parquet support (lazy ``pyarrow``, dependency-light by default).

Corpus Studio's runtime deps stay tiny (pydantic / typer / orjson). Parquet is a
columnar binary format that needs ``pyarrow`` — a large dependency — so it is an
**optional extra** (``pip install corpus-studio-engine[parquet]``), imported lazily
here. When it isn't installed, every Parquet path fails fast with an actionable
message (``ParquetSupportError``) instead of an opaque ``ModuleNotFoundError``, and
the rest of the engine keeps working unchanged.

The import is funnelled through ``_import_pyarrow`` so tests can simulate pyarrow
being absent even in an environment where it happens to be installed.
"""

from __future__ import annotations

from typing import Any

PARQUET_INSTALL_HINT = (
    "Parquet support requires the optional 'pyarrow' dependency. "
    "Install it with: pip install corpus-studio-engine[parquet]"
)


class ParquetSupportError(RuntimeError):
    """Raised when a Parquet operation is requested without ``pyarrow`` installed."""


def _import_pyarrow() -> tuple[Any, Any]:
    """Import and return ``(pyarrow, pyarrow.parquet)``.

    Isolated (and monkeypatchable) so a test can force the not-installed branch.
    Raises ``ImportError`` when pyarrow is missing.
    """
    import pyarrow
    import pyarrow.parquet as pq

    return pyarrow, pq


def load_pyarrow() -> tuple[Any, Any]:
    """Return ``(pyarrow, pyarrow.parquet)`` or raise ``ParquetSupportError`` with the
    install hint. All Parquet read/write code enters through here."""
    try:
        return _import_pyarrow()
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ParquetSupportError(PARQUET_INSTALL_HINT) from exc


def parquet_available() -> bool:
    """True when pyarrow can be imported. Lets the CLI fail fast with the hint before
    doing any work, rather than part-way through an export."""
    try:
        _import_pyarrow()
        return True
    except ImportError:
        return False


def json_default(value: Any) -> Any:
    """``json.dumps`` fallback for the non-JSON scalars a Parquet cell can hold.

    Parquet columns are typed, so a value read back can be ``bytes`` (binary/text
    columns), a ``datetime``/``date``, or a ``Decimal`` — none of which ``json``
    serialises natively. Bytes are decoded as UTF-8 (replacing undecodable bytes,
    since Corpus Studio imports *text* datasets); everything else falls back to its
    string form. Honest boundary: this makes binary columns best-effort text, not a
    faithful round-trip of arbitrary binary payloads (images/audio are out of scope
    for a text-dataset tool)."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)
