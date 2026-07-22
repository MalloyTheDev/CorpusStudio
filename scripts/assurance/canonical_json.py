"""Canonical JSON serialization for CorpusStudio assurance records (Phase 1 kernel).

This is a deliberately NARROW, self-describing profile used only by the assurance subsystem
under ``scripts/``. It is intentionally DISTINCT from the engine's internal content-hash form
(``corpus_studio.platform.trace_records.canonical_sha256``, which uses
``json.dumps(sort_keys=True, separators=(",", ":"))`` with ``ensure_ascii`` at its ``True``
default and stores a bare 64-hex digest). The assurance profile differs on two points, on
purpose:

  * ``ensure_ascii=False`` - assurance records embed repository-relative paths and human text
    verbatim as UTF-8, so records stay diff-readable instead of ``\\uXXXX``-escaped.
  * digests carry an explicit ``sha256:`` algorithm prefix - the assurance trust model reasons
    about digest algorithms as first-class (record integrity vs applicability), so every digest
    is self-describing and algorithm-agile.

Determinism guarantees enforced here (fail closed on violation):
  * object keys sorted lexicographically; compact separators; UTF-8 bytes.
  * NO float anywhere in the payload (floats are not portably deterministic); NaN/Inf are
    impossible by construction (``allow_nan=False`` plus the float rejection below).
  * only JSON-native container/scalar types are allowed; anything else fails closed.

Arrays keep their given order - callers are responsible for sorting semantically unordered
lists (e.g. changed paths) before handing them here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Every assurance digest is emitted with this algorithm prefix (self-describing; see module doc).
DIGEST_PREFIX = "sha256:"


class CanonicalJsonError(ValueError):
    """A payload cannot be canonicalized deterministically (float / unsupported type / bad key)."""


def _reject_nondeterministic(value: Any, path: str = "$") -> None:
    """Walk ``value`` and fail closed on anything that would break byte-determinism.

    ``bool`` is checked before ``int`` because ``bool`` is a subclass of ``int`` (both are
    allowed, but the order keeps the type names honest). ``float`` is rejected outright.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise CanonicalJsonError(f"float is not allowed in a canonical assurance record (at {path})")
    if isinstance(value, dict):
        for key, sub in value.items():
            if not isinstance(key, str):
                raise CanonicalJsonError(f"object key must be a string (at {path})")
            _reject_nondeterministic(sub, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, sub in enumerate(value):
            _reject_nondeterministic(sub, f"{path}[{index}]")
        return
    raise CanonicalJsonError(
        f"unsupported type {type(value).__name__} in a canonical assurance record (at {path})"
    )


def canonical_dumps(payload: Any) -> str:
    """Return the canonical JSON text of ``payload`` (fails closed on non-determinism)."""
    _reject_nondeterministic(payload)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(payload: Any) -> bytes:
    """Return the UTF-8 canonical bytes of ``payload``."""
    return canonical_dumps(payload).encode("utf-8")


def sha256_digest(payload: Any) -> str:
    """Return the ``sha256:``-prefixed digest of the canonical bytes of ``payload``."""
    return DIGEST_PREFIX + hashlib.sha256(canonical_bytes(payload)).hexdigest()


def sha256_of_bytes(data: bytes) -> str:
    """Return the ``sha256:``-prefixed digest of raw bytes (file content or a link target)."""
    return DIGEST_PREFIX + hashlib.sha256(data).hexdigest()
