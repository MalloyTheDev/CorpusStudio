"""Consumption-time verification of sealed execution inputs, shared by every first-party worker lane.

A RunPlan seals ``inputs.dataset.content_sha256`` when it is planned. A worker that later reopens the
dataset path trains on whatever bytes are there at that moment, so the seal would name one dataset while
the run consumed another. This module is the single place a lane turns a sealed dataset binding into rows:

1. read the file ONCE with :func:`~corpus_studio.platform.execution_config.stable_file_bytes` (stat
   identity before, at and after open; a read bounded to the size seen at open; an incremental sha256;
   a capture of the exact bytes),
2. compare that digest with the sealed ``content_sha256``,
3. parse THOSE captured bytes with :func:`~corpus_studio.importers.jsonl_importer.read_jsonl_bytes`.

The path is never read a second time, so there is no window between the check and the use, and a large
dataset costs one I/O pass rather than a verification pass plus a consumption pass.

Who calls what:

* the adapter SFT trainer delegates the dataset half of ``trainer.verify_sealed_runtime`` to
  :func:`verify_sealed_dataset_bytes` and parses the returned bytes itself;
* the DPO, reward, full-parameter SFT and on-policy RL runners call :func:`read_verified_dataset` before
  they dispatch the worker, so a refusal precedes any heavy import, tokenizer or model load, and output
  directory; the worker then accepts only a :class:`VerifiedDataset` bound to its own sealed binding
  (:func:`require_verified_dataset`).

Every refusal is a :class:`SealedInputError` with an ASCII message; the caller maps it to its own
taxonomy. Torch-free by construction: this module imports only the contracts, the execution-config
read primitives and the JSONL importer. Consumption checks for the other sealed inputs (the model and
tokenizer bindings) belong here as well, so every lane shares one implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from corpus_studio.importers.jsonl_importer import read_jsonl_bytes
from corpus_studio.platform.contracts import ExecutionInputBinding
from corpus_studio.platform.execution_config import (
    ExecutionConfigurationError,
    stable_file_bytes,
)

# Byte progress from a multi-GB read arrives once per 1 MiB chunk. Forwarding every chunk would flood
# the run's event stream; forwarding none would let a subprocess parent's silence timer expire during a
# long, healthy read. At most this many progress events are emitted per read.
MAX_DATASET_PROGRESS_EVENTS = 20


class SealedInputError(ValueError):
    """A sealed execution input is missing, changed, unreadable or malformed at consumption time."""


@dataclass(frozen=True)
class VerifiedDataset:
    """Rows parsed from the exact bytes whose sha256 matched the sealed dataset binding.

    Only :func:`read_verified_dataset` constructs one in production. ``content_sha256`` and
    ``byte_count`` describe the bytes that were hashed AND parsed; ``location`` is the binding's
    location, so a worker can confirm the rows belong to its own execution.
    """

    rows: tuple[dict[str, Any], ...]
    content_sha256: str
    byte_count: int
    location: str

    @property
    def row_count(self) -> int:
        return len(self.rows)


def verify_sealed_dataset_bytes(
    location: str,
    expected_sha256: str | None,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[bytes, str]:
    """Read ``location`` once and return its exact bytes and digest, or refuse.

    Refuses a missing digest, a missing/linked/replaced file, a change observed during the read, and
    any digest other than ``expected_sha256``. The caller must parse the returned bytes and must not
    reopen ``location``.
    """

    if expected_sha256 is None:
        raise SealedInputError("sealed execution omitted the dataset content digest")
    try:
        content, observed = stable_file_bytes(location, progress_callback=progress_callback)
    except ExecutionConfigurationError as exc:
        raise SealedInputError(str(exc)) from exc
    if observed != expected_sha256:
        raise SealedInputError("dataset bytes changed after the execution configuration was sealed")
    return content, observed


def read_verified_dataset(
    binding: ExecutionInputBinding,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> VerifiedDataset:
    """Verify a sealed local dataset binding and return the rows parsed from the verified bytes.

    The digest is compared with ``binding.content_sha256`` (the stable-read digest sealed at planning),
    never with ``binding.ref.hash``, which may identify a logical dataset version rather than these
    bytes. A malformed or row-less file is refused even though its bytes match the seal.
    """

    if binding.kind != "dataset" or binding.source != "local_file":
        raise SealedInputError("the sealed dataset binding is not a pinned local dataset file")
    content, digest = verify_sealed_dataset_bytes(
        binding.location, binding.content_sha256, progress_callback=progress_callback
    )
    byte_count = len(content)
    try:
        rows = tuple(read_jsonl_bytes(content))
    except ValueError as exc:
        raise SealedInputError(f"sealed dataset is invalid: {exc}") from exc
    finally:
        del content
    if not rows:
        raise SealedInputError("the sealed dataset contains no rows")
    return VerifiedDataset(
        rows=rows, content_sha256=digest, byte_count=byte_count, location=binding.location
    )


def require_verified_dataset(dataset: object, binding: ExecutionInputBinding) -> VerifiedDataset:
    """The worker-side guard: accept only a :class:`VerifiedDataset` produced for ``binding``.

    A worker called without the runner's verified read (or with rows verified for a different sealed
    execution) must not fall back to reading the mutable path; it refuses instead.
    """

    if not isinstance(dataset, VerifiedDataset):
        raise SealedInputError("the worker requires rows from the verified sealed dataset read")
    if dataset.content_sha256 != binding.content_sha256 or dataset.location != binding.location:
        raise SealedInputError("the verified dataset does not belong to this sealed execution")
    return dataset


def bounded_byte_progress(
    emit: Callable[[str], object],
    *,
    max_events: int = MAX_DATASET_PROGRESS_EVENTS,
) -> Callable[[int, int], None]:
    """Adapt ``(completed, total)`` byte progress into at most ``max_events`` monotonic messages.

    Each message reports real bytes read and hashed, so it is honest progress for a silence timer, not a
    synthetic heartbeat. The final chunk always reports ``total/total``.
    """

    last_bucket = 0

    def _progress(completed: int, total: int) -> None:
        nonlocal last_bucket
        if total <= 0:
            return
        bucket = min(max_events, max(1, completed * max_events // total))
        if bucket <= last_bucket:
            return
        last_bucket = bucket
        emit(f"read and hashed {completed}/{total} sealed dataset bytes")

    return _progress
