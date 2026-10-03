"""The dataset version-store transaction lock (#859).

The row store (``dataset_versions/row_store.jsonl``) and the per-version manifests
(``dataset_versions/<version_id>.rows``) form ONE shared structure: a version is restorable only
while every row its manifest names is in the store. Two protocols mutate that structure:

* **capture / publication** - append the dataset's new rows to the store, publish the manifest,
  commit the record (:func:`~corpus_studio.versions.version_registry.publish_dataset_version`);
* **GC** - read every manifest (the live set), read the store, atomically replace it without the
  unreferenced rows (:func:`~corpus_studio.versions.gc.gc_row_store`).

Unserialized, a capture that appends and publishes between GC's manifest scan and its replace has
its rows pruned (or dropped by the replace), leaving a published version that can never be
restored. Both protocols therefore run under this single cross-process lock,
``dataset_versions/.version_store.lock``: publication holds it from the first store append through
the manifest (and record) write, and GC holds it from the manifest scan through the replace. Every
other row-store append takes it too. Readers (diff, ``restore --output``, suite version pins) stay
lock-free: a published manifest's rows are never pruned while that manifest exists, and GC swaps the
store with an atomic ``os.replace``.

Properties, from :func:`corpus_studio.storage.file_lock.exclusive_file_lock`: portable, a bounded
wait (``DEFAULT_VERSION_STORE_LOCK_TIMEOUT_SECONDS``, read at call time) that ends in
:class:`VersionStoreBusyError`, same-thread reentrant, and released by the kernel when the holder
exits or crashes, so an interrupted writer never needs the lock file deleted.

**Lock order.** The ``examples.jsonl`` single-writer lock
(:func:`corpus_studio.storage.examples_writer.single_writer_lock`, non-blocking) is always acquired
BEFORE this lock, never while holding it. Capture, publication and GC never request the writer lock,
so the order is acyclic; together with the non-blocking writer lock and the bounded wait here, lock
nesting cannot deadlock.

Stdlib-only (torch-free).
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator

from corpus_studio.storage.file_lock import (
    FileLockError,
    FileLockTimeoutError,
    exclusive_file_lock,
)
from corpus_studio.versions.version_registry import registry_dir

VERSION_STORE_LOCK_FILENAME = ".version_store.lock"
# Generous: GC and capture of a large dataset legitimately hold the lock for a while, and a refusal
# is fail-closed and retryable. Read at call time so tests can shorten it.
DEFAULT_VERSION_STORE_LOCK_TIMEOUT_SECONDS = 60.0


class VersionStoreLockError(RuntimeError):
    """The version-store lock could not be acquired; the operation did not start."""


class VersionStoreBusyError(VersionStoreLockError):
    """Another capture or GC held the version-store lock past the bounded wait."""


def version_store_lock_path(project_dir: Path | str) -> Path:
    """``dataset_versions/.version_store.lock`` with the directory resolved, so a symlinked project
    or ``dataset_versions`` directory is not mistaken for a planted lock-directory link."""

    return registry_dir(project_dir).resolve(strict=False) / VERSION_STORE_LOCK_FILENAME


@contextmanager
def version_store_lock(
    project_dir: Path | str, *, operation: str, timeout_seconds: float | None = None
) -> Iterator[None]:
    """Hold the project's version-store lock for the ``with`` body.

    Raises :class:`VersionStoreBusyError` when another capture or GC holds it past the wait, and
    :class:`VersionStoreLockError` when the lock file is unavailable; in both cases the body never
    ran. Only acquisition failures are translated: errors raised by the body propagate unchanged.
    """

    timeout = (
        DEFAULT_VERSION_STORE_LOCK_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    )
    path = version_store_lock_path(project_dir)
    stack = ExitStack()
    try:
        stack.enter_context(exclusive_file_lock(path, timeout_seconds=timeout))
    except FileLockTimeoutError:
        raise VersionStoreBusyError(
            f"the dataset version store is busy: another capture or GC holds {path}; could not "
            f"start {operation} within {timeout:g}s. Retry when it finishes; do not delete the "
            "lock file."
        ) from None
    except FileLockError as exc:
        raise VersionStoreLockError(
            f"the dataset version store lock is unavailable ({exc}); refusing {operation}."
        ) from exc
    with stack:
        yield
