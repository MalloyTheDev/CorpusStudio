"""Portable, bounded, same-thread-reentrant cross-process file lock (stdlib only).

:func:`exclusive_file_lock` excludes other threads of this process (one process-local
``RLock`` per resolved lock path) and other processes (``fcntl.flock`` on POSIX,
``msvcrt.locking`` on Windows). The OS lock belongs to the open file description, so the
kernel drops it when the holder exits or crashes: a dead holder never leaves a stale lock,
and recovery never requires deleting the lock file.

Invariants callers rely on:

* **Bounded.** Acquisition waits at most ``timeout_seconds`` and then raises
  :class:`FileLockTimeoutError`; it never blocks forever.
* **Same-thread reentrant.** A thread that already holds the lock re-enters it, so nested
  critical sections compose. The flip side: code on the SAME thread is never excluded, so
  two critical sections that must exclude each other have to run on different threads or
  processes.
* **Hardened open.** The lock file is opened without following a final-component symlink
  and must be one singly linked regular file owned by the current user, so a planted link
  cannot redirect the lock onto another file.
* **Do not fork while holding it.** The process-local state and the locked file
  description are inherited by a forked child; spawn subprocesses instead.

The algorithm mirrors ``platform.environment_manager._exclusive_file_lock``. It lives
outside ``platform/`` so the data plane can serialize its own stores without importing the
environment manager. Torch-free.
"""

from __future__ import annotations

import errno
import math
import os
import stat
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, cast

if sys.platform == "win32":  # pragma: no cover - Windows-only branch; CI runs Linux
    import msvcrt
else:
    import fcntl

DEFAULT_POLL_SECONDS = 0.05

_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCK_STATE = threading.local()
# errno values meaning "another holder has it" for a non-blocking lock attempt.
_CONTENDED_ERRNOS = frozenset({errno.EACCES, errno.EAGAIN, errno.EDEADLK})


class FileLockError(RuntimeError):
    """The lock file could not be opened or locked safely."""


class FileLockTimeoutError(FileLockError):
    """Another holder kept the lock for longer than the bounded wait."""


def _process_lock_for(key: str) -> threading.RLock:
    with _PROCESS_LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(key, threading.RLock())


def _thread_held_locks() -> set[str]:
    held = getattr(_THREAD_LOCK_STATE, "held", None)
    if held is None:
        held = set()
        _THREAD_LOCK_STATE.held = held
    return cast(set[str], held)


def _open_lock_stream(path: Path) -> BinaryIO:
    """Open one owned regular lock file without following a final-component symlink."""

    lock_dir = path.parent
    try:
        lock_dir.mkdir(parents=True, exist_ok=True)
        if lock_dir.is_symlink():
            raise FileLockError("lock directory cannot be a symbolic link")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise FileLockError(f"lock file is unavailable: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        linked = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
        ):
            raise FileLockError("lock file must be one singly linked regular file")
        if hasattr(os, "getuid") and opened.st_uid != os.getuid():
            raise FileLockError("lock file is owned by another user")
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        if opened.st_size == 0:
            # msvcrt locks a byte range, so the file needs at least one byte to lock.
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return os.fdopen(descriptor, "r+b", buffering=0)
    except Exception as exc:
        os.close(descriptor)
        if isinstance(exc, OSError):
            raise FileLockError(f"lock file is unavailable: {exc}") from exc
        raise


def _try_os_lock(stream: BinaryIO) -> bool:
    try:
        stream.seek(0)
        if sys.platform == "win32":  # pragma: no cover - Windows-only branch
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in _CONTENDED_ERRNOS:
            return False
        raise FileLockError(f"lock file could not be locked: {exc}") from exc


def _unlock_os_lock(stream: BinaryIO) -> None:
    stream.seek(0)
    if sys.platform == "win32":  # pragma: no cover - Windows-only branch
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def exclusive_file_lock(path: Path, *, timeout_seconds: float) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` for the ``with`` body.

    Raises ``ValueError`` unless ``timeout_seconds`` is finite and positive,
    :class:`FileLockTimeoutError` when the lock stays held elsewhere past the timeout, and
    :class:`FileLockError` when the lock file cannot be opened or locked safely. Errors
    raised by the body propagate unchanged; the lock is released on every exit path.
    """

    if not (math.isfinite(timeout_seconds) and timeout_seconds > 0):
        raise ValueError("lock timeout must be a finite number of seconds greater than zero")
    key = str(Path(path).resolve(strict=False))
    deadline = time.monotonic() + timeout_seconds
    process_lock = _process_lock_for(key)
    if not process_lock.acquire(timeout=timeout_seconds):
        raise FileLockTimeoutError(
            f"{key} stayed locked by another thread for {timeout_seconds:g}s"
        )
    held = _thread_held_locks()
    if key in held:
        try:
            yield
        finally:
            process_lock.release()
        return

    stream: BinaryIO | None = None
    os_locked = False
    try:
        stream = _open_lock_stream(Path(path))
        while not (os_locked := _try_os_lock(stream)):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FileLockTimeoutError(
                    f"{key} stayed locked by another process for {timeout_seconds:g}s"
                )
            time.sleep(min(DEFAULT_POLL_SECONDS, remaining))
        held.add(key)
        yield
    finally:
        held.discard(key)
        if stream is not None:
            if os_locked:
                try:
                    _unlock_os_lock(stream)
                except OSError:
                    # Closing the descriptor releases the OS lock even if an explicit unlock fails.
                    pass
            stream.close()
        process_lock.release()
