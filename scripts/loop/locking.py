"""A portable advisory file lock for the loop's shared, cross-process resources (the learning ledger).

Same discipline as the rest of the loop: stdlib-only, fail-closed. An exclusive lock is a lockfile created
with ``O_CREAT | O_EXCL`` (an atomic "create only if absent" on POSIX and Windows); it is released by
unlinking it. Two guards keep a lock from wedging a campaign:

  * ``timeout`` bounds acquisition - a LIVE contended holder makes the waiter fail closed
    (:class:`LockTimeout`) rather than block forever.
  * ``stale_after`` breaks a lock whose file is older than the bound - a holder that CRASHED without
    releasing cannot deadlock the next process.

This is best-effort ADVISORY mutual exclusion between cooperating loop processes (they all take the lock
before a read-modify-write). It is not a kernel mandatory lock and does not defend against a process that
writes the protected file without taking the lock.

Staleness is deliberately CROSS-PROCESS, so it can only use wall-clock time (the lockfile's ``st_mtime``
vs ``time.time()``) - a monotonic clock is per-process and not comparable between the holder and the
waiter. That makes stale-breaking approximate and clock-dependent: a wall-clock jump forward (e.g. an NTP
step) can age a live lock early, and coarse mtime resolution can delay it. Keep ``stale_after`` well above
the longest legitimate hold (the default 60s dwarfs a ledger append), so only a genuinely crashed holder
is broken; a backward clock jump yields a negative age and is treated as NOT stale (the lock is kept).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import TracebackType


class LockError(Exception):
    """Base class for lock failures (fail-closed)."""


class LockTimeout(LockError):
    """The lock was held by a live process for longer than ``timeout`` - the caller fails closed."""


class FileLock:
    """An advisory, cross-process exclusive lock keyed on ``<path>.lock``. Use as a context manager::

        with FileLock(ledger_path):
            ...  # read-modify-write the protected file
    """

    def __init__(self, path: Path | str, *, timeout: float = 10.0, poll: float = 0.05,
                 stale_after: float = 60.0) -> None:
        self.lock_path = Path(f"{path}.lock")
        self.timeout = timeout
        self.poll = poll
        self.stale_after = stale_after
        self._fd: int | None = None

    def acquire(self) -> "FileLock":
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                # O_EXCL makes this the atomic winner: exactly one creator succeeds when the file is absent.
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                self._break_if_stale()
                if time.monotonic() >= deadline:
                    raise LockTimeout(
                        f"could not acquire {self.lock_path} within {self.timeout}s (held by another process)")
                time.sleep(self.poll)
                continue
            try:
                os.write(fd, f"{os.getpid()} {time.time()}\n".encode("ascii"))
            finally:
                self._fd = fd  # own it even if the diagnostic write fails, so release() cleans up
            return self

    def _break_if_stale(self) -> None:
        """Remove the lockfile if it is older than ``stale_after`` (its holder is presumed crashed). The
        age is wall-clock (``time.time()`` vs ``st_mtime``) because staleness is cross-process - see the
        module docstring for the clock-skew caveat. A negative age (future mtime from a backward clock
        step) is NOT stale, so the lock is kept rather than broken under a clock jump."""
        try:
            age = time.time() - self.lock_path.stat().st_mtime
        except FileNotFoundError:
            return  # already gone - the next O_EXCL create will win
        if age > self.stale_after:  # (a negative age never exceeds a positive stale_after -> kept)
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass  # someone else broke or released it first

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass  # already released / broken as stale - releasing is idempotent

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.release()
