"""A portable advisory file lock for the loop's shared, cross-process resources (the learning ledger).

Same discipline as the rest of the loop: stdlib-only, fail-closed. An exclusive lock is a lockfile created
with ``O_CREAT | O_EXCL`` (an atomic "create only if absent" on POSIX and Windows); it is released by
unlinking it. Two guards keep a lock from wedging a campaign:

  * ``timeout`` bounds acquisition - a LIVE contended holder makes the waiter fail closed
    (:class:`LockTimeout`) rather than block forever.
  * a crashed holder is broken so it cannot deadlock the next process - but ONLY when it is genuinely
    gone. Stale-breaking is PID-liveness first: the lockfile records the holder's PID, and a waiter breaks
    the lock only if that PID is DEAD on this host (``os.kill(pid, 0)`` -> ``ProcessLookupError``). A LIVE
    holder is never broken, even a long-running one whose file has aged past ``stale_after`` - so this lock
    is safe to hold across a whole loop run, not just a quick ledger append.

This is best-effort ADVISORY mutual exclusion between cooperating loop processes (they all take the lock
before a read-modify-write). It is not a kernel mandatory lock and does not defend against a process that
writes the protected file without taking the lock.

PID-liveness assumes a SAME-HOST holder (the loop's operational state lives on the local host). When the
PID cannot be probed (a malformed lockfile, or a holder on another host over a shared FS), stale-breaking
falls back to the wall-clock ``st_mtime`` vs ``time.time()`` bound - approximate and clock-dependent (an
NTP step forward can age it early; a backward jump yields a negative age and is treated as NOT stale).
Keep ``stale_after`` well above the longest legitimate hold. The break itself is ATOMIC (rename-to-claim),
so two waiters can never both break a lock and the second delete a lock a third process just re-created.
"""

from __future__ import annotations

import os
import secrets
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
        self._token: str | None = None  # a per-acquisition owner token, so release removes only OUR lock

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
            token = secrets.token_hex(16)  # unforgeable owner id for this acquisition
            try:
                os.write(fd, f"{os.getpid()} {time.time()} {token}\n".encode("ascii"))
            except OSError:
                # The write failed, so the lockfile would carry no token and release() could never clean
                # it up (a leaked lock). Treat this as a FAILED acquisition: drop the empty lockfile we
                # created and propagate - never own a lock we cannot later prove is ours.
                os.close(fd)
                try:
                    self.lock_path.unlink()
                except FileNotFoundError:
                    pass
                raise
            self._fd, self._token = fd, token  # own it only AFTER the token is durably written
            return self

    def _holder_pid(self) -> int | None:
        """The PID recorded in the lockfile (``'<pid> <ts> <token>'``), or None if absent / malformed."""
        try:
            fields = self.lock_path.read_text("ascii").split()
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            return None
        if len(fields) != 3:
            return None
        try:
            return int(fields[0])
        except ValueError:
            return None

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        """Whether a SAME-HOST holder process is still alive (``os.kill(pid, 0)``). This lets a live holder
        keep its lock even past ``stale_after`` (a long-running writer is never wrongly broken). A holder on
        another host (shared FS) cannot be probed and is handled by the mtime fallback in the caller."""
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False           # the holder crashed / exited without releasing
        except (PermissionError, OSError):
            return True            # PID exists (other user) / unknown -> fail SAFE: never break a live lock
        return True

    def _break_if_stale(self) -> None:
        """Break a lock ONLY when its holder is genuinely gone, and break it ATOMICALLY.

        Liveness first: if the recorded PID is ALIVE on this host, keep the lock at any age (a live
        long-running writer must not be broken). Break only when the PID is DEAD (crashed), or - when the
        PID is unknowable (malformed / cross-host holder) - when the wall-clock mtime is older than
        ``stale_after`` (a negative age from a backward clock step is NOT stale). The break itself is atomic
        (:meth:`_atomic_break`), fixing the TOCTOU where two waiters both ``unlink`` and the second deletes
        a lock a third process just freshly created."""
        pid = self._holder_pid()
        if pid is not None:
            if self._pid_is_alive(pid):
                return  # live holder -> keep; the waiter times out (LockTimeout) rather than break it
            stale = True  # holder PID is dead: crashed without releasing
        else:
            try:
                stale = (time.time() - self.lock_path.stat().st_mtime) > self.stale_after
            except FileNotFoundError:
                return  # already gone - the next O_EXCL create will win
        if stale:
            self._atomic_break()

    def _atomic_break(self) -> None:
        """Remove a confirmed-stale lockfile by ATOMICALLY CLAIMING it first: rename it to a unique name.
        ``os.rename`` is atomic, so exactly one racing waiter wins the claim; a loser's rename fails
        (source gone) and it never deletes anyone's lock. If the claimed file turns out to be a LIVE lock
        freshly re-created in the race window, it is restored (via an ``O_EXCL`` create that will not
        clobber a newer lock) rather than broken."""
        claimed = self.lock_path.with_name(f"{self.lock_path.name}.stale-{secrets.token_hex(8)}")
        try:
            os.rename(self.lock_path, claimed)  # only ONE waiter wins this; the rest fail here (safe)
        except (FileNotFoundError, OSError):
            return  # already claimed / broken by another waiter - do not touch the current lock
        try:
            pid = int(claimed.read_text("ascii").split()[0])
        except (OSError, UnicodeDecodeError, ValueError, IndexError):
            pid = -1  # unreadable/malformed -> treat as not-alive -> safe to discard
        if pid >= 0 and self._pid_is_alive(pid):
            # rare fresh/live capture: restore it WITHOUT clobbering a newer lock (O_EXCL create-only).
            try:
                content = claimed.read_bytes()
            except OSError:
                content = b""
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                pass  # a newer lock already holds the slot - leave it
            else:
                try:
                    os.write(fd, content)
                finally:
                    os.close(fd)
        try:
            claimed.unlink()
        except FileNotFoundError:
            pass

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        token, self._token = self._token, None
        if token is None:
            return
        # Remove the lockfile ONLY if we still OWN it. If another process broke our stale lock and
        # re-created its own, the lockfile now carries THAT owner's token - our late release must not
        # delete it. Parse the lockfile ("<pid> <timestamp> <token>") and compare the token field EXACTLY
        # (not a substring, which could match the wrong line); a malformed lockfile is not treated as ours.
        try:
            content = self.lock_path.read_text("ascii")
        except (FileNotFoundError, OSError):
            return  # already gone / unreadable - nothing of ours to remove
        fields = content.split()
        if len(fields) == 3 and fields[2] == token:
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass  # already released / broken as stale - releasing is idempotent

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.release()
