"""Tests for the portable advisory file lock (scripts/loop/locking.py).

Pins mutual exclusion (a live holder makes a waiter fail closed), stale-lock breaking (a crashed holder
cannot wedge the next process), idempotent release, and context-manager use.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from loop.locking import FileLock, LockTimeout  # noqa: E402


def test_a_held_lock_makes_a_waiter_fail_closed(tmp_path: Path) -> None:
    held = FileLock(tmp_path / "res", timeout=5).acquire()
    try:
        with pytest.raises(LockTimeout):
            FileLock(tmp_path / "res", timeout=0.2).acquire()  # a live holder -> fail closed, not block forever
    finally:
        held.release()


def test_release_lets_the_lock_be_reacquired(tmp_path: Path) -> None:
    FileLock(tmp_path / "res", timeout=1).acquire().release()
    FileLock(tmp_path / "res", timeout=1).acquire().release()  # no residue after release


def test_release_is_idempotent(tmp_path: Path) -> None:
    lock = FileLock(tmp_path / "res", timeout=1).acquire()
    lock.release()
    lock.release()  # second release is a no-op, not an error


def test_a_stale_lock_is_broken(tmp_path: Path) -> None:
    # A lockfile older than stale_after belongs to a presumed-crashed holder and is broken so the next
    # process is not deadlocked forever.
    lockfile = Path(str(tmp_path / "res") + ".lock")
    lockfile.write_text("999999 0")  # a bogus PID + ancient timestamp
    os.utime(lockfile, (0, 0))  # force an ancient mtime
    with FileLock(tmp_path / "res", timeout=1, stale_after=1.0):
        pass  # acquires by breaking the stale lock


def test_a_future_dated_lock_is_not_broken_as_stale(tmp_path: Path) -> None:
    # A backward wall-clock jump makes an existing lock's mtime look like the future (negative age). That
    # must NOT be treated as stale - the live lock is kept, so the waiter fails closed instead of stealing it.
    lockfile = Path(str(tmp_path / "res") + ".lock")
    lockfile.write_text("111 0")
    future = time.time() + 3600
    os.utime(lockfile, (future, future))
    with pytest.raises(LockTimeout):
        FileLock(tmp_path / "res", timeout=0.2, stale_after=1.0).acquire()


def test_lock_creates_a_missing_parent_directory(tmp_path: Path) -> None:
    # A lock on a NEW resource whose directory does not exist yet (e.g. run_loop taking the state-file
    # lock before the first save() creates the dir) must create the dir and acquire, not FileNotFoundError.
    target = tmp_path / "deep" / "nested" / "state.json"  # parents do not exist
    with FileLock(target, timeout=1) as lock:
        assert lock._token is not None and lock.lock_path.exists()


def _a_dead_pid() -> int:
    import subprocess  # a child that has EXITED -> its PID is not alive (until reused, unlikely in a test)
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_a_live_holders_lock_is_not_broken_even_when_old(tmp_path: Path) -> None:
    # Hardening D: PID-liveness - a lock whose recorded PID is ALIVE must NOT be broken, even past
    # stale_after. So this lock is safe to hold across a long run; a waiter fails closed, never steals it.
    lockfile = Path(str(tmp_path / "res") + ".lock")
    lockfile.write_text(f"{os.getpid()} 1.0 {'a' * 32}\n")  # our own (live) PID, 3-field
    ancient = time.time() - 10_000
    os.utime(lockfile, (ancient, ancient))  # mtime far older than stale_after
    with pytest.raises(LockTimeout):
        FileLock(tmp_path / "res", timeout=0.2, stale_after=1.0).acquire()
    assert lockfile.exists()  # the live holder's lock was kept, not broken


def test_a_dead_holders_lock_is_broken_even_when_mtime_is_fresh(tmp_path: Path) -> None:
    # PID-liveness also breaks a crashed holder PROMPTLY, without waiting for stale_after: a fresh-mtime
    # lock whose PID is dead is broken so the next process is not deadlocked by a crash.
    lockfile = Path(str(tmp_path / "res") + ".lock")
    lockfile.write_text(f"{_a_dead_pid()} {time.time()} {'b' * 32}\n")  # dead PID, FRESH mtime
    with FileLock(tmp_path / "res", timeout=1.0, stale_after=3600) as lock:  # broken despite the big bound
        assert lock._token is not None and lockfile.exists()  # we acquired a fresh lock


def test_atomic_break_removes_only_the_exact_bytes_it_judged_stale(tmp_path: Path) -> None:
    # PID-reuse immunity: _atomic_break identifies the stale lock by its exact (per-acquisition-token)
    # bytes, not by PID. It breaks the file only when its content still equals what was judged stale.
    lk = FileLock(tmp_path / "res")
    judged = b"12345 6.7 deadbeefdeadbeef\n"
    lk.lock_path.write_bytes(judged)
    lk._atomic_break(expected=judged)  # matches what's on disk -> break it
    assert not lk.lock_path.exists()


def test_atomic_break_restores_a_lock_re_created_in_the_race_window(tmp_path: Path) -> None:
    # If the on-disk lock differs from the judged bytes (a fresh lock re-created in the race - a different
    # token, immune to PID reuse), it is RESTORED, never broken - so no live lock is deleted.
    lk = FileLock(tmp_path / "res")
    fresh = b"999 1.0 aFRESHtokenNOTtheStaleOne\n"
    lk.lock_path.write_bytes(fresh)
    lk._atomic_break(expected=b"the-stale-bytes-i-judged 0 oldtoken\n")  # differs from what's on disk now
    assert lk.lock_path.read_bytes() == fresh  # restored intact, not deleted


def test_release_only_removes_a_lock_we_still_own(tmp_path: Path) -> None:
    # #12: if another process broke our stale lock and re-created its OWN, our LATE release must not delete
    # that new lock. Simulate it: acquire, then overwrite the lockfile with a different owner token.
    lockfile = Path(str(tmp_path / "res") + ".lock")
    ours = FileLock(tmp_path / "res", timeout=2).acquire()
    lockfile.write_text("99999 0 SOMEONE_ELSES_TOKEN\n")  # a different process now owns it
    ours.release()
    assert lockfile.exists() and "SOMEONE_ELSES_TOKEN" in lockfile.read_text()  # not deleted by us
    lockfile.unlink()
    again = FileLock(tmp_path / "res", timeout=2).acquire()  # a normal acquire/release still cleans up
    again.release()
    assert not lockfile.exists()


def test_release_matches_the_owner_token_exactly_not_as_a_substring(tmp_path: Path) -> None:
    # A lockfile whose token field only CONTAINS our token as a substring (e.g. "<token>X") is a
    # different owner - release() must compare the token field exactly and keep it.
    lockfile = Path(str(tmp_path / "res") + ".lock")
    held = FileLock(tmp_path / "res", timeout=2).acquire()
    lockfile.write_text(f"123 456 {held._token}X\n")  # token is a substring, not the exact field
    held.release()
    assert lockfile.exists()  # not deleted - the exact token did not match


def test_context_manager_releases_on_exit(tmp_path: Path) -> None:
    with FileLock(tmp_path / "res", timeout=1):
        pass
    assert not Path(str(tmp_path / "res") + ".lock").exists()  # released on __exit__
