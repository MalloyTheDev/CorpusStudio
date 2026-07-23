"""Tests for the portable advisory file lock (scripts/loop/locking.py).

Pins mutual exclusion (a live holder makes a waiter fail closed), stale-lock breaking (a crashed holder
cannot wedge the next process), idempotent release, and context-manager use.
"""

from __future__ import annotations

import os
import sys
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


def test_context_manager_releases_on_exit(tmp_path: Path) -> None:
    with FileLock(tmp_path / "res", timeout=1):
        pass
    assert not Path(str(tmp_path / "res") + ".lock").exists()  # released on __exit__
