"""The portable, bounded, same-thread-reentrant cross-process file lock (#859).

Deterministic and torch-free. Cross-process cases spawn ``sys.executable`` children with file
barriers (never fork: the process-local lock state would be inherited).
"""

from __future__ import annotations

import errno
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from corpus_studio.storage import file_lock
from corpus_studio.storage.file_lock import (
    FileLockError,
    FileLockTimeoutError,
    exclusive_file_lock,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX lock-file semantics")

_HOLDER_CHILD = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from corpus_studio.storage.file_lock import exclusive_file_lock

    lock, ready, release = (Path(arg) for arg in sys.argv[1:4])
    with exclusive_file_lock(lock, timeout_seconds=10):
        ready.write_text("1")
        deadline = time.monotonic() + 30
        while not release.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
    """
)


def _wait_for(path: Path, child: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 30
    while not path.exists():
        assert child.poll() is None, child.communicate()
        assert time.monotonic() < deadline, "child never signalled"
        time.sleep(0.01)


def _acquire_in_thread(path: Path, timeout: float) -> BaseException | None:
    outcome: dict[str, BaseException | None] = {}

    def run() -> None:
        try:
            with exclusive_file_lock(path, timeout_seconds=timeout):
                outcome["error"] = None
        except BaseException as exc:  # noqa: BLE001 - reported to the asserting thread
            outcome["error"] = exc

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(10)
    assert not worker.is_alive()
    return outcome["error"]


def test_same_thread_reenters_and_other_threads_time_out(tmp_path: Path) -> None:
    lock = tmp_path / "state" / "x.lock"
    with exclusive_file_lock(lock, timeout_seconds=1):
        with exclusive_file_lock(lock, timeout_seconds=1):  # reentrant: no self-deadlock
            error = _acquire_in_thread(lock, 0.05)
    assert isinstance(error, FileLockTimeoutError)
    assert "another thread" in str(error) and str(error).isascii()
    assert _acquire_in_thread(lock, 0.05) is None  # released after the outer exit
    assert lock.read_bytes() == b"\0"  # one lockable byte, created on first use


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_timeout_must_be_finite_and_positive(tmp_path: Path, timeout: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        with exclusive_file_lock(tmp_path / "x.lock", timeout_seconds=timeout):
            pass


def test_other_process_holder_excludes_until_released(tmp_path: Path) -> None:
    lock, ready, release = tmp_path / "x.lock", tmp_path / "ready", tmp_path / "release"
    child = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_CHILD, str(lock), str(ready), str(release)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for(ready, child)
        with pytest.raises(FileLockTimeoutError, match="another process"):
            with exclusive_file_lock(lock, timeout_seconds=0.2):
                pass
    finally:
        release.write_text("1")
        child.communicate(timeout=30)
    assert child.returncode == 0
    with exclusive_file_lock(lock, timeout_seconds=1):
        pass


def test_body_errors_propagate_and_release_the_lock(tmp_path: Path) -> None:
    lock = tmp_path / "x.lock"
    with pytest.raises(KeyError):
        with exclusive_file_lock(lock, timeout_seconds=1):
            raise KeyError("body failure")
    assert _acquire_in_thread(lock, 0.05) is None


def test_symlinked_lock_file_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    target.write_bytes(b"\0")
    lock = tmp_path / "x.lock"
    lock.symlink_to(target)
    with pytest.raises(FileLockError, match="unavailable"):
        with exclusive_file_lock(lock, timeout_seconds=0.2):
            pass


def test_symlinked_lock_directory_is_refused(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(FileLockError, match="symbolic link"):
        with exclusive_file_lock(linked_dir / "x.lock", timeout_seconds=0.2):
            pass


def test_hard_linked_lock_file_is_refused(tmp_path: Path) -> None:
    lock = tmp_path / "x.lock"
    lock.write_bytes(b"\0")
    os.link(lock, tmp_path / "alias.lock")
    with pytest.raises(FileLockError, match="singly linked"):
        with exclusive_file_lock(lock, timeout_seconds=0.2):
            pass


def test_lock_file_owned_by_another_user_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "x.lock"
    lock.write_bytes(b"\0")
    other_uid = lock.stat().st_uid + 1
    monkeypatch.setattr(os, "getuid", lambda: other_uid)
    with pytest.raises(FileLockError, match="another user"):
        with exclusive_file_lock(lock, timeout_seconds=0.2):
            pass


def test_io_error_after_open_is_refused_and_closes_the_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[int] = []
    real_close = os.close

    def failing_fchmod(descriptor: int, mode: int) -> None:
        raise OSError(errno.EPERM, "fchmod refused")

    def spy_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(os, "fchmod", failing_fchmod)
    monkeypatch.setattr(os, "close", spy_close)
    with pytest.raises(FileLockError, match="unavailable"):
        with exclusive_file_lock(tmp_path / "x.lock", timeout_seconds=0.2):
            pass
    assert closed  # the opened descriptor was closed, not leaked


def test_unexpected_lock_errno_is_refused_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_flock(fd: int, operation: int) -> None:
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(file_lock.fcntl, "flock", failing_flock)
    with pytest.raises(FileLockError, match="could not be locked"):
        with exclusive_file_lock(tmp_path / "x.lock", timeout_seconds=5):
            pass


def test_failed_explicit_unlock_still_releases_on_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_unlock(stream: object) -> None:
        raise OSError(errno.EIO, "unlock failed")

    lock = tmp_path / "x.lock"
    monkeypatch.setattr(file_lock, "_unlock_os_lock", failing_unlock)
    with exclusive_file_lock(lock, timeout_seconds=1):
        pass
    monkeypatch.undo()
    assert _acquire_in_thread(lock, 0.2) is None  # closing the descriptor dropped the flock
