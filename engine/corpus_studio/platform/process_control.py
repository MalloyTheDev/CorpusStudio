"""Cross-platform subprocess-group creation and bounded process-tree termination.

Backend workers and environment installers may spawn compiler, data-loader, launcher, or rank
processes.  A timeout must therefore own and stop the whole tree, not only the direct child.  This
module is stdlib-only so both dependency-light control-plane paths can share the same behavior.
"""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any


def process_group_creation_flags() -> int:
    """Return the Windows flag that gives the child its own process group."""

    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))


def start_new_process_session() -> bool:
    """Use a new POSIX session; Windows uses ``creationflags`` instead."""

    return os.name != "nt"


def terminate_process_tree(
    process: subprocess.Popen[Any], *, wait_timeout_seconds: float = 5.0
) -> None:
    """Terminate a process tree, escalate if needed, and reap the direct child.

    Windows uses the fixed system ``taskkill.exe`` utility with ``/T`` and no shell. POSIX workers
    are session leaders, so signaling their process group reaches descendants and distributed ranks.
    """

    if os.name == "nt":
        if process.poll() is None:
            system_root = Path(os.environ.get("SYSTEMROOT", r"C:\Windows"))
            taskkill = system_root / "System32" / "taskkill.exe"
            try:
                subprocess.run(  # noqa: S603 - fixed OS utility and integer pid, no shell
                    [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=wait_timeout_seconds,
                    shell=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                process.terminate()
        try:
            process.wait(timeout=wait_timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=wait_timeout_seconds)
            except subprocess.TimeoutExpired:  # pragma: no cover - OS failed to reap a killed process
                pass
        return

    # A POSIX process group can outlive its leader. Waiting only for the direct child lets a
    # SIGTERM-ignoring compiler/rank survive forever, so track the group independently and escalate.
    kill_process_group = getattr(os, "killpg")
    process_group_id = process.pid

    def _group_exists() -> bool:
        try:
            kill_process_group(process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:  # pragma: no cover - same-user child groups are normally signalable.
            return True
        return True

    try:
        kill_process_group(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + wait_timeout_seconds
    while _group_exists() and time.monotonic() < deadline:
        process.poll()  # reap the direct child as soon as it exits
        time.sleep(0.05)
    if _group_exists():
        try:
            kill_process_group(
                process_group_id,
                getattr(signal, "SIGKILL", signal.SIGTERM),
            )
        except ProcessLookupError:
            pass
        kill_deadline = time.monotonic() + wait_timeout_seconds
        while _group_exists() and time.monotonic() < kill_deadline:
            process.poll()
            time.sleep(0.05)
    if process.poll() is None:
        try:
            process.wait(timeout=wait_timeout_seconds)
        except subprocess.TimeoutExpired:  # pragma: no cover - OS failed to reap a killed process
            pass
