"""The custom-code runtime sandbox launcher (mode 3, slice 3a - control plane, executes NOTHING).

This module turns a :class:`SandboxPolicy` into the concrete OS-level containment for a worker that will
run VETTED-BUT-UNTRUSTED custom-block code, and probes whether this host can provide it. Slice 3a builds
and tests the launcher (argv construction + availability probe + resource-limit hook) WITHOUT running any
custom code; wiring it into the (gated) pretraining worker + a measured run is slice 3b.

Honesty: a static screen is necessary, not sufficient; THIS is the real blast-radius limit. Its own limit
is honest too - a GPU training block must reach the CUDA devices, so a GPU workload is
blast-radius-limited, not fully isolated. A host with no usable backend must REFUSE to run custom code
(fail-closed), never fall back to running it unconfined.

``bubblewrap`` (``bwrap``) is the primary backend: unprivileged user namespaces, a read-only root with a
run-scoped writable bind, no network, a private /tmp, and no-new-privileges by default. The ``unshare``
fallback isolates the network + applies rlimits but does NOT confine the filesystem, so it is weaker and
labelled as such.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass

from corpus_studio.platform.contracts import SandboxPolicy

# Runnable backends, strongest first. "none" is not runnable - a host that resolves to it refuses.
SANDBOX_BACKENDS: tuple[str, ...] = ("bubblewrap", "unshare")


class SandboxUnavailableError(RuntimeError):
    """This host cannot provide a sandbox, so untrusted custom code must not run (fail-closed)."""


@dataclass(frozen=True)
class SandboxAvailability:
    """What containment this host can actually provide (a probe result, not a workload claim)."""

    backend: str | None  # "bubblewrap" | "unshare" | None (none => custom code is refused)
    bubblewrap_path: str | None
    user_namespaces: bool
    detail: str


def _user_namespaces_available() -> bool:
    """Whether unprivileged user namespaces (needed by both backends) look usable on this host."""
    knob = "/proc/sys/kernel/unprivileged_userns_clone"
    try:
        with open(knob, encoding="utf-8") as handle:  # Debian/Ubuntu expose this toggle
            return handle.read().strip() == "1"
    except FileNotFoundError:
        # Most kernels do not expose that knob; presence of the user ns interface is the fallback signal.
        return os.path.exists("/proc/self/ns/user")
    except OSError:
        return False


def probe_sandbox() -> SandboxAvailability:
    """Detect the strongest usable backend on this host. Read-only: it runs no untrusted code and does not
    itself spawn a sandbox - it inspects the environment (PATH + user-namespace support)."""
    if sys.platform != "linux":
        return SandboxAvailability(None, None, False, f"sandbox requires Linux, host is {sys.platform}")
    bwrap = shutil.which("bwrap")
    userns = _user_namespaces_available()
    if bwrap and userns:
        return SandboxAvailability("bubblewrap", bwrap, userns, "bubblewrap with user namespaces")
    if userns:
        return SandboxAvailability(
            "unshare", bwrap, userns, "bubblewrap absent; unshare fallback (no filesystem confinement)"
        )
    return SandboxAvailability(
        None, bwrap, userns, "no user namespaces; custom code cannot be sandboxed on this host"
    )


def _bwrap_argv(policy: SandboxPolicy, inner_argv: list[str]) -> list[str]:
    argv = [
        "bwrap",
        "--die-with-parent",  # do not outlive the supervisor
        "--new-session",  # own session: blocks TIOCSTI terminal-injection
        "--unshare-net",  # no network
        "--unshare-ipc",
        "--unshare-uts",
    ]
    if policy.readonly_root:
        argv += ["--ro-bind", "/", "/"]  # read-only view of the host; rw exceptions bound below
    argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for path in policy.writable_paths:
        argv += ["--bind", path, path]  # the run-scoped output dir(s)
    for device in policy.gpu_devices:
        argv += ["--dev-bind", device, device]  # the documented GPU hole
    argv += ["--", *inner_argv]
    return argv


def _unshare_argv(policy: SandboxPolicy, inner_argv: list[str]) -> list[str]:
    # Weaker fallback: an unprivileged user+network namespace (no network) - but NO filesystem confinement,
    # so the caller relies on rlimits + the static screen for the rest. writable_paths/gpu_devices are not
    # bind-restricted here (the whole FS is visible); they are advisory in this backend.
    return ["unshare", "--user", "--map-root-user", "--net", "--", *inner_argv]


def build_sandboxed_argv(
    policy: SandboxPolicy, inner_argv: list[str], *, backend: str | None
) -> list[str]:
    """Wrap ``inner_argv`` (the worker command) in the chosen backend's containment argv (no shell, a
    plain argv list). Refuses fail-closed when there is no runnable backend - untrusted code never runs
    unconfined."""
    if not inner_argv:
        raise SandboxUnavailableError("no inner command to sandbox")
    if backend == "bubblewrap":
        return _bwrap_argv(policy, list(inner_argv))
    if backend == "unshare":
        return _unshare_argv(policy, list(inner_argv))
    raise SandboxUnavailableError(
        "no usable sandbox backend on this host; refusing to run untrusted custom code unconfined"
    )


def rlimit_preexec(policy: SandboxPolicy) -> Callable[[], None] | None:
    """A ``preexec_fn`` that applies the policy's resource limits in the child before ``exec`` (POSIX
    only). Returns None when nothing to apply or off-POSIX. Built here, invoked by the worker launch in
    slice 3b - not called in 3a."""
    if os.name != "posix":
        return None
    import resource  # noqa: PLC0415 - POSIX-only, imported lazily so import stays cross-platform

    limits: list[tuple[int, int]] = []
    if policy.rlimit_address_space_bytes is not None:
        limits.append((resource.RLIMIT_AS, policy.rlimit_address_space_bytes))
    if policy.rlimit_cpu_seconds is not None:
        limits.append((resource.RLIMIT_CPU, policy.rlimit_cpu_seconds))
    if policy.rlimit_open_files is not None:
        limits.append((resource.RLIMIT_NOFILE, policy.rlimit_open_files))
    if policy.rlimit_processes is not None:
        limits.append((resource.RLIMIT_NPROC, policy.rlimit_processes))
    if not limits:
        return None

    def _apply() -> None:
        for resource_id, value in limits:
            resource.setrlimit(resource_id, (value, value))

    return _apply
