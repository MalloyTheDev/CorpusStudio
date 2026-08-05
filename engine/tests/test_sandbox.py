"""The custom-code runtime sandbox launcher (slice 3a): the SandboxPolicy is safe-by-construction
(network/read-only-root/no-new-privileges are type-locked), the launcher builds the right containment argv
for each backend, refuses fail-closed when no backend exists, and applies resource limits. Nothing here
runs custom code - it builds + probes only."""

import resource

import pytest
from pydantic import ValidationError

from corpus_studio.platform import sandbox as sb
from corpus_studio.platform.contracts import SandboxPolicy


def _subseq(seq, sub):
    """Whether `sub` appears as a contiguous subsequence of `seq`."""
    return any(seq[i : i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1))


# ---- the policy is safe-by-construction --------------------------------------------------------------


def test_policy_locks_the_untrusted_invariants():
    p = SandboxPolicy()
    assert p.network_isolated is True and p.readonly_root is True and p.no_new_privileges is True
    for field in ("network_isolated", "readonly_root", "no_new_privileges"):
        with pytest.raises(ValidationError):  # a Literal[True] cannot be weakened to False
            SandboxPolicy(**{field: False})


def test_policy_requires_absolute_paths():
    with pytest.raises(ValidationError):
        SandboxPolicy(writable_paths=["relative/out"])
    with pytest.raises(ValidationError):
        SandboxPolicy(gpu_devices=[""])
    ok = SandboxPolicy(writable_paths=["/run/out"], gpu_devices=["/dev/nvidia0"])
    assert ok.writable_paths == ["/run/out"]


# ---- argv construction -------------------------------------------------------------------------------


def test_bubblewrap_argv_encodes_the_policy():
    policy = SandboxPolicy(writable_paths=["/run/out"], gpu_devices=["/dev/nvidia0"])
    argv = sb.build_sandboxed_argv(policy, ["python", "-m", "worker"], backend="bubblewrap")
    assert argv[0] == "bwrap"
    assert "--unshare-net" in argv
    assert _subseq(argv, ["--ro-bind", "/", "/"])
    assert _subseq(argv, ["--bind", "/run/out", "/run/out"])  # rw exception
    assert _subseq(argv, ["--dev-bind", "/dev/nvidia0", "/dev/nvidia0"])  # the documented GPU hole
    assert argv[-4:] == ["--", "python", "-m", "worker"]  # the inner command after the separator


def test_unshare_fallback_isolates_the_network():
    argv = sb.build_sandboxed_argv(SandboxPolicy(), ["python", "-m", "worker"], backend="unshare")
    assert argv[0] == "unshare" and "--net" in argv
    assert argv[-3:] == ["python", "-m", "worker"]


def test_no_backend_refuses_fail_closed():
    with pytest.raises(sb.SandboxUnavailableError):
        sb.build_sandboxed_argv(SandboxPolicy(), ["python"], backend=None)


def test_empty_inner_command_is_refused():
    with pytest.raises(sb.SandboxUnavailableError):
        sb.build_sandboxed_argv(SandboxPolicy(), [], backend="bubblewrap")


# ---- availability probe (mocked host) ----------------------------------------------------------------


def test_probe_picks_bubblewrap_when_present(monkeypatch):
    monkeypatch.setattr(sb.sys, "platform", "linux")
    monkeypatch.setattr(sb.shutil, "which", lambda _: "/usr/bin/bwrap")
    monkeypatch.setattr(sb, "_user_namespaces_available", lambda: True)
    avail = sb.probe_sandbox()
    assert avail.backend == "bubblewrap" and avail.bubblewrap_path == "/usr/bin/bwrap"


def test_probe_falls_back_to_unshare_without_bwrap(monkeypatch):
    monkeypatch.setattr(sb.sys, "platform", "linux")
    monkeypatch.setattr(sb.shutil, "which", lambda _: None)
    monkeypatch.setattr(sb, "_user_namespaces_available", lambda: True)
    assert sb.probe_sandbox().backend == "unshare"


def test_probe_reports_none_without_user_namespaces(monkeypatch):
    monkeypatch.setattr(sb.sys, "platform", "linux")
    monkeypatch.setattr(sb.shutil, "which", lambda _: None)
    monkeypatch.setattr(sb, "_user_namespaces_available", lambda: False)
    assert sb.probe_sandbox().backend is None


def test_probe_reports_none_off_linux(monkeypatch):
    monkeypatch.setattr(sb.sys, "platform", "win32")
    assert sb.probe_sandbox().backend is None


# ---- resource limits ---------------------------------------------------------------------------------


def test_rlimit_preexec_applies_the_limits(monkeypatch):
    calls: list[tuple[int, tuple[int, int]]] = []
    monkeypatch.setattr(resource, "setrlimit", lambda res, lim: calls.append((res, lim)))
    policy = SandboxPolicy(rlimit_address_space_bytes=2**30, rlimit_processes=0, rlimit_open_files=1024)
    apply = sb.rlimit_preexec(policy)
    assert apply is not None
    apply()
    applied = dict(calls)
    assert applied[resource.RLIMIT_AS] == (2**30, 2**30)
    assert applied[resource.RLIMIT_NPROC] == (0, 0)  # no forks
    assert applied[resource.RLIMIT_NOFILE] == (1024, 1024)


def test_rlimit_preexec_is_none_without_limits():
    assert sb.rlimit_preexec(SandboxPolicy()) is None
