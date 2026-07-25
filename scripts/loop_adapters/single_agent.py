"""Phase 7.0 runtime adapter: a REAL single agent, PROPOSE-ONLY (zero writes).

This is the second concrete adapter and the first that wires a real Claude-Code agent into the loop - but
strictly read/propose-only. It drives a goal through the whole state machine against live repo state; at
the EXECUTE phase it asks the agent to PROPOSE a unified diff (it never applies it), seals that proposal as
a tamper-evident ``agent_proposal`` record, and ENDS at ESCALATED so a human decides whether to apply it.
It makes NO writes: no source edits, no commits, no push, no PR, no merge, and it declares
``capabilities=frozenset()`` (read-only), so the capability gate lets it run with no ``--allow-capabilities``.

Design (docs/PRODUCTION_SINGLE_AGENT_RUNTIME.md, phase 7.0):
  * The agent is invoked through an INJECTED :class:`AgentClient` (the loop-level seam); the real transport
    is an out-of-process, fixed-argv ``claude`` subprocess with a framed JSON contract over stdio
    (:class:`ClaudeSubprocessClient`). Tests inject a deterministic stub. No SDK (that would break the
    adapter's stdlib-only rule and run the untrusted agent in-process); no shell.
  * The agent's output is UNTRUSTED: it is validated fail-closed into the sealed record; a bad transport /
    unparseable / wrong-shaped response raises, which escalates the loop (never a silent advance).
  * The agent is CONFINED even while it only proposes (phase 7.1.1): the executor runs it with cwd inside a
    disposable, detached worktree at ``base`` (never the developer's tree) and a sanitized (secret-free)
    environment, and the propose diff is stored as a sealed record OUTSIDE the working tree. Whatever the
    agent writes into the throwaway worktree is discarded; only the unified diff it RETURNS is used.

stdlib-only; every git/gh read and the agent transport fail closed. Read-only ``gh`` + the whole-tree diff
building block are reused from the dry-run adapter.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess  # noqa: S404 - fixed-argv git / claude only; never a shell string.
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol

from loop.completeness import Criterion, CriterionKind
from loop.controller import LoopState, Observation, Phase
from loop.driver import Directive
from loop.orchestrate import LoopContext, LoopOrchestrateError
from loop_adapters.dry_run import read_only_gh  # the same default-deny read-only gh the dry run uses

_AGENT_TIMEOUT_S = 300  # a bounded, killable agent call - never an unbounded hang
_GIT_TIMEOUT_S = 60
_MAX_AGENT_OUTPUT_BYTES = 8 * 1024 * 1024  # 8 MiB cap on agent output - a proposal diff is far smaller

# Environment variables that MUST NOT reach the untrusted agent subprocess: anything credential-shaped, plus
# known VCS / cloud / registry auth. Stripped from the inherited env (an explicit allowlist is stronger
# still; this denylist is the fail-safer floor, and PATH / HOME / locale survive).
_SECRET_ENV_SUBSTRINGS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "PRIVATE_KEY", "APIKEY",
                          "API_KEY", "ACCESS_KEY", "SECRET_KEY", "AUTH", "COOKIE", "SESSION")
_SECRET_ENV_PREFIXES = ("GITHUB_", "GH_", "AWS_", "GOOGLE_", "GCP_", "AZURE_", "OPENAI_", "ANTHROPIC_",
                        "HF_", "HUGGINGFACE_", "NPM_", "PYPI_", "TWINE_", "DOCKER_", "SSH_", "GPG_")


# The ONLY things placed in the agent's confined HOME: what the transport needs to authenticate ITSELF.
# Everything else in the real home stays unreachable by `~` expansion - notably ~/.config/gh/hosts.yml (a
# token with WRITE access to this repo), ~/.ssh, ~/.aws, ~/.netrc, and ~/.claude's 284 MB of sessions,
# history and OTHER PROJECTS' transcripts.
#
# `.claude/.credentials.json` IS THE AGENT'S OAUTH TOKEN, and it is here deliberately: an agent that
# cannot authenticate cannot run at all (measured - without it the CLI returns "Not logged in"). So the
# honest statement of this boundary is: THE UNTRUSTED AGENT CAN READ ITS OWN CREDENTIAL. That is
# unavoidable for any design where the agent authenticates as the operator; what the confinement buys is
# that this is the ONLY credential it can reach - not the GitHub token, not SSH keys, not cloud creds,
# not other projects' transcripts. Anyone tightening this further needs a separate agent identity with
# its own scoped credential, which is a 7.2+ concern.
_AGENT_HOME_PASSTHROUGH: tuple[str, ...] = (
    ".claude.json", ".claude/settings.json", ".claude/.credentials.json",
)
_MAX_PASSTHROUGH_BYTES = 4 * 1024 * 1024  # a credential file is small; never copy a session store


def _sanitized_env(home: Path | None = None) -> dict[str, str]:
    """A copy of the environment with credential-shaped variables STRIPPED, so the untrusted agent
    subprocess never inherits GitHub / cloud / release / registry secrets.

    When ``home`` is given, HOME and every XDG base directory are REPOINTED at it, so `~` expansion and
    XDG lookups resolve inside a throwaway directory instead of the operator's real home. That is what
    turns "the agent could read ~/.config/gh/hosts.yml" from reachable into not-at-that-path.

    HONEST LIMIT: this is NOT a sandbox. The subprocess runs as the operator's UID and can still read any
    ABSOLUTE path it can guess. What this buys is that the obvious, `~`-shaped route is gone and the blast
    radius is bounded to what :func:`_confined_home` deliberately links in - the transport's own
    credential. Real isolation (container / user namespace) is the 7.1.5 sandbox."""
    clean: dict[str, str] = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if any(s in upper for s in _SECRET_ENV_SUBSTRINGS) or any(upper.startswith(p) for p in _SECRET_ENV_PREFIXES):
            continue
        clean[key] = value
    if home is not None:
        clean["HOME"] = str(home)
        clean["XDG_CONFIG_HOME"] = str(home / ".config")
        clean["XDG_CACHE_HOME"] = str(home / ".cache")
        clean["XDG_DATA_HOME"] = str(home / ".local" / "share")
        clean["XDG_STATE_HOME"] = str(home / ".local" / "state")
    return clean


@contextmanager
def _confined_home(parent: Path) -> Iterator[Path]:
    """A throwaway HOME for the untrusted agent, containing ONLY symlinks to what the transport needs to
    authenticate itself (:data:`_AGENT_HOME_PASSTHROUGH`). Removed on exit.

    COPIED, not symlinked. Symlinks pointed at the operator's real home, which is NOT bound inside the
    OS sandbox's mount namespace - measured: the agent started and reported "Not logged in", because the
    link dangled. Copying the two small files (a few KB) works both sandboxed and not, and has a second
    benefit: the agent gets its OWN copy, so anything it writes back cannot touch the operator's real
    config. (Only these files - ~/.claude as a whole is ~284 MB of sessions and other projects.)
    A passthrough that is missing, oversized or unreadable is SKIPPED, not fatal: an operator whose
    transport authenticates differently still gets the confinement."""
    home = parent / f"home-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        for rel in _AGENT_HOME_PASSTHROUGH:
            src = Path.home() / rel
            try:
                if not src.is_file() or src.stat().st_size > _MAX_PASSTHROUGH_BYTES:
                    continue
                dst = home / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                dst.chmod(0o600)   # a credential copy keeps restrictive permissions
            except OSError:
                continue  # unreadable passthrough: the transport may still authenticate another way
        (home / ".config").mkdir(parents=True, exist_ok=True)
        yield home
    finally:
        shutil.rmtree(home, ignore_errors=True)


def _git(cwd: Path, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``git -C <cwd> <args>`` (fixed argv, no shell, bounded) with HOOKS AND FSMONITOR PINNED OFF.
    Raises :class:`AgentError` on a non-zero exit / un-runnable git - shared by the read (propose) and
    write adapters.

    ``core.hooksPath=/dev/null`` stops repository hooks; ``core.fsmonitor=`` is ALSO required because
    fsmonitor is a SEPARATE config-driven hook that ``hooksPath`` does not suppress (measured: a
    ``core.fsmonitor`` script still executed under ``git -c core.hooksPath=/dev/null status``). Both run
    with the operator's full environment, so both are disabled on every invocation."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, hooks + fsmonitor disabled.
            ["git", "-C", str(cwd), "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=",
             *args], input=stdin, capture_output=True, text=True, timeout=_GIT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AgentError(f"git {' '.join(args[:2])} could not run: {exc}") from exc
    if proc.returncode != 0:
        raise AgentError(f"git {' '.join(args[:2])} failed (exit {proc.returncode}): {proc.stderr.strip()[:200]}")
    return proc


@contextmanager
def _detached_worktree(repo_root: Path, base_oid: str, worktrees_dir: Path) -> Iterator[Path]:
    """An ISOLATED, disposable, DETACHED ``git worktree`` at ``base_oid`` under ``worktrees_dir`` (outside
    the main working tree) - the confined checkout the agent runs in for a read/propose. Removed on exit
    (best-effort); anything the agent writes here is thrown away."""
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    # A per-invocation UNIQUE name: pid + a random suffix, so a leftover directory from a best-effort
    # disposal that failed can never collide with a later propose for the SAME base in the SAME process.
    wt = worktrees_dir / f"propose-{base_oid[:12]}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    _git(repo_root, "worktree", "add", "--detach", str(wt), base_oid)
    try:
        yield wt
    finally:
        try:
            _git(repo_root, "worktree", "remove", "--force", str(wt))
        except AgentError:
            # The agent can defeat `worktree remove` by corrupting the `.git` gitfile inside its own cwd
            # (measured), which would otherwise leave its bytes on disk AND a stale registration in the
            # operator's repo forever. Fall back to deleting the tree ourselves and pruning the
            # registration, so disposal is not opt-out for the untrusted party.
            shutil.rmtree(wt, ignore_errors=True)
            try:
                _git(repo_root, "worktree", "prune")
            except AgentError:
                pass  # genuinely best-effort now; nothing of the agent's survives on disk


def default_worktrees_dir(repo_root: Path) -> Path:
    """A disposable-worktree root OUTSIDE the repository entirely: ``<cache>/corpusstudio-loop/worktrees/
    <repo-name>-<hash of the absolute repo path>``.

    NOT under ``<git-dir>`` (which is where this used to live) and not under the working tree. The agent's
    confined cwd is a directory inside here, so a naive ``../../..`` from it must NOT land in the
    repository's ``.git``: an agent that can append to ``.git/config`` gets PERSISTENT ARBITRARY CODE
    EXECUTION as the operator with the full un-sanitized environment - measured, via ``core.fsmonitor``,
    which ``core.hooksPath=/dev/null`` does not suppress - and could also plant ``.git/hooks/*``.

    HONEST LIMIT: a cwd is NOT a sandbox. Nothing here stops a determined process from walking to an
    absolute path; the load-bearing restrictions on the agent are the read-only TOOL POLICY (no
    Write/Edit/Bash) and the sanitized environment, and real isolation (container / namespace) is the
    7.1.5 sandbox work. What this function guarantees is only that the operator's own git directory is not
    sitting three levels above the untrusted process's working directory."""
    root = repo_root.resolve()
    key = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    # The XDG spec says a RELATIVE value here is invalid and must be ignored - and it matters:
    # `XDG_CACHE_HOME=.cache` made `git -C <repo> worktree add <relpath>` resolve against the REPO ROOT,
    # putting the confined agent back inside the working tree with the operator's .git a fixed walk away
    # (measured: the pass-5 RCE restored in full, silently). Anything non-absolute is discarded.
    candidates = (os.environ.get("XDG_CACHE_HOME") or "", str(Path.home() / ".cache"))
    base = next((Path(c) for c in candidates if c and Path(c).is_absolute()), None)
    if base is None:  # no absolute cache dir available at all -> fail closed rather than guess
        raise AgentError("cannot resolve an ABSOLUTE cache directory for the confined worktree "
                         "(XDG_CACHE_HOME and HOME are both unusable); refusing to place it relatively")
    return base / "corpusstudio-loop" / "worktrees" / f"{root.name}-{key}"


# ============================================================ agent sandboxing (phase 7.1.5)
# The confinement shipped through 7.1.4 is cwd + a sanitized env + a redirected HOME + a read-only tool
# policy. Every one of those is DEFENCE IN DEPTH, not a boundary: the subprocess runs as the operator's
# UID and can read any ABSOLUTE path it guesses. Encoded exfiltration (base64/hex/split) was measured
# defeating every content scan, so the only real answer is OS-level isolation.
#
# This is the SEAM plus - crucially - a PROBE. Presence of a sandbox binary proves nothing: on the host
# this was written for, `bwrap` is installed and CANNOT WORK (`apparmor_restrict_unprivileged_userns=1`
# blocks unprivileged user namespaces). A sandbox that silently fails open is worse than none, because it
# converts an known gap into a false assurance. So the sandbox is VERIFIED by trying to read a canary
# secret from inside it, and an unverified sandbox REFUSES rather than degrades.
_SANDBOX_PROBE_TIMEOUT_S = 20


class SandboxRunner(Protocol):
    """Wraps an agent argv so it executes under OS-level isolation.

    ``wrap`` returns the argv to actually run. ``cwd`` is the disposable worktree the agent may write;
    ``home`` is its throwaway HOME. An implementation MUST deny read access to the operator's real home
    and SHOULD deny network access."""

    def wrap(self, argv: "tuple[str, ...]", *, cwd: Path, home: Path) -> "tuple[str, ...]":
        ...


class BubblewrapSandbox:
    """A :class:`SandboxRunner` backed by ``bwrap``: the candidate worktree and a throwaway HOME are bound
    read-write, the system runtime read-only, and nothing else of the operator's filesystem is visible.

    Not usable everywhere - it needs unprivileged user namespaces, which several distributions restrict by
    default. :func:`verify_sandbox` is what decides whether it actually works HERE."""

    def __init__(self, bwrap: str = "bwrap") -> None:
        self.bwrap = bwrap

    def wrap(self, argv: tuple[str, ...], *, cwd: Path, home: Path) -> tuple[str, ...]:
        # The agent binary is very often NOT under /usr - here it is a self-contained ELF under an XDG
        # data dir, reached via a ~/.local/bin symlink. Binding only the system runtime made the sandbox
        # fail with "execvp claude: No such file or directory" (measured). Resolve argv[0] and bind the
        # REAL path read-only, then invoke it by that path so the symlink is not needed inside.
        exe = shutil.which(argv[0]) if argv else None
        resolved = str(Path(exe).resolve()) if exe else (argv[0] if argv else "")
        agent_binds: tuple[str, ...] = ()
        if resolved and Path(resolved).exists():
            agent_binds = ("--ro-bind", resolved, resolved)
            argv = (resolved, *argv[1:])
        return (
            # `--unshare-all --share-net`: every namespace isolated EXCEPT the network.
            #
            # HONEST TRADE-OFF, and it bounds what this sandbox can promise: the agent is an API client -
            # it CANNOT function without reaching the network (measured: with --unshare-all it exits 1
            # immediately). So network egress is NOT a control here, and a sandboxed agent could still
            # POST data out. What the sandbox actually buys is FILESYSTEM isolation: the operator's home,
            # credentials, keys and other repositories are simply not present in the mount namespace.
            # The exfiltration bound therefore comes from what the agent can READ (confined HOME, no
            # credential env, read-only tool policy), not from what it can SEND.
            self.bwrap, "--die-with-parent", "--unshare-all", "--share-net",
            # system runtime, read-only
            "--ro-bind", "/usr", "/usr", "--ro-bind", "/bin", "/bin", "--ro-bind", "/lib", "/lib",
            "--ro-bind-try", "/lib64", "/lib64", "--ro-bind-try", "/etc/ssl", "/etc/ssl",
            "--ro-bind-try", "/etc/resolv.conf", "/etc/resolv.conf",
            "--ro-bind-try", "/etc/hosts", "/etc/hosts",
            "--ro-bind-try", "/etc/nsswitch.conf", "/etc/nsswitch.conf",
            "--ro-bind-try", "/etc/ca-certificates.conf", "/etc/ca-certificates.conf",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            # the ONLY writable surfaces: the disposable worktree and the throwaway home
            "--bind", str(cwd), str(cwd), "--bind", str(home), str(home),
            "--setenv", "HOME", str(home), "--chdir", str(cwd),
            *agent_binds,
            *argv,
        )


class SandboxUnavailable(RuntimeError):
    """No sandbox could be verified to actually confine the agent on this host (fail-closed)."""


def verify_sandbox(sandbox: SandboxRunner, worktrees_dir: Path) -> None:
    """PROVE the sandbox confines, or raise :class:`SandboxUnavailable`.

    Presence is not proof. The probe plants a canary "secret" OUTSIDE the sandboxed surfaces and asserts
    the sandboxed process CANNOT read it. A sandbox that runs but does not isolate would otherwise be
    indistinguishable from one that works - and would be actively harmful, since the operator would grant
    unattended use on the strength of it."""
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    probe = worktrees_dir / f"sandbox-probe-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    (probe / "wt").mkdir(parents=True, exist_ok=True)
    (probe / "home").mkdir(parents=True, exist_ok=True)
    canary = probe / "CANARY-SECRET"
    canary.write_text("gho_" + "c" * 36, encoding="utf-8")
    try:
        argv = sandbox.wrap(("/bin/sh", "-c", f"cat {canary} 2>/dev/null || echo __DENIED__"),
                            cwd=probe / "wt", home=probe / "home")
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell at this level.
                list(argv), capture_output=True, text=True, timeout=_SANDBOX_PROBE_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as exc:
            raise SandboxUnavailable(f"sandbox could not run: {type(exc).__name__}: {exc}") from exc
        if proc.returncode != 0:
            raise SandboxUnavailable(
                f"sandbox exited {proc.returncode}: {proc.stderr.strip()[:200] or '(no stderr)'}")
        if "__DENIED__" not in proc.stdout:
            raise SandboxUnavailable(
                "sandbox READ a canary secret outside its bound surfaces - it runs but does not isolate")
    finally:
        shutil.rmtree(probe, ignore_errors=True)


class AgentClient(Protocol):
    """The one seam through which the loop reaches the (untrusted) agent. ``propose`` returns a proposed
    edit for the current goal; it RAISES on any transport / output failure (the caller fails closed)."""

    def propose(self, request: dict[str, Any]) -> dict[str, Any]:  # noqa: C901 - linear validation

        """Given ``{goal, goal_id, base_oid, directive, repo_root}`` return ``{"unified_diff": str,
        "rationale": str}``. Must raise (not return a partial/garbage dict) on failure."""
        ...


class AgentError(LoopOrchestrateError):
    """The agent transport failed or returned output the adapter refuses to trust (fail-closed).

    Subclasses :class:`LoopOrchestrateError` so a routine transport/validation refusal escalates as an
    EXPECTED operational failure rather than being labelled a likely controller bug with a traceback."""


# ---------------------------------------------------------------- the real transport contract (measured)
# `claude -p --output-format json` does NOT return the agent's answer directly: it returns Claude Code's
# OWN envelope, with the model's text nested in `result` as a STRING. Measured:
#   {"type":"result","subtype":"success","is_error":false,"result":"pong","session_id":"...", ...}
# The adapter previously did json.loads(stdout) and looked for `unified_diff` at the TOP level, so against
# a real agent it failed 100% of the time. Nothing caught it because every test injected a stub - the
# seam looked implemented because its tests only ever exercised the fake.
_ENVELOPE_RESULT_KEY = "result"
_FENCE = re.compile(r"\A\s*```(?:json)?\s*\n(?P<body>.*?)\n\s*```\s*\Z", re.DOTALL)

# What the agent is actually ASKED for. The transport used to pipe the raw request dict with NO
# instruction at all, so the model was never told what shape to reply in - the second half of why the
# propose path could not work against a real agent.
_PROMPT = """You are proposing a SINGLE, SMALL code change. You are running in a disposable, throwaway
git worktree: nothing you write to disk is kept. ONLY the JSON you print is used.

GOAL: {goal}

CONSTRAINTS (a proposal violating any of these is refused by the runtime, so do not attempt it):
  * MODIFY EXISTING FILES ONLY - never create, delete, rename or move a file.
  * Only these paths may be changed: {writable}
  * At most {max_paths} file(s), {max_lines} changed lines, {max_bytes} bytes of patch.
  * No secrets, tokens or credentials. No invisible/formatting Unicode characters.
  * Keep it minimal and self-contained; prefer the smallest correct change.

Read the code first (Read/Grep/Glob are available; you cannot edit, run shell, or reach the network).

Reply with ONE JSON object and NOTHING else:
{{"unified_diff": "<a unified diff that applies with `git apply` at the repository root>",
  "rationale": "<one or two sentences on what you changed and why>"}}

The diff must use `--- a/<path>` and `+++ b/<path>` headers with correct @@ hunks."""


def render_prompt(request: dict[str, Any]) -> str:
    """The instruction actually sent to the agent. Carries the goal AND the bounds the runtime will
    enforce - telling the model the rules up front turns most refusals into changes it simply does not
    attempt, which is cheaper and clearer than refusing after the fact."""
    limits = request.get("_limits") or {}
    writable = ", ".join(limits.get("writable_globs") or ["(none declared - nothing is writable)"])
    return _PROMPT.format(
        goal=request.get("goal") or "(no goal supplied)",
        writable=writable,
        max_paths=limits.get("max_changed_paths", "?"),
        max_lines=limits.get("max_changed_lines", "?"),
        max_bytes=limits.get("max_changed_bytes", "?"),
    )


def _unwrap_envelope(stdout: str) -> Any:
    """The agent's own JSON payload, pulled out of Claude Code's result envelope. Fail-closed at every
    step: a non-success subtype, `is_error`, a missing/non-string `result`, or a `result` that is not one
    JSON object all raise rather than being coerced into something usable."""
    try:
        envelope = json.loads(stdout)
    except (ValueError, RecursionError) as exc:
        raise AgentError(f"claude produced no usable JSON envelope: {exc}") from exc
    if not isinstance(envelope, dict):
        raise AgentError(f"claude envelope is not an object (got {type(envelope).__name__})")
    if envelope.get("is_error") is True or envelope.get("subtype") not in (None, "success"):
        raise AgentError(
            f"claude reported failure: subtype={envelope.get('subtype')!r} "
            f"api_error_status={envelope.get('api_error_status')!r}")
    body = envelope.get(_ENVELOPE_RESULT_KEY)
    if not isinstance(body, str):
        raise AgentError(f"claude envelope has no string {_ENVELOPE_RESULT_KEY!r} "
                         f"(got {type(body).__name__}); the transport contract has changed")
    fenced = _FENCE.match(body)
    if fenced:                      # a single ```json fence is unambiguous and extremely common
        body = fenced.group("body")
    try:
        return json.loads(body)
    except (ValueError, RecursionError) as exc:
        raise AgentError(f"the agent's reply is not one JSON object: {exc}") from exc


def _validate_proposal(raw: Any) -> tuple[str, str]:
    """Fail-closed validation of the agent's response into ``(unified_diff, rationale)``. Untrusted input:
    a non-dict, a missing/non-string ``unified_diff``, or a non-string ``rationale`` all raise."""
    if not isinstance(raw, dict):
        raise AgentError(f"agent response is not an object (got {type(raw).__name__})")
    diff = raw.get("unified_diff")
    if not isinstance(diff, str):
        raise AgentError("agent response has no string 'unified_diff'")
    rationale = raw.get("rationale", "")
    if not isinstance(rationale, str):
        raise AgentError("agent response 'rationale' is not a string")
    return diff, rationale


# The read-only tool policy for the PROPOSE phase: the agent may inspect the checkout but not edit it, run
# shell, spawn nested agents, or mutate git/network. Passed to ``claude`` as a DEFAULT; the flag names are
# version-sensitive, so ``argv`` is operator-tunable and MUST be verified against the installed CLI. NOTE:
# prompt/tool restrictions are DEFENCE-IN-DEPTH, not process isolation - the load-bearing confinement is
# that the agent runs with cwd inside a DISPOSABLE worktree with a sanitized (secret-free) environment.
_READONLY_TOOL_ARGV: tuple[str, ...] = (
    "claude", "-p", "--output-format", "json",
    "--allowedTools", "Read,Grep,Glob",
    "--disallowedTools", "Edit,Write,Bash,Task,WebFetch,WebSearch,NotebookEdit",
)


class ClaudeSubprocessClient:
    """The real transport: run ``claude`` OUT OF PROCESS (fixed argv, no shell), CONFINED to a disposable
    worktree and a sanitized environment, feed the request as JSON on stdin, and read one JSON object with
    ``unified_diff`` / ``rationale`` from stdout. Confinement:
      * ``cwd`` = ``request['_cwd']`` (the isolated worktree the executor created) - the agent operates on a
        throwaway checkout, never the developer's tree. This is REQUIRED: a missing / non-directory ``_cwd``
        fails closed (the transport refuses to run UNCONFINED), so isolation cannot be bypassed by omission;
      * ``env`` = :func:`_sanitized_env` - no GitHub / cloud / release / registry secrets are inherited;
      * a read-only tool policy (``argv`` default) denies edit/write/bash/nested-agents/net (defence in depth);
      * the call is bounded by a timeout AND ``max_output_bytes`` (killable) - an oversized/hung agent fails
        closed, never hangs the loop or exhausts memory.
    A non-zero exit / un-runnable binary / unparseable / wrong-shaped / oversized response raises
    :class:`AgentError`. Live behaviour is env-dependent (the CLI must exist + honour the argv)."""

    def __init__(self, *, argv: tuple[str, ...] = _READONLY_TOOL_ARGV, timeout: float = _AGENT_TIMEOUT_S,
                 max_output_bytes: int = _MAX_AGENT_OUTPUT_BYTES,
                 sandbox: "SandboxRunner | None" = None) -> None:
        self.argv = argv
        self.timeout = timeout
        self.max_output_bytes = max_output_bytes
        # When set, the agent argv is wrapped for OS-level isolation. The caller is responsible for having
        # VERIFIED it (verify_sandbox) - this class does not silently degrade to unsandboxed.
        self.sandbox = sandbox

    def propose(self, request: dict[str, Any]) -> dict[str, Any]:  # noqa: C901 - linear validation

        # Confinement is MANDATORY at the transport, not merely a convention the executor follows: a
        # missing / non-directory `_cwd` would run the agent in the process's OWN cwd (the developer's
        # tree) with no isolation. Refuse it - fail closed rather than silently run UNCONFINED.
        cwd = request.get("_cwd")  # the isolated worktree; the agent runs THERE, not the developer's tree
        if not isinstance(cwd, str) or not cwd or not Path(cwd).is_dir():
            raise AgentError("refusing to run the agent unconfined: request['_cwd'] must be an existing worktree")
        home = request.get("_home")
        argv: tuple[str, ...] = tuple(self.argv)
        if self.sandbox is not None:
            argv = self.sandbox.wrap(argv, cwd=Path(cwd), home=Path(home) if home else Path(cwd))
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, bounded timeout, confined cwd+env.
                list(argv), input=render_prompt(request), text=True, capture_output=True,
                timeout=self.timeout, cwd=cwd,
                env=_sanitized_env(Path(request["_home"]) if request.get("_home") else None))
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AgentError(f"claude could not run / timed out: {type(exc).__name__}: {exc}") from exc
        if len(proc.stdout.encode("utf-8", "replace")) > self.max_output_bytes:
            raise AgentError(f"claude output exceeded {self.max_output_bytes} bytes; refusing oversized output")
        if proc.returncode != 0:
            raise AgentError(f"claude exited {proc.returncode}: {proc.stderr.strip()[:200]}")
        diff, rationale = _validate_proposal(_unwrap_envelope(proc.stdout))
        return {"unified_diff": diff, "rationale": rationale}


def _changed_paths_of(unified_diff: str) -> list[str]:
    """Repo-relative paths named by a unified diff's ``--- a/<p>`` / ``+++ b/<p>`` headers (``/dev/null``
    excluded). Descriptive only - the human reviews the diff itself."""
    paths: set[str] = set()
    for line in unified_diff.splitlines():
        for marker in ("--- a/", "+++ b/"):
            if line.startswith(marker) and line[len(marker):] != "/dev/null":
                paths.add(line[len(marker):].split("\t", 1)[0])
    return sorted(paths)


def _seal_proposal(payload: dict[str, Any]) -> dict[str, Any]:
    """A tamper-evident ``agent_proposal`` record: the envelope plus a ``record_digest`` over the canonical
    (sorted-key) bytes of everything but the digest itself - so a reviewer can re-verify it."""
    envelope = {"record_type": "agent_proposal", "schema_version": 1, "payload": payload}
    digest = hashlib.sha256(json.dumps(envelope, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return {**envelope, "record_digest": f"sha256:{digest}"}


def _write_proposal_record(proposals_dir: Path, record: dict[str, Any]) -> Path:
    """Persist a sealed proposal to a CONTENT-ADDRESSED file (named by its record digest) OUTSIDE the
    working tree, and return the path. Content-addressing makes the name collision-free and idempotent -
    no dependence on directory state or a run counter."""
    proposals_dir.mkdir(parents=True, exist_ok=True)
    out = proposals_dir / f"{record['record_digest'].split(':', 1)[-1]}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _resolve_base_oid(repo_root: Path, base: str) -> str:
    """The 40-hex commit the proposal is against (read-only ``git rev-parse``); '' if it cannot resolve."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell.
            ["git", "-C", str(repo_root), "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _make_executor(agent_client: AgentClient, repo_root: Path, base: str, proposals_dir: Path,
                   worktrees_dir: Path, limits: dict[str, Any] | None = None):  # noqa: ANN202
    """Build the propose-only executor. At EXECUTE it runs the agent CONFINED to a disposable worktree and
    seals the returned diff; at DECOMPOSE it installs one self-owned placeholder task (so the graph phases
    proceed without a delegated, write-capable sub-agent); at every other executor phase it advances."""

    def execute(state: LoopState, directive: Directive) -> Observation:
        if state.current_phase is Phase.DECOMPOSE and not state.task_graph:
            state.task_graph = [{
                "id": "propose", "description": "single-agent proposal (no edits applied)",
                "owner": "self", "allowed_paths": [], "depends_on": [], "status": "PENDING",
            }]
            return Observation.SUCCESS
        if state.current_phase is not Phase.EXECUTE:
            return Observation.SUCCESS  # reasoning phases: advance (nothing to propose yet)

        base_oid = _resolve_base_oid(repo_root, base)
        if not base_oid:
            raise AgentError(f"cannot resolve base {base!r} to a commit for the confined worktree")
        # CONFINE the agent: run it with cwd inside a disposable, detached worktree at base (never the
        # developer's tree) + a secret-free env. Anything it writes there is discarded; we use only its
        # returned diff. So even a mis-behaving agent cannot edit the working tree while "proposing".
        with _detached_worktree(repo_root, base_oid, worktrees_dir) as wt, \
                _confined_home(worktrees_dir) as home:
            request = {"goal": state.goal, "goal_id": state.goal_id, "base_oid": base_oid,
                       "repo_root": str(repo_root), "_cwd": str(wt), "_home": str(home),
                       # the SAME bounds the write path would enforce: a propose-only run that suggests
                       # changes 7.1 would refuse teaches the operator nothing.
                       "_limits": limits or {},
                       "directive": {"phase": directive.phase, "action": directive.action,
                                     "allowed_paths": list(directive.allowed_paths)}}
            result = agent_client.propose(request)  # RAISES on failure -> step escalates (fail-closed)
        diff, rationale = _validate_proposal(result)
        record = _seal_proposal({"goal_id": state.goal_id, "base_oid": base_oid, "unified_diff": diff,
                                 "changed_paths": _changed_paths_of(diff), "rationale": rationale})
        # Persist the sealed proposal OUTSIDE the working tree (content-addressed) and reference it.
        out = _write_proposal_record(proposals_dir, record)
        # Invariant: a written proposal is ALWAYS referenced on the state. Normalize a missing/corrupt
        # value to a list (never silently skip the append and leave disk + state disagreeing).
        refs = state.review_state.get("agent_proposals")
        if not isinstance(refs, list):
            refs = state.review_state["agent_proposals"] = []
        refs.append({"record_digest": record["record_digest"], "path": str(out),
                     "changed_paths": record["payload"]["changed_paths"]})
        return Observation.SUCCESS

    return execute


def _signoff_critic(_state: LoopState) -> list[Criterion]:
    """A proposal is not a completed goal: require a HUMAN_APPROVAL sign-off, so the loop ESCALATES at
    VERIFY (surfacing the proposed diff) instead of autonomously finalizing - the agent cannot self-certify."""
    return [Criterion("agent-proposal-signoff", "a human reviews + decides whether to apply the proposed diff",
                      kind=CriterionKind.HUMAN_APPROVAL)]


def build_context(repo_root: Path | str, base: str = "main", *, agent_client: AgentClient | None = None,
                  proposals_dir: Path | str | None = None, worktrees_dir: Path | str | None = None,
                  pr_ref: str | None = None, run_cs_assure: Any = None, gh_runner: Any = None,
                  profile: Any = None, sandbox: Any = None) -> LoopContext:
    """A READ-ONLY, propose-only :class:`LoopContext` driven by a real single agent. ``capabilities`` is
    EMPTY (the capability gate runs it with no opt-in). The executor runs ``agent_client`` (defaulting to
    the out-of-process :class:`ClaudeSubprocessClient`) CONFINED to a disposable, detached worktree at
    ``base`` (never the developer's tree) and seals the diff it returns; nothing is ever applied, pushed,
    or merged. Pass a stub ``agent_client`` + a ``proposals_dir`` in tests. A ``pr_ref`` additionally
    exercises the real CI read + merge gate, still guaranteed not to merge (``dangerous=True`` escalates
    first and the gh runner refuses mutations)."""
    root = Path(repo_root)
    limits: dict[str, Any] = {}
    if profile is not None:
        from loop_adapters.target_profile import TargetProfile, load_profile
        prof = profile if isinstance(profile, TargetProfile) else load_profile(profile)
        limits = {"writable_globs": list(prof.writable_globs),
                  "max_changed_paths": prof.max_changed_paths,
                  "max_changed_lines": prof.max_changed_lines,
                  "max_changed_bytes": prof.max_changed_bytes}
    if sandbox is not None:
        verify_sandbox(sandbox, Path(worktrees_dir) if worktrees_dir else default_worktrees_dir(root))
    client = agent_client if agent_client is not None else ClaudeSubprocessClient(sandbox=sandbox)
    pdir = Path(proposals_dir) if proposals_dir is not None else _default_proposals_dir(root)
    wtdir = Path(worktrees_dir) if worktrees_dir is not None else default_worktrees_dir(root)
    kwargs: dict[str, Any] = {
        "repo_root": root, "base": base,
        "executor": _make_executor(client, root, base, pdir, wtdir, limits),
        "reviewer": lambda _state: [],
        "critic": _signoff_critic,
        "multi_agent": False,                 # single-agent: no delegated (write-capable) sub-agents
        "capabilities": frozenset(),          # READ-ONLY / propose-only - declares no write capability
    }
    if run_cs_assure is not None:
        kwargs["run_cs_assure"] = run_cs_assure
    if pr_ref is not None:
        kwargs.update(gh_runner=gh_runner or read_only_gh(root), pr_ref=pr_ref, dangerous=True)
    return LoopContext(**kwargs)


def _default_proposals_dir(repo_root: Path) -> Path:
    """``<git-dir>/corpusstudio-loop/proposals`` (OUTSIDE the working tree), or a worktree-local fallback
    when not inside a git repo. A read-only ``git rev-parse`` resolves the git dir."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell.
            ["git", "-C", str(repo_root), "rev-parse", "--git-path", "corpusstudio-loop/proposals"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return repo_root / ".corpusstudio-loop-proposals"
    if proc.returncode != 0:
        return repo_root / ".corpusstudio-loop-proposals"
    rel = proc.stdout.strip()
    p = Path(rel)
    return p if p.is_absolute() else repo_root / rel
