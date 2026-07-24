"""Phase 7.1 runtime adapter: write-capable single agent (GATED - needs an explicit operator opt-in).

The write step of the Production Single-Agent Runtime (docs/PRODUCTION_SINGLE_AGENT_RUNTIME.md). The agent
still PROPOSES a unified diff (exactly the 7.0 behaviour, sealed as an ``agent_proposal`` record) - run
CONFINED to a disposable, detached worktree with a secret-free environment; 7.1 then APPLIES that exact
sealed diff in a SEPARATE, pristine ``git worktree``, ASSURES that candidate (phase 7.1.2), and - only if
the candidate is green - pushes the branch and opens a PR. It NEVER merges (the merge gate escalates; a
human merges the PR).

Safety (why this is the least-capable *write* rung):
  * ``capabilities = {"write"}`` - the capability gate REFUSES to run it without ``--allow-capabilities
    write``. The read-only 7.0 adapter is untouched; this write path is a separate, separately-reviewed file.
  * The untrusted agent runs CONFINED (phase 7.1.1): cwd inside a disposable, detached worktree at ``base``
    (never the developer's tree, never the apply worktree) with a sanitized (secret-free) environment and a
    read-only tool policy. Whatever it writes there is discarded; only the diff it RETURNS is used.
  * SENSITIVE-PATH DENIAL (phase 7.1.2, pre-apply, fail closed): the sealed diff is refused before it is
    applied if it touches a protected path (the loop-controller / assurance / agent config, CI + supply
    chain, sealed research), a credential-shaped file (``.env``, ``*.pem``, ``*.key``, ...), a symlink /
    file-mode / binary change, or too many files. A human makes those edits.
  * The unit of change is the agent's OWN sealed diff, applied deterministically with ``git apply`` into a
    SEPARATE pristine DETACHED worktree the agent never touched - the agent does not edit arbitrarily. The
    applied change is verified to match the sealed proposal's ``changed_paths`` (integrity, fail closed),
    and the added lines are scanned for apparent secrets (defense-in-depth; CI gitleaks is the backstop).
  * CANDIDATE ASSURANCE (phase 7.1.2, STATIC): the candidate worktree is classified via ``cs_assure
    impact`` (self-modify / sealed-research / worker-closure / policy obligations) - a STATIC analysis that
    does NOT import or execute the candidate's code. The branch is pushed + the PR opened ONLY when the
    candidate classifies clear; a human-gated / worker / policy obligation publishes NOTHING and returns the
    classified observation so the loop escalates. Because the apply worktree is DETACHED and the branch is
    created by refspec only on a clear candidate, a blocked candidate leaves no orphaned local branch.
    The DYNAMIC gate (ruff/mypy/PYTEST) is deliberately NOT run locally: pytest collection would EXECUTE the
    untrusted candidate's code with the operator's environment (arbitrary code execution + secret exposure -
    the very thing 7.1.1 confinement prevents). It is delegated to CI's isolated sandbox on the opened PR,
    where a human merges only on green. Every cs_assure subprocess also runs with a SANITIZED (secret-free)
    environment; ``gh`` (which needs the token) is separate.
  * All edits/commits/assurance happen in isolated worktrees created from ``base``; the main working tree
    is never touched. Both worktrees are disposed on exit (a pushed branch persists as the PR head).
  * NO autonomous merge: the ``gh`` runner allows ``pr create`` + reads but REFUSES ``pr merge`` (and every
    other mutation), and ``dangerous=True`` escalates the merge gate; a human reviews + merges the PR.

Deferred to later 7.1 rungs (documented in the design doc): a persistent candidate worktree with a
loop-native observe->diagnose->correct loop + an independent REAL reviewer (7.1.2b), exact candidate
identity (7.1.3), and a crash-resumable write-ahead journal + orphan-branch gc + DRAFT PRs (7.1.4).

stdlib-only; every git/gh effect is a fixed-argv subprocess (no shell) and fails closed. Reuses the 7.0
building blocks (the agent client, proposal sealing, diff parsing) and the loop's ``observe`` classifier.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess  # noqa: S404 - fixed-argv git / gh / cs_assure only; never a shell string.
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from loop.controller import HUMAN_GATED_OBLIGATIONS, LoopState, Observation, Phase
from loop.driver import Directive
from loop.orchestrate import CAP_WRITE, LoopContext
from loop_adapters.single_agent import (
    AgentClient,
    AgentError,
    ClaudeSubprocessClient,
    _changed_paths_of,
    _default_proposals_dir,
    _detached_worktree,
    _resolve_base_oid,
    _sanitized_env,
    _seal_proposal,
    _signoff_critic,
    _validate_proposal,
    _write_proposal_record,
    default_worktrees_dir,
)


def _sanitize_branch_suffix(goal_id: str) -> str:
    """A SAFE git branch suffix from a goal id: lowercase, any run of non-``[a-z0-9]`` -> ``-``, leading/
    trailing ``-`` stripped, truncated. So a goal id with spaces / uppercase / punctuation / ``..`` can
    never produce an invalid ref or violate a remote policy; the original goal id stays in the sealed
    proposal record + the PR body for traceability."""
    slug = re.sub(r"[^a-z0-9]+", "-", (goal_id or "").lower()).strip("-")
    return slug[:40].strip("-") or "goal"

_GIT_TIMEOUT_S = 120
_GH_TIMEOUT_S = 60
_CS_ASSURE_TIMEOUT_S = 1800  # a bounded cs_assure call (impact is fast; the dev-tree gate can take minutes)
_MAX_CANDIDATE_FILES = 50  # a single autonomous change touching more than this is refused (review-by-human)


def _sanitized_cs_assure(root: Path | str, *argv: str) -> tuple[int, str, str]:
    """The write adapter's DEFAULT cs_assure runner: like ``loop.observe._run_cs_assure`` but with a
    SANITIZED (secret-free) environment, so no cs_assure subprocess inherits GitHub / cloud / release
    credentials. ``gh`` (which needs the token to open the PR) is a SEPARATE subprocess and keeps its env.
    Fail-closed: a spawn error / timeout returns exit 2 (a refusal), never raises into the loop."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv (python + repo script), no shell, sanitized env.
            [sys.executable, str(Path(root) / "scripts" / "cs_assure.py"), *argv],
            cwd=str(root), capture_output=True, text=True, env=_sanitized_env(), timeout=_CS_ASSURE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        return (2, "", f"cs_assure could not run: {type(exc).__name__}: {exc}")
    return proc.returncode, proc.stdout, proc.stderr

# ---------------------------------------------------------------- sensitive-path denial (7.1.2, pre-apply)
# Paths the write adapter REFUSES to create/modify autonomously; a human must make these edits. Matched
# against every changed path BEFORE the diff is applied - a single hit denies the whole proposal (fail
# closed). This is the "initially deny" list from the 7.1 review: credentials / key material, CI +
# supply-chain, the loop-controller + assurance + agent config (the self-modify class), and sealed research.
_DENIED_DIR_PREFIXES: tuple[str, ...] = (
    "scripts/loop/", "scripts/loop_adapters/", "scripts/assurance/", ".claude/",
    ".github/", "research/", "docs/paper/",
)
_DENIED_EXACT: frozenset[str] = frozenset({
    "scripts/cs_loop.py", "scripts/cs_assure.py", ".gitmodules", ".gitattributes",
    ".npmrc", ".pypirc", ".netrc",
})
_DENIED_NAME_GLOBS: tuple[str, ...] = (
    # credential / key material (by filename, anywhere)
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "*.pkcs12", "*.keystore", "*.jks",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*", "*.ppk", "*.gpg", "*.asc", "credentials",
    # GATE-AFFECTING config - denied so an agent's own diff cannot NEUTRALIZE the ruff/mypy/pytest gate
    # (local or CI): e.g. a `conftest.py` with `collect_ignore_glob=['*']`, or a permissive tool config.
    "conftest.py", "pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg", "setup.py",
    "ruff.toml", ".ruff.toml", "mypy.ini", ".mypy.ini", ".flake8", ".coveragerc",
    ".pre-commit-config.yaml", ".gitignore",
)
# Raw-diff markers that indicate a change the adapter refuses to make autonomously (symlink / mode / binary).
_STRUCTURAL_DENY_MARKERS: tuple[tuple[str, str], ...] = (
    ("mode 120000", "a symlink"), ("old mode ", "a file-mode change"),
    ("mode 100755", "an executable-bit change"),
    ("GIT binary patch", "a binary patch"), ("Binary files ", "a binary file"),
)

# Credential SHAPES scanned in the diff's ADDED lines (defense-in-depth; CI gitleaks is the backstop). A hit
# refuses the candidate rather than committing an apparent secret.
_SECRET_PATTERNS: tuple[tuple["re.Pattern[str]", str], ...] = (
    (re.compile(r"AKIA[0-9A-Z]{16}"), "an AWS access key id"),
    (re.compile(r"ghp_[A-Za-z0-9]{36}"), "a GitHub token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{50,}"), "a GitHub fine-grained PAT"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "a Slack token"),
    (re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"), "a private key block"),
    (re.compile(r"(?i)aws_secret_access_key\s*[=:]\s*[A-Za-z0-9/+]{40}"), "an AWS secret access key"),
    (re.compile(r"""(?i)(?:secret|password|passwd|api[_-]?key|access[_-]?token)\s*[=:]\s*['"][^'"]{12,}['"]"""),
     "a hardcoded credential"),
)


def _classify_sensitive_paths(changed_paths: list[str], unified_diff: str) -> list[str]:
    """Reasons the sealed proposal must NOT be written autonomously (empty = clear). Fail-closed pre-apply
    denial: a protected path, a credential-shaped filename, a symlink / mode / binary change, or too many
    files each denies the whole proposal so a human makes the edit instead."""
    reasons: list[str] = []
    if len(changed_paths) > _MAX_CANDIDATE_FILES:
        reasons.append(f"touches {len(changed_paths)} files (> {_MAX_CANDIDATE_FILES}; needs human review)")
    for p in changed_paths:
        posix = p.replace("\\", "/")
        name = posix.rsplit("/", 1)[-1]
        if posix in _DENIED_EXACT or any(posix.startswith(pre) for pre in _DENIED_DIR_PREFIXES):
            reasons.append(f"{p} (protected path - not an autonomous edit)")
        elif any(fnmatch.fnmatch(name, g) for g in _DENIED_NAME_GLOBS):
            reasons.append(f"{p} (credential-shaped file)")
    for marker, why in _STRUCTURAL_DENY_MARKERS:
        if marker in unified_diff:
            reasons.append(f"diff contains {why}")
    return reasons


def _scan_secrets(unified_diff: str) -> list[str]:
    """Apparent secrets in the diff's ADDED lines (empty = none). Defense-in-depth over the pre-apply path
    denial; the CI gitleaks job is the backstop on the opened PR."""
    hits: set[str] = set()
    for line in unified_diff.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for pat, label in _SECRET_PATTERNS:
            if pat.search(line[1:]):
                hits.add(label)
    return sorted(hits)

# gh subcommands the WRITE adapter permits: the read-only allowlist PLUS ``pr create``. Everything else -
# and crucially ``pr merge`` / ``pr close`` / ``pr edit`` / ``pr review`` / any ``api`` write - is REFUSED,
# so 7.1 can open a PR but can NEVER merge or otherwise mutate a PR autonomously.
_GH_WRITE_ALLOWED = frozenset({
    ("pr", "create"),
    ("pr", "view"), ("pr", "checks"), ("pr", "status"), ("pr", "list"), ("pr", "diff"),
    ("repo", "view"), ("run", "list"), ("run", "view"),
})


class WriteAdapterError(RuntimeError):
    """A git/gh write effect failed or produced output the adapter refuses to trust (fail-closed)."""


def write_gh(repo_root: Path | str) -> Callable[..., "tuple[int, str, str]"]:
    """A ``gh`` runner that runs the write-adapter allowlist (reads + ``pr create``) and REFUSES everything
    else - most importantly ``pr merge``. A refusal is a non-zero exit (never an exception), so the loop's
    CI/merge observation fails closed on it."""
    root = str(repo_root)

    def run(*argv: str) -> tuple[int, str, str]:
        if tuple(argv[:2]) not in _GH_WRITE_ALLOWED:
            return (97, "", f"write_gh: refused non-allowlisted gh '{' '.join(argv[:2])}' (7.1 never merges)")
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv (allowlisted subcommand), no shell.
                ["gh", *argv], cwd=root, capture_output=True, text=True, timeout=_GH_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return (127, "", f"write_gh: gh could not run: {exc}")
        return (proc.returncode, proc.stdout, proc.stderr)

    return run


def _git(cwd: Path, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``git -C <cwd> <args>`` (fixed argv, no shell, bounded). Raises :class:`WriteAdapterError` on a
    non-zero exit or an un-runnable git."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell.
            ["git", "-C", str(cwd), *args], input=stdin, capture_output=True, text=True, timeout=_GIT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WriteAdapterError(f"git {' '.join(args[:2])} could not run: {exc}") from exc
    if proc.returncode != 0:
        raise WriteAdapterError(f"git {' '.join(args[:2])} failed (exit {proc.returncode}): {proc.stderr.strip()[:200]}")
    return proc


@contextmanager
def _apply_worktree(repo_root: Path, base_oid: str, worktrees_dir: Path) -> Iterator[Path]:
    """An ISOLATED, disposable, DETACHED ``git worktree`` at ``base_oid`` under ``worktrees_dir`` (outside
    the main working tree) - the pristine tree the sealed diff is applied + committed + ASSURED in. It is
    DETACHED (no local branch): the committed candidate is pushed to the branch by refspec only AFTER
    assurance is green, so a blocked / red candidate leaves NO orphaned local branch. Removed on exit; a
    best-effort ``worktree remove --force`` never masks the primary error."""
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    wt = worktrees_dir / f"apply-{base_oid[:12]}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    _git(repo_root, "worktree", "add", "--detach", str(wt), base_oid)
    try:
        yield wt
    finally:
        try:
            _git(repo_root, "worktree", "remove", "--force", str(wt))
        except WriteAdapterError:
            pass  # best-effort disposal; a leftover worktree is GC-able, never data loss


def _assure_candidate(wt: Path, base: str, run_cs_assure: Any) -> tuple[Observation, str, str | None]:
    """STATIC candidate assurance: classify the candidate via ``cs_assure impact`` (obligation / policy
    analysis over the diff - it does NOT import or execute the candidate's code) and return
    ``(observation, reason, impact_record_digest)``. ``SUCCESS`` = clear to publish; a human-gated /
    worker-closure / other blocking obligation returns the routing observation so the loop escalates.

    Why static, not the full gate: running ``cs_assure verify`` here would run ruff/mypy/PYTEST inside the
    candidate worktree, and pytest COLLECTION EXECUTES the untrusted agent's committed code with the
    operator's environment - arbitrary code execution + secret exposure, exactly what the 7.1.1 confinement
    prevents for the propose step. The DYNAMIC gate (which must execute the candidate) is therefore
    delegated to CI's isolated sandbox on the opened PR, where the human merges only on green. Fail closed:
    an unusable / refused impact record raises :class:`WriteAdapterError` (-> the loop escalates)."""
    code, out, err = run_cs_assure(wt, "impact", "--base", base)
    if code != 0:
        raise WriteAdapterError(f"candidate impact refused (exit {code}): {err.strip()[:200]}")
    try:
        data = json.loads(out)
    except (ValueError, RecursionError) as exc:
        raise WriteAdapterError(f"candidate impact produced no usable JSON: {exc}") from exc
    payload = data.get("payload") if isinstance(data, dict) else None
    if not isinstance(payload, dict):
        raise WriteAdapterError("candidate impact record has no payload object")
    fired = payload.get("fired_obligations")
    if not isinstance(fired, list):
        # A well-formed impact record ALWAYS has a (possibly empty) list; a missing one is malformed /
        # schema-drifted - fail closed (do not conflate 'uncomputable' with 'nothing fired').
        raise WriteAdapterError("candidate impact record has no fired_obligations list")
    ids = {o["id"] for o in fired if isinstance(o, dict) and isinstance(o.get("id"), str)}
    digest = data.get("record_digest") if isinstance(data, dict) else None
    digest = digest if isinstance(digest, str) else None
    human = sorted((ids & HUMAN_GATED_OBLIGATIONS) - {"worker-closure"})
    if human:
        return Observation.AUTHORIZATION_REQUIRED, f"candidate fires {', '.join(human)} (human review)", digest
    if "worker-closure" in ids:
        return Observation.WORKER_LINEAGE_IMPACT, "candidate touches worker-execution bytes", digest
    blocking = sorted(o["id"] for o in fired if isinstance(o, dict)
                      and o.get("severity") == "blocking" and isinstance(o.get("id"), str))
    if blocking:
        return Observation.POLICY_BLOCK, f"candidate fires blocking obligation(s): {', '.join(blocking)}", digest
    return Observation.SUCCESS, "candidate static assurance clean (dynamic gate delegated to CI)", digest


def _record_candidate_assurance(state: LoopState, record: dict[str, Any], branch: str, observation: Observation,
                                reason: str, impact_digest: str | None, *, published: bool) -> None:
    """Store the CANDIDATE's assurance outcome on the loop state (audit trail): the observation the static
    impact classified for the candidate worktree, why, whether it was published, and the sealed impact
    digest. Kept OUT of the completeness ``evidence`` index (a clean candidate is not a human sign-off - the
    merge stays human-gated via the critic)."""
    if impact_digest and impact_digest not in state.assurance_records:
        state.assurance_records.append(impact_digest)
    state.review_state["candidate_assurance"] = {
        "record_digest": record["record_digest"], "branch": branch,
        "observation": observation.value, "reason": reason,
        "published": published, "impact_record_digest": impact_digest,
    }


def _make_write_executor(agent_client: AgentClient, repo_root: Path, base: str, proposals_dir: Path,
                         worktrees_dir: Path, gh_runner: Callable[..., "tuple[int, str, str]"],
                         branch_prefix: str, run_cs_assure: Any):  # noqa: ANN202
    """The write executor. At EXECUTE: PROPOSE (agent, CONFINED to a disposable detached worktree) -> seal
    -> DENY sensitive/credential paths (pre-apply, fail closed) -> APPLY the exact diff in a SEPARATE,
    pristine detached worktree -> verify the applied change matches the proposal -> scan for secrets ->
    commit -> ASSURE THE CANDIDATE (run the real gate + impact against the candidate worktree, not the dev
    tree) -> publish (push the branch + open a PR) ONLY IF the candidate is green. A non-green candidate
    publishes NOTHING and returns the classified observation so the loop routes it (gate-red -> revise;
    worker/self-modify/policy -> escalate). At DECOMPOSE it installs one self-owned task. Integrity /
    security violations raise (-> the loop escalates)."""

    def execute(state: LoopState, directive: Directive) -> Observation:
        if state.current_phase is Phase.DECOMPOSE and not state.task_graph:
            state.task_graph = [{
                "id": "assure-and-pr", "description": "assure the agent's diff in an isolated worktree + open a PR",
                "owner": "self", "allowed_paths": [], "depends_on": [], "status": "PENDING",
            }]
            return Observation.SUCCESS
        if state.current_phase is not Phase.EXECUTE:
            return Observation.SUCCESS

        base_oid = _resolve_base_oid(repo_root, base)
        if not base_oid:
            raise WriteAdapterError(f"cannot resolve base {base!r} to a commit to branch from")
        # 1) PROPOSE - run the UNTRUSTED agent CONFINED: cwd inside a disposable, detached worktree at base
        #    (never the developer's tree AND never the apply worktree) with a secret-free env. Whatever it
        #    writes into that throwaway checkout is discarded on exit; only the diff it RETURNS is used.
        with _detached_worktree(repo_root, base_oid, worktrees_dir) as propose_wt:
            request = {"goal": state.goal, "goal_id": state.goal_id, "base_oid": base_oid,
                       "repo_root": str(repo_root), "_cwd": str(propose_wt),
                       "directive": {"phase": directive.phase, "action": directive.action,
                                     "allowed_paths": list(directive.allowed_paths)}}
            diff, rationale = _validate_proposal(agent_client.propose(request))  # RAISES -> fail-closed
        changed_paths = _changed_paths_of(diff)
        record = _seal_proposal({"goal_id": state.goal_id, "base_oid": base_oid, "unified_diff": diff,
                                 "changed_paths": changed_paths, "rationale": rationale})
        proposal_path = _write_proposal_record(proposals_dir, record)  # content-addressed, OUTSIDE the tree

        # 2) SENSITIVE-PATH DENIAL (pre-apply, fail closed): refuse to write protected / credential-shaped
        #    paths, symlink/mode/binary changes, or an over-large change - a human makes those edits.
        denials = _classify_sensitive_paths(changed_paths, diff)
        if denials:
            raise WriteAdapterError("refusing to write denied path(s): " + "; ".join(denials[:8]))

        branch = f"{branch_prefix}{_sanitize_branch_suffix(state.goal_id)}"
        # 3) APPLY + ASSURE in a pristine DETACHED worktree the agent never touched, so the commit is
        #    deterministically the sealed diff and nothing else; publish only if the candidate is green.
        with _apply_worktree(repo_root, base_oid, worktrees_dir) as wt:
            _git(wt, "apply", "--index", "-", stdin=diff)  # a diff that does not apply cleanly fails closed
            # INTEGRITY: the staged change must be exactly what the sealed proposal described.
            applied = sorted(p for p in _git(wt, "diff", "--cached", "--name-only").stdout.splitlines() if p)
            if applied != record["payload"]["changed_paths"]:
                raise WriteAdapterError(
                    f"applied paths {applied} != proposed {record['payload']['changed_paths']} (diff drift)")
            secrets = _scan_secrets(diff)  # defense-in-depth over the path denial (CI gitleaks is the backstop)
            if secrets:
                raise WriteAdapterError("refusing: the candidate adds " + ", ".join(secrets))
            _git(wt, "-c", "user.name=corpusstudio-agent", "-c", "user.email=agent@corpusstudio.local",
                 "commit", "-m", f"{state.goal or 'agent change'}\n\n{rationale}\n\n[single-agent proposal, human-reviewed]")
            # CANDIDATE ASSURANCE (STATIC): classify the CANDIDATE worktree via `cs_assure impact`
            # (self-modify / sealed-research / worker-closure / policy) - NOT the dev tree, and NOT the
            # code-executing gate (running the untrusted candidate's pytest locally would be arbitrary code
            # execution with the operator's env; the dynamic gate is CI's job on the opened PR). A refusal
            # raises here -> the loop escalates (fail closed).
            observation, reason, impact_digest = _assure_candidate(wt, base, run_cs_assure)
            if observation is not Observation.SUCCESS:
                # The candidate is NOT clear -> publish NOTHING. Record the block and return the classified
                # observation so the controller routes it (human-gated / worker / policy -> escalate).
                _record_candidate_assurance(state, record, branch, observation, reason, impact_digest,
                                            published=False)
                return observation
            # CLEAR candidate -> PUBLISH: push the committed candidate to the branch (by refspec, from the
            # detached HEAD - no local branch was ever created) and open a PR. NEVER merges. CI runs the
            # dynamic gate on the PR in its sandbox; a human merges only on green.
            _git(wt, "push", "-u", "origin", f"HEAD:refs/heads/{branch}")
            code, pr_out, err = gh_runner("pr", "create", "--head", branch, "--base", base,
                                          "--title", (state.goal or "agent change")[:120],
                                          "--body", rationale or "Opened by the single-agent write runtime (7.1).")
            if code != 0:
                raise WriteAdapterError(f"gh pr create failed (exit {code}): {err.strip()[:200]}")
            _record_candidate_assurance(state, record, branch, observation, reason, impact_digest,
                                        published=True)

        refs = state.review_state.get("agent_proposals")
        if not isinstance(refs, list):
            refs = state.review_state["agent_proposals"] = []
        refs.append({"record_digest": record["record_digest"], "branch": branch, "path": str(proposal_path),
                     "changed_paths": record["payload"]["changed_paths"], "pr": pr_out.strip()})
        return Observation.SUCCESS

    return execute


def build_context(repo_root: Path | str, base: str = "main", *, agent_client: AgentClient | None = None,
                  proposals_dir: Path | str | None = None, worktrees_dir: Path | str | None = None,
                  gh_runner: Any = None, branch_prefix: str = "cs-agent/", run_cs_assure: Any = None,
                  pr_ref: str | None = None) -> LoopContext:
    """A WRITE-CAPABLE, single-agent :class:`LoopContext` (Phase 7.1). ``capabilities={CAP_WRITE}`` - the
    capability gate REFUSES to run it without ``--allow-capabilities write``. It applies the agent's sealed
    diff in an isolated worktree, commits, pushes a branch, and opens a PR; it NEVER merges (``write_gh``
    refuses ``pr merge`` and ``dangerous=True`` escalates the merge gate). Inject a stub ``agent_client`` +
    dirs + ``gh_runner`` in tests."""
    root = Path(repo_root)
    client = agent_client if agent_client is not None else ClaudeSubprocessClient()
    pdir = Path(proposals_dir) if proposals_dir is not None else _default_proposals_dir(root)
    wdir = Path(worktrees_dir) if worktrees_dir is not None else default_worktrees_dir(root)
    gh = gh_runner or write_gh(root)
    # The SAME assurance runner drives BOTH the candidate assurance (static `impact` in the executor,
    # against the candidate worktree) and the loop's own OBSERVE/VERIFY (against the dev tree). Default to
    # the SANITIZED-env cs_assure runner so no assurance subprocess inherits secrets.
    assure = run_cs_assure if run_cs_assure is not None else _sanitized_cs_assure
    kwargs: dict[str, Any] = {
        "repo_root": root, "base": base,
        "executor": _make_write_executor(client, root, base, pdir, wdir, gh, branch_prefix, assure),
        "reviewer": lambda _state: [],
        "critic": _signoff_critic,
        "multi_agent": False,               # single-agent: no delegated wave (verify_paths is a 7.3 concern)
        "capabilities": frozenset({CAP_WRITE}),
        "gh_runner": gh,
        "run_cs_assure": assure,
        "dangerous": True,                  # the merge gate ESCALATES: a human merges the PR, never the loop
    }
    if pr_ref is not None:
        kwargs["pr_ref"] = pr_ref
    return LoopContext(**kwargs)


# Re-export so `AgentError` (the confined-propose transport error) and the shared worktree-dir resolver are
# reachable via this module too - the executor may surface either error, and callers/tests use the resolver.
__all__ = ["AgentError", "WriteAdapterError", "build_context", "default_worktrees_dir", "write_gh"]
