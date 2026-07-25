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
  * WRITABLE-SURFACE ALLOWLIST (phase 7.1.2, fail closed): the change set GIT REALIZED - not the diff text
    - must lie inside ``engine/corpus_studio/**/*.py``, be MODIFY-ONLY (status ``M``), use ordinary-file
    modes (100644), carry a safe module basename, and stay within bounds (2 paths / 60 lines / 8 KiB patch
    / 400 B per line / 1 MiB blob). Anything not positively matched is refused, so an unforeseen path is
    denied by construction and a case variant simply fails to match. A denylist was tried first and failed
    three adversarial passes (case variants, ``rename from`` hiding a denied source, ``copy from``
    exfiltrating a key with no ``+`` lines).
  * The unit of change is the agent's OWN sealed diff, applied deterministically with ``git apply`` into a
    SEPARATE pristine DETACHED worktree the agent never touched - the agent does not edit arbitrarily.
    High-confidence credential formats are scanned over the REALIZED BLOB (which is what catches a
    ``copy from``); a looser heuristic scans only the lines the candidate ADDS, so pre-existing content can
    never make a file permanently unpublishable. Invisible / format characters are refused BY UNICODE
    CATEGORY, so the rendered diff cannot differ from the executed bytes.
  * CANDIDATE ASSURANCE (phase 7.1.2, STATIC): the candidate worktree is classified via ``cs_assure
    impact`` (self-modify / sealed-research / worker-closure / policy obligations) - a STATIC analysis run
    by the TRUSTED dev-tree tool pointed at the candidate with ``--start-dir`` (never the candidate's own
    ``scripts/cs_assure.py``, which would execute a candidate-added shadow stdlib module off
    ``sys.path[0]``; the wholesale ``scripts/`` denial is the independent second layer). No untrusted
    candidate code is imported or executed. The branch is pushed + the PR opened ONLY when the
    candidate classifies clear AND the operator has attested CI safety (``ci_attested_safe``, default
    FALSE - see :func:`_publish_precondition_unmet`); a human-gated / worker / policy obligation, or a hit
    in the worker IMPORT CLOSURE, publishes NOTHING and returns the classified observation so the loop
    escalates. Because the apply worktree is DETACHED and the branch is
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

The later 7.1 rungs have LANDED: a persistent candidate worktree with a loop-native
observe->diagnose->correct loop + an independent REAL reviewer (7.1.2b), exact candidate identity
(7.1.3), a crash-resumable write-ahead journal + DRAFT PRs + :func:`collect_orphan_branches` (7.1.4),
and the VERIFIED sandbox seam (7.1.5). The journal's orphan case is not hypothetical - a run whose
remote is not a GitHub host pushes the candidate and then fails at ``gh pr create``, leaving exactly
one branch no PR references; the collector is what reclaims it, and only ever on request.

stdlib-only; every git/gh effect is a fixed-argv subprocess (no shell) and fails closed. Reuses the 7.0
building blocks (the agent client, proposal sealing, diff parsing) and the loop's ``observe`` classifier.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import time
import re
import subprocess  # noqa: S404 - fixed-argv git / gh / cs_assure only; never a shell string.
import sys
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from loop.controller import HUMAN_GATED_OBLIGATIONS, LoopState, Observation, Phase
from loop.driver import Directive
from loop.orchestrate import CAP_WRITE, LoopContext, LoopOrchestrateError
from loop_adapters.target_profile import TargetProfile, load_profile
from loop_adapters.single_agent import (
    AgentClient,
    AgentError,
    ClaudeSubprocessClient,
    _changed_paths_of,
    _default_proposals_dir,
    _confined_home,
    _detached_worktree,
    _resolve_base_oid,
    _sanitized_env,
    _seal_proposal,
    _signoff_critic,
    _validate_proposal,
    _write_proposal_record,
    SandboxUnavailable,
    default_worktrees_dir,
    verify_sandbox,
)


def _sanitize_branch_suffix(goal_id: str) -> str:
    """A SAFE git branch suffix from a goal id: lowercase, any run of non-``[a-z0-9]`` -> ``-``, leading/
    trailing ``-`` stripped, truncated. So a goal id with spaces / uppercase / punctuation / ``..`` can
    never produce an invalid ref or violate a remote policy; the original goal id stays in the sealed
    proposal record + the PR body for traceability."""
    slug = re.sub(r"[^a-z0-9]+", "-", (goal_id or "").lower()).strip("-")[:40].strip("-") or "goal"
    # A short digest of the FULL goal id keeps distinct goals on distinct branches: truncation alone
    # collided (measured: environment_manager.py and environments.py both slugged to the same 40 chars,
    # so the second goal was permanently unpublishable with an opaque non-fast-forward push error).
    return f"{slug}-{hashlib.sha256((goal_id or '').encode('utf-8')).hexdigest()[:8]}"

_GIT_TIMEOUT_S = 120
_GH_TIMEOUT_S = 60
# A FIXED commit timestamp makes the candidate commit CONTENT-ADDRESSED: identical content yields an
# identical oid, so a resumed or re-run goal reproduces the same commit instead of a new one every time
# (measured: without this, identical content produced two different commit oids).
_FIXED_DATE = "@0 +0000"

_DEFAULT_GATE_RELPATH = "scripts/assurance/policy/gate.json"

_CS_ASSURE_TIMEOUT_S = 1800  # a bounded cs_assure call (impact is fast; the dev-tree gate can take minutes)


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


def _trusted_cs_assure(repo_root: Path, run_cs_assure: Any):  # noqa: ANN202
    """Wrap an assurance runner so a CANDIDATE analysis always executes the TRUSTED dev-tree
    ``scripts/cs_assure.py`` and merely POINTS it at the candidate via ``--start-dir``.

    Why: running the candidate worktree's OWN ``scripts/cs_assure.py`` puts ``<candidate>/scripts`` on
    ``sys.path[0]``, so a candidate that adds e.g. ``scripts/argparse.py`` (or ``json.py`` /
    ``sitecustomize.py``) gets that file IMPORTED AND EXECUTED as the operator before any analysis runs -
    local arbitrary code execution with filesystem + network access (a sanitized env does not stop it
    reading credential FILES). Analyzing the candidate with the trusted tool closes that path; the bare
    ``scripts/`` denial in :func:`_classify_sensitive_paths` is the second, independent layer."""

    def run(root: Path | str, *argv: str) -> tuple[int, str, str]:
        if Path(root) == repo_root:
            return run_cs_assure(root, *argv)  # the dev tree IS the trusted tree - nothing to redirect
        return run_cs_assure(repo_root, *argv, "--start-dir", str(root))

    return run

# ============================================================ the writable surface (7.1.2, ALLOWLIST)
# A DENYLIST of protected paths was tried first and failed three adversarial reviews: it is case-sensitive
# (`Scripts/argparse.py` slipped through), it is blind to `rename from` / `copy from` extended headers, and
# it can never enumerate every future bad path. This is the inverse: an ALLOWLIST of the (tiny) surface an
# autonomous agent may touch. Anything not positively matched is REFUSED, so a path nobody thought of is
# denied by construction - and a case variant simply fails to match.
#
# Rung 7.1 surface = MODIFY-ONLY, one product tree:
#   * `engine/corpus_studio/**/*.py` is the one surface gated end-to-end by ruff + mypy + pytest with an
#     88% coverage floor, and it is where the design doc's stated first target (a docstring / typo / lint
#     fix) lives.
#   * NO CREATES AT ALL (status must be `M`). A new file is how every remaining execution trick lands: a
#     `conftest.py` is auto-executed by pytest; `engine/tests` has NO `__init__.py`, so pytest puts it on
#     `sys.path[0]` and a new `engine/tests/json.py` would shadow stdlib for the whole process.
#   * `engine/tests/test_*.py` is NOT writable yet even though "add a missing test" is a named first
#     target - a new test file is a CREATE, and creates are refused wholesale here.
#
# HONEST CAVEAT about CI (an adversarial pass caught the earlier version of this comment being wrong):
# pushing ANY candidate hands it to CI, which EXECUTES it - `engine-tests.yml` triggers on a bare `push:`
# and runs `pytest --cov=corpus_studio`, and the suite imports the product modules, so a modified
# `engine/corpus_studio/**` file runs at collection. That is the SAME pre-review execution exposure the
# comment previously cited to exclude `engine/tests/**`; it applies to the allowed surface too, and
# "a human merges only on green" does not undo code that already ran. So the exposure is now a
# OPERATOR ATTESTATION, not a machine check: an earlier version tried to VERIFY this by text-scanning
# `.github/workflows/*.yml` and was measured failing OPEN five ways (and it read the dev tree while GitHub
# runs the workflows at `base`). This adapter cannot verify another system's CI, so the operator asserts
# it via `ci_attested_safe` (default False -> never publishes). See `_publish_precondition_unmet`.
_WRITABLE_GLOBS: tuple[str, ...] = (
    "engine/corpus_studio/**/*.py",
)

# Every changed path's basename must be an ordinary, importable, non-shadowing module name.
_SAFE_BASENAME = re.compile(r"(?:__[a-z0-9_]+__|[a-z0-9][a-z0-9_]*)\.py\Z")

# git's raw mode for an ordinary file. A symlink (120000), a gitlink/submodule (160000) and an
# executable (100755) are refused STRUCTURALLY rather than by sniffing the diff text.
_MODE_REGULAR = "100644"

# Bounds. The stated first target is a SINGLE-file change; a merge-sized diff is not something a human
# reviewer skims, so it is refused outright rather than published. BYTES are bounded as well as LINES:
# a line count alone is trivially evaded by putting ~1 MiB on ONE physical line (measured: a 990 KB
# single-line payload reported as "1 insertion(+)"), which defeats the whole point of the bound.
_MAX_CHANGED_PATHS = 2
_MAX_CHANGED_LINES = 60
_MAX_CHANGED_BYTES = 8 * 1024   # total added bytes across the candidate - a human-skimmable change
_MAX_LINE_BYTES = 400           # no single line may hide a payload
_MAX_BLOB_BYTES = 1 << 20       # never slurp an untrusted blob larger than 1 MiB into memory
_MAX_RATIONALE_BYTES = 4096     # agent prose reaches the commit message + PR body; bound it

# Characters that must never appear in candidate content or agent prose. Classified BY UNICODE CATEGORY,
# not by an enumerated codepoint list: an enumeration is unbounded (the first version missed the whole TAG
# block U+E0000-E007F, U+2028/U+2029/U+0085 line separators, U+00AD, U+061C, U+2060, U+180E, U+FFF9 - all
# invisible in a diff view), and this is the same lesson the path ALLOWLIST taught.
#   Cc control, Cf format (bidi overrides, zero-width, TAG), Co private-use, Cs surrogate,
#   Zl/Zp line+paragraph separators.
# TAB and LF are the only control characters source code legitimately needs.
_FORBIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Co", "Cs", "Zl", "Zp"})
_ALLOWED_CONTROLS = frozenset({"\t", "\n"})

# Category is NECESSARY but not SUFFICIENT: some invisible characters live in PRINTABLE categories and are
# valid Python identifier characters - measured, U+FE0F VARIATION SELECTOR-16 (Mn) and U+3164 HANGUL FILLER
# (Lo) both passed the category check, and `('a'+chr(0xFE0F)+'b').isidentifier()` is True, so two names that
# render identically are distinct to the interpreter. The clean exploit is a string literal: a reviewer
# reads `if mode == "strict": enforce()` while the literal carries a VS16, so the branch is silently dead.
# These are exactly Unicode's DEFAULT_IGNORABLE_CODE_POINT property - the authoritative set of "renders as
# nothing" - which CPython does not expose, so the ranges are mirrored here (Unicode 15.1) and must be
# refreshed with the Unicode version. This is an enumeration of a SPEC-DEFINED set, not guesswork.
_DEFAULT_IGNORABLE_RANGES: tuple[tuple[int, int], ...] = (
    (0x00AD, 0x00AD), (0x034F, 0x034F), (0x061C, 0x061C), (0x115F, 0x1160), (0x17B4, 0x17B5),
    (0x180B, 0x180F), (0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x206F), (0x3164, 0x3164),
    (0xFE00, 0xFE0F), (0xFEFF, 0xFEFF), (0xFFA0, 0xFFA0), (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3), (0x1D173, 0x1D17A), (0xE0000, 0xE0FFF),
)


# Flattened, sorted boundaries for an O(log n) membership test. The linear `any()` over 17 ranges per
# CHARACTER cost ~0.85 s on a 1 MiB blob, and this runs on every candidate file.
_DI_BOUNDS: tuple[int, ...] = tuple(
    b for lo, hi in sorted(_DEFAULT_IGNORABLE_RANGES) for b in (lo, hi + 1))


def _is_default_ignorable(ch: str) -> bool:
    """Is ``ch`` a Unicode DEFAULT_IGNORABLE_CODE_POINT (renders as nothing)?"""
    return bisect.bisect_right(_DI_BOUNDS, ord(ch)) % 2 == 1


def _forbidden_char(text: str) -> str | None:
    """The first invisible / format / separator character in ``text`` (as ``U+XXXX``), else None. These make
    the RENDERED text differ from the EXECUTED bytes, so a reviewer cannot see what they are approving."""
    for ch in text:
        if ch in _ALLOWED_CONTROLS:
            continue
        if unicodedata.category(ch) in _FORBIDDEN_CATEGORIES or _is_default_ignorable(ch):
            return f"U+{ord(ch):04X}"
    return None

# Credential SHAPES scanned in the candidate's REALIZED BLOB CONTENT (never the diff's `+` lines: a
# `copy from secrets/deploy.pem` delivers the bytes with zero `+` lines, and `.gitattributes` eol
# normalization means the staged blob can differ from the diff text anyway). CI gitleaks is the backstop.
_SECRET_PATTERNS: tuple[tuple["re.Pattern[str]", str], ...] = (
    (re.compile(r"AKIA[0-9A-Z]{16}"), "an AWS access key id"),
    # gh[p|o|s|u|r]_ - `gho_` is what the local `gh` CLI actually stores in ~/.config/gh/hosts.yml, and
    # it grants write access to this repo; matching only `ghp_` missed the one that matters most.
    (re.compile(r"gh[posur]_[A-Za-z0-9]{36}"), "a GitHub token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{50,}"), "a GitHub fine-grained PAT"),
    (re.compile(r"AIza[A-Za-z0-9_-]{35}"), "a Google API key"),
    (re.compile(r"\bASIA[0-9A-Z]{16}\b"), "an AWS temporary access key id"),
    (re.compile(r"(?<![A-Za-z0-9_-])sk-(?:proj-)?[A-Za-z0-9]{20,}"), "an OpenAI API key"),
    (re.compile(r"glpat-[A-Za-z0-9_-]{20,}"), "a GitLab token"),
    (re.compile(r"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9_-]{10,}"), "a PyPI token"),
    (re.compile(r"(?m)^\s*machine\s+\S+.*\n?\s*(?:login|password)\s+\S+"), "a .netrc credential"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "a Slack token"),
    (re.compile(r"hf_[A-Za-z0-9]{34}"), "a HuggingFace token"),
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"), "an Anthropic API key"),
    (re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"), "a private key block"),
    (re.compile(r"(?i)aws_secret_access_key\s*[=:]\s*[A-Za-z0-9/+]{40}"), "an AWS secret access key"),
)

# A LOOSE heuristic, scanned ONLY over lines the candidate ADDS - never the whole blob. Over the whole
# blob it made files permanently unpublishable: `api_key: Optional[str] = typer.Option(` in cli.py
# (PRE-EXISTING, unmodified) matched, so rung 7.1 could never edit the largest file in its own writable
# surface, and could never fix the line that blocked it. The value charset excludes brackets/parens/spaces
# so a type annotation (`Optional[str]`) or an env lookup (`os.environ["PW"]`) no longer matches, while a
# real quoted secret or a bare `oauth_token: gho_...` (YAML, as ~/.config/gh/hosts.yml writes it) does.
_ADDED_LINE_SECRET_PATTERNS: tuple[tuple["re.Pattern[str]", str], ...] = (
    (re.compile(r"""(?i)(?:secret|password|passwd|api[_-]?key|access[_-]?token|oauth_token)"""
                r"""\s*[=:]\s*(?:['"][^'"]{12,}['"]|[A-Za-z0-9+/=_-]{20,})"""),
     "an apparent hardcoded credential"),
)


def _scan_text(text: str, what: str, max_bytes: int = _MAX_RATIONALE_BYTES) -> list[str]:
    """Reasons a piece of AGENT-CONTROLLED PROSE must not be published (empty = clear).

    The rationale reaches the pushed commit message AND the ``gh pr create --body`` verbatim, so it leaves
    the machine exactly like file content does - but it is not a blob, so the blob scan never sees it.
    Measured: moving a ``.pem``'s bytes from the diff into the rationale published a private key to the
    remote and the PR while every content check passed. Prose is therefore bounded and scanned with the
    SAME credential patterns."""
    reasons: list[str] = []
    if len(text.encode("utf-8", "replace")) > max_bytes:
        reasons.append(f"{what} exceeds {max_bytes} bytes")
    bad = _forbidden_char(text)
    if bad:
        reasons.append(f"{what} contains an invisible/format character ({bad})")
    for pat, label in (*_SECRET_PATTERNS, *_ADDED_LINE_SECRET_PATTERNS):
        if pat.search(text):
            reasons.append(f"{what} contains {label}")
            break
    return reasons


def _publish_precondition_unmet(ci_attested_safe: bool) -> str | None:
    """A REFUSAL REASON unless the OPERATOR has explicitly attested that CI can safely execute an
    agent-authored candidate, else None. Default (no attestation) = REFUSE.

    Why an attestation and NOT a check: pushing hands the candidate to CI, which EXECUTES it before any
    human reviews it (on this repo ``engine-tests.yml`` triggers on a bare ``push:`` and runs
    ``pytest --cov=corpus_studio``, and the suite imports the product modules), so "a human merges only on
    green" cannot undo code that already ran. The previous version tried to VERIFY that property by
    text-scanning ``.github/workflows/*.yml``. An adversarial pass measured it failing OPEN five ways -
    a ``persist-credentials: false`` under a different step, one in a COMMENT, a local composite action
    doing the checkout, a quoted ``'actions/checkout@v7'``, a reusable workflow - and, worse, it read the
    operator's dev WORKING TREE while GitHub runs the workflows at ``base``, so the normal
    dev-branch-ahead-of-main state made it pass exactly when it should refuse.

    That whole approach is wrong in kind: this adapter cannot verify a property of an execution environment
    it does not control by parsing text, and every such attempt fails open. Worse, the property is not even
    expressible as "persist-credentials: false" - a workflow satisfying that STILL runs the candidate in a
    job that later hands ``secrets.CODECOV_TOKEN`` to another step.

    So the human asserts it, once, explicitly, and it is recorded. That is honest about who actually knows,
    it fails closed by default, and it puts the decision with the person who can change the CI."""
    if not ci_attested_safe:
        return ("publishing hands the candidate to CI, which EXECUTES it before review; the operator has "
                "not attested that CI is safe for agent-authored code (pass ci_attested_safe=True after "
                "hardening: no credential-persisting checkout, and no secret-bearing step in a job that "
                "runs candidate code)")
    return None


def _compile_pathglob(glob: str) -> "re.Pattern[str]":
    """Compile a repo-relative path glob with EXPLICIT, path-aware semantics:
    ``**/`` = zero or more directory segments, ``*``/``?`` NEVER cross ``/``, everything else literal.

    Deliberately NOT :mod:`fnmatch` - measured, fnmatch is wrong in both directions for an allowlist:
    ``fnmatch('engine/tests/test_x/conftest.py', 'engine/tests/test_*.py')`` is **True** (it would admit an
    auto-executed pytest conftest - full code execution), while
    ``fnmatch('engine/corpus_studio/cli.py', 'engine/corpus_studio/**/*.py')`` is **False**.
    (``PurePosixPath.full_match`` would do, but it is 3.13+ and this repo targets 3.11.)"""
    out, i = [], 0
    while i < len(glob):
        if glob.startswith("**/", i):
            out.append(r"(?:[^/]+/)*")
            i += 3
        elif glob[i] == "*":
            out.append(r"[^/]*")
            i += 1
        elif glob[i] == "?":
            out.append(r"[^/]")
            i += 1
        else:
            out.append(re.escape(glob[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def _path_is_writable(path: str, profile: TargetProfile) -> bool:
    """Is ``path`` inside the allowed writable surface? Matched CASE-FOLDED, so `Engine/`, `SCRIPTS/` and
    every other case variant can only ever SHRINK the surface, never widen it (the repo carries
    ``core.ignorecase=true``, so on a case-insensitive checkout a variant IS the real path)."""
    folded = path.replace("\\", "/").casefold()
    return any(_compile_pathglob(g.casefold()).match(folded) for g in profile.writable_globs)


@dataclass(frozen=True)
class _ChangeRecord:
    """One realized change, as GIT reports it - not as the diff text claims."""

    status: str      # A C D M R T (U/X are refused)
    src_mode: str    # "000000" for an addition
    dst_mode: str    # "000000" for a deletion
    dst_oid: str     # the realized blob OID ("000...0" for a deletion)
    path: str        # raw bytes decoded; for R/C this is the DESTINATION
    src_path: str    # "" unless the record carries a distinct source


def _realized_changes(wt: Path, base_oid: str, tree_oid: str) -> list[_ChangeRecord]:
    """The change set GIT actually realized in the candidate worktree, via PLUMBING:

        git diff-tree -r --no-renames -z <base_oid> <tree_oid>

    Every element of that command is load-bearing (all measured on git 2.43.0):
      * ``diff-tree`` on the CAPTURED TREE OID, not the index: measured byte-identical to
        ``diff-index --cached``, but it pins the subject. The index is mutable - a concurrent ``git add``
        between classification and commit was measured to change the committed tree - so classifying the
        tree makes "the object classified IS the object committed" true by construction.
      * plumbing IGNORES ``diff.renames`` config; porcelain ``git diff`` honours it, so a repo/global
        ``diff.renames=copies`` would silently change what we see.
      * ``--no-renames`` is passed ALONE. ``--no-renames -M0 -C0`` still prints ``R100 <src> <dst>`` because
        ``-M0`` means "detect a rename at 0% similarity" - it RE-ENABLES detection. With detection off, a
        rename decomposes into ``D <source>`` + ``A <dest>``, which is the ONLY way the deleted source of a
        ``rename from`` becomes visible (the hole that published a diff deleting `.github/dependabot.yml`).
      * ``-r`` recurses into subtrees; ``-z`` gives RAW path bytes so ``core.quotepath`` never applies.
      * the raw format carries MODES and full blob OIDs in one pass, so symlinks (120000), submodules
        (160000) and exec-bit flips (100755) are refused structurally instead of by sniffing diff text.

    Never parses the unified diff: extended headers (``rename from`` / ``copy from`` / mode / binary) are
    invisible to a ``---``/``+++`` scanner, which is exactly how three separate holes got through."""
    proc = _git(wt, "diff-tree", "-r", "--no-renames", "-z", base_oid, tree_oid)
    fields = proc.stdout.split("\0")
    records: list[_ChangeRecord] = []
    i = 0
    while i < len(fields):
        meta = fields[i]
        if not meta:
            i += 1
            continue
        if not meta.startswith(":"):
            raise WriteAdapterError(f"unparseable git diff-index record {meta!r} (refusing to guess)")
        parts = meta[1:].split()
        if len(parts) < 5:
            raise WriteAdapterError(f"malformed git diff-index record {meta!r}")
        src_mode, dst_mode, _src_oid, dst_oid, status = parts[0], parts[1], parts[2], parts[3], parts[4]
        letter = status[0]
        # R/C carry TWO path fields; with --no-renames they should never appear, but parse defensively
        # rather than mis-align the whole stream if a future git emits them.
        if letter in ("R", "C"):
            src_path, path, consumed = fields[i + 1], fields[i + 2], 3
        else:
            src_path, path, consumed = "", fields[i + 1], 2
        records.append(_ChangeRecord(letter, src_mode, dst_mode, dst_oid, path, src_path))
        i += consumed
    return records


def _read_blob(wt: Path, oid: str, path: str, max_bytes: int = _MAX_BLOB_BYTES) -> str | None:
    """The realized blob's text, or None when it is binary / unreadable as text. Bounded: an untrusted blob
    larger than :data:`_MAX_BLOB_BYTES` is refused rather than slurped. Reads the OBJECT (not the worktree
    file), because `.gitattributes` eol normalization means the committed bytes can differ from both the
    worktree and the diff text."""
    size_out = _git(wt, "cat-file", "-s", oid).stdout.strip()
    try:
        size = int(size_out)
    except ValueError as exc:
        raise WriteAdapterError(f"cannot size blob for {path}: {size_out!r}") from exc
    if size > max_bytes:
        raise WriteAdapterError(f"{path} is {size} bytes (> {max_bytes}); refusing an oversized blob")
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell; bytes because the blob is untrusted.
        ["git", "-C", str(wt), "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=",
         "cat-file", "blob", oid], capture_output=True, timeout=_GIT_TIMEOUT_S)
    if proc.returncode != 0:
        # FAIL CLOSED. Ignoring this exit status meant an unreadable blob produced EMPTY content, and an
        # empty string passes every content check - the secret scan, the line bound and the character
        # check all silently succeed on a blob we never actually read.
        raise WriteAdapterError(
            f"cannot read blob for {path} (git cat-file exit {proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'replace').strip()[:160]}")
    raw = proc.stdout
    if b"\0" in raw:
        return None  # binary (NUL test - immune to a candidate-supplied .gitattributes)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _classify_candidate_changes(wt: Path, base_oid: str, tree_oid: str,
                                profile: TargetProfile) -> list[str]:
    """Reasons the realized candidate must NOT be published (empty = clear). Runs against what GIT actually
    staged, so rename sources, copy destinations, mode flips and binary payloads are all visible.

    Refuses, fail-closed: a status other than ``M``; a non-regular file mode (symlink / submodule / exec
    bit); a path outside the case-folded writable ALLOWLIST; an unsafe basename; too many changed paths or
    lines; and any apparent secret in the REALIZED BLOB CONTENT (which is the only layer that catches a
    ``copy from`` - a copy leaves its source untouched, so git reports only the allowed destination)."""
    reasons: list[str] = []
    records = _realized_changes(wt, base_oid, tree_oid)
    if not records:
        return ["the candidate changed nothing"]
    if len(records) > profile.max_changed_paths:
        reasons.append(f"touches {len(records)} paths (> {profile.max_changed_paths}; "
                       "needs human review)")
    for rec in records:
        shown = rec.path or rec.src_path
        # MODIFY-ONLY. A create/delete/rename/copy/typechange is refused outright: creates are how
        # conftest injection and stdlib shadowing land, and a delete/rename is how a protected file
        # disappears (the `rename from` hole).
        if rec.status != "M":
            reasons.append(f"{shown} (status {rec.status}: only in-place modification is allowed)")
            continue
        if rec.src_mode != _MODE_REGULAR or rec.dst_mode != _MODE_REGULAR:
            reasons.append(f"{shown} (mode {rec.src_mode}->{rec.dst_mode}: not an ordinary file)")
            continue
        if not _path_is_writable(rec.path, profile):
            reasons.append(f"{rec.path} (outside the writable surface)")
            continue
        if not profile.basename_ok(rec.path.rsplit("/", 1)[-1]):
            reasons.append(f"{rec.path} (unsafe module basename)")
            continue
        content = _read_blob(wt, rec.dst_oid, rec.path, profile.max_blob_bytes)
        if content is None:
            reasons.append(f"{rec.path} (binary or non-UTF-8 content)")
            continue
        # A single enormous line is how a ~1 MiB payload hides behind "1 insertion(+)", and bidi /
        # control characters make the RENDERED diff differ from the EXECUTED source.
        # split on "\n" ONLY - str.splitlines() also breaks on U+2028/U+2029/U+0085, which git, the diff
        # renderer and the Python tokenizer do NOT, so it under-measured a physical line by ~19x (measured).
        longest = max((len(line.encode("utf-8", "replace")) for line in content.split("\n")), default=0)
        if longest > profile.max_line_bytes:
            reasons.append(f"{rec.path} has a {longest}-byte line (> {profile.max_line_bytes})")
            continue
        bad = _forbidden_char(content)
        if bad:
            reasons.append(f"{rec.path} contains an invisible/format character ({bad})")
            continue
        # HIGH-CONFIDENCE token formats only, over the WHOLE blob - these never legitimately appear in
        # source, and whole-blob is what catches a `copy from` (which delivers content with no `+` lines).
        for pat, label in _SECRET_PATTERNS:
            if pat.search(content):
                reasons.append(f"{rec.path} contains {label}")
                break
    # The LOOSE heuristic runs ONLY over lines the candidate ADDS, computed by comparing the REALIZED
    # BLOBS - never by parsing diff text. Two prefix-sniffing versions of this were defeated: a content
    # line starting "++" renders as "+++..." (read as a file header), and anchoring on a preceding "--- "
    # failed too because a REMOVED content line starting "-- " renders as "--- " (both measured, both
    # published a real credential). Comparing objects has no such surface.
    for rec in records:
        # ADDED files are scanned too, not just modified ones. A create is refused outright by the
        # modify-only rule above, so this is defense in depth today - but the heuristic must not silently
        # stop covering new content if that rule is ever relaxed (Sourcery, #707).
        if rec.status not in ("M", "A") or rec.dst_mode != _MODE_REGULAR:
            continue
        new_text = _read_blob(wt, rec.dst_oid, rec.path)
        if new_text is None:
            continue  # binary/non-UTF-8 was already refused above
        old_lines: set[str] = set()
        if rec.status == "M":   # an ADDED file has no base version - every line is new
            old_proc = subprocess.run(  # noqa: S603 - fixed argv, no shell; unreadable -> treat as empty
                ["git", "-C", str(wt), "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=",
                 "cat-file", "blob", f"{base_oid}:{rec.path}"],
                capture_output=True, timeout=_GIT_TIMEOUT_S)
            if old_proc.returncode == 0:
                try:
                    old_lines = set(old_proc.stdout.decode("utf-8").split("\n"))
                except UnicodeDecodeError:
                    old_lines = set()
        for line in new_text.split("\n"):
            if line in old_lines:
                continue                      # unchanged content can never make a file unpublishable
            for pat, label in _ADDED_LINE_SECRET_PATTERNS:
                if pat.search(line):
                    reasons.append(f"the candidate adds {label}")
                    break

    numstat = _git(wt, "diff-tree", "-r", "--no-renames", "--numstat", base_oid, tree_oid).stdout
    touched = 0
    for line in numstat.splitlines():
        added, removed = line.split("\t", 2)[:2]
        if added == "-" or removed == "-":
            reasons.append("the candidate contains a binary change")
            continue
        touched += int(added) + int(removed)
    if touched > profile.max_changed_lines:
        reasons.append(f"changes {touched} lines (> {profile.max_changed_lines}; "
                       "needs human review)")
    # BYTES as well as lines: `git diff --cached` is the patch a reviewer actually reads, and a line
    # count alone does not bound it (measured: ~990 KB on one physical line reports "1 insertion(+)").
    patch_bytes = len(_git(wt, "diff-tree", "-p", "--no-renames", "--full-index", "--binary",
                           base_oid, tree_oid).stdout.encode("utf-8", "replace"))
    if patch_bytes > profile.max_changed_bytes:
        reasons.append(f"the patch is {patch_bytes} bytes (> {profile.max_changed_bytes}; "
                       "needs human review)")
    return reasons

# gh subcommands the WRITE adapter permits: the read-only allowlist PLUS ``pr create``. Everything else -
# and crucially ``pr merge`` / ``pr close`` / ``pr edit`` / ``pr review`` / any ``api`` write - is REFUSED,
# so 7.1 can open a PR but can NEVER merge or otherwise mutate a PR autonomously.
_GH_WRITE_ALLOWED = frozenset({
    ("pr", "create"),
    ("pr", "view"), ("pr", "checks"), ("pr", "status"), ("pr", "list"), ("pr", "diff"),
    ("repo", "view"), ("run", "list"), ("run", "view"),
})


class WriteAdapterError(LoopOrchestrateError):
    """A git/gh write effect failed, or the adapter REFUSED the candidate (fail-closed).

    Subclasses :class:`LoopOrchestrateError` so the controller counts it among its EXPECTED operational
    refusals. An allowlist refusal is this adapter's PRIMARY SAFETY MECHANISM and normal operation - it
    must escalate labelled as an operational refusal, not as "unrecoverable UNEXPECTED ... (a likely
    controller bug)" with a traceback accumulated in ``review_state`` on every denied path."""


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


def _git(cwd: Path, *args: str, stdin: str | None = None,
         env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``git -C <cwd> <args>`` (fixed argv, no shell, bounded) with HOOKS PINNED OFF. Raises
    :class:`WriteAdapterError` on a non-zero exit or an un-runnable git.

    ``-c core.hooksPath=/dev/null -c core.fsmonitor=`` on EVERY invocation: a linked worktree shares the
    main repo's hooks, and it was measured that ``git worktree add`` runs ``post-checkout`` and
    ``git commit`` runs ``pre-commit``. ``core.fsmonitor`` is a SEPARATE config-driven hook that
    ``hooksPath`` does NOT suppress (measured), so it is disabled too. Both would run with the operator's
    full environment, and whether either is configured is operator state the adapter must not depend on."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, hooks + fsmonitor disabled.
            ["git", "-C", str(cwd), "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=", *args],
            input=stdin, capture_output=True, text=True, timeout=_GIT_TIMEOUT_S,
            env={**os.environ, **env_extra} if env_extra else None)
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


def _worker_reachable_paths(wt: Path, base_oid: str, run_cs_assure: Any) -> frozenset[str]:
    """Every path in the WORKER'S REAL IMPORT CLOSURE for the candidate (declared + undeclared-reachable),
    via ``cs_assure worker-reachability``. FAIL-CLOSED: a refusal / unusable record raises, so an
    uncomputable closure never reads as "touches no worker code"."""
    code, out, err = run_cs_assure(wt, "worker-reachability", "--scope", "head", "--base", base_oid)
    if code != 0:
        raise WriteAdapterError(f"candidate worker-reachability refused (exit {code}): {err.strip()[:200]}")
    try:
        payload = json.loads(out).get("payload")
    except (ValueError, RecursionError, AttributeError) as exc:
        raise WriteAdapterError(f"candidate worker-reachability produced no usable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise WriteAdapterError("candidate worker-reachability record has no payload object")
    # FAIL CLOSED on shape: a payload missing / renaming / nulling these keys previously yielded an EMPTY
    # closure, so the worker gate silently passed (measured: {} -> SUCCESS -> published). A record we
    # cannot read is an UNCOMPUTABLE closure, never "no worker code touched".
    paths: set[str] = set()
    required = ("undeclared_reachable", "worker_roots", "added_reachable", "distribution_impacting_paths")
    for key in required:
        entries = payload.get(key)
        if not isinstance(entries, list):
            raise WriteAdapterError(
                f"worker-reachability record has no usable {key!r} list (got {type(entries).__name__}); "
                "refusing to treat an uncomputable worker closure as empty")
        paths.update(e for e in entries if isinstance(e, str))
    return frozenset(paths)


def _assure_candidate(wt: Path, base_oid: str, run_cs_assure: Any, changed_paths: list[str],
                      candidate_commit: str,
                      profile: TargetProfile) -> tuple[Observation, str, str | None]:
    """STATIC candidate assurance: classify the candidate via ``cs_assure impact`` (obligation / policy
    analysis over the diff) and return ``(observation, reason, impact_record_digest)``. ``SUCCESS`` = clear
    to publish; a human-gated / worker-closure / other blocking obligation returns the routing observation
    so the loop escalates.

    NO UNTRUSTED CODE RUNS. Two independent layers:
      * ``run_cs_assure`` must be the :func:`_trusted_cs_assure` wrapper, so the TRUSTED dev-tree
        ``cs_assure.py`` analyzes the candidate via ``--start-dir`` - never the candidate's own copy (which
        would put ``<candidate>/scripts`` on ``sys.path[0]`` and execute a shadow stdlib module);
      * the whole ``scripts/`` tree is denied pre-apply, so no such file can reach the candidate anyway.
    ``impact`` itself only reads git state + the policy bundle; it does not import the analyzed tree.

    Why static, not the full gate: running ``cs_assure verify`` here would run ruff/mypy/PYTEST inside the
    candidate worktree, and pytest COLLECTION EXECUTES the untrusted agent's committed code with the
    operator's environment - arbitrary code execution + secret exposure, exactly what the 7.1.1 confinement
    prevents for the propose step. The DYNAMIC gate (which must execute the candidate) is therefore
    delegated to CI's isolated sandbox on the opened PR, where the human merges only on green. Fail closed:
    an unusable / refused impact record raises :class:`WriteAdapterError` (-> the loop escalates)."""
    # `--scope head` binds the assessment to the COMMIT. The default (workspace) assesses the MUTABLE
    # WORKING TREE, which is a different object: measured on a candidate whose commit edits
    # platform/worker.py while the worktree is reverted, workspace reported `fired=[]` and head reported
    # `['worker-closure']` - the gate said clean about a commit that fires a human-gated obligation.
    # `--base <base_oid>` pins the same base the classifier used; passing the branch NAME resolved the
    # LOCAL ref while the candidate was branched from origin/<base>, so the assessed change set was not
    # the classified one (measured: it fired obligations belonging to someone else's merged commit).
    argv = ["impact", "--scope", "head", "--base", base_oid]
    if profile.obligations_policy:
        # Resolved against the PROFILE's own directory (operator-owned) and passed ABSOLUTE, so the
        # candidate is judged by a policy it cannot supply or edit. A target repo need not carry any
        # assurance tooling of its own - which is the normal case for anything but CorpusStudio.
        argv += ["--policy", str(profile.resolved_policy())]
    code, out, err = run_cs_assure(wt, *argv)
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
    # BIND the record to this candidate. Without this, "an impact record exists" says nothing about
    # WHICH object it assessed - a naive identity check would certify a record that reported fired=[]
    # for a different subject entirely.
    payload_scope = payload.get("scope")
    provenance = data.get("provenance") if isinstance(data, dict) else None
    head_oid = provenance.get("head_oid") if isinstance(provenance, dict) else None
    if payload_scope != "head":
        raise WriteAdapterError(f"candidate impact scope is {payload_scope!r}, not 'head' (wrong subject)")
    if head_oid != candidate_commit:
        raise WriteAdapterError(
            f"candidate impact assessed {str(head_oid)[:12]!r}, not the candidate commit "
            f"{candidate_commit[:12]!r} (the assessment is about a different object)")
    ids = {o["id"] for o in fired if isinstance(o, dict) and isinstance(o.get("id"), str)}
    digest = data.get("record_digest") if isinstance(data, dict) else None
    digest = digest if isinstance(digest, str) else None
    human = sorted((ids & HUMAN_GATED_OBLIGATIONS) - {"worker-closure"})
    if human:
        return Observation.AUTHORIZATION_REQUIRED, f"candidate fires {', '.join(human)} (human review)", digest
    if "worker-closure" in ids:
        return Observation.WORKER_LINEAGE_IMPACT, "candidate touches worker-execution bytes", digest
    # The worker-closure OBLIGATION only flags the 7 DECLARED worker files - the policy says so itself.
    # But the worker's REAL import closure is far wider (measured: 43 undeclared-reachable modules on this
    # repo, e.g. platform/worker_protocol.py and platform/backends.py, imported at module level by
    # platform/worker.py) and those sit INSIDE the writable surface. Editing one changes worker execution
    # bytes with the lineage gate silent, so consult the reachability analysis too and route it the same way.
    # OPT-IN: the worker import closure is a CorpusStudio concept. Requiring it for a repo with no ML
    # worker would fail closed forever, so a profile that does not declare it skips the check entirely.
    hit: list[str] = []
    if profile.require_worker_reachability:
        reach = _worker_reachable_paths(wt, base_oid, run_cs_assure)
        hit = sorted(p for p in changed_paths if p in reach)
    if hit:
        return (Observation.WORKER_LINEAGE_IMPACT,
                f"candidate touches the worker import closure: {', '.join(hit[:4])}", digest)
    blocking = sorted(o["id"] for o in fired if isinstance(o, dict)
                      and o.get("severity") == "blocking" and isinstance(o.get("id"), str))
    if blocking:
        return Observation.POLICY_BLOCK, f"candidate fires blocking obligation(s): {', '.join(blocking)}", digest
    return Observation.SUCCESS, "candidate static assurance clean (dynamic gate delegated to CI)", digest


def _record_candidate_assurance(state: LoopState, record: dict[str, Any], branch: str, observation: Observation,
                                reason: str, impact_digest: str | None, *, published: bool,
                                identity: dict[str, str] | None = None) -> None:
    """Store the CANDIDATE's assurance outcome on the loop state (audit trail): the observation the static
    impact classified for the candidate worktree, why, whether it was published, and the sealed impact
    digest. Kept OUT of the completeness ``evidence`` index (a clean candidate is not a human sign-off - the
    merge stays human-gated via the critic)."""
    # DELIBERATELY NOT appended to state.assurance_records: that index feeds the semantic completeness
    # check, and this digest is the CANDIDATE'S OWN static self-analysis. Seeding it would let an untrusted
    # executor supply the evidence a DETERMINISTIC criterion is later proven by - exactly the cross-step
    # self-certification vector completeness.py says the #7 write runtime must not open. It is recorded
    # below as an audit field only.
    state.review_state["candidate_assurance"] = {
        "record_digest": record["record_digest"], "branch": branch,
        "observation": observation.value, "reason": reason,
        "published": published, "impact_record_digest": impact_digest,
        # The durable identity of what was assured. Without it the impact digest does not say WHICH object
        # it assessed once the disposable worktree is gone - this is what a human triaging an escalation
        # (and 7.1.4's resume) needs to line the remote branch up against the assurance.
        **(identity or {}),
    }


def _target_has_ci(repo_root: Path) -> bool:
    """Does this repository have CI that will run on the pushed branch?"""
    wf = repo_root / ".github" / "workflows"
    return wf.is_dir() and any(wf.glob("*.yml")) or (wf.is_dir() and any(wf.glob("*.yaml")))


def _pr_body(rationale: str, has_ci: bool) -> str:
    """The PR body, led by a disclosure that is TRUE FOR THIS TARGET.

    The first version hardcoded "the dynamic gate runs in CI on this PR" - and the very first real run
    published to a repository with NO CI AT ALL, so the disclosure asserted a check that would never
    happen. A disclosure a reviewer relies on must not overclaim; where nothing will run, it says so, and
    the reviewer knows the diff is all the assurance there is."""
    dynamic = ("the dynamic gate (lint/type/tests) runs in CI on this PR"
               if has_ci else
               "**this repository has no CI, so NOTHING will run these changes** - static assurance and "
               "your reading of the diff are the only checks")
    return ("> **Machine-authored, NOT yet human-reviewed.** Opened autonomously by the CorpusStudio "
            "single-agent write runtime (7.1). Only STATIC candidate assurance has run; "
            f"{dynamic}, and a human reviews and merges. Read the diff on that basis.\n\n"
            + (rationale or "_No rationale supplied by the agent._"))


def _verify_commit_identity(wt: Path, commit: str, tree_oid: str, base_oid: str) -> None:
    """Assert the built commit IS the classified candidate: its tree is the tree that passed
    classification, and its parent is exactly the base that classification was computed against.

    A tree-only check is NOT enough - ``git commit-tree`` was measured producing a commit with an
    identical tree but a hostile parent (and a message carrying a token), which a tree comparison alone
    accepts. Fail-closed: any mismatch raises rather than publishing an object we did not assure."""
    built_tree = _git(wt, "rev-parse", f"{commit}^{{tree}}").stdout.strip()
    if built_tree != tree_oid:
        raise WriteAdapterError(f"commit tree {built_tree[:12]} != classified tree {tree_oid[:12]}")
    parents = _git(wt, "rev-list", "--parents", "-n", "1", commit).stdout.split()[1:]
    if parents != [base_oid]:
        raise WriteAdapterError(f"commit parents {parents} != [{base_oid}] (base drift)")


# ---------------------------------------------------------------- crash-resumable publish (phase 7.1.4)
# The publish is several EXTERNAL effects in sequence (push, then `gh pr create`). `step()` persists loop
# state after a dispatch and in its except branch, but a SIGKILL or Ctrl-C between them bypasses both, so
# the remote can hold an agent-authored branch that loop state has never heard of. The journal is written
# BEFORE each effect, so a resumed or re-run goal can tell "already done" from "never started".
_JOURNAL_STATES = ("PUSH_INTENDED", "PUSHED", "PR_OPENED")


def _journal_path(proposals_dir: Path, goal_id: str, candidate_commit: str) -> Path:
    """One journal file per (goal, candidate). Keyed by the CONTENT-ADDRESSED commit oid, so re-running a
    goal with identical content lands on the same journal - which is what makes resume idempotent."""
    slug = _sanitize_branch_suffix(goal_id).replace("/", "-")
    return proposals_dir / "journal" / f"{slug}-{candidate_commit[:12]}.json"


def _journal_write(path: Path, state_name: str, **fields: Any) -> None:
    """Record intent BEFORE the effect it describes. Best-effort by design: a journal that cannot be
    written must not block the publish (it is an aid to recovery, not a gate), but a journal that CAN be
    written is always written first."""
    if state_name not in _JOURNAL_STATES:
        raise WriteAdapterError(f"unknown journal state {state_name!r}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(existing, dict):
            existing = {}
        existing.update({"state": state_name, **fields})
        path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError):
        pass


def _journal_read(path: Path) -> dict[str, Any]:
    """The recorded publish state, or {} when absent/unreadable (treated as 'never started')."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _existing_pr_url(gh_runner: Callable[..., "tuple[int, str, str]"], branch: str) -> str:
    """The URL of an OPEN PR already opened for ``branch``, or "". Used to make re-publishing idempotent
    instead of opening a duplicate PR for the same candidate. A gh failure returns "" - the caller then
    attempts creation, and a genuine duplicate is refused by gh itself."""
    code, out, _err = gh_runner("pr", "list", "--head", branch, "--state", "open",
                                "--json", "url", "--limit", "1")
    if code != 0:
        return ""
    try:
        rows = json.loads(out)
    except (ValueError, RecursionError):
        return ""
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        url = rows[0].get("url")
        return url if isinstance(url, str) else ""
    return ""


_AGENT_COMMIT_EMAIL = "agent@corpusstudio.local"
_GC_MIN_AGE_S = 1800  # 30 minutes


def _github_repo_from_url(url: str) -> str:
    """``owner/repo`` for a GitHub push URL, or "" for anything else.

    ``gh`` resolves its target repository from the FETCH remotes and its own heuristics, NOT from the URL
    we push to. Measured: with ``origin=myfork/proj`` + ``upstream=canonical/proj``, ``gh pr list`` queries
    *canonical*; with a ``remote.origin.pushurl`` set, gh queries the FETCH url. So in an ordinary fork
    workflow gh answers about a different repository than the one we are about to delete from - every
    branch reads "no open PR" and gets deleted, including branches with live open PRs. Pinning gh to the
    push URL's repo is what makes the answer be about the ref we are deleting."""
    m = re.match(r"^(?:git@github\.com:|(?:ssh|git|https?)://(?:[^@/]+@)?github\.com/)"
                 r"([^/]+)/(.+?)(?:\.git)?/?$", url.strip())
    return f"{m.group(1)}/{m.group(2)}" if m else ""


def _open_pr_for(gh_runner: Callable[..., "tuple[int, str, str]"], branch: str,
                 repo: str) -> "tuple[str, bool]":
    """``(url, answered)``. ``answered`` is False when gh COULD NOT ANSWER.

    :func:`_existing_pr_url` returns "" on any non-zero exit, which was right for its original caller
    (publish: a duplicate would then be refused by gh itself) and INVERTS when reused as the guard on a
    DELETION - `gh` missing, unauthenticated, rate-limited, offline or under-scoped all read as "no open
    PR exists, delete it". Here the two cases must stay distinguishable so the caller can fail closed.

    ``--state all`` deliberately, not ``open``: a human who CLOSES a candidate PR intending to revisit it
    would otherwise have the branch collected out from under them on the next sweep."""
    code, out, _err = gh_runner("pr", "list", "--head", branch, "--state", "all",
                                "--json", "url", "--limit", "1", "-R", repo)
    if code != 0:
        return "", False
    try:
        rows = json.loads(out)
    except (ValueError, RecursionError):
        return "", False
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        url = rows[0].get("url")
        return (url if isinstance(url, str) else ""), True
    return "", True


def _is_our_candidate(repo_root: Path, rec: dict[str, Any], oid: str) -> str:
    """"" if ``oid`` really is the agent candidate this record describes, else why not.

    A journal record is a FILE - restored from a backup, hand-edited, or written by anything that can
    reach ``<git-dir>/corpusstudio-loop/``. Trusting its ``branch`` field as a delete target was enough to
    destroy an unrelated branch in a measured reproduction. So the record must AGREE with the object
    graph: the oid it claims is the one it recorded as the candidate, and the commit really is a
    single-parent child of the recorded base authored by the pinned agent identity. Anything we cannot
    read, we refuse - an unverifiable branch is kept, never collected."""
    if oid != str(rec.get("candidate_commit", "")):
        return "remote oid is not the recorded candidate commit"
    try:
        meta = _git(repo_root, "cat-file", "-p", f"{oid}^{{commit}}").stdout
    except (WriteAdapterError, AgentError) as exc:
        return f"cannot read the candidate commit to confirm it is ours: {exc}"
    parents = [ln.split()[1] for ln in meta.splitlines() if ln.startswith("parent ")]
    if parents != [str(rec.get("base_oid", ""))]:
        return "candidate is not a single-parent child of the recorded base"
    if not any(ln.startswith("author ") and f"<{_AGENT_COMMIT_EMAIL}>" in ln for ln in meta.splitlines()):
        return f"candidate is not authored by {_AGENT_COMMIT_EMAIL}"
    return ""


def collect_orphan_branches(repo_root: Path | str, *, apply: bool = False,
                            gh_runner: Callable[..., "tuple[int, str, str]"] | None = None,
                            proposals_dir: Path | None = None, branch_prefix: str = "cs-agent/",
                            min_age_s: float = _GC_MIN_AGE_S,
                            repo: str | None = None) -> list[dict[str, str]]:
    """Find - and only with ``apply=True``, delete - candidate branches pushed but never PR'd.

    A publish that dies between ``PUSHED`` and ``PR_OPENED`` leaves a real branch on the remote that no PR
    references and nobody is watching. Reproduced end-to-end: a CLI run against a target whose remote is
    not a GitHub host pushes the candidate, then fails at ``gh pr create``.

    Deleting a remote ref is DESTRUCTIVE and OUTWARD-FACING, and an independent review destroyed an
    unrelated branch through the first version of this function. So EVERY one of these must hold, and each
    is a refusal, never a warning:

    1. the branch is under ``branch_prefix`` - the loop only ever creates branches there, so nothing else
       is even a candidate for collection no matter what a journal file says;
    2. the object really is our candidate: recorded oid, single-parent child of the recorded base,
       authored by the pinned agent identity (:func:`_is_our_candidate`);
    3. the record is at least ``min_age_s`` old. ``{state: PUSHED, no pr_url}`` is not only the crash
       state - it is the NORMAL in-flight state of a publish between the push and ``gh pr create``. Without
       an age floor a concurrent sweep deletes the branch out from under a running publish;
    4. gh ANSWERED and reported no PR - pinned with ``-R`` to the push URL's own repository. gh not being
       able to answer is not evidence of absence (:func:`_open_pr_for`);
    5. the remote still points at exactly the oid we pushed.

    The delete is then ``--force-with-lease``d against that oid, so a branch that moves DURING the sweep is
    refused rather than destroyed. ``apply=False`` is a pure DRY RUN - it writes nothing at all, including
    no journal healing."""
    root = Path(repo_root).resolve()
    jdir = (proposals_dir or _default_proposals_dir(root)) / "journal"
    gh = gh_runner if gh_runner is not None else write_gh(root)
    try:
        url = _git(root, "remote", "get-url", "--push", "origin").stdout.strip()
    except (WriteAdapterError, AgentError) as exc:
        raise WriteAdapterError(f"cannot resolve the push URL for origin; refusing to collect: {exc}") from exc
    # `repo` is an injection seam for tests, exactly like `gh_runner`: against a local bare remote there
    # is no GitHub repo to derive. The CLI NEVER passes it, so operators always get URL-derived pinning.
    repo = repo if repo is not None else _github_repo_from_url(url)
    out: list[dict[str, str]] = []
    for jpath in sorted(jdir.glob("*.json")) if jdir.is_dir() else []:
        rec = _journal_read(jpath)
        branch, pushed = str(rec.get("branch", "")), str(rec.get("remote_oid", ""))
        if rec.get("state") != "PUSHED" or not branch or not pushed or rec.get("pr_url"):
            continue
        row = {"branch": branch, "oid": pushed, "journal": str(jpath), "action": "kept"}

        def keep(why: str) -> None:
            row.update(reason=why)
            out.append(row)

        if not branch.startswith(branch_prefix) or ".." in branch or branch.startswith("-"):
            keep(f"branch is not under {branch_prefix!r}; the loop never created it, so it is not ours")
            continue
        try:
            age = time.time() - jpath.stat().st_mtime
        except OSError as exc:
            keep(f"cannot determine the record age: {exc}")
            continue
        if age < min_age_s:
            keep(f"record is {int(age)}s old (< {int(min_age_s)}s): a publish may still be in flight")
            continue
        if not repo:
            keep(f"push URL {url!r} is not a GitHub repo, so no PR check can be trusted")
            continue
        pr, answered = _open_pr_for(gh, branch, repo)
        if not answered:
            keep("gh could not answer whether a PR exists; absence of an answer is not absence of a PR")
            continue
        if pr:
            if apply:  # a DRY RUN writes nothing, so healing waits for a real sweep
                _journal_write(jpath, "PR_OPENED", pr_url=pr)
            keep(f"a PR exists ({pr}); not an orphan" + (" - journal healed" if apply else ""))
            continue
        try:
            actual = _remote_ref_oid(root, branch)
        except WriteAdapterError as exc:
            keep(f"could not verify the remote ref, refusing to delete blind: {exc}")
            continue
        if not actual:
            row.update(action="gone", reason="already absent from the remote")
            out.append(row)
            continue
        if actual != pushed:
            keep(f"remote moved to {actual[:12]}, not ours; left alone")
            continue
        mismatch = _is_our_candidate(root, rec, actual)
        if mismatch:
            keep(mismatch)
            continue
        if not apply:
            row.update(action="would-delete", reason="orphan: pushed, no PR, unmoved, ours (dry run)")
            out.append(row)
            continue
        try:
            _git(root, "push", f"--force-with-lease=refs/heads/{branch}:{pushed}", url,
                 f":refs/heads/{branch}")
        except (WriteAdapterError, AgentError) as exc:
            # A refused delete is a SUCCESS of the lease, not a failure of the collector: the ref moved
            # under us. Report it and keep going - one stuck branch must not abort the whole sweep.
            # Both error types are caught because this module's `_git` raises WriteAdapterError while the
            # shared helper raises AgentError, and catching only one let a lease rejection - the exact case
            # this exists to handle - escape and kill the sweep.
            keep(f"delete refused, branch kept: {exc}")
        else:
            row.update(action="deleted", reason="orphan removed")
            _journal_write(jpath, "PUSHED", collected="deleted")
            out.append(row)
    return out


def _remote_ref_oid(repo_root: Path, branch: str) -> str:
    """The oid the REMOTE actually has for ``refs/heads/<branch>``, or "" if absent. Fail-closed.

    Two measured traps. (1) ``git ls-remote <url> <pattern>`` matches the ref TAIL, so a decoy
    ``refs/heads/decoy/refs/heads/cs-agent/x`` also matches and a naive ``head -1`` reads the DECOY's oid -
    so every row is parsed and exactly one exact full-ref match is required. (2) ``ls-remote origin``
    ignores ``remote.origin.pushurl``, so with a pushurl configured the push lands on one remote and the
    verification reads another (there is no ``ls-remote --push``) - so the PUSH url is resolved explicitly.
    """
    url = _git(repo_root, "remote", "get-url", "--push", "origin").stdout.strip()
    if not url:
        raise WriteAdapterError("cannot resolve the push URL for origin; refusing to verify blind")
    want = f"refs/heads/{branch}"
    matches = [ln.split("\t") for ln in
               _git(repo_root, "ls-remote", url, want).stdout.splitlines() if "\t" in ln]
    exact = [oid for oid, ref in matches if ref == want]
    if len(exact) > 1:
        raise WriteAdapterError(f"remote reports {len(exact)} refs for {want} (ambiguous); refusing")
    return exact[0] if exact else ""


def _make_write_executor(agent_client: AgentClient, repo_root: Path, base: str, proposals_dir: Path,
                         worktrees_dir: Path, gh_runner: Callable[..., "tuple[int, str, str]"],
                         branch_prefix: str, run_cs_assure: Any, ci_attested_safe: bool,
                         expected_head_box: list, profile: TargetProfile):  # noqa: ANN202
    """The write executor. At EXECUTE: PROPOSE (agent, CONFINED to a disposable detached worktree) -> seal
    -> APPLY the exact diff in a SEPARATE, pristine detached worktree -> CLASSIFY WHAT GIT REALIZED against
    the writable ALLOWLIST (status/mode/path/basename/bounds + a secret scan over blob content) -> commit ->
    ASSURE THE CANDIDATE statically (`cs_assure impact`, run by the TRUSTED dev-tree tool) -> publish (push
    the branch + open a PR) ONLY IF the candidate is clear. A blocked candidate publishes NOTHING; a policy
    obligation returns the classified observation so the loop routes it (human-gated -> escalate), and an
    allowlist violation raises (-> the loop escalates). At DECOMPOSE it installs one self-owned task."""

    def execute(state: LoopState, directive: Directive) -> Observation:
        if state.current_phase is Phase.DECOMPOSE and not state.task_graph:
            state.task_graph = [{
                "id": "assure-and-pr", "description": "assure the agent's diff in an isolated worktree + open a PR",
                "owner": "self", "allowed_paths": [], "depends_on": [], "status": "PENDING",
            }]
            return Observation.SUCCESS
        if state.current_phase is not Phase.EXECUTE:
            return Observation.SUCCESS

        # Branch from the REMOTE base, not the local one. `_resolve_base_oid(repo_root, base)` resolves
        # the LOCAL ref, so if the operator's `main` is ahead of `origin/main` (an ordinary state - local
        # commits, a rebase in progress), the pushed branch and the opened PR would carry the operator's
        # unpushed commits as though the agent had authored them, and the PR diff would not be the
        # candidate. Prefer `origin/<base>`; fall back to the local ref only when there is no
        # remote-tracking ref at all (a repo with no remote - nothing can be published there anyway).
        base_oid = _resolve_base_oid(repo_root, f"origin/{base}") or _resolve_base_oid(repo_root, base)
        if not base_oid:
            raise WriteAdapterError(
                f"cannot resolve base {base!r} (tried origin/{base} then {base}) to a commit to branch from")
        # 1) PROPOSE - run the UNTRUSTED agent CONFINED: cwd inside a disposable, detached worktree at base
        #    (never the developer's tree AND never the apply worktree) with a secret-free env. Whatever it
        #    writes into that throwaway checkout is discarded on exit; only the diff it RETURNS is used.
        with _detached_worktree(repo_root, base_oid, worktrees_dir) as propose_wt, \
                _confined_home(worktrees_dir) as propose_home:
            request = {"goal": state.goal, "goal_id": state.goal_id, "base_oid": base_oid,
                       "repo_root": str(repo_root), "_cwd": str(propose_wt),
                       "_home": str(propose_home),
                       # the bounds the runtime will enforce, so the agent is TOLD the rules rather than
                       # discovering them via a refusal it already paid for
                       "_limits": {"writable_globs": list(profile.writable_globs),
                                   "max_changed_paths": profile.max_changed_paths,
                                   "max_changed_lines": profile.max_changed_lines,
                                   "max_changed_bytes": profile.max_changed_bytes},
                       "directive": {"phase": directive.phase, "action": directive.action,
                                     "allowed_paths": list(directive.allowed_paths)}}
            files, rationale = _validate_proposal(agent_client.propose(request))  # RAISES -> fail-closed
        record_files = files

        branch = f"{branch_prefix}{_sanitize_branch_suffix(state.goal_id)}"
        # 2) APPLY + CLASSIFY + ASSURE in a pristine DETACHED worktree the agent never touched, so the
        #    commit is deterministically the sealed diff and nothing else; publish only if it is clear.
        with _apply_worktree(repo_root, base_oid, worktrees_dir) as wt:
            # MATERIALISE the agent's files, then let GIT compute the diff. The model used to be asked
            # for a unified diff and got the hunk arithmetic wrong (measured on the first real write run:
            # `@@ -1,5 +1,17 @@` over a 20-line body -> "corrupt patch at line 24"). Paths were already
            # validated as safe repo-relative names; each is re-resolved against the worktree root here so
            # nothing can escape it even if that validation is ever loosened.
            for rel, content in record_files.items():
                dest = (wt / rel).resolve()
                if not str(dest).startswith(str(wt.resolve()) + "/"):
                    raise WriteAdapterError(f"agent path {rel!r} resolves outside the candidate worktree")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
            _git(wt, "add", "-A")
            # 3) CAPTURE THE IDENTITY. write-tree freezes the staged content as an immutable object; every
            #    later step is expressed against THIS oid, so the thing classified, committed, assured and
            #    pushed is provably one object rather than four that happened to agree.
            tree_oid = _git(wt, "write-tree").stdout.strip()
            # The sealed record carries the diff GIT produced from the realized tree - authoritative,
            # rather than whatever the model claimed its change was.
            diff = _git(wt, "diff-tree", "-p", "--no-renames", "--full-index", base_oid, tree_oid).stdout
            record = _seal_proposal({"goal_id": state.goal_id, "base_oid": base_oid, "unified_diff": diff,
                                     "changed_paths": _changed_paths_of(diff),
                                     "agent_files": sorted(record_files),
                                     "rationale": rationale})
            proposal_path = _write_proposal_record(proposals_dir, record)
            violations = _classify_candidate_changes(wt, base_oid, tree_oid, profile)
            violations += (_scan_text(rationale, "the rationale", profile.max_rationale_bytes)
                           + _scan_text(state.goal or "", "the goal", profile.max_rationale_bytes))
            if violations:
                raise WriteAdapterError("refusing the candidate: " + "; ".join(violations[:8]))

            # 4) COMMIT FROM THE TREE, not from the index. `git commit` re-reads the index, so a concurrent
            #    `git add` between classification and commit was measured to change the committed tree;
            #    `commit-tree` takes the classified oid directly and touches no index. Dates are pinned so
            #    identical content yields an identical commit oid (content-addressed, resume-safe).
            message = (f"{state.goal or 'agent change'}\n\n{rationale}\n\n"
                       "[autonomous single-agent proposal - NOT yet human-reviewed; static candidate "
                       "assurance only, the dynamic gate runs in CI and a human reviews + merges the PR]")
            candidate_commit = _git(
                wt, "-c", "user.name=corpusstudio-agent", "-c", "user.email=agent@corpusstudio.local",
                "-c", "commit.gpgsign=false",
                "commit-tree", tree_oid, "-p", base_oid, "-m", message,
                env_extra={"GIT_AUTHOR_DATE": _FIXED_DATE, "GIT_COMMITTER_DATE": _FIXED_DATE},
            ).stdout.strip()

            # 5) RE-VERIFY the object we just built (see _verify_commit_identity).
            _verify_commit_identity(wt, candidate_commit, tree_oid, base_oid)

            # `commit-tree` builds the object but does NOT move HEAD, so point the worktree's HEAD at it -
            # otherwise `--scope head` would assess the BASE commit and report a clean, entirely truthful
            # assessment of the wrong object. update-ref (not `reset`) because it touches no index/worktree.
            _git(wt, "update-ref", "HEAD", candidate_commit)

            # 6) ASSURE THE COMMIT (--scope head, --base base_oid) - see _assure_candidate.
            realized = [r.path for r in _realized_changes(wt, base_oid, tree_oid) if r.path]
            observation, reason, impact_digest = _assure_candidate(
                wt, base_oid, run_cs_assure, realized, candidate_commit, profile)
            identity = {"base_oid": base_oid, "candidate_tree_oid": tree_oid,
                        "candidate_commit": candidate_commit}
            if observation is not Observation.SUCCESS:
                _record_candidate_assurance(state, record, branch, observation, reason, impact_digest,
                                            published=False, identity=identity)
                return observation
            unmet = _publish_precondition_unmet(ci_attested_safe)
            if unmet:
                raise WriteAdapterError(f"refusing to publish: {unmet}")

            # 7) PUBLISH - crash-resumable and idempotent. The candidate commit is content-addressed
            #    (7.1.3), so re-running the same goal with the same content produces the SAME oid: a
            #    re-push is a no-op and an existing PR is reused instead of duplicated.
            journal = _journal_path(proposals_dir, state.goal_id or "goal", candidate_commit)
            prior = _journal_read(journal)
            remote_oid = _remote_ref_oid(repo_root, branch)

            if remote_oid and remote_oid != candidate_commit:
                # Someone else's commit is on our branch (a concurrent run, a leftover, a human). NEVER
                # clobber it - the lease would refuse anyway, but refusing here says WHY.
                raise WriteAdapterError(
                    f"remote {branch} already holds {remote_oid[:12]}, not this candidate "
                    f"{candidate_commit[:12]}; refusing to overwrite someone else's branch")
            if remote_oid == candidate_commit:
                reason = f"{reason}; branch already published (resumed)"
            else:
                # WRITE-AHEAD: record the intent before the effect, so a SIGKILL between push and PR
                # leaves a durable trace instead of an orphan nobody knows about.
                _journal_write(journal, "PUSH_INTENDED", goal_id=state.goal_id, branch=branch,
                               base_oid=base_oid, candidate_commit=candidate_commit,
                               candidate_tree_oid=tree_oid)
                #    An explicit <oid>:<ref> refspec cannot publish anything but the assured commit, and
                #    --force-with-lease is enforced REMOTE-SIDE: a plain push over a tip that happens to
                #    be an ANCESTOR was measured to succeed SILENTLY, rewriting an open PR's head.
                _git(wt, "push", "--force-with-lease=" + f"refs/heads/{branch}:{remote_oid}", "origin",
                     f"{candidate_commit}:refs/heads/{branch}")
                remote_oid = _remote_ref_oid(repo_root, branch)
                _journal_write(journal, "PUSHED", remote_oid=remote_oid)
            if remote_oid != candidate_commit:
                raise WriteAdapterError(
                    f"remote {branch} is {remote_oid[:12] or '(absent)'}, not the assured candidate "
                    f"{candidate_commit[:12]}; refusing to open a PR for an object we did not assure")
            identity["remote_oid"] = remote_oid
            _record_candidate_assurance(state, record, branch, observation,
                                        f"{reason}; branch pushed, PR not yet opened", impact_digest,
                                        published=True, identity=identity)
            refs = state.review_state.get("agent_proposals")
            if not isinstance(refs, list):
                refs = state.review_state["agent_proposals"] = []
            refs.append({"record_digest": record["record_digest"], "branch": branch,
                         "path": str(proposal_path),
                         "changed_paths": record["payload"]["changed_paths"], "pr": "",
                         "candidate_commit": candidate_commit, "candidate_tree_oid": tree_oid})
            for box in expected_head_box:
                box.expected_head = candidate_commit

            # Reuse an OPEN PR for this branch rather than duplicating it (a resumed run, or a crash
            # after `gh pr create` succeeded but before its URL was recorded).
            pr_out = prior.get("pr_url", "") or _existing_pr_url(gh_runner, branch)
            if pr_out:
                reason = f"{reason}; existing PR reused"
            else:
                # DRAFT: machine-authored code must not be one click from merge, and draft state is a
                # STRUCTURAL signal that survives a body edit - unlike the text disclosure alone.
                code, pr_out, err = gh_runner("pr", "create", "--draft", "--head", branch, "--base", base,
                                              "--title", (state.goal or "agent change")[:120],
                                              "--body", _pr_body(rationale, _target_has_ci(repo_root)))
                if code != 0:
                    raise WriteAdapterError(f"gh pr create failed (exit {code}): {err.strip()[:200]}")
            _journal_write(journal, "PR_OPENED", pr_url=pr_out.strip())
            _record_candidate_assurance(state, record, branch, observation, reason, impact_digest,
                                        published=True, identity=identity)
            refs[-1]["pr"] = pr_out.strip()
        return Observation.SUCCESS

    return execute


def build_context(repo_root: Path | str, base: str = "main", *, agent_client: AgentClient | None = None,
                  proposals_dir: Path | str | None = None, worktrees_dir: Path | str | None = None,
                  gh_runner: Any = None, branch_prefix: str = "cs-agent/", run_cs_assure: Any = None,
                  pr_ref: str | None = None, ci_attested_safe: bool = False,
                  sandbox: Any = None, require_sandbox: bool = False,
                  profile: TargetProfile | str = "corpusstudio",
                  assurance_root: Path | str | None = None) -> LoopContext:
    """A WRITE-CAPABLE, single-agent :class:`LoopContext` (Phase 7.1). ``capabilities={CAP_WRITE}`` - the
    capability gate REFUSES to run it without ``--allow-capabilities write``. It applies the agent's sealed
    diff in an isolated worktree, commits, pushes a branch, and opens a PR; it NEVER merges (``write_gh``
    refuses ``pr merge`` and ``dangerous=True`` escalates the merge gate). Inject a stub ``agent_client`` +
    dirs + ``gh_runner`` in tests.

    ``ci_attested_safe`` (default FALSE -> never publishes) is the OPERATOR'S EXPLICIT ATTESTATION that CI
    can safely execute an agent-authored candidate: no credential-persisting checkout, and no secret-bearing
    step in a job that runs candidate code. Pushing EXECUTES the candidate in CI before any human reviews
    it, and this adapter cannot verify someone else's CI by parsing it (every attempt failed open), so the
    person who can actually change the workflows asserts it."""
    root = Path(repo_root)
    prof = profile if isinstance(profile, TargetProfile) else load_profile(profile)
    # The TRUSTED assurance tooling need not live in the TARGET: pointing the loop at another
    # repository is exactly the case where it does not. Defaults to the target for CorpusStudio.
    trusted_root = Path(assurance_root) if assurance_root is not None else root
    wdir_probe = Path(worktrees_dir) if worktrees_dir is not None else default_worktrees_dir(root)
    if require_sandbox and sandbox is None:
        raise SandboxUnavailable(
            "require_sandbox=True but no sandbox was supplied; refusing to run the agent unisolated")
    if sandbox is not None:
        # PROVE it confines before trusting it. Presence is not proof: on the host this was written for,
        # `bwrap` is installed and cannot work (apparmor_restrict_unprivileged_userns=1). A sandbox that
        # silently fails open is worse than none - it converts a known gap into a false assurance.
        verify_sandbox(sandbox, wdir_probe)
    client = agent_client if agent_client is not None else ClaudeSubprocessClient(sandbox=sandbox)
    pdir = Path(proposals_dir) if proposals_dir is not None else _default_proposals_dir(root)
    wdir = Path(worktrees_dir) if worktrees_dir is not None else default_worktrees_dir(root)
    gh = gh_runner or write_gh(root)
    # LoopContext is a mutable dataclass and INTEGRATE runs later in the SAME process, so the executor can
    # bind the merge gate to the commit it actually published. The box is filled in below once the context
    # exists (the executor closes over the list, not the context).
    _expected_head_box: list = []
    # The SAME assurance runner drives BOTH the candidate assurance (static `impact` in the executor,
    # against the candidate worktree) and the loop's own OBSERVE/VERIFY (against the dev tree). Default to
    # the SANITIZED-env cs_assure runner so no assurance subprocess inherits secrets, and WRAP it so a
    # candidate is always analyzed by the TRUSTED dev-tree tool (never the candidate's own scripts/).
    assure = _trusted_cs_assure(trusted_root,
                                run_cs_assure if run_cs_assure is not None else _sanitized_cs_assure)
    kwargs: dict[str, Any] = {
        "repo_root": root, "base": base,
        "executor": _make_write_executor(client, root, base, pdir, wdir, gh, branch_prefix, assure,
                                         ci_attested_safe, _expected_head_box, prof),
        "reviewer": lambda _state: [],
        "critic": _signoff_critic,
        "multi_agent": False,               # single-agent: no delegated wave (verify_paths is a 7.3 concern)
        "capabilities": frozenset({CAP_WRITE}),
        "gh_runner": gh,
        "run_cs_assure": assure,
        # A gate the loop can run exists only if the profile names one, or the target carries the
        # default spec itself (CorpusStudio does; a foreign repo does not).
        "local_gate": bool(prof.resolved_gate()) or (root / _DEFAULT_GATE_RELPATH).is_file(),
        "dangerous": True,                  # the merge gate ESCALATES: a human merges the PR, never the loop
    }
    if pr_ref is not None:
        kwargs["pr_ref"] = pr_ref
    ctx = LoopContext(**kwargs)
    _expected_head_box.append(ctx)
    return ctx


# Re-export so `AgentError` (the confined-propose transport error) and the shared worktree-dir resolver are
# reachable via this module too - the executor may surface either error, and callers/tests use the resolver.
__all__ = ["AgentError", "WriteAdapterError", "build_context", "collect_orphan_branches",
           "default_worktrees_dir", "write_gh"]
