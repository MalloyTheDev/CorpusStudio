# Production Single-Agent Runtime (#7) — Design

**Status: PROPOSAL for review. HELD.** This document designs the write-capable single-agent runtime — the
milestone the second external re-review named "Production Single-Agent Runtime." It is **not** a decision
to build write capability. Each capability increase below is a **separately-authorized** step; nothing in
this doc grants any write or merge capability. It exists so we agree on the architecture, the trust model,
and exactly what each phase does and does not do *before* any runtime code is written.

The design **builds on scaffolding that already exists and is hardened** (the 7-auditor pre-#7 audit,
PRs #695–#700). It adds *no* new foundational contract; it fills the injected-effect seams the controller
already exposes.

## 1. What already exists (the scaffolding this design fills)

The autonomous loop (`scripts/loop/`, [`docs/AUTONOMOUS_LOOP.md`](AUTONOMOUS_LOOP.md)) is an
**effect-injected, fail-closed, stdlib-only controller**. Every real effect is a callback on
`LoopContext`; the controller decides retry / stop / escalate / hold / merge / finalize and never
hardcodes an effect. The pieces #7 plugs into:

| Seam (already built) | Where | What #7 uses it for |
| --- | --- | --- |
| `LoopContext.capabilities` + `CAP_WRITE` / `CAP_MERGE` | `orchestrate.py` | An adapter DECLARES its effect capabilities; empty = read-only. |
| `cs_loop --allow-capabilities` + `_enforce_capabilities` | `cs_loop.py` | The runtime REFUSES (exit 2) a context declaring a capability the operator did not permit. |
| `__post_init__` write-guard | `orchestrate.py` | A write-capable **multi-agent** context with no `verify_paths` is refused at construction. |
| `verify_paths` (independent worktree-diff boundary) | `router.py` | Boundary enforcement against the real diff, not agent self-report. |
| `--scope head` / `merge_candidate` change-set scopes | `assurance/records.py` | Bind impact to the exact committed head (the deferred **#15b** wiring). |
| `merge_gate` + `GateEvaluation` + `_resolution_supports` | `integrate.py` | Evidence-bound merge authorization; no auto-merge on obligation identity. |
| `head_bound_merge` (`--match-head-commit`) + `required_checks` | `integrate.py` | Merge only the exact head CI validated; HOLD on a moved head. |
| single-writer `FileLock` (PID-liveness) + durable writes | `locking.py` / `store.py` | Per-run/goal state isolation; a concurrent writer fails closed. |
| exit-code taxonomy (0/3/4/5/6/2) | `cs_loop.py` | A supervising controller reads the outcome without parsing stdout. |
| the dry-run adapter (`read_only_gh`, `dangerous=True`) | `loop_adapters/dry_run.py` | The read-only reference an editing adapter is modeled on. |

**Confirmed boundary:** today the only merge effect (`gh pr merge`) fires solely through an injected
`gh_runner`; the sole shipped adapter wires a default-deny read-only `gh` and forces escalation. #7 is the
first adapter that could change that — which is why it is gated.

## 2. Trust model

The loop keeps two planes (see [`docs/AUTONOMOUS_LOOP.md`](AUTONOMOUS_LOOP.md) §"Two planes"):

- **The control plane is TRUSTED** — the stdlib controller + the assurance/evidence plane. It owns truth:
  only it admits a change, authorizes a merge, or marks a goal complete. (This is the same
  "Rust owns truth; Python computes ML" partition the target architecture states, with the stdlib
  controller standing in for the future Rust core.)
- **The Claude-Code agent is UNTRUSTED** — it *proposes* edits/plans; it does not get to vouch for itself.
  It runs behind the injected `executor` / `reviewer` / `critic` / `agent_runner` seams. The hardening
  already closed the ways it could cheat: it cannot self-grant a `HUMAN_APPROVAL` or self-fabricate
  evidence (pre-critic snapshot + `step()` authorization restore), it cannot edit out of lane
  (`verify_paths` vs the independent worktree diff), and its raises escalate rather than crash the loop.

**Standing invariant across every phase:** a change that fires a **human-gated** obligation
(`loop-controller-self-modify`, `assurance-self-modify`, `sealed-research`, `worker-closure`) **never**
auto-merges — the merge gate escalates it to an independent human. #7 does not touch that.

## 3. The capability ladder

Least-capable-first. Each rung is a distinct PR **and** (from 7.1 on) a distinct explicit authorization.

```
7.0  read / propose-only      capabilities = {}            <- SAFE: zero writes. A normal human-gated PR.
7.1  write-capable single     capabilities = {"write"}     <- GATED: explicit eyes-open go. Edits+PR, NO merge.
7.2  autonomous merge         capabilities = {"write","merge"}  <- SEPARATELY gated. Evidence-bound merge only.
7.3  write-capable multi-agent (later; needs verify_paths, per-goal worktrees)  <- out of scope here.
```

## 4. Phase 7.0 — read / propose-only single-agent adapter (SAFE)

**Goal:** exercise the *entire* loop against a real Claude-Code agent while making **zero** writes to the
repo — no source edits, no commits, no push, no PR, no merge. This is the credible dry-run *with a real
agent* and is within the least-capable-first envelope, so it ships as a normal human-gated
`loop-controller-self-modify` PR (I open, you merge).

**New file:** `scripts/loop_adapters/single_agent.py` exposing `build_context(repo_root, base)`.

- **`capabilities = frozenset()`** (read-only) — passes the capability gate with no `--allow-capabilities`.
- **Isolated git worktree** per run: the adapter creates a throwaway `git worktree add` under the
  operational dir (outside the main worktree), so the agent *reasons over* a checkout it can look at but
  whose edits never touch the developer's tree. Cleanup on exit (auto-removed if unchanged).
- **`executor` = propose-only:** invokes the agent to produce a **proposed unified diff** (+ a rationale),
  and writes it to a **proposal artifact** under the operational dir (e.g.
  `<git-dir>/corpusstudio-loop/proposals/<run-id>.diff`). It does **not** apply the diff. Returns the loop
  `Observation` (SUCCESS/PROGRESS/…) based on whether a proposal was produced — never a merge.
- **`reviewer` / `critic`:** real agent calls that read the proposal + the assurance evidence and return
  `ReviewFinding`s / `Criterion`s — but, per the trust model, the critic's judgment can never *alone*
  finalize (the completeness layer already enforces this).
- **`gh_runner` = read-only** (the `read_only_gh` default-deny allowlist pattern from `dry_run.py`);
  `agent_runner = None` (single-agent); `pr_ref = None` so `INTEGRATE` never calls `gh` for a merge.
- **Outcome:** the loop runs decompose → execute(propose) → observe → review → verify/completeness and
  ESCALATES (a human decides whether to apply the proposal). Nothing is written to the repo.

**Open decision (needs your input): how the adapter invokes Claude Code.** Options: (a) a fixed-`argv`
subprocess to the `claude` CLI in headless/propose mode; (b) the Claude Agent SDK; (c) an injected
callback the host wires. All must honor the **no-shell** rule (argv lists) and produce a *diff artifact*,
not apply edits. 7.0 is the natural place to settle this without any write risk.

**Verification (deterministic, the loop's existing pattern):** an end-to-end test with a **stub agent**
that returns a canned proposal; assert (1) the worktree/source has **no** commits or tracked-file changes
after the run, (2) the proposal artifact is produced, (3) the run ESCALATES (never FINALIZE-with-merge),
(4) `gh` is never called with a write verb.

## 5. Phase 7.1 — write-capable single agent (GATED — needs explicit go)

**Goal:** the agent actually *edits* files, commits, pushes a branch, and opens a PR — **in the isolated
worktree only** — but **does not merge**. A human merges the PR.

### 5a. What 7.1 shipped (#703) + confinement (7.1.1)

- **`capabilities = frozenset({CAP_WRITE})`**; the operator must pass **`--allow-capabilities write`** or
  the runtime refuses (exit 2). This is the machine-checkable opt-in.
- **Effects gain write, but NOT merge:** `write_gh` may `git push` a branch + `gh pr create`; it
  **refuses `gh pr merge`** (and every other mutation). Source edits + commits happen in an **isolated,
  disposable worktree**; the developer's main tree is never touched.
- **The unit of change is the agent's OWN sealed diff.** The agent proposes a unified diff (sealed as the
  7.0 `agent_proposal` record); the executor applies *that exact diff* deterministically with `git apply`
  into a **separate, pristine** branch worktree, then **classifies what git actually realized** against the
  writable allowlist (status / mode / path / basename / bounds / secrets / invisible characters) before
  committing and publishing. *(The original `changed_paths`-match check was replaced in 7.1.2: comparing
  the applied paths to the proposal's own self-reported list only proved the diff agreed with itself. The
  allowlist classification is strictly stronger — it is computed from git plumbing and does not trust the
  proposal at all.)*
- **Agent confinement (7.1.1):** the untrusted agent runs with **cwd inside a disposable, detached
  worktree** at `base` (never the developer's tree, never the apply worktree), a **sanitized (secret-free)
  environment** (no `*TOKEN*`/`*SECRET*`/cloud/registry/VCS auth reaches it), a **version-pinned read-only
  tool policy** (`Read,Grep,Glob` allowed; `Edit,Write,Bash,Task,WebFetch,WebSearch,NotebookEdit` denied),
  and a **bounded output** cap. Whatever it writes into that throwaway checkout is discarded; only the
  diff it *returns* is used.
- **Merge stays human:** the merge gate escalates (`AUTHORIZATION_REQUIRED` via `dangerous=True`); a human
  reviews and merges the PR. 7.1 proves the *write* half end-to-end with the merge button still human-held.

### 5b. Remaining hardening before 7.1 runs against a real repo (7.1.2 – 7.1.5)

The features below were listed here as "lands in 7.1" but were **deferred** — they are **not** in #703 and
7.1 must **not** be run against the real CorpusStudio repository until they land. Each is a distinct,
separately-authorized PR under `loop-controller-self-modify`:

- **7.1.2a — candidate assurance gates the publish. ✅ SHIPPED.** The candidate worktree is classified by
  **`cs_assure impact`** (self-modify / sealed-research / **worker-closure** / policy obligations) — a
  **STATIC** analysis that does **not** import or execute the candidate's code; the branch is pushed + the
  PR opened **only** when the candidate classifies clear; a human-gated / worker / policy obligation
  publishes **nothing** and returns the classified observation so the loop escalates. Every cs_assure
  subprocess runs with a **sanitized (secret-free) environment**, and every git invocation pins hooks off
  (`-c core.hooksPath=/dev/null` — measured: `worktree add` runs `post-checkout` and `commit` runs
  `pre-commit`, so an in-tree hook dir like husky's would otherwise execute candidate content). The apply
  worktree is **detached** and the branch is created by refspec only on a clear candidate, so a blocked
  candidate leaves no orphaned local branch. Worker-reachability is covered by the candidate
  `worker-closure` classification (→ `WORKER_LINEAGE_IMPACT` → escalate).
  - **ALLOWLIST, not denylist — and git is the authority on what changed.** A denylist of protected paths
    was built first and failed three adversarial passes (case variants like `Scripts/argparse.py` slipped
    the case-sensitive matcher; a `rename from` header hid a denied source from the hand-rolled
    `---`/`+++` parser and published a diff deleting `.github/dependabot.yml`; a `copy from
    secrets/deploy.pem` delivered credential bytes with **zero `+` lines**, invisible to both the filename
    denial and the added-line secret scan). Both root causes are unbounded by construction: a denylist
    enumerates badness, and a hand-rolled diff parser disagrees with git's real semantics. The rewrite:
    - **Writable surface = `engine/corpus_studio/**/*.py`, MODIFY-ONLY.** Anything not positively matched
      is refused, so an unforeseen path is denied by default and a case variant simply fails to match
      (matching is **case-folded**; the repo carries `core.ignorecase=true`). **No creates at all** —
      a create is how `conftest.py` injection and stdlib shadowing land.
    - **`engine/tests/test_*.py` is deliberately NOT writable yet**, though "add a missing test" is a
      named first target: CI triggers on a bare `push:` with **no `persist-credentials: false` anywhere**
      in `.github/`, so an agent-authored test would execute on the runner with a `GITHUB_TOKEN` readable
      from `.git/config` *before any human sees the PR*. Allowing it needs that CI hardening first.
    - **Globs are compiled, never `fnmatch`.** Measured: `fnmatch('engine/tests/test_x/conftest.py',
      'engine/tests/test_*.py')` is **True** (an fnmatch allowlist would admit an auto-executed conftest)
      while `fnmatch('engine/corpus_studio/cli.py', 'engine/corpus_studio/**/*.py')` is **False** — wrong
      in both directions. `_compile_pathglob` gives `**/` = zero-or-more segments and `*`/`?` that never
      cross `/`.
    - **The change set comes from `git diff-index --cached -r --no-renames -z <base>`** (plumbing, so it
      ignores `diff.renames` config; raw format carries **modes and blob OIDs**). `--no-renames` is passed
      **alone** — measured, `--no-renames -M0 -C0` still prints `R100` because `-M0` means "detect renames
      at 0% similarity", i.e. it *re-enables* detection. With detection off a rename decomposes into
      `D <source>` + `A <dest>`, which is the only way the deleted source becomes visible.
    - **Status must be `M` and both modes `100644`**, so symlinks (120000), submodules (160000) and
      exec-bit flips are refused *structurally* rather than by sniffing diff text.
    - **Secrets are scanned in the realized blob** (`git cat-file`, size-capped, NUL-tested for binary),
      not in `+` lines — the only layer that catches a `copy from`, since a copy leaves its source
      untouched and git reports only the allowed destination. (`.gitattributes` eol normalization also
      means committed bytes can differ from the diff text.)
    - Bounds: ≤2 changed paths, ≤60 changed lines, ≤1 MiB per blob.
  - **Why static, not the full gate (a security decision, not a shortcut):** running `cs_assure verify`
    against the candidate would run ruff/mypy/**pytest**, and *pytest collection executes the untrusted
    agent's committed code with the operator's environment* — arbitrary code execution + secret exposure,
    the very thing 7.1.1 confinement prevents for the propose step (an adversarial-review workflow found
    this in the first draft, which naively ran the full gate). The **dynamic** gate — which must execute
    the candidate — is therefore delegated to **CI's isolated sandbox** on the opened PR, where a human
    merges only on green. Local dynamic assurance in a hardened local sandbox (container / nsjail, network
    off, sanitized env) is a **7.1.5** option. *(Publish still happens inside EXECUTE, now gated on the
    static candidate assurance.)*
  - **The candidate is analyzed by the TRUSTED tool, never its own copy.** A second adversarial pass found
    that invoking `<candidate>/scripts/cs_assure.py` puts `<candidate>/scripts` on `sys.path[0]`, so a
    candidate that merely *adds* `scripts/argparse.py` (or `json.py` / `sitecustomize.py`) gets it
    **imported and executed as the operator** before any analysis runs — full filesystem + network access,
    so a sanitized env does not stop it reading credential *files* (`~/.config/gh/hosts.yml`, `~/.aws/…`,
    `~/.ssh/id_*`). Verified empirically. Two independent layers now close it: the runner always executes
    the **trusted dev-tree** `cs_assure.py` and points it at the candidate with **`--start-dir`**, and the
    **entire `scripts/` tree** is denied pre-apply so such a file can never reach the candidate at all.
- **7.1.2b — deferred.** A **persistent** candidate worktree with a loop-native observe→diagnose→**correct**
  loop (iterating in the worktree rather than re-proposing from scratch) + an **independent REAL reviewer**
  over the candidate. Needs the worktree lifecycle (create/reuse/dispose across phases) that 7.1.4 builds;
  the current adapter's coarse correction is "re-propose on a routed observation", bounded by the step cap.
- **7.1.3 — exact candidate identity.** Record `candidate_tree_oid` + a staged-patch digest;
  `git commit-tree`/head verification so the pushed commit is provably the assured candidate; wire
  `cs_assure impact --scope head` bound to the pushed commit (`--match-head-commit`).
- **7.1.4 — crash recovery + safe publish.** A write-ahead effect journal (crash-resumable), idempotent
  branch/PR reuse, orphan-branch cleanup (a failed `pr create` currently leaves a pushed branch), a
  **DRAFT** PR, validated remote/PR identity, and a cleanup/gc command.
- **7.1.5 — live canary** with fault injection at every boundary before any unattended use.
- **Sensitive-path denial** (the review's "initially deny" list) — **superseded and subsumed by the 7.1.2a
  ALLOWLIST**, which is strictly stronger: `.env*`, `*.pem`, `*.key`, credential stores, workflow /
  permission changes, sealed research, submodules, symlinks, mode/binary changes, over-large changes,
  gate-affecting config, assurance code and loop-controller code are all outside
  `engine/corpus_studio/**/*.py`, so they are refused *by not being allowed* rather than by being
  enumerated. Nothing needs adding to a denylist when a new sensitive path appears.
- **Two follow-ups this work surfaced** (each its own PR, NOT bundled here):
  1. **`.github/workflows/*` needs `persist-credentials: false`** — prerequisite before `engine/tests/**`
     can ever join the writable surface.
  2. **The `cs_assure` policy matcher is case-sensitive** (`assurance/obligations.py::glob_matches` —
     measured: `Scripts/loop/x.py`, `.GitHub/workflows/e.yml` and `Research/…` fire **zero** obligations).
     That is an assurance-plane bug affecting every `cs_assure` user, not just the loop; fixing it fires
     `assurance-self-modify` and needs its own review. The 7.1.2 allowlist is robust to it regardless,
     since a positive gate does not depend on an obligation firing.

**Not a worker-lineage change.** The adapter is control-plane code (`scripts/loop_adapters/`), not worker
execution bytes, so it does **not** force a fresh worker wheel/env — but it IS under
`loop-controller-self-modify`, so it is admitted only by trusted-base tests + CI + independent human
review (the maintainer; Sourcery is advisory, not the independent gate), never the loop's own gate.

**Verification (as shipped):** an **attack matrix** drives each verified hole end-to-end through the real
git path and asserts it publishes **nothing** (no PR, no remote branch, dev tree pristine): a case-variant
protected path, a `rename from` hiding a denied source, a `conftest.py` create, a symlink, an exec-bit
flip, an edit outside the surface, a secret smuggled into an *allowed* in-place edit, and an oversized
change. Plus: the compiled glob matcher is proven path-aware where `fnmatch` fails open; tests assert edits
appear **only** under the isolated worktree path (the
main tree's `git status` is clean); the agent runs confined (cwd = a disposable worktree, sanitized env,
bounded output); a diff that won't apply fails closed; a **human-gated candidate
obligation**, a **worker-closure candidate**, a **sensitive path** (incl. gate-config), and a **secret in
the diff** each publish **nothing** (no push, no PR) and the candidate outcome is recorded; a **clear
candidate** publishes by refspec from a detached worktree; candidate assurance is proven to *analyze* the
**candidate worktree** while *executing* the **trusted dev-tree tool** (`--start-dir`, never the
candidate's own `cs_assure.py`); a new top-level `scripts/` file is denied; the default cs_assure runner
sanitizes the env; `INTEGRATE` escalates rather than merges; a self-modify-shaped change still escalates.

**Real-run caveat:** the tests drive candidate assurance with an injected fake `run_cs_assure`; the real
default runner runs `cs_assure impact` (static, fast) inside the candidate worktree. The loop's own
dev-tree OBSERVE/VERIFY still runs the full ruff/mypy/pytest gate, but on the **trusted** dev tree (never
the untrusted candidate), which must stay `workspace_stable` (pytest caches must be gitignored). End-to-end
validation against a live candidate — and any *local* dynamic (test) assurance — is the 7.1.5 canary /
sandbox.

## 6. Phase 7.2 — autonomous merge (SEPARATELY gated)

**Goal:** the loop may merge a *product* PR autonomously — only when the evidence proves it safe.

- **`capabilities = frozenset({CAP_WRITE, CAP_MERGE})`**; its own explicit authorization + `--allow-capabilities write merge`.
- **Evidence-bound merge (already built, #14):** `merge_gate` authorizes only when **every** blocking
  obligation has a **trusted, current** `ObligationResolutionRecord` (`RESOLVED` + `APPLICABLE_TO_HEAD` +
  `TRUSTED`), and a `GateEvaluation` is recorded. Human-gated obligations are **never** dischargeable.
- **The obligation-resolution PRODUCER (the remaining #14 half):** derive resolutions from CI — a
  **required** check green **for the exact head** is a `trusted-base-ci` resolution for the obligation it
  discharges (e.g. the web schema-diff check → `contracts`). This is the one net-new evidence producer;
  it consumes `required_checks` + the head-bound CI snapshot the loop already reads.
- **Head-bound merge:** `gh pr merge --match-head-commit <expected_head>`; a commit pushed since CI HOLDs.

**Verification:** a product PR with a matching trusted resolution merges; the same PR without one
escalates; a self-modify / assurance / worker / sealed PR **never** merges regardless of CI.

## 7. Safety invariants (must hold in every phase)

1. A **human-gated** obligation never auto-merges — always escalates. (Enforced by `merge_gate` +
   the shared `HUMAN_GATED_OBLIGATIONS`.)
2. The agent **cannot self-approve**: no self-granting a `HUMAN_APPROVAL`, no self-fabricated evidence
   (pre-critic snapshot + `step()` authorization restore).
3. Any **write-capable MULTI-AGENT** context must carry `verify_paths` (or be refused at construction) —
   `__post_init__` enforces exactly that, and `verify_paths` only bounds a *delegated* wave. The shipped
   single-agent write path (7.1) is `multi_agent=False` and does not carry one; its boundary is the
   git-authoritative allowlist classification instead. (Corrected: this invariant previously read as
   applying to every write-capable context, which the shipped adapter does not satisfy.) Boundary
   enforcement is against the **independent** worktree diff.
4. **Capability declaration + operator opt-in**: a write-capable adapter cannot be loaded without
   `--allow-capabilities`.
5. **Isolated worktree per run/goal**: the main working tree is never edited; goals cannot clobber each
   other; every run gets a fresh run id + run-scoped output.
6. **Fail-closed everywhere**: an unusable assurance read, a raising effect, a locked state file, an
   unmapped terminal phase, or an unbindable head all fail closed (escalate / exit ≥ 2), never advance.
7. **The loop never admits changes to itself** (loop-controller / assurance): the merge gate + OBSERVE
   both escalate them; #7 does not weaken this.

## 8. Open questions for review

- **Agent invocation mechanism** (§4): `claude` CLI subprocess vs Agent SDK vs injected callback. Settle
  in 7.0.
- **Worktree lifecycle**: naming, cleanup policy, disk bounds, and what happens on a crashed run
  (the PID-liveness lock + auto-remove-if-unchanged cover the common cases).
- **Proposal artifact format** (7.0): unified diff + JSON rationale? where stored / retention.
- **7.1 authorization ergonomics**: how the human sees + grants a subject-bound request (`cs_loop
  authorize` extension per #5-slice-2).
- **Scope of the first product target**: which real, low-risk change should 7.1 prove end-to-end first?

## 9. Governance

- **7.0** — normal human-gated `loop-controller-self-modify` PR (read-only adapter; no capability).
- **7.1** — explicit, eyes-open authorization **before** the PR, then a human-gated PR; the write-capable
  step is a *separately-gated commit* from any read-only groundwork.
- **7.2** — a further separate explicit authorization + human-gated PR.
- Multi-agent writes (7.3) are out of scope until 7.1/7.2 are proven.

At no point does the candidate self-merge a controller / assurance / research / worker / dangerous change;
an independent reviewer merges. This design changes none of the honesty invariants — it wires real effects
into seams that already fail closed.

## 10. Chosen defaults + contracts (agreed this review)

The robustness decisions for §8, and the concrete contracts 7.0 implements:

- **Agent invocation = an injected `AgentClient` callback backed by an out-of-process, fixed-argv `claude`
  subprocess with a framed JSON contract over stdio.** NOT the Agent SDK (it violates the adapter
  stdlib-only rule and runs the untrusted agent in-process). The callback is the loop-level seam; the
  subprocess is a swappable transport (a stub is injected in tests). This matches the target
  "out-of-process protocol, not PyO3" architecture, keeps the agent killable/bounded, and honors no-shell.
- **Agent output is untrusted, validated fail-closed** into a sealed record — never trusted as free-form
  text (the same discipline `observe.py` applies to a `cs_assure` record).
- **Two kinds of worktree, both disposable.** *Agent confinement* (7.1.1) applies in **both** adapters:
  the untrusted agent always runs with cwd inside a disposable, detached worktree at `base` (never the
  developer's tree), even when only proposing — the original "7.0 needs no checkout" plan was upgraded so
  a mis-behaving agent cannot edit the working tree while "just proposing". *Write isolation* — applying
  the sealed diff, committing, pushing — happens in a **separate, pristine** branch worktree that only the
  write step (7.1) creates, and only the returned diff (never the confined checkout) crosses into it.
- **7.1 authorization** (subject-bound, deferred to 7.1.3/7.2) = a sealed `AuthorizationRequest` bound to
  `(goal_id, capability, subject = head_sha / change-set fingerprint)`, one-time, stale-on-subject-change
  (no replay). Not in #703: the shipped opt-in is the coarse `--allow-capabilities write` gate only.
- **First 7.1 product target** = a single-file, deterministically-gated change firing **zero** human-gated
  obligations (a docstring / typo / lint fix or a missing-test add); never a `scripts/loop`,
  `scripts/assurance`, `.github`, or worker path.

### `AgentClient` protocol (7.0)

```python
class AgentClient(Protocol):
    def propose(self, request: dict) -> dict:
        """Given {goal, goal_id, base_oid, directive, repo_root, _cwd}, return
        {"unified_diff": str, "rationale": str}. RAISES on transport/output failure (fail-closed).
        `_cwd` is the disposable, confined worktree the transport runs the agent in (7.1.1)."""
```

The real `ClaudeSubprocessClient` runs `["claude", "-p", "--output-format", "json", "--allowedTools",
"Read,Grep,Glob", "--disallowedTools", "Edit,Write,Bash,Task,WebFetch,WebSearch,NotebookEdit"]` (fixed
argv, no shell) **confined** to `cwd = request["_cwd"]` (the disposable worktree) with a **sanitized
(secret-free) environment**, feeds the request as JSON on stdin, and validates the JSON response shape
before returning; a bad exit / unparseable / **oversized** / wrong-shaped response raises (bounded by both
a timeout and a max-output-bytes cap). The tool-policy flag names are version-sensitive, so `argv` is an
operator-tunable default — process-level confinement (worktree cwd + sanitized env) is the load-bearing
boundary; the tool policy is defence-in-depth. Tests inject a deterministic stub.

### `agent_proposal` sealed record (7.0)

```
{ "record_type": "agent_proposal", "schema_version": 1,
  "payload": { "goal_id", "base_oid", "unified_diff", "changed_paths", "rationale" },
  "record_digest": "sha256:<over the record minus this field>" }
```

Written under the operational dir (`<git-dir>/corpusstudio-loop/proposals/<run-id>.json`), **outside** the
working tree. It is the reviewable, tamper-evident artifact a human inspects; 7.0 **applies nothing** and
ends `ESCALATED`.
