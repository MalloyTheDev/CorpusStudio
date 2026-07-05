# Current State

Single source of truth for what Corpus Studio actually does today. When another
doc disagrees with this file, this file wins (and the other doc should be fixed).

Last reconciled: 2026-07-04 (through **v1.3** — Evaluation Suites & Chat Gates —
plus the deep bug/security audit, 19 fixes across data integrity, gate/policy
hardening, and quality/split correctness, PRs #104–118; a residual-audit pass
hardening the v1.3 surface, PRs #133–142; and the CI dependency refresh, PRs
#94–101). Earlier
milestones: the Workspace shell, desktop polish, the LLM-judge evaluation scorer,
a crash-safe / distributable build, the god-object decomposition (Debt + Arena
tabs extracted), a unified JSONL reader, backend retry + per-item error
isolation, and off-thread document opens.

## What works today (implemented and tested)

**Author & validate**
- Local project creation from built-in schema templates with pre-filled examples.
- Schema validation through the Python engine: required non-empty fields,
  declared field types, list element types, enum/label sets, numeric bounds,
  nested object shapes, and chat message structure, with selectable issue
  navigation in the desktop.
- JSONL import preview with failed-row quarantine, review, and retry.
- **Hugging Face Hub dataset import** (read-only, public): `hf-inspect` /
  `hf-import` fetch rows via the public datasets-server JSON API (stdlib urllib,
  no `datasets`/`huggingface_hub` dependency), map columns to a schema, and write
  a *staging* JSONL that flows through the normal import-preview/quarantine path —
  the engine never writes `examples.jsonl`, gated/private datasets are refused, and
  the dataset license is surfaced with a "not assumed training-licensed" caveat. A
  desktop **Import from Hugging Face** dialog (inspect → pick config/split → map
  columns to the project schema → stage) drives it end to end through that same
  preview flow. See [`IMPORT_EXPORT.md`](IMPORT_EXPORT.md).
- Full Unicode correctness end to end: NFKC-aware tokenization and UTF-8 stdio
  so CJK/Cyrillic/accented text round-trips between the desktop and engine.

**Clean & measure**
- Quality report: empty rows, exact + normalized duplicates, low-information
  rows, synthetic-pattern warnings with near-duplicate clustering, PII/secret
  detection (emails, SSNs, private keys, AWS/API keys, JWTs, Luhn-valid cards —
  masked samples), token-length outliers, and category-imbalance warnings, with
  project-level quality history.
- Dataset **debt** ledger (`dataset-debt`): the quality signals, normalized by
  dataset size, ranked by severity, and graded A–F so you know what to fix first.
  Secrets/PII are graded by *presence* (a single leaked key is critical),
  everything else by rate; each item carries a concrete remediation. Surfaced in a
  desktop **Debt** tab (color-coded grade + ranked remediation list) whose grade
  invalidates the moment the dataset changes, so it never shows a stale verdict.
  See [`DEBT.md`](DEBT.md).
- Leakage-checked splits: `detect_split_leakage` reports exact and
  near-duplicate rows shared across train/validation/test.
- Export with an optional cleaning pass (dedupe / drop low-information) that
  writes a removal manifest; verbatim exports warn when duplicates remain.
- Preference exports (DPO/KTO/reward) with a pair-integrity gate.
- Dataset card summarizing metadata, schema, splits, quality, and evaluation.

**Govern & gate**
- Role-based provider/model capability policy, enforced **in the engine** (not
  just the UI): OpenAI/Anthropic evaluator-only by default; Ollama/local
  generation only when explicitly approved via project-local
  `provider_policy_overrides.json`; OpenRouter route-aware. CLI `provider-policy`
  / `provider-approve`; surfaced by a **Provider Generation Policy** panel in the
  desktop Settings tab (which providers may generate, approve/revoke a local
  model). See [`PROVIDER_POLICY.md`](PROVIDER_POLICY.md).
- A gate runner producing serializable pass/warn/block reports over the existing
  schema, quality, leakage, PII, and evaluation logic (no new detection).
  CLI `gate-run`; the export gate blocks on schema/PII failures. Per-project
  thresholds via `gate_thresholds.json` (fail-closed, BOM-tolerant; each report
  records the effective thresholds); `gate-thresholds` prints them. Surfaced by a
  **Run Gates** button in the desktop Quality tab. See [`GATES.md`](GATES.md).
- **Chat gates** (`chat-gate`, `chat_suite` scope): gate a chat dataset's
  conversation *structure* — assistant-not-first, role alternation (tool-aware),
  both-parties-present, no dangling user turn, system placement, turn-count bounds
  — separate from the per-message shape the validator already enforces. Advisory
  by default (warns); `block_chat_malformed` makes training-breaking faults block.
  Verdicts structure, not semantic quality. Surfaced by a **Run chat gates** button in
  the desktop Quality tab (shown for chat datasets), reusing the gate/Problems display.

**Evaluate & assist**
- Evaluation Lab against local Ollama or OpenAI-compatible endpoints: health
  checks, model discovery, report history, two-report comparison, regression
  reruns, tag/failure/score-band summaries, failed-row edit loops, manual
  scoring, saved failure filters. The default automatic score is **keyword-overlap
  recall** (`metric: "keyword_overlap"`) — a lexical proxy, **not** a quality judgment.
  An opt-in **LLM-judge scorer** (`eval-run --judge-model …`, `metric: "llm_judge"`)
  scores each answer 0–100 with a rationale, reusing the evaluator-only judge (provider
  policy enforced; no judge configured ⇒ no cloud call). Manual scoring stays the
  human-trustworthy path. See `docs/EVALUATION_LAB.md`. *(Judge scorer: engine + CLI;
  the desktop Evaluation-tab field is a pending follow-up.)*
- Multi-model benchmark (`benchmark`): one dataset across several models, ranked
  by the same keyword-overlap score, with per-model deltas and the examples every
  model failed.
- **Evaluation suites** (`suite-run`): a file-driven, reusable set of evaluation
  *cases* (dataset × model × metric × pass bars) run as one unit → a `SuiteReport`
  with a **per-metric** roll-up and an aggregate pass/warn/block verdict. Reuses the
  eval run + evaluation gate (no new scoring); per-case failure isolation; each case
  records its dataset fingerprint; advisory by default, `--strict` exits 2 on block.
  Never folds non-comparable metric scales. See [`EVALUATION_SUITES.md`](EVALUATION_SUITES.md).
  A per-project `evaluation_suites/` **registry** (`suite-init` / `suite-list` /
  `suite-run` by name), a desktop **Suites** tab (list / run / view), and
  **`version_id`-pinned cases** (a case re-evaluates the verified reconstruction of a
  pinned dataset version) all ship; suite history/trend is still future.
- Model Arena (`arena-run`): run a prompt suite across several models side by
  side; responses are comparison artifacts, not trainable rows. Optional
  evaluator-only judging (`--judge-model`) scores each response and picks a
  winner (per-model win counts + average judge score); judging is an evaluator
  activity, so OpenAI/Anthropic are permitted as the judge. Saved reports under
  `arena_reports/`. A desktop **Arena** tab surfaces prompts + model list +
  optional judge with side-by-side responses.
- Review-first AI Assist Lab: persistent accept/reject queue, saved views, bulk
  triage with undo, resumable rewrite batches. AI Assist output is always
  `review_required` and never auto-accepted. Generated candidate rows are run
  through the existing dataset gate runner (schema/quality/PII) before review and
  the verdict is attached as `candidate_gate` — a pre-review signal only: a clean
  gate is not approval, a block does not auto-reject, and provider policy is still
  enforced before generation. See [`AI_ASSIST_LAB.md`](AI_ASSIST_LAB.md).

**Train & track**
- Training config export for axolotl / TRL / Unsloth / Hugging Face /
  LLaMA-Factory with compatibility warnings, a real token budget
  (tokens-per-epoch after truncation, over-length counts), a rough arithmetic
  VRAM planning estimate (never inspects hardware), a LoRA rank/alpha suggestion,
  and the exact launch command.
- In-app launch of the user's installed trainer with explicit confirmation of
  the exact argv (no shell), live log streaming, and a Stop that kills the
  process tree.
- Checkpoint tracking during and after runs, resume-from-latest for targets
  with a CLI resume flag, and before/after evaluation comparison against the
  baseline captured at launch.
- A durable **training run registry** under `training_runs/` (`training-run-list`
  / `training-run-update`): launch metadata (argv, config, output dir), status
  lifecycle, pid, exit code, checkpoints, before-eval link. The desktop writes
  records directly; a run left `running` whose process is gone reconciles to
  `interrupted` on load. A read-only run history shows past runs in the Training
  tab. A `training_run` **regression gate** (`training-run-gate`) blocks when the
  trained model's **keyword-overlap score** dropped vs the baseline (a lexical proxy,
  not a quality verdict) and warns with "unverified linkage" when the after-eval
  targeted the base model. Surfaced by a "Gate run" button.
- A durable **model artifact registry** under `model_artifacts/`
  (`artifact-register`/`-list`/`-update`): the adapters/checkpoints a run produced,
  referenced by path (never moved). Base model + eval resolve live through the
  source `run_id`, not stored. **Path integrity** is re-checked on load and flags
  `missing` or `modified` if the weights change on disk: the artifact **list** uses a
  cheap size+mtime check over the weight files + descriptor (fast to glance), while the
  **weight card and promote gate** — the points where a decision is made — do a
  **byte-exact SHA-256** of the weight bytes, so even a size/mtime-preserving swap is
  caught. A live weight card
  (`artifact-card`, never stored) and a `model_artifact` **promote gate**
  (`artifact-gate`) that blocks "keep" when the artifact is `modified`/`missing`
  or the source run regressed. A desktop **Artifacts** tab registers from a run
  and keeps/rejects (promote-gated — a block refuses the keep).

**Version & restore**
- Durable dataset version records under `dataset_versions/` (`dataset-version-create`
  / `-list` / `-show`): each pins the dataset's identity — `row_count` + a
  streaming SHA-256 `content_fingerprint` over the ordered per-row exact
  signatures — plus links to source training runs, model artifacts, an evaluation
  report, and a dataset gate report. Nothing derivable is stored; scores/integrity/
  gate status resolve live in a version card. Live drift detection reports
  `matches` / `drifted` / `unreadable`, so a version can never silently
  misrepresent a changed dataset.
- A content-addressed, deduped **row store** (`dataset_versions/row_store.jsonl`)
  with a per-version ordered manifest, captured in one pass with the fingerprint.
  `dataset-version-diff` compares two versions (multiset added/removed/common +
  sample rows). `dataset-version-restore` reconstructs a version's rows to an
  `--output` file, verified against the recorded fingerprint (all-or-nothing,
  atomic, overwrite-safe). The engine **refuses to write `examples.jsonl`** — the
  dataset has one writer (the desktop).
- A desktop **Versions** tab: read-only history with a live integrity badge, an
  opt-in **Capture version** button, **View card**, a **diff view** ("Set diff
  base" → "Diff base → selected"), and **Restore this version** (in-place). The
  desktop in-place restore first captures the current dataset as an undo version
  and *refuses* if that undo isn't a genuine recovery point (rows couldn't be
  stored); then the engine reconstructs the selected version to a verified temp and
  the desktop atomically swaps it onto `examples.jsonl`. Any failure before the
  swap leaves the dataset untouched. See [`VERSIONING.md`](VERSIONING.md).
- After an import commit that added rows, the desktop **auto-captures a dataset
  version** (best-effort — an honest note if the snapshot can't be written, never a
  fake one).

**Workspace shell & desktop (v1.2.1–v1.2.15)**
- An **IDE-like workspace shell** with an activity bar: a **Start Center** (recent
  workspaces registry, a New Project wizard with a live folder-structure preview,
  open/initialize an existing folder — manifest-primary via `.corpus/project.json`),
  a **Universal Explorer** (VS Code-style file tree with file-type chips + an
  active-tab highlight, document tabs, text/JSON/image/binary viewers, a metadata
  panel; generated reports open read-only; `examples.jsonl` carries a single-writer
  caution), and the classic 15-tab **Studio** (Dashboard, Writing Studio, Examples,
  Preference Review, Quarantine, Splits, Evaluation, AI Assist, Training, Arena,
  Artifacts, **Suites**, Versions, Debt, Settings). Both New Project entry points open
  the one wizard. See [`WORKSPACE_SYSTEM.md`](WORKSPACE_SYSTEM.md).
- Two bottom-docked panels (mutually exclusive), toggled from the activity bar: a
  **Problems** panel (the dataset gate findings as a block-first list with fix hints
  and a count badge) and an **Output / Logs** panel (an ephemeral, local-only record of
  every engine CLI invocation — verb, outcome, duration, stderr on failure).
- Studio desktop refinements: a glanceable **dataset-debt grade badge** in the
  dashboard header (never auto-runs, marks itself stale on dataset change), a **debt
  trend** mini-chart (quality issue rate over recorded runs — labelled a lexical proxy,
  not the A–F grade), a **structured Quality metric grid** (PII-aware status banner +
  coloured rows, replacing the old text blob), a dark-themed **Projects** list, and the
  desktop surfacing of the **AI Assist candidate gate** (verdict + confirm-on-block).

**Distributable, resilient & crash-safe (v1.2.15)**
- A global exception handler (dialog + crash log at `%LOCALAPPDATA%/CorpusStudio/`)
  instead of silent process death; the engine service **never throws from its
  constructor** — a missing engine shows an in-app **"Python engine not found" setup
  screen** (locate the folder / set `CORPUS_STUDIO_ENGINE_DIR` / retry) rather than
  crashing. A **self-contained single-file** Windows publish profile
  (`dotnet publish -p:PublishProfile=win-x64`) needs no installed .NET runtime (the
  Python engine remains a runtime prerequisite). App + engine are versioned together.
- **Backend resilience:** local/OpenAI-compatible HTTP calls retry transient
  failures (HTTP 429/5xx + connection errors) with bounded exponential backoff and
  **fail fast on other 4xx**; health/model-list probes stay single-attempt. Each
  arena/evaluation/judge batch **isolates per-item failures** — one model's outage is
  recorded as a per-response backend error (surfaced in the Arena view) and the run
  finishes for every other model instead of aborting.
- **Data path:** one shared streaming JSONL reader backs strict reads, import
  preview, and file validation, so encoding (BOM-tolerant), blank-line skipping, and
  malformed-line handling can't drift; a declared-but-optional `orjson` accelerates
  the hot path when present with a stdlib fallback. Opening a document in the Explorer
  runs the read **off the UI thread** so a large file never stalls the window.
- Architecture: the desktop view-model is being decomposed from a single god-object
  into per-tab view-models behind interfaces, composed via a DI container
  (`Microsoft.Extensions.DependencyInjection`); a shared `ViewModelBase` and the
  **Debt** and **Arena** tabs are extracted so far (see
  [`CROSS_PLATFORM_ASSESSMENT.md`](CROSS_PLATFORM_ASSESSMENT.md), which relies on it).

## Hard boundaries (by design)

- Corpus Studio orchestrates the user's installed tools; it is **not** a deep
  learning framework. No CUDA, PyTorch internals, backprop, optimizers,
  distributed training, or custom training loops.
- Trainer launches show the exact argv, require explicit confirmation, use no
  shell, and write inspectable run metadata. No hidden trainer behavior.
- The engine never moves, copies, or deletes the user's weight files or
  `examples.jsonl` (reference paths only; the desktop is the single writer of
  `examples.jsonl`).
- No silent cloud behavior, publishing, dataset upload, or auto-acceptance of
  AI-generated dataset rows. Provider permissions are enforced in the engine, not
  just the UI.

## Not built yet (future roadmap)

- **Surface the LLM judge in the Evaluation tab** — the `--judge-model` scorer ships in
  the engine and in suites; the desktop **Evaluation** tab still defaults to keyword
  overlap with no judge-model field.
- **A real tokenizer** (transformers/tokenizers) so token-budget / VRAM numbers are
  exact rather than heuristic. Deliberately deferred: it would break the dependency-light
  engine, and the current estimate is documented as a heuristic.
- **HF export/push** (upload/publishing) — see the hard boundary above; it stays a
  deliberate non-goal for now. (Read-only Hub *import* already ships.)
- **Continue the view-model decomposition** beyond the Debt and Arena tabs; eventually
  an **Avalonia** port for macOS/Linux (see `CROSS_PLATFORM_ASSESSMENT.md`).
- Dataset-version **reorder detection**, row-store **GC** (must never prune
  manifest-referenced rows), and a normalized row identity. (Auto-capture after an
  import commit now ships — see above.)
- Smaller deferrals: PII auto-redaction; a per-project gate-threshold editor in
  the desktop; per-element object shapes for lists-of-objects in the validator; an
  app icon. (CI hardening — ruff, mypy, pytest gate, dependabot, and CodeQL — is
  in place.)
