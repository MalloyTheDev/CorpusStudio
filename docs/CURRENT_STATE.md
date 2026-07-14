# Current State

Single source of truth for what Corpus Studio actually does today. When another
doc disagrees with this file, this file wins (and the other doc should be fixed).

Last reconciled: 2026-07-14 (through **v1.3** — Evaluation Suites & Chat Gates — plus the **platform
run lifecycle** re-scope: profile → plan → predict-fit → run → measure-fit → artifacts, a multi-backend
registry, and subprocess reliability. The pre-Phase-9B real GPU workload evidence remains historical
native Windows/WDDM evidence. The managed native-Linux environments now separately seal the legacy
minimal hardware tuple, the readiness-v2 complete math QLoRA tuple, and the readiness-flash-v1 tiny
forced-flash QLoRA tuple. The first real 0.5B bounded flash smoke loaded the model but failed placement
verification before adapter insertion; a placement-only diagnostic found every parameter and buffer
on `cuda:0`. No real optimizer step has passed through `platform-run`, and sequence length 4096 remains
unverified. These environment/diagnostic results are not a 7B workload or offload result.
The post-v1.3 additions reconciled below include the **reasoning-traces** data loop, a dataset
**truncation guardrail** +
configurable **checkpoint retention**, and a dependency-light **storage safe-spill** profiler —
plus the sealed **Environment Manager** reference lifecycle, the static MoE-safe
**ModelDescriptor / TokenizerDescriptor** foundation with allowlisted **MoE topology inspection**,
the backend-independent **TrainingObjective registry**, and the identity-bound **backend worker
protocol 2.0** — plus the deep bug/security audit, 19 fixes
across data integrity, gate/policy
hardening, and quality/split correctness, PRs #104–118; a residual-audit pass
hardening the v1.3 surface, PRs #133–142; the CI dependency refresh, PRs
#94–101; and the Avalonia-migration decomposition — **all per-tab view-models
extracted** and the WPF code-behind engine handlers being converted to shared
testable commands behind `IEngineService`/`IDialogService`/`IFilePickerService`
seams, PRs #146–243). Earlier
milestones: the Workspace shell, desktop polish, the LLM-judge evaluation scorer,
a crash-safe / distributable build, a unified JSONL reader, backend retry +
per-item error isolation, and off-thread document opens.

## What works today (implemented and tested)

**Author & validate**
- Local project creation from built-in schema templates with pre-filled examples.
- Schema validation through the Python engine: required non-empty fields,
  declared field types, list element types, enum/label sets, numeric bounds,
  nested object shapes, and chat message structure, with selectable issue
  navigation in the desktop.
- JSONL / CSV / TSV / Parquet import preview with failed-row quarantine, review, and
  retry (CSV/TSV/Parquet convert to a staging JSONL and flow through the same
  preview/commit path; CSV/TSV cells import as text so schema type mismatches
  quarantine like a bad JSONL row, while Parquet keeps its column types. Parquet
  needs the engine's optional `[parquet]` extra — a clear install hint when absent).
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
- Export format is JSONL by default (model-ready, all schemas); **CSV/TSV export**
  is available for flat schemas (a chat/nested-object schema is refused, not
  lossy-flattened) and **Parquet export** for every schema (columnar, chat/nested
  included; needs the optional `[parquet]` extra) — all through the same
  validate/gate/clean pipeline.
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
- **Provenance gate** (`provenance-gate`): a per-row *licensing* check — reads each row's
  declared `meta.teacher` and quarantines rows generated by a provider you can't train on
  (e.g. Anthropic/OpenAI), the counterpart to `provider-policy` (generation-time). Buckets
  into trainable / quarantined / unknown; **blocks** on quarantined (`--strict` also on
  unknown). Reuses the provider policy; a `provenance_allowlist.json` / `--allow-teacher`
  clears open teachers. Surfaced by a **Check provenance** button in the desktop Gates panel,
  and enforced before writing a training deliverable via `export --check-provenance`
  (refuses the export on a quarantined row). Trusts the *declared* teacher; unknown ≠ safe.
  See [`GATES.md`](GATES.md).

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
  pinned dataset version) all ship. **Suite run history + trend** (`suite-history`): each run
  appends a timestamped point (verdict + per-status case counts, capped) that the Suites tab shows
  as a newest-first trend — a count over time, never a folded quality score.
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
- The WPF/Avalonia Training Lab launches reviewed, no-shell argv for installed **external** trainers.
  It intentionally emits no direct first-party launch and refuses the old mutable-config path. The
  Tauri/React Platform client plans and executes first-party work through `platform-plan` →
  `platform-run`. Supporting first-party commands are `train-check`, `train-merge`, and `model-fetch`;
  low-level `train-run` is an explicitly acknowledged and labeled development-only route.
- **Truncation guardrail** (`dataset-tokens`): measures a dataset's token-length distribution and how
  many examples a given `sequence_len` would **truncate** (cutting the end - including the model's
  answer). New platform plans render and tokenize the complete hash-pinned JSONL with the exact pinned
  tokenizer/template; over-length rows fail closed unless `allow_truncation` is explicit in the seal.
  The standalone report retains a documented heuristic when the tokenizer extra is absent.
- **Resolved checkpoint/output policy**: cadence and retention are explicit in the effective execution
  seal, and each run writes beneath `<output-root>/runs/<run-id>/`. The trainer saves the **LoRA
  adapter** + tokenizer + model card, not a full base copy. Atomic checkpoint promotion and exact-resume
  verification remain future integrity work and are not claimed by this slice.
- **Versioned reasoning/tool trace foundation** — the language-neutral, hash-sealed `TraceRecord`
  preserves exact source-row lineage, ordered role context, reasoning/action/tool/result/final-answer
  boundaries, producer/model/prompt/request/response evidence, typed validation findings, and a
  separate immutable human-review decision. `trace-generate` now fails closed on requested-provider
  policy before any backend call, reauthorizes the backend-reported model, writes pending records plus
  a sanitized per-attempt report, and never promotes self-filtering to review. `trace-migrate`
  explicitly seals legacy rows; `trace-review` recomputes engine validation and writes
  approved/rejected successors; `trace-validate --require-approved` and first-party trainer admission recheck
  validation, current external project policy (unknown/frontier default-deny), and segment semantics,
  then refuse pending/rejected/tampered/unsupported records before model loading. The built-in `trace` draft
  schema makes the ordinary desktop Writing Studio the authoring foundation. Legacy rows and the
  `<think>reasoning</think>answer` training renderer/no-think baseline remain compatible, and
  `eval-run --reasoning` still scores only the final answer. Generated/imported reasoning remains
  explicitly unverified. No dedicated graphical Trace Studio, tool executor, semantic correctness
  judge, or tool/process trainer is claimed. See [`TRACE_RECORDS.md`](TRACE_RECORDS.md).

**Platform run lifecycle** (headless, contract-first)
- A language-neutral contracts substrate (`engine/corpus_studio/platform/`) turns goal + data +
  hardware into a validated run: **profile** the host + functional capability probes → **plan** a
  hash-sealed RunPlan → predict the **fit** (never `NATIVE_SAFE` from an estimate) → **run** it (via
  the supervisor, in-process or a kill-able **subprocess** worker) → **measure** the fit (watchdog) →
  account for the **artifact** (integrity-checked). CLI: `platform-probe / -plan / -run / -backends /
  -profiles / -storage / -schemas`.
- **Storage safe-spill profiler** (`platform-storage`): a dependency-light, **non-destructive** probe
  of the host's storage topology (mount / capacity / interface — NVMe / SATA / USB / network / virtual;
  no benchmark, no SMART read) plus a **per-role suitability** verdict that refuses an offload /
  checkpoint / scratch path on a USB bridge, a cloud-sync folder, a nearly-full disk, or inside the
  source repository — and flags the *runtime* risks a USB SSD or a WSL `/mnt` host drive creates for
  the model cache, dataset, repo, and venv (small-file / load-latency stalls). `--diagnose "<error>"`
  triages a training failure as storage-implicated (I/O error, dropped drive, full disk) vs a
  VRAM/kernel failure the disk can't explain; `--recommend` prints the per-role storage tier. The
  prerequisite for honest offload planning. See [`HARDWARE_STORAGE_PROFILE.md`](HARDWARE_STORAGE_PROFILE.md).
- **Environment Manager reference lifecycle** (`env-runtimes` / `env-plan` / `env-create` /
  `env-status` / `env-probe` / `env-lock` / `env-remove` / `env-recreate`): the 3-layer dependency
  model in code — a lightweight always-installable **control plane**, opt-in **capability profiles**,
  and **isolated per-backend worker environments**. The side-effectful path is deliberately limited
  to the legacy `backend-corpus-studio` rollback recipe, the exact-pinned math
  `backend-corpus-studio-readiness-v2` recipe, and the exact-pinned flash
  `backend-corpus-studio-readiness-flash-v1` recipe. It discovers Python runtimes, renders concrete
  no-shell argv + explicit indexes, binds a byte-exact worker wheel for readiness recipes, and
  requires the exact plan hash before mutation. Installation captures sanitized pip source/artifact
  evidence, validates the worker wheel's own METADATA/RECORD before install, compares its immutable
  payload files with the installed worker, and inventories every installed `RECORD` byte plus a
  manager-computed digest of every listed file. Unrecorded site-package files and symlinks fail before
  installed torch is imported in a second process; a final lock is sealed only after the required
  probes and a stable post-probe inventory.
  Health/capability probes are also bracketed by clean inventories. Manager 1.2 preserves the sealed
  1.1 readiness-v2 math rollback identity while requiring stronger measured configuration for every
  new creation. The existing manager-1.1 flash lock is preserved as historical evidence but is not
  grandfathered across the new adapter-state equality requirement; it requires replacement before a
  manager-1.2 health claim. Readiness-v2 requires one complete BF16/NF4/double-quant
  QLoRA math-SDPA forward/loss/backward/optimizer/adapter-reload tuple. Flash readiness-v1 requires
  the same complete QLoRA evidence under forced `SDPBackend.FLASH_ATTENTION` (no math/mem-efficient
  fallback), with CUDA bf16 autocast on the forced forward/backward so attention dtypes match real
  TRL/PEFT QLoRA training, plus scoped GPU allocator, `nvidia-smi` process-memory, host-RSS, phase
  timing, and optional temperature/power evidence. Flash readiness is Linux-only; independent probes
  cannot be unioned across either complete tuple. The math readiness-v2 environment is the preserved
  safety/rollback baseline and must not be mutated by flash work. The manager detects worker/dependency drift and safely removes or recreates
  only contained owned paths.
  `RunPlan.environment_ref` pins the immutable
  lock hash and `platform-run --subprocess` dispatches with that managed interpreter after a live
  health check. Tests use fake installers and bounded synthetic probe evidence. Separately, the current
  native-Linux host has preserved seals for all three distinct tuples (the manager-1.1 flash instance
  now requires replacement before a manager-1.2 health claim);
  the first real 0.5B smoke failed before adapter insertion and completed zero optimizer steps. See
  [`HOST_STATE.md`](HOST_STATE.md); none of those facts is full-workload or offload proof. See
  [`ENVIRONMENT_MANAGER.md`](ENVIRONMENT_MANAGER.md); the full
  3-layer + MoE-safe forward plan is [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) +
  [`MOE_ARCHITECTURE.md`](MOE_ARCHITECTURE.md).
- **Static Model/Tokenizer foundation** (`model-inspect`): two versioned root contracts,
  `ModelDescriptor` and `TokenizerDescriptor`, plus deterministic JSON Schema and generated
  TypeScript types. The dependency-light inspector inventories a local snapshot without network
  access, importing torch/transformers/tokenizers, following links, or executing repository code. It
  separates requested revision from an immutable resolved commit; records file/hash/serialization
  risk and independent metadata/integrity/license/custom-code evidence; normalizes tokenizer base,
  added, effective-vocabulary, special-token, chat-template, and context metadata; and reports static
  model/tokenizer compatibility (compatible / resize required / incompatible / unverified). The model
  representation is MoE-safe: counts are scoped records, storage dtype/quantization are per component,
  semantic routing is separate from the `RunPlan` physical-execution specification, and unsupported
  topology remains unknown. A pure static parser recognizes hash-pinned Mixtral, Qwen2-MoE, DeepSeek
  V2, and DeepSeek V3 config structure; emits routed/shared/logical/active-per-token **expert-instance**
  counts; and labels runtime capability unverified. Malformed allowlisted metadata fails closed and
  MoE-like unsupported families are never guessed. This is **not** model loading, tokenizer
  training/editing, MoE execution, backend support proof, parameter-coordinate activity/residency, or
  hardware-fit proof. See [`MODEL_TOKENIZER_CONTRACTS.md`](MODEL_TOKENIZER_CONTRACTS.md) and
  [`MOE_MODEL_INSPECTION.md`](MOE_MODEL_INSPECTION.md) for the exact static evidence boundary.
- **Training objective foundation** (`training-objectives` / `training-objective-check`): a
  hash-sealed registry of 29 backend-independent definitions spanning pretraining, full/adapter SFT,
  completion/response masks, preference and reward optimization, four distillation modes,
  process/verifier/tool-use training, embedding/reranking/classification/multimodal work, and
  evaluation/merge/conversion/quantization-only operations. Every definition carries explicit
  dataset fields, label construction, masks, separately keyed loss components, model requirements,
  expected artifacts, resume/evaluation/hardware implications, and limitations. MoE-safe update
  policies can express shared/router/selected-expert/all-expert scopes, stable identity, exposure,
  optimizer clocks, starvation, and routing-collapse gates without owning physical placement. The
  compatibility report keeps dataset/model/backend evidence independent: a static manifest can earn
  only `declared_compatible`; installed packages or broad `TaskType` support never become verified
  objective support. This does **not** modify `RunPlan` or add new trainers. See
  [`TRAINING_OBJECTIVES.md`](TRAINING_OBJECTIVES.md).
- **MoE-safe parameter-accounting foundation** (`model-inspect --parameter-accounting` /
  `parameter-account`): a hash-sealed `ParameterAccountingReport` keeps logical, per-token,
  per-sequence, touched, resident, updated, exposed, and optional effective counts distinct. Every
  observation carries an exact model/coordinate scope, structured window, evidence source, coverage,
  value relation, identity basis, and resolved tied/shared/replica/quantized/state handling; unknown is
  an explicit gap, never zero. The static producer preserves descriptor declarations and performs
  bounded safetensors-header validation, promoting a header total to exact logical evidence only when
  the complete snapshot matches its recorded content hashes, identity handling is resolved, and the
  declaration agrees. Typed `RunEvent` observations can be reconciled into complete/incomplete/
  conflicting runtime reports, while allocator bytes are never converted into resident coordinates.
  `RunPlan`, `RunManifest`, `ArtifactManifest`, and `EvaluationResult` carry report refs. Existing
  trainers do not yet emit the complete measured runtime axes. Static expert-instance counts do not
  populate active/resident parameter-coordinate evidence, prove fit, or make a runtime speed claim.
  See [`PARAMETER_ACCOUNTING.md`](PARAMETER_ACCOUNTING.md).
- **Physical RunPlan foundation** (`platform-plan --physical-spec`): every newly planned run seals a
  `PhysicalExecutionSpec` containing concrete memory-tier resources, authoritative/cache/replica state
  placements, explicit offload rules, rank bindings, and parallel groups. Parameter-scoped plans pin a
  verified Phase 5 report; storage-backed plans pin and reproduce the exact `StorageProfile`
  assessment, with unsuitable targets refused and marginal/unknown risk requiring explicit acceptance.
  Static backend declarations and passing probes are both required for every non-trivial placement,
  offload, parallelism, and communication token. `platform-run` and workers recompute the canonical
  plan hash before execution. Built-in runners currently support only one explicit CPU or GPU resource;
  non-trivial plans are `PLANNED_UNPROVEN` and refused before trainer invocation. This is a planning
  boundary, not DeepSpeed/FSDP/NVMe or MoE execution proof. See
  [`RUN_PLAN_PHYSICAL_EXECUTION.md`](RUN_PLAN_PHYSICAL_EXECUTION.md).
- **Multi-backend registry with execution-contract admission**: a `BackendManifest` registry currently
  describes `corpus_studio` and `unsloth`. The first-party backend declares and proves the exact dense
  execution-contract surface. Unsloth has no Phase 9B execution-contract declaration or matching
  functional proof, so newly planned training is refused on every host rather than inferred from an
  import or static feature list. Native-Windows/WDDM Blackwell remains an additional hard refusal
  because its required math path is unavailable there.
- **Identity-bound backend worker protocol 2.0**: every newly generated RunPlan hash-pins the exact
  static BackendManifest. A subprocess worker must send `hello` first with that manifest and its exact
  environment/lock ref; only then can the core dispatch. The parent enforces protocol/direction/body,
  correlation and run IDs, unique message IDs, acceptance/order, monotonic event sequence, terminal
  lineage/outcome, and artifact linkage. Managed dispatch additionally checks descriptor/lock recipe
  identity, current recipe digest/layer/target, backend digest, live health, and drift. Legacy unpinned
  plans remain readable but must be regenerated for protocol-2 subprocess dispatch. Both public run
  entry points verify the plan seal before a runner is invoked or spawned. Workers/installers own a
  POSIX session or Windows process group and use bounded process-tree termination; the fake-worker
  suite verifies a timed-out descendant does not survive. See
  [`BACKEND_WORKER_PROTOCOL.md`](BACKEND_WORKER_PROTOCOL.md).
- **Effective execution contract (Phase 9B)**: every new first-party training plan embeds a separately
  hash-sealed `ResolvedExecutionConfiguration`. It pins exact dataset bytes, immutable model/tokenizer
  revisions or local hashes, objective/environment/capability/backend identities, per-state precision,
  quantization, the model attention API and all three PyTorch SDPA toggles, one explicit device map,
  every LoRA/optimizer/loss/schedule/checkpoint/data-format default, and the exact installed trainer
  interface. Backend admission requires one complete passing execution-combination probe plus the
  matching trainer surface; independent precision/quantization/optimizer/loss/kernel results cannot
  be unioned into support. The worker verifies and echoes the configuration hash before model
  loading, reapplies kernel toggles, consumes stabilized dataset bytes, rechecks local model/tokenizer
  roots after load, inventories singleton placement from every parameter and registered buffer, rejects
  hidden Accelerate CPU/disk offload state, and observes post-adapter placement/precision. Echo cannot execute a training
  plan. Runtime lane/`max_steps` flags are assertions only; they cannot alter the seal. Chat-template
  errors block and truncation analysis covers the complete pinned JSONL unless the plan explicitly
  permits truncation. Runner selection derives from the pinned backend manifest, successful training
  requires optimizer-step and adapter-byte evidence, every execution gets a fresh UUIDv7 run ID, and
  adapter/checkpoint output is isolated under `<output-root>/runs/<run-id>/`. Adapter IDs include the
  run, role, and weight-content hash; persisted manifests live under `<record-root>/runs/<run-id>/`.
  Legacy plans remain readable but are not executable by the training runner; regenerate them. See
  [`EFFECTIVE_EXECUTION_CONFIGURATION.md`](EFFECTIVE_EXECUTION_CONFIGURATION.md).
- **Reliability**: an in-process watchdog detects a stall/spill + captures a measured fit; the
  subprocess worker can **KILL a hung run** (→ `KERNEL_STALL`) and isolates a crash. The pre-Phase-9B
  lifecycle ran end to end on a real RTX 5070 under native Windows/WDDM, in-process and subprocess.
  That historical run does not verify the new effective-execution enforcement and is not bare-Linux,
  NVMe/offload, full-sequence 7B, or MoE-runtime proof.
- Consumed by a new **Tauri 2 + React** contract-first client (`apps/web`) alongside the WPF + Avalonia
  heads.
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
  caution), and the **Studio** (Dashboard, Writing Studio, Examples,
  Preference Review, Quarantine, Splits, Evaluation, AI Assist, Training, Arena,
  Artifacts, **Suites**, Versions, Debt, Settings) — a flat 15-tab strip in the
  shipping WPF head, re-skinned to the **Nocturne** grouped workflow-phase sidebar
  (Overview · Author · Measure · Evaluate · Train) on the cross-platform Avalonia
  shell (see [`design/`](design/)). Both New Project entry points open the one
  wizard. See [`WORKSPACE_SYSTEM.md`](WORKSPACE_SYSTEM.md).
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
- Architecture: the desktop view-model has been decomposed from a single god-object
  into per-tab view-models behind interfaces, composed via a DI container
  (`Microsoft.Extensions.DependencyInjection`) with a shared `ViewModelBase`; **all tabs are
  extracted** (18 `IXxxViewModel` + `XxxViewModel`, incl. Training + Evaluation, the Quality
  panel, and the connection/rewrite-batch sub-VMs). Dashboard stays a composition view over the
  extracted children. Run-orchestration is being consolidated off the per-head code-behind into
  the VM as testable commands behind an **`IEngineService`** seam (issue #184): code-behind
  `_Click` handlers are down from 108 to ~59, with ~55 `ICommand`s.
- Cross-platform (Avalonia) migration — **Phases 0–3 done; `ICommand` conversion in progress**
  (not shipped; WPF stays the product head). Head-agnostic seams live on the view-models —
  `IEngineService`, `IDialogService`, `IFilePickerService` (each with a Core `Null*` default and
  WPF/Avalonia DI adapters) — plus a cross-platform venv-path fix; all Models + view-models +
  WPF-free services live in a shared **`CorpusStudio.Core`** (`net8.0`) library; and the proof
  **`CorpusStudio.Avalonia`** head re-authors the *whole* app as `.axaml` over those unchanged
  view-models with compiled bindings (the GO/NO-GO spike passed, then grew to the full shell). See
  [`AVALONIA_MIGRATION_PLAN.md`](AVALONIA_MIGRATION_PLAN.md).

## Hard boundaries (by design)

- The engine **control plane** is dependency-light and is **not** a deep-learning framework
  (no backprop/optimizers/distributed training of its own). CUDA/PyTorch/Transformers live in an
  opt-in `[train]` worker runtime. The authoritative first-party path is sealed
  `platform-plan` -> supervised `platform-run`; the low-level `train-run` compatibility command is
  refused by default and labeled non-reproducible when explicitly enabled. See
  [`TRAINING.md`](TRAINING.md).
- External trainer launches show exact argv and require confirmation. First-party launches bind an
  exact backend/runner/environment/input/effective-configuration chain, use no shell, and write
  inspectable per-run metadata. No hidden lane switching.
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
- **Tokenizer training/editing and isolated functional probes.** Exact per-model counts already work
  when the optional `[model-tokenizer]` (`tokenizers`) extra can load the target tokenizer; tiktoken
  and the documented Unicode heuristic remain fallbacks. The static descriptor inspector deliberately
  does not execute a tokenizer, and the dependency-light core never makes a heavy tokenizer mandatory.
- **HF export/push** (upload/publishing) — see the hard boundary above; it stays a
  deliberate non-goal for now. (Read-only Hub *import* already ships.)
- **Finish the Avalonia port** — Phases 0–3 are done (all view-models extracted; the whole app
  re-authored as `.axaml` over the shared `CorpusStudio.Core`); the **`ICommand` conversion is in
  progress** (WPF code-behind engine handlers → shared testable commands behind `IEngineService`),
  with the process-streaming/timer/undo-state handlers and Fluent-theme styling + per-OS packaging
  still to do. The Avalonia head is not shipped yet. See `AVALONIA_MIGRATION_PLAN.md`.
- Dataset-version **reorder detection** and a normalized row identity are still future.

  _Previously listed here but now **shipped** (see `CLI_REFERENCE.md`): row-store garbage collection
  (`dataset-version-gc`, fail-closed, `--dry-run`), opt-in export PII/secret redaction
  (`export --redact-pii`, with a redaction manifest — known patterns only, not de-identification), the
  desktop per-project gate-threshold editor (`gate-thresholds` read + `gate-thresholds-set` validated
  write), and the validator's recursive **lists-of-objects** checking (`SchemaField.item_fields`)._
- Smaller deferrals: an
  app icon. (CI hardening — ruff, mypy, pytest gate, dependabot, and CodeQL — is
  in place.)
