# Corpus Studio Roadmap

Corpus Studio is a one-stop workspace for creating, validating, testing,
exporting, and launching training on datasets — staged so each release stays
focused and usable. For the authoritative list of what works today, see
[`CURRENT_STATE.md`](CURRENT_STATE.md); this file is the milestone history and the
forward plan.

## Current checkpoint

The full local dataset-to-model loop is shipped end to end: authoring/validation,
import quarantine, quality reports + a graded debt ledger, leakage-checked splits,
JSONL/preference export, Evaluation Lab, multi-model benchmark, Model Arena,
review-first AI Assist (with pre-review candidate gating), a governed provider
policy + gate runner, the local training launcher (in-app launch, live logs,
checkpoints, resume, before/after eval), a training run registry + regression
gate, a model artifact registry + weight card + promote gate, and dataset version
history (capture/card/diff/restore) — all local-first and file-backed.

The most recent re-scope adds a **headless platform run lifecycle**: a language-neutral contracts
substrate (RunPlan / RunEvent / BackendManifest) that turns goal + data + hardware into a validated,
reproducible run — profile → plan → predict-fit → run → measure-fit → artifacts — with a registered
backend-manifest catalog (`corpus_studio`, `unsloth`; only the first-party backend currently declares
and proves the complete Phase 9B execution contract),
a **calibrator + watchdog** (predicted-vs-measured fit, spill/stall detection), and a **supervised
subprocess worker** that can KILL a hung run. Worker protocol 2.0 now binds the exact backend-manifest
digest and environment/lock ref before dispatch, then fail-closes message order and terminal lineage.
It is exercised by a **Tauri 2 + React** contract-first client (`apps/web`) alongside the WPF
(shipping) and Avalonia (interim) heads. A pre-Phase-9B lifecycle ran end to end on a real RTX 5070
(Blackwell/sm_120) under native Windows/WDDM, including a real GPU QLoRA run. That historical run does
not verify the new effective-execution contract. On the current native-Linux host, the managed
`backend-corpus-studio` environment separately passed its minimal CUDA-allocation, 4-bit-construction,
forward/backward, and math-SDPA probe. That environment result is not a native-Linux real workload,
full-sequence 7B, real-workload FlashAttention, or offload result; those remain unverified.

On top of that loop, v1.2.1–v1.2.15 added an **IDE-like workspace shell** (Start
Center, Universal Explorer, Problems + Output panels, one New Project wizard) and
desktop polish; a deep-review pass shipped an **opt-in LLM-judge evaluation
scorer**, a **crash-safe / self-contained distributable** build, an **Avalonia
cross-platform assessment** and the **view-model decomposition** (a shared
`ViewModelBase` + all tabs extracted behind interfaces + DI, then run-orchestration consolidated
into the VMs as testable commands behind an `IEngineService` seam), a
**unified streaming JSONL reader** (soft-`orjson` accelerated, off-thread document
opens), and **backend resilience** (retry/backoff + per-item error isolation). A
subsequent **deep bug/security audit** hardened 19 data-integrity, gate/policy,
and quality/split-correctness issues (PRs #104–118).

Milestones v0.1–v1.2 are complete. The dependency-light **core** pulls no CUDA /
PyTorch / Transformers; those live in an **opt-in `[train]` worker runtime** that adds a
first-party TRL/peft QLoRA trainer + adapter merge + model download. Authoritative
first-party runs use `platform-plan` / `platform-run`; `train-merge` and `model-fetch`
remain supporting commands alongside the original bring-your-own-trainer path.
Cloud publishing remains out of scope.

## Shipped milestones (v0.1 → v1.2)

Each milestone proved one layer of the loop; `CURRENT_STATE.md` documents the
resulting features in full.

- **v0.1 — Dataset Creation Studio.** Local projects from schema templates,
  editors, schema validation, JSONL import (quarantine review/retry) and export,
  train/validation/test splitting, basic quality checks, quality history.
- **v0.2 — Evaluation Lab.** Run datasets against local Ollama / OpenAI-compatible
  models: health checks, model discovery, JSON reports + history + two-report
  comparison, regression reruns, tag/failure/score-band summaries, failed-row edit
  loops, manual scoring, saved failure filters, versioned reviewed-fix tracking.
- **v0.3 — AI Assist Lab.** Review-first accept/reject queue with saved views,
  bulk triage + undo, resumable rewrite batches, synthetic-pattern + preference
  warnings; output is always `review_required`, never auto-accepted.
- **v0.4 — Training Config Generation.** Inspectable configs for Axolotl / TRL /
  Unsloth / HF Trainer / LLaMA-Factory with compatibility warnings, a real token
  budget, a rough VRAM estimate (never inspects hardware), and a LoRA r/alpha helper.
- **v0.5 — Local Training Launcher.** In-app launch of the user's trainer (exact
  argv, explicit confirm, no shell), live log streaming, checkpoint tracking,
  resume-from-latest, a Stop that kills the process tree, before/after eval vs a
  baseline captured at launch.
- **v0.6 — Provider Policy + Gate Foundation.** Role-based provider capability
  policy enforced **in the engine** (OpenAI/Anthropic evaluator-only; local
  generation only when approved; OpenRouter route-aware) + a gate runner with
  serializable pass/warn/block reports and per-project thresholds.
- **v0.7 — Model Arena.** Run a prompt suite across models side by side; responses
  are comparison artifacts, not trainable rows. Optional evaluator-only judging
  (per-model win counts + average judge score). Reports under `arena_reports/`.
- **v0.8 — Training Run Registry.** Durable per-run records under `training_runs/`
  (argv/config/output/status/checkpoints/eval links; `running`→`interrupted`
  reconcile on load) + the `training_run` regression gate with provenance checks.
- **v0.9 — Model Artifact Registry.** Path-referenced artifact records under
  `model_artifacts/` with keep/reject; **path integrity** re-checked on load (cheap
  size+mtime on the list, byte-exact SHA-256 at the weight card + promote gate) so a
  record can never point at altered weights; a live weight card + a promote gate.
- **v1.0 — Full workflow + version history.** Dataset version registry with a
  content fingerprint, live drift detection, lineage links, a version card, a
  content-addressed row store, `dataset-version-diff`/`-restore`, and a desktop
  Versions tab. See [`VERSIONING.md`](VERSIONING.md).
- **v1.1 — Dataset Debt Ledger.** Quality signals normalized by size, ranked,
  graded A–F, each with a remediation (secrets/PII by presence, else by rate) + a
  desktop Debt tab, plus a dashboard grade badge and a debt-trend mini-chart. See
  [`DEBT.md`](DEBT.md).
- **v1.2 — Approved Provider Generation (candidate gating).** `run_ai_assist` runs
  the dataset gate runner over generated candidates and attaches the verdict as
  `candidate_gate` — a pre-review signal only (v1.2.1 surfaces it in the desktop
  with confirm-on-block). See [`AI_ASSIST_LAB.md`](AI_ASSIST_LAB.md).
- **v1.3 — Evaluation Suites & Chat Gates.** Named multi-case evaluation suites with a
  per-metric verdict and optional `version_id`-pinned cases (engine + `suite-*` CLI +
  desktop **Suites** tab); a conversation-structure chat gate (`chat-gate` + desktop
  button). Plus auto-capture of a dataset version after an import commit. See
  [`EVALUATION_SUITES.md`](EVALUATION_SUITES.md) and [`GATES.md`](GATES.md).

## Next

**Platform frontier** (the full local-first AI lifecycle — see [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)).
The run lifecycle (profile→plan→fit→run→artifact) is built. The exact minimal managed-environment
hardware probe is verified on the current host; the Phase-9B real workload is not. The frontier is the
input side. Ordered:

- ✅ **StorageProfile** — dependency-light storage topology + per-role safe-spill suitability (offload
  onto USB / cloud-sync / near-full / in-repo / rotational is refused). See
  [`HARDWARE_STORAGE_PROFILE.md`](HARDWARE_STORAGE_PROFILE.md).
- ✅ **Environment Manager + 3-layer dependency profiles** — the reference `backend-corpus-studio`
  lifecycle now covers runtime discovery, sealed preview/confirmation, isolated creation, durable
  command journals, package/source/hash locks, separate functional and hardware proof, drift, safe
  remove/recreate, and immutable RunPlan association. Other frameworks still get independent,
  verified worker environments — never one `[everything]`. See
  [`ENVIRONMENT_MANAGER.md`](ENVIRONMENT_MANAGER.md).
- ✅ **Model/Tokenizer contract + static-inspection foundation** — `ModelDescriptor` +
  `TokenizerDescriptor` are MoE-safe from their first version, with offline inventory, fail-closed
  custom-code findings, tokenizer vocabulary/compatibility evidence, JSON Schema, and generated TS
  types. An allowlisted Phase 8 parser now adds hash-pinned Mixtral/Qwen2-MoE/DeepSeek V2/V3
  structural expert-instance evidence. This does not load/train/edit models or tokenizers, prove
  backend support, populate active/resident parameter coordinates, or establish MoE runtime
  capability. See [`MODEL_TOKENIZER_CONTRACTS.md`](MODEL_TOKENIZER_CONTRACTS.md) and
  [`MOE_MODEL_INSPECTION.md`](MOE_MODEL_INSPECTION.md).
- ✅ **TrainingObjective contract + registry foundation** — 29 hash-sealed, backend-independent
  definitions with dataset/label/mask/loss semantics, MoE-safe update scope/exposure rules, artifacts,
  resume/eval/hardware implications, and conservative dataset/model/backend compatibility axes. This
  does not yet wire an objective into `RunPlan` or add trainer implementations. See
  [`TRAINING_OBJECTIVES.md`](TRAINING_OBJECTIVES.md).
- ✅ **Parameter-accounting evidence foundation** — sealed dense/MoE-safe reports, bounded static
  descriptor/safetensors evidence, typed runtime observations, strict reconciliation, explicit
  gaps/conflicts, and lifecycle refs. Backend workers still need real coordinate instrumentation. See
  [`PARAMETER_ACCOUNTING.md`](PARAMETER_ACCOUNTING.md).
- ✅ **Offload/placement/parallelism `RunPlan` contract + planner foundation** — concrete resources,
  state placement, offload rules, ranks/groups, evidence pins, capability gates, and tamper checks are
  explicit. Built-in workers remain singleton-only and refuse non-trivial execution. See
  [`RUN_PLAN_PHYSICAL_EXECUTION.md`](RUN_PLAN_PHYSICAL_EXECUTION.md).
- ✅ **Generalized `TraceRecord` + Trace Studio engine/authoring foundation** — versioned source,
  context, reasoning/tool/final-answer segments, producer-policy evidence, typed validation, explicit
  immutable review, legacy migration, atomic generation reports, trainer approval gating, generated
  JSON Schema/TypeScript, and a desktop-selectable trace draft schema. A dedicated graphical Trace
  Studio and tool/process trainers remain future work. See [`TRACE_RECORDS.md`](TRACE_RECORDS.md).
- ✅ **Static MoE model inspection** — allowlisted, hash-pinned topology evidence only.
- ✅ **Backend worker protocol 2.0 + fake-worker conformance foundation** — worker-first
  backend/environment identity, strict typed messages, correlation/order/lineage enforcement, and
  managed recipe-target/lock checks, public-entry plan-seal verification, and bounded worker-tree
  termination. This is not a real new training backend. See
  [`BACKEND_WORKER_PROTOCOL.md`](BACKEND_WORKER_PROTOCOL.md).
- ✅ **Effective execution truth** — newly generated first-party plans now seal immutable inputs,
  per-state precision, exact attention toggles, an explicit device map, all semantic trainer defaults,
  and the exact trainer interface. The worker verifies and echoes that configuration before model
  load and refuses drift; legacy plans require regeneration. This is not a new hardware-verification
  claim. See [`EFFECTIVE_EXECUTION_CONFIGURATION.md`](EFFECTIVE_EXECUTION_CONFIGURATION.md).
- **Next: artifact and environment integrity hardening.** Share the model inspector's safe inventory
  across artifacts/checkpoints/environment inputs, verify installed files against `RECORD`, install a
  reviewed content-hashed wheel, add inter-process lifecycle locks, and make environment replacement
  blue/green. Then add another dense backend only with its own exact contract and functional proof.

- **Surface the LLM judge in the Evaluation tab** — the `--judge-model` scorer ships in
  the engine and in suites, but the desktop Evaluation tab still has no judge-model field.
- **Tokenizer training/editing + isolated functional probes.** Optional target-model `tokenizers`
  and tiktoken counting already exist; the dependency-light core retains a documented heuristic
  fallback, while static descriptors deliberately make no encode/decode claim.
- **Hugging Face Hub export/push** (upload/publishing) — stays a deliberate
  non-goal for now; read-only Hub *import* already ships.
- **Finish the Avalonia cross-platform port.** Phases 0–3 are **done** — a shared
  `CorpusStudio.Core` (`net8.0`) holds all the extracted view-models behind interfaces, and the
  Avalonia head re-authors the whole app as `.axaml` over them (compiled bindings, same DI). The
  **`ICommand` conversion is in progress** (WPF code-behind engine handlers → shared testable
  commands behind `IEngineService`/`IDialogService`/`IFilePickerService`); the remaining handlers
  need a process-streaming seam, timer decoupling, and undo-state migration, after which Fluent-theme
  styling and per-OS packaging follow. See [`AVALONIA_MIGRATION_PLAN.md`](AVALONIA_MIGRATION_PLAN.md)
  and [`CROSS_PLATFORM_ASSESSMENT.md`](CROSS_PLATFORM_ASSESSMENT.md).
- Smaller: dataset-version reorder detection and a normalized row identity. (Row-store GC,
  PII redaction on export, and the desktop gate-threshold editor now ship.)
