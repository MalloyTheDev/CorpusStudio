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

On top of that loop, v1.2.1–v1.2.15 added an **IDE-like workspace shell** (Start
Center, Universal Explorer, Problems + Output panels, one New Project wizard) and
desktop polish; a deep-review pass shipped an **opt-in LLM-judge evaluation
scorer**, a **crash-safe / self-contained distributable** build, an **Avalonia
cross-platform assessment**, the **view-model decomposition** so far (a shared
`ViewModelBase` + the Debt and Arena tabs extracted behind interfaces + DI), a
**unified streaming JSONL reader** (soft-`orjson` accelerated, off-thread document
opens), and **backend resilience** (retry/backoff + per-item error isolation). A
subsequent **deep bug/security audit** hardened 19 data-integrity, gate/policy,
and quality/split-correctness issues (PRs #104–118).

Milestones v0.1–v1.2 are complete. CUDA, PyTorch, Transformers, and cloud
publishing remain deliberately outside the app — Corpus Studio orchestrates the
user's installed trainer, it does not embed a training framework.

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

## Next

- **Finish the evaluation judge in the UI** — a desktop Evaluation-tab "Judge
  model" field wiring the engine's `--judge-model` scorer through `PythonEngineService`.
- **v1.3 — Evaluation Suites & Chat Gates** (not started): reusable evaluation
  suites and chat-scope gates.
- **A real tokenizer** (transformers/tokenizers) so token-budget / VRAM numbers are
  exact rather than heuristic.
- **Hugging Face Hub export/push** (upload/publishing) — stays a deliberate
  non-goal for now; read-only Hub *import* already ships.
- **Continue the view-model decomposition** beyond the Debt and Arena tabs;
  eventually an **Avalonia** port for macOS/Linux (see
  [`CROSS_PLATFORM_ASSESSMENT.md`](CROSS_PLATFORM_ASSESSMENT.md)).
- Smaller: dataset-version auto-capture after import, reorder detection, row-store
  GC (never prune manifest-referenced rows), PII auto-redaction, and a per-project
  gate-threshold editor in the desktop.
