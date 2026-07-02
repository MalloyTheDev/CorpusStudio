# Current State

Single source of truth for what Corpus Studio actually does today. When another
doc disagrees with this file, this file wins (and the other doc should be fixed).

Last reconciled: 2026-07-02 (v0.6 in progress).

## What works today (implemented and tested)

**Author & validate**
- Local project creation from built-in schema templates with pre-filled examples.
- Schema validation through the Python engine: required non-empty fields,
  declared field types, list element types, enum/label sets, numeric bounds,
  nested object shapes, and chat message structure, with selectable issue
  navigation in the desktop.
- JSONL import preview with failed-row quarantine, review, and retry.
- Full Unicode correctness end to end: NFKC-aware tokenization and UTF-8 stdio
  so CJK/Cyrillic/accented text round-trips between the desktop and engine.

**Clean & measure**
- Quality report: empty rows, exact + normalized duplicates, low-information
  rows, synthetic-pattern warnings with near-duplicate clustering, PII/secret
  detection (emails, SSNs, private keys, AWS/API keys, JWTs, Luhn-valid cards —
  masked samples), token-length outliers, and category-imbalance warnings, with
  project-level quality history.
- Leakage-checked splits: `detect_split_leakage` reports exact and
  near-duplicate rows shared across train/validation/test.
- Export with an optional cleaning pass (dedupe / drop low-information) that
  writes a removal manifest; verbatim exports warn when duplicates remain.
- Preference exports (DPO/KTO/reward) with a pair-integrity gate.
- Dataset card summarizing metadata, schema, splits, quality, and evaluation.

**Evaluate & assist**
- Evaluation Lab against local Ollama or OpenAI-compatible endpoints: health
  checks, model discovery, report history, two-report comparison, regression
  reruns, tag/failure/score-band summaries, failed-row edit loops, manual
  scoring, saved failure filters.
- Multi-model benchmark: one dataset across several models, ranked, with
  per-model deltas and the examples every model failed.
- Review-first AI Assist Lab: persistent accept/reject queue, saved views, bulk
  triage with undo, resumable rewrite batches. AI Assist output is always
  `review_required` and never auto-accepted.

**Train (v0.5 launcher — complete)**
- Training config export for axolotl / TRL / Unsloth / Hugging Face /
  LLaMA-Factory with compatibility warnings, a real token budget
  (tokens-per-epoch after truncation, over-length counts), a rough arithmetic
  VRAM planning estimate (never inspects hardware), and a LoRA rank/alpha
  suggestion.
- In-app launch of the user's installed trainer with explicit confirmation of
  the exact argv (no shell), live log streaming, and a Stop that kills the
  process tree.
- Checkpoint tracking during and after runs, resume-from-latest for targets
  with a CLI resume flag, and before/after evaluation comparison against the
  baseline captured at launch.

## Hard boundaries (by design)

- Corpus Studio orchestrates the user's installed tools; it is **not** a deep
  learning framework. No CUDA, PyTorch internals, backprop, optimizers,
  distributed training, or custom training loops.
- Trainer launches show the exact argv, require explicit confirmation, use no
  shell, and write inspectable run metadata. No hidden trainer behavior.
- No silent cloud behavior, publishing, dataset upload, or auto-acceptance of
  AI-generated dataset rows.

## In progress — v0.6 (Provider Policy + Gate Foundation)

- Role-based provider/model capability policy, enforced in the engine:
  OpenAI/Anthropic evaluator-only by default; Ollama/local generation only when
  explicitly approved; OpenRouter route-aware. Surfaced by a **Provider
  Generation Policy** panel in the desktop Settings tab (shows which providers
  may generate, and approve/revoke a local model). See
  [`PROVIDER_POLICY.md`](PROVIDER_POLICY.md).
- A gate runner producing serializable pass/warn/block reports over existing
  schema, quality, leakage, PII, and evaluation logic, surfaced by a **Run Gates**
  button in the desktop Quality tab (overall status + per-gate pass/warn/block
  with repair hints). See [`GATES.md`](GATES.md).

## In progress — v0.7 (Model Chat Lab / Arena)

- Run a prompt suite across several models and capture responses side by side
  (engine `arena-run` → `ArenaReport`). Responses are comparison artifacts, not
  trainable rows. Judging, saved reports, and a desktop surface are next.

## Not built yet (future roadmap)

Model Chat Lab / Arena (v0.7), Training Run Registry (v0.8), Model Artifact /
Weight Registry (v0.9), Dataset Version History & Lineage (v1.0), Dataset Debt
Dashboard (v1.1), Approved Provider Generation into a review queue (v1.2),
Evaluation Suites & Chat Gates (v1.3). Regression gates depend on the
before/after registry (v0.8) and are documented as future work.
