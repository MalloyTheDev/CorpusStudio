# Dataset Version History & Lineage (v1.0)

A **dataset version** is a durable, inspectable *lineage anchor*: it records the
identity of a project's dataset at a moment in time and pins the artifacts that
co-existed with it. It answers "**which exact dataset state produced this model /
evaluation?**" and "**has the dataset changed since?**" — without copying the
dataset. It is the dataset-side mirror of the training-run and model-artifact
registries.

## What a version stores

Records live at `dataset_versions/{version_id}.json` (one atomic JSON per
version, newest-first because `version_id` is timestamp-prefixed) — beside
`training_runs/` and `model_artifacts/`.

| Field | Meaning |
|---|---|
| `version_id` | `YYYYMMDDThhmmss-<suffix>`, validated `^[A-Za-z0-9._-]+$` (injective filename). |
| `created_at` / `updated_at` | ISO-8601 timestamps. |
| `label` / `trigger` | User label; what produced it (`manual`, `manual_add`, `import_commit`, `pre_training`). |
| `row_count` | Rows in `examples.jsonl` at capture time. |
| `content_fingerprint` | 64-char SHA-256 over the **ordered** per-row exact signatures, or `null` if the dataset was missing/unreadable. |
| `fingerprint_algo` | `sha256-ordered-exact-v1` — versioned so a future identity scheme is additive, never a silent reinterpretation. |
| `row_signature_kind` | `exact` (reserved: `normalized` later). |
| `source_run_ids` / `artifact_ids` | Pinned links to training runs / model artifacts. |
| `eval_report_path` / `gate_report_path` | Pinned links to an evaluation report (absolute path) / a dataset-scope gate report. |
| `notes` | Free text. |

**Nothing derivable is stored.** Eval scores, base model, artifact integrity,
and gate status are resolved *live* when the version card is rendered, so a
record can never drift from the state it points at (the same discipline as the
weight card).

### The fingerprint

`content_fingerprint` is a streaming SHA-256 fed the canonical per-row signature
(`exact_row_signature` — `json.dumps(row, sort_keys=True, separators=(',',':'))`,
the *same* primitive used by cleaning/quality/leakage) joined by newlines. It is
deliberately **order-sensitive**: rows have no stable identity today, so a
version means "these exact rows in this exact order." Reordering, adding,
removing, or editing any row changes the fingerprint. It streams in O(1) memory
and never hashes anything but the dataset rows; a missing/unreadable dataset
yields `null` rather than a false alarm.

## Live integrity (`matches` / `drifted` / `unreadable`)

Listing and the card recompute the fingerprint of the *current* `examples.jsonl`
and compare it to the stored one:

- **matches** — the dataset is exactly the state this version recorded.
- **drifted** — the dataset has changed since; linked runs/evals may describe a
  different state (the card leads with this warning).
- **unreadable** — the dataset is missing/unreadable, or no fingerprint was
  recorded.

## CLI

```
# Capture a version (fingerprint + row count of examples.jsonl, with links)
python -m corpus_studio.cli dataset-version-create <project-dir> \
  --label "before cleaning" --trigger pre_training \
  --link-run <run_id> --link-artifact <artifact_id> \
  [--eval-report-path <abs>] [--gate-report-path <path>] [--stamp-run <run_id>]

# List versions (newest first), each annotated with current integrity
python -m corpus_studio.cli dataset-version-list <project-dir>

# Render the live version card (Markdown, or --json for the resolved card)
python -m corpus_studio.cli dataset-version-show <project-dir> --version-id <id> [--json]
```

`dataset-version-create` computes the fingerprint + row count from the project's
`examples.jsonl`, auto-links the newest dataset-scope gate report already on disk
(deterministic — it never *runs* a gate), and with `--stamp-run` writes
`source_snapshot_id=<version_id>` onto that run so the dataset→run link closes in
both directions.

## Hard boundaries

The engine only **reads** `examples.jsonl` and **writes** JSON under
`dataset_versions/`. It never moves, copies, or deletes the dataset or any weight
file; it runs no ML and makes no network calls. Capture is explicit/opt-in, taken
when the dataset is quiescent (after an import/append commits), never as a side
effect.

## Implemented vs deferred

**Implemented (v1.0.0, engine):** the version record + registry
(`versions/version_registry.py`), the streaming fingerprint, live integrity, the
live version card (`versions/version_card.py`), the `source_snapshot_id`
run back-link, and the three CLI commands.

**Implemented (v1.0.1, desktop):** a **Versions** tab surfacing the history —
a read-only list with a live integrity badge (✅ matches / ⚠ drifted /
⛔ unreadable), a one-line summary, a **Capture version** button (opt-in, with an
optional label), and **View card** (the rendered version card). Both capture and
list go **through the engine** (`dataset-version-create` / `-list`), so the
desktop never recomputes the fingerprint and integrity is verified, not guessed.

**Deferred:**
- Auto-capture after an import/append commit (a `trigger` other than `manual`).
- Stable per-row identity + a content-addressed row store (v1.0.2) — the
  prerequisite for the below.
- Version **diff** (added/removed/modified/moved rows) (v1.0.3).
- **Restore-to-version** — the only operation that rewrites `examples.jsonl`,
  deferred last so it respects the append-only write contract (v1.0.4).
