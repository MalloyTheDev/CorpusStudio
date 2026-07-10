# Gates

A **gate** decides whether a dataset, split, export, or evaluation report may
move forward. Gates reuse the existing validation, quality, leakage, PII, and
evaluation logic — they only aggregate results against thresholds and produce a
serializable, project-local report. No gate adds new detection logic.

## Model

- **Scopes**: `dataset`, `row`, `import`, `export`, `split`,
  `evaluation_report`, `training_run`, `model_artifact`, `chat_suite`.
- **Status**: `pass`, `warn`, `block`.
- **`GateResult`** fields: `gate_id`, `name`, `scope`, `status`, `observed`,
  `expected`, `affected`, `message`, `repair`.
- **`GateReport`**: `scope`, `target`, `generated_at`, `overall_status`
  (= worst result), pass/warn/block counts, and `results`. Serializes to JSON
  and reloads via Pydantic.

## Initial gates (wired to existing logic)

| Gate | Reuses | Default behavior |
|---|---|---|
| **schema** | `validate_jsonl_row` | **block** if any row fails validation. |
| **quality** | `build_basic_quality_report` | **block** on exact duplicates; **warn** on near-duplicates, low-information, or synthetic-pattern issues. |
| **leakage** | `detect_split_leakage` | **block** if any row is shared across train/validation/test. |
| **pii** | quality report `pii_findings` | **block** on high-severity (keys/tokens/JWT); **warn** on medium (email/SSN). |
| **eval_score** | `EvaluationReport` | **block** below the average-score or pass-rate threshold. |

The **export gate** is a composite: it **blocks** on empty input, schema, or PII
failure, and **warns** on quality issues (duplicates/low-information) because the
export command has a dedicated cleaning pass. An `input_present` gate ensures an
empty dataset can never pass silently (warn for `dataset` scope, block for
`export`). Thresholds ship as sensible defaults in `GateThresholds` and are
designed for future per-project configuration.

## Running gates

```
python -m corpus_studio.cli gate-run dataset.jsonl instruction --scope dataset \
  --project-dir path/to/project
```

Writes `gate_reports/<scope>-<target>.json` under the project (the target is in
the filename so gating different files in one scope does not clobber earlier
reports) and echoes the report.
`--scope export` runs the export gate. Split and evaluation gates are available
through the engine API (`run_split_gate`, `run_evaluation_gate`).

## Regression gate (training_run scope)

`training-run-gate --project-dir <p> --run-id <id>` reads a training run
record's linked before/after evaluation reports and **blocks** when the trained
model's average score dropped more than `GateThresholds.max_regression_score_drop`
(default 2.0), **passes** on hold/improve, and **warns** with *unverified
linkage* when the after-eval targeted the base model (or no model id was linked)
— because a before/after comparison is only trustworthy if the after-eval ran
against the trained model. Surfaced by a "Gate run" button in the Training tab
that links the newest trained-model eval and runs the gate.

## Promote gate (model_artifact scope)

`artifact-gate --project-dir <p> --artifact-id <id>` is the enforcement point for
keeping a model artifact. It **blocks** when the artifact integrity is `missing`
or `modified` (the weights changed/vanished since evaluation) **or** the source
run's regression gate blocks; it **warns** on unverified linkage or a missing
source run; otherwise **passes**. In the desktop Artifacts tab, "Keep" runs this
gate first and a block refuses the keep. The companion `artifact-card` renders a
live weight card (never stored) carrying the same provenance caveat.

## AI Assist candidate gate (ai_assist_candidates)

`run_ai_assist` runs `run_dataset_gates` over the **generated candidate rows**
(`target="ai_assist_candidates"`) and attaches the `GateReport` to its result as
`candidate_gate`, so an AI-generated batch carries a schema/quality/PII verdict
**before** it reaches the human review queue. This reuses the dataset gate runner
verbatim — no new detection. It is a *pre-review signal, not a decision*:
`review_required` stays true, a clean gate is not approval, and a block does not
auto-reject (the candidate is preserved for the human, block-first). `candidate_gate`
is `null` when a run proposes no gate-able (JSON-object) rows; a batch that proposed
content but no object rows still surfaces those via `validation_errors` plus an
explicit "gate not run" warning. Provider policy is enforced *before* the
provider call, so the gate step can never run on a generation a forbidden provider
was not allowed to perform. See [`AI_ASSIST_LAB.md`](AI_ASSIST_LAB.md).

## Chat gates (chat_suite scope)

`chat-gate <path> [--schema chat] [--project-dir <p>]` gates a chat dataset's
**conversation structure** — the sequence-level shape the per-row validator can't
see. It runs input-present + per-message schema validation + a `chat_structure`
gate over each row's `messages` list, and reports at `chat_suite` scope. It
verdicts **structure, not semantic quality** — it never claims a conversation is
*good*, only *well-formed*.

The `chat_structure` gate flags conversations that: start with an assistant turn
(after any system), have no user or no assistant turn, end on a user turn
(dangling), repeat a role back-to-back (tool-aware — `user→assistant→tool→assistant`
is fine), have more than one system message or a system message not at the start,
or fall outside the turn-count bounds. The per-message shape (valid role, non-empty
content) is **not** re-checked here — that is the schema/validator's job.

Everything **warns** by default (schema validation already blocks truly-invalid
chat). Set `block_chat_malformed: true` to make the training-breaking faults
(assistant-first, no-user, no-assistant, dangling-user) a hard **block**; the
stylistic ones (role repetition, system placement, turn count) stay warnings.
`chat-gate` is **advisory** — it prints/saves the report but exits `0`; the verdict
is in the report. Rows without a `messages` list are skipped (a missing field is the
schema gate's concern), and a dataset with no conversations warns rather than
faking a pass.

## Provenance gate (per-row licensing)

`provenance-gate <path> [--teacher-field meta.teacher] [--strict] [--allow-teacher <name>] [--project-dir <p>]`
reads each row's declared **teacher** — the model/provider that *generated* the row — and
buckets it: **quarantined** (a known restricted provider whose terms forbid training on its
outputs, e.g. Anthropic/OpenAI), **pass** (a recognized open/local provider, or a teacher the
user allow-listed), or **unknown** (untagged / unrecognized → *quarantine-until-verified*). The
verdict **blocks** on any quarantined row (and, under `--strict`, any unknown); unknown rows
otherwise **warn**. It reuses the provider policy (`resolve_policy`) — the licensing counterpart
to `provider-policy` (which gates generation-*time*) and `run-provenance` (a run fingerprint);
neither of those walks a dataset per row. A project `provenance_allowlist.json` (or repeated
`--allow-teacher`) declares an open/MIT teacher trainable-clean.

Standalone it is **advisory** (exit 0, verdict in the report + a human table on stderr). It
becomes an **enforcement point** via `export --check-provenance`, which **refuses the export
(exit 2)** if any row is quarantined — `--provenance-strict` also blocks unknown provenance —
exactly like the PII export gate refuses an export with unmasked secrets.

**Honesty**: the verdict trusts each row's *declared* teacher — a mislabeled or omitted teacher
is not caught by content, `unknown ≠ safe` (it is quarantine-until-verified), and a `pass`
reflects a declared/allow-listed license, not a proof of origin.

## Per-project thresholds

Gate thresholds default to the values in `GateThresholds`, but a project can
override any of them by writing a `gate_thresholds.json` in the project
directory. Overrides are **partial** (unlisted keys keep their default) and
**fail-closed** per key: unknown keys are ignored, and any single value that is
out of range, negative, or non-finite (`NaN`/`inf`) falls back to *that key's*
strict default while the file's other valid overrides still apply — one bad key
never silently discards the rest. A file that is entirely unreadable, not JSON,
or not a JSON object falls back to the strict defaults wholesale. Every field is
bounded (counts and scores are non-negative; the pass-rate is a fraction in
`[0, 1]`), and the file is read as UTF-8 **with a tolerated BOM**, so overrides
saved by Notepad or PowerShell still apply.

Near-duplicate and low-information rows **warn** by default. To make either a
hard block for a project, set `block_normalized_duplicates` / `block_low_information`
to `true` (mirroring `block_exact_duplicates`). These block only at dataset/row
scope; the **export** gate always warns on quality counts (it has a dedicated
cleaning pass) and blocks only on empty input, schema, or PII.

Chat structure (chat_suite scope) has three thresholds: `min_chat_turns` (default
2) and `max_chat_turns` (default 0 = no maximum) bound the message count, and
`block_chat_malformed` (default `false`) turns the training-breaking faults into
blocks as described above.

`training-run-gate` and `artifact-gate` (which always take a project directory)
load the file automatically. `gate-run` applies it **only when you pass
`--project-dir`**; run without it, `gate-run` uses defaults and prints a stderr
note if a `gate_thresholds.json` sits next to the input so the ignored config is
never invisible. Each saved `GateReport` records the effective `thresholds`
behind its verdict, so a gated report stays reproducible even after the file is
edited. `gate-thresholds <project-dir>` prints the effective values so you can
copy them into the file and edit; `gate-thresholds-set <project-dir> --values-json …`
writes a validated `gate_thresholds.json` directly (out-of-range values are refused,
not written — this is what the desktop Settings editor calls). Example:

```json
{ "block_exact_duplicates": false, "max_regression_score_drop": 5.0 }
```

## Future work

A richer per-run selection UI and a desktop threshold editor are follow-ups.


---

## Quality signals & schema-specific checks (reference)

_Consolidated from the former QUALITY_GATES.md — the quality signals the gates
aggregate, plus schema-specific checks and the guiding principle._

### Quality Gates

Quality gates prevent bad examples from silently entering training exports.

#### v0.1 gates

Required:

- valid JSON
- row is a JSON object
- required fields present
- required fields non-empty
- declared field types match schema definitions
- chat messages include valid role/content structure
- schema ID known
- export format supported

Warnings:

- very short output
- very long output
- duplicate example ID
- missing tags
- missing source metadata
- missing license metadata

#### Current and Next Gates

Current:

- duplicate content detection
- normalized duplicate content detection
- low-information text detection
- first-pass synthetic-pattern warnings
- train/test leakage detection (post-split, non-destructive warning)
- first-pass PII / secret detection (emails, SSNs, private keys, AWS/API keys, JWTs, Luhn-valid cards)
- token-length outlier detection (IQR-based, using the Unicode-aware token estimate)
- category-imbalance warnings (low-cardinality field dominated by one value)
- report-level Evaluation failure summaries (by tag, failure reason, and score band)

Next:

- (all previously listed gates are now implemented)

#### Current basic quality report

The engine currently reports:

- example count
- empty rows
- exact duplicate rows
- normalized duplicate rows
- low-information rows under the current token threshold
- dataset-wide synthetic-pattern warnings, including repeated openings,
  repeated closings, and generic AI-style phrases
- structured synthetic issue details with severity, affected row numbers, and
  repair suggestions

The desktop app appends quality snapshots to each project's
`quality_history.jsonl` after user-triggered quality checks and dataset-changing
actions. The history issue count includes duplicate, low-information, empty-row,
and synthetic-pattern warning counts, giving the user a lightweight trend line
for whether edits and imports are improving the dataset.

The desktop app also exposes structured synthetic issues as a triage list. A
selected issue can prepare an AI Assist `rewrite-output` pass by loading the
first affected row into the draft editor and copying the repair suggestion into
the AI Assist instruction. Batch rewrite preparation can also save a project-
local resume record in `ai_assist_rewrite_batches.json`. This is a handoff, not
automatic cleaning.

Synthetic issue severities are heuristic and intentionally conservative. They
help prioritize review; they do not automatically block export or rewrite rows.

#### Code dataset gates

- language field present
- code block not empty
- optional syntax parse
- optional tests field validation

#### Image-caption gates

- image file exists
- caption not empty
- image resolution recorded
- duplicate captions warning
- missing license warning

#### Principle

A quality gate should explain:

1. what failed
2. why it matters
3. how the user can fix it
