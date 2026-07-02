# Quality Gates

Quality gates prevent bad examples from silently entering training exports.

## v0.1 gates

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

## Current and Next Gates

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

## Current basic quality report

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

## Code dataset gates

- language field present
- code block not empty
- optional syntax parse
- optional tests field validation

## Image-caption gates

- image file exists
- caption not empty
- image resolution recorded
- duplicate captions warning
- missing license warning

## Principle

A quality gate should explain:

1. what failed
2. why it matters
3. how the user can fix it
