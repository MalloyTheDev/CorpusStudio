# AI Assist Lab

AI Assist Lab is the staged Corpus Studio workspace for using models to help
review, rewrite, tag, and draft dataset examples.

AI should assist dataset authors. It should not blindly generate garbage and
push it into the dataset. Human review remains required.

AI Assist Lab is the v0.3 lab surface. It comes after the dataset authoring and
evaluation loops, and it must stay review-first.

Current MVP status: Corpus Studio has a review-first `ai-assist` engine command
and a desktop AI Assist tab. The tab sends the current draft through a selected
local model backend, displays the model response and any suggested JSONL, and
lets the user move that suggestion into Writing Studio for editing, validation,
and explicit save. It can also check backend health through the engine
`backend-health` command and list available local models through the engine
`model-list` command. It persists review queue items in the project folder
with `review_required`, `accepted`, and `rejected` states. Queue selection shows
the stored source draft beside the suggested JSONL with a compact comparison
summary. The desktop action control uses schema-aware presets, and the engine
adds first-pass warnings for repetitive synthetic patterns and weak preference
pairs. The queue can be filtered by review state and bulk-mark visible reviews
accepted or rejected. It also supports text search, simple sorting, saved queue
views for repeated triage passes, and undo for recent bulk triage actions. It
does not directly mutate accepted examples. The Quality panel can also send a
selected synthetic-pattern issue into a prepared `rewrite-output` AI Assist
workflow by loading the first affected row into the draft and copying the repair
guidance into the instruction. It can also prepare a batch rewrite draft for
the affected rows across the current structured synthetic issues and persist
that prepared batch in `ai_assist_rewrite_batches.json` for resume after app
restart. Preference
projects also have a first-pass
Preference Review tab that shows prompt/chosen/rejected/reason fields side by
side, ranks pairs by weak/moderate/strong contrast, filters the queue by
contrast strength, and prepares the `judge-preference-strength` AI Assist
action for the selected pair. The visible preference ranking can be exported as
an inspectable JSON artifact for DPO or reward-model review, and the visible
queue can be prepared as a batch judge pass.

## Product Role

AI Assist Lab should help users move faster while preserving dataset quality. It
should make weak examples easier to find, draft candidates easier to review, and
schema violations easier to correct.

The lab should never hide the difference between:

- human-authored accepted examples
- AI-drafted examples awaiting review
- AI-rewritten examples awaiting review
- rejected synthetic examples

## Supported Actions

AI-assisted actions in scope:

- suggest tags; MVP action key: `suggest-tags`
- detect vague examples
- rewrite weak outputs; MVP action key: `rewrite-output`
- generate draft examples; MVP action key: `draft-example`
- create chosen/rejected pairs
- judge preference strength; MVP action key: `judge-preference-strength`
- identify schema violations; MVP surfaces existing validator warnings in the prompt
- detect repetitive synthetic patterns; MVP emits review warnings for repeated openings, repeated text fields, and generic phrases
- review a draft without rewriting; MVP action key: `review`

These actions should use the same model backend abstraction as Evaluation Lab.
Local-first backends should be the default path.

## Human Review Requirement

AI-generated or AI-edited examples must pass through human review before they
enter the accepted dataset.

The review UI should support:

- accept
- edit then accept
- reject
- send back for another rewrite
- mark as needs human rewrite
- preserve original and AI-suggested text for auditability
- prepare preference-strength review for DPO or reward-model pairs
- prepare batch rewrites for structured synthetic-pattern issues
- export and batch-review visible preference rankings

The current desktop MVP supports the middle of that flow: suggestions are saved
to a project-local review queue, can be marked accepted or rejected, and remain
review-only until the user deliberately moves suggested JSONL into Writing
Studio. It also preserves a side-by-side source draft and suggested JSONL view
for auditability. Review-state filters, search, sorting, visible-item bulk
triage, saved queue views, and multi-step undo help process larger queues
without bypassing human decisions. Prepared synthetic batch rewrites can also
be resumed from a project-local rewrite-batch list after restart. From there
the normal validator and save controls remain the acceptance path.

## Synthetic Data Warning

Synthetic data can quickly introduce repetition, shallow phrasing, incorrect
answers, format drift, and hidden leakage. Synthetic data should pass schema
validation and quality gates before being accepted.

AI Assist Lab should flag common synthetic-data problems:

- repeated openings or closings
- overused examples
- generic answers
- unsupported claims
- inconsistent formatting
- near duplicates
- weak chosen/rejected contrast

Current MVP checks are intentionally conservative. They surface warnings in the
review result for repeated suggested text, repeated openings, generic AI-style
phrases, and preference pairs whose chosen/rejected answers are identical or
very high-overlap. Structured dataset-wide synthetic issues can be triaged into
an AI Assist rewrite workflow, but they do not reject or rewrite rows
automatically. Preference pair review similarly prepares a judge pass and leaves
the human in charge of editing, validation, and final acceptance. Batch
synthetic rewrite preparation and preference-ranking export are review aids;
they do not mutate accepted data.

## Workflow

```text
AI drafts or reviews example
-> engine gates the generated candidate rows (schema/quality/PII) as a pre-review signal
-> human accepts/edits/rejects
-> validator checks schema
-> quality engine scores sample
-> example enters dataset
```

## Candidate gate (pre-review safety signal)

When a run produces suggested rows (e.g. `draft-example`, `rewrite-output`), the
engine runs the **existing dataset gate runner** (`run_dataset_gates`) over the
generated candidate rows and attaches the resulting `GateReport` to the result as
`candidate_gate`. This closes the constraint's
`generate -> validate -> quality -> gates -> human review` chain: the candidates
now carry a schema/quality/PII/leakage `pass`/`warn`/`block` verdict **before**
they reach the human review queue.

This is a **signal, not a decision**. It adds no new detection (it reuses the
gate runner verbatim) and it changes nothing about acceptance:

- `review_required` stays `true` regardless of the gate verdict.
- A **clean** gate is *not* approval — the human still reviews.
- A **block** (e.g. a leaked key or secret in generated content) does *not*
  auto-reject: the candidate is still preserved for the human to see and reject.
  The gate leads with the block so it is impossible to miss.
- `candidate_gate` is `null` when the run produced no candidate rows (there is
  nothing to gate — never a fake pass).

Policy is still enforced **first**: `authorize_action` runs before the provider
is ever called, so an evaluator-only provider is blocked from a generating action
before any generation or gating can occur. The gate step cannot bypass provider
policy.

## Implementation Notes

AI Assist Lab should be implemented as a review queue, not as direct mutation of
accepted examples.

Recommended data states:

- `drafted_by_ai`
- `review_requested`
- `accepted`
- `rejected`
- `needs_edit`

Recommended metadata:

- model backend
- model name
- prompt template id
- generation timestamp
- source example id when rewriting
- reviewer action
- quality warnings

Current result shape:

```json
{
  "schema_id": "instruction",
  "action": "rewrite-output",
  "model": "qwen2.5-coder:7b",
  "review_state": "review_required",
  "review_required": true,
  "prompt_template_id": "ai_assist_review_v0.1",
  "model_output": "{...raw model response...}",
  "suggested_jsonl": "{\"instruction\":\"...\",\"output\":\"...\"}\n",
  "warnings": [],
  "validation_errors": [],
  "candidate_gate": {
    "scope": "dataset",
    "target": "ai_assist_candidates",
    "overall_status": "pass",
    "pass_count": 4,
    "warn_count": 0,
    "block_count": 0,
    "results": []
  }
}
```

`candidate_gate` is the gate report over the generated candidate rows, or `null`
when the run proposed no rows. It informs review; it never approves or
auto-accepts (see **Candidate gate** above).

Current desktop queue item shape adds local review metadata around that result:

```json
{
  "review_id": "local-id",
  "created_at": "2026-06-30T12:00:00Z",
  "decided_at": null,
  "review_state": "review_required",
  "source_draft": "{\"instruction\":\"...\"}",
  "suggested_jsonl": "{\"instruction\":\"...\",\"output\":\"...\"}\n"
}
```

Prepared synthetic batch rewrites are stored separately from review results:

```json
{
  "batch_id": "local-id",
  "created_at": "2026-07-01T12:00:00Z",
  "schema_id": "instruction",
  "action": "rewrite-output",
  "row_numbers": [23, 24, 25],
  "issue_count": 3,
  "source_draft": "[{\"instruction\":\"...\"}]",
  "instruction": "Rewrite affected rows 23, 24, 25 as a batch..."
}
```

Near-term hardening should focus on production-grade synthetic pattern
clustering, target-specific preference export preparation, and stronger
reviewed-fix tracking. It should not bypass the validator or auto-save model
output.

## Current CLI MVP

```powershell
python -m corpus_studio.cli ai-assist examples\datasets\instruction\train.jsonl instruction --action review --backend ollama --model qwen2.5-coder:7b
```

The CLI builds a guarded prompt, treats dataset rows as untrusted content,
passes validator warnings into the prompt, calls the selected local backend, and
returns a review-required JSON result. It requires the chosen local backend to
already be running.

## Guardrails

- Do not automatically accept generated rows.
- Do not bypass schema validation.
- Gate generated candidates before review, but never let the gate decide: a
  clean gate is not approval and a block does not auto-reject — the human decides.
- Do not use cloud providers unless the user configures them.
- Do not assume generated data is licensed or correct.
- Do not train on examples marked rejected or review-only.
