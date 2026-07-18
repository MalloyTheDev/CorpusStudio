# TraceRecord and Trace Studio foundation

CorpusStudio has two trace representations:

- A compact legacy/training view: prompt or messages, optional thinking, and answer.
- The versioned language-neutral TraceRecord contract used for durable provenance, review, tool
  boundaries, generated clients, and training gates.

The legacy shape remains readable. New generated candidates use TraceRecord by default. Conversion is
explicit; the engine never silently rewrites an existing JSONL file and never writes a project's
examples.jsonl.

## What a TraceRecord preserves

| Area | Contract evidence |
|---|---|
| Identity | Stable trace_id plus canonical SHA-256 over the fully defaulted record, excluding only trace_hash. |
| Source | One exact source-row ID using sha256(exact_row_signature), and either a hash-pinned DatasetManifest ref or a hash-pinned imported artifact. |
| Context | Ordered, role-preserving messages with stable IDs. Chat context is not concatenated into an ambiguous prompt. |
| Process | Ordered segments with explicit reasoning, action, observation, verifier, tool-call, tool-result, and final-answer kinds. |
| Tool boundary | Unique call IDs; every result follows exactly one call; tool output remains tool-authored and cannot become model-authored reasoning. |
| Producer | Tool/version, backend/provider, separately recorded requested and backend-reported model identity, route, prompt-template hash/version, request and response hashes, safe response metadata, decoding settings, and the effective provider-policy snapshot/decision. Credentials and raw response bodies are not stored. |
| Validation | Validator/version/config hash, timestamp, typed warning/block findings, and a verdict. Normalized-exact answer leakage blocks; semantic correctness is not claimed. |
| Review | pending, approved, or rejected, with reviewer/time/notes. Review is an immutable successor pinned to the previous record hash. |

Exactly one final-answer segment must be last. Segment IDs and sequence numbers are unique and
contiguous. Structured segment content cannot contain think delimiters; those delimiters exist only
at legacy import and model-specific training-render boundaries. Malformed, repeated, reversed, or
partial tag pairs are rejected.

The response hash binds a canonical envelope containing the completion text, backend-reported model
identity, and a hash of the raw provider response. The raw response itself is not retained. If a
backend reports a model other than the requested model, CorpusStudio resolves and authorizes that
actual identity before it preserves the candidate. Provider snapshots remain provenance rather than
authority: approval and training re-resolve both identities against the project-local external
provider-policy override file, reapply unknown-provider default deny, and reassert the built-in
frontier-provider restriction. Revoking or drifting that external approval blocks admission.

Generated or imported reasoning remains unverified, including after record approval. Approval means a
human accepted the record for the intended workflow; it is not proof that the reasoning is true,
complete, optimal, or the model's hidden chain of thought.

## Workflow

### Author a draft

The built-in trace schema is available in the ordinary desktop project wizard and Writing Studio. It
intentionally uses the editable legacy draft fields:

    {
      "prompt": "What is 17 multiplied by 23?",
      "thinking": "Multiply by 20 and 3, then add 340 and 51.",
      "answer": "391"
    }

Seal a separate artifact:

    corpus-studio trace-migrate drafts.jsonl --out trace-records.pending.jsonl
    corpus-studio trace-validate trace-records.pending.jsonl --json

Migration records absent legacy evidence as absent; it never invents a teacher, revision, dataset
version, or verification claim.

### Generate candidates

Trainable generation fails closed. Approve the exact local model or route first in the project policy,
then generate:

    corpus-studio provider-approve ollama qwen-model --project-dir ./project
    corpus-studio trace-generate prompts.jsonl --model qwen-model --project-dir ./project --out trace-records.pending.jsonl

Policy is resolved and authorized before the backend is constructed or called. Accepted records are
still pending; quality filtering is not human review. The adjacent report JSON records every
input-row identity and accepted, rejected, or error outcome without retaining raw response bodies.

The legacy-output option remains a compatibility escape hatch. It writes unsealed, explicitly
non-trainable flat rows plus an adjacent <out>.trace-records.jsonl sidecar containing the actual
pending records. Review and train from that sidecar; the flat compatibility rows never satisfy the
versioned review gate.

### Review and train

Write approved or rejected successors to a new artifact:

    corpus-studio trace-review trace-records.pending.jsonl --out trace-records.approved.jsonl --reviewer reviewer-id --decision approved --all --project-dir ./project
    corpus-studio trace-validate trace-records.approved.jsonl --require-approved --project-dir ./project

Use repeated trace-id options instead of all to review a subset. Approval recomputes the engine's
validator/version/config/findings instead of trusting stored evidence, and blocked validation cannot
be approved. For model-produced records it also requires the current external project policy to
authorize the requested and resolved model and to reproduce the stored snapshot. Training recomputes
that evidence and external authority again. The current first-party trace trainer accepts
approved inline reasoning records and ordinary legacy rows; it refuses pending, rejected, tampered,
foreign-validator, and generated legacy-compatibility rows before runtime probing or model loading.
Ordinary legacy rows remain compatible but emit an explicit unsealed/no-review warning.

The generalized contract can represent tool, agent, verifier, and process-supervision records. The
current trainer deliberately supports only reasoning plus final-answer segments; unsupported segment
kinds are refused rather than silently flattened. Supported segments must be assistant-authored,
must have a producer-consistent model/human/imported origin, cannot be rejected, and cannot carry
target/weight/reward semantics that the current SFT renderer would discard.

## Validation scope

The shipped deterministic checks cover:

- required context and final answer;
- malformed legacy reasoning markup;
- reasoning identical to the final answer;
- normalized-exact answer leakage across context;
- stray reasoning tags in the final answer;
- trivially short reasoning warnings;
- record, source, policy, parent, and inline tool-result hash consistency;
- segment ordering and tool call/result pairing.

Some of those rules are cross-field Pydantic validators and cannot be fully represented by JSON
Schema alone. Non-Python clients can use the generated schema for shape/type validation, but durable
admission, review, and training must still pass through the engine validator.

They do not prove semantic correctness, factuality, causal faithfulness, hidden chain-of-thought
identity, or tool safety.

## Honest non-goals of this foundation

- No graphical dedicated Trace Studio tab yet; the built-in trace draft schema and CLI workflow are
  the engine/desktop authoring foundation.
- No tool execution or agent orchestration.
- No semantic/model-based judge and no automatic approval.
- No process-reward, verifier, or tool-use trainer implementation.
- No dataset mixtures or transformation-graph execution.
- No new training backend.
