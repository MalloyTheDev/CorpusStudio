# Amendment 0007 - Exact-Length (Non-Padding) Sequence Feasibility Fixture Binding

- Amendment: cs-ieee-linux-training-amendment-0007
- Effective protocol version: 1.7.0 (supersedes 1.6.0)
- Authored: 2026-07-17T15:54:00Z
- Status: prospective
- Classification: FIXTURE_IDENTITY_CHANGE_ONLY / NO_WORKER_CHANGE / NO_NEW_ENVIRONMENT_LINEAGE / NO_PRIMARY_MATRIX_CHANGE

## Why

The 1.6.0 feasibility ladder bound a short chat fixture whose rows tokenize well below each rung. Under
the real worker path (trl SFTTrainer, DataCollatorForLanguageModeling, micro-batch 1) the collated
tensor pads to the single row's length, so the actual non-padding width never reached the rung - the
sequence-width claim was not provable from a real batch. This amendment rebinds the feasibility fixture
to a deterministic, license-clear, per-rung EXACT-LENGTH chat fixture so a real mb=1 microbatch carries
EXACTLY the rung in non-padding tokens, with zero padding and zero truncation. No worker code, wheel, or
environment lineage changes; the authorized future lineage remains math-v8 / flash-v8.

## What is bound (effective matrix 1.7.0, arm seven_b_native_linux_nonpadding_sequence_feasibility)

- Fixture cs-ieee-linux-7b-nonpadding-seq-fixture-v1 (CC0-1.0), 12 rows per rung, fixed row order, no
  packing, no truncation, deterministic (base_seed 7, generator e080a94910c47f63...),
  fixture-root SHA256SUMS e1208e2fa930e0c4...; per-rung dataset / rendered / token-id aggregates with
  every exact_non_padding_length equal to its rung (512/1024/2048/3072/4096).
- Model binding: Qwen/Qwen2.5-7B-Instruct @ a09a35458c702b33eeacc393d103063234e8bc28, model/tokenizer/template aggregates, apache-2.0,
  trust_remote_code false.
- Training objective: full-language-model supervision; expected supervised tokens per microbatch = rung.
- Pre-dispatch conformance: PREDISPATCH_RUNTIME_COLLATOR_CONFORMANCE_PASS (actual trl collator, CPU only,
  no model weights, no GPU; evidence SHA256SUMS 95944f122c5a3b31...). This proves tensor width == rung,
  attention mask exactly rung active, zero padding, zero truncation, labels width == rung, supervision
  present, and no source-row mutation. It is explicitly NOT a completed worker execution or GPU result.
- Execution-time admission (per optimizer step): observed_microbatches == 1; nonpadding_tokens == rung;
  supervised_tokens == rung; step_time_seconds > 0; no truncation / packing / fallback; finite loss. A
  step whose non-padding tokens fall below the rung invalidates that rung's sequence-width claim. The
  final claim is sourced from the raw RunEvent measurements, never from sequence_len, fixture metadata,
  or the collator report.

## Validator

validate_protocol.py adds _validate_nonpadding_sequence_feasibility, which rejects a missing/abbreviated
fixture identity, a wrong rung list/order, a row count other than 12, packing/truncation, a missing
objective mode, a non-padding admission weaker than equality to the rung, observed_microbatches other
than exactly 1, a model/tokenizer/template hash that is not a full SHA-256, a private-corpus fixture
substitution, missing conformance evidence, a claim that collator evidence alone proves execution, and
worker-execution-change or new-lineage wording in this fixture-only amendment. The validator self-hash is
re-sealed by this amendment's manifest.

## Preservation

Amendments 0001-0006, effective matrices 1.1.0-1.6.0, and RESERVED_IDENTITIES v1-v6 are preserved
byte-for-byte. Matrix 1.7.0 differs from 1.6.0 only in the fixture identity, the exact-width admission /
objective / conformance bindings, the amendment metadata, and the version stamp. RESERVED v7 is a strict
append-only superset of v6 (no new run/plan/execution/wheel/environment identity is instantiated; the
frozen fixture and its conformance evidence are added as host-evidence source manifests, not as a
reusable identity class).
