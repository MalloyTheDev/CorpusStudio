# Model and Tokenizer Descriptors

CorpusStudio's Phase 3 foundation describes model and tokenizer snapshots without loading them. The
source of truth is the pydantic contracts in
[`engine/corpus_studio/platform/contracts.py`](../engine/corpus_studio/platform/contracts.py); the
dependency-light inspector is
[`engine/corpus_studio/platform/model_inspector.py`](../engine/corpus_studio/platform/model_inspector.py),
with allowlisted MoE config parsing in
[`engine/corpus_studio/platform/moe_inspector.py`](../engine/corpus_studio/platform/moe_inspector.py).

## What ships

- `ModelDescriptor`: source identity, artifact role, architecture/task metadata, formats,
  component-scoped stored representation, scoped parameter-count evidence, optional routing/expert
  topology, vocabulary/context metadata, tokenizer linkage, license/trust findings, file inventory,
  backend-compatibility references, and independent verification axes.
- `TokenizerDescriptor`: source identity, implementation/format, base + added + effective vocabulary,
  maximum token ID, model length, special tokens, chat-template content/hash,
  normalization/pre-tokenization metadata, trust/inventory evidence, and model compatibility results.
- `model-inspect`: bounded, offline inspection of a local model directory and optional local tokenizer
  directory. It can emit one JSON bundle or atomically persist the individual descriptor and
  compatibility records. For four allowlisted `model_type` values it also emits hash-pinned static
  expert-topology evidence.
- Deterministic draft-2020-12 JSON Schemas and generated TypeScript types for both root contracts.

This slice does **not** fetch from a model registry, import torch/transformers/tokenizers, instantiate a
model or tokenizer, execute custom code, train/edit/convert weights, prove a backend can load the
snapshot, predict hardware fit, execute MoE topology, or prove any MoE runtime capability.

## Static inspection

From `C:\CorpusStudio\engine`:

```powershell
.\.venv\Scripts\python.exe -m corpus_studio.cli model-inspect C:\models\tiny `
  --model-id tiny-model `
  --tokenizer C:\models\tiny-tokenizer `
  --tokenizer-id tiny-tokenizer `
  --repository owner/repository `
  --requested-revision main `
  --resolved-commit 0123456789abcdef `
  --tokenizer-repository owner/tokenizer-repository `
  --tokenizer-requested-revision main `
  --tokenizer-resolved-commit fedcba9876543210 `
  --hash-weights `
  --out C:\models\descriptors `
  --json
```

The default human output is intentionally short. `--json` emits the complete bundle. Metadata and
repository code are always hashed. Weight files can be large, so their content hash is opt-in through
`--hash-weights`; hashing streams the file and rejects a source that changes during the read.

The inspector:

- accepts only a regular local directory and never performs network access;
- walks deterministically with a 100,000-file bound and a 16 MiB bound per parsed JSON file;
- does not follow symlinks or Windows junctions;
- records portable POSIX-relative paths, format, role, size, hash status, serialization risk, and link
  status independently;
- flags pickle-based weights and repository `.py` files / `auto_map` metadata;
- always emits `trust_remote_code: false`; custom code requires a future, separate exact-revision
  approval in an isolated environment;
- recognizes only the explicit Mixtral, Qwen2-MoE, DeepSeek V2, and DeepSeek V3 config mappings;
- keeps malformed allowlisted metadata and unsupported MoE-like families **unknown** rather than
  guessing a runnable topology from names or similar keys.

`inventory_sha256` seals the canonical inventory record, including each file's hash status. It is not
automatically a full content digest. `source.snapshot_sha256` is populated only when the complete
inventory has verified content hashes for every recorded file.

## Identity and trust

Requested source identity and resolved evidence are different fields:

- `requested_revision` records user intent such as `main` or a tag;
- `resolved_commit` records the immutable hexadecimal commit actually inspected;
- `revision_pinned` can be true only when that immutable commit exists;
- `snapshot_sha256`, when present, binds the fully hashed local snapshot.

A branch name is never presented as immutable proof. A local path is evidence of what was inspected,
not a claim that it came from a particular repository. When model and tokenizer paths differ, the
tokenizer never silently inherits the model repository identity; use the tokenizer-specific source
options. When both descriptors inspect the same directory, the shared source evidence can be inherited.

Trust findings are also not permission. A descriptor can say custom code appears necessary, list the
files and `auto_map`, and require approval/isolation. It cannot authorize execution. License metadata
and license-file presence are recorded, but neither is converted into an unverified redistribution or
training permission.

## Vocabulary compatibility

Tokenizer size is not assumed to be `len(base_vocab)`. The inspector records:

- base vocabulary size;
- distinct added-token IDs beyond the base IDs;
- maximum observed token ID;
- effective vocabulary size (`max_token_id + 1` when IDs are available);
- special-token IDs and whether each came from the added-token table.

Static compatibility compares that effective size and the maximum special-token ID against the
model's input-embedding and output-head row evidence. Tied embeddings remain tied: a required input
resize also requires the logically shared output head. Results are explicit:

| Status | Meaning |
|---|---|
| `compatible` | Every static check has sufficient evidence and passes. This is not load/runtime proof. |
| `resize_required` | Identity is not contradictory, but a new explicitly resized artifact is required. No silent mutation occurs. |
| `incompatible` | Static identity evidence conflicts, such as a different linked tokenizer ID. |
| `unverified` | Required static evidence is absent, or custom code needs separate approval/isolation. |

Hugging Face's very large `model_max_length` sentinel is normalized to unknown rather than reported as
a real context window. A model/tokenizer context mismatch is surfaced as a warning; the smaller runtime
limit governs.

## MoE-safe representation

The contract provides representation space for sparse and conditional models, and Phase 8 now fills
that space from a narrow static-config allowlist:

- `parameters.components[]` scopes format, stored dtype, quantization, and files to shared weights,
  router, expert group, embeddings, output head, adapter, or another component;
- `parameters.counts[]` replaces a misleading scalar count. Each record carries count kind, scope,
  measurement window, unit, evidence source, and explicit handling of tied/shared/replicated/generated/
  quantized state, optimizer shadows, and decompressed caches;
- `topology.semantic_routing` describes the learned selection policy;
- `topology.expert_groups[]` provides stable group/layer/component identity and partitions the logical
  expert count into routed and always-active shared experts;
- `topology.expert_counts` derives **expert-instance** totals for a full token pass, not parameter
  coordinates;
- `topology.inspection` pins the config hash, evidence paths, parser method, and
  `static_metadata_only` evidence level while keeping runtime capability `unverified`;
- physical placement, residency, paging, and prefetch remain owned by the immutable `RunPlan`
  physical-execution specification.

Phase 3 supplies this representation substrate. The Phase 5
[`ParameterAccountingReport`](PARAMETER_ACCOUNTING.md) now produces and validates separately scoped
logical/active/resident/touched/updated/exposed evidence and can be requested with
`model-inspect --parameter-accounting`. Phase 8 recognizes only `mixtral`, `qwen2_moe`,
`deepseek_v2`, and `deepseek_v3`; see [`MOE_MODEL_INSPECTION.md`](MOE_MODEL_INSPECTION.md). A complete
allowlisted mapping may set `execution_kind: mixture_of_experts` and emit structural expert-instance
counts. Those counts do not populate active or resident parameter-coordinate evidence, establish
loadability/backend support, or prove MoE execution. Every other topology remains unknown.

## Verification axes

`DescriptorVerification` keeps metadata, integrity, license, and custom-code-policy outcomes separate.
Backend compatibility is a separate list of evidence references. A passed metadata parse does not imply
content integrity; integrity does not imply license permission; neither implies functional loadability,
backend support, hardware fit, trainability, or evaluation quality.

## Regenerating the boundary

After a contract change:

```powershell
cd C:\CorpusStudio\engine
.\.venv\Scripts\python.exe -c "from corpus_studio.platform.schema_export import export_json_schemas; export_json_schemas('../docs/contracts')"
cd C:\CorpusStudio\apps\web
npm run gen:contracts
```

The committed artifacts include `docs/contracts/ModelDescriptor.schema.json`,
`docs/contracts/TokenizerDescriptor.schema.json`, `docs/contracts/ParameterAccountingReport.schema.json`,
and their counterparts under `apps/web/src/contracts/`.
