# Static MoE Model Inspection

CorpusStudio Phase 8 adds bounded, dependency-light inspection of expert topology declared by a local
model snapshot's `config.json`. The source of truth is the `ModelTopology` surface in
[`engine/corpus_studio/platform/contracts.py`](../engine/corpus_studio/platform/contracts.py); the pure
family parser lives in
[`engine/corpus_studio/platform/moe_inspector.py`](../engine/corpus_studio/platform/moe_inspector.py)
and is called by `model-inspect`.

This is **static metadata evidence only**. It does not load a model, import torch or Transformers,
execute repository code, test inference or training, prove backend compatibility, measure parameter
residency, predict fit, or establish any MoE runtime capability.

## Evidence states

Every `ModelTopology` carries a `TopologyInspection` record:

| Status | Meaning |
|---|---|
| `not_checked` | No verified `config.json` digest was available, so no classification ran. |
| `no_recognized_moe_evidence` | A verified config was inspected, but no allowlisted MoE family or MoE-like signal was recognized. This is not dense proof. |
| `detected` | One allowlisted family supplied a complete, internally valid structural mapping. |
| `incomplete` | The family is allowlisted, but required fields are missing, malformed, out of bounds, or contradictory. Execution kind stays unknown. |
| `unsupported_family` | MoE-like metadata is present for a family the parser does not map. Execution kind stays unknown. |

Static results bind `config_file`, its SHA-256 digest, the exact config paths used, parser method
`static_config_v1`, and `evidence_level: static_metadata_only`. `runtime_capability` is always
`unverified`.

## Allowlisted families

The first slice intentionally recognizes only mappings with narrow, testable rules:

| `model_type` | Required topology fields | Layer rule | Shared experts |
|---|---|---|---|
| `mixtral` | `num_hidden_layers`, `num_local_experts`, `num_experts_per_tok` | every decoder layer | none declared by this mapping |
| `qwen2_moe` | `num_hidden_layers`, `num_experts`, `num_experts_per_tok`, `decoder_sparse_step`, `shared_expert_intermediate_size`; optional `mlp_only_layers` defaults to `[]` | `(layer + 1) % decoder_sparse_step == 0`, excluding `mlp_only_layers` | one always-active shared expert per sparse layer |
| `deepseek_v2` / `deepseek_v3` | `num_hidden_layers`, `first_k_dense_replace`, `n_routed_experts`, `n_shared_experts`, `num_experts_per_tok`; optional `moe_layer_freq` defaults to compatibility value `1` | `layer >= first_k_dense_replace` and `layer % moe_layer_freq == 0` | the declared `n_shared_experts` per MoE layer |

Counts and layer indices are bounded before materializing structures. Boolean values are rejected as
integers, top-k cannot exceed routed experts, duplicate or out-of-range layer indices fail closed, and
an allowlisted config that resolves to no expert layer remains incomplete. Unsupported families are
not guessed from similar field names.

The mappings follow the corresponding upstream Transformers configurations and decoder-layer
construction:

- [Mixtral configuration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/configuration_mixtral.py)
- [Canonical Qwen1.5-MoE config](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/raw/main/config.json), [Qwen2-MoE configuration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_moe/configuration_qwen2_moe.py), and [decoder implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py)
- [DeepSeek V2 configuration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v2/configuration_deepseek_v2.py) and [official layer-frequency implementation history](https://huggingface.co/deepseek-ai/DeepSeek-V2/commit/44f7caf9e95112ea265b13f837bd43d520253548)
- [DeepSeek V3 configuration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/configuration_deepseek_v3.py) and [decoder implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py)

Additional families require their own explicit parser and fixtures. Similar names or keys are not
enough to claim a topology.

## Structural counts are not parameter counts

`ExpertGroup` records a component-scoped layout:

- stable group, layer namespace, canonical component path, and sorted layer indices;
- total logical expert identities per layer;
- routed and always-active shared expert counts that exactly partition that total;
- routed experts selected per token; and
- the config paths used as metadata evidence.

`ExpertTopologyCounts` derives model-wide **expert-instance** totals: MoE layers, routed/shared/logical
expert instances, and routed/shared/total active expert instances for one token's full model pass.
Its unit is literally `expert_instances`.

Phase 8 also makes the routed/shared partition explicit. Newly emitted records always carry concrete
`routed_expert_count` and `shared_expert_count` values. The schema continues to accept pre-Phase-8
groups that omitted the routed count or serialized a null shared count; validation normalizes those
records to `routed = expert_count - shared` and `shared = 0` when absent. Re-serialize a validated
legacy descriptor to persist the normalized form.

These values never become `N_logical`, `N_active_token`, or `N_resident` parameter coordinates.
Top-k says how many expert identities routing selects; it does not say how many independent parameter
coordinates those experts contain, where they reside, whether weights are tied, or whether a backend
executed them. `ParameterAccountingReport` therefore retains active/resident gaps until an exact,
hash-pinned producer supplies coordinate evidence.

## CLI

From `C:\CorpusStudio\engine`:

```powershell
.\.venv\Scripts\python.exe -m corpus_studio.cli model-inspect C:\models\mixtral --json
```

The human view reports the classification, family, expert-instance counts, and the explicit statements
`Resident experts: unknown - runtime measurement required` and `Execution support: not evaluated`.
JSON output and atomically written descriptor files carry the complete evidence record.

## Deliberate non-claims

This slice does not verify:

- model loading, inference, fine-tuning, or full training;
- backend or kernel support;
- router quality, expert utilization, exposure, starvation, or collapse;
- active, touched, updated, exposed, or resident parameter-coordinate counts;
- GPU/RAM/NVMe placement, offload, parallelism, throughput, or fit; or
- native-Linux RTX 5070 or any other hardware behavior.

Those require isolated backend contracts, functional probes, typed runtime telemetry, and—where
hardware is involved—measurements on the final machine.
