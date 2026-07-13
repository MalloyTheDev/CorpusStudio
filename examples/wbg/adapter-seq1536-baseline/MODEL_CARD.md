# Model card: wbg-7b-cs-3ep (LoRA adapter)

A LoRA adapter trained with Corpus Studio's first-party trainer. This card documents the training **recipe** — it is NOT an evaluation of the model's quality. Evaluate before use.

## Base model

- **Base:** `Qwen/Qwen2.5-7B-Instruct`
- **License:** the *base model's* license governs what you may do with this adapter and any merged model — training on data you can read does not grant redistribution rights to the base. `corpus-studio model-fetch <repo>` reports a base model's license; prefer a permissive (MIT/Apache/BSD) base and verify before distributing.

## Adapter (LoRA)

- Type: LORA / CAUSAL_LM
- r: 16, alpha: 32, dropout: 0.05
- Target modules: o_proj, down_proj, up_proj, k_proj, v_proj, q_proj, gate_proj

## Training

- Format: chat
- Sequence length: 1536, learning rate: 0.0002, seed: 42
- Steps: 177, final train loss: 0.3533
- Mode: 4-bit QLoRA

## Serving

- Merge into the base with `corpus-studio train-merge <adapter-dir>` (auto → GPU / CPU-offload / adapter-only), or serve the base + adapter unmerged (peft / vLLM / TGI accept the adapter).

## Honesty

- A completed run is not a quality signal — run an evaluation suite before promoting this adapter.
- Reproducibility depends on the pinned seed + dataset fingerprint above; changing the data or config changes the result.
