# WBG test corpus + baseline adapter

Test/example data for CorpusStudio: the **World Bible Generator (WBG)** lore-generation dataset and
the first (baseline) LoRA adapter trained on it. Small enough to live in the repo so it clones with
CorpusStudio — no separate copy needed.

## `data/` — the corpus (522 examples, ~5 MB)
Chat-format examples whose assistant turn is a single `AIResult` JSON lore entry (module/status/
summary/tags/storyHooks/…). Token lengths (Qwen2.5 tokenizer): **min 1802 · mean 2241 · max 3445**.
- `wbg_clean_522.jsonl` — the full set.
- `wbg_clean_splits/`… i.e. `train.jsonl` (469) / `validation.jsonl` (26) / `test.jsonl` (27).

Check it before training:
```bash
corpus-studio dataset-tokens examples/wbg/data/wbg_clean_522.jsonl \
    --base-model Qwen/Qwen2.5-7B-Instruct --dataset-format chat --seq-len 4096
```

## `adapter-seq1536-baseline/` — the pre-fix adapter (~78 MB)
The LoRA adapter from the first 3-epoch run. **It was trained at `sequence_len = 1536`, which
truncated 100% of the (1802–3445-token) examples** — cutting the end (the JSON output) off every one,
so the model learned to emit *incomplete* JSON. Kept deliberately as the **"before" baseline** for the
paper's before/after; the proper untruncated **seq-4096 re-train** (on the native-Linux box) **has now
produced the "after" (2026-07-19)** - see the next section. See `MODEL_CARD.md` for the exact settings.

## The "after" - untruncated seq-4096 re-train (2026-07-19)
An **exploratory product run** (NOT a sealed IEEE research cell) re-trained WBG `train.jsonl` (469) at
`sequence_len = 4096` on the native-Linux RTX 5070 host, where **nothing truncates** (real Qwen2.5 token
lengths min 1802 / mean 2241 / max 3445 < 4096; the token-coverage ledger reports 100%). Same corpus,
same `lr = 2e-4`, `seed = 42` as the "before". Two variants were produced end to end (train -> adapter
export), both via the proven seq-4096 config (flash SDPA + liger fused-CE + bnb paged-8bit-AdamW +
`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`):

| Variant | LoRA | Epochs | Final loss | Adapter size | Notes |
| --- | --- | --- | --- | --- | --- |
| **r8** (default) | r=8, alpha=16 | **3 (full)** | ~0.65 | ~80 MB | Peak ~11.77 GB - stays off the 12.3 GB ceiling for the whole run. Recommended for headroom. |
| r16 | r=16, alpha=32 | 1 (converged) | ~0.79 | ~161 MB | Converged by epoch 1; the full 3-epoch r16 deterministically OOMs at the worst-length batch (peak hits the full card even with the optimizer paged to host). |

Per-run **mean** losses (0.93 for r8, 1.17 for r16) differ from the **last-step** losses above and from
the per-adapter `MODEL_CARD.md`; both are honest, just different reductions. The adapters live under
`/mnt/training-nvme/corpusstudio/runs/wbg-after-seq4096/` and are **not committed** (r16 exceeds GitHub's
100 MB/file limit). The r8 full-length variant was enabled by the `platform-plan --lora-r/--lora-alpha`
option (previously locked to r16).

### Eval: does the "after" emit COMPLETE AIResult JSON? (2026-07-19)

Measured on the held-out **test split** (n=27) - deterministic greedy decode of the base in 4-bit nf4 (the
QLoRA training regime) + each adapter. Metric = output parses as one JSON object with all **13 required
AIResult keys present** (empty arrays are valid - the gold carries them). Full evidence (harness, raw
per-example reports, caveats) under `runs/wbg-after-seq4096/eval-closeout/`.

| Model | LoRA / seq | Complete-JSON |
| --- | --- | --- |
| base `Qwen2.5-7B` (no adapter) | - | 0/27 (0%) - valid JSON, wrong schema |
| **before** `adapter-seq1536-baseline` | r16 / 1536 | 9/27 (33.3%) - truncated |
| after-r16 (rank-matched control) | r16 / 4096 | 24/27 (88.9%) |
| **after-r8** (default deliverable) | r8 / 4096 | **26/27 (96.3%)** |

**Before -> after: 33.3% -> 96.3%.** The rank-matched pair (before-r16 -> after-r16, both r16) is
33.3% -> 88.9%, isolating the gain to the **1536 -> 4096 sequence length**. Two identical greedy passes of
after-r8 were reproducible (27/27 per-example). This is exploratory/product evidence, not a sealed research
result.

## The base model — NOT in the repo
`Qwen/Qwen2.5-7B-Instruct` is ~15 GB (individual shards > GitHub's 100 MB/file limit), so it is **not**
committed. It re-downloads automatically:
```bash
corpus-studio model-fetch Qwen/Qwen2.5-7B-Instruct
```
(or copy an existing HF cache to `~/.cache/huggingface` to skip the download).
