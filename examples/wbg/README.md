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
paper's before/after; the proper untruncated **seq-4096 re-train** (on the native-Linux box) produces
the "after". See `MODEL_CARD.md` for the exact settings.

## The base model — NOT in the repo
`Qwen/Qwen2.5-7B-Instruct` is ~15 GB (individual shards > GitHub's 100 MB/file limit), so it is **not**
committed. It re-downloads automatically:
```bash
corpus-studio model-fetch Qwen/Qwen2.5-7B-Instruct
```
(or copy an existing HF cache to `~/.cache/huggingface` to skip the download).
