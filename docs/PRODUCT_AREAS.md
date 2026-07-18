# CorpusStudio product areas (canonical map)

**CorpusStudio is a local-first, end-to-end AI development ecosystem and IDE** covering the complete model
lifecycle:

```
raw data -> ingestion -> cleaning -> transformation -> annotation -> dataset construction ->
schema validation -> provenance & licensing -> train/validation/test splits -> tokenization ->
training planning -> environment management -> fine-tuning & (future) pretraining ->
checkpointing & recovery -> evaluation -> comparison -> behavior analysis & modification ->
model packaging -> export -> release -> reproducible evidence
```

It is **not** a "research platform", "training platform", "experiment runner", "dataset tool", or
"fine-tuning application" - those are individual capabilities. Preferred framing: *AI development
ecosystem*, *dataset-to-model IDE*, *end-to-end AI model development platform*, *local-first AI engineering
workspace*, *data-to-training-to-release system*.

Research protocols, telemetry, experiments, and the planned IEEE paper are a **supporting track** (the
Evidence & Experiments area) that makes the product trustworthy and documents discoveries - they do **not**
define the product. `research/ieee-linux-training/` is one evidence program inside the larger product; its
current development priority must not redefine the whole application. The native-Linux 7B / seq-4096 work is
the highest-priority engineering **gate** (Training Studio must be proven reliable) - not the purpose of
CorpusStudio.

## The seven co-equal product areas

Every architecture doc, roadmap, navigation surface, and plugin route must preserve this hierarchy. The
areas are **first-class equals** - dataset engineering, evaluation, release, Behavior Lab, and evidence are
not subordinate to training.

1. **Data Studio** - import & ingestion; cleaning & transformation; annotation; schema design; validation;
   deduplication; provenance & licensing; dataset versioning.
2. **Training Studio** - model & tokenizer selection; training configuration; fine-tuning; pretraining
   workflows; environment management; hardware planning; checkpoint & resume; run management. (The Training
   Systems architecture - `TRAINING_SYSTEMS_ARCHITECTURE.md` - is the internal design of this area.)
3. **Evaluation Studio** - benchmarks; dataset evaluation; model evaluation; regression testing; A/B
   comparison; quality gates.
4. **Behavior Lab** - activation analysis; steering; causal attribution; behavior modification; weight
   surgery; capability-preservation testing. (A first-class area; implementation is gated - see
   `PRODUCT_VS_RESEARCH.md` and the Behavior Lab issue.)
5. **Model & Release Studio** - artifact inspection; adapter merging; quantization; format conversion; model
   cards; licensing checks; export; release packaging.
6. **Environment & Hardware** - dependency isolation; reproducible environments; GPU capability checks;
   resource planning; runtime health.
7. **Evidence & Experiments** - run lineage; telemetry; reproducibility; scientific comparison; research-paper
   evidence. (Includes the opt-in IEEE research overlay; see `PRODUCT_VS_RESEARCH.md`.)

## Architecture note

The target architecture is a **Rust authoritative core** + **isolated, untrusted Python ML workers** ("Rust
owns truth; Python computes ML and returns evidence"; see `TRAINING_SYSTEMS_ARCHITECTURE.md` and the
Rust-core epic). The standard / verified / sealed-research assurance boundary is in
[`PRODUCT_VS_RESEARCH.md`](PRODUCT_VS_RESEARCH.md). What is built today vs planned is in
[`CURRENT_STATE.md`](CURRENT_STATE.md); milestones are in [`ROADMAP.md`](ROADMAP.md).
