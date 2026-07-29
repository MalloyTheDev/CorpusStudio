# Learning Paradigms & Memory (design/study track)

**Status: gated design/study proposal for review. Docs-only. No training code, dependency, model,
dataset, GPU, or research action is part of this document.** Like Behavior Lab and the MoE overlay,
this track is **not a product default and does not define CorpusStudio's identity, navigation, or
ordinary workflow** - it captures forward direction. Every capability advances the `SupportLevel`
ladder only on **measured evidence**, and all foundational contracts stay **dense-safe and
MoE-compatible** (see [`TRAINING_SYSTEMS_ARCHITECTURE.md`](TRAINING_SYSTEMS_ARCHITECTURE.md),
[`PRODUCT_VS_RESEARCH.md`](PRODUCT_VS_RESEARCH.md)).

CorpusStudio is a model-development platform: it **builds, trains, evaluates, and releases** models and
model-powered systems. This track extends that surface to three learning/memory paradigms it does not
yet architect. What exists today is captured honestly in each section so the gap is explicit, not
implied-away.

## 1. Reinforcement learning as an architected mode (not just labels)

**Today:** RL exists only as *declared objectives* - `ObjectiveKind.preference_optimization` /
`reward_modeling`, `TaskType.grpo` / `preference` / `reward` - with **no training architecture behind
them**. There is no rollout loop, reward-model serving, or on-policy driver.

**Proposed additive architecture:**

- **On-policy training loops**: PPO, GRPO (group-relative), and the RLHF / RLAIF pipeline
  (policy + reference + reward + optional critic).
- **Rollout / experience collection**: a generation phase (sample completions from the current
  policy) feeding an experience buffer - a streaming data source distinct from a static dataset (ties
  to the P3 streaming data-cursor gap).
- **Reward sources**: served reward models (`reward_modeling`), rule / verifier rewards
  (`process_supervision` / `verifier_training`), and RLAIF (an LLM judge as reward, reusing Evaluation
  Studio's judge under the provider policy - never an unsanctioned generation path).
- **Stability controls**: KL-to-reference penalty, entropy bonus, advantage normalization, clip ranges
  - sealed like any other execution field.
- **Additive contracts**: an experience/rollout source, a reward-source reference, and a KL/entropy
  controller config - all **backend-scoped**. RL is not dense-QLoRA-SFT, so it needs the P0c/P0d
  backend-scoped resolved-execution variants, **never a mutation of the sealed SFT execution seal**.

## 2. Memory-augmented training

**Today:** not represented. No memory objective, topology, or data policy.

**Proposed additive architecture** - models that carry state beyond the context window:

- **Episodic / long-term memory**: external memory stores the model reads and writes; learned
  key-value memory layers; **retrieval-augmented training** (train *with* retrieved context so the
  model learns to use memory, not merely be prompted with it).
- **Topology impact**: memory is a new axis alongside dense/MoE - contracts must stay MoE-safe (a
  memory-augmented MoE is valid). Parameter accounting gains a memory-residency coordinate, consistent
  with the existing `N_logical`/`N_active`/`N_resident` model.
- **Additive**: a memory-topology descriptor + an objective/label variant for memory read/write
  supervision.

## 3. Memory synthesis & consolidation ("dreaming")

**Reference: OpenAI's "dreaming" memory system** (introduced April 2025 as Dreaming V0; Dreaming V3 in
2026). Dreaming is a **background process that synthesizes and curates a model's memory *state* from
interaction history**, as opposed to explicit "saved memories" written during a single conversation.
It keeps memory **fresh and non-stale** - e.g. revising "going to Singapore in July" into "went to
Singapore in July 2026" as time passes - and captures context that occurs naturally, without an
explicit "remember this." OpenAI reports large lifts across the 2024 -> 2026 systems on three axes:
**carrying-forward-context / factual recall** (41.5% -> 82.8%), **preference adherence**
(31.4% -> 71.3%), and **staying-correct-over-time** (9.4% -> 75.1%).

**Today:** not represented.

**For a model-development platform, "dreaming" is a memory-consolidation pipeline to build, train, and
- most importantly - evaluate:**

- **The synthesis pipeline**: a background / offline process that consumes an interaction log and emits
  a synthesized, deduplicated, **time-aware** memory state - a *data + system* capability, distinct
  from gradient training. The synthesizer may itself be a trained / fine-tuned model.
- **Training the synthesizer**: fine-tune or RL-train (Section 1) the process that decides what to
  remember, merge, revise, or expire.
- **Memory evaluation (the key deliverable; ties to Evaluation Studio)**: new eval profiles mirroring
  OpenAI's axes - **temporal factual recall**, **preference / constraint adherence**, and
  **staleness resistance / staying-correct-over-time** (does the memory correctly age "next Saturday"
  into the past?). These are **long-horizon, multi-session** evals the current single-turn eval
  surface does not cover.
- **Honesty**: separate *inference-time memory* (a system / product capability) from *gradient
  training of a memory-user*; a synthesized memory state is evidence about that pipeline, not a
  model-quality claim.

## Boundary & sequence

- **Gated design/study.** No implementation PR until reviewed; each capability advances the
  `SupportLevel` ladder only on measured evidence (a QLoRA-SFT proof never implies RL / memory /
  synthesis).
- **Dense-safe + MoE-safe**; **not a product default or identity** - same posture as
  [`MOE_ARCHITECTURE.md`](MOE_ARCHITECTURE.md) and Behavior Lab.
- **Reuses, does not redesign**: RL leans on the existing objective/verifier enums + backend-scoped
  execution (P0c/P0d); memory leans on the MoE-safe topology/accounting contracts + the streaming data
  cursor (P3); dreaming leans on Evaluation Studio + the trace/data loops.
- **Sequence (after the current P0-P7 training-systems ladder; a proposal for review, each
  evidence-gated):**
  - **L1** RL as an architected mode (rollout + reward + KL, backend-scoped);
  - **L2** memory-augmented training (episodic/long-term memory, retrieval-augmented);
  - **L3** memory synthesis / consolidation ("dreaming") + the long-horizon memory eval profiles.
