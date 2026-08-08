# Learning Paradigms & Memory (design/study track)

**Status: gated design/study proposal for review. Docs-only. No training code, dependency, model,
dataset, GPU, or research action is part of this document.** Like Behavior Lab and the MoE overlay,
this track is **not a product default and does not define CorpusStudio's identity, navigation, or
ordinary workflow** - it captures forward direction. Every capability advances the `SupportLevel`
ladder only on **measured evidence**, and all foundational contracts stay **dense-safe and
MoE-compatible** (see [`TRAINING_SYSTEMS_ARCHITECTURE.md`](TRAINING_SYSTEMS_ARCHITECTURE.md),
[`PRODUCT_VS_RESEARCH.md`](PRODUCT_VS_RESEARCH.md)).

> **Reconciled 2026-08-08.** Reward-model *training* has since shipped as a `workload_verified` **PRODUCT**
> capability (RL slice S5a, #833-#838) - a `RewardRunner` lane that trains a QLoRA SEQ_CLS scalar score head
> under a Bradley-Terry loss, promoted on held-out pairwise ranking accuracy. That is a training capability
> in the product S-series, **not** the gated on-policy-RL work here. Section 1 below is updated to reflect it
> and now carries the concrete, **reviewable** on-policy-RL (S5b/S5c) design that a trained reward model
> unblocks. On-policy RL (rollout + PPO/GRPO) stays gated: **no implementation PR until this design is
> reviewed**.

CorpusStudio is a model-development platform: it **builds, trains, evaluates, and releases** models and
model-powered systems. This track extends that surface to three learning/memory paradigms it does not
yet architect. What exists today is captured honestly in each section so the gap is explicit, not
implied-away.

## 1. Reinforcement learning as an architected mode (not just labels)

**Today (reconciled 2026-08-08):**

- **Offline preference optimization (DPO) is `workload_verified`** - the `dpo_qlora` objective +
  `ResolvedPreferenceExecutionConfiguration` + a `PreferenceRunner` lane that trains a QLoRA adapter over a
  frozen reference with a sequence-chunked log-prob, promoted by a measured GPU bring-up.
- **Reward-model training is `workload_verified`** (RL slice S5a, #833-#838) - the `reward_model` objective +
  `ResolvedRewardExecutionConfiguration` + a `RewardRunner` lane that trains a QLoRA **SEQ_CLS** scalar score
  head under a Bradley-Terry loss; the promotion gate is **held-out pairwise ranking accuracy**. So a reward
  **source** now exists as a trainable, supervisor-admittable product artifact.
- **On-policy RL is still NOT built.** There is no rollout / generation loop, no reward-model **serving**
  (using a trained reward model to score fresh completions), and no on-policy driver. `TaskType.grpo` is
  declared-only; there is no `ppo` task/objective, no experience/rollout contract, and no KL/entropy
  controller. That gap is what S5b/S5c below design.

### S5b/S5c design (reviewable proposal - no implementation PR until approved)

On-policy RL is **not** dense-QLoRA-SFT, so it takes its OWN **backend-scoped** resolved-execution config
(P0d, #484 - the same sibling-seal mechanism SFT / DPO / pretraining / full-finetune / **reward** already
use), and **never mutates the byte-locked SFT seal**. It reuses, rather than redesigns: the reward model
(S5a) as the reward source, the frozen-reference-via-`disable_adapter` pattern (DPO), the sealed-optimizer +
gradient/adapter evidence primitives, the admit-at-planning / refuse-at-execution ladder, and (for RLAIF)
Evaluation Studio's judge under the provider policy.

**Execution shape.** A new sibling `ResolvedRolloutExecutionConfiguration` (the 6th resolved-execution
config), carried on `RunPlan` via the existing "exactly one execution authority" tuple, behind a new
`on_policy_rl` execution-variant kind - admitted at planning, refused at execution until a measured run
promotes it (exactly the ladder reward followed).

**Additive contracts (all backend-scoped):**

- **`RolloutSpec`** - the generation phase: sampling params (temperature, top_p, `max_new_tokens`), the
  number of rollouts per prompt (the **GRPO group size**), and the decode policy. Generation reuses the
  sanctioned decode path - never an unsanctioned generation path.
- **`ExperienceSource`** - the on-policy experience buffer: completions are generated FRESH from the current
  policy each iteration (a streaming source distinct from a static dataset), so this ties to the **G2
  streaming data-cursor** gap (shard/offset/consumed/mixture-RNG). On-policy => regenerated, not replayed;
  off-policy replay is a later variant.
- **`RewardSourceRef`** - a hash-pinned reference to what scores each completion: (a) the
  **`workload_verified` `reward_model`** served for scoring [the primary path, reusing S5a], (b) rule /
  verifier rewards (`process_supervision` / `verifier_training`), or (c) RLAIF (an LLM judge reusing
  Evaluation Studio's judge under the provider policy). Fail-closed: a reward source that is not itself
  admittable (e.g. a reward model below `workload_verified`) refuses.
- **`StabilityController`** - sealed on-policy controls: KL-to-reference penalty (coefficient + target),
  entropy bonus, advantage normalization, and the PPO clip range (or GRPO group-relative advantage). Sealed
  like any execution field; no silent defaults.
- **`PolicyOptimizationSpec`** - **GRPO first** (group-relative advantage, NO critic - cheaper, fits the
  12 GB envelope like DPO/reward did), then **PPO** (adds a critic/value head). Reference = the frozen base
  via `disable_adapter` (the DPO pattern); reward = the `RewardSourceRef`.

**Evidence + promotion gate.** A new sibling `RolloutExecutionEvidence` (mirroring
`RewardExecutionEvidence`): per-iteration rollouts generated, mean/spread reward from the source, **KL
divergence to the reference** (must stay within the sealed bound - a blown-up KL is reward-hacking /
collapse), entropy, advantage stats, and finite policy-update steps. The **promotion GATE is a MEASURED
mean-reward lift on a HELD-OUT prompt set while KL stays within bound** - never a falling loss, and never
training-reward alone (which reward-hacks). That held-out-reward-under-bounded-KL signal is the on-policy
analog of reward's held-out pairwise accuracy.

**Boundary posture.** A PRODUCT capability (STANDARD tier), opt-in, **dense + MoE-safe**, backend-scoped;
**not** a product default, navigation, or identity, and **not** the IEEE sealed-research overlay. It
advances the `SupportLevel` ladder only on measured evidence.

**Slice sequence (each evidence-gated, mirroring the reward vertical):**

1. **S5b-1** contracts + enums (`RolloutSpec` / `ExperienceSource` / `RewardSourceRef` /
   `StabilityController` / `PolicyOptimizationSpec` + the `on_policy_rl` variant) at `contract_validated`;
2. **S5b-2** resolver + planner routing (admit-at-planning, refuse-at-execution);
3. **S5b-3** execution-evidence contracts + tracker (rollouts / reward / KL / advantage);
4. **S5b-4** the worker: rollout generation -> reward scoring (served reward model) -> **GRPO** update,
   backend-scoped, reusing the frozen-reference + sealed-optimizer primitives;
5. **S5b-5** runner + supervisor re-verify + the `on_policy_rl` lane;
6. **S5b-6** GPU proof (a small model, GRPO, no critic) -> `on_policy_rl` `workload_verified`.
7. **S5c** then adds **PPO** (critic/value head) and the **RLAIF** reward source (Evaluation Studio judge).

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
  - **L1** RL as an architected mode (rollout + reward + KL, backend-scoped): the **reward source** is
    shipped (`reward_model` `workload_verified`, S5a); the on-policy **rollout + GRPO/PPO** half is designed
    in Section 1 (S5b/S5c) and stays gated until that design is reviewed;
  - **L2** memory-augmented training (episodic/long-term memory, retrieval-augmented);
  - **L3** memory synthesis / consolidation ("dreaming") + the long-horizon memory eval profiles.
