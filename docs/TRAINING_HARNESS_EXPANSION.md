# Training Harness Expansion — the sequenced build plan

**Status: design / architecture-review gate.** Consolidates the full training-method + memory menu into
one ordered backlog with the *exact contract delta* and *evidence gate* per step. Companion detail:
[`LEARNING_PARADIGMS_AND_MEMORY.md`](LEARNING_PARADIGMS_AND_MEMORY.md) (RL / memory / dreaming = L1/L2/L3),
[`PRETRAINING_ARCHITECTURE.md`](PRETRAINING_ARCHITECTURE.md), [`MOE_ARCHITECTURE.md`](MOE_ARCHITECTURE.md),
[`TRAINING_BACKEND_REGISTRY.md`](TRAINING_BACKEND_REGISTRY.md), and the P0-P7 sequence in the
`training-systems` skill. **No implementation PR merges before this architecture is reviewed.**

This revision (2026-08-04) folds in an external architecture review. Its two load-bearing corrections -
**extract a shared objective-worker lifecycle (S0)** and **split RL into reward-modeling / rollout /
optimization** - are adopted below, because without them the harness drifts into a collection of
*individually sealed but operationally disconnected* trainers with duplicated, inconsistent safety logic.

## Where we are
- **dense QLoRA-SFT** is the one `WORKLOAD_VERIFIED` tuple. **Checkpoint/resume** has a CPU
  reference-loop **bitwise-equivalence proof** (seal/verify/admit + worker save/restore/coordinator); the
  production `SFTTrainer` loop is **not yet wired** to the checkpoint coordinator and a **7B GPU resume is
  a gated future step** (`docs/CHECKPOINT_RESUME_PLAN.md`) - do not read the CPU proof as a production 7B
  claim.
- **S1 admission gate** shipped (#775): a task with no executable variant is refused fail-closed. **S2
  control-plane** merged (#778 `PreferenceDataPolicy`; #779 the `dpo_qlora` objective +
  `ResolvedPreferenceExecutionConfiguration`); `preference_dpo` is declared at **`contract_validated`** -
  admitted-then-refused-at-execution, **not executable**.
- **S2 offline DPO worker**: the sequence-chunked log-prob is an **exploratory** primitive. An
  uncommitted scratchpad prototype trained 4B QLoRA DPO at seq 4096 (~9.99 GiB, where trl / off-the-shelf
  liger cap at ~seq 1024) - **exploratory evidence, not a recorded or sealed result** (it is absent from
  `HOST_STATE.md`/`HANDOFF.md` and `preference_dpo` is not `WORKLOAD_VERIFIED`). A sealed run + milestone
  wheel + promotion remain the gated milestone.

The `SupportLevel` ladder is **tuple-scoped**: the one `WORKLOAD_VERIFIED` tuple is `dense_qlora_sft`.
DPO is **expressible and admitted-fail-closed** but its execution is **unproven**; every paradigm below is
`DECLARED`, not `WORKLOAD_VERIFIED`.

## Build-out model (how we work)
- **DEV mode** - the repo + a dev env, iterating on every method with **no pinned worker wheel per code
  change**.
- **Pin a worker wheel + seal an env ONLY at** (a) a **promotion** (capturing `WORKLOAD_VERIFIED`
  evidence for a specific tuple) or (b) a **product release** carrying a batch of methods. Versions
  (v10, v11, ...) track releases/evidence-milestones, **never per training method**.

## S0 - Objective-worker lifecycle kernel (extract + enforce; retroactive)
**Not greenfield.** The objective-generic *contracts* already exist - `RunPlan`,
`ResolvedExecutionConfiguration`, `RunEvent`, `ArtifactManifest` (two-tier + `reload_verified`),
`CheckpointManifest`, `FailureRecord`, worker-protocol-2.0. What is missing is the **enforced worker
INTERFACE + shared preflight/evidence/checkpoint helpers every method's worker MUST call**, so safety
logic (truncation refusal, evidence, checkpoint identity-binding) is *not re-wired per method* - the exact
risk already seen wiring the DPO truncation guard by hand.

- **Delta:** an `ObjectiveWorker` **Protocol** (the method surface) driven by an **enforced template**
  that CALLS the mandatory shared helpers - a structural Protocol proves the methods *exist*, it cannot
  prove a worker *invokes* the safety helpers, so enforcement lives in a template-method runner
  (`run_objective(worker)`) that owns the shared preflight/evidence/checkpoint calls, plus **conformance
  tests** that assert each variant actually goes through it. Each variant supplies its own method-specific
  state; the lifecycle order is:
  `validate_request -> verify_inputs -> prepare_data -> estimate_resources -> allocate_models ->
  initialize_state -> train_step -> evaluate -> checkpoint -> resume -> export -> reload_verify ->
  emit_evidence`.
- **Contracts (mostly ALIGN/extract existing, not net-new):** `ObjectiveExecutionSpec`,
  `ObjectiveRuntimeState`, `ObjectiveCheckpointState`, `ObjectiveEvidenceBundle`, `ObjectiveArtifactSet`,
  `ObjectiveFailureClassification`, `ObjectiveWorkerCapabilities`.
- **Objective-generic checkpoint state:** for **immutable** dependencies bind an *identity*, not the
  bytes (a frozen reference or teacher model is an identity - a hash/ref - never checkpointed weights).
  But **mutable learned state must be checkpointed as contents** (or a content hash + backing store) - a
  version tag and cursor cannot restore it: S6's memory store, an experience buffer, and optimizer state
  are learned/mutable and must round-trip exactly.

| Method | Additional state the lifecycle must carry |
|---|---|
| SFT | sampler cursor, adapter state |
| DPO | reference binding/cache, pair cursor |
| Pretraining | per-rank stream cursor, packing buffer |
| Distillation | teacher binding/cache |
| Reward model | scoring head, calibration state |
| PPO | policy, reference, reward, critic, rollout state |
| GRPO | policy, reference, grouped rollout state |
| Memory training | memory-store **contents** (mutable learned state) + retrieval cursor |

**Gate:** every execution variant proves the same lifecycle on cpu_toy. Rewiring the existing SFT worker
to go through the enforced template **changes its source and therefore mints a new wheel** - "byte
-unchanged" is not achievable and is the wrong bar. The right bar is **behavior-preserving**: the SFT
sealed `ResolvedExecutionConfiguration` stays byte-identical (the contract does not change) and the
re-sealed worker reproduces the SFT evidence (bitwise where already proven) before promotion carries over.

## Sequenced backlog

| # | Slice | New contract/system | Evidence gate (tuple) |
|---|---|---|---|
| **S0** | Objective-worker lifecycle kernel | `ObjectiveWorker` Protocol + shared helpers + objective-generic checkpoint/evidence | every variant proves the same lifecycle on cpu_toy |
| **S1** | Admission gate + backend-scoped execution | fail-closed variant admission + `resolved_execution` variants (P0) | ✅ admission gate shipped (#775); `preference_dpo` at `contract_validated` (#778/#779) |
| **S2a** | **Offline DPO** vertical slice | preference-pair data policy + sequence-chunked DPO worker (reference, no rollout) | ⛔ gated: worker unwired, `preference_dpo` at `contract_validated`. Gate = DPO tuple `WORKLOAD_VERIFIED` on 4B seq 4096 (sealed run + wheel) |
| **S2b** | **Preference family** | IPO / KTO / ORPO - each its OWN objective + provenance (never a loss string under a `dpo` seal). **KTO/unpaired methods need their OWN data contract** - `PreferenceDataPolicy` is prompt/chosen/rejected PAIRS; KTO takes unpaired (prompt + single response + binary desirable/undesirable label), so add an `UnpairedPreferenceDataPolicy` rather than forcing it into pairs | each variant's tuple |
| **S3** | **Pretraining path** | **tokenizer lifecycle** (freeze before tokens, content-hash, token-accounting invalidation) + streaming per-rank data cursor + token budget + continued-pretraining data policy -> dense pretraining | continued-pretrain on a small corpus |
| **S4a** | **Distillation** workers | teacher-serving reference + logit/sequence/rationale distillation loss | KD on 4B (teacher 7B) |
| **S4b** | **Synthetic data + self-training** | teacher generation, best-of-N, rejection sampling, hard-negative mining, self-consistency, verifier/judge filtering, curriculum, iterative student<->teacher - with **full per-row provenance** (generator+revision, settings, prompt source, **the filtering judge/verifier's own identity + version + config**, its decision, rejected candidates, filtering reasons, license/policy, dataset version, contamination status) - binding only a decision without the model that made it makes a filtered set unauditable and unreproducible | a filtered synthetic set feeds S4a/S2 with intact provenance |
| **S5a** | **Reward + verifier modeling** | pairwise / scalar-pointwise / process / outcome reward models, generative verifiers, rule composition, ensembles, uncertainty. Artifacts: weights/adapter + **score head + calibration + input formatter + output scale/direction + provenance**. Eval: pairwise accuracy, tie handling, **calibration, length/verbosity/position bias, reward saturation, OOD, reward-hacking, disagreement vs human/judge** | RM tuple with held-out ranking accuracy + calibration evidence |
| **S5b** | **Rollout + experience plane** (+ value/critic) | `RolloutRequest` / `RolloutRecord` / `RewardRecord` / `TokenLogprobRecord` / `AdvantageRecord` / `ExperienceBatch` / `ExperienceBufferState` / `RolloutWorkerManifest` / `EnvironmentReference`; an explicit **stale-experience policy** (a rollout from an old policy is NOT silently on-policy); the **environment protocol** `reset(seed) / step(action) / snapshot() / restore()` for agentic/tool-use RL; the **value/critic** head+loss+state | an experience batch binds policy version + old/reference logprobs + rewards + validity/staleness |
| **S5c** | **RL optimization** | the update ALGORITHM only - PPO / GRPO / REINFORCE / ... (policy + reference + reward + optional critic) + KL/entropy controller, consuming S5b experience + S5a rewards. **RLHF vs RLAIF is NOT an optimizer** - it is the reward's *feedback source* (human vs AI), a provenance axis on the S5a reward / S5b experience, orthogonal to which algorithm runs (any optimizer x either source) | GRPO on 4B with a rule reward |
| **S6** | **Memory-augmented training** (L2) | memory-topology descriptor (MoE-safe) + retrieval-augmented data policy + read/write objective | retrieval-augmented tuple + held-out memory-use eval |
| **S7** | **Memory synthesis / "dreaming"** (L3) | background memory-consolidation pipeline (synthesize memory *state* from interaction logs) + long-horizon memory eval profiles | temporal-recall / staleness-resistance eval |
| **S8** | **Continual + incremental learning** (anti-forgetting) | replay/rehearsal, domain/task-incremental, adapter-per-domain + composition, regularization against old behavior, data-mixture refresh. Measure **forward/backward transfer, forgetting-per-capability, base regression, behavior/calibration drift, replay coverage** | **new-domain gain must NOT promote unless retained capabilities stay above declared regression floors** |
| **S9** | **Multi-stage training pipelines** | one `TrainingPlan` -> a DAG of **stages**, where a stage lowers to a `RunPlan` (training) **OR to a non-training operation** (eval, merge, quantize, export, release) - `TaskType`/`RunPlan` has no release/eval op, so the pipeline node is a `PipelineStage` union, not "every node is a `RunPlan`". Plus artifact handoff, immutable parent-child lineage, stage-local retry/rollback/branching, stage-specific env/data, **acceptance gates + stop-on-regression**, pipeline resume, branch comparison | e.g. ContinuedPretrain -> SFT -> RM -> PPO -> SafetyEval -> Release runs end to end with gated handoff |

## Orthogonal (land opportunistically; each a registry entry / track, not a new backend id)
- **Update methods** - DoRA, IA3, prefix/prompt tuning, full-parameter fine-tuning.
- **Distributed / offload backends** - DeepSpeed / FSDP / CPU / NVMe (needed by S3/S5 at scale).
- **MoE** - single-device (P6) -> distributed / expert-parallel (P7).
- **Model merging** - TIES / DARE. **QAT + PTQ.**
- **Experiment search plane** - grid / random / Bayesian / successive-halving-pruning + **seed
  replication + confidence intervals + budget-based trial admission + trial ranking / Pareto**. The
  `SupportLevel` ladder gives the *vocabulary*; the sweep engine gives the *statistical evidence*
  (**one successful execution != repeated != statistically stable != production-supported**).
- **Export / serving validation** - adapter merge/unmerge, full-checkpoint export, GGUF / AWQ / GPTQ,
  tokenizer bundling, **generation-config sealing**, inference-runtime compatibility, reload verification,
  deterministic smoke generation, pre- vs post-export eval parity, local serving probe, license manifest,
  model card. *(The artifact registry, weight card, promote gate, `reload_verified`, `train-merge`,
  model-card, and license-fail-closed already exist - this is breadth + serving parity.)*
- **Observability / failure injection.** **Security + method-specific data governance** - bind the
  existing dedup/leakage-split/PII/debt-ledger machinery per method: contamination, exact+semantic dedup,
  **reversed preference pairs, preference ties/ambiguity**, annotator/rubric + teacher-label provenance,
  source proportions, curriculum order, sample weights, hard-negative provenance. **Unlike the rest of
  this section, a method's governance is NOT purely opportunistic - the governance checks a method depends
  on are a PROMOTION PREREQUISITE for that method** (a slice cannot reach `WORKLOAD_VERIFIED` while its
  contamination/dedup/leakage-split evidence is unwired), so S2/S4/S5 carry their governance to their gate.

## Model-family + output-head scope (declare it)
The first-party worker is currently **CausalLM + `adapter_peft`**. Reward/value/embedding/reranker/
multimodal heads are **not** ordinary `adapter_peft` CausalLM artifacts. `ModelTaskClass` today recognizes
`causal_lm / masked_lm / seq2seq_lm / classification / embedding / reranker / reward_model / vision /
speech / multimodal` - so RM/embedding/reranker/classification/encoder-decoder have a *family label* to
build on, but **there is no `value`/`critic` member: the value/critic head is a NET-NEW contract family**
(head + loss + checkpoint state), added in S5b, not an existing one to "catch up". For the recognized
families, **worker execution + artifact semantics catch up per slice** (reward heads in S5a, encoder
-decoder / classification / embedding / reranker / multimodal as planned execution variants).

## The per-method workload-verification ladder (4 gates)
1. **Pure math** - toy tensors validate the loss + gradient equations.
2. **CPU tiny-model** - full formatting, masking, forward, backward, export, reload.
3. **GPU bounded** - the real model/precision/seq on a fixed input set, multiple optimizer steps, real
   peak memory.
4. **Platform** - resolver -> sealed `platform-run` -> checkpoint/resume (where enabled) -> export ->
   reload verification -> **held-out eval** -> milestone wheel -> `WORKLOAD_VERIFIED`.

**A falling training loss is never the promotion gate** - promotion requires a *measured behavioral
improvement on held-out data*, or the evidence must state plainly that the workload proved *execution
only*.

## Invariants (unchanged)
- **Tuple-scoped promotion on measured evidence only** - proving one tuple never implies another; and one
  run is not promotion (require replicated / statistically-stable improvement + hardware portability).
- **Dense-safe / MoE-safe foundational contracts** - `ModelDescriptor`, `TrainingObjective`, `RunPlan`,
  `ArtifactManifest`, checkpoint, telemetry, evaluation stay MoE-compatible.
- **Product vs research** - all product/exploratory; the IEEE sealed ladder is a separate opt-in overlay.
- **Honesty** - "installed != supported"; predicted fit is never `NATIVE_SAFE`; **no silent target
  truncation**; **no silent semantic defaults** (every log-prob / masking / reduction semantic is sealed,
  not an implementation default); single-writer datasets; no-shell argv.

## Recommended entry order + the two highest-priority changes
**S0 (worker-lifecycle kernel) first** as the enforcement backbone, then finish **S2a (DPO)** to
`WORKLOAD_VERIFIED`, then **S5's three-way split** (S5a reward -> S5b rollout -> S5c optimization) before
any RL optimizer. The single highest-priority structural change is **splitting RL into reward-modeling /
rollout / optimization**; the second is **extracting the shared objective-worker lifecycle** - together
they keep the harness one enforceable system rather than disconnected sealed trainers.
