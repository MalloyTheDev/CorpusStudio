# Training Objectives

CorpusStudio has a versioned, dependency-light `TrainingObjective` registry that describes **what a
run is meant to optimize**, independently from the backend that might implement it. A registry entry
is not a support claim: a backend manifest is only a static declaration, and a compatible
`CapabilityReport` must explicitly prove the objective capability before the compatibility report can
say `verified_compatible`.

## Built-in catalog

The registry contains 29 sealed definitions:

- language-model training: `pretraining`, `continued_pretraining`, `full_parameter_sft`, `lora`,
  `qlora`, `other_peft`, `chat_tuning`, `completion_only`, `response_only_loss`, and `tool_use`;
- preference and reward: `dpo`, `ipo`, `kto`, `orpo`, and `reward_model`;
- distillation and supervision: `knowledge_distillation`, `sequence_distillation`,
  `logit_distillation`, `rationale_distillation`, `process_supervision`, and `verifier_training`;
- task-specific training: `embedding`, `reranker`, `classifier`, and `multimodal`;
- non-training operations: `evaluation_only`, `merge_only`, `conversion_only`, and
  `quantization_only`.

Every definition has an `objective_version` and a canonical SHA-256 `objective_hash`. The hash seals
the complete definition except the hash field itself. Catalog loading rejects duplicate id/version
pairs and invalid seals.

## What an objective records

Each `TrainingObjective` explicitly defines:

- accepted dataset schema ids and versions, required fields/types, dataset format, and whether the
  schema is built-in, structural, or only planned;
- label construction, loss-mask construction, and separately keyed loss components;
- compatible model task classes and execution kinds;
- adaptation methods and a semantic update policy;
- backend task/loss/adapter/quantization requirements plus objective capability tokens;
- expected artifacts, exact/fork/restart resume semantics, evaluation requirements, qualitative
  hardware implications, limitations, and independent definition/implementation/hardware evidence.

The `TaskType` enum remains only a coarse bridge. It is not the objective identity.

A loss component's `default_weight` is optional by design. A numeric value is the objective's
declared default coefficient; `None` means the objective intentionally leaves that coefficient
unspecified. Omitted and explicit `None` values therefore deserialize identically and cannot silently
become `1.0` across an `exclude_none` protocol round trip. Built-in objectives that require unit
weight record `1.0` explicitly.

## MoE-safe update semantics

The objective contract does not assume every parameter is active, resident, exposed, or updated.
Update scopes can distinguish shared parameters, adapters, router, selected experts, all experts,
embeddings, heads, and projectors. Router-only and expert-selective policies are independently
representable. Expert-scoped updates require stable expert identity and per-expert exposure tracking;
the contract also carries optimizer-clock, starvation-gate, and routing-collapse requirements.

This is semantic intent only. Device placement, residency, prefetch, storage tiering, and physical
scheduling remain future `RunPlan` responsibilities.

## Compatibility evidence

`ObjectiveCompatibilityReport` keeps three independent axes:

- **dataset**: accepted schema id/version plus required declared fields/types; objectives with more
  than one input stay unverified until role-keyed evidence is available;
- **model**: `ModelDescriptor` task class, topology, tokenizer/head evidence, trust policy, and any
  router/expert requirements;
- **backend**: `BackendManifest` task/loss/adapter/objective declarations, optionally intersected with
  a measured, matching-backend-version `CapabilityReport`.

Statuses are `declared_compatible`, `verified_compatible`, `incompatible`, `unverified`, and
`not_applicable`. Any contradiction makes the overall result incompatible; any evidence gap makes it
unverified. A static backend match can earn only `declared_compatible`. Package installation, config
generation, broad task support, or a lower-level kernel probe never proves an objective.

The current built-in backends statically declare causal-LM SFT plus LoRA/QLoRA capability tokens.
Current host capability probes do not claim an end-to-end objective execution token, so supplying one
of today's reports will remain unverified at that axis until a dedicated objective probe is added.

## CLI

```powershell
corpus-studio training-objectives
corpus-studio training-objectives qlora --json
corpus-studio training-objective-check qlora --schema instruction --backend corpus_studio --json
corpus-studio training-objective-check logit_distillation --schema logit_distillation --schema-version 1.0.0 --fields input_ids:list,teacher_logits:object
```

Built-in schemas automatically provide their version and fields. `--schema-version VERSION` and
`--fields name:type,...` provide explicit evidence for non-built-in shapes. `--model-descriptor FILE`
adds structural model evidence. `--capability-report FILE` adds measured backend evidence and must pin
the selected manifest's backend version. The checker does not predict hardware fit; only a measured
run can prove fit.

## Deliberate non-goals of this slice

- No objective is added to `RunPlan` yet; that belongs to the planned RunPlan-expansion phase.
- No DPO/reward/distillation/MoE trainer is implemented by catalog presence.
- No model loading, training, network access, physical scheduling, parameter-count production, or
  checkpoint expansion is performed here. The later generalized trace contract now ships separately
  as `TraceRecord`; catalog presence still does not implement process/tool/verifier trainers.
- The existing advisory `training-compat` command remains separate.
