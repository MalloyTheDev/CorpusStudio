# CorpusStudio documentation

CorpusStudio is a **local-first, end-to-end AI development ecosystem and IDE** covering the
full model lifecycle. Its surface is organized into **seven co-equal product areas** - see
[`PRODUCT_AREAS.md`](PRODUCT_AREAS.md) for the canonical map. This index groups every doc in
`docs/` under the area it serves, so the 7-area picture stays clear as the set grows.

> New here? Start with [`PRODUCT_SPEC.md`](PRODUCT_SPEC.md) (what CorpusStudio is), then
> [`CURRENT_STATE.md`](CURRENT_STATE.md) (what is actually built), then the area you care about.
> `CLAUDE.md` / `AGENTS.md` / `HANDOFF.md` at the repo root remain the agent contract + session state.

## Product & architecture

- [`PRODUCT_SPEC.md`](PRODUCT_SPEC.md) - product identity, principles, non-goals (incl. Local-First Design).
- [`PRODUCT_AREAS.md`](PRODUCT_AREAS.md) - the canonical seven-product-area map.
- [`PRODUCT_VS_RESEARCH.md`](PRODUCT_VS_RESEARCH.md) - the product vs research boundary.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) - engine / platform / UI, and the Rust-core target.
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) - the forward plan.
- [`ROADMAP.md`](ROADMAP.md) - staged milestones and exit criteria.
- [`CURRENT_STATE.md`](CURRENT_STATE.md) - authoritative "what works today".

## 1. Data Studio

- [`SCHEMA_SYSTEM.md`](SCHEMA_SYSTEM.md) - schema-driven datasets (incl. copyable row examples).
- [`DATA_MODEL.md`](DATA_MODEL.md) - the on-disk data model (incl. Dataset Cards).
- [`DATASET_TESTING_WORKFLOW.md`](DATASET_TESTING_WORKFLOW.md) - dataset testing (incl. Dataset Splitting).
- [`IMPORT_EXPORT.md`](IMPORT_EXPORT.md) - import / export paths.
- [`VERSIONING.md`](VERSIONING.md) - dataset version history & lineage.
- [`DEBT.md`](DEBT.md) - the graded dataset-debt ledger.
- [`GATES.md`](GATES.md) - pre-export gates (schema / quality / PII / leakage).
- [`WORKFLOWS.md`](WORKFLOWS.md) - dataset-task walkthroughs.
- [`TRACE_RECORDS.md`](TRACE_RECORDS.md) - the TraceRecord / Trace Studio foundation.
- [`AI_ASSIST.md`](AI_ASSIST.md) - review-first AI assistance for authors.

## 2. Training Studio

- [`TRAINING.md`](TRAINING.md) - the training surface overview.
- [`TRAINING_SYSTEMS_ARCHITECTURE.md`](TRAINING_SYSTEMS_ARCHITECTURE.md) - the pluggable training-systems architecture (incl. Training Objectives).
- [`TRAINING_BACKEND_REGISTRY.md`](TRAINING_BACKEND_REGISTRY.md) - the training backend registry.
- [`PRETRAINING_ARCHITECTURE.md`](PRETRAINING_ARCHITECTURE.md) - from-scratch / continued pretraining.
- [`MOE_ARCHITECTURE.md`](MOE_ARCHITECTURE.md) - Mixture-of-Experts (incl. Static MoE Model Inspection + MoE Training Architecture).
- [`PARAMETER_ACCOUNTING.md`](PARAMETER_ACCOUNTING.md) - parameter accounting.
- [`CHECKPOINT_RESUME.md`](CHECKPOINT_RESUME.md) - exact checkpoint + resume lineage.
- [`EFFECTIVE_EXECUTION_CONFIGURATION.md`](EFFECTIVE_EXECUTION_CONFIGURATION.md) - the sealed effective execution configuration.
- [`RUN_PLAN_PHYSICAL_EXECUTION.md`](RUN_PLAN_PHYSICAL_EXECUTION.md) - the RunPlan physical-execution contract.
- [`PLATFORM_RUN.md`](PLATFORM_RUN.md) - running a job through the headless platform.
- [`BACKEND_WORKER_PROTOCOL.md`](BACKEND_WORKER_PROTOCOL.md) - the isolated backend worker protocol.
- [`MODEL_TOKENIZER_CONTRACTS.md`](MODEL_TOKENIZER_CONTRACTS.md) - model & tokenizer descriptors.

## 3. Evaluation Studio

- [`EVALUATION_STUDIO.md`](EVALUATION_STUDIO.md) - evaluation runs and Model Arena (incl. Evaluation Suites).
- [`MODEL_BACKENDS.md`](MODEL_BACKENDS.md) - local model-serving backends.
- [`PROVIDER_POLICY.md`](PROVIDER_POLICY.md) - the fail-closed provider policy.

## 4. Behavior Lab

Design/study only - implementation is gated. See the Behavior Lab entry in
[`PRODUCT_AREAS.md`](PRODUCT_AREAS.md); no shipped docs yet.

## 5. Model & Release Studio

- [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) - the release checklist.
- Artifact inspection, adapter merge, and export are covered in [`IMPORT_EXPORT.md`](IMPORT_EXPORT.md)
  and [`CURRENT_STATE.md`](CURRENT_STATE.md) until this area gets its own deep-dive.

## 6. Environment & Hardware

- [`ENVIRONMENT_MANAGER.md`](ENVIRONMENT_MANAGER.md) - sealed, reproducible managed environments.
- [`HARDWARE_STORAGE_PROFILE.md`](HARDWARE_STORAGE_PROFILE.md) - hardware & storage safe-spill profiling.
- [`HOST_STATE.md`](HOST_STATE.md) - verified facts for the current native-Linux host.
- [`RUNNING_ON_LINUX.md`](RUNNING_ON_LINUX.md) - the native-Linux training runbook.

## 7. Evidence & Experiments

- [`MEASUREMENT_HARNESS.md`](MEASUREMENT_HARNESS.md) - the platform measurement harness (Section 11).
- The opt-in IEEE native-Linux research overlay lives under `research/ieee-linux-training/`.

## Workspace, UI & developer reference

- [`WORKSPACE_SYSTEM.md`](WORKSPACE_SYSTEM.md) - the IDE-style universal workspace.
- [`AVALONIA_MIGRATION_PLAN.md`](AVALONIA_MIGRATION_PLAN.md) - the cross-platform / Avalonia plan (superseded by #545, kept as history; incl. the Cross-Platform Assessment).
- [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) - the developer guide (incl. Development Setup).
- [`CLI_REFERENCE.md`](CLI_REFERENCE.md) - the engine CLI reference.

---

_This folder was consolidated from 51 to 42 docs by folding nine narrow sub-topics into their
parent-area doc (each parent keeps the folded content as a clearly-marked "Consolidated from ..."
section); no content was dropped. Add new docs under the area they serve and list them here._
