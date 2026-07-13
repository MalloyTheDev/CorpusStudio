# CorpusStudio Platform Contracts (v1.0.0)

The **language-neutral boundary** between the (Python → Rust) platform core, the Python AI backend
worker(s), and the UI shell (Avalonia now, Tauri later). These JSON Schemas (draft 2020-12) let a
Rust core, a C#/Avalonia client, or a TypeScript/Tauri client validate and generate the same
messages the Python engine does.

## Generated — do not hand-edit

Every `*.schema.json` in this directory is **generated** from the pydantic models in
[`engine/corpus_studio/platform/`](../../engine/corpus_studio/platform) — those models are the
single source of truth. Regenerate after changing a contract:

```
corpus-studio platform-schemas --out docs/contracts
# or: python -m corpus_studio.cli platform-schemas --out docs/contracts
```

`index.json` lists all **26 root contracts** and their shared `contract_version`.

## The contracts

| Contract | Role |
|---|---|
| `ProjectManifest` | Workspace descriptor (identity, schema, splits, registry pointers). |
| `DatasetManifest` | Dataset version identity **+ lineage** (source → transform → teacher/prompt/seed → hashes) + token stats. |
| `ModelDescriptor` | Static model identity, component-scoped representation, scoped parameter counts, topology/trust findings, portable file inventory, and independent verification axes. |
| `ParameterAccountingReport` | Hash-sealed logical/active/resident/touched/updated/exposed evidence with stable coordinate scopes, structured windows, explicit gaps/conflicts, and model/run/artifact/evaluation lineage. |
| `TokenizerDescriptor` | Static tokenizer identity, base/added/effective vocabulary, special tokens, chat-template hash, inventory/trust findings, and model-compatibility evidence. |
| `TrainingObjective` | Hash-sealed, backend-independent objective semantics: dataset/labels/masks/losses, model/update/backend requirements, artifacts, resume/eval/hardware implications, and verification. |
| `ObjectiveCompatibilityReport` | Independent dataset/model/backend evidence axes; static declarations are distinct from functionally verified compatibility. |
| `EnvironmentProfile` | Hashable host + software **signature** (OS, residency model, GPUs, driver/runtime, package locks). |
| `StorageProfile` | Non-destructive storage topology plus per-role safe-spill suitability and risk evidence. |
| `PythonRuntime` | A bounded-probed Python interpreter candidate and recipe compatibility. |
| `EnvironmentRecipe` | Declarative dependency-layer/backend environment recipe; declaration is not support proof. |
| `DependencyResolution` | Hash-sealed preview of exact no-shell install argv, indexes, requirements, and estimates. |
| `EnvironmentInstallation` | Durable bounded-command journal and installation outcome. |
| `EnvironmentLock` | Immutable interpreter/package/source/hash and accelerator lock evidence. |
| `EnvironmentDescriptor` | Managed environment identity, ownership, state, lock, and verification references. |
| `EnvironmentHealthReport` | Independent import/dependency/functional/hardware/drift health axes. |
| `BackendManifest` | A backend's **static** declaration of what it can do (OS/device/precision/quant/adapter/attn/loss plus placement/offload/parallelism/communication axes, deps + conflicts, known-failure modes, probes). |
| `CapabilityReport` | The **measured** counterpart — per-probe outcomes on a specific host (declared ∩ proven). |
| `RunPlan` | The **immutable, fully-resolved** execution plan (`plan_hash`-sealed), including concrete physical resources, state placements, offload rules, and rank/group bindings; accumulation target is in **supervised tokens**. |
| `RunManifest` | A durable run **instance** + state machine + reconciliation. |
| `RunEvent` | The **streamed telemetry** envelope (stage markers + metrics incl. **dedicated vs shared** GPU memory). |
| `ArtifactManifest` | A produced weight artifact (two-tier integrity, `reload_verified`). |
| `EvaluationResult` | Eval outcome with an explicit **as-served vs raw** distinction + gate verdict. |
| `FailureRecord` | Classified terminal outcome (`FailureTaxonomy`: OOM / KERNEL_STALL / ACCIDENTAL_SPILL / …). |
| `FitClassification` | How a plan fits: `NATIVE_*` / `CONTROLLED_*` / `PLANNED_UNPROVEN` / `ACCIDENTAL_WDDM_SPILL` / `THRASHING` / `FAIL`. |
| `WorkerMessage` | The versioned **core ↔ worker** protocol envelope (immutable RunPlan in → RunEvent stream out). |

## Design stance

- **The silent spill is first-class.** `MemoryMetrics` splits `dedicated_gpu_bytes` vs
  `shared_gpu_bytes`; `MemoryResidencyModel` distinguishes `wddm`/`linux_dedicated`/`unified_memory`;
  the `ACCIDENTAL_*`/`KERNEL_STALL` taxonomies make the WDDM spill and the Blackwell sm_120
  fused-attention deadlock machine-actionable instead of buried in warning text.
- **Supervised-token accounting end-to-end** — `RunPlan.batching.supervised_token_accumulation_target`
  pairs with `RunEvent.metrics.supervised_tokens_per_sec` and `TokenStats.supervised_tokens`.
- **One parameter count is not enough** — `ParameterAccountingReport` keeps coordinate universes and
  windows explicit; stored tensor elements and allocator bytes never silently become logical or
  resident coordinates.
- **Semantic routing is not physical placement** — a `RunPlan` may name expert state locations and
  transfer behavior, but residency cannot silently change the learned route. Planned placement is
  labeled `planned_not_measured`; every requested capability needs both declaration and probe proof.
- **Grounded, not invented** — field names/constraints are taken from the existing engine models
  (see each pydantic class docstring for the model it formalizes).
- **Additive versioning** — a MAJOR `contract_version` bump is breaking; readers reject an unknown MAJOR.
