# Training Backend Registry

**Status: architecture proposal for review. Docs-only. This CLASSIFIES candidates; it installs, probes,
downloads, or runs nothing.** Part of
[`TRAINING_SYSTEMS_ARCHITECTURE.md`](TRAINING_SYSTEMS_ARCHITECTURE.md).

A **training backend** is a `(FrameworkBackend, OrchestratorAdapter)` binding with a static
`BackendManifest`. This is distinct from the *inference* model backends in
[`MODEL_BACKENDS.md`](MODEL_BACKENDS.md). Each backend runs in its own isolated, verified environment -
never one `[everything]` env (see [`ENVIRONMENT_MANAGER.md`](ENVIRONMENT_MANAGER.md)).

## 1. Manifest fields (declared per backend)

Shipped in `BackendManifest`: `task_types`, `objective_capabilities`, `model_families`, `supported_os`,
`supported_devices`, `required_compute_capability`, `precision_modes`, `quantization_modes`,
`adapter_methods`, `attention_impls`/`attention_kernels`, `loss_impls`, `checkpoint_impls`/
`checkpoint_semantics`, `optimizers`, `offload_strategies`/`placement_*`, `parallelism_kinds`,
`communication_backends`, `export_formats`/`export_compatibility`, `dependency_requirements`/
`dependency_conflicts`, `known_failure_modes`, `capability_probes`, `telemetry_hooks`.

**Additive (new) fields** required by this architecture: `model_topologies` (dense/moe), `license`,
`security_boundaries`, `config_generator`, `launcher`, `progress_parser`, `failure_parser`, and the
split of `FrameworkBackend` vs `OrchestratorAdapter` (today conflated in one backend id).

Every declared capability carries a **support level** (`DECLARED` .. `PRODUCTION_SUPPORTED` / `REFUSED`;
see [`TRAINING_SYSTEMS_ARCHITECTURE.md`](TRAINING_SYSTEMS_ARCHITECTURE.md) §4). **Installed never means
supported.**

## 2. Candidate classification

Each candidate is classified as one of: `FIRST_PARTY_CANDIDATE` (CorpusStudio owns the loop),
`MANAGED_ADAPTER_CANDIDATE` (CorpusStudio drives an external tool it installs + probes),
`CONFIG_EXPORT_ONLY` (emit a valid config/launcher; user runs it elsewhere), `RESEARCH_ONLY` (sealed
overlay / not a product default), `DEFER` (revisit later), `REJECT` (out of scope / unsafe).

These are **proposed** classifications for review; none is installed or probed here. The eventual
**default** for any workload is **evidence-selected** - it must reach `WORKLOAD_VERIFIED` on the actual
host stack. No project is the default by reputation.

### Frameworks

| Candidate | Class | Note |
|---|---|---|
| PyTorch | `FIRST_PARTY_CANDIDATE` | the current reference substrate |
| JAX | `MANAGED_ADAPTER_CANDIDATE` | P7; separate stack, XLA hardware target |
| TensorFlow / Keras (Keras 3 multi-backend) | `DEFER` | reconsider via Keras-3 config export |
| MLX | `MANAGED_ADAPTER_CANDIDATE` | P7; Apple-Silicon / Metal hardware target |

### Scaling / distributed

| Candidate | Class | Note |
|---|---|---|
| DDP | `FIRST_PARTY_CANDIDATE` | PyTorch-native data parallel |
| FSDP2 | `FIRST_PARTY_CANDIDATE` | P5 |
| DeepSpeed | `MANAGED_ADAPTER_CANDIDATE` | P5; ZeRO/offload via external engine |
| Accelerate | `FIRST_PARTY_CANDIDATE` | already the launch seam under TRL |
| Megatron | `MANAGED_ADAPTER_CANDIDATE` | P6/P7; heavy external, EP/TP/PP |
| Ray Train | `DEFER` | multi-node orchestration, later |

### Update methods (PEFT + full)

| Candidate | Class | Note |
|---|---|---|
| full-parameter | `FIRST_PARTY_CANDIDATE` | P3 |
| LoRA, QLoRA | `FIRST_PARTY_CANDIDATE` | the reference path today (via PEFT) |
| DoRA, PiSSA, VeRA, LoKr, LoHa, AdaLoRA, IA3, prefix/prompt tuning | `MANAGED_ADAPTER_CANDIDATE` | exposed via managed PEFT; per-method probe |
| LoftQ, QA-LoRA | `DEFER` | quantization-aware PEFT, later |

### Orchestrators

| Candidate | Class | Note |
|---|---|---|
| native HF / TRL / PEFT | `FIRST_PARTY_CANDIDATE` | the reference orchestrator |
| Unsloth | `MANAGED_ADAPTER_CANDIDATE` | **not** a default until probed + workload-verified; already `REFUSED` on Windows/WDDM |
| torchtune | `MANAGED_ADAPTER_CANDIDATE` | P7 |
| Axolotl | `MANAGED_ADAPTER_CANDIDATE` | P7; also strong `CONFIG_EXPORT_ONLY` |
| LLaMA-Factory | `CONFIG_EXPORT_ONLY` | emit config; optional managed adapter later |
| ms-swift | `DEFER` | reconsider via config export |
| Ludwig | `DEFER` | declarative; config export candidate |

### Post-training / RL

| Candidate | Class | Note |
|---|---|---|
| TRL | `FIRST_PARTY_CANDIDATE` | preference/RL via the reference stack |
| verl | `MANAGED_ADAPTER_CANDIDATE` | P7; scalable RL |
| OpenRLHF | `MANAGED_ADAPTER_CANDIDATE` | P7 |
| NeMo-RL | `DEFER` | heavy NVIDIA stack |

### Hardware targets

| Candidate | Class | Note |
|---|---|---|
| CUDA | `FIRST_PARTY_CANDIDATE` | the verified RTX 5070 host (env-probe level only; not a 7B workload claim) |
| CPU | `FIRST_PARTY_CANDIDATE` | toy / CI / structural path |
| ROCm | `DEFER` | AMD; revisit with a probed stack |
| Metal / MLX | `MANAGED_ADAPTER_CANDIDATE` | with the MLX framework adapter |
| XLA / TPU | `CONFIG_EXPORT_ONLY` | via JAX/Keras export |

### Rejections

Nothing is `REJECT`ed outright at the architecture level; `REJECT` is reserved for a specific
`(stack, host)` combination proven unsafe or incompatible by a probe (fail-closed), e.g. fused flash-SDPA
on Windows/WDDM.

## 3. Evidence-selected default

The reference `(PyTorch, HF/TRL/PEFT)` backend is the only stack with real GPU workload evidence today
(bounded 0.5B smokes; the 7B workload is **not** claimed). Any promotion of another stack to a default
requires that stack to reach `WORKLOAD_VERIFIED` on this host, recorded as evidence - not asserted.

## 4. Security boundaries (new manifest field)

Each manifest declares its trust surface: `trust_remote_code` posture, network access during
train/install, deserialization surfaces (pickle/`training_args.bin` handling - see the framework
output-tree admission rule), and the isolated-environment boundary. A backend that needs
`trust_remote_code` or network at train time must declare it; the planner can then refuse it under a
stricter assurance tier.
