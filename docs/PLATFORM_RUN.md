# Running a training job through the platform (headless core)

CorpusStudio's headless platform turns **goal + data + hardware** into a validated, reproducible run,
end to end, through language-neutral contracts:

```
profile the host  →  plan the run  →  predict the fit  →  run it  →  account for the artifact
 (platform-probe)    (platform-plan)   (in platform-plan)  (platform-run)  (RunManifest + ArtifactManifest)
```

Everything below is a real `corpus-studio` CLI command. The commands print JSON **results to stdout**
and **telemetry/progress to stderr** (so a result is always a clean, pipeable JSON document). Every
step is dependency-light *except the actual training run*, which needs the `[train]` extra + a GPU.

This is the reference procedure for a real run — e.g. the WBG-7B (Qwen2.5-7B QLoRA) job on an
RTX 5070 (Blackwell / sm_120).

---

## 1. Profile the host + prove its capabilities

```bash
corpus-studio platform-probe --cache
```

- Builds an **EnvironmentProfile** (OS, memory-residency model, GPUs incl. `compute_capability_major`,
  package locks, a deterministic `environment_signature`).
- Runs the **functional capability probes**. `ready` now requires a complete bounded execution tuple
  (precision + quantization + adapter + attention + optimizer + loss + checkpoint together), not a
  package import or a union of unrelated passing probes. Emits a **CapabilityReport** with embedded
  per-probe evidence and the `ready / cpu_toy_only / not_ready` verdict.
- `--cache` stores the report keyed by the signature, so an **unchanged host reuses it** and skips the
  (expensive, torch-loading) probes next time. `platform-profiles` lists what's cached.

On a **native-Windows/WDDM** Blackwell GPU the flash-attention probe short-circuits to
`KERNEL_STALL` without executing (the measured fused-kernel deadlock), so `flash_attention_2` is
correctly absent. Elsewhere the probe executes, and only its result proves that exact environment;
WSL evidence is not bare-Linux evidence.

## 2. Plan the run (and see the predicted fit)

```bash
corpus-studio platform-plan \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --dataset ./my-dataset/examples.jsonl \
  --sequence-len 4096 \
  --out ./plan
```

Resolves an **immutable, hash-sealed RunPlan** — every ambiguous field is decided *ahead of time*
against what PROVED to work on this host:

- **attention / precision / quantization / adapter / optimizer / loss / checkpoint** are selected only
  from one complete passing execution combination. Native-Windows/WDDM Blackwell still requires its
  math path; a standalone flash or bitsandbytes probe cannot upgrade a different tuple.
- **adapter** → `qlora` when quantized, else `lora` (the trainer is LoRA-only; other methods are
  refused rather than silently downgraded).
- `sequence_len` flows from your flag — never a hardcoded calibration value.
- **physical execution** → one concrete CPU or CUDA resource, one whole-model authoritative
  placement, and rank 0 by default. Advanced callers may provide `--physical-spec`, a sealed
  `--parameter-accounting-report`, and an exact `--storage-profile`; see
  [`RUN_PLAN_PHYSICAL_EXECUTION.md`](RUN_PLAN_PHYSICAL_EXECUTION.md).
- **effective execution** → a separately hash-sealed configuration pins the dataset bytes, immutable
  model/tokenizer revisions (or local-directory hashes), objective/environment/capability/backend
  identities, per-state precision, exact attention API/kernel/toggles, explicit device map, all
  semantic LoRA/trainer/checkpoint/data defaults, and the exact installed trainer interface. See
  [`EFFECTIVE_EXECUTION_CONFIGURATION.md`](EFFECTIVE_EXECUTION_CONFIGURATION.md).
- **intermediate checkpoints** → disabled for every new first-party plan (`save_strategy="no"`, no
  cadence or retention). Step-checkpoint plans are refused before training until exact sealed resume
  compatibility and checkpoint lineage exist. The environment probe's adapter save/reload round trip
  is capability evidence for final adapter serialization, not mid-run optimizer-checkpoint resume.

**Pick your framework.** `corpus-studio platform-backends` lists registered backend manifests; pass
`--backend <id>` to resolve against one. A backend is executable only when it declares the complete
execution-contract surface and the selected environment proves the matching functional capabilities.
The current first-party `corpus_studio` backend does so. The Unsloth manifest does not yet declare the
Phase 9B execution contract, so new plans refuse it on every host. Native-Windows Blackwell also has
the independent math-attention incompatibility. An import or static feature declaration is never
support proof.

It also prints the **predicted fit** to stderr and writes `FitClassification.json`, e.g.:

```
predicted fit: NATIVE_UNPROVEN — estimated peak ~10.8 GB fits within 12.0 GB with ~1.2 GB headroom — predicted to fit, NOT measured.
```

**Honesty:** a predicted fit is *never* `NATIVE_SAFE` — only a measured run earns that. Over capacity,
the verdict distinguishes a Windows/WDDM **silent spill** (`ACCIDENTAL_WDDM_SPILL`, 10-25× slowdown)
from a hard OOM (`FAIL`) by the host's residency model.

A non-trivial physical specification is classified `PLANNED_UNPROVEN`, with no peak-memory or native
residency claim, until an estimator and measured backend exist for that topology. Unsuitable storage
is refused; marginal or unknown storage requires both a matching recorded assessment and the explicit
`--allow-marginal-storage` or `--allow-unknown-storage` flag.

If the host isn't ready, `platform-plan` exits non-zero with a clear reason (the ahead-of-time twin of
a training error) — it does **not** silently downgrade a real-training request to a CPU toy.

## 3. Execute the plan through the supervisor

```bash
corpus-studio platform-run ./plan/RunPlan.json --subprocess --out ./run
```

- Runs the plan through the headless **run supervisor** + the **TrainingRunner**. New plans are mapped
  only from `resolved_execution`; the legacy `training_config_snapshot` is not an execution source.
- Revalidates the contract and recomputes both `plan_hash` and the nested execution-configuration hash
  before dispatch; the worker verifies them again and echoes the effective hash before model loading.
  Editing an input, dtype, attention toggle, trainer field, device/placement, rank, selector, or
  offload rule after planning is refused.
- Newly planned runs hash-pin the exact static `BackendManifest`. Subprocess protocol 2.0 waits for a
  worker-first `hello`, validates backend and environment/lock identity, and only then dispatches.
  Correlation/run IDs, message order, event sequence, terminal lineage, and artifacts are fail-closed.
  The parent owns a dedicated process group/session and terminates the full worker tree on timeout or
  protocol failure; public in-process and subprocess entry points both reject a broken plan seal before
  runner invocation.
  See [`BACKEND_WORKER_PROTOCOL.md`](BACKEND_WORKER_PROTOCOL.md).
- Refuses non-trivial physical execution before importing or invoking a trainer. The current built-in
  runner proves the singleton path only; a representable offload/distributed plan is not support proof.
- Revalidates local model/tokenizer inputs and pinned package versions. The dataset file is opened,
  read, and SHA-256 hashed once; the trainer parses those exact captured bytes rather than reopening
  the path. It then applies exactly one attention-kernel policy and explicit device map and observes
  the model attention API and actual loaded placement. Any mismatch refuses the run;
  `device_map="auto"` and silent semantic trainer-field removal are invalid. Chat-template failure
  blocks, and truncation analysis covers the complete pinned dataset unless an explicit allow policy
  was sealed.
- Gives resolved training setup one absolute `--preflight-timeout` budget (1800 seconds by default).
  Bounded same-thread byte/row events expose dataset verification, formatting, and tokenization;
  tokenizer/model-load events mark the real call boundaries. These events and heartbeats cannot
  extend that absolute deadline. At optimizer creation, the ordinary `--timeout` silence deadline
  resumes. Setup expiry is `TIMEOUT` at the last stage; ordinary execution silence is
  `KERNEL_STALL`.
- Streams **RunEvent** envelopes to **stderr** (ordered `seq`, `stage` / `metric` with per-step loss /
  `artifact_produced` / `terminal`).
- Mints a fresh UUIDv7 `run_id` for every execution. A resolved run derives its trainer directory from
  the sealed output-root policy as `<output-root>/runs/<run-id>/artifacts/adapter`, so rerunning one
  plan cannot mix adapters. Intermediate checkpoints are disabled; an unexpected one is a failed-run
  deviation and remains preserved as evidence. The control-plane design + verifier for exact resume
  lineage (a hash-sealed checkpoint manifest, fail-closed byte integrity, and resume admission) lands
  in [`CHECKPOINT_RESUME.md`](CHECKPOINT_RESUME.md) / `corpus-studio checkpoint-verify`
  ([#440](https://github.com/MalloyTheDev/CorpusStudio/issues/440)); automatic resume stays blocked
  until a separately reviewed trainer change consumes it, so runs expected to exceed 30 minutes remain
  gated.
- Writes the terminal **RunManifest** to stdout (and `./run/runs/<run-id>/RunManifest.json`), classifying the
  terminal state — `succeeded / failed / cancelled` — with a **FailureRecord taxonomy** on abnormal
  termination (`OOM`, `NUMERICAL_FAILURE`, `ENVIRONMENT_FAILURE`, …). Subprocess terminal events are
  held by the parent until terminal identity, artifacts, and durable records are admitted; a child
  success overturned by parent admission is exposed only as the final classified failure.
- A training success requires canonical before/after identities for the complete trainable adapter
  state with at least one changed tensor, at least one verified materialized gradient with honest
  eligible/observed name inventories, a real optimizer observed before step evidence, and exactly one
  finite loss for every completed sealed step. Final trainable tensors must remain finite, and the
  exact trained PEFT export state/config must equal the independently parsed saved Safetensors/config.
  The runner then verifies the exact non-link output path, one recognized adapter payload, and an
  **ArtifactManifest** with independent weight and config hashes; the subprocess parent reconstructs
  the same gate and the claimed fit from raw peak evidence. Artifact evidence and the terminal record
  are persisted before a succeeded terminal event. The adapter ID is
  `<run-id>-adapter-<content-hash-prefix>`. The platform never moves your weights; it only references
  and re-checks them.

### Capture measurement telemetry

```bash
corpus-studio platform-run ./plan/RunPlan.json --subprocess --out ./run \
  --telemetry --telemetry-interval-ms 200
corpus-studio telemetry-summarize ./run/runs/<run-id> --plan ./plan/RunPlan.json --table
```

`--telemetry` samples raw GPU/host telemetry into the run directory and, after the run, derives a
`RunTelemetrySummary` purely from the durable raw records (`RunManifest` + `RunEvents.jsonl` +
`TelemetrySamples.jsonl`). Raw is authoritative and is written before the summary; CSV, tables, and
plot series all render from the one derived object. A telemetry gap never converts a workload success
into paper data - a successful run can still report `scientifically_complete=false`. See
[`MEASUREMENT_HARNESS.md`](MEASUREMENT_HARNESS.md).

### Smoke-test the pipeline without a GPU

```bash
corpus-studio platform-run --demo --runner cpu_toy   # tiny CPU model, needs the [train] extra
corpus-studio platform-run --demo --runner echo      # no-op, needs nothing — proves the harness
```

On a machine without the `[train]` extra, `--runner cpu_toy` reports `failed / ENVIRONMENT_FAILURE`
cleanly (not a crash) — the honest "this host can't run it" signal.

---

## Historical pre-Phase-9B RTX 5070 evidence (2026-07-12)

The pre-Phase-9B lifecycle was executed on an actual RTX 5070 (12 GB, driver 610.74, `cc 12.0`,
Windows/WDDM) with `torch 2.11.0+cu128` + the `[train]` extra. This evidence remains useful for the old
native-Windows path, but it predates `ResolvedExecutionConfiguration` and does not verify the new
attention/precision/placement/input enforcement. The Phase 9B workload must be rerun on the current
native-Linux host; the managed-environment hardware probe does not satisfy that workload requirement.

**Profile + probe** (`platform-probe`) — `READINESS: ready`, kernels actually ran:

```
GPU: NVIDIA GeForce RTX 5070
PASS         cuda_available   — 1 CUDA device(s)
PASS         bf16_matmul      — bf16 matmul on cuda
PASS         bnb_4bit_load    — Linear4bit forward ok
KERNEL_STALL flash_attn_backward — sm_120 (Blackwell): not executed to avoid the deadlock
PASS         checkpoint_reload — save/reload round-trip ok
proven on this host: bf16, int4, nf4
```

**Plan** (`platform-plan`) resolved `bf16 / nf4 / math / qlora`, backend `corpus_studio`. For the
WBG-7B target (Qwen2.5-7B, seq 4096) the fit was honestly **`ACCIDENTAL_WDDM_SPILL`** (est. peak
~19.3 GB > 12.8 GB — *predicted, not measured*). `--backend unsloth` was refused:
`attention 'math' not supported. Backends that fit: corpus_studio.`

**Run** (`platform-run --runner training`) — a real GPU QLoRA (Qwen2.5-0.5B, nf4, math attention,
3 steps) reached `state: succeeded`; the RunEvent stream showed the loss **decreasing** across steps
(`3.70 → 2.49`), and it did **not** deadlock (the sm_120 flash kernel was correctly disabled). The
saved LoRA adapter (17.6 MB) got an integrity-checked ArtifactManifest — a real 64-char sha256
`content_hash`, and a live re-hash of the on-disk weights re-verified as `ok`.

That run exercised the pre-Phase-9B native-Windows lifecycle. It did not exercise the new nested
execution seal, immutable-input checks, exact trainer-interface admission, or runtime placement
observation. It is also not native-Linux, offload, NVMe-throughput, full-sequence 7B, FSDP/DeepSpeed,
or MoE-runtime evidence.

## What's proven vs. what still needs a measured run

| Step | Guarantee |
|------|-----------|
| profile / probe | **measured** — a kernel ran (or was safely refused) |
| plan | **resolved** — valid + sealed against proven capabilities and immutable inputs |
| fit | **predicted** — an estimate, explicitly *not* a proven fit |
| failed/cancelled run with peak memory | **measured but unproven** — native non-spill stays `NATIVE_UNPROVEN`; an observed spill stays a spill |
| succeeded run after every success gate | **measured and proven** — output, adapter bytes, artifact integrity, losses, optimizer, gradients, and update all passed |

A peak-memory sample makes fit *measured*, not automatically proven. Native fit becomes proven only
after the complete succeeded-run evidence gate above. Until then the platform says **"predicted"** or
**`NATIVE_UNPROVEN`**, not **"it fits"**.
