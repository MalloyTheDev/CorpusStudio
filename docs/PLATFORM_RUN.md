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
- Runs the **functional capability probes** — *readiness = a kernel actually ran*, not "the package
  imports". Emits a **CapabilityReport** with a per-probe `PASS / KERNEL_STALL / …` and the
  `ready / cpu_toy_only / not_ready` verdict.
- `--cache` stores the report keyed by the signature, so an **unchanged host reuses it** and skips the
  (expensive, torch-loading) probes next time. `platform-profiles` lists what's cached.

On a Blackwell GPU the flash-attention probe **short-circuits to `KERNEL_STALL` without executing**
(the fused kernel deadlocks on sm_120), so `flash_attention_2` is correctly absent from the proven
capabilities.

## 2. Plan the run (and see the predicted fit)

```bash
corpus-studio platform-plan \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --dataset ./my-dataset/examples.jsonl \
  --sequence-len 4096 \
  --out ./plan
```

Resolves an **immutable, hash-sealed RunPlan** — every ambiguous field is decided *ahead of time*
against what PROVED to work on this host:

- **attention** → `math` on Blackwell (asserted from `compute_capability_major >= 12`); an explicit
  fused/flash override on Blackwell is **refused**, not silently sealed into a deadlock.
- **precision** → `bf16` only if proven, else `fp32`.
- **quantization** → `nf4` only if bitsandbytes passed, else `none`.
- **adapter** → `qlora` when quantized, else `lora` (the trainer is LoRA-only; other methods are
  refused rather than silently downgraded).
- `sequence_len` flows from your flag — never a hardcoded calibration value.
- **physical execution** → one concrete CPU or CUDA resource, one whole-model authoritative
  placement, and rank 0 by default. Advanced callers may provide `--physical-spec`, a sealed
  `--parameter-accounting-report`, and an exact `--storage-profile`; see
  [`RUN_PLAN_PHYSICAL_EXECUTION.md`](RUN_PLAN_PHYSICAL_EXECUTION.md).

**Pick your framework.** `corpus-studio platform-backends` lists the registered training backends
(`corpus_studio`, `unsloth`, …); pass `--backend <id>` to `platform-plan` to resolve the plan on that
framework. The planner validates the chosen backend against the *resolved* plan — so on Blackwell,
where `math` attention is mandatory, `--backend unsloth` is **honestly refused** (Unsloth's fused
kernels declare no `math` path) with the backends that *do* fit named. A backend is never "supported"
for a config it doesn't declare.

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
corpus-studio platform-run ./plan/RunPlan.json --runner training --out ./run
```

- Runs the plan through the headless **run supervisor** + the **TrainingRunner** (which drives the
  first-party trainer, reading the plan's `training_config_snapshot`).
- Revalidates the contract and recomputes `plan_hash` before dispatch; the worker verifies it again.
  Editing a device, placement, rank, selector, or offload rule after planning is refused.
- Refuses non-trivial physical execution before importing or invoking a trainer. The current built-in
  runner proves the singleton path only; a representable offload/distributed plan is not support proof.
- Streams **RunEvent** envelopes to **stderr** (ordered `seq`, `stage` / `metric` with per-step loss /
  `artifact_produced` / `terminal`).
- Writes the terminal **RunManifest** to stdout (and `./run/RunManifest.json`), classifying the
  terminal state — `succeeded / failed / cancelled` — with a **FailureRecord taxonomy** on abnormal
  termination (`OOM`, `NUMERICAL_FAILURE`, `ENVIRONMENT_FAILURE`, …).
- For each produced weight artifact, writes an integrity-checked **ArtifactManifest** to
  `./run/artifacts/<id>.json` (a cheap size+mtime fingerprint + a byte-exact sha256 content hash — the
  promote gate). The platform never moves your weights; it only references + re-checks them.

### Smoke-test the pipeline without a GPU

```bash
corpus-studio platform-run --demo --runner cpu_toy   # tiny CPU model, needs the [train] extra
corpus-studio platform-run --demo --runner echo      # no-op, needs nothing — proves the harness
```

On a machine without the `[train]` extra, `--runner cpu_toy` reports `failed / ENVIRONMENT_FAILURE`
cleanly (not a crash) — the honest "this host can't run it" signal.

---

## Verified on a real RTX 5070 (Blackwell / sm_120), 2026-07-12

The full lifecycle was executed on an actual RTX 5070 (12 GB, driver 610.74, `cc 12.0`, Windows/WDDM)
with `torch 2.11.0+cu128` + the `[train]` extra — the exact hardware this runbook targets.

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

So every row of the table below is not just designed but **exercised on the real target hardware**;
the only thing still unmeasured is a full-length 7B run's *actual* peak memory (the fit above stays
`predicted` until then).

## What's proven vs. what still needs a measured run

| Step | Guarantee |
|------|-----------|
| profile / probe | **measured** — a kernel ran (or was safely refused) |
| plan | **resolved** — valid + sealed against proven capabilities |
| fit | **predicted** — an estimate, explicitly *not* a proven fit |
| run | **measured** — the terminal state + artifact are real |

The fit prediction becomes *measured* truth only after a real run reports its peak memory. Until then
the platform is careful to say **"predicted"**, not **"it fits"**.
