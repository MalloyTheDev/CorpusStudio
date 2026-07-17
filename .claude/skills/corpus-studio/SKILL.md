---
name: corpus-studio
description: "Project-specific guardrails for editing the CorpusStudio repo - a local-first AI dataset and model-development application at /mnt/training-nvme/repos/CorpusStudio (torch-free control plane + isolated [train] QLoRA worker + WPF/Avalonia/Tauri UI). Use for ANY change here - dataset features, the training engine, the Environment Manager, model/tokenizer management, evaluation, the CLI, UI, or docs - so every edit keeps the dependency-light boundary, the honesty invariants, the contract-regeneration dance, and the one-coherent-CI-green-PR-per-slice workflow. The native-Linux 7B research paper (research/ieee-linux-training, docs/paper) is a SEPARATE opt-in overlay that uses CorpusStudio to verify the training engine; its amendments, reserved identities, and sealed-research gates apply only when the task is the paper, and never define normal product behavior. Pair with the general existing-repo-engineer skill (that one owns generic repo mechanics; this one owns CorpusStudio's invariants and program state). Keep trivial questions lightweight; load this whenever files or repository behavior in CorpusStudio actually matter."
---

# CorpusStudio

Operate as a senior engineer on **CorpusStudio** - a **local-first AI dataset and model-development
application**: a dependency-light, torch-free **control plane** (`engine/corpus_studio/platform/`), an
opt-in `[train]` QLoRA **worker**, and WPF/Avalonia/Tauri **UI** clients. The product surface is the
local builder lifecycle - dataset creation / cleaning / validation / versioning, schema support, dataset
inspection and quality, model + tokenizer selection, local fine-tuning / training, evaluation, and
export - all hardware-aware. The bar here is not "it works" - it is "the evidence, contracts, and honesty
invariants still hold." A passing run obtained by weakening a gate is a defect.

The **native-Linux 7B research paper** (`research/ieee-linux-training/`, `docs/paper/`) is a **separate
project that uses** CorpusStudio to verify the training engine can train a 7B model at seq 4096 on this
host. Its experiment matrices, amendments, reserved identities, paper-performance gates, and
sealed-research evidence rules are an **opt-in overlay** (the "Research overlay" sections below) and
**must not define normal product behavior**. The standard-vs-sealed boundary is
[`docs/PRODUCT_VS_RESEARCH.md`](../../../docs/PRODUCT_VS_RESEARCH.md).

This skill is the CorpusStudio-specific overlay. It assumes the general **existing-repo-engineer**
skill for repo mechanics (recon, minimal-diff, verify-before-handoff) and adds the invariants a generic
skill cannot know. When they conflict, **this skill and the repo's own `CLAUDE.md` / `AGENTS.md` win.**

## Match effort to the task

- **Answer/inspect** - a question or a read. Don't spin up the workflow; just read the right files.
- **Small edit** - one coherent change. Follow the verify gate before you claim done.
- **Slice** - a feature/fix worth a PR. Branch, keep it one coherent CI-green PR, regenerate what the
  change touches, merge after CI. This is the default unit of work here.
- **Research/hardware/GPU** - stop and read the boundary sections below before touching anything
  hardware-adjacent or protocol-adjacent. These have hard stop conditions.

## Read-first, always

Before non-trivial work, ground yourself in the repo's own source of truth (do not duplicate it here):

- [`CLAUDE.md`](../../../CLAUDE.md) + [`AGENTS.md`](../../../AGENTS.md) - the agent contract (imported).
- [`docs/HOST_STATE.md`](../../../docs/HOST_STATE.md) - verified host facts; read before anything
  hardware-adjacent. Where older docs show Windows `C:`/`F:` paths, HOST_STATE supersedes them.
- [`HANDOFF.md`](../../../HANDOFF.md) - session state + roadmap.
- [`docs/CURRENT_STATE.md`](../../../docs/CURRENT_STATE.md) - authoritative feature state.
- [`docs/IMPLEMENTATION_PLAN.md`](../../../docs/IMPLEMENTATION_PLAN.md) - forward plan;
  [`docs/PRODUCT_VS_RESEARCH.md`](../../../docs/PRODUCT_VS_RESEARCH.md) - the product vs research boundary.
  MoE (`docs/MOE_ARCHITECTURE.md`) is a forward research direction, not a product-wide contract mandate.

## The verify gate (run before you claim a change is done)

From `engine/` with the venv (`engine/.venv`, CPython 3.12.3, torch-free core + `[dev]`):

```bash
.venv/bin/python -m ruff check corpus_studio tests
.venv/bin/python -m mypy corpus_studio
.venv/bin/python -m pytest -q --no-header --basetemp=.pytest_tmp
```

- CI runs on **Linux / Python 3.11** with `pytest --cov=corpus_studio --cov-fail-under=88` plus C#/web
  jobs. The coverage floor is **88%** and this host is native Linux, so local coverage ~= CI coverage
  (no Windows under-measurement). Keep a margin above 88, not 88.0x exactly.
- Do not report "done" on red. If tests fail, say so with the output.

## Do not break - the non-negotiable invariants

1. **Dependency-light boundary.** `import corpus_studio.platform` and the engine core must pull **no
   torch** (nor transformers/trl/peft/bitsandbytes). All heavy deps are lazy-imported and live behind
   the `[train]` extra. `tests/test_platform_contracts.py::test_platform_import_is_torch_free` guards
   this; a new `platform/` module that imports torch at module load is a defect.
2. **Contracts are the boundary.** Change `platform/contracts.py` (pydantic) -> **regenerate**
   `docs/contracts/*.schema.json`:
   `python -c "from corpus_studio.platform.schema_export import export_json_schemas; export_json_schemas('../docs/contracts')"`
   -> regenerate the TS types (`cd apps/web && npm run gen:contracts`) -> **update the two counts** in
   `tests/test_platform_contracts.py` (root-contract count + the schema-writer count) when you add/remove
   a `ROOT_CONTRACTS` entry. Schemas and TS are committed and CI diffs them; a drifted schema fails CI.
3. **No-shell execution.** Installers and trainer launches are `argv` lists, never shell strings.
4. **Honesty invariants.** License fail-closed; provenance gate blocks frontier teachers; "a completed
   step != proven fit"; "installed != supported"; no silent target truncation; predicted fit is never
   `NATIVE_SAFE` (only a measured run is); single-writer `examples.jsonl`. **An unavailable metric must
   be null with a typed reason, never a plausible-looking zero** - the v6 lesson: token throughput read
   `0.0` because the observer captured no batches, but a real supervised step processed tokens, so the
   honest value is unavailable, not measured-zero. Capture at the lowest un-bypassable boundary the
   trainer actually consumes (observe `inputs` at `training_step`, not a collate wrapper the
   accelerate-prepared dataloader silently bypasses), and gate throughput separately from resource
   completeness. Never weaken an evidence contract, precision requirement, kernel enforcement, artifact
   integrity, failure taxonomy, or provenance rule to obtain a passing run.
5. **Blackwell / sm_120.** The **math** SDPA attention path is the verified-safe default (fused
   flash-SDPA deadlocks on native Windows/WDDM). Unsloth is refused on Windows/WDDM. Bare-Linux
   forced-`torch_sdpa_flash` is VERIFIED only at the bounded 0.5B / seq-256 / 12-step tuple (v6, and
   re-validated with valid per-step token throughput in v7); it is NOT claimed for the real 7B /
   full-corpus workload, nor for `flash_attention_2` / external `flash-attn` / seq 4096 / Windows-WDDM.
6. **ASCII in CLI-facing strings** (Windows console UTF-8): use `-`, not the em dash; no non-ASCII in
   `typer` help, `raise ...Error(...)` messages, or anything printed to a console.
7. **One training authority.** Shipping clients use `platform-plan` -> `platform-run`; never the
   development-only `train-run`. Every execution gets a fresh run ID and a run-scoped output directory;
   runner identity derives from the pinned backend manifest.
8. **Hardware claims stay evidence-bound.** The managed `backend-corpus-studio` environment is
   `HARDWARE_VERIFIED` (env-manager GPU probe) - that is NOT a workload result. Do not claim
   full-sequence 7B success, DeepSpeed/FSDP/CPU/NVMe offload, real offload fit, PCIe/NVMe throughput,
   bare-Linux FlashAttention for the workload, or MoE runtime capability without a measured run.

## Where the product stands

Product identity, scope, and roadmap are the source of truth for ordinary work: read
[`docs/PRODUCT_SPEC.md`](../../../docs/PRODUCT_SPEC.md) (who it is for and what it is),
[`docs/CURRENT_STATE.md`](../../../docs/CURRENT_STATE.md) (what is built today), and
[`docs/ROADMAP.md`](../../../docs/ROADMAP.md) (milestones). CorpusStudio is a dataset + model-development
application; the 7B paper is a capability test that lives in the overlay below, not the product's
identity.

## Research overlay (paper project only)

**The sections from here down to "Never do" apply ONLY when the task is the native-Linux 7B research
paper** (`research/ieee-linux-training/`, `docs/paper/`). Ordinary product work - dataset features, the
training engine, the Environment Manager for normal environments, model/tokenizer management, evaluation,
UI, packaging - does **not** need amendments, reserved identities, sealed-research admission, or paper
telemetry. The sealed-research provenance gate fires only for `requires_worker_wheel` recipes; the
standard product training backend (`backend-corpus-studio`) does not. If you are not working on the
paper, skip to "Never do (project policy)". Boundary detail: [`docs/PRODUCT_VS_RESEARCH.md`](../../../docs/PRODUCT_VS_RESEARCH.md).

## Research protocol (research/ieee-linux-training/) - append-only

- The protocol is **append-only after any visible result**. Never edit `PROTOCOL.md`, the base
  `EXPERIMENT_MATRIX.yaml`, a frozen amendment, a frozen effective-matrix JSON, or a `RESERVED_IDENTITIES.v*`
  in place. A change is a **new dated amendment** -> new effective-matrix version -> superset
  `RESERVED_IDENTITIES.vN+1` (append-only over the prior).
- `validate_protocol.py` must stay green (`--verify-host-evidence` too), effective-matrix reconstruction
  must be **byte-deterministic**, and a new amendment must hash-bind the prior one (supersession) and be
  set-disjoint from the reserved identities. Current effective version and the v5 bring-up procedure are
  in [`research/ieee-linux-training/README.md`](../../../research/ieee-linux-training/README.md) and
  `RUNBOOK_v5_bringup.md`.
- Metric definitions are authoritative in `research/ieee-linux-training/METRICS.md` (warm-up steps 1-2,
  measured 3-12; trapezoidal energy; n=3 t=4.3026527299). Any measurement code must implement them
  exactly, not approximately.

## Research plan pre-dispatch semantic gate

Before ANY research GPU dispatch, prove the sealed plan is not merely schema-valid but semantically
executable. **A syntactically valid RunPlan is not necessarily a semantically executable RunPlan.**
GPU time must never be spent discovering a defect a CPU-only planning preflight can reject.

1. **No trusted product defaults for matrix-bearing values.** Every one is explicitly supplied or
   compared against the effective matrix: dataset format; model + tokenizer revisions; chat-template
   hash; sequence length; microbatch; gradient accumulation; max steps; optimizer; precision;
   quantization; adapter config; attention path; checkpoint policy; truncation and packing. A product
   default is not a research choice.
2. **Dataset structure must match the selected formatter.** `platform-plan` runs a torch-free
   structural conformance preflight (`platform/dataset_conformance.py`) that refuses a plan with zero
   structurally compatible rows before any id is minted - but you still select the correct
   `--dataset-format` explicitly; the preflight is a backstop, not the decision.
3. **For chat training:** the tokenizer exposes a non-empty chat template; its exact raw-template
   SHA-256 (`sha256(tokenizer.chat_template.encode("utf-8"))`, no normalization) matches the plan; all
   expected rows render non-empty; tokenization yields non-empty examples; the assistant/trainable
   region exists.
4. **CPU rejects before GPU allocates.** Render + tokenize with the exact immutable tokenizer on the CPU
   first; never let GPU work discover a defect a planning/preflight step could have rejected.
5. **The pre-dispatch review reports** (all of): source / compatible / rendered / tokenized row counts;
   dataset hash; rendered-example hash; tokenizer + chat-template hashes; batching tuple;
   sequence/truncation policy; the normalized math/flash plan diff; the candidate-identity disjointness
   result.
6. **Rejected or superseded plans are preserved, never edited or reused** - their identities are burned.
7. **Schema validity != semantic executability.** Prove the selected formatter can consume the EXACT
   immutable dataset before any GPU allocation.

**Lessons already learned (do not relearn on the GPU):** product default **GA=8** vs research **GA=1**
(a matrix-bearing value silently defaulted); a **chat** fixture planned as **instruction** (zero usable
rows, discovered only after model load); an environment **probe** (`HARDWARE_VERIFIED`) is NOT workload
evidence; a **completed step is not a proven fit** (predicted fit is never `NATIVE_SAFE`). Keep volatile
commit ids, lock hashes, plan ids, and run ids OUT of this file - `HANDOFF.md` + `docs/HOST_STATE.md`
own those.

## Worker execution closure and identity impact

Before editing any `platform/` module, decide whether it runs INSIDE the managed worker child, because
that decides whether the change needs a new worker wheel + new sealed environments (v-lineage bump) or
is control-plane-only.

- **The worker execution closure** = everything the `--subprocess` child imports/executes:
  `platform/worker.py::run_worker` -> `supervisor.py::execute_run` -> the success-admission chain
  (`validate_training_success_evidence` -> `validate_sealed_adapter_artifact` in `platform/artifacts.py`)
  -> the runner (`platform/runners.py`) -> the trainer (`training/trainer.py`). **`artifacts.py` and
  `runners.py` are worker code even though they live under `platform/`.** Success admission runs in the
  child; the parent `subprocess_supervisor.py` re-validates for defense in depth but does not move the
  boundary.
- **Consequence:** changing any file in that closure changes worker execution bytes. A sealed
  environment is pinned to a wheel content hash and is IMMUTABLE - it cannot be patched in place and
  must not be recreated under the same id. So a worker-closure change needs: a fresh amendment ->
  effective-matrix bump -> superset `RESERVED_IDENTITIES` -> a new wheel (built reproducibly, twice,
  byte-identical) -> new `-vN` environments -> fresh plans. Control-plane-only changes (planner, CLI
  wiring, schema, summary aggregation, the parent-side telemetry sampler) do not.
- **Once identities are instantiated** (wheel built, environments sealed, a plan dispatched, a visible
  run produced) a "reuse the existing lineage" rationale that depended on non-instantiation no longer
  holds - prove worker non-impact by call-graph, or bump the lineage. Do not guess; trace the import
  path in source and write the classification down (WORKER_CHANGE_REQUIRED vs control-plane-only).

## Framework output-tree admission

The trainer runs a third-party framework (TRL/transformers) that writes its own files into the adapter
output dir. The sealed artifact validator must admit exactly the benign framework metadata and still
fail-closed on real alternate weight payloads.

- TRL's `SFTTrainer.save_model` writes `training_args.bin` (a `TrainingArguments` pickle - NOT weights)
  next to `adapter_model.safetensors`. The `.bin` suffix is in `_WEIGHT_SUFFIXES`, so a naive check
  rejects it as an "alternate or nested model-weight payload" (a false rejection at the export gate).
- The fix pattern (see `platform/artifacts.py`): a **narrow, named allowlist** of root auxiliary
  metadata files, admitted only under strict guards (regular file, single hard link, small size cap,
  never deserialized). Everything else with a weight suffix is still rejected. Never widen the check to
  "ignore all `.bin`" - that would let a real payload through. The file still enters the content hash.
- General rule: when a framework legitimately emits a non-weight file into a sealed tree, admit it by
  an explicit name + structural guards, never by relaxing the payload class.

## Scientific telemetry gate

A run can be a workload success yet not paper-usable. `RunTelemetrySummary.completeness
.scientifically_complete` is true only when every `REQUIRED_PAPER_FIELDS` entry
(`platform/telemetry.py`) is genuinely present; it is reported, never hidden, and never faked.

- **Each field comes from a real measurement and is null (never zero-filled) when its source is
  unavailable.** Do not manufacture a value to flip completeness.
- **Know which process owns each source.** The parent telemetry sampler is torch-free, so the torch
  allocator memory probe returns null there; the CUDA-owning child owns torch allocator + token counts +
  step boundaries. GPU memory therefore has two honest sources: parent nvidia-smi device used/free
  (`probe_gpu`) and worker torch-allocator memory emitted per step into `RunEvents.jsonl`.
- **Identity lineage the plan cannot carry** (worker wheel sha256, source repository commit) is threaded
  as a `worker_identity_overlay` from the sealed environment lock (+ the wheel's `BUILD_PROVENANCE.json`)
  at `platform-run` time; the sealed `execution_configuration_hash` is the resolved config's real
  `configuration_hash` field. Resume lineage stays manifest-authoritative - an overlay never supplies it.
- **Worker-side instrumentation is untestable in the torch-free venv**, so keep the pure logic (token
  counting, hash derivation) at module level and unit-test it, keep the framework wiring in the
  `pragma: no cover` integration path, and make it degrade-to-null and pass-through so it can never alter
  or fail training. Field-by-field root cause before implementing: **do not guess why a field was
  missing - read the preserved raw records.**

## Never do (project policy)

- **No multi-agent workflow fan-outs** for CorpusStudio work (cost) - verify inline, one Claude process.
- **No `git clean`.** Do not delete, rewrite, relabel, reuse, or mutate ANY historical worker wheel,
  environment, lock, capability report, execution probe, RunPlan, execution config, run, adapter, output
  directory, protocol version, amendment, evidence directory, or `SHA256SUMS`.
- **No writes to `/mnt/windows-c` or `/mnt/windows-f`** - read-only historical mounts by policy, even
  though the filesystem permits writes. Develop only from `/mnt/training-nvme/repos/CorpusStudio`.
- Do not build a worker wheel, create/recreate/remove a managed environment, generate an executable
  RunPlan, load model weights, or dispatch a GPU workload unless the task explicitly authorizes it.
- **STOP and surface** before: a destructive/irreversible op; a credential or legal-authorization
  decision; a full 7B run (substantial time); a scientific change that would amend the study after
  results are visible; or a hardware-risk condition. For GPU work: **unload Ollama first, one GPU
  operation at a time.**

## Slice workflow (the default unit of work)

1. `git checkout -b feat/<slice>` (or `fix/`, `research/`) - branch first; `main` is protected.
2. Make one coherent change. Regenerate schemas/TS + update contract counts if you touched
   `platform/contracts.py`. Update the doc that the change makes stale (HANDOFF/HOST_STATE/CURRENT_STATE).
3. Run the full verify gate green (ruff + mypy + pytest at >=88% coverage). Add deterministic tests -
   fixture-driven, no GPU, torch-free where the code is torch-free.
4. Commit with the repo's trailers; open one PR; **wait for CI**; merge after green (admin-merge is used
   here for CI-green PRs when review is the only blocker, per standing authorization). Keep one coherent
   CI-green PR per slice - do not batch unrelated changes.

## Environment gotchas (learned the hard way)

- **cwd drift:** a `cd engine && ...` in a compound Bash command shifts the persisted working directory
  and scopes later `git`/greps to `engine/`. Prefer absolute paths or `git -C`; verify with `pwd`.
- **CRLF working tree:** the working tree is all CRLF, committed is LF. `git diff --check` is noisy;
  preserve per-file line endings, never bulk-normalize. Write new files with LF.
- **pytest basetemp:** a stale custom `--basetemp` dir causes `FileNotFoundError`; `rm -rf` it before a
  run, and never commit `.pytest_tmp*` artifacts (stage explicit paths, not `git add -A`).
- **Do not commit generated junk:** stray `*.schema.json`/`index.json` exported into `engine/` are
  session artifacts - clean them; the canonical schemas live in `docs/contracts/`.

## Research overlay: where the paper program stands (update this as it moves)

*This is the state of the native-Linux 7B **paper** project, not the product roadmap; it is relevant only
when working on the paper overlay.* The active thread is proving the training engine can train at the 7B
/ seq-4096 target. The **v7 0.5B GPU
bring-up PASSED (`V7_MATH_AND_FLASH_THROUGHPUT_PASS`, 2026-07-16)**: the v6 token-throughput observer gap
(`tokens/sec = 0.0`, UNAVAILABLE not zero) was fixed in the worker child by PR **#466** (merge
`25c901ec`) - observe `inputs` at `SFTTrainer.training_step` (the trainer's un-bypassable consumption
boundary), emit raw per-step counts, and gate `scientific_throughput_complete` /
`paper_performance_complete` separately from resource completeness. That worker-byte change forced a
fresh v7 lineage: amendment 0004 -> effective matrix **1.4.0** -> `RESERVED_IDENTITIES.v4` (reserves all
v1-v6) -> reproducible v7 wheel `090f879b...` (source `21aa81d9`) -> `-math-v7`/`-flash-v7` environments
-> fresh matched chat plans (A7 normalized comparison UNEXPECTED=0) -> **both smokes succeeded** (runs
`run-019f6956...` / `run-019f6966...`; 12 steps, loss ~5.43->~0.38, adapter admitted 336/336, measured
`NATIVE_SAFE`), with the token observer now firing on **every** step (positive non-padding + supervised,
`observed_microbatches=1`, rates == observed tokens / duration, `scientific_throughput_complete=True`
as-dispatched). **Lesson (build-provenance):** the shipped telemetry reader
(`_build_provenance_source_commit`) reads key **`source_commit`** from the wheel's `BUILD_PROVENANCE.json`
to populate `identity.repository_commit`; a build-provenance generator that writes the commit under a
different key (v7 used `audited_commit`) leaves the auto summary `scientific_resource_complete=false`.
The commit is authentic and recoverable, so re-deriving the summary from the PRESERVED raw records with
the sealed commit via the identity overlay restores `paper_performance_complete=true` with zero
measurement change - but future wheel builds must emit `source_commit`. This is a 0.5B feasibility result,
NOT a 7B or full-training claim. Remaining gates each need separate human authorization: the corpus
freeze, then the 7B sequence ladder and full runs. This paragraph goes stale fast: the authoritative,
volatile identity + readiness detail always lives in `HANDOFF.md` + `docs/HOST_STATE.md` - trust those
over this paragraph.
