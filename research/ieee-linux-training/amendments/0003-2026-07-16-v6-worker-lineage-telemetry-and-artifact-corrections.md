# Amendment 0003 - v6 worker lineage (artifact-admission + telemetry-population corrections)

- **Amendment id:** `cs-ieee-linux-training-amendment-0003`
- **Study:** `cs-ieee-linux-training-v1`
- **Base protocol version:** 1.0.0
- **Effective protocol version:** 1.3.0 (supersedes 1.2.0)
- **Status:** prospective (authored before any v6 wheel build, environment, plan, or run)
- **Authored:** 2026-07-16
- **Analysis role:** primary

This amendment is **append-only**. It does not edit `PROTOCOL.md`, `EXPERIMENT_MATRIX.yaml`, amendment
0001 or 0002, effective matrices 1.1.0 or 1.2.0, or `RESERVED_IDENTITIES.v1`/`.v2` in place. It adds
effective matrix **1.3.0**, reserved-identity registry **v3** (a strict append-only superset of v2),
and this narrative + manifest. The prior amendment 0002 is bound by exact raw-file hash in the manifest
`supersedes` block, so the amendment chain stays ordered and 0002 stays provably byte-frozen.

## Why this amendment exists

The v5 0.5B bring-up (effective matrix 1.2.0, worker wheel
`271447bf0b82ab14ddb4848bd0d47cc0f2f5ad8ff378f291f04cff65be2fc895`, worker source
`713fde14c8a6bae1888c43851fa6a487a031a242`) produced the study's **first real GPU training** on the
native-Linux RTX 5070 and then terminally failed. The visible, preserved facts:

1. The corrected chat-format math run
   (`run-019f67e6-8d14-782a-8b10-ecdb4e74a6e0`, forced `torch_sdpa_math`) completed **12 QLoRA
   optimizer steps with monotonically decreasing loss** (5.43 -> 0.39) and serialized a clean 336-tensor
   LoRA adapter (all `lora_A`/`lora_B`, zero base weights).
2. Final success admission **failed** (taxonomy `ARTIFACT_FAILURE`, stage `export`) because TRL's
   `SFTTrainer.save_model` writes a benign `training_args.bin` (a `TrainingArguments` pickle - not model
   weights) into the adapter directory, and the sealed adapter validator classified any `.bin` file as
   an alternate/nested model-weight payload. The adapter payload itself was correct; the trigger was a
   benign framework metadata file.
3. **flash-v5 was withheld** - the export/artifact-admission path is kernel-independent and shared with
   math, so a flash dispatch would fail identically and would only burn flash-v5 identities.
4. The v5 result therefore remained **NATIVE_UNPROVEN** (a failed run is never a proven fit), and its
   telemetry was **scientifically incomplete** (missing `gpu.memory`, `step.step_time_seconds`, and the
   worker-wheel / repository-commit / execution-configuration identity fields).

An earlier GA=1 dispatch on the same wheel
(`run-019f67ac-275f-7405-ac92-f407c91be70f`) failed `UNSUPPORTED_CONFIGURATION` at `adapter_attached`
because a chat fixture was planned as instruction format (zero usable rows); that planning defect was
fixed by a pre-dispatch dataset-format conformance preflight (PR #460, control-plane).

## The corrections and why they force a fresh worker lineage

Two corrections landed on `main` after the v5 result was visible:

- **PR #461** (`4078825a82309223b3b39d1192cc08f8c26afc85`) - the sealed adapter validator now admits
  `training_args.bin` under a narrow, named root auxiliary-metadata allowlist (regular file, single hard
  link, size-capped, **never deserialized**), while still rejecting every real alternate weight payload,
  nested `.bin`, extra Safetensors, or checkpoint state.
- **PR #462** (`9a15b5c095accd451fbb57e887bf4b2c1ad0a7e7`) - telemetry now populates every required paper
  field: driver device memory (nvidia-smi) plus worker torch-allocator memory into `RunEvents.jsonl`;
  per-step wall time at the optimizer-step boundary; non-padding and supervised token throughput;
  the sealed `execution_configuration_hash`; and the worker-wheel-sha256 + source-commit identity
  overlay threaded from the environment lock and build provenance.

**Both corrections run inside the managed worker child** - the artifact validator executes in
`worker.py -> execute_run -> validate_sealed_adapter_artifact`, and the telemetry emission is in the
runner/trainer that the child executes. They therefore change worker execution bytes. Because the v5
environments are sealed to the immutable wheel `271447bf...`, they cannot be patched in place and must
not be recreated under the same ids. **A fresh v6 wheel, v6 environments, v6 plans, and v6 runs are
required.**

## What 1.3.0 changes

- `first_party_execution_paths[first-party-math].environment_id` ->
  `backend-corpus-studio-research-math-v6`
- `first_party_execution_paths[first-party-flash].environment_id` ->
  `backend-corpus-studio-research-flash-v6`
- `schema_version` / `protocol_version` -> `1.3.0`
- Immutable bindings gain (as in 1.2.0, reconstructed from base): the exact effective-matrix hash,
  reserved-identities hash, and worker source commit, each required per environment plan and trial.

The v6 worker source must **descend from `af28be9b42e0b91d72eaf7e9f24462ef3f2a189d`** (which itself
descends from the `df86db5` worker-source floor and from both correction commits `4078825` and
`9a15b5c`). To avoid an impossible self-referential commit pin, the amendment does not pin an exact v6
commit; it requires that the final wheel source descend from `af28be9`, that the exact final
post-amendment source commit be recorded in the wheel evidence, that the exact wheel sha-256 be recorded
in each environment and trial, and that historical worker-wheel reuse is prohibited.

## What 1.3.0 does NOT change (the scientific tuple is preserved)

Qwen2.5-0.5B bring-up model; chat fixture; sequence length 256; microbatch 1; gradient accumulation 1;
12 optimizer steps; QLoRA r16 / alpha 32 / dropout 0.05 / bias none / all-linear; NF4 + double
quantization; BF16 compute; FP32 adapter weights and materialized gradients; `adamw_torch`;
checkpoint-free; no offload; no compile; no truncation; no packing; forced math versus forced flash. The
only differences between the matched math and flash arms remain the attention capability/probe, the SDP
toggles, and the environment identity.

## Reserved identities

`RESERVED_IDENTITIES.v3.json` is an append-only strict superset of v2. Beyond every v1-v4 identity it
already carried, it reserves the fully instantiated v5 identities: the v5 worker wheel
(`271447bf...`) and source commit (`713fde1...`); the `math-v5`/`flash-v5` environment ids, lock hashes,
capability-report and execution-probe evidence hashes, recipe and resolution refs; all six generated v5
plans (rejected GA=8, rejected GA=1 instruction, and corrected chat pairs) with their plan ids, plan
hashes, execution-configuration ids and hashes; both dispatched math run ids; the preserved adapter
artifact identity; and the v5 output and evidence roots. No v1-v5 identity may be reused.

## Preregistration boundary (unchanged)

Feasibility characterization remains distinct from the gated full-training phase over the
~500-output corpus; that phase is not authorized here. This amendment authorizes only a fresh v6 0.5B
feasibility bring-up (one math smoke, one conditional flash smoke). No 7B model is loaded and the
sequence-length ladder is not started.
