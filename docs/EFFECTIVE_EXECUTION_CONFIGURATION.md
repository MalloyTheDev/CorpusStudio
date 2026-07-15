# Effective execution configuration

Phase 9B closes the first-party dense trainer's intent-to-runtime gap. A new `RunPlan` now contains a
separately hash-sealed `ResolvedExecutionConfiguration`; the training runner consumes that typed
configuration directly. It may refuse the configuration, but it may not fill in, filter, reinterpret,
or override semantic fields after planning.

This is an execution-contract and enforcement milestone. It does not add or verify DeepSpeed, FSDP,
offload, distributed, or MoE execution.

## Trust chain

```text
immutable inputs + exact capability evidence + backend/environment identity
    -> ResolvedExecutionConfiguration.configuration_hash
    -> RunPlan.plan_hash
    -> worker verifies both seals and echoes the execution hash
    -> trainer applies the sealed policy and observes attention/precision/placement
    -> deviation refuses the run
```

The configuration pins:

- exact dataset bytes and either immutable Hugging Face model/tokenizer commits or stable local
  directory digests;
- hash-pinned objective, backend manifest, managed environment, and capability-report identities;
- unquantized or quantized weight storage, dequantization, forward-compute, gradient, optimizer-state,
  optimizer-auxiliary, and trainable master-weight dtypes;
- the model attention API, one required effective kernel, FlashAttention package identity when
  applicable, and all three PyTorch SDPA toggles;
- an explicit root model device map (`auto` is invalid);
- every LoRA-family default, optimizer, loss, sequence/batching policy, stop condition, checkpoint
  cadence/retention, formatter, truncation policy, packing choice, seed, and output directory;
- exact installed trainer package versions, required `SFTConfig` fields, sequence-length field, and
  tokenizer parameter name.

`RunPlan.training_config_snapshot` remains only as a legacy read-compatibility field. Newly generated
plans leave it empty and use `resolved_execution`.

## Plan-time admission

An explicit request is not permission to bypass evidence. The selected backend must declare the
requested optimizer, loss, checkpoint implementation, execution-contract version, trainer fields,
attention API, and effective kernel. The matching environment must also carry passing functional
evidence.

Independent capability axes are diagnostic only. `ready` and execution-contract support require an
embedded `ExecutionCapabilityCombination` from one passing probe. The tuple binds runtime/device,
precision, quantization, adapter, attention API/kernel, optimizer, loss, checkpoint, export, and
contract version together. The reference probes implement a bounded FP32/eager/LoRA CPU tuple and a
BF16/NF4/QLoRA/math-SDPA CUDA tuple, each including a real model backward, AdamW update, and PEFT
safetensors save/reload; the CUDA tuple is not a passing hardware claim until it runs successfully in
the final managed environment. Trainer-surface inspection is an additional conjunctive gate. A
standalone PyTorch flash-SDPA result proves only that kernel, not a complete training tuple or the
external `flash_attention_2` package.

The current `corpus_studio` backend declares this contract. The existing Unsloth manifest does not,
so the planner refuses it on every host for new Phase 9B plans. Adding a backend requires an isolated
recipe plus its own exact declarations and functional probes.

## Worker enforcement

Before model loading, the execution path:

1. Recomputes the `RunPlan` and execution-configuration hashes.
2. Re-hashes local model/tokenizer inputs and rejects changed or unstable bytes, linked paths, and
   root escapes. The dataset is read and hashed once by the trainer, then parsed from those exact
   stabilized bytes rather than reopened or rehashed.
3. Verifies the sealed package versions and formatter/chat-template identity. Managed planning also
   requires every trainer package copied into the configuration to retain its sealed artifact,
   RECORD, and installed-tree integrity evidence; version-only unknown integrity is refused. A
   new complete package uses `record_count_semantics="all_record_rows_v2"` and must have positive counts with
   `record_verified_entries == record_entries == installed_file_count`, no failed row, and both the
   RECORD-text and installed-file-tree SHA-256 identities.
4. Applies the exact Flash, memory-efficient, and math SDPA toggles and checks the observed global
   state; SDPA paths run a tiny forward/backward with only the required kernel enabled.
5. Loads the model with the immutable revision, `trust_remote_code=False`, safetensors-only policy,
   sealed quantization compute dtype, and explicit device map.
6. Observes the model attention API and initial placement, then re-hashes local model/tokenizer roots
   after third-party loading.
7. After PEFT attachment, observes every parameter device, base storage/NF4 identity,
   dequantization dtype, and trainable master dtype; a post-accumulation hook verifies each
   materialized leaf gradient (not an earlier autocast edge tensor), records observed versus eligible
   coverage without claiming every eligible tensor received a gradient, and requires at least one
   changed adapter tensor to intersect that observed set. Optimizer-state dtype/device guards run
   when those tensors materialize. A mismatch retains a structured taxonomy and last verified stage.
8. Requires the exact sealed TRL configuration surface; semantic fields are never silently removed.
   In particular, `logging_strategy="steps"`, `logging_steps=1`, and
   `logging_nan_inf_filter=false` are part of the execution meaning.

For subprocess runs, protocol 2.0 includes the execution-configuration hash in `run_accepted`. The
parent compares it with the dispatched plan before accepting any run events.

Resolved training setup is supervised separately from optimizer execution. Its first recognized
setup stage starts one absolute `--preflight-timeout` budget; bounded same-thread dataset and
tokenization progress plus actual tokenizer/model-load boundaries are observable but cannot extend
that deadline. `optimizer_created` is emitted only from `on_train_begin` after the callback observes
a real optimizer (the first optimizer metric is still a defensive supervision boundary). That event
permanently restores the ordinary `--timeout` silence rule. Heartbeats extend neither deadline.

The runner lane is part of the execution boundary: an echo worker cannot consume a training plan.
`platform-run` defaults to `--runner auto`, which selects the one lane allowed by the seal.
`--max-steps` and explicit `--runner cpu_toy|training` are compatibility assertions only. To change
execution semantics, generate a new plan and therefore new hashes.

Runner identity is derived from the pinned backend manifest, not accepted as an independent source of
training authority. Echo is valid only for an explicit evaluation/demo plan bound to the echo backend.
A resolved training success must contain canonical before/after hashes for the complete trainable
adapter state with at least one changed tensor, observed materialized-gradient coverage, a verified
real optimizer, exactly one finite loss for every completed optimizer step, and finite final adapter
tensors. Because the pinned TRL constructor may recast QLoRA trainable state, the worker restores the
sealed master dtype only on the same identity-bound parameters after trainer construction, then
re-runs placement, quantization, and precision verification before the first backward pass. The exact
post-training PEFT export state must equal the independently parsed Safetensors
tensor state; the complete PEFT config semantics must equal parsed `adapter_config.json`. The runner
also requires the trainer's reported output and adapter paths to equal the derived run-scoped path and
rejects symlink/junction descendants, alternate weight payloads, and checkpoints. A readable path
elsewhere is not an artifact. Promotion uses recognized adapter/model weight bytes only; a descriptor
cannot stand in for trained weights, but its independent metadata hash is also mandatory. Artifact
construction, content/config rehashing, durable manifest writes, and raw peak-memory reconciliation
must pass before terminal success is released or measured fit receives `proven=true`. The subprocess
parent independently reconstructs these gates and defers the child's terminal event until admission
and persistence finish. A failed non-spilling run remains `NATIVE_UNPROVEN`; only observed spill may
retain a spill classification.

Every invocation mints a fresh UUIDv7 `run_id`, even when the same immutable plan is executed again.
The sealed `output_dir` is an output **root** with `output_layout=run_scoped_v1`; the worker derives:

```text
<output-root>/runs/<run-id>/artifacts/adapter
```

New first-party plans explicitly seal `save_strategy="no"` with null checkpoint cadence and retention,
so the trainer writes no intermediate checkpoint directories. Legacy step-checkpoint documents remain
parseable but are refused before training because there is no exact resume lineage. A worker reporting
an intermediate checkpoint under the disabled policy fails the run. The final adapter cannot collide
across executions; its artifact ID
contains the run ID, `adapter` role, and weight-content-hash prefix. `platform-run --out RECORD_ROOT`
persists the terminal records under `RECORD_ROOT/runs/<run-id>/`.

Exact source-run/checkpoint identity, optimizer/scheduler/RNG/sampler state, and fresh resumed-run
lineage for first-party runs expected to exceed 30 minutes are tracked separately in issue #440.

The shipping WPF/Avalonia desktop no longer constructs or launches `train-run`. A first-party config
export has no executable argv and directs the user to Platform planning. The retained low-level CLI
command refuses unless `--allow-unsealed-direct-execution` is supplied; when explicitly used for
development, it labels the result `UNSEALED_DIRECT_EXECUTION`, `NON_REPRODUCIBLE`, and
`NO_PLATFORM_LINEAGE`.

## Data formatting and truncation

Chat-template errors block by default. A chat plan must carry the exact template hash; the trainer
does not silently replace it with `role: content` text.

Truncation analysis renders and tokenizes the complete pinned JSONL, not the first 256 rows. Any
over-length record blocks a default plan. `--allow-truncation` makes that policy explicit in the seal;
it does not silently truncate an otherwise refusing plan.

Corpus-scale streaming preparation is still future work. This guard is correct for the current
file-backed trainer, but the trainer still materializes the corpus in memory.

## CLI

Hub models require an immutable commit:

```bash
corpus-studio platform-plan \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --dataset ./my-dataset/examples.jsonl \
  --sequence-len 1024 \
  --epochs 1 \
  --out ./plan

corpus-studio platform-run ./plan/RunPlan.json \
  --subprocess \
  --out ./run
```

A local model directory is hashed instead of requiring `--model-revision`. A separate tokenizer Hub
commit can be supplied with `--tokenizer-revision`. Chat datasets additionally require
`--chat-template-sha256`.

Legacy plans are still parseable for inspection and migration. The training runner refuses them:
regenerate the plan against the current capability report and immutable inputs.

## Verification boundary

The contract, planner, worker protocol, fake workers, trainer adapters, schema generation, and UI
integration are covered by CPU/unit/integration tests. Those tests verify enforcement logic, not the
final hardware stack.

The historical native-Windows RTX 5070 run predates this contract and is not proof that the Phase 9B
path works on that hardware. The current native-Linux host's managed `backend-corpus-studio`
environment passed its exact minimal hardware-probe tuple, but that is a separate prerequisite rather
than Phase-9B workload proof. Native-Linux real-workload execution, full-sequence 7B behavior,
DeepSpeed NVMe offload, Linux FSDP, CPU/NVMe offload, bare-Linux FlashAttention for the real workload,
PCIe 4.0 NVMe throughput, sustained NVMe writes, real offload fit, and MoE runtime capability remain
explicitly unverified.

The repository now has CPU/unit evidence for the stricter training-success gate and complete RECORD
count admission. It has not been exercised by another model load or GPU workload. The preserved
manager-1.2 v3 wheel, environments, and unexecuted plans predate these semantics and cannot be reused
for new admission. The next operational slice, after merge and separate approval, is a new audited
worker wheel, new immutable environment IDs/locks, completely fresh matched plans, and at most one
non-retried 0.5B math/flash smoke pair. The user's in-progress 500-output corpus and every 7B workload
remain out of scope until the user marks them ready.
