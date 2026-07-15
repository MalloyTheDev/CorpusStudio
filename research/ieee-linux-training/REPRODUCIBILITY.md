# Reproducibility and evidence plan

## Identity chain

Every result must resolve the following immutable chain without manual inference:

`study spec -> cell -> trial -> repository commit -> worker wheel -> environment lock -> capability report -> execution probe -> RunPlan -> ResolvedExecutionConfiguration -> model/tokenizer/dataset inputs -> worker event stream -> telemetry -> artifacts -> aggregate`

Each link records an ID, SHA-256, source path, size where applicable, and contract version. Hashes are recomputed by both the control plane and worker where the platform contract requires it. A missing or mismatched link prevents a successful result.

## Host and software record

Before each execution series, preserve: operating system and kernel; CPU and memory; swap; filesystem and free disk; GPU index, UUID, PCI identity, name, VRAM, compute capability, driver, CUDA runtime; relevant clocks and power limit as read-only facts; background GPU processes; repository commit; Python interpreter; worker-wheel filename, size, SHA-256 and METADATA SHA-256; environment recipe, resolution, package sources, exact pins, installed inventory, lock, health, and capability evidence.

No driver, system CUDA, clock, voltage, or power-limit change is part of this study.

## Immutable inputs

Models and tokenizers are local immutable snapshots with exact resolved revisions, per-file hashes, aggregate content hashes, license evidence, and `trust_remote_code=false`. Source-corpus bytes and provenance/license evidence are frozen before the first new matrix result. Deterministic sequence views are keyed by model-tokenizer hash and sequence length; each view freezes exact-length rendered examples, row order, labels, masks, and content hashes before planning. Paired paths consume the identical view.

No model, dataset, or result is uploaded. Network access is not used during training. Package installation is permitted only while building reviewed managed environments from exact plans and package indexes; package sources are preserved in the environment lock.

## Raw evidence layout

Large outputs live outside the checkout under a study root on `/mnt/training-nvme`. The harness creates immutable per-trial directories and never overwrites a completed or failed attempt.

```text
results/
  raw/<cell-id>/<trial-id>/
  manifests/
  aggregates/
  tables/
  latex/
  figures/
  logs/
  failures/
```

A raw trial directory contains the experiment specification hash, cell/trial manifest, pre/post environment health, pre/post GPU state, exact argv, RunPlan, resolved configuration, worker stdout/stderr and authenticated events, process status, 200 ms telemetry, adapter/artifact manifest when produced, failure record when applicable, and a `SHA256SUMS` over preserved evidence.

The source repository may contain only small reviewable specifications, harness code, manifests, deterministic summaries, tables, plots, and paper drafts. Environments, wheels, model snapshots, datasets, raw logs, and adapters remain outside it.

## Execution replay

The experiment harness consumes sealed RunPlans and invokes `platform-run`; it never calls the trainer directly or applies semantic overrides. It is resumable at immutable trial granularity and runs only one GPU cell at a time. Completed trials are detected by their validated terminal manifest and evidence hashes, not by directory existence.

Before replay, verify the exact repository commit, worker bytes, environment lock and health, capability tuple, immutable input hashes, GPU UUID, and study specification. A changed worker, environment lock, input, or semantic field requires a fresh plan and distinct trial identity.

## Deterministic aggregation

Aggregates are generated only from validated raw manifests. The aggregation program:

- sorts cells and trials by stable IDs;
- validates all input hashes and protocol versions;
- applies only preregistered warm-up exclusions;
- retains failures without numeric imputation;
- computes metrics and confidence intervals from unrounded raw values;
- writes JSON, CSV, Markdown, LaTeX, and figures from the same normalized records;
- records its repository commit, command, interpreter, input manifest hash, and output hashes;
- produces byte-for-byte stable outputs when timestamps and paths are represented as normalized data fields rather than generation metadata.

Every plotted point and table cell links back to raw trial IDs. Manually entered measured values are prohibited.

## Failures and amendments

Partial telemetry, logs, plans, manifests, and post-failure health evidence are preserved. Failed cells remain present in completion and failure matrices. An infrastructure retry is a new attempt linked to the original, not an overwrite.

After any new result exists, protocol changes use a versioned amendment as defined in `PROTOCOL.md`. Aggregates group by protocol version; retrospective changes never silently alter primary analyses.

## Current non-claims

At protocol freeze, no real optimizer step has passed through `platform-run` and sequence 4096 is unverified. Historical tiny-probe and placement evidence cannot be promoted into benchmark evidence. Reproduction instructions will not claim a cell until its exact sealed run and complete hashes exist.
