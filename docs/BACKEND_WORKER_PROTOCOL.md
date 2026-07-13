# Backend worker protocol

Status: **protocol 2.0 contract and fake-worker conformance foundation shipped**. This is an isolated
process boundary, not a real DeepSpeed, FSDP, NVMe-offload, or MoE runtime backend.

## Why protocol 2.0

Protocol 1.0 accepted a JSON object with a familiar `type` and `body`, but the parent did not enforce
the declared protocol version, direction, correlation, message order, backend identity, or managed
environment identity. A stale or wrong worker could therefore receive a RunPlan before the mismatch
was detected.

Protocol 2.0 is intentionally wire-incompatible: the worker must identify itself **before** the core
dispatches a run.

## Identity handshake

1. A newly generated `RunPlan.backend_ref` pins the SHA-256 digest of the exact static
   `BackendManifest` used by the planner.
2. The worker starts with literal `--backend-id`, `--environment-id`, and optional
   `--environment-hash` argv tokens. No shell command is constructed.
3. The worker emits `hello` first, containing its full static backend manifest and exact environment
   ref.
4. The core recomputes the backend-manifest digest and compares both identities with the RunPlan.
5. Only after those comparisons pass does the core send `run_dispatch`.

For a managed backend environment, the environment hash is the immutable `EnvironmentLock` digest.
Before launch, the Environment Manager also checks descriptor/lock recipe identity, current recipe
digest, recipe layer, recipe target versus backend ID, backend-manifest digest, functional state, and
live drift.

## Run state machine

The supervised run channel accepts this order:

```text
worker: hello
core:   run_dispatch
worker: run_accepted
worker: event | heartbeat ...
worker: terminal_result
```

`run_rejected` is legal before acceptance. A structured `failure` may terminate the exchange. The
parent rejects:

- a non-2.0 protocol version, wrong message direction, unknown field, or wrong body schema;
- anything other than `hello` as the first worker message, or a repeated hello;
- missing/wrong correlation IDs and duplicate message IDs;
- acceptance twice, telemetry before acceptance, or rejection after acceptance;
- a run ID that differs from the dispatched run;
- non-increasing `RunEvent.seq` values;
- a terminal outcome inconsistent with its `RunManifest` or `FailureRecord`;
- terminal plan/environment/dataset lineage that differs from the dispatched plan;
- artifact records whose producer or ordered IDs disagree with the terminal manifest.

stdout is reserved for one `WorkerMessage` JSON object per line. Worker diagnostics belong on stderr;
non-JSON stdout is a protocol failure, not ignored telemetry.

The parent resets its hang deadline only for the handshake, acceptance, and real `RunEvent` progress.
It validates heartbeats but does not let them extend the deadline, so a heartbeat thread cannot make a
hung training thread look healthy.

## Failure behavior

A protocol or identity mismatch becomes an `ENVIRONMENT_FAILURE` manifest and the worker process tree
is terminated and the direct child reaped. Workers are launched in a dedicated POSIX session or
Windows process group; timeout/cancellation reaches compiler, data-loader, launcher, or rank
descendants rather than killing only the direct child. A silent child still reaches the existing
timeout and is killed as `KERNEL_STALL`. A child that exits before a terminal result remains an
isolated environment/crash failure. No mismatch can be converted into success. Both public execution
entry points verify the RunPlan seal before invoking or spawning a runner, and the worker verifies it
again before constructing the runner.

## Compatibility

Legacy unpinned RunPlans remain parseable and retain their historical hash verification. They can
still be inspected or used by the in-process compatibility path, but protocol-2 subprocess dispatch
rejects them before sending a run. Regenerate the plan to obtain a hash-pinned backend ref.

The protocol and contracts remain torch-free. The conformance suite uses the real echo worker plus
small fake subprocesses for valid roundtrips, hangs, crashes, malformed bodies, wrong versions and
directions, backend/environment mismatches, correlation and ID violations, ordering errors, and
terminal-lineage failures. A real fake-worker descendant test also proves that timeout cleanup reaches
the worker's child process on the development host.

## Evidence boundary

These tests prove contract parsing, process supervision, identity binding, and deterministic failure
classification on the development host. They do **not** verify native-Linux RTX 5070 training,
DeepSpeed or FSDP, FlashAttention on bare Linux, NVMe performance/offload, full-sequence 7B training,
real offload fit, or MoE runtime capability.
