"""The supervised training WORKER — the child process of the subprocess run supervisor.

First emits a ``hello`` that binds its static backend manifest and environment identity, then reads a
``run_dispatch`` :class:`WorkerMessage` (the immutable RunPlan, by value) from stdin. After accepting
the dispatch it streams JSON-lines on **stdout**: ``run_accepted`` (with the pid) → ``event`` per
RunEvent → ``terminal_result`` (the RunManifest + a FailureTaxonomy outcome). stdout is exclusively
the protocol channel (one JSON WorkerMessage per line, flushed); **stderr is free** for telemetry.

Running in its own process is the whole point: it lets the PARENT time out and KILL a hung run — the
thing the in-process watchdog can only *detect* — and isolates a backend crash (a segfault, a CUDA
abort) from the core. Dependency-light at import (the heavy stack is lazy-imported by the runner it
dispatches to, never here).
"""

from __future__ import annotations

import itertools
import os
import sys
from typing import Any

from corpus_studio.platform.backends import backend_manifest_digest, get_worker_backend
from corpus_studio.platform.common import HashRef, JsonObject, Ref
from corpus_studio.platform.contracts import (
    FailureRecord,
    HelloBody,
    RunDispatchBody,
    WorkerBody,
    WorkerMessageType,
)
from corpus_studio.platform.enums import FailureTaxonomy
from corpus_studio.platform.worker_protocol import (
    PROTOCOL_VERSION as PROTOCOL_VERSION,
    WorkerProtocolError,
    build_worker_message,
    decode_worker_message,
    encode_worker_message,
    parse_worker_body,
)

_message_ids = itertools.count()


def _send(
    msg_type: WorkerMessageType,
    body: WorkerBody | JsonObject,
    *,
    correlation_id: str | None = None,
    out: Any = None,
) -> None:
    """Write one worker→core :class:`WorkerMessage` as a single flushed JSON line on stdout. Only the
    run's own (single) thread writes here — an event per completed step — so PROGRESS, not a liveness
    ping, is what reaches the parent (a bare liveness heartbeat would keep ticking on an independent
    thread even while the TRAINING thread is hung, defeating the parent's kill-on-stall)."""
    stream = out if out is not None else sys.stdout
    envelope = build_worker_message(
        msg_type,
        body,
        message_id=f"w-{next(_message_ids)}",
        correlation_id=correlation_id,
        direction="worker_to_core",
    )
    stream.write(encode_worker_message(envelope) + "\n")
    stream.flush()


def _build_runner(runner_name: str, max_steps: int | None) -> Any:
    """The Runner for ``runner_name`` — mirrors the ``platform-run`` selection (echo needs nothing;
    cpu_toy/training lazy-import the trainer)."""
    from corpus_studio.platform.supervisor import EchoRunner  # noqa: PLC0415

    if runner_name == "echo":
        return EchoRunner()
    from corpus_studio.platform.runners import TrainingRunner  # noqa: PLC0415

    return TrainingRunner(cpu_toy=(runner_name == "cpu_toy"), max_steps=max_steps)


def run_worker(
    dispatch_line: str,
    *,
    runner_name: str,
    backend_id: str,
    environment_ref: Ref,
    max_steps: int | None = None,
    out: Any = None,
) -> int:
    """Execute one dispatched run and stream it back. ``dispatch_line`` is the raw ``run_dispatch``
    JSON. Returns a process exit code (0 on a clean terminal — including a classified failure, which is
    a *result*, not a worker crash; non-zero only when the worker itself couldn't run the dispatch)."""
    from corpus_studio.platform.supervisor import execute_run  # noqa: PLC0415

    # CAPTURE the real stdout NOW, before running the trainer. The protocol channel is stdout, but the
    # trainer wraps trainer.train() in redirect_stdout(sys.stderr) (to keep tqdm/transformers off the
    # CLI's stdout) — so a live `sys.stdout` lookup inside the per-step sink would land on stderr, and
    # the parent (reading the stdout pipe) would see silence during training and false-kill a healthy
    # run as KERNEL_STALL. Binding the stream here makes every message reach the pipe regardless.
    stream = out if out is not None else sys.stdout

    correlation_id: str | None = None
    run_id = "unknown"
    try:
        envelope = decode_worker_message(dispatch_line, expected_direction="core_to_worker")
        correlation_id = envelope.message_id
        if envelope.type != "run_dispatch":
            raise WorkerProtocolError(
                f"first core message must be 'run_dispatch', received {envelope.type!r}"
            )
        if envelope.correlation_id is not None:
            raise WorkerProtocolError("run_dispatch must not carry a correlation_id")
        parsed = parse_worker_body(envelope)
        if not isinstance(parsed, RunDispatchBody):  # pragma: no cover - map is canonical
            raise WorkerProtocolError("run_dispatch selected the wrong body contract")
        plan = parsed.plan
        run_id = parsed.run_id
        from corpus_studio.platform.planner import verify_run_plan_hash  # noqa: PLC0415

        if not verify_run_plan_hash(plan):
            raise ValueError("plan_hash does not match the canonical plan body")
        backend = get_worker_backend(backend_id)
        if backend is None:
            raise WorkerProtocolError(f"unknown worker backend {backend_id!r}")
        expected_backend_hash = backend_manifest_digest(backend)
        actual_backend_hash = plan.backend_ref.hash.value if plan.backend_ref.hash else None
        backend_hash_algo = plan.backend_ref.hash.algo if plan.backend_ref.hash else None
        if (
            plan.backend_ref.id != backend_id
            or backend_hash_algo != "sha256"
            or actual_backend_hash != expected_backend_hash
        ):
            raise WorkerProtocolError(
                "RunPlan backend_ref does not match this worker's backend manifest identity"
            )
        if plan.environment_ref != environment_ref:
            raise WorkerProtocolError(
                "RunPlan environment_ref does not match this worker's environment identity"
            )
    except (ValueError, KeyError, TypeError, WorkerProtocolError) as exc:
        _send(
            "run_rejected",
            FailureRecord(
                run_id=run_id,
                taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                message=f"malformed or incompatible run_dispatch: {exc}",
            ),
            correlation_id=correlation_id,
            out=stream,
        )
        return 2

    _send(
        "run_accepted",
        {"run_id": run_id, "pid": os.getpid()},
        correlation_id=correlation_id,
        out=stream,
    )

    runner = _build_runner(runner_name, max_steps)
    # Stream each RunEvent to the parent as it is produced (the sink runs synchronously inside
    # execute_run, so ordering + backpressure are preserved over the pipe). Each metric event is a
    # COMPLETED STEP — real progress — which is what resets the parent's silence timer; a hung training
    # thread emits none, so the parent times out and kills it (the point of the subprocess model).
    result = execute_run(
        plan,
        runner,
        run_id=run_id,
        sink=lambda event: _send(
            "event", event, correlation_id=correlation_id, out=stream
        ),
    )
    manifest = result.manifest
    outcome = (
        FailureTaxonomy.PASS
        if manifest.failure is None
        else manifest.failure.taxonomy
    )
    _send(
        "terminal_result",
        {
            "run_id": run_id,
            "outcome": outcome.value,
            "run_manifest": manifest.model_dump(mode="json"),
            "artifacts": [a.model_dump(mode="json") for a in result.artifacts],
            "failure": manifest.failure.model_dump(mode="json") if manifest.failure else None,
        },
        correlation_id=correlation_id,
        out=stream,
    )
    return 0


def main() -> None:
    """CLI entrypoint: read the single ``run_dispatch`` line from stdin and run it. Invoked as
    ``python -m corpus_studio.platform.worker --runner <name>`` by the subprocess supervisor."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="corpus-studio-worker")
    parser.add_argument("--runner", default="echo", choices=["echo", "cpu_toy", "training"])
    parser.add_argument("--backend-id", required=True)
    parser.add_argument("--environment-id", required=True)
    parser.add_argument("--environment-hash")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    environment_ref = Ref(
        id=args.environment_id,
        hash=HashRef(value=args.environment_hash) if args.environment_hash else None,
    )
    backend = get_worker_backend(args.backend_id)
    if backend is None:
        _send(
            "failure",
            FailureRecord(
                taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                message=f"unknown worker backend {args.backend_id!r}",
            ),
        )
        raise SystemExit(2)
    _send(
        "hello",
        HelloBody(
            worker_id=f"{args.backend_id}-{os.getpid()}",
            backend=backend,
            environment_ref=environment_ref,
        ),
    )

    dispatch_line = sys.stdin.readline()
    if not dispatch_line.strip():
        _send(
            "run_rejected",
            FailureRecord(
                run_id="unknown",
                taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                message="no run_dispatch received on stdin",
            ),
        )
        raise SystemExit(2)
    raise SystemExit(
        run_worker(
            dispatch_line,
            runner_name=args.runner,
            backend_id=args.backend_id,
            environment_ref=environment_ref,
            max_steps=args.max_steps,
        )
    )


if __name__ == "__main__":
    main()
