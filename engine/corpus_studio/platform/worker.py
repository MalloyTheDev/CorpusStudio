"""The supervised training WORKER — the child process of the subprocess run supervisor.

Reads a ``run_dispatch`` :class:`WorkerMessage` (the immutable RunPlan, by value) from stdin, runs it
through the in-process supervisor + the chosen :class:`Runner`, and streams the result back as
WorkerMessages (JSON-lines) on **stdout**: ``run_accepted`` (with the pid) → ``event`` per RunEvent →
``terminal_result`` (the RunManifest + a FailureTaxonomy outcome). stdout is the protocol channel (one
JSON WorkerMessage per line, flushed); **stderr is free** for telemetry.

Running in its own process is the whole point: it lets the PARENT time out and KILL a hung run — the
thing the in-process watchdog can only *detect* — and isolates a backend crash (a segfault, a CUDA
abort) from the core. Dependency-light at import (the heavy stack is lazy-imported by the runner it
dispatches to, never here).
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from typing import Any

from corpus_studio.platform.contracts import RunPlan
from corpus_studio.platform.enums import FailureTaxonomy

PROTOCOL_VERSION = "1.0.0"

_message_ids = itertools.count()


def _send(msg_type: str, body: dict[str, Any], *, out: Any = None) -> None:
    """Write one worker→core :class:`WorkerMessage` as a single flushed JSON line on stdout. Only the
    run's own (single) thread writes here — an event per completed step — so PROGRESS, not a liveness
    ping, is what reaches the parent (a bare liveness heartbeat would keep ticking on an independent
    thread even while the TRAINING thread is hung, defeating the parent's kill-on-stall)."""
    stream = out if out is not None else sys.stdout
    envelope = {
        "protocol_version": PROTOCOL_VERSION,
        "message_id": f"w-{next(_message_ids)}",
        "direction": "worker_to_core",
        "type": msg_type,
        "body": body,
    }
    stream.write(json.dumps(envelope, separators=(",", ":")) + "\n")
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
    max_steps: int | None = None,
    out: Any = None,
) -> int:
    """Execute one dispatched run and stream it back. ``dispatch_line`` is the raw ``run_dispatch``
    JSON. Returns a process exit code (0 on a clean terminal — including a classified failure, which is
    a *result*, not a worker crash; non-zero only when the worker itself couldn't run the dispatch)."""
    from corpus_studio.platform.supervisor import execute_run  # noqa: PLC0415

    try:
        envelope = json.loads(dispatch_line)
        body = envelope["body"]
        plan = RunPlan.model_validate(body["plan"])
        run_id = body["run_id"]
    except (ValueError, KeyError, TypeError) as exc:
        _send("run_rejected", {"run_id": "unknown", "taxonomy": "ENVIRONMENT_FAILURE",
                               "message": f"malformed run_dispatch: {exc}"}, out=out)
        return 2

    _send("run_accepted", {"run_id": run_id, "pid": os.getpid()}, out=out)

    runner = _build_runner(runner_name, max_steps)
    # Stream each RunEvent to the parent as it is produced (the sink runs synchronously inside
    # execute_run, so ordering + backpressure are preserved over the pipe). Each metric event is a
    # COMPLETED STEP — real progress — which is what resets the parent's silence timer; a hung training
    # thread emits none, so the parent times out and kills it (the point of the subprocess model).
    result = execute_run(
        plan,
        runner,
        run_id=run_id,
        sink=lambda event: _send("event", event.model_dump(mode="json"), out=out),
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
        out=out,
    )
    return 0


def main() -> None:
    """CLI entrypoint: read the single ``run_dispatch`` line from stdin and run it. Invoked as
    ``python -m corpus_studio.platform.worker --runner <name>`` by the subprocess supervisor."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="corpus-studio-worker")
    parser.add_argument("--runner", default="echo", choices=["echo", "cpu_toy", "training"])
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    dispatch_line = sys.stdin.readline()
    if not dispatch_line.strip():
        _send("run_rejected", {"run_id": "unknown", "taxonomy": "ENVIRONMENT_FAILURE",
                               "message": "no run_dispatch received on stdin"})
        raise SystemExit(2)
    raise SystemExit(run_worker(dispatch_line, runner_name=args.runner, max_steps=args.max_steps))


if __name__ == "__main__":
    main()
