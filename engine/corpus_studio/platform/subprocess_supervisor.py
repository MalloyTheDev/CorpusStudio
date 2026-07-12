"""The subprocess run supervisor — the PARENT side of the worker protocol.

``execute_run_subprocess`` spawns a :mod:`corpus_studio.platform.worker` child, dispatches the
immutable RunPlan to it (a ``run_dispatch`` WorkerMessage on the child's stdin), and consumes the
child's WorkerMessage stream (``run_accepted`` → ``event`` × N → ``terminal_result``) from its stdout,
forwarding each :class:`RunEvent` to the sink and returning the same :class:`SupervisedRun` the
in-process :func:`~corpus_studio.platform.supervisor.execute_run` returns.

Why this exists: it owns a PROCESS it can terminate, so it can do the one thing the in-process
watchdog cannot — **time out and KILL a hung run** (e.g. the sm_120 fused-attention deadlock) and
classify it honestly as ``KERNEL_STALL``. It also isolates a backend crash (a segfault / CUDA abort)
from the core: the child dying is a classified failure here, not a core crash.

Reading a pipe with a timeout is done with a reader thread + a queue (cross-platform; ``select`` on
pipes isn't portable to Windows). Dependency-light: stdlib + platform contracts only.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from corpus_studio.platform.contracts import (
    ArtifactManifest,
    FailureRecord,
    RunEvent,
    RunManifest,
    RunPlan,
)
from corpus_studio.platform.enums import FailureTaxonomy
from corpus_studio.platform.artifacts import write_artifact_manifest
from corpus_studio.platform.supervisor import (
    RunEventSink,
    SupervisedRun,
    _now_iso,
    _sanitize_id,
    write_run_manifest,
)
from corpus_studio.platform.worker import PROTOCOL_VERSION

# How long the parent waits for a KILLED/exiting child to actually die before giving up on the join.
_REAP_TIMEOUT_S = 10.0


def _default_worker_argv(runner_name: str, max_steps: int | None) -> list[str]:
    argv = [sys.executable, "-m", "corpus_studio.platform.worker", "--runner", runner_name]
    if max_steps is not None:
        argv += ["--max-steps", str(max_steps)]
    return argv


def _dispatch_line(plan: RunPlan, run_id: str, heartbeat_interval_s: int) -> str:
    """The single ``run_dispatch`` WorkerMessage JSON the parent writes to the child's stdin."""
    envelope = {
        "protocol_version": PROTOCOL_VERSION,
        "message_id": f"c-{run_id}",
        "direction": "core_to_worker",
        "type": "run_dispatch",
        "body": {
            "run_id": run_id,
            "plan": plan.model_dump(mode="json"),
            "heartbeat_interval_seconds": heartbeat_interval_s,
        },
    }
    return json.dumps(envelope, separators=(",", ":"))


def _failed_manifest(
    plan: RunPlan,
    rid: str,
    *,
    taxonomy: FailureTaxonomy,
    message: str,
    target: str,
    started: str,
    finished: str,
    out_dir: str | None,
    remediation: str | None = None,
    exit_code: int | None = None,
) -> RunManifest:
    """A parent-built terminal manifest for the cases where the child's own ``terminal_result`` is
    unusable — a killed (hung) worker, a crash, a rejection, or a malformed terminal. ``target`` mirrors
    the success convention (the runner name), not a synthetic ``worker:<id>``."""
    from corpus_studio.platform.common import HashRef, Ref  # noqa: PLC0415

    failure = FailureRecord(
        run_id=rid,
        taxonomy=taxonomy,
        message=message,
        exit_code=exit_code,
        detected_at=finished,
        remediation=remediation,
    )
    return RunManifest(
        run_id=rid,
        plan_ref=Ref(id=plan.plan_id, hash=HashRef(value=plan.plan_hash)),
        environment_ref=plan.environment_ref,
        dataset_ref=plan.dataset_ref,
        created_at=started,
        updated_at=finished,
        started_at=started,
        finished_at=finished,
        state="failed",
        base_model=plan.base_model,
        target=target,
        output_dir=out_dir if out_dir is not None else plan.export.output_dir,
        failure=failure,
    )


def execute_run_subprocess(
    plan: RunPlan,
    *,
    run_id: str | None = None,
    runner_name: str = "echo",
    max_steps: int | None = None,
    sink: RunEventSink | None = None,
    silence_timeout_s: float = 600.0,
    heartbeat_interval_s: int = 30,
    out_dir: str | Path | None = None,
    clock: Callable[[], str] = _now_iso,
    worker_argv: list[str] | None = None,
) -> SupervisedRun:
    """Run ``plan`` in a supervised child process and return its :class:`SupervisedRun`.

    The child streams ``event`` messages (one per completed step = real progress) that are forwarded to
    ``sink`` + collected. **If no message arrives within ``silence_timeout_s``** the child is presumed
    hung and is terminated + killed → ``KERNEL_STALL`` (the in-process-impossible outcome). A child that
    exits without a ``terminal_result`` → ``ENVIRONMENT_FAILURE`` (a crash). The child's stderr is
    drained on a thread (so a chatty runner can never wedge on a full pipe) and forwarded here. A
    try/finally guarantees the child is terminated + reaped on EVERY path — a raising sink or a dispatch
    BrokenPipe can never orphan a live GPU worker. ``worker_argv`` is injectable for tests.

    NOTE on the load window: there is no progress signal during the initial model download/load, so set
    ``silence_timeout_s`` above your cold-cache load time (or pre-fetch the model). A silent load longer
    than the timeout is honestly indistinguishable from a hang here — and a bare liveness heartbeat is
    NOT the fix (it keeps ticking on its own thread while the TRAINING thread is hung, defeating the
    kill)."""
    rid = _sanitize_id(run_id or plan.plan_id)
    out_dir_str = str(out_dir) if out_dir is not None else None
    argv = worker_argv or _default_worker_argv(runner_name, max_steps)
    started = clock()

    events: list[RunEvent] = []
    artifacts: list[ArtifactManifest] = []
    terminal_manifest: RunManifest | None = None
    terminal_seen = False
    rejection: dict[str, Any] | None = None
    killed_reason: str | None = None

    def _forward(event: RunEvent) -> None:
        events.append(event)
        if sink is not None:
            sink(event)

    # stderr is INHERITED (not a pipe): the trainer's tqdm/transformers write \r-based progress bars
    # with no newlines, which would deadlock a line-drained stderr pipe (buffer fills → the child wedges
    # on write). Inheriting sends telemetry straight to our stderr, and — crucially — with no bogus
    # liveness heartbeat, a child that DID wedge on stderr makes no progress, emits no events, and is
    # killed by the silence timeout anyway. So the wedge is handled by the kill, not masked.
    proc = subprocess.Popen(  # noqa: S603 - argv is our own worker command (or a test injection)
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    try:
        # Reader thread → a queue, so the main loop can read WITH A TIMEOUT (a silent child never
        # enqueues → queue.get(timeout) trips → we kill it).
        lines: queue.Queue[tuple[str, str | None]] = queue.Queue()
        threading.Thread(
            target=_reader, args=(proc, lines), name=f"worker-reader-{rid}", daemon=True
        ).start()

        _write_dispatch(proc, plan, rid, heartbeat_interval_s)

        while True:
            try:
                kind, payload = lines.get(timeout=silence_timeout_s)
            except queue.Empty:
                killed_reason = (
                    f"the worker produced no output for {silence_timeout_s:.0f}s and was killed "
                    "(a hung run — e.g. the sm_120 fused-attention deadlock)"
                )
                break
            if kind == "eof":
                break
            message = _parse_message(payload or "")
            if message is None:
                continue
            mtype, mbody = message
            if mtype == "event":
                try:
                    _forward(RunEvent.model_validate(mbody))
                except (ValueError, TypeError):
                    continue  # a malformed event line is dropped, not fatal
            elif mtype == "terminal_result":
                terminal_seen = True
                terminal_manifest = _parse_terminal(mbody, artifacts)
                break  # the terminal is the last message
            elif mtype in ("run_rejected", "failure"):
                rejection = mbody  # capture the reason/taxonomy for honest classification
    finally:
        # GUARANTEE no orphan: terminate + reap the child on EVERY path (no-op if it already exited).
        _terminate(proc)

    finished = clock()
    manifest = _finalize(
        plan, rid, runner_name, terminal_manifest, terminal_seen, rejection, killed_reason,
        proc.returncode, started, finished, out_dir_str,
    )
    if out_dir_str is not None:
        write_run_manifest(manifest, out_dir_str)
        # Persist the child's integrity-checked ArtifactManifests too (the child ran execute_run
        # WITHOUT out_dir, so it built them but didn't write them). Same machine → the paths are valid.
        for artifact_manifest in artifacts:
            write_artifact_manifest(artifact_manifest, out_dir_str)
    return SupervisedRun(manifest=manifest, events=events, artifacts=artifacts)


def _reader(proc: subprocess.Popen[str], lines: queue.Queue[tuple[str, str | None]]) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.put(("line", line))
    lines.put(("eof", None))


def _write_dispatch(proc: subprocess.Popen[str], plan: RunPlan, rid: str, hb: int) -> None:
    """Send the run_dispatch to the child's stdin, then close it. Guarded: a fast-crashing child may
    already be gone, so the write/close can raise — the eof/exit path then classifies it, no orphan."""
    if proc.stdin is None:  # pragma: no cover - stdin is always PIPE here
        return
    try:
        proc.stdin.write(_dispatch_line(plan, rid, hb) + "\n")
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        pass
    finally:
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError):  # pragma: no cover - close-after-broken-pipe is benign
            pass


def _terminate(proc: subprocess.Popen[str]) -> None:
    """Terminate then (if it lingers) hard-kill the child, and reap it so no zombie/orphan is left. A
    no-op if the child already exited (Popen.terminate/kill/wait short-circuit once returncode is set)."""
    proc.terminate()
    try:
        proc.wait(timeout=_REAP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=_REAP_TIMEOUT_S)
        except subprocess.TimeoutExpired:  # pragma: no cover - the OS failed to reap a killed pid
            pass


def _taxonomy(value: Any) -> FailureTaxonomy:
    """A FailureTaxonomy string from a child message → the enum, defaulting to ENVIRONMENT_FAILURE."""
    try:
        return FailureTaxonomy(str(value))
    except ValueError:
        return FailureTaxonomy.ENVIRONMENT_FAILURE


def _parse_message(line: str) -> tuple[str, dict[str, Any]] | None:
    """A WorkerMessage line → ``(type, body)``, or ``None`` for a blank/non-JSON line (telemetry a
    misbehaving child might interleave — dropped, never fatal)."""
    line = line.strip()
    if not line:
        return None
    try:
        envelope = json.loads(line)
        return str(envelope["type"]), dict(envelope.get("body") or {})
    except (ValueError, KeyError, TypeError):
        return None


def _parse_terminal(body: dict[str, Any], artifacts: list[ArtifactManifest]) -> RunManifest | None:
    try:
        for raw in body.get("artifacts") or []:
            artifacts.append(ArtifactManifest.model_validate(raw))
        raw_manifest = body.get("run_manifest")
        return RunManifest.model_validate(raw_manifest) if raw_manifest else None
    except (ValueError, TypeError):
        return None


def _finalize(
    plan: RunPlan,
    rid: str,
    runner_name: str,
    terminal_manifest: RunManifest | None,
    terminal_seen: bool,
    rejection: dict[str, Any] | None,
    killed_reason: str | None,
    returncode: int | None,
    started: str,
    finished: str,
    out_dir: str | None,
) -> RunManifest:
    if killed_reason is not None:
        return _failed_manifest(
            plan, rid, taxonomy=FailureTaxonomy.KERNEL_STALL, message=killed_reason,
            target=runner_name, started=started, finished=finished, out_dir=out_dir,
            exit_code=returncode,
            remediation="use math/eager attention (the planner forces it on sm_120), lower "
            "sequence_len, pre-fetch the model, or raise --timeout if the load/step is just slow.",
        )
    if terminal_manifest is not None:
        # The child's own authoritative manifest (success OR its classified failure). Align output_dir
        # with the parent's out_dir (where the manifest is written), matching in-process execute_run.
        if out_dir is not None and terminal_manifest.output_dir != out_dir:
            return terminal_manifest.model_copy(update={"output_dir": out_dir})
        return terminal_manifest
    if rejection is not None:
        return _failed_manifest(
            plan, rid, taxonomy=_taxonomy(rejection.get("taxonomy")),
            message=str(rejection.get("message") or "the worker rejected the dispatch"),
            target=runner_name, started=started, finished=finished, out_dir=out_dir,
            exit_code=returncode,
            remediation="the worker could not accept the plan — check the dispatch / worker version",
        )
    if terminal_seen:
        # A terminal_result arrived but its manifest didn't validate — a protocol/schema mismatch, NOT a
        # crash. Say so honestly rather than blaming a phantom exit code.
        return _failed_manifest(
            plan, rid, taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
            message="the worker sent a malformed terminal_result (protocol/schema drift) — the run's "
            "own outcome could not be recovered",
            target=runner_name, started=started, finished=finished, out_dir=out_dir,
            exit_code=returncode,
            remediation="check that the worker + core protocol/contract versions match",
        )
    # No terminal at all → the child's stdout closed without a result → a crash.
    return _failed_manifest(
        plan, rid, taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
        message=f"the worker process exited (code {returncode}) without a terminal result — it crashed "
        "before completing the run",
        target=runner_name, started=started, finished=finished, out_dir=out_dir, exit_code=returncode,
        remediation="check the worker's stderr; run 'corpus-studio train-check' to verify the runtime",
    )
