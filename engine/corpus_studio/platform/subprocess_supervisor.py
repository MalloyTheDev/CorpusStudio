"""The subprocess run supervisor — the PARENT side of the worker protocol.

``execute_run_subprocess`` spawns a :mod:`corpus_studio.platform.worker` child, dispatches the
child's worker-first ``hello``, validates its backend/environment identity, then dispatches the
immutable RunPlan (a ``run_dispatch`` WorkerMessage on stdin). It consumes ``run_accepted`` →
``event`` × N → ``terminal_result`` from stdout, forwarding each :class:`RunEvent` to the sink and
returning the same :class:`SupervisedRun` the in-process supervisor returns.

Why this exists: it owns a PROCESS it can terminate, so it can do the one thing the in-process
watchdog cannot — **time out and KILL a hung run** (e.g. the sm_120 fused-attention deadlock) and
classify it honestly as ``KERNEL_STALL``. It also isolates a backend crash (a segfault / CUDA abort)
from the core: the child dying is a classified failure here, not a core crash.

Reading a pipe with a timeout is done with a reader thread + a queue (cross-platform; ``select`` on
pipes isn't portable to Windows). Dependency-light: stdlib + platform contracts only.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from corpus_studio.platform.contracts import (
    ArtifactManifest,
    FailureRecord,
    HeartbeatBody,
    HelloBody,
    RunAcceptedBody,
    RunEvent,
    RunManifest,
    RunPlan,
    TerminalResultBody,
)
from corpus_studio.platform.backends import backend_manifest_digest
from corpus_studio.platform.enums import FailureTaxonomy
from corpus_studio.platform.artifacts import write_artifact_manifest
from corpus_studio.platform.common import new_uuid7_id
from corpus_studio.platform.process_control import (
    process_group_creation_flags,
    start_new_process_session,
    terminate_process_tree,
)
from corpus_studio.platform.supervisor import (
    RunEventSink,
    SupervisedRun,
    _now_iso,
    _sanitize_id,
    run_record_directory,
    write_run_manifest,
)
from corpus_studio.platform.worker_protocol import (
    WorkerProtocolError,
    build_worker_message,
    decode_worker_message,
    encode_worker_message,
    parse_worker_body,
)

def worker_identity_argv(plan: RunPlan) -> list[str]:
    """Literal argv tokens that bind a worker process to the plan identities it must present."""

    argv = [
        "--backend-id",
        plan.backend_ref.id,
        "--environment-id",
        plan.environment_ref.id,
    ]
    if plan.environment_ref.hash is not None and plan.environment_ref.hash.value is not None:
        argv += ["--environment-hash", plan.environment_ref.hash.value]
    return argv


def _default_worker_argv(plan: RunPlan, runner_name: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "corpus_studio.platform.worker",
        "--runner",
        runner_name,
        *worker_identity_argv(plan),
    ]


def _dispatch_line(plan: RunPlan, run_id: str, heartbeat_interval_s: int) -> str:
    """The single ``run_dispatch`` WorkerMessage JSON the parent writes to the child's stdin."""
    envelope = build_worker_message(
        "run_dispatch",
        {
            "run_id": run_id,
            "plan": plan.model_dump(mode="json"),
            "heartbeat_interval_seconds": heartbeat_interval_s,
        },
        message_id=f"c-{run_id}",
        direction="core_to_worker",
    )
    return encode_worker_message(envelope)


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
    unusable — a killed (hung) worker, a crash, a rejection, or a malformed terminal. The manifest
    target is always the backend identity pinned by the RunPlan, never an independently selected lane."""
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
        target=plan.backend_ref.id,
        output_dir=plan.export.output_dir,
        failure=failure,
    )


def execute_run_subprocess(
    plan: RunPlan,
    *,
    run_id: str | None = None,
    runner_name: str = "auto",
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
    ``sink`` + collected. **If no accepted run/event progress arrives within ``silence_timeout_s``**
    the child is presumed hung and is terminated + killed → ``KERNEL_STALL``. Heartbeats are parsed but
    deliberately do not extend that deadline, so an independent liveness thread cannot mask a hung
    training thread. A child that
    exits without a ``terminal_result`` → ``ENVIRONMENT_FAILURE`` (a crash). The child inherits stderr
    so trainer telemetry cannot fill a parent-owned pipe. A try/finally guarantees the child is
    terminated + reaped on EVERY path — a raising sink or a dispatch BrokenPipe can never orphan a live
    GPU worker. ``worker_argv`` is injectable for tests.

    NOTE on the load window: there is no progress signal during the initial model download/load, so set
    ``silence_timeout_s`` above your cold-cache load time (or pre-fetch the model). A silent load longer
    than the timeout is honestly indistinguishable from a hang here — and a bare liveness heartbeat is
    NOT the fix (it keeps ticking on its own thread while the TRAINING thread is hung, defeating the
    kill)."""
    rid = _sanitize_id(run_id or new_uuid7_id("run"))
    out_dir_str = str(out_dir) if out_dir is not None else None
    record_dir = run_record_directory(out_dir_str, rid) if out_dir_str is not None else None
    started = clock()

    # Refuse a broken seal at the public parent boundary, before a worker sees identities or input.
    from corpus_studio.platform.planner import verify_run_plan_hash  # noqa: PLC0415

    if not verify_run_plan_hash(plan):
        manifest = _failed_manifest(
            plan,
            rid,
            taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
            message="RunPlan hash verification failed; regenerate the plan before execution",
            target=runner_name,
            started=started,
            finished=clock(),
            out_dir=out_dir_str,
            remediation="regenerate the RunPlan from immutable inputs; do not mutate it after sealing",
        )
        if record_dir is not None:
            write_run_manifest(manifest, record_dir)
        return SupervisedRun(manifest=manifest, events=[], artifacts=[])

    if plan.resolved_execution is not None:
        from corpus_studio.platform.execution_config import (  # noqa: PLC0415
            verify_execution_configuration_hash,
        )

        if not verify_execution_configuration_hash(plan.resolved_execution):
            manifest = _failed_manifest(
                plan,
                rid,
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                message="resolved execution configuration hash verification failed",
                target=runner_name,
                started=started,
                finished=clock(),
                out_dir=out_dir_str,
                remediation="regenerate the RunPlan; do not mutate resolved execution fields",
            )
            if record_dir is not None:
                write_run_manifest(manifest, record_dir)
            return SupervisedRun(manifest=manifest, events=[], artifacts=[])
        if (
            max_steps is not None
            and max_steps != plan.resolved_execution.schedule.max_steps
        ):
            manifest = _failed_manifest(
                plan,
                rid,
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                message="max_steps cannot override the sealed execution schedule",
                target=runner_name,
                started=started,
                finished=clock(),
                out_dir=out_dir_str,
                remediation="create a derived RunPlan with a new execution hash",
            )
            if record_dir is not None:
                write_run_manifest(manifest, record_dir)
            return SupervisedRun(manifest=manifest, events=[], artifacts=[])

    from corpus_studio.platform.execution_config import (  # noqa: PLC0415
        ExecutionConfigurationError,
        verify_runner_lane,
    )

    try:
        if runner_name == "auto":
            from corpus_studio.platform.execution_config import required_runner_lane  # noqa: PLC0415

            runner_name = required_runner_lane(plan)
        verify_runner_lane(plan, runner_name)
    except ExecutionConfigurationError as exc:
        manifest = _failed_manifest(
            plan,
            rid,
            taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
            message=str(exc),
            target=runner_name,
            started=started,
            finished=clock(),
            out_dir=out_dir_str,
            remediation="dispatch the plan through its sealed runner lane",
        )
        if record_dir is not None:
            write_run_manifest(manifest, record_dir)
        return SupervisedRun(manifest=manifest, events=[], artifacts=[])

    argv = worker_argv or _default_worker_argv(plan, runner_name)

    events: list[RunEvent] = []
    artifacts: list[ArtifactManifest] = []
    terminal_manifest: RunManifest | None = None
    terminal_seen = False
    rejection: dict[str, Any] | None = None
    killed_reason: str | None = None
    protocol_failure: str | None = None
    dispatched = False
    accepted = False
    seen_message_ids: set[str] = set()
    last_event_seq: int | None = None
    dispatch_message_id = f"c-{rid}"
    progress_deadline = time.monotonic() + silence_timeout_s

    def _reset_progress_deadline() -> None:
        nonlocal progress_deadline
        progress_deadline = time.monotonic() + silence_timeout_s

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
        creationflags=process_group_creation_flags(),
        start_new_session=start_new_process_session(),
    )
    try:
        # Reader thread → a queue, so the main loop can read WITH A TIMEOUT (a silent child never
        # enqueues → queue.get(timeout) trips → we kill it).
        lines: queue.Queue[tuple[str, str | None]] = queue.Queue()
        threading.Thread(
            target=_reader, args=(proc, lines), name=f"worker-reader-{rid}", daemon=True
        ).start()

        while True:
            remaining = progress_deadline - time.monotonic()
            if remaining <= 0:
                killed_reason = (
                    f"the worker made no run progress for {silence_timeout_s:.0f}s and was killed "
                    "(a hung run - for example, a fused-attention deadlock)"
                )
                break
            try:
                kind, payload = lines.get(timeout=remaining)
            except queue.Empty:
                killed_reason = (
                    f"the worker made no run progress for {silence_timeout_s:.0f}s and was killed "
                    "(a hung run - for example, a fused-attention deadlock)"
                )
                break
            if kind == "eof":
                break
            try:
                message = decode_worker_message(
                    payload or "", expected_direction="worker_to_core"
                )
                if message.message_id in seen_message_ids:
                    raise WorkerProtocolError(
                        f"duplicate worker message_id {message.message_id!r}"
                    )
                seen_message_ids.add(message.message_id)

                if not dispatched:
                    _validate_hello(message.type, parse_worker_body(message), plan)
                    if message.correlation_id is not None:
                        raise WorkerProtocolError("hello must not carry a correlation_id")
                    _write_dispatch(proc, plan, rid, heartbeat_interval_s)
                    dispatched = True
                    _reset_progress_deadline()
                    continue

                if message.correlation_id != dispatch_message_id:
                    raise WorkerProtocolError(
                        f"message correlation_id {message.correlation_id!r} does not match "
                        f"dispatch {dispatch_message_id!r}"
                    )
                body = parse_worker_body(message)
                if message.type == "hello":
                    raise WorkerProtocolError("worker sent a second hello")
                if message.type == "run_accepted":
                    if accepted:
                        raise WorkerProtocolError("worker sent duplicate run_accepted")
                    if not isinstance(body, RunAcceptedBody):  # pragma: no cover - canonical map
                        raise WorkerProtocolError("run_accepted selected the wrong body contract")
                    _require_run_id(body.run_id, rid, "run_accepted")
                    expected_execution_hash = (
                        plan.resolved_execution.configuration_hash
                        if plan.resolved_execution is not None
                        else None
                    )
                    if body.execution_configuration_hash != expected_execution_hash:
                        raise WorkerProtocolError(
                            "run_accepted execution configuration hash does not match the dispatch"
                        )
                    accepted = True
                    _reset_progress_deadline()
                elif message.type == "event":
                    if not accepted:
                        raise WorkerProtocolError("event arrived before run_accepted")
                    if not isinstance(body, RunEvent):  # pragma: no cover - canonical map
                        raise WorkerProtocolError("event selected the wrong body contract")
                    _require_run_id(body.run_id, rid, "event")
                    if last_event_seq is not None and body.seq <= last_event_seq:
                        raise WorkerProtocolError(
                            f"event seq {body.seq} is not greater than prior seq {last_event_seq}"
                        )
                    last_event_seq = body.seq
                    _forward(body)
                    _reset_progress_deadline()
                elif message.type == "heartbeat":
                    if not accepted:
                        raise WorkerProtocolError("heartbeat arrived before run_accepted")
                    if not isinstance(body, HeartbeatBody):  # pragma: no cover - canonical map
                        raise WorkerProtocolError("heartbeat selected the wrong body contract")
                    _require_run_id(body.run_id, rid, "heartbeat")
                elif message.type == "terminal_result":
                    if not accepted:
                        raise WorkerProtocolError("terminal_result arrived before run_accepted")
                    terminal_seen = True
                    if not isinstance(body, TerminalResultBody):  # pragma: no cover - canonical map
                        raise WorkerProtocolError(
                            "terminal_result selected the wrong body contract"
                        )
                    terminal_manifest = _parse_terminal(body, plan, rid, events, artifacts)
                    break  # terminal_result is the last legal message
                elif message.type in {"run_rejected", "failure"}:
                    if message.type == "run_rejected" and accepted:
                        raise WorkerProtocolError("run_rejected arrived after run_accepted")
                    if not isinstance(body, FailureRecord):  # pragma: no cover - canonical map
                        raise WorkerProtocolError(
                            f"{message.type} selected the wrong body contract"
                        )
                    if body.run_id is not None:
                        _require_run_id(body.run_id, rid, message.type)
                    rejection = body.model_dump(mode="json")
                    break
                else:
                    raise WorkerProtocolError(
                        f"message type {message.type!r} is illegal during a run"
                    )
            except WorkerProtocolError as exc:
                protocol_failure = str(exc)
                break
    finally:
        # GUARANTEE no orphan: terminate the full worker tree and reap the direct child on EVERY path.
        terminate_process_tree(proc, wait_timeout_seconds=10.0)

    finished = clock()
    manifest = _finalize(
        plan, rid, runner_name, terminal_manifest, terminal_seen, rejection, killed_reason,
        protocol_failure, proc.returncode, started, finished, out_dir_str,
    )
    if record_dir is not None:
        write_run_manifest(manifest, record_dir)
        # Persist the child's integrity-checked ArtifactManifests too (the child ran execute_run
        # WITHOUT out_dir, so it built them but didn't write them). Same machine → the paths are valid.
        for artifact_manifest in artifacts:
            write_artifact_manifest(artifact_manifest, record_dir)
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

def _taxonomy(value: Any) -> FailureTaxonomy:
    """A FailureTaxonomy string from a child message → the enum, defaulting to ENVIRONMENT_FAILURE."""
    try:
        return FailureTaxonomy(str(value))
    except ValueError:
        return FailureTaxonomy.ENVIRONMENT_FAILURE


def _require_run_id(actual: str, expected: str, message_type: str) -> None:
    if actual != expected:
        raise WorkerProtocolError(
            f"{message_type} run_id {actual!r} != dispatched run_id {expected!r}"
        )


def _validate_hello(message_type: str, body: object, plan: RunPlan) -> None:
    """Bind the worker's self-declared backend and environment to the immutable RunPlan."""

    if message_type != "hello" or not isinstance(body, HelloBody):
        raise WorkerProtocolError("the first worker message must be hello")
    if (
        plan.backend_ref.hash is None
        or plan.backend_ref.hash.algo != "sha256"
        or plan.backend_ref.hash.value is None
    ):
        raise WorkerProtocolError(
            "RunPlan backend_ref is not hash-pinned; regenerate it for worker protocol 2.0.0"
        )
    if body.backend.backend_id != plan.backend_ref.id:
        raise WorkerProtocolError(
            f"worker backend {body.backend.backend_id!r} != plan backend {plan.backend_ref.id!r}"
        )
    actual_backend_hash = backend_manifest_digest(body.backend)
    if actual_backend_hash != plan.backend_ref.hash.value:
        raise WorkerProtocolError(
            "worker backend manifest digest does not match the RunPlan backend_ref"
        )
    if body.environment_ref != plan.environment_ref:
        raise WorkerProtocolError(
            "worker environment_ref does not match the RunPlan environment_ref"
        )


def _parse_terminal(
    body: TerminalResultBody,
    plan: RunPlan,
    rid: str,
    events: list[RunEvent],
    artifacts: list[ArtifactManifest],
) -> RunManifest:
    """Validate terminal identity/linkage before accepting any child-produced artifacts."""

    _require_run_id(body.run_id, rid, "terminal_result")
    manifest = body.run_manifest
    expected_plan_hash = manifest.plan_ref.hash.value if manifest.plan_ref.hash else None
    expected_plan_algo = manifest.plan_ref.hash.algo if manifest.plan_ref.hash else None
    if (
        manifest.plan_ref.id != plan.plan_id
        or expected_plan_algo != "sha256"
        or expected_plan_hash != plan.plan_hash
    ):
        raise WorkerProtocolError("terminal run_manifest does not link to the dispatched RunPlan")
    if manifest.environment_ref != plan.environment_ref:
        raise WorkerProtocolError(
            "terminal run_manifest environment_ref does not match the dispatched RunPlan"
        )
    if manifest.dataset_ref != plan.dataset_ref:
        raise WorkerProtocolError(
            "terminal run_manifest dataset_ref does not match the dispatched RunPlan"
        )
    if manifest.target != plan.backend_ref.id:
        raise WorkerProtocolError(
            "terminal run_manifest target does not match the dispatched backend"
        )
    if body.failure != manifest.failure:
        raise WorkerProtocolError(
            "terminal_result failure does not match run_manifest.failure"
        )

    parsed_artifacts = list(body.artifacts)
    for artifact in parsed_artifacts:
        if artifact.producer_run_ref.id != rid:
            raise WorkerProtocolError(
                f"artifact {artifact.artifact_id!r} links to the wrong producer run"
            )
    artifact_ids = [artifact.artifact_id for artifact in parsed_artifacts]
    if artifact_ids != manifest.artifact_ids:
        raise WorkerProtocolError(
            "terminal artifact list does not match run_manifest.artifact_ids"
        )
    if plan.resolved_execution is not None and manifest.state == "succeeded":
        if not any(
            event.event_type == "metric"
            and event.optimizer_step is not None
            and event.optimizer_step > 0
            for event in events
        ):
            raise WorkerProtocolError(
                "successful training terminal has no optimizer-step evidence"
            )
        adapters = [artifact for artifact in parsed_artifacts if artifact.kind == "adapter"]
        if not adapters:
            raise WorkerProtocolError(
                "successful training terminal has no required adapter artifact"
            )
        from corpus_studio.platform.execution_config import (  # noqa: PLC0415
            run_scoped_training_output,
        )

        execution = plan.resolved_execution
        assert execution is not None
        expected_output = run_scoped_training_output(execution, rid).resolve(strict=False)
        if Path(manifest.output_dir).resolve(strict=False) != expected_output or any(
            Path(artifact.path).resolve(strict=False) != expected_output
            for artifact in adapters
        ):
            raise WorkerProtocolError(
                "successful training terminal deviated from the sealed run-scoped output"
            )
        if any(
            artifact.integrity is None
            or artifact.integrity.current_integrity != "ok"
            or artifact.integrity.content_hash is None
            for artifact in adapters
        ):
            raise WorkerProtocolError(
                "successful training terminal has no integrity-checked adapter bytes"
            )
        from corpus_studio.training.artifact_registry import (  # noqa: PLC0415
            compute_weight_content_hash,
        )

        if any(
            compute_weight_content_hash(artifact.path) != artifact.integrity.content_hash
            for artifact in adapters
            if artifact.integrity is not None
        ):
            raise WorkerProtocolError(
                "successful training terminal adapter weight bytes do not match its integrity hash"
            )
    artifacts.extend(parsed_artifacts)
    return manifest


def _finalize(
    plan: RunPlan,
    rid: str,
    runner_name: str,
    terminal_manifest: RunManifest | None,
    terminal_seen: bool,
    rejection: dict[str, Any] | None,
    killed_reason: str | None,
    protocol_failure: str | None,
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
            remediation="use math/eager attention (forced for native-Windows/WDDM sm_120), lower "
            "sequence_len, pre-fetch the model, or raise --timeout if the load/step is just slow.",
        )
    if terminal_manifest is not None:
        return terminal_manifest
    if protocol_failure is not None:
        return _failed_manifest(
            plan,
            rid,
            taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
            message=f"worker protocol violation: {protocol_failure}",
            target=runner_name,
            started=started,
            finished=finished,
            out_dir=out_dir,
            exit_code=returncode,
            remediation="regenerate the RunPlan and align the core, worker, backend manifest, and "
            "managed environment lock",
        )
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
