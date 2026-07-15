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

import math
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
    FitClassification,
    HeartbeatBody,
    HelloBody,
    RunAcceptedBody,
    RunEvent,
    RunManifest,
    RunPlan,
    TerminalResultBody,
)
from corpus_studio.platform.backends import backend_manifest_digest
from corpus_studio.platform.enums import FailureTaxonomy, StageMarker
from corpus_studio.platform.artifacts import write_artifact_manifest
from corpus_studio.platform.common import new_uuid7_id
from corpus_studio.platform.process_control import (
    process_group_creation_flags,
    start_new_process_session,
    terminate_process_tree,
)
from corpus_studio.platform.supervisor import (
    RUN_EVENTS_FILENAME,
    ProducedArtifact,
    RunnerFailure,
    RunEventSink,
    SupervisedRun,
    TelemetryControl,
    _append_event_line,
    _now_iso,
    _sanitize_id,
    run_record_directory,
    validate_training_success_evidence,
    write_run_manifest,
)
from corpus_studio.platform.worker_protocol import (
    WorkerProtocolError,
    build_worker_message,
    decode_worker_message,
    encode_worker_message,
    parse_worker_body,
)


_PREFLIGHT_STAGES = frozenset(
    {
        StageMarker.process_start,
        StageMarker.dataset_verification,
        StageMarker.execution_config_verified,
        StageMarker.env_loaded,
        StageMarker.cuda_init,
        StageMarker.tokenizer_load,
        StageMarker.dataset_formatting,
        StageMarker.truncation_analysis,
        StageMarker.attention_policy_applied,
        StageMarker.model_load,
        StageMarker.placement_verified,
        StageMarker.model_loaded,
        StageMarker.quantized,
        StageMarker.adapter_attached,
    }
)
_KilledFailure = tuple[FailureTaxonomy, str, StageMarker | None, str]


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
    stage: StageMarker | None = None,
    remediation: str | None = None,
    detail: str | None = None,
    exit_code: int | None = None,
) -> RunManifest:
    """A parent-built terminal manifest for the cases where the child's own ``terminal_result`` is
    unusable — a killed (hung) worker, a crash, a rejection, or a malformed terminal. The manifest
    target is always the backend identity pinned by the RunPlan, never an independently selected lane."""
    from corpus_studio.platform.common import HashRef, Ref  # noqa: PLC0415

    failure = FailureRecord(
        run_id=rid,
        taxonomy=taxonomy,
        stage=stage,
        message=message,
        exit_code=exit_code,
        detected_at=finished,
        remediation=remediation,
        detail=detail,
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


def _append_manifest_note(manifest: RunManifest, note: str) -> RunManifest:
    if not note or note in manifest.notes:
        return manifest
    return manifest.model_copy(
        update={"notes": manifest.notes + ("; " if manifest.notes else "") + note}
    )


def _bind_failure_record(
    failure: FailureRecord,
    *,
    run_id: str,
    current_stage: StageMarker | None,
    finished: str,
    returncode: int | None,
) -> FailureRecord:
    """Preserve authenticated child evidence while filling only absent parent-known fields."""

    return failure.model_copy(
        update={
            "run_id": run_id,
            "stage": failure.stage or current_stage,
            "detected_at": failure.detected_at or finished,
            "exit_code": (
                failure.exit_code
                if failure.exit_code is not None
                else (returncode if returncode not in (None, 0) else None)
            ),
        }
    )


def _measured_failure_fit(manifest: RunManifest) -> FitClassification | None:
    evidence = manifest.training_success_evidence
    if evidence is None or evidence.measured_peak is None:
        return None
    from corpus_studio.platform.watchdog import reconcile_measured_fit  # noqa: PLC0415

    return reconcile_measured_fit(evidence.measured_peak, proven=False)


def execute_run_subprocess(
    plan: RunPlan,
    *,
    run_id: str | None = None,
    runner_name: str = "auto",
    max_steps: int | None = None,
    sink: RunEventSink | None = None,
    silence_timeout_s: float = 600.0,
    preflight_timeout_s: float = 1800.0,
    heartbeat_interval_s: int = 30,
    out_dir: str | Path | None = None,
    clock: Callable[[], str] = _now_iso,
    worker_argv: list[str] | None = None,
    telemetry: TelemetryControl | None = None,
    warmup_steps: int = 2,
) -> SupervisedRun:
    """Run ``plan`` in a supervised child process and return its :class:`SupervisedRun`.

    Normal execution requires a genuine RunEvent within ``silence_timeout_s`` or the complete child
    tree is killed as ``KERNEL_STALL``. Resolved training preflight is different: once the worker emits
    a recognized, real setup stage, the entire preflight receives one non-extendable
    ``preflight_timeout_s`` budget. This prevents a large immutable-input check, full-corpus
    tokenization, or local model load from being mislabeled as a kernel stall while still bounding a
    genuinely stuck setup operation as ``TIMEOUT``. Repeated stages and heartbeats cannot extend that
    hard deadline. On ``optimizer_created`` (or the first optimizer metric), the ordinary silence
    deadline resumes.

    A child that exits without a ``terminal_result`` becomes ``ENVIRONMENT_FAILURE``. The child
    inherits stderr so trainer telemetry cannot fill a parent-owned pipe. A try/finally guarantees the
    child tree is terminated and reaped on every path. ``worker_argv`` is injectable for tests.
    """
    if not math.isfinite(silence_timeout_s) or silence_timeout_s <= 0:
        raise ValueError("silence_timeout_s must be finite and positive")
    if not math.isfinite(preflight_timeout_s) or preflight_timeout_s <= 0:
        raise ValueError("preflight_timeout_s must be finite and positive")
    if heartbeat_interval_s <= 0:
        raise ValueError("heartbeat_interval_s must be positive")
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
    events_handle: Any = None
    if record_dir is not None:
        record_dir.mkdir(parents=True, exist_ok=True)
        events_handle = (record_dir / RUN_EVENTS_FILENAME).open("w", encoding="utf-8")  # noqa: SIM115
    artifacts: list[ArtifactManifest] = []
    terminal_manifest: RunManifest | None = None
    deferred_terminal: RunEvent | None = None
    terminal_seen = False
    rejection: FailureRecord | None = None
    terminal_admission_failure: RunnerFailure | None = None
    killed_failure: _KilledFailure | None = None
    protocol_failure: str | None = None
    sink_errors: list[str] = []
    dispatched = False
    accepted = False
    seen_message_ids: set[str] = set()
    last_event_seq: int | None = None
    current_stage: StageMarker | None = None
    preflight_deadline: float | None = None
    preflight_finished = False
    resolved_training = plan.resolved_execution is not None
    dispatch_message_id = f"c-{rid}"
    progress_deadline = time.monotonic() + silence_timeout_s

    def _reset_progress_deadline(now: float | None = None) -> None:
        nonlocal progress_deadline
        progress_deadline = (time.monotonic() if now is None else now) + silence_timeout_s

    def _observe_stage(stage: StageMarker, now: float) -> None:
        nonlocal current_stage, preflight_deadline, preflight_finished
        current_stage = stage
        if resolved_training and not preflight_finished and stage in _PREFLIGHT_STAGES:
            if preflight_deadline is None:
                preflight_deadline = now + preflight_timeout_s
            return
        if resolved_training:
            preflight_finished = True
        preflight_deadline = None

    def _deadline_failure(now: float) -> _KilledFailure | None:
        if preflight_deadline is not None:
            if now < preflight_deadline:
                return None
            stage_name = current_stage.value if current_stage is not None else "unknown"
            return (
                FailureTaxonomy.TIMEOUT,
                f"training preflight exceeded its non-extendable {preflight_timeout_s:.0f}s "
                f"deadline during {stage_name!r} and was killed",
                current_stage,
                "inspect worker stderr and preflight events; fix the blocked operation or set an "
                "evidence-based --preflight-timeout",
            )
        if now < progress_deadline:
            return None
        return (
            FailureTaxonomy.KERNEL_STALL,
            f"the worker made no run progress for {silence_timeout_s:.0f}s and was killed "
            "(a hung run - for example, a fused-attention deadlock)",
            current_stage,
            "use a verified attention path, lower sequence_len, or raise --timeout only when the "
            "observed training operation is expected to be slow",
        )

    def _drive_telemetry(event: RunEvent) -> None:
        if telemetry is None:
            return
        try:
            if event.event_type == "stage" and event.stage == StageMarker.process_start:
                telemetry.set_phase("setup")
            elif (
                event.event_type == "metric"
                and event.optimizer_step is not None
                and event.optimizer_step > 0
            ):
                phase = "warmup" if event.optimizer_step <= warmup_steps else "measured"
                telemetry.mark_step(event.optimizer_step, phase=phase)
            elif event.event_type == "terminal":
                telemetry.set_phase("teardown")
        except Exception:  # noqa: BLE001 - an observer cannot rewrite run truth.
            pass

    def _forward(event: RunEvent) -> None:
        events.append(event)
        if events_handle is not None:
            try:
                _append_event_line(events_handle, event)
            except Exception as exc:  # noqa: BLE001 - a durable-log write cannot rewrite run truth.
                label = f"event_log:{type(exc).__name__}"
                if label not in sink_errors:
                    sink_errors.append(label)
        _drive_telemetry(event)
        if sink is not None:
            try:
                sink(event)
            except Exception as exc:  # noqa: BLE001 - an observer cannot rewrite run truth.
                label = type(exc).__name__
                if label not in sink_errors:
                    sink_errors.append(label)

    # stderr is INHERITED (not a pipe): the trainer's tqdm/transformers write \r-based progress bars
    # with no newlines, which would deadlock a line-drained stderr pipe (buffer fills and the child
    # wedges on write). Inheriting sends telemetry straight to our stderr. Heartbeats never move either
    # deadline, and preflight has one absolute budget, so a wedged child remains bounded.
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
    # In subprocess mode the WORKER child does the GPU/host work, so root an attached sampler's
    # process-tree probe at the child, not the control-plane parent.
    if telemetry is not None and hasattr(telemetry, "set_root_pid"):
        telemetry.set_root_pid(proc.pid)  # type: ignore[attr-defined]
    try:
        # Reader thread → a queue, so the main loop can read WITH A TIMEOUT (a silent child never
        # enqueues → queue.get(timeout) trips → we kill it).
        lines: queue.Queue[tuple[str, str | None]] = queue.Queue()
        threading.Thread(
            target=_reader, args=(proc, lines), name=f"worker-reader-{rid}", daemon=True
        ).start()

        while True:
            now = time.monotonic()
            killed_failure = _deadline_failure(now)
            if killed_failure is not None:
                break
            active_deadline = (
                preflight_deadline if preflight_deadline is not None else progress_deadline
            )
            try:
                kind, payload = lines.get(timeout=max(0.0, active_deadline - now))
            except queue.Empty:
                continue
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
                if deferred_terminal is not None and message.type != "terminal_result":
                    raise WorkerProtocolError(
                        "worker sent a message after its terminal RunEvent"
                    )
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
                    if body.event_type == "terminal":
                        if deferred_terminal is not None:  # pragma: no cover - guarded above.
                            raise WorkerProtocolError("worker sent duplicate terminal RunEvents")
                        deferred_terminal = body
                    else:
                        _forward(body)
                    now = time.monotonic()
                    if body.stage is not None:
                        _observe_stage(body.stage, now)
                    elif body.event_type == "metric" and body.optimizer_step is not None:
                        _observe_stage(StageMarker.optimizer_step, now)
                    _reset_progress_deadline(now)
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
                    try:
                        if deferred_terminal is not None:
                            terminal_state = (deferred_terminal.payload or {}).get("state")
                            if terminal_state != body.run_manifest.state:
                                raise WorkerProtocolError(
                                    "terminal RunEvent state does not match terminal_result"
                                )
                        terminal_manifest = _parse_terminal(body, plan, rid, events, artifacts)
                    except RunnerFailure as exc:
                        terminal_admission_failure = exc
                    except WorkerProtocolError:
                        raise
                    except Exception as exc:  # noqa: BLE001 - filesystem admission must be total.
                        terminal_admission_failure = RunnerFailure(
                            f"terminal artifact admission failed: {exc}",
                            taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                            stage=StageMarker.export,
                            remediation="preserve the run and inspect its artifact paths and bytes",
                        )
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
                    rejection = body
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
        plan, rid, runner_name, terminal_manifest, terminal_seen, rejection, killed_failure,
        terminal_admission_failure, protocol_failure, proc.returncode, started, finished,
        out_dir_str, current_stage,
    )
    if record_dir is not None:
        # Persist the child's integrity-checked ArtifactManifests too (the child ran execute_run
        # WITHOUT out_dir, so it built them but didn't write them). Write these before the terminal
        # manifest so a durable succeeded manifest can never point at missing artifact evidence.
        try:
            for artifact_manifest in artifacts:
                write_artifact_manifest(artifact_manifest, record_dir)
        except Exception as exc:  # noqa: BLE001 - classify the durable artifact gate.
            if manifest.state == "succeeded":
                failed_fit = _measured_failure_fit(manifest)
                manifest = _failed_manifest(
                    plan,
                    rid,
                    taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                    message=f"artifact manifest persistence failed: {exc}",
                    target=runner_name,
                    started=started,
                    finished=finished,
                    out_dir=out_dir_str,
                    stage=StageMarker.export,
                    remediation="preserve the run directory and repair durable artifact storage",
                )
                manifest = manifest.model_copy(update={"final_fit": failed_fit})
        if sink_errors:
            manifest = _append_manifest_note(
                manifest,
                "event sink failures were isolated: " + ", ".join(sink_errors),
            )
        try:
            write_run_manifest(manifest, record_dir)
        except Exception as exc:  # noqa: BLE001 - classify terminal-record durability.
            if manifest.state == "succeeded":
                failed_fit = _measured_failure_fit(manifest)
                manifest = _failed_manifest(
                    plan,
                    rid,
                    taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                    message=f"run manifest persistence failed: {exc}",
                    target=runner_name,
                    started=started,
                    finished=clock(),
                    out_dir=out_dir_str,
                    stage=StageMarker.export,
                    remediation="preserve the run directory and repair durable run-record storage",
                )
                manifest = manifest.model_copy(update={"final_fit": failed_fit})
                try:
                    write_run_manifest(manifest, record_dir)
                except Exception as retry_exc:  # noqa: BLE001 - no durable truth channel remains.
                    manifest = _append_manifest_note(
                        manifest,
                        "failed terminal manifest could not be persisted: "
                        + type(retry_exc).__name__,
                    )
            else:
                manifest = _append_manifest_note(
                    manifest,
                    "terminal manifest could not be persisted: " + type(exc).__name__,
                )
    elif sink_errors:
        manifest = _append_manifest_note(
            manifest,
            "event sink failures were isolated: " + ", ".join(sink_errors),
        )

    terminal_event = None
    if accepted:
        terminal_message = (
            manifest.failure.message
            if manifest.failure is not None and manifest.failure.message
            else manifest.state
        )
        terminal_event = RunEvent(
            event_type="terminal",
            run_id=rid,
            seq=(
                deferred_terminal.seq
                if deferred_terminal is not None
                else ((last_event_seq + 1) if last_event_seq is not None else 0)
            ),
            emitted_at=deferred_terminal.emitted_at if deferred_terminal is not None else clock(),
            message=terminal_message,
            payload={"state": manifest.state},
        )
        events.append(terminal_event)
    if sink is not None and terminal_event is not None:
        try:
            sink(terminal_event)
        except Exception as exc:  # noqa: BLE001 - observer failure cannot rewrite terminal truth.
            label = type(exc).__name__
            if label not in sink_errors:
                sink_errors.append(label)
            note = "event sink failures were isolated: " + ", ".join(sink_errors)
            manifest = _append_manifest_note(manifest, note)
            if record_dir is not None:
                try:
                    write_run_manifest(manifest, record_dir)
                except Exception:  # noqa: BLE001 - observer notes are not execution truth.
                    pass
    if events_handle is not None:
        try:
            events_handle.close()
        except Exception:  # noqa: BLE001 - the durable stream is already flushed per line.
            pass
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
    if manifest.base_model != plan.base_model:
        raise WorkerProtocolError(
            "terminal run_manifest base_model does not match the dispatched RunPlan"
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
        if artifact.base_model != plan.base_model:
            raise WorkerProtocolError(
                f"artifact {artifact.artifact_id!r} links to the wrong base model"
            )
    artifact_ids = [artifact.artifact_id for artifact in parsed_artifacts]
    if artifact_ids != manifest.artifact_ids:
        raise WorkerProtocolError(
            "terminal artifact list does not match run_manifest.artifact_ids"
        )
    if plan.resolved_execution is not None and manifest.state == "succeeded":
        adapters = [artifact for artifact in parsed_artifacts if artifact.kind == "adapter"]
        if manifest.training_success_evidence is None:
            raise RunnerFailure(
                "successful training terminal has no sealed training-success evidence",
                taxonomy=FailureTaxonomy.UPDATE_FAILURE,
                stage=StageMarker.optimizer_step,
            )
        from corpus_studio.platform.execution_config import (  # noqa: PLC0415
            ExecutionConfigurationError,
            verify_run_scoped_output_path,
        )

        execution = plan.resolved_execution
        assert execution is not None
        try:
            verify_run_scoped_output_path(
                execution,
                rid,
                observed_path=manifest.output_dir,
                require_exists=True,
            )
            for artifact in adapters:
                verify_run_scoped_output_path(
                    execution,
                    rid,
                    observed_path=artifact.path,
                    require_exists=True,
                )
        except ExecutionConfigurationError as exc:
            raise RunnerFailure(
                str(exc),
                taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                stage=StageMarker.export,
            ) from exc
        if any(
            artifact.integrity is None
            or artifact.integrity.current_integrity != "ok"
            or artifact.integrity.content_hash is None
            or artifact.integrity.metadata_hash is None
            for artifact in adapters
        ):
            raise RunnerFailure(
                "successful training terminal has no integrity-checked adapter bytes",
                taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                stage=StageMarker.export,
            )
        from corpus_studio.training.artifact_registry import (  # noqa: PLC0415
            compute_weight_content_hash,
        )

        if any(
            compute_weight_content_hash(artifact.path) != artifact.integrity.content_hash
            for artifact in adapters
            if artifact.integrity is not None
        ):
            raise RunnerFailure(
                "successful training terminal adapter weight bytes do not match its integrity hash",
                taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                stage=StageMarker.export,
            )
        validated = validate_training_success_evidence(
            plan,
            rid,
            events,
            [
                ProducedArtifact(
                    artifact_id=artifact.artifact_id,
                    kind=artifact.kind,
                    path=artifact.path,
                )
                for artifact in parsed_artifacts
            ],
            parsed_artifacts,
            manifest.training_success_evidence.execution,
            manifest.training_success_evidence.measured_peak,
        )
        if validated != manifest.training_success_evidence:
            raise RunnerFailure(
                "successful training terminal evidence does not match reconstructed admission",
                taxonomy=FailureTaxonomy.UPDATE_FAILURE,
                stage=StageMarker.optimizer_step,
            )
        measured_peak = manifest.training_success_evidence.measured_peak
        if (measured_peak is None) != (manifest.final_fit is None):
            raise RunnerFailure(
                "successful training terminal fit is not bound to raw peak-memory evidence",
                taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                stage=StageMarker.export,
            )
        if measured_peak is not None:
            from corpus_studio.platform.watchdog import (  # noqa: PLC0415
                reconcile_measured_fit,
            )

            if reconcile_measured_fit(measured_peak, proven=True) != manifest.final_fit:
                raise RunnerFailure(
                    "successful training terminal fit differs from parent-reconstructed evidence",
                    taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                    stage=StageMarker.export,
                )
    artifacts.extend(parsed_artifacts)
    return manifest


def _finalize(
    plan: RunPlan,
    rid: str,
    runner_name: str,
    terminal_manifest: RunManifest | None,
    terminal_seen: bool,
    rejection: FailureRecord | None,
    killed_failure: _KilledFailure | None,
    terminal_admission_failure: RunnerFailure | None,
    protocol_failure: str | None,
    returncode: int | None,
    started: str,
    finished: str,
    out_dir: str | None,
    current_stage: StageMarker | None,
) -> RunManifest:
    if killed_failure is not None:
        taxonomy, message, stage, remediation = killed_failure
        return _failed_manifest(
            plan, rid, taxonomy=taxonomy, message=message,
            target=runner_name, started=started, finished=finished, out_dir=out_dir,
            stage=stage,
            exit_code=returncode,
            remediation=remediation,
        )
    if terminal_manifest is not None:
        if terminal_manifest.failure is not None:
            terminal_manifest = terminal_manifest.model_copy(
                update={
                    "failure": _bind_failure_record(
                        terminal_manifest.failure,
                        run_id=rid,
                        current_stage=current_stage,
                        finished=finished,
                        returncode=returncode,
                    )
                }
            )
        return terminal_manifest
    if terminal_admission_failure is not None:
        return _failed_manifest(
            plan,
            rid,
            taxonomy=terminal_admission_failure.taxonomy,
            message=str(terminal_admission_failure),
            target=runner_name,
            started=started,
            finished=finished,
            out_dir=out_dir,
            stage=terminal_admission_failure.stage,
            remediation=terminal_admission_failure.remediation,
            exit_code=returncode,
        )
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
            stage=current_stage,
            exit_code=returncode,
            remediation="regenerate the RunPlan and align the core, worker, backend manifest, and "
            "managed environment lock",
        )
    if rejection is not None:
        bound_failure = _bind_failure_record(
            rejection,
            run_id=rid,
            current_stage=current_stage,
            finished=finished,
            returncode=returncode,
        )
        manifest = _failed_manifest(
            plan, rid, taxonomy=rejection.taxonomy,
            message=rejection.message or "the worker rejected the dispatch",
            target=runner_name, started=started, finished=finished, out_dir=out_dir,
            stage=bound_failure.stage,
            exit_code=bound_failure.exit_code,
            remediation=rejection.remediation
            or "the worker could not accept the plan — check the dispatch / worker version",
            detail=rejection.detail,
        )
        return manifest.model_copy(update={"failure": bound_failure})
    if terminal_seen:
        # A terminal_result arrived but its manifest didn't validate — a protocol/schema mismatch, NOT a
        # crash. Say so honestly rather than blaming a phantom exit code.
        return _failed_manifest(
            plan, rid, taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
            message="the worker sent a malformed terminal_result (protocol/schema drift) — the run's "
            "own outcome could not be recovered",
            target=runner_name, started=started, finished=finished, out_dir=out_dir,
            stage=current_stage,
            exit_code=returncode,
            remediation="check that the worker + core protocol/contract versions match",
        )
    # No terminal at all → the child's stdout closed without a result → a crash.
    return _failed_manifest(
        plan, rid, taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
        message=f"the worker process exited (code {returncode}) without a terminal result — it crashed "
        "before completing the run",
        target=runner_name, started=started, finished=finished, out_dir=out_dir, exit_code=returncode,
        stage=current_stage,
        remediation="check the worker's stderr; run 'corpus-studio train-check' to verify the runtime",
    )
