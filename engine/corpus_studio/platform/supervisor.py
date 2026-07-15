"""The headless run supervisor — platform slice 3.

Consumes an immutable :class:`RunPlan`, executes it through a pluggable in-process
:class:`Runner`, streams :class:`RunEvent` envelopes (monotonic per-run ``seq``) to an injectable
sink, classifies the terminal outcome into the :class:`FailureTaxonomy`, and produces a crash-safe
:class:`RunManifest` (written with the atomic temp-then-replace convention proven in
``training.run_registry``, and its terminal state machine — ``succeeded / failed / cancelled``).

This is the first vertical that actually CONSUMES a ``RunPlan`` and EMITS the ``RunEvent`` stream the
whole platform boundary was designed around — dismantling the "the UI owns the trainer process"
gap that slices 1 (contracts) and 2 (env probes) only described.

Dependency-light: this module imports only the platform contracts + the standard library — there is
**no torch import at module load**. The :class:`EchoRunner` is a no-op that proves the harness (seq
ordering, terminal classification, manifest write, cancellation) on a core-only install with no GPU
and no ``[train]`` extra. Real runners (the cpu-toy / GPU trainer, which lazy-import the heavy stack)
plug into the same :class:`Runner` seam in a later slice.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from corpus_studio.platform.artifacts import build_artifact_manifest, write_artifact_manifest
from corpus_studio.platform.common import HashRef, MemoryMetrics, Ref, new_uuid7_id
from corpus_studio.platform.contracts import (
    ArtifactManifest,
    EventMetrics,
    FailureRecord,
    FitClassification,
    OptimizerStepLossEvidence,
    RunEvent,
    RunManifest,
    RunPlan,
    TrainingExecutionEvidence,
    TrainingSuccessEvidence,
)
from corpus_studio.platform.enums import FailureTaxonomy, StageMarker

EventType = Literal[
    "stage",
    "metric",
    "log",
    "warning",
    "checkpoint_written",
    "eval_result",
    "artifact_produced",
    "heartbeat",
    "terminal",
]

RunState = Literal["prepared", "running", "succeeded", "failed", "cancelled", "interrupted"]

_ID_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _now_iso() -> str:
    """UTC ISO-8601 timestamp — mirrors ``platform.profiler._now_iso`` (which is private there)."""
    return datetime.now(timezone.utc).isoformat()


def _sanitize_id(value: str) -> str:
    """Coerce an arbitrary string into a value matching ``RunManifest.run_id`` (``_ID``)."""
    cleaned = _ID_UNSAFE.sub("-", value).strip("-")
    return cleaned or "run"


def run_record_directory(root: str | Path, run_id: str) -> Path:
    """Return the collision-free platform record directory for one immutable run instance."""

    return Path(root) / "runs" / _sanitize_id(run_id)


# --------------------------------------------------------------------------------------------------
# Cancellation + runner-failure signalling
# --------------------------------------------------------------------------------------------------
class CancelToken:
    """A cooperative cancellation token (a ``threading.Event`` under the hood) that a runner polls
    between units of work. Cancelling is idempotent; a runner that observes :attr:`cancelled` must
    stop promptly and raise :class:`RunCancelled`."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


class RunCancelled(Exception):
    """Raised by a cooperative runner when it observes cancellation — the supervisor maps it to the
    ``cancelled`` terminal state (no :class:`FailureRecord`, since a cancel is not a failure)."""


class RunnerFailure(Exception):
    """A classified runner failure. Carries the taxonomy/stage the supervisor records in the
    manifest's :class:`FailureRecord` — so "it died" becomes an actionable category."""

    def __init__(
        self,
        message: str,
        *,
        taxonomy: FailureTaxonomy = FailureTaxonomy.FAIL,
        stage: StageMarker | None = None,
        remediation: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.taxonomy = taxonomy
        self.stage = stage
        self.remediation = remediation
        self.exit_code = exit_code


# --------------------------------------------------------------------------------------------------
# Produced artifacts + the event sink/context handed to a runner
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ProducedArtifact:
    """A weight artifact a runner produced. The supervisor records the id on the manifest; the
    platform NEVER moves/copies/deletes the underlying bytes (a full ``ArtifactManifest`` write is a
    later slice)."""

    artifact_id: str
    kind: str = "adapter"
    path: str = ""


RunEventSink = Callable[[RunEvent], None]


class RunContext:
    """Handed to a :class:`Runner`. The runner emits telemetry with the ``emit_*`` helpers (the
    ``seq`` + timestamp are assembled here, never by the runner) and polls :attr:`cancelled`."""

    def __init__(
        self,
        plan: RunPlan,
        run_id: str,
        sink: RunEventSink,
        cancel: CancelToken,
        clock: Callable[[], str] = _now_iso,
    ) -> None:
        self.plan = plan
        self.run_id = run_id
        self._sink = sink
        self._cancel = cancel
        self._clock = clock
        self._seq = 0
        # A runner may set the MEASURED fit (from the watchdog's observed peak) — the post-run
        # reconciliation of the calibrator's *predicted* fit. The supervisor records it on the manifest.
        self.final_fit: FitClassification | None = None
        # The training runner may report an observed peak and trainer-side execution evidence, but it
        # cannot promote either to success. ``execute_run`` owns the output/artifact gates and is the
        # only layer that may create TrainingSuccessEvidence or a proven native fit.
        self.measured_peak: MemoryMetrics | None = None
        self.training_execution_evidence: TrainingExecutionEvidence | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancel.cancelled

    def _event(self, event_type: EventType, **fields: Any) -> RunEvent:
        event = RunEvent(
            event_type=event_type,
            run_id=self.run_id,
            seq=self._seq,
            emitted_at=self._clock(),
            **fields,
        )
        self._seq += 1
        self._sink(event)
        return event

    def emit_stage(self, marker: StageMarker, message: str | None = None) -> RunEvent:
        return self._event("stage", stage=marker, message=message)

    def emit_metric(
        self,
        *,
        optimizer_step: int | None = None,
        loss: float | None = None,
        metrics: EventMetrics | None = None,
        message: str | None = None,
    ) -> RunEvent:
        if metrics is None and loss is not None:
            metrics = EventMetrics(loss=loss)
        return self._event(
            "metric", optimizer_step=optimizer_step, metrics=metrics, message=message
        )

    def emit_log(self, message: str) -> RunEvent:
        return self._event("log", message=message)

    def emit_warning(self, message: str) -> RunEvent:
        return self._event("warning", message=message)

    def emit_artifact(self, artifact: ProducedArtifact) -> RunEvent:
        return self._event(
            "artifact_produced",
            message=f"{artifact.kind}:{artifact.path}",
            payload={
                "artifact_id": artifact.artifact_id,
                "kind": artifact.kind,
                "path": artifact.path,
            },
        )

    def emit_terminal(self, state: RunState, message: str | None = None) -> RunEvent:
        return self._event("terminal", message=message or state, payload={"state": state})


# --------------------------------------------------------------------------------------------------
# The Runner protocol + the dependency-light EchoRunner
# --------------------------------------------------------------------------------------------------
class Runner(Protocol):
    """The pluggable unit of work. A runner performs the run, emitting telemetry through ``ctx`` and
    returning the artifacts it produced; it raises :class:`RunCancelled` on cooperative cancel or
    :class:`RunnerFailure` (or any exception) on failure."""

    name: str

    def run(self, ctx: RunContext) -> Sequence[ProducedArtifact]: ...


class EchoRunner:
    """A no-op runner that emits a canned ``process_start`` → ``metric``×N → ``export`` script so the
    supervisor is provable end-to-end on a core-only install (no GPU, no ``[train]`` extra). It does
    no real work; it exists to prove the harness, not to train."""

    name = "echo"

    def __init__(self, steps: int = 3) -> None:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.steps = steps

    def run(self, ctx: RunContext) -> Sequence[ProducedArtifact]:
        ctx.emit_stage(StageMarker.process_start, "echo run started")
        for step in range(1, self.steps + 1):
            if ctx.cancelled:
                raise RunCancelled
            ctx.emit_metric(
                optimizer_step=step,
                loss=round(1.0 / step, 4),
                message=f"[{step}/{self.steps}] step",
            )
        ctx.emit_stage(StageMarker.export, "echo run complete")
        return []


# --------------------------------------------------------------------------------------------------
# The supervisor
# --------------------------------------------------------------------------------------------------
@dataclass
class SupervisedRun:
    """The outcome of one supervised execution: the terminal :class:`RunManifest`, the full ordered
    :class:`RunEvent` stream that produced it, and an :class:`ArtifactManifest` (integrity-checked)
    for each weight artifact the run produced."""

    manifest: RunManifest
    events: list[RunEvent]
    artifacts: list[ArtifactManifest]


def validate_training_success_evidence(
    plan: RunPlan,
    run_id: str,
    events: Sequence[RunEvent],
    produced: Sequence[ProducedArtifact],
    artifacts: Sequence[ArtifactManifest],
    execution_evidence: TrainingExecutionEvidence | None,
    measured_peak: MemoryMetrics | None,
) -> TrainingSuccessEvidence:
    """Reconcile every resolved-training success gate before terminal PASS or proven fit."""

    execution = plan.resolved_execution
    if execution is None:  # pragma: no cover - caller restricts this helper to resolved training.
        raise RunnerFailure(
            "training success evidence requires a resolved execution",
            taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
            stage=StageMarker.process_start,
        )
    if execution_evidence is None:
        raise RunnerFailure(
            "resolved training returned without trainer execution evidence",
            taxonomy=FailureTaxonomy.UPDATE_FAILURE,
            stage=StageMarker.optimizer_step,
        )
    if (
        execution.schedule.max_steps is not None
        and execution_evidence.completed_optimizer_steps != execution.schedule.max_steps
    ):
        raise RunnerFailure(
            "completed optimizer steps do not match the sealed schedule",
            taxonomy=FailureTaxonomy.OPTIMIZER_FAILURE,
            stage=StageMarker.optimizer_step,
        )
    optimizer_events = [
        event
        for event in events
        if event.event_type == "stage" and event.stage == StageMarker.optimizer_created
    ]
    if len(optimizer_events) != 1 or not execution_evidence.optimizer_created:
        raise RunnerFailure(
            "resolved training requires exactly one real optimizer-created event",
            taxonomy=FailureTaxonomy.OPTIMIZER_FAILURE,
            stage=StageMarker.optimizer_created,
        )
    optimizer_event = optimizer_events[0]
    step_metrics = [
        event
        for event in events
        if event.event_type == "metric"
        and event.optimizer_step is not None
        and event.optimizer_step > 0
    ]
    if any(
        event.seq <= optimizer_event.seq
        or event.metrics is None
        or event.metrics.loss is None
        for event in step_metrics
    ):
        raise RunnerFailure(
            "every completed-step metric must follow optimizer creation and carry one finite loss",
            taxonomy=FailureTaxonomy.LOSS_EVIDENCE_FAILURE,
            stage=StageMarker.loss,
        )
    event_losses: list[OptimizerStepLossEvidence] = []
    for event in step_metrics:
        assert event.optimizer_step is not None
        assert event.metrics is not None and event.metrics.loss is not None
        event_losses.append(
            OptimizerStepLossEvidence(
                optimizer_step=event.optimizer_step,
                loss=event.metrics.loss,
            )
        )
    if event_losses != execution_evidence.step_losses:
        raise RunnerFailure(
            "RunEvent losses do not provide exactly one finite record for every completed step",
            taxonomy=FailureTaxonomy.LOSS_EVIDENCE_FAILURE,
            stage=StageMarker.loss,
        )

    from corpus_studio.platform.execution_config import (  # noqa: PLC0415
        ExecutionConfigurationError,
        verify_run_scoped_output_path,
    )

    try:
        expected_output = verify_run_scoped_output_path(
            execution,
            run_id,
            require_exists=True,
        )
    except ExecutionConfigurationError as exc:
        raise RunnerFailure(
            str(exc),
            taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
            stage=StageMarker.export,
        ) from exc
    adapters = [artifact for artifact in produced if artifact.kind == "adapter"]
    if (
        len(produced) != 1
        or len(adapters) != 1
        or Path(adapters[0].path).absolute() != expected_output
    ):
        raise RunnerFailure(
            "resolved training must return only one adapter at the sealed output path",
            taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
            stage=StageMarker.export,
        )
    adapter_manifests = [artifact for artifact in artifacts if artifact.kind == "adapter"]
    if (
        len(artifacts) != 1
        or len(adapter_manifests) != 1
        or adapter_manifests[0].artifact_id != adapters[0].artifact_id
    ):
        raise RunnerFailure(
            "adapter artifact manifest does not match the produced adapter",
            taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
            stage=StageMarker.export,
        )
    manifest = adapter_manifests[0]
    if Path(manifest.path).absolute() != expected_output:
        raise RunnerFailure(
            "adapter artifact manifest escaped the sealed output path",
            taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
            stage=StageMarker.export,
        )
    integrity = manifest.integrity
    if (
        integrity is None
        or integrity.current_integrity != "ok"
        or integrity.content_hash is None
        or integrity.metadata_hash is None
        or integrity.cheap_fingerprint is None
    ):
        raise RunnerFailure(
            "adapter artifact lacks complete byte and integrity evidence",
            taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
            stage=StageMarker.export,
        )
    from corpus_studio.platform.artifacts import (  # noqa: PLC0415
        validate_sealed_adapter_artifact,
    )
    from corpus_studio.training.artifact_registry import (  # noqa: PLC0415
        compute_weight_content_hash,
    )

    try:
        adapter_evidence = validate_sealed_adapter_artifact(
            manifest.path,
            execution,
            execution_evidence.adapter_export_state,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise RunnerFailure(
            f"adapter Safetensors/config validation failed: {exc}",
            taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
            stage=StageMarker.export,
        ) from exc
    if compute_weight_content_hash(manifest.path) != integrity.content_hash:
        raise RunnerFailure(
            "adapter weight bytes changed before terminal admission",
            taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
            stage=StageMarker.export,
        )
    if adapter_evidence.adapter_config_sha256 != integrity.metadata_hash:
        raise RunnerFailure(
            "adapter config bytes changed before terminal admission",
            taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
            stage=StageMarker.export,
        )
    return TrainingSuccessEvidence(
        execution=execution_evidence,
        output_path_verified=True,
        adapter_bytes_verified=True,
        artifact_integrity_verified=True,
        adapter_safetensors_sha256=adapter_evidence.safetensors_sha256,
        adapter_config_sha256=adapter_evidence.adapter_config_sha256,
        measured_peak=measured_peak,
    )


def execute_run(
    plan: RunPlan,
    runner: Runner,
    *,
    run_id: str | None = None,
    sink: RunEventSink | None = None,
    cancel: CancelToken | None = None,
    out_dir: str | Path | None = None,
    clock: Callable[[], str] = _now_iso,
) -> SupervisedRun:
    """Execute ``plan`` through ``runner``, collecting the ``RunEvent`` stream and returning the
    terminal :class:`RunManifest`. Terminal classification is total: :class:`RunCancelled` →
    ``cancelled``; :class:`RunnerFailure` → ``failed`` with its taxonomy; any other exception →
    ``failed`` / ``FAIL`` (the supervisor never leaks a runner crash). Every invocation mints a fresh
    UUIDv7 run identity unless a test/integration caller supplies one. With ``out_dir`` the manifest
    is written atomically to ``<out_dir>/runs/<run_id>/RunManifest.json``. Events are appended to the
    returned list and, if ``sink`` is given, forwarded to it live."""
    rid = _sanitize_id(run_id or new_uuid7_id("run"))
    cancel = cancel or CancelToken()
    events: list[RunEvent] = []
    sink_errors: list[str] = []
    record_dir = run_record_directory(out_dir, rid) if out_dir is not None else None

    def _collect(event: RunEvent) -> None:
        events.append(event)
        if sink is not None:
            try:
                sink(event)
            except Exception as exc:  # noqa: BLE001 - an observer cannot rewrite run truth.
                label = type(exc).__name__
                if label not in sink_errors:
                    sink_errors.append(label)

    ctx = RunContext(plan, rid, _collect, cancel, clock)
    started = clock()
    plan_ref = Ref(id=plan.plan_id, hash=HashRef(value=plan.plan_hash))

    state: RunState = "running"
    failure: FailureRecord | None = None
    artifact_ids: list[str] = []
    produced: Sequence[ProducedArtifact] = []
    artifact_manifests: list[ArtifactManifest] = []
    training_success_evidence: TrainingSuccessEvidence | None = None

    try:
        # Defense in depth: callers can reach this public library boundary without going through the
        # CLI or protocol worker. Never let a mutated plan reach a runner under its old seal.
        from corpus_studio.platform.planner import verify_run_plan_hash  # noqa: PLC0415

        if not verify_run_plan_hash(plan):
            raise RunnerFailure(
                "RunPlan hash verification failed; regenerate the plan before execution",
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.env_loaded,
                remediation="regenerate the RunPlan from immutable inputs; do not mutate it after sealing",
            )
        if plan.resolved_execution is not None:
            from corpus_studio.platform.execution_config import (  # noqa: PLC0415
                verify_execution_configuration_hash,
            )

            if not verify_execution_configuration_hash(plan.resolved_execution):
                raise RunnerFailure(
                    "resolved execution configuration hash verification failed",
                    taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                    stage=StageMarker.env_loaded,
                    remediation="regenerate the RunPlan; do not mutate resolved execution fields",
                )
        from corpus_studio.platform.execution_config import (  # noqa: PLC0415
            ExecutionConfigurationError,
            verify_runner_lane,
        )

        try:
            verify_runner_lane(plan, runner.name)
        except ExecutionConfigurationError as exc:
            raise RunnerFailure(
                str(exc),
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.env_loaded,
                remediation="dispatch the plan through its sealed runner lane",
            ) from exc
        if plan.resolved_execution is not None:
            from corpus_studio.platform.runners import TrainingRunner  # noqa: PLC0415

            if type(runner) is not TrainingRunner:
                raise RunnerFailure(
                    "resolved training plans require the first-party TrainingRunner adapter",
                    taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                    stage=StageMarker.env_loaded,
                )
        elif not isinstance(runner, EchoRunner):
            raise RunnerFailure(
                "echo plans require the built-in EchoRunner adapter",
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.env_loaded,
            )
        produced = runner.run(ctx)
        artifact_ids = [artifact.artifact_id for artifact in produced]
        try:
            artifact_manifests = [
                build_artifact_manifest(
                    artifact_id=artifact.artifact_id,
                    path=artifact.path,
                    kind=artifact.kind,
                    run_id=rid,
                    base_model=plan.base_model,
                    now=clock(),
                )
                for artifact in produced
            ]
        except Exception as exc:  # noqa: BLE001 - artifact admission is a classified gate.
            raise RunnerFailure(
                f"artifact manifest creation failed: {exc}",
                taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                stage=StageMarker.export,
            ) from exc
        if plan.resolved_execution is not None:
            training_success_evidence = validate_training_success_evidence(
                plan,
                rid,
                events,
                produced,
                artifact_manifests,
                ctx.training_execution_evidence,
                ctx.measured_peak,
            )
        if record_dir is not None:
            try:
                for artifact_manifest in artifact_manifests:
                    write_artifact_manifest(artifact_manifest, record_dir)
            except Exception as exc:  # noqa: BLE001 - durable artifact evidence is a success gate.
                artifact_ids = []
                raise RunnerFailure(
                    f"artifact manifest persistence failed: {exc}",
                    taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                    stage=StageMarker.export,
                    remediation="preserve the run directory and repair durable artifact storage",
                ) from exc
        if plan.resolved_execution is not None and ctx.measured_peak is not None:
            # Promotion follows every semantic, byte-integrity, and durable-artifact gate.
            from corpus_studio.platform.watchdog import (  # noqa: PLC0415
                reconcile_measured_fit,
            )

            ctx.final_fit = reconcile_measured_fit(ctx.measured_peak, proven=True)
        state = "succeeded"
    except RunCancelled:
        state = "cancelled"
    except RunnerFailure as exc:
        state = "failed"
        failure = FailureRecord(
            run_id=rid,
            taxonomy=exc.taxonomy,
            stage=exc.stage,
            exit_code=exc.exit_code,
            message=str(exc),
            exception_type="RunnerFailure",
            detected_at=clock(),
            remediation=exc.remediation,
        )
    except Exception as exc:  # noqa: BLE001 — the supervisor must classify, not propagate, a crash
        state = "failed"
        failure = FailureRecord(
            run_id=rid,
            taxonomy=FailureTaxonomy.FAIL,
            message=str(exc) or type(exc).__name__,
            exception_type=type(exc).__name__,
            detected_at=clock(),
        )

    if state != "succeeded" and ctx.measured_peak is not None:
        from corpus_studio.platform.watchdog import reconcile_measured_fit  # noqa: PLC0415

        ctx.final_fit = reconcile_measured_fit(ctx.measured_peak, proven=False)
    finished = clock()
    manifest = RunManifest(
        run_id=rid,
        plan_ref=plan_ref,
        environment_ref=plan.environment_ref,
        dataset_ref=plan.dataset_ref,
        created_at=started,
        updated_at=finished,
        started_at=started,
        finished_at=finished,
        state=state,
        base_model=plan.base_model,
        target=plan.backend_ref.id if plan.resolved_execution is not None else runner.name,
        output_dir=(
            next(
                (artifact.path for artifact in produced if artifact.kind == "adapter"),
                plan.export.output_dir,
            )
            if plan.resolved_execution is not None
            else plan.export.output_dir
        ),
        artifact_ids=artifact_ids,
        failure=failure,
        final_fit=ctx.final_fit,  # the MEASURED fit, when a runner captured one (via the watchdog)
        training_success_evidence=(
            training_success_evidence if state == "succeeded" else None
        ),
        notes=(
            "event sink failures were isolated: " + ", ".join(sink_errors)
            if sink_errors
            else ""
        ),
    )
    if record_dir is not None:
        try:
            write_run_manifest(manifest, record_dir)
        except Exception as exc:  # noqa: BLE001 - terminal truth follows durable admission.
            if manifest.state == "succeeded":
                from corpus_studio.platform.watchdog import (  # noqa: PLC0415
                    reconcile_measured_fit,
                )

                failed_fit = (
                    reconcile_measured_fit(ctx.measured_peak, proven=False)
                    if ctx.measured_peak is not None
                    else None
                )
                payload = manifest.model_dump(mode="json")
                payload.update(
                    {
                        "state": "failed",
                        "updated_at": clock(),
                        "finished_at": clock(),
                        "failure": FailureRecord(
                            run_id=rid,
                            taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                            stage=StageMarker.export,
                            message=f"run manifest persistence failed: {exc}",
                            exception_type=type(exc).__name__,
                            detected_at=clock(),
                            remediation=(
                                "preserve the run directory and repair durable run-record storage"
                            ),
                        ).model_dump(mode="json"),
                        "final_fit": (
                            failed_fit.model_dump(mode="json")
                            if failed_fit is not None
                            else None
                        ),
                        "training_success_evidence": None,
                    }
                )
                manifest = RunManifest.model_validate(payload)
                try:
                    write_run_manifest(manifest, record_dir)
                except Exception as retry_exc:  # noqa: BLE001 - no durable channel remains.
                    manifest = manifest.model_copy(
                        update={
                            "notes": (
                                manifest.notes
                                + ("; " if manifest.notes else "")
                                + "failed terminal manifest could not be persisted: "
                                + type(retry_exc).__name__
                            )
                        }
                    )
            else:
                manifest = manifest.model_copy(
                    update={
                        "notes": (
                            manifest.notes
                            + ("; " if manifest.notes else "")
                            + "terminal manifest could not be persisted: "
                            + type(exc).__name__
                        )
                    }
                )

    terminal_message = (
        manifest.failure.message
        if manifest.failure is not None and manifest.failure.message
        else manifest.state
    )
    ctx.emit_terminal(manifest.state, terminal_message)
    if sink_errors:
        note = "event sink failures were isolated: " + ", ".join(sink_errors)
        if note not in manifest.notes:
            manifest = manifest.model_copy(
                update={"notes": manifest.notes + ("; " if manifest.notes else "") + note}
            )
            if record_dir is not None:
                try:
                    write_run_manifest(manifest, record_dir)
                except Exception:  # noqa: BLE001 - observer notes are not execution truth.
                    pass
    return SupervisedRun(manifest=manifest, events=events, artifacts=artifact_manifests)


def write_run_manifest(manifest: RunManifest, out_dir: str | Path) -> Path:
    """Crash-safe write: serialize to a temp file then ``os.replace`` onto the final path (atomic on
    the same filesystem) — the ``run_registry`` durability convention, so a torn write can never
    leave a half-written manifest."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "RunManifest.json"
    tmp = directory / f".RunManifest.{os.getpid()}.tmp"
    tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, final)
    return final


def demo_run_plan(plan_id: str = "demo-echo") -> RunPlan:
    """A minimal, valid :class:`RunPlan` for exercising the supervisor with the :class:`EchoRunner` —
    no dataset, backend, or GPU required. This is NOT a real training plan (the planner that resolves
    those is a later slice); it exists so ``platform-run --demo`` and the tests can prove the harness
    end-to-end. Attention defaults to ``math`` (the Blackwell-safe path). Built via
    ``model_validate`` so the plan body reads as the language-neutral JSON a real planner emits."""
    from corpus_studio.platform.backends import (  # noqa: PLC0415
        backend_manifest_ref,
        get_worker_backend,
    )

    echo_backend = get_worker_backend("echo")
    assert echo_backend is not None  # built-in protocol fixture
    draft = RunPlan.model_validate(
        {
            "plan_id": plan_id,
            "plan_hash": "0" * 64,
            "backend_ref": backend_manifest_ref(echo_backend).model_dump(mode="json"),
            "environment_ref": {"id": "a" * 64},
            "dataset_ref": {"id": "none"},
            "task_type": "evaluation",
            "base_model": "none",
            "precision": "bf16",
            "quantization": "none",
            "adapter": {"method": "none"},
            "optimizer": {"impl": "adamw_torch", "learning_rate": 2e-4},
            "loss_impl": "cross_entropy",
            "attention_backend": "math",
            "sequence": {"max_sequence_len": 512},
            "batching": {"micro_batch_size": 1, "supervised_token_accumulation_target": 1024},
            "checkpoint_policy": {"impl": "adapter_only"},
            "export": {"format": "adapter_peft"},
        }
    )
    from corpus_studio.platform.planner import compute_plan_hash, run_plan_hash_payload  # noqa: PLC0415

    return draft.model_copy(update={"plan_hash": compute_plan_hash(run_plan_hash_payload(draft))})
