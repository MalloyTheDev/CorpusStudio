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
from corpus_studio.platform.common import HashRef, Ref
from corpus_studio.platform.contracts import (
    ArtifactManifest,
    EventMetrics,
    FailureRecord,
    FitClassification,
    RunEvent,
    RunManifest,
    RunPlan,
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
    ``failed`` / ``FAIL`` (the supervisor never leaks a runner crash). With ``out_dir`` the manifest
    is written atomically to ``<out_dir>/RunManifest.json``. Events are appended to the returned
    list and, if ``sink`` is given, forwarded to it live."""
    rid = _sanitize_id(run_id or plan.plan_id)
    cancel = cancel or CancelToken()
    events: list[RunEvent] = []

    def _collect(event: RunEvent) -> None:
        events.append(event)
        if sink is not None:
            sink(event)

    ctx = RunContext(plan, rid, _collect, cancel, clock)
    started = clock()
    plan_ref = Ref(id=plan.plan_id, hash=HashRef(value=plan.plan_hash))

    state: RunState = "running"
    failure: FailureRecord | None = None
    artifact_ids: list[str] = []
    produced: Sequence[ProducedArtifact] = []

    try:
        produced = runner.run(ctx)
        artifact_ids = [artifact.artifact_id for artifact in produced]
        state = "succeeded"
        ctx.emit_terminal("succeeded")
    except RunCancelled:
        state = "cancelled"
        ctx.emit_terminal("cancelled", "run cancelled")
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
        ctx.emit_terminal("failed", str(exc))
    except Exception as exc:  # noqa: BLE001 — the supervisor must classify, not propagate, a crash
        state = "failed"
        failure = FailureRecord(
            run_id=rid,
            taxonomy=FailureTaxonomy.FAIL,
            message=str(exc) or type(exc).__name__,
            exception_type=type(exc).__name__,
            detected_at=clock(),
        )
        ctx.emit_terminal("failed", str(exc))

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
        target=runner.name,
        output_dir=str(out_dir) if out_dir is not None else plan.export.output_dir,
        artifact_ids=artifact_ids,
        failure=failure,
        final_fit=ctx.final_fit,  # the MEASURED fit, when a runner captured one (via the watchdog)
    )
    artifact_manifests = [
        build_artifact_manifest(
            artifact_id=artifact.artifact_id,
            path=artifact.path,
            kind=artifact.kind,
            run_id=rid,
            base_model=plan.base_model,
            now=finished,
        )
        for artifact in produced
    ]
    if out_dir is not None:
        write_run_manifest(manifest, out_dir)
        for artifact_manifest in artifact_manifests:
            write_artifact_manifest(artifact_manifest, out_dir)
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
    return RunPlan.model_validate(
        {
            "plan_id": plan_id,
            "plan_hash": "0" * 64,
            "backend_ref": {"id": "echo"},
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
