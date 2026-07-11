"""Platform slice 3 — the headless run supervisor. Pure tests (no torch): the EchoRunner + the
terminal-state classification + the crash-safe RunManifest write are all provable on a core-only
install. Mirrors the round-trip idiom of test_platform_contracts.py."""

import pytest
from pydantic import ValidationError

import corpus_studio.platform as P
from corpus_studio.platform.enums import FailureTaxonomy, StageMarker
from corpus_studio.platform.supervisor import (
    CancelToken,
    EchoRunner,
    ProducedArtifact,
    RunContext,
    RunnerFailure,
    demo_run_plan,
    execute_run,
    write_run_manifest,
)

# A fixed clock so timestamps are deterministic in assertions.
_CLOCK = lambda: "2026-07-11T00:00:00+00:00"  # noqa: E731


# ---- demo plan + happy path --------------------------------------------------


def test_demo_run_plan_is_a_valid_roundtrippable_runplan():
    plan = demo_run_plan()
    assert P.RunPlan.model_validate_json(plan.model_dump_json()) == plan


def test_echo_run_succeeds_and_emits_ordered_events():
    result = execute_run(demo_run_plan(), EchoRunner(steps=3), clock=_CLOCK)

    assert result.manifest.state == "succeeded"
    assert result.manifest.failure is None
    assert result.manifest.target == "echo"

    kinds = [(e.event_type, e.stage) for e in result.events]
    assert kinds[0] == ("stage", StageMarker.process_start)
    assert kinds[-1] == ("terminal", None)
    assert [e.event_type for e in result.events].count("metric") == 3
    # process_start + 3 metrics + export stage + terminal
    assert len(result.events) == 6


def test_event_seq_is_monotonic_from_zero():
    result = execute_run(demo_run_plan(), EchoRunner(steps=4), clock=_CLOCK)
    assert [e.seq for e in result.events] == list(range(len(result.events)))


def test_every_run_event_roundtrips():
    result = execute_run(demo_run_plan(), EchoRunner(steps=2), clock=_CLOCK)
    for event in result.events:
        assert P.RunEvent.model_validate_json(event.model_dump_json()) == event


def test_metric_events_carry_step_and_loss():
    result = execute_run(demo_run_plan(), EchoRunner(steps=2), clock=_CLOCK)
    metrics = [e for e in result.events if e.event_type == "metric"]
    assert [m.optimizer_step for m in metrics] == [1, 2]
    assert metrics[0].metrics is not None
    assert metrics[0].metrics.loss == pytest.approx(1.0)


# ---- terminal classification -------------------------------------------------


def test_cancellation_yields_cancelled_state_and_no_failure():
    token = CancelToken()
    token.cancel()  # pre-cancelled: the EchoRunner stops before the first metric
    result = execute_run(demo_run_plan(), EchoRunner(steps=3), cancel=token, clock=_CLOCK)

    assert result.manifest.state == "cancelled"
    assert result.manifest.failure is None
    assert result.events[-1].event_type == "terminal"
    assert result.events[-1].payload == {"state": "cancelled"}
    # No metric emitted — only process_start + terminal.
    assert [e.event_type for e in result.events] == ["stage", "terminal"]


def test_runner_failure_is_classified_with_its_taxonomy():
    class _FailRunner:
        name = "boom"

        def run(self, ctx: RunContext):
            ctx.emit_stage(StageMarker.process_start)
            raise RunnerFailure(
                "dependency missing",
                taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                stage=StageMarker.env_loaded,
                remediation="pip install '.[train]'",
            )

    result = execute_run(demo_run_plan(), _FailRunner(), clock=_CLOCK)

    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.ENVIRONMENT_FAILURE
    assert result.manifest.failure.stage == StageMarker.env_loaded
    assert result.manifest.failure.remediation == "pip install '.[train]'"
    assert result.manifest.failure.exception_type == "RunnerFailure"


def test_unexpected_exception_becomes_failed_fail():
    class _CrashRunner:
        name = "crash"

        def run(self, ctx: RunContext):
            raise ValueError("kaboom")

    result = execute_run(demo_run_plan(), _CrashRunner(), clock=_CLOCK)

    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.FAIL
    assert result.manifest.failure.exception_type == "ValueError"
    assert result.manifest.failure.message == "kaboom"


def test_produced_artifacts_are_recorded_on_the_manifest():
    class _ArtifactRunner:
        name = "artifact"

        def run(self, ctx: RunContext):
            art = ProducedArtifact(artifact_id="run-1-adapter", kind="adapter", path="out/adapter")
            ctx.emit_artifact(art)
            return [art]

    result = execute_run(demo_run_plan(), _ArtifactRunner(), clock=_CLOCK)
    assert result.manifest.artifact_ids == ["run-1-adapter"]
    assert any(e.event_type == "artifact_produced" for e in result.events)


# ---- manifest persistence ----------------------------------------------------


def test_manifest_roundtrips():
    result = execute_run(demo_run_plan(), EchoRunner(), clock=_CLOCK)
    assert P.RunManifest.model_validate_json(result.manifest.model_dump_json()) == result.manifest


def test_manifest_written_atomically_to_out_dir(tmp_path):
    result = execute_run(demo_run_plan(), EchoRunner(), out_dir=tmp_path, clock=_CLOCK)
    written = tmp_path / "RunManifest.json"
    assert written.is_file()
    reloaded = P.RunManifest.model_validate_json(written.read_text(encoding="utf-8"))
    assert reloaded == result.manifest
    assert reloaded.output_dir == str(tmp_path)
    # No temp file left behind.
    assert list(tmp_path.glob(".RunManifest.*.tmp")) == []


def test_write_run_manifest_creates_missing_dirs(tmp_path):
    result = execute_run(demo_run_plan(), EchoRunner(), clock=_CLOCK)
    nested = tmp_path / "a" / "b"
    path = write_run_manifest(result.manifest, nested)
    assert path == nested / "RunManifest.json"
    assert path.is_file()


# ---- sink + run_id -----------------------------------------------------------


def test_sink_receives_the_live_event_stream():
    seen: list = []
    result = execute_run(demo_run_plan(), EchoRunner(steps=2), sink=seen.append, clock=_CLOCK)
    assert seen == result.events


def test_run_id_is_sanitized_to_the_id_pattern():
    result = execute_run(demo_run_plan(), EchoRunner(), run_id="My Run!! #1", clock=_CLOCK)
    assert result.manifest.run_id == "My-Run-1"


def test_run_id_all_punctuation_falls_back_to_run():
    result = execute_run(demo_run_plan(), EchoRunner(), run_id="###", clock=_CLOCK)
    assert result.manifest.run_id == "run"


def test_log_and_warning_events_are_emitted():
    class _ChattyRunner:
        name = "chatty"

        def run(self, ctx: RunContext):
            ctx.emit_log("loading")
            ctx.emit_warning("running on CPU")
            return []

    result = execute_run(demo_run_plan(), _ChattyRunner(), clock=_CLOCK)
    types = [e.event_type for e in result.events]
    assert "log" in types
    assert "warning" in types
    warning = next(e for e in result.events if e.event_type == "warning")
    assert warning.message == "running on CPU"


def test_default_clock_stamps_a_real_timestamp():
    # No clock override → exercises the real _now_iso default.
    result = execute_run(demo_run_plan(), EchoRunner(steps=1))
    assert result.manifest.state == "succeeded"
    assert result.events[0].emitted_at.endswith("+00:00")


def test_echo_runner_rejects_non_positive_steps():
    with pytest.raises(ValueError):
        EchoRunner(steps=0)


def test_bad_manifest_field_is_rejected_by_the_contract():
    # Sanity that RunManifest keeps extra="forbid" (a stale producer is caught, not silently dropped).
    with pytest.raises(ValidationError):
        P.RunManifest(
            run_id="r1",
            plan_ref=P.Ref(id="p1"),
            created_at="t",
            updated_at="t",
            bogus=1,
        )
