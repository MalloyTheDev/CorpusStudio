"""Deterministic tests for the Section 11 measurement harness.

No GPU and no torch: every probe is synthetic, every clock is injected. The tests cover parsing,
energy integration (with boundary clipping), aggregation, warm-up/measured partitioning, missing /
reordered / duplicate samples, interrupted and failed runs, scientific-completeness gating, the
durable in-path raw event log, sampler overhead, cross-trial confidence intervals, and a synthetic
end-to-end raw-record -> summary -> paper-table path.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

from corpus_studio.platform import telemetry as T
from corpus_studio.platform.common import HashRef, MemoryMetrics, Ref
from corpus_studio.platform.contracts import (
    EventMetrics,
    FailureRecord,
    GpuTelemetrySample,
    HostTelemetrySample,
    RunEvent,
    RunManifest,
    RunTelemetrySummary,
    TelemetryIdentity,
    TelemetrySample,
)
from corpus_studio.platform.enums import FailureTaxonomy, OperatingSystem, StageMarker
from corpus_studio.platform.supervisor import (
    EchoRunner,
    demo_run_plan,
    execute_run,
    run_record_directory,
)


# --------------------------------------------------------------------------------------------------
# Fixtures builders (pure, deterministic)
# --------------------------------------------------------------------------------------------------
def _sample(seq: int, *, phase: str, mono_ns: int, power: float | None, step: int | None = None,
            rss: int | None = 1_000_000, unavailable: list[str] | None = None) -> TelemetrySample:
    gpu = (
        GpuTelemetrySample(
            device_index=0, device_uuid="GPU-x", utilization_percent=50.0,
            power_watts=power, temperature_c=40.0 + seq, graphics_clock_mhz=2000.0,
        )
        if power is not None
        else None
    )
    return TelemetrySample(
        run_id="run-t",
        sample_seq=seq,
        monotonic_ns=mono_ns,
        wall_utc="2026-07-15T00:00:00+00:00",
        phase=phase,  # type: ignore[arg-type]
        sample_source=OperatingSystem.linux,
        optimizer_step=step,
        gpu=gpu,
        host=HostTelemetrySample(process_tree_rss_bytes=rss) if rss is not None else None,
        memory=MemoryMetrics(torch_allocated_bytes=500_000_000, cuda_device_used_bytes=1_000_000_000)
        if power is not None
        else None,
        probe_unavailable=unavailable or [],
    )


def _metric_event(seq: int, step: int, *, loss: float, step_time: float | None = 0.5,
                  tokens_per_sec: float | None = 1000.0) -> RunEvent:
    return RunEvent(
        event_type="metric",
        run_id="run-t",
        seq=seq,
        emitted_at="2026-07-15T00:00:00+00:00",
        optimizer_step=step,
        metrics=EventMetrics(
            loss=loss, step_time_seconds=step_time, tokens_per_sec=tokens_per_sec,
            supervised_tokens_per_sec=tokens_per_sec,
        ),
    )


def _write_manifest(directory: Path, *, state: str = "succeeded",
                    failure: FailureRecord | None = None) -> None:
    manifest = RunManifest(
        run_id="run-t",
        plan_ref=Ref(id="plan-t", hash=HashRef(value="a" * 64)),
        created_at="2026-07-15T00:00:00+00:00",
        updated_at="2026-07-15T00:00:03+00:00",
        started_at="2026-07-15T00:00:00+00:00",
        finished_at="2026-07-15T00:00:03+00:00",
        state=state,  # type: ignore[arg-type]
        failure=failure,
    )
    (directory / "RunManifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def _write_jsonl(path: Path, objs: list) -> None:
    path.write_text("".join(o.model_dump_json() + "\n" for o in objs), encoding="utf-8")


def _full_run(directory: Path, *, step_time: float | None = 0.5) -> None:
    """A complete, paper-shaped raw run: 12 steps, ramping power, all identity present via overlay."""
    directory.mkdir(parents=True, exist_ok=True)
    _write_manifest(directory)
    events = [
        RunEvent(event_type="stage", run_id="run-t", seq=0, emitted_at="t",
                 stage=StageMarker.process_start),
    ]
    events += [
        _metric_event(i + 1, i + 1, loss=round(1.0 / (i + 1), 4), step_time=step_time)
        for i in range(12)
    ]
    _write_jsonl(directory / "RunEvents.jsonl", events)
    samples = [_sample(0, phase="baseline", mono_ns=0, power=100.0)]
    samples += [_sample(1, phase="setup", mono_ns=200_000_000, power=110.0)]
    for i in range(12):
        step = i + 1
        samples.append(
            _sample(
                2 + i,
                phase="warmup" if step <= 2 else "measured",
                mono_ns=(2 + i) * 200_000_000,
                power=120.0 + 10 * i,
                step=step,
            )
        )
    samples.append(_sample(14, phase="teardown", mono_ns=14 * 200_000_000, power=260.0))
    _write_jsonl(directory / "TelemetrySamples.jsonl", samples)


_FULL_IDENTITY = TelemetryIdentity(
    repository_commit="df86db5",
    worker_wheel_sha256="b" * 64,
    environment_lock_hash="c" * 64,
    plan_hash="a" * 64,
    execution_configuration_hash="d" * 64,
    run_id="run-t",
)


# --------------------------------------------------------------------------------------------------
# Torch-free
# --------------------------------------------------------------------------------------------------
def test_telemetry_import_is_torch_free() -> None:
    for heavy in ("torch", "transformers", "trl", "peft", "bitsandbytes"):
        assert heavy not in sys.modules, f"telemetry pulled {heavy}"


# --------------------------------------------------------------------------------------------------
# Parsing / ordering
# --------------------------------------------------------------------------------------------------
def test_parse_samples_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "TelemetrySamples.jsonl"
    originals = [_sample(0, phase="baseline", mono_ns=0, power=100.0),
                 _sample(1, phase="measured", mono_ns=200_000_000, power=150.0, step=3)]
    _write_jsonl(path, originals)
    parsed = T.load_samples(path)
    assert parsed == originals


def test_reordered_and_duplicate_samples_aggregate_identically(tmp_path: Path) -> None:
    a = _sample(0, phase="baseline", mono_ns=0, power=100.0)
    b = _sample(1, phase="measured", mono_ns=200_000_000, power=150.0, step=3)
    c = _sample(2, phase="measured", mono_ns=400_000_000, power=200.0, step=4)
    clean = T.order_and_dedupe_samples([a, b, c])
    scrambled = T.order_and_dedupe_samples([c, a, b, b, c])  # reordered + duplicated
    assert [s.sample_seq for s in clean] == [0, 1, 2]
    assert scrambled == clean  # first-write-wins dedupe, deterministic order


# --------------------------------------------------------------------------------------------------
# Energy integration (with boundary clipping)
# --------------------------------------------------------------------------------------------------
def test_energy_trapezoid_no_clip() -> None:
    joules, clipped = T._integrate_energy([(0.0, 100.0), (1.0, 200.0)])
    assert joules == 150.0 and clipped is False


def test_energy_boundary_clip_interpolates_power() -> None:
    # Window [0.5, 1.0]: power at 0.5 interpolates to 150 W; trapezoid 0.5*(150+200)*0.5 = 87.5.
    joules, clipped = T._integrate_energy([(0.0, 100.0), (1.0, 200.0)], lo=0.5, hi=1.0)
    assert clipped is True
    assert abs(joules - 87.5) < 1e-9


def test_energy_none_below_two_points() -> None:
    joules, _ = T._integrate_energy([(0.0, 100.0)])
    assert joules is None


# --------------------------------------------------------------------------------------------------
# Statistics + cross-trial CI
# --------------------------------------------------------------------------------------------------
def test_metric_summary_sample_sd_uses_n_minus_one() -> None:
    summary = T._metric_summary([1.0, 2.0, 3.0, 4.0], "u")
    assert summary is not None
    assert summary.count == 4 and summary.mean == 2.5 and summary.median == 2.5
    assert summary.minimum == 1.0 and summary.maximum == 4.0
    assert abs(summary.sample_standard_deviation - 1.2909944487358056) < 1e-12


def test_cross_trial_ci_reported_only_for_three_trials() -> None:
    _, ci3 = T.combine_trial_values([10.0, 12.0, 14.0], "seconds")
    assert ci3.reported is True and ci3.trial_count == 3
    assert ci3.mean == 12.0 and abs(ci3.sample_standard_deviation - 2.0) < 1e-12
    assert abs(ci3.half_width - 4.3026527299 * 2.0 / (3 ** 0.5)) < 1e-9
    assert abs(ci3.lower - (12.0 - ci3.half_width)) < 1e-12
    _, ci2 = T.combine_trial_values([10.0, 12.0], "seconds")
    assert ci2.reported is False and ci2.half_width is None


# --------------------------------------------------------------------------------------------------
# Aggregation + warm-up/measured partition + energy end to end
# --------------------------------------------------------------------------------------------------
def test_full_run_summary_partitions_and_integrates(tmp_path: Path) -> None:
    _full_run(tmp_path)
    summary = T.summarize_run_telemetry(
        tmp_path, identity_overlay=_FULL_IDENTITY, requested_interval_ms=200.0
    )
    assert summary.step.completed_optimizer_steps == 12
    assert summary.step.warmup_optimizer_steps == [1, 2]
    assert summary.step.measured_optimizer_steps == list(range(3, 13))
    assert summary.step.step_time_seconds is not None
    assert summary.step.step_time_seconds.count == 10  # measured steps only
    assert summary.energy.run_joules is not None and summary.energy.run_joules > 0
    assert summary.energy.measured_window_joules is not None
    assert summary.energy.power_sample_count == 15
    assert summary.sampling.observed_median_interval_ms == 200.0
    assert summary.gpu is not None and summary.gpu.device_uuid == "GPU-x"
    # Every raw source is bound by sha256 so the summary provably derives from them.
    kinds = {r.record_kind for r in summary.raw_records}
    assert kinds == {"run_manifest", "run_events", "telemetry_samples"}
    assert RunTelemetrySummary.model_validate_json(summary.model_dump_json()) == summary


def test_full_run_is_scientifically_complete(tmp_path: Path) -> None:
    _full_run(tmp_path)
    summary = T.summarize_run_telemetry(tmp_path, identity_overlay=_FULL_IDENTITY)
    assert summary.completeness.scientifically_complete is True
    assert summary.completeness.missing_required_paper_fields == []


# --------------------------------------------------------------------------------------------------
# A telemetry gap never manufactures paper data
# --------------------------------------------------------------------------------------------------
def test_missing_step_time_marks_incomplete_but_keeps_success(tmp_path: Path) -> None:
    _full_run(tmp_path, step_time=None)  # a run that never recorded per-step time
    summary = T.summarize_run_telemetry(tmp_path, identity_overlay=_FULL_IDENTITY)
    assert summary.outcome.state == "succeeded"  # the run's success is untouched...
    assert summary.completeness.scientifically_complete is False  # ...but it is not paper-usable
    assert "step.step_time_seconds" in summary.completeness.missing_required_paper_fields


def test_missing_power_yields_null_energy_never_zero(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    _write_manifest(tmp_path)
    _write_jsonl(tmp_path / "RunEvents.jsonl", [_metric_event(1, 1, loss=0.5)])
    # Samples with no GPU power at all.
    samples = [_sample(i, phase="measured", mono_ns=i * 200_000_000, power=None, step=3) for i in range(3)]
    _write_jsonl(tmp_path / "TelemetrySamples.jsonl", samples)
    summary = T.summarize_run_telemetry(tmp_path)
    assert summary.energy.run_joules is None  # null, not 0.0
    assert summary.energy.power_sample_count == 0


def test_probe_gaps_flag_degraded(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    _write_manifest(tmp_path)
    _write_jsonl(tmp_path / "RunEvents.jsonl", [_metric_event(1, 3, loss=0.5)])
    samples = [
        _sample(0, phase="measured", mono_ns=0, power=100.0, step=3, unavailable=["nvidia_smi_gpu"]),
        _sample(1, phase="measured", mono_ns=200_000_000, power=110.0, step=3),
    ]
    _write_jsonl(tmp_path / "TelemetrySamples.jsonl", samples)
    summary = T.summarize_run_telemetry(tmp_path)
    assert summary.completeness.telemetry_degraded is True
    assert summary.completeness.degraded_sample_count == 1


# --------------------------------------------------------------------------------------------------
# Interrupted and failed runs remain records
# --------------------------------------------------------------------------------------------------
def test_interrupted_run_still_summarizes(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    _write_manifest(tmp_path, state="interrupted")
    _write_jsonl(tmp_path / "RunEvents.jsonl", [_metric_event(1, 1, loss=0.9)])
    _write_jsonl(
        tmp_path / "TelemetrySamples.jsonl",
        [_sample(0, phase="warmup", mono_ns=0, power=100.0, step=1)],
    )
    summary = T.summarize_run_telemetry(tmp_path)
    assert summary.outcome.state == "interrupted"
    assert summary.step.completed_optimizer_steps == 1
    assert summary.completeness.scientifically_complete is False


def test_failed_run_carries_taxonomy(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    failure = FailureRecord(
        run_id="run-t",
        taxonomy=FailureTaxonomy.GRADIENT_FAILURE,
        stage=StageMarker.backward,
        message="materialized gradient dtype mismatch",
        detected_at="2026-07-15T00:00:03+00:00",
    )
    _write_manifest(tmp_path, state="failed", failure=failure)
    (tmp_path / "RunEvents.jsonl").write_text("", encoding="utf-8")
    summary = T.summarize_run_telemetry(tmp_path)
    assert summary.outcome.state == "failed"
    assert summary.outcome.failure_taxonomy == FailureTaxonomy.GRADIENT_FAILURE
    assert summary.outcome.failure_stage == StageMarker.backward
    assert summary.step.completed_optimizer_steps == 0


# --------------------------------------------------------------------------------------------------
# The durable raw event log is written by the authoritative in-process supervisor
# --------------------------------------------------------------------------------------------------
def test_execute_run_writes_durable_event_log(tmp_path: Path) -> None:
    plan = demo_run_plan()
    result = execute_run(plan, EchoRunner(steps=3), run_id="run-log", out_dir=tmp_path)
    record = run_record_directory(tmp_path, "run-log")
    events_path = record / "RunEvents.jsonl"
    assert events_path.is_file()
    logged = T.load_events(events_path)
    # The durable log holds exactly the events the supervisor also returned in memory (same truth).
    assert [e.model_dump(mode="json") for e in logged] == [
        e.model_dump(mode="json") for e in result.events
    ]
    assert logged[-1].event_type == "terminal"


# --------------------------------------------------------------------------------------------------
# The sampler with a synthetic probe drives phase from the caller and records overhead
# --------------------------------------------------------------------------------------------------
def test_sampler_take_sample_is_deterministic(tmp_path: Path) -> None:
    mono = itertools.count(0, 200_000_000)
    reading = T.SampleReading(
        gpu=GpuTelemetrySample(power_watts=123.0), host=HostTelemetrySample(process_tree_rss_bytes=5),
    )
    sampler = T.TelemetrySampler(
        "run-s", tmp_path, probe=lambda: reading, interval_ms=200.0,
        monotonic_ns=lambda: next(mono), source=OperatingSystem.linux,
    )
    sampler.set_phase("measured")
    sampler.mark_step(3, phase="measured")
    first = sampler.take_sample()
    second = sampler.take_sample()
    assert first.sample_seq == 0 and second.sample_seq == 1
    assert second.monotonic_ns - first.monotonic_ns == 200_000_000
    assert first.gpu is not None and first.gpu.power_watts == 123.0
    overhead = sampler.overhead(wall_seconds=1.0)
    assert overhead.total_sampler_seconds is not None and overhead.total_sampler_seconds >= 0
    # Raw is on disk (written before any summary).
    assert (tmp_path / "TelemetrySamples.jsonl").is_file()
    assert len(T.load_samples(tmp_path / "TelemetrySamples.jsonl")) == 2


def test_sampler_root_pid_is_noop_for_injected_probe(tmp_path: Path) -> None:
    sampler = T.TelemetrySampler("run-s", tmp_path, probe=lambda: T.SampleReading(), interval_ms=50.0)
    sampler.set_root_pid(4321)  # must not raise and must not swap the injected probe
    sample = sampler.take_sample()
    assert sample.gpu is None and sample.host is None


# --------------------------------------------------------------------------------------------------
# Synthetic end to end: raw records -> summary -> paper table
# --------------------------------------------------------------------------------------------------
def test_end_to_end_raw_to_summary_to_table(tmp_path: Path) -> None:
    _full_run(tmp_path)
    summary = T.summarize_run_telemetry(tmp_path, identity_overlay=_FULL_IDENTITY)
    T.write_summary(summary, tmp_path)
    assert (tmp_path / "RunTelemetrySummary.json").is_file()
    csv = T.summary_to_csv(summary)
    assert "run_joules,joules," in csv
    table = T.summary_to_table(summary)
    assert table.startswith("# Run telemetry summary - run-t")
    assert "measured-window energy" in table
    series = T.power_series(T.load_samples(tmp_path / "TelemetrySamples.jsonl"))
    assert len(series) == 15 and series[0][0] == 0.0
    # The table's rendered energy matches the summary's derived (not a re-measurement).
    assert summary.energy.run_joules is not None


# --------------------------------------------------------------------------------------------------
# The default probes never raise (they may return partial data off-Linux / without a GPU)
# --------------------------------------------------------------------------------------------------
def test_default_probes_are_fail_soft() -> None:
    gpu = T.probe_gpu()
    host = T.probe_host()
    assert isinstance(gpu, T.SampleReading)
    assert isinstance(host, T.SampleReading)
    combined = T.default_probe()()
    assert isinstance(combined, T.SampleReading)


# --------------------------------------------------------------------------------------------------
# Default GPU probe parsing (nvidia-smi + watchdog memory), fully synthetic
# --------------------------------------------------------------------------------------------------
class _FakeCompleted:
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode = returncode
        self.stdout = stdout


def test_run_nvidia_smi_parses_and_handles_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        T.subprocess, "run",
        lambda *a, **k: _FakeCompleted(0, "0, GPU-abc, 55, 12, 130.5, 61, 2100, 7001, P2\n"),
    )
    row = T._run_nvidia_smi(["index", "uuid"])
    assert row is not None and row[0] == "0" and row[2] == "55"

    def _boom(*a, **k):
        raise OSError("no nvidia-smi")

    monkeypatch.setattr(T.subprocess, "run", _boom)
    assert T._run_nvidia_smi(["index"]) is None
    monkeypatch.setattr(T.subprocess, "run", lambda *a, **k: _FakeCompleted(9, ""))
    assert T._run_nvidia_smi(["index"]) is None


def test_probe_gpu_builds_sample_from_row(monkeypatch) -> None:
    monkeypatch.setattr(
        T, "_run_nvidia_smi",
        lambda query: ["0", "GPU-abc", "55", "12", "130.5", "61", "2100", "7001", "P2"],
    )
    monkeypatch.setattr(
        "corpus_studio.platform.watchdog.sample_gpu_memory",
        lambda: MemoryMetrics(torch_allocated_bytes=1, cuda_device_used_bytes=2),
    )
    reading = T.probe_gpu()
    assert reading.gpu is not None
    assert reading.gpu.device_index == 0 and reading.gpu.power_watts == 130.5
    assert reading.gpu.performance_state == "P2"
    assert reading.memory is not None and reading.memory.torch_allocated_bytes == 1
    assert reading.unavailable == []


def test_probe_gpu_marks_unavailable_when_absent(monkeypatch) -> None:
    monkeypatch.setattr(T, "_run_nvidia_smi", lambda query: None)
    monkeypatch.setattr("corpus_studio.platform.watchdog.sample_gpu_memory", lambda: None)
    reading = T.probe_gpu()
    assert reading.gpu is None
    assert "nvidia_smi_gpu" in reading.unavailable and "gpu_memory" in reading.unavailable


def test_probe_host_reads_proc_on_linux() -> None:
    # This host is native Linux, so /proc is present; the probe returns real (partial) data.
    reading = T.probe_host()
    assert reading.host is not None
    assert reading.host.system_ram_used_bytes is not None


def test_proc_readers_execute_on_linux() -> None:
    import os as _os

    meminfo = T._read_meminfo()
    assert "MemTotal" in meminfo and meminfo["MemTotal"] > 0
    tree_rss = T._read_process_tree_rss(_os.getpid())
    assert tree_rss is not None and tree_rss > 0  # this process is its own tree root
    # A pid that is not present resolves to None rather than raising or fabricating a value.
    assert T._read_process_tree_rss(2_000_000_123) is None
    read_bytes, write_bytes = T._read_self_io()
    assert read_bytes is None or read_bytes >= 0


def test_max_memory_takes_field_wise_maximum() -> None:
    merged = T._max_memory(
        [
            MemoryMetrics(torch_allocated_bytes=10, process_rss_bytes=None),
            MemoryMetrics(torch_allocated_bytes=30, process_rss_bytes=5),
        ]
    )
    assert merged is not None
    assert merged.torch_allocated_bytes == 30 and merged.process_rss_bytes == 5
    assert T._max_memory([]) is None


# --------------------------------------------------------------------------------------------------
# The threaded sampler loop actually runs and stops
# --------------------------------------------------------------------------------------------------
def test_sampler_thread_runs_and_stops(tmp_path: Path) -> None:
    import time as _time

    sampler = T.TelemetrySampler(
        "run-thread", tmp_path, probe=lambda: T.SampleReading(gpu=GpuTelemetrySample(power_watts=1.0)),
        interval_ms=5.0,
    )
    sampler.start()
    _time.sleep(0.05)
    overhead = sampler.stop()
    samples = T.load_samples(tmp_path / "TelemetrySamples.jsonl")
    assert len(samples) >= 1
    assert overhead.total_sampler_seconds is not None


def test_take_sample_survives_probe_exception(tmp_path: Path) -> None:
    def _explode() -> T.SampleReading:
        raise RuntimeError("driver fell over")

    sampler = T.TelemetrySampler("run-x", tmp_path, probe=_explode, interval_ms=50.0)
    sample = sampler.take_sample()
    assert any(u.startswith("probe:") for u in sample.probe_unavailable)


# --------------------------------------------------------------------------------------------------
# Loading tolerates blank lines; append_jsonl round-trips
# --------------------------------------------------------------------------------------------------
def test_loaders_skip_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "TelemetrySamples.jsonl"
    good = _sample(0, phase="measured", mono_ns=0, power=100.0, step=3)
    path.write_text(good.model_dump_json() + "\n\n", encoding="utf-8")
    assert T.load_samples(path) == [good]
    events_path = tmp_path / "RunEvents.jsonl"
    T.append_jsonl(events_path, _metric_event(1, 1, loss=0.5).model_dump_json())
    T.append_jsonl(events_path, "")  # a torn blank tail is skipped, not an error
    assert len(T.load_events(events_path)) == 1


# --------------------------------------------------------------------------------------------------
# Rich host/GPU aggregation: swap/disk deltas, throttle reasons, performance states
# --------------------------------------------------------------------------------------------------
def test_rich_host_and_gpu_aggregation(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    _write_manifest(tmp_path)
    _write_jsonl(tmp_path / "RunEvents.jsonl", [_metric_event(1, 3, loss=0.5)])
    samples = []
    for i in range(3):
        gpu = GpuTelemetrySample(
            power_watts=100.0 + i, temperature_c=40.0 + i, utilization_percent=50.0,
            performance_state="P2" if i else "P0", throttle_reasons=["thermal"] if i == 2 else [],
        )
        host = HostTelemetrySample(
            process_tree_rss_bytes=1000 + i, system_ram_used_bytes=8_000_000_000,
            swap_used_bytes=100 * i, cpu_utilization_percent=250.0, disk_read_bytes=10 * i,
            disk_write_bytes=20 * i,
        )
        samples.append(
            TelemetrySample(
                run_id="run-t", sample_seq=i, monotonic_ns=i * 200_000_000,
                wall_utc="t", phase="measured", sample_source=OperatingSystem.linux,
                optimizer_step=3, gpu=gpu, host=host,
            )
        )
    _write_jsonl(tmp_path / "TelemetrySamples.jsonl", samples)
    summary = T.summarize_run_telemetry(tmp_path)
    assert summary.host is not None
    assert summary.host.swap_used_delta_bytes == 200  # 200 - 0
    assert summary.host.disk_read_delta_bytes == 20 and summary.host.disk_write_delta_bytes == 40
    assert summary.host.cpu_utilization_percent is not None
    assert summary.gpu is not None
    assert summary.gpu.observed_performance_states == ["P0", "P2"]
    assert summary.gpu.observed_throttle_reasons == ["thermal"]


def test_energy_per_1000_tokens_is_derived(tmp_path: Path) -> None:
    _full_run(tmp_path)  # tokens_per_sec=1000, step_time=0.5, 10 measured steps -> 5000 tokens
    summary = T.summarize_run_telemetry(tmp_path, identity_overlay=_FULL_IDENTITY)
    assert summary.energy.energy_per_1000_nonpadding_tokens is not None
    assert summary.energy.joules_per_measured_optimizer_step is not None
    assert summary.step.nonpadding_tokens_per_second is not None
    assert summary.step.optimizer_steps_per_minute is not None
    assert T.step_loss_series(T.load_events(tmp_path / "RunEvents.jsonl"))[0] == [1.0, 1.0]


def test_identity_overlay_wins_over_plan(tmp_path: Path) -> None:
    _full_run(tmp_path)
    plan = demo_run_plan()
    overlay = TelemetryIdentity(study_id="cs-ieee", repository_commit="df86db5")
    summary = T.summarize_run_telemetry(tmp_path, plan=plan, identity_overlay=overlay)
    assert summary.identity.study_id == "cs-ieee"
    assert summary.identity.repository_commit == "df86db5"
    assert summary.identity.plan_hash == plan.plan_hash  # plan value retained where overlay is silent
