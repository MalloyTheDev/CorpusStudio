"""CLI coverage for the measurement harness commands (GPU-free).

``telemetry-summarize`` derives the summary from pre-written raw records; ``platform-run --telemetry``
argument validation is exercised without starting a GPU sampler.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.platform.common import HashRef, Ref
from corpus_studio.platform.contracts import EventMetrics, RunEvent, RunManifest, TelemetrySample
from corpus_studio.platform.enums import OperatingSystem, StageMarker

runner = CliRunner()


def _build_raw_run(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(
        run_id="run-cli",
        plan_ref=Ref(id="plan-cli", hash=HashRef(value="a" * 64)),
        created_at="2026-07-15T00:00:00+00:00",
        updated_at="2026-07-15T00:00:03+00:00",
        state="succeeded",
    )
    (directory / "RunManifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    events = [
        RunEvent(event_type="stage", run_id="run-cli", seq=0, emitted_at="t",
                 stage=StageMarker.process_start),
        RunEvent(event_type="metric", run_id="run-cli", seq=1, emitted_at="t", optimizer_step=3,
                 metrics=EventMetrics(loss=0.5, step_time_seconds=0.4, tokens_per_sec=900.0)),
    ]
    (directory / "RunEvents.jsonl").write_text(
        "".join(e.model_dump_json() + "\n" for e in events), encoding="utf-8"
    )
    samples = [
        TelemetrySample(run_id="run-cli", sample_seq=i, monotonic_ns=i * 200_000_000, wall_utc="t",
                        phase="measured", sample_source=OperatingSystem.linux, optimizer_step=3)
        for i in range(2)
    ]
    (directory / "TelemetrySamples.jsonl").write_text(
        "".join(s.model_dump_json() + "\n" for s in samples), encoding="utf-8"
    )


def test_telemetry_summarize_writes_and_prints_json(tmp_path: Path) -> None:
    _build_raw_run(tmp_path)
    result = runner.invoke(app, ["telemetry-summarize", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert '"run_id": "run-cli"' in result.output
    assert (tmp_path / "RunTelemetrySummary.json").is_file()


def test_telemetry_summarize_csv_and_table(tmp_path: Path) -> None:
    _build_raw_run(tmp_path)
    csv = runner.invoke(app, ["telemetry-summarize", str(tmp_path), "--csv", "--no-write"])
    assert csv.exit_code == 0 and "metric,unit,value" in csv.output
    assert not (tmp_path / "RunTelemetrySummary.json").is_file()  # --no-write honored
    table = runner.invoke(app, ["telemetry-summarize", str(tmp_path), "--table", "--no-write"])
    assert table.exit_code == 0 and "Run telemetry summary" in table.output


def test_telemetry_summarize_missing_manifest_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["telemetry-summarize", str(tmp_path)])
    assert result.exit_code == 2
    assert "No RunManifest.json" in result.output


def test_platform_run_telemetry_requires_out() -> None:
    result = runner.invoke(app, ["platform-run", "--demo", "--telemetry"])
    assert result.exit_code == 2
    assert "--telemetry requires --out" in result.output
