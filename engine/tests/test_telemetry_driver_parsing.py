"""Pre-live validation of the nvidia-smi parser + process-tree RSS + sampler cleanup (Section 4).

No CUDA, no model load. Captured RTX 5070 driver output (driver 595.71.05) is replayed to prove every
supported query field parses, that missing/unsupported fields stay null (never a fabricated zero or a
literal sentinel string), and that partial/failed driver output degrades safely. Process-tree RSS is
validated against a real synthetic parent -> child -> grandchild tree, and the sampler is proven to
leave no live thread after stop or a failing probe. A read-only live-driver query runs where a GPU is
present and is skipped otherwise. This reduces the chance of wasting the first smoke; it does NOT
replace first-smoke validation.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from corpus_studio.platform import telemetry as T

# Captured verbatim from `nvidia-smi --query-gpu=index,uuid,utilization.gpu,utilization.memory,
# power.draw,temperature.gpu,clocks.gr,clocks.mem,pstate,memory.used,memory.total
# --format=csv,noheader,nounits --id=0` on the RTX 5070 host (driver 595.71.05). The trailing two
# columns are device memory (MiB) - the driver-side used/free the torch-free sampler folds in.
REAL_NOUNITS = "0, GPU-0944f0c7-0790-6bc8-5d8a-a3e6287594b5, 0, 25, 6.47, 38, 180, 405, P8, 10, 12227"
# The same query WITHOUT --nounits (units + spacing the parser must tolerate defensively).
REAL_UNITS = (
    "0, GPU-0944f0c7-0790-6bc8-5d8a-a3e6287594b5, 0 %, 25 %, 6.47 W, 38, 180 MHz, 405 MHz, P8, "
    "10 MiB, 12227 MiB"
)

PROBE_QUERY = [
    "index", "uuid", "utilization.gpu", "utilization.memory", "power.draw",
    "temperature.gpu", "clocks.gr", "clocks.mem", "pstate", "memory.used", "memory.total",
]


class _FakeCompleted:
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode = returncode
        self.stdout = stdout


def _row(monkeypatch, line: str) -> list[str] | None:
    monkeypatch.setattr(T.subprocess, "run", lambda *a, **k: _FakeCompleted(0, line + "\n"))
    return T._run_nvidia_smi(PROBE_QUERY)


# --------------------------------------------------------------------------------------------------
# Field parsing against real captured driver output
# --------------------------------------------------------------------------------------------------
def test_every_probe_field_parses_from_real_nounits_output(monkeypatch) -> None:
    monkeypatch.setattr(T, "_run_nvidia_smi", lambda query: REAL_NOUNITS.split(", "))
    monkeypatch.setattr("corpus_studio.platform.watchdog.sample_gpu_memory", lambda: None)
    reading = T.probe_gpu()
    gpu = reading.gpu
    assert gpu is not None
    assert gpu.device_index == 0
    assert gpu.device_uuid == "GPU-0944f0c7-0790-6bc8-5d8a-a3e6287594b5"
    assert gpu.utilization_percent == 0.0
    assert gpu.memory_controller_utilization_percent == 25.0
    assert gpu.power_watts == 6.47
    assert gpu.temperature_c == 38.0
    assert gpu.graphics_clock_mhz == 180.0
    assert gpu.memory_clock_mhz == 405.0
    assert gpu.performance_state == "P8"
    # Driver device memory (MiB -> bytes) rides on the same probe even with no torch allocator sample.
    assert reading.memory is not None
    assert reading.memory.cuda_device_used_bytes == 10 * 1024 * 1024
    assert reading.memory.cuda_device_free_bytes == (12227 - 10) * 1024 * 1024


def test_units_and_extra_spacing_are_tolerated(monkeypatch) -> None:
    # The sealed probe uses --nounits, but a driver that ignores it (or a locale that pads spacing)
    # must not crash or fabricate: numeric cells keep their value, string cells pass through.
    monkeypatch.setattr(T, "_run_nvidia_smi", lambda query: [c.strip() for c in REAL_UNITS.split(",")])
    monkeypatch.setattr("corpus_studio.platform.watchdog.sample_gpu_memory", lambda: None)
    gpu = T.probe_gpu().gpu
    assert gpu is not None
    assert gpu.power_watts == 6.47 and gpu.graphics_clock_mhz == 180.0
    assert gpu.utilization_percent == 0.0 and gpu.performance_state == "P8"


@pytest.mark.parametrize("sentinel", ["[N/A]", "[Not Supported]", "[Requested functionality has been deprecated]", "[Unknown Error]"])
def test_unsupported_fields_become_null_not_a_sentinel_string(monkeypatch, sentinel: str) -> None:
    # Every field reports the driver's bracketed sentinel; the parser preserves each as null.
    row = [sentinel] * len(PROBE_QUERY)
    monkeypatch.setattr(T, "_run_nvidia_smi", lambda query: row)
    monkeypatch.setattr("corpus_studio.platform.watchdog.sample_gpu_memory", lambda: None)
    gpu = T.probe_gpu().gpu
    assert gpu is not None
    assert gpu.device_index is None and gpu.device_uuid is None
    assert gpu.power_watts is None and gpu.temperature_c is None
    assert gpu.performance_state is None  # not the literal "[N/A]" string


def test_float_and_str_helpers_never_fabricate() -> None:
    assert T._float_or_none("6.47") == 6.47
    assert T._float_or_none("180 MHz") == 180.0  # stray unit tolerated
    assert T._float_or_none("[N/A]") is None
    assert T._float_or_none("") is None
    assert T._float_or_none("P8") is None  # a non-numeric token stays null, never 0
    assert T._str_or_none("GPU-abc") == "GPU-abc"
    assert T._str_or_none("[Not Supported]") is None
    assert T._str_or_none("  ") is None


def test_partial_output_marks_gpu_unavailable(monkeypatch) -> None:
    # Fewer cells than the query (a truncated line) is not force-fit: the whole GPU reading degrades.
    monkeypatch.setattr(T, "_run_nvidia_smi", lambda query: ["0", "GPU-abc", "55"])
    monkeypatch.setattr("corpus_studio.platform.watchdog.sample_gpu_memory", lambda: None)
    reading = T.probe_gpu()
    assert reading.gpu is None and "nvidia_smi_gpu" in reading.unavailable


def test_nvidia_smi_failure_and_partial_lines(monkeypatch) -> None:
    # Nonzero exit -> None; empty stdout -> None; OSError (tool absent) -> None.
    monkeypatch.setattr(T.subprocess, "run", lambda *a, **k: _FakeCompleted(9, "some error\n"))
    assert T._run_nvidia_smi(PROBE_QUERY) is None
    monkeypatch.setattr(T.subprocess, "run", lambda *a, **k: _FakeCompleted(0, "   \n"))
    assert T._run_nvidia_smi(PROBE_QUERY) is None

    def _boom(*a, **k):
        raise OSError("nvidia-smi missing")

    monkeypatch.setattr(T.subprocess, "run", _boom)
    assert T._run_nvidia_smi(PROBE_QUERY) is None
    # A well-formed row still parses through the real _run_nvidia_smi (split on ',' + strip each cell).
    parsed = _row(monkeypatch, REAL_NOUNITS)
    assert parsed is not None and parsed[0] == "0" and parsed[4] == "6.47"


def test_live_driver_query_parses_when_a_gpu_is_present(monkeypatch) -> None:
    # Read-only: run the REAL nvidia-smi. Skipped where no GPU/driver is present (e.g. CI runners).
    monkeypatch.setattr("corpus_studio.platform.watchdog.sample_gpu_memory", lambda: None)
    row = T._run_nvidia_smi(PROBE_QUERY)
    if row is None:
        pytest.skip("no nvidia-smi / GPU on this host")
    assert len(row) == len(PROBE_QUERY)
    gpu = T.probe_gpu().gpu
    assert gpu is not None
    # Sanity, not exact values: numeric fields are float-or-null, pstate matches P<n> or is null.
    assert gpu.power_watts is None or gpu.power_watts >= 0
    assert gpu.temperature_c is None or gpu.temperature_c >= 0
    assert gpu.performance_state is None or (
        gpu.performance_state.startswith("P") and gpu.performance_state[1:].isdigit()
    )


# --------------------------------------------------------------------------------------------------
# Process-tree RSS rooting: synthetic parent -> child -> grandchild
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not Path("/proc").is_dir(), reason="process-tree RSS needs /proc (Linux)")
def test_process_tree_rss_sums_a_three_level_tree() -> None:
    # grandchild allocates ~128 MiB; parent + child are near-idle. If the RSS walk reaches depth 2,
    # the parent-rooted tree total includes the grandchild's memory (well over 100 MiB).
    grandchild = "import time; buf = bytearray(128*1024*1024); buf[::4096] = b'x'*len(buf[::4096]); time.sleep(60)"
    child = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {grandchild!r}]); time.sleep(60)"
    parent_code = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(60)"
    parent = subprocess.Popen([sys.executable, "-c", parent_code], start_new_session=True)
    try:
        deadline = time.monotonic() + 15.0
        tree_rss = 0
        while time.monotonic() < deadline:
            tree_rss = T._read_process_tree_rss(parent.pid) or 0
            if tree_rss > 100 * 1024 * 1024:
                break
            time.sleep(0.2)
        # The grandchild (depth 2) is included: the tree total exceeds 100 MiB.
        assert tree_rss > 100 * 1024 * 1024, f"tree RSS {tree_rss} did not include the grandchild"
        # A pid that does not exist resolves to None, never a fabricated value.
        assert T._read_process_tree_rss(2_000_000_123) is None
    finally:
        try:
            os.killpg(os.getpgid(parent.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):  # pragma: no cover - teardown best-effort
            parent.kill()
        parent.wait(timeout=10)


# --------------------------------------------------------------------------------------------------
# Sampler overhead measurement + no lingering thread after stop / failing probe
# --------------------------------------------------------------------------------------------------
def _telemetry_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("telemetry-")]


def test_overhead_is_measured_and_thread_is_reaped_on_stop(tmp_path: Path) -> None:
    sampler = T.TelemetrySampler(
        "run-overhead", tmp_path, probe=lambda: T.SampleReading(), interval_ms=5.0
    )
    sampler.start()
    time.sleep(0.05)
    overhead = sampler.stop()
    assert overhead.total_sampler_seconds is not None and overhead.total_sampler_seconds >= 0
    assert overhead.per_sample_mean_seconds is not None
    # No telemetry thread survives the stop.
    assert not any(t.name == "telemetry-run-overhead" and t.is_alive() for t in _telemetry_threads())


def test_failing_probe_does_not_wedge_or_leak_the_sampler(tmp_path: Path) -> None:
    before = len(_telemetry_threads())

    def _always_raises() -> T.SampleReading:
        raise RuntimeError("probe blew up")

    sampler = T.TelemetrySampler("run-fail", tmp_path, probe=_always_raises, interval_ms=5.0)
    sampler.start()
    time.sleep(0.05)
    sampler.stop()  # a probe that always raises still degrades samples, never kills the thread wedged
    # Samples were still written (each degraded, not zero-filled), and the thread is gone.
    samples = T.load_samples(sampler.samples_path)
    assert samples and all(s.probe_unavailable for s in samples)
    assert len(_telemetry_threads()) == before
