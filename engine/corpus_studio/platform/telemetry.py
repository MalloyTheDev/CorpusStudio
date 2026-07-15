"""The measurement harness (platform Section 11).

Two halves, one rule. The **sampler** takes raw environmental samples (GPU + host) on a fixed cadence
and appends each as one :class:`TelemetrySample` JSON line to ``<run-dir>/TelemetrySamples.jsonl`` as
it is taken. The **aggregator** reads that raw series plus the durable ``RunEvent`` stream and the
authoritative ``RunManifest`` and DERIVES a single :class:`RunTelemetrySummary`; CSV, tables, and plot
series all render from that derived object. The rule: raw is authoritative and is written before any
summary, the summary binds its raw sources by sha256, and a telemetry gap is never zero-filled or
allowed to convert a workload success into paper data.

Dependency-light: importing this module pulls no torch and no third-party runtime dependency. The
default GPU/host probes lazy-shell to ``nvidia-smi`` and read ``/proc`` only when actually sampling,
and every probe is fail-soft (a missing driver field stays ``null`` and names the probe on the
sample). All interval math uses the monotonic clock; wall-clock UTC is lineage only. Native-Linux,
WSL, and Windows samples carry a distinct ``sample_source`` and are never collapsed together.
"""

from __future__ import annotations

import hashlib
import os
import statistics
import subprocess  # noqa: S404 - fixed nvidia-smi argv, never a shell string.
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO, Literal, cast

from corpus_studio.platform.common import MemoryMetrics
from corpus_studio.platform.contracts import (
    EnergyIntegration,
    GpuTelemetrySample,
    GpuTelemetrySummary,
    HostTelemetrySample,
    HostTelemetrySummary,
    MeasurementOverhead,
    MemoryWindowSummary,
    MetricSummary,
    OptimizerStepLossEvidence,
    RawRecordBinding,
    RunManifest,
    RunOutcomeSummary,
    RunEvent,
    RunPlan,
    RunTelemetrySummary,
    SamplingCadence,
    ScientificCompleteness,
    StepTelemetrySummary,
    TelemetryIdentity,
    TelemetrySample,
    TrialConfidenceInterval,
)
from corpus_studio.platform.enums import OperatingSystem
from corpus_studio.platform.host_platform import detect_operating_system

Phase = Literal["baseline", "setup", "warmup", "measured", "teardown"]

# Warm-up optimizer steps excluded from the steady-state window (research METRICS.md: steps 1-2).
DEFAULT_WARMUP_STEPS = 2
# The n=3 planned-trial two-sided 95% Student-t multiplier, t_(0.975,2) (research METRICS.md).
T_MULTIPLIER_N3 = 4.3026527299

SAMPLES_FILENAME = "TelemetrySamples.jsonl"
EVENTS_FILENAME = "RunEvents.jsonl"
MANIFEST_FILENAME = "RunManifest.json"
SUMMARY_FILENAME = "RunTelemetrySummary.json"

# The fields a run must carry to be usable as paper data. A run can succeed with any of these missing;
# the summary then reports scientifically_complete=False rather than silently presenting partial data.
REQUIRED_PAPER_FIELDS: tuple[str, ...] = (
    "step.step_losses",
    "step.step_time_seconds",
    "gpu.power_watts",
    "gpu.memory",
    "energy.run_joules",
    "host.process_tree_rss",
    "identity.repository_commit",
    "identity.worker_wheel_sha256",
    "identity.environment_lock_hash",
    "identity.plan_hash",
    "identity.execution_configuration_hash",
    "identity.run_id",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------------------------------
# Raw probes — fail-soft, lazy, torch-free
# --------------------------------------------------------------------------------------------------
@dataclass
class SampleReading:
    """One probe's contribution to a sample: partial GPU/host/memory data plus the names of any
    probes that could not produce a value this tick (never zero-filled)."""

    gpu: GpuTelemetrySample | None = None
    host: HostTelemetrySample | None = None
    memory: MemoryMetrics | None = None
    unavailable: list[str] = field(default_factory=list)


Probe = Callable[[], SampleReading]


def _run_nvidia_smi(query: Sequence[str]) -> list[str] | None:
    """Return one CSV row (already split) from ``nvidia-smi --query-gpu`` for GPU index 0, or None if
    the tool is absent, errors, or times out. Fixed argv - never a shell string."""

    argv = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(query),
        "--format=csv,noheader,nounits",
        "--id=0",
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, bounded timeout.
            argv,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    first_line = completed.stdout.strip().splitlines()[0]
    return [cell.strip() for cell in first_line.split(",")]


def _float_or_none(cell: str) -> float | None:
    """Parse a numeric driver cell to float, or None. Tolerates a stray trailing unit the driver can
    emit even under ``--nounits`` (e.g. ``6.47 W`` -> 6.47, ``180 MHz`` -> 180) and preserves any
    ``[N/A]``-style sentinel or otherwise-unparseable value as null - never a fabricated zero."""

    text = cell.strip()
    try:
        return float(text)
    except (TypeError, ValueError):
        pass
    head = text.split()
    if head:
        try:
            return float(head[0])
        except (TypeError, ValueError):
            return None
    return None


def _str_or_none(cell: str) -> str | None:
    """A driver string field, or None for an empty cell or a bracketed not-available sentinel (e.g.
    ``[N/A]``, ``[Not Supported]``, ``[Requested functionality has been deprecated]``). A missing value
    is preserved as null, never surfaced as a literal sentinel string."""

    value = cell.strip()
    if not value or (value.startswith("[") and value.endswith("]")):
        return None
    return value


def probe_gpu() -> SampleReading:
    """Default GPU probe: nvidia-smi for utilization/power/temperature/clocks + the watchdog memory
    sampler for allocator/CUDA/process memory. Every field is optional and fail-soft."""

    reading = SampleReading()
    query = [
        "index",
        "uuid",
        "utilization.gpu",
        "utilization.memory",
        "power.draw",
        "temperature.gpu",
        "clocks.gr",
        "clocks.mem",
        "pstate",
    ]
    row = _run_nvidia_smi(query)
    if row is not None and len(row) == len(query):
        index = None
        try:
            index = int(row[0])
        except (TypeError, ValueError):
            index = None
        reading.gpu = GpuTelemetrySample(
            device_index=index if index is not None and index >= 0 else None,
            device_uuid=_str_or_none(row[1]),
            utilization_percent=_float_or_none(row[2]),
            memory_controller_utilization_percent=_float_or_none(row[3]),
            power_watts=_float_or_none(row[4]),
            temperature_c=_float_or_none(row[5]),
            graphics_clock_mhz=_float_or_none(row[6]),
            memory_clock_mhz=_float_or_none(row[7]),
            performance_state=_str_or_none(row[8]),
        )
    else:
        reading.unavailable.append("nvidia_smi_gpu")

    try:
        from corpus_studio.platform.watchdog import sample_gpu_memory  # noqa: PLC0415

        memory = sample_gpu_memory()
    except Exception:  # noqa: BLE001 - a memory-probe failure must not break sampling.
        memory = None
    if memory is not None:
        reading.memory = memory
    else:
        reading.unavailable.append("gpu_memory")
    return reading


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - /proc/meminfo absent; non-Linux host, validated off-CI.
        return values
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            # /proc/meminfo is in kB.
            values[key.strip()] = int(parts[0]) * 1024
    return values


def _read_process_tree_rss(root_pid: int) -> int | None:
    """Best-effort sum of RSS bytes over ``root_pid`` and its descendants from ``/proc``. Returns
    None when ``/proc`` is unavailable (non-Linux) or unreadable."""

    proc = Path("/proc")
    if not proc.is_dir():  # pragma: no cover - non-Linux host; validated only off-CI.
        return None
    page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    children: dict[int, list[int]] = {}
    rss_pages: dict[int, int] = {}
    try:
        pids = [int(entry.name) for entry in proc.iterdir() if entry.name.isdigit()]
    except OSError:  # pragma: no cover - /proc iteration race; defensive only.
        return None
    for pid in pids:
        try:
            stat = (proc / str(pid) / "stat").read_text(encoding="utf-8")
            statm = (proc / str(pid) / "statm").read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - the pid vanished between listing and read (race).
            continue
        # ppid is field 4 of stat, but comm (field 2) can contain spaces/parens; split after ')'.
        close = stat.rfind(")")
        if close == -1:  # pragma: no cover - malformed /proc stat; defensive only.
            continue
        after = stat[close + 2 :].split()
        if len(after) < 2:  # pragma: no cover - malformed /proc stat; defensive only.
            continue
        try:
            ppid = int(after[1])
            resident_pages = int(statm.split()[1])
        except (ValueError, IndexError):  # pragma: no cover - malformed /proc entry; defensive only.
            continue
        children.setdefault(ppid, []).append(pid)
        rss_pages[pid] = resident_pages
    if root_pid not in rss_pages:
        return None
    total_pages = 0
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:  # pragma: no cover - only a pathological /proc ppid cycle reaches this.
            continue
        seen.add(pid)
        total_pages += rss_pages.get(pid, 0)
        stack.extend(children.get(pid, []))
    return total_pages * page_size


def _read_self_io() -> tuple[int | None, int | None]:
    try:
        text = Path("/proc/self/io").read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - /proc/self/io absent or restricted; non-Linux/off-CI.
        return None, None
    read_bytes = write_bytes = None
    for line in text.splitlines():
        if line.startswith("read_bytes:"):
            read_bytes = int(line.split()[1])
        elif line.startswith("write_bytes:"):
            write_bytes = int(line.split()[1])
    return read_bytes, write_bytes


def probe_host(root_pid: int | None = None) -> SampleReading:
    """Default host probe: /proc meminfo + process-tree RSS + self IO. Fail-soft on any non-Linux or
    unreadable source."""

    reading = SampleReading()
    pid = root_pid if root_pid is not None else os.getpid()
    meminfo = _read_meminfo()
    tree_rss = _read_process_tree_rss(pid)
    read_bytes, write_bytes = _read_self_io()
    total = meminfo.get("MemTotal")
    available = meminfo.get("MemAvailable")
    swap_total = meminfo.get("SwapTotal")
    swap_free = meminfo.get("SwapFree")
    system_used = total - available if total is not None and available is not None else None
    swap_used = (
        swap_total - swap_free if swap_total is not None and swap_free is not None else None
    )
    if not meminfo and tree_rss is None:
        reading.unavailable.append("proc_host")
        return reading
    reading.host = HostTelemetrySample(
        process_tree_rss_bytes=tree_rss,
        system_ram_used_bytes=system_used,
        system_ram_available_bytes=available,
        swap_used_bytes=swap_used,
        disk_read_bytes=read_bytes,
        disk_write_bytes=write_bytes,
    )
    if tree_rss is None:
        reading.unavailable.append("process_tree_rss")
    return reading


def default_probe(root_pid: int | None = None) -> Probe:
    """Compose the default GPU + host probes into one :class:`Probe`."""

    def _probe() -> SampleReading:
        gpu = probe_gpu()
        host = probe_host(root_pid)
        return SampleReading(
            gpu=gpu.gpu,
            host=host.host,
            memory=gpu.memory,
            unavailable=sorted(set(gpu.unavailable) | set(host.unavailable)),
        )

    return _probe


# --------------------------------------------------------------------------------------------------
# The sampler
# --------------------------------------------------------------------------------------------------
class TelemetrySampler:
    """Appends raw :class:`TelemetrySample` lines to ``<record_dir>/TelemetrySamples.jsonl`` on a
    fixed cadence. The probe, monotonic clock, and wall clock are injectable so the sampler is
    deterministic in tests without a GPU. Phase and current optimizer step are mutable so the caller
    marks the warm-up/measured boundary; interval math uses ``monotonic_ns``."""

    def __init__(
        self,
        run_id: str,
        record_dir: str | Path,
        *,
        probe: Probe | None = None,
        interval_ms: float = 200.0,
        source: OperatingSystem | None = None,
        root_pid: int | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        clock: Callable[[], str] = _now_iso,
    ) -> None:
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self.run_id = run_id
        self._dir = Path(record_dir)
        self._uses_default_probe = probe is None
        self._probe = probe or default_probe(root_pid)
        self._interval_s = interval_ms / 1000.0
        self.interval_ms = interval_ms
        self._source = source or detect_operating_system()[0]
        self._monotonic_ns = monotonic_ns
        self._clock = clock
        self._seq = 0
        self._phase: Phase = "baseline"
        self._optimizer_step: int | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle: TextIO | None = None
        self._busy_seconds = 0.0
        self._samples_written = 0

    @property
    def samples_path(self) -> Path:
        return self._dir / SAMPLES_FILENAME

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = cast(Phase, phase)

    def set_root_pid(self, pid: int) -> None:
        """Re-root the default host process-tree probe (subprocess mode points it at the worker
        child). A no-op when a custom probe was injected - the caller owns that probe's target."""

        if self._uses_default_probe:
            with self._lock:
                self._probe = default_probe(pid)

    def mark_step(self, optimizer_step: int | None, phase: str | None = None) -> None:
        with self._lock:
            self._optimizer_step = optimizer_step
            if phase is not None:
                self._phase = cast(Phase, phase)

    def _open(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._handle = self.samples_path.open("a", encoding="utf-8")  # noqa: SIM115

    def take_sample(self) -> TelemetrySample:
        """Take, write, and return exactly one sample. Public so tests drive the sampler
        deterministically without the timing thread. Accrues the probe's busy time as overhead."""

        if self._handle is None:
            self._open()
        start = time.perf_counter()
        try:
            reading = self._probe()
        except Exception as exc:  # noqa: BLE001 - a probe crash degrades this sample, never the run.
            reading = SampleReading(unavailable=[f"probe:{type(exc).__name__}"])
        with self._lock:
            phase = self._phase
            step = self._optimizer_step
            seq = self._seq
            self._seq += 1
        sample = TelemetrySample(
            run_id=self.run_id,
            sample_seq=seq,
            monotonic_ns=self._monotonic_ns(),
            wall_utc=self._clock(),
            phase=phase,
            sample_source=self._source,
            optimizer_step=step,
            gpu=reading.gpu,
            host=reading.host,
            memory=reading.memory,
            probe_unavailable=sorted(set(reading.unavailable)),
        )
        assert self._handle is not None
        self._handle.write(sample.model_dump_json() + "\n")
        self._handle.flush()
        self._samples_written += 1
        self._busy_seconds += time.perf_counter() - start
        return sample

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.take_sample()
            self._stop.wait(self._interval_s)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("sampler already started")
        self._open()
        self._thread = threading.Thread(target=self._loop, name=f"telemetry-{self.run_id}", daemon=True)
        self._thread.start()

    def stop(self) -> MeasurementOverhead:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s * 4 + 2.0)
            self._thread = None
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None
        return self.overhead()

    def overhead(self, wall_seconds: float | None = None) -> MeasurementOverhead:
        per_sample = (
            self._busy_seconds / self._samples_written if self._samples_written else None
        )
        fraction = (
            self._busy_seconds / wall_seconds
            if wall_seconds is not None and wall_seconds > 0
            else None
        )
        return MeasurementOverhead(
            total_sampler_seconds=self._busy_seconds,
            per_sample_mean_seconds=per_sample,
            wall_seconds=wall_seconds,
            overhead_fraction_of_wall=fraction,
        )


# --------------------------------------------------------------------------------------------------
# Loading raw records
# --------------------------------------------------------------------------------------------------
def _sha256_and_lines(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            lines += chunk.count(b"\n")
    return digest.hexdigest(), lines


def append_jsonl(path: str | Path, line_obj: str) -> None:
    """Append one already-serialized JSON line durably (append + flush). The atomic-per-line
    convention makes a torn tail at most one lost trailing sample, never a corrupt record."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line_obj + "\n")
        handle.flush()


def load_samples(path: str | Path) -> list[TelemetrySample]:
    """Parse every non-blank line as a :class:`TelemetrySample`. Raises on a malformed line so a
    corrupt raw record is never silently dropped."""

    samples: list[TelemetrySample] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        samples.append(TelemetrySample.model_validate_json(raw))
    return samples


def load_events(path: str | Path) -> list[RunEvent]:
    events: list[RunEvent] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        events.append(RunEvent.model_validate_json(raw))
    return events


def order_and_dedupe_samples(samples: Iterable[TelemetrySample]) -> list[TelemetrySample]:
    """Deterministically order raw samples by ``sample_seq`` and keep the first record for each seq,
    so a reordered or duplicated stream aggregates identically to the clean stream."""

    by_seq: dict[int, TelemetrySample] = {}
    for sample in samples:
        by_seq.setdefault(sample.sample_seq, sample)
    return [by_seq[key] for key in sorted(by_seq)]


# --------------------------------------------------------------------------------------------------
# Statistics + energy
# --------------------------------------------------------------------------------------------------
def _metric_summary(values: Sequence[float], unit: str) -> MetricSummary | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    sd = statistics.stdev(clean) if len(clean) >= 2 else None
    return MetricSummary(
        unit=unit,
        count=len(clean),
        mean=statistics.fmean(clean),
        median=statistics.median(clean),
        sample_standard_deviation=sd,
        minimum=min(clean),
        maximum=max(clean),
    )


def _integrate_energy(
    points: Sequence[tuple[float, float]],
    lo: float | None = None,
    hi: float | None = None,
) -> tuple[float | None, bool]:
    """Trapezoidal integral of power over time: ``sum 0.5*(P_i+P_{i+1})*(t_{i+1}-t_i)``. ``points``
    are ``(t_seconds, watts)`` sorted by time. When ``lo``/``hi`` bound a window, an interval that
    crosses a boundary is linearly clipped and its power linearly interpolated at the boundary.
    Returns ``(joules, boundary_clipped)``; joules is None when fewer than two points contribute."""

    ordered = sorted(points, key=lambda item: item[0])
    joules = 0.0
    contributing = 0
    clipped = False
    for (t0, p0), (t1, p1) in zip(ordered, ordered[1:]):
        if t1 <= t0:
            continue
        a, b = t0, t1
        pa, pb = p0, p1
        if lo is not None and b <= lo:
            continue
        if hi is not None and a >= hi:
            continue
        if lo is not None and a < lo:
            pa = p0 + (p1 - p0) * (lo - t0) / (t1 - t0)
            a = lo
            clipped = True
        if hi is not None and b > hi:
            pb = p0 + (p1 - p0) * (hi - t0) / (t1 - t0)
            b = hi
            clipped = True
        joules += 0.5 * (pa + pb) * (b - a)
        contributing += 1
    if contributing == 0:
        return None, clipped
    return joules, clipped


def _time_weighted_mean_power(points: Sequence[tuple[float, float]]) -> float | None:
    ordered = sorted(points, key=lambda item: item[0])
    if len(ordered) < 2:
        return None
    energy, _ = _integrate_energy(ordered)
    span = ordered[-1][0] - ordered[0][0]
    if energy is None or span <= 0:
        return None
    return energy / span


# --------------------------------------------------------------------------------------------------
# Memory windows
# --------------------------------------------------------------------------------------------------
_MEMORY_FIELDS = (
    "torch_allocated_bytes",
    "torch_reserved_bytes",
    "torch_peak_allocated_bytes",
    "torch_peak_reserved_bytes",
    "cuda_device_used_bytes",
    "cuda_device_free_bytes",
    "dedicated_gpu_bytes",
    "shared_gpu_bytes",
    "system_ram_used_bytes",
    "process_rss_bytes",
)


def _max_memory(samples: Sequence[MemoryMetrics]) -> MemoryMetrics | None:
    if not samples:
        return None
    result: dict[str, int | None] = {}
    for name in _MEMORY_FIELDS:
        observed = [getattr(s, name) for s in samples if getattr(s, name) is not None]
        result[name] = max(observed) if observed else None
    return MemoryMetrics(**result)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------------
# Identity extraction
# --------------------------------------------------------------------------------------------------
def identity_from_plan(
    plan: RunPlan | None,
    manifest: RunManifest,
    overlay: TelemetryIdentity | None = None,
) -> TelemetryIdentity:
    """Fill the lineage block from the plan + manifest, then apply the caller's ``overlay`` (study /
    protocol / amendment / cell / trial / commit / wheel / lock / capability / probe) which the plan
    does not carry. Overlay values win only where they are set."""

    base = TelemetryIdentity(run_id=manifest.run_id)
    if manifest.resume_lineage is not None:
        # A resumed trial is marked so paper aggregation never averages it in with an uninterrupted
        # one; the parent run and continued-from step come straight from the manifest's lineage.
        base.resumed = True
        base.parent_run_id = manifest.resume_lineage.parent_run_id
        base.resumed_from_global_step = manifest.resume_lineage.resumed_from_global_step
    if plan is not None:
        base.plan_id = plan.plan_id
        base.plan_hash = plan.plan_hash
        base.model_ref = plan.base_model or None
        if plan.dataset_ref is not None and plan.dataset_ref.hash is not None:
            base.dataset_fingerprint = plan.dataset_ref.hash.value
        if plan.environment_ref is not None and plan.environment_ref.hash is not None:
            base.environment_lock_hash = plan.environment_ref.hash.value
        execution = plan.resolved_execution
        if execution is not None:
            base.execution_configuration_hash = getattr(
                execution, "execution_configuration_hash", None
            )
            base.sequence_view = plan.sequence.max_sequence_len
    if overlay is not None:
        for name, value in overlay.model_dump(exclude_none=True).items():
            setattr(base, name, value)
    return base


# --------------------------------------------------------------------------------------------------
# Step evidence extraction
# --------------------------------------------------------------------------------------------------
@dataclass
class _StepRow:
    optimizer_step: int
    loss: float | None
    step_time_seconds: float | None
    tokens_per_sec: float | None
    supervised_tokens_per_sec: float | None


def _step_rows(events: Sequence[RunEvent]) -> list[_StepRow]:
    rows: dict[int, _StepRow] = {}
    for event in events:
        if event.event_type != "metric" or event.optimizer_step is None or event.optimizer_step <= 0:
            continue
        metrics = event.metrics
        rows[event.optimizer_step] = _StepRow(
            optimizer_step=event.optimizer_step,
            loss=metrics.loss if metrics else None,
            step_time_seconds=metrics.step_time_seconds if metrics else None,
            tokens_per_sec=metrics.tokens_per_sec if metrics else None,
            supervised_tokens_per_sec=metrics.supervised_tokens_per_sec if metrics else None,
        )
    return [rows[key] for key in sorted(rows)]


def _step_summary(
    events: Sequence[RunEvent],
    manifest: RunManifest,
    warmup_steps: int,
) -> StepTelemetrySummary:
    rows = _step_rows(events)
    completed = len(rows)
    step_numbers = [row.optimizer_step for row in rows]
    warmup = sorted(step_numbers[:warmup_steps])
    measured = sorted(step_numbers[warmup_steps:])
    measured_set = set(measured)
    losses = [
        OptimizerStepLossEvidence(optimizer_step=row.optimizer_step, loss=row.loss)
        for row in rows
        if row.loss is not None
    ]
    loss_values = [row.loss for row in rows if row.loss is not None]
    measured_rows = [row for row in rows if row.optimizer_step in measured_set]
    step_times = [r.step_time_seconds for r in measured_rows if r.step_time_seconds is not None]
    opt_per_min = None
    if step_times and sum(step_times) > 0:
        opt_per_min = 60.0 * len(step_times) / sum(step_times)
    samples_per_sec_values: list[float] = []  # placeholder until per-step sample counts flow through

    evidence = manifest.training_success_evidence
    gradient_count = changed_count = None
    before_hash = after_hash = None
    if evidence is not None:
        gradient_count = evidence.execution.gradient_coverage.observed_tensor_count
        changed_count = evidence.execution.trainable_state.changed_tensor_count
        before_hash = evidence.execution.trainable_state.before_sha256
        after_hash = evidence.execution.trainable_state.after_sha256

    return StepTelemetrySummary(
        completed_optimizer_steps=completed,
        warmup_optimizer_steps=warmup,
        measured_optimizer_steps=measured,
        step_losses=losses,
        first_loss=loss_values[0] if loss_values else None,
        last_loss=loss_values[-1] if loss_values else None,
        min_loss=min(loss_values) if loss_values else None,
        step_time_seconds=_metric_summary(step_times, "seconds"),
        nonpadding_tokens_per_second=_metric_summary(
            [r.tokens_per_sec for r in measured_rows if r.tokens_per_sec is not None],
            "tokens/second",
        ),
        supervised_tokens_per_second=_metric_summary(
            [
                r.supervised_tokens_per_sec
                for r in measured_rows
                if r.supervised_tokens_per_sec is not None
            ],
            "tokens/second",
        ),
        samples_per_second=_metric_summary(samples_per_sec_values, "samples/second"),
        optimizer_steps_per_minute=opt_per_min,
        gradient_observed_tensor_count=gradient_count,
        changed_adapter_tensor_count=changed_count,
        trainable_state_before_sha256=before_hash,
        trainable_state_after_sha256=after_hash,
    )


# --------------------------------------------------------------------------------------------------
# GPU / host / energy / sampling summaries
# --------------------------------------------------------------------------------------------------
def _seconds(sample: TelemetrySample, origin_ns: int) -> float:
    return (sample.monotonic_ns - origin_ns) / 1e9


def _gpu_summary(
    all_samples: Sequence[TelemetrySample],
    measured: Sequence[TelemetrySample],
) -> GpuTelemetrySummary | None:
    gpu_samples = [s.gpu for s in measured if s.gpu is not None]
    all_gpu = [s.gpu for s in all_samples if s.gpu is not None]
    mem_measured = [s.memory for s in measured if s.memory is not None]
    mem_all = [s.memory for s in all_samples if s.memory is not None]
    baseline_mem = next(
        (s.memory for s in all_samples if s.phase == "baseline" and s.memory is not None),
        None,
    )
    if not gpu_samples and not all_gpu and not mem_all:
        return None
    device_index = next((g.device_index for g in all_gpu if g.device_index is not None), None)
    device_uuid = next((g.device_uuid for g in all_gpu if g.device_uuid is not None), None)
    temps_all = [g.temperature_c for g in all_gpu if g.temperature_c is not None]
    starting_temp = next(
        (g.temperature_c for g in all_gpu if g.temperature_c is not None), None
    )
    memory_window = MemoryWindowSummary(
        baseline=baseline_mem,
        measured_window_max=_max_memory(mem_measured),
        whole_run_max=_max_memory(mem_all),
    )
    return GpuTelemetrySummary(
        device_index=device_index,
        device_uuid=device_uuid,
        utilization_percent=_metric_summary(
            [g.utilization_percent for g in gpu_samples if g.utilization_percent is not None],
            "percent",
        ),
        memory_controller_utilization_percent=_metric_summary(
            [
                g.memory_controller_utilization_percent
                for g in gpu_samples
                if g.memory_controller_utilization_percent is not None
            ],
            "percent",
        ),
        power_watts=_metric_summary(
            [g.power_watts for g in gpu_samples if g.power_watts is not None], "watts"
        ),
        temperature_c=_metric_summary(
            [g.temperature_c for g in gpu_samples if g.temperature_c is not None], "celsius"
        ),
        starting_temperature_c=starting_temp,
        max_run_temperature_c=max(temps_all) if temps_all else None,
        graphics_clock_mhz=_metric_summary(
            [g.graphics_clock_mhz for g in gpu_samples if g.graphics_clock_mhz is not None], "MHz"
        ),
        memory_clock_mhz=_metric_summary(
            [g.memory_clock_mhz for g in gpu_samples if g.memory_clock_mhz is not None], "MHz"
        ),
        observed_performance_states=sorted(
            {g.performance_state for g in all_gpu if g.performance_state is not None}
        ),
        observed_throttle_reasons=sorted(
            {reason for g in all_gpu for reason in g.throttle_reasons}
        ),
        memory=memory_window,
    )


def _host_summary(
    all_samples: Sequence[TelemetrySample],
    measured: Sequence[TelemetrySample],
) -> HostTelemetrySummary | None:
    host_measured = [s.host for s in measured if s.host is not None]
    host_all = [s.host for s in all_samples if s.host is not None]
    if not host_all:
        return None
    rss_measured = [h.process_tree_rss_bytes for h in host_measured if h.process_tree_rss_bytes is not None]
    rss_all = [h.process_tree_rss_bytes for h in host_all if h.process_tree_rss_bytes is not None]
    baseline_rss = next(
        (
            s.host.process_tree_rss_bytes
            for s in all_samples
            if s.phase == "baseline" and s.host is not None and s.host.process_tree_rss_bytes is not None
        ),
        None,
    )
    swaps = [h.swap_used_bytes for h in host_all if h.swap_used_bytes is not None]
    reads = [h.disk_read_bytes for h in host_all if h.disk_read_bytes is not None]
    writes = [h.disk_write_bytes for h in host_all if h.disk_write_bytes is not None]
    rss_window = MemoryWindowSummary()
    if baseline_rss is not None or rss_measured or rss_all:
        rss_window = MemoryWindowSummary(
            baseline=MemoryMetrics(process_rss_bytes=baseline_rss) if baseline_rss is not None else None,
            measured_window_max=(
                MemoryMetrics(process_rss_bytes=max(rss_measured)) if rss_measured else None
            ),
            whole_run_max=MemoryMetrics(process_rss_bytes=max(rss_all)) if rss_all else None,
        )
    return HostTelemetrySummary(
        process_tree_rss=rss_window,
        process_tree_rss_bytes=_metric_summary(rss_measured, "bytes"),
        system_ram_used_bytes=_metric_summary(
            [h.system_ram_used_bytes for h in host_measured if h.system_ram_used_bytes is not None],
            "bytes",
        ),
        swap_used_bytes=_metric_summary(swaps, "bytes"),
        swap_used_delta_bytes=(swaps[-1] - swaps[0]) if len(swaps) >= 2 else None,
        cpu_utilization_percent=_metric_summary(
            [h.cpu_utilization_percent for h in host_measured if h.cpu_utilization_percent is not None],
            "percent",
        ),
        disk_read_delta_bytes=(reads[-1] - reads[0]) if len(reads) >= 2 and reads[-1] >= reads[0] else None,
        disk_write_delta_bytes=(writes[-1] - writes[0]) if len(writes) >= 2 and writes[-1] >= writes[0] else None,
    )


def _energy(
    all_samples: Sequence[TelemetrySample],
    measured: Sequence[TelemetrySample],
    origin_ns: int,
    measured_step_count: int,
    measured_nonpadding_tokens: float | None,
) -> EnergyIntegration:
    all_points = [
        (_seconds(s, origin_ns), s.gpu.power_watts)
        for s in all_samples
        if s.gpu is not None and s.gpu.power_watts is not None
    ]
    measured_powers = [
        s.gpu.power_watts for s in measured if s.gpu is not None and s.gpu.power_watts is not None
    ]
    run_joules, _ = _integrate_energy(all_points)
    window_joules = None
    clipped = False
    coverage = None
    if measured:
        lo = _seconds(measured[0], origin_ns)
        hi = _seconds(measured[-1], origin_ns)
        window_joules, clipped = _integrate_energy(all_points, lo=lo, hi=hi)
        window_span = hi - lo
        if window_span > 0 and len([p for p in all_points if lo <= p[0] <= hi]) >= 2:
            covered = [p[0] for p in all_points if lo <= p[0] <= hi]
            coverage = min(1.0, (max(covered) - min(covered)) / window_span) if covered else None
    per_step = (
        window_joules / measured_step_count
        if window_joules is not None and measured_step_count > 0
        else None
    )
    per_1000 = (
        1000.0 * window_joules / measured_nonpadding_tokens
        if window_joules is not None
        and measured_nonpadding_tokens is not None
        and measured_nonpadding_tokens > 0
        else None
    )
    return EnergyIntegration(
        run_joules=run_joules,
        measured_window_joules=window_joules,
        joules_per_measured_optimizer_step=per_step,
        energy_per_1000_nonpadding_tokens=per_1000,
        power_sample_count=len(all_points),
        time_weighted_mean_power_watts=_time_weighted_mean_power(all_points),
        median_measured_power_watts=(
            statistics.median(measured_powers) if measured_powers else None
        ),
        max_power_watts=max((p for _, p in all_points), default=None) if all_points else None,
        coverage_fraction=coverage,
        boundary_clipped=clipped,
    )


def _cadence(
    samples: Sequence[TelemetrySample],
    requested_interval_ms: float | None,
    source: OperatingSystem | None,
) -> SamplingCadence:
    deltas_ms = [
        (b.monotonic_ns - a.monotonic_ns) / 1e6
        for a, b in zip(samples, samples[1:])
        if b.monotonic_ns >= a.monotonic_ns
    ]
    return SamplingCadence(
        requested_interval_ms=requested_interval_ms,
        sample_count=len(samples),
        observed_median_interval_ms=statistics.median(deltas_ms) if deltas_ms else None,
        observed_min_interval_ms=min(deltas_ms) if deltas_ms else None,
        observed_max_interval_ms=max(deltas_ms) if deltas_ms else None,
        source=source,
    )


# --------------------------------------------------------------------------------------------------
# Completeness
# --------------------------------------------------------------------------------------------------
def _present(summary: RunTelemetrySummary, field_path: str) -> bool:
    step, gpu, host, energy, identity = (
        summary.step,
        summary.gpu,
        summary.host,
        summary.energy,
        summary.identity,
    )
    checks: dict[str, bool] = {
        "step.step_losses": bool(step.step_losses),
        "step.step_time_seconds": step.step_time_seconds is not None,
        "gpu.power_watts": gpu is not None and gpu.power_watts is not None,
        "gpu.memory": gpu is not None
        and gpu.memory is not None
        and gpu.memory.whole_run_max is not None,
        "energy.run_joules": energy.run_joules is not None,
        "host.process_tree_rss": host is not None
        and host.process_tree_rss is not None
        and host.process_tree_rss.whole_run_max is not None,
        "identity.repository_commit": identity.repository_commit is not None,
        "identity.worker_wheel_sha256": identity.worker_wheel_sha256 is not None,
        "identity.environment_lock_hash": identity.environment_lock_hash is not None,
        "identity.plan_hash": identity.plan_hash is not None,
        "identity.execution_configuration_hash": identity.execution_configuration_hash is not None,
        "identity.run_id": identity.run_id is not None,
    }
    return checks.get(field_path, False)


def _completeness(
    summary: RunTelemetrySummary,
    degraded_sample_count: int,
) -> ScientificCompleteness:
    missing = sorted(f for f in REQUIRED_PAPER_FIELDS if not _present(summary, f))
    complete = not missing
    reason = (
        "all required paper telemetry present"
        if complete
        else "missing required paper telemetry: " + ", ".join(missing)
    )
    return ScientificCompleteness(
        scientifically_complete=complete,
        missing_required_paper_fields=missing,
        telemetry_degraded=degraded_sample_count > 0,
        degraded_sample_count=degraded_sample_count,
        reason=reason,
    )


# --------------------------------------------------------------------------------------------------
# The top-level aggregation
# --------------------------------------------------------------------------------------------------
def summarize_run_telemetry(
    record_dir: str | Path,
    *,
    plan: RunPlan | None = None,
    identity_overlay: TelemetryIdentity | None = None,
    requested_interval_ms: float | None = None,
    overhead: MeasurementOverhead | None = None,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    generated_at: str | None = None,
) -> RunTelemetrySummary:
    """Derive the single :class:`RunTelemetrySummary` from the raw records under ``record_dir``. The
    manifest is authoritative for the outcome; the event stream is authoritative for step evidence;
    the telemetry series is authoritative for environmental metrics. Each raw file is bound by sha256
    in ``raw_records`` so the summary is provably a function of its source."""

    directory = Path(record_dir)
    manifest_path = directory / MANIFEST_FILENAME
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    events_path = directory / EVENTS_FILENAME
    samples_path = directory / SAMPLES_FILENAME

    events = load_events(events_path) if events_path.exists() else []
    raw_samples = load_samples(samples_path) if samples_path.exists() else []
    samples = order_and_dedupe_samples(raw_samples)
    measured = [s for s in samples if s.phase == "measured"]
    origin_ns = samples[0].monotonic_ns if samples else 0
    source = samples[0].sample_source if samples else None

    step = _step_summary(events, manifest, warmup_steps)
    gpu = _gpu_summary(samples, measured)
    host = _host_summary(samples, measured)
    tokens = step.nonpadding_tokens_per_second
    measured_nonpadding = None
    if tokens is not None and tokens.mean is not None and step.step_time_seconds is not None:
        # tokens/sec * measured seconds; best-effort measured non-padding token total.
        measured_seconds = (step.step_time_seconds.mean or 0) * len(step.measured_optimizer_steps)
        measured_nonpadding = tokens.mean * measured_seconds if measured_seconds else None
    energy = _energy(samples, measured, origin_ns, len(step.measured_optimizer_steps), measured_nonpadding)
    sampling = _cadence(samples, requested_interval_ms, source)

    outcome = RunOutcomeSummary(
        state=manifest.state,
        training_success=manifest.training_success_evidence is not None,
        failure_taxonomy=manifest.failure.taxonomy if manifest.failure else None,
        failure_stage=manifest.failure.stage if manifest.failure else None,
        measured_fit=manifest.final_fit,
    )
    identity = identity_from_plan(plan, manifest, identity_overlay)
    degraded = sum(1 for s in samples if s.probe_unavailable)

    raw_records: list[RawRecordBinding] = []
    for kind, path in (
        ("run_manifest", manifest_path),
        ("run_events", events_path),
        ("telemetry_samples", samples_path),
    ):
        if path.exists():
            sha256, line_count = _sha256_and_lines(path)
            raw_records.append(
                RawRecordBinding(
                    record_kind=kind,  # type: ignore[arg-type]
                    path=str(path),
                    sha256=sha256,
                    line_count=line_count,
                )
            )

    summary = RunTelemetrySummary(
        run_id=manifest.run_id,
        generated_at=generated_at or _now_iso(),
        identity=identity,
        outcome=outcome,
        step=step,
        gpu=gpu,
        host=host,
        energy=energy,
        sampling=sampling,
        overhead=overhead or MeasurementOverhead(),
        completeness=ScientificCompleteness(scientifically_complete=False),
        raw_records=raw_records,
    )
    summary.completeness = _completeness(summary, degraded)
    return summary


def write_summary(summary: RunTelemetrySummary, record_dir: str | Path) -> Path:
    """Write ``RunTelemetrySummary.json`` atomically (temp-then-replace) AFTER the raw records exist."""

    directory = Path(record_dir)
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / SUMMARY_FILENAME
    tmp = directory / f".{SUMMARY_FILENAME}.{os.getpid()}.tmp"
    tmp.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, final)
    return final


# --------------------------------------------------------------------------------------------------
# Rendering — every renderer derives from the summary/raw, never from a re-measurement
# --------------------------------------------------------------------------------------------------
def _fmt(value: float | None, digits: int) -> str:
    return "" if value is None else f"{round(value, digits)}"


def summary_to_csv(summary: RunTelemetrySummary) -> str:
    """Flat ``metric,unit,value`` CSV at full collected precision (no rounding) - the machine-facing
    export. Tables round; this does not."""

    rows: list[tuple[str, str, object]] = [
        ("run_id", "", summary.run_id),
        ("state", "", summary.outcome.state),
        ("training_success", "", summary.outcome.training_success),
        ("completed_optimizer_steps", "count", summary.step.completed_optimizer_steps),
        ("measured_step_count", "count", len(summary.step.measured_optimizer_steps)),
        ("first_loss", "", summary.step.first_loss),
        ("last_loss", "", summary.step.last_loss),
        ("min_loss", "", summary.step.min_loss),
    ]
    if summary.step.step_time_seconds is not None:
        rows.append(("median_step_time", "seconds", summary.step.step_time_seconds.median))
    rows.extend(
        [
            ("optimizer_steps_per_minute", "steps/min", summary.step.optimizer_steps_per_minute),
            ("run_joules", "joules", summary.energy.run_joules),
            ("measured_window_joules", "joules", summary.energy.measured_window_joules),
            (
                "time_weighted_mean_power",
                "watts",
                summary.energy.time_weighted_mean_power_watts,
            ),
            ("max_power", "watts", summary.energy.max_power_watts),
            ("observed_median_interval", "ms", summary.sampling.observed_median_interval_ms),
            ("sampler_overhead_seconds", "seconds", summary.overhead.total_sampler_seconds),
            (
                "scientifically_complete",
                "",
                summary.completeness.scientifically_complete,
            ),
        ]
    )
    lines = ["metric,unit,value"]
    for name, unit, value in rows:
        rendered = "" if value is None else str(value)
        lines.append(f"{name},{unit},{rendered}")
    return "\n".join(lines) + "\n"


def summary_to_table(summary: RunTelemetrySummary) -> str:
    """A rounded Markdown table (research METRICS.md rounding: time 3dp, throughput 2dp, power 2W,
    energy 2J, temperature 1C) that still links back to the raw-precision CSV/JSON."""

    step = summary.step
    energy = summary.energy
    gpu = summary.gpu
    lines = [
        f"# Run telemetry summary - {summary.run_id}",
        "",
        f"- state: **{summary.outcome.state}** (training_success={summary.outcome.training_success})",
        f"- scientifically_complete: **{summary.completeness.scientifically_complete}**",
        "",
        "| metric | value | unit |",
        "| --- | ---: | --- |",
        f"| completed optimizer steps | {step.completed_optimizer_steps} | count |",
        f"| measured steps | {len(step.measured_optimizer_steps)} | count |",
        f"| first / last loss | {_fmt(step.first_loss, 4)} / {_fmt(step.last_loss, 4)} | - |",
    ]
    if step.step_time_seconds is not None:
        lines.append(
            f"| median step time | {_fmt(step.step_time_seconds.median, 3)} | seconds |"
        )
    if step.optimizer_steps_per_minute is not None:
        lines.append(
            f"| optimizer steps / min | {_fmt(step.optimizer_steps_per_minute, 2)} | steps/min |"
        )
    lines.extend(
        [
            f"| run energy | {_fmt(energy.run_joules, 2)} | joules |",
            f"| measured-window energy | {_fmt(energy.measured_window_joules, 2)} | joules |",
            f"| time-weighted mean power | {_fmt(energy.time_weighted_mean_power_watts, 2)} | watts |",
            f"| max power | {_fmt(energy.max_power_watts, 2)} | watts |",
        ]
    )
    if gpu is not None:
        lines.append(
            f"| starting / max temperature | {_fmt(gpu.starting_temperature_c, 1)} / "
            f"{_fmt(gpu.max_run_temperature_c, 1)} | celsius |"
        )
    if summary.completeness.missing_required_paper_fields:
        lines.extend(
            [
                "",
                "Missing required paper telemetry: "
                + ", ".join(summary.completeness.missing_required_paper_fields),
            ]
        )
    return "\n".join(lines) + "\n"


def power_series(samples: Sequence[TelemetrySample]) -> list[list[float]]:
    """Plot-ready ``[t_seconds, watts]`` rows derived from the raw samples (the same source the energy
    integral uses). The control plane stays dependency-light: it emits the series a plotting tool
    renders, not a rasterized image."""

    ordered = order_and_dedupe_samples(samples)
    origin = ordered[0].monotonic_ns if ordered else 0
    return [
        [(s.monotonic_ns - origin) / 1e9, s.gpu.power_watts]
        for s in ordered
        if s.gpu is not None and s.gpu.power_watts is not None
    ]


def step_loss_series(events: Sequence[RunEvent]) -> list[list[float]]:
    """Plot-ready ``[optimizer_step, loss]`` rows from the durable event stream."""

    return [[float(row.optimizer_step), row.loss] for row in _step_rows(events) if row.loss is not None]


# --------------------------------------------------------------------------------------------------
# Cross-trial statistics (research METRICS.md) — consumes several per-run summaries
# --------------------------------------------------------------------------------------------------
def combine_trial_values(values: Sequence[float], unit: str) -> tuple[MetricSummary, TrialConfidenceInterval]:
    """Descriptive statistics + the two-sided 95% Student-t interval for a per-trial mean. The
    interval is reported only for exactly three successful trials (the planned n=3 design,
    t=4.3026527299); with fewer, values are returned but no confirmatory interval is fabricated."""

    clean = [float(v) for v in values]
    summary = _metric_summary(clean, unit) or MetricSummary(unit=unit, count=0)
    ci = TrialConfidenceInterval(unit=unit, trial_count=len(clean))
    if len(clean) == 3 and summary.mean is not None and summary.sample_standard_deviation is not None:
        half = T_MULTIPLIER_N3 * summary.sample_standard_deviation / (3 ** 0.5)
        ci = TrialConfidenceInterval(
            unit=unit,
            trial_count=3,
            mean=summary.mean,
            sample_standard_deviation=summary.sample_standard_deviation,
            t_multiplier=T_MULTIPLIER_N3,
            half_width=half,
            lower=summary.mean - half,
            upper=summary.mean + half,
            reported=True,
        )
    return summary, ci
