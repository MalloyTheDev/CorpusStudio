"""The run watchdog — the reliability layer slice 7 deferred to "a watchdog + a memory-signature
classifier". Slice 7 classifies failures that RAISE (OOM, NaN) by string-matching; this catches the
two that DON'T raise, so nothing else can see them:

* **KERNEL_STALL** — a hang. On Blackwell (sm_120) the fused flash/mem-efficient attention deadlocks
  on the first backward and the training thread simply stops making progress; there is no exception.
  The watchdog notices the heartbeat has gone silent past a timeout and reports the stall.
* **ACCIDENTAL_WDDM_SPILL** — a silent slowdown. On Windows/WDDM, once a process reserves more GPU
  memory than the physical VRAM the driver quietly pages the excess to shared system RAM (10-25×
  slowdown), *not* a clean OOM. The tell is ``torch_reserved > dedicated_vram`` → a non-zero
  ``shared_gpu_bytes``, the documented ACCIDENTAL_SPILL fingerprint. NOTE: this is a **lower bound** —
  it sees only *this process's* torch reservation crossing the physical VRAM. A spill driven by
  other-process/display VRAM (torch under dedicated while the device as a whole pages) is not caught;
  a true OS shared-memory-in-use counter would, but torch/CUDA exposes none cheaply.

Honesty on the stall: an in-process CUDA hang cannot be force-killed (the training thread is stuck in
the kernel), and it cannot be classified onto the manifest (the run never returns). So the watchdog
treats a stall as an **observability signal only** — it prints a heads-up and flags it — never an
abort and never a ``KERNEL_STALL`` manifest verdict. Detecting-and-KILLING a hang is the job of the
supervised subprocess worker (a later slice), which owns a process it can time out and terminate.

It also captures the **measured peak memory**, so a predicted fit becomes a *measured* one — the only
way a plan legitimately earns ``NATIVE_SAFE`` (an estimate never can).

Dependency-light: this module imports only the platform contracts + stdlib. The default memory
sampler lazy-imports torch INSIDE the function; tests inject a fake sampler and a fake clock, so every
classifier and the thread loop are provable with no torch and no real time.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from corpus_studio.platform.contracts import FitClassification, MemoryMetrics
from corpus_studio.platform.enums import FitClass, MemoryResidencyModel, OperatingSystem
from corpus_studio.platform.gpu_health import spill_remediation

# A sampler returns a point-in-time MemoryMetrics, or None when there's no CUDA device to read.
MemorySampler = Callable[[], "MemoryMetrics | None"]

# WDDM headroom: treat within this fraction of capacity as TIGHT rather than SAFE (a measured run that
# peaked at >90% of VRAM is real but leaves no margin for a longer sequence / a bigger batch).
_TIGHT_FRACTION = 0.90


# ------------------------------------------------------------------------------------------------
# The default memory sampler (lazy torch + nvidia-smi)
# ------------------------------------------------------------------------------------------------
def sample_gpu_memory() -> MemoryMetrics | None:
    """Sample the current CUDA memory via torch (``torch.cuda.mem_get_info`` gives the physical VRAM
    total + free). Returns ``None`` when torch/CUDA is unavailable (CPU host) — never raises. The
    **spill fingerprint** is
    computed here: ``shared_gpu_bytes = max(0, torch_reserved - dedicated_vram)`` — non-zero only when
    the process has reserved beyond physical VRAM (the WDDM silent spill)."""
    try:
        import torch  # noqa: PLC0415 - lazy; the sampler is the only torch touch-point
    except ImportError:
        return None
    if not torch.cuda.is_available():  # pragma: no cover - needs a CUDA host to exercise the else
        return None

    # pragma: no cover below — the live-GPU branch is user/GPU-verified, not runnable in CI.
    allocated = int(torch.cuda.memory_allocated())  # pragma: no cover
    reserved = int(torch.cuda.memory_reserved())  # pragma: no cover
    peak_allocated = int(torch.cuda.max_memory_allocated())  # pragma: no cover
    peak_reserved = int(torch.cuda.max_memory_reserved())  # pragma: no cover
    free_bytes, total_bytes = torch.cuda.mem_get_info()  # pragma: no cover
    dedicated = int(total_bytes)  # pragma: no cover
    used = dedicated - int(free_bytes)  # pragma: no cover
    shared = max(0, peak_reserved - dedicated)  # pragma: no cover - the spill estimate
    return MemoryMetrics(  # pragma: no cover
        torch_allocated_bytes=allocated,
        torch_reserved_bytes=reserved,
        torch_peak_allocated_bytes=peak_allocated,
        torch_peak_reserved_bytes=peak_reserved,
        cuda_device_used_bytes=used,
        cuda_device_free_bytes=int(free_bytes),
        dedicated_gpu_bytes=dedicated,
        shared_gpu_bytes=shared,
    )


# ------------------------------------------------------------------------------------------------
# Pure classifiers (no threads, no torch — fully unit-tested)
# ------------------------------------------------------------------------------------------------
def is_stalled(now_s: float, last_beat_s: float, timeout_s: float) -> bool:
    """True when no heartbeat has arrived within ``timeout_s`` — the hang signature. A non-positive
    timeout disables stall detection (returns False)."""
    if timeout_s <= 0:
        return False
    return (now_s - last_beat_s) > timeout_s


def observed_spill(sample: MemoryMetrics) -> bool:
    """True when the sample shows GPU memory spilled to shared system RAM — the documented
    ACCIDENTAL_SPILL fingerprint (``shared_gpu_bytes > 0``)."""
    return bool(sample.shared_gpu_bytes and sample.shared_gpu_bytes > 0)


def _max_opt(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def merge_peak(acc: MemoryMetrics | None, sample: MemoryMetrics) -> MemoryMetrics:
    """Element-wise max of two samples — the running peak. ``dedicated_gpu_bytes`` (physical VRAM,
    constant) is carried from whichever sample has it."""
    if acc is None:
        return sample
    return MemoryMetrics(
        torch_allocated_bytes=_max_opt(acc.torch_allocated_bytes, sample.torch_allocated_bytes),
        torch_reserved_bytes=_max_opt(acc.torch_reserved_bytes, sample.torch_reserved_bytes),
        torch_peak_allocated_bytes=_max_opt(
            acc.torch_peak_allocated_bytes, sample.torch_peak_allocated_bytes
        ),
        torch_peak_reserved_bytes=_max_opt(
            acc.torch_peak_reserved_bytes, sample.torch_peak_reserved_bytes
        ),
        cuda_device_used_bytes=_max_opt(acc.cuda_device_used_bytes, sample.cuda_device_used_bytes),
        cuda_device_free_bytes=sample.cuda_device_free_bytes,  # free shrinks; keep the latest
        dedicated_gpu_bytes=acc.dedicated_gpu_bytes or sample.dedicated_gpu_bytes,
        shared_gpu_bytes=_max_opt(acc.shared_gpu_bytes, sample.shared_gpu_bytes),
        system_ram_used_bytes=_max_opt(acc.system_ram_used_bytes, sample.system_ram_used_bytes),
        process_rss_bytes=_max_opt(acc.process_rss_bytes, sample.process_rss_bytes),
    )


def reconcile_measured_fit(
    peak: MemoryMetrics,
    *,
    residency: MemoryResidencyModel = MemoryResidencyModel.unknown,
    proven: bool = True,
    os_value: OperatingSystem = OperatingSystem.unknown,
) -> FitClassification:
    """The MEASURED fit from an observed peak — the post-run reconciliation of the calibrator's
    *predicted* fit. Unlike the estimate, a **completed** run that stayed on-device legitimately earns
    ``NATIVE_SAFE`` (tight-margin caveat → ``NATIVE_TIGHT``); an observed spill is classified by
    residency (WDDM vs unified-memory paging). ``proven=False`` (the run FAILED or was cancelled) never
    asserts a native fit from the partial peak — a run that OOM'd on a later step "peaked" only at what
    torch reserved *before* the failing alloc, so ``NATIVE_SAFE`` would be a lie — it downgrades to
    ``NATIVE_UNPROVEN``. A spill stays classified on any path: it's a fact, and it's likely what caused
    the failure."""
    peak_bytes = peak.torch_peak_reserved_bytes or peak.torch_reserved_bytes or 0
    capacity = peak.dedicated_gpu_bytes
    headroom = (capacity - peak_bytes) if capacity is not None else None

    if peak_bytes <= 0:
        # Nothing measurable was captured — do NOT fabricate NATIVE_SAFE from an empty peak (that would
        # assert a proven fit from no data). Honestly unproven.
        return FitClassification(
            classification=FitClass.NATIVE_UNPROVEN,
            estimated_peak_bytes=None,
            device_capacity_bytes=capacity,
            rationale="No usable memory sample was captured during the run — fit unproven.",
        )

    if observed_spill(peak):
        classification = (
            FitClass.ACCIDENTAL_UNIFIED_MEMORY_PAGING
            if residency == MemoryResidencyModel.unified_memory
            else FitClass.ACCIDENTAL_WDDM_SPILL
        )
        rationale = (
            f"MEASURED: the run reserved ~{_gb(peak_bytes)} GB, ~{_gb(peak.shared_gpu_bytes)} GB of it "
            "spilled to shared system RAM (not a clean OOM). "
            + spill_remediation(classification, os_value)
        )
    elif not proven:
        # The run did NOT complete — the sampled peak is only what it reserved before failing, so a
        # NATIVE_SAFE/TIGHT verdict would falsely claim "fit proven" on a run that demonstrably didn't.
        classification = FitClass.NATIVE_UNPROVEN
        rationale = (
            f"MEASURED: peaked at ~{_gb(peak_bytes)} GB before the run ended without completing — fit "
            "NOT proven (only a completed run proves a fit; a later OOM/cancel is not reflected here)."
        )
    elif capacity is not None and peak_bytes > capacity * _TIGHT_FRACTION:
        classification = FitClass.NATIVE_TIGHT
        rationale = (
            f"MEASURED: peaked at ~{_gb(peak_bytes)} GB of {_gb(capacity)} GB on-device — real, but no "
            "margin for a longer sequence or a bigger batch."
        )
    else:
        classification = FitClass.NATIVE_SAFE  # a MEASURED run earns this; an estimate never does
        rationale = (
            f"MEASURED: peaked at ~{_gb(peak_bytes)} GB on-device"
            + (f" of {_gb(capacity)} GB ({_gb(headroom)} GB free)" if capacity is not None else "")
            + " — fit proven by a real run."
        )
    return FitClassification(
        classification=classification,
        estimated_peak_bytes=peak_bytes,  # here the "estimate" IS the observed peak
        device_capacity_bytes=capacity,
        headroom_bytes=headroom,
        rationale=rationale,
    )


def _gb(b: int | None) -> str:
    return "?" if b is None else f"{b / 1_000_000_000:.1f}"


# ------------------------------------------------------------------------------------------------
# The threaded watchdog
# ------------------------------------------------------------------------------------------------
class RunWatchdog:
    """Samples memory on a background thread while a (blocking) run proceeds, tracking the peak,
    detecting a spill, and — since a true hang stops the heartbeat — detecting a stall past a timeout.

    The run beats the heart with :meth:`beat` (from the trainer's per-step progress callback); the
    thread polls independently, so it keeps sampling even while ``run_training`` blocks. It fires
    ``on_stall`` once per stall *episode* (a heads-up only — an in-process CUDA hang can't be
    force-killed or classified, that's the subprocess-worker slice — so silent death becomes a
    signal). :attr:`stalled` is the CURRENT state (cleared by a beat on recovery); :attr:`ever_stalled`
    latches whether the run went silent at least once. ``sampler`` must be side-effect-free / thread-
    safe (it's called from both the per-step callback and the daemon thread). Injectable ``sampler`` /
    ``monotonic`` make it deterministic in tests."""

    def __init__(
        self,
        *,
        sampler: MemorySampler = sample_gpu_memory,
        heartbeat_timeout_s: float = 180.0,
        poll_interval_s: float = 5.0,
        monotonic: Callable[[], float] | None = None,
        on_stall: Callable[[], None] | None = None,
    ) -> None:
        import time  # noqa: PLC0415 - only for the default monotonic clock

        self._sampler = sampler
        self._timeout = heartbeat_timeout_s
        self._interval = poll_interval_s
        self._monotonic = monotonic or time.monotonic
        self._on_stall = on_stall
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_beat = self._monotonic()
        self._peak: MemoryMetrics | None = None
        self._stalled = False  # CURRENT state — cleared by beat() on recovery
        self._ever_stalled = False  # latched — the run went silent at least once
        self._spilled = False

    def beat(self) -> None:
        """Record progress — call once per training step. Resets the stall timer AND clears any stall
        flag: a step completing means the run recovered, so a slow-load / slow-step heads-up that fired
        earlier should not linger onto the manifest. A run that ends while still stalled keeps the
        flag."""
        with self._lock:
            self._last_beat = self._monotonic()
            self._stalled = False

    def sample(self) -> None:
        """Take one memory sample + fold it into the running peak (and the spill flag). Thread-safe;
        called both from the run's per-step progress callback (deterministic per-step capture) and the
        watchdog thread (to catch a spill/peak that happens BETWEEN steps). The memory PROBE is
        best-effort observability — it must NEVER fail the run: a sampler that raises (e.g. a torch
        memory query on a faulting GPU) is swallowed and treated as "no sample", so it can't abort the
        training thread or kill the watchdog thread."""
        try:
            sample = self._sampler()
        except Exception:  # noqa: BLE001 - a probe fault is not a run failure; drop the sample
            return
        if sample is None:
            return
        with self._lock:
            self._peak = merge_peak(self._peak, sample)
            if observed_spill(sample):
                self._spilled = True

    def poll_once(self) -> None:
        """One sample + stall-check pass (the thread loop body; also directly unit-testable). The
        stall check is independent of the heartbeat, so a true hang — where the beat stops — is caught
        even though no step is completing."""
        self.sample()
        with self._lock:
            stalled_now = is_stalled(self._monotonic(), self._last_beat, self._timeout)
            fire = stalled_now and not self._stalled
            if stalled_now:
                self._stalled = True
                self._ever_stalled = True  # latched — beat() never clears this
        if fire and self._on_stall is not None:
            self._on_stall()

    def _loop(self) -> None:  # pragma: no cover - thread body exercised via poll_once in tests
        while not self._stop.wait(self._interval):
            self.poll_once()

    def start(self) -> None:
        self._last_beat = self._monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="run-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self._interval + 1.0)
        self._thread = None

    def __enter__(self) -> RunWatchdog:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    @property
    def peak(self) -> MemoryMetrics | None:
        with self._lock:
            return self._peak

    @property
    def stalled(self) -> bool:
        """The CURRENT stall state — set when the heartbeat is silent past the timeout, cleared by the
        next :meth:`beat` (recovery)."""
        with self._lock:
            return self._stalled

    @property
    def ever_stalled(self) -> bool:
        """Whether the run went silent past the timeout at least once — latched, never cleared. Used
        for the manifest warning (a run that stalled then recovered is still worth flagging)."""
        with self._lock:
            return self._ever_stalled

    @property
    def spilled(self) -> bool:
        with self._lock:
            return self._spilled

    def measured_fit(
        self,
        residency: MemoryResidencyModel = MemoryResidencyModel.unknown,
        *,
        proven: bool = True,
        os_value: OperatingSystem = OperatingSystem.unknown,
    ) -> FitClassification | None:
        """The measured fit from the tracked peak, or ``None`` if nothing was ever sampled (CPU run).
        ``proven=False`` (the run failed/cancelled) downgrades a would-be NATIVE fit to
        ``NATIVE_UNPROVEN`` — only a completed run proves a fit (a spill still classifies honestly).
        ``os_value`` tunes a spill's remediation (e.g. bare Linux would OOM where WDDM spills)."""
        peak = self.peak
        return (
            None
            if peak is None
            else reconcile_measured_fit(peak, residency=residency, proven=proven, os_value=os_value)
        )
