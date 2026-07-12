"""The run watchdog — the reliability layer for the two non-raising failure modes (a hang →
KERNEL_STALL, a silent WDDM spill) + measured-peak capture. Pure tests: a fake sampler + a fake
monotonic clock make every classifier and the poll loop deterministic with no torch and no real
time. The live-GPU sampler body is user/GPU-verified (pragma: no cover)."""

from corpus_studio.platform.contracts import MemoryMetrics
from corpus_studio.platform.enums import FitClass, MemoryResidencyModel
from corpus_studio.platform.watchdog import (
    RunWatchdog,
    is_stalled,
    merge_peak,
    observed_spill,
    reconcile_measured_fit,
    sample_gpu_memory,
)

GB = 1_000_000_000


def _mem(*, reserved=None, peak_reserved=None, dedicated=None, shared=None, used=None, free=None):
    return MemoryMetrics(
        torch_reserved_bytes=reserved,
        torch_peak_reserved_bytes=peak_reserved,
        dedicated_gpu_bytes=dedicated,
        shared_gpu_bytes=shared,
        cuda_device_used_bytes=used,
        cuda_device_free_bytes=free,
    )


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


# ---- pure classifiers --------------------------------------------------------


def test_is_stalled():
    assert is_stalled(now_s=100, last_beat_s=10, timeout_s=30) is True
    assert is_stalled(now_s=100, last_beat_s=80, timeout_s=30) is False
    assert is_stalled(now_s=100, last_beat_s=10, timeout_s=0) is False  # disabled


def test_observed_spill():
    assert observed_spill(_mem(shared=2 * GB)) is True
    assert observed_spill(_mem(shared=0)) is False
    assert observed_spill(_mem(shared=None)) is False


def test_merge_peak_is_elementwise_max():
    a = _mem(reserved=3 * GB, peak_reserved=3 * GB, dedicated=12 * GB, shared=0, free=9 * GB)
    b = _mem(reserved=5 * GB, peak_reserved=5 * GB, dedicated=12 * GB, shared=1 * GB, free=7 * GB)
    m = merge_peak(a, b)
    assert m.torch_peak_reserved_bytes == 5 * GB
    assert m.shared_gpu_bytes == 1 * GB
    assert m.dedicated_gpu_bytes == 12 * GB
    assert m.cuda_device_free_bytes == 7 * GB  # keeps the latest (free shrinks)
    assert merge_peak(None, a) is a  # first sample seeds the peak


def test_reconcile_measured_fit_earns_native_safe_from_a_measured_run():
    # peaked at 6 GB of 12 GB, no spill → NATIVE_SAFE (only a MEASURED run earns this)
    fit = reconcile_measured_fit(_mem(peak_reserved=6 * GB, dedicated=12 * GB, shared=0))
    assert fit.classification == FitClass.NATIVE_SAFE
    assert fit.estimated_peak_bytes == 6 * GB
    assert "MEASURED" in (fit.rationale or "")


def test_reconcile_measured_fit_tight_when_over_ninety_percent():
    fit = reconcile_measured_fit(_mem(peak_reserved=11 * GB, dedicated=12 * GB, shared=0))
    assert fit.classification == FitClass.NATIVE_TIGHT


def test_reconcile_measured_fit_wddm_spill():
    fit = reconcile_measured_fit(
        _mem(peak_reserved=19 * GB, dedicated=12 * GB, shared=7 * GB),
        residency=MemoryResidencyModel.wddm,
    )
    assert fit.classification == FitClass.ACCIDENTAL_WDDM_SPILL


def test_reconcile_measured_fit_unified_paging():
    fit = reconcile_measured_fit(
        _mem(peak_reserved=19 * GB, dedicated=12 * GB, shared=7 * GB),
        residency=MemoryResidencyModel.unified_memory,
    )
    assert fit.classification == FitClass.ACCIDENTAL_UNIFIED_MEMORY_PAGING


# ---- the threaded watchdog (driven via poll_once + a fake clock) --------------


def test_poll_once_tracks_peak_and_spill():
    samples = iter(
        [
            _mem(peak_reserved=3 * GB, dedicated=12 * GB, shared=0),
            _mem(peak_reserved=8 * GB, dedicated=12 * GB, shared=0),
            _mem(peak_reserved=14 * GB, dedicated=12 * GB, shared=2 * GB),  # spill
        ]
    )
    clock = _Clock()
    wd = RunWatchdog(sampler=lambda: next(samples), heartbeat_timeout_s=100, monotonic=clock)
    for _ in range(3):
        wd.beat()  # keep the heart alive so no stall fires
        wd.poll_once()
    assert wd.peak is not None and wd.peak.torch_peak_reserved_bytes == 14 * GB
    assert wd.spilled is True
    assert wd.measured_fit(MemoryResidencyModel.wddm).classification == FitClass.ACCIDENTAL_WDDM_SPILL


def test_stall_fires_on_stall_once_when_the_heartbeat_goes_silent():
    fired = []
    clock = _Clock()
    wd = RunWatchdog(
        sampler=lambda: None,  # CPU / no GPU
        heartbeat_timeout_s=10,
        monotonic=clock,
        on_stall=lambda: fired.append(True),
    )
    clock.t = 5  # within timeout
    wd.poll_once()
    assert wd.stalled is False and fired == []
    clock.t = 30  # past timeout with no beat
    wd.poll_once()
    wd.poll_once()  # a second pass must NOT re-fire on_stall
    assert wd.stalled is True
    assert fired == [True]


def test_beat_resets_the_stall_timer():
    clock = _Clock()
    wd = RunWatchdog(sampler=lambda: None, heartbeat_timeout_s=10, monotonic=clock)
    clock.t = 9
    wd.beat()  # last_beat -> 9
    clock.t = 15  # 15 - 9 = 6 < 10 → not stalled
    wd.poll_once()
    assert wd.stalled is False


def test_beat_clears_a_prior_stall_flag_on_recovery():
    # A step completing after a stall means the run recovered — the stall flag must clear so a slow-load
    # heads-up doesn't linger onto the manifest; only a run that ENDS while stalled keeps it.
    clock = _Clock()
    wd = RunWatchdog(sampler=lambda: None, heartbeat_timeout_s=10, monotonic=clock)
    clock.t = 30
    wd.poll_once()
    assert wd.stalled is True
    wd.beat()  # progress resumed
    assert wd.stalled is False
    clock.t = 45  # a NEW silent gap re-arms the stall
    wd.poll_once()
    assert wd.stalled is True


def test_reconcile_measured_fit_is_unproven_from_an_empty_peak():
    # An all-None peak must NOT fabricate NATIVE_SAFE (that would assert a proven fit from no data).
    fit = reconcile_measured_fit(MemoryMetrics())
    assert fit.classification == FitClass.NATIVE_UNPROVEN
    assert fit.estimated_peak_bytes is None


def test_measured_fit_none_when_nothing_sampled():
    wd = RunWatchdog(sampler=lambda: None, heartbeat_timeout_s=100, monotonic=_Clock())
    wd.poll_once()
    assert wd.peak is None
    assert wd.measured_fit() is None


def test_measured_fit_unproven_from_a_degenerate_sample():
    # A sampler that returns a real MemoryMetrics with all-None fields: the peak IS set (not None), but
    # measured_fit must be NATIVE_UNPROVEN — not silently None, and never a fabricated NATIVE_SAFE.
    wd = RunWatchdog(sampler=lambda: MemoryMetrics(), heartbeat_timeout_s=100, monotonic=_Clock())
    wd.poll_once()
    assert wd.peak is not None
    assert wd.measured_fit().classification == FitClass.NATIVE_UNPROVEN


def test_on_stall_refires_once_per_stall_episode():
    # on_stall fires once per stall EPISODE: a recovery (beat clears _stalled) re-arms it, so a later
    # silence fires again. ever_stalled stays latched across both.
    fired = []
    clock = _Clock()
    wd = RunWatchdog(
        sampler=lambda: None, heartbeat_timeout_s=10, monotonic=clock, on_stall=lambda: fired.append(1)
    )
    clock.t = 30
    wd.poll_once()  # episode 1 → fire
    wd.beat()  # recover (last_beat -> 30, _stalled cleared)
    clock.t = 45
    wd.poll_once()  # 45-30=15 > 10 → episode 2 → fire again
    assert fired == [1, 1]
    assert wd.ever_stalled is True


def test_reconcile_not_proven_downgrades_native_to_unproven():
    # A FAILED/cancelled run's partial peak must not claim NATIVE_SAFE — only a completed run proves a
    # fit. The SAME peak: proven=True → NATIVE_SAFE, proven=False → NATIVE_UNPROVEN (peak preserved).
    peak = _mem(peak_reserved=6 * GB, dedicated=12 * GB, shared=0)
    assert reconcile_measured_fit(peak, proven=True).classification == FitClass.NATIVE_SAFE
    downgraded = reconcile_measured_fit(peak, proven=False)
    assert downgraded.classification == FitClass.NATIVE_UNPROVEN
    assert downgraded.estimated_peak_bytes == 6 * GB  # the measured peak is still recorded


def test_reconcile_spill_stays_classified_even_when_not_proven():
    # A spill is a FACT (and likely the cause of the failure) — it classifies honestly on any path.
    fit = reconcile_measured_fit(
        _mem(peak_reserved=19 * GB, dedicated=12 * GB, shared=7 * GB), proven=False
    )
    assert fit.classification == FitClass.ACCIDENTAL_WDDM_SPILL


def test_sampler_that_raises_is_swallowed_not_propagated():
    # A probe fault must not abort the run or kill the watchdog thread — it's dropped as "no sample".
    def _boom():
        raise RuntimeError("CUDA error querying memory")

    wd = RunWatchdog(sampler=_boom, heartbeat_timeout_s=100, monotonic=_Clock())
    wd.sample()  # must not raise
    wd.poll_once()  # must not raise
    assert wd.peak is None
    assert wd.measured_fit() is None


def test_reconcile_uses_torch_reserved_when_peak_reserved_absent():
    fit = reconcile_measured_fit(_mem(reserved=5 * GB, dedicated=12 * GB, shared=0))
    assert fit.classification == FitClass.NATIVE_SAFE
    assert fit.estimated_peak_bytes == 5 * GB


def test_reconcile_native_safe_when_capacity_unknown():
    fit = reconcile_measured_fit(_mem(peak_reserved=5 * GB, dedicated=None, shared=0))
    assert fit.classification == FitClass.NATIVE_SAFE
    assert fit.headroom_bytes is None


def test_start_stop_thread_smoke():
    # A real thread pass: a fast interval + one sample, then stop cleanly.
    samples = [_mem(peak_reserved=2 * GB, dedicated=12 * GB, shared=0)]
    wd = RunWatchdog(
        sampler=lambda: samples[0], heartbeat_timeout_s=100, poll_interval_s=0.01
    )
    with wd:
        wd.beat()
    # after the context exits the thread is joined; peak may or may not be set depending on timing,
    # but stop() must not hang and the object stays usable.
    assert wd.stalled is False


def test_default_sampler_returns_none_without_torch():
    # The engine venv is torch-free → the lazy import fails → a clean None, never a raise.
    assert sample_gpu_memory() is None
