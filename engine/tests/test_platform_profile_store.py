"""Platform slice 9 — the host profile / capability store. Pure tests (no torch): the profiler +
prober are injected as fakes, so caching, recalibrate-on-change, corrupt-entry tolerance, and the
skip-the-probes optimization are all provable on a core-only install."""

import corpus_studio.platform as P
from corpus_studio.platform.common import Ref
from corpus_studio.platform.contracts import CapabilityReport, EnvHost, EnvironmentProfile
from corpus_studio.platform.profile_store import (
    ResolvedEnvironment,
    default_store_dir,
    list_signatures,
    load_profile,
    load_report,
    resolve_capabilities,
    save_environment,
)


def _profile(signature):
    return EnvironmentProfile(environment_signature=signature, host=EnvHost(os="linux"))


def _report(signature, readiness="ready"):
    return CapabilityReport(
        backend_id="corpus_studio", environment_ref=Ref(id=signature), readiness=readiness
    )


# ---- persistence ------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path):
    sig = "a" * 64
    profile, report = _profile(sig), _report(sig)
    entry = save_environment(profile, report, tmp_path)
    assert entry == tmp_path / sig
    assert load_profile(sig, tmp_path) == profile
    assert load_report(sig, tmp_path) == report


def test_missing_entry_loads_none(tmp_path):
    assert load_profile("b" * 64, tmp_path) is None
    assert load_report("b" * 64, tmp_path) is None


def test_corrupt_entry_is_a_miss_not_a_crash(tmp_path):
    sig = "c" * 64
    (tmp_path / sig).mkdir()
    (tmp_path / sig / "CapabilityReport.json").write_text("{ not json", encoding="utf-8")
    assert load_report(sig, tmp_path) is None


def test_list_signatures_only_returns_complete_entries(tmp_path):
    full = "d" * 64
    save_environment(_profile(full), _report(full), tmp_path)
    # A half-written entry (profile only) must not be listed.
    partial = "e" * 64
    (tmp_path / partial).mkdir()
    (tmp_path / partial / "EnvironmentProfile.json").write_text(
        _profile(partial).model_dump_json(), encoding="utf-8"
    )
    assert list_signatures(tmp_path) == [full]


def test_list_signatures_on_missing_dir_is_empty(tmp_path):
    assert list_signatures(tmp_path / "nope") == []


# ---- resolve_capabilities (cache + recalibrate) -----------------------------


def test_first_resolve_probes_and_caches(tmp_path):
    sig = "f" * 64
    probe_calls = []

    def _build():
        return _profile(sig)

    def _probe(profile):
        probe_calls.append(profile.environment_signature)
        return _report(sig)

    resolved = resolve_capabilities(tmp_path, build_profile=_build, run_probes=_probe)
    assert isinstance(resolved, ResolvedEnvironment)
    assert resolved.cached is False
    assert probe_calls == [sig]  # probes ran once
    assert load_report(sig, tmp_path) is not None  # and were persisted


def test_second_resolve_reuses_cache_and_skips_probes(tmp_path):
    sig = "1" * 64
    save_environment(_profile(sig), _report(sig, readiness="ready"), tmp_path)
    probe_calls = []

    def _probe(profile):
        probe_calls.append(1)
        return _report(sig)

    resolved = resolve_capabilities(tmp_path, build_profile=lambda: _profile(sig), run_probes=_probe)
    assert resolved.cached is True
    assert resolved.report.readiness == "ready"
    assert probe_calls == []  # the expensive probes were NOT re-run


def test_changed_signature_recalibrates(tmp_path):
    save_environment(_profile("2" * 64), _report("2" * 64), tmp_path)
    probe_calls = []

    def _probe(profile):
        probe_calls.append(profile.environment_signature)
        return _report("3" * 64)

    # The host now reports a DIFFERENT signature → cache miss → re-probe + persist.
    resolved = resolve_capabilities(
        tmp_path, build_profile=lambda: _profile("3" * 64), run_probes=_probe
    )
    assert resolved.cached is False
    assert probe_calls == ["3" * 64]
    assert list_signatures(tmp_path) == ["2" * 64, "3" * 64]  # both retained


def test_refresh_reprobes_even_on_a_cache_hit(tmp_path):
    sig = "4" * 64
    save_environment(_profile(sig), _report(sig, readiness="not_ready"), tmp_path)
    probe_calls = []

    def _probe(profile):
        probe_calls.append(1)
        return _report(sig, readiness="ready")  # a fresh, different verdict

    resolved = resolve_capabilities(
        tmp_path, build_profile=lambda: _profile(sig), run_probes=_probe, refresh=True
    )
    assert resolved.cached is False
    assert probe_calls == [1]
    assert resolved.report.readiness == "ready"
    # The cache was updated with the fresh verdict.
    assert load_report(sig, tmp_path).readiness == "ready"


# ---- misc -------------------------------------------------------------------


def test_default_store_dir_is_under_home():
    assert default_store_dir().name == "profiles"
    assert "corpus_studio" in str(default_store_dir()).replace("\\", "/")


def test_roundtrip_via_public_models(tmp_path):
    sig = "5" * 64
    save_environment(_profile(sig), _report(sig), tmp_path)
    loaded = load_report(sig, tmp_path)
    assert P.CapabilityReport.model_validate_json(loaded.model_dump_json()) == loaded
