"""The platform CLI JSON contracts — the exact stdout the Tauri shell / apps/web live flow parses.
`platform-plan` runs the real profiler+probes, so on a torch-less CI host it would refuse (not_ready);
we inject a synthetic READY host to exercise the resolve→fit→bundle path, and prove the --backend
selection flows through the CLI (incl. the honest Unsloth-on-Blackwell refusal)."""

import json

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.platform.common import Ref
from corpus_studio.platform.contracts import (
    CapabilityReport,
    EffectiveCapabilities,
    EnvironmentProfile,
    EnvHost,
    GpuDevice,
)

runner = CliRunner()
_SIG = "b" * 64


def _ready_profile(cc_major: int = 8) -> EnvironmentProfile:
    return EnvironmentProfile(
        environment_signature=_SIG,
        host=EnvHost(os="linux"),
        gpus=[
            GpuDevice(
                index=0, kind="cuda", name="Synthetic", vram_total_bytes=12_000_000_000,
                compute_capability=f"{cc_major}.0", compute_capability_major=cc_major,
            )
        ],
    )


def _ready_report(attn: str = "sdpa") -> CapabilityReport:
    return CapabilityReport(
        backend_id="corpus_studio", environment_ref=Ref(id=_SIG), readiness="ready",
        bitsandbytes_ok=True,
        effective_capabilities=EffectiveCapabilities(
            precision_modes=["bf16"], quantization_modes=["nf4"], attention_impls=[attn],
            adapter_methods=["qlora"],
        ),
    )


def _ready_host(monkeypatch, *, cc_major: int = 8, attn: str = "sdpa") -> None:
    """Inject a synthetic ready host so the planner produces a plan without torch/a GPU present."""
    monkeypatch.setattr(
        "corpus_studio.platform.profiler.build_environment_profile",
        lambda: _ready_profile(cc_major),
    )
    monkeypatch.setattr(
        "corpus_studio.platform.probes.run_capability_probes", lambda _profile: _ready_report(attn)
    )


# ---- platform-backends -------------------------------------------------------


def test_platform_backends_json_lists_the_registry():
    result = runner.invoke(app, ["platform-backends", "--json"])
    assert result.exit_code == 0
    ids = [b["backend_id"] for b in json.loads(result.stdout)]
    assert ids == ["corpus_studio", "unsloth"]


# ---- platform-probe (works torch-less: readiness present) --------------------


def test_platform_probe_json_bundles_profile_and_report():
    result = runner.invoke(app, ["platform-probe", "--json"])
    assert result.exit_code == 0
    bundle = json.loads(result.stdout)
    assert "environment_profile" in bundle
    assert "capability_report" in bundle
    assert bundle["capability_report"]["readiness"] in {"ready", "cpu_toy_only", "not_ready"}


# ---- platform-plan --json (the live-flow bundle) -----------------------------


def test_platform_plan_json_bundles_plan_and_fit(monkeypatch):
    _ready_host(monkeypatch)
    result = runner.invoke(
        app,
        ["platform-plan", "--base-model", "Qwen/Qwen2.5-7B", "--dataset", "d.jsonl", "--json"],
    )
    assert result.exit_code == 0
    bundle = json.loads(result.stdout)
    assert len(bundle["run_plan"]["plan_hash"]) == 64
    assert bundle["run_plan"]["backend_ref"]["id"] == "corpus_studio"
    # a predicted fit rides along — and it is never the measured-only NATIVE_SAFE
    assert bundle["fit_classification"]["classification"] != "NATIVE_SAFE"


def test_platform_plan_backend_flows_through_the_cli(monkeypatch):
    # A proven-sdpa host: Unsloth declares sdpa, so it can run this plan → backend_ref reflects it.
    _ready_host(monkeypatch, cc_major=8, attn="sdpa")
    result = runner.invoke(
        app,
        ["platform-plan", "--base-model", "m", "--dataset", "d.jsonl", "--backend", "unsloth", "--json"],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["run_plan"]["backend_ref"]["id"] == "unsloth"


def test_platform_plan_rejects_unsloth_on_a_blackwell_math_plan(monkeypatch):
    # Blackwell forces math; Unsloth declares no math → the planner refuses through the CLI (exit 2).
    _ready_host(monkeypatch, cc_major=12, attn="sdpa")
    result = runner.invoke(
        app,
        ["platform-plan", "--base-model", "m", "--dataset", "d.jsonl", "--backend", "unsloth"],
    )
    assert result.exit_code == 2
    assert "can't run this plan" in result.stderr or "can't run this plan" in result.output


def test_platform_plan_json_survives_stdout_noise_from_a_probe(monkeypatch):
    # A probe/import that prints a banner to STDOUT (historically older bitsandbytes) must not corrupt
    # the JSON bundle the Tauri shell parses — the CLI redirects probe-time stdout to stderr.
    def _noisy_profile() -> EnvironmentProfile:
        print("=== Welcome to bitsandbytes! (BUG REPORT banner) ===")  # noqa: T201 - simulates the lib
        return _ready_profile()

    monkeypatch.setattr("corpus_studio.platform.profiler.build_environment_profile", _noisy_profile)
    monkeypatch.setattr(
        "corpus_studio.platform.probes.run_capability_probes", lambda _p: _ready_report()
    )
    result = runner.invoke(
        app, ["platform-plan", "--base-model", "m", "--dataset", "d.jsonl", "--json"]
    )
    assert result.exit_code == 0
    bundle = json.loads(result.stdout)  # pure JSON despite the banner
    assert len(bundle["run_plan"]["plan_hash"]) == 64
    assert "bitsandbytes" in result.stderr  # the banner was redirected off stdout, onto stderr
