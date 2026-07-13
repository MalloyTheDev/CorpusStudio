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
    ParameterAccountingReport,
    ParameterEvidenceGap,
    ParameterScope,
    ParameterWindow,
)
from corpus_studio.platform.parameter_accounting import parameter_accounting_hash_for

runner = CliRunner()
_SIG = "b" * 64


def _ready_profile(cc_major: int = 8, os_name: str = "linux") -> EnvironmentProfile:
    return EnvironmentProfile(
        environment_signature=_SIG,
        host=EnvHost(os=os_name),
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


def _ready_host(monkeypatch, *, cc_major: int = 8, attn: str = "sdpa", os_name: str = "linux") -> None:
    """Inject a synthetic ready host so the planner produces a plan without torch/a GPU present.
    ``os_name`` matters on Blackwell: only native Windows forces the math mandate (WSL/Linux keep sdpa)."""
    monkeypatch.setattr(
        "corpus_studio.platform.profiler.build_environment_profile",
        lambda: _ready_profile(cc_major, os_name),
    )
    monkeypatch.setattr(
        "corpus_studio.platform.probes.run_capability_probes", lambda _profile: _ready_report(attn)
    )


def _accounting_report():
    model_ref = Ref(id="model", hash={"value": "c" * 64})
    scope = ParameterScope(
        scope_id="model",
        kind="model",
        model_ref=model_ref,
        coordinate_universe_id="model-coordinates",
        coordinate_universe_sha256="c" * 64,
        definition="One model coordinate universe.",
    )
    draft = ParameterAccountingReport(
        report_id="parameter-report",
        report_hash="0" * 64,
        generated_at="2026-07-13T00:00:00Z",
        profile="model_static",
        status="incomplete",
        model_ref=model_ref,
        gaps=[
            ParameterEvidenceGap(
                gap_id="logical-gap",
                kind="logical",
                scope=scope,
                window=ParameterWindow(
                    window_id="static-model",
                    kind="static_snapshot",
                    definition="One static snapshot.",
                ),
                reason="missing_observation",
                explanation="Logical evidence is absent.",
                resolution="Supply measured evidence.",
            )
        ],
    )
    return draft.model_copy(update={"report_hash": parameter_accounting_hash_for(draft)})


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
    physical = bundle["run_plan"]["physical_execution"]
    assert physical["evidence_status"] == "planned_not_measured"
    assert physical["parallelism"]["world_size"] == 1
    assert physical["offload_rules"] == []
    # a predicted fit rides along — and it is never the measured-only NATIVE_SAFE
    assert bundle["fit_classification"]["classification"] != "NATIVE_SAFE"


def test_platform_plan_dataset_format_flows_through_the_cli(monkeypatch):
    # A chat dataset (messages) must not be planned as instruction (Alpaca) — the snapshot's
    # dataset_format is what the trainer uses to format rows.
    _ready_host(monkeypatch)
    result = runner.invoke(
        app,
        ["platform-plan", "--base-model", "m", "--dataset", "d.jsonl", "--dataset-format", "chat",
         "--json"],
    )
    assert result.exit_code == 0
    snap = json.loads(result.stdout)["run_plan"]["training_config_snapshot"]
    assert snap["dataset_format"] == "chat"
    assert "format" not in snap  # the trainer key is dataset_format, never a silently-dropped "format"


def test_platform_plan_output_dir_flows_into_the_snapshot(monkeypatch):
    # --output-dir controls where the trainer saves the adapter (so a run can target a project dir,
    # not the CWD). It must reach the training snapshot verbatim.
    _ready_host(monkeypatch)
    result = runner.invoke(
        app,
        ["platform-plan", "--base-model", "m", "--dataset", "d.jsonl", "--output-dir",
         "/some/project/adapters/wbg", "--json"],
    )
    assert result.exit_code == 0
    snap = json.loads(result.stdout)["run_plan"]["training_config_snapshot"]
    assert snap["output_dir"] == "/some/project/adapters/wbg"


def test_platform_plan_memory_efficient_sets_the_levers(monkeypatch):
    # --memory-efficient (a tight-GPU shortcut) must seal the paged optimizer + fused Liger loss into
    # the snapshot the trainer replays, so the platform path — not just train-run — gets them.
    _ready_host(monkeypatch)
    result = runner.invoke(
        app,
        ["platform-plan", "--base-model", "m", "--dataset", "d.jsonl", "--memory-efficient", "--json"],
    )
    assert result.exit_code == 0
    snap = json.loads(result.stdout)["run_plan"]["training_config_snapshot"]
    assert snap["optim"] == "paged_adamw_8bit"
    assert snap["use_liger"] is True


def test_platform_plan_backend_flows_through_the_cli(monkeypatch):
    # A proven-sdpa host: Unsloth declares sdpa, so it can run this plan → backend_ref reflects it.
    _ready_host(monkeypatch, cc_major=8, attn="sdpa")
    result = runner.invoke(
        app,
        ["platform-plan", "--base-model", "m", "--dataset", "d.jsonl", "--backend", "unsloth", "--json"],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["run_plan"]["backend_ref"]["id"] == "unsloth"


def test_platform_plan_rejects_unsloth_on_a_native_windows_blackwell_math_plan(monkeypatch):
    # NATIVE WINDOWS + Blackwell forces math (WDDM flash deadlock); Unsloth declares no math → the
    # planner refuses through the CLI (exit 2).
    _ready_host(monkeypatch, cc_major=12, attn="sdpa", os_name="windows")
    result = runner.invoke(
        app,
        ["platform-plan", "--base-model", "m", "--dataset", "d.jsonl", "--backend", "unsloth"],
    )
    assert result.exit_code == 2
    assert "can't run this plan" in result.stderr or "can't run this plan" in result.output


def test_platform_plan_allows_unsloth_on_wsl_blackwell(monkeypatch):
    # WSL is its own platform: the flash deadlock is Windows-WDDM-only, so a WSL Blackwell host does
    # NOT force math — it seals sdpa, which Unsloth declares, so the plan is accepted (exit 0). This is
    # the whole reason to train under WSL (verified on a real 5070: flash SDPA runs on WSL2 Blackwell).
    _ready_host(monkeypatch, cc_major=12, attn="sdpa", os_name="wsl")
    result = runner.invoke(
        app,
        ["platform-plan", "--base-model", "m", "--dataset", "d.jsonl", "--backend", "unsloth", "--json"],
    )
    assert result.exit_code == 0
    bundle = json.loads(result.stdout)
    assert bundle["run_plan"]["backend_ref"]["id"] == "unsloth"
    assert bundle["run_plan"]["attention_backend"] == "sdpa"  # sdpa sealed, NOT math (Windows-only)


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


def test_platform_plan_loads_and_copies_a_hash_sealed_parameter_report(monkeypatch, tmp_path):
    _ready_host(monkeypatch)
    report = _accounting_report()
    report_path = tmp_path / "ParameterAccountingReport.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    out = tmp_path / "plan"
    result = runner.invoke(
        app,
        [
            "platform-plan",
            "--base-model",
            "m",
            "--dataset",
            "d.jsonl",
            "--parameter-accounting-report",
            str(report_path),
            "--out",
            str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    bundle = json.loads(result.stdout)
    pinned = bundle["run_plan"]["parameter_accounting_ref"]
    assert pinned["id"] == report.report_id
    assert pinned["hash"]["value"] == report.report_hash
    assert (out / "ParameterAccountingReport.json").exists()


def test_platform_plan_refuses_a_tampered_parameter_report(monkeypatch, tmp_path):
    _ready_host(monkeypatch)
    report = _accounting_report().model_copy(update={"report_hash": "0" * 64})
    path = tmp_path / "tampered.json"
    path.write_text(report.model_dump_json(), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "platform-plan",
            "--base-model",
            "m",
            "--dataset",
            "d.jsonl",
            "--parameter-accounting-report",
            str(path),
        ],
    )
    assert result.exit_code == 2
    assert "hash mismatch" in result.output


def test_platform_plan_refuses_unverified_nontrivial_physical_spec(monkeypatch, tmp_path):
    _ready_host(monkeypatch)
    spec = {
        "resources": [
            {
                "resource_id": "compute-0",
                "tier": "gpu",
                "device_kind": "cuda",
                "device_id": "cuda:0",
            },
            {
                "resource_id": "host-ram",
                "tier": "pageable_ram",
                "device_kind": "cpu",
                "device_id": "cpu:0",
            },
        ],
        "placements": [
            {
                "placement_id": "parameters-authoritative",
                "state": "parameters",
                "selector": {"whole_model": True},
                "resource_id": "compute-0",
                "role": "authoritative",
            }
        ],
        "offload_rules": [
            {
                "rule_id": "parameter-offload",
                "state": "parameters",
                "selector": {"whole_model": True},
                "source_resource_id": "compute-0",
                "target_resource_id": "host-ram",
                "mechanism": "cpu_copy",
                "trigger": "after_use",
            }
        ],
        "parallelism": {
            "world_size": 1,
            "ranks": [{"rank": 0, "resource_id": "compute-0"}],
        },
    }
    path = tmp_path / "physical.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "platform-plan",
            "--base-model",
            "m",
            "--dataset",
            "d.jsonl",
            "--physical-spec",
            str(path),
        ],
    )
    assert result.exit_code == 2
    assert "can't run the physical plan" in result.output


def test_platform_run_refuses_a_tampered_plan_hash(tmp_path):
    from corpus_studio.platform.supervisor import demo_run_plan

    body = demo_run_plan().model_dump(mode="json")
    body["seed"] += 1
    path = tmp_path / "RunPlan.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    result = runner.invoke(app, ["platform-run", str(path)])
    assert result.exit_code == 2
    assert "plan_hash does not match" in result.output
