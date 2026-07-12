"""Platform END-TO-END composition. Proves the 9 slices actually FIT TOGETHER — profile → plan →
predicted-fit → run → integrity-checked artifact, persisted — with the trainer mocked, so the whole
chain is provable on a core-only install (no GPU / no [train]). This is the regression that would
catch a break between any two slices (a snapshot key the runner drops, a plan field the calibrator
misreads, an artifact that isn't written)."""

import corpus_studio.platform as P
from corpus_studio.platform.artifacts import recheck_artifact_integrity
from corpus_studio.platform.calibrator import classify_fit
from corpus_studio.platform.contracts import (
    CapabilityReport,
    EffectiveCapabilities,
    EnvHost,
    EnvironmentProfile,
    GpuDevice,
)
from corpus_studio.platform.common import Ref
from corpus_studio.platform.enums import FitClass
from corpus_studio.platform.planner import PlannerConstraints, build_run_plan
from corpus_studio.platform.profile_store import resolve_capabilities
from corpus_studio.platform.runners import TrainingRunner
from corpus_studio.platform.supervisor import execute_run
from corpus_studio.training.trainer import TrainRunConfig, TrainResult

_SIG = "a" * 64


def _blackwell_profile():
    return EnvironmentProfile(
        environment_signature=_SIG,
        host=EnvHost(os="linux"),
        gpus=[GpuDevice(index=0, kind="cuda", name="RTX 5070", vram_total_bytes=12_000_000_000,
                        compute_capability_major=12)],
    )


def _ready_report():
    return CapabilityReport(
        backend_id="corpus_studio", environment_ref=Ref(id=_SIG), readiness="ready",
        bitsandbytes_ok=True,
        effective_capabilities=EffectiveCapabilities(
            precision_modes=["bf16"], quantization_modes=["nf4"], attention_impls=["sdpa"],
            adapter_methods=["qlora"],
        ),
    )


def _fake_trainer(adapter_dir, *, steps=2):
    """A stand-in run_training that writes a real adapter dir (so integrity is checkable) and drives
    the progress callback — no torch, no model, no dataset."""

    def _run(config, *, progress_callback=None, **_kw):
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"trained-weights")
        for step in range(1, steps + 1):
            if progress_callback is not None:
                progress_callback(step, steps, 1.0 / step)
        return TrainResult(
            output_dir=str(adapter_dir.parent), adapter_path=str(adapter_dir),
            base_model=config.base_model, cpu_toy=config.cpu_toy, steps=steps, final_loss=0.5,
            checkpoints=[],
        )

    return _run


def test_full_plan_run_artifact_chain_composes(tmp_path, monkeypatch):
    profile, report = _blackwell_profile(), _ready_report()

    # 1. PLAN — resolve an immutable, sealed RunPlan from the host + proven capabilities.
    plan = build_run_plan(
        profile=profile, capabilities=report, dataset_ref=Ref(id="ds-1"),
        constraints=PlannerConstraints(
            base_model="Qwen/Qwen2.5-7B-Instruct",
            dataset_path=str(tmp_path / "examples.jsonl"),
            sequence_len=4096,
        ),
        plan_id="wbg-plan",
    )
    assert plan.attention_backend.value == "math"  # Blackwell mandate held end-to-end
    assert plan.quantization.value == "nf4" and plan.precision.value == "bf16"
    assert len(plan.plan_hash) == 64 and plan.plan_hash != "0" * 64

    # 2. The plan's snapshot must be a valid config for the trainer the runner will drive.
    cfg = TrainRunConfig.model_validate(plan.training_config_snapshot)
    assert cfg.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.dataset_format == "instruction"  # not silently defaulted from a wrong "format" key

    # 3. FIT — a predicted classification (never a proven NATIVE_SAFE from an estimate).
    fit = classify_fit(plan, profile)
    assert fit.classification != FitClass.NATIVE_SAFE
    assert fit.estimated_peak_bytes and fit.device_capacity_bytes == 12_000_000_000

    # 4. RUN — execute the plan through the supervisor + TrainingRunner (trainer mocked).
    adapter = tmp_path / "out" / "adapter"
    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _fake_trainer(adapter))
    rundir = tmp_path / "run"
    result = execute_run(plan, TrainingRunner(), run_id="run-1", out_dir=rundir)

    # 5. The whole chain landed: succeeded manifest linked to the plan, streamed metrics, an
    #    integrity-checked artifact, all persisted beside each other.
    assert result.manifest.state == "succeeded"
    assert result.manifest.plan_ref.id == "wbg-plan"
    assert result.manifest.plan_ref.hash is not None
    assert result.manifest.plan_ref.hash.value == plan.plan_hash
    assert [e.event_type for e in result.events].count("metric") == 2
    assert result.manifest.artifact_ids == ["run-1-adapter"]
    assert len(result.artifacts) == 1
    art = result.artifacts[0]
    assert art.integrity is not None and art.integrity.current_integrity == "ok"
    assert art.producer_run_ref.id == "run-1"
    assert (rundir / "RunManifest.json").is_file()
    assert (rundir / "artifacts" / "run-1-adapter.json").is_file()

    # 6. The two-tier integrity re-check confirms the persisted artifact still matches.
    assert recheck_artifact_integrity(art).integrity.current_integrity == "ok"


def test_cpu_toy_chain_composes(tmp_path, monkeypatch):
    # A cpu-toy-only host, opted in → a cpu-toy plan runs end-to-end.
    profile = EnvironmentProfile(environment_signature="b" * 64, host=EnvHost(os="windows"))
    report = CapabilityReport(
        backend_id="corpus_studio", environment_ref=Ref(id="b" * 64), readiness="cpu_toy_only"
    )
    plan = build_run_plan(
        profile=profile, capabilities=report, dataset_ref=Ref(id="ds"),
        constraints=PlannerConstraints(
            base_model="hf-internal-testing/tiny-random-gpt2", dataset_path="d.jsonl",
            allow_cpu_toy=True,
        ),
        plan_id="toy",
    )
    assert plan.training_config_snapshot["cpu_toy"] is True
    assert plan.precision.value == "fp32" and plan.quantization.value == "none"

    adapter = tmp_path / "toy-out"
    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _fake_trainer(adapter, steps=1))
    result = execute_run(plan, TrainingRunner(cpu_toy=True), run_id="toy-1")
    assert result.manifest.state == "succeeded"
    assert result.manifest.target == "cpu_toy"


def test_profile_store_feeds_the_planner(tmp_path):
    # The cache resolves the capabilities (skipping probes on a hit); the planner builds against them.
    profile, report = _blackwell_profile(), _ready_report()
    probe_calls = []

    def _probe(_):
        probe_calls.append(1)
        return report

    first = resolve_capabilities(tmp_path, build_profile=_blackwell_profile, run_probes=_probe)
    second = resolve_capabilities(tmp_path, build_profile=_blackwell_profile, run_probes=_probe)
    assert first.cached is False and second.cached is True
    assert probe_calls == [1]  # the expensive probes ran once, then the cache served

    plan = build_run_plan(
        profile=second.profile, capabilities=second.report, dataset_ref=Ref(id="ds"),
        constraints=PlannerConstraints(base_model="Qwen/Qwen2.5-7B", dataset_path="d.jsonl"),
        plan_id="from-cache",
    )
    assert plan.environment_ref.id == profile.environment_signature
    assert P.RunPlan.model_validate_json(plan.model_dump_json()) == plan
