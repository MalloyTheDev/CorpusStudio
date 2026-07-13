"""Platform END-TO-END composition. Proves the 9 slices actually FIT TOGETHER — profile → plan →
predicted-fit → run → integrity-checked artifact, persisted — with the trainer mocked, so the whole
chain is provable on a core-only install (no GPU / no [train]). This is the regression that would
catch a break between any two slices (a snapshot key the runner drops, a plan field the calibrator
misreads, an artifact that isn't written)."""

from pathlib import Path

import corpus_studio.platform as P
from corpus_studio.platform.artifacts import recheck_artifact_integrity
from corpus_studio.platform.calibrator import classify_fit
from corpus_studio.platform.contracts import (
    CapabilityReport,
    EffectiveCapabilities,
    EnvHost,
    EnvironmentProfile,
    ExecutionCapabilityCombination,
    GpuDevice,
    ProbeResult,
)
from corpus_studio.platform.backends import get_backend
from corpus_studio.platform.common import HashRef, PackageLock, Ref
from corpus_studio.platform.enums import FitClass
from corpus_studio.platform.execution_config import stable_file_sha256
from corpus_studio.platform.planner import PlannerConstraints, build_run_plan
from corpus_studio.platform.profile_store import resolve_capabilities
from corpus_studio.platform.runners import TrainingRunner
from corpus_studio.platform.supervisor import execute_run
from corpus_studio.training.trainer import TrainResult, train_config_from_resolved

_SIG = "a" * 64


def _blackwell_profile():
    # Native Windows + Blackwell — the host whose WDDM flash deadlock forces the math mandate this
    # e2e chain asserts. (A WSL/Linux Blackwell host would instead seal sdpa; covered in the planner
    # + CLI tests.)
    return EnvironmentProfile(
        environment_signature=_SIG,
        host=EnvHost(os="windows"),
        gpus=[GpuDevice(index=0, kind="cuda", name="RTX 5070", vram_total_bytes=12_000_000_000,
                        compute_capability_major=12)],
    )


def _ready_report(*, signature=_SIG, readiness="ready", bitsandbytes=True):
    backend = get_backend("corpus_studio")
    assert backend is not None
    combination = ExecutionCapabilityCombination.model_validate(
        {
            "runtime_mode": "training" if readiness == "ready" else "cpu_toy",
            "device": "cuda" if readiness == "ready" else "cpu",
            "precision": "bf16" if readiness == "ready" else "fp32",
            "quantization": "nf4" if bitsandbytes else "none",
            "adapter_method": "qlora" if bitsandbytes else "lora",
            "attention_impl": "math" if readiness == "ready" else "eager",
            "attention_kernel": "torch_sdpa_math" if readiness == "ready" else "eager",
            "optimizer": "adamw_torch",
            "loss_impl": "cross_entropy",
            "checkpoint_impl": "adapter_only",
            "export_format": "adapter_peft",
            "execution_contract_version": "1.0.0",
            "probe": "synthetic_execution",
        }
    )
    axis_proofs = {
        "adapter": ["lora", "qlora"],
        "attention": ["eager", "math", "sdpa"],
        "attention_kernel": ["eager", "torch_sdpa_math"],
        "checkpoint": ["adapter_only"],
        "loss": ["cross_entropy"],
        "optimizer": ["adamw_torch"],
        "precision": ["bf16", "fp32"],
    }
    probe_results = [
        ProbeResult(probe="synthetic_axes", outcome="PASS", proves=axis_proofs),
        ProbeResult(
            probe="trainer_contract",
            outcome="PASS",
            proves={
                "trainer_field": backend.trainer_fields,
                "trainer_init_field": backend.trainer_init_fields,
            },
        ),
        ProbeResult(
            probe="synthetic_execution",
            outcome="PASS",
            execution_combinations=[combination],
        ),
    ]
    if bitsandbytes:
        probe_results.append(
            ProbeResult(
                probe="bnb_4bit_load", outcome="PASS", proves={"quantization": ["nf4"]}
            )
        )
    return CapabilityReport(
        backend_id="corpus_studio", backend_version=backend.backend_version,
        environment_ref=Ref(id=signature), readiness=readiness,
        bitsandbytes_ok=bitsandbytes,
        probe_results=probe_results,
        installed_packages=[
            PackageLock(name=name, version="1.0")
            for name in [
                "accelerate",
                "bitsandbytes",
                "datasets",
                "peft",
                "torch",
                "transformers",
                "trl",
            ]
        ],
        effective_capabilities=EffectiveCapabilities(
            precision_modes=["bf16", "fp32"],
            quantization_modes=["nf4"] if bitsandbytes else [],
            attention_impls=["eager", "math", "sdpa"],
            attention_kernels=["eager", "torch_sdpa_math"],
            adapter_methods=["lora", "qlora"],
            optimizers=["adamw_torch"],
            loss_impls=["cross_entropy"],
            checkpoint_impls=["adapter_only"],
            execution_contract_versions=["1.0.0"],
            execution_combinations=[combination],
            trainer_fields=backend.trainer_fields,
            trainer_init_fields=backend.trainer_init_fields,
        ),
    )


def _pinned_dataset(tmp_path, name="examples.jsonl"):
    dataset = tmp_path / name
    dataset.write_text('{"instruction":"Say hi","output":"Hi"}\n', encoding="utf-8")
    digest = stable_file_sha256(dataset)
    return dataset, digest, Ref(id=f"dataset-{name}", hash=HashRef(value=digest))


def _fake_trainer(*, steps=2):
    """A stand-in run_training that writes a real adapter dir (so integrity is checkable) and drives
    the progress callback — no torch, no model, no dataset."""

    def _run(config, *, progress_callback=None, **_kw):
        adapter_dir = Path(config.output_dir)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"trained-weights")
        for step in range(1, steps + 1):
            if progress_callback is not None:
                progress_callback(step, steps, 1.0 / step)
        return TrainResult(
            output_dir=str(adapter_dir), adapter_path=str(adapter_dir),
            base_model=config.base_model, cpu_toy=config.cpu_toy, steps=steps, final_loss=0.5,
            checkpoints=[],
        )

    return _run


def test_full_plan_run_artifact_chain_composes(tmp_path, monkeypatch):
    profile, report = _blackwell_profile(), _ready_report()
    dataset, dataset_digest, dataset_ref = _pinned_dataset(tmp_path)

    # 1. PLAN — resolve an immutable, sealed RunPlan from the host + proven capabilities.
    plan = build_run_plan(
        profile=profile, capabilities=report, dataset_ref=dataset_ref,
        constraints=PlannerConstraints(
            base_model="Qwen/Qwen2.5-7B-Instruct",
            model_revision="1" * 40,
            dataset_path=str(dataset),
            dataset_content_sha256=dataset_digest,
            sequence_len=4096,
            output_dir=str(tmp_path / "planned-output"),
        ),
        plan_id="wbg-plan",
    )
    assert plan.attention_backend.value == "math"  # Blackwell mandate held end-to-end
    assert plan.quantization.value == "nf4" and plan.precision.value == "bf16"
    assert len(plan.plan_hash) == 64 and plan.plan_hash != "0" * 64

    # 2. The independently sealed execution config maps exactly to the trainer boundary.
    assert plan.resolved_execution is not None
    assert plan.training_config_snapshot == {}
    cfg = train_config_from_resolved(plan.resolved_execution)
    assert cfg.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.dataset_format == "instruction"  # not silently defaulted from a wrong "format" key

    # 3. FIT — a predicted classification (never a proven NATIVE_SAFE from an estimate).
    fit = classify_fit(plan, profile)
    assert fit.classification != FitClass.NATIVE_SAFE
    assert fit.estimated_peak_bytes and fit.device_capacity_bytes == 12_000_000_000

    # 4. RUN — execute the plan through the supervisor + TrainingRunner (trainer mocked).
    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _fake_trainer())
    rundir = tmp_path / "run"
    result = execute_run(plan, TrainingRunner(), run_id="run-1", out_dir=rundir)

    # 5. The whole chain landed: succeeded manifest linked to the plan, streamed metrics, an
    #    integrity-checked artifact, all persisted beside each other.
    assert result.manifest.state == "succeeded"
    assert result.manifest.plan_ref.id == "wbg-plan"
    assert result.manifest.plan_ref.hash is not None
    assert result.manifest.plan_ref.hash.value == plan.plan_hash
    assert [e.event_type for e in result.events].count("metric") == 2
    expected_adapter = tmp_path / "planned-output" / "runs" / "run-1" / "artifacts" / "adapter"
    assert result.manifest.output_dir == str(expected_adapter)
    assert len(result.manifest.artifact_ids) == 1
    artifact_id = result.manifest.artifact_ids[0]
    assert artifact_id.startswith("run-1-adapter-")
    assert len(result.artifacts) == 1
    art = result.artifacts[0]
    assert art.integrity is not None and art.integrity.current_integrity == "ok"
    assert art.producer_run_ref.id == "run-1"
    record_dir = rundir / "runs" / "run-1"
    assert (record_dir / "RunManifest.json").is_file()
    assert (record_dir / "artifacts" / f"{artifact_id}.json").is_file()

    # 6. The two-tier integrity re-check confirms the persisted artifact still matches.
    assert recheck_artifact_integrity(art).integrity.current_integrity == "ok"


def test_cpu_toy_chain_composes(tmp_path, monkeypatch):
    # A cpu-toy-only host, opted in → a cpu-toy plan runs end-to-end.
    profile = EnvironmentProfile(environment_signature="b" * 64, host=EnvHost(os="windows"))
    report = _ready_report(signature="b" * 64, readiness="cpu_toy_only", bitsandbytes=False)
    dataset, dataset_digest, dataset_ref = _pinned_dataset(tmp_path, "toy.jsonl")
    plan = build_run_plan(
        profile=profile, capabilities=report, dataset_ref=dataset_ref,
        constraints=PlannerConstraints(
            base_model="hf-internal-testing/tiny-random-LlamaForCausalLM",
            model_revision="9fb191250dd56d0ba7ec9785a025ed29c03d5998",
            dataset_path=str(dataset),
            dataset_content_sha256=dataset_digest,
            allow_cpu_toy=True,
            output_dir=str(tmp_path / "toy-planned-output"),
        ),
        plan_id="toy",
    )
    assert plan.resolved_execution is not None
    assert plan.resolved_execution.runtime_mode == "cpu_toy"
    assert plan.precision.value == "fp32" and plan.quantization.value == "none"

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _fake_trainer(steps=1))
    result = execute_run(plan, TrainingRunner(cpu_toy=True), run_id="toy-1")
    assert result.manifest.state == "succeeded"
    assert result.manifest.target == "corpus_studio"


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

    dataset, dataset_digest, dataset_ref = _pinned_dataset(tmp_path, "cached.jsonl")
    plan = build_run_plan(
        profile=second.profile, capabilities=second.report, dataset_ref=dataset_ref,
        constraints=PlannerConstraints(
            base_model="Qwen/Qwen2.5-7B",
            model_revision="1" * 40,
            dataset_path=str(dataset),
            dataset_content_sha256=dataset_digest,
        ),
        plan_id="from-cache",
    )
    assert plan.environment_ref.id == profile.environment_signature
    assert P.RunPlan.model_validate_json(plan.model_dump_json()) == plan
