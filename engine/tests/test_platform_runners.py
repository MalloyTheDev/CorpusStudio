"""Platform slice 4 — the TrainingRunner that plugs training.trainer.run_training into the run
supervisor. Pure tests (no torch): ``run_training`` is monkeypatched, so the progress→RunEvent
adaptation, the produced-artifact handoff, and the failure/cancel classification are all provable on
a core-only install. The real training path is user-smoke-tested (a GPU + the [train] extra)."""

from pathlib import Path

import pytest

import corpus_studio.platform as P
from corpus_studio.platform.enums import FailureTaxonomy, StageMarker
from corpus_studio.platform.execution_config import execution_configuration_hash_for
from corpus_studio.platform.planner import compute_plan_hash, run_plan_hash_payload
from corpus_studio.platform.runners import (
    TrainingRunner,
    classify_training_error,
    demo_training_plan,
)
from corpus_studio.platform.supervisor import execute_run
from corpus_studio.training.trainer import (
    ExecutionPlacementDeviation,
    TrainerError,
    TrainResult,
)

_CLOCK = lambda: "2026-07-11T00:00:00+00:00"  # noqa: E731


@pytest.fixture(autouse=True)
def _isolate_relative_training_outputs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _write_fake_adapter(path: str) -> str:
    adapter = Path(path)
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"fake-adapter-weights")
    return str(adapter)


def _reseal(body: dict) -> P.RunPlan:
    """Build a valid test plan after intentionally changing a field covered by its seal."""

    draft = P.RunPlan.model_validate(body)
    return draft.model_copy(update={"plan_hash": compute_plan_hash(run_plan_hash_payload(draft))})


def _fake_run_training(steps, *, loss_by_step=None, capture=None, checkpoints=False):
    """Build a stand-in for run_training that drives the progress callback `steps` times then returns
    a TrainResult — no torch, no model, no dataset."""

    def _run(config, *, progress_callback=None, **_kw):
        if capture is not None:
            capture["config"] = config
        for step in range(1, steps + 1):
            if progress_callback is not None:
                loss = (loss_by_step or {}).get(step)
                progress_callback(step, steps, loss)
        adapter_path = _write_fake_adapter(config.output_dir)
        return TrainResult(
            output_dir=config.output_dir,
            adapter_path=adapter_path,
            base_model=config.base_model,
            cpu_toy=config.cpu_toy,
            steps=steps,
            final_loss=(loss_by_step or {}).get(steps),
            checkpoints=[f"{config.output_dir}/checkpoint-{steps}"] if checkpoints else [],
        )

    return _run


# ---- demo plan ---------------------------------------------------------------


def test_demo_training_plan_is_valid_and_carries_resolved_execution():
    plan = demo_training_plan()
    assert P.RunPlan.model_validate_json(plan.model_dump_json()) == plan
    assert plan.resolved_execution is not None
    assert plan.resolved_execution.inputs.model.location.startswith("hf-internal-testing")
    assert plan.training_config_snapshot == {}


def test_training_runner_leaves_dataset_hash_and_capture_to_the_trainer(monkeypatch):
    plan = demo_training_plan()

    def unexpected_dataset_read(_path):
        raise AssertionError("runner must not rehash the trainer-owned dataset")

    monkeypatch.setattr(
        "corpus_studio.platform.execution_config.stable_file_sha256",
        unexpected_dataset_read,
    )
    monkeypatch.setattr(
        "corpus_studio.training.trainer.run_training",
        _fake_run_training(1),
    )

    result = execute_run(plan, TrainingRunner(cpu_toy=True), clock=_CLOCK)

    assert result.manifest.state == "succeeded"


def test_training_runner_starts_preflight_before_input_revalidation(monkeypatch):
    from corpus_studio.platform.execution_config import ExecutionConfigurationError

    def fail_validation(_execution):
        raise ExecutionConfigurationError("synthetic input drift")

    monkeypatch.setattr(
        "corpus_studio.platform.execution_config.verify_execution_non_dataset_inputs",
        fail_validation,
    )

    result = execute_run(demo_training_plan(), TrainingRunner(cpu_toy=True), clock=_CLOCK)

    assert result.manifest.state == "failed"
    assert result.events[0].event_type == "stage"
    assert result.events[0].stage == StageMarker.process_start
    assert "validating sealed execution inputs" in (result.events[0].message or "")


def test_resolved_training_plan_cannot_succeed_through_echo_runner():
    from corpus_studio.platform.supervisor import EchoRunner

    result = execute_run(demo_training_plan(), EchoRunner(), clock=_CLOCK)
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert "sealed lane" in result.manifest.failure.message


def test_resolved_training_plan_refuses_a_training_runner_subclass():
    class _ImpostorTrainingRunner(TrainingRunner):
        pass

    result = execute_run(
        demo_training_plan(), _ImpostorTrainingRunner(cpu_toy=True), clock=_CLOCK
    )

    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert "first-party TrainingRunner adapter" in result.manifest.failure.message


# ---- success path (mocked trainer) -------------------------------------------


def test_training_runner_success_adapts_progress_and_produces_the_adapter(monkeypatch):
    capture: dict = {}
    monkeypatch.setattr(
        "corpus_studio.training.trainer.run_training",
        _fake_run_training(3, loss_by_step={1: 0.9, 2: 0.5, 3: 0.3}, capture=capture),
    )
    result = execute_run(
        demo_training_plan(), TrainingRunner(cpu_toy=True), run_id="run-1", clock=_CLOCK
    )

    assert result.manifest.state == "succeeded"
    assert result.manifest.target == "corpus_studio"
    # The sealed cpu_toy mode flowed into the trainer config unchanged.
    assert capture["config"].cpu_toy is True

    metrics = [e for e in result.events if e.event_type == "metric"]
    assert [m.optimizer_step for m in metrics] == [1, 2, 3]
    assert metrics[0].metrics is not None
    assert metrics[0].metrics.loss == 0.9

    # The saved adapter is recorded as a produced artifact + an artifact_produced event.
    from corpus_studio.training.artifact_registry import compute_content_hash

    adapter_path = Path("output") / "runs" / "run-1" / "artifacts" / "adapter"
    content_hash = compute_content_hash(str(adapter_path))
    assert content_hash is not None
    artifact_id = f"run-1-adapter-{content_hash[:12]}"
    assert result.manifest.artifact_ids == [artifact_id]
    assert result.manifest.output_dir == str(adapter_path)
    produced = next(e for e in result.events if e.event_type == "artifact_produced")
    assert produced.payload == {
        "artifact_id": artifact_id,
        "kind": "adapter",
        "path": str(adapter_path),
    }
    assert not any("checkpoint-" in (e.message or "") for e in result.events)


def test_training_runner_refuses_unexpected_intermediate_checkpoint_output(monkeypatch):
    monkeypatch.setattr(
        "corpus_studio.training.trainer.run_training",
        _fake_run_training(1, checkpoints=True),
    )
    result = execute_run(
        demo_training_plan(), TrainingRunner(cpu_toy=True), run_id="run-checkpoint", clock=_CLOCK
    )

    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.CHECKPOINT_FAILURE
    assert "disabled save policy" in result.manifest.failure.message
    assert result.manifest.artifact_ids == []


def test_training_runner_refuses_a_readable_adapter_outside_the_run_scope(monkeypatch):
    def _rogue(config, *, progress_callback=None, **_kw):
        if progress_callback is not None:
            progress_callback(1, 1, 0.1)
        rogue = _write_fake_adapter(str(Path(config.output_dir).parent / "rogue-adapter"))
        return TrainResult(
            output_dir=rogue,
            adapter_path=rogue,
            base_model=config.base_model,
            cpu_toy=config.cpu_toy,
            steps=1,
        )

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _rogue)
    result = execute_run(
        demo_training_plan(), TrainingRunner(cpu_toy=True), run_id="run-rogue", clock=_CLOCK
    )

    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.CHECKPOINT_FAILURE
    assert "run-scoped" in result.manifest.failure.message
    assert result.manifest.artifact_ids == []


def test_training_runner_refuses_descriptor_only_adapter_output(monkeypatch):
    def _descriptor_only(config, *, progress_callback=None, **_kw):
        if progress_callback is not None:
            progress_callback(1, 1, 0.1)
        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
        return TrainResult(
            output_dir=str(output),
            adapter_path=str(output),
            base_model=config.base_model,
            cpu_toy=config.cpu_toy,
            steps=1,
        )

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _descriptor_only)
    result = execute_run(
        demo_training_plan(), TrainingRunner(cpu_toy=True), run_id="run-no-weights", clock=_CLOCK
    )

    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.CHECKPOINT_FAILURE
    assert "weight bytes" in result.manifest.failure.message
    assert result.manifest.artifact_ids == []


def test_max_steps_override_is_refused_without_calling_the_trainer(monkeypatch):
    capture: dict = {}
    monkeypatch.setattr(
        "corpus_studio.training.trainer.run_training", _fake_run_training(1, capture=capture)
    )
    result = execute_run(
        demo_training_plan(), TrainingRunner(cpu_toy=True, max_steps=5), clock=_CLOCK
    )
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert "cannot override" in result.manifest.failure.message
    assert "config" not in capture


def test_training_runner_refuses_legacy_sealed_step_checkpoint_plan(monkeypatch):
    called = []
    monkeypatch.setattr(
        "corpus_studio.training.trainer.run_training",
        lambda *_args, **_kwargs: called.append(True),
    )
    plan = demo_training_plan()
    execution_body = plan.resolved_execution.model_dump(mode="json")
    execution_body["configuration_hash"] = "0" * 64
    execution_body["save_strategy"] = "steps"
    execution_body["checkpoint_policy"]["cadence_optimizer_steps"] = 1
    execution_body["checkpoint_policy"]["keep_last"] = 1
    execution = P.ResolvedExecutionConfiguration.model_validate(execution_body)
    execution = execution.model_copy(
        update={"configuration_hash": execution_configuration_hash_for(execution)}
    )
    body = plan.model_dump(mode="json")
    body["resolved_execution"] = execution.model_dump(mode="json")
    body["checkpoint_policy"] = execution.checkpoint_policy.model_dump(mode="json")

    result = execute_run(_reseal(body), TrainingRunner(cpu_toy=True), clock=_CLOCK)

    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert result.manifest.failure.stage == StageMarker.process_start
    assert "resume compatibility" in result.manifest.failure.message
    assert called == []


def test_training_runner_rejects_unvalidated_disabled_policy_fields(monkeypatch):
    called = []
    monkeypatch.setattr(
        "corpus_studio.training.trainer.run_training",
        lambda *_args, **_kwargs: called.append(True),
    )
    plan = demo_training_plan()
    execution = plan.resolved_execution
    assert execution is not None
    policy = execution.checkpoint_policy.model_copy(update={"keep_last": 1})
    execution = execution.model_copy(
        update={"checkpoint_policy": policy, "configuration_hash": "0" * 64}
    )
    execution = execution.model_copy(
        update={"configuration_hash": execution_configuration_hash_for(execution)}
    )
    tampered = plan.model_copy(
        update={"checkpoint_policy": policy, "resolved_execution": execution}
    )
    tampered = tampered.model_copy(
        update={"plan_hash": compute_plan_hash(run_plan_hash_payload(tampered))}
    )

    result = execute_run(tampered, TrainingRunner(cpu_toy=True), clock=_CLOCK)

    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert "resume compatibility" in result.manifest.failure.message
    assert called == []


@pytest.mark.parametrize(
    "save_strategy, cadence, keep_last, message",
    [
        ("no", 1, None, "disabled checkpointing"),
        ("no", None, 1, "disabled checkpointing"),
        ("steps", None, None, "requires an optimizer-step cadence"),
    ],
)
def test_resolved_execution_rejects_inconsistent_checkpoint_policy(
    save_strategy, cadence, keep_last, message
):
    plan = demo_training_plan()
    execution = plan.resolved_execution
    assert execution is not None
    body = execution.model_dump(mode="json")
    body["save_strategy"] = save_strategy
    body["checkpoint_policy"]["cadence_optimizer_steps"] = cadence
    body["checkpoint_policy"]["keep_last"] = keep_last

    with pytest.raises(ValueError, match=message):
        P.ResolvedExecutionConfiguration.model_validate(body)


def test_training_runner_name_reflects_cpu_toy_flag():
    assert TrainingRunner(cpu_toy=True).name == "cpu_toy"
    assert TrainingRunner().name == "training"


def test_training_runner_refuses_a_physical_plan_it_cannot_consume(monkeypatch):
    called = []
    monkeypatch.setattr(
        "corpus_studio.training.trainer.run_training",
        lambda *_args, **_kwargs: called.append(True),
    )
    body = demo_training_plan().model_dump(mode="json")
    body["offload_strategy"] = "controlled_parameter_offload"
    body["physical_execution"] = {
        "resources": [
            {
                "resource_id": "compute-0",
                "tier": "pageable_ram",
                "device_kind": "cpu",
                "device_id": "cpu:0",
            },
            {
                "resource_id": "host-ram",
                "tier": "pinned_ram",
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
    result = execute_run(_reseal(body), TrainingRunner(cpu_toy=True), clock=_CLOCK)
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert "cannot consume" in result.manifest.failure.message
    assert called == []


# ---- backend and lane identity ----------------------------------------------


def test_runplan_rejects_backend_identity_drift():
    body = demo_training_plan().model_dump(mode="json")
    body["backend_ref"] = {"id": "unsloth", "hash": {"value": "f" * 64}}
    with pytest.raises(ValueError, match="backend_ref must match"):
        P.RunPlan.model_validate(body)


def test_cpu_toy_lane_cannot_override_a_training_mode(monkeypatch):
    called = []
    monkeypatch.setattr(
        "corpus_studio.training.trainer.run_training",
        lambda *_args, **_kwargs: called.append(True),
    )
    plan = demo_training_plan()
    execution = plan.resolved_execution
    assert execution is not None
    changed = execution.model_copy(update={"runtime_mode": "training"})
    from corpus_studio.platform.execution_config import execution_configuration_hash_for

    changed = changed.model_copy(update={"configuration_hash": execution_configuration_hash_for(changed)})
    tampered = plan.model_copy(update={"resolved_execution": changed})
    tampered = tampered.model_copy(
        update={"plan_hash": compute_plan_hash(run_plan_hash_payload(tampered))}
    )
    result = execute_run(tampered, TrainingRunner(cpu_toy=True), clock=_CLOCK)
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert called == []


# ---- the run watchdog (measured fit / spill / stall) -------------------------

GB = 1_000_000_000


def _sample(*, peak_reserved, dedicated=12 * GB, shared=0):
    from corpus_studio.platform.contracts import MemoryMetrics

    return MemoryMetrics(
        torch_peak_reserved_bytes=peak_reserved, dedicated_gpu_bytes=dedicated, shared_gpu_bytes=shared
    )


def test_runner_streams_setup_stage_events_from_the_trainer(monkeypatch):
    # A trainer that reports setup milestones → the runner emits them as `stage` RunEvents. Over the
    # worker pipe these reset the subprocess supervisor's silence timer during the silent model-load —
    # real progress, the honest alternative to a liveness heartbeat.
    def _trainer_with_stages(config, *, progress_callback=None, stage_callback=None, **_kw):
        if stage_callback is not None:
            stage_callback("model_loaded", "loaded the base model")
            stage_callback("adapter_attached", "LoRA attached")
            stage_callback("optimizer_created", "trainer ready")
        if progress_callback is not None:
            progress_callback(1, 1, 0.5)
        adapter_path = _write_fake_adapter(config.output_dir)
        return TrainResult(
            output_dir=config.output_dir,
            adapter_path=adapter_path,
            base_model=config.base_model,
            cpu_toy=True,
            steps=1,
        )

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _trainer_with_stages)
    result = execute_run(demo_training_plan(), TrainingRunner(cpu_toy=True), clock=_CLOCK)
    stages = [e.stage.value for e in result.events if e.event_type == "stage" and e.stage is not None]
    assert "model_loaded" in stages
    assert "adapter_attached" in stages
    assert "optimizer_created" in stages


def test_runner_records_the_measured_fit_from_the_watchdog(monkeypatch):
    # The per-step progress callback samples memory; the observed peak reconciles to a MEASURED fit on
    # the manifest — a run that stayed on-device earns NATIVE_SAFE (an estimate never does).
    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _fake_run_training(3))
    runner = TrainingRunner(cpu_toy=True, memory_sampler=lambda: _sample(peak_reserved=6 * GB))
    result = execute_run(demo_training_plan(), runner, clock=_CLOCK)
    assert result.manifest.state == "succeeded"
    assert result.manifest.final_fit is not None
    assert result.manifest.final_fit.classification.value == "NATIVE_SAFE"
    assert result.manifest.final_fit.estimated_peak_bytes == 6 * GB


def test_runner_warns_on_a_measured_spill(monkeypatch):
    # A sample showing memory spilled to shared RAM → a warning event + an ACCIDENTAL_WDDM_SPILL fit.
    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _fake_run_training(2))
    runner = TrainingRunner(
        cpu_toy=True, memory_sampler=lambda: _sample(peak_reserved=19 * GB, shared=7 * GB)
    )
    result = execute_run(demo_training_plan(), runner, clock=_CLOCK)
    assert result.manifest.final_fit is not None
    assert result.manifest.final_fit.classification.value == "ACCIDENTAL_WDDM_SPILL"
    assert any(
        e.event_type == "warning" and "spill" in (e.message or "").lower() for e in result.events
    )


def test_runner_does_not_abort_on_a_stall_and_warns_instead(monkeypatch):
    # A detected stall is an OBSERVABILITY SIGNAL, never an abort (an in-process CUDA hang can't be
    # force-killed or classified — that's the subprocess-worker slice). A run that goes silent past the
    # timeout must still SUCCEED (no false KERNEL_STALL abort) and only carry a warning. Uses a real
    # (short) thread: the mocked trainer beats once, then goes silent past the tiny timeout, then
    # returns WITHOUT another beat, so watchdog.stalled is still set at the end.
    import time

    def _stalling_trainer(config, *, progress_callback=None, **_kw):
        if progress_callback is not None:
            progress_callback(1, 3, 0.9)  # one beat, then go silent
        time.sleep(0.4)  # > heartbeat_timeout_s; the watchdog thread trips (heads-up only, no cancel)
        adapter_path = _write_fake_adapter(config.output_dir)
        return TrainResult(
            output_dir=config.output_dir,
            adapter_path=adapter_path,
            base_model=config.base_model,
            cpu_toy=True,
            steps=3,
        )

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _stalling_trainer)
    runner = TrainingRunner(
        cpu_toy=True, memory_sampler=lambda: None, heartbeat_timeout_s=0.1, poll_interval_s=0.02
    )
    result = execute_run(demo_training_plan(), runner, clock=_CLOCK)
    assert result.manifest.state == "succeeded"  # NOT aborted / KERNEL_STALL
    assert result.manifest.failure is None
    assert any(
        e.event_type == "warning" and "without progress" in (e.message or "") for e in result.events
    )


def test_runner_records_the_measured_fit_even_on_failure(monkeypatch):
    # The watchdog samples per step; a run that trains then FAILS must still record the measured peak
    # (the richest diagnostic) — captured in a finally, on every terminal path, not only on success.
    def _sample_then_boom(config, *, progress_callback=None, **_kw):
        if progress_callback is not None:
            progress_callback(1, 2, 0.9)  # a step happens → the watchdog samples a real peak
        raise RuntimeError("kaboom mid-training")

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _sample_then_boom)
    runner = TrainingRunner(cpu_toy=True, memory_sampler=lambda: _sample(peak_reserved=6 * GB))
    result = execute_run(demo_training_plan(), runner, clock=_CLOCK)
    assert result.manifest.state == "failed"
    assert result.manifest.final_fit is not None  # the measured peak survived the failure
    assert result.manifest.final_fit.estimated_peak_bytes == 6 * GB
    # HONESTY: a FAILED run must NOT be stamped NATIVE_SAFE "fit proven" from its partial peak — the
    # run didn't complete, so the fit is UNPROVEN (a spill would still classify; this one didn't spill).
    assert result.manifest.final_fit.classification.value == "NATIVE_UNPROVEN"


def test_runner_records_a_spill_even_on_failure(monkeypatch):
    # The richest diagnostic: a run that SPILLED then failed. The finally must record the spill fit +
    # emit the spill warning even though the run raised (this is what the finally exists for).
    def _spill_then_boom(config, *, progress_callback=None, **_kw):
        if progress_callback is not None:
            progress_callback(1, 2, 0.9)  # samples a spilling peak
        raise RuntimeError("OOM after the spill")

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _spill_then_boom)
    runner = TrainingRunner(
        cpu_toy=True, memory_sampler=lambda: _sample(peak_reserved=19 * GB, shared=7 * GB)
    )
    result = execute_run(demo_training_plan(), runner, clock=_CLOCK)
    assert result.manifest.state == "failed"
    assert result.manifest.final_fit is not None
    assert result.manifest.final_fit.classification.value == "ACCIDENTAL_WDDM_SPILL"
    assert result.manifest.final_fit.estimated_peak_bytes == 19 * GB
    assert any(
        e.event_type == "warning" and "spill" in (e.message or "").lower() for e in result.events
    )


def test_runner_records_the_fit_on_cancel_after_a_step(monkeypatch):
    # A run that completes a step (real peak sampled) and is THEN cancelled must still record the
    # measured fit — the finally captures it on the cancel path, not just success.
    from corpus_studio.platform.supervisor import CancelToken

    token = CancelToken()

    def _cancel_after_step1(config, *, progress_callback=None, **_kw):
        progress_callback(1, 2, 0.9)  # a step completes → the watchdog samples a 6 GB peak
        token.cancel()  # user cancels between steps
        progress_callback(2, 2, 0.8)  # observes the cancel → _CancelTraining
        return TrainResult(
            output_dir="o", adapter_path="o", base_model=config.base_model, cpu_toy=True, steps=2
        )

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _cancel_after_step1)
    runner = TrainingRunner(cpu_toy=True, memory_sampler=lambda: _sample(peak_reserved=6 * GB))
    result = execute_run(demo_training_plan(), runner, cancel=token, clock=_CLOCK)
    assert result.manifest.state == "cancelled"
    assert result.manifest.failure is None
    assert result.manifest.final_fit is not None  # measured peak survived the cancel
    assert result.manifest.final_fit.estimated_peak_bytes == 6 * GB
    assert result.manifest.final_fit.classification.value == "NATIVE_UNPROVEN"  # not "proven"


def test_runner_survives_a_raising_memory_sampler(monkeypatch):
    # The memory probe is best-effort observability — a sampler that RAISES (e.g. a torch memory query
    # on a faulting GPU) must NEVER abort an otherwise-successful run.
    def _boom_sampler():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _fake_run_training(2))
    result = execute_run(
        demo_training_plan(), TrainingRunner(cpu_toy=True, memory_sampler=_boom_sampler), clock=_CLOCK
    )
    assert result.manifest.state == "succeeded"  # the probe fault didn't fail the run
    assert result.manifest.final_fit is None  # nothing usable was sampled


# ---- failure + cancel classification -----------------------------------------


def test_missing_runtime_is_environment_failure(monkeypatch):
    def _raise(config, *, progress_callback=None, **_kw):
        raise TrainerError("CPU toy training needs torch + transformers + …")

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _raise)
    result = execute_run(demo_training_plan(), TrainingRunner(cpu_toy=True), clock=_CLOCK)

    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.ENVIRONMENT_FAILURE
    assert result.manifest.failure.stage == StageMarker.env_loaded
    assert "train-check" in (result.manifest.failure.remediation or "")


def test_placement_deviation_is_structured_and_fails_closed(monkeypatch):
    def _raise(config, *, stage_callback=None, **_kw):
        if stage_callback is not None:
            stage_callback("placement_deviation", "requested CPU, observed CUDA")
        raise ExecutionPlacementDeviation(
            "PLACEMENT_DEVIATION: requested CPU, observed CUDA"
        )

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _raise)
    result = execute_run(demo_training_plan(), TrainingRunner(cpu_toy=True), clock=_CLOCK)
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert result.manifest.failure.stage == StageMarker.placement_deviation
    assert any(event.stage == StageMarker.placement_deviation for event in result.events)


def test_cancellation_during_training_yields_cancelled(monkeypatch):
    from corpus_studio.platform.supervisor import CancelToken

    token = CancelToken()
    token.cancel()  # pre-cancelled: the first progress callback aborts the run
    monkeypatch.setattr(
        "corpus_studio.training.trainer.run_training", _fake_run_training(3)
    )
    result = execute_run(
        demo_training_plan(), TrainingRunner(cpu_toy=True), cancel=token, clock=_CLOCK
    )

    assert result.manifest.state == "cancelled"
    assert result.manifest.failure is None


# ---- fine-grained failure classification ------------------------------------


class OutOfMemoryError(Exception):
    """Stands in for torch.cuda.OutOfMemoryError — the classifier matches by type NAME, so no torch
    is needed and the message can be empty (torch's OOM often carries the signal only in the type)."""


def test_classify_wedged_gpu_recommends_a_reset():
    # 'device not ready' (the WSL2 GPU-PV wedge from a prior crashed run) is NOT a config bug — the
    # classifier must surface a RESET, not a generic FAIL, or the operator burns runs re-running it.
    taxonomy, remediation = classify_training_error(
        RuntimeError("CUDA error: device not ready\nCUDA kernel errors might be reported asynchronously")
    )
    assert taxonomy == FailureTaxonomy.ENVIRONMENT_FAILURE
    assert remediation and "wsl --terminate" in remediation and "NOT a config problem" in remediation


def test_classify_wedged_beats_a_coincidental_oom_word():
    # A wedged GPU is checked first; an unrelated 'out of memory' phrasing shouldn't hide the wedge.
    taxonomy, _ = classify_training_error(RuntimeError("device not ready (was out of memory earlier)"))
    assert taxonomy == FailureTaxonomy.ENVIRONMENT_FAILURE


def test_classify_oom_from_message():
    taxonomy, remediation = classify_training_error(
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    )
    assert taxonomy == FailureTaxonomy.OOM
    assert remediation and "sequence_len" in remediation


def test_classify_oom_from_exception_type_name():
    assert classify_training_error(OutOfMemoryError(""))[0] == FailureTaxonomy.OOM


def test_classify_numerical_failure():
    assert classify_training_error(ValueError("Loss is nan"))[0] == FailureTaxonomy.NUMERICAL_FAILURE
    assert classify_training_error(RuntimeError("gradients contain inf"))[0] == (
        FailureTaxonomy.NUMERICAL_FAILURE
    )


def test_classify_unrecognized_stays_fail():
    taxonomy, remediation = classify_training_error(ValueError("something unexpected"))
    assert taxonomy == FailureTaxonomy.FAIL
    assert remediation is None
    # 'information' contains 'inf' but must not be mislabeled numerical (no loss/grad co-signal).
    assert classify_training_error(RuntimeError("missing information"))[0] == FailureTaxonomy.FAIL


def test_grad_information_message_is_not_numerical():
    # "gradient" (grad co-signal) + "information" (contains 'inf') co-occur, but there is no whole-word
    # NaN/Inf numeric signal → must stay FAIL, not be promoted to NUMERICAL_FAILURE.
    assert classify_training_error(
        RuntimeError("failed to gather gradient information from rank 0")
    )[0] == FailureTaxonomy.FAIL
    assert classify_training_error(
        RuntimeError("reinforcement gradient step failed")
    )[0] == FailureTaxonomy.FAIL


def test_runner_oom_is_classified_as_oom(monkeypatch):
    def _oom(config, *, progress_callback=None, **_kw):
        raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB")

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _oom)
    result = execute_run(demo_training_plan(), TrainingRunner(cpu_toy=True), clock=_CLOCK)
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.OOM
    assert "micro_batch_size" in (result.manifest.failure.remediation or "")


def test_runner_unrecognized_error_stays_fail(monkeypatch):
    def _boom(config, *, progress_callback=None, **_kw):
        raise RuntimeError("an unexpected internal error")

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _boom)
    result = execute_run(demo_training_plan(), TrainingRunner(cpu_toy=True), clock=_CLOCK)
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.FAIL


def test_missing_resolved_execution_is_unsupported_configuration():
    # The plain echo demo plan has no training execution contract.
    from corpus_studio.platform.supervisor import demo_run_plan

    result = execute_run(demo_run_plan(), TrainingRunner(cpu_toy=True), clock=_CLOCK)
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION


def test_stale_inner_execution_hash_is_unsupported_configuration():
    plan = demo_training_plan()
    execution = plan.resolved_execution
    assert execution is not None
    stale = execution.model_copy(update={"seed": execution.seed + 1})
    tampered = plan.model_copy(update={"resolved_execution": stale})
    # Reseal only the outer plan to prove the independent inner seal is checked at execution.
    tampered = tampered.model_copy(
        update={"plan_hash": compute_plan_hash(run_plan_hash_payload(tampered))}
    )
    result = execute_run(tampered, TrainingRunner(cpu_toy=True), clock=_CLOCK)
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION


def test_stale_objective_hash_is_refused_before_trainer_invocation(monkeypatch):
    from corpus_studio.platform.execution_config import execution_configuration_hash_for

    called = False

    def _trainer(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("trainer must not be called")

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _trainer)
    plan = demo_training_plan()
    execution = plan.resolved_execution
    assert execution is not None
    stale_objective = execution.objective_ref.model_copy(
        update={"hash": P.HashRef(value="f" * 64)}
    )
    changed = execution.model_copy(
        update={"configuration_hash": "0" * 64, "objective_ref": stale_objective}
    )
    changed = changed.model_copy(
        update={"configuration_hash": execution_configuration_hash_for(changed)}
    )
    tampered = plan.model_copy(update={"resolved_execution": changed})
    tampered = tampered.model_copy(
        update={"plan_hash": compute_plan_hash(run_plan_hash_payload(tampered))}
    )
    result = execute_run(tampered, TrainingRunner(cpu_toy=True), clock=_CLOCK)
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert "objective hash" in result.manifest.failure.message
    assert called is False
