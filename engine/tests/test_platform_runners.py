"""Platform slice 4 — the TrainingRunner that plugs training.trainer.run_training into the run
supervisor. Pure tests (no torch): ``run_training`` is monkeypatched, so the progress→RunEvent
adaptation, the produced-artifact handoff, and the failure/cancel classification are all provable on
a core-only install. The real training path is user-smoke-tested (a GPU + the [train] extra)."""

import corpus_studio.platform as P
from corpus_studio.platform.enums import FailureTaxonomy, StageMarker
from corpus_studio.platform.runners import (
    TrainingRunner,
    classify_training_error,
    demo_training_plan,
)
from corpus_studio.platform.supervisor import execute_run
from corpus_studio.training.trainer import TrainerError, TrainResult

_CLOCK = lambda: "2026-07-11T00:00:00+00:00"  # noqa: E731


def _fake_run_training(steps, *, loss_by_step=None, capture=None):
    """Build a stand-in for run_training that drives the progress callback `steps` times then returns
    a TrainResult — no torch, no model, no dataset."""

    def _run(config, *, progress_callback=None, **_kw):
        if capture is not None:
            capture["config"] = config
        for step in range(1, steps + 1):
            if progress_callback is not None:
                loss = (loss_by_step or {}).get(step)
                progress_callback(step, steps, loss)
        return TrainResult(
            output_dir=config.output_dir,
            adapter_path=f"{config.output_dir}/adapter",
            base_model=config.base_model,
            cpu_toy=config.cpu_toy,
            steps=steps,
            final_loss=(loss_by_step or {}).get(steps),
            checkpoints=[f"{config.output_dir}/checkpoint-{steps}"],
        )

    return _run


# ---- demo plan ---------------------------------------------------------------


def test_demo_training_plan_is_valid_and_carries_a_snapshot():
    plan = demo_training_plan()
    assert P.RunPlan.model_validate_json(plan.model_dump_json()) == plan
    assert plan.training_config_snapshot["base_model"].startswith("hf-internal-testing")


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
    assert result.manifest.target == "cpu_toy"
    # cpu_toy override flowed into the resolved trainer config.
    assert capture["config"].cpu_toy is True

    metrics = [e for e in result.events if e.event_type == "metric"]
    assert [m.optimizer_step for m in metrics] == [1, 2, 3]
    assert metrics[0].metrics is not None
    assert metrics[0].metrics.loss == 0.9

    # The saved adapter is recorded as a produced artifact + an artifact_produced event.
    assert result.manifest.artifact_ids == ["run-1-adapter"]
    produced = next(e for e in result.events if e.event_type == "artifact_produced")
    assert produced.payload == {
        "artifact_id": "run-1-adapter",
        "kind": "adapter",
        "path": "output/adapter",
    }
    # A checkpoint log line was emitted.
    assert any(e.event_type == "log" and "checkpoint-3" in (e.message or "") for e in result.events)


def test_max_steps_override_flows_into_the_config(monkeypatch):
    capture: dict = {}
    monkeypatch.setattr(
        "corpus_studio.training.trainer.run_training", _fake_run_training(1, capture=capture)
    )
    execute_run(demo_training_plan(), TrainingRunner(cpu_toy=True, max_steps=5), clock=_CLOCK)
    assert capture["config"].max_steps == 5


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
    result = execute_run(P.RunPlan.model_validate(body), TrainingRunner(), clock=_CLOCK)
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert "cannot consume" in result.manifest.failure.message
    assert called == []


# ---- multi-backend dispatch --------------------------------------------------


def _plan_with_backend(backend_id: str):
    body = demo_training_plan().model_dump(mode="json")
    body["backend_ref"] = {"id": backend_id}
    return P.RunPlan.model_validate(body)


def test_training_runner_dispatches_to_the_plan_backend_corpus_studio(monkeypatch):
    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _fake_run_training(1))
    result = execute_run(_plan_with_backend("corpus_studio"), TrainingRunner(), clock=_CLOCK)
    assert result.manifest.state == "succeeded"
    assert result.manifest.target == "corpus_studio"  # the manifest names the framework that ran


def test_training_runner_dispatches_to_unsloth(monkeypatch):
    # The 'training' runner reads the plan's backend_ref and drives the Unsloth trainer for it. The
    # Unsloth function is mocked (the real one needs a GPU + unsloth); dispatch + labeling is what we
    # prove here.
    monkeypatch.setattr(
        "corpus_studio.training.unsloth_trainer.run_unsloth_training", _fake_run_training(2)
    )
    result = execute_run(_plan_with_backend("unsloth"), TrainingRunner(), clock=_CLOCK)
    assert result.manifest.state == "succeeded"
    assert result.manifest.target == "unsloth"
    assert [m.optimizer_step for m in result.events if m.event_type == "metric"] == [1, 2]


def test_unknown_backend_is_unsupported_configuration():
    result = execute_run(_plan_with_backend("megatron"), TrainingRunner(), clock=_CLOCK)
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert "megatron" in (result.manifest.failure.message or "")


def test_cpu_toy_always_uses_the_first_party_path_regardless_of_backend(monkeypatch):
    # A plan can carry any backend_ref, but --runner cpu_toy is the first-party CPU smoke path — it must
    # NOT silently route to Unsloth (which has no CPU path). It runs run_training and labels 'cpu_toy'.
    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _fake_run_training(1))
    result = execute_run(_plan_with_backend("unsloth"), TrainingRunner(cpu_toy=True), clock=_CLOCK)
    assert result.manifest.state == "succeeded"
    assert result.manifest.target == "cpu_toy"


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
        return TrainResult(
            output_dir="o", adapter_path="o", base_model=config.base_model, cpu_toy=True, steps=1
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
        return TrainResult(
            output_dir="o", adapter_path="o", base_model=config.base_model, cpu_toy=True, steps=3
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


def test_empty_snapshot_is_unsupported_configuration():
    # The plain echo demo plan has no training_config_snapshot.
    from corpus_studio.platform.supervisor import demo_run_plan

    result = execute_run(demo_run_plan(), TrainingRunner(cpu_toy=True), clock=_CLOCK)
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION


def test_invalid_snapshot_is_unsupported_configuration():
    # A snapshot missing the required base_model / dataset_path can't build a TrainRunConfig.
    plan = demo_training_plan().model_copy(update={"training_config_snapshot": {"lora_r": 4}})
    result = execute_run(plan, TrainingRunner(cpu_toy=True), clock=_CLOCK)
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
