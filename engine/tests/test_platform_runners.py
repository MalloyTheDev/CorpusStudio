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

    def _run(config, *, progress_callback=None):
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


# ---- failure + cancel classification -----------------------------------------


def test_missing_runtime_is_environment_failure(monkeypatch):
    def _raise(config, *, progress_callback=None):
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
    def _oom(config, *, progress_callback=None):
        raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB")

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _oom)
    result = execute_run(demo_training_plan(), TrainingRunner(cpu_toy=True), clock=_CLOCK)
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.OOM
    assert "micro_batch_size" in (result.manifest.failure.remediation or "")


def test_runner_unrecognized_error_stays_fail(monkeypatch):
    def _boom(config, *, progress_callback=None):
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
