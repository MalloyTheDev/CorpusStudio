"""Platform slice 4 — the TrainingRunner that plugs training.trainer.run_training into the run
supervisor. Pure tests (no torch): ``run_training`` is monkeypatched, so the progress→RunEvent
adaptation, the produced-artifact handoff, and the failure/cancel classification are all provable on
a core-only install. The real training path is user-smoke-tested (a GPU + the [train] extra)."""

import json
import hashlib
from pathlib import Path
import struct

import pytest

import corpus_studio.platform as P
from corpus_studio.platform.common import MemoryMetrics
from corpus_studio.platform.enums import FailureTaxonomy, StageMarker
from corpus_studio.platform.execution_config import (
    canonical_sha256,
    execution_configuration_hash_for,
    verify_execution_configuration_hash,
)
from corpus_studio.platform.planner import (
    compute_plan_hash,
    run_plan_hash_payload,
    verify_run_plan_hash,
)
from corpus_studio.platform.runners import (
    TrainingRunner,
    classify_training_error,
    demo_training_plan,
)
from corpus_studio.platform.supervisor import execute_run
from corpus_studio.training.trainer import (
    ExecutionPlacementDeviation,
    TrainingEvidenceError,
    TrainerEnvironmentError,
    TrainerError,
    TrainResult,
    train_config_from_resolved,
)

_CLOCK = lambda: "2026-07-11T00:00:00+00:00"  # noqa: E731


@pytest.fixture(autouse=True)
def _isolate_relative_training_outputs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _write_fake_adapter(path: str) -> str:
    adapter = Path(path)
    adapter.mkdir(parents=True, exist_ok=True)
    header = json.dumps(_fake_adapter_header(), separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    header += b" " * (-len(header) % 8)
    (adapter / "adapter_model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + _fake_adapter_bytes()
    )
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.05,
                "bias": "none",
                "target_modules": ["q_proj"],
                "base_model_name_or_path": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "inference_mode": True,
                "peft_version": "not-installed",
                "use_dora": False,
                "use_rslora": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return str(adapter)


def _fake_adapter_header() -> dict[str, object]:
    return {
        "base_model.model.layers.0.q_proj.lora_A.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 4],
        },
        "base_model.model.layers.0.q_proj.lora_B.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
    }


def _fake_adapter_bytes() -> bytes:
    return struct.pack("<ff", 1.0, 2.0)


def _fake_export_evidence() -> dict[str, object]:
    from corpus_studio.platform.artifacts import canonical_adapter_config_sha256
    from corpus_studio.platform.parameter_accounting import canonical_tensor_state_sha256

    names = sorted(_fake_adapter_header())
    payload = _fake_adapter_bytes()
    records = []
    for index, name in enumerate(names):
        records.append(
            {
                "name": name,
                "dtype": "F32",
                "shape": [1],
                "content_sha256": hashlib.sha256(
                    payload[index * 4 : (index + 1) * 4]
                ).hexdigest(),
            }
        )
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": ["q_proj"],
        "base_model_name_or_path": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "inference_mode": True,
        "peft_version": "not-installed",
        "use_dora": False,
        "use_rslora": False,
    }
    return {
        "before_sha256": "c" * 64,
        "after_sha256": canonical_tensor_state_sha256(records),
        "tensor_count": 2,
        "tensor_names": names,
        "changed_tensor_count": 1,
        "changed_tensor_names": [names[1]],
        "adapter_config_semantic_sha256": canonical_adapter_config_sha256(config),
    }


def _execution_evidence(steps: int, losses: dict[int, float] | None = None):
    losses = losses or {step: round(1.0 / step, 6) for step in range(1, steps + 1)}
    return P.TrainingExecutionEvidence(
        trainable_state={
            "before_sha256": "a" * 64,
            "after_sha256": "b" * 64,
            "trainable_tensor_count": 2,
            "trainable_tensor_names": ["adapter.other_weight", "adapter.weight"],
            "changed_tensor_count": 1,
            "changed_tensor_names": ["adapter.weight"],
        },
        adapter_export_state=_fake_export_evidence(),
        gradient_coverage={
            "eligible_tensor_count": 2,
            "eligible_tensor_names": ["adapter.other_weight", "adapter.weight"],
            "observed_tensor_count": 1,
            "observed_tensor_names": ["adapter.weight"],
        },
        optimizer_created=True,
        completed_optimizer_steps=steps,
        step_losses=[
            {"optimizer_step": step, "loss": losses[step]}
            for step in range(1, steps + 1)
        ],
    )


def _reseal(body: dict) -> P.RunPlan:
    """Build a valid test plan after intentionally changing a field covered by its seal."""

    draft = P.RunPlan.model_validate(body)
    return draft.model_copy(update={"plan_hash": compute_plan_hash(run_plan_hash_payload(draft))})


def test_legacy_resolved_plan_remains_hash_verifiable_but_is_not_executable():
    body = demo_training_plan().model_dump(mode="json")
    execution = body["resolved_execution"]
    interface = execution["trainer_interface"]
    interface.pop("logging_strategy")
    interface.pop("logging_nan_inf_filter")
    interface["required_sft_config_fields"] = [
        name
        for name in interface["required_sft_config_fields"]
        if name not in {"logging_strategy", "logging_nan_inf_filter"}
    ]
    execution["configuration_hash"] = canonical_sha256(
        {key: value for key, value in execution.items() if key != "configuration_hash"}
    )
    plan_payload = {
        key: value for key, value in body.items() if key not in {"plan_hash", "created_at"}
    }
    body["plan_hash"] = compute_plan_hash(plan_payload)

    legacy = P.RunPlan.model_validate(body)
    assert legacy.resolved_execution is not None
    assert verify_execution_configuration_hash(legacy.resolved_execution)
    assert verify_run_plan_hash(legacy)
    with pytest.raises(TrainerError, match="predates exact per-step loss logging"):
        train_config_from_resolved(legacy.resolved_execution)


def _fake_run_training(steps, *, loss_by_step=None, capture=None, checkpoints=False):
    """Build a stand-in for run_training that drives the progress callback `steps` times then returns
    a TrainResult — no torch, no model, no dataset."""

    def _run(config, *, progress_callback=None, stage_callback=None, **_kw):
        if capture is not None:
            capture["config"] = config
        if stage_callback is not None:
            stage_callback("optimizer_created", "observed the real optimizer")
        losses = loss_by_step or {
            step: round(1.0 / step, 6) for step in range(1, steps + 1)
        }
        for step in range(1, steps + 1):
            if progress_callback is not None:
                progress_callback(step, steps, losses[step])
        adapter_path = _write_fake_adapter(config.output_dir)
        return TrainResult(
            output_dir=config.output_dir,
            adapter_path=adapter_path,
            base_model=config.base_model,
            cpu_toy=config.cpu_toy,
            steps=steps,
            final_loss=losses[steps],
            checkpoints=[f"{config.output_dir}/checkpoint-{steps}"] if checkpoints else [],
            execution_evidence=_execution_evidence(steps, losses),
        )

    return _run


# ---- per-step telemetry emission ---------------------------------------------


def test_progress_emits_step_time_worker_memory_and_token_rates(monkeypatch):
    # The runner runs in the CUDA-owning child, so it measures per-step wall time (monotonic deltas at
    # the optimizer-step boundary), samples the worker torch allocator memory, and folds in the
    # trainer's per-step token counts - all into the one metric RunEvent for that step.
    import itertools

    ticks = itertools.count()
    monkeypatch.setattr("time.monotonic", lambda: next(ticks) * 0.5)  # strictly increasing

    def _fake(config, *, progress_callback=None, stage_callback=None, token_callback=None, **_kw):
        if stage_callback is not None:
            stage_callback("optimizer_created", "observed the real optimizer")
        losses = {1: 1.0, 2: 0.5}
        for step in range(1, 3):  # the demo plan seals a 2-step schedule
            if token_callback is not None:
                # nonpadding, supervised, observed_microbatches (one microbatch observed per step).
                token_callback(step, 100 * step, 40 * step, 1)
            if progress_callback is not None:
                progress_callback(step, 2, losses[step])
        adapter_path = _write_fake_adapter(config.output_dir)
        return TrainResult(
            output_dir=config.output_dir,
            adapter_path=adapter_path,
            base_model=config.base_model,
            cpu_toy=config.cpu_toy,
            steps=2,
            final_loss=losses[2],
            checkpoints=[],
            execution_evidence=_execution_evidence(2, losses),
        )

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _fake)
    runner = TrainingRunner(
        cpu_toy=True,
        memory_sampler=lambda: MemoryMetrics(torch_allocated_bytes=42),
    )
    result = execute_run(demo_training_plan(), runner, clock=_CLOCK)
    assert result.manifest.state == "succeeded"

    metrics = {
        event.optimizer_step: event.metrics
        for event in result.events
        if event.event_type == "metric" and event.optimizer_step
    }
    # Worker allocator memory rides on every step's metric record (persisted into RunEvents.jsonl).
    assert metrics[1].memory is not None and metrics[1].memory.torch_allocated_bytes == 42
    assert metrics[1].loss == 1.0
    # Raw observed counts ride on every step (the source of truth the rates are validated against),
    # regardless of whether a wall time exists yet to form a rate.
    assert metrics[1].observed_microbatches == 1
    assert metrics[1].nonpadding_tokens == 100 and metrics[1].supervised_tokens == 40
    assert metrics[2].nonpadding_tokens == 200 and metrics[2].supervised_tokens == 80
    # Step 1 has no prior boundary, so it carries no wall time and thus no derived token rate.
    assert metrics[1].step_time_seconds is None
    assert metrics[1].tokens_per_sec is None
    # Step 2 has a measured wall time and derived non-padding / supervised token rates.
    assert metrics[2].step_time_seconds is not None and metrics[2].step_time_seconds > 0
    assert metrics[2].tokens_per_sec is not None and metrics[2].tokens_per_sec > 0
    assert metrics[2].supervised_tokens_per_sec is not None
    # Each derived rate equals observed tokens / observed duration (no fabrication).
    assert metrics[2].tokens_per_sec == 200 / metrics[2].step_time_seconds
    assert metrics[2].supervised_tokens_per_sec == 80 / metrics[2].step_time_seconds


def test_progress_step_time_is_null_without_token_or_time_data(monkeypatch):
    # A trainer that never reports tokens still emits step_time + loss; token rates stay null (honest).
    def _fake(config, *, progress_callback=None, stage_callback=None, token_callback=None, **_kw):
        if stage_callback is not None:
            stage_callback("optimizer_created", "opt")
        for step in range(1, 3):
            if progress_callback is not None:
                progress_callback(step, 2, 1.0 / step)
        adapter_path = _write_fake_adapter(config.output_dir)
        return TrainResult(
            output_dir=config.output_dir,
            adapter_path=adapter_path,
            base_model=config.base_model,
            cpu_toy=config.cpu_toy,
            steps=2,
            final_loss=0.5,
            checkpoints=[],
            execution_evidence=_execution_evidence(2, {1: 1.0, 2: 0.5}),
        )

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _fake)
    result = execute_run(demo_training_plan(), TrainingRunner(cpu_toy=True), clock=_CLOCK)
    metrics = {
        event.optimizer_step: event.metrics
        for event in result.events
        if event.event_type == "metric" and event.optimizer_step
    }
    assert metrics[2].tokens_per_sec is None and metrics[2].supervised_tokens_per_sec is None


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
        _fake_run_training(2),
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
        _fake_run_training(2, loss_by_step={1: 0.9, 2: 0.5}, capture=capture),
    )
    result = execute_run(
        demo_training_plan(), TrainingRunner(cpu_toy=True), run_id="run-1", clock=_CLOCK
    )

    assert result.manifest.state == "succeeded"
    assert result.manifest.target == "corpus_studio"
    # The sealed cpu_toy mode flowed into the trainer config unchanged.
    assert capture["config"].cpu_toy is True

    metrics = [e for e in result.events if e.event_type == "metric"]
    assert [m.optimizer_step for m in metrics] == [1, 2]
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
    assert result.manifest.training_success_evidence is not None
    success = result.manifest.training_success_evidence
    assert success.output_path_verified is True
    assert success.adapter_bytes_verified is True
    assert success.artifact_integrity_verified is True
    assert success.execution.gradient_coverage.observed_tensor_count == 1
    assert [item.loss for item in success.execution.step_losses] == [0.9, 0.5]
    produced = next(e for e in result.events if e.event_type == "artifact_produced")
    assert produced.payload == {
        "artifact_id": artifact_id,
        "kind": "adapter",
        "path": str(adapter_path),
    }
    assert not any("checkpoint-" in (e.message or "") for e in result.events)


@pytest.mark.parametrize("tamper", ["tensor", "config", "nested_weight"])
def test_training_success_rejects_bytes_not_bound_to_trainer_evidence(monkeypatch, tamper):
    base = _fake_run_training(2)

    def _tampered(*args, **kwargs):
        result = base(*args, **kwargs)
        adapter = Path(result.adapter_path)
        if tamper == "tensor":
            path = adapter / "adapter_model.safetensors"
            payload = bytearray(path.read_bytes())
            payload[-1] ^= 1
            path.write_bytes(payload)
        elif tamper == "config":
            path = adapter / "adapter_config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config["target_modules"] = ["k_proj"]
            path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
        else:
            nested = adapter / "checkpoint-shadow"
            nested.mkdir()
            (nested / "model.safetensors").write_bytes(b"unsealed")
        return result

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _tampered)
    result = execute_run(
        demo_training_plan(),
        TrainingRunner(cpu_toy=True),
        run_id=f"run-tamper-{tamper}",
        clock=_CLOCK,
    )

    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.ARTIFACT_FAILURE
    assert result.manifest.failure.stage == StageMarker.export
    assert result.manifest.training_success_evidence is None


def test_training_runner_rejects_linked_output_root_before_invoking_trainer(
    tmp_path, monkeypatch
):
    outside = tmp_path / "outside"
    outside.mkdir()
    Path("output").symlink_to(outside, target_is_directory=True)
    invoked = False

    def _unexpected(*_args, **_kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("trainer must not run through a linked output root")

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _unexpected)
    result = execute_run(
        demo_training_plan(),
        TrainingRunner(cpu_toy=True),
        run_id="run-linked-output",
        clock=_CLOCK,
    )

    assert invoked is False
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.ARTIFACT_FAILURE
    assert "link-like" in result.manifest.failure.message


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
    assert result.manifest.failure.taxonomy == FailureTaxonomy.ARTIFACT_FAILURE
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
    assert result.manifest.failure.taxonomy == FailureTaxonomy.ARTIFACT_FAILURE
    assert "weight bytes" in result.manifest.failure.message
    assert result.manifest.artifact_ids == []


def test_training_runner_refuses_missing_execution_evidence(monkeypatch):
    def _missing_evidence(config, *, progress_callback=None, stage_callback=None, **_kw):
        if stage_callback is not None:
            stage_callback("optimizer_created", "synthetic")
        if progress_callback is not None:
            progress_callback(1, 2, 0.9)
            progress_callback(2, 2, 0.5)
        adapter = _write_fake_adapter(config.output_dir)
        return TrainResult(
            output_dir=config.output_dir,
            adapter_path=adapter,
            base_model=config.base_model,
            cpu_toy=True,
            steps=2,
        )

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _missing_evidence)
    result = execute_run(demo_training_plan(), TrainingRunner(cpu_toy=True), clock=_CLOCK)
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.UPDATE_FAILURE
    assert result.manifest.failure.stage == StageMarker.optimizer_step


def test_supervisor_rejects_loss_or_optimizer_event_evidence_mismatch(monkeypatch):
    def _mismatched(config, *, progress_callback=None, stage_callback=None, **_kw):
        if stage_callback is not None:
            stage_callback("optimizer_created", "first")
            stage_callback("optimizer_created", "duplicate")
        if progress_callback is not None:
            progress_callback(1, 2, 0.9)
            progress_callback(2, 2, 0.4)
        adapter = _write_fake_adapter(config.output_dir)
        return TrainResult(
            output_dir=config.output_dir,
            adapter_path=adapter,
            base_model=config.base_model,
            cpu_toy=True,
            steps=2,
            execution_evidence=_execution_evidence(2, {1: 0.9, 2: 0.5}),
        )

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _mismatched)
    result = execute_run(demo_training_plan(), TrainingRunner(cpu_toy=True), clock=_CLOCK)
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.OPTIMIZER_FAILURE
    assert result.manifest.failure.stage == StageMarker.optimizer_created


def test_supervisor_rejects_step_loss_values_that_disagree_with_terminal_evidence(monkeypatch):
    def _mismatched(config, *, progress_callback=None, stage_callback=None, **_kw):
        if stage_callback is not None:
            stage_callback("optimizer_created", "observed")
        if progress_callback is not None:
            progress_callback(1, 2, 0.9)
            progress_callback(2, 2, 0.4)
        adapter = _write_fake_adapter(config.output_dir)
        return TrainResult(
            output_dir=config.output_dir,
            adapter_path=adapter,
            base_model=config.base_model,
            cpu_toy=True,
            steps=2,
            execution_evidence=_execution_evidence(2, {1: 0.9, 2: 0.5}),
        )

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _mismatched)
    result = execute_run(demo_training_plan(), TrainingRunner(cpu_toy=True), clock=_CLOCK)
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.LOSS_EVIDENCE_FAILURE
    assert result.manifest.failure.stage == StageMarker.loss


@pytest.mark.parametrize("loss_before_optimizer", [True, False])
def test_supervisor_rejects_reversed_or_lossless_completed_step_events(
    monkeypatch, loss_before_optimizer
):
    def _invalid(config, *, progress_callback=None, stage_callback=None, **_kw):
        if loss_before_optimizer and progress_callback is not None:
            progress_callback(1, 1, 0.5)
        if stage_callback is not None:
            stage_callback("optimizer_created", "observed")
        if progress_callback is not None:
            if not loss_before_optimizer:
                progress_callback(1, 1, None)
                progress_callback(1, 1, 0.5)
            progress_callback(2, 2, 0.4)
        adapter = _write_fake_adapter(config.output_dir)
        return TrainResult(
            output_dir=config.output_dir,
            adapter_path=adapter,
            base_model=config.base_model,
            cpu_toy=True,
            steps=2,
            execution_evidence=_execution_evidence(2, {1: 0.5, 2: 0.4}),
        )

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _invalid)
    result = execute_run(demo_training_plan(), TrainingRunner(cpu_toy=True), clock=_CLOCK)
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.LOSS_EVIDENCE_FAILURE
    assert result.manifest.failure.stage == StageMarker.loss


def test_a_forward_stage_oom_is_classified_at_forward_not_a_stale_loss(monkeypatch):
    # The trainer marks the forward/backward compute region at each optimizer step's start (on_step_begin
    # -> stage_callback("forward")). A runtime OOM during that step must then be classified at `forward` -
    # a stage the research protocol's flash-eligibility mapping lists - instead of the prior step's `loss`
    # log, which fail-closes flash as NOT_RUN. Regression: the 7B ladder rung-2048 OOM landed on `loss`
    # (structurally unmapped) and withheld flash; forward marking makes the eligibility rule reachable.
    def _oom_in_forward(config, *, progress_callback=None, stage_callback=None, **_kw):
        if stage_callback is not None:
            stage_callback("optimizer_created", "trainer ready")
            stage_callback("forward", "step 1: forward/backward compute")
        raise RuntimeError("CUDA out of memory. Tried to allocate 448.00 MiB")

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _oom_in_forward)
    result = execute_run(demo_training_plan(), TrainingRunner(cpu_toy=True), clock=_CLOCK)
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.OOM
    assert result.manifest.failure.stage == StageMarker.forward


def test_event_sink_failure_cannot_rewrite_a_success(monkeypatch):
    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _fake_run_training(2))

    def _raising_sink(_event):
        raise RuntimeError("observer failed")

    result = execute_run(
        demo_training_plan(), TrainingRunner(cpu_toy=True), sink=_raising_sink, clock=_CLOCK
    )
    assert result.manifest.state == "succeeded"
    assert result.manifest.notes == "event sink failures were isolated: RuntimeError"
    assert sum(event.event_type == "terminal" for event in result.events) == 1


def test_artifact_manifest_persistence_fails_before_success_manifest(
    tmp_path, monkeypatch
):
    from corpus_studio.platform import supervisor as supervisor_module

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _fake_run_training(2))
    monkeypatch.setattr(
        supervisor_module,
        "write_artifact_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    result = execute_run(
        demo_training_plan(),
        TrainingRunner(cpu_toy=True),
        out_dir=tmp_path / "records",
        clock=_CLOCK,
    )
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.ARTIFACT_FAILURE
    assert result.manifest.failure.stage == StageMarker.export
    persisted = P.RunManifest.model_validate_json(
        (tmp_path / "records" / "runs" / result.manifest.run_id / "RunManifest.json").read_text()
    )
    assert persisted.state == "failed"
    assert persisted.training_success_evidence is None


def test_artifact_gate_failure_cannot_promote_measured_fit(monkeypatch):
    from corpus_studio.platform import supervisor as supervisor_module

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _fake_run_training(2))
    real_builder = supervisor_module.build_artifact_manifest

    def _modified_manifest(**kwargs):
        manifest = real_builder(**kwargs)
        assert manifest.integrity is not None
        return manifest.model_copy(
            update={
                "integrity": manifest.integrity.model_copy(
                    update={"current_integrity": "modified"}
                )
            }
        )

    monkeypatch.setattr(supervisor_module, "build_artifact_manifest", _modified_manifest)
    result = execute_run(
        demo_training_plan(),
        TrainingRunner(
            cpu_toy=True,
            memory_sampler=lambda: _sample(peak_reserved=6 * GB),
        ),
        clock=_CLOCK,
    )
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.ARTIFACT_FAILURE
    assert result.manifest.failure.stage == StageMarker.export
    assert result.manifest.final_fit is not None
    assert result.manifest.final_fit.classification.value == "NATIVE_UNPROVEN"
    assert result.manifest.training_success_evidence is None


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
            execution_evidence=_execution_evidence(1, {1: 0.5}),
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
    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _fake_run_training(2))
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

    def _stalling_trainer(config, *, progress_callback=None, stage_callback=None, **_kw):
        if stage_callback is not None:
            stage_callback("optimizer_created", "observed the real optimizer")
        losses = {1: 0.9, 2: 0.8}
        if progress_callback is not None:
            for step in range(1, 3):
                progress_callback(step, 2, losses[step])
        time.sleep(0.4)  # > heartbeat_timeout_s; the watchdog thread trips (heads-up only, no cancel)
        adapter_path = _write_fake_adapter(config.output_dir)
        return TrainResult(
            output_dir=config.output_dir,
            adapter_path=adapter_path,
            base_model=config.base_model,
            cpu_toy=True,
            steps=2,
            execution_evidence=_execution_evidence(2, losses),
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
    assert result.manifest.failure is not None
    assert result.manifest.failure.stage == StageMarker.loss
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
            raise TrainerEnvironmentError("CPU toy training needs torch + transformers + …")

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _raise)
    result = execute_run(demo_training_plan(), TrainingRunner(cpu_toy=True), clock=_CLOCK)

    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == FailureTaxonomy.ENVIRONMENT_FAILURE
    assert result.manifest.failure.stage == StageMarker.process_start
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


@pytest.mark.parametrize(
    ("taxonomy", "stage"),
    [
        (FailureTaxonomy.GRADIENT_FAILURE, StageMarker.backward),
        (FailureTaxonomy.NUMERICAL_FAILURE, StageMarker.loss),
        (FailureTaxonomy.LOSS_EVIDENCE_FAILURE, StageMarker.loss),
        (FailureTaxonomy.OPTIMIZER_FAILURE, StageMarker.optimizer_created),
        (FailureTaxonomy.UPDATE_FAILURE, StageMarker.optimizer_step),
    ],
)
def test_training_evidence_failures_retain_exact_taxonomy_and_stage(
    monkeypatch, taxonomy, stage
):
    def _raise(config, **_kw):
        raise TrainingEvidenceError(
            "synthetic evidence failure",
            taxonomy=taxonomy,
            stage=stage,
        )

    monkeypatch.setattr("corpus_studio.training.trainer.run_training", _raise)
    result = execute_run(demo_training_plan(), TrainingRunner(cpu_toy=True), clock=_CLOCK)
    assert result.manifest.state == "failed"
    assert result.manifest.failure is not None
    assert result.manifest.failure.taxonomy == taxonomy
    assert result.manifest.failure.stage == stage


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
