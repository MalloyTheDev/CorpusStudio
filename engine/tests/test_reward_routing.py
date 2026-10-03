"""The reward-model runner + supervisor routing (RL slice S5a PR 3c-1): the torch-free admission gates of
``validate_reward_success_evidence`` + the ``RewardRunner`` dispatch/refusal + the lane factory. The full
routed run (execute_run -> RewardRunner -> run_reward -> independent re-verify -> manifest) needs torch +
the promoting wheel, so it is proven by the PR 3c-2 GPU run; here the dispatch is exercised with a fake
worker. Reward stays gated upstream (required_runner_lane still refuses) until that run promotes it."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from corpus_studio.platform import supervisor
from corpus_studio.platform.contracts import (
    AdapterExportStateEvidence,
    GradientCoverageEvidence,
    OptimizerStepLossEvidence,
    PreferenceRewardMarginEvidence,
    RewardExecutionEvidence,
    RewardSuccessEvidence,
    TrainableStateChangeEvidence,
)
from corpus_studio.platform.enums import FailureTaxonomy, StageMarker
from corpus_studio.platform.execution_config import local_input_binding
from corpus_studio.platform.runners import RewardRunner, build_lane_runner
from corpus_studio.platform.supervisor import (
    ProducedArtifact,
    RunnerFailure,
    validate_reward_success_evidence,
)

_A, _B, _C, _D = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)


def _success(steps: int = 3) -> RewardSuccessEvidence:
    execution = RewardExecutionEvidence(
        trainable_state=TrainableStateChangeEvidence(
            before_sha256=_A, after_sha256=_B, trainable_tensor_count=2,
            trainable_tensor_names=["p.0", "p.1"], changed_tensor_count=1, changed_tensor_names=["p.0"],
        ),
        adapter_export_state=AdapterExportStateEvidence(
            before_sha256=_C, after_sha256=_D, tensor_count=2, tensor_names=["p.0", "p.1"],
            changed_tensor_count=1, changed_tensor_names=["p.0"], adapter_config_semantic_sha256=_A,
        ),
        gradient_coverage=GradientCoverageEvidence(
            eligible_tensor_count=2, eligible_tensor_names=["p.0", "p.1"],
            observed_tensor_count=1, observed_tensor_names=["p.0"],
        ),
        optimizer_created=True,
        completed_optimizer_steps=steps,
        step_losses=[OptimizerStepLossEvidence(optimizer_step=i, loss=0.5) for i in range(1, steps + 1)],
        reward_pairs_consumed=8,
        step_reward_margins=[
            PreferenceRewardMarginEvidence(
                optimizer_step=i, chosen_reward=0.4, rejected_reward=-0.1, margin=0.5,
            )
            for i in range(1, steps + 1)
        ],
    )
    return RewardSuccessEvidence(
        execution=execution, output_path_verified=True, adapter_bytes_verified=True,
        artifact_integrity_verified=True, adapter_safetensors_sha256=_A, adapter_config_sha256=_B,
        heldout_pairwise_accuracy=0.9, heldout_pairs_evaluated=10,
    )


def _plan(max_steps: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        resolved_reward_execution=SimpleNamespace(schedule=SimpleNamespace(max_steps=max_steps))
    )


def _adapter_artifact() -> ProducedArtifact:
    return ProducedArtifact(artifact_id="run-x-adapter-abc", kind="adapter", path="/tmp/adapter")


# --- validate_reward_success_evidence: the independent admission gate --------------------------------


def test_validate_refuses_missing_success_evidence() -> None:
    with pytest.raises(RunnerFailure, match="without adapter success evidence"):
        validate_reward_success_evidence(_plan(), None, [_adapter_artifact()], None)


def test_validate_refuses_schedule_mismatch() -> None:
    with pytest.raises(RunnerFailure, match="do not match the sealed schedule"):
        validate_reward_success_evidence(
            _plan(max_steps=5), _success(steps=3), [_adapter_artifact()], None
        )


def test_validate_refuses_epoch_scheduled_zero_steps() -> None:
    # An epoch-scheduled plan (max_steps None) can never admit zero completed steps.
    plan = SimpleNamespace(
        resolved_reward_execution=SimpleNamespace(schedule=SimpleNamespace(max_steps=None))
    )
    proposed = _success(steps=1).model_copy(
        update={"execution": _success(steps=1).execution.model_copy(
            update={"completed_optimizer_steps": 0}
        )}
    )
    with pytest.raises(RunnerFailure, match="zero completed optimizer steps"):
        validate_reward_success_evidence(plan, proposed, [_adapter_artifact()], None)


def test_validate_refuses_missing_adapter_artifact() -> None:
    with pytest.raises(RunnerFailure, match="no adapter artifact"):
        validate_reward_success_evidence(_plan(), _success(), [], None)


def test_validate_refuses_failed_reverification(monkeypatch) -> None:
    monkeypatch.setattr(supervisor, "_reload_verify_adapter", lambda *a, **k: (False, "bytes changed"))
    with pytest.raises(RunnerFailure, match="failed independent re-verification"):
        validate_reward_success_evidence(_plan(), _success(), [_adapter_artifact()], None)


def test_validate_admits_on_independent_reverification(monkeypatch) -> None:
    monkeypatch.setattr(supervisor, "_reload_verify_adapter", lambda *a, **k: (True, None))
    admitted = validate_reward_success_evidence(
        _plan(), _success(), [_adapter_artifact()], measured_peak=None
    )
    assert admitted.execution.completed_optimizer_steps == 3
    assert admitted.adapter_bytes_verified is True
    # the held-out pairwise accuracy gate rides through admission unchanged
    assert admitted.heldout_pairwise_accuracy == pytest.approx(0.9)


# --- RewardRunner: dispatch + fail-closed refusal + the lane factory --------------------------------


def test_reward_runner_refuses_without_resolved_execution() -> None:
    runner = RewardRunner()
    ctx = SimpleNamespace(plan=SimpleNamespace(resolved_reward_execution=None))
    with pytest.raises(RunnerFailure, match="requires a resolved reward execution"):
        runner.run(ctx)  # type: ignore[arg-type]


def test_build_lane_runner_maps_the_reward_lane() -> None:
    assert isinstance(build_lane_runner("reward"), RewardRunner)


class _RecordingCtx:
    """A minimal RunContext stand-in for a torch-free dispatch test."""

    def __init__(self, execution) -> None:
        self.plan = SimpleNamespace(resolved_reward_execution=execution)
        self.run_id = "run-x"
        self.reward_success_evidence = None
        self.measured_peak = None
        self.stages: list[str] = []
        self.payloads: list[dict] = []
        self.artifacts: list[object] = []

    def emit_stage(self, _marker, message: str, *, payload=None) -> None:
        self.stages.append(message)
        if payload is not None:
            self.payloads.append(payload)

    def emit_artifact(self, artifact) -> None:
        self.artifacts.append(artifact)


_PAIR = {"prompt": "p", "chosen": "c", "rejected": "r"}


def _reward_execution(tmp_path) -> SimpleNamespace:
    # A real sealed dataset binding: the runner verifies these bytes before it dispatches the worker.
    dataset = tmp_path / "pairs.jsonl"
    dataset.write_text(json.dumps(_PAIR) + "\n", encoding="utf-8")
    binding = local_input_binding(
        kind="dataset", location=str(dataset), ref_id="dataset", directory=False
    )
    return SimpleNamespace(
        output_dir="/tmp/out",
        output_layout="run_scoped_v1",
        inputs=SimpleNamespace(dataset=binding),
        configuration_hash="e" * 64,
    )


def _skip_loader_admission(monkeypatch) -> None:
    # These dispatch tests use a stand-in execution with only a dataset binding; the model/tokenizer
    # admission gate (#863) is exercised with real sealed configs in test_sealed_loader_admission.py.
    import corpus_studio.platform.runners as runners

    monkeypatch.setattr(runners, "_admit_sealed_loader", lambda *_a, **_k: None)


def test_reward_runner_dispatches_and_reports_worker_evidence(monkeypatch, tmp_path) -> None:
    import corpus_studio.platform.execution_config as exec_cfg
    import corpus_studio.training.reward_worker as reward_worker

    _skip_loader_admission(monkeypatch)
    success = _success()

    def _fake_run_reward(execution, *, dataset, output_dir=None, stage_callback=None):
        # the worker receives the rows parsed from the verified sealed bytes, never a path to reopen
        assert dataset.content_sha256 == execution.inputs.dataset.content_sha256
        assert dataset.rows == (_PAIR,)
        return SimpleNamespace(output_dir=output_dir, success_evidence=success)

    monkeypatch.setattr(reward_worker, "run_reward", _fake_run_reward)
    monkeypatch.setattr(exec_cfg, "run_scoped_training_output", lambda execution, run_id, leaf="adapter": "/tmp/out/runs/run-x/artifacts/adapter")
    monkeypatch.setattr(exec_cfg, "verify_run_scoped_output_path", lambda *a, **k: None)

    execution = _reward_execution(tmp_path)
    ctx = _RecordingCtx(execution)
    runner = RewardRunner(memory_sampler=lambda: None)
    produced = runner.run(ctx)  # type: ignore[arg-type]

    assert ctx.reward_success_evidence is success  # runner REPORTS, execute_run re-verifies before admit
    assert len(produced) == 1 and produced[0].kind == "adapter"
    assert produced[0].artifact_id.startswith("run-x-adapter-")
    assert any("reward adapter saved" in message for message in ctx.stages)
    assert ctx.payloads == [
        {
            "content_sha256": execution.inputs.dataset.content_sha256,
            "byte_count": len(json.dumps(_PAIR)) + 1,
            "row_count": 1,
            "execution_configuration_hash": "e" * 64,
        }
    ]


def test_reward_runner_maps_a_worker_error_to_a_classified_failure(monkeypatch, tmp_path) -> None:
    import corpus_studio.platform.execution_config as exec_cfg
    import corpus_studio.training.reward_worker as reward_worker

    _skip_loader_admission(monkeypatch)

    def _boom(execution, *, dataset, output_dir=None, stage_callback=None):
        raise reward_worker.RewardWorkerError("nf4 requires CUDA")

    monkeypatch.setattr(reward_worker, "run_reward", _boom)
    monkeypatch.setattr(exec_cfg, "run_scoped_training_output", lambda execution, run_id, leaf="adapter": "/tmp/out/runs/run-x/artifacts/adapter")

    ctx = _RecordingCtx(_reward_execution(tmp_path))
    with pytest.raises(RunnerFailure, match="nf4 requires CUDA"):
        RewardRunner(memory_sampler=lambda: None).run(ctx)  # type: ignore[arg-type]


def test_reward_runner_refuses_a_changed_dataset_before_dispatch(monkeypatch, tmp_path) -> None:
    import corpus_studio.platform.execution_config as exec_cfg
    import corpus_studio.training.reward_worker as reward_worker

    _skip_loader_admission(monkeypatch)

    def _must_not_dispatch(*_a, **_k):
        pytest.fail("the reward worker ran on a dataset that no longer matches its seal")

    monkeypatch.setattr(reward_worker, "run_reward", _must_not_dispatch)
    monkeypatch.setattr(exec_cfg, "run_scoped_training_output", lambda execution, run_id, leaf="adapter": "/tmp/out/runs/run-x/artifacts/adapter")

    execution = _reward_execution(tmp_path)
    Path(execution.inputs.dataset.location).write_text(
        json.dumps({**_PAIR, "chosen": "TAMPERED"}) + "\n", encoding="utf-8"
    )
    ctx = _RecordingCtx(execution)
    with pytest.raises(RunnerFailure, match="dataset bytes changed after") as refused:
        RewardRunner(memory_sampler=lambda: None).run(ctx)  # type: ignore[arg-type]
    assert refused.value.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert refused.value.stage == StageMarker.dataset_verification
    assert ctx.payloads == []
