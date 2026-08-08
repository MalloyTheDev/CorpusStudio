"""The on-policy RL (GRPO) runner + supervisor routing (RL slice S5b-5): the torch-free admission gates of
``validate_rollout_success_evidence`` + the ``RolloutRunner`` dispatch/refusal + the lane factory. The full
routed run (execute_run -> RolloutRunner -> run_rollout -> independent re-verify -> manifest) needs torch +
the promoting wheel, so it is proven by the S5b-6 GPU run; here the dispatch is exercised with a fake
worker. On-policy RL stays gated upstream (required_runner_lane still refuses) until that run promotes it."""

from types import SimpleNamespace

import pytest

from corpus_studio.platform import supervisor
from corpus_studio.platform.contracts import (
    AdapterExportStateEvidence,
    GradientCoverageEvidence,
    OptimizerStepLossEvidence,
    RolloutExecutionEvidence,
    RolloutStepEvidence,
    RolloutSuccessEvidence,
    TrainableStateChangeEvidence,
)
from corpus_studio.platform.runners import RolloutRunner, build_lane_runner
from corpus_studio.platform.supervisor import (
    ProducedArtifact,
    RunnerFailure,
    validate_rollout_success_evidence,
)

_A, _B, _C, _D = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)


def _success(steps: int = 3) -> RolloutSuccessEvidence:
    execution = RolloutExecutionEvidence(
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
        reference_model_frozen=True,
        total_rollouts_sampled=steps * 4,
        step_rollout_stats=[
            RolloutStepEvidence(
                optimizer_step=i, rollouts_sampled=4, mean_reward=0.5,
                kl_to_reference=0.02, entropy=1.5, mean_advantage=0.0,
            )
            for i in range(1, steps + 1)
        ],
    )
    return RolloutSuccessEvidence(
        execution=execution, output_path_verified=True, adapter_bytes_verified=True,
        artifact_integrity_verified=True, adapter_safetensors_sha256=_A, adapter_config_sha256=_B,
        heldout_prompts_evaluated=10, heldout_baseline_mean_reward=0.2,
        heldout_policy_mean_reward=0.7, heldout_mean_reward_lift=0.5,
        heldout_max_kl_to_reference=0.05, kl_bound=0.1,
    )


def _plan(max_steps: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        resolved_rollout_execution=SimpleNamespace(schedule=SimpleNamespace(max_steps=max_steps))
    )


def _adapter_artifact() -> ProducedArtifact:
    return ProducedArtifact(artifact_id="run-x-adapter-abc", kind="adapter", path="/tmp/adapter")


# --- validate_rollout_success_evidence: the independent admission gate -------------------------------


def test_validate_refuses_missing_success_evidence() -> None:
    with pytest.raises(RunnerFailure, match="without adapter success evidence"):
        validate_rollout_success_evidence(_plan(), None, [_adapter_artifact()], None)


def test_validate_refuses_schedule_mismatch() -> None:
    with pytest.raises(RunnerFailure, match="do not match the sealed schedule"):
        validate_rollout_success_evidence(
            _plan(max_steps=5), _success(steps=3), [_adapter_artifact()], None
        )


def test_validate_refuses_epoch_scheduled_zero_steps() -> None:
    # An epoch-scheduled plan (max_steps None) can never admit zero completed steps.
    plan = SimpleNamespace(
        resolved_rollout_execution=SimpleNamespace(schedule=SimpleNamespace(max_steps=None))
    )
    proposed = _success(steps=1).model_copy(
        update={"execution": _success(steps=1).execution.model_copy(
            update={"completed_optimizer_steps": 0}
        )}
    )
    with pytest.raises(RunnerFailure, match="zero completed optimizer steps"):
        validate_rollout_success_evidence(plan, proposed, [_adapter_artifact()], None)


def test_validate_refuses_missing_adapter_artifact() -> None:
    with pytest.raises(RunnerFailure, match="no adapter artifact"):
        validate_rollout_success_evidence(_plan(), _success(), [], None)


def test_validate_refuses_failed_reverification(monkeypatch) -> None:
    monkeypatch.setattr(supervisor, "_reload_verify_adapter", lambda *a, **k: (False, "bytes changed"))
    with pytest.raises(RunnerFailure, match="failed independent re-verification"):
        validate_rollout_success_evidence(_plan(), _success(), [_adapter_artifact()], None)


def test_validate_admits_on_independent_reverification(monkeypatch) -> None:
    monkeypatch.setattr(supervisor, "_reload_verify_adapter", lambda *a, **k: (True, None))
    admitted = validate_rollout_success_evidence(
        _plan(), _success(), [_adapter_artifact()], measured_peak=None
    )
    assert admitted.execution.completed_optimizer_steps == 3
    assert admitted.adapter_bytes_verified is True
    # the held-out reward-LIFT + bounded-KL promotion gate rides through admission unchanged
    assert admitted.heldout_mean_reward_lift == pytest.approx(0.5)
    assert admitted.heldout_max_kl_to_reference <= admitted.kl_bound
    assert admitted.execution.reference_model_frozen is True


# --- RolloutRunner: dispatch + fail-closed refusal + the lane factory --------------------------------


def test_rollout_runner_refuses_without_resolved_execution() -> None:
    runner = RolloutRunner()
    ctx = SimpleNamespace(plan=SimpleNamespace(resolved_rollout_execution=None))
    with pytest.raises(RunnerFailure, match="requires a resolved rollout execution"):
        runner.run(ctx)  # type: ignore[arg-type]


def test_build_lane_runner_maps_the_rollout_lane() -> None:
    assert isinstance(build_lane_runner("rollout"), RolloutRunner)


class _RecordingCtx:
    """A minimal RunContext stand-in for a torch-free dispatch test."""

    def __init__(self, execution) -> None:
        self.plan = SimpleNamespace(resolved_rollout_execution=execution)
        self.run_id = "run-x"
        self.rollout_success_evidence = None
        self.measured_peak = None
        self.stages: list[str] = []
        self.artifacts: list[object] = []

    def emit_stage(self, _marker, message: str) -> None:
        self.stages.append(message)

    def emit_artifact(self, artifact) -> None:
        self.artifacts.append(artifact)


def _rollout_execution() -> SimpleNamespace:
    return SimpleNamespace(output_dir="/tmp/out", output_layout="run_scoped_v1")


def test_rollout_runner_dispatches_and_reports_worker_evidence(monkeypatch) -> None:
    import corpus_studio.platform.execution_config as exec_cfg
    import corpus_studio.training.rollout_worker as rollout_worker

    success = _success()
    monkeypatch.setattr(
        rollout_worker, "run_rollout",
        lambda execution, output_dir=None: SimpleNamespace(
            output_dir=output_dir, success_evidence=success
        ),
    )
    monkeypatch.setattr(exec_cfg, "run_scoped_training_output", lambda execution, run_id, leaf="adapter": "/tmp/out/runs/run-x/artifacts/adapter")
    monkeypatch.setattr(exec_cfg, "verify_run_scoped_output_path", lambda *a, **k: None)

    ctx = _RecordingCtx(_rollout_execution())
    runner = RolloutRunner(memory_sampler=lambda: None)
    produced = runner.run(ctx)  # type: ignore[arg-type]

    assert ctx.rollout_success_evidence is success  # runner REPORTS, execute_run re-verifies before admit
    assert len(produced) == 1 and produced[0].kind == "adapter"
    assert produced[0].artifact_id.startswith("run-x-adapter-")
    assert any("rollout policy adapter saved" in message for message in ctx.stages)


def test_rollout_runner_maps_a_worker_error_to_a_classified_failure(monkeypatch) -> None:
    import corpus_studio.platform.execution_config as exec_cfg
    import corpus_studio.training.rollout_worker as rollout_worker

    def _boom(execution, output_dir=None):
        raise rollout_worker.RolloutWorkerError("nf4 requires CUDA")

    monkeypatch.setattr(rollout_worker, "run_rollout", _boom)
    monkeypatch.setattr(exec_cfg, "run_scoped_training_output", lambda execution, run_id, leaf="adapter": "/tmp/out/runs/run-x/artifacts/adapter")

    ctx = _RecordingCtx(_rollout_execution())
    with pytest.raises(RunnerFailure, match="nf4 requires CUDA"):
        RolloutRunner(memory_sampler=lambda: None).run(ctx)  # type: ignore[arg-type]
