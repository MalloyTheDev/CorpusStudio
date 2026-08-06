"""The pretraining runner + supervisor routing (S3b Phase 2 slice 2b): the torch-free admission gates of
``validate_pretraining_success_evidence`` and the ``PretrainingRunner`` fail-closed refusal. The full
routed run (execute_run -> PretrainingRunner -> run_pretraining -> independent re-verify -> manifest) is
proven by a cpu_toy run, since the worker dispatch needs torch."""

from types import SimpleNamespace

import pytest

from corpus_studio.platform import supervisor
from corpus_studio.platform.contracts import (
    FullModelExportStateEvidence,
    GradientCoverageEvidence,
    OptimizerStepLossEvidence,
    PretrainingExecutionEvidence,
    PretrainingSuccessEvidence,
    TrainableStateChangeEvidence,
)
from corpus_studio.platform.runners import PretrainingRunner
from corpus_studio.platform.supervisor import (
    ProducedArtifact,
    RunnerFailure,
    validate_pretraining_success_evidence,
)

_A, _B, _C, _D = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)


def _success(steps: int = 3) -> PretrainingSuccessEvidence:
    execution = PretrainingExecutionEvidence(
        trainable_state=TrainableStateChangeEvidence(
            before_sha256=_A, after_sha256=_B, trainable_tensor_count=2,
            trainable_tensor_names=["p.0", "p.1"], changed_tensor_count=1, changed_tensor_names=["p.0"],
        ),
        model_export_state=FullModelExportStateEvidence(
            before_sha256=_C, after_sha256=_D, tensor_count=2, tensor_names=["p.0", "p.1"],
            changed_tensor_count=1, changed_tensor_names=["p.0"], model_config_semantic_sha256=_A,
        ),
        gradient_coverage=GradientCoverageEvidence(
            eligible_tensor_count=2, eligible_tensor_names=["p.0", "p.1"],
            observed_tensor_count=1, observed_tensor_names=["p.0"],
        ),
        optimizer_created=True,
        completed_optimizer_steps=steps,
        step_losses=[OptimizerStepLossEvidence(optimizer_step=i, loss=3.0) for i in range(1, steps + 1)],
    )
    return PretrainingSuccessEvidence(
        execution=execution, output_path_verified=True, model_bytes_verified=True,
        artifact_integrity_verified=True, model_safetensors_sha256=_A, model_config_sha256=_B,
    )


def _plan(max_steps: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        resolved_pretraining_execution=SimpleNamespace(schedule=SimpleNamespace(max_steps=max_steps))
    )


def _model_artifact() -> ProducedArtifact:
    return ProducedArtifact(artifact_id="run-x-model-abc", kind="model", path="/tmp/model")


def test_validate_refuses_missing_success_evidence() -> None:
    with pytest.raises(RunnerFailure, match="without full-model success evidence"):
        validate_pretraining_success_evidence(_plan(), None, [_model_artifact()], None)


def test_validate_refuses_schedule_mismatch() -> None:
    with pytest.raises(RunnerFailure, match="do not match the sealed schedule"):
        validate_pretraining_success_evidence(_plan(max_steps=5), _success(steps=3), [_model_artifact()], None)


def test_validate_refuses_missing_model_artifact() -> None:
    with pytest.raises(RunnerFailure, match="no model artifact"):
        validate_pretraining_success_evidence(_plan(), _success(), [], None)


def test_validate_refuses_failed_reverification(monkeypatch) -> None:
    monkeypatch.setattr(
        supervisor, "_reload_verify_full_model", lambda *a, **k: (False, "bytes changed")
    )
    with pytest.raises(RunnerFailure, match="failed independent re-verification"):
        validate_pretraining_success_evidence(_plan(), _success(), [_model_artifact()], None)


def test_validate_admits_on_independent_reverification(monkeypatch) -> None:
    monkeypatch.setattr(supervisor, "_reload_verify_full_model", lambda *a, **k: (True, None))
    admitted = validate_pretraining_success_evidence(_plan(), _success(), [_model_artifact()], measured_peak=None)
    assert admitted.execution.completed_optimizer_steps == 3
    assert admitted.model_bytes_verified is True


def test_pretraining_runner_refuses_without_resolved_execution() -> None:
    runner = PretrainingRunner(cpu_toy=True)
    ctx = SimpleNamespace(plan=SimpleNamespace(resolved_pretraining_execution=None))
    with pytest.raises(RunnerFailure, match="requires a resolved pretraining execution"):
        runner.run(ctx)  # type: ignore[arg-type]
