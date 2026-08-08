"""The pairwise reward-model evidence CAPTURE logic (RL slice S5a): the RewardExecutionTracker's assembly
of the formal RewardExecutionEvidence / RewardSuccessEvidence from a reward run, proven torch-free with
fakes. The live gradient hooks + raw capture + adapter-bytes build (wired into run_reward_training) are
pragma, proven by a GPU run."""

from typing import Any

import pytest

from corpus_studio.platform.contracts import RewardSuccessEvidence
from corpus_studio.training.reward_evidence import RewardEvidenceError, RewardExecutionTracker
from corpus_studio.training.trainer import (
    AdapterExportStateSnapshot,
    GradientObservationTracker,
    TrainableStateSnapshot,
)

_A = "a" * 64


def _export_snapshot(state_sha: str, tensor_sha: dict[str, str]) -> AdapterExportStateSnapshot:
    return AdapterExportStateSnapshot(
        state_sha256=state_sha,
        tensor_sha256=dict(tensor_sha),
        tensor_shapes={n: (2,) for n in tensor_sha},
        tensor_dtypes={n: "F32" for n in tensor_sha},
    )


def _trainable_snapshot(state_sha: str, tensor_sha: dict[str, str]) -> TrainableStateSnapshot:
    return TrainableStateSnapshot(
        state_sha256=state_sha,
        tensor_sha256=dict(tensor_sha),
        tensor_shapes={n: (2,) for n in tensor_sha},
        tensor_dtypes={n: "torch.float32" for n in tensor_sha},
    )


class _FakeParam:
    requires_grad = True


class _FakeOptimizer:
    def __init__(self, params: list[Any]) -> None:
        self.param_groups = [{"params": list(params)}]

    def step(self) -> None:  # pragma: no cover - identity only
        ...

    def zero_grad(self) -> None:  # pragma: no cover - identity only
        ...


def _grad_tracker(
    names: list[str], param_objs: dict[str, Any], observed: list[str]
) -> GradientObservationTracker:
    tracker = GradientObservationTracker(names)
    tracker.eligible_names = tuple(sorted(names))
    tracker.eligible_parameter_ids = {n: id(param_objs[n]) for n in names}
    for name in observed:
        tracker.observe(name)
    return tracker


def _begun_tracker(expected_steps: int = 3) -> RewardExecutionTracker:
    p0, p1 = _FakeParam(), _FakeParam()
    params = {"a.0": p0, "a.1": p1}
    grads = _grad_tracker(["a.0", "a.1"], params, ["a.0"])  # a.0 observed a materialized gradient
    tracker = RewardExecutionTracker(expected_steps=expected_steps, gradients=grads)
    tracker.on_train_begin(_FakeOptimizer([p0, p1]))
    return tracker


def _finalize(tracker: RewardExecutionTracker, steps: int) -> Any:
    before_t = _trainable_snapshot(_A, {"a.0": "1" * 64, "a.1": "2" * 64})
    after_t = _trainable_snapshot("b" * 64, {"a.0": "9" * 64, "a.1": "2" * 64})  # a.0 changed
    before_e = _export_snapshot(_A, {"a.0": "1" * 64, "a.1": "2" * 64})
    after_e = _export_snapshot("b" * 64, {"a.0": "9" * 64, "a.1": "2" * 64})
    return tracker.finalize(
        steps=steps,
        before=before_t,
        after=after_t,
        before_export=before_e,
        after_export=after_e,
        adapter_config_semantic_sha256=_A,
        reward_pairs_consumed=10,
    )


def test_reward_execution_evidence_assembles_from_a_run() -> None:
    tracker = _begun_tracker()
    # the chosen/rejected values are the EXPLICIT SEQ_CLS scores; margin = chosen - rejected
    tracker.record_step(1, loss=0.69, chosen_reward=0.2, rejected_reward=-0.1, margin=0.3)
    tracker.record_step(2, loss=0.40, chosen_reward=0.9, rejected_reward=-0.4, margin=1.3)
    ev = _finalize(tracker, 2)
    assert ev.completed_optimizer_steps == 2
    assert ev.reward_pairs_consumed == 10
    assert [s.optimizer_step for s in ev.step_losses] == [1, 2]
    assert [m.optimizer_step for m in ev.step_reward_margins] == [1, 2]
    assert ev.step_reward_margins[1].margin == pytest.approx(1.3)
    assert ev.trainable_state.changed_tensor_names == ["a.0"]
    assert ev.adapter_export_state.tensor_count == 2


def test_reward_record_step_refuses_inconsistent_margin() -> None:
    tracker = _begun_tracker()
    with pytest.raises(RewardEvidenceError, match="margin must equal"):
        tracker.record_step(1, loss=0.7, chosen_reward=0.5, rejected_reward=-0.3, margin=0.1)


def test_reward_record_step_refuses_non_finite_signal() -> None:
    tracker = _begun_tracker()
    with pytest.raises(RewardEvidenceError, match="must be finite"):
        tracker.record_step(1, loss=float("nan"), chosen_reward=0.2, rejected_reward=-0.1, margin=0.3)


def test_reward_finalize_refuses_overrun_past_the_ceiling() -> None:
    tracker = _begun_tracker(expected_steps=1)
    tracker.record_step(1, loss=0.7, chosen_reward=0.2, rejected_reward=-0.1, margin=0.3)
    tracker.record_step(2, loss=0.5, chosen_reward=0.9, rejected_reward=-0.4, margin=1.3)
    with pytest.raises(RewardEvidenceError, match="exceeds the sealed schedule ceiling"):
        _finalize(tracker, 2)


def test_reward_success_evidence_binds_the_heldout_pairwise_gate() -> None:
    tracker = _begun_tracker()
    tracker.record_step(1, loss=0.69, chosen_reward=0.2, rejected_reward=-0.1, margin=0.3)
    execution = _finalize(tracker, 1)
    success = RewardSuccessEvidence(
        execution=execution,
        output_path_verified=True,
        adapter_bytes_verified=True,
        artifact_integrity_verified=True,
        adapter_safetensors_sha256="c" * 64,
        adapter_config_sha256="d" * 64,
        heldout_pairwise_accuracy=0.92,
        heldout_pairs_evaluated=25,
    )
    # the promotion gate is held-out pairwise ranking accuracy, not a falling loss
    assert success.heldout_pairwise_accuracy == pytest.approx(0.92)
    assert success.heldout_pairs_evaluated == 25
    # the accuracy gate is a fraction in [0, 1] - an out-of-range value fails closed
    with pytest.raises(ValueError, match="less than or equal to 1"):
        RewardSuccessEvidence(
            execution=execution,
            output_path_verified=True,
            adapter_bytes_verified=True,
            artifact_integrity_verified=True,
            adapter_safetensors_sha256="c" * 64,
            adapter_config_sha256="d" * 64,
            heldout_pairwise_accuracy=1.5,
            heldout_pairs_evaluated=25,
        )
