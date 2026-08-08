"""The on-policy RL (GRPO) evidence CAPTURE logic (RL slice S5b): the RolloutExecutionTracker's assembly of
the formal RolloutExecutionEvidence / RolloutSuccessEvidence from a rollout run, proven torch-free with
fakes. The live gradient hooks + raw capture + adapter-bytes build (wired into run_rollout_training) are
pragma, proven by a GPU run."""

from typing import Any

import pytest

from corpus_studio.platform.contracts import RolloutSuccessEvidence
from corpus_studio.training.rollout_evidence import RolloutEvidenceError, RolloutExecutionTracker
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


def _begun_tracker(expected_steps: int = 3) -> RolloutExecutionTracker:
    p0, p1 = _FakeParam(), _FakeParam()
    params = {"a.0": p0, "a.1": p1}
    grads = _grad_tracker(["a.0", "a.1"], params, ["a.0"])  # a.0 observed a materialized gradient
    tracker = RolloutExecutionTracker(expected_steps=expected_steps, gradients=grads)
    tracker.on_train_begin(_FakeOptimizer([p0, p1]))
    return tracker


def _finalize(tracker: RolloutExecutionTracker, steps: int) -> Any:
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
    )


def _step(tracker: RolloutExecutionTracker, step: int, **over: Any) -> None:
    fields = dict(
        loss=0.5, rollouts_sampled=4, mean_reward=0.2, kl_to_reference=0.03, entropy=1.1,
        mean_advantage=0.0,
    )
    fields.update(over)
    tracker.record_step(step, **fields)


def test_rollout_execution_evidence_assembles_from_a_run() -> None:
    tracker = _begun_tracker()
    _step(tracker, 1, mean_reward=0.1, kl_to_reference=0.02)
    _step(tracker, 2, mean_reward=0.6, kl_to_reference=0.05)
    ev = _finalize(tracker, 2)
    assert ev.completed_optimizer_steps == 2
    assert ev.reference_model_frozen is True
    assert ev.total_rollouts_sampled == 8  # 4 + 4
    assert [s.optimizer_step for s in ev.step_losses] == [1, 2]
    assert [r.optimizer_step for r in ev.step_rollout_stats] == [1, 2]
    assert ev.step_rollout_stats[1].mean_reward == pytest.approx(0.6)
    assert ev.step_rollout_stats[1].kl_to_reference == pytest.approx(0.05)
    assert ev.trainable_state.changed_tensor_names == ["a.0"]


def test_rollout_record_step_refuses_a_non_finite_signal() -> None:
    tracker = _begun_tracker()
    with pytest.raises(RolloutEvidenceError, match="must be finite"):
        _step(tracker, 1, mean_reward=float("inf"))


def test_rollout_record_step_refuses_a_negative_kl() -> None:
    tracker = _begun_tracker()
    with pytest.raises(RolloutEvidenceError, match="KL divergence.*non-negative"):
        _step(tracker, 1, kl_to_reference=-0.01)


def test_rollout_record_step_refuses_a_degenerate_group() -> None:
    tracker = _begun_tracker()
    with pytest.raises(RolloutEvidenceError, match="group of at least two rollouts"):
        _step(tracker, 1, rollouts_sampled=1)


def test_rollout_finalize_refuses_overrun_past_the_ceiling() -> None:
    tracker = _begun_tracker(expected_steps=1)
    _step(tracker, 1)
    _step(tracker, 2)
    with pytest.raises(RolloutEvidenceError, match="exceeds the sealed schedule ceiling"):
        _finalize(tracker, 2)


def test_rollout_success_evidence_binds_the_lift_and_kl_gate() -> None:
    tracker = _begun_tracker()
    _step(tracker, 1)
    execution = _finalize(tracker, 1)
    success = RolloutSuccessEvidence(
        execution=execution,
        output_path_verified=True,
        adapter_bytes_verified=True,
        artifact_integrity_verified=True,
        adapter_safetensors_sha256="c" * 64,
        adapter_config_sha256="d" * 64,
        heldout_prompts_evaluated=16,
        heldout_baseline_mean_reward=0.1,
        heldout_policy_mean_reward=0.8,
        heldout_mean_reward_lift=0.7,
        heldout_max_kl_to_reference=0.09,
        kl_bound=0.1,
    )
    # the promotion gate is a MEASURED held-out reward LIFT under a bounded KL, not a falling loss
    assert success.heldout_mean_reward_lift == pytest.approx(0.7)
    # the lift must be internally consistent (policy - baseline)
    with pytest.raises(ValueError, match="heldout_mean_reward_lift must equal"):
        RolloutSuccessEvidence(
            execution=execution,
            output_path_verified=True,
            adapter_bytes_verified=True,
            artifact_integrity_verified=True,
            adapter_safetensors_sha256="c" * 64,
            adapter_config_sha256="d" * 64,
            heldout_prompts_evaluated=16,
            heldout_baseline_mean_reward=0.1,
            heldout_policy_mean_reward=0.8,
            heldout_mean_reward_lift=0.2,  # inconsistent: should be 0.7
            heldout_max_kl_to_reference=0.05,
            kl_bound=0.1,
        )
    # a run that blew the KL bound is NOT an admissible on-policy success (KL is the safety rail)
    with pytest.raises(ValueError, match="within kl_bound"):
        RolloutSuccessEvidence(
            execution=execution,
            output_path_verified=True,
            adapter_bytes_verified=True,
            artifact_integrity_verified=True,
            adapter_safetensors_sha256="c" * 64,
            adapter_config_sha256="d" * 64,
            heldout_prompts_evaluated=16,
            heldout_baseline_mean_reward=0.1,
            heldout_policy_mean_reward=0.8,
            heldout_mean_reward_lift=0.7,
            heldout_max_kl_to_reference=0.5,  # exceeds kl_bound
            kl_bound=0.1,
        )
