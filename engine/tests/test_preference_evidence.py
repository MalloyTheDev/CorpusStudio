"""The offline-DPO (preference) evidence CAPTURE logic (DPO slice 2b): the PreferenceExecutionTracker's
assembly of the formal PreferenceExecutionEvidence (#813) from a DPO run, proven torch-free with fakes.
The live gradient hooks + raw capture + adapter-bytes build (wired into run_dpo_training) are pragma,
proven by a GPU run."""

from typing import Any

import pytest

from corpus_studio.training.preference_evidence import (
    PreferenceEvidenceError,
    PreferenceExecutionTracker,
)
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


def _begun_tracker(expected_steps: int = 3) -> PreferenceExecutionTracker:
    p0, p1 = _FakeParam(), _FakeParam()
    params = {"a.0": p0, "a.1": p1}
    grads = _grad_tracker(["a.0", "a.1"], params, ["a.0"])  # a.0 observed a materialized gradient
    tracker = PreferenceExecutionTracker(expected_steps=expected_steps, gradients=grads)
    tracker.on_train_begin(_FakeOptimizer([p0, p1]))
    return tracker


def _finalize(tracker: PreferenceExecutionTracker, steps: int) -> Any:
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
        preference_pairs_consumed=10,
    )


def test_preference_execution_evidence_assembles_from_a_dpo_run() -> None:
    tracker = _begun_tracker()
    tracker.record_step(1, loss=0.69, chosen_reward=0.2, rejected_reward=-0.1, margin=0.3)
    tracker.record_step(2, loss=0.60, chosen_reward=0.5, rejected_reward=-0.3, margin=0.8)
    ev = _finalize(tracker, 2)
    assert ev.completed_optimizer_steps == 2
    assert ev.reference_model_frozen is True
    assert ev.preference_pairs_consumed == 10
    assert [s.optimizer_step for s in ev.step_losses] == [1, 2]
    assert [m.optimizer_step for m in ev.step_reward_margins] == [1, 2]
    assert ev.step_reward_margins[1].margin == pytest.approx(0.8)
    assert ev.trainable_state.changed_tensor_names == ["a.0"]
    assert ev.adapter_export_state.tensor_count == 2


def test_preference_record_step_refuses_inconsistent_margin() -> None:
    tracker = _begun_tracker()
    with pytest.raises(PreferenceEvidenceError, match="margin must equal"):
        tracker.record_step(1, loss=0.7, chosen_reward=0.5, rejected_reward=-0.3, margin=0.1)


def test_preference_record_step_refuses_non_finite_signal() -> None:
    tracker = _begun_tracker()
    with pytest.raises(PreferenceEvidenceError, match="finite"):
        tracker.record_step(1, loss=float("nan"), chosen_reward=0.2, rejected_reward=-0.1, margin=0.3)


def test_preference_record_step_refuses_out_of_order() -> None:
    tracker = _begun_tracker()
    with pytest.raises(PreferenceEvidenceError, match="sequence deviation"):
        tracker.record_step(2, loss=0.7, chosen_reward=0.2, rejected_reward=-0.1, margin=0.3)


def test_preference_finalize_refuses_overrun_past_the_ceiling() -> None:
    tracker = _begun_tracker(expected_steps=1)
    tracker.record_step(1, loss=0.7, chosen_reward=0.2, rejected_reward=-0.1, margin=0.3)
    tracker.record_step(2, loss=0.6, chosen_reward=0.5, rejected_reward=-0.3, margin=0.8)
    with pytest.raises(PreferenceEvidenceError, match="exceeds the sealed schedule ceiling"):
        _finalize(tracker, 2)


def test_preference_on_train_begin_refuses_optimizer_inventory_mismatch() -> None:
    p0, p1 = _FakeParam(), _FakeParam()
    params = {"a.0": p0, "a.1": p1}
    grads = _grad_tracker(["a.0", "a.1"], params, ["a.0"])
    tracker = PreferenceExecutionTracker(expected_steps=3, gradients=grads)
    with pytest.raises(PreferenceEvidenceError, match="do not exactly match"):
        tracker.on_train_begin(_FakeOptimizer([p0]))  # optimizer omits a.1
