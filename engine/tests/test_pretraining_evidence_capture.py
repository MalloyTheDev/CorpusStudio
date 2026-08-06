"""The full-parameter pretraining evidence CAPTURE logic (S3b Phase 2): the full-model export comparison
and the execution tracker's assembly, proven torch-free with fakes. The live gradient hooks + raw tensor
capture (register_full_model_gradient_hooks / capture_* in the worker) are pragma, proven by a run."""

from typing import Any

import pytest

from corpus_studio.training.pretraining_evidence import (
    PretrainingEvidenceError,
    PretrainingExecutionTracker,
    compare_full_model_export_states,
)
from corpus_studio.training.trainer import (
    AdapterExportStateSnapshot,
    GradientObservationTracker,
    TrainableStateSnapshot,
)

_A, _B = "a" * 64, "b" * 64


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
    """A stand-in trainable parameter - the tracker only reads identity + requires_grad."""

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


# ---- compare_full_model_export_states ----------------------------------------------------------


def test_compare_full_model_export_ok() -> None:
    before = _export_snapshot(_A, {"p.0": "1" * 64, "p.1": "2" * 64})
    after = _export_snapshot(_B, {"p.0": "9" * 64, "p.1": "2" * 64})  # only p.0 changed
    ev = compare_full_model_export_states(before, after, model_config_semantic_sha256=_A)
    assert ev.changed_tensor_names == ["p.0"]
    assert ev.tensor_count == 2
    assert ev.model_config_semantic_sha256 == _A


def test_compare_full_model_export_no_change_refused() -> None:
    snap = _export_snapshot(_A, {"p.0": "1" * 64})
    with pytest.raises(PretrainingEvidenceError, match="did not change"):
        compare_full_model_export_states(snap, snap, model_config_semantic_sha256=_A)


def test_compare_full_model_export_inventory_drift_refused() -> None:
    before = _export_snapshot(_A, {"p.0": "1" * 64})
    after = _export_snapshot(_B, {"p.0": "9" * 64, "p.1": "2" * 64})  # membership grew
    with pytest.raises(PretrainingEvidenceError, match="shape, dtype, or membership"):
        compare_full_model_export_states(before, after, model_config_semantic_sha256=_A)


# ---- PretrainingExecutionTracker ---------------------------------------------------------------


def _run_two_steps(tracker: PretrainingExecutionTracker, opt: _FakeOptimizer) -> None:
    tracker.on_train_begin(opt)
    tracker.on_step_end(1, opt)
    tracker.on_log(1, {"loss": 3.0})
    tracker.on_step_end(2, opt)
    tracker.on_log(2, {"loss": 2.5})


def test_tracker_full_lifecycle_builds_evidence() -> None:
    p0, p1 = _FakeParam(), _FakeParam()
    grads = _grad_tracker(["p.0", "p.1"], {"p.0": p0, "p.1": p1}, observed=["p.0"])
    tracker = PretrainingExecutionTracker(expected_steps=2, gradients=grads)
    _run_two_steps(tracker, _FakeOptimizer([p0, p1]))
    ev = tracker.finalize(
        steps=2,
        before=_trainable_snapshot(_A, {"p.0": "1" * 64, "p.1": "2" * 64}),
        after=_trainable_snapshot(_B, {"p.0": "9" * 64, "p.1": "2" * 64}),  # p.0 changed
        before_export=_export_snapshot(_A, {"p.0": "1" * 64, "p.1": "2" * 64}),
        after_export=_export_snapshot(_B, {"p.0": "9" * 64, "p.1": "2" * 64}),
        model_config_semantic_sha256=_A,
    )
    assert ev.completed_optimizer_steps == 2
    assert [s.loss for s in ev.step_losses] == [3.0, 2.5]
    assert ev.gradient_coverage.observed_tensor_names == ["p.0"]
    assert ev.trainable_state.changed_tensor_names == ["p.0"]


def test_tracker_refuses_optimizer_not_covering_trainable_inventory() -> None:
    p0, p1 = _FakeParam(), _FakeParam()
    grads = _grad_tracker(["p.0", "p.1"], {"p.0": p0, "p.1": p1}, observed=["p.0"])
    tracker = PretrainingExecutionTracker(expected_steps=1, gradients=grads)
    # optimizer only steps p0 - not the full trainable inventory
    with pytest.raises(PretrainingEvidenceError, match="do not exactly match the complete trainable"):
        tracker.on_train_begin(_FakeOptimizer([p0]))


def test_tracker_refuses_out_of_order_step() -> None:
    p0 = _FakeParam()
    grads = _grad_tracker(["p.0"], {"p.0": p0}, observed=["p.0"])
    tracker = PretrainingExecutionTracker(expected_steps=2, gradients=grads)
    opt = _FakeOptimizer([p0])
    tracker.on_train_begin(opt)
    with pytest.raises(PretrainingEvidenceError, match="sequence deviation"):
        tracker.on_step_end(2, opt)  # skipped step 1


def test_tracker_refuses_duplicate_loss() -> None:
    p0 = _FakeParam()
    grads = _grad_tracker(["p.0"], {"p.0": p0}, observed=["p.0"])
    tracker = PretrainingExecutionTracker(expected_steps=1, gradients=grads)
    opt = _FakeOptimizer([p0])
    tracker.on_train_begin(opt)
    tracker.on_step_end(1, opt)
    tracker.on_log(1, {"loss": 3.0})
    with pytest.raises(PretrainingEvidenceError, match="duplicate loss"):
        tracker.on_log(1, {"loss": 2.0})


def test_tracker_refuses_missing_loss_at_finalize() -> None:
    p0 = _FakeParam()
    grads = _grad_tracker(["p.0"], {"p.0": p0}, observed=["p.0"])
    tracker = PretrainingExecutionTracker(expected_steps=1, gradients=grads)
    opt = _FakeOptimizer([p0])
    tracker.on_train_begin(opt)
    tracker.on_step_end(1, opt)  # no on_log -> no loss recorded
    with pytest.raises(PretrainingEvidenceError, match="one finite loss for every completed"):
        tracker.finalize(
            steps=1,
            before=_trainable_snapshot(_A, {"p.0": "1" * 64}),
            after=_trainable_snapshot(_B, {"p.0": "9" * 64}),
            before_export=_export_snapshot(_A, {"p.0": "1" * 64}),
            after_export=_export_snapshot(_B, {"p.0": "9" * 64}),
            model_config_semantic_sha256=_A,
        )


def test_tracker_refuses_step_overrun_past_schedule() -> None:
    p0 = _FakeParam()
    grads = _grad_tracker(["p.0"], {"p.0": p0}, observed=["p.0"])
    tracker = PretrainingExecutionTracker(expected_steps=1, gradients=grads)  # sealed ceiling = 1
    opt = _FakeOptimizer([p0])
    tracker.on_train_begin(opt)
    tracker.on_step_end(1, opt)
    tracker.on_log(1, {"loss": 3.0})
    tracker.on_step_end(2, opt)  # ran past the sealed ceiling
    tracker.on_log(2, {"loss": 2.5})
    with pytest.raises(PretrainingEvidenceError, match="exceeds the sealed schedule"):
        tracker.finalize(
            steps=2,
            before=_trainable_snapshot(_A, {"p.0": "1" * 64}),
            after=_trainable_snapshot(_B, {"p.0": "9" * 64}),
            before_export=_export_snapshot(_A, {"p.0": "1" * 64}),
            after_export=_export_snapshot(_B, {"p.0": "9" * 64}),
            model_config_semantic_sha256=_A,
        )


def test_tracker_allows_early_stop_short_of_schedule() -> None:
    # data-limited: the sealed ceiling is 5 but only 2 steps completed - record the ACTUAL 2 steps.
    p0, p1 = _FakeParam(), _FakeParam()
    grads = _grad_tracker(["p.0", "p.1"], {"p.0": p0, "p.1": p1}, observed=["p.0"])
    tracker = PretrainingExecutionTracker(expected_steps=5, gradients=grads)
    _run_two_steps(tracker, _FakeOptimizer([p0, p1]))
    ev = tracker.finalize(
        steps=2,
        before=_trainable_snapshot(_A, {"p.0": "1" * 64, "p.1": "2" * 64}),
        after=_trainable_snapshot(_B, {"p.0": "9" * 64, "p.1": "2" * 64}),
        before_export=_export_snapshot(_A, {"p.0": "1" * 64, "p.1": "2" * 64}),
        after_export=_export_snapshot(_B, {"p.0": "9" * 64, "p.1": "2" * 64}),
        model_config_semantic_sha256=_A,
    )
    assert ev.completed_optimizer_steps == 2


def test_tracker_refuses_non_parameter_in_optimizer_group() -> None:
    p0 = _FakeParam()
    grads = _grad_tracker(["p.0"], {"p.0": p0}, observed=["p.0"])
    tracker = PretrainingExecutionTracker(expected_steps=1, gradients=grads)
    # a stray non-parameter entry (no requires_grad) must be rejected at the interface
    with pytest.raises(PretrainingEvidenceError, match="non-parameter entry"):
        tracker.on_train_begin(_FakeOptimizer(["not-a-parameter"]))
