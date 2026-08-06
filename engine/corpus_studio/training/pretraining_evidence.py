"""Full-parameter pretraining execution-evidence capture (S3b Phase 2 - the worker-side honesty core).

The full-model sibling of the SFT evidence machinery in ``trainer.py``. It REUSES the generic capture
primitives unchanged (``capture_trainable_state`` / ``compare_trainable_states`` /
``capture_adapter_export_state`` / ``GradientObservationTracker``) and adds only what full-parameter
pretraining needs: a full-model export comparison (-> :class:`FullModelExportStateEvidence`) and a focused
execution tracker that assembles a :class:`PretrainingExecutionEvidence`.

Unlike the QLoRA-SFT path there is no adapter and no nf4 / master-dtype enforcement (a from-scratch model
trains in its native dtype), so the tracker asserts the config-INDEPENDENT honesty invariants: a real
optimizer that steps EXACTLY the complete trainable inventory, one finite loss per completed step,
materialized gradients on the trainable tensors, and a proven byte change in both the trainable state and
the exported ``model.safetensors``.

``torch`` is always passed in (never imported at module load), and ``finalize`` operates on already-captured
snapshots, so the whole tracker + comparison is unit-tested in the base gate. Only the live gradient-hook
registration and the raw tensor capture (in ``pretraining_trainer``) are ``# pragma: no cover`` and proven
by a cpu_toy run plus the measured GPU run."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from corpus_studio.platform.contracts import (
    FullModelExportStateEvidence,
    OptimizerStepLossEvidence,
    PretrainingExecutionEvidence,
)
from corpus_studio.training.trainer import (
    AdapterExportStateSnapshot,
    GradientObservationTracker,
    TrainableStateSnapshot,
    compare_trainable_states,
)


class PretrainingEvidenceError(RuntimeError):
    """A pretraining run that cannot produce honest execution evidence (fail-closed)."""


def compare_full_model_export_states(
    before: AdapterExportStateSnapshot,
    after: AdapterExportStateSnapshot,
    *,
    model_config_semantic_sha256: str,
) -> FullModelExportStateEvidence:
    """The full-model sibling of ``compare_adapter_export_states``: require one stable
    ``model.safetensors`` inventory (shape / dtype / membership) and at least one exact byte change."""
    if (
        before.tensor_shapes != after.tensor_shapes
        or before.tensor_dtypes != after.tensor_dtypes
        or set(before.tensor_sha256) != set(after.tensor_sha256)
    ):
        raise PretrainingEvidenceError(
            "model export inventory changed shape, dtype, or membership during training"
        )
    changed = sorted(
        name for name, digest in before.tensor_sha256.items() if after.tensor_sha256[name] != digest
    )
    if not changed or before.state_sha256 == after.state_sha256:
        raise PretrainingEvidenceError("the canonical model export state did not change during training")
    return FullModelExportStateEvidence(
        before_sha256=before.state_sha256,
        after_sha256=after.state_sha256,
        tensor_count=len(after.tensor_sha256),
        tensor_names=sorted(after.tensor_sha256),
        changed_tensor_count=len(changed),
        changed_tensor_names=changed,
        model_config_semantic_sha256=model_config_semantic_sha256,
    )


def _optimizer_parameter_ids(optimizer: Any, *, eligible_ids: list[int]) -> tuple[int, ...]:
    """The optimizer's materialized parameter ids, verified to equal the complete trainable inventory -
    proves the optimizer steps every trainable parameter and nothing else. Config-independent."""
    param_groups = getattr(optimizer, "param_groups", None)
    if (
        not isinstance(param_groups, Sequence)
        or isinstance(param_groups, (str, bytes, bytearray))
        or not param_groups
        or not all(isinstance(group, Mapping) for group in param_groups)
    ):
        raise PretrainingEvidenceError("optimizer does not expose materialized parameter groups")
    observed_ids: list[int] = []
    for group in param_groups:
        group_parameters = group.get("params")
        if not isinstance(group_parameters, Sequence) or isinstance(
            group_parameters, (str, bytes, bytearray)
        ):
            raise PretrainingEvidenceError("optimizer parameter groups are not materialized sequences")
        observed_ids.extend(id(parameter) for parameter in group_parameters)
    if (
        not eligible_ids
        or len(observed_ids) != len(set(observed_ids))
        or sorted(observed_ids) != sorted(eligible_ids)
    ):
        raise PretrainingEvidenceError(
            "optimizer parameters do not exactly match the complete trainable inventory"
        )
    return tuple(sorted(eligible_ids))


class PretrainingExecutionTracker:
    """Collect optimizer, loss, gradient, and update evidence for one full-parameter pretraining run.

    A focused, config-independent sibling of ``TrainingExecutionTracker`` (no adapter, no sealed nf4 /
    master-dtype enforcement). The callback methods are driven from an HF ``TrainerCallback`` bridge in
    ``pretraining_trainer``; ``finalize`` takes already-captured before/after snapshots so the whole class
    is pure and base-gate testable."""

    def __init__(
        self,
        *,
        expected_steps: int,
        gradients: GradientObservationTracker,
    ) -> None:
        self.expected_steps = expected_steps
        self.gradients = gradients
        self.optimizer: Any | None = None
        self.optimizer_parameter_ids: tuple[int, ...] = ()
        self.completed_steps: list[int] = []
        self.losses: dict[int, float] = {}

    def _eligible_ids(self) -> list[int]:
        return list(self.gradients.eligible_parameter_ids.values())

    def on_train_begin(self, optimizer: Any) -> None:
        if self.optimizer is not None:
            raise PretrainingEvidenceError("optimizer creation was reported more than once")
        if (
            optimizer is None
            or not callable(getattr(optimizer, "step", None))
            or not callable(getattr(optimizer, "zero_grad", None))
        ):
            raise PretrainingEvidenceError(
                "on_train_begin did not expose a real optimizer with materialized parameter groups"
            )
        self.optimizer = optimizer
        self.optimizer_parameter_ids = _optimizer_parameter_ids(
            optimizer, eligible_ids=self._eligible_ids()
        )

    def on_step_end(self, step: int, optimizer: Any) -> None:
        if self.optimizer is None or optimizer is not self.optimizer:
            raise PretrainingEvidenceError(
                "optimizer identity was absent or changed at optimizer-step completion"
            )
        if (
            _optimizer_parameter_ids(optimizer, eligible_ids=self._eligible_ids())
            != self.optimizer_parameter_ids
        ):
            raise PretrainingEvidenceError("optimizer parameter identity changed after creation")
        expected = len(self.completed_steps) + 1
        if step != expected:
            raise PretrainingEvidenceError(
                f"optimizer-step sequence deviation: expected {expected}, observed {step}"
            )
        self.completed_steps.append(step)

    def on_log(self, step: int, logs: Mapping[str, Any] | None) -> None:
        if not logs or "loss" not in logs:
            return
        if step not in self.completed_steps:
            raise PretrainingEvidenceError(
                f"loss record for optimizer step {step} arrived before that step completed"
            )
        if step in self.losses:
            raise PretrainingEvidenceError(f"duplicate loss record for optimizer step {step}")
        raw_loss = logs["loss"]
        if isinstance(raw_loss, bool) or not isinstance(raw_loss, Real):
            raise PretrainingEvidenceError(f"loss record for optimizer step {step} is not numeric")
        loss = float(raw_loss)
        if not math.isfinite(loss):
            raise PretrainingEvidenceError(f"loss record for optimizer step {step} is non-finite")
        self.losses[step] = loss

    def finalize(
        self,
        *,
        steps: int,
        before: TrainableStateSnapshot,
        after: TrainableStateSnapshot,
        before_export: AdapterExportStateSnapshot,
        after_export: AdapterExportStateSnapshot,
        model_config_semantic_sha256: str,
    ) -> PretrainingExecutionEvidence:
        """Assemble the sealed pretraining execution evidence from the collected callbacks + the
        before/after snapshots the caller captured around ``trainer.train()``."""
        if steps != self.expected_steps:
            raise PretrainingEvidenceError(
                f"completed step count {steps} does not match the sealed schedule {self.expected_steps}"
            )
        if self.optimizer is None:
            raise PretrainingEvidenceError(
                "training completed without real optimizer-creation evidence"
            )
        expected = list(range(1, steps + 1))
        if self.completed_steps != expected:
            raise PretrainingEvidenceError(
                "completed optimizer-step evidence does not match trainer global_step"
            )
        if sorted(self.losses) != expected:
            raise PretrainingEvidenceError(
                "training did not emit exactly one finite loss for every completed optimizer step"
            )
        return PretrainingExecutionEvidence(
            trainable_state=compare_trainable_states(before, after),
            model_export_state=compare_full_model_export_states(
                before_export,
                after_export,
                model_config_semantic_sha256=model_config_semantic_sha256,
            ),
            gradient_coverage=self.gradients.evidence(),
            optimizer_created=True,
            completed_optimizer_steps=steps,
            step_losses=[
                OptimizerStepLossEvidence(optimizer_step=step, loss=self.losses[step])
                for step in expected
            ],
        )


def register_full_model_gradient_hooks(  # pragma: no cover - torch hooks; proven by cpu_toy + GPU run
    model: Any, torch_module: Any
) -> GradientObservationTracker:
    """Register post-accumulation gradient-observation hooks on every trainable parameter of a
    from-scratch model, so a materialized gradient is verified present + finite and recorded. The
    full-parameter analog of ``enforce_trainable_precision`` WITHOUT the QLoRA master/nf4 dtype + device
    enforcement (which does not apply to a natively-typed from-scratch model)."""
    trainable: list[str] = []
    tracker = GradientObservationTracker(trainable)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        register_post_accumulate = getattr(parameter, "register_post_accumulate_grad_hook", None)
        if not callable(register_post_accumulate):
            raise PretrainingEvidenceError(
                f"the training runtime cannot verify materialized gradients after accumulation: {name}"
            )

        def _verify_gradient(
            materialized_parameter: Any,
            *,
            parameter_name: str = name,
            expected_parameter: Any = parameter,
        ) -> None:
            if materialized_parameter is not expected_parameter:
                raise PretrainingEvidenceError(
                    f"post-accumulation gradient hook identity changed for {parameter_name}"
                )
            gradient = getattr(expected_parameter, "grad", None)
            if gradient is None:
                raise PretrainingEvidenceError(f"materialized gradient is missing for {parameter_name}")
            isfinite = getattr(torch_module, "isfinite", None)
            if callable(isfinite) and not bool(isfinite(gradient).all().item()):
                raise PretrainingEvidenceError(
                    f"materialized gradient is non-finite for {parameter_name}"
                )
            tracker.observe(parameter_name)

        register_post_accumulate(_verify_gradient)
        trainable.append(name)
    if not trainable:
        raise PretrainingEvidenceError("the model has no trainable parameters")
    # Bind the eligible inventory only after all hooks registered (closures share one tracker object).
    tracker.eligible_names = tuple(sorted(trainable))
    tracker.eligible_parameter_ids = {
        name: id(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return tracker
