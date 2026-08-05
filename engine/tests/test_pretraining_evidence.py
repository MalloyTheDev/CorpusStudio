"""The from-scratch / continued pretraining execution-evidence contract family (Phase 2, S3b): the
full-parameter sibling of the SFT ``TrainingSuccessEvidence`` family. Torch-free contract validation
only - the worker capture that PRODUCES this evidence, and the runner admission that CONSUMES it, are a
separate gated slice; here we prove the honesty gates fail closed."""

import pytest
from pydantic import ValidationError

from corpus_studio.platform.common import Ref
from corpus_studio.platform.contracts import (
    AdapterExportStateEvidence,
    FullModelExportStateEvidence,
    GradientCoverageEvidence,
    OptimizerStepLossEvidence,
    PretrainingExecutionEvidence,
    PretrainingSuccessEvidence,
    RunManifest,
    TrainableStateChangeEvidence,
    TrainingExecutionEvidence,
    TrainingSuccessEvidence,
)

_A, _B, _C, _D = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)


def _trainable() -> TrainableStateChangeEvidence:
    # For pretraining the trainable set is the COMPLETE parameter inventory (reused generic model).
    return TrainableStateChangeEvidence(
        before_sha256=_A,
        after_sha256=_B,
        trainable_tensor_count=2,
        trainable_tensor_names=["p.0", "p.1"],
        changed_tensor_count=1,
        changed_tensor_names=["p.0"],
    )


def _export(*, before: str = _C, after: str = _D, **overrides: object) -> FullModelExportStateEvidence:
    kwargs: dict[str, object] = {
        "before_sha256": before,
        "after_sha256": after,
        "tensor_count": 2,
        "tensor_names": ["p.0", "p.1"],
        "changed_tensor_count": 1,
        "changed_tensor_names": ["p.0"],
        "model_config_semantic_sha256": _A,
    }
    kwargs.update(overrides)
    return FullModelExportStateEvidence(**kwargs)  # type: ignore[arg-type]


def _gradients(**overrides: object) -> GradientCoverageEvidence:
    kwargs: dict[str, object] = {
        "eligible_tensor_count": 2,
        "eligible_tensor_names": ["p.0", "p.1"],
        "observed_tensor_count": 1,
        "observed_tensor_names": ["p.0"],
    }
    kwargs.update(overrides)
    return GradientCoverageEvidence(**kwargs)  # type: ignore[arg-type]


def _execution(**overrides: object) -> PretrainingExecutionEvidence:
    kwargs: dict[str, object] = {
        "trainable_state": _trainable(),
        "model_export_state": _export(),
        "gradient_coverage": _gradients(),
        "optimizer_created": True,
        "completed_optimizer_steps": 2,
        "step_losses": [
            OptimizerStepLossEvidence(optimizer_step=1, loss=3.0),
            OptimizerStepLossEvidence(optimizer_step=2, loss=2.5),
        ],
    }
    kwargs.update(overrides)
    return PretrainingExecutionEvidence(**kwargs)  # type: ignore[arg-type]


def test_valid_pretraining_execution_evidence() -> None:
    ev = _execution()
    assert ev.completed_optimizer_steps == 2
    # the full-model export inventory equals the complete trainable-state inventory
    assert ev.model_export_state.tensor_count == ev.trainable_state.trainable_tensor_count


def test_full_model_export_must_change() -> None:
    with pytest.raises(ValidationError, match="must change during successful pretraining"):
        _export(before=_C, after=_C)


def test_full_model_export_changed_names_must_be_subset() -> None:
    with pytest.raises(ValidationError, match="belong to the export inventory"):
        _export(changed_tensor_names=["p.9"])


def test_full_model_export_names_must_be_sorted_unique() -> None:
    with pytest.raises(ValidationError, match="sorted and unique"):
        _export(tensor_names=["p.1", "p.0"])


def test_gradient_eligibility_must_equal_trainable_inventory() -> None:
    # gradients eligible over a strict subset of the trainable params - dishonest, refuse.
    bad = _gradients(
        eligible_tensor_count=1,
        eligible_tensor_names=["p.0"],
        observed_tensor_count=1,
        observed_tensor_names=["p.0"],
    )
    with pytest.raises(ValidationError, match="gradient eligibility must equal"):
        _execution(gradient_coverage=bad)


def test_changed_tensor_needs_an_observed_gradient() -> None:
    # the only changed tensor (p.0) has NO observed gradient - refuse the unsupported update claim.
    no_overlap = _gradients(observed_tensor_count=1, observed_tensor_names=["p.1"])
    with pytest.raises(ValidationError, match="observed materialized gradient"):
        _execution(gradient_coverage=no_overlap)


def test_step_losses_gap_refused() -> None:
    with pytest.raises(ValidationError, match="one ordered finite loss for every completed step"):
        _execution(
            completed_optimizer_steps=2,
            step_losses=[OptimizerStepLossEvidence(optimizer_step=1, loss=3.0)],
        )


def test_export_count_must_equal_trainable_inventory() -> None:
    # export claims 3 tensors but only 2 params are trainable - refuse the mismatch.
    big_export = _export(
        tensor_count=3,
        tensor_names=["p.0", "p.1", "p.2"],
    )
    with pytest.raises(ValidationError, match="model export tensor count must equal"):
        _execution(model_export_state=big_export)


def test_optimizer_created_is_type_locked() -> None:
    with pytest.raises(ValidationError):
        _execution(optimizer_created=False)


def test_pretraining_success_evidence_rides_the_run_manifest() -> None:
    success = PretrainingSuccessEvidence(
        execution=_execution(),
        output_path_verified=True,
        model_bytes_verified=True,
        artifact_integrity_verified=True,
        model_safetensors_sha256=_A,
        model_config_sha256=_B,
    )
    manifest = RunManifest(
        run_id="run-pretrain-1",
        plan_ref=Ref(id="plan-pretrain-1"),
        created_at="2026-08-05T00:00:00+00:00",
        updated_at="2026-08-05T00:00:03+00:00",
        pretraining_success_evidence=success,
    )
    assert manifest.pretraining_success_evidence is not None
    # a from-scratch pretraining run carries the model evidence, not the adapter evidence
    assert manifest.training_success_evidence is None


def test_success_model_verified_flags_are_type_locked() -> None:
    with pytest.raises(ValidationError):
        PretrainingSuccessEvidence(
            execution=_execution(),
            output_path_verified=True,
            model_bytes_verified=False,  # cannot claim success with unverified model bytes
            artifact_integrity_verified=True,
            model_safetensors_sha256=_A,
            model_config_sha256=_B,
        )


def _pretraining_success() -> PretrainingSuccessEvidence:
    return PretrainingSuccessEvidence(
        execution=_execution(),
        output_path_verified=True,
        model_bytes_verified=True,
        artifact_integrity_verified=True,
        model_safetensors_sha256=_A,
        model_config_sha256=_B,
    )


def _adapter_success() -> TrainingSuccessEvidence:
    # A minimal valid SFT adapter success, to prove the RunManifest XOR guard (not the adapter contract).
    execution = TrainingExecutionEvidence(
        trainable_state=_trainable(),
        adapter_export_state=AdapterExportStateEvidence(
            before_sha256=_C,
            after_sha256=_D,
            tensor_count=2,
            tensor_names=["p.0", "p.1"],
            changed_tensor_count=1,
            changed_tensor_names=["p.0"],
            adapter_config_semantic_sha256=_A,
        ),
        gradient_coverage=_gradients(),
        optimizer_created=True,
        completed_optimizer_steps=1,
        step_losses=[OptimizerStepLossEvidence(optimizer_step=1, loss=3.0)],
    )
    return TrainingSuccessEvidence(
        execution=execution,
        output_path_verified=True,
        adapter_bytes_verified=True,
        artifact_integrity_verified=True,
        adapter_safetensors_sha256=_A,
        adapter_config_sha256=_B,
    )


def test_run_manifest_refuses_both_success_evidence_families() -> None:
    # honesty invariant: a run cannot be BOTH an adapter success and a full-model pretraining success.
    with pytest.raises(ValidationError, match="at most one success-evidence family"):
        RunManifest(
            run_id="run-both",
            plan_ref=Ref(id="plan-both"),
            created_at="2026-08-05T00:00:00+00:00",
            updated_at="2026-08-05T00:00:03+00:00",
            training_success_evidence=_adapter_success(),
            pretraining_success_evidence=_pretraining_success(),
        )


def test_run_manifest_allows_neither_success_evidence() -> None:
    # a prepared / running / failed run carries neither family - the XOR guard must not force one.
    manifest = RunManifest(
        run_id="run-prepared",
        plan_ref=Ref(id="plan-prepared"),
        created_at="2026-08-05T00:00:00+00:00",
        updated_at="2026-08-05T00:00:00+00:00",
    )
    assert manifest.training_success_evidence is None
    assert manifest.pretraining_success_evidence is None
