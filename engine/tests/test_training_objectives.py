"""TrainingObjective registry and compatibility evidence tests (torch-free)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.platform.backends import get_backend
from corpus_studio.platform.common import Ref
from corpus_studio.platform.contracts import (
    CapabilityReport,
    EffectiveCapabilities,
    ModelDescriptor,
    ObjectiveBackendRequirement,
    ObjectiveCompatibilityAxis,
    ObjectiveCompatibilityReport,
    ObjectiveLossComponent,
    ObjectiveUpdatePolicy,
    ObjectiveVerification,
    TrainingObjective,
)
from corpus_studio.platform.enums import ObjectiveCompatibilityStatus
from corpus_studio.platform.objectives import (
    builtin_objectives,
    check_objective_compatibility,
    get_objective,
    objective_hash_for,
    validate_objective_catalog,
    verify_objective_hash,
)
from corpus_studio.schemas.registry import load_builtin_schema

runner = CliRunner()
_EXPECTED_IDS = sorted(
    [
        "pretraining",
        "continued_pretraining",
        "full_parameter_sft",
        "lora",
        "qlora",
        "other_peft",
        "chat_tuning",
        "completion_only",
        "response_only_loss",
        "dpo",
        "ipo",
        "kto",
        "orpo",
        "reward_model",
        "knowledge_distillation",
        "sequence_distillation",
        "logit_distillation",
        "rationale_distillation",
        "process_supervision",
        "verifier_training",
        "tool_use",
        "embedding",
        "reranker",
        "classifier",
        "multimodal",
        "evaluation_only",
        "merge_only",
        "conversion_only",
        "quantization_only",
    ]
)


def _dense_model(
    *, topology: str | dict[str, object] = "dense", task_class: str = "causal_lm"
) -> ModelDescriptor:
    return ModelDescriptor(
        model_id="model-1",
        source={"kind": "local", "local_path": "C:/models/model-1"},
        task_classes=[task_class],
        topology={"execution_kind": topology} if isinstance(topology, str) else topology,
        tokenizer_ref=Ref(id="tokenizer-1"),
        vocabulary={
            "output_head_rows": {"value": 32_000, "source": "config", "evidence": "declared"}
        },
    )


def _schema_evidence(schema_id: str) -> dict[str, object]:
    schema = load_builtin_schema(schema_id)
    return {
        "dataset_schema_id": schema_id,
        "dataset_schema_version": schema.version,
        "dataset_fields": {item.name: item.type for item in schema.fields},
    }


def _ready_qlora_report(
    *,
    prove_objective: bool,
    prove_quantization: bool = True,
    backend_version: str | None = "1.0.0",
) -> CapabilityReport:
    return CapabilityReport(
        backend_id="corpus_studio",
        backend_version=backend_version,
        environment_ref=Ref(id="environment-1"),
        readiness="ready",
        effective_capabilities=EffectiveCapabilities(
            adapter_methods=["qlora"],
            quantization_modes=["nf4"] if prove_quantization else [],
            objective_capabilities=(
                ["adapter_qlora", "causal_lm_sft"] if prove_objective else []
            ),
        ),
    )


def _expert_selective_objective() -> TrainingObjective:
    base = get_objective("full_parameter_sft")
    assert base is not None
    payload = base.model_dump(mode="json")
    payload["objective_id"] = "expert_selective_sft"
    payload["objective_hash"] = "0" * 64
    payload["update_policy"] = {
        "scopes": ["selected_experts"],
        "selection_mode": "selected_experts",
        "stable_expert_identity": "required",
        "exposure_tracking": "per_expert",
        "optimizer_clock": "per_expert",
        "update_window_definition": "One optimizer window per stable expert identity.",
        "starvation_gate_required_when_expert_scoped": True,
        "routing_collapse_gate_required_when_routed": True,
        "notes": [],
    }
    draft = TrainingObjective.model_validate(payload)
    return draft.model_copy(update={"objective_hash": objective_hash_for(draft)})


def test_builtin_catalog_covers_all_requested_objectives_and_is_sealed():
    objectives = builtin_objectives()
    assert [item.objective_id for item in objectives] == _EXPECTED_IDS
    assert len(objectives) == 29
    assert all(item.objective_version == "1.0.0" for item in objectives)
    assert all(verify_objective_hash(item) for item in objectives)
    qlora = next(item for item in objectives if item.objective_id == "qlora")
    conditional_artifacts = {
        item.kind.value: item for item in qlora.expected_artifacts if not item.required
    }
    assert set(conditional_artifacts) == {"expert_shards", "routing_state"}
    assert all(item.condition for item in conditional_artifacts.values())
    # Callers receive copies, not mutable registry instances.
    assert builtin_objectives()[0] is not objectives[0]


def test_catalog_rejects_duplicate_and_tampered_definitions():
    objective = get_objective("qlora")
    assert objective is not None
    with pytest.raises(ValueError, match="duplicate objective"):
        validate_objective_catalog([objective, objective.model_copy(deep=True)])
    tampered = objective.model_copy(update={"description": "changed without resealing"})
    assert objective_hash_for(tampered) != tampered.objective_hash
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_objective_catalog([tampered])


def test_objective_contract_rejects_unknown_loss_references_and_extras():
    objective = get_objective("qlora")
    assert objective is not None
    payload = objective.model_dump(mode="json")
    payload["objective_hash"] = "0" * 64
    payload["loss_components"][0]["mask_ref"] = "missing"
    with pytest.raises(ValidationError, match="unknown mask"):
        TrainingObjective.model_validate(payload)
    payload = objective.model_dump(mode="json")
    payload["bogus"] = True
    with pytest.raises(ValidationError):
        TrainingObjective.model_validate(payload)


def test_objective_capabilities_are_controlled_tokens_and_orpo_keeps_both_losses():
    with pytest.raises(ValidationError):
        ObjectiveBackendRequirement(objective_capabilities=["bad capability"])
    orpo = get_objective("orpo")
    assert orpo is not None
    assert [item.kind.value for item in orpo.loss_components] == ["odds_ratio", "cross_entropy"]
    assert {item.mask_ref for item in orpo.loss_components} == {
        "chosen_response_mask",
        "primary_mask",
    }


def test_sparse_update_policies_require_identity_and_exposure():
    with pytest.raises(ValidationError, match="stable expert identity"):
        ObjectiveUpdatePolicy(
            scopes=["selected_experts"],
            selection_mode="selected_experts",
            stable_expert_identity="not_required",
            exposure_tracking="per_expert",
            optimizer_clock="per_expert",
            update_window_definition="one exposure window",
        )
    with pytest.raises(ValidationError, match="per-expert exposure"):
        ObjectiveUpdatePolicy(
            scopes=["selected_experts"],
            selection_mode="selected_experts",
            stable_expert_identity="required",
            exposure_tracking="per_component",
            optimizer_clock="per_expert",
            update_window_definition="one exposure window",
        )
    policy = ObjectiveUpdatePolicy(
        scopes=["router"],
        selection_mode="router_only",
        stable_expert_identity="when_expert_scoped",
        exposure_tracking="per_component",
        optimizer_clock="per_component",
        update_window_definition="one routing window",
    )
    assert policy.selection_mode.value == "router_only"


@pytest.mark.parametrize(
    ("scopes", "selection_mode", "exposure_tracking", "optimizer_clock", "message"),
    [
        (["all_parameters"], "router_only", "per_component", "per_component", "router_only"),
        (["adapters", "router"], "adapter_only", "per_component", "per_component", "adapter_only"),
        (
            ["all_parameters", "task_head"],
            "task_head_only",
            "per_component",
            "per_component",
            "task_head_only",
        ),
        (
            ["adapters", "all_parameters"],
            "all",
            "per_component",
            "per_component",
            "all_parameters",
        ),
        (
            ["selected_experts"],
            "selected_experts",
            "per_expert",
            "global",
            "per-expert optimizer",
        ),
    ],
)
def test_update_policy_rejects_contradictory_scope_and_selection_claims(
    scopes, selection_mode, exposure_tracking, optimizer_clock, message
):
    with pytest.raises(ValidationError, match=message):
        ObjectiveUpdatePolicy(
            scopes=scopes,
            selection_mode=selection_mode,
            stable_expert_identity="required",
            exposure_tracking=exposure_tracking,
            optimizer_clock=optimizer_clock,
            update_window_definition="one update window",
        )


def test_verification_tiers_form_a_strict_evidence_ladder():
    with pytest.raises(ValidationError, match="validated definition"):
        ObjectiveVerification(
            definition="declared",
            implementation="functional_verified",
            hardware="not_verified",
            evidence_refs=["implementation:test"],
        )
    with pytest.raises(ValidationError, match="functionally verified implementation"):
        ObjectiveVerification(
            definition="contract_validated",
            implementation="not_verified",
            hardware="hardware_verified",
            evidence_refs=["hardware:test"],
        )
    verified = ObjectiveVerification(
        definition="contract_validated",
        implementation="functional_verified",
        hardware="hardware_verified",
        evidence_refs=["hardware:test", "implementation:test"],
    )
    assert verified.hardware.value == "hardware_verified"


def test_objective_can_keep_sparse_auxiliary_losses_separate_from_primary_loss():
    objective = get_objective("full_parameter_sft")
    assert objective is not None
    payload = objective.model_dump(mode="json")
    payload["objective_hash"] = "0" * 64
    payload["loss_components"].extend(
        [
            {
                "component_id": "load_balance_loss",
                "kind": "load_balancing",
                "construction": "Per-router expert utilization balance term.",
                "label_ref": None,
                "mask_ref": None,
                "default_weight": None,
                "reduction": "mean",
            },
            {
                "component_id": "router_z_loss",
                "kind": "router_z_loss",
                "construction": "Router-logit stabilization term.",
                "label_ref": None,
                "mask_ref": None,
                "default_weight": None,
                "reduction": "mean",
            },
        ]
    )
    payload["loss_components"].sort(key=lambda item: item["component_id"])
    parsed = TrainingObjective.model_validate(payload)
    assert [item.component_id for item in parsed.loss_components] == [
        "load_balance_loss",
        "primary_loss",
        "router_z_loss",
    ]


def test_unspecified_loss_weight_survives_exclude_none_and_objective_seal_roundtrips():
    omitted = ObjectiveLossComponent(
        component_id="router_z_loss",
        kind="router_z_loss",
        construction="Router-logit stabilization term.",
    )
    explicit = ObjectiveLossComponent(
        component_id="router_z_loss",
        kind="router_z_loss",
        construction="Router-logit stabilization term.",
        default_weight=None,
    )
    assert omitted.default_weight is None
    assert explicit == omitted
    assert ObjectiveLossComponent.model_validate_json(explicit.model_dump_json()) == explicit

    wire_payload = explicit.model_dump(mode="json", exclude_none=True)
    assert "default_weight" not in wire_payload
    assert ObjectiveLossComponent.model_validate(wire_payload) == explicit

    objective = get_objective("full_parameter_sft")
    assert objective is not None
    payload = objective.model_dump(mode="json")
    payload["objective_hash"] = "0" * 64
    payload["loss_components"].append(explicit.model_dump(mode="json"))
    payload["loss_components"].sort(key=lambda item: item["component_id"])
    unsealed = TrainingObjective.model_validate(payload)
    sealed = unsealed.model_copy(update={"objective_hash": objective_hash_for(unsealed)})

    reloaded = TrainingObjective.model_validate(
        sealed.model_dump(mode="json", exclude_none=True)
    )
    assert reloaded.loss_components[-1].default_weight is None
    assert reloaded.objective_hash == sealed.objective_hash
    assert verify_objective_hash(reloaded)

    qlora = get_objective("qlora")
    assert qlora is not None
    assert qlora.loss_components[0].default_weight == 1.0


def test_all_not_applicable_compatibility_axes_keep_a_not_applicable_overall():
    axis = ObjectiveCompatibilityAxis(status="not_applicable")
    report = ObjectiveCompatibilityReport(
        objective_ref=Ref(id="noop"),
        objective_version="1.0.0",
        dataset=axis,
        model=axis,
        backend=axis,
        overall_status="not_applicable",
    )
    assert report.overall_status == ObjectiveCompatibilityStatus.not_applicable


def test_qlora_static_backend_is_only_declared_compatible():
    objective = get_objective("qlora")
    backend = get_backend("corpus_studio")
    assert objective is not None and backend is not None
    report = check_objective_compatibility(
        objective,
        **_schema_evidence("instruction"),
        model_descriptor=_dense_model(),
        backend_manifest=backend,
    )
    assert report.dataset.status == ObjectiveCompatibilityStatus.verified_compatible
    assert report.model.status == ObjectiveCompatibilityStatus.verified_compatible
    assert report.backend.status == ObjectiveCompatibilityStatus.declared_compatible
    assert report.overall_status == ObjectiveCompatibilityStatus.declared_compatible
    assert [item.value for item in objective.backend_requirement.quantization_modes] == [
        "int4",
        "nf4",
    ]
    no_four_bit = check_objective_compatibility(
        objective,
        **_schema_evidence("instruction"),
        model_descriptor=_dense_model(),
        backend_manifest=backend.model_copy(update={"quantization_modes": []}),
    )
    assert no_four_bit.backend.status == ObjectiveCompatibilityStatus.incompatible


def test_capability_report_needs_explicit_objective_proof():
    objective = get_objective("qlora")
    backend = get_backend("corpus_studio")
    assert objective is not None and backend is not None
    base = dict(
        **_schema_evidence("instruction"),
        model_descriptor=_dense_model(),
        backend_manifest=backend,
    )
    unproven = check_objective_compatibility(
        objective, capability_report=_ready_qlora_report(prove_objective=False), **base
    )
    assert unproven.backend.status == ObjectiveCompatibilityStatus.unverified
    no_quantization_proof = check_objective_compatibility(
        objective,
        capability_report=_ready_qlora_report(
            prove_objective=True, prove_quantization=False
        ),
        **base,
    )
    assert no_quantization_proof.backend.status == ObjectiveCompatibilityStatus.unverified
    proven = check_objective_compatibility(
        objective, capability_report=_ready_qlora_report(prove_objective=True), **base
    )
    assert proven.backend.status == ObjectiveCompatibilityStatus.verified_compatible
    assert proven.overall_status == ObjectiveCompatibilityStatus.verified_compatible
    assert proven.capability_environment_ref == Ref(id="environment-1")


@pytest.mark.parametrize("backend_version", [None, "0.9.0"])
def test_capability_report_must_pin_the_selected_backend_version(backend_version):
    objective = get_objective("qlora")
    backend = get_backend("corpus_studio")
    assert objective is not None and backend is not None
    report = check_objective_compatibility(
        objective,
        **_schema_evidence("instruction"),
        model_descriptor=_dense_model(),
        backend_manifest=backend,
        capability_report=_ready_qlora_report(
            prove_objective=True,
            backend_version=backend_version,
        ),
    )
    assert report.backend.status == ObjectiveCompatibilityStatus.unverified
    assert "backend version" in " ".join(report.backend.reasons)


def test_dataset_mismatch_and_unknown_model_evidence_fail_conservatively():
    objective = get_objective("qlora")
    assert objective is not None
    wrong_schema = check_objective_compatibility(
        objective,
        **_schema_evidence("classification"),
    )
    assert wrong_schema.overall_status == ObjectiveCompatibilityStatus.incompatible
    unknown_model = check_objective_compatibility(
        objective,
        **_schema_evidence("instruction"),
        model_descriptor=_dense_model(topology="unknown"),
    )
    assert unknown_model.model.status == ObjectiveCompatibilityStatus.unverified
    assert unknown_model.overall_status == ObjectiveCompatibilityStatus.unverified


def test_dataset_version_is_required_and_must_match_the_objective():
    objective = get_objective("qlora")
    assert objective is not None
    schema = load_builtin_schema("instruction")
    fields = {item.name: item.type for item in schema.fields}
    missing = check_objective_compatibility(
        objective,
        dataset_schema_id=schema.id,
        dataset_fields=fields,
    )
    assert missing.dataset.status == ObjectiveCompatibilityStatus.unverified
    assert "version evidence" in " ".join(missing.dataset.reasons)
    mismatched = check_objective_compatibility(
        objective,
        dataset_schema_id=schema.id,
        dataset_schema_version="9.9.9",
        dataset_fields=fields,
    )
    assert mismatched.dataset.status == ObjectiveCompatibilityStatus.incompatible
    assert "is not accepted" in " ".join(mismatched.dataset.reasons)


def test_dataset_checker_selects_an_exact_version_when_ids_repeat():
    base = get_objective("qlora")
    assert base is not None
    payload = base.model_dump(mode="json")
    payload["objective_id"] = "versioned_schema_qlora"
    payload["objective_hash"] = "0" * 64
    newer = {
        **payload["dataset_inputs"][0]["variants"][-1],
        "schema_version": "0.2.0",
    }
    payload["dataset_inputs"][0]["variants"].append(newer)
    draft = TrainingObjective.model_validate(payload)
    objective = draft.model_copy(update={"objective_hash": objective_hash_for(draft)})
    report = check_objective_compatibility(
        objective,
        dataset_schema_id="instruction",
        dataset_schema_version="0.2.0",
        dataset_fields={"instruction": "text", "output": "markdown"},
    )
    assert report.dataset.status == ObjectiveCompatibilityStatus.verified_compatible


def test_multi_input_objective_requires_role_keyed_dataset_evidence():
    base = get_objective("qlora")
    assert base is not None
    payload = base.model_dump(mode="json")
    payload["objective_id"] = "multi_input_qlora"
    payload["objective_hash"] = "0" * 64
    train_input = payload["dataset_inputs"][0]
    teacher_input = {**train_input, "role": "teacher"}
    payload["dataset_inputs"] = [teacher_input, train_input]
    draft = TrainingObjective.model_validate(payload)
    objective = draft.model_copy(update={"objective_hash": objective_hash_for(draft)})
    report = check_objective_compatibility(
        objective,
        **_schema_evidence("instruction"),
    )
    assert report.dataset.status == ObjectiveCompatibilityStatus.unverified
    assert "multi-input" in " ".join(report.dataset.reasons)


def test_expert_selective_objective_rejects_a_dense_model():
    objective = _expert_selective_objective()
    report = check_objective_compatibility(
        objective,
        **_schema_evidence("instruction"),
        model_descriptor=_dense_model(),
    )
    assert report.model.status == ObjectiveCompatibilityStatus.incompatible
    assert "dense model" in " ".join(report.model.reasons)


def test_routed_expert_update_requires_semantic_routing_evidence():
    base = _expert_selective_objective()
    payload = base.model_dump(mode="json")
    payload["objective_id"] = "routed_expert_sft"
    payload["objective_hash"] = "0" * 64
    payload["update_policy"]["selection_mode"] = "routed_experts"
    draft = TrainingObjective.model_validate(payload)
    objective = draft.model_copy(update={"objective_hash": objective_hash_for(draft)})
    model = _dense_model(
        topology={
            "execution_kind": "mixture_of_experts",
            "expert_groups": [
                {
                    "group_id": "experts",
                    "layer_indices": [0],
                    "expert_count": 2,
                    "routed_expert_count": 2,
                    "expert_identity_scheme": "layer-index",
                }
            ],
        }
    )
    report = check_objective_compatibility(
        objective,
        **_schema_evidence("instruction"),
        model_descriptor=model,
    )
    assert report.model.status == ObjectiveCompatibilityStatus.unverified
    assert "semantic routing" in " ".join(report.model.reasons)


def test_planned_distillation_schema_stays_unverified():
    objective = get_objective("logit_distillation")
    assert objective is not None
    report = check_objective_compatibility(
        objective,
        dataset_schema_id="logit_distillation",
        dataset_fields={"input_ids": "list", "teacher_logits": "object"},
    )
    assert report.dataset.status == ObjectiveCompatibilityStatus.unverified


def test_reference_model_requirement_remains_unverified_without_reference_evidence():
    objective = get_objective("dpo")
    assert objective is not None
    report = check_objective_compatibility(
        objective,
        **_schema_evidence("preference"),
        model_descriptor=_dense_model(),
    )
    assert report.model.status == ObjectiveCompatibilityStatus.unverified
    assert "reference-model" in " ".join(report.model.reasons)


def test_training_objectives_cli_lists_shows_and_rejects_unknown():
    listed = runner.invoke(app, ["training-objectives", "--json"])
    assert listed.exit_code == 0
    assert [item["objective_id"] for item in json.loads(listed.stdout)] == _EXPECTED_IDS
    shown = runner.invoke(app, ["training-objectives", "qlora", "--json"])
    assert shown.exit_code == 0
    assert json.loads(shown.stdout)["objective_id"] == "qlora"
    unknown = runner.invoke(app, ["training-objectives", "does_not_exist"])
    assert unknown.exit_code == 2
    assert "Unknown training objective" in unknown.stderr


def test_training_objective_check_cli_reports_independent_axes(tmp_path):
    descriptor = tmp_path / "ModelDescriptor.json"
    descriptor.write_text(_dense_model().model_dump_json(indent=2), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "training-objective-check",
            "qlora",
            "--schema",
            "instruction",
            "--model-descriptor",
            str(descriptor),
            "--backend",
            "corpus_studio",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["dataset_schema_version"] == "0.1.0"
    assert report["dataset"]["status"] == "verified_compatible"
    assert report["model"]["status"] == "verified_compatible"
    assert report["backend"]["status"] == "declared_compatible"
    assert report["overall_status"] == "declared_compatible"


def test_objective_registry_import_is_torch_free_in_a_fresh_process():
    code = (
        "import sys; import corpus_studio.platform.objectives as o; "
        "assert len(o.builtin_objectives()) == 29; "
        "assert all(x not in sys.modules for x in "
        "('torch','transformers','trl','peft','bitsandbytes'))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
