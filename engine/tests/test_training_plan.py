"""TrainingPlan composition + resolver + thin registries (Training Systems P0b, #482).

The load-bearing property is that the resolver adds NO sealing: lowering a TrainingPlan reproduces the
exact plan_hash a direct build_run_plan produces. The ready-capability fixture is reused from the
planner test (pytest prepend import) rather than duplicated.
"""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import ValidationError

from corpus_studio.platform.common import HashRef, Ref
from corpus_studio.platform.contracts import (
    TrainingPlan,
    TrainingPlanComposition,
    TrainingPlanParameters,
    TrainingPlanResolution,
)
from corpus_studio.platform.enums import AssuranceTier, SupportLevel
from corpus_studio.platform.planner import PlannerConstraints, build_run_plan
from corpus_studio.platform.training_plan import (
    BackendSecurityRefused,
    THIN_REGISTRIES,
    framework_registry,
    resolve_training_plan,
    training_plan_precheck,
)
from test_platform_planner import _NOW, _profile, _report

_DATASET_REF = Ref(id="ds-1", hash=HashRef(value="d" * 64))


def _params(**over) -> TrainingPlanParameters:
    base = {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "dataset_path": "data/examples.jsonl",
        "model_revision": "1" * 40,
        "dataset_content_sha256": "d" * 64,
    }
    base.update(over)
    return TrainingPlanParameters(**base)


def _training_plan(**param_over) -> TrainingPlan:
    return TrainingPlan(
        plan_intent_id="intent-1",
        composition=TrainingPlanComposition(objective_id="qlora-sft"),
        parameters=_params(**param_over),
    )


def test_resolver_reproduces_the_direct_build_run_plan_hash():
    # THE GATE: lowering a TrainingPlan produces the SAME plan_hash as a direct build_run_plan with the
    # equivalent PlannerConstraints. The resolver seals nothing of its own.
    profile, caps = _profile(cc_major=12), _report()
    plan = _training_plan()
    resolution = resolve_training_plan(
        plan, profile=profile, capabilities=caps, dataset_ref=_DATASET_REF, plan_id="p1", now=_NOW
    )
    direct = build_run_plan(
        profile=profile,
        capabilities=caps,
        dataset_ref=_DATASET_REF,
        constraints=PlannerConstraints(**plan.parameters.model_dump(exclude={"contract_version"})),
        plan_id="p1",
        now=_NOW,
    )
    assert len(resolution.run_plan_refs) == 1
    assert resolution.run_plan_refs[0].hash is not None
    assert resolution.run_plan_refs[0].hash.value == direct.plan_hash


def test_parameters_mirror_planner_constraints_field_for_field():
    # The resolver copies parameters -> PlannerConstraints verbatim, so the two MUST stay in lockstep.
    constraint_fields = {f.name for f in dataclasses.fields(PlannerConstraints)}
    param_fields = set(TrainingPlanParameters.model_fields) - {"contract_version"}
    assert param_fields == constraint_fields


def test_training_plan_carries_no_sealed_execution_hash():
    # Invariant 1: a TrainingPlan is pre-resolution intent; it never carries a sealed execution field.
    fields = set(TrainingPlan.model_fields)
    assert "plan_hash" not in fields
    assert "configuration_hash" not in fields
    with pytest.raises(ValidationError):  # extra=forbid: one cannot be smuggled in either
        TrainingPlan(
            plan_intent_id="i",
            composition=TrainingPlanComposition(objective_id="o"),
            parameters=_params(),
            plan_hash="0" * 64,
        )


def test_precheck_is_advisory_only():
    # Invariant 2: cross-dimension checks are UX hints, never a gate. A mismatch is reported, not raised.
    plan = _training_plan(adapter_method="lora")  # composition update_method defaults to qlora
    findings = training_plan_precheck(plan)
    assert "update_method_mismatch" in {f.code for f in findings}
    assert all(f.severity in {"info", "warning"} for f in findings)  # never 'block'


def test_precheck_surfaces_the_backend_tuple_support():
    # #485 wired into the pre-check: the proven first-party tuple (pytorch x corpus_studio) draws NO
    # tuple finding, but a merely config-generation tuple is flagged (info) - proving another tuple
    # never implies this one.
    proven = {f.code for f in training_plan_precheck(_training_plan())}
    assert "backend_tuple_not_workload_verified" not in proven
    assert "backend_tuple_unresolvable" not in proven
    unsloth = TrainingPlan(
        plan_intent_id="i",
        composition=TrainingPlanComposition(objective_id="o", orchestrator="unsloth"),
        parameters=_params(backend="unsloth"),
    )
    findings = {f.code: f.severity for f in training_plan_precheck(unsloth)}
    assert findings.get("backend_tuple_not_workload_verified") == "info"


def test_precheck_flags_an_unresolvable_backend_tuple():
    # corpus_studio binds pytorch, not jax -> the framework x orchestrator tuple is unresolvable.
    mismatch = TrainingPlan(
        plan_intent_id="i",
        composition=TrainingPlanComposition(
            objective_id="o", framework="jax", orchestrator="corpus_studio"
        ),
        parameters=_params(),
    )
    assert "backend_tuple_unresolvable" in {f.code for f in training_plan_precheck(mismatch)}


def test_framework_and_orchestrator_registries_derive_from_the_backend_registry():
    # Single source of truth (#485): the thin framework/orchestrator registries mirror the canonical
    # backend registry EXACTLY, so their support levels can never drift from the resolver's admission
    # gate and an orchestrator the resolver would refuse as unknown is not falsely advertised.
    from corpus_studio.platform.backend_registry import (
        reference_framework_backends,
        reference_orchestrator_adapters,
    )
    from corpus_studio.platform.training_plan import framework_registry, orchestrator_registry

    assert {(e.name, e.support_level) for e in framework_registry()} == {
        (f.framework_id, f.support_level) for f in reference_framework_backends()
    }
    assert {(e.name, e.support_level) for e in orchestrator_registry()} == {
        (o.orchestrator_id, o.support_level) for o in reference_orchestrator_adapters()
    }
    # torchtune / megatron are NOT vetted in the backend registry, so they are no longer advertised.
    assert "torchtune" not in {e.name for e in orchestrator_registry()}


def test_resolution_attaches_precheck_findings_without_blocking():
    profile, caps = _profile(cc_major=12), _report()
    plan = _training_plan()
    resolution = resolve_training_plan(
        plan, profile=profile, capabilities=caps, dataset_ref=_DATASET_REF, plan_id="p1", now=_NOW
    )
    assert isinstance(resolution, TrainingPlanResolution)
    assert isinstance(resolution.precheck_findings, tuple)  # carried, advisory


def test_thin_registries_carry_support_levels():
    for name, registry in THIN_REGISTRIES.items():
        entries = registry()
        assert entries, f"registry {name} is empty"
        assert all(isinstance(entry.support_level, SupportLevel) for entry in entries)
    # the reference framework is the one with a measured workload
    assert any(
        entry.name == "pytorch" and entry.support_level == SupportLevel.workload_verified
        for entry in framework_registry()
    )


def test_thin_registry_keys_are_composition_field_names():
    # #742 review: registries are keyed by the COMPOSITION FIELD name so a consumer can do
    # THIN_REGISTRIES[field] without special-casing. Guard against drift (e.g. 'training_preset' vs
    # 'preset'): every registry key must be an actual TrainingPlanComposition field.
    assert set(THIN_REGISTRIES) <= set(TrainingPlanComposition.model_fields)


def test_resolution_refs_must_carry_the_sealed_plan_hash():
    # The resolution POINTS at sealed RunPlans; it never re-seals. A ref without its hash is refused.
    with pytest.raises(ValidationError):
        TrainingPlanResolution(
            plan_intent_id="i",
            composition=TrainingPlanComposition(objective_id="o"),
            run_plan_refs=(Ref(id="rp-1"),),
        )


def test_resolver_refuses_an_over_tier_backend_security_posture():
    # #483 wiring: the security refusal is REACHABLE from the admission path, not just its own unit
    # test. unsloth declares allowlisted network/download; sealed_research permits only 'none', so the
    # resolver refuses fail-closed BEFORE any RunPlan is built (an unreachable control is not a control).
    plan = TrainingPlan(
        plan_intent_id="intent-1",
        composition=TrainingPlanComposition(objective_id="qlora-sft", orchestrator="unsloth"),
        parameters=_params(),
    )
    with pytest.raises(BackendSecurityRefused, match="sealed_research"):
        resolve_training_plan(
            plan,
            profile=_profile(cc_major=12),
            capabilities=_report(),
            dataset_ref=_DATASET_REF,
            plan_id="p1",
            assurance_tier=AssuranceTier.sealed_research,
        )


def test_resolver_admits_the_clean_first_party_backend_at_every_tier():
    # the default corpus_studio orchestrator declares a clean posture, so it resolves at every tier.
    profile, caps = _profile(cc_major=12), _report()
    for tier in AssuranceTier:
        resolution = resolve_training_plan(
            _training_plan(),
            profile=profile,
            capabilities=caps,
            dataset_ref=_DATASET_REF,
            plan_id="p1",
            now=_NOW,
            assurance_tier=tier,
        )
        assert isinstance(resolution, TrainingPlanResolution)


def test_resolver_refuses_an_unknown_orchestrator_fail_closed():
    # #744 review: a fail-closed gate must not silently bypass an orchestrator it cannot vet.
    plan = TrainingPlan(
        plan_intent_id="intent-1",
        composition=TrainingPlanComposition(objective_id="qlora-sft", orchestrator="mystery-backend"),
        parameters=_params(),
    )
    with pytest.raises(BackendSecurityRefused, match="not in the backend registry"):
        resolve_training_plan(
            plan,
            profile=_profile(cc_major=12),
            capabilities=_report(),
            dataset_ref=_DATASET_REF,
            plan_id="p1",
        )
