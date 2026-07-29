"""TrainingPlan resolution + the thin registries (Training Systems P0b, #482).

A :class:`TrainingPlan` composes one entry per registry dimension + free parameters. This module:

- exposes the **thin registries** (the capability entries per dimension, each carrying a
  :class:`SupportLevel`);
- **lowers** a :class:`TrainingPlan` into one-or-more RunPlans via the planner's ``build_run_plan``,
  reproducing a direct build's ``plan_hash`` byte for byte (the resolver seals NOTHING of its own);
- runs an **advisory** cross-dimension compatibility pre-check (never the authoritative gate).

RunPlan and ResolvedExecutionConfiguration remain the sole sealing/execution authority. Control-plane
only: stdlib + platform contracts, no torch (the planner is imported lazily inside the resolver).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from corpus_studio.platform.backend_registry import (
    reference_orchestrator_adapters,
    refuse_backend_security,
)
from corpus_studio.platform.common import HashRef, Ref
from corpus_studio.platform.contracts import (
    TrainingPlan,
    TrainingPlanCompatibilityFinding,
    TrainingPlanResolution,
)
from corpus_studio.platform.enums import AssuranceTier, ParallelismKind, SupportLevel


class BackendSecurityRefused(ValueError):
    """A TrainingPlan's composed orchestrator declares a security posture the run's assurance tier
    refuses (see ``backend_registry.refuse_backend_security``). Fail-closed: resolution never proceeds."""

if TYPE_CHECKING:  # pragma: no cover - typing only; the planner is imported lazily at call time.
    from corpus_studio.platform.contracts import (
        CapabilityReport,
        EnvironmentProfile,
        ParameterAccountingReport,
        PhysicalExecutionSpec,
        StorageProfile,
    )


@dataclass(frozen=True)
class RegistryEntry:
    """One capability entry in a thin registry: its name + how far it is proven (:class:`SupportLevel`)
    + a short note. Support levels are conservative on purpose - "installed is never supported"; a
    higher level requires the evidence its name implies."""

    name: str
    support_level: SupportLevel
    note: str = ""


# The thin registries: valid entries per registry dimension, each carrying its SupportLevel (#481).
# Only the PyTorch + corpus_studio reference path has a measured workload on this host; everything else
# is declared / config-generation intent until its exact stack is probed and workload-verified.


def framework_registry() -> tuple[RegistryEntry, ...]:
    return (
        RegistryEntry("pytorch", SupportLevel.workload_verified, "the reference framework; measured"),
        RegistryEntry("jax", SupportLevel.declared),
        RegistryEntry("tf_keras", SupportLevel.declared),
        RegistryEntry("mlx", SupportLevel.declared),
    )


def orchestrator_registry() -> tuple[RegistryEntry, ...]:
    return (
        RegistryEntry("corpus_studio", SupportLevel.workload_verified, "first-party TRL/PEFT QLoRA loop"),
        RegistryEntry("unsloth", SupportLevel.config_generation_only),
        RegistryEntry("torchtune", SupportLevel.declared),
        RegistryEntry("axolotl", SupportLevel.config_generation_only),
        RegistryEntry("megatron", SupportLevel.declared),
    )


def model_topology_registry() -> tuple[RegistryEntry, ...]:
    return (
        RegistryEntry("dense", SupportLevel.workload_verified, "measured dense QLoRA"),
        RegistryEntry("moe", SupportLevel.declared, "static topology inspection only; no runtime proof"),
    )


def update_method_registry() -> tuple[RegistryEntry, ...]:
    return (
        RegistryEntry("qlora", SupportLevel.workload_verified),
        RegistryEntry("lora", SupportLevel.probed),
        RegistryEntry("full_finetune", SupportLevel.declared),
        RegistryEntry("dora", SupportLevel.declared),
        RegistryEntry("ia3", SupportLevel.declared),
    )


def parallelism_registry() -> tuple[RegistryEntry, ...]:
    return (
        RegistryEntry("data", SupportLevel.probed, "single-rank data parallel"),
        RegistryEntry("tensor", SupportLevel.declared),
        RegistryEntry("pipeline", SupportLevel.declared),
        RegistryEntry("expert", SupportLevel.declared),
        RegistryEntry("sequence", SupportLevel.declared),
        RegistryEntry("context", SupportLevel.declared),
    )


def precision_registry() -> tuple[RegistryEntry, ...]:
    return (
        RegistryEntry("bf16", SupportLevel.workload_verified),
        RegistryEntry("fp16", SupportLevel.declared),
        RegistryEntry("fp32", SupportLevel.probed),
        RegistryEntry("fp8", SupportLevel.declared),
    )


def hardware_target_registry() -> tuple[RegistryEntry, ...]:
    return (
        RegistryEntry("cuda", SupportLevel.workload_verified),
        RegistryEntry("rocm", SupportLevel.declared),
        RegistryEntry("mps", SupportLevel.declared),
        RegistryEntry("cpu", SupportLevel.probed, "cpu-toy smoke only"),
    )


def checkpoint_strategy_registry() -> tuple[RegistryEntry, ...]:
    return (
        RegistryEntry("adapter_only", SupportLevel.workload_verified),
        RegistryEntry("full_state", SupportLevel.declared),
        RegistryEntry("sharded", SupportLevel.declared),
        RegistryEntry("distcp", SupportLevel.declared),
    )


def evaluation_profile_registry() -> tuple[RegistryEntry, ...]:
    return (
        RegistryEntry("keyword_overlap", SupportLevel.probed, "lexical proxy, not a quality judgment"),
        RegistryEntry("schema_conformance", SupportLevel.probed),
        RegistryEntry("llm_judge", SupportLevel.config_generation_only),
    )


def training_preset_registry() -> tuple[RegistryEntry, ...]:
    return (
        RegistryEntry("qlora-qwen-seq4096", SupportLevel.workload_verified, "the measured seq-4096 recipe"),
    )


THIN_REGISTRIES = {
    "framework": framework_registry,
    "orchestrator": orchestrator_registry,
    "model_topology": model_topology_registry,
    "update_method": update_method_registry,
    "parallelism": parallelism_registry,
    "precision": precision_registry,
    "hardware_target": hardware_target_registry,
    "checkpoint_strategy": checkpoint_strategy_registry,
    "evaluation_profile": evaluation_profile_registry,
    "preset": training_preset_registry,  # keyed by the composition FIELD name (not the registry name)
}


def training_plan_precheck(training_plan: TrainingPlan) -> list[TrainingPlanCompatibilityFinding]:
    """Advisory cross-dimension compatibility checks BEFORE resolution. UX hints ONLY - they never gate
    resolution or execution (the authoritative gate stays the planner's declared-and-proven capability
    match plus the exact execution-tuple probe). Findings are sorted by code for determinism."""
    findings: list[TrainingPlanCompatibilityFinding] = []
    comp = training_plan.composition
    params = training_plan.parameters

    if params.adapter_method is not None and params.adapter_method != comp.update_method.value:
        findings.append(
            TrainingPlanCompatibilityFinding(
                code="update_method_mismatch",
                message=(
                    f"composition update_method '{comp.update_method.value}' does not match parameters "
                    f"adapter_method '{params.adapter_method}'; the parameters are what resolve"
                ),
            )
        )
    if comp.orchestrator != params.backend:
        findings.append(
            TrainingPlanCompatibilityFinding(
                code="backend_mismatch",
                message=(
                    f"composition orchestrator '{comp.orchestrator}' does not match parameters backend "
                    f"'{params.backend}'; the parameters backend is what runs"
                ),
            )
        )
    if comp.update_method.value == "qlora" and comp.quantization.value == "none":
        findings.append(
            TrainingPlanCompatibilityFinding(
                code="qlora_without_quantization",
                message="update_method 'qlora' with quantization 'none' - qlora requires a 4-bit base",
            )
        )
    if comp.model_topology == "moe" and ParallelismKind.expert not in comp.parallelism:
        findings.append(
            TrainingPlanCompatibilityFinding(
                code="moe_without_expert_parallelism",
                severity="info",
                message="model_topology 'moe' without expert parallelism (fine for small resident experts)",
            )
        )
    return sorted(findings, key=lambda finding: finding.code)


def resolve_training_plan(
    training_plan: TrainingPlan,
    *,
    profile: EnvironmentProfile,
    capabilities: CapabilityReport,
    dataset_ref: Ref,
    plan_id: str,
    assurance_tier: AssuranceTier = AssuranceTier.standard,
    environment_ref: Ref | None = None,
    parameter_accounting: ParameterAccountingReport | None = None,
    physical_execution: PhysicalExecutionSpec | None = None,
    storage_profile: StorageProfile | None = None,
    allow_marginal_storage: bool = False,
    allow_unknown_storage: bool = False,
    now: str | None = None,
) -> TrainingPlanResolution:
    """Lower a :class:`TrainingPlan` into a RunPlan via the authoritative planner, then wrap it in a
    :class:`TrainingPlanResolution`. The resolver builds the SAME ``PlannerConstraints`` a direct caller
    would and calls ``build_run_plan``, so the resolved RunPlan carries the SAME ``plan_hash`` as a
    direct build - the resolver seals nothing. Raises the planner's ``PlannerError`` when the host
    cannot honor the request. Fail-closed SECURITY GATE: if the composed orchestrator's declared
    security posture exceeds ``assurance_tier``, resolution is REFUSED with
    :class:`BackendSecurityRefused` before any RunPlan is built."""
    from corpus_studio.platform.planner import PlannerConstraints, build_run_plan  # noqa: PLC0415

    adapter = {a.orchestrator_id: a for a in reference_orchestrator_adapters()}.get(
        training_plan.composition.orchestrator
    )
    if adapter is not None:
        refusal = refuse_backend_security(adapter.security_posture, tier=assurance_tier)
        if refusal is not None:
            raise BackendSecurityRefused(
                f"orchestrator '{adapter.orchestrator_id}' refused at assurance tier "
                f"'{assurance_tier.value}': {refusal}"
            )
    constraints = PlannerConstraints(
        **training_plan.parameters.model_dump(exclude={"contract_version"})
    )
    run_plan = build_run_plan(
        profile=profile,
        capabilities=capabilities,
        dataset_ref=dataset_ref,
        constraints=constraints,
        plan_id=plan_id,
        environment_ref=environment_ref,
        parameter_accounting=parameter_accounting,
        physical_execution=physical_execution,
        storage_profile=storage_profile,
        allow_marginal_storage=allow_marginal_storage,
        allow_unknown_storage=allow_unknown_storage,
        now=now,
    )
    return TrainingPlanResolution(
        plan_intent_id=training_plan.plan_intent_id,
        composition=training_plan.composition,
        run_plan_refs=(Ref(id=run_plan.plan_id, hash=HashRef(value=run_plan.plan_hash)),),
        precheck_findings=tuple(training_plan_precheck(training_plan)),
        resolved_at=now,
    )
