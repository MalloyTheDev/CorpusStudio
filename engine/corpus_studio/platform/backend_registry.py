"""FrameworkBackend / OrchestratorAdapter registries + the security-tier refusal (P0c, #483).

Splits the conflated ``BackendManifest`` into a ``FrameworkBackend`` x ``OrchestratorAdapter`` binding
(docs/TRAINING_BACKEND_REGISTRY.md) and adds the assurance-tier SECURITY GATE: a backend whose declared
security posture exceeds the run's assurance tier is REFUSED before it can run. Additive + append-only
- ``BackendManifest`` and its digest are untouched. Control-plane only: stdlib + platform contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

from corpus_studio.platform.contracts import (
    BackendSecurityPosture,
    FrameworkBackend,
    OrchestratorAdapter,
)
from corpus_studio.platform.enums import (
    AccessScope,
    AssuranceTier,
    BackendCandidateClass,
    DeviceKind,
    OperatingSystem,
    SupportLevel,
)

# The maximum network/download scope each assurance tier permits. ``sealed_research`` demands full
# reproducibility (no network/download); ``standard`` is the permissive product tier.
_MAX_ACCESS: dict[AssuranceTier, AccessScope] = {
    AssuranceTier.standard: AccessScope.unrestricted,
    AssuranceTier.verified: AccessScope.allowlisted,
    AssuranceTier.sealed_research: AccessScope.none,
}
_ACCESS_ORDER: dict[AccessScope, int] = {
    AccessScope.none: 0,
    AccessScope.allowlisted: 1,
    AccessScope.unrestricted: 2,
}


def refuse_backend_security(posture: BackendSecurityPosture, *, tier: AssuranceTier) -> str | None:
    """Return a refusal reason if a backend's DECLARED security posture exceeds what ``tier`` permits,
    else ``None``. Fail-closed by design (the TrainingPlan resolver calls this before admitting a
    composed backend - see ``training_plan.resolve_training_plan``):

    * ``trust_remote_code=True`` is arbitrary code execution and is refused at EVERY tier;
    * ``network_access`` / ``download_access`` may not exceed the tier's maximum scope
      (standard -> unrestricted, verified -> allowlisted, sealed_research -> none).
    """
    if posture.trust_remote_code:
        return (
            "backend declares trust_remote_code=True (arbitrary code execution), refused at every "
            f"assurance tier (including '{tier.value}')"
        )
    allowed = _MAX_ACCESS[tier]
    for label, scope in (
        ("network_access", posture.network_access),
        ("download_access", posture.download_access),
    ):
        if _ACCESS_ORDER[scope] > _ACCESS_ORDER[allowed]:
            return (
                f"backend declares {label}='{scope.value}', refused at assurance tier '{tier.value}' "
                f"(maximum permitted: '{allowed.value}')"
            )
    return None


def reference_framework_backends() -> tuple[FrameworkBackend, ...]:
    """The declared framework substrates. Only PyTorch is workload-verified on this host; the rest are
    declared / managed-adapter candidates until their exact stack is probed."""
    return (
        FrameworkBackend(
            framework_id="pytorch",
            display_name="PyTorch",
            supported_os=(OperatingSystem.linux, OperatingSystem.wsl, OperatingSystem.windows),
            supported_devices=(DeviceKind.cuda, DeviceKind.cpu),
            model_topologies=("dense", "moe"),
            candidate_class=BackendCandidateClass.first_party,
            support_level=SupportLevel.workload_verified,
        ),
        FrameworkBackend(
            framework_id="jax",
            display_name="JAX",
            candidate_class=BackendCandidateClass.managed_adapter,
            support_level=SupportLevel.declared,
        ),
        FrameworkBackend(
            framework_id="mlx",
            display_name="MLX",
            candidate_class=BackendCandidateClass.managed_adapter,
            support_level=SupportLevel.declared,
        ),
        FrameworkBackend(
            framework_id="tf_keras",
            display_name="TensorFlow / Keras",
            candidate_class=BackendCandidateClass.defer,
            support_level=SupportLevel.declared,
        ),
    )


def reference_orchestrator_adapters() -> tuple[OrchestratorAdapter, ...]:
    """The declared training-loop drivers. ``corpus_studio`` (first-party, measured) declares NO
    network / download and no ``trust_remote_code``; the external adapters declare a posture the planner
    refuses by tier."""
    return (
        OrchestratorAdapter(
            orchestrator_id="corpus_studio",
            display_name="CorpusStudio first-party TRL/PEFT loop",
            framework_ref="pytorch",
            candidate_class=BackendCandidateClass.first_party,
            security_posture=BackendSecurityPosture(),  # all none; trust_remote_code False
            support_level=SupportLevel.workload_verified,
        ),
        OrchestratorAdapter(
            orchestrator_id="unsloth",
            display_name="Unsloth",
            framework_ref="pytorch",
            candidate_class=BackendCandidateClass.config_export_only,
            security_posture=BackendSecurityPosture(
                network_access=AccessScope.allowlisted, download_access=AccessScope.allowlisted
            ),
            support_level=SupportLevel.config_generation_only,
        ),
        OrchestratorAdapter(
            orchestrator_id="axolotl",
            display_name="Axolotl",
            framework_ref="pytorch",
            candidate_class=BackendCandidateClass.managed_adapter,
            security_posture=BackendSecurityPosture(
                network_access=AccessScope.allowlisted, download_access=AccessScope.allowlisted
            ),
            support_level=SupportLevel.config_generation_only,
        ),
    )


# --------------------------------------------------------------------------------------------------
# Capability-tuple resolver (P1, #485): bind a FrameworkBackend x OrchestratorAdapter and resolve its
# EVIDENCE-SELECTED support - never inferring one tuple's support from another.
# --------------------------------------------------------------------------------------------------

# The SupportLevel ladder, weakest to strongest. ``refused`` is a terminal fail-closed state, not a
# rung; a tuple is only as proven as its WEAKEST member.
_SUPPORT_LADDER: tuple[SupportLevel, ...] = (
    SupportLevel.declared,
    SupportLevel.config_generation_only,
    SupportLevel.installed,
    SupportLevel.probed,
    SupportLevel.workload_verified,
    SupportLevel.production_supported,
)
_SUPPORT_RANK: dict[SupportLevel, int] = {level: rank for rank, level in enumerate(_SUPPORT_LADDER)}


@dataclass(frozen=True)
class BackendBindingResolution:
    """The resolution of one ``FrameworkBackend`` x ``OrchestratorAdapter`` capability tuple (#485).

    ``support_level`` is the WEAKEST of the two members (proving one member - or a sibling tuple - never
    lifts the other); ``admissible`` is False when a member is unknown / framework-mismatched / REFUSED
    or the orchestrator's security posture exceeds the tier; ``default_eligible`` additionally requires
    WORKLOAD_VERIFIED (or higher) evidence - no measured workload is ever a silent default."""

    framework_id: str
    orchestrator_id: str
    support_level: SupportLevel
    admissible: bool
    default_eligible: bool
    refusal_reason: str | None = None


def _tuple_support_level(
    framework: FrameworkBackend, orchestrator: OrchestratorAdapter
) -> SupportLevel:
    if SupportLevel.refused in (framework.support_level, orchestrator.support_level):
        return SupportLevel.refused
    return min(
        (framework.support_level, orchestrator.support_level),
        key=lambda level: _SUPPORT_RANK[level],
    )


def resolve_backend_binding(
    framework_id: str, orchestrator_id: str, *, tier: AssuranceTier
) -> BackendBindingResolution:
    """Bind a ``FrameworkBackend`` x ``OrchestratorAdapter`` into a capability tuple and resolve its
    evidence-selected support, admissibility (the assurance-tier security gate), and
    default-eligibility.

    Fails closed: an unknown member, a framework mismatch (the orchestrator binds a different
    framework), a REFUSED member, or a security-posture violation is NOT admissible. Proving one tuple
    never implies another - each binding is resolved from ITS OWN two members' declared/measured
    SupportLevel, never inferred from a sibling tuple."""
    framework = next(
        (f for f in reference_framework_backends() if f.framework_id == framework_id), None
    )
    orchestrator = next(
        (o for o in reference_orchestrator_adapters() if o.orchestrator_id == orchestrator_id), None
    )
    # Structural fail-closed checks -> a refused, inadmissible tuple.
    if framework is None:
        structural: str | None = f"unknown framework '{framework_id}'"
    elif orchestrator is None:
        structural = f"unknown orchestrator '{orchestrator_id}'"
    elif orchestrator.framework_ref != framework_id:
        structural = (
            f"orchestrator '{orchestrator_id}' binds framework '{orchestrator.framework_ref}', "
            f"not '{framework_id}'"
        )
    else:
        structural = None
    if framework is None or orchestrator is None or structural is not None:
        return BackendBindingResolution(
            framework_id, orchestrator_id, SupportLevel.refused, False, False, structural
        )
    # Both members known + correctly bound: the tuple's evidence (weakest member) + the security gate.
    support = _tuple_support_level(framework, orchestrator)
    reason = refuse_backend_security(orchestrator.security_posture, tier=tier)
    if reason is None and support is SupportLevel.refused:
        reason = f"a member of {framework_id} x {orchestrator_id} is REFUSED on this stack"
    admissible = reason is None
    default_eligible = admissible and (
        _SUPPORT_RANK[support] >= _SUPPORT_RANK[SupportLevel.workload_verified]
    )
    return BackendBindingResolution(
        framework_id, orchestrator_id, support, admissible, default_eligible, reason
    )


def select_default_binding(*, tier: AssuranceTier) -> BackendBindingResolution | None:
    """Evidence-select the DEFAULT framework x orchestrator tuple for ``tier``: the admissible binding
    with the strongest support, but only if that support is WORKLOAD_VERIFIED or higher. No such tuple
    -> ``None`` (never a guessed default). On the reference stack only pytorch x corpus_studio is
    workload-verified, so it is the default wherever the tier admits it."""
    eligible: list[BackendBindingResolution] = []
    for orchestrator in reference_orchestrator_adapters():
        binding = resolve_backend_binding(
            orchestrator.framework_ref, orchestrator.orchestrator_id, tier=tier
        )
        if binding.default_eligible:
            eligible.append(binding)
    if not eligible:
        return None
    return max(eligible, key=lambda binding: _SUPPORT_RANK[binding.support_level])
