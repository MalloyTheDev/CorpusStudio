"""FrameworkBackend / OrchestratorAdapter registries + the security-tier refusal (P0c, #483).

Splits the conflated ``BackendManifest`` into a ``FrameworkBackend`` x ``OrchestratorAdapter`` binding
(docs/TRAINING_BACKEND_REGISTRY.md) and adds the assurance-tier SECURITY GATE: a backend whose declared
security posture exceeds the run's assurance tier is REFUSED before it can run. Additive + append-only
- ``BackendManifest`` and its digest are untouched. Control-plane only: stdlib + platform contracts.
"""

from __future__ import annotations

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
