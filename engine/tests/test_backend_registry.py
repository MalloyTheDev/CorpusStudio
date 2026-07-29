"""FrameworkBackend/OrchestratorAdapter split + security-tier refusal (P0c, #483).

The security refusal is fail-closed: trust_remote_code is refused at every tier; network/download
tighten as the tier hardens. The split is APPEND-ONLY - BackendManifest is not mutated.
"""

from __future__ import annotations

from corpus_studio.platform.backend_registry import (
    reference_framework_backends,
    reference_orchestrator_adapters,
    refuse_backend_security,
)
from corpus_studio.platform.contracts import BackendManifest, BackendSecurityPosture
from corpus_studio.platform.enums import (
    AccessScope,
    AssuranceTier,
    BackendCandidateClass,
    SupportLevel,
)


def test_split_is_append_only_backend_manifest_untouched():
    # #483 is append-only: the split + security posture are NEW contracts, never fields ON
    # BackendManifest (whose digest pins backend_ref across RunPlan / ResolvedExecutionConfiguration /
    # CheckpointBoundIdentities). Mutating it would silently break the sealed reference lineage.
    fields = set(BackendManifest.model_fields)
    for smuggled in (
        "security_posture",
        "framework_ref",
        "trust_remote_code",
        "network_access",
        "download_access",
        "candidate_class",
        "model_topologies",
    ):
        assert smuggled not in fields, f"BackendManifest gained '{smuggled}' - not append-only"


def test_trust_remote_code_is_refused_at_every_tier():
    posture = BackendSecurityPosture(trust_remote_code=True)
    for tier in AssuranceTier:
        reason = refuse_backend_security(posture, tier=tier)
        assert reason is not None and "trust_remote_code" in reason


def test_network_download_scope_tightens_with_the_tier():
    clean = BackendSecurityPosture()  # all none -> passes every tier
    assert all(refuse_backend_security(clean, tier=tier) is None for tier in AssuranceTier)

    allow = BackendSecurityPosture(network_access=AccessScope.allowlisted)
    assert refuse_backend_security(allow, tier=AssuranceTier.standard) is None
    assert refuse_backend_security(allow, tier=AssuranceTier.verified) is None
    assert refuse_backend_security(allow, tier=AssuranceTier.sealed_research) is not None

    unrestricted = BackendSecurityPosture(download_access=AccessScope.unrestricted)
    assert refuse_backend_security(unrestricted, tier=AssuranceTier.standard) is None
    assert refuse_backend_security(unrestricted, tier=AssuranceTier.verified) is not None
    assert refuse_backend_security(unrestricted, tier=AssuranceTier.sealed_research) is not None


def test_reference_registries_carry_support_and_candidate_class():
    frameworks = reference_framework_backends()
    orchestrators = reference_orchestrator_adapters()
    assert frameworks and orchestrators
    assert all(isinstance(f.support_level, SupportLevel) for f in frameworks)
    assert all(isinstance(o.support_level, SupportLevel) for o in orchestrators)
    assert any(
        f.framework_id == "pytorch"
        and f.candidate_class == BackendCandidateClass.first_party
        and f.support_level == SupportLevel.workload_verified
        for f in frameworks
    )


def test_first_party_orchestrator_passes_every_tier_external_does_not():
    # the first-party corpus_studio loop declares no network/download and no trust_remote_code, so it is
    # admissible even at sealed_research; an external adapter with an allowlisted posture is not.
    corpus = next(o for o in reference_orchestrator_adapters() if o.orchestrator_id == "corpus_studio")
    assert all(refuse_backend_security(corpus.security_posture, tier=tier) is None for tier in AssuranceTier)
    external = next(o for o in reference_orchestrator_adapters() if o.orchestrator_id != "corpus_studio")
    assert (
        refuse_backend_security(external.security_posture, tier=AssuranceTier.sealed_research) is not None
    )
