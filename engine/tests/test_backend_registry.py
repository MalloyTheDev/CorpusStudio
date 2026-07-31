"""FrameworkBackend/OrchestratorAdapter split + security-tier refusal (P0c, #483).

The security refusal is fail-closed: trust_remote_code is refused at every tier; network/download
tighten as the tier hardens. The split is APPEND-ONLY - BackendManifest is not mutated.
"""

from __future__ import annotations

from corpus_studio.platform.backend_registry import (
    reference_framework_backends,
    reference_orchestrator_adapters,
    refuse_backend_security,
    resolve_backend_binding,
    select_default_binding,
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


# ---- capability-tuple resolver (P1, #485) -----------------------------------


def test_resolve_binding_first_party_tuple_is_workload_verified_and_default_eligible():
    r = resolve_backend_binding("pytorch", "corpus_studio", tier=AssuranceTier.standard)
    assert r.support_level is SupportLevel.workload_verified
    assert r.admissible and r.default_eligible and r.refusal_reason is None


def test_tuple_support_is_the_weakest_member_proving_one_never_implies_another():
    # pytorch (framework) is workload_verified, but unsloth (orchestrator) is only
    # config_generation_only, so the TUPLE is config_generation_only - proving pytorch x corpus_studio
    # never lifts pytorch x unsloth.
    strong = resolve_backend_binding("pytorch", "corpus_studio", tier=AssuranceTier.standard)
    weak = resolve_backend_binding("pytorch", "unsloth", tier=AssuranceTier.standard)
    assert strong.support_level is SupportLevel.workload_verified and strong.default_eligible
    assert weak.support_level is SupportLevel.config_generation_only
    assert weak.admissible and not weak.default_eligible  # may generate config, never a silent default


def test_binding_refused_when_security_posture_exceeds_the_tier():
    # unsloth declares allowlisted network/download; sealed_research permits none -> not admissible.
    r = resolve_backend_binding("pytorch", "unsloth", tier=AssuranceTier.sealed_research)
    assert not r.admissible and not r.default_eligible
    assert r.refusal_reason and "sealed_research" in r.refusal_reason


def test_binding_fails_closed_on_unknown_member_or_framework_mismatch():
    unknown_f = resolve_backend_binding("nope", "corpus_studio", tier=AssuranceTier.standard)
    assert not unknown_f.admissible and unknown_f.support_level is SupportLevel.refused
    assert "unknown framework" in (unknown_f.refusal_reason or "")
    unknown_o = resolve_backend_binding("pytorch", "nope", tier=AssuranceTier.standard)
    assert "unknown orchestrator" in (unknown_o.refusal_reason or "")
    # corpus_studio binds pytorch, not jax -> a mismatched binding is refused.
    mismatch = resolve_backend_binding("jax", "corpus_studio", tier=AssuranceTier.standard)
    assert not mismatch.admissible and mismatch.support_level is SupportLevel.refused
    assert "binds framework 'pytorch'" in (mismatch.refusal_reason or "")


def test_default_binding_is_evidence_selected_across_tiers():
    # The only workload-verified reference tuple is pytorch x corpus_studio; it is the default wherever
    # the tier admits it (corpus_studio declares no network/download, so even sealed_research admits it).
    for tier in (AssuranceTier.standard, AssuranceTier.verified, AssuranceTier.sealed_research):
        default = select_default_binding(tier=tier)
        assert default is not None
        assert (default.framework_id, default.orchestrator_id) == ("pytorch", "corpus_studio")
        assert default.support_level is SupportLevel.workload_verified


def test_default_binding_is_none_when_no_tuple_is_workload_verified(monkeypatch):
    # Fail closed: a stack with no workload-verified tuple has NO default (never a guessed one).
    import corpus_studio.platform.backend_registry as reg

    only_config = tuple(
        o for o in reg.reference_orchestrator_adapters() if o.orchestrator_id == "unsloth"
    )
    monkeypatch.setattr(reg, "reference_orchestrator_adapters", lambda: only_config)
    assert reg.select_default_binding(tier=AssuranceTier.standard) is None


def test_default_binding_is_none_when_the_verified_tuple_is_security_inadmissible(monkeypatch):
    # Evidence alone is not enough - the tier's security gate must also admit the tuple. A
    # workload-verified tuple whose posture the tier REFUSES is not default-eligible, so there is no
    # default at that tier.
    import corpus_studio.platform.backend_registry as reg

    patched = tuple(
        o.model_copy(
            update={"security_posture": BackendSecurityPosture(network_access=AccessScope.allowlisted)}
        )
        if o.orchestrator_id == "corpus_studio"
        else o
        for o in reg.reference_orchestrator_adapters()
    )
    monkeypatch.setattr(reg, "reference_orchestrator_adapters", lambda: patched)
    # standard admits an allowlisted posture -> still a default; sealed_research refuses it -> none.
    assert reg.select_default_binding(tier=AssuranceTier.standard) is not None
    assert reg.select_default_binding(tier=AssuranceTier.sealed_research) is None
