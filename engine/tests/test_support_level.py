"""Property tests for the SupportLevel rollup + its lossy partial projection (P0a, #481).

The projection must COEXIST with the shipped ladders (never replace them), stay anti-collapse (never
report a level higher than the proven axes justify), and CARRY the unproven axes instead of promoting
them. The property test is exhaustive over every axis combination the ObjectiveVerification validator
accepts.
"""

from __future__ import annotations

import itertools

from pydantic import ValidationError

from corpus_studio.platform.contracts import ObjectiveVerification
from corpus_studio.platform.enums import (
    ObjectiveVerificationStatus,
    RecipeVerification,
    SupportLevel,
    VerificationOutcome,
)
from corpus_studio.platform.support_level import (
    SupportLevelRollup,
    project_objective_verification,
)

_FUNCTIONALLY_PROVEN = {
    ObjectiveVerificationStatus.functional_verified,
    ObjectiveVerificationStatus.hardware_verified,
}
_UNPROVEN = {ObjectiveVerificationStatus.not_verified, ObjectiveVerificationStatus.not_applicable}
_NET_NEW = {
    SupportLevel.config_generation_only,
    SupportLevel.installed,
    SupportLevel.production_supported,
    SupportLevel.refused,
}


def test_support_level_has_the_seven_states_in_evidence_order():
    assert [s.value for s in SupportLevel] == [
        "declared",
        "config_generation_only",
        "installed",
        "probed",
        "workload_verified",
        "production_supported",
        "refused",
    ]


def test_support_level_coexists_with_the_ladders_it_does_not_replace():
    # #481 is ADDITIVE: the shipped multi-axis ladders remain, unchanged.
    assert len(ObjectiveVerificationStatus) == 6
    assert VerificationOutcome.partial in set(VerificationOutcome)  # the anti-collapse axis still exists
    assert len(RecipeVerification) == 4


def _valid_objective_verifications():
    statuses = list(ObjectiveVerificationStatus)
    for definition, implementation, hardware in itertools.product(statuses, repeat=3):
        try:
            yield ObjectiveVerification(
                definition=definition,
                implementation=implementation,
                hardware=hardware,
                evidence_refs=["evidence-0000"],
            )
        except ValidationError:
            continue  # the ladder's own validator rejects incoherent axis combinations


def test_projection_is_anti_collapse_and_carries_every_unproven_axis():
    seen_valid = 0
    for verification in _valid_objective_verifications():
        seen_valid += 1
        rollup = project_objective_verification(verification)
        hw_proven = verification.hardware == ObjectiveVerificationStatus.hardware_verified
        impl_proven = verification.implementation in _FUNCTIONALLY_PROVEN

        # lossy rollup, but NEVER higher than the proven axes justify
        assert (rollup.level == SupportLevel.workload_verified) == hw_proven
        if rollup.level == SupportLevel.probed:
            assert impl_proven and not hw_proven
        if rollup.level == SupportLevel.declared:
            assert not impl_proven
        # the objective ladder cannot evidence the env/manifest/ops-only states
        assert rollup.level not in _NET_NEW

        # every unproven axis is CARRIED verbatim; every projected axis is absent from carried
        for axis, status in (
            ("definition", verification.definition),
            ("implementation", verification.implementation),
            ("hardware", verification.hardware),
        ):
            token = f"{axis}:{status.value}"
            assert (token in rollup.carried) == (status in _UNPROVEN)

    assert seen_valid > 0  # the enumeration actually exercised valid records


def test_projection_spot_checks():
    # nothing proven -> declared, with the two unproven axes carried (definition defaults to 'declared',
    # which is projected into the level, not carried)
    bare = project_objective_verification(ObjectiveVerification())
    assert bare == SupportLevelRollup(
        level=SupportLevel.declared,
        carried=("implementation:not_verified", "hardware:not_verified"),
    )

    # a functional probe but no hardware run -> probed, hardware carried
    probed = project_objective_verification(
        ObjectiveVerification(
            definition=ObjectiveVerificationStatus.contract_validated,
            implementation=ObjectiveVerificationStatus.functional_verified,
            hardware=ObjectiveVerificationStatus.not_verified,
            evidence_refs=["evidence-0000"],
        )
    )
    assert probed.level == SupportLevel.probed
    assert probed.carried == ("hardware:not_verified",)

    # a real measured run -> workload_verified, nothing carried
    proven = project_objective_verification(
        ObjectiveVerification(
            definition=ObjectiveVerificationStatus.contract_validated,
            implementation=ObjectiveVerificationStatus.functional_verified,
            hardware=ObjectiveVerificationStatus.hardware_verified,
            evidence_refs=["evidence-0000"],
        )
    )
    assert proven.level == SupportLevel.workload_verified
    assert proven.carried == ()
