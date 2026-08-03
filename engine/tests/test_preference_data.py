"""Preference-pair data policy (S2 / DPO). Additive + dense/MoE-safe, PARALLEL to the SFT-only
TrainingDataPolicy; it fails closed when the prompt length budget leaves no room for the response."""

import pytest

from corpus_studio.platform.contracts import PreferenceDataPolicy


def _policy(**over):
    base = dict(
        schema_id="preference",
        schema_version="0.1.0",
        schema_sha256="e" * 64,
        formatter_id="corpus-studio:preference-chat-v1",
        formatter_sha256="a" * 64,
        max_prompt_length=512,
        max_length=1024,
    )
    base.update(over)
    return PreferenceDataPolicy(**base)


def test_preference_data_policy_round_trips_a_chosen_rejected_pair_schema():
    p = _policy()
    assert p.pair_schema == "chosen_rejected"
    # the sealed resolved-schema identity: id + version + the content digest
    assert p.schema_id == "preference"
    assert p.schema_version == "0.1.0"
    assert p.schema_sha256 == "e" * 64
    assert p.truncation_policy == "refuse"  # never silently truncate a pair
    assert PreferenceDataPolicy.model_validate_json(p.model_dump_json()) == p


def test_preference_data_policy_requires_the_resolved_schema_digest():
    # id + version alone do not fail closed (a project-local schema can edit fields without bumping the
    # version), so the content digest is mandatory - and schema_id must be non-empty.
    base = dict(
        schema_id="preference",
        schema_version="0.1.0",
        formatter_id="corpus-studio:preference-chat-v1",
        formatter_sha256="a" * 64,
        max_prompt_length=512,
        max_length=1024,
    )
    with pytest.raises(ValueError):
        PreferenceDataPolicy(**base)  # no schema_sha256
    with pytest.raises(ValueError):
        _policy(schema_id="")  # min_length=1


def test_preference_data_policy_refuses_a_prompt_budget_with_no_room_for_the_response():
    with pytest.raises(ValueError, match="below max_length"):
        _policy(max_prompt_length=1024, max_length=1024)
