"""Preference-pair data policy (S2 / DPO). Additive + dense/MoE-safe, PARALLEL to the SFT-only
TrainingDataPolicy; it fails closed when the prompt length budget leaves no room for the response."""

import pytest

from corpus_studio.platform.contracts import PreferenceDataPolicy


def _policy(**over):
    base = dict(
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
    assert p.truncation_policy == "refuse"  # never silently truncate a pair
    assert PreferenceDataPolicy.model_validate_json(p.model_dump_json()) == p


def test_preference_data_policy_refuses_a_prompt_budget_with_no_room_for_the_response():
    with pytest.raises(ValueError, match="below max_length"):
        _policy(max_prompt_length=1024, max_length=1024)
