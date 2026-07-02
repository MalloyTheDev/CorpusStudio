import pytest

from corpus_studio.providers.policy import (
    ProviderPolicyError,
    ProviderRole,
    authorize_action,
    infer_provider_id,
    resolve_policy,
)


# --- evaluator-only defaults -------------------------------------------------

def test_openai_is_evaluator_only_by_default():
    policy = resolve_policy("openai")
    assert policy.can_evaluate() is True
    assert policy.can_generate_trainable() is False
    with pytest.raises(ProviderPolicyError):
        authorize_action(policy, "rewrite-output")
    with pytest.raises(ProviderPolicyError):
        authorize_action(policy, "draft-example")


def test_anthropic_is_evaluator_only_by_default():
    policy = resolve_policy("anthropic")
    assert policy.can_generate_trainable() is False
    with pytest.raises(ProviderPolicyError):
        authorize_action(policy, "draft-example")


def test_evaluator_actions_allowed_for_evaluator_only_providers():
    for provider in ("openai", "anthropic"):
        policy = resolve_policy(provider)
        # None of these should raise.
        authorize_action(policy, "review")
        authorize_action(policy, "suggest-tags")
        authorize_action(policy, "judge-preference-strength")


def test_frontier_providers_cannot_be_overridden_into_generation():
    # Even if a user override flips the flags, the blocked role wins.
    policy = resolve_policy(
        "openai",
        overrides={"openai": {"outputs_trainable": True, "user_approved_generation": True}},
    )
    assert policy.can_generate_trainable() is False
    with pytest.raises(ProviderPolicyError):
        authorize_action(policy, "rewrite-output")


# --- ollama / local generation approval --------------------------------------

def test_ollama_unapproved_cannot_generate():
    policy = resolve_policy("ollama", model_id="llama3")
    assert policy.can_generate_trainable() is False
    with pytest.raises(ProviderPolicyError):
        authorize_action(policy, "rewrite-output")


def test_ollama_approved_model_can_generate():
    policy = resolve_policy(
        "ollama",
        model_id="llama3",
        overrides={
            "ollama/model:llama3": {
                "outputs_trainable": True,
                "user_approved_generation": True,
            }
        },
    )
    assert policy.can_generate_trainable() is True
    assert policy.requires_human_review is True
    authorize_action(policy, "rewrite-output")  # does not raise


def test_ollama_approval_is_per_model():
    overrides = {
        "ollama/model:approved": {"outputs_trainable": True, "user_approved_generation": True}
    }
    assert resolve_policy("ollama", model_id="approved", overrides=overrides).can_generate_trainable()
    assert not resolve_policy("ollama", model_id="other", overrides=overrides).can_generate_trainable()


# --- openrouter route awareness ----------------------------------------------

def test_openrouter_openai_route_is_blocked():
    policy = resolve_policy("openrouter", route_id="openai/gpt-4o")
    assert policy.route_parent == "openai"
    assert policy.can_generate_trainable() is False
    with pytest.raises(ProviderPolicyError):
        authorize_action(policy, "rewrite-output")


def test_openrouter_anthropic_route_is_blocked():
    policy = resolve_policy("openrouter", route_id="anthropic/claude-3.5-sonnet")
    assert policy.route_parent == "anthropic"
    with pytest.raises(ProviderPolicyError):
        authorize_action(policy, "draft-example")


def test_openrouter_frontier_route_cannot_be_overridden():
    policy = resolve_policy(
        "openrouter",
        route_id="openai/gpt-4o",
        overrides={
            "openrouter/route:openai/gpt-4o": {
                "outputs_trainable": True,
                "user_approved_generation": True,
            }
        },
    )
    # Route inheritance blocks the generator role; override cannot re-enable it.
    assert policy.can_generate_trainable() is False


def test_openrouter_non_frontier_route_can_be_approved():
    policy = resolve_policy(
        "openrouter",
        route_id="meta-llama/llama-3-70b-instruct",
        overrides={
            "openrouter/route:meta-llama/llama-3-70b-instruct": {
                "outputs_trainable": True,
                "user_approved_generation": True,
            }
        },
    )
    assert policy.route_parent == "meta-llama"
    assert policy.can_generate_trainable() is True
    authorize_action(policy, "rewrite-output")


def test_openrouter_non_frontier_route_unapproved_cannot_generate():
    policy = resolve_policy("openrouter", route_id="meta-llama/llama-3-70b-instruct")
    assert policy.can_generate_trainable() is False


# --- provider inference ------------------------------------------------------

def test_infer_provider_id():
    assert infer_provider_id("ollama", None) == "ollama"
    assert infer_provider_id("openai-compatible", "https://api.openai.com/v1") == "openai"
    assert infer_provider_id("openai-compatible", "https://api.anthropic.com") == "anthropic"
    assert infer_provider_id("openai-compatible", "https://openrouter.ai/api/v1") == "openrouter"
    assert infer_provider_id("openai-compatible", "http://localhost:1234/v1") == "openai_compatible"


def test_unknown_provider_falls_back_to_evaluator_only():
    policy = resolve_policy("some-new-provider")
    assert policy.default_policy_source == "fallback"
    assert policy.can_generate_trainable() is False
    assert policy.can_evaluate() is True


# --- audit hardening regressions ---------------------------------------------

def test_frontier_block_survives_role_key_override():
    # An override that tries to wipe the block/roles must NOT re-enable generation.
    policy = resolve_policy(
        "openai",
        overrides={
            "openai": {
                "outputs_trainable": True,
                "user_approved_generation": True,
                "blocked_roles": [],
                "allowed_roles": ["evaluator", "trainable_output_generator"],
            }
        },
    )
    assert policy.can_generate_trainable() is False
    assert ProviderRole.TRAINABLE_OUTPUT_GENERATOR in policy.blocked_roles


def test_openrouter_bare_slug_route_is_frontier_blocked():
    # A slash-less route id (e.g. 'gpt-4o') cannot be vetted -> deny generation.
    policy = resolve_policy(
        "openrouter",
        route_id="gpt-4o",
        overrides={
            "openrouter/route:gpt-4o": {"outputs_trainable": True, "user_approved_generation": True}
        },
    )
    assert policy.can_generate_trainable() is False


def test_infer_provider_id_matches_exact_host_only():
    # Substrings in the path must not misclassify a local server.
    assert infer_provider_id("openai-compatible", "http://localhost:1234/openrouter") == "openai_compatible"
    assert infer_provider_id("openai-compatible", "http://localhost/api.openai.com/proxy") == "openai_compatible"
    # Look-alike attacker domain is not treated as OpenAI.
    assert infer_provider_id("openai-compatible", "https://api.openai.com.evil.example/v1") == "openai_compatible"
    # Real hosts still resolve.
    assert infer_provider_id("openai-compatible", "https://api.openai.com/v1") == "openai"
    assert infer_provider_id("openai-compatible", "https://openrouter.ai/api/v1") == "openrouter"


def test_unknown_action_is_denied():
    from corpus_studio.providers.policy import ProviderPolicyError, authorize_action

    policy = resolve_policy("ollama", model_id="llama3")
    with pytest.raises(ProviderPolicyError):
        authorize_action(policy, "generate")  # not a categorized action
