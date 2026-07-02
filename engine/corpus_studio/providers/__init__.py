"""Provider role/capability policy, enforced in the engine.

Providers/models/routes carry roles (evaluator vs trainable-output generator).
OpenAI and Anthropic are evaluator-only by default; local providers (Ollama,
local OpenAI-compatible servers) may generate trainable output only when a user
explicitly approves the specific model; OpenRouter is route-aware and inherits
the restriction of the route's upstream provider.
"""

from corpus_studio.providers.policy import (
    DEFAULT_PROVIDER_POLICIES,
    EVALUATOR_ACTIONS,
    TRAINABLE_ACTIONS,
    ProviderPolicy,
    ProviderPolicyError,
    ProviderRole,
    authorize_action,
    authorize_evaluation,
    infer_provider_id,
    is_trainable_action,
    resolve_policy,
)

__all__ = [
    "DEFAULT_PROVIDER_POLICIES",
    "EVALUATOR_ACTIONS",
    "TRAINABLE_ACTIONS",
    "ProviderPolicy",
    "ProviderPolicyError",
    "ProviderRole",
    "authorize_action",
    "authorize_evaluation",
    "infer_provider_id",
    "is_trainable_action",
    "resolve_policy",
]
