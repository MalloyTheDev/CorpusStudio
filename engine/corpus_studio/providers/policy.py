"""Role-based provider/model capability policy.

Enforced in the engine (not only the desktop UI): the same
``authorize_action`` guard runs for the CLI, the desktop, and tests. Trainable
generation is only permitted when the resolved policy explicitly allows it AND
a human review step follows.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProviderRole(str, Enum):
    """What a provider/model/route is permitted to do."""

    TRAINABLE_OUTPUT_GENERATOR = "trainable_output_generator"
    EVALUATOR = "evaluator"


# AI Assist actions that produce trainable dataset content (rows a user could
# accept into the dataset). These require generation-approved policy.
TRAINABLE_ACTIONS = frozenset({"rewrite-output", "draft-example"})
# Evaluator/critic actions producing non-trainable metadata only.
EVALUATOR_ACTIONS = frozenset({"review", "suggest-tags", "judge-preference-strength"})


class ProviderPolicyError(Exception):
    """Raised when an action is not permitted for the resolved provider policy."""


class ProviderPolicy(BaseModel):
    """Capability policy for a provider, or a specific model/route under it."""

    provider_id: str
    provider_kind: str = "unknown"  # hosted | local | router | unknown
    model_id: str | None = None
    route_id: str | None = None
    display_name: str = ""
    allowed_roles: list[ProviderRole] = Field(default_factory=list)
    blocked_roles: list[ProviderRole] = Field(default_factory=list)
    outputs_trainable: bool = False
    requires_human_review: bool = True
    local_only: bool = False
    requires_api_key: bool = False
    route_parent: str | None = None
    license_or_terms_note: str = ""
    safety_notes: str = ""
    default_policy_source: str = "builtin"
    user_approved_generation: bool = False

    def can_generate_trainable(self) -> bool:
        """True only when this policy is explicitly cleared to generate trainable rows."""

        if ProviderRole.TRAINABLE_OUTPUT_GENERATOR in self.blocked_roles:
            return False
        if ProviderRole.TRAINABLE_OUTPUT_GENERATOR not in self.allowed_roles:
            return False
        return self.outputs_trainable and self.user_approved_generation

    def can_evaluate(self) -> bool:
        return (
            ProviderRole.EVALUATOR in self.allowed_roles
            and ProviderRole.EVALUATOR not in self.blocked_roles
        )


_EVALUATOR_ONLY = {
    "allowed_roles": [ProviderRole.EVALUATOR],
    "blocked_roles": [ProviderRole.TRAINABLE_OUTPUT_GENERATOR],
    "outputs_trainable": False,
}

# Built-in defaults. Trainable generation is off by default everywhere; hosted
# frontier providers are hard-blocked from it, local providers can be approved.
DEFAULT_PROVIDER_POLICIES: dict[str, ProviderPolicy] = {
    "openai": ProviderPolicy(
        provider_id="openai",
        provider_kind="hosted",
        display_name="OpenAI",
        requires_api_key=True,
        license_or_terms_note=(
            "Provider terms restrict using outputs to train competing models; "
            "evaluator-only by default."
        ),
        default_policy_source="builtin",
        **_EVALUATOR_ONLY,
    ),
    "anthropic": ProviderPolicy(
        provider_id="anthropic",
        provider_kind="hosted",
        display_name="Anthropic",
        requires_api_key=True,
        license_or_terms_note=(
            "Provider terms restrict using outputs to train competing models; "
            "evaluator-only by default."
        ),
        default_policy_source="builtin",
        **_EVALUATOR_ONLY,
    ),
    "openrouter": ProviderPolicy(
        provider_id="openrouter",
        provider_kind="router",
        display_name="OpenRouter",
        requires_api_key=True,
        safety_notes="Route-aware: routes to OpenAI/Anthropic inherit their restrictions.",
        default_policy_source="builtin",
        # Base is evaluator-only; a specific approved non-frontier route may
        # upgrade to generation via resolve_policy + overrides.
        **_EVALUATOR_ONLY,
    ),
    "ollama": ProviderPolicy(
        provider_id="ollama",
        provider_kind="local",
        display_name="Ollama (local)",
        local_only=True,
        allowed_roles=[ProviderRole.EVALUATOR, ProviderRole.TRAINABLE_OUTPUT_GENERATOR],
        outputs_trainable=False,  # requires explicit per-model approval
        user_approved_generation=False,
        default_policy_source="builtin",
    ),
    "openai_compatible": ProviderPolicy(
        provider_id="openai_compatible",
        provider_kind="local",
        display_name="Local OpenAI-compatible server",
        local_only=True,
        allowed_roles=[ProviderRole.EVALUATOR, ProviderRole.TRAINABLE_OUTPUT_GENERATOR],
        outputs_trainable=False,  # requires explicit per-model approval
        user_approved_generation=False,
        safety_notes="A base_url can point at a hosted provider; approve generation deliberately.",
        default_policy_source="builtin",
    ),
}

_FRONTIER_ROUTE_PARENTS = {"openai", "anthropic"}


def route_parent(route_id: str) -> str:
    """OpenRouter route ids look like 'openai/gpt-4o' — the parent is the prefix."""

    return route_id.split("/", 1)[0].strip().lower()


def infer_provider_id(backend: str | None, base_url: str | None) -> str:
    """Best-effort provider identity from the transport backend and base URL.

    Heuristic: a base_url host of api.openai.com / api.anthropic.com /
    openrouter.ai maps to that provider; otherwise an OpenAI-compatible backend
    is treated as a local server. Documented as a heuristic — set an explicit
    provider_id when it matters.
    """

    normalized = (backend or "").replace("_", "-").lower()
    if normalized == "ollama":
        return "ollama"

    host = (base_url or "").lower()
    if "openrouter" in host:
        return "openrouter"
    if "api.openai.com" in host:
        return "openai"
    if "api.anthropic.com" in host:
        return "anthropic"
    return "openai_compatible"


def _fallback_policy(provider_id: str) -> ProviderPolicy:
    """Unknown providers default to the safest posture: evaluator-only."""

    return ProviderPolicy(
        provider_id=provider_id,
        provider_kind="unknown",
        display_name=provider_id,
        default_policy_source="fallback",
        **_EVALUATOR_ONLY,
    )


def resolve_policy(
    provider_id: str,
    model_id: str | None = None,
    route_id: str | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> ProviderPolicy:
    """Resolve the effective policy for a provider/model/route.

    Applies OpenRouter route inheritance, then any user overrides keyed by the
    most specific match (``provider/route``, ``provider/model``, or ``provider``).
    """

    base = DEFAULT_PROVIDER_POLICIES.get(provider_id) or _fallback_policy(provider_id)
    policy = base.model_copy(update={"model_id": model_id, "route_id": route_id})

    if provider_id == "openrouter" and route_id:
        parent = route_parent(route_id)
        if parent in _FRONTIER_ROUTE_PARENTS:
            policy = policy.model_copy(
                update={
                    "route_parent": parent,
                    "allowed_roles": [ProviderRole.EVALUATOR],
                    "blocked_roles": [ProviderRole.TRAINABLE_OUTPUT_GENERATOR],
                    "outputs_trainable": False,
                }
            )
        else:
            # Non-frontier route: generation is *possible* but still requires
            # explicit approval (outputs_trainable + user_approved_generation).
            policy = policy.model_copy(
                update={
                    "route_parent": parent,
                    "allowed_roles": [
                        ProviderRole.EVALUATOR,
                        ProviderRole.TRAINABLE_OUTPUT_GENERATOR,
                    ],
                    "blocked_roles": [],
                }
            )

    if overrides:
        for key in _override_keys(provider_id, model_id, route_id):
            override = overrides.get(key)
            if override:
                policy = policy.model_copy(update={**override, "default_policy_source": "user_override"})
                break

    return policy


def most_specific_override_key(
    provider_id: str, model_id: str | None = None, route_id: str | None = None
) -> str:
    """The single most-specific override key for a provider/model/route."""

    if route_id:
        return f"{provider_id}/route:{route_id}"
    if model_id:
        return f"{provider_id}/model:{model_id}"
    return provider_id


def _override_keys(
    provider_id: str, model_id: str | None, route_id: str | None
) -> list[str]:
    """Override lookup keys, most specific first."""

    keys: list[str] = []
    if route_id:
        keys.append(f"{provider_id}/route:{route_id}")
    if model_id:
        keys.append(f"{provider_id}/model:{model_id}")
    keys.append(provider_id)
    return keys


def is_trainable_action(action: str) -> bool:
    return action in TRAINABLE_ACTIONS


def authorize_action(policy: ProviderPolicy, action: str) -> None:
    """Raise ``ProviderPolicyError`` if ``action`` is not permitted for ``policy``.

    Trainable-generating actions require a generation-approved policy; all other
    (evaluator/critic) actions require the evaluator role.
    """

    label = policy.provider_id
    if policy.route_id:
        label += f" route '{policy.route_id}'"
    elif policy.model_id:
        label += f" model '{policy.model_id}'"

    if is_trainable_action(action):
        if not policy.can_generate_trainable():
            if ProviderRole.TRAINABLE_OUTPUT_GENERATOR in policy.blocked_roles:
                reason = "this provider is evaluator-only by default and is blocked from trainable generation"
            else:
                reason = (
                    "this model is not generation-approved; approve it explicitly "
                    "(outputs_trainable + user_approved_generation) before generating trainable rows"
                )
            raise ProviderPolicyError(
                f"{label} may not run trainable-generating action '{action}': {reason}. "
                "Trainable candidates still require human review before they can be saved."
            )
    elif not policy.can_evaluate():
        raise ProviderPolicyError(
            f"{label} may not run evaluator action '{action}': the evaluator role is not allowed."
        )
