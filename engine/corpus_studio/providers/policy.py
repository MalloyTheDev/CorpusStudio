"""Role-based provider/model capability policy.

Enforced in the engine (not only the desktop UI): the same
``authorize_action`` guard runs for the CLI, the desktop, and tests. Trainable
generation is only permitted when the resolved policy explicitly allows it AND
a human review step follows.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, computed_field


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
    # A host-inferred ``openai_compatible`` endpoint is unverifiable — it could be a trusted
    # local model OR a proxy fronting a frontier API, and the engine does NOT probe it (a proxy
    # could spoof any probe anyway). Generation approval for it additionally requires this
    # explicit acknowledgment — a HUMAN attestation that it is a trusted local model, not a
    # verification the engine performed — so a frontier proxy can't be silently approved to
    # launder outputs into training data. This flag is the speed-bump by design: it converts an
    # unverifiable identity into a deliberate, on-the-record user decision.
    acknowledge_untrusted_endpoint: bool = False

    def can_generate_trainable(self) -> bool:
        """True only when this policy is explicitly cleared to generate trainable rows."""

        if ProviderRole.TRAINABLE_OUTPUT_GENERATOR in self.blocked_roles:
            return False
        if ProviderRole.TRAINABLE_OUTPUT_GENERATOR not in self.allowed_roles:
            return False
        if not (self.outputs_trainable and self.user_approved_generation):
            return False
        # An unverifiable OpenAI-compatible endpoint needs the extra explicit acknowledgment.
        if self.provider_id == "openai_compatible" and not self.acknowledge_untrusted_endpoint:
            return False
        return True

    def can_evaluate(self) -> bool:
        return (
            ProviderRole.EVALUATOR in self.allowed_roles
            and ProviderRole.EVALUATOR not in self.blocked_roles
        )

    # Serialized so any consumer (desktop, tooling) can show the effective
    # decision without re-implementing the role logic.
    # mypy doesn't model pydantic's @computed_field stacked on @property; the
    # pattern is correct at runtime, so silence the known false positive.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def generation_allowed(self) -> bool:
        return self.can_generate_trainable()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def evaluation_allowed(self) -> bool:
        return self.can_evaluate()


# Typed as dict[str, Any] so it can be splatted into ProviderPolicy(**...) without
# mypy inferring the heterogeneous literal as dict[str, object].
_EVALUATOR_ONLY: dict[str, Any] = {
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

    Trust boundary (honest scope): identity is inferred from the URL HOSTNAME only —
    it is not a probe and does not verify what actually answers at ``base_url``. A
    reverse proxy on a benign-looking host can front any frontier API, and this cannot
    tell. That is exactly why generation for a host-inferred ``openai_compatible``
    endpoint additionally requires ``acknowledge_untrusted_endpoint`` (a human vouch,
    see ProviderPolicy): the classification narrows the risk, the acknowledgment — not
    this function — is what authorizes trainable generation.
    """

    normalized = (backend or "").replace("_", "-").lower()
    if normalized == "ollama":
        return "ollama"

    # Match the exact hostname, not a substring of the whole URL, so a path or
    # query containing 'openai'/'openrouter' cannot misclassify a local server
    # (and 'api.openai.com.evil.example' is not treated as OpenAI).
    host = ""
    if base_url:
        parsed = urlsplit(base_url if "://" in base_url else f"//{base_url}")
        host = (parsed.hostname or "").lower()

    if host == "openrouter.ai" or host.endswith(".openrouter.ai"):
        return "openrouter"
    if host == "openai.com" or host.endswith(".openai.com"):
        return "openai"
    if host == "anthropic.com" or host.endswith(".anthropic.com"):
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


# User overrides (from an editable JSON file) may only touch these fields. Role
# and blocking fields are NOT overridable, so an override can never re-enable a
# hard-blocked frontier provider.
_OVERRIDE_ALLOWED_KEYS = frozenset(
    {
        "outputs_trainable",
        "user_approved_generation",
        "acknowledge_untrusted_endpoint",
        "license_or_terms_note",
        "safety_notes",
        "display_name",
        "requires_api_key",
    }
)


def _is_frontier(provider_id: str, route_id: str | None) -> bool:
    """Whether generation must be hard-blocked regardless of overrides/approval."""

    if provider_id in _FRONTIER_ROUTE_PARENTS:
        return True
    if provider_id == "openrouter" and route_id:
        # A bare (slash-less) route id can't be vetted, so treat it as frontier
        # and deny generation; a fully-qualified route inherits its vendor.
        if "/" not in route_id:
            return True
        return route_parent(route_id) in _FRONTIER_ROUTE_PARENTS
    return False


def _apply_frontier_block(policy: ProviderPolicy) -> ProviderPolicy:
    return policy.model_copy(
        update={
            "allowed_roles": [ProviderRole.EVALUATOR],
            "blocked_roles": [ProviderRole.TRAINABLE_OUTPUT_GENERATOR],
            "outputs_trainable": False,
        }
    )


def resolve_policy(
    provider_id: str,
    model_id: str | None = None,
    route_id: str | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> ProviderPolicy:
    """Resolve the effective policy for a provider/model/route.

    Applies OpenRouter route inheritance, then any user overrides (restricted to
    a safe key allowlist), then re-asserts the frontier block last so no override
    or route-id spelling can grant generation to a frontier provider/route.
    """

    base = DEFAULT_PROVIDER_POLICIES.get(provider_id) or _fallback_policy(provider_id)
    policy = base.model_copy(update={"model_id": model_id, "route_id": route_id})
    frontier = _is_frontier(provider_id, route_id)

    if provider_id == "openrouter" and route_id:
        policy = policy.model_copy(update={"route_parent": route_parent(route_id)})
        if not frontier:
            # Fully-qualified non-frontier route: generation is *possible* but
            # still requires explicit approval.
            policy = policy.model_copy(
                update={
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
                safe = {k: v for k, v in override.items() if k in _OVERRIDE_ALLOWED_KEYS}
                policy = policy.model_copy(
                    update={**safe, "default_policy_source": "user_override"}
                )
                break

    if frontier:
        policy = _apply_frontier_block(policy)

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


def authorize_evaluation(policy: ProviderPolicy) -> None:
    """Raise if the policy may not act as an evaluator/judge (non-trainable).

    Judging (e.g. arena response ranking) is an evaluator activity, so
    evaluator-only providers like OpenAI/Anthropic are permitted.
    """

    if not policy.can_evaluate():
        raise ProviderPolicyError(
            f"{policy.provider_id} is not permitted to evaluate/judge."
        )


def _policy_label(policy: ProviderPolicy) -> str:
    label = policy.provider_id
    if policy.route_id:
        label += f" route '{policy.route_id}'"
    elif policy.model_id:
        label += f" model '{policy.model_id}'"
    return label


def authorize_action(policy: ProviderPolicy | None, action: str) -> None:
    """Raise ``ProviderPolicyError`` if ``action`` is not permitted for ``policy``.

    Trainable-generating actions require a generation-approved policy; all other
    (evaluator/critic) actions require the evaluator role.

    A trainable action with **no policy** FAILS CLOSED: a missing policy is an
    unverified provider, and trainable data must never be generated without an
    explicitly approved one. Evaluator actions with no policy are permitted — they
    create no trainable data, and the CLI always resolves a policy in practice.
    """

    # Default-deny: an action that is neither a known trainable nor a known
    # evaluator action must not fall through to the permissive evaluator path.
    if action not in TRAINABLE_ACTIONS and action not in EVALUATOR_ACTIONS:
        who = _policy_label(policy) if policy is not None else "the provider"
        raise ProviderPolicyError(
            f"{who} was asked to run unrecognized action '{action}'; "
            "only explicitly categorized trainable/evaluator actions are permitted."
        )

    if is_trainable_action(action):
        if policy is None:
            raise ProviderPolicyError(
                f"Trainable-generating action '{action}' requires an explicit, "
                "generation-approved provider policy, but none was resolved (fail-closed). "
                "Trainable candidates still require human review before they can be saved."
            )
        if not policy.can_generate_trainable():
            label = _policy_label(policy)
            if ProviderRole.TRAINABLE_OUTPUT_GENERATOR in policy.blocked_roles:
                reason = "this provider is evaluator-only by default and is blocked from trainable generation"
            elif (
                policy.provider_id == "openai_compatible"
                and policy.outputs_trainable
                and policy.user_approved_generation
                and not policy.acknowledge_untrusted_endpoint
            ):
                reason = (
                    "this is a host-inferred OpenAI-compatible endpoint, which could be a "
                    "trusted local model OR a proxy fronting a frontier API; set "
                    "'acknowledge_untrusted_endpoint: true' in the override to vouch it is a "
                    "trusted local model before generating trainable rows"
                )
            else:
                reason = (
                    "this model is not generation-approved; approve it explicitly "
                    "(outputs_trainable + user_approved_generation) before generating trainable rows"
                )
            raise ProviderPolicyError(
                f"{label} may not run trainable-generating action '{action}': {reason}. "
                "Trainable candidates still require human review before they can be saved."
            )
    elif policy is not None and not policy.can_evaluate():
        raise ProviderPolicyError(
            f"{_policy_label(policy)} may not run evaluator action '{action}': "
            "the evaluator role is not allowed."
        )
