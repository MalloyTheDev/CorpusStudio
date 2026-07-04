# Provider Policy

Corpus Studio enforces **role-based provider/model capability policy in the
engine** — not only in the desktop UI. The same guard runs for the CLI, the
desktop, and tests, so no surface can bypass it.

## Roles

- **Trainable Output Generator** — may produce trainable dataset content
  (rewritten rows, drafted/synthetic examples, preference-pair candidates) only
  when the exact provider/model/route policy allows it.
- **Evaluator / Judge / Critic** — may score, critique, classify, label, compare
  outputs, and judge preference strength, producing **non-trainable metadata**.

Trainable-generating AI Assist actions today: `rewrite-output`, `draft-example`.
Evaluator actions: `review`, `suggest-tags`, `judge-preference-strength`.

## Defaults

| Provider | Trainable generation | Notes |
|---|---|---|
| **OpenAI** | **Blocked** (evaluator-only) | The generator role is on `blocked_roles`; a user override cannot re-enable it. |
| **Anthropic** | **Blocked** (evaluator-only) | Same as OpenAI. |
| **OpenRouter** | **Route-aware** | Routes to `openai/*` and `anthropic/*` inherit the block; other routes may generate only when that exact route is approved. |
| **Ollama** (local) | **Off until approved** | Approve a specific model (`outputs_trainable` + `user_approved_generation`). |
| **Local OpenAI-compatible** | **Off until *explicitly acknowledged*** | Host-inferred, so unverifiable (could be a local model OR a proxy fronting a frontier API). Generation approval requires `acknowledge_untrusted_endpoint` **in addition to** `outputs_trainable` + `user_approved_generation`. |
| Unknown provider | Blocked (evaluator-only) | Safest fallback. |

`requires_human_review` is always true for trainable generation. Every generated
candidate is returned `review_required` and must go through
**generate → validate → quality check → gates → human review → accept/edit/reject → save**.
Evaluator-only output never enters trainable fields through the normal save flow.

## Policy resolution

`resolve_policy(provider_id, model_id, route_id, overrides)`:
1. Start from the built-in default for `provider_id` (evaluator-only fallback if unknown).
2. For OpenRouter, apply route inheritance (`openai/*`, `anthropic/*` → blocked; others → generator role allowed but still needs approval).
3. Apply the most-specific project override (`provider/route:…`, `provider/model:…`, or `provider`).

Provider identity is inferred from the transport backend + `base_url` by exact
hostname (`api.openai.com → openai`, `openrouter.ai → openrouter`, otherwise a
local OpenAI-compatible server). This is a **heuristic** — set an explicit
provider id when it matters (a provider reached through a proxy/gateway URL
cannot be identified from the host alone).

**Safety invariants (enforced):**
- User overrides may only set a safe key allowlist (`outputs_trainable`,
  `user_approved_generation`, `acknowledge_untrusted_endpoint`, notes, display
  name). Role and blocking fields are **not** overridable, and the frontier block
  is re-asserted last — so no override can re-enable generation for
  OpenAI/Anthropic/frontier routes.
- A host-inferred `openai_compatible` endpoint can't generate trainable rows on
  `user_approved_generation` alone — it also needs `acknowledge_untrusted_endpoint`
  (a conscious "this is a trusted local model, not a frontier proxy"), so a local
  proxy fronting a frontier API can't be silently approved to launder its outputs
  into training data.
- OpenRouter routes must be fully qualified (`vendor/model`). A bare, slash-less
  route id (e.g. `gpt-4o`) cannot be vetted and is treated as frontier — denied.
- `authorize_action` is default-deny: an action that is neither a known trainable
  nor a known evaluator action is rejected, not permitted as an evaluator call.

## Approving a local model (local-first, inspectable)

Approvals live in project-local `provider_policy_overrides.json`:

```
python -m corpus_studio.cli provider-approve --provider ollama \
  --project-dir path/to/project --model llama3
```

Inspect effective policy:

```
python -m corpus_studio.cli provider-policy --provider ollama \
  --model llama3 --project-dir path/to/project
```

Revoke with `--revoke`. Approving an evaluator-only provider (OpenAI/Anthropic)
is refused (exit code 2).

## Not implemented

Real OpenAI/Anthropic/OpenRouter API clients — policy is enforced regardless of
whether a given transport is implemented (evaluator-only providers are configured,
not embedded). The approved-generation → review-queue pipeline shipped in v1.2 as
candidate gating; see [`AI_ASSIST_LAB.md`](AI_ASSIST_LAB.md).
