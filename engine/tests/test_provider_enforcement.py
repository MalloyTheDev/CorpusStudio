import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from corpus_studio.ai_assist.assistant import run_ai_assist
from corpus_studio.cli import app
from corpus_studio.model_backends.base import BackendGenerateResponse
from corpus_studio.providers.overrides import (
    approve_generation,
    load_overrides,
    revoke_generation,
)
from corpus_studio.providers.policy import (
    ProviderPolicyError,
    authorize_action,
    resolve_policy,
)

runner = CliRunner()

ROWS = [{"instruction": "Explain recursion.", "output": "A function calls itself."}]


class FakeBackend:
    """Returns a fixed JSON body; never touches the network."""

    config = None

    def generate(self, request):
        return BackendGenerateResponse(
            text=json.dumps({"summary": "ok", "suggested_jsonl": "", "tags": [], "warnings": []}),
            model_name="fake-local",
        )


# --- engine-level enforcement (not desktop) ----------------------------------

def test_engine_blocks_generation_for_evaluator_only_policy():
    policy = resolve_policy("openai")
    with pytest.raises(ProviderPolicyError):
        run_ai_assist(
            schema_id="instruction",
            action="rewrite-output",
            rows=ROWS,
            backend=FakeBackend(),
            model="gpt-4o",
            policy=policy,
        )


def test_engine_allows_evaluator_action_for_evaluator_only_policy():
    policy = resolve_policy("anthropic")
    result = run_ai_assist(
        schema_id="instruction",
        action="review",
        rows=ROWS,
        backend=FakeBackend(),
        model="claude-3.5-sonnet",
        policy=policy,
    )
    assert result.review_required is True


def test_engine_blocks_unapproved_ollama_generation():
    policy = resolve_policy("ollama", model_id="llama3")
    with pytest.raises(ProviderPolicyError):
        run_ai_assist(
            schema_id="instruction",
            action="draft-example",
            rows=ROWS,
            backend=FakeBackend(),
            model="llama3",
            policy=policy,
        )


def test_engine_allows_approved_ollama_generation_but_still_review_required():
    policy = resolve_policy(
        "ollama",
        model_id="llama3",
        overrides={"ollama/model:llama3": {"outputs_trainable": True, "user_approved_generation": True}},
    )
    result = run_ai_assist(
        schema_id="instruction",
        action="rewrite-output",
        rows=ROWS,
        backend=FakeBackend(),
        model="llama3",
        policy=policy,
    )
    # Generation is permitted, but the candidate must still be reviewed.
    assert result.review_required is True
    assert result.review_state == "review_required"


# --- project-local overrides round-trip --------------------------------------

def test_overrides_roundtrip_and_resolve(tmp_path: Path):
    key = approve_generation(tmp_path, "ollama", model_id="llama3")
    assert key == "ollama/model:llama3"

    loaded = load_overrides(tmp_path)
    assert loaded[key]["user_approved_generation"] is True
    assert resolve_policy("ollama", model_id="llama3", overrides=loaded).can_generate_trainable()

    assert revoke_generation(tmp_path, "ollama", model_id="llama3") is True
    after = load_overrides(tmp_path)
    assert resolve_policy("ollama", model_id="llama3", overrides=after).can_generate_trainable() is False


def test_overrides_missing_file_is_empty(tmp_path: Path):
    assert load_overrides(tmp_path) == {}


# --- CLI enforcement (engine entrypoint) -------------------------------------

def test_cli_ai_assist_blocks_openai_generation(tmp_path: Path):
    src = tmp_path / "rows.jsonl"
    src.write_text(json.dumps(ROWS[0]) + "\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "ai-assist",
            str(src),
            "instruction",
            "--action",
            "rewrite-output",
            "--model",
            "gpt-4o",
            "--backend",
            "openai-compatible",
            "--base-url",
            "https://api.openai.com/v1",
        ],
    )
    # Blocked by policy before any provider call (exit code 2), no network used.
    assert result.exit_code == 2


# --- item 9c: host-inferred openai_compatible needs an explicit acknowledgment -----------

_OAI_COMPAT_KEY = "openai_compatible/model:local-model"


def test_openai_compatible_generation_needs_explicit_acknowledgment():
    # user_approved_generation is NOT enough for an unverifiable OpenAI-compatible endpoint
    # (it could front a frontier API): generation stays blocked without the acknowledgment.
    no_ack = resolve_policy(
        "openai_compatible",
        model_id="local-model",
        overrides={_OAI_COMPAT_KEY: {"outputs_trainable": True, "user_approved_generation": True}},
    )
    assert no_ack.can_generate_trainable() is False

    acknowledged = resolve_policy(
        "openai_compatible",
        model_id="local-model",
        overrides={
            _OAI_COMPAT_KEY: {
                "outputs_trainable": True,
                "user_approved_generation": True,
                "acknowledge_untrusted_endpoint": True,
            }
        },
    )
    assert acknowledged.can_generate_trainable() is True


def test_openai_compatible_without_ack_refuses_draft_with_actionable_message():
    policy = resolve_policy(
        "openai_compatible",
        model_id="local-model",
        overrides={_OAI_COMPAT_KEY: {"outputs_trainable": True, "user_approved_generation": True}},
    )
    with pytest.raises(ProviderPolicyError, match="acknowledge_untrusted_endpoint"):
        authorize_action(policy, "draft-example")


def test_ollama_generation_does_not_need_the_acknowledgment():
    # Ollama is a specific known local runtime, not an arbitrary endpoint — approval unchanged.
    approved = resolve_policy(
        "ollama",
        model_id="llama3",
        overrides={"ollama/model:llama3": {"outputs_trainable": True, "user_approved_generation": True}},
    )
    assert approved.can_generate_trainable() is True
