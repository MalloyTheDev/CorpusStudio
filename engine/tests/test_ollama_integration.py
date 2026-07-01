"""Opt-in local integration tests against a real Ollama backend.

These do NOT run by default and are skipped in CI. They exercise the real
network path for model discovery, health checks, Evaluation, and AI Assist so a
maintainer can smoke-test the local stack before a release.

Enable them with a running Ollama server:

    CORPUS_STUDIO_OLLAMA_INTEGRATION=1 pytest -m integration

Optional overrides:
    CORPUS_STUDIO_OLLAMA_MODEL      (default: llama3.2)
    CORPUS_STUDIO_OLLAMA_BASE_URL   (default: http://localhost:11434)

Each test additionally self-skips if the backend is unreachable or has no
models pulled, so an enabled-but-unavailable environment reports skips, never
failures.
"""

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.model_backends.ollama import OllamaBackend, default_ollama_config

runner = CliRunner()


def _flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


INTEGRATION_ENABLED = _flag_enabled("CORPUS_STUDIO_OLLAMA_INTEGRATION")
OLLAMA_MODEL = os.environ.get("CORPUS_STUDIO_OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.environ.get("CORPUS_STUDIO_OLLAMA_BASE_URL", "http://localhost:11434")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not INTEGRATION_ENABLED,
        reason="Set CORPUS_STUDIO_OLLAMA_INTEGRATION=1 to run Ollama integration tests.",
    ),
]


@pytest.fixture(scope="module")
def ollama_models() -> list[str]:
    config = default_ollama_config(OLLAMA_MODEL).model_copy(
        update={"base_url": OLLAMA_BASE_URL, "timeout_seconds": 15}
    )
    backend = OllamaBackend(config)
    try:
        models = list(backend.list_models())
    except Exception as exc:  # noqa: BLE001 - any network failure means "not available".
        pytest.skip(f"Ollama not reachable at {OLLAMA_BASE_URL}: {exc}")
    if not models:
        pytest.skip("Ollama is reachable but has no models pulled.")
    return models


@pytest.fixture(scope="module")
def resolved_model(ollama_models: list[str]) -> str:
    for model in ollama_models:
        if model == OLLAMA_MODEL or model.startswith(f"{OLLAMA_MODEL}:"):
            return model
    return ollama_models[0]


def test_model_list_discovers_models(ollama_models: list[str]):
    result = runner.invoke(
        app, ["model-list", "--backend", "ollama", "--base-url", OLLAMA_BASE_URL]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["reachable"] is True
    assert payload["models"]


def test_backend_health_reports_reachable(resolved_model: str):
    result = runner.invoke(
        app,
        [
            "backend-health",
            "--backend",
            "ollama",
            "--base-url",
            OLLAMA_BASE_URL,
            "--model",
            resolved_model,
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["reachable"] is True
    assert payload["model_available"] is True


def test_eval_run_produces_report(tmp_path: Path, resolved_model: str):
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        json.dumps({"instruction": "Reply with the single word: ping.", "output": "ping"})
        + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "eval-run",
            str(dataset),
            "instruction",
            "--backend",
            "ollama",
            "--base-url",
            OLLAMA_BASE_URL,
            "--model",
            resolved_model,
            "--limit",
            "1",
            "--timeout-seconds",
            "180",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["examples_tested"] == 1
    assert len(report["results"]) == 1


def test_ai_assist_review_returns_review_only_result(tmp_path: Path, resolved_model: str):
    dataset = tmp_path / "draft.jsonl"
    dataset.write_text(
        json.dumps({"instruction": "Explain what a variable is.", "output": "It stores a value."})
        + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "ai-assist",
            str(dataset),
            "instruction",
            "--action",
            "review",
            "--backend",
            "ollama",
            "--base-url",
            OLLAMA_BASE_URL,
            "--model",
            resolved_model,
            "--timeout-seconds",
            "180",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["action"] == "review"
    assert payload["review_required"] is True
