from corpus_studio.model_backends.base import ModelBackendConfig
from corpus_studio.model_backends.base import BackendGenerateRequest
from corpus_studio.model_backends.ollama import OllamaBackend, default_ollama_config
from corpus_studio.model_backends.openai_compatible import (
    OpenAICompatibleBackend,
    default_openai_compatible_config,
)


class FakeResponse:
    def __init__(self, payload: str):
        self._payload = payload.encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_model_backend_config_can_be_constructed_without_api_key():
    config = ModelBackendConfig(
        provider_name="custom_http",
        base_url="http://localhost:9000",
        model_name="local-model",
    )

    assert config.api_key_optional is None
    assert config.timeout_seconds == 120
    assert config.streaming_enabled is False


def test_local_backend_defaults_do_not_require_network_calls():
    ollama = default_ollama_config("qwen2.5-coder:7b")
    compatible = default_openai_compatible_config("local-chat-model")

    assert ollama.provider_name == "ollama"
    assert ollama.base_url == "http://localhost:11434"
    assert compatible.provider_name == "openai_compatible"
    assert compatible.base_url == "http://localhost:1234/v1"


def test_ollama_backend_generate_uses_injected_opener_without_real_network():
    calls = []

    def fake_opener(request, timeout):
        calls.append((request, timeout))
        return FakeResponse('{"response": "A variable stores a value."}')

    backend = OllamaBackend(default_ollama_config("local-model"), opener=fake_opener)

    response = backend.generate(BackendGenerateRequest(prompt="Explain variables."))

    assert response.text == "A variable stores a value."
    assert calls[0][0].full_url == "http://localhost:11434/api/generate"
    assert calls[0][1] == 120


def test_openai_compatible_backend_generate_uses_injected_opener_without_real_network():
    calls = []
    config = default_openai_compatible_config(
        "local-chat-model",
        base_url="http://localhost:1234/v1",
        api_key="test-key",
    )

    def fake_opener(request, timeout):
        calls.append((request, timeout))
        return FakeResponse(
            '{"choices": [{"message": {"content": "Recursion calls itself."}}]}'
        )

    backend = OpenAICompatibleBackend(config, opener=fake_opener)

    response = backend.generate(BackendGenerateRequest(prompt="Explain recursion."))

    assert response.text == "Recursion calls itself."
    assert calls[0][0].full_url == "http://localhost:1234/v1/chat/completions"
    assert calls[0][0].headers["Authorization"] == "Bearer test-key"
