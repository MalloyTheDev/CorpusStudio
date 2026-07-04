"""Ollama backend support for local Evaluation Lab runs."""

from collections.abc import Callable, Iterator, Sequence
import json
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from corpus_studio.model_backends.base import (
    BackendGenerateRequest,
    BackendGenerateResponse,
    ModelBackendConfig,
)
from corpus_studio.model_backends.retry import RetryPolicy, call_with_retry

UrlOpen = Callable[..., Any]


def default_ollama_config(model_name: str) -> ModelBackendConfig:
    """Return local-first defaults for an Ollama model."""

    return ModelBackendConfig(
        provider_name="ollama",
        base_url="http://localhost:11434",
        model_name=model_name,
        streaming_enabled=False,
    )


class OllamaBackend:
    """Small Ollama HTTP adapter.

    The adapter only makes network calls when a method is invoked. Tests can
    inject a fake opener so no real Ollama process is required.
    """

    def __init__(
        self,
        config: ModelBackendConfig,
        opener: UrlOpen = urlopen,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], Any] = time.sleep,
    ):
        self.config = config
        self._opener = opener
        # Generation retries transient failures; probes (health/model-list) stay
        # single-attempt so an unreachable server fails fast for the UI.
        self._retry_policy = retry_policy or RetryPolicy()
        self._sleep = sleep

    def list_models(self, retry_policy: RetryPolicy | None = None) -> Sequence[str]:
        payload = self._request_json("GET", "/api/tags", retry_policy=retry_policy)
        return [
            str(model["name"])
            for model in payload.get("models", [])
            if isinstance(model, dict) and model.get("name")
        ]

    def generate(self, request: BackendGenerateRequest) -> BackendGenerateResponse:
        if request.messages:
            payload = self._request_json(
                "POST",
                "/api/chat",
                {
                    "model": self.config.model_name,
                    "messages": request.messages,
                    "stream": False,
                    "options": self._options(request),
                },
            )
            text = payload.get("message", {}).get("content", "")
        else:
            payload = self._request_json(
                "POST",
                "/api/generate",
                {
                    "model": self.config.model_name,
                    "prompt": request.prompt or "",
                    "stream": False,
                    "options": self._options(request),
                },
            )
            text = payload.get("response", "")

        return BackendGenerateResponse(
            text=str(text),
            model_name=self.config.model_name,
            raw=payload,
        )

    def stream_generate(self, request: BackendGenerateRequest) -> Iterator[str]:
        raise NotImplementedError("Streaming generation is not implemented for the MVP adapter.")

    def health_check(self) -> bool:
        try:
            # A health probe should fail fast, not sit through backoff retries.
            self.list_models(retry_policy=RetryPolicy.single())
            return True
        except (OSError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
            return False

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        url = _join_url(self.config.base_url, path)

        def _do() -> dict[str, Any]:
            request = Request(
                url,
                data=data,
                method=method,
                headers={"Content-Type": "application/json"},
            )
            with self._opener(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))

        return call_with_retry(_do, retry_policy or self._retry_policy, sleep=self._sleep)

    def _options(self, request: BackendGenerateRequest) -> dict[str, Any]:
        return {
            "temperature": request.temperature
            if request.temperature is not None
            else self.config.temperature,
            "top_p": request.top_p if request.top_p is not None else self.config.top_p,
            "num_predict": request.max_tokens
            if request.max_tokens is not None
            else self.config.max_tokens,
        }


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"
