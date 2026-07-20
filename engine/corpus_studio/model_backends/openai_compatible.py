"""OpenAI-compatible backend support.

This covers local servers such as LM Studio and later hosted compatible
providers. The adapter only performs network calls when methods are invoked.
"""
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
    read_bounded_json,
)
from corpus_studio.model_backends.retry import RetryPolicy, call_with_retry

UrlOpen = Callable[..., Any]


def default_openai_compatible_config(
    model_name: str,
    base_url: str = "http://localhost:1234/v1",
    api_key: str | None = None,
) -> ModelBackendConfig:
    """Return defaults for a local OpenAI-compatible endpoint."""

    return ModelBackendConfig(
        provider_name="openai_compatible",
        base_url=base_url,
        api_key_optional=api_key,
        model_name=model_name,
        streaming_enabled=False,
    )


class OpenAICompatibleBackend:
    """Small adapter for OpenAI-compatible chat completions endpoints."""

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
        payload = self._request_json("GET", "/models", retry_policy=retry_policy)
        return [
            str(model["id"])
            for model in payload.get("data", [])
            if isinstance(model, dict) and model.get("id")
        ]

    def generate(self, request: BackendGenerateRequest) -> BackendGenerateResponse:
        messages = request.messages or [{"role": "user", "content": request.prompt or ""}]
        body: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "temperature": request.temperature
            if request.temperature is not None
            else self.config.temperature,
            "top_p": request.top_p if request.top_p is not None else self.config.top_p,
            "stream": False,
        }
        if request.seed is not None:
            # best-effort determinism: some OpenAI-compatible servers honor `seed`, some ignore it
            # (recorded as a caveat in the eval report, not claimed as a guarantee).
            body["seed"] = request.seed
        payload = self._request_json("POST", "/chat/completions", body)
        choice = (payload.get("choices") or [{}])[0]
        text = choice.get("message", {}).get("content") or choice.get("text") or ""
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
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_optional:
            headers["Authorization"] = f"Bearer {self.config.api_key_optional}"
        url = _join_url(self.config.base_url, path)

        def _do() -> dict[str, Any]:
            request = Request(url, data=data, method=method, headers=headers)
            with self._opener(request, timeout=self.config.timeout_seconds) as response:
                return read_bounded_json(response)

        return call_with_retry(_do, retry_policy or self._retry_policy, sleep=self._sleep)


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"
