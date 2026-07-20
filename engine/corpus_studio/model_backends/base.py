"""Shared model backend contracts.

These contracts are intentionally lightweight and make no network calls.
Provider-specific modules can implement them in v0.2.
"""

import json
from collections.abc import Iterator, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field

# Cap for reading an untrusted HTTP eval-backend response. Eval responses are tiny; a huge or slowly
# trickled body (a runaway/hostile --base-url endpoint) must not OOM the eval process. 64 MiB is generous.
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def read_bounded_json(response: Any, max_bytes: int = MAX_RESPONSE_BYTES) -> dict[str, Any]:
    """Read + parse a JSON HTTP response with a byte cap. Raises ValueError on overflow - the evaluator and
    health paths already treat ValueError as a transient backend error, so an oversized body isolates one
    row (null-with-reason) rather than OOM-crashing the whole batch. Never a silent truncation."""
    body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise ValueError(f"backend response exceeded {max_bytes} bytes (runaway or hostile server)")
    return json.loads(body.decode("utf-8"))


class ModelBackendConfig(BaseModel):
    """Configuration for a model provider."""

    provider_name: str
    base_url: str
    model_name: str
    api_key_optional: str | None = None
    timeout_seconds: int = 120
    max_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 1.0
    streaming_enabled: bool = False


class BackendGenerateRequest(BaseModel):
    """Provider-neutral generation request."""

    prompt: str | None = None
    messages: list[dict[str, str]] = Field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    # Sampler seed for reproducible decoding. None omits it (server default). With temperature 0 the
    # decode is greedy; seed pins any residual sampling so two identical eval passes agree.
    seed: int | None = None


class BackendGenerateResponse(BaseModel):
    """Provider-neutral generation response."""

    text: str
    model_name: str
    raw: dict | None = None


class BackendHealthReport(BaseModel):
    """Provider-neutral backend health report."""

    provider_name: str
    base_url: str
    model_name: str
    reachable: bool
    model_available: bool = False
    available_models: list[str] = Field(default_factory=list)
    error: str | None = None


class BackendModelListReport(BaseModel):
    """Provider-neutral available-models report."""

    provider_name: str
    base_url: str
    reachable: bool
    models: list[str] = Field(default_factory=list)
    error: str | None = None


class ModelBackend(Protocol):
    """Conceptual interface future provider implementations should satisfy."""

    config: ModelBackendConfig

    def list_models(self) -> Sequence[str]:
        """Return available model names."""

    def generate(self, request: BackendGenerateRequest) -> BackendGenerateResponse:
        """Generate one response."""

    def stream_generate(self, request: BackendGenerateRequest) -> Iterator[str]:
        """Stream response chunks."""

    def health_check(self) -> bool:
        """Return whether the backend is reachable."""
