"""Tests for the shared transport retry policy and its wiring into the adapters."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError

import pytest

from corpus_studio.model_backends.base import BackendGenerateRequest, ModelBackendConfig
from corpus_studio.model_backends.ollama import OllamaBackend
from corpus_studio.model_backends.retry import (
    RetryPolicy,
    call_with_retry,
    format_backend_error,
    is_transient,
)


def _http_error(code: int) -> HTTPError:
    return HTTPError(url="http://x", code=code, msg="err", hdrs=None, fp=None)


@pytest.mark.parametrize(
    "exc, expected",
    [
        (_http_error(429), True),
        (_http_error(500), True),
        (_http_error(503), True),
        (_http_error(400), False),
        (_http_error(401), False),
        (_http_error(404), False),
        (URLError("connection refused"), True),
        (TimeoutError("timed out"), True),
        (ConnectionResetError("reset"), True),
        (ValueError("bad json"), False),
        (KeyError("bug"), False),
    ],
)
def test_is_transient_classifies_only_retryable_failures(exc, expected):
    assert is_transient(exc) is expected


def test_call_with_retry_returns_on_first_success_without_sleeping():
    slept: list[float] = []
    result = call_with_retry(lambda: "ok", RetryPolicy(), sleep=slept.append)
    assert result == "ok"
    assert slept == []


def test_call_with_retry_retries_transient_then_succeeds_with_backoff_schedule():
    slept: list[float] = []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _http_error(503)
        return "recovered"

    result = call_with_retry(
        flaky,
        RetryPolicy(max_attempts=3, base_delay_seconds=0.5, backoff_factor=2.0),
        sleep=slept.append,
        rng=lambda: 1.0,  # pin jitter to the top of the band => the full backoff schedule
    )
    assert result == "recovered"
    assert calls["n"] == 3
    assert slept == [0.5, 1.0]  # backoff after attempt 1 and 2; attempt 3 succeeds


def test_call_with_retry_does_not_retry_non_transient():
    slept: list[float] = []
    calls = {"n": 0}

    def failing():
        calls["n"] += 1
        raise _http_error(400)  # client error — retrying is pointless

    with pytest.raises(HTTPError):
        call_with_retry(failing, RetryPolicy(), sleep=slept.append)
    assert calls["n"] == 1
    assert slept == []


def test_call_with_retry_raises_last_error_after_exhausting_attempts():
    slept: list[float] = []
    calls = {"n": 0}

    def always_503():
        calls["n"] += 1
        raise _http_error(503)

    with pytest.raises(HTTPError):
        call_with_retry(always_503, RetryPolicy(max_attempts=3), sleep=slept.append)
    assert calls["n"] == 3  # tried the max, no more
    assert len(slept) == 2  # slept between the 3 attempts


def test_single_policy_never_retries_even_on_transient():
    slept: list[float] = []
    calls = {"n": 0}

    def always_503():
        calls["n"] += 1
        raise _http_error(503)

    with pytest.raises(HTTPError):
        call_with_retry(always_503, RetryPolicy.single(), sleep=slept.append)
    assert calls["n"] == 1
    assert slept == []


def test_jitter_keeps_backoff_within_the_equal_jitter_band():
    # Equal jitter: each sleep is in [0.5*delay, delay]. rng=0.0 -> the low edge (half).
    slept: list[float] = []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _http_error(503)
        return "ok"

    call_with_retry(
        flaky,
        RetryPolicy(max_attempts=3, base_delay_seconds=0.5, backoff_factor=2.0),
        sleep=slept.append,
        rng=lambda: 0.0,
    )
    assert slept == [0.25, 0.5]  # half of [0.5, 1.0]


def test_wall_clock_deadline_stops_retrying_before_the_budget_is_blown():
    # A fake clock that advances by each sleep (mimicking real elapsed time). With a 1.5s
    # deadline the first backoff (0.5s) fits, but the second (1.0s at elapsed 0.5) would reach
    # exactly the budget, so the call gives up instead of sleeping past it.
    clock = {"t": 0.0}
    slept: list[float] = []
    calls = {"n": 0}

    def fake_now() -> float:
        return clock["t"]

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        clock["t"] += seconds

    def always_503():
        calls["n"] += 1
        raise _http_error(503)

    with pytest.raises(HTTPError):
        call_with_retry(
            always_503,
            RetryPolicy(max_attempts=5, base_delay_seconds=0.5, backoff_factor=2.0),
            sleep=fake_sleep,
            rng=lambda: 1.0,
            now=fake_now,
            deadline_seconds=1.5,
        )
    assert calls["n"] == 2  # gave up on the wall-clock budget, not the 5-attempt cap
    assert slept == [0.5]  # only the first backoff fit inside the deadline


def _http_error_with_retry_after(code: int, retry_after: str) -> HTTPError:
    from email.message import Message

    headers = Message()
    headers["Retry-After"] = retry_after
    return HTTPError(url="http://x", code=code, msg="err", hdrs=headers, fp=None)


def test_retry_after_header_is_honored_verbatim_and_capped():
    slept: list[float] = []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            # 3s requested (over the 8s cap it's honored; a huge value would be capped).
            raise _http_error_with_retry_after(429, "3")
        return "ok"

    result = call_with_retry(
        flaky,
        RetryPolicy(max_attempts=3, base_delay_seconds=0.5, max_delay_seconds=8.0),
        sleep=slept.append,
        rng=lambda: 0.0,  # would give tiny jittered delays; Retry-After overrides, no jitter
    )
    assert result == "ok"
    assert slept == [3.0, 3.0]  # honored verbatim, NOT the exponential/jittered schedule


def test_retry_after_over_cap_is_clamped_to_max_delay():
    slept: list[float] = []

    def always_429():
        raise _http_error_with_retry_after(429, "3600")  # 1h requested

    with pytest.raises(HTTPError):
        call_with_retry(
            always_429,
            RetryPolicy(max_attempts=2, max_delay_seconds=8.0),
            sleep=slept.append,
        )
    assert slept == [8.0]  # clamped to max_delay_seconds


def test_delay_is_capped_at_max():
    policy = RetryPolicy(base_delay_seconds=1.0, backoff_factor=10.0, max_delay_seconds=8.0)
    assert policy.delay_for(1) == 1.0
    assert policy.delay_for(2) == 8.0  # 10.0 capped to 8.0
    assert policy.delay_for(5) == 8.0


def test_format_backend_error_leads_with_http_status():
    assert format_backend_error(_http_error(503)).startswith("HTTP 503")
    assert "URLError" in format_backend_error(URLError("boom"))


# --- Adapter integration: retry is actually wired into the HTTP path ---


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size: int = -1) -> bytes:
        return self._body if size < 0 else self._body[:size]


class _FlakyOpener:
    """Raise ``error`` for the first ``fail_times`` calls, then return ``body``."""

    def __init__(self, error: Exception, fail_times: int, body: bytes):
        self._error = error
        self._remaining = fail_times
        self._body = body
        self.calls = 0

    def __call__(self, request, timeout=None):
        self.calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise self._error
        return _FakeResponse(self._body)


def test_ollama_generate_recovers_after_transient_failures():
    opener = _FlakyOpener(
        error=_http_error(503),
        fail_times=2,
        body=json.dumps({"response": "hello"}).encode("utf-8"),
    )
    backend = OllamaBackend(
        ModelBackendConfig(provider_name="ollama", base_url="http://x", model_name="m"),
        opener=opener,
        sleep=lambda _seconds: None,  # no real waiting in tests
    )
    result = backend.generate(BackendGenerateRequest(prompt="hi"))
    assert result.text == "hello"
    assert opener.calls == 3  # two failures retried, third succeeded


def test_ollama_generate_fails_fast_on_client_error():
    opener = _FlakyOpener(error=_http_error(404), fail_times=5, body=b"{}")
    backend = OllamaBackend(
        ModelBackendConfig(provider_name="ollama", base_url="http://x", model_name="m"),
        opener=opener,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(HTTPError):
        backend.generate(BackendGenerateRequest(prompt="hi"))
    assert opener.calls == 1  # 404 is not retried


def test_ollama_health_check_is_single_attempt():
    opener = _FlakyOpener(error=_http_error(503), fail_times=5, body=b"{}")
    backend = OllamaBackend(
        ModelBackendConfig(provider_name="ollama", base_url="http://x", model_name="m"),
        opener=opener,
        sleep=lambda _seconds: None,
    )
    assert backend.health_check() is False
    assert opener.calls == 1  # probe fails fast, no backoff retries
