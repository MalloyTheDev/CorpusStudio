"""Transport-level retry policy shared by the HTTP model backends.

Model calls fail transiently all the time — a provider hiccups with a 503, rate
limits with a 429, or a socket drops mid-request. A bounded exponential-backoff
retry turns most of those blips into a successful call instead of a failed batch
item. The classification is deliberately conservative: only connection-level
errors and HTTP 429/5xx are retried; every other 4xx (400/401/403/404) is a
client/config problem where retrying only wastes time, so it fails fast.

The primitive is dependency-free and its ``sleep`` is injectable so tests run
instantly and can assert the backoff schedule without real waits.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar
from urllib.error import HTTPError, URLError

T = TypeVar("T")

# Errors that represent a genuine backend/transport failure (network down,
# timeout, bad HTTP status, unreadable/invalid JSON body) as opposed to a bug in
# our own code. Callers that isolate one item's failure from a batch catch these
# so a failed model call is recorded, not fatal — HTTPError/URLError/TimeoutError
# are all OSError subclasses, and json.JSONDecodeError is a ValueError.
BACKEND_ERROR_TYPES: tuple[type[BaseException], ...] = (OSError, ValueError)


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential-backoff schedule for a single HTTP call.

    ``max_attempts`` counts the FIRST try, so ``max_attempts=3`` means one call
    plus up to two retries. ``single()`` disables retries for latency-sensitive
    probes (health checks, interactive model listing).
    """

    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    backoff_factor: float = 2.0
    max_delay_seconds: float = 8.0

    @classmethod
    def single(cls) -> "RetryPolicy":
        """A no-retry policy: try once, surface the failure immediately."""
        return cls(max_attempts=1)

    def delay_for(self, attempt: int) -> float:
        """Seconds to wait after ``attempt`` (1-based) fails, before the next try."""
        raw = self.base_delay_seconds * (self.backoff_factor ** (attempt - 1))
        return min(raw, self.max_delay_seconds)


def is_transient(exc: BaseException) -> bool:
    """Whether ``exc`` is worth retrying.

    Transient: HTTP 429 (rate limit) and 5xx (server), plus connection-level
    failures (DNS/connection refused/reset, timeouts). Not transient: other 4xx
    (the request itself is wrong) and anything that is not a transport error.
    """
    # HTTPError is a subclass of both URLError and OSError, so it must be checked
    # first to read its status code.
    if isinstance(exc, HTTPError):
        return exc.code == 429 or 500 <= exc.code <= 599
    if isinstance(exc, (URLError, TimeoutError, ConnectionError, OSError)):
        return True
    return False


def call_with_retry(
    fn: Callable[[], T],
    policy: RetryPolicy,
    sleep: Callable[[float], Any] = time.sleep,
    on_retry: Callable[[int, BaseException, float], Any] | None = None,
) -> T:
    """Call ``fn`` with bounded backoff, retrying only transient failures.

    Re-raises immediately on a non-transient error or once attempts are
    exhausted, so the caller always sees the underlying exception (never a
    wrapped/generic one). ``on_retry(attempt, exc, delay)`` is invoked before
    each backoff sleep for observability/tests.
    """
    attempt = 1
    while True:
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised unless transient + attempts remain
            if attempt >= policy.max_attempts or not is_transient(exc):
                raise
            delay = policy.delay_for(attempt)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            sleep(delay)
            attempt += 1


def format_backend_error(exc: BaseException) -> str:
    """A concise, single-line description of a backend failure for a report field.

    HTTP errors lead with their status code (the most actionable fact); others
    use the exception type and message. Bounded so a giant error body can't bloat
    a report.
    """
    if isinstance(exc, HTTPError):
        detail = f"HTTP {exc.code}"
        reason = str(getattr(exc, "reason", "") or "").strip()
        return f"{detail} {reason}".strip()[:300]
    message = str(exc).strip() or exc.__class__.__name__
    return f"{exc.__class__.__name__}: {message}"[:300]
