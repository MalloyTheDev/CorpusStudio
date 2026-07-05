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

import random
import time
from collections.abc import Callable
from datetime import datetime, timezone
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
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


def _apply_jitter(delay: float, rng: Callable[[], float]) -> float:
    """Equal jitter: keep at least half the computed backoff and randomize the rest, so many
    clients backing off the same provider don't retry in lockstep (a thundering herd). ``rng``
    returns a value in [0, 1); with ``rng`` fixed at 1.0 this is the full delay (deterministic
    for tests)."""

    return delay * 0.5 + delay * 0.5 * rng()


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Seconds requested by a 429/503 ``Retry-After`` header, or ``None`` if absent/unparseable.

    The header is either an integer number of seconds or an HTTP date. Honoring it lets a
    provider that knows its own rate-limit window tell us exactly when to come back.
    """

    if not isinstance(exc, HTTPError):
        return None
    headers = getattr(exc, "headers", None)
    raw = headers.get("Retry-After") if headers is not None else None
    if not raw:
        return None
    text = str(raw).strip()
    if text.isdigit():
        return float(text)
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


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
    *,
    deadline_seconds: float | None = None,
    rng: Callable[[], float] = random.random,
    now: Callable[[], float] = time.monotonic,
) -> T:
    """Call ``fn`` with bounded backoff, retrying only transient failures.

    Re-raises immediately on a non-transient error or once attempts are
    exhausted, so the caller always sees the underlying exception (never a
    wrapped/generic one). ``on_retry(attempt, exc, delay)`` is invoked before
    each backoff sleep for observability/tests.

    The backoff delay is: a provider's ``Retry-After`` (429/503) when present, capped at
    ``max_delay_seconds`` and honored verbatim (no jitter); otherwise the exponential schedule
    with equal jitter so concurrent clients don't retry in lockstep. ``deadline_seconds`` is an
    optional wall-clock budget for the whole sequence — when the next backoff would exceed it the
    call gives up and re-raises rather than sleeping past the caller's timeout. ``rng`` and
    ``now`` are injectable so tests stay deterministic.
    """
    start = now()
    attempt = 1
    while True:
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised unless transient + attempts remain
            if attempt >= policy.max_attempts or not is_transient(exc):
                raise
            retry_after = _retry_after_seconds(exc)
            if retry_after is not None:
                delay = min(retry_after, policy.max_delay_seconds)
            else:
                delay = _apply_jitter(policy.delay_for(attempt), rng)
            # Give up rather than sleep past a caller's wall-clock budget.
            if deadline_seconds is not None and (now() - start) + delay >= deadline_seconds:
                raise
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
