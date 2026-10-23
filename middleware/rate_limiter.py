import hashlib
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from observability.metrics import record_audit_event


class RateLimitExceeded(RuntimeError):
    pass


class SlidingWindowRateLimiter:
    def __init__(self, limit: int = 30, window_seconds: int = 60):
        if limit < 1 or window_seconds < 1:
            raise ValueError("Rate limit and window must be positive.")
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            requests = self._requests[key]
            while requests and now - requests[0] >= self.window_seconds:
                requests.popleft()
            if len(requests) >= self.limit:
                raise RateLimitExceeded(
                    f"Rate limit exceeded: maximum {self.limit} queries per "
                    f"{self.window_seconds} seconds."
                )
            requests.append(now)


QUERY_RATE_LIMIT = int(os.getenv("AI_QUERY_RATE_LIMIT", "30"))
rate_limiter = SlidingWindowRateLimiter(limit=QUERY_RATE_LIMIT)


def audit_query(
    session_id: str,
    query: str | None,
    route: str,
    outcome: str,
    request_id: str | None = None,
) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "route": route,
        "outcome": outcome,
    }
    if request_id is not None:
        event["request_id"] = request_id
    if isinstance(query, str):
        event["query_sha256"] = hashlib.sha256(query.encode("utf-8")).hexdigest()
    record_audit_event(event)
