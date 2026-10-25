import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """Per-process sliding-window limiter; use an API gateway for multi-replica deployments."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            queue = self.events[key]
            while queue and queue[0] <= now - 60:
                queue.popleft()
            if len(queue) >= self.limit:
                return False
            queue.append(now)
            if len(self.events) > 10000:
                expired = [item for item, timestamps in self.events.items() if not timestamps or timestamps[-1] <= now - 60]
                for item in expired:
                    self.events.pop(item, None)
            return True


class EventDeduplicator:
    """Bounds Slack Events API retries so a retried delivery does not run a query twice."""

    def __init__(self, retention_seconds: int = 3600, max_entries: int = 20000) -> None:
        self.retention_seconds = retention_seconds
        self.max_entries = max_entries
        self.seen_ids: dict[str, float] = {}
        self._lock = threading.Lock()

    def is_duplicate(self, event_id: str | None) -> bool:
        if not event_id:
            return False
        now = time.monotonic()
        with self._lock:
            previous = self.seen_ids.get(event_id)
            if previous is not None and previous > now - self.retention_seconds:
                return True
            self.seen_ids[event_id] = now
            if len(self.seen_ids) > self.max_entries:
                self.seen_ids = {key: stamp for key, stamp in self.seen_ids.items()
                                 if stamp > now - self.retention_seconds}
            return False
