from collections import deque
from threading import Lock
from time import time
from typing import Deque, Dict


class InMemoryRateLimiter:
    def __init__(self):
        self._events: Dict[str, Deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str, limit: int, window_sec: int) -> bool:
        if limit <= 0 or window_sec <= 0:
            return True

        now = time()
        cutoff = now - float(window_sec)

        with self._lock:
            bucket = self._events.setdefault(key, deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


rate_limiter = InMemoryRateLimiter()
