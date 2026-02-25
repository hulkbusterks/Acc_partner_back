import time
from typing import Any, Optional


class TTLCache:
    def __init__(self, ttl: float = 5.0):
        self.ttl = ttl
        self.store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        item = self.store.get(key)
        if not item:
            return None
        ts, val = item
        if now - ts > self.ttl:
            del self.store[key]
            return None
        return val

    def set(self, key: str, val: Any):
        self.store[key] = (time.time(), val)

    def clear(self):
        self.store.clear()

    def invalidate(self, key: str):
        self.store.pop(key, None)


cache = TTLCache(ttl=5.0)
