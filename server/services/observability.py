from collections import deque
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Deque, Dict


class InMemoryObservability:
    def __init__(self, max_traces: int = 200):
        self._lock = Lock()
        self._counters: Dict[str, int] = {}
        self._timers: Dict[str, Dict[str, float]] = {}
        self._traces: Deque[Dict[str, Any]] = deque(maxlen=max_traces)

    def incr(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = int(self._counters.get(name, 0) + value)

    def observe_ms(self, name: str, value_ms: float) -> None:
        val = float(value_ms)
        with self._lock:
            stat = self._timers.get(name)
            if not stat:
                self._timers[name] = {
                    "count": 1,
                    "sum_ms": val,
                    "min_ms": val,
                    "max_ms": val,
                }
                return
            stat["count"] += 1
            stat["sum_ms"] += val
            stat["min_ms"] = min(stat["min_ms"], val)
            stat["max_ms"] = max(stat["max_ms"], val)

    def add_trace(self, event: Dict[str, Any]) -> None:
        payload = dict(event or {})
        payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with self._lock:
            self._traces.append(payload)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            timers = {}
            for name, stat in self._timers.items():
                count = float(stat["count"])
                timers[name] = {
                    "count": int(stat["count"]),
                    "avg_ms": (stat["sum_ms"] / count) if count > 0 else 0.0,
                    "min_ms": stat["min_ms"],
                    "max_ms": stat["max_ms"],
                }
            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "counters": dict(self._counters),
                "timers": timers,
                "recent_traces": list(self._traces),
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._timers.clear()
            self._traces.clear()


observability = InMemoryObservability()
