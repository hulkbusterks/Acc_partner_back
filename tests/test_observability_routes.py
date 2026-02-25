import os
from fastapi.testclient import TestClient

from server.main import app
from server.services.observability import observability


client = TestClient(app)


def test_internal_metrics_and_reset_roundtrip():
    observability.reset()
    observability.incr("unit_test_counter")
    observability.observe_ms("unit_test_timer", 12.5)
    observability.add_trace({"event": "unit-test"})

    token = os.getenv("ADMIN_TOKEN")
    headers = {"X-Admin-Token": token} if token else {}

    metrics = client.get("/internal/metrics", headers=headers)
    assert metrics.status_code == 200
    data = metrics.json()
    assert data["counters"]["unit_test_counter"] == 1
    assert data["timers"]["unit_test_timer"]["count"] == 1
    assert len(data["recent_traces"]) >= 1

    reset = client.post("/internal/metrics/reset", headers=headers)
    assert reset.status_code == 200

    post = client.get("/internal/metrics", headers=headers)
    assert post.status_code == 200
    post_data = post.json()
    assert post_data["counters"] == {}
    assert post_data["timers"] == {}
    assert post_data["recent_traces"] == []
