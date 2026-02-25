import uuid
from server.main import app
from fastapi.testclient import TestClient
from server.db import create_db_and_tables

client = TestClient(app)


def setup_module(module):
    create_db_and_tables()


def _auth_header() -> dict:
    suffix = uuid.uuid4().hex[:8]
    email = f"lbagg_{suffix}@example.com"
    password = "Passw0rd!"
    r = client.post("/auth/register", json={"email": email, "password": password, "display_name": "Agg Tester"})
    assert r.status_code == 200
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_submit_and_aggregate():
    headers = _auth_header()
    resp = client.post("/leaderboard/submit", params={"score": 10}, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    uid = data["user_id"]  # now comes from current_user
    assert data["best_score"] == 10

    resp = client.post("/leaderboard/submit", params={"score": 15}, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["best_score"] == 15
    assert data["sessions"] == 2

    resp = client.get("/leaderboard/aggregates", params={"order_by":"best"}, headers=headers)
    assert resp.status_code == 200
    arr = resp.json()
    assert any(e["user_id"] == uid for e in arr)
