import uuid
from server.main import app
from fastapi.testclient import TestClient
from server.db import create_db_and_tables

client = TestClient(app)


def setup_module(module):
    create_db_and_tables()


def _auth_header() -> dict:
    suffix = uuid.uuid4().hex[:8]
    email = f"lbind_{suffix}@example.com"
    password = "Passw0rd!"
    r = client.post("/auth/register", json={"email": email, "password": password, "display_name": "Ind Tester"})
    assert r.status_code == 200
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_individual_aggregate_endpoint():
    headers = _auth_header()
    # submit two sessions
    resp = client.post("/leaderboard/submit", params={"score": 3}, headers=headers)
    assert resp.status_code == 200
    uid = resp.json()["user_id"]  # now comes from current_user
    resp = client.post("/leaderboard/submit", params={"score": 2}, headers=headers)
    assert resp.status_code == 200
    # fetch individual aggregate
    resp = client.get(f"/leaderboard/aggregate/{uid}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == uid
    assert data["best_score"] >= 3
    assert data["sessions"] >= 2