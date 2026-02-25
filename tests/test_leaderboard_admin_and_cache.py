from server.main import app
from fastapi.testclient import TestClient
from server.db import create_db_and_tables
import os
import uuid

client = TestClient(app)


def setup_module(module):
    create_db_and_tables()


def _auth_header() -> dict:
    suffix = uuid.uuid4().hex[:8]
    email = f"lbadmin_{suffix}@example.com"
    password = "Passw0rd!"
    r = client.post("/auth/register", json={"email": email, "password": password, "display_name": "Admin Tester"})
    assert r.status_code == 200
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_reset_and_cache():
    headers = _auth_header()
    # submit
    r = client.post("/leaderboard/submit", params={"score": 5}, headers=headers)
    assert r.status_code == 200
    uid = r.json()["user_id"]  # now comes from current_user
    # fetch to populate cache
    r = client.get(f"/leaderboard/aggregate/{uid}", headers=headers)
    assert r.status_code == 200
    data1 = r.json()
    # reset without token should fail
    r = client.post("/leaderboard/reset", headers={"X-Admin-Token": "wrong"})
    assert r.status_code == 401
    # set token and reset
    os.environ["ADMIN_TOKEN"] = "letmein"
    r = client.post("/leaderboard/reset", headers={"X-Admin-Token": "letmein"})
    assert r.status_code == 200
    # subsequent fetch should 404
    r = client.get(f"/leaderboard/aggregate/{uid}", headers=headers)
    assert r.status_code == 404
