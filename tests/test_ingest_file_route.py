import uuid
from fastapi.testclient import TestClient

from server.main import app


client = TestClient(app)


def _auth_token() -> str:
    suffix = uuid.uuid4().hex[:8]
    email = f"ingest_file_{suffix}@example.com"
    password = "Passw0rd!"

    r = client.post("/auth/register", json={"email": email, "password": password, "display_name": "Ingest File"})
    assert r.status_code == 200

    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200
    return r.json()["token"]


def test_ingest_file_txt():
    token = _auth_token()
    files = {"file": ("sample.txt", b"This is content from uploaded file.", "text/plain")}
    r = client.post(
        "/ingest/file",
        params={"title": "Uploaded TXT", "authors": "Tester"},
        headers={"Authorization": f"Bearer {token}"},
        files=files,
    )
    assert r.status_code == 200
    data = r.json()
    assert "book_id" in data
    assert data.get("chars", 0) > 0
