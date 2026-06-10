"""Auth tests for the dashboard notes write endpoints (branch fix/auth-unauthenticated-writes).

June 9 audit follow-up: the four /dashboard/api/notes* write endpoints were the
only mutating routes without the X-API-Key requirement. They now use the same
`verify_api_key` dependency as every other write route:
  * missing X-API-Key header -> 401 "API key required"
  * wrong key                -> 403 "Invalid API key"
  * correct key              -> unchanged behavior (success or normal domain error)

Same TestClient + savepoint-proxy pattern as test_write_response_contract.py.
"""

from contextlib import contextmanager

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    pytest.skip("fastapi/httpx not installed", allow_module_level=True)

try:
    import main
except Exception as e:  # pragma: no cover
    pytest.skip(f"cannot import main ({e})", allow_module_level=True)


class _ConnProxy:
    """Wrap a real psycopg2 connection so commit()/rollback() operate on an
    inner SAVEPOINT; the outer db fixtures roll everything back on teardown."""

    def __init__(self, conn, sp_name):
        self._conn = conn
        self._sp = sp_name
        with self._conn.cursor() as c:
            c.execute(f"SAVEPOINT {self._sp}")

    def cursor(self, *args, **kwargs):
        return self._conn.cursor(*args, **kwargs)

    def commit(self):
        with self._conn.cursor() as c:
            c.execute(f"RELEASE SAVEPOINT {self._sp}")
            c.execute(f"SAVEPOINT {self._sp}")

    def rollback(self):
        with self._conn.cursor() as c:
            c.execute(f"ROLLBACK TO SAVEPOINT {self._sp}")
            c.execute(f"SAVEPOINT {self._sp}")


@pytest.fixture
def client(_db_connection, monkeypatch):
    """TestClient with NO default X-API-Key header — each test passes (or
    omits) the key explicitly via `headers=`."""

    @contextmanager
    def _fake_get_conn():
        proxy = _ConnProxy(_db_connection, "notes_auth_inner")
        try:
            yield proxy
            proxy.commit()
        except Exception:
            proxy.rollback()
            raise

    monkeypatch.setattr(main, "get_db_connection", _fake_get_conn)
    with TestClient(main.app) as c:
        yield c
    _db_connection.rollback()


def _key():
    return {"X-API-Key": main.API_KEY}


def _create_note(client, title="notes auth test"):
    resp = client.post(
        "/dashboard/api/notes",
        json={"category": "note", "title": title},
        headers=_key(),
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ─────────────────────────────────────────────────────────────────
# (a) Without the key: 401. With the wrong key: 403.
# ─────────────────────────────────────────────────────────────────

def _no_key_calls(client):
    return [
        ("POST /notes", client.post(
            "/dashboard/api/notes", json={"category": "note", "title": "x"})),
        ("PUT /notes/{id}", client.put(
            "/dashboard/api/notes/1", json={"title": "x"})),
        ("DELETE /notes/{id}", client.delete("/dashboard/api/notes/1")),
        ("PUT /notes/{id}/toggle", client.put("/dashboard/api/notes/1/toggle")),
    ]


@pytest.mark.db
def test_notes_writes_401_without_key(client):
    for label, resp in _no_key_calls(client):
        assert resp.status_code == 401, f"{label}: expected 401, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert body["detail"] == "API key required", label
        # write_response_envelope still applies on auth failures
        assert body["success"] is False, label


@pytest.mark.db
def test_notes_writes_403_with_wrong_key(client):
    bad = {"X-API-Key": "definitely-not-the-key"}
    calls = [
        client.post("/dashboard/api/notes",
                    json={"category": "note", "title": "x"}, headers=bad),
        client.put("/dashboard/api/notes/1", json={"title": "x"}, headers=bad),
        client.delete("/dashboard/api/notes/1", headers=bad),
        client.put("/dashboard/api/notes/1/toggle", headers=bad),
    ]
    for resp in calls:
        assert resp.status_code == 403, f"{resp.request.method} {resp.request.url}: {resp.status_code}"
        assert resp.json()["detail"] == "Invalid API key"


@pytest.mark.db
def test_notes_create_rejected_without_key_leaves_no_row(client):
    title = "must-not-exist-no-key-zqx"
    resp = client.post("/dashboard/api/notes",
                       json={"category": "note", "title": title})
    assert resp.status_code == 401
    listing = client.get("/dashboard/api/notes")  # list endpoint is read-only, still no auth
    assert listing.status_code == 200
    assert all(n["title"] != title for n in listing.json()["notes"])


# ─────────────────────────────────────────────────────────────────
# (b) With the key: behavior unchanged (success or normal domain error)
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_notes_create_with_key(client):
    note = _create_note(client, title="auth test create")
    assert note["id"]
    assert note["title"] == "auth test create"


@pytest.mark.db
def test_notes_update_with_key(client):
    note = _create_note(client, title="before update")
    resp = client.put(f"/dashboard/api/notes/{note['id']}",
                      json={"title": "after update"}, headers=_key())
    assert resp.status_code == 200, resp.text
    assert resp.json()["title"] == "after update"


@pytest.mark.db
def test_notes_delete_with_key(client):
    note = _create_note(client, title="to be deleted")
    resp = client.delete(f"/dashboard/api/notes/{note['id']}", headers=_key())
    assert resp.status_code == 200, resp.text
    assert resp.json()["deleted"] is True


@pytest.mark.db
def test_notes_toggle_with_key(client):
    note = _create_note(client, title="to be toggled")
    assert note["status"] == "open"
    resp = client.put(f"/dashboard/api/notes/{note['id']}/toggle", headers=_key())
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "done"


@pytest.mark.db
def test_notes_domain_errors_unchanged_with_key(client):
    """Auth must not swallow the endpoints' normal 404s."""
    for resp in [
        client.put("/dashboard/api/notes/999999999",
                   json={"title": "x"}, headers=_key()),
        client.delete("/dashboard/api/notes/999999999", headers=_key()),
        client.put("/dashboard/api/notes/999999999/toggle", headers=_key()),
    ]:
        assert resp.status_code == 404, resp.text
        assert resp.json()["error"] == "Note not found"
