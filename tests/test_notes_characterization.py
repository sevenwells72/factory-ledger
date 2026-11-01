"""Characterization of the dashboard Notes endpoints (main.py modularization PR0).

PR1 moves `# DASHBOARD API — Notes / To-Dos / Reminders` and its two models out
of main.py. These tests pin what the five routes do, so the move can be proven
behaviour-neutral. The access rules (docs/security/notes-endpoints.md) changed
in their own reviewed PR and must only change that way again:

  * GET /dashboard/api/notes needs a key, like the writes: master, the
    dashboard key, a personal actor key or a session. The read-only GPT key is
    refused;
  * every Notes write by a named person (personal actor key or session) records
    an actor_write_audit row; the shared master and dashboard keys name no one.

Already covered elsewhere and not repeated here: 401/403 without/with a wrong
key and the 404s (test_notes_auth.py), session attribution of actor id and
name (test_key_sessions.py), the POST readonly tripwire
(test_readonly_tripwire.py), the write envelope (test_write_response_contract.py).

All DB writes run inside the test's rolled-back transaction on the local test
database (tests/conftest.py refuses production hosts).
"""

import re
from uuid import uuid4

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.routing import Mount

import main
from tests.test_actor_attribution import actors, client  # noqa: F401  (fixtures)
from tests.test_readonly_tripwire import _ReadOnlyError, _make_raising_get_db_connection

NOTES = "/dashboard/api/notes"
READONLY_KEY = "notes-characterization-readonly-key"
NOTE_KEYS = {"id", "category", "title", "body", "priority", "status", "due_date",
             "entity_type", "entity_id", "created_at", "updated_at"}
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} (AM|PM) ET$")

# (method, path template, handler name) in registration order.
NOTES_ROUTES = [
    ("GET", NOTES, "dashboard_api_notes"),
    ("POST", NOTES, "dashboard_api_notes_create"),
    ("PUT", NOTES + "/{note_id}", "dashboard_api_notes_update"),
    ("DELETE", NOTES + "/{note_id}", "dashboard_api_notes_delete"),
    ("PUT", NOTES + "/{note_id}/toggle", "dashboard_api_notes_toggle"),
]
WRITE_ROUTES = NOTES_ROUTES[1:]


def _key(key):
    return {"X-API-Key": key}


def _master():
    return _key(main.API_KEY)


def _create(client, key=None, **fields):
    body = {"category": "note", "title": "char " + uuid4().hex[:8], **fields}
    resp = client.post(NOTES, json=body, headers=_key(key or main.API_KEY))
    assert resp.status_code == 200, resp.text
    return resp.json()


def _write_calls(note_id):
    """One request per write route: (method, url, json body)."""
    return [
        ("POST", NOTES, {"category": "note", "title": "char write"}),
        ("PUT", f"{NOTES}/{note_id}", {"title": "char updated"}),
        ("PUT", f"{NOTES}/{note_id}/toggle", None),
        ("DELETE", f"{NOTES}/{note_id}", None),
    ]


def _audit_rows(db_cursor, note_id):
    db_cursor.execute(
        "SELECT actor_id, operator_id, method, route, target_table, target_id "
        "FROM actor_write_audit WHERE target_table = 'notes' AND target_id = %s ORDER BY id",
        (note_id,))
    return db_cursor.fetchall()


@pytest.fixture
def readonly_key(monkeypatch):
    monkeypatch.setattr(main, "READONLY_API_KEY", READONLY_KEY)
    return READONLY_KEY


@pytest.fixture
def sessions_on(monkeypatch):
    monkeypatch.setenv("KEY_SESSIONS_ENABLED", "1")
    monkeypatch.setenv("PIN_LOGIN_ENABLED", "0")


def _session_headers(client, actor):
    resp = client.post("/auth/session", json={"actor_key": actor["key"]})
    assert resp.status_code == 200, resp.text
    return _key(resp.json()["session_token"])


# ─────────────────────────────────────────────────────────────────
# Route registration (no database)
# ─────────────────────────────────────────────────────────────────

def _notes_routes():
    return [(i, r) for i, r in enumerate(main.app.routes)
            if isinstance(r, APIRoute) and r.path.startswith(NOTES)]


def test_notes_routes_registered_in_order():
    found = [(next(iter(r.methods)), r.path, r.name) for _, r in _notes_routes()]
    assert found == NOTES_ROUTES
    indexes = [i for i, _ in _notes_routes()]
    assert indexes == list(range(indexes[0], indexes[0] + len(indexes))), "notes routes must stay contiguous"


def test_notes_routes_precede_the_dashboard_static_mount():
    mount = next(i for i, r in enumerate(main.app.routes) if isinstance(r, Mount) and r.path == "/dashboard")
    assert all(i < mount for i, _ in _notes_routes())


def test_notes_route_contracts():
    routes = {(next(iter(r.methods)), r.path): r for _, r in _notes_routes()}
    for method, path, _ in NOTES_ROUTES:
        route = routes[(method, path)]
        assert route.status_code is None and route.response_model is None
        assert route.response_class is main.DecimalSafeJSONResponse
    get = routes[("GET", NOTES)]
    # The list endpoint takes four optional filters; its key arrives through the
    # same dependency as the writes.
    assert [p.name for p in get.dependant.query_params] == ["category", "status", "entity_type", "entity_id"]
    assert not any(p.required for p in get.dependant.query_params)
    assert get.dependant.header_params == []
    for method, path, _ in NOTES_ROUTES:
        deps = [d.call for d in routes[(method, path)].dependant.dependencies]
        assert deps == [main.verify_api_key], (method, path)
    assert routes[("POST", NOTES)].dependant.body_params[0].field_info.annotation is main.NoteCreate
    assert routes[("PUT", NOTES + "/{note_id}")].dependant.body_params[0].field_info.annotation is main.NoteUpdate


def test_note_models_contract():
    assert list(main.NoteCreate.model_fields) == [
        "category", "title", "body", "priority", "due_date", "entity_type", "entity_id"]
    assert [n for n, f in main.NoteCreate.model_fields.items() if f.is_required()] == ["category", "title"]
    assert list(main.NoteUpdate.model_fields) == [
        "title", "body", "priority", "status", "due_date", "entity_type", "entity_id"]
    assert not any(f.is_required() for f in main.NoteUpdate.model_fields.values())


# ─────────────────────────────────────────────────────────────────
# Authentication matrix
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
@pytest.mark.parametrize("which", ["master", "dashboard", "actor_floor", "actor_office", "session"])
def test_get_notes_accepted(client, actors, sessions_on, which):
    keys = {"master": _master(), "dashboard": _key(main.DASHBOARD_API_KEY),
            "actor_floor": _key(actors["floor"]["key"]), "actor_office": _key(actors["office"]["key"])}
    h = _session_headers(client, actors["office"]) if which == "session" else keys[which]
    note = _create(client, title="listed " + uuid4().hex[:6])
    resp = client.get(NOTES, headers=h)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"notes"}  # no write envelope on the read
    assert any(n["id"] == note["id"] for n in body["notes"])


@pytest.mark.db
@pytest.mark.parametrize("which,status,detail", [
    ("none", 401, "API key required"),
    ("bogus", 403, "Invalid API key"),
    ("readonly", 403, "API key not authorized for this endpoint"),
    ("retired_actor", 403, "Invalid API key"),
])
def test_get_notes_rejected(client, actors, readonly_key, which, status, detail):
    keys = {"none": {}, "bogus": _key("not-a-key"), "readonly": _key(readonly_key),
            "retired_actor": _key(actors["retired"]["key"])}
    _create(client, title="hidden " + uuid4().hex[:6])
    resp = client.get(NOTES, headers=keys[which])
    assert resp.status_code == status, resp.text
    assert resp.json() == {"detail": detail}  # reads carry no write envelope


@pytest.mark.db
@pytest.mark.parametrize("which", ["master", "dashboard", "actor_floor", "actor_office", "session"])
def test_writes_accepted(client, actors, sessions_on, which):
    keys = {"master": _master(), "dashboard": _key(main.DASHBOARD_API_KEY),
            "actor_floor": _key(actors["floor"]["key"]), "actor_office": _key(actors["office"]["key"])}
    h = _session_headers(client, actors["floor"]) if which == "session" else keys[which]
    note = _create(client)
    for method, url, body in _write_calls(note["id"]):
        resp = client.request(method, url, json=body, headers=h)
        assert resp.status_code == 200, (which, method, url, resp.text)
        assert resp.json()["success"] is True


@pytest.mark.db
@pytest.mark.parametrize("which,detail", [
    ("readonly", "API key not authorized for this endpoint"),
    ("retired_actor", "Invalid API key"),
])
def test_writes_rejected_and_nothing_changes(client, actors, readonly_key, db_cursor, which, detail):
    h = _key(readonly_key) if which == "readonly" else _key(actors["retired"]["key"])
    note = _create(client, title="untouched")
    db_cursor.execute("SELECT count(*) AS n FROM notes")
    before = db_cursor.fetchone()["n"]
    for method, url, body in _write_calls(note["id"]):
        resp = client.request(method, url, json=body, headers=h)
        assert resp.status_code == 403, (which, method, url, resp.text)
        assert resp.json()["detail"] == detail
        assert resp.json()["success"] is False
    db_cursor.execute("SELECT count(*) AS n FROM notes")
    assert db_cursor.fetchone()["n"] == before
    db_cursor.execute("SELECT title, status FROM notes WHERE id = %s", (note["id"],))
    assert dict(db_cursor.fetchone()) == {"title": "untouched", "status": "open"}


# ─────────────────────────────────────────────────────────────────
# Actor attribution
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
@pytest.mark.parametrize("which", ["session", "actor_key"])
def test_named_writes_record_method_route_and_target(client, actors, sessions_on, db_cursor, which):
    h = _session_headers(client, actors["office"]) if which == "session" else _key(actors["office"]["key"])
    created = client.post(NOTES, json={"category": "todo", "title": "attributed"}, headers=h)
    assert created.status_code == 200, created.text
    nid = created.json()["id"]
    for method, url, body in _write_calls(nid)[1:]:
        assert client.request(method, url, json=body, headers=h).status_code == 200
    rows = [dict(r) for r in _audit_rows(db_cursor, nid)]
    office = actors["office"]
    assert rows == [
        {"actor_id": office["id"], "operator_id": office["name"], "method": m,
         "route": route, "target_table": "notes", "target_id": nid}
        for m, route in [("POST", NOTES), ("PUT", NOTES + "/{note_id}"),
                         ("PUT", NOTES + "/{note_id}/toggle"), ("DELETE", NOTES + "/{note_id}")]
    ]


@pytest.mark.db
def test_floor_actor_key_writes_are_attributed(client, actors, sessions_on, db_cursor):
    floor = actors["floor"]
    h = _key(floor["key"])
    created = client.post(NOTES, json={"category": "note", "title": "floor note"}, headers=h)
    assert created.status_code == 200, created.text
    nid = created.json()["id"]
    for method, url, body in _write_calls(nid)[1:]:
        assert client.request(method, url, json=body, headers=h).status_code == 200
    rows = _audit_rows(db_cursor, nid)
    assert len(rows) == 4
    assert all((r["actor_id"], r["operator_id"]) == (floor["id"], floor["name"]) for r in rows)


@pytest.mark.db
@pytest.mark.parametrize("which", ["master", "dashboard"])
def test_shared_key_writes_record_no_attribution(client, actors, sessions_on, db_cursor, which):
    """The shared keys name a surface, not a person (`_record_actor_write` is a no-op)."""
    h = {"master": _master(), "dashboard": _key(main.DASHBOARD_API_KEY)}[which]
    created = client.post(NOTES, json={"category": "note", "title": "unattributed"}, headers=h)
    assert created.status_code == 200, created.text
    nid = created.json()["id"]
    for method, url, body in _write_calls(nid)[1:]:
        assert client.request(method, url, json=body, headers=h).status_code == 200
    assert _audit_rows(db_cursor, nid) == []


# ─────────────────────────────────────────────────────────────────
# Request / response contracts
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_create_defaults_and_shape(client):
    note = _create(client, title="shape")
    assert set(note) == NOTE_KEYS | {"success"}
    assert note["success"] is True
    assert (note["category"], note["body"], note["priority"], note["status"]) == ("note", "", "normal", "open")
    assert note["due_date"] is None and note["entity_type"] is None and note["entity_id"] is None
    assert TIMESTAMP.match(note["created_at"]) and TIMESTAMP.match(note["updated_at"])


@pytest.mark.db
def test_create_with_every_field(client):
    note = _create(client, category="reminder", title="full", body="b", priority="high",
                   due_date="2026-11-05", entity_type="lot", entity_id="42")
    assert {k: note[k] for k in ("category", "title", "body", "priority", "due_date", "entity_type", "entity_id")} == {
        "category": "reminder", "title": "full", "body": "b", "priority": "high",
        "due_date": "2026-11-05", "entity_type": "lot", "entity_id": "42"}


@pytest.mark.db
def test_update_is_partial_and_due_date_empty_string_clears(client):
    note = _create(client, title="partial", body="keep", due_date="2026-12-01")
    resp = client.put(f"{NOTES}/{note['id']}", json={"priority": "low", "status": "dismissed"}, headers=_master())
    assert resp.status_code == 200, resp.text
    updated = resp.json()
    assert (updated["title"], updated["body"], updated["due_date"]) == ("partial", "keep", "2026-12-01")
    assert (updated["priority"], updated["status"]) == ("low", "dismissed")
    cleared = client.put(f"{NOTES}/{note['id']}", json={"due_date": ""}, headers=_master()).json()
    assert cleared["due_date"] is None and cleared["title"] == "partial"


@pytest.mark.db
def test_toggle_flips_open_and_done(client):
    note = _create(client)
    url = f"{NOTES}/{note['id']}/toggle"
    assert client.put(url, headers=_master()).json()["status"] == "done"
    assert client.put(url, headers=_master()).json()["status"] == "open"
    client.put(f"{NOTES}/{note['id']}", json={"status": "dismissed"}, headers=_master())
    assert client.put(url, headers=_master()).json()["status"] == "open"  # anything not open -> open


@pytest.mark.db
def test_delete_response(client):
    note = _create(client)
    resp = client.delete(f"{NOTES}/{note['id']}", headers=_master())
    assert resp.status_code == 200
    assert resp.json() == {"deleted": True, "id": note["id"], "success": True}


@pytest.mark.db
@pytest.mark.parametrize("method,path_suffix,body,status,expected", [
    ("POST", "", {"category": "memo", "title": "x"}, 422,
     "Value error, category must be 'note', 'todo', or 'reminder'"),
    ("POST", "", {"category": "note", "title": "x", "priority": "urgent"}, 422,
     "Value error, priority must be 'low', 'normal', or 'high'"),
    ("POST", "", {"category": "note"}, 422, "Field required"),
    ("POST", "", {"category": "note", "title": "x", "due_date": "13/01/2026"}, 400,
     "Invalid isoformat string: '13/01/2026'"),
    ("PUT", "/{id}", {}, 400, "No fields to update"),
    ("PUT", "/{id}", {"due_date": "nope"}, 400, "Invalid isoformat string: 'nope'"),
    ("PUT", "/{id}", {"status": "bogus"}, 422,
     "Value error, status must be 'open', 'done', or 'dismissed'"),
    ("PUT", "/{id}", {"priority": "urgent"}, 422,
     "Value error, priority must be 'low', 'normal', or 'high'"),
    ("PUT", "/abc", {"title": "x"}, 422,
     "Input should be a valid integer, unable to parse string as an integer"),
])
def test_validation_errors(client, method, path_suffix, body, status, expected):
    note = _create(client)
    resp = client.request(method, NOTES + path_suffix.replace("{id}", str(note["id"])), json=body, headers=_master())
    assert resp.status_code == status, resp.text
    data = resp.json()
    assert data["success"] is False
    assert data["error_detail"]["message"] == expected
    if status == 400:
        assert data["error"] == expected and data["error_detail"]["code"] == "HTTP_400"
    else:
        assert data["error_detail"]["code"] == "VALIDATION_ERROR"


@pytest.mark.db
def test_list_filters_and_ordering(client):
    tag = uuid4().hex[:10]
    made = {}
    for label, fields in [
        ("done_high", {"priority": "high"}),
        ("open_low", {"priority": "low"}),
        ("open_normal_late", {"priority": "normal", "due_date": "2026-12-31"}),
        ("open_normal_soon", {"priority": "normal", "due_date": "2026-11-02"}),
        ("open_normal_nodate", {"priority": "normal"}),
        ("open_high", {"priority": "high"}),
    ]:
        made[label] = _create(client, category="todo", title=label, entity_type="lot", entity_id=tag, **fields)["id"]
    client.put(f"{NOTES}/{made['done_high']}/toggle", headers=_master())
    _create(client, category="note", title="other category", entity_type="lot", entity_id=tag)

    listed = client.get(NOTES, params={"category": "todo", "entity_type": "lot", "entity_id": tag},
                        headers=_master()).json()["notes"]
    # open before closed; then high, normal, low; then due date ascending, undated last
    assert [n["title"] for n in listed] == [
        "open_high", "open_normal_soon", "open_normal_late", "open_normal_nodate", "open_low", "done_high"]
    assert all(set(n) == NOTE_KEYS for n in listed)
    only_done = client.get(NOTES, params={"status": "done", "entity_id": tag}, headers=_master()).json()["notes"]
    assert [n["title"] for n in only_done] == ["done_high"]
    assert client.get(NOTES, params={"entity_id": "no-such-" + tag}, headers=_master()).json() == {"notes": []}


# ─────────────────────────────────────────────────────────────────
# Read-only database and error paths (no database: get_db_connection is faked)
# ─────────────────────────────────────────────────────────────────

def _raw_client():
    # No `with`: no startup, so no database. Server errors become responses.
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.mark.parametrize("method,path,body", [
    ("PUT", NOTES + "/1", {"title": "x"}),
    ("DELETE", NOTES + "/1", None),
    ("PUT", NOTES + "/1/toggle", None),
])
def test_readonly_database_reaches_global_tripwire(monkeypatch, method, path, body):
    """Each write handler re-raises a read-only error (`if _is_readonly_error(e): raise`)
    instead of returning its own 500. POST is covered in test_readonly_tripwire.py."""
    monkeypatch.setattr(main, "get_db_connection", _make_raising_get_db_connection(_ReadOnlyError()))
    monkeypatch.setattr(main, "_capture_readonly_diagnostics", lambda: {"db": "fake"})
    resp = _raw_client().request(method, path, json=body, headers=_master())
    assert resp.status_code == 503, resp.text
    assert resp.json()["error_code"] == "READONLY_TRANSACTION"


@pytest.mark.parametrize("method,path,body", [
    ("GET", NOTES, None),
    ("POST", NOTES, {"category": "note", "title": "x"}),
    ("PUT", NOTES + "/1", {"title": "x"}),
    ("DELETE", NOTES + "/1", None),
    ("PUT", NOTES + "/1/toggle", None),
])
def test_other_database_errors_return_route_500(monkeypatch, method, path, body):
    monkeypatch.setattr(main, "get_db_connection", _make_raising_get_db_connection(RuntimeError("boom")))
    resp = _raw_client().request(method, path, json=body, headers=_master())
    assert resp.status_code == 500, resp.text
    data = resp.json()
    assert data["error"] == "boom"
    assert "error_code" not in data
    if method == "GET":
        assert data == {"error": "boom"}  # reads carry no write envelope
    else:
        assert data["success"] is False
