"""Tests for the uniform write-response contract (branch fix/write-response-contract).

Contract under test (additive — June 10 2026 audit follow-up):
  * Every write endpoint's happy path returns `success: true` AND the id of
    the record it created/modified.
  * Every failure returns `success: false` plus a structured
    `error_detail: {code, message}` — regardless of whether the endpoint
    raised an HTTPException or returned a JSONResponse directly.
  * Existing response fields are untouched (additive envelope only), and
    endpoints that already set `success` themselves are not overwritten.

The envelope is applied by the `write_response_envelope` HTTP middleware in
main.py, so these tests go through a real ASGI round-trip (TestClient), not
direct function calls. DB writes are rolled back via the same savepoint
proxy pattern as test_ship_order_service_line.py.
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
    """TestClient against the real app; every endpoint DB write goes through
    the savepoint proxy and is rolled back when the test ends."""

    @contextmanager
    def _fake_get_conn():
        proxy = _ConnProxy(_db_connection, "contract_inner")
        try:
            yield proxy
            proxy.commit()
        except Exception:
            proxy.rollback()
            raise

    monkeypatch.setattr(main, "get_db_connection", _fake_get_conn)
    with TestClient(main.app) as c:
        c.headers["X-API-Key"] = main.API_KEY
        yield c
    # Roll the whole test's work back past the proxy savepoints.
    _db_connection.rollback()


# ─────────────────────────────────────────────────────────────────
# Seed helpers (all rolled back by the fixture)
# ─────────────────────────────────────────────────────────────────

def _seed_customer(conn, name="Contract Test Customer"):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO customers (name, active) VALUES (%s, true) RETURNING id",
            (name,),
        )
        return cur.fetchone()[0]


def _seed_product(conn, name="CONTRACT TEST GRANOLA ZQX"):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO products (name, type, uom, active) "
            "VALUES (%s, 'finished', 'lb', true) RETURNING id",
            (name,),
        )
        return cur.fetchone()[0]


def _seed_lot(conn, product_id, lot_code="CONTRACT-LOT-001"):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id",
            (product_id, lot_code),
        )
        return cur.fetchone()[0]


def _create_order(client, customer="Contract Test Customer",
                  product="CONTRACT TEST GRANOLA ZQX"):
    resp = client.post("/sales/orders", json={
        "customer_name": customer,
        "lines": [{"product_name": product, "quantity_lb": 100}],
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


# ─────────────────────────────────────────────────────────────────
# Happy paths: success + id on every write endpoint
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_create_sales_order_success_and_ids(client, _db_connection):
    _seed_customer(_db_connection)
    _seed_product(_db_connection)
    data = _create_order(client)
    assert data["success"] is True
    assert data["order_id"]
    assert data["lines"][0]["line_id"]
    # pre-existing fields untouched
    assert data["order_number"] and data["status"] == "confirmed"


@pytest.mark.db
def test_update_order_status_success_and_id(client, _db_connection):
    _seed_customer(_db_connection)
    _seed_product(_db_connection)
    order = _create_order(client)
    resp = client.patch(f"/sales/orders/{order['order_id']}/status",
                        json={"status": "in_production"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    assert data["order_id"] == order["order_id"]
    assert data["previous_status"] == "confirmed"
    assert data["status"] == "in_production"


@pytest.mark.db
def test_update_order_header_success_and_id(client, _db_connection):
    _seed_customer(_db_connection)
    _seed_product(_db_connection)
    order = _create_order(client)
    resp = client.patch(f"/sales/orders/{order['order_id']}",
                        json={"notes": "contract test note"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    assert data["order_id"] == order["order_id"]
    assert data["fields_updated"] == ["notes"]


@pytest.mark.db
def test_add_order_lines_success_and_ids(client, _db_connection):
    _seed_customer(_db_connection)
    _seed_product(_db_connection)
    order = _create_order(client)
    resp = client.post(f"/sales/orders/{order['order_id']}/lines", json={
        "lines": [{"product_name": "CONTRACT TEST GRANOLA ZQX", "quantity_lb": 50}],
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    assert data["order_id"] == order["order_id"]
    assert data["lines_added"][0]["line_id"]


@pytest.mark.db
def test_cancel_order_line_success_and_ids(client, _db_connection):
    _seed_customer(_db_connection)
    _seed_product(_db_connection)
    order = _create_order(client)
    line_id = order["lines"][0]["line_id"]
    resp = client.patch(f"/sales/orders/{order['order_id']}/lines/{line_id}/cancel")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    assert data["order_id"] == order["order_id"]
    assert data["line_id"] == line_id
    assert data["line_status"] == "cancelled"


@pytest.mark.db
def test_update_order_line_success_and_id(client, _db_connection):
    _seed_customer(_db_connection)
    _seed_product(_db_connection)
    order = _create_order(client)
    line_id = order["lines"][0]["line_id"]
    resp = client.patch(
        f"/sales/orders/{order['order_id']}/lines/{line_id}/update",
        params={"quantity_lb": 120},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    assert data["line_id"] == line_id
    assert data["quantity_lb"] == 120


@pytest.mark.db
def test_create_customer_success_and_id(client):
    resp = client.post("/customers", json={"name": "Contract New Customer LLC"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    assert data["customer_id"]
    assert data["name"] == "Contract New Customer LLC"


@pytest.mark.db
def test_note_create_success_and_id(client):
    resp = client.post("/dashboard/api/notes",
                       json={"category": "note", "title": "contract test"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    assert data["id"]
    assert data["title"] == "contract test"


@pytest.mark.db
def test_note_update_success_and_id(client):
    note = client.post("/dashboard/api/notes",
                       json={"category": "todo", "title": "before"}).json()
    resp = client.put(f"/dashboard/api/notes/{note['id']}",
                      json={"title": "after"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    assert data["id"] == note["id"]
    assert data["title"] == "after"


@pytest.mark.db
def test_note_delete_success_and_id(client):
    note = client.post("/dashboard/api/notes",
                       json={"category": "note", "title": "to delete"}).json()
    resp = client.delete(f"/dashboard/api/notes/{note['id']}")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    assert data["id"] == note["id"]
    assert data["deleted"] is True  # legacy field preserved


@pytest.mark.db
def test_note_toggle_success_and_id(client):
    note = client.post("/dashboard/api/notes",
                       json={"category": "todo", "title": "toggle me"}).json()
    resp = client.put(f"/dashboard/api/notes/{note['id']}/toggle")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    assert data["id"] == note["id"]
    assert data["status"] == "done"


@pytest.mark.db
def test_products_resolve_success(client, _db_connection):
    pid = _seed_product(_db_connection)
    resp = client.post("/products/resolve",
                       json={"names": ["CONTRACT TEST GRANOLA ZQX"]})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    # resolve creates no record; the ids it returns are the matched products
    assert data["resolved"][0]["match"]["id"] == pid
    assert data["summary"]["resolved"] == 1


@pytest.mark.db
def test_lot_reassign_success_and_reassignment_id(client, _db_connection):
    from_pid = _seed_product(_db_connection, "CONTRACT FROM PRODUCT ZQX")
    to_pid = _seed_product(_db_connection, "CONTRACT TO PRODUCT ZQX")
    lot_id = _seed_lot(_db_connection, from_pid)
    resp = client.post(f"/lots/{lot_id}/reassign", json={
        "to_product_id": to_pid,
        "reason_code": "data_entry_error",
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # pre-existing envelope fields (untouched by middleware)
    assert data["success"] is True
    assert data["lot_id"] == lot_id
    assert data["from_product"] == "CONTRACT FROM PRODUCT ZQX"
    # new: id of the lot_reassignments history row
    assert data["reassignment_id"], "reassignment_id missing/null on happy path"


# ─────────────────────────────────────────────────────────────────
# Failure paths: success=false + structured error_detail
# ─────────────────────────────────────────────────────────────────

def _assert_failure_envelope(resp, expected_status):
    assert resp.status_code == expected_status, resp.text
    data = resp.json()
    assert data["success"] is False
    assert isinstance(data["error_detail"], dict)
    assert data["error_detail"]["code"]
    assert data["error_detail"]["message"]
    return data


@pytest.mark.db
def test_duplicate_customer_fails_with_envelope(client):
    """Raised-HTTPException path (string detail)."""
    first = client.post("/customers", json={"name": "Contract Dup Customer"})
    assert first.json()["success"] is True
    resp = client.post("/customers", json={"name": "Contract Dup Customer"})
    data = _assert_failure_envelope(resp, 409)
    assert "detail" in data  # legacy key preserved


@pytest.mark.db
def test_invalid_status_transition_fails_with_envelope(client, _db_connection):
    _seed_customer(_db_connection)
    _seed_product(_db_connection)
    order = _create_order(client)
    resp = client.patch(f"/sales/orders/{order['order_id']}/status",
                        json={"status": "invoiced"})  # confirmed → invoiced: invalid
    _assert_failure_envelope(resp, 400)


@pytest.mark.db
def test_order_not_found_fails_with_structured_code(client):
    """Raised-HTTPException path with dict detail → error_code surfaces."""
    resp = client.patch("/sales/orders/SO-NOPE-999", json={"notes": "x"})
    data = _assert_failure_envelope(resp, 404)
    assert data["error_detail"]["code"] == "ORDER_NOT_FOUND"
    assert data["detail"]["error_code"] == "ORDER_NOT_FOUND"  # legacy preserved


@pytest.mark.db
def test_header_update_no_fields_fails_with_structured_code(client, _db_connection):
    _seed_customer(_db_connection)
    _seed_product(_db_connection)
    order = _create_order(client)
    resp = client.patch(f"/sales/orders/{order['order_id']}", json={})
    data = _assert_failure_envelope(resp, 400)
    assert data["error_detail"]["code"] == "NO_FIELDS_TO_UPDATE"


@pytest.mark.db
def test_note_bad_due_date_fails_with_envelope(client):
    """Returned-JSONResponse path ({"error": str}) — not a raised exception."""
    resp = client.post("/dashboard/api/notes", json={
        "category": "reminder", "title": "bad date", "due_date": "not-a-date",
    })
    data = _assert_failure_envelope(resp, 400)
    assert "error" in data  # legacy string key preserved alongside error_detail


@pytest.mark.db
def test_note_delete_missing_fails_with_envelope(client):
    resp = client.delete("/dashboard/api/notes/999999999")
    data = _assert_failure_envelope(resp, 404)
    assert data["error"] == "Note not found"


@pytest.mark.db
def test_lot_reassign_same_product_fails_with_envelope(client, _db_connection):
    pid = _seed_product(_db_connection, "CONTRACT SAME PRODUCT ZQX")
    lot_id = _seed_lot(_db_connection, pid, "CONTRACT-LOT-002")
    resp = client.post(f"/lots/{lot_id}/reassign", json={
        "to_product_id": pid, "reason_code": "data_entry_error",
    })
    data = _assert_failure_envelope(resp, 400)
    assert "already assigned" in data["error"]


@pytest.mark.db
def test_validation_error_gets_envelope(client):
    """FastAPI 422 (list-shaped detail) also carries the failure envelope."""
    resp = client.post("/customers", json={})  # missing required 'name'
    data = _assert_failure_envelope(resp, 422)
    assert data["error_detail"]["code"] == "VALIDATION_ERROR"


# ─────────────────────────────────────────────────────────────────
# Envelope hygiene
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_no_write_returns_200_without_an_id(client, _db_connection):
    """Sweep: every happy-path write above must include a record id. This
    re-asserts it in one place so a future endpoint regression fails loudly."""
    _seed_customer(_db_connection)
    _seed_product(_db_connection)
    order = _create_order(client)
    checks = [
        (order, "order_id"),
        (client.post("/customers", json={"name": "Sweep Customer"}).json(), "customer_id"),
        (client.post("/dashboard/api/notes",
                     json={"category": "note", "title": "sweep"}).json(), "id"),
    ]
    for payload, id_key in checks:
        assert payload.get("success") is True
        assert payload.get(id_key), f"write returned 200 without {id_key}: {payload}"


@pytest.mark.db
def test_get_requests_are_not_enveloped(client, _db_connection):
    """The envelope applies to write methods only — reads stay untouched."""
    resp = client.get("/sales/orders")
    assert resp.status_code == 200
    assert "success" not in resp.json()
