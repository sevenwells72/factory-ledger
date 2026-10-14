"""O-4: every order-close/status shortcut obeys the recorded-shipment gate.

Real HTTP routes and a disposable PostgreSQL database cover shared/personal
keys and owner/office sessions, the immutable America/New_York creation date,
notes, previews, ticket commits, voids, and the two ship-all paths that can change status without
posting stock. Refusals must leave orders, lines, reservations and audit intact.
"""
from uuid import uuid4

import pytest

import main
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import commit, headers

pytestmark = pytest.mark.db
CUTOFF = "2026-11-20"
CREATED = {"before": "2026-11-20 04:59:59+00", "after": "2026-11-20 05:00:00+00"}
ROLES = ("owner", "office", "floor", "master", "dashboard", "owner_session", "office_session")
CLOSE_PATHS = ("direct-preview", "direct-commit", "ticket")
STATUS_PATHS = ("direct", "ticket")
NOTE = "Michael confirmed the historical truck log"


@pytest.fixture(autouse=True)
def cutoff(monkeypatch):
    monkeypatch.setenv("ORDER_LEGACY_CLOSE_CUTOFF", CUTOFF)
    monkeypatch.setenv("KEY_SESSIONS_ENABLED", "1")


def _role(role):
    return role.removesuffix("_session")


def _key(role, actors, client):
    if role.endswith("_session"):
        response = client.post("/auth/session", json={"actor_key": actors[_role(role)]["key"]})
        assert response.status_code == 200
        assert response.json()["actor"]["id"] == actors[_role(role)]["id"]
        return response.json()["session_token"]
    if role == "master":
        return main.API_KEY
    if role == "dashboard":
        return main.DASHBOARD_API_KEY
    return actors[role]["key"]


def _seed(cur, *, created="before", recorded=False, status="confirmed", service=False,
          fake_fulfilled=False, link_style="unified", voided=False):
    token = uuid4().hex[:12]
    cur.execute("INSERT INTO customers(name,active) VALUES(%s,true) RETURNING id", ("O4 " + token,))
    customer = cur.fetchone()["id"]
    cur.execute("INSERT INTO products(name,type,is_service,active) VALUES(%s,'finished',%s,true) RETURNING id",
                ("O4 product " + token, service))
    product = cur.fetchone()["id"]
    cur.execute("INSERT INTO sales_orders(order_number,customer_id,status,order_date) VALUES(%s,%s,%s,%s) RETURNING id",
                ("O4-" + token, customer, status, "2020-01-01" if created == "after" else "2030-01-01"))
    oid = cur.fetchone()["id"]
    # The production trigger makes created_at immutable. Only the rolled-back
    # disposable fixture can represent an existing historical creation date.
    cur.execute("ALTER TABLE sales_orders DISABLE TRIGGER trg_sales_orders_created_at")
    cur.execute("UPDATE sales_orders SET created_at=%s WHERE id=%s", (CREATED[created], oid))
    cur.execute("ALTER TABLE sales_orders ENABLE TRIGGER trg_sales_orders_created_at")
    qty = 0 if service else 10
    cur.execute("""INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb,
                   quantity_shipped_lb,line_status,ordered_quantity,ordered_unit)
                   VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (oid, product, qty, qty if recorded or fake_fulfilled else 0,
                 "fulfilled" if recorded or fake_fulfilled else "pending", 1 if service else None,
                 "each" if service else None))
    line = cur.fetchone()["id"]
    cur.execute("""INSERT INTO sales_order_allocations(sales_order_id,sales_order_line_id,
                   product_id,quantity_lb,source) VALUES(%s,%s,%s,1,'manual') RETURNING id""",
                (oid, line, product))
    allocation = cur.fetchone()["id"]
    txn = None
    if recorded:
        if service:
            cur.execute("INSERT INTO shipments(sales_order_id,customer_id,shipped_at) VALUES(%s,%s,NOW())", (oid, customer))
        else:
            cur.execute("INSERT INTO transactions(type,timestamp) VALUES('ship',NOW()) RETURNING id")
            txn = cur.fetchone()["id"]
            cur.execute("INSERT INTO lots(product_id,lot_code) VALUES(%s,%s) RETURNING id", (product, "O4-LOT-" + token.upper()))
            lot = cur.fetchone()["id"]
            cur.execute("INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES(%s,%s,%s,-10)", (txn, product, lot))
            cur.execute("INSERT INTO sales_order_shipments(sales_order_line_id,transaction_id,quantity_lb) VALUES(%s,%s,10)", (line, txn))
            if link_style == "unified":
                cur.execute("INSERT INTO shipments(sales_order_id,customer_id,shipped_at,transaction_id) VALUES(%s,%s,NOW(),%s) RETURNING id", (oid, customer, txn))
                sid = cur.fetchone()["id"]
                cur.execute("INSERT INTO shipment_lines(shipment_id,transaction_id,sales_order_line_id,product_id,quantity_lb) VALUES(%s,%s,%s,%s,10)", (sid, txn, line, product))
            if voided:
                cur.execute("""INSERT INTO ledger_corrections(target_table,target_id,event_type,previous_values,replacement_values,reason,operator_id)
                               VALUES('transactions',%s,'void','{}','{}','O4 synthetic void','test')""", (txn,))
    return {"id": oid, "customer": customer, "product": product, "line": line,
            "allocation": allocation, "transaction": txn}


def _snapshot(cur, oid):
    result = {}
    for name, sql in (
        ("order", "SELECT * FROM sales_orders WHERE id=%s"),
        ("lines", "SELECT * FROM sales_order_lines WHERE sales_order_id=%s ORDER BY id"),
        ("allocations", "SELECT * FROM sales_order_allocations WHERE sales_order_id=%s ORDER BY id"),
        ("shipments", "SELECT * FROM shipments WHERE sales_order_id=%s ORDER BY id"),
    ):
        cur.execute(sql, (oid,))
        result[name] = [dict(row) for row in cur.fetchall()]
    # Ticket rows and actor last-used telemetry are intentional prepare/auth
    # writes; business audit, inventory and trace postings must not change.
    for table in ("actor_write_audit", "transactions", "transaction_lines", "trace_events", "shipment_lines"):
        cur.execute(f"SELECT count(*) AS n FROM {table}")
        result[table] = cur.fetchone()["n"]
    return result


def _code(response):
    detail = response.json().get("detail")
    return detail.get("error_code") if isinstance(detail, dict) else None


def _assert_refused(response, status, code=None):
    assert response.status_code == status, response.text
    if code:
        assert _code(response) == code, response.text


def _ticket(client, cur, oid, path, body, key, expected_status=200, code=None):
    before = _snapshot(cur, oid)
    prepared = client.post(f"/sales/orders/{oid}/{path}/prepare", json=body, headers=headers(key))
    if expected_status in (403, 422):
        _assert_refused(prepared, expected_status, code)
        assert _snapshot(cur, oid) == before
        return prepared
    assert prepared.status_code == 200, prepared.text
    assert _snapshot(cur, oid) == before, "prepare must leave the business state intact"
    draft = prepared.json()
    if expected_status != 200:
        assert draft["can_commit"] is False, prepared.text
        if code:
            assert code in {blocker["code"] for blocker in draft["blockers"]}, prepared.text
        response = commit(client, draft, key)
        _assert_refused(response, 409, "TICKET_STALE")
        assert _snapshot(cur, oid) == before
        return response
    assert draft["can_commit"] is True, prepared.text
    response = commit(client, draft, key)
    assert response.status_code == 200, response.text
    return response


def _close_expected(path, role, created, recorded, note, reason):
    if _role(role) == "floor" or (path == "ticket" and role == "dashboard"):
        return 403, None
    if reason == "shipped_not_recorded":
        if _role(role) != "owner":
            return 403, "ROLE_NOT_ALLOWED"
        if not note:
            return 422, "NOTE_REQUIRED"
        if not recorded and created == "after":
            return 403, "LEGACY_CLOSE_CUTOFF"
    elif not recorded:
        return 409, "SHIPMENT_REQUIRED"
    return 200, None


@pytest.mark.parametrize("path", CLOSE_PATHS)
@pytest.mark.parametrize("reason", ("shipped_recorded", "shipped_not_recorded", "short_closed"))
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("created", CREATED)
@pytest.mark.parametrize("recorded", (False, True))
@pytest.mark.parametrize("note", (False, True))
def test_close_guard_matrix(client, db_cursor, actors, path, reason, role, created, recorded, note):
    order = _seed(db_cursor, created=created, recorded=recorded)
    oid, key = order["id"], _key(role, actors, client)
    body = {"reason": reason}
    if note:
        body["note"] = NOTE
    expected, code = _close_expected(path, role, created, recorded, note, reason)
    before = _snapshot(db_cursor, oid)
    if path == "ticket":
        response = _ticket(client, db_cursor, oid, "close", body, key, expected, code)
    else:
        body["mode"] = "preview" if path == "direct-preview" else "commit"
        response = client.post(f"/sales/orders/{oid}/close", json=body, headers=headers(key))
        if expected != 200:
            _assert_refused(response, expected, code)
        else:
            assert response.status_code == 200, response.text
    if expected != 200 or path == "direct-preview":
        assert _snapshot(db_cursor, oid) == before
    else:
        after = _snapshot(db_cursor, oid)
        assert after["order"][0]["state"] == "closed"
        assert after["order"][0]["state_reason"] == reason
        assert after["order"][0]["status"] == "shipped"
        assert after["allocations"][0]["status"] == "released"
        assert after["transactions"] == before["transactions"]
        assert after["transaction_lines"] == before["transaction_lines"]
        assert after["trace_events"] == before["trace_events"]


@pytest.mark.parametrize("path", STATUS_PATHS)
@pytest.mark.parametrize("status", ("invoiced", "shipped", "partial_ship"))
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("created", CREATED)
@pytest.mark.parametrize("recorded", (False, True))
@pytest.mark.parametrize("note", (False, True))
def test_status_guard_matrix(client, db_cursor, actors, path, status, role, created, recorded, note):
    order = _seed(db_cursor, created=created, recorded=recorded, status="shipped")
    oid, key = order["id"], _key(role, actors, client)
    body = {"status": status}
    if note:
        body["note"] = NOTE  # An ignored/unsupported note cannot grant an exemption.
    before = _snapshot(db_cursor, oid)
    if path == "ticket" and role == "dashboard":
        expected, code = 403, None
    elif path == "ticket" and note:
        expected, code = 422, None
    elif _role(role) == "floor":
        expected, code = 403, None
    elif status != "invoiced":
        expected, code = 400, None  # These values can only be set by real shipping.
    elif not recorded:
        expected, code = 409, "SHIPMENT_REQUIRED"
    else:
        expected, code = 200, None
    if path == "ticket":
        response = _ticket(client, db_cursor, oid, "status", body, key, expected, code)
    else:
        response = client.patch(f"/sales/orders/{oid}/status", json=body, headers=headers(key))
        if expected != 200:
            _assert_refused(response, expected, code)
        else:
            assert response.status_code == 200, response.text
    if expected != 200:
        assert _snapshot(db_cursor, oid) == before
    else:
        after = _snapshot(db_cursor, oid)
        assert after["order"][0]["state"] == "closed"
        assert after["order"][0]["status"] == "invoiced"
        assert after["order"][0]["state_reason"] == "shipped_recorded"
        assert after["allocations"][0]["status"] == "released"
        assert after["transactions"] == before["transactions"]


@pytest.mark.parametrize("path", ("direct-preview", "direct-commit", "ticket", "direct-status", "ticket-status"))
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("created", CREATED)
@pytest.mark.parametrize("recorded", (False, True))
@pytest.mark.parametrize("note", (False, True))
def test_cancel_paths_still_obey_the_existing_policy(client, db_cursor, actors, path, role, created, recorded, note):
    order = _seed(db_cursor, created=created, recorded=recorded)
    oid, key = order["id"], _key(role, actors, client)
    is_status = path.endswith("status")
    body = {"status": "cancelled"} if is_status else {"reason": "customer_cancelled"}
    if note:
        body["note"] = NOTE
    before = _snapshot(db_cursor, oid)
    if path.startswith("ticket") and role == "dashboard":
        expected, code = 403, None
    elif path == "ticket-status" and note:
        expected, code = 422, None
    elif _role(role) == "floor":
        expected, code = 403, None
    elif recorded:
        expected, code = 409, "ORDER_ALREADY_SHIPPED"
    else:
        expected, code = 200, None
    if path.startswith("ticket"):
        _ticket(client, db_cursor, oid, "status" if is_status else "cancel", body, key, expected, code)
    else:
        if is_status:
            response = client.patch(f"/sales/orders/{oid}/status", json=body, headers=headers(key))
        else:
            body["mode"] = "preview" if path == "direct-preview" else "commit"
            response = client.post(f"/sales/orders/{oid}/cancel", json=body, headers=headers(key))
        if expected != 200:
            _assert_refused(response, expected, code)
        else:
            assert response.status_code == 200, response.text
    if expected != 200 or path == "direct-preview":
        assert _snapshot(db_cursor, oid) == before
    else:
        after = _snapshot(db_cursor, oid)
        assert after["order"][0]["state"] == "cancelled"
        assert after["allocations"][0]["status"] == "released"
        assert after["transactions"] == before["transactions"]


@pytest.mark.parametrize("path", ("legacy", "commit"))
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("created", CREATED)
@pytest.mark.parametrize("recorded", (False, True))
@pytest.mark.parametrize("note", (False, True))
def test_ship_all_no_remaining_lines_cannot_forge_shipped(client, db_cursor, actors, path, role, created, recorded, note):
    order = _seed(db_cursor, created=created, recorded=recorded, status="partial_ship", fake_fulfilled=True)
    oid, key = order["id"], _key(role, actors, client)
    body = {"ship_all": True}
    if note:
        body["note"] = NOTE
    if path == "legacy":
        body["mode"] = "commit"
    before = _snapshot(db_cursor, oid)
    response = client.post(f"/sales/orders/{oid}/ship" + ("/commit" if path == "commit" else ""), json=body, headers=headers(key))
    if path == "legacy" and role == "dashboard":
        _assert_refused(response, 403)
    elif path == "commit" and note:
        _assert_refused(response, 422)
    elif not recorded:
        _assert_refused(response, 409, "SHIPMENT_REQUIRED")
    else:
        assert response.status_code == 200, response.text
        assert response.json()["order_status"] == "shipped"
        assert response.json()["shipment_id"] is None
    after = _snapshot(db_cursor, oid)
    if response.status_code != 200:
        assert after == before
    else:
        assert after["order"][0]["state"] == "open"
        assert after["order"][0]["status"] == "shipped"
        assert after["transactions"] == before["transactions"]
        assert after["shipments"] == before["shipments"]


@pytest.mark.parametrize("path", ("legacy", "commit"))
@pytest.mark.parametrize("recorded", (False, True))
def test_fulfilled_counter_plus_service_line_is_not_a_recorded_physical_ship(client, db_cursor, actors, path, recorded):
    order = _seed(db_cursor, created="after", recorded=recorded, fake_fulfilled=True, status="partial_ship")
    db_cursor.execute("INSERT INTO products(name,type,is_service) VALUES(%s,'packaging',true) RETURNING id", ("O4 service " + uuid4().hex,))
    service = db_cursor.fetchone()["id"]
    db_cursor.execute("INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb,ordered_quantity,ordered_unit) VALUES(%s,%s,0,1,'each')", (order["id"], service))
    body = {"ship_all": True}
    if path == "legacy":
        body["mode"] = "commit"
    before = _snapshot(db_cursor, order["id"])
    response = client.post(f"/sales/orders/{order['id']}/ship" + ("/commit" if path == "commit" else ""), json=body, headers=headers(actors["office"]["key"]))
    if recorded:
        assert response.status_code == 200, response.text
        assert response.json()["order_status"] == "shipped"
        assert response.json()["shipment_id"] is not None
        assert _snapshot(db_cursor, order["id"])["transactions"] == before["transactions"]
    else:
        _assert_refused(response, 409, "SHIPMENT_REQUIRED")
        assert _snapshot(db_cursor, order["id"]) == before


@pytest.mark.parametrize("path", ("legacy", "commit"))
def test_service_only_shipping_records_a_header_and_can_close_after_cutoff(client, db_cursor, actors, path):
    order = _seed(db_cursor, created="after", service=True)
    body = {"ship_all": True}
    if path == "legacy":
        body["mode"] = "commit"
    key = actors["office"]["key"]
    before = _snapshot(db_cursor, order["id"])
    response = client.post(f"/sales/orders/{order['id']}/ship" + ("/commit" if path == "commit" else ""), json=body, headers=headers(key))
    assert response.status_code == 200, response.text
    assert response.json()["order_status"] == "shipped"
    assert response.json()["shipment_id"] is not None
    assert _snapshot(db_cursor, order["id"])["transactions"] == before["transactions"]
    response = client.post(f"/sales/orders/{order['id']}/close", json={"reason": "shipped_recorded", "mode": "commit"}, headers=headers(key))
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("path", ("direct-commit", "ticket"))
@pytest.mark.parametrize("note", ("", "   ", "\t\n"))
def test_legacy_close_rejects_blank_notes(client, db_cursor, actors, path, note):
    order = _seed(db_cursor)
    body = {"reason": "shipped_not_recorded", "note": note}
    key = actors["owner"]["key"]
    if path == "ticket":
        _ticket(client, db_cursor, order["id"], "close", body, key, 422, "NOTE_REQUIRED")
    else:
        body["mode"] = "commit"
        before = _snapshot(db_cursor, order["id"])
        response = client.post(f"/sales/orders/{order['id']}/close", json=body, headers=headers(key))
        _assert_refused(response, 422, "NOTE_REQUIRED")
        assert _snapshot(db_cursor, order["id"]) == before


@pytest.mark.parametrize("path", ("direct-commit", "ticket"))
@pytest.mark.parametrize("reason", ("shipped_recorded", "short_closed"))
@pytest.mark.parametrize("link_style", ("unified", "legacy"))
@pytest.mark.parametrize("voided", (False, True))
def test_effective_recorded_links_count_but_voided_shipments_do_not(client, db_cursor, actors, path, reason, link_style, voided):
    order = _seed(db_cursor, created="after", recorded=True, link_style=link_style, voided=voided)
    key = actors["office"]["key"]
    body = {"reason": reason}
    expected = 409 if voided else 200
    code = "SHIPMENT_REQUIRED" if voided else None
    if path == "ticket":
        _ticket(client, db_cursor, order["id"], "close", body, key, expected, code)
    else:
        body["mode"] = "commit"
        before = _snapshot(db_cursor, order["id"])
        response = client.post(f"/sales/orders/{order['id']}/close", json=body, headers=headers(key))
        if voided:
            _assert_refused(response, expected, code)
            assert _snapshot(db_cursor, order["id"]) == before
        else:
            assert response.status_code == 200, response.text


def test_ticket_commit_rechecks_cutoff_without_mutation(client, db_cursor, actors, monkeypatch):
    order = _seed(db_cursor)
    key = actors["owner"]["key"]
    response = client.post(f"/sales/orders/{order['id']}/close/prepare", json={"reason": "shipped_not_recorded", "note": NOTE}, headers=headers(key))
    assert response.status_code == 200 and response.json()["can_commit"], response.text
    before = _snapshot(db_cursor, order["id"])
    monkeypatch.setenv("ORDER_LEGACY_CLOSE_CUTOFF", "2026-11-19")
    response = commit(client, response.json(), key)
    _assert_refused(response, 403, "LEGACY_CLOSE_CUTOFF")
    assert _snapshot(db_cursor, order["id"]) == before


@pytest.mark.parametrize("path", ("direct-commit", "ticket"))
@pytest.mark.parametrize("bad_cutoff", ("2026-13-01", "20261120"))
def test_invalid_cutoff_fails_closed(client, db_cursor, actors, monkeypatch, path, bad_cutoff):
    order = _seed(db_cursor)
    key = actors["owner"]["key"]
    monkeypatch.setenv("ORDER_LEGACY_CLOSE_CUTOFF", bad_cutoff)
    body = {"reason": "shipped_not_recorded", "note": NOTE}
    before = _snapshot(db_cursor, order["id"])
    if path == "direct-commit":
        body["mode"] = "commit"
    # Runtime corruption of validated configuration may either return a server
    # refusal or raise before routing; it must never authorize a close.
    try:
        response = client.post(f"/sales/orders/{order['id']}/close" + ("/prepare" if path == "ticket" else ""), json=body, headers=headers(key))
    except RuntimeError as exc:
        assert "ORDER_LEGACY_CLOSE_CUTOFF" in str(exc)
    else:
        assert response.status_code >= 500, response.text
    assert _snapshot(db_cursor, order["id"]) == before


def _reopen(client, cur, actors, path, role, created, recorded, note, restore):
    order = _seed(cur, created=created, recorded=recorded)
    oid, key = order["id"], _key(role, actors, client)
    cur.execute("""UPDATE sales_orders SET state='cancelled',state_reason='customer_cancelled',
                   status='cancelled',status_before_exit=%s WHERE id=%s""", (restore, oid))
    before = _snapshot(cur, oid)
    body = {"note": NOTE} if note else {}
    if _role(role) in ("office", "floor") or (path == "ticket" and role == "dashboard"):
        expected, code = 403, None
    elif restore in ("shipped", "invoiced") and not recorded:
        expected, code = 409, "SHIPMENT_REQUIRED"
    else:
        expected, code = 200, None
    if path == "ticket":
        response = _ticket(client, cur, oid, "reopen", body, key, expected, code)
    else:
        body["mode"] = "preview" if path == "direct-preview" else "commit"
        response = client.post(f"/sales/orders/{oid}/reopen", json=body, headers=headers(key))
        if expected != 200:
            _assert_refused(response, expected, code)
        else:
            assert response.status_code == 200, response.text
    after = _snapshot(cur, oid)
    if expected != 200 or path == "direct-preview":
        assert after == before
    else:
        assert after["order"][0]["state"] == "open"
        assert after["order"][0]["status"] == restore
        assert after["allocations"] == before["allocations"]
        assert after["transactions"] == before["transactions"]
        assert after["shipments"] == before["shipments"]
    return response


@pytest.mark.parametrize("path", CLOSE_PATHS)
@pytest.mark.parametrize("restore", ("shipped", "invoiced"))
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("created", CREATED)
@pytest.mark.parametrize("recorded", (False, True))
@pytest.mark.parametrize("note", (False, True))
def test_reopen_cannot_restore_shipped_or_invoiced_without_a_recorded_shipment(client, db_cursor, actors, path, restore, role, created, recorded, note):
    _reopen(client, db_cursor, actors, path, role, created, recorded, note, restore)


@pytest.mark.parametrize("path", CLOSE_PATHS)
@pytest.mark.parametrize("restore", ("confirmed", "ready"))
@pytest.mark.parametrize("role", ROLES)
def test_normal_reopen_remains_available_without_recorded_shipment(client, db_cursor, actors, path, restore, role):
    _reopen(client, db_cursor, actors, path, role, "after", False, False, restore)


@pytest.mark.parametrize("path", STATUS_PATHS)
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("note", (False, True))
def test_recorded_partial_invoice_uses_short_close_reason_without_legacy_exception(client, db_cursor, actors, path, role, note):
    # A legacy stored shipped status may overstate the actual recorded pounds.
    order = _seed(db_cursor, created="after", status="shipped")
    oid, key = order["id"], _key(role, actors, client)
    db_cursor.execute("INSERT INTO lots(product_id,lot_code) VALUES(%s,%s) RETURNING id",
                      (order["product"], "O4-PARTIAL-" + uuid4().hex[:12].upper()))
    lot = db_cursor.fetchone()["id"]
    db_cursor.execute("INSERT INTO transactions(type,timestamp) VALUES('ship',NOW()) RETURNING id")
    txn = db_cursor.fetchone()["id"]
    db_cursor.execute("INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES(%s,%s,%s,-1)",
                      (txn, order["product"], lot))
    db_cursor.execute("INSERT INTO sales_order_shipments(sales_order_line_id,transaction_id,quantity_lb) VALUES(%s,%s,1)",
                      (order["line"], txn))
    db_cursor.execute("INSERT INTO shipments(sales_order_id,customer_id,shipped_at,transaction_id) VALUES(%s,%s,NOW(),%s) RETURNING id",
                      (oid, order["customer"], txn))
    shipment = db_cursor.fetchone()["id"]
    db_cursor.execute("INSERT INTO shipment_lines(shipment_id,transaction_id,sales_order_line_id,product_id,quantity_lb) VALUES(%s,%s,%s,%s,1)",
                      (shipment, txn, order["line"], order["product"]))
    db_cursor.execute("UPDATE sales_order_lines SET quantity_shipped_lb=1,line_status='partial' WHERE id=%s", (order["line"],))
    body = {"status": "invoiced"}
    if note:
        body["note"] = NOTE
    if path == "ticket" and role == "dashboard":
        expected = 403
    elif path == "ticket" and note:
        expected = 422  # Status tickets never accept a legacy-close note.
    elif _role(role) == "floor":
        expected = 403
    else:
        expected = 200
    before = _snapshot(db_cursor, oid)
    if path == "ticket":
        _ticket(client, db_cursor, oid, "status", body, key, expected)
    else:
        response = client.patch(f"/sales/orders/{oid}/status", json=body, headers=headers(key))
        if expected != 200:
            _assert_refused(response, expected)
        else:
            assert response.status_code == 200, response.text
    after = _snapshot(db_cursor, oid)
    if expected != 200:
        assert after == before
    else:
        assert after["order"][0]["state"] == "closed"
        assert after["order"][0]["status"] == "invoiced"
        assert after["order"][0]["state_reason"] == "short_closed"
        assert after["lines"] == before["lines"]
        assert after["allocations"][0]["status"] == "released"
        assert after["transactions"] == before["transactions"]
        assert after["transaction_lines"] == before["transaction_lines"]
        assert after["shipments"] == before["shipments"]


def _legacy_closed(cur, *, created="before", recorded=False, restore="ready"):
    order = _seed(cur, created=created, recorded=recorded)
    cur.execute("""UPDATE sales_orders SET state='closed',state_reason='shipped_not_recorded',
                   state_note='Original historical close',status='shipped',status_before_exit=%s
                   WHERE id=%s""", (restore, order["id"]))
    return order


@pytest.mark.parametrize("path", CLOSE_PATHS)
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("cutover", ("before", "after", "unset"))
@pytest.mark.parametrize("recorded", (False, True))
@pytest.mark.parametrize("note", (False, True))
@pytest.mark.parametrize("restore", ("ready", "shipped", "invoiced"))
def test_legacy_reopen_guard_matrix(client, db_cursor, actors, monkeypatch, path, role,
                                   cutover, recorded, note, restore):
    # immutable created_at straddles New York midnight; order_date deliberately
    # lies in the opposite direction, and a later recorded shipment cannot
    # grant anyone a cutover or owner exemption.
    order = _legacy_closed(db_cursor, created="after" if cutover == "unset" else cutover,
                           recorded=recorded, restore=restore)
    if cutover == "unset":
        monkeypatch.delenv("ORDER_LEGACY_CLOSE_CUTOFF")
    oid, key = order["id"], _key(role, actors, client)
    before = _snapshot(db_cursor, oid)
    body = {"note": NOTE} if note else {}
    if _role(role) != "owner":
        expected, code = 403, "ROLE_NOT_ALLOWED" if role != "dashboard" else None
    elif not note:
        expected, code = 422, "NOTE_REQUIRED"
    elif cutover == "after":
        expected, code = 403, "LEGACY_CLOSE_CUTOFF"
    else:
        expected, code = 200, None
    if path == "ticket":
        response = _ticket(client, db_cursor, oid, "reopen", body, key, expected, code)
    else:
        body["mode"] = "preview" if path == "direct-preview" else "commit"
        response = client.post(f"/sales/orders/{oid}/reopen", json=body, headers=headers(key))
        if expected != 200:
            _assert_refused(response, expected, code)
        else:
            assert response.status_code == 200, response.text
    after = _snapshot(db_cursor, oid)
    if expected != 200 or path == "direct-preview":
        assert after == before
        if expected == 200:
            assert response.json()["resulting_status"] == restore
    else:
        assert after["order"][0]["state"] == "open"
        assert after["order"][0]["status"] == restore
        assert after["order"][0]["status_before_exit"] is None
        assert after["order"][0]["state_note"] == NOTE
        assert after["order"][0]["state_reason"] is None
        assert after["order"][0]["state_changed_by"] == actors["owner"]["name"]
        for field in ("allocations", "lines", "shipments", "transactions", "transaction_lines", "trace_events"):
            assert after[field] == before[field]


@pytest.mark.parametrize("path", CLOSE_PATHS)
@pytest.mark.parametrize("role", ("owner", "owner_session"))
@pytest.mark.parametrize("note", ("", "   ", "\t\n"))
def test_legacy_reopen_requires_its_own_nonblank_note(client, db_cursor, actors, path, role, note):
    order = _legacy_closed(db_cursor)
    key = _key(role, actors, client)
    before = _snapshot(db_cursor, order["id"])
    body = {"note": note}
    if path == "ticket":
        _ticket(client, db_cursor, order["id"], "reopen", body, key, 422, "NOTE_REQUIRED")
    else:
        body["mode"] = "preview" if path == "direct-preview" else "commit"
        response = client.post(f"/sales/orders/{order['id']}/reopen", json=body, headers=headers(key))
        _assert_refused(response, 422, "NOTE_REQUIRED")
    assert _snapshot(db_cursor, order["id"]) == before


@pytest.mark.parametrize("role", ("owner", "owner_session"))
@pytest.mark.parametrize("recorded", (False, True))
@pytest.mark.parametrize("change", ("cutoff", "exit_reason"))
def test_legacy_reopen_ticket_commit_rechecks_policy_without_mutation(client, db_cursor, actors,
                                                                    monkeypatch, role, recorded, change):
    order = _legacy_closed(db_cursor, recorded=recorded, restore="shipped")
    key = _key(role, actors, client)
    prepared = client.post(f"/sales/orders/{order['id']}/reopen/prepare", json={"note": NOTE}, headers=headers(key))
    assert prepared.status_code == 200 and prepared.json()["can_commit"], prepared.text
    if change == "cutoff":
        monkeypatch.setenv("ORDER_LEGACY_CLOSE_CUTOFF", "2026-11-19")
    else:
        # A draft prepared for the legacy exception cannot keep its exemption
        # after another writer changes the exit reason.
        db_cursor.execute("UPDATE sales_orders SET state_reason='shipped_recorded' WHERE id=%s", (order["id"],))
        if recorded:
            db_cursor.execute("""INSERT INTO ledger_corrections(target_table,target_id,event_type,previous_values,
                               replacement_values,reason,operator_id)
                               VALUES('transactions',%s,'void','{}','{}','O4 synthetic void','test')""", (order["transaction"],))
    before = _snapshot(db_cursor, order["id"])
    response = commit(client, prepared.json(), key)
    _assert_refused(response, 403 if change == "cutoff" else 409,
                    "LEGACY_CLOSE_CUTOFF" if change == "cutoff" else "TICKET_STALE")
    if change != "cutoff":
        assert "SHIPMENT_REQUIRED" in {b["code"] for b in response.json()["detail"]["blockers"]}
    assert _snapshot(db_cursor, order["id"]) == before


@pytest.mark.parametrize("role", ("master", "owner", "owner_session"))
@pytest.mark.parametrize("recorded", (False, True))
@pytest.mark.parametrize("change", ("legacy_reason", "missing_note"))
def test_ordinary_reopen_ticket_rechecks_new_legacy_reason(client, db_cursor, actors,
                                                          monkeypatch, role, recorded, change):
    order = _seed(db_cursor, recorded=recorded)
    db_cursor.execute("""UPDATE sales_orders SET state='closed',state_reason='shipped_recorded',
                       status='shipped',status_before_exit='ready' WHERE id=%s""", (order["id"],))
    key = _key(role, actors, client)
    body = {} if change == "missing_note" else {"note": NOTE}
    prepared = client.post(f"/sales/orders/{order['id']}/reopen/prepare", json=body, headers=headers(key))
    assert prepared.status_code == 200 and prepared.json()["can_commit"], prepared.text
    db_cursor.execute("UPDATE sales_orders SET state_reason='shipped_not_recorded' WHERE id=%s", (order["id"],))
    monkeypatch.setenv("ORDER_LEGACY_CLOSE_CUTOFF", "2026-11-19")
    before = _snapshot(db_cursor, order["id"])
    response = commit(client, prepared.json(), key)
    status, code = ((403, "ROLE_NOT_ALLOWED") if role == "master"
                    else (422, "NOTE_REQUIRED") if change == "missing_note"
                    else (403, "LEGACY_CLOSE_CUTOFF"))
    _assert_refused(response, status, code)
    assert _snapshot(db_cursor, order["id"]) == before


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("cutover", ("before", "after", "unset"))
@pytest.mark.parametrize("recorded", (False, True))
def test_legacy_reopen_office_options_match_policy(client, db_cursor, actors, monkeypatch,
                                                  role, cutover, recorded):
    order = _legacy_closed(db_cursor, created="after" if cutover == "unset" else cutover, recorded=recorded)
    if cutover == "unset":
        monkeypatch.delenv("ORDER_LEGACY_CLOSE_CUTOFF")
    key = _key(role, actors, client)
    before = _snapshot(db_cursor, order["id"])
    response = client.get(f"/sales/orders/{order['id']}/office-options", headers=headers(key))
    if role == "dashboard":
        _assert_refused(response, 403)  # Shared dashboard keys cannot read ticket options.
        assert _snapshot(db_cursor, order["id"]) == before
        return
    assert response.status_code == 200, response.text
    options = response.json()
    assert options["reopen_note_required"] is True
    assert options["permissions"]["reopen_order"] is (_role(role) == "owner" and cutover != "after")
    assert _snapshot(db_cursor, order["id"]) == before


@pytest.mark.parametrize("role", ("owner", "owner_session"))
@pytest.mark.parametrize("path", ("direct-preview", "direct-commit", "ticket"))
@pytest.mark.parametrize("bad_cutoff", ("2026-13-01", "20261120"))
def test_legacy_reopen_invalid_cutoff_fails_closed(client, db_cursor, actors, monkeypatch,
                                                  role, path, bad_cutoff):
    order = _legacy_closed(db_cursor)
    key = _key(role, actors, client)
    monkeypatch.setenv("ORDER_LEGACY_CLOSE_CUTOFF", bad_cutoff)
    body = {"note": NOTE}
    if path != "ticket":
        body["mode"] = "preview" if path == "direct-preview" else "commit"
    before = _snapshot(db_cursor, order["id"])
    try:
        response = client.post(f"/sales/orders/{order['id']}/reopen" + ("/prepare" if path == "ticket" else ""),
                               json=body, headers=headers(key))
    except RuntimeError as exc:
        assert "ORDER_LEGACY_CLOSE_CUTOFF" in str(exc)
    else:
        assert response.status_code >= 500, response.text
    assert _snapshot(db_cursor, order["id"]) == before
