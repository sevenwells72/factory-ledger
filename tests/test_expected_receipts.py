"""FR-2: Expected / incoming inventory records (migration 041).

Covers:
  * supplier resolution failure → 422 with up to 5 candidates, no auto-create
  * case/whitespace-insensitive supplier match on create
  * receive auto-match picks the FIFO (oldest expected_date) open record
  * partial receipt → remaining computed from the ledger SUM
  * full receipt → auto-close
  * over-receipt → auto-close, remaining floored at 0
  * unknown supplier / no open record → receipt posts normally, unlinked
  * voided linked receipt no longer counts (posted-only SUM)
  * PATCH edits / status transitions and their guards
  * expected_receipts never affect any inventory balance query
"""

from contextlib import contextmanager
from datetime import date, timedelta

import pytest

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    pytest.skip("fastapi/httpx not installed", allow_module_level=True)

try:
    import main
except Exception as e:  # pragma: no cover
    pytest.skip(f"cannot import main ({e})", allow_module_level=True)


class _ConnProxy:
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
    @contextmanager
    def _fake_get_conn():
        proxy = _ConnProxy(_db_connection, "fr2_inner")
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
    _db_connection.rollback()


@pytest.fixture
def cur(_db_connection):
    """Cursor on the same connection the client uses (so seeded rows are visible)."""
    from psycopg2.extras import RealDictCursor
    c = _db_connection.cursor(cursor_factory=RealDictCursor)
    yield c
    c.close()


# ─────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────

def _seed_product(cur, name):
    cur.execute(
        "INSERT INTO products (name, type, odoo_code, active) VALUES (%s, 'ingredient', %s, true) RETURNING id",
        (name, f"SKU-{name[:20]}"),
    )
    return cur.fetchone()["id"]


def _seed_supplier(cur, name, active=True):
    cur.execute("INSERT INTO suppliers (name, active) VALUES (%s, %s) RETURNING id", (name, active))
    return cur.fetchone()["id"]


def _create_er(client, product_id, supplier_name, qty, expected_date=None, **extra):
    body = {"product_id": product_id, "supplier_name": supplier_name, "expected_qty": qty}
    if expected_date:
        body["expected_date"] = expected_date
    body.update(extra)
    r = client.post("/expected-receipts", json=body)
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["success"] is True
    return data["expected_receipt"]


def _receive(client, product_name, shipper_name, cases, case_size_lb, lot_code=None):
    body = {
        "mode": "commit",
        "product_name": product_name,
        "cases": cases,
        "case_size_lb": case_size_lb,
        "shipper_name": shipper_name,
        "bol_reference": "BOL-FR2",
    }
    if lot_code:
        body["lot_code"] = lot_code
    r = client.post("/receive", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _get_er(client, er_id):
    r = client.get(f"/expected-receipts/{er_id}")
    assert r.status_code == 200, r.text
    return r.json()


# ─────────────────────────────────────────────────────────────────
# supplier resolution
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_unknown_supplier_returns_422_with_candidates_and_never_autocreates(client, cur):
    pid = _seed_product(cur, "FR2 Oats Candidates")
    for n in ["Dutch Valley", "Dutch Gold", "Dutch Valley Foods", "Euro Good", "Quali Pack",
              "Star Snacks", "Tri State"]:
        _seed_supplier(cur, n)
    cur.execute("SELECT COUNT(*) AS n FROM suppliers")
    before = cur.fetchone()["n"]

    r = client.post("/expected-receipts", json={
        "product_id": pid, "supplier_name": "Dutch Valey", "expected_qty": 100,
    })
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["success"] is False
    assert body["error_detail"]["code"] == "SUPPLIER_NOT_FOUND"
    detail = body["detail"]
    assert detail["error_code"] == "SUPPLIER_NOT_FOUND"
    names = [c["name"] for c in detail["candidates"]]
    assert 1 <= len(names) <= 5
    assert names[0] == "Dutch Valley", names
    assert "Dutch Gold" in names or "Dutch Valley Foods" in names
    assert detail["suggestions"] == names

    cur.execute("SELECT COUNT(*) AS n FROM suppliers")
    assert cur.fetchone()["n"] == before, "422 path must never auto-create a supplier"
    cur.execute("SELECT COUNT(*) AS n FROM expected_receipts WHERE product_id = %s", (pid,))
    assert cur.fetchone()["n"] == 0


@pytest.mark.db
def test_supplier_match_is_case_and_whitespace_insensitive(client, cur):
    pid = _seed_product(cur, "FR2 Oats CaseMatch")
    sid = _seed_supplier(cur, "Jack's Eggs")
    er = _create_er(client, pid, "  jack’s   EGGS ", 50)
    assert er["supplier_id"] == sid
    assert er["supplier_name"] == "Jack's Eggs"
    assert er["status"] == "open"
    assert er["remaining"] == 50
    assert er["received_qty"] == 0


@pytest.mark.db
def test_inactive_supplier_rejected(client, cur):
    pid = _seed_product(cur, "FR2 Oats Inactive")
    _seed_supplier(cur, "Old Vendor Co", active=False)
    r = client.post("/expected-receipts", json={"product_id": pid, "supplier_name": "old vendor co", "expected_qty": 10})
    assert r.status_code == 422
    assert r.json()["detail"]["error_code"] == "SUPPLIER_INACTIVE"


# ─────────────────────────────────────────────────────────────────
# receive auto-match
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_auto_match_selects_fifo_oldest_expected_date(client, cur):
    pname = "FR2 Almonds FIFO"
    pid = _seed_product(cur, pname)
    _seed_supplier(cur, "FIFO Nut Co")
    # Created LATER but due EARLIER must win; NULL-dated record must lose to any dated one.
    er_late = _create_er(client, pid, "FIFO Nut Co", 500, "2026-09-15")
    er_null = _create_er(client, pid, "FIFO Nut Co", 500)
    er_early = _create_er(client, pid, "FIFO Nut Co", 500, "2026-09-01")

    resp = _receive(client, pname, "fifo nut co", cases=4, case_size_lb=25)  # 100 lb
    assert resp["expected_receipt"]["id"] == er_early["id"]
    assert resp["expected_receipt"]["auto_closed"] is False
    assert resp["expected_receipt"]["remaining"] == pytest.approx(400)

    cur.execute("SELECT expected_receipt_id FROM transactions WHERE id = %s", (resp["transaction_id"],))
    assert cur.fetchone()["expected_receipt_id"] == er_early["id"]

    assert _get_er(client, er_late["id"])["received_qty"] == 0
    assert _get_er(client, er_null["id"])["received_qty"] == 0

    # Once the early one closes, the next receipt rolls to the next-oldest dated record.
    _receive(client, pname, "FIFO Nut Co", cases=16, case_size_lb=25)  # 400 → closes early
    assert _get_er(client, er_early["id"])["status"] == "closed"
    resp3 = _receive(client, pname, "FIFO Nut Co", cases=1, case_size_lb=25)
    assert resp3["expected_receipt"]["id"] == er_late["id"]


@pytest.mark.db
def test_partial_receipt_remaining_is_computed_from_ledger(client, cur):
    pname = "FR2 Honey Partial"
    pid = _seed_product(cur, pname)
    _seed_supplier(cur, "Partial Honey Farm")
    er = _create_er(client, pid, "Partial Honey Farm", 500, "2026-09-10", reference_number="PO-77")

    resp = _receive(client, pname, "Partial Honey Farm", cases=8, case_size_lb=25)  # 200 lb
    assert resp["expected_receipt"]["remaining"] == pytest.approx(300)
    assert "still expected" in resp["message"]

    detail = _get_er(client, er["id"])
    assert detail["status"] == "open"
    assert detail["received_qty"] == pytest.approx(200)
    assert detail["remaining"] == pytest.approx(300)
    assert detail["over_receipt_qty"] == 0
    assert detail["receipt_count"] == 1
    assert detail["linked_receipts"][0]["transaction_id"] == resp["transaction_id"]
    assert detail["linked_receipts"][0]["quantity_lb"] == pytest.approx(200)

    # No stored balance column anywhere: remaining/received are not on the table.
    cur.execute("""SELECT column_name FROM information_schema.columns
                   WHERE table_name = 'expected_receipts'""")
    cols = {r["column_name"] for r in cur.fetchall()}
    assert not any(c for c in cols if "remain" in c or "received" in c), cols

    # A second partial receipt accumulates via SUM.
    _receive(client, pname, "Partial Honey Farm", cases=2, case_size_lb=25)  # +50
    detail = _get_er(client, er["id"])
    assert detail["received_qty"] == pytest.approx(250)
    assert detail["remaining"] == pytest.approx(250)
    assert detail["receipt_count"] == 2
    assert detail["status"] == "open"


@pytest.mark.db
def test_full_receipt_auto_closes(client, cur):
    pname = "FR2 Cocoa Full"
    pid = _seed_product(cur, pname)
    _seed_supplier(cur, "Full Cocoa Ltd")
    er = _create_er(client, pid, "Full Cocoa Ltd", 200, "2026-09-10")

    resp = _receive(client, pname, "Full Cocoa Ltd", cases=8, case_size_lb=25)  # exactly 200
    assert resp["expected_receipt"]["auto_closed"] is True
    assert resp["expected_receipt"]["status"] == "closed"
    assert "now closed" in resp["message"]

    detail = _get_er(client, er["id"])
    assert detail["status"] == "closed"
    assert detail["remaining"] == 0
    assert detail["received_qty"] == pytest.approx(200)

    # Closed record must not attract further receipts.
    resp2 = _receive(client, pname, "Full Cocoa Ltd", cases=1, case_size_lb=25)
    assert resp2["expected_receipt"] is None
    cur.execute("SELECT expected_receipt_id FROM transactions WHERE id = %s", (resp2["transaction_id"],))
    assert cur.fetchone()["expected_receipt_id"] is None


@pytest.mark.db
def test_over_receipt_closes_and_remaining_floors_at_zero(client, cur):
    pname = "FR2 Sugar Over"
    pid = _seed_product(cur, pname)
    _seed_supplier(cur, "Over Sugar Inc")
    er = _create_er(client, pid, "Over Sugar Inc", 100)

    resp = _receive(client, pname, "Over Sugar Inc", cases=6, case_size_lb=25)  # 150 lb
    assert resp["expected_receipt"]["auto_closed"] is True
    assert resp["expected_receipt"]["remaining"] == 0
    assert resp["expected_receipt"]["over_receipt_qty"] == pytest.approx(50)

    detail = _get_er(client, er["id"])
    assert detail["status"] == "closed"
    assert detail["received_qty"] == pytest.approx(150)
    assert detail["remaining"] == 0, "remaining must floor at 0 in API responses"
    assert detail["over_receipt_qty"] == pytest.approx(50)

    listed = client.get("/expected-receipts", params={"status": "closed", "product_id": pid}).json()
    row = next(x for x in listed["expected_receipts"] if x["id"] == er["id"])
    assert row["remaining"] == 0


@pytest.mark.db
def test_receipt_with_unknown_supplier_or_no_open_record_posts_unlinked(client, cur):
    pname = "FR2 Salt Unlinked"
    pid = _seed_product(cur, pname)
    _seed_supplier(cur, "Known Salt Co")
    _create_er(client, pid, "Known Salt Co", 100)

    # Unknown supplier → normal receipt, no error, no link.
    resp = _receive(client, pname, "Totally New Vendor", cases=1, case_size_lb=25)
    assert resp["success"] is True
    assert resp["expected_receipt"] is None
    cur.execute("SELECT expected_receipt_id FROM transactions WHERE id = %s", (resp["transaction_id"],))
    assert cur.fetchone()["expected_receipt_id"] is None
    # ...and it never auto-created the supplier.
    cur.execute("SELECT COUNT(*) AS n FROM suppliers WHERE supplier_name_norm(name) = supplier_name_norm(%s)", ("Totally New Vendor",))
    assert cur.fetchone()["n"] == 0

    # Known supplier but a different product → no match, no link.
    other = _seed_product(cur, "FR2 Pepper Unlinked")
    resp2 = _receive(client, "FR2 Pepper Unlinked", "Known Salt Co", cases=1, case_size_lb=25)
    assert resp2["expected_receipt"] is None


@pytest.mark.db
def test_voided_linked_receipt_no_longer_counts(client, cur):
    pname = "FR2 Flour Void"
    pid = _seed_product(cur, pname)
    _seed_supplier(cur, "Void Mills")
    er = _create_er(client, pid, "Void Mills", 300)
    resp = _receive(client, pname, "Void Mills", cases=4, case_size_lb=25)  # 100
    assert _get_er(client, er["id"])["received_qty"] == pytest.approx(100)

    v = client.post(f"/void/{resp['transaction_id']}", json={"reason": "fr2 test void"})
    assert v.status_code == 200, v.text

    detail = _get_er(client, er["id"])
    assert detail["received_qty"] == 0, "voided receipts must drop out of the posted-only SUM"
    assert detail["remaining"] == pytest.approx(300)
    assert detail["linked_receipts"][0]["counted"] is False


@pytest.mark.db
def test_receive_preview_reports_match_without_linking(client, cur):
    pname = "FR2 Raisins Preview"
    pid = _seed_product(cur, pname)
    _seed_supplier(cur, "Preview Fruit")
    er = _create_er(client, pid, "Preview Fruit", 40, "2026-09-05")
    r = client.post("/receive", json={
        "mode": "preview", "product_name": pname, "cases": 1, "case_size_lb": 10,
        "shipper_name": "preview fruit", "bol_reference": "B",
    })
    assert r.status_code == 200, r.text
    assert r.json()["expected_receipt_match"]["id"] == er["id"]
    assert _get_er(client, er["id"])["receipt_count"] == 0


# ─────────────────────────────────────────────────────────────────
# list / overdue / PATCH
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_list_filters_and_overdue_flag(client, cur):
    pid = _seed_product(cur, "FR2 Overdue Prod")
    _seed_supplier(cur, "Overdue Supply")
    yesterday = (main.get_plant_now().date() - timedelta(days=1)).isoformat()
    tomorrow = (main.get_plant_now().date() + timedelta(days=1)).isoformat()
    late = _create_er(client, pid, "Overdue Supply", 10, yesterday)
    ontime = _create_er(client, pid, "Overdue Supply", 10, tomorrow)
    undated = _create_er(client, pid, "Overdue Supply", 10)

    data = client.get("/expected-receipts", params={"status": "open", "product_id": pid}).json()
    by_id = {x["id"]: x for x in data["expected_receipts"]}
    assert set(by_id) == {late["id"], ontime["id"], undated["id"]}
    assert by_id[late["id"]]["is_overdue"] is True and by_id[late["id"]]["days_overdue"] == 1
    assert by_id[ontime["id"]]["is_overdue"] is False
    assert by_id[undated["id"]]["is_overdue"] is False
    assert data["overdue_count"] == 1
    # FIFO ordering: dated ascending, undated last
    assert [x["id"] for x in data["expected_receipts"]] == [late["id"], ontime["id"], undated["id"]]

    only = client.get("/expected-receipts", params={"overdue_only": "true", "product_id": pid}).json()
    assert [x["id"] for x in only["expected_receipts"]] == [late["id"]]

    # A cancelled record is never overdue and drops out of status=open.
    client.patch(f"/expected-receipts/{late['id']}", json={"status": "cancelled"})
    data = client.get("/expected-receipts", params={"status": "open", "product_id": pid}).json()
    assert late["id"] not in {x["id"] for x in data["expected_receipts"]}
    assert _get_er(client, late["id"])["is_overdue"] is False

    bad = client.get("/expected-receipts", params={"status": "bogus"})
    assert bad.status_code == 422


@pytest.mark.db
def test_patch_edits_then_status_transitions_are_guarded(client, cur):
    pid = _seed_product(cur, "FR2 Patch Prod")
    _seed_supplier(cur, "Patch Supply")
    er = _create_er(client, pid, "Patch Supply", 100, "2026-09-01", reference_number="REF-1", notes="n1")

    r = client.patch(f"/expected-receipts/{er['id']}", json={
        "expected_qty": 250, "expected_date": "2026-09-20", "reference_number": "REF-2", "notes": None,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert set(body["changed_fields"]) == {"expected_qty", "expected_date", "reference_number", "notes"}
    rec = body["expected_receipt"]
    assert rec["expected_qty"] == 250 and rec["remaining"] == 250
    assert rec["expected_date"] == "2026-09-20"
    assert rec["reference_number"] == "REF-2"
    assert rec["notes"] is None

    assert client.patch(f"/expected-receipts/{er['id']}", json={"expected_qty": 0}).status_code == 422
    assert client.patch(f"/expected-receipts/{er['id']}", json={}).status_code == 422
    assert client.patch(f"/expected-receipts/{er['id']}", json={"status": "open"}).status_code == 422
    assert client.patch("/expected-receipts/999999", json={"notes": "x"}).status_code == 404

    r = client.patch(f"/expected-receipts/{er['id']}", json={"status": "closed"})
    assert r.status_code == 200
    assert r.json()["expected_receipt"]["status"] == "closed"

    # Closed → no more edits and no re-status.
    r = client.patch(f"/expected-receipts/{er['id']}", json={"notes": "late edit"})
    assert r.status_code == 409
    assert r.json()["error_detail"]["code"] == "EXPECTED_RECEIPT_NOT_OPEN"
    r = client.patch(f"/expected-receipts/{er['id']}", json={"status": "cancelled"})
    assert r.status_code == 409


@pytest.mark.db
def test_create_validation_and_product_name_resolution(client, cur):
    pid = _seed_product(cur, "FR2 Unique Product Name Zed")
    _seed_supplier(cur, "Zed Supply")
    r = client.post("/expected-receipts", json={"product_id": pid, "supplier_name": "Zed Supply", "expected_qty": 0})
    assert r.status_code == 422
    r = client.post("/expected-receipts", json={"supplier_name": "Zed Supply", "expected_qty": 5})
    assert r.status_code == 422
    r = client.post("/expected-receipts", json={"product_id": 987654321, "supplier_name": "Zed Supply", "expected_qty": 5})
    assert r.status_code == 404
    er = _create_er(client, None, "Zed Supply", 5, product_name="FR2 Unique Product Name Zed")
    assert er["product_id"] == pid


@pytest.mark.db
def test_suppliers_endpoints(client, cur):
    _seed_supplier(cur, "Endpoint Supply A")
    r = client.post("/suppliers", json={"name": "  Endpoint   Supply B "})
    assert r.status_code == 201, r.text
    assert r.json()["name"] == "Endpoint Supply B"
    dup = client.post("/suppliers", json={"name": "endpoint supply b"})
    assert dup.status_code == 409
    assert dup.json()["error_detail"]["code"] == "SUPPLIER_EXISTS"
    lst = client.get("/suppliers", params={"q": "endpoint supply"}).json()
    assert {"Endpoint Supply A", "Endpoint Supply B"} <= {s["name"] for s in lst["suppliers"]}


# ─────────────────────────────────────────────────────────────────
# the invariant: expected receipts never touch inventory balances
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_expected_receipts_never_affect_inventory_balances(client, cur):
    pname = "FR2 Invariant Walnuts"
    pid = _seed_product(cur, pname)
    _seed_supplier(cur, "Invariant Nuts")

    # Real on-hand: one posted receipt of 100 lb (unlinked — no ER exists yet).
    resp = _receive(client, pname, "Invariant Nuts", cases=4, case_size_lb=25, lot_code="FR2-INV-LOT")
    lot_id = resp["lot_id"]
    on_hand_before = main.lot_on_hand(cur, lot_id)
    assert on_hand_before == pytest.approx(100)
    lookup_before = client.get("/inventory/lookup", params={"q": pname}).json()
    inv_before = next(x for x in lookup_before["results"] if x["product"] == pname)

    # A large open expected receipt, then an over-receipt-closed one, then a cancelled one.
    _create_er(client, pid, "Invariant Nuts", 10000)
    er2 = _create_er(client, pid, "Invariant Nuts", 1)
    client.patch(f"/expected-receipts/{er2['id']}", json={"status": "cancelled"})

    assert main.lot_on_hand(cur, lot_id) == pytest.approx(on_hand_before)
    lookup_after = client.get("/inventory/lookup", params={"q": pname}).json()
    inv_after = next(x for x in lookup_after["results"] if x["product"] == pname)
    assert inv_after["total_on_hand"] == pytest.approx(inv_before["total_on_hand"]) == pytest.approx(100)

    # Product-level balance straight from the canonical posted-lines subquery.
    cur.execute(
        f"SELECT COALESCE(SUM(quantity_lb), 0) AS bal FROM {main.POSTED_LINES} tl WHERE product_id = %s",
        (pid,),
    )
    assert float(cur.fetchone()["bal"]) == pytest.approx(100)

    # Static guard: the canonical balance subquery never references the FR-2 tables.
    assert "expected_receipt" not in main.POSTED_LINES
