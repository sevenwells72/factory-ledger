"""Supplies feature (migration 043): packaging / consumables inventory view,
per-product low-stock thresholds, supply-request queue.

Covers:
  * GET /supplies/inventory lists EVERY product, zero-inventory included, on_hand
    as the posted-only ledger SUM (voided lines drop out), alphabetical
  * is_low = on_hand < low_stock_threshold; false when threshold is NULL;
    threshold applies to ingredients too
  * ?category= filter (ingredient|packaging|consumable), invalid → 422;
    'consumable' accepted by the products.type CHECK
  * GET /supplies/inventory/{id}/lots — FIFO order (COALESCE(received_at,
    created_at)), per-lot remaining, depleted lots hidden unless include_empty,
    and the order MATCHES the ship path's multi-lot FIFO allocation
  * POST /supply-requests XOR rule (product_id vs item_text), 404 unknown
    product, qty > 0, requested_by required
  * GET /supply-requests newest first, status filter open|done|all
  * PATCH open → done sets done_at; done → done is 409; DB CHECKs back it up
  * dashboard key is accepted on all five routes (allowlist)
"""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

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
        proxy = _ConnProxy(_db_connection, "supplies_inner")
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
    from psycopg2.extras import RealDictCursor
    c = _db_connection.cursor(cursor_factory=RealDictCursor)
    yield c
    c.close()


# ─────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────

def _seed_product(cur, name, ptype="ingredient", uom="lb", threshold=None, active=True):
    cur.execute(
        """INSERT INTO products (name, type, odoo_code, uom, active, low_stock_threshold)
           VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
        (name, ptype, f"SKU-{name[:20]}", uom, active, threshold),
    )
    return cur.fetchone()["id"]


def _seed_lot(cur, product_id, lot_code, received_at=None):
    cur.execute(
        "INSERT INTO lots (product_id, lot_code, received_at) VALUES (%s, %s, %s) RETURNING id",
        (product_id, lot_code, received_at),
    )
    return cur.fetchone()["id"]


def _post_line(cur, product_id, lot_id, qty, txn_type="receive"):
    """Posted transaction with one line (positive = credit, negative = debit)."""
    cur.execute(
        "INSERT INTO transactions (type, timestamp, status) VALUES (%s, %s, 'posted') RETURNING id",
        (txn_type, datetime.now(timezone.utc).replace(tzinfo=None)),
    )
    tid = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)",
        (tid, product_id, lot_id, qty),
    )
    return tid


def _inv_item(client, product_id, **params):
    r = client.get("/supplies/inventory", params=params)
    assert r.status_code == 200, r.text
    items = [i for i in r.json()["items"] if i["product_id"] == product_id]
    assert len(items) == 1, f"product {product_id} not in inventory listing"
    return items[0]


def _create_sr(client, **body):
    r = client.post("/supply-requests", json=body)
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["success"] is True
    return data["supply_request"]


# ─────────────────────────────────────────────────────────────────
# GET /supplies/inventory
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_inventory_lists_zero_inventory_products_with_on_hand_zero(client, cur):
    pid = _seed_product(cur, "SUP Zero Boxes", ptype="packaging", uom="unit")
    item = _inv_item(client, pid)
    assert item["on_hand"] == 0
    assert item["category"] == "packaging"
    assert item["unit"] == "unit"
    assert item["low_stock_threshold"] is None
    assert item["is_low"] is False


@pytest.mark.db
def test_inventory_on_hand_is_posted_only_ledger_sum(client, cur):
    pid = _seed_product(cur, "SUP Ledger Tape", ptype="consumable", uom="unit")
    lot_a = _seed_lot(cur, pid, "SUP-TAPE-A")
    lot_b = _seed_lot(cur, pid, "SUP-TAPE-B")
    _post_line(cur, pid, lot_a, 40)
    _post_line(cur, pid, lot_b, 25)
    _post_line(cur, pid, lot_a, -10, txn_type="adjust")
    voided = _post_line(cur, pid, lot_b, 500)  # will be voided → must not count
    assert _inv_item(client, pid)["on_hand"] == pytest.approx(555)

    v = client.post(f"/void/{voided}", json={"reason": "supplies test void"})
    assert v.status_code == 200, v.text
    assert _inv_item(client, pid)["on_hand"] == pytest.approx(55), "voided lines must drop out"


@pytest.mark.db
def test_inventory_is_low_uses_threshold_and_applies_to_ingredients(client, cur):
    low = _seed_product(cur, "SUP Low Oats", ptype="ingredient", threshold=100)
    ok = _seed_product(cur, "SUP Ok Oats", ptype="ingredient", threshold=100)
    equal = _seed_product(cur, "SUP Equal Oats", ptype="ingredient", threshold=50)
    unset = _seed_product(cur, "SUP Unset Oats", ptype="ingredient", threshold=None)
    for pid, qty in ((low, 99.5), (ok, 100.5), (equal, 50), (unset, 0)):
        lot = _seed_lot(cur, pid, f"SUP-OATS-{pid}")
        if qty:
            _post_line(cur, pid, lot, qty)

    assert _inv_item(client, low)["is_low"] is True
    assert _inv_item(client, low)["low_stock_threshold"] == 100
    assert _inv_item(client, ok)["is_low"] is False
    assert _inv_item(client, equal)["is_low"] is False, "on_hand == threshold is not low (strict <)"
    assert _inv_item(client, unset)["is_low"] is False, "NULL threshold = no alerting even at 0"

    r = client.get("/supplies/inventory", params={"low_only": "true"})
    ids = {i["product_id"] for i in r.json()["items"]}
    assert low in ids and ok not in ids and unset not in ids
    assert r.json()["low_count"] == len(r.json()["items"])


@pytest.mark.db
def test_inventory_category_filter_and_alphabetical_order(client, cur):
    a = _seed_product(cur, "SUP zz Consumable Gloves", ptype="consumable", uom="unit")
    b = _seed_product(cur, "SUP aa Consumable Sanitizer", ptype="consumable", uom="unit")
    c = _seed_product(cur, "SUP Packaging Liner", ptype="packaging", uom="unit")

    r = client.get("/supplies/inventory", params={"category": "consumable"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["category"] == "consumable"
    assert all(i["category"] == "consumable" for i in body["items"])
    ids = [i["product_id"] for i in body["items"]]
    assert a in ids and b in ids and c not in ids
    names = [i["name"] for i in body["items"]]
    assert names == sorted(names, key=str.lower)
    assert names.index("SUP aa Consumable Sanitizer") < names.index("SUP zz Consumable Gloves")

    r = client.get("/supplies/inventory", params={"category": "packaging"})
    ids = [i["product_id"] for i in r.json()["items"]]
    assert c in ids and a not in ids

    r = client.get("/supplies/inventory", params={"category": "widgets"})
    assert r.status_code == 422
    assert r.json()["detail"]["error_code"] == "INVALID_CATEGORY"


@pytest.mark.db
def test_products_type_check_rejects_unknown_but_accepts_consumable(cur):
    import psycopg2
    cur.execute("SAVEPOINT typechk")
    with pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute("INSERT INTO products (name, type) VALUES ('SUP Bad Type', 'widget')")
    cur.execute("ROLLBACK TO SAVEPOINT typechk")
    cur.execute("INSERT INTO products (name, type) VALUES ('SUP Consumable OK', 'consumable') RETURNING type")
    assert cur.fetchone()["type"] == "consumable"
    with pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute("UPDATE products SET low_stock_threshold = -1 WHERE name = 'SUP Consumable OK'")
    cur.execute("ROLLBACK TO SAVEPOINT typechk")


# ─────────────────────────────────────────────────────────────────
# GET /supplies/inventory/{product_id}/lots
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_lots_are_fifo_with_per_lot_remaining_and_hide_depleted(client, cur):
    pid = _seed_product(cur, "SUP FIFO Film", ptype="packaging", uom="unit")
    base = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    # Inserted newest-first on purpose: id order != FIFO order.
    lot_new = _seed_lot(cur, pid, "SUP-FILM-NEW", base + timedelta(days=10))
    lot_mid = _seed_lot(cur, pid, "SUP-FILM-MID", base + timedelta(days=5))
    lot_old = _seed_lot(cur, pid, "SUP-FILM-OLD", base)
    lot_empty = _seed_lot(cur, pid, "SUP-FILM-EMPTY", base - timedelta(days=1))
    lot_norecv = _seed_lot(cur, pid, "SUP-FILM-NORECV", None)  # falls back to created_at (now → last)
    _post_line(cur, pid, lot_new, 30)
    _post_line(cur, pid, lot_mid, 20)
    _post_line(cur, pid, lot_old, 10)
    _post_line(cur, pid, lot_old, -4, txn_type="pack")
    _post_line(cur, pid, lot_empty, 7)
    _post_line(cur, pid, lot_empty, -7, txn_type="pack")
    _post_line(cur, pid, lot_norecv, 5)

    r = client.get(f"/supplies/inventory/{pid}/lots")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["product_id"] == pid and body["unit"] == "unit"
    codes = [l["lot_code"] for l in body["lots"]]
    assert codes == ["SUP-FILM-OLD", "SUP-FILM-MID", "SUP-FILM-NEW", "SUP-FILM-NORECV"], codes
    assert [l["fifo_rank"] for l in body["lots"]] == [1, 2, 3, 4]
    by_code = {l["lot_code"]: l for l in body["lots"]}
    assert by_code["SUP-FILM-OLD"]["remaining"] == pytest.approx(6)
    assert by_code["SUP-FILM-MID"]["remaining"] == pytest.approx(20)
    assert by_code["SUP-FILM-NEW"]["remaining"] == pytest.approx(30)
    assert by_code["SUP-FILM-OLD"]["lot_date"].startswith("2026-08-01")
    assert body["on_hand"] == pytest.approx(61)
    assert body["lot_count"] == 4

    r2 = client.get(f"/supplies/inventory/{pid}/lots", params={"include_empty": "true"})
    codes2 = [l["lot_code"] for l in r2.json()["lots"]]
    assert codes2 == ["SUP-FILM-EMPTY", "SUP-FILM-OLD", "SUP-FILM-MID", "SUP-FILM-NEW", "SUP-FILM-NORECV"], codes2
    assert r2.json()["on_hand"] == pytest.approx(61)

    assert client.get("/supplies/inventory/999999999/lots").status_code == 404


@pytest.mark.db
def test_lots_order_matches_ship_multi_lot_fifo_allocation(client, cur):
    """The lots endpoint must present lots in the same order the ship path
    consumes them (POST /ship preview → multi_lot_fifo allocations)."""
    pname = "SUP FIFO Match Cocoa"
    pid = _seed_product(cur, pname, ptype="finished")
    base = datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)
    lot_c = _seed_lot(cur, pid, "SUP-MATCH-C", base + timedelta(days=2))
    lot_a = _seed_lot(cur, pid, "SUP-MATCH-A", base)
    lot_b = _seed_lot(cur, pid, "SUP-MATCH-B", base + timedelta(days=1))
    for lot in (lot_a, lot_b, lot_c):
        _post_line(cur, pid, lot, 10)

    ship = client.post("/ship", json={
        "mode": "preview", "product_name": pname, "quantity_lb": 25,
        "customer_name": "SUP Nobody Customer Inc", "order_reference": "SUP-TEST",
    })
    assert ship.status_code == 200, ship.text
    assert ship.json()["ship_mode"] == "multi_lot_fifo"
    ship_order = [a["lot_code"] for a in ship.json()["allocations"]]

    lots = client.get(f"/supplies/inventory/{pid}/lots").json()["lots"]
    supplies_order = [l["lot_code"] for l in lots]
    assert supplies_order[: len(ship_order)] == ship_order
    assert supplies_order == ["SUP-MATCH-A", "SUP-MATCH-B", "SUP-MATCH-C"]


# ─────────────────────────────────────────────────────────────────
# supply requests
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_create_supply_request_xor_rule(client, cur):
    pid = _seed_product(cur, "SUP Req Boxes", ptype="packaging", uom="unit")

    r = client.post("/supply-requests", json={"requested_by": "Maria"})
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["error_code"] == "SUPPLY_REQUEST_TARGET_REQUIRED"
    assert r.json()["success"] is False

    r = client.post("/supply-requests", json={"product_id": pid, "item_text": "boxes", "requested_by": "Maria"})
    assert r.status_code == 422
    assert r.json()["detail"]["error_code"] == "SUPPLY_REQUEST_TARGET_AMBIGUOUS"

    r = client.post("/supply-requests", json={"item_text": "   ", "requested_by": "Maria"})
    assert r.status_code == 422
    assert r.json()["detail"]["error_code"] == "SUPPLY_REQUEST_TARGET_REQUIRED"

    r = client.post("/supply-requests", json={"product_id": 999999999, "requested_by": "Maria"})
    assert r.status_code == 404
    assert r.json()["detail"]["error_code"] == "PRODUCT_NOT_FOUND"

    r = client.post("/supply-requests", json={"product_id": pid, "requested_by": "  "})
    assert r.status_code == 422
    assert r.json()["detail"]["error_code"] == "REQUESTED_BY_REQUIRED"

    r = client.post("/supply-requests", json={"product_id": pid, "qty": 0, "requested_by": "Maria"})
    assert r.status_code == 422
    assert r.json()["detail"]["error_code"] == "INVALID_QUANTITY"

    sr = _create_sr(client, product_id=pid, qty=12, note="running out", requested_by="Maria")
    assert sr["product_id"] == pid and sr["item_text"] is None
    assert sr["display_name"] == "SUP Req Boxes" and sr["unit"] == "unit"
    assert sr["qty"] == 12 and sr["status"] == "open" and sr["done_at"] is None
    assert sr["requested_by"] == "Maria"

    sr2 = _create_sr(client, item_text="  blue   nitrile gloves ", requested_by="Jose")
    assert sr2["product_id"] is None and sr2["item_text"] == "blue nitrile gloves"
    assert sr2["display_name"] == "blue nitrile gloves" and sr2["qty"] is None

    cur.execute("SELECT COUNT(*) AS n FROM supply_requests WHERE requested_by IN ('Maria','Jose')")
    assert cur.fetchone()["n"] == 2


@pytest.mark.db
def test_list_supply_requests_newest_first_and_status_filter(client, cur):
    pid = _seed_product(cur, "SUP List Labels", ptype="packaging", uom="unit")
    first = _create_sr(client, product_id=pid, requested_by="A")
    second = _create_sr(client, item_text="SUP paper towels", requested_by="B")
    third = _create_sr(client, item_text="SUP degreaser", requested_by="C")
    done = client.patch(f"/supply-requests/{first['id']}", json={"status": "done"})
    assert done.status_code == 200, done.text

    ids_all = [s["id"] for s in client.get("/supply-requests").json()["supply_requests"]]
    mine = [i for i in ids_all if i in (first["id"], second["id"], third["id"])]
    assert mine == [third["id"], second["id"], first["id"]], "newest first"

    body = client.get("/supply-requests", params={"status": "open"}).json()
    ids_open = [s["id"] for s in body["supply_requests"]]
    assert second["id"] in ids_open and third["id"] in ids_open and first["id"] not in ids_open
    assert all(s["status"] == "open" for s in body["supply_requests"])

    ids_done = [s["id"] for s in client.get("/supply-requests", params={"status": "done"}).json()["supply_requests"]]
    assert first["id"] in ids_done and second["id"] not in ids_done

    ids_all2 = [s["id"] for s in client.get("/supply-requests", params={"status": "all"}).json()["supply_requests"]]
    assert set(ids_all2) == set(ids_all)

    r = client.get("/supply-requests", params={"status": "cancelled"})
    assert r.status_code == 422
    assert r.json()["detail"]["error_code"] == "INVALID_STATUS"


@pytest.mark.db
def test_patch_open_to_done_only(client, cur):
    sr = _create_sr(client, item_text="SUP zip ties", requested_by="Dana")
    r = client.patch(f"/supply-requests/{sr['id']}", json={"status": "done"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["supply_request"]["status"] == "done"
    assert body["supply_request"]["done_at"] is not None
    assert body["changed_fields"] == ["status", "done_at"]

    again = client.patch(f"/supply-requests/{sr['id']}", json={"status": "done"})
    assert again.status_code == 409
    assert again.json()["detail"]["error_code"] == "SUPPLY_REQUEST_NOT_OPEN"
    assert again.json()["success"] is False

    # Body may only carry status='done'
    bad = client.patch(f"/supply-requests/{sr['id']}", json={"status": "open"})
    assert bad.status_code == 422

    assert client.patch("/supply-requests/999999999", json={"status": "done"}).status_code == 404

    # DB-level guards behind the API
    import psycopg2
    cur.execute("SAVEPOINT srchk")
    with pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute("INSERT INTO supply_requests (item_text, requested_by, status) VALUES ('x', 'y', 'done')")
    cur.execute("ROLLBACK TO SAVEPOINT srchk")
    with pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute("INSERT INTO supply_requests (product_id, item_text, requested_by) VALUES (NULL, NULL, 'y')")
    cur.execute("ROLLBACK TO SAVEPOINT srchk")


@pytest.mark.db
def test_dashboard_key_allowed_on_supplies_routes(client, cur):
    pid = _seed_product(cur, "SUP Dash Key Bags", ptype="packaging", uom="unit")
    h = {"X-API-Key": main.DASHBOARD_API_KEY}
    assert client.get("/supplies/inventory", headers=h).status_code == 200
    assert client.get(f"/supplies/inventory/{pid}/lots", headers=h).status_code == 200
    r = client.post("/supply-requests", json={"product_id": pid, "requested_by": "dash"}, headers=h)
    assert r.status_code == 201, r.text
    assert client.get("/supply-requests", headers=h).status_code == 200
    assert client.patch(f"/supply-requests/{r.json()['supply_request_id']}", json={"status": "done"}, headers=h).status_code == 200
    # and a route not on the allowlist still 403s for the dashboard key
    assert client.get("/admin/readonly-probe", headers=h).status_code in (403, 404)


@pytest.mark.db
def test_supply_requests_never_touch_inventory(client, cur):
    pid = _seed_product(cur, "SUP NoInv Sleeves", ptype="packaging", uom="unit")
    lot = _seed_lot(cur, pid, "SUP-SLEEVE-1")
    _post_line(cur, pid, lot, 12)
    before = _inv_item(client, pid)["on_hand"]
    sr = _create_sr(client, product_id=pid, qty=100, requested_by="Q")
    client.patch(f"/supply-requests/{sr['id']}", json={"status": "done"})
    assert _inv_item(client, pid)["on_hand"] == before == 12
