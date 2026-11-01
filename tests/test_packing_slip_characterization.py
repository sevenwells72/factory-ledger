"""Packing slip PDF: what the document says, not only that it is a PDF.

Characterization tests written before moving GET /sales/orders/{id}/packing-slip
out of main.py (modularization PR3). They pin the current output so the move
can prove it changed nothing:

  * pre-shipment FIFO preview: lots oldest first by COALESCE(received_at,
    created_at), posted-only balances, "lb · N units" from case_size_lb,
    supplier lot reference, INSUFFICIENT rows with the unpacked-batch hint,
    service lines as N/A, cancelled lines left out;
  * post-shipment: the lots actually shipped;
  * header, purchase order, footer (plant time via main.get_plant_now, so a
    monkeypatch on main still reaches the handler), download headers;
  * errors and the header / ?key= auth paths. Signed links are covered by
    tests/test_floor_packing_slip.py, staff keys by
    tests/test_dashboard_packing_slip.py, log redaction by
    tests/test_actor_attribution.py.
"""
import io
from datetime import datetime
from uuid import uuid4

import pytest
from pypdf import PdfReader

import main
from tests.test_released_by_attribution import client, _seed  # noqa: F401

pytestmark = pytest.mark.db

FROZEN = datetime(2026, 10, 26, 9, 5, tzinfo=main.PLANT_TIMEZONE)


@pytest.fixture(autouse=True)
def _frozen_plant_time(monkeypatch):
    monkeypatch.setattr(main, "get_plant_now", lambda: FROZEN)


def _product(cur, token, name, *, ptype="finished", case_size=None, is_service=False, parent=None):
    cur.execute(
        "INSERT INTO products (name, type, odoo_code, uom, is_service, active, case_size_lb, parent_batch_product_id) "
        "VALUES (%s, %s, %s, 'lb', %s, true, %s, %s) RETURNING id",
        (f"{name} {token}", ptype, f"PSC-{name[:4]}-{token}", is_service, case_size, parent),
    )
    return cur.fetchone()["id"]


def _lot(cur, product_id, code, qty, received_at, supplier_lot=None):
    cur.execute(
        "INSERT INTO lots (product_id, lot_code, entry_source, received_at, supplier_lot_code) "
        "VALUES (%s, %s, 'received', %s, %s) RETURNING id",
        (product_id, code, received_at, supplier_lot),
    )
    lot_id = cur.fetchone()["id"]
    cur.execute("INSERT INTO transactions (type, timestamp) VALUES ('receive', NOW()) RETURNING id")
    txn = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)",
        (txn, product_id, lot_id, qty),
    )
    return lot_id


def _line(cur, order_id, product_id, qty, *, ordered_quantity=None, status="pending"):
    cur.execute(
        "INSERT INTO sales_order_lines (sales_order_id, product_id, quantity_lb, quantity_shipped_lb, line_status, ordered_quantity) "
        "VALUES (%s, %s, %s, 0, %s, %s) RETURNING id",
        (order_id, product_id, qty, status, ordered_quantity),
    )
    return cur.fetchone()["id"]


@pytest.fixture
def preview_order(db_cursor):
    """One open order exercising every pre-shipment branch."""
    cur = db_cursor
    t = uuid4().hex[:6].upper()
    cur.execute(
        "INSERT INTO customers (name, address, active) VALUES (%s, %s, true) RETURNING id",
        (f"PSC Customer {t}", "12 Test Street\nBayonne, NJ 07002"),
    )
    customer_id = cur.fetchone()["id"]
    order_number = f"PSC-SO-{t}"
    cur.execute(
        "INSERT INTO sales_orders (customer_id, order_number, status, order_date, requested_ship_date, notes) "
        "VALUES (%s, %s, 'confirmed', '2026-10-20', '2026-10-25', %s) RETURNING id",
        (customer_id, order_number, f"PO-{t}"),
    )
    order_id = cur.fetchone()["id"]

    # Spans two lots, oldest first; 25 lb cases; the older lot has a supplier lot.
    granola = _product(cur, t, "Granola", case_size=25)
    _lot(cur, granola, f"PSC-NEW-{t}", 100, "2026-10-05")
    _lot(cur, granola, f"PSC-OLD-{t}", 50, "2026-10-01", supplier_lot=f"SUP-{t}")
    _line(cur, order_id, granola, 100)
    # No finished lots, but unpacked batch inventory -> INSUFFICIENT + pack hint.
    batch = _product(cur, t, "Batch Mix", ptype="batch")
    _lot(cur, batch, f"PSC-BATCH-{t}", 300, "2026-10-02")
    packed = _product(cur, t, "Packed Bar", case_size=10, parent=batch)
    _line(cur, order_id, packed, 40)
    # Partial shortfall, no case size.
    oats = _product(cur, t, "Oats")
    _lot(cur, oats, f"PSC-OATS-{t}", 30, "2026-10-03")
    _line(cur, order_id, oats, 50)
    # Service line counted, not weighed.
    pallet = _product(cur, t, "Pallet Charge", is_service=True)
    _line(cur, order_id, pallet, 3, ordered_quantity=3)
    # Cancelled line never appears.
    ghost = _product(cur, t, "Ghost Item")
    _lot(cur, ghost, f"PSC-GHOST-{t}", 10, "2026-10-01")
    _line(cur, order_id, ghost, 5, status="cancelled")
    return {"order_id": order_id, "order_number": order_number, "t": t}


def _get(client, path, key=main.API_KEY, **kw):
    headers = {"X-API-Key": key} if key else {}
    return client.get(path, headers=headers, **kw)


def _text(resp):
    assert resp.status_code == 200, resp.text[:300]
    reader = PdfReader(io.BytesIO(resp.content))
    assert len(reader.pages) == 1
    return reader.pages[0].extract_text()


def _in_order(text, *parts):
    pos = -1
    for part in parts:
        nxt = text.find(part, pos + 1)
        assert nxt > pos, f"{part!r} missing or out of order"
        pos = nxt


PREVIEW_TEXT = """CNS Confectionery Products LLC
33 Hook Road
Bayonne, NJ 07002 US
(201) 823-1400
miriam@cnscoinc.com
Packing Slip
BILL TO
PSC Customer {t}
12 Test Street
Bayonne, NJ 07002
SHIP TO
PSC Customer {t}
12 Test Street
Bayonne, NJ 07002
SHIP DATE
2026-10-25
SHIP VIA
Customer Pick Up
SO #
PSC-SO-{t}
DATE
2026-10-20
PURCHASE ORDER
PO-{t}
DATE
ACTIVITY
LOT #
QTY
2026-10-20
Granola {t}
PSC-OLD-{t}
(Supplier: SUP-{t})
50 lb · 2 units
2026-10-20
Granola {t}
PSC-NEW-{t}
50 lb · 2 units
2026-10-20
Packed Bar {t}
INSUFFICIENT
40 lb · 4 units
Note: 300 lb of Batch Mix {t} is available — run /pack to convert
to finished goods.
2026-10-20
Oats {t}
PSC-OATS-{t}
30 lb
2026-10-20
Oats {t}
INSUFFICIENT
20 lb
2026-10-20
Pallet Charge {t}
N/A
3
WAREHOUSE CONFIRMATION
Picked By: ___________________________________
Verified By: ___________________________________
Date: ___________________________________
All lot numbers are system-assigned internal codes from Factory Ledger. Do not substitute with supplier lot numbers.
Generated by Factory Ledger | 2026-10-26 09:05 AM ET
 Page 1 of 1
"""


def test_preview_pdf_content(client, preview_order):
    """Whole page, line by line: FIFO order (older lot first), 25 lb cases,
    supplier lot only where recorded, INSUFFICIENT with the unpacked-batch
    hint, a partial shortfall, the service line as N/A, no cancelled line."""
    o = preview_order
    resp = _get(client, f"/sales/orders/{o['order_id']}/packing-slip")
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.headers["content-disposition"] == f'inline; filename="packing_slip_{o["order_number"]}.pdf"'
    assert _text(resp).splitlines() == PREVIEW_TEXT.format(t=o["t"]).splitlines()


def test_shipped_order_lists_the_lots_actually_shipped(client, db_cursor):
    seeded = _seed(db_cursor)
    r = client.post(
        f"/sales/orders/{seeded['order_id']}/ship/commit",
        json={"lines": [{"line_id": seeded["line_id"], "quantity_lb": 10}]},
        headers={"X-API-Key": main.API_KEY},
    )
    assert r.status_code == 200, r.text
    db_cursor.execute("SELECT lot_code FROM lots WHERE id = %s", (seeded["lot_id"],))
    lot_code = db_cursor.fetchone()["lot_code"]
    text = _text(_get(client, f"/sales/orders/{seeded['order_id']}/packing-slip"))
    _in_order(text, lot_code, "10 lb")
    assert "INSUFFICIENT" not in text and "SHIP DATE" in text and "TBD" in text


def test_order_number_resolves_like_the_id(client, preview_order):
    by_id = _text(_get(client, f"/sales/orders/{preview_order['order_id']}/packing-slip"))
    by_number = _text(_get(client, f"/sales/orders/{preview_order['order_number']}/packing-slip"))
    assert by_id == by_number


def test_errors(client, preview_order, db_cursor):
    r = _get(client, "/sales/orders/999999999/packing-slip")
    assert r.status_code == 404, r.text
    assert r.json()["detail"] == "Order #999999999 not found"

    db_cursor.execute("UPDATE sales_orders SET status = 'cancelled' WHERE id = %s", (preview_order["order_id"],))
    r = _get(client, f"/sales/orders/{preview_order['order_id']}/packing-slip")
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"Order {preview_order['order_number']} is cancelled — cannot generate packing slip"


def test_query_key_and_missing_or_wrong_keys(client, preview_order):
    path = f"/sales/orders/{preview_order['order_id']}/packing-slip"
    assert _get(client, path, key=None, params={"key": main.API_KEY}).content.startswith(b"%PDF")
    r = _get(client, path, key=None)
    assert (r.status_code, r.json()["detail"]) == (401, "API key required")
    r = _get(client, path, key=None, params={"key": "not-a-real-key"})
    assert (r.status_code, r.json()["detail"]) == (401, "Invalid API key")
    r = _get(client, path, key="not-a-real-key")
    assert (r.status_code, r.json()["detail"]) == (401, "Invalid API key")


def test_render_failure_returns_json_500(client, preview_order, monkeypatch):
    def boom():
        raise RuntimeError("clock unavailable")
    monkeypatch.setattr(main, "get_plant_now", boom)
    r = _get(client, f"/sales/orders/{preview_order['order_id']}/packing-slip")
    assert r.status_code == 500 and r.json() == {"error": "clock unavailable"}
