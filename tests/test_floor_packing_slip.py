"""Floor GPT commitShipOrder returns a 10-minute signed packing-slip link.

The legacy route has no write ticket, so the link signs 'shipment-<id>' in the
receipt slot of the same PACKING_SLIP_SIGNING_KEY HMAC that ticket shipments
use. No API key ever appears in the URL, and the link opens only the order it
was minted for.
"""
from datetime import datetime, timezone
import time
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
from fastapi import HTTPException

import main
import ship_tickets
from tests.test_released_by_attribution import client, _seed  # noqa: F401

pytestmark = pytest.mark.db


def _commit(client, seeded, qty=10):
    return client.post(
        f"/sales/orders/{seeded['order_id']}/ship/commit",
        json={"lines": [{"line_id": seeded["line_id"], "quantity_lb": qty}]},
        headers={"X-API-Key": main.API_KEY},
    )


def _on_hand(cur, lot_id):
    cur.execute("SELECT COALESCE(SUM(quantity_lb),0) AS q FROM transaction_lines WHERE lot_id=%s", (lot_id,))
    return float(cur.fetchone()["q"])


def _link(order_id, receipt, expires):
    return f"/sales/orders/{order_id}/packing-slip?" + urlencode(
        {"receipt": receipt, "expires": expires, "sig": ship_tickets._sign(order_id, receipt, expires)})


def test_commit_returns_working_signed_link_and_still_ships(client, db_cursor, monkeypatch):
    monkeypatch.delenv("PUBLIC_API_BASE_URL", raising=False)
    monkeypatch.delenv("RAILWAY_PUBLIC_DOMAIN", raising=False)
    seeded = _seed(db_cursor)
    before = time.time()
    r = _commit(client, seeded)
    assert r.status_code == 200, r.text
    body = r.json()
    sid = body["shipment_id"]
    assert sid and body["lines_shipped"]
    assert _on_hand(db_cursor, seeded["lot_id"]) == 490.0

    url = body["packing_slip_url"]
    parts = urlsplit(url)
    assert (parts.scheme, parts.netloc, parts.path) == ("http", "testserver", f"/sales/orders/{seeded['order_id']}/packing-slip")
    query = parse_qs(parts.query)
    assert set(query) == {"receipt", "expires", "sig"} and query["receipt"] == [f"shipment-{sid}"]
    assert main.API_KEY not in url and "key=" not in url
    expires = int(query["expires"][0])
    assert before + 590 <= expires <= time.time() + 600
    assert datetime.fromisoformat(body["packing_slip_expires_at"]) == datetime.fromtimestamp(expires, timezone.utc)

    slip = client.get(parts.path + "?" + parts.query)  # no X-API-Key header
    assert slip.status_code == 200 and slip.content.startswith(b"%PDF"), slip.text


def test_link_cannot_be_edited_to_reach_other_shipments_or_live_longer(client, db_cursor):
    a, b = _seed(db_cursor), _seed(db_cursor)
    ra, rb = _commit(client, a).json(), _commit(client, b).json()
    path = urlsplit(ra["packing_slip_url"])
    q = {k: v[0] for k, v in parse_qs(path.query).items()}
    get = lambda order_id, **over: client.get(f"/sales/orders/{order_id}/packing-slip?" + urlencode({**q, **over}))

    assert get(a["order_id"]).status_code == 200
    assert get(b["order_id"]).status_code == 401                                   # other order, same sig
    assert get(a["order_id"], receipt=f"shipment-{rb['shipment_id']}").status_code == 401
    assert get(a["order_id"], expires=int(q["expires"]) + 60).status_code == 401  # stretched expiry
    assert get(a["order_id"], sig="0" * 64).status_code == 401

    # Correctly signed, but the shipment belongs to another order.
    assert client.get(_link(a["order_id"], f"shipment-{rb['shipment_id']}", int(time.time()) + 300)).status_code == 401
    # Correctly signed, but expired or beyond the 10-minute window.
    assert client.get(_link(a["order_id"], f"shipment-{ra['shipment_id']}", int(time.time()) - 1)).status_code == 401
    assert client.get(_link(a["order_id"], f"shipment-{ra['shipment_id']}", int(time.time()) + 660)).status_code == 401
    # Malformed shipment receipts fall through to the ticket lookup and match nothing.
    for receipt in ("shipment-0", f"shipment-{ra['shipment_id']} ", f"shipment--{ra['shipment_id']}"):
        assert client.get(_link(a["order_id"], receipt, int(time.time()) + 300)).status_code == 401


def test_signing_failure_keeps_the_committed_shipment(client, db_cursor, monkeypatch):
    def unavailable(*args):
        raise HTTPException(503, "Packing slip signing is unavailable.")
    monkeypatch.setattr(ship_tickets, "_sign", unavailable)
    seeded = _seed(db_cursor)
    r = _commit(client, seeded)
    assert r.status_code == 200, r.text
    assert r.json()["shipment_id"] and r.json()["packing_slip_url"] is None
    assert _on_hand(db_cursor, seeded["lot_id"]) == 490.0


@pytest.mark.parametrize("env,expected", [
    ({"PUBLIC_API_BASE_URL": "https://api.example/"}, "https://api.example/sales/orders/"),
    ({"RAILWAY_PUBLIC_DOMAIN": "fl.up.railway.app"}, "https://fl.up.railway.app/sales/orders/"),
])
def test_public_base_comes_from_configuration_not_the_host_header(client, db_cursor, monkeypatch, env, expected):
    monkeypatch.delenv("PUBLIC_API_BASE_URL", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    r = client.post(
        f"/sales/orders/{_seed(db_cursor)['order_id']}/ship/commit",
        json={"ship_all": True}, headers={"X-API-Key": main.API_KEY, "Host": "evil.example"})
    assert r.status_code == 200, r.text
    assert r.json()["packing_slip_url"].startswith(expected)


def test_preview_and_nothing_shipped_return_no_link(client, db_cursor):
    seeded = _seed(db_cursor, stock=0)
    r = client.post(f"/sales/orders/{seeded['order_id']}/ship/preview", json={}, headers={"X-API-Key": main.API_KEY})
    assert r.status_code == 200 and "packing_slip_url" not in r.json()
