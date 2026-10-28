"""Dashboard "Packing slip" button: named staff open the PDF with their own key.

The browser sends the person's key or session in the X-API-Key header, so no
credential appears in a URL. The served dashboard key stays refused (the slip
carries the customer's address), and so does a deactivated person.
"""
import pytest

import main
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_released_by_attribution import _seed

pytestmark = pytest.mark.db


def _slip(client, order_id, key):
    return client.get(f"/sales/orders/{order_id}/packing-slip", headers={"X-API-Key": key})


@pytest.mark.parametrize("label", ["floor", "office", "owner"])
def test_every_staff_role_opens_the_pdf_with_its_own_key(client, actors, db_cursor, label):
    seeded = _seed(db_cursor)
    resp = _slip(client, seeded["order_id"], actors[label]["key"])
    assert resp.status_code == 200, resp.text[:300]
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF")


def test_served_dashboard_key_is_still_refused_in_the_header(client, actors, db_cursor):
    seeded = _seed(db_cursor)
    resp = _slip(client, seeded["order_id"], main.DASHBOARD_API_KEY)
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "API key not authorized for this endpoint"


def test_deactivated_person_cannot_open_the_pdf(client, actors, db_cursor):
    seeded = _seed(db_cursor)
    resp = _slip(client, seeded["order_id"], actors["retired"]["key"])
    assert resp.status_code == 401, resp.text
    assert resp.json()["detail"] == "Invalid API key"
