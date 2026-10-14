"""Ready-state receipt labels reflect the recorded action after later flag changes."""
import pytest

from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_order_tickets import catalog, create_order, prepare  # noqa: F401
from tests.test_write_tickets import commit, headers

pytestmark = pytest.mark.db


@pytest.mark.parametrize('ready,label_en,label_es', [
    (True, 'order marked ready', 'pedido marcado listo'),
    (False, 'order marked not ready', 'pedido marcado no listo'),
])
def test_shift_ready_label_keeps_recorded_state_after_opposite_action(
        client, db_cursor, catalog, actors, ready, label_en, label_es):
    order = create_order(client, catalog, actors['office']['key'])
    floor = actors['floor']['key']
    path = f"/sales/orders/{order['order_id']}/ready/prepare"
    prepared = prepare(client, path, {'ready': ready, 'note': 'recorded state'}, floor)
    result = commit(client, prepared, floor)
    assert result.status_code == 200, result.text
    receipt = result.json()
    assert receipt['ready'] is ready

    # The current flag disagrees with the earlier receipt; each label must still
    # describe its own immutable response, in both languages.
    opposite = prepare(client, path, {'ready': not ready, 'note': 'later state'}, floor)
    later = commit(client, opposite, floor)
    assert later.status_code == 200, later.text
    db_cursor.execute('SELECT ready FROM sales_order_flags WHERE so_number=%s', (order['order_number'],))
    assert db_cursor.fetchone()['ready'] is not ready

    docs = []
    for lang, label in [('en', label_en), ('es', label_es)]:
        response = client.get('/reports/shift-summary', params={'lang': lang}, headers=headers(floor))
        assert response.status_code == 200, response.text
        doc = response.json()
        docs.append(doc)
        rows = {row['receipt_number']: row for row in doc['other_receipts']['rows']}
        row = rows[receipt['receipt_number']]
        assert row['action'] == 'mark_order_ready'
        assert row['line_en'].startswith(f"{receipt['receipt_number']}  {label_en}  ")
        assert row['line_es'].startswith(f"{receipt['receipt_number']}  {label_es}  ")
        assert row['line'].startswith(f"{receipt['receipt_number']}  {label}  ")
        assert row['refs']['order_id'] == order['order_id']
    assert docs[0]['summary_hash'] == docs[1]['summary_hash']
