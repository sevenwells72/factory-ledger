"""Open Charge/Product warnings show the live item without rewriting history."""
from uuid import uuid4

import pytest

import charge_setting
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import headers


@pytest.mark.db
@pytest.mark.parametrize('surface', ['list', 'detail'])
@pytest.mark.parametrize('status', ['open', 'escalated'])
@pytest.mark.parametrize('original_kind,current_kind', [('product', 'charge'), ('charge', 'product')])
def test_open_warning_uses_current_name_and_setting_and_keeps_original_detail(
        client, actors, db_cursor, surface, status, original_kind, current_kind):
    token = str(uuid4().int)
    original_name = ('Coffee original ' if original_kind == 'product' else 'Shrink wrap original ') + token
    current_name = ('Coffee current ' if current_kind == 'product' else 'Shrink wrap current ') + token
    auth = headers(actors['office']['key'])
    response = client.post('/catalog/items', headers=auth,
                           json={'name': original_name, 'item_kind': original_kind, 'type': 'finished'})
    assert response.status_code == 200, response.text
    product = response.json()['item']
    check_id = product['check_exception_id']
    db_cursor.execute('SELECT detail FROM exceptions WHERE id=%s', (check_id,))
    original_detail = db_cursor.fetchone()['detail']

    response = client.post(f"/catalog/items/{product['id']}/item-kind", headers=auth,
                           json={'item_kind': current_kind})
    assert response.status_code == 200, response.text
    db_cursor.execute('UPDATE products SET name=%s WHERE id=%s', (current_name, product['id']))
    if status == 'escalated':
        db_cursor.execute("UPDATE exceptions SET status='escalated', escalated_at=clock_timestamp() WHERE id=%s", (check_id,))
    path = f'/exceptions/{check_id}' if surface == 'detail' else \
        f"/exceptions?kind=CHARGE_SETTING_CHECK&product_id={product['id']}"
    response = client.get(path, headers=auth)
    assert response.status_code == 200, response.text
    warning = response.json() if surface == 'detail' else response.json()['exceptions'][0]
    detail = warning['detail']
    assert warning['product']['name'] == current_name
    assert detail['current_name'] == current_name and detail['current_marked_as'] == current_kind
    assert f'{current_name} is now marked {charge_setting.LABELS[current_kind][0]}.' in detail['message']
    assert f'{current_name} ahora está marcado como {charge_setting.LABELS[current_kind][1]}.' in detail['message_es']
    assert original_name in detail['message'] and 'Originally asked about' in detail['message']
    assert original_name in detail['message_es'] and 'Se preguntó originalmente' in detail['message_es']
    for key, value in original_detail.items():
        assert detail[key] == value
    db_cursor.execute('SELECT detail FROM exceptions WHERE id=%s', (check_id,))
    assert db_cursor.fetchone()['detail'] == original_detail
    db_cursor.execute("SELECT count(*) AS n FROM exceptions WHERE product_id=%s AND kind='CHARGE_SETTING_CHECK'",
                      (product['id'],))
    assert db_cursor.fetchone()['n'] == 1
