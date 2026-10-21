"""A stock-holding batch cannot become a charge through a generic route."""
from uuid import uuid4

import pytest

from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_write_tickets import error, headers


def _name():
    return 'Batch guard ' + uuid4().hex[:10]


def _item(cur, *, product_type='batch', is_service=False):
    cur.execute('''INSERT INTO products(name, type, uom, is_service, active)
                   VALUES (%s, %s, 'lb', %s, true) RETURNING id''',
                (_name(), product_type, is_service))
    return cur.fetchone()['id']


def _saved(cur, product_id):
    cur.execute('SELECT type, is_service, default_batch_lb FROM products WHERE id=%s', (product_id,))
    return cur.fetchone()


@pytest.mark.db
@pytest.mark.parametrize('choice', ['charge', ' Charge '])
def test_general_creation_refuses_a_batch_charge_without_writing(client, db_cursor, choice):
    name = _name()
    response = client.post('/products/quick-create', json={
        'product_name': name, 'product_type': 'batch', 'item_kind': choice,
    }, headers=headers())
    error(response, 422, 'ITEM_KIND_NOT_ALLOWED')
    assert 'Product' in response.json()['detail']['message']
    assert 'Producto' in response.json()['detail']['message_es']
    db_cursor.execute('SELECT count(*) AS n FROM products WHERE name=%s', (name,))
    assert db_cursor.fetchone()['n'] == 0


@pytest.mark.db
def test_general_creation_accepts_a_batch_explicitly_marked_product(client, db_cursor):
    response = client.post('/products/quick-create', json={
        'product_name': _name(), 'product_type': 'batch', 'item_kind': 'product',
    }, headers=headers())
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['product_type'] == 'batch' and result['item_kind'] == 'product'
    assert _saved(db_cursor, result['product_id'])['is_service'] is False


@pytest.mark.db
def test_general_creation_still_requires_a_choice_for_a_batch(client, db_cursor):
    name = _name()
    error(client.post('/products/quick-create', json={
        'product_name': name, 'product_type': 'batch',
    }, headers=headers()), 422, 'ITEM_KIND_REQUIRED')
    db_cursor.execute('SELECT count(*) AS n FROM products WHERE name=%s', (name,))
    assert db_cursor.fetchone()['n'] == 0


@pytest.mark.db
@pytest.mark.parametrize('choice', ['charge', ' Charge '])
def test_admin_cannot_mark_a_batch_as_charge_or_save_other_changes(client, db_cursor, choice):
    product_id = _item(db_cursor)
    before = _saved(db_cursor, product_id)
    error(client.put(f'/admin/products/{product_id}', json={
        'item_kind': choice, 'default_batch_lb': 321,
    }, headers=headers()), 422, 'ITEM_KIND_NOT_ALLOWED')
    assert _saved(db_cursor, product_id) == before


@pytest.mark.db
@pytest.mark.parametrize('body', [{'default_batch_lb': 321}, {'item_kind': 'product', 'default_batch_lb': 321}])
def test_admin_allows_batch_metadata_and_explicit_product(client, db_cursor, body):
    product_id = _item(db_cursor)
    response = client.put(f'/admin/products/{product_id}', json=body, headers=headers())
    assert response.status_code == 200, response.text
    saved = _saved(db_cursor, product_id)
    assert saved['type'] == 'batch' and saved['is_service'] is False
    assert float(saved['default_batch_lb']) == 321


@pytest.mark.db
def test_admin_metadata_keeps_an_existing_nonbatch_charge_setting(client, db_cursor):
    product_id = _item(db_cursor, product_type='finished', is_service=True)
    response = client.put(f'/admin/products/{product_id}', json={'active': False}, headers=headers())
    assert response.status_code == 200, response.text
    assert _saved(db_cursor, product_id)['is_service'] is True


@pytest.mark.db
@pytest.mark.parametrize('path', ['item-kind', 'confirm-kind'])
def test_catalog_edit_and_confirmation_also_refuse_a_batch_charge(client, actors, db_cursor, path):
    product_id = _item(db_cursor)
    error(client.post(f'/catalog/items/{product_id}/{path}', json={'item_kind': 'charge'},
                      headers=headers(actors['office']['key'])), 422, 'ITEM_KIND_NOT_ALLOWED')
    assert _saved(db_cursor, product_id)['is_service'] is False
