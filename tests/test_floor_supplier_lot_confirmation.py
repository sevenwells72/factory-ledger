"""The unchanged Floor Make/Pack operations honor explicit supplier-lot inputs."""
import pytest

import main
from tests.test_actor_attribution import client  # noqa: F401
from tests.test_write_tickets import headers
from tests.test_write_tickets_part2 import items  # noqa: F401

pytestmark = pytest.mark.db


def floor_body(action, items):
    common = {'lot_confirmations': [], 'ingredient_lot_allocations': []}
    if action == 'make':
        return common | {'product_name': items['batch']['name'], 'batches': 1}
    return common | {'source_product': items['batch']['name'],
                     'target_product': items['finished']['name'], 'cases': 2}


@pytest.mark.parametrize('action', ['make', 'pack'])
def test_floor_preview_fifo_never_confirms_and_commit_requires_choice(client, items, action):
    payload = floor_body(action, items)
    preview = client.post('/'+action, json=payload, headers=headers())
    assert preview.status_code == 200, preview.text
    draft = preview.json()
    assert draft['can_commit'] is False
    assert draft['lot_confirmation_group']['ingredients']
    assert all(row['confirmed'] is False for row in draft['input_plan'])
    refused = client.post('/'+action, json=payload | {'mode': 'commit'}, headers=headers())
    assert refused.status_code == 422, refused.text
    assert refused.json()['detail']['error_code'] == 'LOT_NOT_CONFIRMED'


@pytest.mark.parametrize('action', ['make', 'pack'])
def test_floor_supplier_lot_commit_records_evidence(client, db_cursor, items, action):
    item = items['ingredient' if action == 'make' else 'batch']
    db_cursor.execute('UPDATE lots SET supplier_lot_code=%s WHERE id=%s', ('Bag Lot 22', item['lot_id']))
    payload = floor_body(action, items) | {'mode': 'commit',
        'ingredient_lot_allocations': [{'ingredient_product_id': item['id'],
            'lot_id': item['lot_id'], 'quantity_lb': 10}],
        'lot_confirmations': [{'lot_id': item['lot_id'], 'method': 'supplier_lot', 'value': ' bAGLOT 22 '}]}
    result = client.post('/'+action, json=payload, headers=headers())
    assert result.status_code == 200, result.text
    receipt = result.json()
    assert receipt['input_plan'][0]['confirmed'] is True
    db_cursor.execute('SELECT method,value FROM transaction_lot_confirmations WHERE transaction_id=%s',
                      (receipt['transaction_id'],))
    assert dict(db_cursor.fetchone()) == {'method': 'supplier_lot', 'value': 'bAGLOT 22'}


@pytest.mark.parametrize('action', ['make', 'pack'])
def test_floor_split_consumes_only_selected_quantities(client, db_cursor, items, action):
    item = items['ingredient' if action == 'make' else 'batch']
    db_cursor.execute("INSERT INTO lots(product_id,lot_code,supplier_lot_code) VALUES (%s,%s,'BAG-SECOND') RETURNING id",
                      (item['id'], item['lot_code']+'-SECOND'))
    second = db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO transactions(type) VALUES ('receive') RETURNING id")
    tx = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,100)',
                      (tx, item['id'], second))
    db_cursor.execute("UPDATE lots SET supplier_lot_code='BAG-FIRST' WHERE id=%s", (item['lot_id'],))
    payload = floor_body(action, items) | {'mode': 'commit',
        'ingredient_lot_allocations': [
            {'ingredient_product_id': item['id'], 'lot_id': item['lot_id'], 'quantity_lb': 3},
            {'ingredient_product_id': item['id'], 'lot_id': second, 'quantity_lb': 7}],
        'lot_confirmations': [
            {'lot_id': item['lot_id'], 'method': 'supplier_lot', 'value': 'BAG-FIRST', 'selected': True},
            {'lot_id': second, 'method': 'supplier_lot', 'value': 'BAG-SECOND', 'selected': True}]}
    result = client.post('/'+action, json=payload, headers=headers())
    assert result.status_code == 200, result.text
    db_cursor.execute('SELECT lot_id,quantity_lb FROM transaction_lines WHERE transaction_id=%s AND product_id=%s ORDER BY lot_id',
                      (result.json()['transaction_id'], item['id']))
    assert {row['lot_id']: float(row['quantity_lb']) for row in db_cursor.fetchall()} == {item['lot_id']: -3, second: -7}


@pytest.mark.parametrize('action', ['make', 'pack'])
def test_floor_rejects_split_total_before_posting(client, db_cursor, items, action):
    item = items['ingredient' if action == 'make' else 'batch']
    payload = floor_body(action, items) | {'mode': 'commit',
        'ingredient_lot_allocations': [{'ingredient_product_id': item['id'], 'lot_id': item['lot_id'], 'quantity_lb': 9}],
        'lot_confirmations': [{'lot_id': item['lot_id'], 'method': 'full_code', 'value': item['lot_code']}]}
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    before = db_cursor.fetchone()['n']
    result = client.post('/'+action, json=payload, headers=headers())
    assert result.status_code in (400, 409, 422), result.text
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    assert db_cursor.fetchone()['n'] == before


def test_floor_selected_shortage_keeps_existing_atomic_followup(client, db_cursor, items):
    item = items['ingredient']
    db_cursor.execute("UPDATE lots SET supplier_lot_code='BAG-LOW' WHERE id=%s", (item['lot_id'],))
    payload = floor_body('make', items) | {'mode': 'commit', 'batches': 11,
        'ingredient_lot_allocations': [{'ingredient_product_id': item['id'], 'lot_id': item['lot_id'], 'quantity_lb': 110}],
        'lot_confirmations': [{'lot_id': item['lot_id'], 'method': 'supplier_lot', 'value': 'BAG-LOW', 'selected': True}]}
    result = client.post('/make', json=payload, headers=headers())
    assert result.status_code == 200, result.text
    response = result.json()
    assert response['shortages']
    db_cursor.execute("SELECT id FROM exceptions WHERE transaction_id=%s AND kind='SHORTAGE'",
                      (response['transaction_id'],))
    assert db_cursor.fetchone() is not None
    assert main.lot_on_hand(db_cursor, item['lot_id']) == -10
