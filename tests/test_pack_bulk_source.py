"""A bought-in bulk item repacked into its finished case, through the real Pack ticket.

Same path the dashboard Pack form uses: /resolve with product_type 'pack_source',
then /pack/prepare, physical-tag confirmation and /tickets/{ticket}/commit under a named actor.
"""
import pytest
import main
from tests.test_actor_attribution import client, actors  # noqa
from tests.test_write_tickets_part2 import items, body  # noqa
from tests.test_write_tickets import headers, commit, posted_count, error  # noqa

pytestmark = pytest.mark.db


def balance(cur, lot_id):
    cur.execute("""SELECT COALESCE(SUM(tl.quantity_lb), 0) AS lb FROM transaction_lines tl
                   JOIN transactions t ON t.id = tl.transaction_id
                   WHERE tl.lot_id = %s AND t.status = 'posted'""", (lot_id,))
    return float(cur.fetchone()['lb'])


def test_bulk_item_is_offered_packed_confirmed_and_traced(client, db_cursor, items, actors):
    key = actors['floor']['key']
    bulk, packed = items['ingredient'], items['finished']
    db_cursor.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=%s RETURNING name',
                      (bulk['id'], packed['id']))
    db_cursor.execute('SELECT name FROM products WHERE id=%s', (bulk['id'],))
    bulk_name = db_cursor.fetchone()['name']

    # The source list offers the bulk item only because a finished product is linked to it.
    def sources(product_type):
        r = client.post('/resolve', json={'kind': 'product', 'query': bulk_name, 'context': {
            'action': 'pack', 'group': 'floor', 'product_type': product_type}}, headers=headers(key))
        assert r.status_code == 200, r.text
        data = r.json()
        return [row['id'] for row in ([data['match']] if data.get('match') else data['candidates'])]
    assert bulk['id'] in sources('pack_source')
    assert bulk['id'] not in sources('batch')

    # The packed-product list offers what is linked to the chosen source.
    db_cursor.execute('SELECT name FROM products WHERE id=%s', (packed['id'],))
    r = client.post('/resolve', json={'kind': 'product', 'query': db_cursor.fetchone()['name'], 'context': {
        'action': 'pack', 'group': 'floor', 'product_id': bulk['id']}}, headers=headers(key))
    assert r.status_code == 200 and r.json()['match']['id'] == packed['id'], r.text

    before = balance(db_cursor, bulk['lot_id'])
    payload = body('pack', items) | {'source_product_id': bulk['id'], 'cases': 1,
        'target_lot_code': 'BULK-REPACK-01'}
    r = client.post('/pack/prepare', json=payload, headers=headers(key))
    assert r.status_code == 200, r.text
    draft = r.json()
    plan = draft['draft']['input_plan']
    total = float(draft['draft']['total_lb'])
    assert [row['product_id'] for row in plan] == [bulk['id']]
    assert sum(float(row['quantity_lb']) for row in plan) == total
    # No physical tag confirmed yet: nothing posts.
    assert not draft['can_commit']
    error(commit(client, draft, key), 422, 'LOT_NOT_CONFIRMED')
    assert posted_count(db_cursor, draft) == 0

    confirmations = [{'lot_id': row['lot_id'], 'method': 'full_code', 'value': row['lot_code']} for row in plan]
    done = commit(client, draft, key, lot_confirmations=confirmations)
    assert done.status_code == 200, done.text
    tx = done.json()['transaction_id']

    db_cursor.execute("""SELECT t.type, t.operator_id, tl.product_id, tl.quantity_lb, l.lot_code
                         FROM transactions t JOIN transaction_lines tl ON tl.transaction_id = t.id
                         JOIN lots l ON l.id = tl.lot_id WHERE t.id = %s ORDER BY tl.quantity_lb""", (tx,))
    lines = db_cursor.fetchall()
    assert {row['type'] for row in lines} == {'pack'}
    assert {row['operator_id'] for row in lines} == {actors['floor']['name']}
    assert [(row['product_id'], float(row['quantity_lb'])) for row in lines] == [
        (bulk['id'], -total), (packed['id'], total)]
    assert lines[1]['lot_code'] == 'BULK-REPACK-01'
    assert balance(db_cursor, bulk['lot_id']) == before - total

    # Trace: the new case lot names the bulk lot it came from, and the bulk lot names the case lot.
    back = client.get('/trace/batch/BULK-REPACK-01', headers=headers(key))
    assert back.status_code == 200, back.text
    assert [(row['product_id'], float(row['quantity_lb'])) for row in back.json()['ingredients']] == [(bulk['id'], total)]
    forward = client.get('/trace/ingredient/' + bulk['lot_code'], headers=headers(key))
    assert forward.status_code == 200, forward.text
    assert 'BULK-REPACK-01' in [row['batch_lot_code'] for row in forward.json()['used_in_batches']]
