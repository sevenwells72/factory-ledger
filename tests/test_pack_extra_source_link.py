"""A regular finished product packed from an Extra-Kosher (SS) twin batch.

2026-11-25: Granola Cocoa Vibes 25 LB (linked to the regular Cocoa Crunch batch)
could not be chosen on the Pack form when the source was the SS Cocoa Crunch batch,
under any name or code. Migration 099 adds approved extra pack sources.
"""
from pathlib import Path
from uuid import uuid4

import pytest

from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets_part2 import items, body, prepare  # noqa: F401
from tests.test_write_tickets import headers, commit

pytestmark = pytest.mark.db
MIGRATION = Path(__file__).parents[1] / 'migrations/099_product_pack_sources.sql'


@pytest.fixture
def twin(db_cursor, items):
    """An Extra-Kosher twin of the fixture batch, with 100 lb on one lot."""
    token = uuid4().hex[:8].upper()
    db_cursor.execute("""INSERT INTO products(name,odoo_code,type,uom,default_batch_lb,active,kosher_tier)
        VALUES (%s,%s,'batch','lb',10,true,'extra_kosher') RETURNING id,name""",
                      (f'Twin batch {token} — EXTRA-KOSHER (SS)', f'TWIN-{token}'))
    product = dict(db_cursor.fetchone())
    db_cursor.execute("INSERT INTO lots(product_id,lot_code,kosher_tier) VALUES (%s,%s,'extra_kosher') RETURNING id,lot_code",
                      (product['id'], f'TWIN-LOT-{token}'))
    product.update(lot_id=(lot := db_cursor.fetchone())['id'], lot_code=lot['lot_code'])
    db_cursor.execute("INSERT INTO transactions(type,notes) VALUES ('receive','twin fixture') RETURNING id")
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,100)',
                      (db_cursor.fetchone()['id'], product['id'], product['lot_id']))
    db_cursor.execute("UPDATE products SET kosher_tier='regular' WHERE id=%s", (items['finished']['id'],))
    return product


def packed_choices(client, db_cursor, key, source_id, query):
    r = client.post('/resolve', json={'kind': 'product', 'query': query, 'context': {
        'action': 'pack', 'group': 'floor', 'product_id': source_id}}, headers=headers(key))
    assert r.status_code == 200, r.text
    data = r.json()
    return [row['id'] for row in ([data['match']] if data.get('match') else data['candidates'])]


def link(db_cursor, items, twin):
    db_cursor.execute(MIGRATION.read_text())
    db_cursor.execute(MIGRATION.read_text())  # idempotent
    db_cursor.execute("""INSERT INTO product_pack_sources(finished_product_id,source_product_id,reason,created_by)
        VALUES (%s,%s,'Regular SKU packed from the SS twin batch','test')""", (items['finished']['id'], twin['id']))


def test_unlinked_twin_batch_still_hides_the_regular_product(client, db_cursor, items, twin, actors):
    key = actors['floor']['key']
    db_cursor.execute('SELECT name, odoo_code FROM products WHERE id=%s', (items['finished']['id'],))
    finished = db_cursor.fetchone()
    for query in (finished['name'], finished['odoo_code'], str(items['finished']['id'])):
        assert items['finished']['id'] not in packed_choices(client, db_cursor, key, twin['id'], query)
    # The usual batch link is unchanged.
    assert packed_choices(client, db_cursor, key, items['batch']['id'], finished['name']) == [items['finished']['id']]


def test_linked_twin_batch_offers_packs_and_records_the_downgrade(client, db_cursor, items, twin, actors):
    key = actors['floor']['key']
    link(db_cursor, items, twin)
    db_cursor.execute('SELECT name, odoo_code FROM products WHERE id=%s', (items['finished']['id'],))
    finished = db_cursor.fetchone()
    for query in (finished['name'], finished['odoo_code']):
        assert packed_choices(client, db_cursor, key, twin['id'], query) == [items['finished']['id']]
    # Still offered from its usual batch; another finished product is not offered from the twin.
    assert packed_choices(client, db_cursor, key, items['batch']['id'], finished['name']) == [items['finished']['id']]

    payload = body('pack', items) | {'source_product_id': twin['id'], 'cases': 2}
    prepared = prepare(client, 'pack', payload, key)
    assert 'mismatch' not in str(prepared['draft'].get('warning') or '').lower()
    assert [row['lot_id'] for row in prepared['draft']['input_plan']] == [twin['lot_id']]
    done = commit(client, prepared, key)
    assert done.status_code == 200, done.text
    result = done.json()
    assert result['transaction_kosher_downgrade'] is True
    db_cursor.execute("""SELECT tl.product_id, tl.lot_id, tl.quantity_lb FROM transaction_lines tl
                         WHERE tl.transaction_id=%s ORDER BY tl.quantity_lb""", (result['transaction_id'],))
    lines = db_cursor.fetchall()
    assert [(r['product_id'], float(r['quantity_lb'])) for r in lines] == [
        (twin['id'], -10.0), (items['finished']['id'], 10.0)]
    assert lines[0]['lot_id'] == twin['lot_id']


def test_pack_source_list_offers_a_linked_non_batch_item(client, db_cursor, items, actors):
    key = actors['floor']['key']
    bulk = items['ingredient']
    db_cursor.execute('SELECT name FROM products WHERE id=%s', (bulk['id'],))
    name = db_cursor.fetchone()['name']

    def sources():
        r = client.post('/resolve', json={'kind': 'product', 'query': name, 'context': {
            'action': 'pack', 'group': 'floor', 'product_type': 'pack_source'}}, headers=headers(key))
        assert r.status_code == 200, r.text
        data = r.json()
        return [row['id'] for row in ([data['match']] if data.get('match') else data['candidates'])]
    assert bulk['id'] not in sources()
    db_cursor.execute(MIGRATION.read_text())
    db_cursor.execute("""INSERT INTO product_pack_sources(finished_product_id,source_product_id,reason,created_by)
        VALUES (%s,%s,'bulk repack','test')""", (items['finished']['id'], bulk['id']))
    assert bulk['id'] in sources()
