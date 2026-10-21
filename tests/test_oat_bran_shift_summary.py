"""Persisted substitution evidence stays visible in both kinds of shift make."""
import json

import pytest

import main
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets_part2 import items  # noqa: F401
from tests.test_shift_summary import summary, section
from tests.test_write_tickets import headers

pytestmark = pytest.mark.db


def direct_swap(cur, items, actor_id=None, *, date=None):
    cur.execute("INSERT INTO products(name,type,odoo_code,is_service) VALUES ('Oat Bran','ingredient','TEST-BRAN',false) RETURNING id")
    bran = cur.fetchone()['id']
    cur.execute("""INSERT INTO transactions(type,notes,occurred_at,entered_by_actor_id)
                   VALUES ('make','Automatic oat substitution',%s,%s) RETURNING id""",
                (date or main.get_plant_now(), actor_id))
    txn = cur.fetchone()['id']
    for name, qty in [('batch', 10), ('ingredient', -10)]:
        cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,%s)',
                    (txn, items[name]['id'], items[name]['lot_id'], qty))
    evidence = {'automatic': True, 'quantity_lb': 3, 'available_bran_lb': 0,
                'message_en': 'No oat bran in stock — used 3 lb extra oats instead.',
                'message_es': 'No hay salvado de avena — se usaron 3 lb adicionales de avena.'}
    cur.execute('''INSERT INTO transaction_substitutions
        (transaction_id,ingredient_product_id,substitute_product_id,lot_id,reason_code,note,actor_id)
        VALUES (%s,%s,%s,%s,'oat_bran_auto_substitute',%s,%s)''',
        (txn, bran, items['ingredient']['id'], items['ingredient']['lot_id'], json.dumps(evidence), actor_id))
    return txn


def test_direct_swap_summary_has_real_oats_lots_bilingual_notice_and_no_fake_ticket(client, db_cursor, items, actors):
    before = summary(client, main.API_KEY)
    txn = direct_swap(db_cursor, items)
    en = summary(client, main.API_KEY)
    es = summary(client, main.API_KEY, lang='es')
    block = en['direct_make_substitutions']
    assert block['count'] == 1
    [row] = block['rows']
    assert row['transaction_id'] == txn and row['quantity_lb'] == 10
    assert row['consumed_lots'] == [{'lot_id': items['ingredient']['lot_id'],
                                   'lot_code': items['ingredient']['lot_code'], 'quantity_lb': 10}]
    assert 'used 3 lb extra oats' in row['line']
    assert 'se usaron 3 lb adicionales' in es['direct_make_substitutions']['rows'][0]['line']
    assert en['summary_hash'] == es['summary_hash'] != before['summary_hash']
    assert en['receipts_today'] == 0 and section(en, 'made')['rows'] == []
    assert 'receipt_number' not in row
    # A shared-key make is visible in the all-person report, not attributed to a floor actor.
    assert summary(client, actors['floor']['key'])['direct_make_substitutions']['count'] == 0
    # Display-only lot labels cannot invalidate a confirmed immutable swap.
    db_cursor.execute('UPDATE lots SET lot_code=%s WHERE id=%s', ('RENAMED-OATS', items['ingredient']['lot_id']))
    renamed = summary(client, main.API_KEY)
    assert renamed['summary_hash'] == en['summary_hash']
    assert renamed['direct_make_substitutions']['rows'][0]['consumed_lots'][0]['lot_code'] == 'RENAMED-OATS'


def test_direct_swap_summary_filters_actor_day_and_voids(client, db_cursor, items, actors):
    from datetime import timedelta
    txn = direct_swap(db_cursor, items, actors['floor']['id'])
    key = actors['floor']['key']
    assert summary(client, key)['direct_make_substitutions']['count'] == 1
    assert summary(client, actors['owner']['key'])['direct_make_substitutions']['count'] == 0
    yesterday = (main.get_plant_now() - timedelta(days=1)).date().isoformat()
    assert summary(client, key, date=yesterday)['direct_make_substitutions']['count'] == 0
    response = client.post(f'/void/{txn}', json={'reason': 'entered twice'}, headers=headers())
    assert response.status_code == 200, response.text
    assert summary(client, key)['direct_make_substitutions']['count'] == 0
