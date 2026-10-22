"""Round B: one physical-tag card, including explicit unmatched-lot traceability."""
import pytest
import main
import lot_confirmation
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_write_tickets_part2 import items, body  # noqa: F401
from tests.test_write_tickets import headers, commit, posted_count, error  # noqa: F401
from tests.test_lot_confirmation import prepare, evidence

pytestmark = pytest.mark.db


@pytest.mark.parametrize('language,instruction', [
    ('en', 'Select the supplier lot on each bag you used. If you used more than one lot, split the needed lb between them. FIFO is a suggestion; select every lot yourself.'),
    ('es', 'Seleccione el lote del proveedor de cada bolsa que usó. Si usó más de un lote, divida las lb necesarias entre ellos. FIFO es una sugerencia; seleccione cada lote usted mismo.'),
])
def test_single_lot_requires_tag_on_one_complete_card(client, items, language, instruction):
    prepared = prepare(client, 'make', body('make', items))
    group = prepared['draft']['lot_confirmation_group']
    assert group['kind'] == 'lot_confirmation'
    assert group['instruction_' + language] == instruction
    ingredient = group['ingredients'][0]
    assert ingredient['product_name'] == items['ingredient']['name']
    assert ingredient['suggested_lots'][0]['lot_id'] == items['ingredient']['lot_id']
    assert ingredient['suggested_lots'][0]['confirmed'] is False
    assert ingredient['candidates'][0]['lot_id'] == items['ingredient']['lot_id']
    assert ingredient['different_lot']['label_' + language] == ('Otro lote' if language == 'es' else 'Different lot')
    assert prepared['can_commit'] is False
    error(commit(client, prepared), 422, 'LOT_NOT_CONFIRMED')


def test_all_ingredients_share_one_group_and_keep_fifo_candidates(client, db_cursor, items):
    db_cursor.execute("INSERT INTO products(name,type,uom) VALUES ('Second Round B ingredient','ingredient','lb') RETURNING id")
    pid = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES (%s,%s,2)', (items['batch']['id'], pid))
    for code in ('ROUND-B-OLD-1111', 'ROUND-B-NEW-2222'):
        db_cursor.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id', (pid, code))
        lid = db_cursor.fetchone()['id']
        db_cursor.execute("INSERT INTO transactions(type) VALUES ('receive') RETURNING id")
        tid = db_cursor.fetchone()['id']
        db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,20)', (tid,pid,lid))
    prepared = prepare(client, 'make', body('make', items))
    group = prepared['draft']['lot_confirmation_group']
    assert len(group['ingredients']) == 2
    second = next(i for i in group['ingredients'] if i['product_id'] == pid)
    assert [r['lot_code'] for r in second['candidates']] == ['ROUND-B-OLD-1111', 'ROUND-B-NEW-2222']
    assert second['suggested_lots'][0]['lot_code'] == 'ROUND-B-OLD-1111'
    assert all(i['product_name'] for i in prepared['draft']['input_plan'])


def test_wrong_tag_rejected_and_matching_tag_records(client, db_cursor, items):
    prepared = prepare(client, 'make', body('make', items))
    assert prepared['draft']['lot_confirmation_group']['ingredients']
    error(commit(client, prepared, lot_confirmations=[evidence(items['ingredient'], value='NOPE')]), 422, 'LOT_CONFIRMATION_MISMATCH')
    assert posted_count(db_cursor, prepared) == 0
    result = commit(client, prepared, lot_confirmations=[evidence(items['ingredient'])])
    assert result.status_code == 200, result.text
    assert result.json()['input_plan'][0]['confirmed'] is True
    assert result.json().get('lot_exceptions', []) == []


@pytest.mark.parametrize('physical_tag', ['TAG-NOT-IN-SYSTEM-7654', '7654', 'Otro lote'])
def test_different_lot_posts_exception_without_debiting_fifo(client, db_cursor, items, actors, physical_tag):
    prepared = prepare(client, 'make', body('make', items), actors['floor']['key'])
    original = items['ingredient']
    db_cursor.execute('SELECT COALESCE(sum(quantity_lb),0) AS qty FROM ledger_current_transaction_lines WHERE lot_id=%s', (original['lot_id'],))
    before = db_cursor.fetchone()['qty']
    confirmation = evidence(original, 'different_lot', physical_tag)
    result = commit(client, prepared, actors['floor']['key'], lot_confirmations=[confirmation])
    assert result.status_code == 200, result.text
    result = result.json()
    row = result['input_plan'][0]
    assert row['lot_id'] != original['lot_id']
    assert row['confirmed'] is False
    assert row['lot_exception']['physical_tag'] == physical_tag
    assert row['lot_exception']['suggested_lot_id'] == original['lot_id']
    assert result['lot_exceptions'][0]['exception_id']
    db_cursor.execute('SELECT COALESCE(sum(quantity_lb),0) AS qty FROM ledger_current_transaction_lines WHERE lot_id=%s', (original['lot_id'],))
    assert db_cursor.fetchone()['qty'] == before
    db_cursor.execute('SELECT lot_id,quantity_lb FROM transaction_lines WHERE transaction_id=%s AND product_id=%s', (result['transaction_id'], original['id']))
    line = db_cursor.fetchone()
    assert line['lot_id'] == row['lot_id'] and float(line['quantity_lb']) == -10
    db_cursor.execute('SELECT count(*) AS n FROM transaction_lot_confirmations WHERE transaction_id=%s', (result['transaction_id'],))
    assert db_cursor.fetchone()['n'] == 0
    db_cursor.execute('SELECT kind,detail,owner_actor_id FROM exceptions WHERE id=%s', (result['lot_exceptions'][0]['exception_id'],))
    exception = db_cursor.fetchone()
    assert exception['kind'] == 'SHIFT_DISCREPANCY'
    assert exception['detail']['physical_tag'] == physical_tag
    assert exception['detail']['suggested_lot_id'] == original['lot_id']
    assert exception['owner_actor_id'] is not None
    assert result['shortages'][0]['lot_id'] == row['lot_id']
    assert commit(client, prepared, actors['floor']['key']).json() == result | {'replayed': True}


def test_unmatched_lot_exception_is_atomic(client, db_cursor, items, monkeypatch):
    prepared = prepare(client, 'make', body('make', items))
    def crash(*args, **kwargs):
        raise RuntimeError('round B exception failure')
    monkeypatch.setattr(lot_confirmation, 'record_lot_exceptions', crash)
    with pytest.raises(RuntimeError, match='round B exception failure'):
        commit(client, prepared, lot_confirmations=[evidence(items['ingredient'], 'different_lot', '7654')])
    assert posted_count(db_cursor, prepared) == 0
    db_cursor.execute("SELECT count(*) AS n FROM lots WHERE lot_code LIKE 'UNVERIFIED-%'")
    assert db_cursor.fetchone()['n'] == 0


@pytest.mark.parametrize('identity', ['ingredient_catalog', 'ingredient_lot'])
def test_different_lot_does_not_bypass_extra_kosher_input_provenance(client, db_cursor, items, identity):
    # Historical lot tiers remain immutable when the catalog later changes.
    db_cursor.execute("UPDATE products SET kosher_tier='extra_kosher' WHERE id=%s", (items['ingredient']['id'],))
    payload = body('make', items)
    if identity == 'ingredient_lot':
        db_cursor.execute("INSERT INTO lots(product_id,lot_code) VALUES (%s,'ROUND-B-EXTRA-1234') RETURNING id", (items['ingredient']['id'],))
        extra_lot = db_cursor.fetchone()['id']
        db_cursor.execute("UPDATE products SET kosher_tier='regular' WHERE id=%s", (items['ingredient']['id'],))
        payload['ingredient_lots'] = [{'ingredient_product_id': items['ingredient']['id'], 'lot_id': extra_lot}]
    prepared = prepare(client, 'make', payload)
    assert prepared['draft']['lot_confirmation_group']['ingredients']
    selected = prepared['draft']['input_plan'][0]
    response = commit(client, prepared, lot_confirmations=[evidence(selected, 'different_lot', '7654')])
    assert response.status_code in (409, 422), response.text
    assert 'PHYSICAL_LOT_KOSHER_REVIEW_REQUIRED' in response.text
    assert posted_count(db_cursor, prepared) == 0


def test_extra_kosher_output_different_regular_input_waits_for_owner_and_keeps_trace(client, db_cursor, items, actors):
    db_cursor.execute("UPDATE products SET kosher_tier='extra_kosher' WHERE id=%s", (items['batch']['id'],))
    key = actors['floor']['key']
    prepared = prepare(client, 'make', body('make', items), key)
    before = main.lot_on_hand(db_cursor, items['ingredient']['lot_id'])
    held = commit(client, prepared, key,
                  lot_confirmations=[evidence(items['ingredient'], 'different_lot', 'PHYSICAL-7654')])
    assert held.status_code == 202, held.text
    assert held.json()['status'] == 'awaiting_approval' and held.json()['posted'] is False
    assert posted_count(db_cursor, prepared) == 0
    db_cursor.execute("SELECT count(*) AS n FROM lots WHERE lot_code LIKE 'UNVERIFIED-%'")
    assert db_cursor.fetchone()['n'] == 0
    approved = client.post(f"/exceptions/{held.json()['exception_id']}/approve", json={}, headers=headers(actors['owner']['key']))
    assert approved.status_code == 200, approved.text
    result = approved.json()
    [actual] = result['input_plan']
    assert actual['confirmed'] is False and actual['lot_id'] != items['ingredient']['lot_id']
    assert actual['lot_exception']['physical_tag'] == 'PHYSICAL-7654'
    assert actual['lot_exception']['suggested_lot_id'] == items['ingredient']['lot_id']
    assert result['lot_exceptions'][0]['exception_id']
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == before
    db_cursor.execute('SELECT kosher_tier,kosher_attested_by_actor_id FROM lots WHERE id=%s', (actual['lot_id'],))
    assert dict(db_cursor.fetchone()) == {'kosher_tier': 'regular', 'kosher_attested_by_actor_id': None}
    db_cursor.execute('SELECT kosher_tier,kosher_attested_by_actor_id FROM lots WHERE id=%s', (result['lot_id'],))
    assert dict(db_cursor.fetchone()) == {'kosher_tier': 'extra_kosher', 'kosher_attested_by_actor_id': actors['owner']['id']}
    db_cursor.execute('SELECT ingredient_lot_id FROM ingredient_lot_consumption WHERE transaction_id=%s', (result['transaction_id'],))
    assert db_cursor.fetchone()['ingredient_lot_id'] == actual['lot_id']
    db_cursor.execute("SELECT l.lot_id FROM trace_event_lots l JOIN trace_events e ON e.id=l.trace_event_id WHERE e.transaction_id=%s AND l.role='input'", (result['transaction_id'],))
    assert db_cursor.fetchone()['lot_id'] == actual['lot_id']
    db_cursor.execute('SELECT count(*) AS n FROM transaction_lot_confirmations WHERE transaction_id=%s', (result['transaction_id'],))
    assert db_cursor.fetchone()['n'] == 0
    assert commit(client, prepared, key).json() == result | {'replayed': True}


@pytest.mark.parametrize('language', ['en', 'es'])
def test_shift_summary_uses_actual_exception_lot_not_fifo_suggestion(client, db_cursor, items, actors, language):
    key = actors['floor']['key']
    prepared = prepare(client, 'make', body('make', items), key)
    result = commit(client, prepared, key,
                    lot_confirmations=[evidence(items['ingredient'], 'different_lot', 'TAG-7654')])
    assert result.status_code == 200, result.text
    result = result.json()
    response = client.get('/reports/shift-summary', params={'lang': language}, headers=headers(key))
    assert response.status_code == 200, response.text
    made = next(s for s in response.json()['sections'] if s['key'] == 'made')
    [row] = made['rows']
    [ingredient] = row['detail']['ingredients']
    assert ingredient['lot']['id'] == result['input_plan'][0]['lot_id']
    assert ingredient['lot']['id'] != items['ingredient']['lot_id']
    assert ingredient['confirmation'] is None and ingredient['short_lb'] == 10
    assert result['input_plan'][0]['lot_code'] in row['line']
    assert items['ingredient']['lot_code'] not in row['line']
    db_cursor.execute('SELECT ingredient_lot_id FROM ingredient_lot_consumption WHERE transaction_id=%s', (result['transaction_id'],))
    assert db_cursor.fetchone()['ingredient_lot_id'] == ingredient['lot']['id']
    db_cursor.execute("SELECT l.lot_id FROM trace_event_lots l JOIN trace_events e ON e.id=l.trace_event_id WHERE e.transaction_id=%s AND l.role='input'", (result['transaction_id'],))
    assert db_cursor.fetchone()['lot_id'] == ingredient['lot']['id']
