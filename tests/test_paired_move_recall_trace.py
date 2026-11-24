"""Owner rule (2026-11-24): a paired wrong-lot move shows BOTH lots in every recall trace.

Real HTTP against the local rollback-only ledger. Lot A is received and feeds a batch;
lot B is a second delivery of the same ingredient; a wrong-lot correction then says
10 lb recorded on A was really taken from B. Every trace must connect the two lots,
labeled, in both directions. Also the two leftovers from the PR #232 review.
"""
from uuid import uuid4

import pytest

import main
import paired_trace
from tests.test_actor_attribution import client, actors
from tests.test_write_tickets_part2 import items, prepare, body, isolated_database
from tests.test_write_tickets import headers, commit, error, isolated_database as a1_isolated_database
from tests.test_exceptions_enforcement import adjust
from tests.test_no_stock_by_hand import rules, received, ROOT


def _make_from(client, cur, items, actors, lot_id):
    """One 10 lb make that consumes the named ingredient lot; returns (transaction id, batch lot id, batch lot code)."""
    data = body('make', items) | {'ingredient_lots': [{'ingredient_product_id': items['ingredient']['id'], 'lot_id': lot_id}]}
    posted = commit(client, prepare(client, 'make', data, actors['floor']['key']), actors['floor']['key'])
    assert posted.status_code == 200, posted.text
    txn = posted.json()['transaction_id']
    cur.execute("""SELECT l.id, l.lot_code FROM transaction_lines i JOIN lots l ON l.id=i.lot_id
                   WHERE i.transaction_id=%s AND i.quantity_lb>0""", (txn,))
    out = cur.fetchone()
    return txn, out['id'], out['lot_code']


def _second_delivery(client, cur, items, actors):
    """Another delivery of the same ingredient on the same day: its own lot, duplicate warning acknowledged."""
    cur.execute('INSERT INTO suppliers(name) VALUES (%s) RETURNING id', ('Second delivery ' + uuid4().hex,))
    payload = dict(product_id=items['ingredient']['id'], supplier_id=cur.fetchone()['id'], cases=3, case_size_lb=40,
                   bol_reference='DEL-' + uuid4().hex, supplier_lot_code='SUP-02', occurred_at=main.get_plant_now().isoformat())
    key = actors['floor']['key']
    prepared = prepare(client, 'receive', payload, key)
    posted = commit(client, prepared, key, acknowledged_warnings=[w['code'] for w in prepared.get('warnings') or []])
    assert posted.status_code == 200, posted.text
    return dict(id=posted.json()['lot_id'], code=posted.json()['lot_code'])


@pytest.fixture
def scene(client, db_cursor, items, actors, rules):
    """A (received, feeds one batch), B (second delivery, feeds nothing), not yet paired."""
    received(client, db_cursor, items, actors, 50, 4)
    a = dict(id=items['ingredient']['lot_id'], code=items['ingredient']['lot_code'])
    make_txn, batch_id, batch_code = _make_from(client, db_cursor, items, actors, a['id'])
    b = _second_delivery(client, db_cursor, items, actors)
    assert a['id'] != b['id']
    db_cursor.execute("UPDATE lots SET supplier_lot_code='SUP-B-ONLY' WHERE id=%s", (b['id'],))
    items['ingredient'].update(lot_id=a['id'], lot_code=a['code'])   # adjust() targets A, the lot to restore
    return dict(a=a, b=b, make_txn=make_txn, batch=dict(id=batch_id, code=batch_code),
                product_id=items['ingredient']['id'], batch_product_id=items['batch']['id'],
                key=actors['owner']['key'], floor=actors['floor']['key'])


def _pair(client, items, scene, pounds=10):
    """Wrong-lot correction: `pounds` recorded on A were really taken from B."""
    key = scene['floor']
    move = commit(client, prepare(client, 'adjust', adjust(items, pounds, 'wrong_lot', note='Tag mixup at the mixer',
                                                           paired_lot_id=scene['b']['id']), key), key)
    assert move.status_code == 200, move.text
    return move.json()


def _report(client, scene, lot, direction='forward', product_id=None):
    response = client.get('/dashboard/api/lot-report/' + lot['code'],
                          params={'product_id': product_id or scene['product_id'], 'direction': direction},
                          headers=headers(scene['key']))
    assert response.status_code == 200, response.text
    return response.json()


def _ids(report):
    return [n['lot_id'] for n in report['forward_trace']['lots']]


# ── Recall report (forward exposure) ──

def test_before_any_correction_the_recall_of_b_reaches_nothing_else(client, scene):
    report = _report(client, scene, scene['b'])
    assert _ids(report) == [scene['b']['id']]
    assert report['paired_moves'] == [] and report['forward_trace']['paired_moves'] == []
    assert report['forward_trace']['consumption'] == []
    assert 'wrong-lot correction' not in report['forward_trace']['quantity_basis']


def test_recall_of_the_lot_really_used_reaches_the_batch_recorded_on_the_other_lot(client, db_cursor, items, scene):
    move = _pair(client, items, scene)
    report = _report(client, scene, scene['b'])
    lots = {n['lot_id']: n for n in report['forward_trace']['lots']}
    assert set(lots) == {scene['b']['id'], scene['a']['id'], scene['batch']['id']}
    assert lots[scene['b']['id']]['reached_via'] == 'start'
    assert lots[scene['a']['id']]['reached_via'] == 'paired_move'
    assert lots[scene['batch']['id']]['reached_via'] == 'consumption'
    assert [(e['source_lot_id'], e['lot_id'], e['transaction_id']) for e in report['forward_trace']['consumption']] == [
        (scene['a']['id'], scene['batch']['id'], scene['make_txn'])]
    [shown] = report['paired_moves']
    assert shown['transaction_id'] == move['transaction_id'] and shown['receipt_number'] == move['receipt_number']
    assert (shown['restored_lot_id'], shown['used_lot_id'], shown['quantity_lb']) == (scene['a']['id'], scene['b']['id'], 10.0)
    assert scene['a']['code'] in shown['label'] and scene['b']['code'] in shown['label'] and 'Both lots are included' in shown['label']
    assert scene['a']['code'] in shown['label_es'] and 'Se incluyen ambos lotes' in shown['label_es']
    assert shown['basis'] == 'assumed_link_not_physically_verified'
    assert 'assumed link, not a verified one' in report['forward_trace']['quantity_basis']
    # Each of the two lots names the other one.
    assert lots[scene['b']['id']]['paired_moves'][0]['other_lot'] == dict(
        lot_id=scene['a']['id'], lot_code=scene['a']['code'], supplier_lot_code='SUP-01', role='first_recorded')
    assert lots[scene['a']['id']]['paired_moves'][0]['other_lot']['role'] == 'corrected_to'
    assert 'paired_moves' not in lots[scene['batch']['id']]


def test_recall_of_the_lot_first_recorded_also_shows_the_lot_it_was_corrected_to(client, items, scene):
    _pair(client, items, scene)
    report = _report(client, scene, scene['a'])
    lots = {n['lot_id']: n for n in report['forward_trace']['lots']}
    assert set(lots) == {scene['a']['id'], scene['b']['id'], scene['batch']['id']}
    assert lots[scene['b']['id']]['reached_via'] == 'paired_move'
    assert len(report['paired_moves']) == 1


def test_a_voided_correction_links_nothing(client, items, scene):
    move = _pair(client, items, scene)
    void = client.post(f"/void/{move['transaction_id']}", json={'reason': 'data_entry_error'}, headers=headers(scene['key']))
    assert void.status_code == 200, void.text
    report = _report(client, scene, scene['b'])
    assert _ids(report) == [scene['b']['id']] and report['paired_moves'] == []


def test_link_survives_taking_the_098_marker_down(client, db_cursor, items, scene):
    _pair(client, items, scene)
    db_cursor.execute((ROOT / 'migrations/down/098_no_stock_by_hand_down.sql').read_text())
    report = _report(client, scene, scene['b'])
    assert set(_ids(report)) == {scene['b']['id'], scene['a']['id'], scene['batch']['id']}
    assert len(report['paired_moves']) == 1


def test_before_migration_098_every_trace_answers_with_no_links(db_cursor):
    assert paired_trace.available(db_cursor) is False
    assert paired_trace.moves(db_cursor, main) == []


def test_a_chain_of_corrections_links_every_lot_in_it():
    moves = [dict(transaction_id=1, restored_lot_id=1, restored_lot_code='A', restored_supplier_lot_code=None,
                  used_lot_id=2, used_lot_code='B', used_supplier_lot_code=None),
             dict(transaction_id=2, restored_lot_id=2, restored_lot_code='B', restored_supplier_lot_code=None,
                  used_lot_id=3, used_lot_code='C', used_supplier_lot_code=None)]
    by_lot = paired_trace.links(moves)
    assert paired_trace.linked_lot_ids(by_lot, 1) == [2, 3]
    assert paired_trace.linked_lot_ids(by_lot, 3) == [2, 1]
    assert paired_trace.linked_lot_ids(by_lot, 9) == []


# ── Backward trace of the batch ──

def test_backward_trace_of_the_batch_names_both_lots(client, items, scene):
    move = _pair(client, items, scene)
    report = _report(client, scene, scene['batch'], 'backward', scene['batch_product_id'])
    inputs = [i for t in report['input_tree']['transactions'] for i in t['inputs']]
    [a_input] = [i for i in inputs if i['lot_id'] == scene['a']['id']]
    assert a_input['paired_moves'][0]['transaction_id'] == move['transaction_id']
    assert a_input['paired_moves'][0]['other_lot']['lot_id'] == scene['b']['id']
    [other] = a_input['paired_lots']
    assert (other['lot_id'], other['lot_code'], other['supplier_lot_code']) == (scene['b']['id'], scene['b']['code'], 'SUP-B-ONLY')
    assert other['origin']['kind'] == 'received' and other['origin']['receipt_number']
    assert [m['transaction_id'] for m in report['paired_moves']] == [move['transaction_id']]


def test_backward_trace_without_a_correction_is_unchanged(client, scene):
    report = _report(client, scene, scene['batch'], 'backward', scene['batch_product_id'])
    inputs = [i for t in report['input_tree']['transactions'] for i in t['inputs']]
    assert all('paired_moves' not in i and 'paired_lots' not in i for i in inputs)
    assert report['paired_moves'] == []


# ── Older trace lookups ──

def test_trace_batch_lists_the_other_lot_beside_the_ingredient(client, items, scene):
    move = _pair(client, items, scene)
    response = client.get('/trace/batch/' + scene['batch']['code'], params={'product_id': scene['batch_product_id']},
                          headers=headers(scene['key']))
    assert response.status_code == 200, response.text
    result = response.json()
    [ingredient] = [i for i in result['ingredients'] if i['lot_id'] == scene['a']['id']]
    assert ingredient['paired_moves'][0]['transaction_id'] == move['transaction_id']
    [other] = result['paired_lots']
    assert other['lot_id'] == scene['b']['id'] and other['supplier_lot_code'] == 'SUP-B-ONLY'
    assert other['supplier']['bol_reference'] and other['paired_moves'][0]['other_lot']['lot_id'] == scene['a']['id']
    tree_inputs = [i for t in result['input_tree']['transactions'] for i in t['inputs']]
    assert [i['paired_lots'][0]['lot_id'] for i in tree_inputs if i['lot_id'] == scene['a']['id']] == [scene['b']['id']]


def test_trace_ingredient_of_the_lot_really_used_shows_the_batch_through_the_other_lot(client, items, scene):
    _pair(client, items, scene)
    response = client.get('/trace/ingredient/' + scene['b']['code'], params={'product_id': scene['product_id']},
                          headers=headers(scene['key']))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['used_in_batches'] == []          # nothing was recorded against B itself
    assert len(result['paired_moves']) == 1
    [other] = result['paired_lots']
    assert other['lot_id'] == scene['a']['id']
    assert [b['lot_id'] for b in other['downstream_batches']] == [scene['batch']['id']]


def test_trace_batch_on_an_ingredient_lot_shows_both_lots(client, items, scene):
    _pair(client, items, scene)
    response = client.get('/trace/batch/' + scene['b']['code'], params={'product_id': scene['product_id']},
                          headers=headers(scene['key']))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['trace_type'] == 'ingredient' and result['downstream_batches'] == []
    assert [p['lot_id'] for p in result['paired_lots']] == [scene['a']['id']]
    assert [b['lot_id'] for b in result['paired_lots'][0]['downstream_batches']] == [scene['batch']['id']]


def test_trace_supplier_lot_recall_includes_the_linked_lot_outside_the_totals(client, items, scene):
    move = _pair(client, items, scene)
    response = client.get('/trace/supplier-lot/SUP-B-ONLY', headers=headers(scene['key']))
    assert response.status_code == 200, response.text
    result = response.json()
    [matched] = result['matched_internal_lots']
    assert matched['lot_id'] == scene['b']['id'] and matched['paired_moves'][0]['transaction_id'] == move['transaction_id']
    assert matched['production_usage'] == [] and result['total_exposure_summary']['total_in_production_lb'] == 0
    [other] = result['paired_lots']
    assert other['lot_id'] == scene['a']['id']
    assert [b['lot_id'] for b in other['downstream_batches']] == [scene['batch']['id']]


def test_older_lookups_add_nothing_when_there_is_no_correction(client, scene):
    for path, params in [('/trace/batch/' + scene['batch']['code'], {'product_id': scene['batch_product_id']}),
                         ('/trace/ingredient/' + scene['a']['code'], {'product_id': scene['product_id']}),
                         ('/trace/supplier-lot/SUP-B-ONLY', {})]:
        response = client.get(path, params=params, headers=headers(scene['key']))
        assert response.status_code == 200, response.text
        assert 'paired_moves' not in response.json() and 'paired_lots' not in response.json(), path


# ── PR #232 review leftovers ──

def test_direct_adjust_preview_refuses_what_the_commit_refuses(client, db_cursor, items, actors, rules):
    received(client, db_cursor, items, actors)
    key = actors['owner']['key']
    data = dict(product_name=items['ingredient']['name'], lot_code=items['ingredient']['lot_code'],
                adjustment_lb=25, reason='data_entry_error', note='typo')
    db_cursor.execute('SELECT count(*) AS n FROM transactions'); before = db_cursor.fetchone()['n']
    for path, mode in [('/adjust', 'preview'), ('/adjust/preview', 'commit')]:   # the shortcut forces preview
        response = client.post(path, json=data | {'mode': mode}, headers=headers(key))
        error(response, 422, 'NO_STOCK_BY_HAND')
        assert response.json()['detail']['message'] and response.json()['detail']['message_es']
    error(client.post('/adjust', json=data | {'mode': 'commit'}, headers=headers(key)), 422, 'NO_STOCK_BY_HAND')
    # A removal is still previewed normally.
    removal = client.post('/adjust', json=data | {'mode': 'preview', 'adjustment_lb': -5, 'reason': 'damaged_thrown_out'},
                          headers=headers(key))
    assert removal.status_code == 200, removal.text
    assert removal.json()['new_balance_lb'] == 45
    db_cursor.execute('SELECT count(*) AS n FROM transactions'); assert db_cursor.fetchone()['n'] == before


def test_direct_adjust_preview_is_unchanged_before_098(client, db_cursor, items, actors):
    key = actors['owner']['key']
    data = dict(mode='preview', product_name=items['ingredient']['name'], lot_code=items['ingredient']['lot_code'],
                adjustment_lb=25, reason='count_correction')
    response = client.post('/adjust', json=data, headers=headers(key))
    assert response.status_code == 200, response.text
    assert response.json()['new_balance_lb'] == 125


@pytest.mark.parametrize('kind', ['unknown', 'inactive'])
@pytest.mark.parametrize('path,mode', [('/receive', 'preview'), ('/receive', 'commit'),
                                       ('/receive/preview', 'commit'), ('/receive/commit', 'preview')])
def test_direct_receive_refuses_unregistered_primary_supplier(client, db_cursor, items, actors, rules, kind, path, mode):
    name = 'Primary ' + uuid4().hex
    if kind == 'inactive':
        db_cursor.execute('INSERT INTO suppliers(name,active) VALUES (%s,false)', (name,))
    db_cursor.execute('SELECT count(*) AS n FROM transactions'); before = db_cursor.fetchone()['n']
    data = dict(mode=mode, product_name=items['ingredient']['name'], shipper_name=name, cases=1, case_size_lb=50,
                bol_reference='TEST', supplier_lot_code='SUP-1')
    error(client.post(path, json=data, headers=headers(actors['owner']['key'])), 422, 'SUPPLIER_REQUIRED')
    db_cursor.execute('SELECT count(*) AS n FROM transactions'); assert db_cursor.fetchone()['n'] == before


def test_direct_receive_posts_for_a_registered_primary_supplier(client, db_cursor, items, actors, rules):
    name = 'Primary ' + uuid4().hex
    db_cursor.execute('INSERT INTO suppliers(name) VALUES (%s)', (name,))
    data = dict(mode='commit', product_name=items['ingredient']['name'], shipper_name=name, cases=1, case_size_lb=50,
                bol_reference='TEST-' + uuid4().hex[:6], supplier_lot_code='SUP-1')
    response = client.post('/receive', json=data, headers=headers(actors['owner']['key']))
    assert response.status_code == 200, response.text
    assert response.json().get('lot_id') or response.json().get('lot_code')
