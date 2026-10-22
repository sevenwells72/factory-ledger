"""Owner policy: correct the selected lot when total positive stock covers need."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import psycopg2
from psycopg2.extras import RealDictCursor
import pytest

import main
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import enabled, reason_catalog, chat, model  # noqa: F401
from tests.test_supplier_lot_clients import make_card
from tests.test_supplier_lot_confirmation import stocked_lot, allocation, supplier_evidence
from tests.test_lot_confirmation import prepare, evidence
from tests.test_floor_supplier_lot_confirmation import floor_body
from tests.test_write_tickets import headers, commit, posted_count, error
from tests.test_write_tickets import isolated_database  # noqa: F401
from tests.test_write_tickets_part2 import items, body, seed  # noqa: F401

pytestmark = pytest.mark.db


def balance(cur, item, pounds):
    before = float(main.lot_on_hand(cur, item['lot_id']))
    cur.execute("INSERT INTO transactions(type) VALUES ('adjust') RETURNING id")
    tid = cur.fetchone()['id']
    cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,%s)',
                (tid, item['id'], item['lot_id'], pounds - before))


def choice(item, pounds=10):
    return {'ingredient_lot_allocations': [allocation(item['id'], item['lot_id'], pounds)],
            'lot_confirmations': [evidence(item, 'full_code')]}


@pytest.mark.parametrize('action', ['make', 'pack'])
def test_wrong_lot_overage_refuses_when_positive_product_stock_covers(client, db_cursor, items, action):
    item = items['ingredient' if action == 'make' else 'batch']
    balance(db_cursor, item, 2)
    second = stocked_lot(db_cursor, item['id'], 'ENOUGH-OTHER-LOT', pounds=8)
    stocked_lot(db_cursor, item['id'], 'NEGATIVE-NOT-POSITIVE-STOCK', pounds=-20)
    prepared = prepare(client, action, body(action, items) | choice(item))
    assert not prepared['can_commit']
    blocker = prepared['blockers'][0]
    assert blocker['code'] == 'LOT_ALLOCATION_EXCEEDS_STOCK'
    assert 'Pick the right lot or split' in blocker['message']
    assert 'Elija el lote correcto o divida' in blocker['message_es']
    group = prepared['draft']['lot_confirmation_group']['ingredients'][0]
    assert group['total_available_lb'] == 10 and group['stock_shortage'] is False
    assert {row['lot_id'] for row in group['candidates']} == {item['lot_id'], second}
    error(commit(client, prepared), 422, 'LOT_ALLOCATION_EXCEEDS_STOCK')
    assert posted_count(db_cursor, prepared) == 0


@pytest.mark.parametrize('action', ['make', 'pack'])
def test_right_split_posts_without_manufacturing_shortage(client, db_cursor, items, action):
    item = items['ingredient' if action == 'make' else 'batch']
    balance(db_cursor, item, 2)
    second = stocked_lot(db_cursor, item['id'], 'RIGHT-SPLIT-SECOND', pounds=8)
    prepared = prepare(client, action, body(action, items) | {'ingredient_lot_allocations': [
        allocation(item['id'], item['lot_id'], 2), allocation(item['id'], second, 8)]})
    result = commit(client, prepared, lot_confirmations=[evidence(row, 'full_code') for row in prepared['draft']['input_plan']])
    assert result.status_code == 200, result.text
    assert result.json()['shortages'] == []
    assert main.lot_on_hand(db_cursor, item['lot_id']) == 0
    assert main.lot_on_hand(db_cursor, second) == 0


@pytest.mark.parametrize('action', ['make', 'pack'])
def test_real_product_shortage_keeps_post_and_flag_policy(client, db_cursor, items, action):
    item = items['ingredient' if action == 'make' else 'batch']
    balance(db_cursor, item, 2)
    stocked_lot(db_cursor, item['id'], 'STILL-NOT-ENOUGH', pounds=3)
    prepared = prepare(client, action, body(action, items) | choice(item))
    assert prepared['can_commit']
    group = prepared['draft']['lot_confirmation_group']['ingredients'][0]
    assert group['total_available_lb'] == 5 and group['stock_shortage'] is True
    result = commit(client, prepared)
    assert result.status_code == 200, result.text
    assert result.json()['shortages'][0]['short_lb'] == 8
    assert main.lot_on_hand(db_cursor, item['lot_id']) == -8


@pytest.mark.parametrize('action', ['make', 'pack'])
@pytest.mark.parametrize('total_now_short', [False, True])
def test_frozen_plan_rechecks_stock_at_commit(client, db_cursor, items, action, total_now_short):
    item = items['ingredient' if action == 'make' else 'batch']
    stocked_lot(db_cursor, item['id'], 'COMMIT-OTHER-STOCK', pounds=3 if total_now_short else 10)
    prepared = prepare(client, action, body(action, items) | choice(item))
    assert prepared['can_commit']
    balance(db_cursor, item, 2)
    result = commit(client, prepared)
    if total_now_short:
        assert result.status_code == 200, result.text
        assert result.json()['shortages'][0]['short_lb'] == 8
    else:
        error(result, 422, 'LOT_ALLOCATION_EXCEEDS_STOCK')
        assert posted_count(db_cursor, prepared) == 0
        assert main.lot_on_hand(db_cursor, item['lot_id']) == 2


def test_new_stock_between_prepare_and_commit_requires_correct_lot(client, db_cursor, items):
    item = items['ingredient']
    balance(db_cursor, item, 2)
    prepared = prepare(client, 'make', body('make', items) | choice(item))
    assert prepared['can_commit']
    stocked_lot(db_cursor, item['id'], 'ARRIVED-AFTER-PREPARE', pounds=10)
    error(commit(client, prepared), 422, 'LOT_ALLOCATION_EXCEEDS_STOCK')
    assert posted_count(db_cursor, prepared) == 0


def test_pack_addin_uses_the_same_product_stock_policy(client, db_cursor, items):
    source, ingredient = items['batch'], items['ingredient']
    db_cursor.execute("INSERT INTO products(name,type,uom) VALUES ('Review stock intermediate','batch','lb') RETURNING id")
    intermediate = db_cursor.fetchone()['id']
    db_cursor.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=%s', (intermediate, items['finished']['id']))
    for pid, pounds in [(source['id'], 10), (ingredient['id'], 2)]:
        db_cursor.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES (%s,%s,%s)',
                          (intermediate, pid, pounds))
    balance(db_cursor, ingredient, .5)
    second = stocked_lot(db_cursor, ingredient['id'], 'ADDIN-OTHER-ENOUGH', pounds=1.5)
    payload = body('pack', items) | {'ingredient_lot_allocations': [allocation(ingredient['id'], ingredient['lot_id'], 2)]}
    refused = prepare(client, 'pack', payload)
    assert refused['blockers'][0]['code'] == 'LOT_ALLOCATION_EXCEEDS_STOCK'
    accepted = prepare(client, 'pack', payload | {'ingredient_lot_allocations': [
        allocation(ingredient['id'], ingredient['lot_id'], .5), allocation(ingredient['id'], second, 1.5)]})
    result = commit(client, accepted, lot_confirmations=[evidence(row, 'full_code') for row in accepted['draft']['input_plan']])
    assert result.status_code == 200, result.text
    assert result.json()['shortages'] == []


def test_extra_pack_shortage_counts_only_eligible_source_stock(client, db_cursor, items):
    source = items['batch']
    balance(db_cursor, source, 8)
    db_cursor.execute("UPDATE products SET kosher_tier='extra_kosher' WHERE id IN (%s,%s)",
                      (source['id'], items['finished']['id']))
    extra = stocked_lot(db_cursor, source['id'], 'ELIGIBLE-EXTRA-ONLY-2', pounds=2)
    prepared = prepare(client, 'pack', body('pack', items) | {
        'ingredient_lot_allocations': [allocation(source['id'], extra, 10)]})
    group = prepared['draft']['lot_confirmation_group']['ingredients'][0]
    assert group['total_available_lb'] == 2 and group['stock_shortage'] is True
    assert [row['lot_id'] for row in group['candidates']] == [extra]
    result = commit(client, prepared, lot_confirmations=[evidence(row, 'full_code') for row in prepared['draft']['input_plan']])
    assert result.status_code == 200, result.text
    assert result.json()['kosher_tier'] == 'extra_kosher'
    assert result.json()['shortages'][0]['short_lb'] == 8
    assert main.lot_on_hand(db_cursor, source['lot_id']) == 8
    assert main.lot_on_hand(db_cursor, extra) == -8


def test_regular_pack_counts_extra_stock_and_keeps_downgrade_evidence(client, db_cursor, items):
    source = items['batch']
    balance(db_cursor, source, 2)
    db_cursor.execute("UPDATE products SET kosher_tier='extra_kosher' WHERE id=%s", (source['id'],))
    extra = stocked_lot(db_cursor, source['id'], 'ELIGIBLE-EXTRA-FOR-REGULAR', pounds=8)
    db_cursor.execute("UPDATE products SET kosher_tier='regular' WHERE id=%s", (source['id'],))
    refused = prepare(client, 'pack', body('pack', items) | choice(source))
    group = refused['draft']['lot_confirmation_group']['ingredients'][0]
    assert group['total_available_lb'] == 10 and group['stock_shortage'] is False
    assert refused['blockers'][0]['code'] == 'LOT_ALLOCATION_EXCEEDS_STOCK'
    prepared = prepare(client, 'pack', body('pack', items) | {'ingredient_lot_allocations': [
        allocation(source['id'], source['lot_id'], 2), allocation(source['id'], extra, 8)]})
    result = commit(client, prepared, lot_confirmations=[evidence(row, 'full_code') for row in prepared['draft']['input_plan']])
    assert result.status_code == 200, result.text
    assert result.json()['shortages'] == []
    assert result.json()['transaction_kosher_downgrade'] is True


def test_make_substitute_and_pack_lock_unused_older_lot_in_same_order(isolated_database, monkeypatch):
    """Force the former Make(seed100)->old1 / Pack(old1)->seed100 cycle."""
    import lot_confirmation
    import ticket_actions
    import write_tickets

    with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        products = seed(cur)
        cur.execute("INSERT INTO products(name,type,uom,default_batch_lb,active) VALUES ('Lock-order replacement','batch','lb',10,true) RETURNING id")
        replacement = cur.fetchone()['id']
        older = stocked_lot(cur, replacement, 'LOCK-OLDER-UNUSED')
        newer = stocked_lot(cur, replacement, 'LOCK-NEWER-SEED')
        cur.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=%s', (replacement, products['finished']['id']))
    make_seed_read, pack_has_older, make_requests_all = Event(), Event(), Event()
    original = lot_confirmation.prepare_substitutions

    def pause_after_seed(*args, **kwargs):
        changes = original(*args, **kwargs)
        make_seed_read.set()
        assert pack_has_older.wait(5), 'Pack did not acquire the unused older lot.'
        return changes

    monkeypatch.setattr(lot_confirmation, 'prepare_substitutions', pause_after_seed)

    class CoordinatedCursor:
        def __init__(self, inner, action):
            self.inner, self.action = inner, action

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def execute(self, query, params=None):
            if query.startswith('SELECT id FROM lots WHERE product_id=ANY') and 'FOR UPDATE' in query:
                if self.action == 'pack':
                    self.inner.execute('SELECT id FROM lots WHERE id=%s FOR UPDATE', (older,))
                    self.inner.fetchall()
                    pack_has_older.set()
                    assert make_requests_all.wait(5), 'Make did not reach the ordered lock set.'
                else:
                    make_requests_all.set()
            return self.inner.execute(query, params)

    make_payload = write_tickets.MakePrepareRequest(product_id=products['batch']['id'], batches=1,
        occurred_at=main.get_plant_now(), substitutions=[{
            'ingredient_product_id': products['ingredient']['id'], 'substitute_product_id': replacement,
            'lot_id': newer, 'reason_code': 'ingredient_unavailable'}],
        ingredient_lot_allocations=[allocation(replacement, newer, 10)],
        lot_confirmations=[{'lot_id': newer, 'method': 'full_code', 'value': 'LOCK-NEWER-SEED'}]).model_dump(mode='json')
    pack_payload = write_tickets.PackPrepareRequest(source_product_id=replacement,
        target_product_id=products['finished']['id'], cases=2, occurred_at=main.get_plant_now(),
        ingredient_lot_allocations=[allocation(replacement, older, 10)],
        lot_confirmations=[{'lot_id': older, 'method': 'full_code', 'value': 'LOCK-OLDER-UNUSED'}]).model_dump(mode='json')

    def validate(action, payload):
        if action == 'pack':
            assert make_seed_read.wait(5)
        with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SET LOCAL lock_timeout='4s'")
            result = ticket_actions.validate(main, CoordinatedCursor(cur, action), action, payload, lock=True)
            return result[0]['input_plan']

    with ThreadPoolExecutor(max_workers=2) as executor:
        making = executor.submit(validate, 'make', make_payload)
        packing = executor.submit(validate, 'pack', pack_payload)
        assert [row['lot_id'] for row in making.result(timeout=12)] == [newer]
        assert [row['lot_id'] for row in packing.result(timeout=12)] == [older]


@pytest.mark.parametrize('action', ['make', 'pack'])
@pytest.mark.parametrize('enough_stock', [False, True])
def test_floor_checked_api_applies_conditional_stock_policy(client, db_cursor, items, monkeypatch, action, enough_stock):
    monkeypatch.setenv('REQUIRE_MAKE_PACK_LOT_EVIDENCE', '1')
    item = items['ingredient' if action == 'make' else 'batch']
    balance(db_cursor, item, 2)
    stocked_lot(db_cursor, item['id'], 'FLOOR-OTHER-STOCK', pounds=8 if enough_stock else 3)
    payload = floor_body(action, items) | choice(item)
    preview = client.post('/' + action, headers=headers(), json=payload)
    assert preview.status_code == 200, preview.text
    assert preview.json()['lot_confirmation_group']['ingredients'][0]['stock_shortage'] is not enough_stock
    result = client.post('/' + action, headers=headers(), json=payload | {'mode': 'commit'})
    if enough_stock:
        assert preview.json()['blockers'][0]['code'] == 'LOT_ALLOCATION_EXCEEDS_STOCK'
        error(result, 422, 'LOT_ALLOCATION_EXCEEDS_STOCK')
        assert main.lot_on_hand(db_cursor, item['lot_id']) == 2
    else:
        assert preview.json()['can_commit'] is True
        assert result.status_code == 200, result.text
        assert result.json()['shortages'][0]['short_lb'] == 8
        assert main.lot_on_hand(db_cursor, item['lot_id']) == -8


@pytest.mark.parametrize('action', ['make', 'pack'])
@pytest.mark.parametrize('enough_stock', [False, True])
def test_assistant_selection_applies_conditional_stock_policy(client, chat, items, db_cursor, monkeypatch, action, enough_stock):
    item = items['ingredient' if action == 'make' else 'batch']
    balance(db_cursor, item, 2)
    second = stocked_lot(db_cursor, item['id'], 'ASSISTANT-RIGHT-SPLIT', pounds=8 if enough_stock else 3)
    card = make_card(client, chat, items, db_cursor, monkeypatch, action)
    response = client.post('/assistant/select-lots', headers=chat[1], json={'draft_id': card['id'], **choice(item)})
    assert response.status_code == 200, response.text
    corrected = response.json()['card']
    if not enough_stock:
        assert corrected['prepared']['can_commit'] is True
        result = client.post('/assistant/record', headers=chat[1], json={'draft_id': corrected['id']})
        assert result.status_code == 200, result.text
        assert result.json()['result']['shortages'][0]['short_lb'] == 8
        return
    assert corrected['prepared']['blockers'][0]['code'] == 'LOT_ALLOCATION_EXCEEDS_STOCK'
    assert corrected['prepared']['draft']['lot_confirmation_group']['ingredients'][0]['stock_shortage'] is False
    rows = [allocation(item['id'], item['lot_id'], 2), allocation(item['id'], second, 8)]
    lot_codes = {row['lot_id']: row['lot_code'] for row in corrected['prepared']['draft']['lot_confirmation_group']['ingredients'][0]['candidates']}
    response = client.post('/assistant/select-lots', headers=chat[1], json={'draft_id': corrected['id'],
        'ingredient_lot_allocations': rows,
        'lot_confirmations': [{'lot_id': row['lot_id'], 'method': 'full_code', 'value': lot_codes[row['lot_id']]} for row in rows]})
    assert response.status_code == 200, response.text
    selected = response.json()['card']
    result = client.post('/assistant/record', headers=chat[1], json={'draft_id': selected['id']})
    assert result.status_code == 200, result.text
    assert result.json()['kind'] == 'receipt'
    assert result.json()['result']['shortages'] == []


@pytest.mark.parametrize('placeholder', ['UNKNOWN', ' n / a ', 'NA'])
def test_placeholder_supplier_codes_are_blank_but_real_aliases_survive(client, db_cursor, items, placeholder):
    item = items['ingredient']
    db_cursor.execute('UPDATE lots SET supplier_lot_code=%s WHERE id=%s', (placeholder, item['lot_id']))
    for code in ('UNKNOWN', 'N/A'):
        db_cursor.execute('INSERT INTO lot_supplier_codes(lot_id,supplier_lot_code) VALUES (%s,%s)', (item['lot_id'], code))
    prepared = prepare(client, 'make', body('make', items))
    candidate = prepared['draft']['lot_confirmation_group']['ingredients'][0]['candidates'][0]
    assert candidate['supplier_lot_code'] is None and candidate['supplier_lot_codes'] == []
    error(commit(client, prepared, lot_confirmations=[supplier_evidence(item['lot_id'], placeholder, selected=True)]),
          422, 'SUPPLIER_LOT_NOT_IN_STOCK')
    db_cursor.execute("INSERT INTO lot_supplier_codes(lot_id,supplier_lot_code) VALUES (%s,'REAL BAG 22')", (item['lot_id'],))
    prepared = prepare(client, 'make', body('make', items))
    candidate = prepared['draft']['lot_confirmation_group']['ingredients'][0]['candidates'][0]
    assert candidate['supplier_lot_code'] == 'REAL BAG 22' and candidate['supplier_lot_codes'] == ['REAL BAG 22']
    result = commit(client, prepared, lot_confirmations=[supplier_evidence(item['lot_id'], 'realbag22')])
    assert result.status_code == 200, result.text
