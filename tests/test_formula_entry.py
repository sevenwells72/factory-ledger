"""Formula-driven make entry (Phase A, docs/production/formula-driven-entry.md).

Real HTTP prepare/commit against the local test database: pans math, bags
pre-filled by the existing FIFO spill, edited/split pounds, fixed reasons for
not-used and swapped ingredients, the > 10% owner-review flag, short stock that
still saves, and nothing posted before the final Record.
"""
from datetime import timedelta
import json
import re
from pathlib import Path
from uuid import uuid4

import pytest

import assistant_tools
import formula_entry
import main
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_write_tickets import commit, headers, posted_count, ticket_row

pytestmark = pytest.mark.db
ROOT = Path(__file__).resolve().parents[1]


def product(cur, name, kind, batch_lb=None, code=None):
    cur.execute("""INSERT INTO products(id,name,odoo_code,type,uom,default_batch_lb,active)
                   SELECT COALESCE(max(id),0)+1,%s,%s,%s,'lb',%s,true FROM products RETURNING id""",
                (f'FE {name} {uuid4().hex[:6]}', code or f'FE-{uuid4().hex[:8]}', kind, batch_lb))
    return cur.fetchone()['id']


def recipe_line(cur, batch, ingredient, lb):
    cur.execute('''INSERT INTO batch_formulas(id,product_id,ingredient_product_id,quantity_lb)
                   SELECT COALESCE(max(id),0)+1,%s,%s,%s FROM batch_formulas''', (batch, ingredient, lb))


def bag(cur, pid, quantity, days_old):
    code = f'FE-{uuid4().hex[:8].upper()}'
    received = main.get_plant_now() - timedelta(days=days_old)
    cur.execute('INSERT INTO lots(product_id,lot_code,received_at) VALUES (%s,%s,%s) RETURNING id',
                (pid, code, received))
    lot_id = cur.fetchone()['id']
    cur.execute("INSERT INTO transactions(type,notes) VALUES ('receive','formula entry fixture') RETURNING id")
    tx = cur.fetchone()['id']
    cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,%s)',
                (tx, pid, lot_id, quantity))
    return {'id': lot_id, 'code': code}


@pytest.fixture
def granola(db_cursor):
    """100 lb pan: 80 lb oats + 20 lb honey. Oats: an old 50 lb bag, then a 500 lb bag."""
    batch = product(db_cursor, 'Granola', 'batch', 100)
    oats, honey, syrup = (product(db_cursor, n, 'ingredient') for n in ('Plain Oats', 'Honey', 'Syrup'))
    recipe_line(db_cursor, batch, oats, 80)
    recipe_line(db_cursor, batch, honey, 20)
    return {'batch': batch, 'oats': oats, 'honey': honey, 'syrup': syrup,
            'oats_old': bag(db_cursor, oats, 50, 30), 'oats_new': bag(db_cursor, oats, 500, 5),
            'oats_spare': bag(db_cursor, oats, 300, 1),
            'honey_bag': bag(db_cursor, honey, 100, 10), 'syrup_bag': bag(db_cursor, syrup, 100, 10)}


def body(g, pans=2, **changes):
    return {'product_id': g['batch'], 'batches': pans, 'occurred_at': main.get_plant_now().isoformat(),
            'confirmed_sku': True, 'formula_entry': True, **changes}


def prepare(client, actors, data, confirm=None):
    if confirm is not None:
        data = data | {'lot_confirmations': confirm}
    response = client.post('/make/prepare', json=data, headers=headers(actors['floor']['key']))
    assert response.status_code == 200, response.text
    return response.json()


def confirmations(prepared):
    return [{'lot_id': row['lot_id'], 'method': 'full_code', 'value': row['lot_code']}
            for row in prepared['draft']['input_plan']]


def confirmed(client, actors, data):
    first = prepare(client, actors, data)
    return prepare(client, actors, data, confirmations(first))


def record(client, actors, prepared):
    return commit(client, prepared, actors['floor']['key'],
                  acknowledged_warnings=[w['code'] for w in prepared['warnings']])


def card(prepared, pid):
    return next(i for i in prepared['draft']['formula_entry']['ingredients'] if i['product_id'] == pid)


def line(pid, *bags, reason_code=None, note=None):
    return {'ingredient_product_id': pid, 'bags': [{'lot_id': lot['id'], 'quantity_lb': lb} for lot, lb in bags],
            'reason_code': reason_code, 'note': note}


def consumed(cur, transaction_id):
    cur.execute('''SELECT lot_id,quantity_lb FROM transaction_lines
                   WHERE transaction_id=%s AND quantity_lb<0 ORDER BY lot_id''', (transaction_id,))
    return {r['lot_id']: -float(r['quantity_lb']) for r in cur.fetchall()}


def output(cur, transaction_id):
    cur.execute('SELECT sum(quantity_lb) AS lb FROM transaction_lines WHERE transaction_id=%s AND quantity_lb>0',
                (transaction_id,))
    return float(cur.fetchone()['lb'])


def test_pans_multiply_the_recipe_and_suggest_one_lot_unconfirmed(client, db_cursor, actors, granola):
    prepared = prepare(client, actors, body(granola, pans=2))
    block = prepared['draft']['formula_entry']
    assert (block['pans'], block['pan_lb'], block['output_lb']) == (2, 100.0, 200.0)
    oats, honey = card(prepared, granola['oats']), card(prepared, granola['honey'])
    assert (oats['standard_lb'], honey['standard_lb']) == (160.0, 40.0)
    # PR-1 (make-entry simplification): ONE suggested lot, the oldest with stock, for the
    # whole amount. FL never spills into the newer lot on book stock alone; the
    # uncovered 110 lb is a shortage on that lot unless the worker adds a real second lot.
    assert [(b['lot_id'], b['quantity_lb']) for b in oats['bags']] == [(granola['oats_old']['id'], 160.0)]
    assert [(s['lot_id'], s['short_lb']) for s in prepared['draft']['shortages']] == [(granola['oats_old']['id'], 110.0)]
    assert 'WILL_CREATE_SHORTAGE' in {w['code'] for w in prepared['warnings']}
    assert [b['lot_id'] for b in oats['candidates']] == [
        granola['oats_old']['id'], granola['oats_new']['id'], granola['oats_spare']['id']]
    assert not any(b['confirmed'] for i in block['ingredients'] for b in i['bags'])
    assert prepared['can_commit'] is False
    assert {b['code'] for b in prepared['blockers']} == {'LOT_NOT_CONFIRMED'}
    assert [r['code'] for r in block['reasons']] == list(formula_entry.CODES)
    assert posted_count(db_cursor, prepared) == 0


def test_confirmed_standard_run_posts_one_record_for_all_pans(client, db_cursor, actors, granola):
    prepared = confirmed(client, actors, body(granola, pans=2))
    assert prepared['can_commit'] is True, prepared['blockers']
    assert prepared['draft']['formula_entry']['review'] == []
    response = record(client, actors, prepared)
    assert response.status_code == 200, response.text
    result = response.json()
    assert output(db_cursor, result['transaction_id']) == 200.0
    assert consumed(db_cursor, result['transaction_id']) == {
        granola['oats_old']['id']: 160.0, granola['honey_bag']['id']: 40.0}
    # Existing shortage policy (R3): the suggested lot goes negative and is flagged.
    assert [(s['lot_id'], s['short_lb']) for s in result['shortages']] == [(granola['oats_old']['id'], 110.0)]
    assert result['ingredient_review'] == []
    db_cursor.execute("SELECT count(*) AS n FROM transactions WHERE ticket_id=%s AND type='make'",
                      (prepared['ticket_id'],))
    assert db_cursor.fetchone()['n'] == 1


def test_bag_split_posts_exact_pounds_per_bag(client, db_cursor, actors, granola):
    lines = [line(granola['oats'], (granola['oats_new'], 100), (granola['oats_spare'], 60)),
             line(granola['honey'], (granola['honey_bag'], 41))]
    prepared = confirmed(client, actors, body(granola, ingredient_lines=lines))
    assert prepared['can_commit'] is True, prepared['blockers']
    oats = card(prepared, granola['oats'])
    assert oats['entered_lb'] == 160.0 and not oats['over_tolerance']
    assert prepared['warnings'] == [] or all(w['code'] != formula_entry.WARNING_CODE for w in prepared['warnings'])
    result = record(client, actors, prepared).json()
    assert consumed(db_cursor, result['transaction_id']) == {
        granola['oats_new']['id']: 100.0, granola['oats_spare']['id']: 60.0, granola['honey_bag']['id']: 41.0}
    db_cursor.execute('SELECT lot_id FROM transaction_lot_confirmations WHERE transaction_id=%s ORDER BY lot_id',
                      (result['transaction_id'],))
    assert [r['lot_id'] for r in db_cursor.fetchall()] == sorted(
        [granola['oats_new']['id'], granola['oats_spare']['id'], granola['honey_bag']['id']])


def test_changing_pounds_keeps_confirmation_but_changing_the_bag_does_not(client, db_cursor, actors, granola):
    lines = [line(granola['oats'], (granola['oats_old'], 40), (granola['oats_new'], 120)),
             line(granola['honey'], (granola['honey_bag'], 40))]
    evidence = confirmations(confirmed(client, actors, body(granola, ingredient_lines=lines)))
    # Same bags, different pounds (D7): the typed confirmations still count.
    lines[0] = line(granola['oats'], (granola['oats_old'], 40), (granola['oats_new'], 125))
    again = prepare(client, actors, body(granola, ingredient_lines=lines), evidence)
    assert again['can_commit'] is True, again['blockers']
    # Swap the old bag for the spare bag: only the new bag needs confirming.
    lines[0] = line(granola['oats'], (granola['oats_spare'], 40), (granola['oats_new'], 125))
    kept = [c for c in evidence if c['lot_id'] != granola['oats_old']['id']]
    changed = prepare(client, actors, body(granola, ingredient_lines=lines), kept)
    assert changed['can_commit'] is False
    unconfirmed = [b for b in card(changed, granola['oats'])['bags'] if not b['confirmed']]
    assert [b['lot_id'] for b in unconfirmed] == [granola['oats_spare']['id']]
    assert posted_count(db_cursor, changed) == 0


@pytest.mark.parametrize('reason,note,blocker', [
    (None, None, 'MAKE_REASON_REQUIRED'), ('other', None, 'MAKE_REASON_NOTE_REQUIRED'),
    ('other', '  ', 'MAKE_REASON_NOTE_REQUIRED'), ('not_used', None, None), ('other', 'Honey jar cracked', None)])
def test_zero_ingredient_needs_a_fixed_reason(client, db_cursor, actors, granola, reason, note, blocker):
    lines = [line(granola['honey'], reason_code=reason, note=note)]
    prepared = confirmed(client, actors, body(granola, ingredient_lines=lines))
    honey = card(prepared, granola['honey'])
    assert honey['not_used'] and honey['bags'] == [] and honey['standard_lb'] == 40.0
    codes = {b['code'] for b in prepared['blockers']}
    if blocker:
        assert blocker in codes and prepared['can_commit'] is False
        assert record(client, actors, prepared).status_code != 200
        assert posted_count(db_cursor, prepared) == 0
        return
    assert prepared['can_commit'] is True, prepared['blockers']
    result = record(client, actors, prepared).json()
    assert granola['honey_bag']['id'] not in consumed(db_cursor, result['transaction_id'])
    [row] = [r for r in result['ingredient_review'] if r['product_id'] == granola['honey']]
    assert (row['kind'], row['reason_code'], row['used_lb'], row['standard_lb']) == ('not_used', reason, 0.0, 40.0)
    db_cursor.execute('''SELECT substitute_product_id,reason_code,note FROM transaction_substitutions
                         WHERE transaction_id=%s AND ingredient_product_id=%s''',
                      (result['transaction_id'], granola['honey']))
    saved = db_cursor.fetchone()
    assert (saved['substitute_product_id'], saved['reason_code'], saved['note']) == (None, reason, note)
    # The master recipe is never changed by an entry.
    db_cursor.execute('SELECT quantity_lb FROM batch_formulas WHERE product_id=%s AND ingredient_product_id=%s',
                      (granola['batch'], granola['honey']))
    assert float(db_cursor.fetchone()['quantity_lb']) == 20.0


def test_free_text_reason_is_refused_by_the_request(client, actors, granola):
    lines = [line(granola['honey'], reason_code='ran out')]
    response = client.post('/make/prepare', json=body(granola, ingredient_lines=lines),
                           headers=headers(actors['floor']['key']))
    assert response.status_code == 422


@pytest.mark.parametrize('reason,ok', [('physical_substitution', False), ('substituted', True)])
def test_swapped_ingredient_needs_a_fixed_reason(client, db_cursor, actors, granola, reason, ok):
    swap = [{'ingredient_product_id': granola['honey'], 'substitute_product_id': granola['syrup'],
             'lot_id': granola['syrup_bag']['id'], 'reason_code': reason}]
    prepared = confirmed(client, actors, body(granola, substitutions=swap))
    if not ok:
        assert 'MAKE_REASON_REQUIRED' in {b['code'] for b in prepared['blockers']}
        assert prepared['can_commit'] is False
        return
    assert prepared['can_commit'] is True, prepared['blockers']
    syrup = card(prepared, granola['syrup'])
    assert syrup['substituted_for'] == granola['honey'] and syrup['standard_lb'] == 40.0
    result = record(client, actors, prepared).json()
    [row] = result['ingredient_review']
    assert (row['kind'], row['reason_code'], row['substituted_for']) == ('substituted', 'substituted', granola['honey'])
    assert consumed(db_cursor, result['transaction_id'])[granola['syrup_bag']['id']] == 40.0


@pytest.mark.parametrize('oats_lb,flagged', [(176, False), (144, False), (176.5, True), (143, True), (200, True)])
def test_more_than_ten_percent_off_recipe_is_flagged_for_the_owner(client, db_cursor, actors, granola, oats_lb, flagged):
    lines = [line(granola['oats'], (granola['oats_new'], oats_lb))]
    prepared = confirmed(client, actors, body(granola, ingredient_lines=lines))
    assert prepared['can_commit'] is True, prepared['blockers']
    warning = [w for w in prepared['warnings'] if w['code'] == formula_entry.WARNING_CODE]
    assert card(prepared, granola['oats'])['over_tolerance'] is flagged
    if not flagged:
        assert warning == [] and prepared['draft']['formula_entry']['review'] == []
        return
    assert warning[0]['requires_ack'] is True and 'weekly review' in warning[0]['message']
    unacknowledged = commit(client, prepared, actors['floor']['key'],
                            acknowledged_warnings=[w['code'] for w in prepared['warnings']
                                                   if w['code'] != formula_entry.WARNING_CODE])
    assert unacknowledged.status_code == 409
    assert unacknowledged.json()['detail']['error_code'] == 'WARNING_NOT_ACKNOWLEDGED'
    assert posted_count(db_cursor, prepared) == 0
    result = record(client, actors, prepared).json()
    [row] = result['ingredient_review']
    assert (row['kind'], row['weekly_view'], row['used_lb']) == ('off_recipe', True, float(oats_lb))
    assert row['difference_pct'] == round((oats_lb - 160) / 160 * 100, 1)
    # Saved with the ticket receipt for the weekly view (no new table in Phase A).
    saved = ticket_row(db_cursor, prepared)['response']
    saved = saved if isinstance(saved, dict) else json.loads(saved)
    assert saved['ingredient_review'] == result['ingredient_review']


def test_short_stock_still_saves_and_flags_the_shortage(client, db_cursor, actors, granola):
    lines = [line(granola['honey'], (granola['honey_bag'], 130))]   # bag holds 100 lb
    prepared = confirmed(client, actors, body(granola, pans=2, ingredient_lines=lines))
    assert prepared['can_commit'] is True, prepared['blockers']
    assert [s['short_lb'] for s in prepared['draft']['shortages'] if s['product_id'] == granola['honey']] == [30.0]
    response = record(client, actors, prepared)
    assert response.status_code == 200, response.text
    result = response.json()
    assert sum(s['short_lb'] for s in result['shortages'] if s['product_id'] == granola['honey']) == 30.0
    assert float(main.lot_on_hand(db_cursor, granola['honey_bag']['id'])) == -30.0
    [row] = result['ingredient_review']
    assert row['kind'] == 'off_recipe' and row['product_id'] == granola['honey']


def test_lots_are_rechecked_at_record_and_nothing_posts_before(client, db_cursor, actors, granola):
    prepared = confirmed(client, actors, body(granola))
    assert prepared['can_commit'] is True
    assert posted_count(db_cursor, prepared) == 0
    db_cursor.execute("UPDATE lots SET status='merged' WHERE id=%s", (granola['oats_old']['id'],))
    response = record(client, actors, prepared)
    assert response.status_code == 409 or response.status_code == 422, response.text
    assert posted_count(db_cursor, prepared) == 0


def test_ingredient_lines_must_belong_to_the_recipe_and_bag(client, actors, granola):
    stray = prepare(client, actors, body(granola, ingredient_lines=[line(granola['syrup'], (granola['syrup_bag'], 5))]))
    assert stray['can_commit'] is False and stray['blockers'][0]['code'] == 'UNUSED_INGREDIENT_LINE'
    wrong = prepare(client, actors, body(granola, ingredient_lines=[line(granola['oats'], (granola['honey_bag'], 5))]))
    assert wrong['can_commit'] is False and wrong['blockers'][0]['code'] == 'LOT_NOT_FOUND'


def test_other_make_callers_keep_their_payload_and_draft(client, db_cursor, actors, granola):
    data = {k: v for k, v in body(granola).items() if k != 'formula_entry'}
    prepared = prepare(client, actors, data)
    assert 'formula_entry' not in prepared['draft']
    payload = ticket_row(db_cursor, prepared)['payload']
    payload = payload if isinstance(payload, dict) else json.loads(payload)
    assert 'formula_entry' not in payload and 'ingredient_lines' not in payload


def test_assistant_cannot_send_formula_entry_fields():
    schema = next(t for t in assistant_tools.tools([]) if t['name'] == 'prepare_make')['parameters']
    assert 'formula_entry' not in schema['properties'] and 'ingredient_lines' not in schema['properties']
    assert assistant_tools.has_ui_fields({'ingredient_lines': []})


def test_screen_reason_list_matches_the_server():
    source = (ROOT / 'dashboard/make-formula.js').read_text()
    listed = re.findall(r"\['(\w+)','[^']+','[^']+',(?:true|false)\]", source)
    assert tuple(listed) == formula_entry.CODES


# --- Review fixes (reviews/formula-entry-f8ba30d.md, Blubber 2026-11-02) ---------------

from tests.test_oat_bran_auto_substitute import BRAN, OATS, stock, sunshine  # noqa: E402,F401


def ss_body(**changes):
    # SS #1 (116): 28 lb oat bran + 185 lb rolled oats per pan (migration 081).
    return {'product_id': 116, 'batches': 1, 'occurred_at': main.get_plant_now().isoformat(),
            'confirmed_sku': True, 'formula_entry': True, **changes}


def bran_bag(item, lb):
    return {'id': item['lot_id']}, lb


def substitution_rows(cur, transaction_id):
    cur.execute('''SELECT ingredient_product_id,substitute_product_id,reason_code FROM transaction_substitutions
                   WHERE transaction_id=%s ORDER BY ingredient_product_id''', (transaction_id,))
    return [dict(r) for r in cur.fetchall()]


def test_finding1_bran_above_book_stock_posts_what_was_confirmed_and_flags_the_shortage(
        client, db_cursor, actors, sunshine):
    bran = stock(db_cursor, BRAN, 10)
    lines = [line(BRAN, bran_bag(bran, 28)), line(OATS, bran_bag(sunshine[OATS], 185))]
    prepared = confirmed(client, actors, ss_body(ingredient_lines=lines))
    assert prepared['can_commit'] is True, prepared['blockers']
    codes = {w['code'] for w in prepared['warnings']}
    assert 'WILL_CREATE_SHORTAGE' in codes and formula_entry.WARNING_CODE not in codes
    assert prepared['draft']['auto_substitutions'] == []      # the worker used the bran: no swap
    response = record(client, actors, prepared)
    assert response.status_code == 200, response.text          # Prepare and Record agree
    result = response.json()
    taken = consumed(db_cursor, result['transaction_id'])
    assert taken[bran['lot_id']] == 28.0 and taken[sunshine[OATS]['lot_id']] == 185.0
    assert [(s['lot_id'], s['short_lb']) for s in result['shortages']] == [(bran['lot_id'], 18.0)]
    assert float(main.lot_on_hand(db_cursor, bran['lot_id'])) == -18.0
    assert result['ingredient_review'] == [] and result['auto_substitutions'] == []
    assert substitution_rows(db_cursor, result['transaction_id']) == []


def test_finding2_exact_recipe_on_ss_product_gives_no_warning(client, db_cursor, actors, sunshine):
    bran = stock(db_cursor, BRAN, 100)
    prepared = confirmed(client, actors, ss_body())
    assert prepared['can_commit'] is True, prepared['blockers']
    assert prepared['warnings'] == []
    assert (card(prepared, BRAN)['standard_lb'], card(prepared, OATS)['standard_lb']) == (28.0, 185.0)
    assert prepared['draft']['formula_entry']['review'] == []
    result = record(client, actors, prepared).json()
    assert consumed(db_cursor, result['transaction_id'])[bran['lot_id']] == 28.0
    assert result['ingredient_review'] == []


def test_finding2_swap_cards_keep_the_recipe_and_check_bran_plus_swapped_oats(client, db_cursor, actors, sunshine):
    bran = stock(db_cursor, BRAN, 10)
    prepared = confirmed(client, actors, ss_body())
    assert prepared['can_commit'] is True, prepared['blockers']
    block = prepared['draft']['formula_entry']
    bran_card, oats_card = card(prepared, BRAN), card(prepared, OATS)
    # Recipe × pans, never the stock-adjusted split (spec 3.2).
    assert (bran_card['standard_lb'], bran_card['entered_lb'], bran_card['checked_lb']) == (28.0, 10.0, 28.0)
    assert (oats_card['standard_lb'], oats_card['entered_lb'], oats_card['checked_lb']) == (185.0, 203.0, 185.0)
    assert bran_card['swap_role'] == 'from' and oats_card['swap_role'] == 'to' and block['swap']['swap_lb'] == 18.0
    assert not bran_card['over_tolerance'] and not oats_card['over_tolerance']
    assert formula_entry.WARNING_CODE not in {w['code'] for w in prepared['warnings']}
    response = record(client, actors, prepared)
    assert response.status_code == 200, response.text
    result = response.json()
    [swap] = [r for r in result['ingredient_review'] if r['product_id'] == BRAN]
    assert (swap['kind'], swap['standard_lb'], swap['used_lb'], swap['swap_lb']) == ('swap', 28.0, 10.0, 18.0)
    db_cursor.execute("SELECT payload FROM write_tickets WHERE id=%s", (prepared['ticket_id'],))
    standard = db_cursor.fetchone()['payload']['specification']['standard_requirements']
    assert (standard[str(BRAN)], standard[str(OATS)]) == (28.0, 185.0)
    assert [r['reason_code'] for r in substitution_rows(db_cursor, result['transaction_id'])] == ['oat_bran_auto_substitute']


def test_finding2_swap_label_card_with_no_bran_in_stock(client, db_cursor, actors, sunshine):
    prepared = confirmed(client, actors, ss_body())
    bran_card = card(prepared, BRAN)
    assert (bran_card['standard_lb'], bran_card['entered_lb'], bran_card['checked_lb']) == (28.0, 0.0, 28.0)
    assert not bran_card['not_used'] and prepared['can_commit'] is True, prepared['blockers']
    # The Make screen sends every card, so the empty bran card comes back as a 0 lb line: still no reason needed.
    lines = [line(BRAN), line(OATS, bran_bag(sunshine[OATS], 213))]
    again = prepare(client, actors, ss_body(ingredient_lines=lines), confirmations(prepared))
    assert again['can_commit'] is True, again['blockers']
    result = record(client, actors, again).json()
    assert consumed(db_cursor, result['transaction_id'])[sunshine[OATS]['lot_id']] == 213.0
    assert [r['kind'] for r in result['ingredient_review']] == ['swap']


def test_finding4_bag_lots_are_locked_in_lot_id_order_at_record(client, db_cursor, actors, granola, monkeypatch):
    import ticket_actions
    calls, original = [], ticket_actions.lot

    def spy(api, cur, lot_id, product_id=None, lock=False):
        calls.append((lot_id, lock))
        return original(api, cur, lot_id, product_id, lock)
    lines = [line(granola['oats'], (granola['oats_spare'], 60), (granola['oats_new'], 100)),
             line(granola['honey'], (granola['honey_bag'], 40))]
    prepared = confirmed(client, actors, body(granola, ingredient_lines=lines))
    monkeypatch.setattr(ticket_actions, 'lot', spy)
    assert record(client, actors, prepared).status_code == 200
    plan = {granola['oats_spare']['id'], granola['oats_new']['id'], granola['honey_bag']['id']}
    locked = [lot_id for lot_id, lock in calls if lock and lot_id in plan]
    assert locked[:3] == sorted(plan)          # the first lock taken on any bag is in lot-ID order
    assert granola['oats_spare']['id'] > granola['oats_new']['id']   # screen order was the reverse


def shift_flags(client, actors, receipt):
    response = client.get('/reports/shift-summary', headers=headers(actors['floor']['key']))
    assert response.status_code == 200, response.text
    made = next(s for s in response.json()['sections'] if s['key'] == 'made')
    return next(r for r in made['rows'] if r['receipt_number'] == receipt)['flags']


def test_finding5_and_3_zero_lb_row_in_warnings_and_on_the_shift_summary(client, db_cursor, actors, granola):
    lines = [line(granola['honey'], reason_code='out_of_stock'), line(granola['oats'], (granola['oats_new'], 200))]
    prepared = confirmed(client, actors, body(granola, ingredient_lines=lines))
    assert prepared['can_commit'] is True, prepared['blockers']
    by_code = {w['code']: w for w in prepared['warnings']}
    changed = by_code[formula_entry.CHANGED_CODE]
    assert changed['requires_ack'] is False and changed['refs']['ingredients'] == [granola['honey']]
    assert 'not used' in changed['message'] and 'Out of stock' in changed['message']
    assert 'no se usó' in changed['message_es'] and 'Sin existencias' in changed['message_es']
    assert by_code[formula_entry.WARNING_CODE]['requires_ack'] is True
    result = record(client, actors, prepared).json()
    flags = {f['code']: f for f in shift_flags(client, actors, result['receipt_number'])}
    honey = flags['INGREDIENT_NOT_USED']
    assert honey['severity'] == 'info' and honey['refs']['reason_code'] == 'out_of_stock'
    assert 'Out of stock' in honey['message_en'] and 'recorded by' in honey['message_en']
    off = flags['INGREDIENT_OFF_RECIPE']
    assert off['severity'] == 'warn' and off['refs']['standard_lb'] == 160.0 and off['refs']['used_lb'] == 200.0
    assert off['refs']['difference_pct'] == 25.0 and 'weekly view' in off['message_en']


def test_finding5_and_3_swap_row_in_warnings_and_on_the_shift_summary(client, db_cursor, actors, sunshine):
    stock(db_cursor, BRAN, 10)
    prepared = confirmed(client, actors, ss_body())
    changed = next(w for w in prepared['warnings'] if w['code'] == formula_entry.CHANGED_CODE)
    assert changed['requires_ack'] is False and changed['refs']['ingredients'] == [BRAN]
    assert '(swap)' in changed['message'] and 'automatic swap: oat bran short' in changed['message']
    result = record(client, actors, prepared).json()
    flags = [f for f in shift_flags(client, actors, result['receipt_number']) if f['code'].startswith('INGREDIENT_')]
    assert [f['code'] for f in flags] == ['INGREDIENT_SWAP']
    assert flags[0]['refs']['swap_lb'] == 18.0 and flags[0]['refs']['reason_code'] == 'oat_bran_auto_substitute'


def test_n1_bran_recipe_without_oats_line_exact_recipe_needs_no_reason(client, db_cursor, actors, sunshine):
    """Re-review N1: rolled oats offered only for the automatic swap is not a recipe line, so 0 lb
    of it needs no reason; the worker who used exactly the recipe records cleanly."""
    batch, honey = product(db_cursor, 'Bran No Oats', 'batch', 100), product(db_cursor, 'N1 Honey', 'ingredient')
    recipe_line(db_cursor, batch, BRAN, 30)
    recipe_line(db_cursor, batch, honey, 70)
    bran, honey_bag = stock(db_cursor, BRAN, 10), bag(db_cursor, honey, 500, 5)
    data = {'product_id': batch, 'batches': 1, 'occurred_at': main.get_plant_now().isoformat(),
            'confirmed_sku': True, 'formula_entry': True}
    proposal = prepare(client, actors, data)
    assert card(proposal, OATS)['standard_lb'] == 0.0           # offered only for the swap
    # The Make screen sends every card: exact recipe, rolled oats left at 0 lb with no reason.
    lines = [line(BRAN, bran_bag(bran, 30)), line(OATS), line(honey, (honey_bag, 70))]
    prepared = confirmed(client, actors, data | {'ingredient_lines': lines})
    assert prepared['can_commit'] is True, prepared['blockers']
    assert not any(b['code'].startswith('MAKE_REASON') for b in prepared['blockers'])
    oats = card(prepared, OATS)
    assert not oats['not_used'] and not oats['over_tolerance']
    codes = {w['code'] for w in prepared['warnings']}
    assert formula_entry.WARNING_CODE not in codes and formula_entry.CHANGED_CODE not in codes
    response = record(client, actors, prepared)
    assert response.status_code == 200, response.text
    result = response.json()
    assert consumed(db_cursor, result['transaction_id']) == {bran['lot_id']: 30.0, honey_bag['id']: 70.0}
    assert result['ingredient_review'] == []
    assert [(s['lot_id'], s['short_lb']) for s in result['shortages']] == [(bran['lot_id'], 20.0)]
    assert substitution_rows(db_cursor, result['transaction_id']) == []
