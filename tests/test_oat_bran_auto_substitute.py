"""Real HTTP/ledger regressions for Michael's Sunshine oat-bran fallback.

Use the production recipe snapshot plus migration 081, not a reduced recipe;
only the OpenAI response is mocked for the Assistant entry point.
"""
from collections import defaultdict
import json
from pathlib import Path
from uuid import uuid4

import pytest

import main
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import model, record, resolve_product, turn
from tests.test_oat_bran_081 import seed as seed_recipes
from tests.test_write_tickets import commit, headers, posted_count

pytestmark = pytest.mark.db
ROOT = Path(__file__).resolve().parents[1]
BRAN, OATS = 31, 32
RECIPES = {116: (28, 185, 350), 114: (29, 200, 393), 118: (28, 185, 379)}


def stock(cur, pid, quantity, code=None):
    code = code or f'AUTO-{pid}-{uuid4().hex[:8].upper()}'
    cur.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id', (pid, code))
    lot_id = cur.fetchone()['id']
    cur.execute("INSERT INTO transactions(type,notes) VALUES ('receive','oat fallback fixture') RETURNING id")
    tx = cur.fetchone()['id']
    cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,%s)',
                (tx, pid, lot_id, quantity))
    return {'id': pid, 'lot_id': lot_id, 'lot_code': code}


def change_balance(cur, item, quantity):
    delta = quantity - float(main.lot_on_hand(cur, item['lot_id']))
    cur.execute("INSERT INTO transactions(type,notes) VALUES ('adjust','oat fallback stock changed') RETURNING id")
    tx = cur.fetchone()['id']
    cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,%s)',
                (tx, item['id'], item['lot_id'], delta))


@pytest.fixture
def sunshine(db_cursor, monkeypatch):
    monkeypatch.setenv('ASSISTANT_ENABLED', '1')
    seed_recipes(db_cursor)
    db_cursor.execute((ROOT / 'migrations/081_restore_ss_oat_bran.sql').read_text())
    db_cursor.execute("SELECT id FROM products WHERE type='ingredient' AND id<>%s ORDER BY id", (BRAN,))
    return {r['id']: stock(db_cursor, r['id'], 5000) for r in db_cursor.fetchall()}


def payload(pid=116, **changes):
    return {'product_id': pid, 'batches': 1, 'occurred_at': main.get_plant_now().isoformat(),
            'confirmed_sku': True, **changes}


def confirmations(draft):
    return [{'lot_id': row['lot_id'], 'method': 'full_code', 'value': row['lot_code']}
            for row in draft.get('input_plan', [])]


def ticket_prepare(client, actors, data):
    key = actors['floor']['key']
    response = client.post('/make/prepare', json=data, headers=headers(key))
    assert response.status_code == 200, response.text
    prepared = response.json()
    response = client.post('/make/prepare', json=data | {'lot_confirmations': confirmations(prepared['draft'])},
                           headers=headers(key))
    assert response.status_code == 200, response.text
    return response.json()


def run_make(path, client, db_cursor, actors, monkeypatch, data):
    """Return the real user-visible draft and HTTP posting result for each path."""
    if path == 'direct':
        db_cursor.execute('SELECT name FROM products WHERE id=%s', (data['product_id'],))
        request = {k: v for k, v in data.items() if k != 'product_id'}
        request['product_name'] = db_cursor.fetchone()['name']
        preview = client.post('/make', json=request | {'mode': 'preview'}, headers=headers())
        assert preview.status_code == 200, preview.text
        return preview.json(), client.post('/make', json=request | {'mode': 'commit'}, headers=headers())
    if path == 'ticket':
        prepared = ticket_prepare(client, actors, data)
        return prepared['draft'], commit(client, prepared, actors['floor']['key'])
    session = client.post('/assistant/session', headers=headers(actors['floor']['key']))
    assert session.status_code == 200, session.text
    chat = session.json()['session_id'], headers(actors['floor']['key'])
    model_data = {key: value for key, value in data.items() if key != 'confirmed_sku'}
    model(monkeypatch, resolve_product(db_cursor, data['product_id']), ('prepare_make', model_data))
    response = turn(client, chat, 'Make one Sunshine batch')
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['kind'] == 'draft', card
    draft = card['prepared']['draft']
    return draft, record(client, chat, card, lot_confirmations=confirmations(draft))


def receipt(path, response):
    assert response.status_code == 200, response.text
    body = response.json()
    return body['result'] if path == 'assistant' else body


def consumption(cur, transaction_id):
    cur.execute('''SELECT product_id,lot_id,quantity_lb FROM transaction_lines
                   WHERE transaction_id=%s AND quantity_lb<0 ORDER BY id''', (transaction_id,))
    rows = cur.fetchall()
    totals = defaultdict(float)
    for row in rows:
        totals[row['product_id']] -= float(row['quantity_lb'])
    return totals, rows


def assert_notice(document, extra, preview=False):
    [swap] = document['auto_substitutions']
    assert swap['ingredient_product_id'] == BRAN and swap['substitute_product_id'] == OATS
    assert swap['quantity_lb'] == extra and swap['automatic'] is True
    assert swap['reason_code'] == 'oat_bran_auto_substitute'
    en, es = swap['message_en'].lower(), swap['message_es'].lower()
    assert 'oat bran' in en and 'oats' in en and str(extra) in en
    assert 'salvado de avena' in es and 'avena' in es and str(extra) in es
    field = 'preview_message' if preview else 'message'
    assert swap['message_en'] in document[field]
    assert swap['message_es'] in document[field + '_es']


@pytest.mark.parametrize('path', ['direct', 'ticket', 'assistant'])
@pytest.mark.parametrize('pid', [116, 114, 118], ids=['SS1', 'SS2', 'SS3'])
@pytest.mark.parametrize('bran_on_hand', [0, 9], ids=['no-bran', 'partial-bran'])
def test_every_make_path_uses_available_bran_then_extra_oats(
        client, db_cursor, sunshine, actors, monkeypatch, path, pid, bran_on_hand):
    bran_needed, ordinary_oats, output = RECIPES[pid]
    bran_lot = stock(db_cursor, BRAN, bran_on_hand) if bran_on_hand else None
    draft, response = run_make(path, client, db_cursor, actors, monkeypatch, payload(pid))
    made = receipt(path, response)
    extra = bran_needed - bran_on_hand
    assert_notice(draft, extra, preview=True)
    assert_notice(made, extra)
    used, lines = consumption(db_cursor, made['transaction_id'])
    assert used[BRAN] == bran_on_hand
    assert used[OATS] == ordinary_oats + extra
    assert made['output_lb'] == output and sum(used.values()) == output
    assert {r['lot_id'] for r in lines if r['product_id'] == OATS} == {sunshine[OATS]['lot_id']}
    if bran_lot:
        assert main.lot_on_hand(db_cursor, bran_lot['lot_id']) == 0
    db_cursor.execute('SELECT ingredient_product_id,ingredient_lot_id,quantity_lb FROM ingredient_lot_consumption WHERE transaction_id=%s',
                      (made['transaction_id'],))
    trace = db_cursor.fetchall()
    assert sum(float(r['quantity_lb']) for r in trace if r['ingredient_product_id'] == OATS) == ordinary_oats + extra
    assert {r['ingredient_lot_id'] for r in trace if r['ingredient_product_id'] == OATS} == {sunshine[OATS]['lot_id']}
    db_cursor.execute('SELECT * FROM transaction_substitutions WHERE transaction_id=%s', (made['transaction_id'],))
    evidence = db_cursor.fetchall()
    assert evidence and all(r['ingredient_product_id'] == BRAN and r['substitute_product_id'] == OATS for r in evidence)
    assert {r['lot_id'] for r in evidence} == {sunshine[OATS]['lot_id']}
    assert all(r['reason_code'] == 'oat_bran_auto_substitute' for r in evidence)
    db_cursor.execute('SELECT count(*) AS n FROM shortage_flags WHERE transaction_id=%s', (made['transaction_id'],))
    assert db_cursor.fetchone()['n'] == 0
    # The permanent formula remains the precise October 14 recipe.
    db_cursor.execute('SELECT quantity_lb FROM batch_formulas WHERE product_id=%s AND ingredient_product_id=%s', (pid, BRAN))
    assert float(db_cursor.fetchone()['quantity_lb']) == bran_needed


@pytest.mark.parametrize('path', ['direct', 'ticket', 'assistant'])
def test_no_bran_and_short_oats_uses_the_normal_oats_shortage_policy(
        client, db_cursor, sunshine, actors, monkeypatch, path):
    change_balance(db_cursor, sunshine[OATS], 200)  # Recipe needs 185 + 28, short 13.
    draft, response = run_make(path, client, db_cursor, actors, monkeypatch, payload())
    assert_notice(draft, 28, preview=True)
    if path == 'direct':
        assert response.status_code == 400, response.text
        assert 'ingredient ID 32' in response.text and '13.00' in response.text
        db_cursor.execute("SELECT count(*) AS n FROM transactions WHERE type='make'")
        assert db_cursor.fetchone()['n'] == 0
    else:
        made = receipt(path, response)
        [shortage] = made['shortages']
        assert shortage['product_id'] == OATS and shortage['lot_id'] == sunshine[OATS]['lot_id']
        assert shortage['short_lb'] == 13
        db_cursor.execute('SELECT product_id,lot_id,short_lb FROM shortage_flags WHERE transaction_id=%s', (made['transaction_id'],))
        [flag] = db_cursor.fetchall()
        assert (flag['product_id'], flag['lot_id'], float(flag['short_lb'])) == (OATS, sunshine[OATS]['lot_id'], 13)
        assert main.lot_on_hand(db_cursor, sunshine[OATS]['lot_id']) == -13
        db_cursor.execute('SELECT count(*) AS n FROM transaction_lines WHERE transaction_id=%s AND product_id=%s',
                          (made['transaction_id'], BRAN))
        assert db_cursor.fetchone()['n'] == 0


@pytest.mark.parametrize('path', ['direct', 'ticket'])
def test_real_oats_lots_are_consumed_and_audited_when_fallback_splits_lots(
        client, db_cursor, sunshine, actors, monkeypatch, path):
    change_balance(db_cursor, sunshine[OATS], 190)
    second = stock(db_cursor, OATS, 23)
    draft, response = run_make(path, client, db_cursor, actors, monkeypatch, payload())
    made = receipt(path, response)
    used, lines = consumption(db_cursor, made['transaction_id'])
    oats = {r['lot_id']: -float(r['quantity_lb']) for r in lines if r['product_id'] == OATS}
    assert used[OATS] == 213 and used[BRAN] == 0
    assert oats == {sunshine[OATS]['lot_id']: 190, second['lot_id']: 23}
    db_cursor.execute('SELECT lot_id,note FROM transaction_substitutions WHERE transaction_id=%s', (made['transaction_id'],))
    [audit] = db_cursor.fetchall()
    assert audit['lot_id'] in oats
    evidence = json.loads(audit['note'])
    assert {r['lot_id']: r['quantity_lb'] for r in evidence['consumed_oats_lots']} == oats
    if path == 'ticket':
        plan = {r['lot_id'] for r in draft['input_plan'] if r['product_id'] == OATS}
        assert plan == set(oats)


@pytest.mark.parametrize('path', ['direct', 'ticket'])
def test_two_batches_scale_fallback_pound_for_pound(client, db_cursor, sunshine, actors, monkeypatch, path):
    stock(db_cursor, BRAN, 11)
    draft, response = run_make(path, client, db_cursor, actors, monkeypatch, payload(batches=2))
    made = receipt(path, response)
    assert_notice(draft, 45, preview=True)
    used, _ = consumption(db_cursor, made['transaction_id'])
    assert used[BRAN] == 11 and used[OATS] == 415 and made['output_lb'] == 700


@pytest.mark.parametrize('bran_on_hand', [0, 9, 50])
def test_manual_oat_bran_swap_still_overrides_automatic_stock_choice(client, db_cursor, sunshine, actors, bran_on_hand):
    bran = stock(db_cursor, BRAN, bran_on_hand)
    data = payload(substitutions=[{'ingredient_product_id': BRAN, 'substitute_product_id': OATS,
                                  'lot_id': sunshine[OATS]['lot_id'], 'reason_code': 'ingredient_unavailable',
                                  'note': 'Operator selected oats for this make'}])
    prepared = ticket_prepare(client, actors, data)
    assert prepared['can_commit'], prepared
    made = receipt('ticket', commit(client, prepared, actors['floor']['key']))
    used, _ = consumption(db_cursor, made['transaction_id'])
    assert used[BRAN] == 0 and used[OATS] == 213
    assert main.lot_on_hand(db_cursor, bran['lot_id']) == bran_on_hand
    assert not made.get('auto_substitutions')
    db_cursor.execute('SELECT reason_code,note FROM transaction_substitutions WHERE transaction_id=%s', (made['transaction_id'],))
    assert [dict(r) for r in db_cursor.fetchall()] == [{'reason_code': 'ingredient_unavailable', 'note': 'Operator selected oats for this make'}]


@pytest.mark.parametrize('path', ['direct', 'ticket'])
def test_enough_bran_keeps_the_original_recipe(client, db_cursor, sunshine, actors, monkeypatch, path):
    stock(db_cursor, BRAN, 50)
    draft, response = run_make(path, client, db_cursor, actors, monkeypatch, payload())
    made = receipt(path, response)
    used, _ = consumption(db_cursor, made['transaction_id'])
    assert used[BRAN] == 28 and used[OATS] == 185
    assert not draft.get('auto_substitutions') and not made.get('auto_substitutions')


def test_non_sunshine_recipe_with_oat_bran_keeps_its_existing_shortage(client, db_cursor, sunshine, actors):
    db_cursor.execute("""INSERT INTO products(id,name,odoo_code,type,default_batch_lb)
                         SELECT max(id)+1,'Other oat-bran recipe','AUTO-OTHER','batch',28 FROM products RETURNING id""")
    pid = db_cursor.fetchone()['id']
    db_cursor.execute('''INSERT INTO batch_formulas(id,product_id,ingredient_product_id,quantity_lb)
                        SELECT max(id)+1,%s,%s,28 FROM batch_formulas''', (pid, BRAN))
    prepared = ticket_prepare(client, actors, payload(pid))
    assert not prepared['can_commit']
    assert prepared['blockers'][0]['code'] == 'INSUFFICIENT_STOCK'
    assert 'Oat Bran' in prepared['blockers'][0]['message']
    assert posted_count(db_cursor, prepared) == 0


@pytest.mark.parametrize('path', ['direct', 'ticket'])
def test_explicit_oats_exclusion_preserves_manual_recipe_choice(
        client, db_cursor, sunshine, actors, monkeypatch, path):
    bran = stock(db_cursor, BRAN, 0)
    data = payload(excluded_ingredients=[OATS], reason_code='trial_without_ingredient')
    draft, response = run_make(path, client, db_cursor, actors, monkeypatch, data)
    assert not draft.get('auto_substitutions')
    if path == 'direct':
        assert response.status_code == 400, response.text
        assert 'ingredient ID 31' in response.text
        db_cursor.execute("SELECT count(*) AS n FROM transactions WHERE type='make'")
        assert db_cursor.fetchone()['n'] == 0
    else:
        made = receipt(path, response)
        assert not made.get('auto_substitutions')
        used, _ = consumption(db_cursor, made['transaction_id'])
        assert used[BRAN] == 28 and used[OATS] == 0
        assert main.lot_on_hand(db_cursor, bran['lot_id']) == -28
        assert made['shortages'][0]['product_id'] == BRAN
        db_cursor.execute('SELECT reason_code FROM transaction_substitutions WHERE transaction_id=%s', (made['transaction_id'],))
        assert [r['reason_code'] for r in db_cursor.fetchall()] == ['trial_without_ingredient']
    assert main.lot_on_hand(db_cursor, sunshine[OATS]['lot_id']) == 5000


@pytest.mark.parametrize('selected', [False, True], ids=['fifo', 'selected-empty-bran-lot'])
def test_empty_bran_lot_does_not_create_a_bran_shortage(client, db_cursor, sunshine, actors, selected):
    empty = stock(db_cursor, BRAN, 0)
    selection = {'ingredient_lots': [{'ingredient_product_id': BRAN, 'lot_id': empty['lot_id']}]} if selected else {}
    prepared = ticket_prepare(client, actors, payload(**selection))
    made = receipt('ticket', commit(client, prepared, actors['floor']['key']))
    assert made['shortages'] == []
    used, _ = consumption(db_cursor, made['transaction_id'])
    assert used[BRAN] == 0 and used[OATS] == 213
    assert main.lot_on_hand(db_cursor, empty['lot_id']) == 0


def test_ticket_does_not_silently_change_the_confirmed_swap_when_bran_arrives(client, db_cursor, sunshine, actors):
    prepared = ticket_prepare(client, actors, payload())
    assert prepared['can_commit'], prepared
    new_bran = stock(db_cursor, BRAN, 100)
    made = receipt('ticket', commit(client, prepared, actors['floor']['key']))
    used, _ = consumption(db_cursor, made['transaction_id'])
    assert used[BRAN] == 0 and used[OATS] == 213
    assert main.lot_on_hand(db_cursor, new_bran['lot_id']) == 100
    assert_notice(made, 28)


def test_ticket_rejects_depleted_bran_instead_of_silently_posting_a_bran_shortage(client, db_cursor, sunshine, actors):
    bran = stock(db_cursor, BRAN, 9)
    prepared = ticket_prepare(client, actors, payload())
    assert prepared['can_commit'], prepared
    change_balance(db_cursor, bran, 3)
    response = commit(client, prepared, actors['floor']['key'])
    assert response.status_code == 409, response.text
    assert response.json()['detail']['error_code'] == 'TICKET_STALE'
    assert posted_count(db_cursor, prepared) == 0


def test_ticket_oats_depletion_retains_normal_shortage_and_replay_policy(client, db_cursor, sunshine, actors):
    prepared = ticket_prepare(client, actors, payload())
    assert prepared['can_commit'], prepared
    change_balance(db_cursor, sunshine[OATS], 200)
    made = receipt('ticket', commit(client, prepared, actors['floor']['key']))
    assert made['shortages'][0]['product_id'] == OATS and made['shortages'][0]['short_lb'] == 13
    replay = commit(client, prepared, actors['floor']['key'])
    assert replay.json() == made | {'replayed': True}
    assert posted_count(db_cursor, prepared) == 1
    db_cursor.execute('SELECT count(*) AS n FROM transaction_substitutions WHERE transaction_id=%s', (made['transaction_id'],))
    assert db_cursor.fetchone()['n'] == 1


def test_saved_receipt_and_shift_summary_explain_the_swap_in_both_languages(client, db_cursor, sunshine, actors):
    prepared = ticket_prepare(client, actors, payload())
    made = receipt('ticket', commit(client, prepared, actors['floor']['key']))
    response = client.get('/receipts/' + made['receipt_number'], headers=headers(actors['floor']['key']))
    assert response.status_code == 200, response.text
    assert_notice(response.json()['response'], 28)
    response = client.get('/reports/shift-summary', headers=headers(actors['floor']['key']))
    assert response.status_code == 200, response.text
    rows = [row for section in response.json()['sections'] for row in section['rows']
            if row.get('receipt_number') == made['receipt_number']]
    assert len(rows) == 1
    assert 'oat bran' in rows[0]['line_en'].lower() and '28' in rows[0]['line_en']
    assert 'salvado de avena' in rows[0]['line_es'].lower() and '28' in rows[0]['line_es']
