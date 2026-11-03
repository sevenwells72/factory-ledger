"""Make-entry simplification PR-1: lot list instead of typed codes, one suggested lot,
retry-safe Record, and the legacy make-route guard.

Real HTTP prepare/commit against the local test database (never staging/prod).
"""
from pathlib import Path
from uuid import uuid4

import pytest

import main
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_formula_entry import bag, body, card, consumed, granola, line, output, prepare  # noqa: F401
from tests.test_kosher_tiers import approve, items  # noqa: F401
from tests.test_write_tickets import commit, headers, posted_count, ticket_row
from tests.test_write_tickets_part2 import body as ticket_body

pytestmark = pytest.mark.db
MIGRATION = Path(__file__).resolve().parents[1] / 'migrations/090_make_lot_selection.sql'
DOWN = Path(__file__).resolve().parents[1] / 'migrations/down/090_make_lot_selection_down.sql'


@pytest.fixture
def migrated(db_cursor):
    db_cursor.execute(MIGRATION.read_text())


def selections(prepared):
    """What the Make screen sends at Record after the worker taps each lot."""
    return [{'lot_id': row['lot_id'], 'method': 'selected', 'value': row['lot_code']}
            for row in prepared['draft']['input_plan']]


def record(client, actors, prepared, **extra):
    return commit(client, prepared, actors['floor']['key'],
                  acknowledged_warnings=[w['code'] for w in prepared['warnings'] if w.get('requires_ack')], **extra)


def evidence(cur, transaction_id):
    cur.execute('SELECT lot_id,method,value,actor_id FROM transaction_lot_confirmations WHERE transaction_id=%s ORDER BY lot_id',
                (transaction_id,))
    return [dict(r) for r in cur.fetchall()]


def makes_for_attempt(cur, attempt):
    cur.execute("""SELECT count(*) AS n FROM transactions t JOIN write_tickets wt ON wt.id=t.ticket_id
                   WHERE t.type='make' AND wt.payload->>'entry_attempt_id'=%s""", (attempt,))
    return cur.fetchone()['n']


# ── Lot list replaces typed codes ────────────────────────────────────────────

def test_tapped_lots_record_who_selected_them(client, db_cursor, actors, granola, migrated):
    prepared = prepare(client, actors, body(granola))
    assert prepared['can_commit'] is False   # suggestions are never pre-confirmed (R4)
    oats = card(prepared, granola['oats'])
    # The list shows supplier lot, received date and pounds for every in-stock lot.
    assert {'supplier_lot', 'received_on', 'available_lb'} <= set(oats['candidates'][0])
    response = record(client, actors, prepared, lot_confirmations=selections(prepared))
    assert response.status_code == 200, response.text
    rows = evidence(db_cursor, response.json()['transaction_id'])
    assert {r['method'] for r in rows} == {'selected'}
    assert {r['actor_id'] for r in rows} == {actors['floor']['id']}
    assert {r['lot_id'] for r in rows} == {granola['oats_old']['id'], granola['honey_bag']['id']}


def test_a_lot_not_tapped_is_not_recorded(client, db_cursor, actors, granola, migrated):
    prepared = prepare(client, actors, body(granola))
    partial = [s for s in selections(prepared) if s['lot_id'] != granola['honey_bag']['id']]
    response = record(client, actors, prepared, lot_confirmations=partial)
    assert response.status_code == 422, response.text
    assert response.json()['detail']['error_code'] == 'LOT_NOT_CONFIRMED'
    assert posted_count(db_cursor, prepared) == 0


def test_tapped_lot_must_still_be_the_lot_shown(client, db_cursor, actors, granola, migrated):
    prepared = prepare(client, actors, body(granola))
    wrong = [dict(s, value='SOMETHING-ELSE') if s['lot_id'] == granola['oats_old']['id'] else s
             for s in selections(prepared)]
    response = record(client, actors, prepared, lot_confirmations=wrong)
    assert response.status_code == 422 and response.json()['detail']['error_code'] == 'LOT_CONFIRMATION_MISMATCH'
    assert posted_count(db_cursor, prepared) == 0


def test_selection_is_refused_until_migration_090(client, db_cursor, actors, granola):
    prepared = prepare(client, actors, body(granola))
    response = record(client, actors, prepared, lot_confirmations=selections(prepared))
    assert response.status_code == 503, response.text
    assert response.json()['detail']['error_code'] == 'LOT_SELECTION_UNAVAILABLE'
    assert posted_count(db_cursor, prepared) == 0


def test_selection_is_only_for_the_make_screen(client, db_cursor, actors, granola, migrated):
    data = {k: v for k, v in body(granola).items() if k != 'formula_entry'}
    first = prepare(client, actors, data)
    response = client.post('/make/prepare', json=data | {'lot_confirmations': selections(first)},
                           headers=headers(actors['floor']['key']))
    assert response.status_code == 422 and response.json()['detail']['error_code'] == 'LOT_SELECTION_MAKE_SCREEN_ONLY'
    late = record(client, actors, first, lot_confirmations=selections(first))
    assert late.status_code == 422 and late.json()['detail']['error_code'] == 'LOT_SELECTION_MAKE_SCREEN_ONLY'
    assert posted_count(db_cursor, first) == 0


def test_shared_key_cannot_select_lots(client, db_cursor, granola, migrated):
    first = client.post('/make/prepare', json=body(granola), headers=headers()).json()
    response = client.post('/make/prepare', json=body(granola) | {'lot_confirmations': selections(first)},
                           headers=headers())
    assert response.status_code == 403 and response.json()['detail']['error_code'] == 'ACTOR_REQUIRED'


def test_typed_codes_still_work_for_other_callers(client, db_cursor, actors, granola, migrated):
    prepared = prepare(client, actors, body(granola))
    typed = [{'lot_id': r['lot_id'], 'method': 'full_code', 'value': r['lot_code']} for r in prepared['draft']['input_plan']]
    response = record(client, actors, prepared, lot_confirmations=typed)
    assert response.status_code == 200, response.text
    assert {r['method'] for r in evidence(db_cursor, response.json()['transaction_id'])} == {'full_code'}


def test_migration_090_is_idempotent_and_down_refuses_with_evidence(client, db_cursor, actors, granola, migrated):
    db_cursor.execute(MIGRATION.read_text())   # second apply is a no-op
    prepared = prepare(client, actors, body(granola))
    assert record(client, actors, prepared, lot_confirmations=selections(prepared)).status_code == 200
    db_cursor.execute('SAVEPOINT down')
    with pytest.raises(Exception, match='selected lot confirmations exist'):
        db_cursor.execute(DOWN.read_text())
    db_cursor.execute('ROLLBACK TO SAVEPOINT down')


# ── One suggested lot; real multi-lot use with known pounds ─────────────────

def test_no_automatic_second_lot_short_stock_posts_as_shortage(client, db_cursor, actors, granola, migrated):
    prepared = prepare(client, actors, body(granola, pans=2))
    oats = card(prepared, granola['oats'])
    assert [b['lot_id'] for b in oats['bags']] == [granola['oats_old']['id']]
    shortage = [w for w in prepared['warnings'] if w['code'] == 'WILL_CREATE_SHORTAGE']
    assert shortage and shortage[0]['requires_ack'] is False   # shortage policy unchanged: warns, never blocks
    response = record(client, actors, prepared, lot_confirmations=selections(prepared))
    assert response.status_code == 200, response.text
    result = response.json()
    assert granola['oats_new']['id'] not in consumed(db_cursor, result['transaction_id'])
    assert [(s['lot_id'], s['short_lb']) for s in result['shortages']] == [(granola['oats_old']['id'], 110.0)]


def test_two_lots_with_known_pounds_post_exactly(client, db_cursor, actors, granola, migrated):
    lines = [line(granola['oats'], (granola['oats_old'], 50), (granola['oats_new'], 110))]
    prepared = prepare(client, actors, body(granola, pans=2, ingredient_lines=lines))
    response = record(client, actors, prepared, lot_confirmations=selections(prepared))
    assert response.status_code == 200, response.text
    result = response.json()
    assert consumed(db_cursor, result['transaction_id']) == {
        granola['oats_old']['id']: 50.0, granola['oats_new']['id']: 110.0, granola['honey_bag']['id']: 40.0}
    assert result['shortages'] == []
    assert len(evidence(db_cursor, result['transaction_id'])) == 3


# ── Retry safety ─────────────────────────────────────────────────────────────

def test_same_ticket_retry_replays_without_a_second_make(client, db_cursor, actors, granola, migrated):
    attempt = uuid4().hex
    prepared = prepare(client, actors, body(granola, entry_attempt_id=attempt))
    first = record(client, actors, prepared, lot_confirmations=selections(prepared))
    again = record(client, actors, prepared, lot_confirmations=selections(prepared))
    assert first.status_code == again.status_code == 200
    assert again.json()['replayed'] is True and again.json()['receipt_number'] == first.json()['receipt_number']
    assert makes_for_attempt(db_cursor, attempt) == 1


def test_new_draft_after_a_lost_answer_shows_the_receipt_not_a_second_make(client, db_cursor, actors, granola, migrated):
    attempt = uuid4().hex
    prepared = prepare(client, actors, body(granola, entry_attempt_id=attempt))
    receipt = record(client, actors, prepared, lot_confirmations=selections(prepared)).json()['receipt_number']
    retry = prepare(client, actors, body(granola, entry_attempt_id=attempt))
    assert retry['can_commit'] is False
    assert retry['blockers'][0]['code'] == 'MAKE_ALREADY_RECORDED'
    assert receipt in retry['blockers'][0]['message'] and retry['draft']['already_recorded']['receipt_number'] == receipt
    response = record(client, actors, retry, lot_confirmations=selections(retry))
    assert response.status_code == 409 and response.json()['detail']['error_code'] == 'MAKE_ALREADY_RECORDED'
    assert makes_for_attempt(db_cursor, attempt) == 1


def test_two_open_drafts_of_one_make_post_once(client, db_cursor, actors, granola, migrated):
    attempt = uuid4().hex
    a = prepare(client, actors, body(granola, entry_attempt_id=attempt))
    b = prepare(client, actors, body(granola, pans=1, entry_attempt_id=attempt))
    assert record(client, actors, a, lot_confirmations=selections(a)).status_code == 200
    second = record(client, actors, b, lot_confirmations=selections(b))
    assert second.status_code == 409, second.text
    detail = second.json()['detail']
    assert detail['error_code'] == 'MAKE_ALREADY_RECORDED' and detail['receipt_number'].startswith('MK-')
    assert ticket_row(db_cursor, b)['status'] == 'rejected'
    assert makes_for_attempt(db_cursor, attempt) == 1


def test_different_attempts_are_separate_makes(client, db_cursor, actors, granola, migrated):
    for _ in range(2):
        prepared = prepare(client, actors, body(granola, pans=1, entry_attempt_id=uuid4().hex))
        response = record(client, actors, prepared, lot_confirmations=selections(prepared))
        assert response.status_code == 200, response.text


def test_bad_attempt_id_is_refused(client, actors, granola):
    response = client.post('/make/prepare', json=body(granola, entry_attempt_id='x; drop'),
                           headers=headers(actors['floor']['key']))
    assert response.status_code == 422


# ── Extra-kosher safeguards unchanged ───────────────────────────────────────

def test_extra_kosher_make_with_tapped_lots_is_held_then_retry_blocked(client, db_cursor, items, actors, migrated):
    key, attempt = actors['floor']['key'], uuid4().hex
    data = ticket_body('make', items) | {'confirmed_sku': True, 'formula_entry': True, 'entry_attempt_id': attempt}
    prepared = client.post('/make/prepare', json=data, headers=headers(key)).json()
    held = commit(client, prepared, key, lot_confirmations=selections(prepared))
    assert held.status_code == 202, held.text
    assert held.json()['posted'] is False and posted_count(db_cursor, prepared) == 0
    retry = client.post('/make/prepare', json=data, headers=headers(key)).json()
    assert retry['blockers'][0]['code'] == 'MAKE_ALREADY_RECORDED' and retry['can_commit'] is False
    approved = approve(client, held.json(), actors)
    assert approved.status_code == 200, approved.text
    assert {r['method'] for r in evidence(db_cursor, approved.json()['transaction_id'])} == {'selected'}


# ── Legacy make route ────────────────────────────────────────────────────────

def test_legacy_make_commit_is_marked_unverified(client, db_cursor, granola):
    db_cursor.execute('SELECT name FROM products WHERE id=%s', (granola['batch'],))
    name = db_cursor.fetchone()['name']
    response = client.post('/make', json={'mode': 'commit', 'product_name': name, 'batches': 1, 'confirmed_sku': True},
                           headers=headers())
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['lot_verification'] == 'none'
    assert 'LEGACY_MAKE_UNVERIFIED_LOTS' in {w['code'] for w in data['warnings'] if isinstance(w, dict)}


@pytest.mark.parametrize('path,mode', [('/make', 'commit'), ('/make/commit', None)])
def test_legacy_make_commit_can_be_blocked(client, db_cursor, granola, monkeypatch, path, mode):
    monkeypatch.setenv('LEGACY_MAKE_COMMIT', 'block')
    db_cursor.execute('SELECT name FROM products WHERE id=%s', (granola['batch'],))
    name = db_cursor.fetchone()['name']
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    before = db_cursor.fetchone()['n']
    payload = {'product_name': name, 'batches': 1, 'confirmed_sku': True} | ({'mode': mode} if mode else {})
    response = client.post(path, json=payload, headers=headers())
    assert response.status_code == 403 and response.json()['detail']['error_code'] == 'LEGACY_MAKE_DISABLED'
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    assert db_cursor.fetchone()['n'] == before
    preview = client.post('/make', json=payload | {'mode': 'preview'}, headers=headers())
    assert preview.status_code == 200, preview.text


@pytest.mark.parametrize('row,kind,code', [
    ({'lot_type': 'commingled', 'supplier_lot_code': 'ABC'}, 'pooled', None),
    ({'supplier_lot_code': 'N/A'}, 'no_lot', None),
    ({'supplier_lot_code': None}, 'not_recorded', None),
    ({'supplier_lot_code': ' L-778 '}, 'supplier', 'L-778')])
def test_lot_list_supplier_lot_is_never_guessed(row, kind, code):
    import formula_entry
    assert formula_entry.supplier_lot(row) == {'kind': kind, 'code': code}


# ── PR #176 staging fixes (ChatGPT Work acceptance, 2026-11-02) ─────────────

def _ledger(cur):
    cur.execute("SELECT count(*) AS n FROM transactions")
    tx = cur.fetchone()['n']
    cur.execute("SELECT count(*) AS n FROM transaction_lines")
    return tx, cur.fetchone()['n']


@pytest.mark.parametrize('pounds', [-25, -0.5, 0])
def test_negative_or_zero_lot_pounds_are_refused_by_the_server(client, db_cursor, actors, granola, migrated, pounds):
    before = _ledger(db_cursor)
    data = body(granola, entry_attempt_id=uuid4().hex,
                ingredient_lines=[line(granola['oats'], (granola['oats_old'], pounds))])
    response = client.post('/make/prepare', json=data, headers=headers(actors['floor']['key']))
    assert response.status_code == 422, response.text
    db_cursor.execute("SELECT count(*) AS n FROM write_tickets WHERE payload->>'entry_attempt_id'=%s",
                      (data['entry_attempt_id'],))
    assert db_cursor.fetchone()['n'] == 0, 'no draft ticket for a refused amount'
    assert _ledger(db_cursor) == before


def test_explicit_not_used_still_records_with_a_reason(client, db_cursor, actors, granola, migrated):
    lines = [line(granola['honey'], reason_code='out_of_stock')]
    prepared = prepare(client, actors, body(granola, ingredient_lines=lines))
    response = record(client, actors, prepared, lot_confirmations=selections(prepared))
    assert response.status_code == 200, response.text
    assert granola['honey_bag']['id'] not in consumed(db_cursor, response.json()['transaction_id'])


def test_chosen_lot_with_negative_book_stock_keeps_identity_and_posts_a_shortage(client, db_cursor, actors, granola, migrated):
    honey = granola['honey_bag']
    db_cursor.execute("UPDATE lots SET supplier_lot_code='HSUP-1' WHERE id=%s", (honey['id'],))
    db_cursor.execute("INSERT INTO transactions(type,notes) VALUES ('adjust','fixture: honey below zero') RETURNING id")
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,-110)',
                      (db_cursor.fetchone()['id'], granola['honey'], honey['id']))
    fresh = bag(db_cursor, granola['honey'], 500, 1)   # another honey lot WITH stock
    lines = [line(granola['honey'], (honey, 40))]
    prepared = prepare(client, actors, body(granola, ingredient_lines=lines))
    honey_card = card(prepared, granola['honey'])
    assert [b['lot_id'] for b in honey_card['bags']] == [honey['id']], 'FL never switches to the stocked lot'
    [shown] = honey_card['unstocked_lots']
    assert shown['lot_code'] == honey['code'] and shown['supplier_lot'] == {'kind': 'supplier', 'code': 'HSUP-1'}
    assert shown['received_on'] and shown['book_lb'] == -10.0 and shown['available_lb'] == 0.0
    assert honey['id'] not in [c['lot_id'] for c in honey_card['candidates']]
    assert fresh['id'] in [c['lot_id'] for c in honey_card['candidates']]
    response = record(client, actors, prepared, lot_confirmations=selections(prepared))
    assert response.status_code == 200, response.text
    result = response.json()
    assert consumed(db_cursor, result['transaction_id'])[honey['id']] == 40.0
    assert fresh['id'] not in consumed(db_cursor, result['transaction_id'])
    assert [s['lot_id'] for s in result['shortages'] if s['product_id'] == granola['honey']] == [honey['id']]


def test_lost_response_retries_never_double_post(client, db_cursor, actors, granola, migrated):
    """Scenario A: FL commits, the phone never hears back. Retrying the same ticket and
    re-preparing the same entry both lead to the one existing receipt."""
    attempt = uuid4().hex
    balances = lambda: {k: float(main.lot_on_hand(db_cursor, granola[k]['id'])) for k in ('oats_old', 'honey_bag')}
    before, ledger_before = balances(), _ledger(db_cursor)
    prepared = prepare(client, actors, body(granola, entry_attempt_id=attempt))
    first = record(client, actors, prepared, lot_confirmations=selections(prepared))   # answer "lost"
    assert first.status_code == 200
    receipt, txn = first.json()['receipt_number'], first.json()['transaction_id']
    after_one = balances()
    ledger_after_one = _ledger(db_cursor)
    retry = record(client, actors, prepared, lot_confirmations=selections(prepared))
    assert retry.status_code == 200 and retry.json()['replayed'] is True
    assert (retry.json()['receipt_number'], retry.json()['transaction_id']) == (receipt, txn)
    again = prepare(client, actors, body(granola, entry_attempt_id=attempt))
    assert again['blockers'][0]['code'] == 'MAKE_ALREADY_RECORDED' and again['blockers'][0]['receipt_number'] == receipt
    refused = record(client, actors, again, lot_confirmations=selections(again))
    assert refused.status_code == 409 and refused.json()['detail']['receipt_number'] == receipt
    assert balances() == after_one and after_one != before, 'ingredients deducted exactly once'
    assert _ledger(db_cursor) == ledger_after_one and ledger_after_one != ledger_before
    assert makes_for_attempt(db_cursor, attempt) == 1


def test_extra_kosher_retry_after_owner_approval_returns_the_receipt(client, db_cursor, items, actors, migrated):
    """Scenario B: held for Michael, approved and posted; retrying the original entry
    (same ticket, or a new draft of the same entry) never posts again."""
    key, attempt = actors['floor']['key'], uuid4().hex
    data = ticket_body('make', items) | {'confirmed_sku': True, 'formula_entry': True, 'entry_attempt_id': attempt}
    prepared = client.post('/make/prepare', json=data, headers=headers(key)).json()
    second_draft = client.post('/make/prepare', json=data | {'note': 'second tab'}, headers=headers(key)).json()
    held = commit(client, prepared, key, lot_confirmations=selections(prepared))
    assert held.status_code == 202, held.text
    ingredient_before = float(main.lot_on_hand(db_cursor, items['ingredient']['lot_id']))
    approved = approve(client, held.json(), actors)
    assert approved.status_code == 200, approved.text
    receipt, txn = approved.json()['receipt_number'], approved.json()['transaction_id']
    ingredient_after = float(main.lot_on_hand(db_cursor, items['ingredient']['lot_id']))
    assert ingredient_after < ingredient_before
    ledger = _ledger(db_cursor)
    db_cursor.execute("SELECT count(*) AS n FROM lots WHERE product_id=%s", (items['batch']['id'],))
    output_lots = db_cursor.fetchone()['n']
    # Same ticket again: the saved receipt, marked as a replay.
    replay = commit(client, prepared, key, lot_confirmations=selections(prepared))
    assert replay.status_code == 200, replay.text
    assert (replay.json()['receipt_number'], replay.json()['transaction_id'], replay.json()['replayed']) == (receipt, txn, True)
    # A new draft of the same entry: already recorded, with the receipt.
    again = client.post('/make/prepare', json=data, headers=headers(key)).json()
    assert again['can_commit'] is False and again['blockers'][0]['code'] == 'MAKE_ALREADY_RECORDED'
    assert again['blockers'][0]['receipt_number'] == receipt
    # A second draft opened before approval cannot post either.
    late = commit(client, second_draft, key, lot_confirmations=selections(second_draft))
    assert late.status_code == 409 and late.json()['detail']['error_code'] == 'MAKE_ALREADY_RECORDED'
    assert late.json()['detail']['receipt_number'] == receipt
    assert _ledger(db_cursor) == ledger
    db_cursor.execute("SELECT count(*) AS n FROM lots WHERE product_id=%s", (items['batch']['id'],))
    assert db_cursor.fetchone()['n'] == output_lots, 'no second finished lot'
    assert float(main.lot_on_hand(db_cursor, items['ingredient']['lot_id'])) == ingredient_after
    assert makes_for_attempt(db_cursor, attempt) == 1


def test_shortage_warning_on_a_lot_already_below_zero_states_new_shortage_and_balance_after(client, db_cursor, actors, granola, migrated):
    """Final acceptance: -25 lb before, 25 lb used → 25 lb more short, -50 lb after (EN and ES)."""
    honey = granola['honey_bag']
    db_cursor.execute("INSERT INTO transactions(type,notes) VALUES ('adjust','fixture: honey to -25') RETURNING id")
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,-125)',
                      (db_cursor.fetchone()['id'], granola['honey'], honey['id']))
    lines = [line(granola['honey'], (honey, 25))]
    prepared = prepare(client, actors, body(granola, ingredient_lines=lines))
    [warning] = [w for w in prepared['warnings'] if w['code'] == 'WILL_CREATE_SHORTAGE' and w['refs']['lot_id'] == honey['id']]
    assert warning['requires_ack'] is False
    name = card(prepared, granola['honey'])['product_name']
    assert warning['message'].startswith(
        f"{name} lot {honey['code']} is already at -25 lb on the books (none available); this entry uses 25 lb, "
        "adding 25 lb of shortage. Book balance after: -50 lb.")
    assert warning['message_es'].startswith(
        f"{name} lote {honey['code']} ya está en -25 lb en el sistema (nada disponible); esta entrada usa 25 lb "
        "y agrega 25 lb de faltante. Saldo en el sistema después: -50 lb.")
    assert warning['refs'] == {'lot_id': honey['id'], 'short_lb': 25.0, 'on_hand_lb': -25.0}   # refs unchanged
    response = record(client, actors, prepared, lot_confirmations=selections(prepared))
    assert response.status_code == 200, response.text
    [flag] = [s for s in response.json()['shortages'] if s['lot_id'] == honey['id']]
    assert flag['short_lb'] == 25.0   # posting rule unchanged: the new shortage only
    assert float(main.lot_on_hand(db_cursor, honey['id'])) == -50.0


def test_shortage_warning_on_a_lot_with_some_stock(client, db_cursor, actors, granola, migrated):
    prepared = prepare(client, actors, body(granola, pans=2))
    [warning] = [w for w in prepared['warnings'] if w['code'] == 'WILL_CREATE_SHORTAGE']
    assert ' has 50 lb on the books; this entry uses 160 lb, 110 lb more than the lot has. Book balance after: -110 lb.' in warning['message']
    assert ' tiene 50 lb en el sistema; esta entrada usa 160 lb, 110 lb más de lo que tiene el lote. Saldo en el sistema después: -110 lb.' in warning['message_es']
