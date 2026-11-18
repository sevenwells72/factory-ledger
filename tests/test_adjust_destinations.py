"""Adjustment rules, owner decision 2026-11-18 (migration 096).

1. A ticket adjustment that REMOVES stock names where it went — one pick from the
   five destination reasons; "Shipped but not recorded" also names the sales order.
   "Wrong lot used" and "Data-entry error" are also accepted, each with a note.
2. Over 500 lb: never held, photo optional, posts at once; the owner is emailed,
   and the email can neither block nor undo the adjustment.
3. Unchanged: named-actor attribution, the 10 % / weekly-view highlighting, "never
   add stock to cover a shortage", shortage-close evidence.
Everything is gated on the migration's marker; without it the old rules hold.
"""
from pathlib import Path
from uuid import uuid4

import psycopg2
import pytest

import main
import adjustment_alerts
import exceptions_enforcement as a3b
from scripts import exceptions_sweep
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import headers, commit, ticket_row, posted_count, error, isolated_database  # noqa: F401
from tests.test_write_tickets_part2 import items, prepare  # noqa: F401
from tests.test_exceptions_enforcement import (adjust, found, exceptions_for, set_balance, seed_photo, hold,  # noqa: F401
                                               open_shortage, resolve, private_evidence_volume)

pytestmark = pytest.mark.db
ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / 'migrations/096_adjust_destinations.sql'
DOWN = ROOT / 'migrations/down/096_adjust_destinations_down.sql'
DESTINATIONS = ['shipped_unrecorded', 'sent_to_sunshine_bulk', 'used_in_production', 'damaged_thrown_out', 'count_unknown']
NEVER_THERE = ['wrong_lot', 'data_entry_error']        # accepted on a removal too, with a required note
REMOVAL_REASONS = DESTINATIONS + NEVER_THERE
EMAIL_ENV = {'RESEND_API_KEY': 're_test_not_real', 'ALERT_EMAIL_TO': 'owner@example.test', 'ENVIRONMENT': 'production'}


@pytest.fixture
def migrated(db_cursor):
    db_cursor.execute(MIGRATION.read_text())


@pytest.fixture
def order(db_cursor):
    token = uuid4().hex[:8].upper()
    db_cursor.execute('INSERT INTO customers(name) VALUES (%s) RETURNING id', (f'Dest Customer {token}',))
    customer_id = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO sales_orders(order_number, customer_id) VALUES (%s,%s) RETURNING id, order_number',
                      (f'SO-DEST-{token}', customer_id))
    return dict(db_cursor.fetchone(), customer_name=f'Dest Customer {token}')


def refuse(client_, payload, key, status, code):
    response = client_.post('/adjust/prepare', json=payload, headers=headers(key))
    error(response, status, code)
    return response.json()['detail']


def alerts(cur):
    cur.execute('SELECT * FROM adjustment_alerts ORDER BY id')
    return cur.fetchall()


def cursor_transaction(cur):
    """`deliver` wants a commit-on-exit transaction factory; in tests it is the suite's cursor."""
    from contextlib import contextmanager

    @contextmanager
    def transaction():
        yield cur
    return transaction


class Mailbox:
    def __init__(self, status='sent'):
        self.status, self.sent = status, []

    def __call__(self, email, env=None):
        self.sent.append(email)
        if self.status == 'sent':
            return {'status': 'sent', 'to': 'owner@example.test', 'provider_id': 'msg_1', 'error': None}
        return {'status': 'failed', 'to': 'owner@example.test', 'provider_id': None, 'error': 'Email service answered 500: down'}


# ── gate: without migration 096 nothing changes ─────────────────────────────

def test_without_migration_096_the_old_rules_hold(client, db_cursor, items, actors):
    key = actors['floor']['key']
    assert a3b.destination_rules(db_cursor) is False
    # A removal still takes the old reasons, and the receipt has no destination.
    small = prepare(client, 'adjust', adjust(items, -5, 'physical_count'), key)
    assert 'destination' not in small['draft']
    posted = commit(client, small, key)
    assert posted.status_code == 200 and 'destination' not in posted.json()
    # Over 500 lb without a photo is still held for the owner.
    big = prepare(client, 'adjust', adjust(items, 600, 'missing_receipt'), key)
    assert big['draft']['correction_review']['photo_required'] is True
    assert 'owner_email' not in big['draft']['correction_review']
    assert commit(client, big, key).status_code == 202
    # A client that already sends the new field is not broken by the old rules.
    assert client.post('/adjust/prepare', json=adjust(items, -5, 'physical_count', sales_order_id=1),
                       headers=headers(key)).status_code == 200


# ── 1. destination on removals ──────────────────────────────────────────────

def test_migration_adds_five_destinations_and_keeps_the_old_reasons(db_cursor, migrated):
    db_cursor.execute('SELECT code, label_en, label_es, adjust_sign, note_required, active, removal_destination, '
                      'removal_allowed, sales_order_required, applies_to FROM correction_reasons ORDER BY sort_order')
    rows = {r['code']: r for r in db_cursor.fetchall()}
    assert [c for c in rows if rows[c]['removal_destination']] == DESTINATIONS
    assert {c: (rows[c]['label_en'], rows[c]['label_es']) for c in DESTINATIONS} == {
        'shipped_unrecorded': ('Shipped but not recorded', 'Enviado pero no registrado'),
        'sent_to_sunshine_bulk': ('Sent to Sunshine as bulk', 'Enviado a Sunshine a granel'),
        'used_in_production': ('Used in production but not recorded', 'Usado en producción pero no registrado'),
        'damaged_thrown_out': ('Damaged or thrown out', 'Dañado o desechado'),
        'count_unknown': ('Count correction — cause unknown', 'Corrección de conteo — causa desconocida')}
    assert all(rows[c]['adjust_sign'] == 'negative' and rows[c]['applies_to'] == ['adjust'] and rows[c]['active']
               for c in DESTINATIONS)
    assert [c for c in DESTINATIONS if rows[c]['sales_order_required']] == ['shipped_unrecorded']
    assert [c for c in DESTINATIONS if rows[c]['note_required']] == ['count_unknown']
    # Allowed on a removal: the five destinations plus the two "never on this lot" reasons,
    # whose own row is otherwise unchanged (any direction, no blanket note requirement).
    assert sorted(c for c in rows if rows[c]['removal_allowed']) == sorted(REMOVAL_REASONS)
    assert all((rows[c]['adjust_sign'], rows[c]['note_required'], rows[c]['removal_destination'], rows[c]['active']) ==
               ('any', False, False, True) for c in NEVER_THERE)
    assert (rows['wrong_lot']['label_en'], rows['data_entry_error']['label_en']) == ('Wrong lot used', 'Data-entry error')
    # The eight original rows are all still there, labels untouched; only the two
    # removal-only reasons a destination replaces are retired for new entries.
    assert (rows['damage_disposal']['label_en'], rows['unrecorded_usage']['label_en']) == ('Damage/disposal', 'Unrecorded usage')
    assert [c for c in rows if not rows[c]['active']] == ['damage_disposal', 'unrecorded_usage']
    assert len(rows) == 13


@pytest.mark.parametrize('reason', ['physical_count', 'missing_production', 'unknown',
                                    'damage_disposal', 'unrecorded_usage', 'count_correction', 'damage'])
def test_a_removal_must_say_where_the_stock_went(client, db_cursor, items, actors, migrated, reason):
    detail = refuse(client, adjust(items, -5, reason, note='n'), actors['floor']['key'], 422, 'REASON_CODE_INVALID')
    # Destinations first, then the two "never on this lot" reasons, which need a note here.
    assert [r['code'] for r in detail['valid_reasons']] == REMOVAL_REASONS
    assert [r['code'] for r in detail['valid_reasons'] if r['note_required']] == ['count_unknown'] + NEVER_THERE
    assert [r['code'] for r in detail['valid_reasons'] if r['removal_destination']] == DESTINATIONS
    assert detail['valid_reasons'][0]['sales_order_required'] is True
    db_cursor.execute('SELECT count(*) AS n FROM write_tickets WHERE action=%s', ('adjust',))
    assert db_cursor.fetchone()['n'] == 0                    # refused at prepare: no ticket exists


def test_a_removal_without_any_reason_lists_the_destinations(client, items, actors, migrated):
    payload = adjust(items, -5, ' ')
    detail = refuse(client, payload, actors['floor']['key'], 422, 'REASON_CODE_REQUIRED')
    assert detail['destination_required'] is True and [r['code'] for r in detail['valid_reasons']] == REMOVAL_REASONS


@pytest.mark.parametrize('reason,label', [('wrong_lot', 'Wrong lot used'), ('data_entry_error', 'Data-entry error')])
def test_wrong_lot_and_data_entry_error_remove_stock_only_with_a_note(client, db_cursor, items, actors, migrated, reason, label):
    key = actors['floor']['key']
    for blank in ({}, {'note': ''}, {'note': '   '}):
        detail = refuse(client, adjust(items, -8, reason, **blank), key, 422, 'NOTE_REQUIRED')
        assert detail['reason_code'] == reason
    prepared = prepare(client, 'adjust', adjust(items, -8, reason, note='Used lot B by mistake; moved to lot B'), key)
    assert prepared['draft']['reason']['note_required'] is True
    assert prepared['draft']['destination'] is None               # it names no destination: the note explains it
    receipt = commit(client, prepared, key).json()
    assert receipt['reason_code'] == reason and receipt['destination'] is None
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 92.0
    db_cursor.execute('SELECT reason_code, adjust_reason, notes, entered_by_actor_id FROM transactions WHERE id=%s', (receipt['transaction_id'],))
    txn = db_cursor.fetchone()
    assert (txn['reason_code'], txn['adjust_reason'], txn['entered_by_actor_id']) == (reason, label, actors['floor']['id'])
    assert 'Used lot B by mistake' in txn['notes']
    # A sales order does not belong on these, and the note rule is for removals only:
    refuse(client, adjust(items, -8, reason, note='n', sales_order_id=1), key, 422, 'SALES_ORDER_NOT_APPLICABLE')
    added = prepare(client, 'adjust', adjust(items, 8, reason), key)             # adding stock: note still optional
    assert added['can_commit'] and added['draft']['reason']['note_required'] is False


def test_large_wrong_lot_removal_emails_the_owner_with_the_note(client, db_cursor, items, actors, migrated):
    key = actors['floor']['key']
    set_balance(db_cursor, items, 'ingredient', 2000.0)
    commit(client, prepare(client, 'adjust', adjust(items, -700, 'wrong_lot', note='Belongs to lot B'), key), key)
    mailbox = Mailbox()
    assert adjustment_alerts.deliver(cursor_transaction(db_cursor), env=EMAIL_ENV, send=mailbox)['status'] == 'sent'
    text = mailbox.sent[0]['text']
    assert 'Reason: Wrong lot used' in text and 'Destination: None: recorded as Wrong lot used' in text
    assert 'Belongs to lot B' in text


@pytest.mark.parametrize('reason', ['sent_to_sunshine_bulk', 'used_in_production', 'damaged_thrown_out'])
def test_a_destination_is_one_pick_and_lands_on_the_receipt(client, db_cursor, items, actors, migrated, reason):
    key = actors['floor']['key']
    prepared = prepare(client, 'adjust', adjust(items, -12, reason), key)
    assert prepared['can_commit'] and prepared['draft']['destination']['code'] == reason
    assert prepared['draft']['destination']['sales_order'] is None
    response = commit(client, prepared, key)
    assert response.status_code == 200, response.text
    receipt = response.json()
    assert receipt['destination']['code'] == reason and receipt['reason_code'] == reason
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 88.0
    # Named-actor attribution is unchanged.
    db_cursor.execute('SELECT reason_code, entered_by_actor_id, operator_id, adjust_reason FROM transactions WHERE id=%s',
                      (receipt['transaction_id'],))
    txn = db_cursor.fetchone()
    assert (txn['reason_code'], txn['entered_by_actor_id'], txn['operator_id']) == (reason, actors['floor']['id'], actors['floor']['name'])
    assert txn['adjust_reason'] == receipt['destination']['label_en']
    assert alerts(db_cursor) == []                           # 12 lb: no owner email


def test_shipped_but_not_recorded_must_name_the_sales_order(client, db_cursor, items, actors, migrated, order):
    key = actors['floor']['key']
    refuse(client, adjust(items, -20, 'shipped_unrecorded'), key, 422, 'SALES_ORDER_REQUIRED')
    refuse(client, adjust(items, -20, 'shipped_unrecorded', sales_order_id=2_000_000_000), key, 422, 'SALES_ORDER_NOT_FOUND')
    db_cursor.execute('SAVEPOINT cancelled_order')
    db_cursor.execute("UPDATE sales_orders SET state='cancelled', state_reason='customer_cancelled' WHERE id=%s", (order['id'],))
    refuse(client, adjust(items, -20, 'shipped_unrecorded', sales_order_id=order['id']), key, 422, 'SALES_ORDER_CANCELLED')
    db_cursor.execute('ROLLBACK TO SAVEPOINT cancelled_order')
    # A closed order is fine: closing it is often how the office found the gap.
    db_cursor.execute("UPDATE sales_orders SET status='confirmed' WHERE id=%s", (order['id'],))
    # A sales order on a reason that is not about shipping is refused, not kept.
    refuse(client, adjust(items, -20, 'damaged_thrown_out', sales_order_id=order['id']), key, 422, 'SALES_ORDER_NOT_APPLICABLE')
    prepared = prepare(client, 'adjust', adjust(items, -20, 'shipped_unrecorded', sales_order_id=order['id'], note='Friday truck'), key)
    named = prepared['draft']['destination']['sales_order']
    assert (named['id'], named['order_number'], named['customer_name']) == (order['id'], order['order_number'], order['customer_name'])
    receipt = commit(client, prepared, key).json()
    assert receipt['destination']['sales_order']['order_number'] == order['order_number']
    assert ticket_row(db_cursor, prepared)['payload']['sales_order_id'] == order['id']
    db_cursor.execute('SELECT notes FROM transactions WHERE id=%s', (receipt['transaction_id'],))
    notes = db_cursor.fetchone()['notes']
    assert order['order_number'] in notes and 'Friday truck' in notes
    # Naming the order explains the removal only: the order itself is untouched.
    db_cursor.execute('SELECT status, state FROM sales_orders WHERE id=%s', (order['id'],))
    assert tuple(db_cursor.fetchone().values()) == ('confirmed', 'open')
    db_cursor.execute('SELECT count(*) AS n FROM shipments WHERE sales_order_id=%s', (order['id'],))
    assert db_cursor.fetchone()['n'] == 0


def test_count_correction_cause_unknown_needs_a_note_and_is_highlighted(client, db_cursor, items, actors, migrated):
    key = actors['floor']['key']
    refuse(client, adjust(items, -3, 'count_unknown'), key, 422, 'NOTE_REQUIRED')
    refuse(client, adjust(items, -3, 'count_unknown', note='   '), key, 422, 'NOTE_REQUIRED')
    prepared = prepare(client, 'adjust', adjust(items, -3, 'count_unknown', note='Counted twice, 3 lb short'), key)
    review = prepared['draft']['correction_review']
    assert review['rules'] == ['REASON_UNKNOWN'] and review['highlighted'] is True      # small, still on the weekly view
    assert commit(client, prepared, key).status_code == 200


def test_adding_stock_keeps_the_old_reasons_and_never_takes_a_destination(client, db_cursor, items, actors, migrated):
    key = actors['floor']['key']
    for reason in DESTINATIONS:
        refuse(client, adjust(items, 5, reason, note='n'), key, 422, 'REASON_SIGN_MISMATCH')
    prepared = prepare(client, 'adjust', adjust(items, 5, 'physical_count'), key)
    assert prepared['draft']['destination'] is None
    assert commit(client, prepared, key).json()['destination'] is None
    detail = refuse(client, adjust(items, 5, 'nonsense'), key, 422, 'REASON_CODE_INVALID')
    assert [r['code'] for r in detail['valid_reasons']] == ['physical_count', 'missing_receipt', 'missing_production',
                                                            'wrong_lot', 'data_entry_error', 'unknown']
    # Found inventory is untouched by the destination rule.
    assert prepare(client, 'found', found(items, 4, 'physical_count'), key)['can_commit']


# ── 2. over 500 lb: never held, photo optional, owner emailed ───────────────

def test_over_500_lb_posts_at_once_without_a_photo_and_queues_the_owner_email(client, db_cursor, items, actors, migrated):
    key = actors['floor']['key']
    set_balance(db_cursor, items, 'ingredient', 2000.0)
    prepared = prepare(client, 'adjust', adjust(items, -600, 'sent_to_sunshine_bulk'), key)
    review = prepared['draft']['correction_review']
    assert review['photo_required'] is False and review['owner_email'] is True
    assert review['rules'] == ['OVER_10_PERCENT_OF_BOOK', 'OVER_500_LB'] and review['highlighted'] is True
    assert 'optional' in review['message'] and 'opcional' in review['message_es']
    response = commit(client, prepared, key)
    assert response.status_code == 200, response.text        # recorded, not 202 held
    receipt = response.json()
    assert receipt['receipt_number'].startswith('ADJ-') and receipt.get('held') is not True
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 1400.0
    assert ticket_row(db_cursor, prepared)['status'] == 'committed'
    assert exceptions_for(db_cursor, ticket_id=prepared['ticket_id']) == []          # no LARGE_CORRECTION, no owner queue
    [alert] = alerts(db_cursor)
    assert (alert['transaction_id'], alert['action'], float(alert['lb']), alert['status'], alert['attempts']) == \
        (receipt['transaction_id'], 'adjust', -600.0, 'pending', 0)
    # A replay returns the same receipt and queues nothing new.
    assert commit(client, prepared, key).json() == receipt | {'replayed': True}
    assert len(alerts(db_cursor)) == 1


@pytest.mark.parametrize('delta,queued', [(-500, False), (-500.5, True), (501, True), (499, False)])
def test_the_line_is_strictly_over_500_lb_either_direction(client, db_cursor, items, actors, migrated, delta, queued):
    key = actors['floor']['key']
    set_balance(db_cursor, items, 'ingredient', 5000.0)
    reason = 'damaged_thrown_out' if delta < 0 else 'missing_receipt'
    assert commit(client, prepare(client, 'adjust', adjust(items, delta, reason), key), key).status_code == 200
    assert len(alerts(db_cursor)) == (1 if queued else 0)


def test_ten_percent_highlight_is_unchanged(client, db_cursor, items, actors, migrated):
    key = actors['floor']['key']                                  # fixture lot: 100 lb on the book
    review = prepare(client, 'adjust', adjust(items, -10, 'damaged_thrown_out'), key)['draft']['correction_review']
    assert review['rules'] == [] and review['highlighted'] is False
    review = prepare(client, 'adjust', adjust(items, -11, 'damaged_thrown_out'), key)['draft']['correction_review']
    assert review['rules'] == ['OVER_10_PERCENT_OF_BOOK'] and review['highlighted'] is True and 'owner_email' not in review


def test_found_over_500_lb_posts_at_once_too(client, db_cursor, items, actors, migrated):
    key = actors['floor']['key']
    prepared = prepare(client, 'found', found(items, 600, 'missing_receipt'), key)
    assert prepared['draft']['correction_review']['photo_required'] is False
    response = commit(client, prepared, key)
    assert response.status_code == 200, response.text
    [alert] = alerts(db_cursor)
    assert (alert['action'], float(alert['lb']), alert['transaction_id']) == ('found', 600.0, response.json()['transaction_id'])


def test_a_photo_is_still_accepted_as_optional_evidence(client, db_cursor, items, actors, migrated):
    key = actors['floor']['key']
    set_balance(db_cursor, items, 'ingredient', 2000.0)
    seed_photo(db_cursor, actors['floor']['id'], 'photos/pallet.jpg')
    prepared = prepare(client, 'adjust', adjust(items, -700, 'damaged_thrown_out', attachment_ref='photos/pallet.jpg'), key)
    receipt = commit(client, prepared, key).json()
    assert receipt['correction_review']['attachment_ref'] == 'photos/pallet.jpg'
    db_cursor.execute('SELECT transaction_id FROM attachments WHERE attachment_ref=%s', ('photos/pallet.jpg',))
    assert db_cursor.fetchone()['transaction_id'] == receipt['transaction_id']


def test_shared_key_large_adjust_posts_too_and_key_access_is_unchanged(client, db_cursor, items, actors, migrated):
    # Which keys may adjust is NOT part of this change: the master key still can, office still cannot.
    set_balance(db_cursor, items, 'ingredient', 2000.0)
    prepared = prepare(client, 'adjust', adjust(items, -600, 'damaged_thrown_out'))
    assert commit(client, prepared).status_code == 200
    error(client.post('/adjust/prepare', json=adjust(items, -5, 'damaged_thrown_out'), headers=headers(actors['office']['key'])),
          403, 'ROLE_NOT_ALLOWED')
    assert len(alerts(db_cursor)) == 1


def test_direct_legacy_adjust_route_also_queues_the_email(client, db_cursor, items, migrated):
    set_balance(db_cursor, items, 'ingredient', 2000.0)
    db_cursor.execute('SELECT lot_code FROM lots WHERE id=%s', (items['ingredient']['lot_id'],))
    response = client.post('/adjust', headers=headers(), json={
        'mode': 'commit', 'product_name': items['ingredient']['name'], 'lot_code': db_cursor.fetchone()['lot_code'],
        'adjustment_lb': -650, 'reason': 'damage', 'reason_es': 'daño'})
    assert response.status_code == 200, response.text
    [alert] = alerts(db_cursor)
    assert alert['transaction_id'] == response.json()['transaction_id'] and float(alert['lb']) == -650.0


# ── 2b. the email: content, failure, retry ──────────────────────────────────

def large_removal(client_, db_cursor, items_, actors_, order_, **extra):
    key = actors_['floor']['key']
    set_balance(db_cursor, items_, 'ingredient', 2000.0)
    prepared = prepare(client_, 'adjust', adjust(items_, -680, 'shipped_unrecorded', sales_order_id=order_['id'],
                                                 note='Went on the Friday truck', **extra), key)
    response = commit(client_, prepared, key)
    assert response.status_code == 200, response.text
    return response.json()


def test_email_names_product_lot_pounds_person_reason_destination_and_photo(client, db_cursor, items, actors, migrated, order):
    seed_photo(db_cursor, actors['floor']['id'], 'photos/truck.jpg')
    receipt = large_removal(client, db_cursor, items, actors, order, attachment_ref='photos/truck.jpg')
    mailbox = Mailbox()
    result = adjustment_alerts.deliver(cursor_transaction(db_cursor), env=EMAIL_ENV, send=mailbox)
    assert result['status'] == 'sent' and result['count'] == 1
    [email] = mailbox.sent
    db_cursor.execute('SELECT lot_code FROM lots WHERE id=%s', (items['ingredient']['lot_id'],))
    lot_code = db_cursor.fetchone()['lot_code']
    assert email['subject'] == f'Large inventory adjustment: −680 lb {items["ingredient"]["name"]}, lot {lot_code}'
    for line in (f'Product: {items["ingredient"]["name"]}', f'Lot: {lot_code}', 'Change: −680 lb (removed)',
                 f'Person: {actors["floor"]["name"]}', 'Reason: Shipped but not recorded',
                 f'Destination: Sales order {order["order_number"]} ({order["customer_name"]})',
                 f'Receipt: {receipt["receipt_number"]}',
                 f'Photo: Attached. Open the receipt to see it: https://cns-factory-ledger.netlify.app/receipt.html?number={receipt["receipt_number"]}'):
        assert line in email['text'], line
    assert 'Went on the Friday truck' in email['text'] and 'nothing is waiting for your approval' in email['text']
    assert order['order_number'] in email['html'] and '<script' not in email['html']
    [alert] = alerts(db_cursor)
    assert (alert['status'], alert['attempts'], alert['email_to'], alert['provider_id'], alert['last_error']) == \
        ('sent', 1, 'owner@example.test', 'msg_1', None)
    assert alert['sent_at'] is not None
    # Sent once: another pass has nothing to do.
    assert adjustment_alerts.deliver(cursor_transaction(db_cursor), env=EMAIL_ENV, send=mailbox)['status'] == 'nothing_pending'
    assert len(mailbox.sent) == 1


def test_email_without_photo_or_order_says_so(client, db_cursor, items, actors, migrated):
    key = actors['floor']['key']
    set_balance(db_cursor, items, 'ingredient', 2000.0)
    commit(client, prepare(client, 'adjust', adjust(items, -600, 'sent_to_sunshine_bulk'), key), key)
    commit(client, prepare(client, 'adjust', adjust(items, 800, 'missing_receipt'), key), key)
    mailbox = Mailbox()
    assert adjustment_alerts.deliver(cursor_transaction(db_cursor), env=EMAIL_ENV | {'ENVIRONMENT': 'staging'}, send=mailbox)['count'] == 2
    [email] = mailbox.sent                                    # one email covers both
    assert email['subject'] == '[staging] 2 large inventory adjustments (over 500 lb each)'
    assert 'Destination: Sent to Sunshine as bulk' in email['text'] and 'Photo: None attached' in email['text']
    assert 'Change: +800 lb (added)' in email['text'] and 'Destination: Not applicable: stock was added' in email['text']


def test_a_failed_email_never_undoes_the_adjustment_and_is_retried(client, db_cursor, items, actors, migrated, order):
    receipt = large_removal(client, db_cursor, items, actors, order)
    lot_id = items['ingredient']['lot_id']
    failing = Mailbox('failed')
    result = adjustment_alerts.deliver(cursor_transaction(db_cursor), env=EMAIL_ENV, send=failing)
    assert result['status'] == 'failed' and 'answered 500' in result['error']
    [alert] = alerts(db_cursor)
    assert (alert['status'], alert['attempts'], alert['sent_at']) == ('pending', 1, None) and 'answered 500' in alert['last_error']
    # The adjustment stands exactly as recorded.
    assert main.lot_on_hand(db_cursor, lot_id) == 1320.0
    db_cursor.execute('SELECT effective_status FROM ledger_current_transactions WHERE id=%s', (receipt['transaction_id'],))
    assert db_cursor.fetchone()['effective_status'] == 'posted'
    # Straight away it is inside the cool-down; later the retry sends it.
    assert adjustment_alerts.deliver(cursor_transaction(db_cursor), env=EMAIL_ENV, send=failing)['status'] == 'nothing_pending'
    db_cursor.execute("UPDATE adjustment_alerts SET last_attempt_at = last_attempt_at - interval '10 minutes'")
    working = Mailbox()
    assert adjustment_alerts.deliver(cursor_transaction(db_cursor), env=EMAIL_ENV, send=working)['status'] == 'sent'
    [alert] = alerts(db_cursor)
    assert (alert['status'], alert['attempts'], alert['last_error']) == ('sent', 2, None)


def test_a_sender_that_raises_leaves_the_alert_pending(client, db_cursor, items, actors, migrated, order):
    large_removal(client, db_cursor, items, actors, order)

    def explode(email, env=None):
        raise RuntimeError('network down')
    result = adjustment_alerts.deliver(cursor_transaction(db_cursor), env=EMAIL_ENV, send=explode)
    assert result['status'] == 'error'
    assert [a['status'] for a in alerts(db_cursor)] == ['pending']


def test_without_email_settings_nothing_is_claimed_and_no_thread_starts(client, db_cursor, items, actors, migrated, order, monkeypatch):
    monkeypatch.delenv('RESEND_API_KEY', raising=False)
    monkeypatch.delenv('ALERT_EMAIL_TO', raising=False)
    large_removal(client, db_cursor, items, actors, order)
    assert adjustment_alerts.kick(main) is False
    assert adjustment_alerts.deliver(cursor_transaction(db_cursor), env={})['status'] == 'not_configured'
    assert [(a['status'], a['attempts']) for a in alerts(db_cursor)] == [('pending', 0)]      # waits for the nightly job


def test_real_resend_sender_failure_is_reported_not_raised(client, db_cursor, items, actors, migrated, order, monkeypatch):
    """The reused reconciliation sender: an unreachable service is a failed send, nothing more."""
    import httpx
    large_removal(client, db_cursor, items, actors, order)

    def unreachable(*args, **kwargs):
        raise httpx.ConnectError('no route')
    monkeypatch.setattr(httpx, 'post', unreachable)
    result = adjustment_alerts.deliver(cursor_transaction(db_cursor), env=EMAIL_ENV)
    assert result['status'] == 'failed' and 'ConnectError' in result['error']
    assert [a['status'] for a in alerts(db_cursor)] == ['pending']
    assert 're_test_not_real' not in str(result)


def test_queue_failure_never_blocks_or_undoes_the_adjustment(client, db_cursor, items, actors, migrated, monkeypatch):
    key = actors['floor']['key']
    set_balance(db_cursor, items, 'ingredient', 2000.0)

    def broken(*args, **kwargs):
        raise RuntimeError('alert code is broken')
    monkeypatch.setattr(a3b, 'is_large', broken)
    # correction_review also uses is_large; break only the queue's call.
    monkeypatch.setattr(a3b, 'correction_review', lambda *a, **k: {
        'rules': ['OVER_500_LB'], 'highlighted': True, 'photo_required': False, 'owner_email': True, 'message': None})
    prepared = prepare(client, 'adjust', adjust(items, -600, 'damaged_thrown_out'), key)
    response = commit(client, prepared, key)
    assert response.status_code == 200, response.text
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 1400.0
    assert alerts(db_cursor) == []
    # The transaction is still usable after the failed savepoint.
    db_cursor.execute('SELECT 1 AS ok')
    assert db_cursor.fetchone()['ok'] == 1


def test_queue_insert_error_rolls_back_only_the_savepoint(client, db_cursor, items, actors, migrated):
    key = actors['floor']['key']
    set_balance(db_cursor, items, 'ingredient', 2000.0)
    # A database-level failure inside the queue: the table refuses the row.
    db_cursor.execute("ALTER TABLE adjustment_alerts ADD CONSTRAINT test_refuse_all CHECK (false) NOT VALID")
    prepared = prepare(client, 'adjust', adjust(items, -600, 'damaged_thrown_out'), key)
    response = commit(client, prepared, key)
    assert response.status_code == 200, response.text
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 1400.0
    assert ticket_row(db_cursor, prepared)['status'] == 'committed' and alerts(db_cursor) == []


def test_background_sender_runs_its_passes_and_stops(monkeypatch):
    monkeypatch.setenv('RESEND_API_KEY', 're_test_not_real')
    monkeypatch.setenv('ALERT_EMAIL_TO', 'owner@example.test')
    passes, slept = [], []
    monkeypatch.setattr(adjustment_alerts, 'deliver', lambda transaction: passes.append(transaction))

    class Api:
        get_transaction = object()
    assert adjustment_alerts.kick(Api, delays=(1, 2, 3), sleep=slept.append) is True
    adjustment_alerts._worker['thread'].join(timeout=5) if adjustment_alerts._worker['thread'] else None
    assert slept == [1, 2, 3] and passes == [Api.get_transaction] * 3
    assert adjustment_alerts._worker['thread'] is None


def test_nightly_sweep_hook_never_raises(monkeypatch, capsys):
    monkeypatch.setenv('DATABASE_URL', 'postgresql://user:secret@db.example.com:5432/prod')     # unguarded target
    monkeypatch.delenv('ENVIRONMENT', raising=False)
    exceptions_sweep.send_adjustment_alerts()
    captured = capsys.readouterr()
    assert 'stay pending' in captured.err and 'secret' not in captured.err + captured.out


# ── 3. what stays ───────────────────────────────────────────────────────────

def test_never_add_stock_to_cover_a_shortage_is_unchanged(client, db_cursor, items, actors, migrated):
    key = actors['floor']['key']
    open_shortage(client, db_cursor, items, actors)
    error(client.post('/adjust/prepare', json=adjust(items, 6, 'missing_receipt'), headers=headers(key)),
          409, 'SHORTAGE_OPEN_RESOLVE_INSTEAD')
    error(client.post('/inventory/found/prepare', json=found(items, 6, 'physical_count'), headers=headers(key)),
          409, 'SHORTAGE_OPEN_RESOLVE_INSTEAD')


def test_counted_shortage_close_over_500_lb_no_longer_needs_a_photo(client, db_cursor, items, actors, migrated):
    key = actors['floor']['key']
    exception_id, _ = open_shortage(client, db_cursor, items, actors)
    # Evidence rules stay: a count is still required to close as counted.
    error(resolve(client, exception_id, key, resolution_kind='counted', note='counted it'), 422, 'COUNT_REQUIRED')
    ok = resolve(client, exception_id, key, resolution_kind='counted', counted_lb=600, note='a whole pallet was mis-shelved')
    assert ok.status_code == 200, ok.text
    res = ok.json()['detail']['resolution']
    assert res['correction_review']['owner_email'] is True and res['correction_review']['photo_required'] is False
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 600.0
    db_cursor.execute('SELECT reason_code FROM transactions WHERE id=%s', (res['transaction_id'],))
    assert db_cursor.fetchone()['reason_code'] == 'physical_count'
    assert [(a['transaction_id'], float(a['lb'])) for a in alerts(db_cursor)] == [(res['transaction_id'], 606.0)]


def test_owner_count_ticket_that_closes_shortages_keeps_physical_count(client, db_cursor, items, actors, migrated):
    owner = actors['owner']['key']
    exception_id, _ = open_shortage(client, db_cursor, items, actors)
    set_balance(db_cursor, items, 'ingredient', 50.0)             # stock arrived; the shortage is still open
    prepared = prepare(client, 'adjust', adjust(items, -10, 'physical_count', resolves_shortage_ids=[exception_id]), owner)
    assert 'destination' not in prepared['draft']
    response = commit(client, prepared, owner)
    assert response.status_code == 200, response.text
    assert exceptions_for(db_cursor, id=exception_id)[0]['resolution_kind'] == 'counted'


def test_a_hold_from_before_the_change_can_still_be_approved(client, db_cursor, items, actors):
    prepared, exception_id = hold(client, db_cursor, items, actors)           # old rules: held without a photo
    set_balance(db_cursor, items, 'ingredient', 2000.0)
    # A second, removal hold whose reason the new rules would refuse.
    key = actors['floor']['key']
    old_removal = prepare(client, 'adjust', adjust(items, -600, 'physical_count'), key)
    assert commit(client, old_removal, key).status_code == 202
    db_cursor.execute(MIGRATION.read_text())
    approved = client.post(f'/exceptions/{exception_id}/approve', json={}, headers=headers(actors['owner']['key']))
    assert approved.status_code == 200, approved.text
    [removal_exc] = exceptions_for(db_cursor, ticket_id=old_removal['ticket_id'])
    approved = client.post(f'/exceptions/{removal_exc["id"]}/approve', json={}, headers=headers(actors['owner']['key']))
    assert approved.status_code == 200, approved.text
    assert approved.json()['reason_code'] == 'physical_count'
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 2000.0
    assert len(alerts(db_cursor)) == 2                                         # both crossed 500 lb: the owner still gets the email


def test_a_hold_from_before_the_change_posts_when_committed_again(client, db_cursor, items, actors):
    prepared, exception_id = hold(client, db_cursor, items, actors)
    db_cursor.execute(MIGRATION.read_text())
    released = commit(client, prepared, actors['floor']['key'])               # no photo: no longer needed
    assert released.status_code == 200, released.text
    [exc] = exceptions_for(db_cursor, id=exception_id)
    assert (exc['status'], exc['resolution_kind']) == ('resolved', 'hold_lifted')
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 700.0


def test_a_hold_from_before_the_change_can_still_be_rejected(client, db_cursor, items, actors):
    prepared, exception_id = hold(client, db_cursor, items, actors)
    db_cursor.execute(MIGRATION.read_text())
    rejected = client.post(f'/exceptions/{exception_id}/reject', json={'resolution_kind': 'declined', 'note': 'wrong lot'},
                           headers=headers(actors['owner']['key']))
    assert rejected.status_code == 200, rejected.text
    assert posted_count(db_cursor, prepared) == 0 and alerts(db_cursor) == []


# ── migration ───────────────────────────────────────────────────────────────

def test_migration_096_is_rerunnable_and_touches_no_ledger_row(client, db_cursor, items, actors):
    key = actors['floor']['key']
    old = commit(client, prepare(client, 'adjust', adjust(items, -5, 'damage_disposal'), key), key).json()
    db_cursor.execute('SELECT id, reason_code, adjust_reason, notes, status FROM transactions ORDER BY id')
    before = db_cursor.fetchall()
    db_cursor.execute(MIGRATION.read_text())
    db_cursor.execute(MIGRATION.read_text())
    db_cursor.execute('SELECT id, reason_code, adjust_reason, notes, status FROM transactions ORDER BY id')
    assert db_cursor.fetchall() == before
    # The historical row keeps its retired reason and its label still resolves.
    db_cursor.execute('''SELECT r.label_en, r.active FROM transactions t JOIN correction_reasons r ON r.code=t.reason_code
                         WHERE t.id=%s''', (old['transaction_id'],))
    assert tuple(db_cursor.fetchone().values()) == ('Damage/disposal', False)
    db_cursor.execute('SELECT count(*) AS n FROM correction_reasons')
    assert db_cursor.fetchone()['n'] == 13
    db_cursor.execute("SELECT count(*) AS n FROM migration_markers WHERE name='096_adjust_destinations'")
    assert db_cursor.fetchone()['n'] == 1


def test_migration_096_down_refuses_while_used_and_restores_when_unused(client, db_cursor, items, actors, migrated):
    key = actors['floor']['key']
    db_cursor.execute('SAVEPOINT clean096')
    commit(client, prepare(client, 'adjust', adjust(items, -5, 'damaged_thrown_out'), key), key)
    db_cursor.execute('SAVEPOINT down096')
    with pytest.raises(psycopg2.Error, match='keep migration 096'):
        db_cursor.execute(DOWN.read_text())
    db_cursor.execute('ROLLBACK TO SAVEPOINT down096')
    db_cursor.execute('ROLLBACK TO SAVEPOINT clean096')
    db_cursor.execute(DOWN.read_text())
    assert a3b.destination_rules(db_cursor) is False
    db_cursor.execute("SELECT to_regclass('public.adjustment_alerts') AS relation")
    assert db_cursor.fetchone()['relation'] is None
    db_cursor.execute('SELECT code FROM correction_reasons WHERE active ORDER BY sort_order')
    assert [r['code'] for r in db_cursor.fetchall()] == ['physical_count', 'missing_receipt', 'missing_production', 'wrong_lot',
                                                         'damage_disposal', 'unrecorded_usage', 'data_entry_error', 'unknown']
    db_cursor.execute("SELECT count(*) AS n FROM information_schema.columns WHERE table_name='correction_reasons' "
                      "AND column_name IN ('removal_destination','removal_allowed','sales_order_required')")
    assert db_cursor.fetchone()['n'] == 0
    # Back on the old rules.
    assert prepare(client, 'adjust', adjust(items, -5, 'physical_count'), key)['can_commit']
