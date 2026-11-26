"""Weighed batch output on the Make screen (sweetened coconut yield, 2026-11-26).

Sweetened coconut batches are booked at 360 lb per pan, but real runs weigh about
1% more, so packs ran over the batch lot and opened SHORTAGE exceptions almost
every run. The Make screen can now send `weighed_output_lb` (the scale reading for
the whole run): it is booked on the output lot, while ingredient pounds stay
recipe x pans. Blank keeps today's behaviour, byte for byte (payload and hash).

Real HTTP prepare/commit against the local test database.
"""
import json
from pathlib import Path

import pytest

import assistant_tools
import formula_entry
import main
import reconciliation
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_daily_reconciliation import Kit, at, night
from tests.test_formula_entry import (body, confirmed, consumed, granola, output, prepare,  # noqa: F401
                                      record, ticket_row)
from tests.test_write_tickets import headers

pytestmark = pytest.mark.db
ROOT = Path(__file__).resolve().parents[1]


def payload_of(cur, prepared):
    payload = ticket_row(cur, prepared)['payload']
    return payload if isinstance(payload, dict) else json.loads(payload)


def notes_of(cur, transaction_id):
    cur.execute('SELECT notes FROM transactions WHERE id=%s', (transaction_id,))
    return cur.fetchone()['notes']


def codes(items):
    return [item['code'] for item in items]


def test_weighed_output_books_the_scale_reading_and_keeps_recipe_consumption(client, db_cursor, actors, granola):
    # 2 pans x 100 lb = 200 lb standard; the scale reads 202.5 lb (+1.25%).
    prepared = confirmed(client, actors, body(granola, pans=2, weighed_output_lb=202.5))
    assert prepared['can_commit'] is True, prepared['blockers']
    draft = prepared['draft']
    assert (draft['total_output_lb'], draft['standard_output_lb'], draft['weighed_difference_pct']) == (202.5, 200.0, 1.2)
    assert draft['formula_entry']['output_lb'] == 200.0          # pans x pan lb, unchanged
    assert draft['formula_entry']['weighed_output_lb'] == 202.5
    assert '202.5 lb' in draft['preview_message'] and 'standard 200 lb' in draft['preview_message']
    # A normal ~1% coconut gain needs no acknowledgement.
    assert formula_entry.OUTPUT_WARNING_CODE not in codes(prepared['warnings'])
    specification = payload_of(db_cursor, prepared)['specification']
    assert (specification['output_lb'], specification['standard_output_lb']) == (202.5, 200.0)

    response = record(client, actors, prepared)
    assert response.status_code == 200, response.text
    result = response.json()
    assert output(db_cursor, result['transaction_id']) == 202.5
    # Ingredients: exactly recipe x pans, as for an unweighed make.
    assert consumed(db_cursor, result['transaction_id']) == {
        granola['oats_old']['id']: 160.0, granola['honey_bag']['id']: 40.0}
    assert (result['weighed_output_lb'], result['standard_output_lb']) == (202.5, 200.0)
    assert 'weighed output 202.5 lb (standard 200 lb, +1.2%)' in notes_of(db_cursor, result['transaction_id'])
    # The lot trace output matches the booked pounds.
    db_cursor.execute("""SELECT el.quantity_lb FROM trace_event_lots el JOIN trace_events e ON e.id=el.trace_event_id
                         WHERE e.transaction_id=%s AND el.role='output'""", (result['transaction_id'],))
    assert [float(r['quantity_lb']) for r in db_cursor.fetchall()] == [202.5]


def test_blank_weighed_output_keeps_the_payload_draft_and_posting_unchanged(client, db_cursor, actors, granola):
    prepared = confirmed(client, actors, body(granola, pans=2))
    payload = payload_of(db_cursor, prepared)
    assert 'weighed_output_lb' not in payload
    assert 'standard_output_lb' not in payload['specification']
    draft = prepared['draft']
    assert 'weighed_output_lb' not in draft and 'standard_output_lb' not in draft
    assert 'weighed_output_lb' not in draft['formula_entry']
    response = record(client, actors, prepared)
    assert response.status_code == 200, response.text
    result = response.json()
    assert output(db_cursor, result['transaction_id']) == 200.0
    assert 'weighed' not in notes_of(db_cursor, result['transaction_id'])
    assert 'weighed_output_lb' not in result


def test_explicit_null_is_the_same_as_blank(client, db_cursor, actors, granola):
    prepared = prepare(client, actors, body(granola, pans=2, weighed_output_lb=None))
    assert 'weighed_output_lb' not in payload_of(db_cursor, prepared)


@pytest.mark.parametrize('weighed, warned', [(210.0, False), (210.2, True), (189.0, True), (250.0, True)])
def test_more_than_five_percent_off_needs_an_acknowledgement(client, actors, granola, weighed, warned):
    prepared = confirmed(client, actors, body(granola, pans=2, weighed_output_lb=weighed))
    assert prepared['can_commit'] is True, prepared['blockers']
    warning = [w for w in prepared['warnings'] if w['code'] == formula_entry.OUTPUT_WARNING_CODE]
    assert bool(warning) is warned
    if warned:
        assert warning[0]['requires_ack'] is True
        assert f'{weighed:,g} lb' in warning[0]['message'] and 'standard 200 lb' in warning[0]['message']


@pytest.mark.parametrize('weighed', [250.5, 149.0, 2025.0])
def test_more_than_twenty_five_percent_off_is_refused(client, db_cursor, actors, granola, weighed):
    prepared = prepare(client, actors, body(granola, pans=2, weighed_output_lb=weighed))
    assert prepared['can_commit'] is False
    assert 'WEIGHED_OUTPUT_IMPLAUSIBLE' in codes(prepared['blockers'])
    before = output_rows(db_cursor, granola['batch'])
    response = client.post(f"/tickets/{prepared['ticket']}/commit", json={'acknowledged_warnings': []},
                           headers=headers(actors['floor']['key']))
    assert response.status_code in (409, 422), response.text
    assert output_rows(db_cursor, granola['batch']) == before


def output_rows(cur, product_id):
    cur.execute('SELECT count(*) AS n FROM transaction_lines WHERE product_id=%s', (product_id,))
    return cur.fetchone()['n']


@pytest.mark.parametrize('weighed', [0, -5, 'abc'])
def test_zero_negative_or_text_weight_is_refused_by_the_request(client, actors, granola, weighed):
    response = client.post('/make/prepare', json=body(granola, weighed_output_lb=weighed),
                           headers=headers(actors['floor']['key']))
    assert response.status_code == 422


def test_assistant_cannot_send_a_weighed_output():
    schema = next(t for t in assistant_tools.tools([]) if t['name'] == 'prepare_make')['parameters']
    assert 'weighed_output_lb' not in schema['properties']
    assert assistant_tools.has_ui_fields({'weighed_output_lb': 4365})


def test_legacy_make_route_and_gpt_schema_are_unchanged():
    """POST /make (GPT schema) has no weighed field and still books batches x pan lb."""
    assert 'weighed_output_lb' not in main.MakeRequest.model_fields
    assert 'weighed_output_lb' not in (ROOT / 'openapi-gpt-v3.yaml').read_text()


def test_screen_sends_the_weighed_field_and_shows_it():
    entry = (ROOT / 'dashboard/entry.js').read_text()
    assert "field('weighed_output_lb'" in entry
    formula_js = (ROOT / 'dashboard/make-formula.js').read_text()
    assert 'block.weighed_output_lb' in formula_js


# ───────────── daily reconciliation: recipe check uses the pans, not output / pan lb ─────────────

def _make_ticket(kit, batches, weighed):
    payload = {'batches': batches, **({'weighed_output_lb': weighed} if weighed is not None else {})}
    kit.cur.execute("""INSERT INTO write_tickets (ticket_hash, action, operator_id, key_kind, client_source, payload,
                           payload_hash, state_hash, draft, status, expires_at, committed_at)
                       VALUES (md5(random()::text) || md5(random()::text), 'make', 'test', 'legacy_dashboard',
                               'dashboard', %s, %s, %s, '{}', 'committed', now() + interval '1 hour', now())
                       RETURNING id""", (json.dumps(payload), 'a' * 64, 'b' * 64))
    return kit.cur.fetchone()['id']


def heavy_make(kit, batches, output_lb, when, *, weighed):
    """A ticket make whose output lot got `output_lb`; ingredients exactly recipe x batches."""
    ticket = _make_ticket(kit, batches, output_lb if weighed else None)
    kit.cur.execute("""INSERT INTO transactions (type, timestamp, status, occurred_at, business_date, created_at,
                                                  operator_id, entered_by_actor_id, ticket_id)
                       VALUES ('make', %s, 'posted', %s, %s, %s, %s, %s, %s) RETURNING id""",
                    (when.replace(tzinfo=None), when, when.date(), when, f'{kit.tag} Worker', kit.actor, ticket))
    tx = kit.cur.fetchone()['id']
    kit.txs.add(tx)
    lines = [(kit.lot(kit.granola), output_lb), (kit.oats_lot, -60 * batches),
             (kit.honey_lot, -30 * batches), (kit.nuts_lot, -10 * batches)]
    for lot, quantity in lines:
        kit.cur.execute('INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb, created_at) '
                        'SELECT %s, product_id, id, %s, %s FROM lots WHERE id = %s', (tx, quantity, when, lot))
    kit.side_records(tx, 'make', lines, when)
    return tx


@pytest.fixture
def kit(db_cursor):
    return Kit(db_cursor)


def test_reconciliation_checks_a_weighed_make_against_its_pans(kit):
    kit.run(night(-1))
    # 10 pans of 100 lb, weighed at 1,060 lb (+6%), ingredients exactly recipe x 10.
    weighed = heavy_make(kit, 10, 1060, at(0, 9), weighed=True)
    outcome = kit.run(night(0))
    assert not [f for f in kit.flags(outcome) if f['kind'].startswith('MAKE_')]
    stored = next(m for m in kit.row(outcome['run_id'])['summary']['makes'] if m['transaction_id'] == weighed)
    assert stored['batches'] == 10 and stored['output_lb'] == 1060


def test_reconciliation_still_infers_pans_from_output_for_unweighed_makes(kit):
    kit.run(night(-1))
    # The same +6% output without a weighed ticket reads as 10.6 batches: ingredients look short.
    unweighed = heavy_make(kit, 10, 1060, at(0, 9), weighed=False)
    outcome = kit.run(night(0))
    assert 'MAKE_OFF_RECIPE' in {f['kind'] for f in kit.flags(outcome)}
