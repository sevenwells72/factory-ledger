"""One-query candidate balances must preserve the old FIFO/traceability result."""
from datetime import datetime, timedelta, timezone
import json

from fastapi import HTTPException
import pytest

import main
import ticket_actions as actions
import write_tickets
from tests.test_write_tickets_part2 import items  # noqa: F401

pytestmark = pytest.mark.db


class CountCursor:
    def __init__(self, cursor):
        self.cursor, self.queries = cursor, 0

    def execute(self, *args, **kwargs):
        self.queries += 1
        return self.cursor.execute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.cursor, name)


def prior_choose_inputs(api, cur, requirements, overrides):
    """Frozen before implementation: same reads and policy as reviewed C head."""
    plan = []
    for pid, needed in sorted(requirements.items()):
        if pid in overrides:
            rows = [actions.lot(api, cur, overrides[pid], pid)]
        else:
            cur.execute("""SELECT id FROM lots WHERE product_id=%s AND status IS DISTINCT FROM 'merged'
                           ORDER BY COALESCE(received_at,created_at),id""", (pid,))
            rows = [actions.lot(api, cur, row['id'], pid) for row in cur.fetchall()]
        remaining = needed
        for row in rows:
            take = min(max(0, float(row['on_hand_lb'])), remaining)
            if take > api.BALANCE_EPSILON:
                plan.append({'product_id': pid, 'lot_id': row['id'], 'quantity_lb': take})
                remaining -= take
        if remaining > api.BALANCE_EPSILON:
            if actions.a3b.assign_shortfall(plan, pid, remaining, rows, overrides.get(pid)) is None:
                name = actions.product(api, cur, pid)['name']
                actions.fail('INSUFFICIENT_STOCK', f'{name} needs {float(needed)} lb and has no lot to record the shortage against; receive it first.')
    return plan


@pytest.fixture
def varied_lots(db_cursor, items):
    cur = db_cursor
    pid = items['ingredient']['id']
    now = datetime.now(timezone.utc)
    result = {'product_id': pid, 'base': items['ingredient']['lot_id']}
    for index, (key, amount, voided, merged) in enumerate([
        ('first', 15, False, False), ('tie', 25, False, False),
        ('negative', -5, False, False), ('empty', 0, False, False),
        ('voided', 80, True, False), ('merged', 100, False, True),
    ]):
        # first/tie share the date and use the ID tie-breaker from the old code.
        when = now - timedelta(days=5 if key in ('first', 'tie') else 2)
        cur.execute('INSERT INTO lots(product_id,lot_code,received_at) VALUES (%s,%s,%s) RETURNING id',
                    (pid, 'D-LOT-' + key.upper(), when))
        lid = cur.fetchone()['id']
        result[key] = lid
        if amount:
            cur.execute("INSERT INTO transactions(type,notes,status) VALUES ('receive','D candidate batching',%s) RETURNING id", ('voided' if voided else 'posted',))
            tid = cur.fetchone()['id']
            cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,%s)', (tid,pid,lid,amount))
        if merged:
            cur.execute("UPDATE lots SET status='merged',merged_into_lot_id=%s WHERE id=%s", (result['base'],lid))
    return result


@pytest.mark.parametrize('needed', [10, 35, 125, 175])
@pytest.mark.parametrize('override', [False, True])
def test_batch_candidate_query_preserves_fifo_and_shortfall(db_cursor, varied_lots, needed, override):
    pid = varied_lots['product_id']
    overrides = {pid: varied_lots['tie']} if override else {}
    before, after = CountCursor(db_cursor), CountCursor(db_cursor)
    expected = prior_choose_inputs(main, before, {pid: needed}, overrides)
    actual = actions.choose_inputs(main, after, {pid: needed}, overrides)
    assert actual == expected
    assert after.queries == 1
    assert before.queries > after.queries
    assert all(item['lot_id'] not in (varied_lots['merged'],varied_lots['voided']) for item in actual)


def test_all_ingredient_candidates_share_one_balance_query(db_cursor, varied_lots, items):
    reqs = {varied_lots['product_id']: 30, items['batch']['id']: 40}
    expected = prior_choose_inputs(main, db_cursor, reqs, {})
    cur = CountCursor(db_cursor)
    assert actions.choose_inputs(main, cur, reqs, {}) == expected
    assert cur.queries == 1


@pytest.mark.parametrize('kind', ['missing', 'merged', 'wrong_product'])
def test_override_refusals_match_before_batching(db_cursor, varied_lots, items, kind):
    pid = varied_lots['product_id']
    lid = {'missing': 2147483647, 'merged': varied_lots['merged'], 'wrong_product': items['batch']['lot_id']}[kind]
    errors = []
    for select in (prior_choose_inputs, actions.choose_inputs):
        with pytest.raises(HTTPException) as error:
            select(main, db_cursor, {pid: 10}, {pid: lid})
        errors.append((error.value.status_code,error.value.detail))
    assert errors[0] == errors[1]


def test_make_draft_confirmation_and_fifo_evidence_are_identical(db_cursor, varied_lots, items, monkeypatch):
    payload = json.loads(write_tickets.MakePrepareRequest(product_id=items['batch']['id'], batches=4,
        occurred_at=main.get_plant_now(), confirmed_sku=True).model_dump_json(exclude={'client_source'}))
    current = actions.choose_inputs
    monkeypatch.setattr(actions, 'choose_inputs', prior_choose_inputs)
    before = actions.validate(main, db_cursor, 'make', payload)
    monkeypatch.setattr(actions, 'choose_inputs', current)
    after = actions.validate(main, db_cursor, 'make', payload)
    assert after[7] == before[7]
    assert after[6] == before[6]
    for field in ('input_plan','blockers','lot_confirmation_group','shortages'):
        assert after[0][field] == before[0][field]
    assert after[0]['blockers']
    assert all(not entry['confirmed'] for entry in after[7])
    assert all(not lot['confirmed'] for ingredient in after[0]['lot_confirmation_group']['ingredients'] for lot in ingredient['suggested_lots'])
    with pytest.raises(HTTPException) as error:
        actions.validate(main, db_cursor, 'make', payload, lock=True)
    assert error.value.detail['error_code'] == 'LOT_NOT_CONFIRMED'
    bad = dict(payload, lot_confirmations=[{'lot_id':after[7][0]['lot_id'],'method':'full_code','value':'WRONG-TAG'}])
    with pytest.raises(HTTPException) as error:
        actions.validate(main, db_cursor, 'make', bad)
    assert error.value.detail['error_code'] == 'LOT_CONFIRMATION_MISMATCH'


def test_empty_requirements_do_not_query_database(db_cursor):
    cur = CountCursor(db_cursor)
    assert actions.choose_inputs(main, cur, {}, {}) == []
    assert cur.queries == 0
