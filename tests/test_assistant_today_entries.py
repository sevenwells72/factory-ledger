"""Fast, complete Today reads: real day boundaries, pagination and effective lines."""
from datetime import datetime, time, timedelta
from hashlib import sha256
from time import perf_counter
from uuid import uuid4

import pytest
from psycopg2.extras import Json, execute_values

import main
import write_tickets as tickets
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_write_tickets import commit, headers, payload, prepare  # noqa: F401

pytestmark = pytest.mark.db


def seed_day(cur, actors, count=601, *, day=None, happened_at=None, entered_at=None):
    day = day or main.get_plant_now().date()
    start = datetime.combine(day, time(8), main.PLANT_TIMEZONE)
    tag = uuid4().hex.upper()
    cur.execute("INSERT INTO products(name,odoo_code,type,uom) VALUES (%s,%s,'ingredient','lb') RETURNING id",
                ('Sunflower Seeds ' + tag, 'FA-' + tag))
    product_id = cur.fetchone()['id']
    cur.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id', (product_id, 'FA-' + tag))
    lot_id = cur.fetchone()['id']
    rows = []
    for index in range(count):
        actor = actors['floor' if index % 2 == 0 else 'office']
        happened = happened_at or start + timedelta(seconds=index)
        entered = entered_at or start + timedelta(seconds=index)
        number = f'RCV-{day:%y%m%d}-{900000 + index}'
        digest = sha256(f'{tag}-{index}'.encode()).hexdigest()
        draft = {'actor': {key: actor[key] for key in ('id', 'name', 'role')},
                 'product_name': 'Sunflower Seeds ' + tag, 'total_lb': 50, 'cases': 2, 'case_size_lb': 25}
        rows.append((digest, 'receive', actor['id'], actor['name'], 'actor', 'fl_assistant',
                     Json({'occurred_at': happened.isoformat()}), digest, digest, Json(draft),
                     'committed', entered + timedelta(minutes=10), entered, number,
                     Json({'product_name': draft['product_name'], 'total_lb': 50})))
    execute_values(cur, '''INSERT INTO write_tickets(ticket_hash,action,actor_id,operator_id,key_kind,
        client_source,payload,payload_hash,state_hash,draft,status,expires_at,committed_at,receipt_number,response)
        VALUES %s''', rows)
    cur.execute('''INSERT INTO transactions(type,occurred_at,business_date,operator_id,receipt_number,ticket_id)
        SELECT 'receive',(payload->>'occurred_at')::timestamptz,%s,operator_id,receipt_number,id
        FROM write_tickets WHERE ticket_hash=ANY(%s) RETURNING id,ticket_id''',
        (day, [row[0] for row in rows]))
    txns = cur.fetchall()
    execute_values(cur, 'INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES %s',
                   [(row['id'], product_id, lot_id, 50) for row in txns])
    return [row[13] for row in rows]


class CountingCursor:
    def __init__(self, cursor):
        self.cursor = cursor
        self.calls = 0

    def execute(self, *args):
        self.calls += 1
        return self.cursor.execute(*args)

    def fetchall(self):
        return self.cursor.fetchall()


def test_today_all_601_entries_newest_first_across_pages_without_actor_filter(client, db_cursor, actors):
    expected = seed_day(db_cursor, actors)
    start = perf_counter()
    first = client.get('/receipts?summary=true&limit=500', headers=headers(actors['floor']['key']))
    assert first.status_code == 200, first.text
    page = first.json()
    assert [row['receipt_number'] for row in page['receipts']] == expected[::-1][:500]
    assert {row['actor']['name'] for row in page['receipts']} == {actors['floor']['name'], actors['office']['name']}
    assert page['has_more'] is True and page['next_offset'] == 500
    second = client.get('/receipts?summary=true&limit=500&offset=500', headers=headers(actors['floor']['key']))
    assert second.status_code == 200, second.text
    tail = second.json()
    assert [row['receipt_number'] for row in tail['receipts']] == expected[::-1][500:]
    assert tail['has_more'] is False and tail['next_offset'] is None
    assert perf_counter() - start < 5
    assert page['receipts'][0]['transactions'][0]['lines'][0]['quantity_lb'] == 50


def test_today_summary_query_count_stays_two_for_601_entries(db_cursor, actors, monkeypatch):
    seed_day(db_cursor, actors)
    counted = CountingCursor(db_cursor)
    monkeypatch.setattr(tickets, 'receipt_detail', lambda *args: pytest.fail('Today must not load individual receipt details'))
    result = tickets.receipt_summaries(main, counted, main.get_plant_now().date())
    assert len(result['receipts']) == 500
    assert counted.calls == 2


def test_today_empty_summary_has_no_extra_queries(db_cursor):
    counted = CountingCursor(db_cursor)
    day = main.get_plant_now().date()
    assert tickets.receipt_summaries(main, counted, day) == {
        'date': day, 'receipts': [], 'has_more': False, 'next_offset': None}
    assert counted.calls == 1


def test_today_summary_keeps_actor_action_status_and_source_filters(client, db_cursor, actors):
    expected = seed_day(db_cursor, actors, 6)
    params = {'summary': 'true', 'actor': actors['floor']['id'], 'action': 'receive',
              'status': 'committed', 'client_source': 'fl_assistant'}
    result = client.get('/receipts', params=params, headers=headers()).json()
    assert [row['receipt_number'] for row in result['receipts']] == expected[::2][::-1]
    for field, value in [('actor', 'nobody'), ('action', 'make'), ('status', 'rejected'), ('client_source', 'dashboard')]:
        assert client.get('/receipts', params=params | {field: value}, headers=headers()).json()['receipts'] == []


@pytest.mark.parametrize('happened,entered,included,late', [
    ('2026-11-01T00:00:00-04:00', '2026-11-02T00:00:00-05:00', True, True),
    ('2026-10-31T23:59:59-04:00', '2026-11-01T23:59:59-05:00', True, True),
    ('2026-11-01T01:30:00-04:00', '2026-11-01T01:30:00-05:00', True, False),
    ('2026-10-31T23:59:59-04:00', '2026-11-02T00:00:00-05:00', False, None),
])
def test_today_summary_uses_new_york_day_on_both_clocks_across_dst(client, db_cursor, actors,
                                                                 happened, entered, included, late):
    day = datetime(2026, 11, 1).date()
    expected = seed_day(db_cursor, actors, 1, day=day, happened_at=datetime.fromisoformat(happened),
                        entered_at=datetime.fromisoformat(entered))
    response = client.get('/receipts', params={'summary': 'true', 'date': day.isoformat()}, headers=headers())
    assert response.status_code == 200, response.text
    entries = response.json()['receipts']
    assert [row['receipt_number'] for row in entries] == (expected if included else [])
    if included:
        assert entries[0]['late_entry'] is late


def test_today_equal_entry_times_use_newest_ticket_first(client, db_cursor, actors):
    entered = datetime.combine(main.get_plant_now().date(), time(8), main.PLANT_TIMEZONE)
    expected = seed_day(db_cursor, actors, 3, entered_at=entered)
    result = client.get('/receipts?summary=true', headers=headers()).json()
    assert [row['receipt_number'] for row in result['receipts']] == expected[::-1]


def test_today_effective_void_status_and_amended_lines_without_full_receipt(client, db_cursor, payload):
    receipt = commit(client, prepare(client, payload)).json()
    db_cursor.execute('SELECT id FROM transaction_lines WHERE transaction_id=%s', (receipt['transaction_id'],))
    line_id = db_cursor.fetchone()['id']
    db_cursor.execute('''INSERT INTO ledger_corrections(target_table,target_id,event_type,previous_values,replacement_values,reason)
        VALUES ('transaction_lines',%s,'amend','{}','{"quantity_lb":42}','Test effective quantity'),
               ('transactions',%s,'void','{}','{}','Test void status')''', (line_id, receipt['transaction_id']))
    result = client.get('/receipts?summary=true', headers=headers()).json()['receipts'][0]
    assert result['effective_status'] == 'voided'
    assert result['transactions'][0]['effective_status'] == 'voided'
    assert result['transactions'][0]['lines'][0]['quantity_lb'] == 42
    assert 'allowed_actions' not in result['transactions'][0]  # live permissions belong to the full receipt
    assert 'lots' not in result


@pytest.mark.parametrize('params', ['limit=0', 'limit=501', 'offset=-1'])
def test_today_page_bounds_are_validated(client, params):
    response = client.get('/receipts?summary=true&' + params, headers=headers())
    assert response.status_code == 422
