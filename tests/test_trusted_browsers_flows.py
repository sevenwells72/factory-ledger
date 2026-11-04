"""Selected-person end-to-end flows (091): real FL handlers and the LOCAL test DB.

Receive by each of the four people; the FL Assistant from a selected session
(mocked model only) through prepare_receive -> Record -> DB attribution,
duplicate replay and Change-user refusal; Make/Pack/adjust correction through
the Assistant; Ship through the dashboard ticket routes. No remote service.
"""
from pathlib import Path

import pytest

import main
import trusted_browsers
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_write_tickets import payload, headers  # noqa: F401
from tests.test_write_tickets_part2 import items, body as action_body  # noqa: F401
from tests.test_trusted_browsers import migrated, people, approved, as_person  # noqa: F401
from tests.test_fl_assistant import model, turn, make_draft, record, resolve_product
from tests.test_ship_tickets import shipping, ship_body, PNG  # noqa: F401
from tests.test_order_tickets import create_body

pytestmark = pytest.mark.db
ROOT = Path(__file__).resolve().parents[1]
NAMES = {'michael': 'owner', 'luz': 'office', 'arturo': 'floor', 'miriam': None}


@pytest.fixture(autouse=True)
def selection_on(monkeypatch, db_cursor, tmp_path):
    monkeypatch.setenv('KEY_SESSIONS_ENABLED', '1')
    monkeypatch.setenv('PERSON_SELECT_ENABLED', '1')
    monkeypatch.setenv('ASSISTANT_ENABLED', '1')
    monkeypatch.setenv('EVIDENCE_STORAGE_DIR', str(tmp_path / 'evidence'))
    sql = (ROOT / 'migrations/061_exceptions_tables.sql').read_text()
    seed = sql[sql.index('INSERT INTO correction_reasons'):]
    db_cursor.execute(seed[:seed.index('ON CONFLICT (code) DO NOTHING;') + len('ON CONFLICT (code) DO NOTHING;')])
    trusted_browsers._MIGRATED.clear()
    yield
    trusted_browsers._MIGRATED.clear()


def actor_name(db_cursor, actor_id):
    db_cursor.execute('SELECT name FROM actors WHERE id=%s', (actor_id,))
    return db_cursor.fetchone()['name']


def browser_id(db_cursor, token):
    db_cursor.execute('SELECT id FROM trusted_browsers WHERE secret_hash=%s', (trusted_browsers.digest(token),))
    return db_cursor.fetchone()['id']


def assert_selected_ticket(db_cursor, ticket_id, actor_id, token):
    db_cursor.execute('SELECT actor_id, operator_id, key_kind, identity_via, trusted_browser_id FROM write_tickets WHERE id=%s', (ticket_id,))
    t = db_cursor.fetchone()
    assert (t['actor_id'], t['key_kind'], t['identity_via']) == (actor_id, 'actor', 'selection')
    assert t['operator_id'] == actor_name(db_cursor, actor_id)            # actors.name snapshot (Blubber for id 1 in prod)
    assert t['trusted_browser_id'] == browser_id(db_cursor, token)
    db_cursor.execute('SELECT operator_id, entered_by_actor_id FROM transactions WHERE ticket_id=%s', (ticket_id,))
    rows = db_cursor.fetchall()
    assert rows and all(r['operator_id'] == t['operator_id'] and r['entered_by_actor_id'] == actor_id for r in rows)


@pytest.mark.parametrize('who', ['michael', 'luz', 'arturo', 'miriam'])
def test_receive_by_each_of_the_four_people(client, db_cursor, actors, people, payload, who):
    token = approved(client, actors)
    h = as_person(token, people[who])
    prepared = client.post('/receive/prepare', json=payload, headers=h)
    assert prepared.status_code == 200, prepared.text
    p = prepared.json()
    r = client.post('/tickets/' + p['ticket'] + '/commit', json={'payload_hash': p['payload_hash']}, headers=h)
    assert r.status_code == 200, r.text
    assert_selected_ticket(db_cursor, p['ticket_id'], people[who], token)
    replay = client.post('/tickets/' + p['ticket'] + '/commit', json={'payload_hash': p['payload_hash']}, headers=h)
    assert replay.json()['replayed'] is True and replay.json()['transaction_id'] == r.json()['transaction_id']
    db_cursor.execute('SELECT count(*) AS n FROM transactions WHERE ticket_id=%s', (p['ticket_id'],))
    assert db_cursor.fetchone()['n'] == 1


@pytest.fixture
def selected_chat(client, actors, people):
    token = approved(client, actors)
    h = as_person(token, people['arturo'])
    r = client.post('/assistant/session', headers=h)
    assert r.status_code == 200, r.text
    assert r.json()['actor']['id'] == people['arturo']
    return r.json()['session_id'], h, token


def test_assistant_selected_receive_records_attribution_replay_and_refuses_change_user(
        client, db_cursor, actors, people, payload, selected_chat, monkeypatch):
    sid, h, token = selected_chat
    card, _ = make_draft(client, (sid, h), payload, db_cursor, monkeypatch)
    assert card['prepared']['draft']['actor']['via'] == 'selection'
    # Change user before Record: Luz cannot record Arturo's draft or read his chat.
    luz = as_person(token, people['luz'])
    stolen = client.post('/assistant/record', headers=luz, json={'draft_id': card['id']})
    assert stolen.status_code in (403, 404), stolen.text
    peek = client.post('/assistant/resume', headers=luz, json={'session_id': sid})
    assert peek.status_code in (403, 404), peek.text
    first = record(client, (sid, h), card)
    assert first.status_code == 200, first.text
    result = first.json()['result']
    assert result['receipt_number'].startswith('RCV-')
    db_cursor.execute('SELECT ticket FROM assistant_drafts WHERE id=%s', (card['id'],))
    db_cursor.execute('SELECT id, client_source FROM write_tickets WHERE ticket_hash=%s',
                      (main.write_tickets.token_hash(db_cursor.fetchone()['ticket']),))
    ticket = db_cursor.fetchone()
    assert ticket['client_source'] == 'fl_assistant'
    assert_selected_ticket(db_cursor, ticket['id'], people['arturo'], token)
    again = record(client, (sid, h), card)
    assert again.json()['result'] == result | {'replayed': True}
    db_cursor.execute('SELECT count(*) AS n FROM transactions WHERE ticket_id=%s', (ticket['id'],))
    assert db_cursor.fetchone()['n'] == 1
    # After a real Change user, Luz's replay of the same Record is still refused.
    assert client.post('/assistant/record', headers=luz, json={'draft_id': card['id']}).status_code in (403, 404)


@pytest.mark.parametrize('action', ['make', 'pack', 'adjust'])
def test_assistant_selected_make_pack_and_correction(client, db_cursor, actors, people, items, selected_chat, monkeypatch, action):
    sid, h, token = selected_chat
    chat = (sid, h)
    body = action_body(action, items)
    body.pop('reason', None)
    if action == 'adjust':
        body['reason_code'] = 'physical_count'
    calls = [resolve_product(db_cursor, body[f]) for f in ('product_id', 'source_product_id', 'target_product_id') if f in body]
    if 'lot_id' in body:
        calls.append(('resolve', {'kind': 'lot', 'query': items['ingredient']['lot_code']}))
    calls.append(('prepare_' + action, body))
    model(monkeypatch, *calls)
    card = turn(client, chat, action).json()['cards'][0]
    if action == 'pack':
        model(monkeypatch, ('prepare_pack', body))
        card = turn(client, chat, text='', choice_id=card['id'], selected_id=items['batch']['lot_id']).json()['cards'][0]
    assert card['kind'] == 'draft', card
    confirmations = [{'lot_id': i['lot_id'], 'method': 'full_code', 'value': i['lot_code']}
                     for i in card['prepared']['draft'].get('input_plan', [])] if action in ('make', 'pack') else []
    committed = record(client, chat, card, lot_confirmations=confirmations)
    assert committed.status_code == 200, committed.text
    assert_selected_ticket(db_cursor, card['prepared']['ticket_id'], people['arturo'], token)


def test_assistant_selected_office_person_cannot_make(client, actors, people, items, monkeypatch, db_cursor):
    token = approved(client, actors)
    h = as_person(token, people['miriam'])
    sid = client.post('/assistant/session', headers=h).json()['session_id']
    r = client.post('/make/prepare', headers=h, json=action_body('make', items))
    assert r.status_code == 403 and r.json()['detail']['error_code'] == 'ROLE_NOT_ALLOWED'
    assert sid


def test_ship_order_by_selected_luz(client, db_cursor, actors, people, shipping):
    token = approved(client, actors)
    h = as_person(token, people['luz'])
    photo = client.post('/attachments', data={'kind': 'bol'}, files={'file': ('proof.png', PNG, 'image/png')}, headers=h)
    assert photo.status_code == 200, photo.text
    order = client.post('/sales/orders/prepare', json=create_body(shipping), headers=h).json()
    created = client.post('/tickets/' + order['ticket'] + '/commit', json={'payload_hash': order['payload_hash']}, headers=h)
    assert created.status_code == 200, created.text
    body = ship_body(shipping, attachment_ids=[photo.json()['id']], ship_all=True)
    p = client.post(f"/sales/orders/{created.json()['order_id']}/ship/prepare", json=body, headers=h)
    assert p.status_code == 200, p.text
    r = client.post('/tickets/' + p.json()['ticket'] + '/commit', json={'payload_hash': p.json()['payload_hash']}, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()['receipt_number'].startswith('SHP-')
    assert_selected_ticket(db_cursor, p.json()['ticket_id'], people['luz'], token)
    db_cursor.execute('SELECT entered_by_actor_id FROM shipments WHERE ticket_id=%s', (p.json()['ticket_id'],))
    assert db_cursor.fetchone()['entered_by_actor_id'] == people['luz']
    replay = client.post('/tickets/' + p.json()['ticket'] + '/commit', json={'payload_hash': p.json()['payload_hash']}, headers=h)
    assert replay.json()['replayed'] is True
    db_cursor.execute('SELECT count(*) AS n FROM shipments WHERE ticket_id=%s', (p.json()['ticket_id'],))
    assert db_cursor.fetchone()['n'] == 1
