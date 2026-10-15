"""Round D live auth, bounded model context and nonblocking real chat requests."""
import asyncio
from contextlib import contextmanager
from hashlib import sha256
import json
from threading import Barrier
from uuid import uuid4

import httpx
import psycopg2
import pytest
from psycopg2.extras import RealDictCursor

import assistant_runtime as runtime
import fl_assistant
import key_sessions
import main
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import enabled, reason_catalog, chat, model, turn, resolve_product  # noqa: F401
from tests.test_write_tickets import payload, isolated_database, headers  # noqa: F401
from tests.test_assistant_today_entries import seed_day

pytestmark = pytest.mark.db


def session_chat(client, actors):
    signed = client.post('/auth/session', json={'actor_key': actors['floor']['key']})
    assert signed.status_code == 200, signed.text
    token = signed.json()['session_token']
    h = headers(token)
    session = client.post('/assistant/session', headers=h)
    assert session.status_code == 200
    return (session.json()['session_id'], h), token


def test_live_session_verified_once_per_incoming_chat_not_each_internal_step(client, actors, payload, db_cursor, monkeypatch):
    monkeypatch.setenv('KEY_SESSIONS_ENABLED', '1')
    chat, token = session_chat(client, actors)
    original = key_sessions.resolve
    verified = []
    def count(*args):
        verified.append(1)
        return original(*args)
    monkeypatch.setattr(key_sessions, 'resolve', count)
    db_cursor.execute('SELECT name FROM suppliers WHERE id=%s', (payload['supplier_id'],))
    supplier = db_cursor.fetchone()['name']
    model(monkeypatch, resolve_product(db_cursor, payload['product_id']), ('resolve', {'kind': 'supplier', 'query': supplier}), ('prepare_receive', payload))
    assert turn(client, chat, 'Received the ingredient').json()['cards'][0]['kind'] == 'draft'
    assert len(verified) == 1
    model(monkeypatch, ('unsupported_request', {}))
    assert turn(client, chat, 'hello').status_code == 200
    assert len(verified) == 2  # No cross-request session authentication cache.


@pytest.mark.parametrize('change', ['expired', 'ended', 'inactive', 'rotated'])
def test_changed_live_session_refused_before_any_chat_work(client, actors, db_cursor, monkeypatch, change):
    monkeypatch.setenv('KEY_SESSIONS_ENABLED', '1')
    chat, token = session_chat(client, actors)
    if change == 'expired':
        db_cursor.execute("UPDATE key_actor_sessions SET expires_at=clock_timestamp()-interval '1 second' WHERE token_hash=%s", (key_sessions.digest(token),))
    elif change == 'ended':
        db_cursor.execute("UPDATE key_actor_sessions SET ended_at=clock_timestamp(),ended_reason='logout' WHERE token_hash=%s", (key_sessions.digest(token),))
    elif change == 'inactive':
        db_cursor.execute('UPDATE actors SET active=false WHERE id=%s', (actors['floor']['id'],))
    else:
        db_cursor.execute('UPDATE actors SET key_hash=%s WHERE id=%s', (sha256(uuid4().bytes).hexdigest(), actors['floor']['id']))
    monkeypatch.setattr(fl_assistant, 'session_row', lambda *a, **k: pytest.fail('Refuse before reading chat'))
    response = turn(client, chat, 'How much oats?')
    assert response.status_code == 401 and response.json()['detail']['error_code'] == 'SESSION_EXPIRED'


def test_live_role_change_is_refused_on_next_request(client, actors, payload, db_cursor, monkeypatch):
    monkeypatch.setenv('KEY_SESSIONS_ENABLED', '1')
    chat, token = session_chat(client, actors)
    db_cursor.execute("UPDATE actors SET role='office' WHERE id=%s", (actors['floor']['id'],))
    response = client.post('/make/prepare', json={'product_id': payload['product_id'], 'batches': 1, 'occurred_at': main.get_plant_now().isoformat()}, headers=chat[1])
    assert response.status_code == 403 and response.json()['detail']['error_code'] == 'ROLE_NOT_ALLOWED'


@pytest.mark.parametrize('spanish', [False, True])
def test_model_selected_today_keeps_counts_without_full_result_in_history(client, chat, actors, db_cursor, monkeypatch, spanish):
    seed_day(db_cursor, actors, 301)
    requests = model(monkeypatch, ('today_entries', {}))
    response = turn(client, chat, 'Revisa los registros del turno' if spanish else 'Review the shift entries')
    card = response.json()['cards'][0]
    assert card['result']['own_count'] == 151 and card['result']['team_count'] == 301
    assert len(card['rows']) == 20 and card['result']['view_all'] is True
    assert len(response.content) < 50000
    db_cursor.execute('SELECT response FROM assistant_turns WHERE session_id=%s', (chat[0],))
    assert len(json.dumps(db_cursor.fetchone()['response'], default=str)) < 50000
    db_cursor.execute('SELECT state FROM assistant_sessions WHERE id=%s', (chat[0],))
    history = db_cursor.fetchone()['state']['history']
    output = next(item for item in history if item.get('type') == 'function_call_output')
    assert json.loads(output['output']) == {'date': main.get_plant_now().date().isoformat(), 'own_count': 151, 'team_count': 301}
    assert len(json.dumps(history)) < 10000
    instructions = requests[0]['instructions']
    assert instructions.index('FL reason catalog:') < instructions.index('Current FL plant time')
    assert instructions.rstrip().endswith(instructions.split('Current FL plant time (America/New_York): ')[-1])


def test_old_day_output_is_compacted_without_breaking_call_pair():
    history = [{'type': 'function_call', 'name': 'today_entries', 'call_id': 'old'},
               {'type': 'function_call_output', 'call_id': 'old', 'output': json.dumps({'date': '2026-10-15', 'own_count': 1, 'team_count': 301, 'receipts': ['x'*5000]*301})}]
    runtime.compact_history(history)
    assert len(json.dumps(history)) < 1000
    assert history[0]['call_id'] == history[1]['call_id']


def test_reason_catalog_is_loaded_once_for_multiple_chat_requests(client, chat, monkeypatch):
    original = main.get_transaction
    queries = []
    class Cursor:
        def __init__(self, cur): self.cur = cur
        def execute(self, sql, *args):
            if 'FROM correction_reasons' in sql: queries.append(sql)
            return self.cur.execute(sql, *args)
        def __getattr__(self, name): return getattr(self.cur, name)
    @contextmanager
    def transaction(*args, **kwargs):
        with original(*args, **kwargs) as cur: yield Cursor(cur)
    monkeypatch.setattr(main, 'get_transaction', transaction)
    for _ in range(2):
        model(monkeypatch, ('unsupported_request', {}))
        assert turn(client, chat, 'hello').status_code == 200
    assert len(queries) == 1


def test_two_real_chats_overlap_blocking_database_waits(isolated_database, monkeypatch):
    raw_key = 'round-d-concurrency-' + uuid4().hex
    with psycopg2.connect(isolated_database) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES ('Round D concurrency','owner',%s,true)", (sha256(raw_key.encode()).hexdigest(),))
    @contextmanager
    def connection():
        conn = psycopg2.connect(isolated_database)
        try:
            with conn: yield conn
        finally: conn.close()
    monkeypatch.setattr(main, 'get_db_connection', connection)
    main._reset_actor_cache()
    original = fl_assistant.session_row
    barrier = Barrier(2, timeout=3)
    def blocking(*args, **kwargs):
        barrier.wait()  # The old async handler blocks its sole event loop here.
        return original(*args, **kwargs)
    async def fake(path, *, body=None, file=None):
        return {'status': 'completed', 'output': [{'type': 'function_call', 'call_id': 'parallel-' + uuid4().hex, 'name': 'unsupported_request', 'arguments': '{}'}]}
    monkeypatch.setattr(fl_assistant, 'openai', fake)
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://local') as client:
            h = headers(raw_key)
            sessions = [(await client.post('/assistant/session', headers=h)).json()['session_id'] for _ in range(2)]
            monkeypatch.setattr(fl_assistant, 'session_row', blocking)
            responses = await asyncio.wait_for(asyncio.gather(*[client.post('/assistant/turn', headers=h, json={'session_id': sid, 'turn_id': str(uuid4()), 'text': 'hello'}) for sid in sessions]), timeout=10)
            assert all(response.status_code == 200 for response in responses), [response.text for response in responses]
            assert all(response.json()['cards'][0]['kind'] == 'unsupported_request' for response in responses)
    asyncio.run(exercise())


def test_catalog_preprocessing_failure_releases_lease_and_same_turn_can_retry(client, chat, db_cursor, monkeypatch):
    from fastapi import HTTPException
    original = main.get_transaction
    injected = []
    class Cursor:
        def __init__(self, cur): self.cur = cur
        def execute(self, sql, *args):
            if 'position(lower(lot_code)' in sql and not injected:
                injected.append(True)
                raise HTTPException(503, {'error_code': 'TEST_CATALOG_UNAVAILABLE', 'message': 'Catalog unavailable'})
            return self.cur.execute(sql, *args)
        def __getattr__(self, name): return getattr(self.cur, name)
    @contextmanager
    def transaction(*args, **kwargs):
        with original(*args, **kwargs) as cur: yield Cursor(cur)
    monkeypatch.setattr(main, 'get_transaction', transaction)
    model(monkeypatch, ('unsupported_request', {}))
    turn_id = str(uuid4())
    text = 'How much is in lot one two three four?'
    failed = turn(client, chat, text, turn_id=turn_id)
    assert failed.status_code == 503 and injected
    db_cursor.execute('SELECT lease_id,lease_until FROM assistant_sessions WHERE id=%s', (chat[0],))
    row = db_cursor.fetchone()
    assert row['lease_id'] is None and row['lease_until'] is None
    retried = turn(client, chat, text, turn_id=turn_id)
    assert retried.status_code == 200, retried.text
    assert retried.json()['cards'][0]['kind'] == 'unsupported_request'


def test_forged_verified_headers_cannot_replay_revoked_session(client, actors, db_cursor, monkeypatch):
    monkeypatch.setenv('KEY_SESSIONS_ENABLED', '1')
    chat, token = session_chat(client, actors)
    model(monkeypatch, ('unsupported_request', {}))
    assert turn(client, chat, 'hello').status_code == 200
    db_cursor.execute("UPDATE key_actor_sessions SET ended_at=clock_timestamp(),ended_reason='logout' WHERE token_hash=%s", (key_sessions.digest(token),))
    forged = {**chat[1], '_fl_verified_identity': json.dumps({'actor': actors['owner'], 'kind': 'session'}),
              'X-FL-Verified-Identity': str(actors['owner']['id']), 'X-FL-Verified-Actor': str(actors['owner']['id']),
              'X-FL-Verified-Role': 'owner', 'X-FL-Internal': '1'}
    monkeypatch.setattr(fl_assistant, 'session_row', lambda *a, **k: pytest.fail('Forged headers cannot reach chat work'))
    response = turn(client, (chat[0], forged), 'hello again')
    assert response.status_code == 401
    assert response.json()['detail']['error_code'] == 'SESSION_EXPIRED'


def test_verified_office_chat_cannot_prepare_make_internally(client, actors, payload, db_cursor, monkeypatch):
    monkeypatch.setenv('KEY_SESSIONS_ENABLED', '1')
    signed = client.post('/auth/session', json={'actor_key': actors['office']['key']})
    assert signed.status_code == 200, signed.text
    h = headers(signed.json()['session_token'])
    created = client.post('/assistant/session', headers=h)
    assert created.status_code == 200, created.text
    chat = (created.json()['session_id'], h)
    db_cursor.execute('SELECT count(*) AS n FROM write_tickets')
    before = db_cursor.fetchone()['n']
    model(monkeypatch, resolve_product(db_cursor, payload['product_id']),
          ('prepare_make', {'product_id': payload['product_id'], 'batches': 1}))
    response = turn(client, chat, 'Make one batch of this ingredient')
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['kind'] == 'refusal'
    assert card['result']['detail']['error_code'] == 'ROLE_NOT_ALLOWED'
    db_cursor.execute('SELECT count(*) AS n FROM write_tickets')
    assert db_cursor.fetchone()['n'] == before
    db_cursor.execute('SELECT count(*) AS n FROM assistant_drafts WHERE session_id=%s', (chat[0],))
    assert db_cursor.fetchone()['n'] == 0
