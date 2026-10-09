"""Real FL handlers/DB, mocked OpenAI only. No remote ledger or API key required."""
import asyncio
import copy
import json
from uuid import uuid4
from pathlib import Path

import pytest
from fastapi import HTTPException

import assistant_tools
import fl_assistant
import main
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_write_tickets import payload, headers  # noqa: F401
from tests.test_write_tickets import isolated_database, seed  # noqa: F401
from tests.test_write_tickets_part2 import items, body as action_body  # noqa: F401

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv('ASSISTANT_ENABLED', '1')


@pytest.fixture(autouse=True)
def reason_catalog(db_cursor):
    # Schema-only snapshots deliberately contain no seeds. Use A3a's migration.
    sql = (Path(__file__).resolve().parents[1] / 'migrations/061_exceptions_tables.sql').read_text()
    seed = sql[sql.index('INSERT INTO correction_reasons'):]
    db_cursor.execute(seed[:seed.index('ON CONFLICT (code) DO NOTHING;') + len('ON CONFLICT (code) DO NOTHING;')])


@pytest.fixture
def chat(client, actors):
    h = headers(actors['floor']['key'])
    r = client.post('/assistant/session', headers=h)
    assert r.status_code == 200, r.text
    return r.json()['session_id'], h


def model(monkeypatch, *calls):
    queue = list(calls)
    requests = []

    async def fake(path, *, body=None, file=None):
        assert path == 'responses'
        assert body['tool_choice'] == 'required'
        assert body['parallel_tool_calls'] is False
        assert body['store'] is False
        assert 'Current FL plant time (America/New_York):' in body['instructions']
        assert all(t['type'] == 'function' and t['strict'] for t in body['tools'])
        assert not any('commit' in t['name'] or 'chat' in t['name'] for t in body['tools'])
        requests.append(copy.deepcopy(body))
        call = queue.pop(0)
        if isinstance(call, Exception):
            raise call
        if isinstance(call, dict):
            return call
        name, args = call
        return {'status': 'completed', 'output': [
            {'type': 'message', 'content': [{'type': 'output_text', 'text': 'RECORDED! MK-261008-999'}]},
            {'type': 'function_call', 'call_id': 'call_' + uuid4().hex, 'name': name, 'arguments': json.dumps(args)}]}

    monkeypatch.setattr(fl_assistant, 'openai', fake)
    return requests


def turn(client, chat, text='test', **extra):
    sid, h = chat
    return client.post('/assistant/turn', headers=h, json={
        'session_id': sid, 'turn_id': str(uuid4()), 'text': text, **extra})


def resolve_product(cur, pid):
    cur.execute('SELECT name FROM products WHERE id=%s', (pid,))
    return ('resolve', {'kind': 'product', 'query': cur.fetchone()['name']})


def make_draft(client, chat, payload, db_cursor, monkeypatch):
    calls = [resolve_product(db_cursor, payload['product_id'])]
    db_cursor.execute('SELECT name FROM suppliers WHERE id=%s', (payload['supplier_id'],))
    calls += [('resolve', {'kind': 'supplier', 'query': db_cursor.fetchone()['name']}), ('prepare_receive', payload)]
    reqs = model(monkeypatch, *calls)
    r = turn(client, chat, 'receive test stock')
    assert r.status_code == 200, r.text
    card = r.json()['cards'][0]
    assert card['kind'] == 'draft', card
    return card, reqs


def record(client, chat, card, **extra):
    return client.post('/assistant/record', headers=chat[1], json={'draft_id': card['id'], **extra})


def test_forced_tools_real_prepare_exact_record_replay_and_resume(client, chat, payload, db_cursor, monkeypatch):
    card, reqs = make_draft(client, chat, payload, db_cursor, monkeypatch)
    assert 'ticket' not in card['prepared'] and 'payload_hash' not in card['prepared']
    assert 'RECORDED!' not in json.dumps(card)
    assert 'wt_' not in json.dumps(reqs)
    db_cursor.execute('SELECT * FROM assistant_drafts WHERE id=%s', (card['id'],))
    draft = db_cursor.fetchone()
    db_cursor.execute('SELECT * FROM write_tickets WHERE ticket_hash=%s', (main.write_tickets.token_hash(draft['ticket']),))
    ticket = db_cursor.fetchone()
    assert ticket['client_source'] == 'fl_assistant'
    assert ticket['status'] == 'prepared'
    assert ticket['expires_at'] - ticket['prepared_at'] == __import__('datetime').timedelta(minutes=10)
    first = record(client, chat, card)
    assert first.status_code == 200, first.text
    assert first.json()['kind'] == 'receipt'
    result = first.json()['result']
    again = record(client, chat, card)
    assert again.json()['result'] == result | {'replayed': True}
    db_cursor.execute('SELECT count(*) AS n FROM transactions WHERE ticket_id=%s', (ticket['id'],))
    assert db_cursor.fetchone()['n'] == 1
    resume = client.post('/assistant/resume', headers=chat[1], json={'session_id': chat[0]})
    assert resume.json()['drafts'][0]['result'] == again.json()['result']
    assert 'wt_' not in resume.text and 'payload_hash' not in resume.text


@pytest.mark.parametrize('action', ['make', 'pack', 'adjust', 'found'])
def test_each_action_uses_real_fl_prepare_and_commit(client, chat, items, db_cursor, monkeypatch, action):
    payload = action_body(action, items)
    payload.pop('reason', None)
    if action == 'adjust':
        payload['reason_code'] = 'physical_count'
    calls = []
    for field in ('product_id', 'source_product_id', 'target_product_id'):
        if field in payload:
            calls.append(resolve_product(db_cursor, payload[field]))
    if 'lot_id' in payload:
        calls.append(('resolve', {'kind': 'lot', 'query': items['ingredient']['lot_code']}))
    calls.append(('prepare_' + action, payload))
    model(monkeypatch, *calls)
    r = turn(client, chat, action)
    assert r.status_code == 200, r.text
    card = r.json()['cards'][0]
    assert card['kind'] == 'draft', card
    confirmations = []
    if action in ('make', 'pack'):
        assert not card['prepared']['can_commit']
        assert {b['code'] for b in card['prepared']['blockers']} == {'LOT_NOT_CONFIRMED'}
        denied = record(client, chat, card)
        assert denied.status_code == 422
        assert denied.json()['result']['detail']['error_code'] == 'LOT_NOT_CONFIRMED'
        assert client.post('/assistant/resume', headers=chat[1], json={'session_id': chat[0]}).json()['drafts'][0]['record_started_at'] is None
        confirmations = [{'lot_id': i['lot_id'], 'method': 'full_code', 'value': i['lot_code']}
                         for i in card['prepared']['draft']['input_plan']]
    else:
        assert card['prepared']['can_commit']
    committed = record(client, chat, card, lot_confirmations=confirmations)
    assert committed.status_code == 200, committed.text
    assert committed.json()['result']['receipt_number'].startswith({'make': 'MK-', 'pack': 'PK-', 'adjust': 'ADJ-', 'found': 'FND-'}[action])


def test_sku_confirmation_is_human_evidence_and_prepares_a_new_ticket(client, chat, items, db_cursor, monkeypatch):
    monkeypatch.setattr(main, 'get_sibling_skus', lambda *args: [{'name': 'Other finished SKU'}])
    payload = action_body('make', items)
    model(monkeypatch, resolve_product(db_cursor, payload['product_id']), ('prepare_make', payload))
    result = turn(client, chat)
    card = result.json()['cards'][0]
    assert card['prepared']['blockers'][0]['code'] == 'SKU_CONFIRMATION_REQUIRED'
    assert not card['prepared']['can_commit']
    confirmed = client.post('/assistant/confirm-sku', headers=chat[1], json={'draft_id': card['id']})
    assert confirmed.status_code == 200, confirmed.text
    new = confirmed.json()['card']
    assert new['id'] != card['id']
    assert {b['code'] for b in new['prepared']['blockers']} == {'LOT_NOT_CONFIRMED'}
    confirmations = [{'lot_id': i['lot_id'], 'method': 'full_code', 'value': i['lot_code']}
                     for i in new['prepared']['draft']['input_plan']]
    assert record(client, chat, new, lot_confirmations=confirmations).status_code == 200
    assert record(client, chat, card).status_code == 409
    db_cursor.execute('SELECT payload FROM write_tickets WHERE id=%s', (new['prepared']['ticket_id'],))
    assert db_cursor.fetchone()['payload']['confirmed_sku'] is True


def test_lost_response_retries_same_ticket(client, chat, payload, db_cursor, monkeypatch):
    card, _ = make_draft(client, chat, payload, db_cursor, monkeypatch)
    real = fl_assistant.relay

    async def lost(app, request, method, path, **kwargs):
        result = await real(app, request, method, path, **kwargs)
        if path.startswith('/tickets/'):
            raise HTTPException(502, 'response lost')
        return result

    monkeypatch.setattr(fl_assistant, 'relay', lost)
    r = record(client, chat, card)
    assert r.status_code == 502
    monkeypatch.setattr(fl_assistant, 'relay', real)
    r = record(client, chat, card)
    assert r.status_code == 200, r.text
    assert r.json()['kind'] == 'receipt'
    # Fixture wraps all route transactions in one rollback scope; durable race
    # acceptance uses independent DB connections (see live staging smoke).


@pytest.mark.parametrize('status,result', [(403, {'detail': {'error_code': 'ROLE_NOT_ALLOWED', 'action': 'make', 'role': 'office'}}),
    (422, {'detail': {'error_code': 'LOT_NOT_CONFIRMED'}}), (500, {'detail': 'database failed'}), (200, {'success': True})])
def test_no_receipt_on_denial_failure_or_missing_receipt(client, chat, payload, db_cursor, monkeypatch, status, result):
    card, _ = make_draft(client, chat, payload, db_cursor, monkeypatch)
    async def response(*args, **kwargs):
        return status, result
    monkeypatch.setattr(fl_assistant, 'relay', response)
    r = record(client, chat, card)
    assert r.status_code >= 400
    assert r.json()['kind'] == 'not_recorded' and r.json()['result'] == result
    db_cursor.execute('SELECT status,record_started_at FROM assistant_drafts WHERE id=%s', (card['id'],))
    row = db_cursor.fetchone()
    assert row['status'] == 'pending'
    definitive = 400 <= status < 500
    assert (row['record_started_at'] is None) == definitive
    cancelled = client.post('/assistant/cancel', headers=chat[1], json={'draft_id': card['id']})
    assert cancelled.status_code == (200 if definitive else 409)


def test_cross_actor_and_tampering_rejected(client, chat, actors, payload, db_cursor, monkeypatch):
    card, _ = make_draft(client, chat, payload, db_cursor, monkeypatch)
    other = (chat[0], headers(actors['office']['key']))
    assert record(client, other, card).status_code == 404
    assert record(client, chat, card, payload_hash='forged').status_code == 422
    assert record(client, chat, card, ticket='forged').status_code == 422
    assert client.post('/assistant/resume', headers=other[1], json={'session_id': chat[0]}).status_code == 404


def test_cancel_is_durable_and_prevents_record(client, chat, payload, db_cursor, monkeypatch):
    card, _ = make_draft(client, chat, payload, db_cursor, monkeypatch)
    r = client.post('/assistant/cancel', headers=chat[1], json={'draft_id': card['id']})
    assert r.status_code == 200
    assert record(client, chat, card).status_code == 409


def test_turn_replay_does_not_prepare_twice(client, chat, payload, db_cursor, monkeypatch):
    calls = [resolve_product(db_cursor, payload['product_id'])]
    db_cursor.execute('SELECT name FROM suppliers WHERE id=%s', (payload['supplier_id'],))
    calls.append(('resolve', {'kind': 'supplier', 'query': db_cursor.fetchone()['name']}))
    calls.append(('prepare_receive', payload))
    requests = model(monkeypatch, *calls)
    tid = str(uuid4())
    first = turn(client, chat, turn_id=tid)
    assert first.status_code == 200, first.text
    assert turn(client, chat, turn_id=tid).json() == first.json()
    assert first.json()['cards'][0]['kind'] == 'draft'
    assert len(requests) == 3
    assert turn(client, chat, 'changed', turn_id=tid).status_code == 409


def test_unmatched_words_logged_no_alias_learning(client, chat, monkeypatch, db_cursor):
    model(monkeypatch, ('resolve', {'kind': 'product', 'query': 'zzzxxyunmatched'}))
    r = turn(client, chat)
    assert r.json()['cards'][0]['kind'] == 'choices'
    db_cursor.execute('SELECT query FROM assistant_unmatched_words WHERE session_id=%s', (chat[0],))
    assert db_cursor.fetchone()['query'] == 'zzzxxyunmatched'


def test_ambiguous_candidates_require_real_selection(client, chat, monkeypatch, db_cursor):
    token = 'F1AMB' + uuid4().hex[:8]
    for suffix in ('one', 'two'):
        db_cursor.execute("INSERT INTO products(name,type,uom) VALUES (%s,'ingredient','lb')", (token + ' ' + suffix,))
    calls = model(monkeypatch, ('resolve', {'kind': 'product', 'query': token}))
    r = turn(client, chat)
    card = r.json()['cards'][0]
    assert card['result']['outcome'] == 'ambiguous'
    assert len(calls) == 1
    bad = turn(client, chat, choice_id=card['id'], selected_id=999999999)
    assert bad.status_code == 422
    selection = card['result']['candidates'][0]['id']
    model(monkeypatch, ('prepare_found', {'product_id': selection, 'quantity': 1, 'reason_code': 'physical_count'}))
    r = turn(client, chat, choice_id=card['id'], selected_id=selection)
    assert r.json()['cards'][0]['kind'] == 'draft'


def test_model_cannot_escape_or_commit(client, chat, monkeypatch):
    for response in [ {'status': 'completed', 'output': [{'type': 'message', 'content': 'recorded'}]},
                     {'status': 'completed', 'output': [{'type': 'function_call', 'name': 'commit', 'arguments': '{}'}]},
                     {'status': 'incomplete', 'output': []}]:
        model(monkeypatch, response)
        r = turn(client, chat)
        assert r.status_code == 502
        assert 'receipt_number' not in r.text


def test_model_ids_need_resolution(client, chat, monkeypatch, payload):
    calls = [('prepare_receive', payload)] * fl_assistant.MAX_STEPS
    model(monkeypatch, *calls)
    r = turn(client, chat)
    assert r.status_code == 502


def test_prepare_refusal_is_exact_fl_response(client, chat, monkeypatch):
    model(monkeypatch, ('prepare_receive', {}))
    r = turn(client, chat)
    card = r.json()['cards'][0]
    expected = client.post('/receive/prepare', headers=chat[1], json={'client_source': 'fl_assistant'})
    assert card['status'] == expected.status_code
    assert card['result'] == expected.json()
    assert len(card['reasons']) == 8


def test_today_is_actor_bound_and_model_prose_hidden(client, chat, actors, monkeypatch):
    requests = model(monkeypatch, ('today_entries', {}))
    r = turn(client, chat, 'what did I enter today?')
    assert r.status_code == 200, r.text
    card = r.json()['cards'][0]
    expected = client.get('/receipts', headers=chat[1], params={'actor': actors['floor']['id']})
    assert card['kind'] == 'read' and card['result'] == expected.json()
    assert 'RECORDED!' not in r.text


def test_a5_evidence_relay_preserves_hash_and_values(client, chat, payload, db_cursor, monkeypatch):
    card, _ = make_draft(client, chat, payload, db_cursor, monkeypatch)
    evidence = [{'lot_id': 17, 'method': 'last4', 'value': '1234'}]
    observed = []
    async def capture(app, request, method, path, *, body=None, **kwargs):
        observed.append((path, body))
        return 422, {'detail': {'error_code': 'LOT_NOT_CONFIRMED'}}
    monkeypatch.setattr(fl_assistant, 'relay', capture)
    record(client, chat, card, lot_confirmations=evidence)
    db_cursor.execute('SELECT ticket,payload_hash FROM assistant_drafts WHERE id=%s', (card['id'],))
    row = db_cursor.fetchone()
    assert observed == [('/tickets/' + row['ticket'] + '/commit', {
        'payload_hash': row['payload_hash'], 'acknowledged_warnings': [], 'lot_confirmations': evidence})]


def test_photo_private_durable_and_never_model_input(client, chat, actors, monkeypatch, db_cursor):
    data = b'\x89PNG\r\n\x1a\nattachment-only-test'
    r = client.post('/assistant/attachment', params={'session_id': chat[0]}, headers=chat[1], files={'file': ('photo.png', data, 'image/png')})
    assert r.status_code == 200, r.text
    aid = r.json()['id']
    read = client.post('/assistant/attachment/read', headers=chat[1], json={'attachment_id': aid})
    assert read.content == data
    assert client.post('/assistant/attachment/read', headers=headers(actors['office']['key']), json={'attachment_id': aid}).status_code == 404
    requests = model(monkeypatch, ('today_entries', {}))
    r = turn(client, chat, 'today', attachment_ids=[aid])
    assert r.status_code == 200
    assert 'attachment-only-test' not in json.dumps(requests)
    assert 'input_image' not in json.dumps(requests)
    assert turn(client, chat, attachment_ids=[str(uuid4())]).status_code == 404


@pytest.mark.parametrize('name,data,mime', [('x.svg', b'<svg/>', 'image/svg+xml'), ('x.jpg', b'bad', 'image/jpeg'),
    ('large.png', b'\x89PNG\r\n\x1a\n' + b'x' * fl_assistant.MAX_PHOTO, 'image/png')])
def test_bad_photos_refused(client, chat, name, data, mime):
    r = client.post('/assistant/attachment', params={'session_id': chat[0]}, headers=chat[1], files={'file': (name, data, mime)})
    assert r.status_code == 422


def test_dictation_returns_editable_unsent_text(client, chat, monkeypatch):
    async def fake(path, *, file):
        assert path == 'audio/transcriptions' and file[0] == 'dictation.webm'
        return {'text': 'Recibí dos cajas'}
    monkeypatch.setattr(fl_assistant, 'openai', fake)
    r = client.post('/assistant/transcribe', headers=chat[1], files={'file': ('audio.webm', b'fake audio', 'audio/webm')})
    assert r.json() == {'text': 'Recibí dos cajas', 'editable': True, 'sent': False, 'success': True}


def test_identity_and_disabled_service(client, actors, monkeypatch):
    for key in (main.API_KEY, main.DASHBOARD_API_KEY, 'invalid'):
        assert client.post('/assistant/session', headers=headers(key)).status_code == 403
    monkeypatch.delenv('ASSISTANT_ENABLED')
    assert client.post('/assistant/session', headers=headers(actors['floor']['key'])).status_code == 503


def test_openai_failure_never_leaks_secret(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'should-never-appear')
    async def broken(*args, **kwargs):
        raise __import__('httpx').ConnectError('should-never-appear')
    monkeypatch.setattr(__import__('httpx').AsyncClient, 'post', broken)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(fl_assistant.openai('responses', body={}))
    assert 'should-never-appear' not in str(exc.value.detail)


def test_committed_races_lost_reply_and_cancel_safety(isolated_database, monkeypatch):
    """Independent committed DB connections, unlike the rollback fixtures above."""
    from contextlib import contextmanager
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from hashlib import sha256
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    from fastapi.testclient import TestClient
    root = Path(__file__).resolve().parents[1]
    raw_key = 'f1-race-' + uuid4().hex
    with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute((root / 'migrations/068_fl_assistant.sql').read_text())
        cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES ('F1 race','floor',%s,true) RETURNING id", (sha256(raw_key.encode()).hexdigest(),))
        aid = cur.fetchone()['id']
        payload = seed(cur)
        sid = str(uuid4())
        cur.execute('INSERT INTO assistant_sessions(id,actor_id) VALUES (%s,%s)', (sid, aid))
    @contextmanager
    def connection():
        with psycopg2.connect(isolated_database) as conn:
            yield conn
    monkeypatch.setattr(main, 'get_db_connection', connection)
    main._reset_actor_cache()
    try:
        with TestClient(main.app) as http:
            def draft():
                prepared = http.post('/receive/prepare', headers=headers(raw_key), json=payload).json()
                did = str(uuid4())
                with connection() as conn, conn.cursor() as cur:
                    cur.execute('INSERT INTO assistant_drafts(id,session_id,actor_id,ticket,payload_hash,card) VALUES (%s,%s,%s,%s,%s,%s)',
                                (did, sid, aid, prepared['ticket'], prepared['payload_hash'], Json({'kind': 'draft'})))
                return {'id': did}, prepared
            chat = (sid, headers(raw_key))
            card, prepared = draft()
            real = fl_assistant.relay
            gate = Barrier(2)
            async def simultaneous(*args, **kwargs):
                # Barrier in a worker avoids blocking either request's event loop.
                await asyncio.to_thread(gate.wait, 10)
                return await real(*args, **kwargs)
            monkeypatch.setattr(fl_assistant, 'relay', simultaneous)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(record, http, chat, card) for _ in range(2)]
                responses = [f.result(timeout=30) for f in futures]
            assert all(r.status_code == 200 for r in responses), [r.text for r in responses]
            assert responses[0].json()['result']['receipt_number'] == responses[1].json()['result']['receipt_number']
            with connection() as conn, conn.cursor() as cur:
                cur.execute('SELECT count(*) FROM transactions WHERE ticket_id=%s', (prepared['ticket_id'],))
                assert cur.fetchone()[0] == 1
            monkeypatch.setattr(fl_assistant, 'relay', real)
            payload['bol_reference'] += '-LOST'
            card, prepared = draft()
            async def lost(*args, **kwargs):
                await real(*args, **kwargs)
                raise HTTPException(502, 'lost response after real commit')
            monkeypatch.setattr(fl_assistant, 'relay', lost)
            assert record(http, chat, card, acknowledged_warnings=['POSSIBLE_DUPLICATE']).status_code == 502
            cancelled = http.post('/assistant/cancel', headers=chat[1], json={'draft_id': card['id']})
            assert cancelled.status_code == 409
            assert cancelled.json()['detail']['error_code'] == 'RECORD_OUTCOME_PENDING'
            monkeypatch.setattr(fl_assistant, 'relay', real)
            replay = record(http, chat, card)
            assert replay.status_code == 200 and replay.json()['result']['replayed'] is True, replay.text
    finally:
        main._reset_actor_cache()


@pytest.mark.parametrize('flag', [None, '', '0', 'true', '01', '1'])
def test_page_hidden_unless_explicitly_enabled(client, monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv('ASSISTANT_ENABLED', raising=False)
    else:
        monkeypatch.setenv('ASSISTANT_ENABLED', flag)
    response = client.get('/dash/fl-assistant')
    assert response.status_code == (200 if flag == '1' else 404)
    if flag != '1':
        assert '<html' not in response.text.lower() and 'actor-key' not in response.text
    else:
        assert response.headers['cache-control'] == 'no-store'


@pytest.mark.parametrize('status', [400, 401, 403, 429, 500, 503])
def test_openai_http_failure_logs_only_status(monkeypatch, caplog, status):
    import httpx
    monkeypatch.setenv('OPENAI_API_KEY', 'private-test-credential')
    async def rejected(*args, **kwargs):
        return httpx.Response(status, text='private-test-response')
    monkeypatch.setattr(httpx.AsyncClient, 'post', rejected)
    with caplog.at_level('WARNING', logger='fl_assistant'), pytest.raises(HTTPException) as exc:
        asyncio.run(fl_assistant.openai('responses', body={}))
    assert exc.value.status_code == 502
    assert [r.getMessage() for r in caplog.records if r.name == 'fl_assistant'] == [f'OpenAI request failed: status={status}']
    assert 'private-test-' not in caplog.text + str(exc.value.detail)


def test_correction_catalog_is_readable_with_master_and_actor_keys_when_disabled(client, actors, monkeypatch):
    monkeypatch.delenv('ASSISTANT_ENABLED')
    for key in [main.API_KEY, *(actors[role]['key'] for role in ('owner', 'floor', 'office'))]:
        response = client.get('/correction-reasons', headers=headers(key))
        assert response.status_code == 200 and len(response.json()['reasons']) == 8
    assert client.get('/correction-reasons', headers=headers(main.DASHBOARD_API_KEY)).status_code == 403


def test_expired_record_can_be_cancelled(client, chat, payload, db_cursor, monkeypatch):
    card, _ = make_draft(client, chat, payload, db_cursor, monkeypatch)
    db_cursor.execute("UPDATE write_tickets SET expires_at=clock_timestamp()-interval '1 second' WHERE id=%s", (card['prepared']['ticket_id'],))
    response = record(client, chat, card)
    assert response.status_code == 409
    assert response.json()['result']['detail']['error_code'] == 'TICKET_EXPIRED'
    assert client.post('/assistant/cancel', headers=chat[1], json={'draft_id': card['id']}).status_code == 200


def test_role_denied_record_can_be_cancelled(client, chat, actors, items, db_cursor, monkeypatch):
    payload = action_body('found', items)
    model(monkeypatch, resolve_product(db_cursor, payload['product_id']), ('prepare_found', payload))
    card = turn(client, chat).json()['cards'][0]
    db_cursor.execute("UPDATE actors SET role='office' WHERE id=%s", (actors['floor']['id'],))
    response = record(client, chat, card)
    assert response.status_code == 403
    assert response.json()['result']['detail']['error_code'] == 'ROLE_NOT_ALLOWED'
    assert client.post('/assistant/cancel', headers=chat[1], json={'draft_id': card['id']}).status_code == 200


@pytest.mark.parametrize('uncertainty', ['prior', 'concurrent'])
def test_rejection_preserves_other_uncertain_attempts(client, chat, payload, db_cursor, monkeypatch, uncertainty):
    card, _ = make_draft(client, chat, payload, db_cursor, monkeypatch)
    if uncertainty == 'prior':
        db_cursor.execute('UPDATE assistant_drafts SET record_started_at=clock_timestamp() WHERE id=%s', (card['id'],))
    async def denied(*args, **kwargs):
        if uncertainty == 'concurrent':
            # Another request durably marks its attempt while this relay runs.
            db_cursor.execute("UPDATE assistant_drafts SET record_started_at=record_started_at+interval '1 second' WHERE id=%s", (card['id'],))
        return 403, {'detail': {'error_code': 'ROLE_NOT_ALLOWED'}}
    monkeypatch.setattr(fl_assistant, 'relay', denied)
    assert record(client, chat, card).status_code == 403
    cancelled = client.post('/assistant/cancel', headers=chat[1], json={'draft_id': card['id']})
    assert cancelled.status_code == 409 and cancelled.json()['detail']['error_code'] == 'RECORD_OUTCOME_PENDING'


def test_crashed_chat_recovers_and_active_turn_renews_lease(client, chat, db_cursor, monkeypatch):
    db_cursor.execute("UPDATE assistant_sessions SET lease_id=%s,lease_until=clock_timestamp()+interval '2 minutes' WHERE id=%s", (str(uuid4()), chat[0]))
    assert turn(client, chat).json()['detail']['error_code'] == 'CHAT_BUSY'
    db_cursor.execute("UPDATE assistant_sessions SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s", (chat[0],))
    calls = []
    async def inspect(path, **kwargs):
        db_cursor.execute('SELECT extract(epoch FROM lease_until-clock_timestamp()) AS remaining FROM assistant_sessions WHERE id=%s', (chat[0],))
        assert 110 < db_cursor.fetchone()['remaining'] <= 120
        calls.append(path)
        # Simulate elapsed work between model calls; the next iteration renews.
        db_cursor.execute("UPDATE assistant_sessions SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s", (chat[0],))
        name, args = 'resolve', {'kind': 'product', 'query': 'missing-fixture'}
        return {'status': 'completed', 'output': [{'type': 'function_call', 'call_id': str(uuid4()), 'name': name, 'arguments': json.dumps(args)}]}
    # A matched resolver call continues the loop; an unmatched one finishes it.
    real = fl_assistant.relay
    async def lookup(app, request, method, path, **kwargs):
        if path == '/resolve' and len(calls) == 1:
            return 200, {'outcome': 'match', 'match': {'id': 123}, 'needs_clarification': False}
        return await real(app, request, method, path, **kwargs)
    monkeypatch.setattr(fl_assistant, 'relay', lookup)
    monkeypatch.setattr(fl_assistant, 'openai', inspect)
    assert turn(client, chat).status_code == 200
    assert len(calls) == 2
    db_cursor.execute('SELECT lease_id,lease_until FROM assistant_sessions WHERE id=%s', (chat[0],))
    assert dict(db_cursor.fetchone()) == {'lease_id': None, 'lease_until': None}


def test_replaced_worker_cannot_save_or_clear_new_lease(client, chat, db_cursor, monkeypatch):
    replacement = str(uuid4())
    async def replaced(path, **kwargs):
        db_cursor.execute("UPDATE assistant_sessions SET lease_id=%s,lease_until=clock_timestamp()+interval '2 minutes' WHERE id=%s", (replacement, chat[0]))
        return {'status': 'completed', 'output': [{'type': 'function_call', 'call_id': 'read', 'name': 'today_entries', 'arguments': '{}'}]}
    monkeypatch.setattr(fl_assistant, 'openai', replaced)
    assert turn(client, chat).status_code == 409
    db_cursor.execute('SELECT lease_id FROM assistant_sessions WHERE id=%s', (chat[0],))
    assert str(db_cursor.fetchone()['lease_id']) == replacement
    db_cursor.execute('SELECT count(*) AS n FROM assistant_turns WHERE session_id=%s', (chat[0],))
    assert db_cursor.fetchone()['n'] == 0


@pytest.mark.parametrize('action', ['make', 'pack'])
def test_a3b_shortage_warning_and_receipt_survive_assistant(client, chat, items, db_cursor, monkeypatch, action):
    payload = action_body(action, items)
    payload['batches' if action == 'make' else 'cases'] = 11 if action == 'make' else 22
    calls = [resolve_product(db_cursor, payload[f]) for f in
             ('product_id', 'source_product_id', 'target_product_id') if f in payload]
    model(monkeypatch, *calls, ('prepare_' + action, payload))
    card = turn(client, chat).json()['cards'][0]
    warnings = card['prepared']['warnings']
    assert any(w['code'] == 'WILL_CREATE_SHORTAGE' and w['message_es'] and not w['requires_ack'] for w in warnings)
    confirmations = [{'lot_id': i['lot_id'], 'method': 'full_code', 'value': i['lot_code']}
                     for i in card['prepared']['draft']['input_plan']]
    response = record(client, chat, card, lot_confirmations=confirmations)
    assert response.status_code == 200, response.text
    result = response.json()['result']
    assert response.json()['kind'] == 'receipt'
    assert result['shortages'][0]['short_lb'] == 10
    assert result['shortages'][0]['exception_id']
    assert record(client, chat, card, lot_confirmations=confirmations).json()['result'] == result | {'replayed': True}
    db_cursor.execute('SELECT count(*) AS n FROM shortage_flags WHERE transaction_id=%s', (result['transaction_id'],))
    assert db_cursor.fetchone()['n'] == 1


@pytest.mark.parametrize('action', ['adjust', 'found'])
@pytest.mark.parametrize('decision', ['approve', 'reject'])
def test_a3b_hold_resume_cancel_and_owner_outcome(client, chat, actors, items, db_cursor, monkeypatch, action, decision):
    payload = action_body(action, items)
    payload.pop('reason', None)
    payload['reason_code'] = 'physical_count'
    payload['delta_lb' if action == 'adjust' else 'quantity'] = -600 if action == 'adjust' else 600
    calls = ([('resolve', {'kind': 'lot', 'query': items['ingredient']['lot_code']})] if action == 'adjust'
             else [resolve_product(db_cursor, payload['product_id'])])
    model(monkeypatch, *calls, ('prepare_' + action, payload))
    card = turn(client, chat).json()['cards'][0]
    response = record(client, chat, card)
    assert response.status_code == 202, response.text
    held = response.json()
    assert held['kind'] == 'awaiting_approval'
    assert held['result']['held'] and held['result']['status'] == 'awaiting_approval'
    assert 'receipt_number' not in held['result']
    assert record(client, chat, card).json() == held
    saved = client.post('/assistant/resume', headers=chat[1], json={'session_id': chat[0]}).json()['drafts'][0]
    assert saved['status'] == 'pending' and saved['result'] == held['result']
    assert saved['record_started_at']
    assert client.post('/assistant/cancel', headers=chat[1], json={'draft_id': card['id']}).status_code == 409
    ticket_id = card['prepared']['ticket_id']
    db_cursor.execute('SELECT count(*) AS n FROM transactions WHERE ticket_id=%s', (ticket_id,))
    assert db_cursor.fetchone()['n'] == 0
    decision_response = client.post(f"/exceptions/{held['result']['exception_id']}/{decision}",
        headers=headers(actors['owner']['key']), json={'note': 'Synthetic owner review',
            **({'resolution_kind': 'declined'} if decision == 'reject' else {})})
    assert decision_response.status_code == 200, decision_response.text
    final = record(client, chat, card)
    if decision == 'approve':
        assert final.status_code == 200 and final.json()['kind'] == 'receipt'
        assert final.json()['result']['receipt_number'] == decision_response.json()['receipt_number']
        assert record(client, chat, card).json() == final.json()
    else:
        assert final.status_code == 409 and final.json()['kind'] == 'not_recorded'
    db_cursor.execute('SELECT count(*) AS n FROM transactions WHERE ticket_id=%s', (ticket_id,))
    assert db_cursor.fetchone()['n'] == (1 if decision == 'approve' else 0)


def test_model_cannot_invent_a3b_photo_evidence(client, chat, monkeypatch):
    for tool in assistant_tools.tools([{'code': 'physical_count'}]):
        assert 'attachment_ref' not in json.dumps(tool['parameters'])
    model(monkeypatch, ('prepare_found', {'quantity': 600, 'attachment_ref': 'invented-photo'}))
    response = turn(client, chat)
    assert response.status_code == 502
    assert response.json()['detail']['error_code'] == 'MODEL_EVIDENCE_FORBIDDEN'
