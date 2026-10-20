"""Plain bilingual progress uses actual response chunks and retains safe failures."""
import asyncio
from contextlib import contextmanager
import json
from threading import Event
import time
from uuid import uuid4

from fastapi import HTTPException
from starlette.requests import Request
import pytest

import assistant_runtime as runtime
import db_busy
import fl_assistant
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import enabled, reason_catalog, chat, model  # noqa: F401
from tests.test_write_tickets import payload  # noqa: F401

pytestmark = pytest.mark.db

import main  # noqa: E402
REAL_GET_DB_CONNECTION = main.get_db_connection   # before the client fixture swaps it


@pytest.fixture(autouse=True)
def deferred(monkeypatch):
    """Deferred chat-lock releases, recorded instead of run on a thread that
    would outlive the test and touch its shared connection."""
    calls = []
    monkeypatch.setattr(fl_assistant, '_in_background', lambda target, *args: calls.append((target, args)),
                        raising=False)
    return calls


@pytest.mark.parametrize('spanish', [False, True])
def test_actual_chat_stream_reports_localized_lookup_before_result(client, chat, payload, db_cursor, monkeypatch, spanish):
    db_cursor.execute('SELECT name FROM products WHERE id=%s', (payload['product_id'],))
    name = db_cursor.fetchone()['name']
    model(monkeypatch, ('inventory_lookup', {'q': name}))
    text = ('¿Cuánto tenemos de ' if spanish else 'How much stock of ') + name + '?'
    response = client.post('/assistant/turn', headers={**chat[1], 'Accept': 'application/x-ndjson'}, json={
        'session_id': chat[0], 'turn_id': str(uuid4()), 'text': text})
    assert response.status_code == 200 and response.headers['content-type'].startswith('application/x-ndjson')
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]['type'] == 'result'
    statuses = [event['message'] for event in events if event['type'] == 'progress']
    assert statuses[0] == ('Leyendo tu mensaje…' if spanish else 'Reading your message…')
    assert any(message.startswith('Buscando ' if spanish else 'Looking up ') for message in statuses)
    assert events[-1]['result']['cards'][0]['kind'] == 'read'


def test_stream_failure_clears_lease_and_does_not_expose_transport_secrets(client, chat, db_cursor, monkeypatch):
    async def unavailable(*args, **kwargs):
        raise HTTPException(502, 'OpenAI request failed.')
    monkeypatch.setattr(fl_assistant, 'openai', unavailable)
    response = client.post('/assistant/turn', headers={**chat[1], 'Accept': 'application/x-ndjson'}, json={
        'session_id': chat[0], 'turn_id': str(uuid4()), 'text': 'How much oats?'})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1] == {'type': 'error', 'status': 502, 'detail': 'OpenAI request failed.'}
    assert not any(event['type'] == 'result' for event in events)
    assert chat[1]['X-API-Key'] not in response.text
    db_cursor.execute('SELECT lease_id,lease_until FROM assistant_sessions WHERE id=%s', (chat[0],))
    assert dict(db_cursor.fetchone()) == {'lease_id': None, 'lease_until': None}
    db_cursor.execute('SELECT count(*) AS n FROM assistant_drafts WHERE session_id=%s', (chat[0],))
    assert db_cursor.fetchone()['n'] == 0


def test_disconnected_progress_reader_waits_for_draft_work_to_finish():
    finished = Event()
    request = Request({'type': 'http', 'method': 'POST', 'path': '/assistant/turn',
                       'headers': [(b'accept', b'application/x-ndjson')]})
    async def work():
        runtime.progress(request, 'lookup', 'oats')
        time.sleep(0.05)  # Stand-in for an in-flight blocking PostgreSQL call.
        finished.set()
        return {'cards': []}
    async def exercise():
        response = await runtime.run_turn(work, request)
        iterator = response.body_iterator
        first = await anext(iterator)
        assert json.loads(first)['type'] == 'progress'
        await iterator.aclose()
        assert finished.is_set()
    asyncio.run(exercise())


# PR #141 re-check (comment 6294160112) item 3: once the 200 NDJSON headers
# are sent a pool timeout cannot become HTTP 503, so the stream reports it as
# a plain, retryable busy event instead of an error 500.

def _stream_events(handler):
    request = Request({'type': 'http', 'method': 'POST', 'path': '/assistant/turn',
                       'headers': [(b'accept', b'application/x-ndjson')]})
    async def exercise():
        # As inside DatabaseBusyMiddleware: wait ids are collected per request.
        token, _ = db_busy.begin_request()
        try:
            response = await runtime.run_turn(handler, request)
            return [json.loads(chunk) async for chunk in response.body_iterator]
        finally:
            db_busy.end_request(token)
    return asyncio.run(exercise())


def _pool_timeout():
    return main.WaitingConnectionPool(0, 1, wait_seconds=0.01)._timeout()


BUSY_EVENT = {'type': 'error', 'status': 503, 'error_code': 'DATABASE_BUSY', 'retryable': True,
              'detail': {'error_code': 'DATABASE_BUSY', 'retryable': True,
                         'message': 'FL is busy right now — try again in a moment.',
                         'message_es': 'FL está ocupado en este momento — inténtalo de nuevo en un momento.'}}


def test_stream_reports_a_pool_timeout_as_a_retryable_busy_event():
    async def escapes():
        raise _pool_timeout()

    async def quoted_in_a_500():
        try:
            raise _pool_timeout()
        except Exception as e:
            raise HTTPException(500, str(e))

    async def raised_from():
        try:
            raise _pool_timeout()
        except Exception as exc:
            raise RuntimeError('turn failed') from exc

    for handler in (escapes, quoted_in_a_500, raised_from):
        events = _stream_events(handler)
        assert events[-1] == BUSY_EVENT, handler.__name__


def test_stream_keeps_real_failures_and_normal_results():
    async def handled_then_bug():
        try:
            raise _pool_timeout()
        except Exception:
            pass
        raise RuntimeError('unrelated bug')

    async def upstream():
        raise HTTPException(502, 'OpenAI request failed.')

    async def ok():
        return {'cards': [{'kind': 'read'}]}

    assert _stream_events(handled_then_bug)[-1] == {
        'type': 'error', 'status': 500,
        'detail': 'Factory Ledger is temporarily unavailable — try again in a minute.'}
    assert _stream_events(upstream)[-1] == {'type': 'error', 'status': 502, 'detail': 'OpenAI request failed.'}
    assert _stream_events(ok) == [{'type': 'result', 'result': {'cards': [{'kind': 'read'}]}}]


def test_real_turn_with_the_pool_full_streams_busy_not_500(client, chat, db_cursor, monkeypatch):
    """The reviewer's case: authentication already cached, then every pool
    connection stays busy while the turn runs."""
    import main
    db_pool = main.WaitingConnectionPool(0, 1, wait_seconds=0.1, dsn=db_cursor.connection.dsn)
    app_pool, held = main.db_pool, []
    progress = runtime.progress

    def fill_pool_once_turn_starts(request, stage, *args, **kwargs):
        if stage == 'read' and not held:
            monkeypatch.setattr(main, 'db_pool', db_pool)
            monkeypatch.setattr(main, 'get_db_connection', REAL_GET_DB_CONNECTION)
            held.append(db_pool.getconn())
        return progress(request, stage, *args, **kwargs)
    monkeypatch.setattr(runtime, 'progress', fill_pool_once_turn_starts)
    try:
        response = client.post('/assistant/turn', headers={**chat[1], 'Accept': 'application/x-ndjson'}, json={
            'session_id': chat[0], 'turn_id': str(uuid4()), 'text': 'How much oats?'})
    finally:
        for conn in held:
            db_pool.putconn(conn)
        db_pool.closeall()
        monkeypatch.setattr(main, 'db_pool', app_pool)   # app shutdown closes its own pool
    assert held, 'the turn never started'
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]['type'] == 'progress'
    assert events[-1] == BUSY_EVENT


# PR #141 re-check round 2 (comment 6295966584) item 2: a lookup inside the
# turn that meets a full pool answered 503 DATABASE_BUSY, which the turn saved
# as a "lookup_failed" card (unavailable, no Retry) and replayed on resend.

@pytest.mark.parametrize('streamed', [False, True])
def test_a_busy_lookup_aborts_the_turn_unsaved_and_the_retry_runs_it_again(client, chat, db_cursor, monkeypatch, streamed):
    db_pool = main.WaitingConnectionPool(0, 1, wait_seconds=0.1, dsn=db_cursor.connection.dsn)
    app_pool, held = main.db_pool, db_pool.getconn()   # every pool connection busy
    relay, test_connection = fl_assistant.relay, main.get_db_connection
    busy = [True]

    async def lookup_while_pool_is_full(app, request, method, path, **kwargs):
        if not (busy[0] and path.startswith('/lots/')):
            return await relay(app, request, method, path, **kwargs)
        main.db_pool, main.get_db_connection = db_pool, REAL_GET_DB_CONNECTION
        try:
            return await relay(app, request, method, path, **kwargs)
        finally:   # capacity is back before the turn saves
            main.db_pool, main.get_db_connection = app_pool, test_connection
    monkeypatch.setattr(fl_assistant, 'relay', lookup_while_pool_is_full)
    model(monkeypatch, ('lot_lookup', {'lot_code': 'NOPE-1'}), ('lot_lookup', {'lot_code': 'NOPE-1'}))
    headers = {**chat[1], **({'Accept': 'application/x-ndjson'} if streamed else {})}
    turn = {'session_id': chat[0], 'turn_id': str(uuid4()), 'text': 'Where is lot NOPE-1?'}
    try:
        first = client.post('/assistant/turn', headers=headers, json=turn)
    finally:
        db_pool.putconn(held)
        db_pool.closeall()
    if streamed:
        assert first.status_code == 200
        assert [json.loads(line) for line in first.text.splitlines()][-1] == BUSY_EVENT
    else:
        assert first.status_code == 503, first.text
        assert first.json()['error_code'] == 'DATABASE_BUSY' and first.json()['retryable'] is True
    db_cursor.execute('SELECT count(*) AS n FROM assistant_turns WHERE id=%s', (turn['turn_id'],))
    assert db_cursor.fetchone()['n'] == 0   # nothing saved to replay
    db_cursor.execute('SELECT lease_id FROM assistant_sessions WHERE id=%s', (chat[0],))
    assert db_cursor.fetchone()['lease_id'] is None

    busy[0] = False   # Retry: same turn id, pool free again
    again = client.post('/assistant/turn', headers=headers, json=turn)
    assert again.status_code == 200
    result = [json.loads(line) for line in again.text.splitlines()][-1]['result'] if streamed else again.json()
    card = result['cards'][0]
    assert card['kind'] == 'lookup_failed' and card['result']['detail']['error_code'] == 'LOT_NOT_FOUND'


# PR #141 re-check round 3 (comment 6297413580) item 2: the lookup timed out,
# then the chat-lock release timed out too, so Retry met 409 CHAT_BUSY for two
# minutes; and a release that timed out after the turn was saved answered busy.

def _turn_events(response, streamed):
    return [json.loads(line) for line in response.text.splitlines()][-1] if streamed else response.json()


@pytest.mark.parametrize('streamed', [False, True])
def test_retry_runs_at_once_after_the_lock_release_also_met_a_full_pool(client, chat, db_cursor, monkeypatch,
                                                                       streamed, deferred):
    db_pool = main.WaitingConnectionPool(0, 1, wait_seconds=0.1, dsn=db_cursor.connection.dsn)
    app_pool, test_connection, held = main.db_pool, main.get_db_connection, db_pool.getconn()
    relay = fl_assistant.relay
    busy = [True]

    async def lookup_fills_the_pool(app, request, method, path, **kwargs):
        if busy[0] and path.startswith('/lots/'):   # and it stays full through the turn's cleanup
            main.db_pool, main.get_db_connection = db_pool, REAL_GET_DB_CONNECTION
        return await relay(app, request, method, path, **kwargs)
    monkeypatch.setattr(fl_assistant, 'relay', lookup_fills_the_pool)
    requests = model(monkeypatch, ('lot_lookup', {'lot_code': 'NOPE-1'}), ('lot_lookup', {'lot_code': 'NOPE-1'}))
    headers = {**chat[1], **({'Accept': 'application/x-ndjson'} if streamed else {})}
    turn = {'session_id': chat[0], 'turn_id': str(uuid4()), 'text': 'Where is lot NOPE-1?'}
    try:
        first = client.post('/assistant/turn', headers=headers, json=turn)
    finally:
        db_pool.putconn(held)
        db_pool.closeall()
        main.db_pool, main.get_db_connection = app_pool, test_connection   # FL is free again
    if streamed:
        assert _turn_events(first, True) == BUSY_EVENT
    else:
        assert first.status_code == 503 and first.json()['error_code'] == 'DATABASE_BUSY', first.text
    db_cursor.execute('SELECT lease_id FROM assistant_sessions WHERE id=%s', (chat[0],))
    assert db_cursor.fetchone()['lease_id'] is not None   # the release met the full pool too

    busy[0] = False
    again = client.post('/assistant/turn', headers=headers, json=turn)   # Retry, right away
    assert again.status_code == 200, again.text
    card = (_turn_events(again, streamed)['result'] if streamed else again.json())['cards'][0]
    assert card['kind'] == 'lookup_failed' and card['result']['detail']['error_code'] == 'LOT_NOT_FOUND'
    assert len(requests) == 2   # the model ran again; nothing was replayed
    assert len(deferred) == 1
    for target, args in deferred:   # the late row clear finds a newer lock owner: no effect
        target(*args)
    db_cursor.execute('SELECT lease_id FROM assistant_sessions WHERE id=%s', (chat[0],))
    assert db_cursor.fetchone()['lease_id'] is None


@pytest.mark.parametrize('streamed', [False, True])
def test_a_turn_saved_just_before_the_pool_filled_reports_its_saved_result(client, chat, db_cursor, monkeypatch,
                                                                         streamed, deferred):
    db_pool = main.WaitingConnectionPool(0, 1, wait_seconds=0.1, dsn=db_cursor.connection.dsn)
    app_pool, test_connection, held = main.db_pool, main.get_db_connection, db_pool.getconn()
    turn = {'session_id': chat[0], 'turn_id': str(uuid4()), 'text': 'Where is lot NOPE-1?'}
    saved = [False]

    @contextmanager
    def pool_full_once_saved():
        if saved[0]:
            with REAL_GET_DB_CONNECTION() as conn:   # main.db_pool is the full pool now
                yield conn
            return
        with test_connection() as conn:
            yield conn
        db_cursor.execute('SELECT 1 FROM assistant_turns WHERE id=%s', (turn['turn_id'],))
        if db_cursor.fetchone():
            saved[0] = True
            main.db_pool = db_pool
    monkeypatch.setattr(main, 'get_db_connection', pool_full_once_saved)
    model(monkeypatch, ('lot_lookup', {'lot_code': 'NOPE-1'}), ('lot_lookup', {'lot_code': 'NOPE-2'}))
    headers = {**chat[1], **({'Accept': 'application/x-ndjson'} if streamed else {})}
    try:
        first = client.post('/assistant/turn', headers=headers, json=turn)
    finally:
        db_pool.putconn(held)
        db_pool.closeall()
        main.db_pool, main.get_db_connection = app_pool, test_connection
    assert saved[0], 'the turn never saved'
    assert first.status_code == 200, first.text
    result = _turn_events(first, True)['result'] if streamed else first.json()
    assert result['turn_id'] == turn['turn_id'] and result['cards'][0]['kind'] == 'lookup_failed'
    db_cursor.execute('SELECT lease_id FROM assistant_sessions WHERE id=%s', (chat[0],))
    assert db_cursor.fetchone()['lease_id'] is None   # freed in the same commit as the save
    assert deferred == []

    following = client.post('/assistant/turn', headers=headers, json={
        'session_id': chat[0], 'turn_id': str(uuid4()), 'text': 'Where is lot NOPE-2?'})
    assert following.status_code == 200, following.text   # not CHAT_BUSY
