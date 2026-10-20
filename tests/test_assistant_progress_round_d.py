"""Plain bilingual progress uses actual response chunks and retains safe failures."""
import asyncio
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
