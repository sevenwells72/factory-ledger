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
import fl_assistant
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import enabled, reason_catalog, chat, model  # noqa: F401
from tests.test_write_tickets import payload  # noqa: F401

pytestmark = pytest.mark.db


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
