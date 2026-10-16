"""Mobile UX round: dated chat history and the photo errors the phone page relies on."""
from datetime import datetime
from uuid import uuid4

import pytest

import fl_assistant
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_write_tickets import isolated_database, seed  # noqa: F401
from tests.test_fl_assistant import chat, enabled, model, reason_catalog, turn  # noqa: F401

pytestmark = pytest.mark.db


def test_resume_dates_each_turn_without_changing_its_reply(client, chat, monkeypatch):
    model(monkeypatch, ('unsupported_request', {}), ('unsupported_request', {}))
    first, second = turn(client, chat, 'Photo').json(), turn(client, chat, 'hello').json()
    resumed = client.post('/assistant/resume', headers=chat[1], json={'session_id': chat[0]})
    assert resumed.status_code == 200, resumed.text
    turns = resumed.json()['turns']
    assert [t['text'] for t in turns] == ['Photo', 'hello']
    stamps = [datetime.fromisoformat(t['created_at']) for t in turns]
    assert all(stamp.tzinfo is not None for stamp in stamps) and stamps[0] <= stamps[1]
    for saved, live in zip(turns, (first, second)):
        assert {k: saved[k] for k in ('turn_id', 'text', 'cards')} == {k: live[k] for k in ('turn_id', 'text', 'cards')}
    # A replayed turn still returns its original saved reply, unchanged.
    sid, h = chat
    turn_id = str(uuid4())
    model(monkeypatch, ('unsupported_request', {}))
    body = {'session_id': sid, 'turn_id': turn_id, 'text': 'again'}
    original = client.post('/assistant/turn', headers=h, json=body).json()
    assert client.post('/assistant/turn', headers=h, json=body).json() == original
    assert 'created_at' not in original


def test_reopening_resumes_the_same_chat(client, chat, monkeypatch):
    model(monkeypatch, ('unsupported_request', {}))
    turn(client, chat, 'first')
    for _ in range(2):
        resumed = client.post('/assistant/resume', headers=chat[1], json={'session_id': chat[0]}).json()
        assert resumed['session_id'] == chat[0] and len(resumed['turns']) == 1


@pytest.mark.parametrize('data,name', [
    (b'\xff\xd8\xff' + b'0' * fl_assistant.MAX_PHOTO, 'big.jpg'),
    (b'%PDF-1.4 not a photo', 'fake.jpg'),
], ids=['over-5mb', 'not-a-photo'])
def test_photo_rejections_use_the_code_the_page_reads(client, chat, data, name):
    sid, h = chat
    response = client.post('/assistant/attachment', headers=h, params={'session_id': sid},
                           files={'file': (name, data, 'image/jpeg')}, data={'kind': 'lot_tag'})
    assert response.status_code == 422
    assert response.json()['detail']['error_code'] == 'PHOTO_INVALID'
