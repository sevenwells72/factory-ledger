"""Pilot sessions retain merged A9/A12 authority on the isolated local DB."""
import pytest

import key_sessions
import main
import shift_summary
from tests.test_key_sessions import actors, client, payload, pilot, sign_in  # noqa: F401
from tests.test_kosher_tiers import items  # noqa: F401
from tests.test_a12_review_fixes import found_payload
from tests.test_shift_summary import confirm, receive, section, summary
from tests.test_write_tickets import commit, error, headers, posted_count
from tests.test_write_tickets_part2 import body, prepare

pytestmark = pytest.mark.db


@pytest.mark.parametrize('lang', ['en', 'es'])
def test_session_shift_summary_confirms_person_attributed_receipt(client, actors, payload, db_cursor, lang):
    session, _ = sign_in(client, actors)
    token = session['session_token']
    result, _, prepared = receive(client, db_cursor, token, payload)
    doc = summary(client, token, lang=lang)
    assert doc['actor'] == session['actor']
    assert doc['confirmation']['can_confirm'] is True and doc['receipts_today'] == 1
    [row] = section(doc, 'received')['rows']
    assert row['receipt_number'] == result['receipt_number']
    assert row['actor'] == session['actor'] and row['line'] == row['line_' + lang]
    assert row['kosher_tier'] == 'regular'
    assert row['transaction_kosher_downgrade'] is False
    assert row['kosher_attested_by_actor_id'] is None
    assert posted_count(db_cursor, prepared) == 1

    recorded = confirm(client, token, doc, lang=lang)
    assert recorded.status_code == 201
    confirmed = recorded.json()
    assert confirmed['actor'] == session['actor']
    assert confirmed['summary_hash'] == doc['summary_hash']
    db_cursor.execute('SELECT actor_id,receipts_seen,summary_snapshot FROM shift_confirmations WHERE id=%s',
                      (confirmed['confirmation_id'],))
    saved = db_cursor.fetchone()
    assert saved['actor_id'] == actors['floor']['id'] and saved['receipts_seen'] == 1
    assert saved['summary_snapshot'] == doc['sections']
    db_cursor.execute('SELECT entered_by_actor_id,operator_id FROM transactions WHERE id=%s',
                      (result['transaction_id'],))
    assert dict(db_cursor.fetchone()) == {'entered_by_actor_id': actors['floor']['id'],
                                        'operator_id': actors['floor']['name']}
    denied = client.get('/reports/shift-summary', params={'actor': actors['office']['id']}, headers=headers(token))
    error(denied, 403, 'ACTOR_NOT_ALLOWED')


def held_with_session(client, items, token, action):
    payload_ = (body('make', items) | {'confirmed_sku': True} if action == 'make'
                else found_payload(items, 'batch'))
    prepared = prepare(client, action, payload_, token)
    assert prepared['can_commit'] is True
    held = commit(client, prepared, token)
    assert held.status_code == 202
    return prepared, held.json()


@pytest.mark.parametrize('action', ['make', 'found'])
def test_session_kosher_hold_approval_preserves_preparer_and_owner(client, actors, items, db_cursor, action):
    floor, floor_headers = sign_in(client, actors)
    owner, owner_headers = sign_in(client, actors, 'owner')
    token = floor['session_token']
    before = summary(client, token)
    prepared, held = held_with_session(client, items, token, action)
    waiting = summary(client, token)
    [hold] = [r for r in section(waiting, 'not_entered_yet')['rows'] if r.get('kind') == 'kosher_hold']
    assert hold['receipt_number'] == held['receipt_number']
    assert hold['actor'] == floor['actor'] and hold['approval']['owner_actor_id'] == owner['actor']['id']
    assert hold['held'] is True and hold['posted'] is False
    assert hold['kosher_attested_by_actor_id'] is None
    assert waiting['summary_hash'] == before['summary_hash'] and waiting['receipts_today'] == 0
    assert posted_count(db_cursor, prepared) == 0

    path = f"/exceptions/{held['exception_id']}/approve"
    error(client.post(path, json={}, headers=floor_headers), 403, 'ROLE_NOT_ALLOWED')
    approved = client.post(path, json={}, headers=owner_headers)
    assert approved.status_code == 200
    result = approved.json()
    assert result['receipt_number'] == held['receipt_number']
    assert result['kosher_attested_by_actor_id'] == owner['actor']['id'] != floor['actor']['id']
    assert posted_count(db_cursor, prepared) == 1
    after = summary(client, token)
    assert not [r for r in section(after, 'not_entered_yet')['rows']
                if r.get('receipt_number') == held['receipt_number']]
    [row] = section(after, shift_summary.ACTION_SECTION[action])['rows']
    assert row['receipt_number'] == held['receipt_number'] and row['actor'] == floor['actor']
    assert row['kosher_tier'] == 'extra_kosher' and row['transaction_kosher_downgrade'] is False
    assert row['kosher_attested_by_actor_id'] == owner['actor']['id']
    assert row['kosher_attested_at'] == result['kosher_attested_at']
    assert 'EXTRA_KOSHER' in {flag['code'] for flag in row['flags']}
    assert after['receipts_today'] == 1 and after['summary_hash'] != waiting['summary_hash']
    error(confirm(client, token, waiting), 409, 'SHIFT_SUMMARY_STALE')
    assert confirm(client, token, after).status_code == 201
    replay = client.post(path, json={}, headers=owner_headers)
    assert replay.status_code == 200 and replay.json()['replayed'] is True
    assert replay.json()['receipt_number'] == held['receipt_number']
    assert commit(client, prepared, token).json()['replayed'] is True
    assert posted_count(db_cursor, prepared) == 1
    db_cursor.execute('SELECT entered_by_actor_id,kosher_attested_by_actor_id FROM transactions WHERE ticket_id=%s',
                      (prepared['ticket_id'],))
    assert dict(db_cursor.fetchone()) == {'entered_by_actor_id': floor['actor']['id'],
                                        'kosher_attested_by_actor_id': owner['actor']['id']}
    db_cursor.execute('SELECT actor_id FROM ticket_attestations WHERE ticket_id=%s', (prepared['ticket_id'],))
    assert db_cursor.fetchone()['actor_id'] == owner['actor']['id']


@pytest.mark.parametrize('kind', ['master', 'dashboard'])
def test_shared_summary_read_never_becomes_person_confirmation(client, actors, kind):
    key = main.API_KEY if kind == 'master' else main.DASHBOARD_API_KEY
    doc = summary(client, key)
    assert doc['actor'] is None and doc['confirmation']['can_confirm'] is False
    error(confirm(client, key, doc), 403, 'ACTOR_REQUIRED')
    response = client.post('/auth/session', json={'actor_key': key})
    error(response, 401, 'ACTOR_KEY_INVALID')


@pytest.mark.parametrize('end', ['idle', 'logout'])
def test_ended_sessions_cannot_confirm_or_approve_new_upstream_routes(client, actors, items, db_cursor, end):
    floor, floor_headers = sign_in(client, actors)
    owner, owner_headers = sign_in(client, actors, 'owner')
    prepared, held = held_with_session(client, items, floor['session_token'], 'make')
    doc = summary(client, floor['session_token'])
    for session, auth in ((floor, floor_headers), (owner, owner_headers)):
        if end == 'idle':
            db_cursor.execute("UPDATE key_actor_sessions SET expires_at=clock_timestamp()-interval '1 second' WHERE token_hash=%s",
                              (key_sessions.digest(session['session_token']),))
        else:
            assert client.post('/auth/session/logout', headers=auth).status_code == 200
    error(client.get('/reports/shift-summary', headers=floor_headers), 401, 'SESSION_EXPIRED')
    error(confirm(client, floor['session_token'], doc), 401, 'SESSION_EXPIRED')
    error(client.post(f"/exceptions/{held['exception_id']}/approve", json={}, headers=owner_headers),
          401, 'SESSION_EXPIRED')
    assert posted_count(db_cursor, prepared) == 0
    db_cursor.execute('SELECT count(*) AS n FROM shift_confirmations WHERE actor_id=%s', (actors['floor']['id'],))
    assert db_cursor.fetchone()['n'] == 0
