"""A11 security acceptance on real PostgreSQL; no actual credentials in fixtures."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta
import secrets
from pathlib import Path
from types import SimpleNamespace

import psycopg2
from psycopg2.extras import RealDictCursor
from starlette.requests import Request
from fastapi import HTTPException
import pytest

import main
import pin_sessions as pins
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_write_tickets import isolated_database, payload, prepare, commit, headers, error, posted_count  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def people(db_cursor, actors, monkeypatch):
    monkeypatch.setenv('PIN_PEPPER', secrets.token_hex(32))
    db_cursor.execute((ROOT/'migrations/072_pin_sessions.sql').read_text())
    for label, person in actors.items():
        while True:
            value = str(secrets.randbelow(8000)+2000)
            if pins.valid_pin(value) and value not in [p.get('pin') for p in actors.values()]:
                break
        person['pin'] = value
        db_cursor.execute('UPDATE actors SET pin_hash=%s,pin_set_at=clock_timestamp() WHERE id=%s', (pins.pin_hash(value), person['id']))
    return actors


def login(http, person):
    response = http.post('/auth/session', json={'pin': person['pin']})
    assert response.status_code == 200  # never put credential-bearing response in assertion output
    return response.json()['session_token']


def request(ip='198.51.100.1', device=None):
    return Request({'type': 'http', 'method': 'POST', 'path': '/auth/session', 'client': (ip, 1234),
                    'headers': [(b'cookie', ('fl_device='+(device or secrets.token_urlsafe(32))).encode())]})


def test_weak_pin_policy():
    bad = [str(i)*4 for i in range(10)]
    bad += [''.join(str((start+step*i)%10) for i in range(4)) for start in range(10) for step in (1,-1)]
    bad += [str(i).zfill(2)*2 for i in range(100)] + ['', 'abc', '１２３４', ' 123', '12345']
    assert all(not pins.valid_pin(value) for value in bad)
    assert pins.valid_pin(str(5800+37))


def test_session_identity_permissions_and_hash_only_storage(client, db_cursor, people):
    token = login(client, people['floor'])
    result = client.get('/auth/whoami', headers=headers(token)).json()
    assert result['key_kind'] == 'session'
    assert result['actor']['id'] == people['floor']['id']
    assert result['permissions']['make'] and not result['permissions']['create_order']
    db_cursor.execute('SELECT * FROM actor_sessions WHERE token_hash=%s', (pins.digest(token),))
    row = db_cursor.fetchone()
    assert token not in str(row) and row['actor_id'] == people['floor']['id']
    assert row['expires_at'] - row['last_seen_at'] <= pins.IDLE + timedelta(seconds=1)
    assert client.get('/actors/pins', headers=headers(token)).status_code == 403
    assert client.post('/auth/session/key', json={'actor_key': people['floor']['key']}).status_code == 401


def test_source_lock_precedes_correct_pin_and_survives_cookie_rotation(client, db_cursor, people):
    for _ in range(5):
        response = client.post('/auth/session', json={'pin': 'invalid'})
        assert response.status_code == 401
    assert 'Retry-After' in response.headers
    client.cookies.clear()
    denied = client.post('/auth/session', json={'pin': people['floor']['pin']})
    error(denied, 401, 'PIN_INVALID')
    assert denied.json()['detail'] == response.json()['detail']
    db_cursor.execute('SELECT count(*) AS n FROM actor_sessions')
    assert db_cursor.fetchone()['n'] == 0


def test_device_lock_survives_ip_rotation(db_cursor, people, client):
    device = secrets.token_urlsafe(32)
    for i in range(5):
        _, error_, _ = pins.verify_pin(main, request('198.51.100.'+str(i), device), 'invalid', purpose='test')
        assert error_
    actor, error_, _ = pins.verify_pin(main, request('203.0.113.100', device), people['owner']['pin'], purpose='test')
    assert actor is None and error_.headers['Retry-After']


def test_distributed_global_ceiling(db_cursor, people, client):
    for i in range(50):
        actor, error_, _ = pins.verify_pin(main, request('198.51.100.'+str(i)), 'invalid', purpose='test')
        assert actor is None and error_
    actor, error_, _ = pins.verify_pin(main, request('203.0.113.200'), people['owner']['pin'], purpose='test')
    assert actor is None and int(error_.headers['Retry-After']) >= 3599


def test_all_ten_thousand_candidates_stopped(db_cursor, people, client):
    # All seeded valid values lie beyond the first five candidates. This
    # proves the acceptance sweep is stopped, not that guessing has zero risk.
    accepted = 0
    for i in range(10000):
        actor, _, _ = pins.verify_pin(main, request(), str(i).zfill(4), purpose='sweep')
        accepted += actor is not None
    assert accepted == 0
    db_cursor.execute("SELECT count(*) AS total,count(*) FILTER (WHERE blocked) AS blocked FROM pin_attempts WHERE purpose='sweep'")
    row = db_cursor.fetchone()
    assert row['total'] == 10000 and row['blocked'] == 9995


def test_idle_expiry_background_reads_logout_and_revocation(client, db_cursor, people):
    token = login(client, people['floor'])
    db_cursor.execute("UPDATE actor_sessions SET expires_at=clock_timestamp()-interval '1 second' WHERE token_hash=%s", (pins.digest(token),))
    error(client.get('/auth/whoami', headers=headers(token)), 401, 'SESSION_EXPIRED')
    token = login(client, people['floor'])
    db_cursor.execute('SELECT expires_at FROM actor_sessions WHERE token_hash=%s', (pins.digest(token),))
    before = db_cursor.fetchone()['expires_at']
    assert client.get('/auth/whoami', headers=headers(token)|{'X-FL-Background':'1'}).status_code == 200
    db_cursor.execute('SELECT expires_at FROM actor_sessions WHERE token_hash=%s', (pins.digest(token),))
    assert db_cursor.fetchone()['expires_at'] == before
    assert client.delete('/auth/session', headers=headers(token)).status_code == 200
    assert client.get('/auth/whoami', headers=headers(token)).status_code == 401
    token = login(client, people['floor'])
    db_cursor.execute('UPDATE actors SET active=false WHERE id=%s', (people['floor']['id'],))
    assert client.get('/auth/whoami', headers=headers(token)).status_code == 401


def test_session_wrong_person_cannot_commit_or_replay(client, db_cursor, people, payload):
    floor = login(client, people['floor'])
    prepared = prepare(client, payload, floor)
    office = login(client, people['office'])
    error(commit(client, prepared, office), 403, 'TICKET_WRONG_USER')
    new_floor = login(client, people['floor'])
    response = commit(client, prepared, new_floor)
    assert response.status_code == 200
    error(commit(client, prepared, office), 403, 'TICKET_WRONG_USER')
    assert commit(client, prepared, new_floor).json()['replayed'] is True
    db_cursor.execute('SELECT entered_by_actor_id FROM transactions WHERE ticket_id=%s', (prepared['ticket_id'],))
    assert db_cursor.fetchone()['entered_by_actor_id'] == people['floor']['id']


def test_owner_step_up_is_per_request_and_wrong_person_fails(client, db_cursor, people, payload):
    owner = login(client, people['owner'])
    # Owner PIN is checked before the held-correction handler is reached.
    path = '/exceptions/2147483000/approve'
    error(client.post(path, json={}, headers=headers(owner)), 403, 'OWNER_PIN_REQUIRED')
    error(client.post(path, json={}, headers=headers(owner)|{'X-FL-Owner-PIN':people['office']['pin']}), 401, 'PIN_INVALID')
    assert client.post(path, json={}, headers=headers(owner)|{'X-FL-Owner-PIN':people['owner']['pin']}).status_code == 404
    error(client.post(path, json={}, headers=headers(owner)), 403, 'OWNER_PIN_REQUIRED')
    floor = login(client, people['floor'])
    assert client.post(path, json={}, headers=headers(floor)|{'X-FL-Owner-PIN':people['owner']['pin']}).status_code == 403
    # A12's one-batch attestation seam may verify Michael without switching
    # the signed-in person; no elevated token/session is created.
    req = request(); req.state.actor = people['floor']; req.state.key_kind = 'session'
    req.scope['headers'].append((b'x-fl-owner-pin', people['owner']['pin'].encode()))
    assert pins.require_owner_pin(main, req, purpose='kosher_attestation', allow_other_owner=True)['id'] == people['owner']['id']
    assert req.state.actor['id'] == people['floor']['id']


def test_bootstrap_unique_reset_and_own_pin_change(client, db_cursor, people):
    owner = people['owner']; db_cursor.execute('UPDATE actors SET pin_hash=NULL WHERE id=%s', (owner['id'],))
    response = client.post('/auth/session/key', json={'actor_key':owner['key']})
    assert response.status_code == 200
    session = response.json()['session_token']
    endpoint = '/actors/'+str(owner['id'])+'/pin'
    response = client.post(endpoint, json={'pin':owner['pin']}, headers=headers(session))
    assert response.status_code == 200 and response.json()['sign_in_again']
    assert client.get('/auth/session', headers=headers(session)).status_code == 401
    session = login(client, owner)
    admin_headers = headers(session)|{'X-FL-Owner-PIN':owner['pin']}
    endpoint = '/actors/'+str(people['floor']['id'])+'/pin'
    error(client.post(endpoint, json={'pin':people['office']['pin']}, headers=admin_headers), 409, 'PIN_IN_USE')
    error(client.post(endpoint, json={'pin':str(1)*4}, headers=admin_headers), 422, 'PIN_WEAK')
    floor = login(client, people['floor'])
    new_pin = next(str(i) for i in range(1100,2000) if pins.valid_pin(str(i)))
    assert client.post(endpoint, json={'pin':new_pin,'current_pin':people['floor']['pin']}, headers=headers(floor)).status_code == 200
    assert client.get('/auth/whoami', headers=headers(floor)).status_code == 401
    people['floor']['pin'] = new_pin
    floor = login(client, people['floor'])
    assert client.post('/actors/'+str(people['office']['id'])+'/pin', json={'pin':new_pin}, headers=headers(floor)).status_code == 403
    assert client.post(endpoint, json={'pin':str(8100+37)}, headers=headers(main.DASHBOARD_API_KEY)).status_code == 403


def test_unconfigured_pepper_fails_closed(client, monkeypatch):
    monkeypatch.delenv('PIN_PEPPER', raising=False)
    assert client.post('/auth/session', json={'pin':'invalid'}).status_code == 503


def test_parallel_requests_share_atomic_global_limit(isolated_database, monkeypatch):
    monkeypatch.setenv('PIN_PEPPER', secrets.token_hex(32))
    with psycopg2.connect(isolated_database) as conn, conn.cursor() as cur:
        cur.execute((ROOT/'migrations/072_pin_sessions.sql').read_text())
    @contextmanager
    def transaction():
        with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
    api = SimpleNamespace(get_transaction=transaction)
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda i: pins.verify_pin(api, request('203.0.113.'+str(i)), 'invalid', purpose='race'), range(75)))
    assert all(actor is None and error_ for actor,error_,_ in results)
    with transaction() as cur:
        cur.execute("SELECT count(*) FILTER (WHERE NOT blocked) AS tried,count(*) FILTER (WHERE blocked) AS blocked FROM pin_attempts")
        assert dict(cur.fetchone()) == {'tried':50,'blocked':25}


def test_malformed_auth_never_echoes_input(client, people):
    candidate = int(people['owner']['pin'])
    response = client.post('/auth/session', json={'pin':candidate})
    assert response.status_code == 422
    assert str(candidate) not in response.text


def test_backdated_ticket_requires_pin_on_commit(client, db_cursor, people, payload):
    token = login(client, people['owner'])
    prepared = prepare(client, payload | {'occurred_at': (main.get_plant_now()-timedelta(days=16)).isoformat(), 'backfill':True}, token)
    error(commit(client, prepared, token), 403, 'OWNER_PIN_REQUIRED')
    assert posted_count(db_cursor, prepared) == 0
    response = client.post('/tickets/'+prepared['ticket']+'/commit', json={'payload_hash':prepared['payload_hash']},
                           headers=headers(token)|{'X-FL-Owner-PIN':people['owner']['pin']})
    assert response.status_code == 200
    assert posted_count(db_cursor, prepared) == 1


def test_migration_replay_preserves_sessions_and_protects_actor_hashes(client, db_cursor, people):
    token = login(client, people['owner'])
    db_cursor.execute((ROOT/'migrations/072_pin_sessions.sql').read_text())
    assert client.get('/auth/session', headers=headers(token)).status_code == 200
    db_cursor.execute("SELECT relrowsecurity FROM pg_class WHERE relname IN ('actors','actor_sessions','pin_attempts','pin_rate_limits','pin_management_audit')")
    assert all(r['relrowsecurity'] for r in db_cursor.fetchall())


def test_demotion_between_owner_proof_and_pin_write_denies(client, db_cursor, people, monkeypatch):
    owner = login(client, people['owner'])
    verify = pins.require_owner_pin
    def demote_after_proof(api, request, **options):
        result = verify(api, request, **options)
        db_cursor.execute("UPDATE actors SET role='office' WHERE id=%s", (people['owner']['id'],))
        return result
    monkeypatch.setattr(pins, 'require_owner_pin', demote_after_proof)
    target = people['floor']['id']
    before = pins.pin_hash(people['floor']['pin'])
    candidate = next(str(i) for i in range(1100,2000) if pins.valid_pin(str(i)))
    response = client.post('/actors/'+str(target)+'/pin',json={'pin':candidate},
                           headers=headers(owner)|{'X-FL-Owner-PIN':people['owner']['pin']})
    error(response,403,'ROLE_NOT_ALLOWED')
    db_cursor.execute('SELECT pin_hash FROM actors WHERE id=%s',(target,))
    assert db_cursor.fetchone()['pin_hash'] == before
