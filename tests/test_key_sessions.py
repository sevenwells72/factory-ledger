"""Pilot sessions: real HTTP/DB, mocked model only for Assistant transport."""
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from uuid import uuid4
import pytest
import main
import key_sessions
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_write_tickets import payload, headers  # noqa: F401
from tests.test_fl_assistant import make_draft, record

pytestmark = pytest.mark.db

@pytest.fixture(autouse=True)
def pilot(monkeypatch):
    monkeypatch.setenv('KEY_SESSIONS_ENABLED', '1')
    monkeypatch.setenv('PIN_LOGIN_ENABLED', '0')


def sign_in(client, actors, role='floor'):
    response = client.post('/auth/session', json={'actor_key': actors[role]['key']})
    assert response.status_code == 200
    result = response.json()
    assert result['actor']['id'] == actors[role]['id']
    return result, headers(result['session_token'])


@pytest.mark.parametrize('role', ['floor','office','owner'])
def test_exchange_hash_only_and_personal_permissions(client, actors, db_cursor, role):
    result, h = sign_in(client, actors, role)
    assert result['key_kind'] == 'session' and result['idle_minutes'] == 180
    assert len(result['session_token']) >= 47
    db_cursor.execute('SELECT * FROM actor_sessions WHERE token_hash=%s', (key_sessions.digest(result['session_token']),))
    row = db_cursor.fetchone()
    assert result['session_token'] not in str(row) and actors[role]['key'] not in str(row)
    assert row['expires_at'] - row['created_at'] >= timedelta(minutes=179, seconds=59)
    who = client.get('/auth/whoami', headers=h)
    assert who.status_code == 200 and who.json()['key_kind'] == 'session'
    assert who.json()['actor']['name'] == actors[role]['name']
    personal = client.get('/auth/whoami', headers=headers(actors[role]['key'])).json()
    assert who.json()['permissions'] == personal['permissions']
    if role == 'office':
        denied = client.post('/make', headers=h, json={})
        assert denied.status_code == 403 and denied.json()['detail']['error_code'] == 'ROLE_NOT_ALLOWED'


@pytest.mark.parametrize('minutes,expected', [(179.99,200),(180,401),(181,401)])
def test_idle_boundary_slides_or_ends(client, actors, db_cursor, minutes, expected):
    result,h = sign_in(client,actors)
    db_cursor.execute("""UPDATE actor_sessions SET last_seen_at=clock_timestamp()-(%s * interval '1 minute'),
        expires_at=clock_timestamp()+((180-%s) * interval '1 minute') WHERE token_hash=%s""",
        (minutes,minutes,key_sessions.digest(result['session_token'])))
    r=client.get('/auth/whoami',headers=h)
    assert r.status_code == expected
    db_cursor.execute('SELECT * FROM actor_sessions WHERE token_hash=%s',(key_sessions.digest(result['session_token']),))
    row=db_cursor.fetchone()
    if expected == 200:
        assert row['expires_at']-row['last_seen_at'] == timedelta(minutes=180)
    else:
        assert r.json()['detail']['error_code']=='SESSION_EXPIRED'
        assert row['ended_reason']=='idle'


def test_logout_ends_session_and_prevents_commit(client,actors,payload,db_cursor):
    _,h=sign_in(client,actors)
    prepared=client.post('/receive/prepare',headers=h,json=payload).json()
    assert client.post('/auth/session/logout',headers=h).json()['signed_out'] is True
    committed=client.post('/tickets/'+prepared['ticket']+'/commit',headers=h,json={'payload_hash':prepared['payload_hash']})
    assert committed.status_code==401 and committed.json()['detail']['error_code']=='SESSION_EXPIRED'
    db_cursor.execute("SELECT ended_reason FROM actor_sessions WHERE actor_id=%s",(actors['floor']['id'],))
    assert db_cursor.fetchone()['ended_reason']=='logout'


@pytest.mark.parametrize('change',['inactive','rotated'])
def test_revoke_live_key_ends_all_sessions_without_cache_delay(client,actors,db_cursor,change):
    _,h=sign_in(client,actors)
    _,other=sign_in(client,actors)
    assert client.get('/auth/whoami',headers=h).status_code==200
    if change=='inactive':
        db_cursor.execute('UPDATE actors SET active=false WHERE id=%s',(actors['floor']['id'],))
    else:
        db_cursor.execute('UPDATE actors SET key_hash=%s WHERE id=%s',(sha256(uuid4().bytes).hexdigest(),actors['floor']['id']))
    for credential in (h,other):
        r=client.get('/auth/whoami',headers=credential)
        assert r.status_code==401 and r.json()['detail']['error_code']=='SESSION_EXPIRED'
    db_cursor.execute('SELECT ended_reason FROM actor_sessions WHERE actor_id=%s',(actors['floor']['id'],))
    assert [r['ended_reason'] for r in db_cursor.fetchall()]==['revoked','revoked']


@pytest.mark.parametrize('kind',['master','dashboard','retired','unknown','session'])
def test_shared_inactive_unknown_and_session_credentials_cannot_exchange(client,actors,kind):
    key={'master':main.API_KEY,'dashboard':main.DASHBOARD_API_KEY,'retired':actors['retired']['key'],
         'unknown':'invalid-key','session':'fls_invalid'}[kind]
    r=client.post('/auth/session',json={'actor_key':key})
    assert r.status_code==401 and 'session_token' not in r.text and key not in r.text


def test_shared_key_refused_even_if_actor_hash_collides(client,actors,db_cursor):
    db_cursor.execute('UPDATE actors SET key_hash=%s WHERE id=%s',(key_sessions.digest(main.API_KEY),actors['floor']['id']))
    assert client.post('/auth/session',json={'actor_key':main.API_KEY}).status_code==401


def test_actor_key_with_session_prefix_still_resolves_as_actor(client,actors,db_cursor):
    key='fls_personal_'+uuid4().hex
    db_cursor.execute('UPDATE actors SET key_hash=%s WHERE id=%s',(key_sessions.digest(key),actors['floor']['id']))
    main._reset_actor_cache()
    assert client.get('/auth/whoami',headers=headers(key)).json()['key_kind']=='actor'
    assert client.post('/auth/session',json={'actor_key':key}).status_code==200


def test_rate_limits_failures_and_ignores_forwarded_header(client,actors,db_cursor):
    for n in range(5):
        assert client.post('/auth/session',json={'actor_key':'invalid'},headers={'X-Forwarded-For':str(n)}).status_code==401
    locked=client.post('/auth/session',json={'actor_key':actors['floor']['key']})
    assert locked.status_code==429 and int(locked.headers['retry-after'])>0
    db_cursor.execute('SELECT count(*) AS n FROM key_session_attempts WHERE NOT ok')
    assert db_cursor.fetchone()['n']==5


def test_successful_exchanges_also_rate_limited(client,actors):
    for n in range(10): sign_in(client,actors)
    assert client.post('/auth/session',json={'actor_key':actors['floor']['key']}).status_code==429


def test_global_limit_spans_sources(client,actors,db_cursor):
    db_cursor.execute("INSERT INTO key_session_attempts(source_hash,ok) SELECT 'source-'||n,false FROM generate_series(1,50) n")
    assert client.post('/auth/session',json={'actor_key':actors['floor']['key']}).status_code==429


def test_flag_off_needs_no_session_tables_and_preserves_existing_auth(client,actors,monkeypatch,db_cursor):
    monkeypatch.setenv('KEY_SESSIONS_ENABLED','0')
    db_cursor.execute('DROP TABLE actor_sessions,key_session_attempts')
    for key,kind in [(main.API_KEY,'legacy_ledger'),(main.DASHBOARD_API_KEY,'legacy_dashboard'),(actors['floor']['key'],'actor')]:
        r=client.get('/auth/whoami',headers=headers(key))
        assert r.status_code==200 and r.json()['key_kind']==kind
    for route in ['/auth/session','/auth/session/logout']:
        assert client.post(route,json={'actor_key':'test'}).status_code==404
    assert client.get('/auth/whoami',headers=headers('fls_invalid')).status_code==403


def test_pilot_independent_of_pin_flag(client,actors,monkeypatch):
    for pin in ['0','1']:
        monkeypatch.setenv('PIN_LOGIN_ENABLED',pin)
        sign_in(client,actors)
    assert client.post('/auth/session',json={'pin':'1234'}).status_code==422


def test_assistant_session_prepare_record_replay_and_attribution(client,actors,payload,db_cursor,monkeypatch):
    monkeypatch.setenv('ASSISTANT_ENABLED','1')
    _,h=sign_in(client,actors)
    sid=client.post('/assistant/session',headers=h).json()['session_id']
    card,_=make_draft(client,(sid,h),payload,db_cursor,monkeypatch)
    first=record(client,(sid,h),card)
    assert first.status_code==200 and first.json()['kind']=='receipt'
    replay=record(client,(sid,h),card)
    assert replay.json()['result']['receipt_number']==first.json()['result']['receipt_number']
    db_cursor.execute('SELECT entered_by_actor_id FROM transactions WHERE id=%s',(first.json()['result']['transaction_id'],))
    assert db_cursor.fetchone()['entered_by_actor_id']==actors['floor']['id']
    db_cursor.execute('SELECT actor_id,expires_at-prepared_at AS ttl FROM write_tickets WHERE receipt_number=%s',(first.json()['result']['receipt_number'],))
    ticket=db_cursor.fetchone()
    assert ticket['actor_id']==actors['floor']['id'] and ticket['ttl']==timedelta(minutes=10)


def test_new_session_same_person_can_commit_but_handover_cannot(client,actors,payload):
    _,h=sign_in(client,actors)
    prepared=client.post('/receive/prepare',headers=h,json=payload).json()
    _,other=sign_in(client,actors,'office')
    path='/tickets/'+prepared['ticket']+'/commit'
    body={'payload_hash':prepared['payload_hash']}
    assert client.post(path,headers=other,json=body).json()['detail']['error_code']=='TICKET_WRONG_USER'
    client.post('/auth/session/logout',headers=h)
    _,renewed=sign_in(client,actors)
    assert client.post(path,headers=renewed,json=body).status_code==200


def test_migration_rerun_preserves_sessions_and_private_tables(client,actors,db_cursor):
    result,_=sign_in(client,actors)
    migration=Path(__file__).resolve().parents[1]/'migrations/077_key_sessions.sql'
    db_cursor.execute(migration.read_text())
    db_cursor.execute('SELECT token_hash FROM actor_sessions WHERE token_hash=%s',(key_sessions.digest(result['session_token']),))
    assert db_cursor.fetchone()
    db_cursor.execute("SELECT relrowsecurity FROM pg_class WHERE oid IN ('actor_sessions'::regclass,'key_session_attempts'::regclass)")
    assert all(r['relrowsecurity'] for r in db_cursor.fetchall())


@pytest.mark.parametrize('body',[{'actor_key':123}, {'actor_key':'secret-value','unexpected':'secret-value'}, {'actor_key':'secret-value'*60}])
def test_exchange_validation_never_echoes_credentials(client,body):
    r=client.post('/auth/session',json=body)
    assert r.status_code==422 and 'secret-value' not in r.text and '123' not in r.text
