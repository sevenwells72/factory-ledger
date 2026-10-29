"""Phone PIN sign-in (migration 089) on top of key sessions: real HTTP/DB."""
from datetime import timedelta
from pathlib import Path
import pytest
import main
import key_sessions
import device_pins
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_write_tickets import payload, headers  # noqa: F401

pytestmark = pytest.mark.db
MIGRATION = Path(__file__).resolve().parents[1] / 'migrations/089_device_pins.sql'
DOWN = Path(__file__).resolve().parents[1] / 'migrations/down/089_device_pins_down.sql'


@pytest.fixture(autouse=True)
def pins(monkeypatch, db_cursor):
    monkeypatch.setenv('KEY_SESSIONS_ENABLED', '1')
    monkeypatch.setenv('PIN_SIGNIN_ENABLED', '1')
    # Real PBKDF2, fewer rounds so the suite stays fast; one test checks production rounds.
    monkeypatch.setattr(device_pins, 'PIN_ITERATIONS', 1000)
    db_cursor.execute(MIGRATION.read_text())


def key_session(client, actors, role='floor'):
    r = client.post('/auth/session', json={'actor_key': actors[role]['key']})
    assert r.status_code == 200, r.text
    return r.json()['session_token']


def enroll(client, actors, role='floor', pin='482916', label='Phone'):
    token = key_session(client, actors, role)
    r = client.post('/auth/device/enroll', headers=headers(token), json={'actor_key': actors[role]['key'], 'pin': pin, 'label': label})
    assert r.status_code == 200, r.text
    return r.json(), token


def pin(client, secret, value):
    return client.post('/auth/pin', json={'device_secret': secret, 'pin': value})


def session_row(db_cursor, token):
    db_cursor.execute('SELECT * FROM key_actor_sessions WHERE token_hash=%s', (key_sessions.digest(token),))
    return db_cursor.fetchone()


def test_enroll_from_personal_key_stores_only_hashes_and_opens_12_hour_phone_session(client, actors, db_cursor):
    result, key_token = enroll(client, actors)
    secret = result['device_secret']
    assert len(secret) == 43 and result['pin_session_hours'] == 12 and result['actor']['id'] == actors['floor']['id']
    db_cursor.execute('SELECT * FROM key_devices WHERE actor_id=%s', (actors['floor']['id'],))
    device = db_cursor.fetchone()
    assert device['secret_hash'] == key_sessions.digest(secret)
    assert secret not in str(device) and '482916' not in str(device) and actors['floor']['key'] not in str(device)
    assert device['pin_iterations'] == 1000 and len(device['pin_hash']) == 64
    # The personal-key session was turned into the phone session.
    assert session_row(db_cursor, key_token)['ended_reason'] == 'logout'
    row = session_row(db_cursor, result['session_token'])
    assert row['device_id'] == device['id']
    assert timedelta(hours=11, minutes=59) < row['expires_at'] - row['created_at'] <= timedelta(hours=12, seconds=5)
    who = client.get('/auth/whoami', headers=headers(result['session_token'])).json()
    assert who['key_kind'] == 'session' and who['actor']['name'] == actors['floor']['name']


def test_production_rounds_are_strong(monkeypatch):
    monkeypatch.undo()
    assert device_pins.PIN_ITERATIONS >= 200_000


def test_pin_sign_in_works_only_with_the_enrolled_phone(client, actors, db_cursor):
    floor, _ = enroll(client, actors, 'floor', '482916')
    office, _ = enroll(client, actors, 'office', '730518')
    ok = pin(client, floor['device_secret'], '482916')
    assert ok.status_code == 200 and ok.json()['actor']['id'] == actors['floor']['id']
    # Right PIN, wrong or made-up phone: refused the same way.
    for secret in [office['device_secret'], 'x' * 43, 'short', floor['device_secret'][:-1] + 'A']:
        r = pin(client, secret, '482916')
        assert r.status_code == 401 and r.json()['detail']['error_code'] in ('DEVICE_NOT_RECOGNIZED', 'PIN_INVALID')
    assert pin(client, office['device_secret'], '482916').json()['detail']['error_code'] == 'PIN_INVALID'
    assert pin(client, 'x' * 43, '482916').json()['detail']['error_code'] == 'DEVICE_NOT_RECOGNIZED'


def test_phone_session_keeps_fixed_12_hours_and_same_permissions_as_personal_key(client, actors, db_cursor):
    for role in ['floor', 'office', 'owner']:
        result, _ = enroll(client, actors, role, '482916')
        h = headers(result['session_token'])
        before = session_row(db_cursor, result['session_token'])['expires_at']
        who = client.get('/auth/whoami', headers=h).json()
        assert session_row(db_cursor, result['session_token'])['expires_at'] == before
        personal = client.get('/auth/whoami', headers=headers(actors[role]['key'])).json()
        assert who['permissions'] == personal['permissions'] and who['actor']['role'] == role
        if role == 'office':
            denied = client.post('/make', headers=h, json={})
            assert denied.status_code == 403 and denied.json()['detail']['error_code'] == 'ROLE_NOT_ALLOWED'
    # At the 12-hour mark the phone session ends like any expired session.
    db_cursor.execute("UPDATE key_actor_sessions SET expires_at=clock_timestamp()-interval '1 second' WHERE token_hash=%s",
                      (key_sessions.digest(result['session_token']),))
    r = client.get('/auth/whoami', headers=h)
    assert r.status_code == 401 and r.json()['detail']['error_code'] == 'SESSION_EXPIRED'


def test_five_wrong_pins_lock_only_that_phone(client, actors, db_cursor):
    floor, _ = enroll(client, actors, 'floor', '482916')
    office, _ = enroll(client, actors, 'office', '730518')
    lefts = [pin(client, floor['device_secret'], '111112').json()['detail'].get('attempts_left') for _ in range(4)]
    assert lefts == [4, 3, 2, 1]
    locked = pin(client, floor['device_secret'], '111112')
    assert locked.status_code == 429 and locked.json()['detail']['error_code'] == 'PIN_LOCKED' and int(locked.headers['retry-after']) > 0
    # Even the right PIN waits on the locked phone...
    assert pin(client, floor['device_secret'], '482916').status_code == 429
    # ...while other people, and personal keys, are unaffected.
    assert pin(client, office['device_secret'], '730518').status_code == 200
    assert client.post('/auth/session', json={'actor_key': actors['owner']['key']}).status_code == 200
    db_cursor.execute("UPDATE key_devices SET locked_until=clock_timestamp()-interval '1 second' WHERE actor_id=%s", (actors['floor']['id'],))
    assert pin(client, floor['device_secret'], '482916').status_code == 200
    db_cursor.execute('SELECT failed_count FROM key_devices WHERE actor_id=%s', (actors['floor']['id'],))
    assert db_cursor.fetchone()['failed_count'] == 0


def test_ten_wrong_pins_block_until_owner_reset_then_new_pin(client, actors, db_cursor):
    floor, _ = enroll(client, actors, 'floor', '482916')
    owner, _ = enroll(client, actors, 'owner', '602817')
    phone = headers(floor['session_token'])
    for n in range(10):
        db_cursor.execute('UPDATE key_devices SET locked_until=NULL WHERE actor_id=%s', (actors['floor']['id'],))
        r = pin(client, floor['device_secret'], '999998')
    assert r.status_code == 423 and r.json()['detail']['error_code'] == 'PIN_BLOCKED'
    assert pin(client, floor['device_secret'], '482916').status_code == 423
    # Blocking ends that phone's open session too.
    assert client.get('/auth/whoami', headers=phone).status_code == 401
    listed = client.get('/auth/devices', headers=headers(owner['session_token'])).json()
    device = next(p for p in listed['people'] if p['id'] == actors['floor']['id'])['devices'][0]
    assert device['status'] == 'blocked'
    reset = client.post(f"/auth/devices/{device['id']}/reset-pin", headers=headers(owner['session_token']))
    assert reset.status_code == 200
    need = pin(client, floor['device_secret'], '482916')
    assert need.status_code == 409 and need.json()['detail'] == {**need.json()['detail'], 'error_code': 'PIN_RESET_REQUIRED', 'reset_open': True}
    weak = client.post('/auth/pin/reset', json={'device_secret': floor['device_secret'], 'new_pin': '123456'})
    assert weak.status_code == 422 and weak.json()['detail']['error_code'] == 'PIN_TOO_SIMPLE'
    done = client.post('/auth/pin/reset', json={'device_secret': floor['device_secret'], 'new_pin': '593017'})
    assert done.status_code == 200 and done.json()['actor']['id'] == actors['floor']['id']
    assert pin(client, floor['device_secret'], '593017').status_code == 200
    assert pin(client, floor['device_secret'], '482916').status_code == 401


def test_reset_window_expires(client, actors, db_cursor):
    floor, _ = enroll(client, actors, 'floor', '482916')
    owner, _ = enroll(client, actors, 'owner', '602817')
    db_cursor.execute('SELECT id FROM key_devices WHERE actor_id=%s', (actors['floor']['id'],))
    device_id = db_cursor.fetchone()['id']
    client.post(f'/auth/devices/{device_id}/reset-pin', headers=headers(owner['session_token']))
    db_cursor.execute("UPDATE key_devices SET pin_reset_until=clock_timestamp()-interval '1 second' WHERE id=%s", (device_id,))
    assert pin(client, floor['device_secret'], '482916').json()['detail']['reset_open'] is False
    late = client.post('/auth/pin/reset', json={'device_secret': floor['device_secret'], 'new_pin': '593017'})
    assert late.status_code == 409 and late.json()['detail']['error_code'] == 'PIN_RESET_NOT_OPEN'


def test_unrecognised_phones_share_a_bucket_that_never_locks_enrolled_phones(client, actors, db_cursor):
    floor, _ = enroll(client, actors, 'floor', '482916')
    db_cursor.execute('INSERT INTO key_device_attempts(device_id,ok) SELECT NULL,false FROM generate_series(1,%s)', (device_pins.UNKNOWN_LIMIT,))
    r = pin(client, 'y' * 43, '482916')
    assert r.status_code == 429 and r.json()['detail']['error_code'] == 'PIN_RATE_LIMITED'
    assert pin(client, floor['device_secret'], '482916').status_code == 200


@pytest.mark.parametrize('value,code', [('12345', 'PIN_FORMAT'), ('1234567', 'PIN_FORMAT'), ('12a456', 'PIN_FORMAT'),
                                        ('000000', 'PIN_TOO_SIMPLE'), ('123456', 'PIN_TOO_SIMPLE'), ('654321', 'PIN_TOO_SIMPLE'),
                                        ('121212', 'PIN_TOO_SIMPLE'), ('437437', 'PIN_TOO_SIMPLE')])
def test_enroll_refuses_bad_pins_without_echo(client, actors, value, code):
    token = key_session(client, actors)
    r = client.post('/auth/device/enroll', headers=headers(token), json={'actor_key': actors['floor']['key'], 'pin': value, 'label': 'Phone'})
    assert r.status_code == 422 and r.json()['detail']['error_code'] == code and value not in r.text


@pytest.mark.parametrize('route,body', [('/auth/pin', {'device_secret': 'secret-value', 'pin': 4829}),
                                        ('/auth/pin', {'device_secret': 'secret-value', 'pin': '482916', 'extra': 'secret-value'}),
                                        ('/auth/device/enroll', {'pin': '482916-secret-value' * 4, 'label': 'x'})])
def test_validation_never_echoes_pins_or_secrets(client, route, body):
    r = client.post(route, json=body)
    assert r.status_code in (401, 403, 422) and 'secret-value' not in r.text and '4829' not in r.text


def test_enrollment_needs_a_personal_key_session(client, actors):
    phone, _ = enroll(client, actors)
    body = {'actor_key': actors['floor']['key'], 'pin': '593017', 'label': 'Second'}
    # Not from a PIN session, not with the personal key directly, not with shared keys.
    r = client.post('/auth/device/enroll', headers=headers(phone['session_token']), json=body)
    assert r.status_code == 403 and r.json()['detail']['error_code'] == 'PERSONAL_KEY_REQUIRED'
    for key in [actors['floor']['key'], main.API_KEY, main.DASHBOARD_API_KEY]:
        assert client.post('/auth/device/enroll', headers=headers(key), json=body).status_code == 403
    assert client.post('/auth/device/enroll', json=body).status_code == 401


def test_device_limit_and_replacing_this_phone(client, actors, db_cursor):
    first, _ = enroll(client, actors, label='One')
    enroll(client, actors, label='Two')
    enroll(client, actors, label='Three')
    token = key_session(client, actors)
    over = client.post('/auth/device/enroll', headers=headers(token), json={'actor_key': actors['floor']['key'], 'pin': '593017', 'label': 'Four'})
    assert over.status_code == 409 and over.json()['detail']['error_code'] == 'DEVICE_LIMIT'
    again = client.post('/auth/device/enroll', headers=headers(token),
                        json={'actor_key': actors['floor']['key'], 'pin': '593017', 'label': 'One again', 'replace_device_secret': first['device_secret']})
    assert again.status_code == 200
    assert pin(client, first['device_secret'], '482916').json()['detail']['error_code'] == 'DEVICE_NOT_RECOGNIZED'
    assert client.get('/auth/whoami', headers=headers(first['session_token'])).status_code == 401


def test_change_own_pin_from_phone_session(client, actors, db_cursor):
    phone, key_token = enroll(client, actors)
    body = {'device_secret': phone['device_secret'], 'current_pin': '482916', 'new_pin': '593017'}
    # A personal-key session is not this phone's PIN session.
    other = key_session(client, actors)
    assert client.post('/auth/pin/change', headers=headers(other), json=body).json()['detail']['error_code'] == 'DEVICE_SESSION_REQUIRED'
    wrong = client.post('/auth/pin/change', headers=headers(phone['session_token']), json={**body, 'current_pin': '000001'})
    assert wrong.status_code == 401 and wrong.json()['detail']['error_code'] == 'PIN_INVALID'
    weak = client.post('/auth/pin/change', headers=headers(phone['session_token']), json={**body, 'new_pin': '111111'})
    assert weak.status_code == 422
    assert client.post('/auth/pin/change', headers=headers(phone['session_token']), json=body).json()['pin_changed'] is True
    assert pin(client, phone['device_secret'], '482916').status_code == 401
    assert pin(client, phone['device_secret'], '593017').status_code == 200
    db_cursor.execute("SELECT action FROM key_device_audit WHERE actor_id=%s ORDER BY id", (actors['floor']['id'],))
    assert [r['action'] for r in db_cursor.fetchall()] == ['enroll', 'pin_change']


def test_forget_this_phone(client, actors):
    phone, _ = enroll(client, actors)
    h = headers(phone['session_token'])
    assert client.post('/auth/device/forget', headers=h, json={'device_secret': phone['device_secret']}).json()['forgotten'] is True
    assert client.get('/auth/whoami', headers=h).status_code == 401
    assert pin(client, phone['device_secret'], '482916').json()['detail']['error_code'] == 'DEVICE_NOT_RECOGNIZED'


def test_owner_screen_lists_without_secrets_and_refuses_everyone_else(client, actors):
    floor, _ = enroll(client, actors, 'floor')
    owner, _ = enroll(client, actors, 'owner', '602817')
    listed = client.get('/auth/devices', headers=headers(owner['session_token']))
    assert listed.status_code == 200
    text = listed.text
    for secret in [floor['device_secret'], owner['device_secret'], '482916', actors['floor']['key']]:
        assert secret not in text
    assert 'pin_hash' not in text and 'secret_hash' not in text and 'key_hash' not in text
    person = next(p for p in listed.json()['people'] if p['id'] == actors['floor']['id'])
    assert person['devices'][0]['status'] == 'active' and person['devices'][0]['label'] == 'Phone'
    assert client.get('/auth/devices', headers=headers(floor['session_token'])).json()['detail']['error_code'] == 'OWNER_REQUIRED'
    # The owner's personal key, the master (GPT) key and the dashboard key are not sessions.
    for key in [actors['owner']['key'], main.API_KEY, main.DASHBOARD_API_KEY]:
        assert client.get('/auth/devices', headers=headers(key)).status_code == 403


def test_owner_revokes_one_phone_and_only_its_sessions_end(client, actors, db_cursor):
    one, _ = enroll(client, actors, label='One')
    two, _ = enroll(client, actors, label='Two')
    personal = key_session(client, actors)
    owner, _ = enroll(client, actors, 'owner', '602817')
    db_cursor.execute("SELECT id FROM key_devices WHERE label='One' AND actor_id=%s", (actors['floor']['id'],))
    r = client.post(f"/auth/devices/{db_cursor.fetchone()['id']}/revoke", headers=headers(owner['session_token']))
    assert r.status_code == 200
    assert client.get('/auth/whoami', headers=headers(one['session_token'])).status_code == 401
    assert client.get('/auth/whoami', headers=headers(two['session_token'])).status_code == 200
    assert client.get('/auth/whoami', headers=headers(personal)).status_code == 200


def test_remove_access_and_deactivate(client, actors, db_cursor):
    phone, _ = enroll(client, actors)
    personal = key_session(client, actors)
    owner, _ = enroll(client, actors, 'owner', '602817')
    oh = headers(owner['session_token'])
    fid = actors['floor']['id']
    assert client.post(f'/auth/actors/{fid}/remove-access', headers=oh).status_code == 200
    for token in [phone['session_token'], personal]:
        assert client.get('/auth/whoami', headers=headers(token)).status_code == 401
    assert pin(client, phone['device_secret'], '482916').status_code == 401
    # The personal key still works after "remove access" (it is not deactivation).
    assert client.post('/auth/session', json={'actor_key': actors['floor']['key']}).status_code == 200
    assert client.post(f'/auth/actors/{fid}/active', headers=oh, json={'active': False}).json()['active'] is False
    main._reset_actor_cache()
    assert client.post('/auth/session', json={'actor_key': actors['floor']['key']}).status_code == 401
    db_cursor.execute('SELECT role,active FROM actors WHERE id=%s', (fid,))
    assert dict(db_cursor.fetchone()) == {'role': 'floor', 'active': False}
    assert client.post(f'/auth/actors/{fid}/active', headers=oh, json={'active': True}).json()['active'] is True
    main._reset_actor_cache()
    assert client.post('/auth/session', json={'actor_key': actors['floor']['key']}).status_code == 200
    # Not yourself, and not another owner's account.
    assert client.post(f"/auth/actors/{actors['owner']['id']}/active", headers=oh, json={'active': False}).status_code == 409
    db_cursor.execute("SELECT action FROM key_device_audit WHERE actor_id=%s ORDER BY id", (fid,))
    assert [r['action'] for r in db_cursor.fetchall()] == ['enroll', 'remove_access', 'deactivate', 'reactivate']


def test_rotating_personal_key_retires_the_phone(client, actors, db_cursor):
    phone, _ = enroll(client, actors)
    db_cursor.execute("UPDATE actors SET key_hash=%s WHERE id=%s", ('0' * 64, actors['floor']['id']))
    assert client.get('/auth/whoami', headers=headers(phone['session_token'])).status_code == 401
    assert pin(client, phone['device_secret'], '482916').json()['detail']['error_code'] == 'DEVICE_NOT_RECOGNIZED'
    db_cursor.execute('SELECT revoked_reason FROM key_devices WHERE actor_id=%s', (actors['floor']['id'],))
    assert db_cursor.fetchone()['revoked_reason'] == 'key_rotated'


def test_pin_switch_off_hides_routes_and_ends_phone_sessions_only(client, actors, monkeypatch):
    phone, _ = enroll(client, actors)
    personal = key_session(client, actors)
    monkeypatch.setenv('PIN_SIGNIN_ENABLED', '0')
    config = client.get('/auth/session/config').json()
    assert config['key_sessions_enabled'] is True and config['pin_sign_in'] is False
    assert pin(client, phone['device_secret'], '482916').status_code == 404
    assert client.get('/auth/whoami', headers=headers(phone['session_token'])).status_code == 401
    assert client.get('/auth/whoami', headers=headers(personal)).status_code == 200
    assert client.get('/auth/whoami', headers=headers(main.API_KEY)).json()['key_kind'] == 'legacy_ledger'


def test_config_reports_pin_sign_in(client):
    config = client.get('/auth/session/config').json()
    assert {k: config[k] for k in ('key_sessions_enabled', 'idle_minutes', 'pin_sign_in', 'pin_session_hours')} == {
        'key_sessions_enabled': True, 'idle_minutes': 180, 'pin_sign_in': True, 'pin_session_hours': 12}


def test_migration_rerun_and_down_keep_personal_sessions(client, actors, db_cursor, monkeypatch):
    phone, _ = enroll(client, actors)
    personal = key_session(client, actors)
    db_cursor.execute(MIGRATION.read_text())
    db_cursor.execute("SELECT relrowsecurity FROM pg_class WHERE oid IN ('key_devices'::regclass,'key_device_attempts'::regclass,'key_device_audit'::regclass)")
    assert all(r['relrowsecurity'] for r in db_cursor.fetchall())
    assert pin(client, phone['device_secret'], '482916').status_code == 200
    # The runbook turns the switch off before rolling the migration back.
    monkeypatch.setenv('PIN_SIGNIN_ENABLED', '0')
    db_cursor.execute(DOWN.read_text())
    assert session_row(db_cursor, phone['session_token'])['ended_reason'] == 'revoked'
    assert client.get('/auth/whoami', headers=headers(personal)).status_code == 200


def test_enrollment_asks_for_the_personal_key_again(client, actors, db_cursor):
    # A signed-in screen left open is not enough to add a phone.
    token = key_session(client, actors)
    for wrong in ['not-my-key', actors['office']['key'], main.API_KEY]:
        r = client.post('/auth/device/enroll', headers=headers(token), json={'actor_key': wrong, 'pin': '482916', 'label': 'Phone'})
        assert r.status_code == 401 and r.json()['detail']['error_code'] == 'ACTOR_KEY_INVALID' and wrong not in r.text
    db_cursor.execute('SELECT count(*) AS n FROM key_devices WHERE actor_id=%s', (actors['floor']['id'],))
    assert db_cursor.fetchone()['n'] == 0
    db_cursor.execute('SELECT count(*) AS n FROM key_session_attempts WHERE NOT ok')
    assert db_cursor.fetchone()['n'] == 3
    for _ in range(2):
        client.post('/auth/device/enroll', headers=headers(token), json={'actor_key': 'x', 'pin': '482916', 'label': 'Phone'})
    limited = client.post('/auth/device/enroll', headers=headers(token), json={'actor_key': actors['floor']['key'], 'pin': '482916', 'label': 'Phone'})
    assert limited.status_code == 429


def test_owner_cannot_manage_another_owner_but_can_manage_own_phone(client, actors, db_cursor):
    db_cursor.execute("INSERT INTO actors(name,role,key_hash,active) VALUES ('Second owner','owner',%s,true) RETURNING id",
                      (key_sessions.digest('second-owner-key'),))
    second = db_cursor.fetchone()['id']
    main._reset_actor_cache()
    token = client.post('/auth/session', json={'actor_key': 'second-owner-key'}).json()['session_token']
    other = client.post('/auth/device/enroll', headers=headers(token),
                        json={'actor_key': 'second-owner-key', 'pin': '730518', 'label': 'Other owner phone'}).json()
    mine, _ = enroll(client, actors, 'owner', '602817', label='My phone')
    oh = headers(mine['session_token'])
    db_cursor.execute('SELECT id,actor_id FROM key_devices WHERE revoked_at IS NULL')
    ids = {r['actor_id']: r['id'] for r in db_cursor.fetchall()}
    for action in ['revoke', 'reset-pin']:
        r = client.post(f"/auth/devices/{ids[second]}/{action}", headers=oh)
        assert r.status_code == 409 and r.json()['detail']['error_code'] == 'OWNER_PROTECTED'
    assert client.post(f'/auth/actors/{second}/remove-access', headers=oh).status_code == 409
    assert client.post(f'/auth/actors/{second}/active', headers=oh, json={'active': False}).status_code == 409
    assert pin(client, other['device_secret'], '730518').status_code == 200
    # A lost phone of your own can be removed from another device.
    assert client.post(f"/auth/devices/{ids[actors['owner']['id']]}/revoke", headers=oh).status_code == 200


def test_switch_on_before_migration_keeps_ordinary_sign_in_working(client, actors, db_cursor, monkeypatch):
    db_cursor.execute(DOWN.read_text())
    key_sessions._PINS_MIGRATED.clear()
    token = key_session(client, actors)
    assert client.get('/auth/whoami', headers=headers(token)).status_code == 200
    assert client.get('/auth/session/config').json()['pin_sign_in'] is False
    assert pin(client, 'x' * 43, '482916').status_code == 404
