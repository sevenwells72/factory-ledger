"""Phone PIN sign-in on top of the live key sessions (migration 089).

A phone is enrolled once from a personal-key session. It then holds a random
32-byte device secret; the server stores only its sha256. Everyday sign-in
sends that secret plus a 4-digit PIN and opens a 12-hour session on that phone
only. The PIN hash is PBKDF2 salted with a random salt AND the device secret,
so the database alone cannot be brute-forced back to a PIN.

Limits are per phone: 5 wrong PINs in a row lock that phone for 15 minutes,
10 block it until the owner resets the PIN. Unrecognised phones share one
separate bucket. One person's mistakes never lock anyone else out.

Off unless KEY_SESSIONS_ENABLED=1 and PIN_SIGNIN_ENABLED=1; with either off
these routes return 404 and nothing reads the 089 tables.
"""
import hashlib
import hmac
import re
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, StrictBool, StrictStr

import key_sessions
from key_sessions import digest

PIN_ITERATIONS = 200_000
MAX_DEVICES = 3
LOCK_AFTER = 5
LOCK_MINUTES = 15
BLOCK_AFTER = 10
RESET_MINUTES = 60
UNKNOWN_LIMIT = 30          # unrecognised-phone attempts per 10 minutes, all together
SECRET = re.compile(r'[A-Za-z0-9_-]{43}')
LOCK_UNKNOWN = 770182       # advisory lock for the shared unrecognised bucket
LOCK_EXCHANGE = 770180      # key_sessions' personal-key limiter
LOCK_PERSON = 770181        # key_sessions.resolve's per-person lock, taken first everywhere


class Strict(BaseModel):
    class Config:
        extra = 'forbid'


class Enroll(Strict):
    # The personal key again: a signed-in screen left open is not enough to add a phone.
    actor_key: StrictStr = Field(..., min_length=1, max_length=512)
    pin: StrictStr = Field(..., max_length=16)
    label: StrictStr = Field(..., min_length=1, max_length=60)
    replace_device_secret: Optional[StrictStr] = Field(None, max_length=64)


class PinSignIn(Strict):
    device_secret: StrictStr = Field(..., max_length=64)
    pin: StrictStr = Field(..., max_length=16)


class PinReset(Strict):
    device_secret: StrictStr = Field(..., max_length=64)
    new_pin: StrictStr = Field(..., max_length=16)


class PinChange(Strict):
    device_secret: StrictStr = Field(..., max_length=64)
    current_pin: StrictStr = Field(..., max_length=16)
    new_pin: StrictStr = Field(..., max_length=16)


class Forget(Strict):
    device_secret: StrictStr = Field(..., max_length=64)


class Active(Strict):
    active: StrictBool


def problem(status, code, en, es, **extra):
    return HTTPException(status, {'error_code': code, 'message': f'{en} / {es}', **extra})


def new_pin_problem(pin):
    # Owner decision 2026-10-29: the PIN identifies who is entering, it is not a
    # security control, so any 4 digits are fine (1111, 2222 …).
    if not re.fullmatch(r'[0-9]{4}', pin):
        return problem(422, 'PIN_FORMAT', 'Use exactly 4 digits.', 'Usa exactamente 4 dígitos.')
    return None


def pin_hash(secret, pin, salt, iterations):
    return hashlib.pbkdf2_hmac('sha256', b'fl-pin-v1:' + pin.encode(),
                               bytes.fromhex(salt) + secret.encode(), iterations).hex()


def pin_columns(secret, pin):
    salt = secrets.token_hex(16)
    return pin_hash(secret, pin, salt, PIN_ITERATIONS), salt, PIN_ITERATIONS


def pin_matches(device, secret, pin):
    if not device['pin_hash'] or not re.fullmatch(r'[0-9]{4}', pin):
        return False
    return hmac.compare_digest(pin_hash(secret, pin, device['pin_salt'], device['pin_iterations']), device['pin_hash'])


def lock_person(cur, actor_id):
    # Same lock, same order as key_sessions.resolve (person, then rows): no deadlocks.
    cur.execute('SELECT pg_advisory_xact_lock(%s,%s)', (LOCK_PERSON, actor_id))


def end_sessions(cur, where, params):
    cur.execute("UPDATE key_actor_sessions SET ended_at=clock_timestamp(), ended_reason='revoked' "
                "WHERE ended_at IS NULL AND " + where, params)


def audit(cur, actor_id, action, device_id=None, by=None):
    cur.execute('INSERT INTO key_device_audit(actor_id,device_id,action,by_actor_id) VALUES (%s,%s,%s,%s)',
                (actor_id, device_id, action, by if by is not None else actor_id))


def open_session(cur, device, actor):
    token = key_sessions.PREFIX + secrets.token_urlsafe(32)
    cur.execute(f"""INSERT INTO key_actor_sessions(token_hash,actor_id,actor_key_hash,expires_at,device_id)
        VALUES (%s,%s,%s,clock_timestamp()+interval '{key_sessions.PIN_SESSION_HOURS} hours',%s)
        RETURNING expires_at""", (digest(token), actor['id'], actor['key_hash'], device['id']))
    return {'session_token': token, 'key_kind': 'session',
            'actor': {k: actor[k] for k in ('id', 'name', 'role')},
            'expires_at': cur.fetchone()['expires_at'],
            'pin_session_hours': key_sessions.PIN_SESSION_HOURS,
            'device': {'id': device['id'], 'label': device['label']}}


def find_device(cur, secret):
    """The live device row for a secret (locked), with its person, or None."""
    if not isinstance(secret, str) or not SECRET.fullmatch(secret):
        return None
    cur.execute('SELECT actor_id FROM key_devices WHERE secret_hash=%s', (digest(secret),))
    owner = cur.fetchone()
    if not owner:
        return None
    lock_person(cur, owner['actor_id'])
    cur.execute("""SELECT d.*, a.name, a.role, a.active, a.key_hash, clock_timestamp() AS now
                   FROM key_devices d JOIN actors a ON a.id=d.actor_id
                   WHERE d.secret_hash=%s FOR UPDATE OF d FOR SHARE OF a""", (digest(secret),))
    device = cur.fetchone()
    if not device or device['revoked_at'] is not None:
        return None
    if not device['active'] or device['actor_key_hash'] != device['key_hash']:
        # The person was deactivated or their personal key rotated: retire the phone.
        reason = 'deactivated' if not device['active'] else 'key_rotated'
        cur.execute('UPDATE key_devices SET revoked_at=clock_timestamp(), revoked_reason=%s WHERE id=%s',
                    (reason, device['id']))
        end_sessions(cur, 'device_id=%s', (device['id'],))
        audit(cur, device['actor_id'], 'revoke', device['id'])
        return None
    return device


def unknown_attempt(cur):
    """Count one attempt from an unrecognised phone; a refusal to raise."""
    cur.execute('SELECT pg_advisory_xact_lock(%s)', (LOCK_UNKNOWN,))
    cur.execute("""SELECT count(*) AS n FROM key_device_attempts
                   WHERE device_id IS NULL AND attempted_at > clock_timestamp()-interval '10 minutes'""")
    if cur.fetchone()['n'] >= UNKNOWN_LIMIT:
        return problem(429, 'PIN_RATE_LIMITED', 'Too many sign-in attempts from unrecognised phones. Try again later.',
                       'Demasiados intentos desde teléfonos no reconocidos. Intenta más tarde.'), {'Retry-After': '600'}
    cur.execute('INSERT INTO key_device_attempts(device_id,ok) VALUES (NULL,false)')
    return problem(401, 'DEVICE_NOT_RECOGNIZED', 'This phone is not set up for a PIN. Sign in with your personal key.',
                   'Este teléfono no tiene PIN. Inicia sesión con tu clave personal.'), None


def gate(device):
    """A refusal for a phone that may not try a PIN right now, or None."""
    now = device['now']
    if device['failed_count'] >= BLOCK_AFTER:
        return problem(423, 'PIN_BLOCKED', 'Too many wrong PINs. Ask the owner to reset your PIN.',
                       'Demasiados PIN incorrectos. Pide al dueño que restablezca tu PIN.'), None
    if device['locked_until'] and device['locked_until'] > now:
        wait = max(1, int((device['locked_until'] - now).total_seconds()) + 1)
        return problem(429, 'PIN_LOCKED', f'Too many wrong PINs on this phone. Try again in {(wait + 59) // 60} minutes.',
                       f'Demasiados PIN incorrectos en este teléfono. Intenta en {(wait + 59) // 60} minutos.'), {'Retry-After': str(wait)}
    if not device['pin_hash']:
        reset_open = bool(device['pin_reset_until'] and device['pin_reset_until'] > now)
        return problem(409, 'PIN_RESET_REQUIRED',
                       'Your PIN was reset. Open My PIN to choose a new one.' if reset_open else 'Your PIN reset has expired. Ask the owner to reset it again, or sign in with your personal key.',
                       'Tu PIN fue restablecido. Abre Mi PIN para elegir uno nuevo.' if reset_open else 'El restablecimiento venció. Pide al dueño que lo haga otra vez, o usa tu clave personal.',
                       reset_open=reset_open), None
    return None


def wrong_pin(cur, device):
    failed = device['failed_count'] + 1
    lock = failed % LOCK_AFTER == 0 and failed < BLOCK_AFTER
    cur.execute(f"""UPDATE key_devices SET failed_count=%s,
                    locked_until=CASE WHEN %s THEN clock_timestamp()+interval '{LOCK_MINUTES} minutes' ELSE locked_until END
                    WHERE id=%s""", (failed, lock, device['id']))
    cur.execute('INSERT INTO key_device_attempts(device_id,ok) VALUES (%s,false)', (device['id'],))
    if failed >= BLOCK_AFTER:
        end_sessions(cur, 'device_id=%s', (device['id'],))
        audit(cur, device['actor_id'], 'blocked', device['id'])
        return problem(423, 'PIN_BLOCKED', 'Too many wrong PINs. Ask the owner to reset your PIN.',
                       'Demasiados PIN incorrectos. Pide al dueño que restablezca tu PIN.'), None
    if lock:
        return problem(429, 'PIN_LOCKED', f'Too many wrong PINs on this phone. Try again in {LOCK_MINUTES} minutes.',
                       f'Demasiados PIN incorrectos en este teléfono. Intenta en {LOCK_MINUTES} minutos.'), {'Retry-After': str(LOCK_MINUTES * 60)}
    return problem(401, 'PIN_INVALID', 'Wrong PIN.', 'PIN incorrecto.',
                   attempts_left=LOCK_AFTER - failed % LOCK_AFTER), None


def good_pin(cur, device):
    cur.execute('UPDATE key_devices SET failed_count=0, locked_until=NULL, last_used_at=clock_timestamp() WHERE id=%s',
                (device['id'],))
    cur.execute('INSERT INTO key_device_attempts(device_id,ok) VALUES (%s,true)', (device['id'],))


def finish(response, failure, headers, result):
    # Counted refusals commit before the HTTP error is raised.
    if failure:
        if headers:
            failure.headers = headers
        raise failure
    response.headers['Cache-Control'] = 'no-store'
    return result


def register_routes(app, api):
    def require_pins():
        if not key_sessions.pin_enabled(api):
            raise HTTPException(404, 'Not found')

    def session_actor(request: Request, _: bool = Depends(api.verify_api_key)):
        if getattr(request.state, 'key_kind', None) != 'session' or not request.state.actor:
            raise problem(403, 'SESSION_REQUIRED', 'Sign in first.', 'Inicia sesión primero.')
        return request.state.actor

    def owner_actor(actor=Depends(session_actor)):
        if actor.get('role') != 'owner':
            raise problem(403, 'OWNER_REQUIRED', 'Only the owner can manage staff phones.',
                          'Solo el dueño puede administrar los teléfonos del personal.')
        return actor

    @app.post('/auth/device/enroll', dependencies=[Depends(require_pins)])
    def enroll(body: Enroll, request: Request, response: Response, actor=Depends(session_actor)):
        source = digest(request.client.host if request.client else 'unknown')
        if getattr(request.state, 'session_device_id', None) is not None:
            raise problem(403, 'PERSONAL_KEY_REQUIRED', 'Set up a phone after signing in with your personal key.',
                          'Configura un teléfono después de iniciar sesión con tu clave personal.')
        failure = new_pin_problem(body.pin)
        if failure:
            raise failure
        label = body.label.strip()
        if not label:
            raise problem(422, 'LABEL_REQUIRED', 'Name this phone.', 'Ponle nombre a este teléfono.')
        failure, headers, result = None, None, None
        with api.get_transaction() as cur:
            # Re-checking the personal key counts against the same limits as signing in with it.
            cur.execute('SELECT pg_advisory_xact_lock(%s)', (LOCK_EXCHANGE,))
            cur.execute("""SELECT count(*) AS n FROM key_session_attempts
                           WHERE source_hash=%s AND NOT ok AND attempted_at>clock_timestamp()-interval '15 minutes'""", (source,))
            if cur.fetchone()['n'] >= 5:
                failure, headers = problem(429, 'SESSION_RATE_LIMITED', 'Too many sign-in attempts. Try again later.',
                                           'Demasiados intentos. Intenta más tarde.'), {'Retry-After': '900'}
            else:
                lock_person(cur, actor['id'])
                cur.execute('SELECT id,name,role,key_hash,active FROM actors WHERE id=%s FOR UPDATE', (actor['id'],))
                person = cur.fetchone()
                if not hmac.compare_digest(digest(body.actor_key.strip()), person['key_hash']):
                    cur.execute('INSERT INTO key_session_attempts(source_hash,ok) VALUES (%s,false)', (source,))
                    failure = problem(401, 'ACTOR_KEY_INVALID', 'Enter your own personal key to set up this phone.',
                                      'Escribe tu propia clave personal para configurar este teléfono.')
            if not failure:
                failure, result = _enroll(cur, body, person, label, request)
        return finish(response, failure, headers, result)

    def _enroll(cur, body, person, label, request):
        """Replace this phone if asked, check the limit, store the phone: (failure, result)."""
        if body.replace_device_secret and SECRET.fullmatch(body.replace_device_secret):
            cur.execute("""UPDATE key_devices SET revoked_at=clock_timestamp(), revoked_reason='replaced'
                           WHERE secret_hash=%s AND actor_id=%s AND revoked_at IS NULL RETURNING id""",
                        (digest(body.replace_device_secret), person['id']))
            for old in cur.fetchall():
                end_sessions(cur, 'device_id=%s', (old['id'],))
                audit(cur, person['id'], 'revoke', old['id'])
        cur.execute('SELECT count(*) AS n FROM key_devices WHERE actor_id=%s AND revoked_at IS NULL', (person['id'],))
        if cur.fetchone()['n'] >= MAX_DEVICES:
            return problem(409, 'DEVICE_LIMIT', f'{person["name"]} already has {MAX_DEVICES} phones set up. Ask the owner to remove one.',
                           f'{person["name"]} ya tiene {MAX_DEVICES} teléfonos. Pide al dueño que quite uno.'), None
        secret = secrets.token_urlsafe(32)
        hashed, salt, iterations = pin_columns(secret, body.pin)
        cur.execute("""INSERT INTO key_devices(actor_id,secret_hash,actor_key_hash,label,pin_hash,pin_salt,pin_iterations,pin_set_at)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,clock_timestamp()) RETURNING id,label""",
                    (person['id'], digest(secret), person['key_hash'], label, hashed, salt, iterations))
        device = cur.fetchone()
        audit(cur, person['id'], 'enroll', device['id'])
        # The personal-key session becomes this phone's 12-hour session.
        cur.execute("""UPDATE key_actor_sessions SET ended_at=clock_timestamp(), ended_reason='logout'
                       WHERE id=%s AND ended_at IS NULL""", (request.state.actor_session_id,))
        return None, {**open_session(cur, device, person), 'device_secret': secret}

    @app.post('/auth/pin', dependencies=[Depends(require_pins)])
    def pin_sign_in(body: PinSignIn, response: Response):
        failure, headers, result = None, None, None
        with api.get_transaction() as cur:
            device = find_device(cur, body.device_secret)
            if device is None:
                failure, headers = unknown_attempt(cur)
            else:
                failure, headers = gate(device) or (None, None)
                if not failure:
                    if pin_matches(device, body.device_secret, body.pin):
                        good_pin(cur, device)
                        result = open_session(cur, device, device | {'id': device['actor_id']})
                    else:
                        failure, headers = wrong_pin(cur, device)
        return finish(response, failure, headers, result)

    @app.post('/auth/pin/reset', dependencies=[Depends(require_pins)])
    def pin_reset(body: PinReset, response: Response):
        failure, headers, result = None, None, None
        with api.get_transaction() as cur:
            device = find_device(cur, body.device_secret)
            if device is None:
                failure, headers = unknown_attempt(cur)
            elif device['pin_hash'] or not (device['pin_reset_until'] and device['pin_reset_until'] > device['now']):
                failure = problem(409, 'PIN_RESET_NOT_OPEN', 'This phone has no open PIN reset. Ask the owner.',
                                  'Este teléfono no tiene un restablecimiento abierto. Pide al dueño.')
            else:
                failure = new_pin_problem(body.new_pin)
                if not failure:
                    hashed, salt, iterations = pin_columns(body.device_secret, body.new_pin)
                    cur.execute("""UPDATE key_devices SET pin_hash=%s,pin_salt=%s,pin_iterations=%s,pin_set_at=clock_timestamp(),
                                   pin_reset_until=NULL,failed_count=0,locked_until=NULL,last_used_at=clock_timestamp() WHERE id=%s""",
                                (hashed, salt, iterations, device['id']))
                    audit(cur, device['actor_id'], 'pin_reset_set', device['id'])
                    result = open_session(cur, device, device | {'id': device['actor_id']})
        return finish(response, failure, headers, result)

    @app.post('/auth/pin/change', dependencies=[Depends(require_pins)])
    def pin_change(body: PinChange, request: Request, response: Response, actor=Depends(session_actor)):
        failure = new_pin_problem(body.new_pin)
        if failure:
            raise failure
        headers, result = None, None
        with api.get_transaction() as cur:
            device = find_device(cur, body.device_secret)
            if device is None or device['actor_id'] != actor['id'] or device['id'] != getattr(request.state, 'session_device_id', None):
                failure = problem(403, 'DEVICE_SESSION_REQUIRED', 'Sign in with your PIN on this phone to change it.',
                                  'Inicia sesión con tu PIN en este teléfono para cambiarlo.')
            else:
                failure, headers = gate(device) or (None, None)
                if not failure:
                    if not pin_matches(device, body.device_secret, body.current_pin):
                        failure, headers = wrong_pin(cur, device)
                    else:
                        hashed, salt, iterations = pin_columns(body.device_secret, body.new_pin)
                        cur.execute("""UPDATE key_devices SET pin_hash=%s,pin_salt=%s,pin_iterations=%s,pin_set_at=clock_timestamp(),
                                       failed_count=0,locked_until=NULL WHERE id=%s""", (hashed, salt, iterations, device['id']))
                        audit(cur, actor['id'], 'pin_change', device['id'])
                        result = {'pin_changed': True}
        return finish(response, failure, headers, result)

    @app.post('/auth/device/forget', dependencies=[Depends(require_pins)])
    def forget(body: Forget, response: Response, actor=Depends(session_actor)):
        with api.get_transaction() as cur:
            device = find_device(cur, body.device_secret)
            if device and device['actor_id'] == actor['id']:
                cur.execute("UPDATE key_devices SET revoked_at=clock_timestamp(), revoked_reason='self', revoked_by=%s WHERE id=%s",
                            (actor['id'], device['id']))
                end_sessions(cur, 'device_id=%s', (device['id'],))
                audit(cur, actor['id'], 'revoke', device['id'])
        # The phone forgets its secret either way; nothing here says whether it was valid.
        return finish(response, None, None, {'forgotten': True})

    @app.get('/auth/devices', dependencies=[Depends(require_pins)])
    def list_devices(response: Response, _=Depends(owner_actor)):
        with api.get_transaction() as cur:
            cur.execute('SELECT id,name,role,active,last_used_at FROM actors ORDER BY id')
            people = [dict(r, devices=[]) for r in cur.fetchall()]
            cur.execute("""SELECT id,actor_id,label,enrolled_at,last_used_at,failed_count,locked_until,pin_reset_until,
                                  pin_hash IS NOT NULL AS has_pin, clock_timestamp() AS now
                           FROM key_devices WHERE revoked_at IS NULL ORDER BY enrolled_at""")
            by_id = {p['id']: p for p in people}
            for d in cur.fetchall():
                status = ('blocked' if d['failed_count'] >= BLOCK_AFTER
                          else 'locked' if d['locked_until'] and d['locked_until'] > d['now']
                          else 'reset_pending' if not d['has_pin'] else 'active')
                by_id[d['actor_id']]['devices'].append({k: d[k] for k in ('id', 'label', 'enrolled_at', 'last_used_at', 'pin_reset_until')} | {'status': status})
        return finish(response, None, None, {'people': people, 'reset_minutes': RESET_MINUTES})

    def owned_device(cur, owner, device_id):
        cur.execute("""SELECT d.actor_id, a.role FROM key_devices d JOIN actors a ON a.id=d.actor_id
                       WHERE d.id=%s AND d.revoked_at IS NULL""", (device_id,))
        target = cur.fetchone()
        # Your own phones (say, a lost one) yes; another owner's, no.
        if target and target['role'] == 'owner' and target['actor_id'] != owner['id']:
            raise problem(409, 'OWNER_PROTECTED', 'Owner accounts are not changed here.', 'Las cuentas de dueño no se cambian aquí.')
        if target:
            lock_person(cur, target['actor_id'])
        cur.execute('SELECT * FROM key_devices WHERE id=%s AND revoked_at IS NULL FOR UPDATE', (device_id,))
        device = cur.fetchone()
        if not device:
            raise problem(404, 'DEVICE_NOT_FOUND', 'That phone is not set up (or was already removed).',
                          'Ese teléfono no está configurado (o ya fue quitado).')
        return device

    @app.post('/auth/devices/{device_id}/revoke', dependencies=[Depends(require_pins)])
    def revoke_device(device_id: int, response: Response, owner=Depends(owner_actor)):
        with api.get_transaction() as cur:
            device = owned_device(cur, owner, device_id)
            cur.execute("UPDATE key_devices SET revoked_at=clock_timestamp(), revoked_reason='owner', revoked_by=%s WHERE id=%s",
                        (owner['id'], device['id']))
            end_sessions(cur, 'device_id=%s', (device['id'],))
            audit(cur, device['actor_id'], 'revoke', device['id'], owner['id'])
        return finish(response, None, None, {'revoked': True})

    @app.post('/auth/devices/{device_id}/reset-pin', dependencies=[Depends(require_pins)])
    def reset_device_pin(device_id: int, response: Response, owner=Depends(owner_actor)):
        with api.get_transaction() as cur:
            device = owned_device(cur, owner, device_id)
            cur.execute(f"""UPDATE key_devices SET pin_hash=NULL,pin_salt=NULL,pin_iterations=NULL,failed_count=0,locked_until=NULL,
                            pin_reset_until=clock_timestamp()+interval '{RESET_MINUTES} minutes' WHERE id=%s RETURNING pin_reset_until""",
                        (device['id'],))
            until = cur.fetchone()['pin_reset_until']
            end_sessions(cur, 'device_id=%s', (device['id'],))
            audit(cur, device['actor_id'], 'pin_reset', device['id'], owner['id'])
        return finish(response, None, None, {'reset': True, 'pin_reset_until': until})

    def other_person(cur, owner, actor_id):
        if actor_id == owner['id']:
            raise problem(409, 'NOT_YOURSELF', 'You cannot change your own access here.', 'No puedes cambiar tu propio acceso aquí.')
        lock_person(cur, actor_id)
        cur.execute('SELECT id,name,role,active FROM actors WHERE id=%s FOR UPDATE', (actor_id,))
        person = cur.fetchone()
        if not person:
            raise problem(404, 'PERSON_NOT_FOUND', 'No such person.', 'No existe esa persona.')
        if person['role'] == 'owner':
            raise problem(409, 'OWNER_PROTECTED', 'Owner accounts are not changed here.', 'Las cuentas de dueño no se cambian aquí.')
        return person

    def remove_all(cur, person, reason, by):
        cur.execute("""UPDATE key_devices SET revoked_at=clock_timestamp(), revoked_reason=%s, revoked_by=%s
                       WHERE actor_id=%s AND revoked_at IS NULL""", (reason, by, person['id']))
        end_sessions(cur, 'actor_id=%s', (person['id'],))

    @app.post('/auth/actors/{actor_id}/remove-access', dependencies=[Depends(require_pins)])
    def remove_access(actor_id: int, response: Response, owner=Depends(owner_actor)):
        with api.get_transaction() as cur:
            person = other_person(cur, owner, actor_id)
            remove_all(cur, person, 'access_removed', owner['id'])
            audit(cur, person['id'], 'remove_access', None, owner['id'])
        return finish(response, None, None, {'access_removed': True})

    @app.post('/auth/actors/{actor_id}/active', dependencies=[Depends(require_pins)])
    def set_active(actor_id: int, body: Active, response: Response, owner=Depends(owner_actor)):
        with api.get_transaction() as cur:
            person = other_person(cur, owner, actor_id)
            if person['active'] != body.active:
                cur.execute('UPDATE actors SET active=%s WHERE id=%s', (body.active, person['id']))
                if not body.active:
                    remove_all(cur, person, 'deactivated', owner['id'])
                audit(cur, person['id'], 'reactivate' if body.active else 'deactivate', None, owner['id'])
        return finish(response, None, None, {'active': body.active})
