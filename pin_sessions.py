"""A11: PIN-only person lookup, durable throttles and opaque ten-minute sessions.

Secrets never appear in audit rows or error responses. PIN verification commits
its security evidence independently of any business transaction it protects.
"""
from datetime import timedelta
import hashlib
import hmac
import logging
import os
import re
import secrets

from fastapi import Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, SecretStr

IDLE = timedelta(minutes=10)
SOURCE_WINDOW = timedelta(minutes=15)
GLOBAL_WINDOW = timedelta(hours=1)
SOURCE_LIMIT = 5
GLOBAL_LIMIT = 50
SESSION_PREFIX = 'fls_'
AUTH_ROUTES = frozenset({('GET', '/auth/session'), ('DELETE', '/auth/session'),
                         ('GET', '/actors/pins'), ('POST', '/actors/{actor_id}/pin')})
log = logging.getLogger(__name__)


def fail(status, code, message, headers=None):
    raise HTTPException(status, {'error_code': code, 'message': message}, headers=headers)


def pepper():
    value = os.environ.get('PIN_PEPPER', '')
    if len(value) < 32:
        fail(503, 'PIN_AUTH_UNAVAILABLE', 'PIN sign-in is not configured.')
    return value.encode()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def pin_hash(pin):
    return hmac.new(pepper(), b'fl-pin-v1:' + pin.encode(), hashlib.sha256).hexdigest()


def valid_pin(pin):
    if not isinstance(pin, str) or re.fullmatch(r'[0-9]{4}', pin) is None:
        return False
    steps = [(int(pin[i+1]) - int(pin[i])) % 10 for i in range(3)]
    # Also reject alternating repeated pairs; wraparound sequences count.
    return len(set(pin)) > 1 and steps not in ([1]*3, [9]*3) and pin[:2] != pin[2:]


def sources(request):
    device = request.cookies.get('fl_device', '')
    if re.fullmatch(r'[a-zA-Z0-9_-]{32,64}', device) is None:
        device = secrets.token_urlsafe(32)
    # Only the ASGI peer: the deployment must configure uvicorn's trusted
    # proxy list. Never trust an arbitrary client X-Forwarded-For header here.
    ip = request.client.host if request.client else 'unknown'
    def source_hash(kind, value):
        return kind + ':' + hmac.new(pepper(), (kind + ':' + value).encode(), hashlib.sha256).hexdigest()
    return device, source_hash('ip', ip), source_hash('device', device)


def verify_pin(api, request, pin, *, purpose, actor_id=None, owner=False):
    """Return (actor, error, device). Every attempt shares source/global limits.

    A DB advisory lock makes check+count atomic across workers/restarts. A
    blocked attempt is recorded but cannot keep extending an existing lock.
    The lock is tested BEFORE querying the actor/PIN, including correct PINs.
    """
    device, ip_key, device_key = sources(request)
    result, error, alert = None, None, False
    with api.get_transaction() as cur:
        # This fast path ONLY denies and logs. Misses take the serialized
        # path below, which re-checks all locks before looking up any PIN.
        cur.execute('''WITH gate AS (
                         SELECT max(locked_until) AS until FROM pin_rate_limits
                         WHERE source=ANY(%s) AND locked_until>clock_timestamp())
                       INSERT INTO pin_attempts(source,device_hash,purpose,ok,blocked)
                       SELECT %s,%s,%s,false,true FROM gate WHERE until IS NOT NULL
                       RETURNING (SELECT until FROM gate) AS until,clock_timestamp() AS now''',
                    (['global', ip_key, device_key], ip_key, device_key, purpose))
        locked = cur.fetchone()
        if locked:
            retry = max(1, int((locked['until']-locked['now']).total_seconds())+1)
            return None, HTTPException(401, {'error_code': 'PIN_INVALID',
                'message': 'PIN not accepted. Try again or contact Michael.'},
                headers={'Retry-After': str(retry)}), device
        cur.execute('SELECT pg_advisory_xact_lock(724110)')
        keys = ['global', ip_key, device_key]
        cur.execute('INSERT INTO pin_rate_limits(source) SELECT unnest(%s::text[]) ON CONFLICT DO NOTHING', (keys,))
        cur.execute('SELECT *,clock_timestamp() AS now FROM pin_rate_limits WHERE source=ANY(%s) FOR UPDATE', (keys,))
        buckets = {row['source']: row for row in cur.fetchall()}
        now = buckets['global']['now']
        locked = [r['locked_until'] for r in buckets.values() if r['locked_until'] and r['locked_until'] > now]
        blocked = bool(locked)
        if not blocked:
            candidate_hash = pin_hash(pin) if isinstance(pin, str) and re.fullmatch(r'[0-9]{4}', pin) else ''
            cur.execute('SELECT id,name,role,pin_hash FROM actors WHERE pin_hash=%s AND active', (candidate_hash,))
            candidate = cur.fetchone()
            if candidate and (actor_id is None or candidate['id'] == actor_id) and (not owner or candidate['role'] == 'owner'):
                result = dict(candidate)
            else:
                for key in keys:
                    window, limit = (GLOBAL_WINDOW, GLOBAL_LIMIT) if key == 'global' else (SOURCE_WINDOW, SOURCE_LIMIT)
                    recent = [t for t in buckets[key]['failed_at'] if t > now-window] + [now]
                    until = now+window if len(recent) >= limit else None
                    cur.execute('UPDATE pin_rate_limits SET failed_at=%s,locked_until=%s WHERE source=%s', (recent, until, key))
                    if until:
                        locked.append(until)
                        if key == 'global':
                            alert = True
        cur.execute('''INSERT INTO pin_attempts(source,device_hash,purpose,ok,blocked,actor_id)
                       VALUES (%s,%s,%s,%s,%s,%s)''',
                    (ip_key, device_key, purpose, result is not None, blocked, result['id'] if result else None))
        if result is None:
            retry = max(1, int((max(locked)-now).total_seconds())+1) if locked else None
            error = HTTPException(401, {'error_code': 'PIN_INVALID', 'message': 'PIN not accepted. Try again or contact Michael.'},
                                  headers={'Retry-After': str(retry)} if retry else None)
    if alert:
        # Durable event is also visible in GET /actors/pins. Monitoring can
        # route this fixed, credential-free event to the owner's alert channel.
        log.error('PIN_GLOBAL_LOCKOUT: PIN authentication locked for one hour')
    return result, error, device


def public_actor(actor):
    return {k: actor[k] for k in ('id', 'name', 'role')}


def issue(api, actor, method, device_label=None):
    token = SESSION_PREFIX + secrets.token_urlsafe(32)
    with api.get_transaction() as cur:
        cur.execute('SELECT id,name,role,active,pin_hash FROM actors WHERE id=%s FOR SHARE', (actor['id'],))
        current = cur.fetchone()
        if not current or not current['active'] or (method == 'pin' and current['pin_hash'] != actor['pin_hash']):
            fail(401, 'PIN_INVALID', 'Sign-in no longer valid. Try again.')
        if method == 'actor_key' and current['role'] not in ('owner', 'office'):
            fail(403, 'ROLE_NOT_ALLOWED', 'Personal-key sign-in is available to office and owner only.')
        cur.execute('''INSERT INTO actor_sessions(token_hash,actor_id,auth_method,device_label,expires_at)
                       VALUES (%s,%s,%s,%s,clock_timestamp()+interval '10 minutes') RETURNING expires_at''',
                    (digest(token), actor['id'], method, device_label))
        expiry = cur.fetchone()['expires_at']
    return {'session_token': token, 'actor': public_actor(current), 'key_kind': 'session', 'expires_at': expiry.isoformat()}


def resolve(api, request, token):
    error, actor = None, None
    with api.get_transaction() as cur:
        # Lock the actor before the session, matching PIN reset lock order.
        cur.execute('''SELECT a.id,a.name,a.role,a.active FROM actors a JOIN actor_sessions s ON s.actor_id=a.id
                       WHERE s.token_hash=%s FOR SHARE OF a''', (digest(token),))
        actor = cur.fetchone()
        cur.execute('SELECT *,clock_timestamp() AS now FROM actor_sessions WHERE token_hash=%s FOR UPDATE', (digest(token),))
        row = cur.fetchone()
        if not row or not actor or row['ended_at'] or not actor['active']:
            error = 'SESSION_EXPIRED'
        elif row['expires_at'] <= row['now']:
            cur.execute("UPDATE actor_sessions SET ended_at=clock_timestamp(),ended_reason='idle' WHERE id=%s", (row['id'],))
            error = 'SESSION_EXPIRED'
        else:
            # Browser polling is not human activity. The common browser client
            # marks passive reads; it also erases its credential on idle.
            if request.headers.get('X-FL-Background') != '1':
                cur.execute("""UPDATE actor_sessions SET last_seen_at=clock_timestamp(),
                               expires_at=clock_timestamp()+interval '10 minutes' WHERE id=%s RETURNING expires_at""", (row['id'],))
                row['expires_at'] = cur.fetchone()['expires_at']
            request.state.session = dict(row)
    if error:
        fail(401, error, 'Session ended. Enter your PIN to continue.')
    return public_actor(actor)


def require_owner_pin(api, request, *, purpose, allow_other_owner=False):
    """A PIN proves ONE request only. Never issues a session or an elevated token.

    A12 can use allow_other_owner=True for a floor-owned batch attestation;
    the returned owner id must be stored on that exact ticket in its transaction.
    Ordinary owner actions must match the currently authenticated owner.
    """
    actor = api.request_actor(request)
    if not actor or (not allow_other_owner and actor['role'] != 'owner'):
        fail(403, 'ROLE_NOT_ALLOWED', 'This action requires the owner.')
    with api.get_transaction() as cur:
        cur.execute('SELECT active,role FROM actors WHERE id=%s', (actor['id'],))
        current = cur.fetchone()
        if not current or not current['active'] or (not allow_other_owner and current['role'] != 'owner'):
            fail(403, 'ROLE_NOT_ALLOWED', 'This action requires an active owner.')
    # Cache only inside this Request, so nested validation of the same action
    # does not double-count. There is no cross-request step-up grace period.
    verified = getattr(request.state, 'pin_owner', None)
    if verified:
        return verified
    value = request.headers.get('X-FL-Owner-PIN')
    if not value:
        fail(403, 'OWNER_PIN_REQUIRED', 'Michael must re-enter his PIN for this action.')
    owner, error, _ = verify_pin(api, request, value, purpose=purpose,
                                 actor_id=None if allow_other_owner else actor['id'], owner=True)
    if error:
        raise error
    request.state.pin_owner = public_actor(owner)
    return request.state.pin_owner


class Login(BaseModel):
    pin: SecretStr
    device_label: str | None = None

    class Config:
        extra = 'forbid'


class KeyLogin(BaseModel):
    actor_key: SecretStr


class SetPin(BaseModel):
    pin: SecretStr
    current_pin: SecretStr | None = None


def register_routes(app, api):
    @app.post('/auth/session')
    def login(body: Login, request: Request):
        actor, error, device = verify_pin(api, request, body.pin.get_secret_value(), purpose='login')
        if error:
            response = JSONResponse({'detail': error.detail}, status_code=error.status_code, headers=error.headers)
        else:
            response = JSONResponse(issue(api, actor, 'pin', (body.device_label or '')[:100] or None))
        response.headers['Cache-Control'] = 'no-store'
        # A hint for throttling, never an identity credential. IP limit still
        # applies when third-party cookie policy prevents cookie persistence.
        response.set_cookie('fl_device', device, max_age=31536000, secure=True, httponly=True, samesite='none')
        return response

    @app.post('/auth/session/key')
    def key_login(body: KeyLogin):
        pepper()  # don't offer sessions before PIN support is configured
        actor = api._resolve_actor(body.actor_key.get_secret_value())
        if not actor or actor['role'] not in ('owner', 'office'):
            fail(401, 'SIGN_IN_INVALID', 'Personal sign-in not accepted.')
        return JSONResponse(issue(api, actor, 'actor_key'), headers={'Cache-Control': 'no-store'})

    @app.get('/auth/session')
    def session(request: Request, _: bool = Depends(api.verify_api_key)):
        row = getattr(request.state, 'session', None)
        if not row:
            fail(403, 'SESSION_REQUIRED', 'Sign in with a PIN first.')
        return JSONResponse({'actor': public_actor(api.request_actor(request)), 'key_kind': 'session',
                             'expires_at': row['expires_at'].isoformat(),
                             'permissions': api.permissions.permissions_for(api._actor_identity(request))},
                            headers={'Cache-Control': 'no-store'})

    @app.delete('/auth/session')
    def logout(request: Request, _: bool = Depends(api.verify_api_key)):
        row = getattr(request.state, 'session', None)
        if not row:
            fail(403, 'SESSION_REQUIRED', 'A session is required.')
        with api.get_transaction() as cur:
            cur.execute("UPDATE actor_sessions SET ended_at=clock_timestamp(),ended_reason='logout' WHERE id=%s", (row['id'],))
        return {'signed_out': True}

    @app.get('/actors/pins')
    def actors(request: Request, _: bool = Depends(api.verify_api_key)):
        actor = api.request_actor(request)
        if not actor or actor['role'] != 'owner' or not getattr(request.state, 'session', None):
            fail(403, 'ROLE_NOT_ALLOWED', 'PIN administration requires an owner session.')
        with api.get_transaction() as cur:
            cur.execute('SELECT id,name,role,active,pin_set_at,pin_hash IS NOT NULL AS pin_set FROM actors ORDER BY name')
            people = [dict(row) for row in cur.fetchall()]
            cur.execute("SELECT locked_until FROM pin_rate_limits WHERE source='global'")
            lock = cur.fetchone()
            cur.execute("SELECT count(*) AS failures FROM pin_attempts WHERE NOT ok AND at > clock_timestamp()-interval '1 hour'")
            failures = cur.fetchone()['failures']
        return {'actors': people, 'global_locked_until': lock['locked_until'] if lock else None, 'failures_last_hour': failures}

    @app.post('/actors/{actor_id}/pin')
    def set_pin(actor_id: int, body: SetPin, request: Request, _: bool = Depends(api.verify_api_key)):
        actor = api.request_actor(request)
        session = getattr(request.state, 'session', None)
        if not actor or not session:
            fail(403, 'SESSION_REQUIRED', 'PIN changes require a personal session.')
        if actor['id'] != actor_id and actor['role'] != 'owner':
            fail(403, 'ROLE_NOT_ALLOWED', 'Only Michael can reset another person’s PIN.')
        value = body.pin.get_secret_value()
        if not valid_pin(value):
            fail(422, 'PIN_WEAK', 'Choose four digits without repeated digits, repeated pairs or a counting sequence.')
        with api.get_transaction() as cur:
            cur.execute('SELECT pin_hash FROM actors WHERE id=%s', (actor['id'],))
            previous = cur.fetchone()['pin_hash']
        bootstrap = (actor['role'] == 'owner' and actor_id == actor['id'] and previous is None and session['auth_method'] == 'actor_key')
        if not bootstrap:
            if actor['role'] == 'owner':
                require_owner_pin(api, request, purpose='pin_management')
            else:
                _, error, _ = verify_pin(api, request, body.current_pin.get_secret_value() if body.current_pin else '',
                                         purpose='pin_change', actor_id=actor['id'])
                if error:
                    raise error
        hashed = pin_hash(value)
        # Serialize PIN uniqueness changes. Re-check the setter after step-up
        # so a simultaneous reset cannot authorize a stale credential.
        with api.get_transaction() as cur:
            cur.execute('SELECT pg_advisory_xact_lock(724111)')
            cur.execute('SELECT id,active,pin_hash FROM actors WHERE id=ANY(%s) ORDER BY id FOR UPDATE', ([actor['id'], actor_id],))
            rows = {r['id']: r for r in cur.fetchall()}
            if actor_id not in rows or not rows[actor_id]['active']:
                fail(404, 'ACTOR_NOT_FOUND', 'Active person not found.')
            if not rows[actor['id']]['active'] or rows[actor['id']]['pin_hash'] != previous:
                fail(401, 'SESSION_EXPIRED', 'Identity changed. Sign in again.')
            cur.execute('SELECT id FROM actors WHERE pin_hash=%s AND id<>%s', (hashed, actor_id))
            if cur.fetchone():
                fail(409, 'PIN_IN_USE', 'That PIN is unavailable. Choose another.')
            cur.execute('''UPDATE actors SET pin_hash=%s,pin_set_at=clock_timestamp(),pin_failed_attempts=0,pin_locked_until=NULL
                           WHERE id=%s''', (hashed, actor_id))
            cur.execute("""UPDATE actor_sessions SET ended_at=clock_timestamp(),ended_reason='pin_reset'
                           WHERE actor_id=%s AND ended_at IS NULL""", (actor_id,))
            cur.execute('INSERT INTO pin_management_audit(actor_id,changed_by_actor_id) VALUES (%s,%s)', (actor_id, actor['id']))
        return {'pin_set': True, 'sessions_revoked': True, 'sign_in_again': actor_id == actor['id']}
