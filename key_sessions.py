"""Pilot key sessions, independent of dormant PIN login. No raw key persistence.

Every session checks the live actor/key hash, bypassing the legacy actor cache.
"""
import hashlib
import os
import secrets
from datetime import timedelta
from fastapi import Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, StrictStr

PREFIX = 'fls_'
IDLE_MINUTES = 180
ROUTES = frozenset({('POST', '/auth/session/logout')})


def enabled():
    return os.getenv('KEY_SESSIONS_ENABLED', '0') == '1'


def require_enabled():
    if not enabled():
        raise HTTPException(404, 'Not found')


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def expired():
    raise HTTPException(401, {'error_code': 'SESSION_EXPIRED',
                             'message': 'Sign in again with your personal key.'})


class Exchange(BaseModel):
    actor_key: StrictStr = Field(..., min_length=1, max_length=512)
    class Config:
        extra = 'forbid'


def resolve(api, token, request):
    """Serialize sliding activity with logout/expiry; return a fresh actor."""
    if not enabled() or not token.startswith(PREFIX):
        return None
    actor = None
    with api.get_transaction() as cur:
        cur.execute('SELECT actor_id FROM key_actor_sessions WHERE token_hash=%s', (digest(token),))
        owner = cur.fetchone()
        if owner:
            # A revoked key may end several sessions. Serialize by actor first
            # to avoid two session requests locking each other's rows.
            cur.execute('SELECT pg_advisory_xact_lock(770181,%s)', (owner['actor_id'],))
        cur.execute("""SELECT s.*, a.name, a.role, a.active,
                              a.key_hash AS current_key_hash
                       FROM key_actor_sessions s JOIN actors a ON a.id=s.actor_id
                       WHERE s.token_hash=%s FOR UPDATE OF s FOR SHARE OF a""", (digest(token),))
        row = cur.fetchone()
        cur.execute('SELECT clock_timestamp() AS now')
        now = cur.fetchone()['now']
        if row and row['ended_at'] is None:
            reason = ('revoked' if not row['active'] or row['actor_key_hash'] != row['current_key_hash']
                      else 'idle' if now >= row['expires_at'] else None)
            if reason:
                # End all sessions issued against the revoked key on detection.
                cur.execute("""UPDATE key_actor_sessions SET ended_at=%s, ended_reason=%s
                               WHERE ended_at IS NULL AND """ +
                            ('actor_id=%s AND actor_key_hash=%s' if reason == 'revoked' else 'id=%s'),
                            (now, reason, row['actor_id'], row['actor_key_hash']) if reason == 'revoked'
                            else (now, reason, row['id']))
            else:
                expiry = now + timedelta(minutes=IDLE_MINUTES)
                cur.execute('UPDATE key_actor_sessions SET last_seen_at=%s,expires_at=%s WHERE id=%s',
                            (now, expiry, row['id']))
                request.state.actor_session_id = row['id']
                request.state.session_expires_at = expiry
                actor = {'id': row['actor_id'], 'name': row['name'], 'role': row['role']}
    if actor is None:
        expired()
    return actor


def register_routes(app, api):
    @app.get('/auth/session/config')
    def config(response: Response):
        response.headers['Cache-Control'] = 'no-store'
        return {'key_sessions_enabled': enabled(), 'idle_minutes': IDLE_MINUTES}

    @app.post('/auth/session', dependencies=[Depends(require_enabled)])
    def exchange(body: Exchange, request: Request, response: Response):
        # Do not trust arbitrary X-Forwarded-For as a fresh limiter source.
        source = digest(request.client.host if request.client else 'unknown')
        failure, result = None, None
        with api.get_transaction() as cur:
            cur.execute('SELECT pg_advisory_xact_lock(770180)')
            cur.execute("""SELECT
                count(*) FILTER (WHERE source_hash=%s AND attempted_at>clock_timestamp()-interval '1 minute') AS recent,
                count(*) FILTER (WHERE source_hash=%s AND NOT ok AND attempted_at>clock_timestamp()-interval '15 minutes') AS failures,
                count(*) FILTER (WHERE NOT ok) AS global_failures
                FROM key_session_attempts WHERE attempted_at>clock_timestamp()-interval '1 hour'""", (source, source))
            limits = cur.fetchone()
            if limits['recent'] >= 10 or limits['failures'] >= 5 or limits['global_failures'] >= 50:
                failure = HTTPException(429, {'error_code': 'SESSION_RATE_LIMITED',
                    'message': 'Too many sign-in attempts. Try again later.'}, headers={'Retry-After': '900'})
            else:
                key = body.actor_key.strip()
                # Shared keys never open a session: master, dashboard, and the
                # G1 read-only GPT key (refused even if an actor's hash matches it).
                shared = any(value and secrets.compare_digest(digest(key), digest(value))
                             for value in (api.API_KEY, api.DASHBOARD_API_KEY,
                                           getattr(api, 'READONLY_API_KEY', '')))
                cur.execute('SELECT id,name,role,key_hash FROM actors WHERE key_hash=%s AND active FOR SHARE', (digest(key),))
                actor = cur.fetchone()
                ok = bool(actor and not shared)
                cur.execute('INSERT INTO key_session_attempts(source_hash,ok) VALUES (%s,%s)', (source, ok))
                if not ok:
                    failure = HTTPException(401, {'error_code': 'ACTOR_KEY_INVALID',
                        'message': 'Use your active personal actor key.'})
                else:
                    token = PREFIX + secrets.token_urlsafe(32)
                    cur.execute("""INSERT INTO key_actor_sessions(token_hash,actor_id,actor_key_hash,expires_at)
                        VALUES (%s,%s,%s,clock_timestamp()+interval '180 minutes') RETURNING expires_at""",
                        (digest(token), actor['id'], actor['key_hash']))
                    result = {'session_token': token, 'key_kind': 'session',
                              'actor': {k: actor[k] for k in ('id','name','role')},
                              'expires_at': cur.fetchone()['expires_at'], 'idle_minutes': IDLE_MINUTES}
        # Failed attempts must commit before raising the HTTP refusal.
        if failure:
            raise failure
        response.headers['Cache-Control'] = 'no-store'
        return result

    @app.post('/auth/session/logout', dependencies=[Depends(require_enabled)])
    def logout(request: Request, response: Response, _: bool = Depends(api.verify_api_key)):
        if request.state.key_kind != 'session':
            raise HTTPException(403, {'error_code': 'SESSION_REQUIRED', 'message': 'Use a session to sign out.'})
        with api.get_transaction() as cur:
            cur.execute("""UPDATE key_actor_sessions SET ended_at=clock_timestamp(),ended_reason='logout'
                           WHERE id=%s AND ended_at IS NULL""", (request.state.actor_session_id,))
        response.headers['Cache-Control'] = 'no-store'
        return {'signed_out': True}
