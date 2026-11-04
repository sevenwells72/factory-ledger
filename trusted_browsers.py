"""Four-name entry plus retained legacy browser-credential compatibility.

Current dashboard: X-FL-Person with Michael/Luz/Arturo/Miriam enters directly
through select_name(), without a token, feature flag, approval or expiry.
Migration 091 choice rows supply stable actor IDs; historical browser/audit
records remain intact. The following contract applies ONLY to old flb_ clients:

Owner-approved browsers and one-tap person selection (migration 091).

Three separate things, kept separate on purpose:

* BROWSER (machine): a random `flb_` token the browser keeps in localStorage.
  The server stores only its sha256. It becomes usable only after the owner
  approves the pairing code it shows, from his own signed-in session (owner
  proof). It never expires; the owner revokes it. It is a ROUTINE capability:
  it never reaches owner-only decisions, admin routes or the legacy direct
  inventory writes, and it is never owner proof.
* PERSON (identification): the `X-FL-Person` header names one of the exactly
  four configured `person_choices`. It is checked on every request against the
  live table (no cache). Choosing a name proves nothing; `via='selection'`
  travels with the identity so `permissions.role_of` caps a selected owner.
* OWNER PROOF: unchanged — the owner's personal key, key session or phone PIN.

Off unless PERSON_SELECT_ENABLED=1; with the flag off every `flb_` token is
an unknown key (403) and these routes return 404. With the flag on but 091
missing, selection is unavailable (503) and nothing else changes.
"""
import hashlib
import hmac
import os
import re
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, StrictStr

PREFIX = 'flb_'
TOKEN = re.compile(r'flb_[A-Za-z0-9_-]{43}')          # 'flb_' + token_urlsafe(32)
PERSON = re.compile(r'[1-9][0-9]{0,9}')
CODE_ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789'      # no 0/O, 1/I/L
CODE_LENGTH = 8
PAIRING_MINUTES = 10
# The exact people offered, in display order (owner decision 2026-11-03).
# person_choices must match this list EXACTLY — names, routine roles and order —
# or selection is unavailable. Display names are UI only; ledger snapshots keep
# actors.name (production owner = id 1 'Blubber', shown as Michael).
EXPECTED_CHOICES = (('Michael', 'Owner', 'owner'), ('Luz', 'Office', 'office'),
                    ('Arturo', 'Floor Lead', 'floor'), ('Miriam', 'Office', 'office'))
CHOICE_COUNT = len(EXPECTED_CHOICES)
REQUEST_LIMIT = 30          # new pairing requests per 10 minutes, all browsers together
APPROVE_FAILURE_LIMIT = 10  # wrong codes per 10 minutes, all owners together
SEEN_THROTTLE = "interval '10 minutes'"
LOCK_ATTEMPTS = 770190
PERSON_HEADER = 'X-FL-Person'
# Owner-proof routes: a key session of the owner only (key_sessions.ROUTES).
OWNER_ROUTES = frozenset({('GET', '/auth/browsers'), ('POST', '/auth/browsers/approve'),
                          ('POST', '/auth/browsers/{browser_id}/revoke')})

_MIGRATED = set()


def enabled():
    return os.getenv('PERSON_SELECT_ENABLED', '0') == '1'


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def problem(status, code, en, es, **extra):
    return HTTPException(status, {'error_code': code, 'message': en, 'message_es': es, **extra})


def ready(api):
    """Flag on AND migration 091 present (checked once per database)."""
    if not enabled():
        return False
    source = (api.get_db_connection, os.getenv('DATABASE_URL'))
    if source in _MIGRATED:
        return True
    with api.get_transaction() as cur:
        cur.execute("SELECT to_regclass('public.trusted_browsers') IS NOT NULL AS ready")
        ok = cur.fetchone()['ready']
    if ok:
        _MIGRATED.add(source)
    return ok


def unavailable():
    return problem(503, 'PERSON_SELECT_UNAVAILABLE', 'Name selection is not available right now. Nothing was recorded.',
                   'La selección de nombre no está disponible ahora. No se registró nada.')


def misconfigured():
    return problem(503, 'PERSON_CHOICES_MISCONFIGURED',
                   'The list of people is not set up correctly. Ask the owner. Nothing was recorded.',
                   'La lista de personas no está bien configurada. Pregunta al dueño. No se registró nada.')


def choices_valid(rows):
    """CONFIGURATION check (Codex R3/R5): exactly four slots — Michael, Luz, Arturo,
    Miriam in that order — each a distinct actor whose role matches its slot. A
    wrong, missing, duplicate or reordered configuration fails every browser
    closed (never substitutes). Whether a person is active is NOT checked here."""
    if len(rows) != CHOICE_COUNT or len({r['id'] for r in rows}) != CHOICE_COUNT:
        return False
    for row, (display, role_label, role) in zip(rows, EXPECTED_CHOICES):
        if row['display_name'] != display or row['role_label'] != role_label or row['role'] != role:
            return False
    return True


def eligible(row):
    """PER-PERSON check: the slot is switched on and the actor is active. An
    ineligible person is shown disabled and refused; the other three continue."""
    return bool(row['choice_active'] and row['actor_active'])


def choices(cur):
    """All four configured slots in display order (each with 'eligible'), or 503
    PERSON_CHOICES_MISCONFIGURED when the configuration itself is wrong."""
    cur.execute("""SELECT c.actor_id AS id, c.display_name, c.role_label, c.sort_order, a.name, a.role,
                          c.active AS choice_active, a.active AS actor_active
                   FROM person_choices c JOIN actors a ON a.id=c.actor_id
                   ORDER BY c.sort_order""")
    rows = [dict(r) for r in cur.fetchall()]
    if not choices_valid(rows):
        raise misconfigured()
    for row in rows:
        row['eligible'] = eligible(row)
    return rows


def availability(api):
    """Health for /auth/session/config, separate from the mode flag. Never raises."""
    if not enabled():
        return {'person_select_available': False, 'person_select_problem': None}
    try:
        if not ready(api):
            return {'person_select_available': False, 'person_select_problem': 'PERSON_SELECT_UNAVAILABLE'}
        with api.get_transaction() as cur:
            choices(cur)
    except HTTPException as exc:
        return {'person_select_available': False, 'person_select_problem': exc.detail['error_code']}
    except Exception:
        return {'person_select_available': False, 'person_select_problem': 'PERSON_SELECT_UNAVAILABLE'}
    return {'person_select_available': True, 'person_select_problem': None}


def public_choice(row):
    return {'id': row['id'], 'display_name': row['display_name'], 'role_label': row['role_label'],
            'available': row['eligible']}


def browser_row(cur, token, *, lock=False):
    if not isinstance(token, str) or not TOKEN.fullmatch(token):
        return None
    cur.execute('SELECT * FROM trusted_browsers WHERE secret_hash=%s' + (' FOR UPDATE' if lock else ''), (digest(token),))
    return cur.fetchone()


def refused(row):
    if row is None:
        return problem(401, 'BROWSER_NOT_APPROVED', 'This browser is not approved. Ask Michael to approve it.',
                       'Este navegador no está aprobado. Pide a Michael que lo apruebe.')
    if row['status'] == 'revoked':
        return problem(401, 'BROWSER_REVOKED', 'This browser was removed by the owner. Ask Michael to approve it again.',
                       'El dueño quitó este navegador. Pide a Michael que lo apruebe otra vez.')
    return problem(401, 'BROWSER_NOT_APPROVED', 'This browser is waiting for approval. Nothing was recorded.',
                   'Este navegador espera aprobación. No se registró nada.')


def select_name(api, name, request):
    """Owner-authorized name-only entry. No secret, approval, session or expiry.

    The existing choice row supplies only the stable actor foreign key. Its
    active flag and the actor's credential/role settings do not gate this path.
    The tapped label is the new write's operator snapshot; history is untouched.
    """
    fixed = {display: role for display, _label, role in EXPECTED_CHOICES}
    if name not in fixed:
        raise problem(403, 'PERSON_NOT_SELECTABLE', 'Choose one of the four names.',
                      'Elige uno de los cuatro nombres.')
    with api.get_transaction() as cur:
        cur.execute('SELECT actor_id FROM person_choices WHERE display_name=%s ORDER BY sort_order', (name,))
        rows = cur.fetchall()
    if len(rows) != 1:
        raise unavailable()
    actor = {'id': rows[0]['actor_id'], 'name': name, 'role': fixed[name],
             'via': 'selection', 'trusted_browser_id': None}
    request.state.actor = actor
    request.state.key_kind = 'selection'
    request.state.identity_via = 'selection'
    request.state.trusted_browser_id = None
    return actor


def authenticate(api, token, request):
    """Resolve an `flb_` token (+ optional person) onto request.state. Returns the actor or None.

    Raises 403 'Invalid API key' while the flag is off (an unknown key, as before)."""
    if not enabled():
        raise HTTPException(status_code=403, detail='Invalid API key')
    if not ready(api):
        raise unavailable()
    raw_person = request.headers.get(PERSON_HEADER)
    with api.get_transaction() as cur:
        row = browser_row(cur, token)
        if row is None or row['status'] != 'approved':
            raise refused(row)
        if request.headers.get('X-FL-Background') != '1':
            cur.execute(f"""UPDATE trusted_browsers SET last_seen_at=clock_timestamp()
                            WHERE id=%s AND (last_seen_at IS NULL OR last_seen_at < clock_timestamp()-{SEEN_THROTTLE})""",
                        (row['id'],))
        actor = None
        if raw_person is not None:
            if not PERSON.fullmatch(raw_person.strip()):
                raise problem(403, 'PERSON_NOT_SELECTABLE', 'Choose your name again.', 'Elige tu nombre otra vez.')
            people = {r['id']: r for r in choices(cur)}
            person = people.get(int(raw_person.strip()))
            if person is not None and not person['eligible']:
                raise problem(403, 'PERSON_NOT_AVAILABLE',
                              f"{person['display_name']} cannot record right now (switched off by the owner). Nothing was recorded.",
                              f"{person['display_name']} no puede registrar ahora (desactivado por el dueño). No se registró nada.",
                              person_id=person['id'])
            if person is None:
                raise problem(403, 'PERSON_NOT_SELECTABLE', 'That name is not on this list any more. Choose your name again.',
                              'Ese nombre ya no está en la lista. Elige tu nombre otra vez.')
            # The marker and provenance live IN the actor dict, so request_actor()
            # and every raw-actor helper sees them, not only request.state.
            actor = {'id': person['id'], 'name': person['name'], 'role': person['role'],
                     'via': 'selection', 'trusted_browser_id': row['id']}
    request.state.trusted_browser_id = row['id']
    request.state.identity_via = 'selection' if actor else None
    request.state.actor = actor
    request.state.key_kind = 'selection' if actor else 'browser'
    return actor


class Approve(BaseModel):
    pairing_code: StrictStr = Field(..., min_length=1, max_length=20)
    label: StrictStr = Field(..., min_length=1, max_length=60)

    class Config:
        extra = 'forbid'


def normal_code(value):
    return re.sub(r'[\s-]', '', value or '').upper()


def attempts(cur, kind, limit):
    cur.execute('SELECT pg_advisory_xact_lock(%s)', (LOCK_ATTEMPTS,))
    cur.execute("""SELECT count(*) AS n FROM trusted_browser_attempts
                   WHERE kind=%s AND attempted_at > clock_timestamp()-interval '10 minutes'"""
                + (" AND NOT ok" if kind == 'approve' else ''), (kind,))
    return cur.fetchone()['n'] >= limit


def no_store(response):
    response.headers['Cache-Control'] = 'no-store'


def register_routes(app, api):
    def require_ready(request: Request):
        if request.headers.get(PERSON_HEADER) == 'Michael':
            return
        if not enabled():
            raise HTTPException(404, 'Not found')
        if not ready(api):
            raise unavailable()

    def owner_proof(request: Request, _: bool = Depends(api.verify_api_key)):
        # Only the owner's own key session (personal key or phone PIN). A browser
        # token or a tapped name never reaches here (key_sessions.ROUTES is session-only).
        actor = getattr(request.state, 'actor', None)
        if api.permissions.name_only(api._actor_identity(request)) and actor['name'] == 'Michael':
            return actor
        if getattr(request.state, 'key_kind', None) != 'session' or not actor or actor.get('role') != 'owner':
            raise problem(403, 'OWNER_PROOF_REQUIRED', 'Only Michael, signed in with his own key or PIN, can manage browsers.',
                          'Solo Michael, con su propia clave o PIN, puede administrar navegadores.')
        return actor

    @app.post('/auth/browsers/request', dependencies=[Depends(require_ready)], status_code=201)
    def request_approval(response: Response):
        """Unauthenticated: hands out a PENDING token that can do nothing until approved."""
        no_store(response)
        failure, result = None, None
        with api.get_transaction() as cur:
            if attempts(cur, 'request', REQUEST_LIMIT):
                failure = problem(429, 'BROWSER_REQUEST_RATE_LIMITED', 'Too many approval requests. Try again in a few minutes.',
                                  'Demasiadas solicitudes. Intenta en unos minutos.')
            else:
                cur.execute("""UPDATE trusted_browsers SET status='expired', pairing_code_hash=NULL
                               WHERE status='pending' AND pairing_expires_at <= clock_timestamp()""")
                token = PREFIX + secrets.token_urlsafe(32)
                code = ''.join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
                cur.execute(f"""INSERT INTO trusted_browsers(secret_hash,status,pairing_code_hash,pairing_expires_at)
                                VALUES (%s,'pending',%s,clock_timestamp()+interval '{PAIRING_MINUTES} minutes')
                                RETURNING pairing_expires_at""", (digest(token), digest(code)))
                expires = cur.fetchone()['pairing_expires_at']
                cur.execute("INSERT INTO trusted_browser_attempts(kind,ok) VALUES ('request',true)")
                result = {'browser_token': token, 'pairing_code': code[:4] + '-' + code[4:],
                          'pairing_expires_at': expires, 'pairing_minutes': PAIRING_MINUTES}
        if failure:
            failure.headers = {'Retry-After': '600'}
            raise failure
        return result

    @app.get('/auth/browsers/status', dependencies=[Depends(require_ready)])
    def status(request: Request, response: Response):
        """For the browser itself: pending / approved (+ the people) / revoked / expired."""
        no_store(response)
        token = request.headers.get('X-API-Key') or ''
        with api.get_transaction() as cur:
            row = browser_row(cur, token)
            if row is None:
                raise refused(None)
            state = row['status']
            if state == 'pending' and row['pairing_expires_at'] is not None:
                cur.execute('SELECT clock_timestamp() >= %s AS gone', (row['pairing_expires_at'],))
                if cur.fetchone()['gone']:
                    state = 'expired'
            result = {'status': state, 'label': row['label'],
                      'pairing_expires_at': row['pairing_expires_at'] if state == 'pending' else None}
            if state == 'approved':
                result['people'] = [public_choice(r) for r in choices(cur)]
        return result

    @app.get('/auth/browsers', dependencies=[Depends(require_ready)])
    def list_browsers(response: Response, _=Depends(owner_proof)):
        no_store(response)
        with api.get_transaction() as cur:
            cur.execute("""SELECT b.id,b.label,b.status,b.requested_at,b.approved_at,b.revoked_at,b.last_seen_at,
                                  a.name AS approved_by, b.pairing_expires_at
                           FROM trusted_browsers b LEFT JOIN actors a ON a.id=b.approved_by_actor_id
                           WHERE b.status IN ('approved','revoked')
                              OR (b.status='pending' AND b.pairing_expires_at > clock_timestamp())
                           ORDER BY b.status, b.id DESC""")
            rows = [dict(r) for r in cur.fetchall()]
            try:
                people, problem_code = [public_choice(r) for r in choices(cur)], None
            except HTTPException as exc:
                people, problem_code = [], exc.detail['error_code']
        return {'browsers': rows, 'people': people, 'people_problem': problem_code,
                'pairing_minutes': PAIRING_MINUTES}

    @app.post('/auth/browsers/approve', dependencies=[Depends(require_ready)])
    def approve(body: Approve, response: Response, owner=Depends(owner_proof)):
        no_store(response)
        code = normal_code(body.pairing_code)
        label = body.label.strip()
        if not label:
            raise problem(422, 'LABEL_REQUIRED', 'Name this browser (for example "Floor tablet").',
                          'Ponle nombre a este navegador (por ejemplo "Tableta del piso").')
        failure, result = None, None
        with api.get_transaction() as cur:
            if attempts(cur, 'approve', APPROVE_FAILURE_LIMIT):
                failure = problem(429, 'BROWSER_APPROVE_RATE_LIMITED', 'Too many wrong codes. Wait 10 minutes.',
                                  'Demasiados códigos incorrectos. Espera 10 minutos.')
            else:
                row = None
                if len(code) == CODE_LENGTH and all(c in CODE_ALPHABET for c in code):
                    cur.execute("""SELECT * FROM trusted_browsers WHERE status='pending' AND pairing_code_hash=%s
                                   AND pairing_expires_at > clock_timestamp() FOR UPDATE""", (digest(code),))
                    row = cur.fetchone()
                cur.execute('INSERT INTO trusted_browser_attempts(kind,ok) VALUES (%s,%s)', ('approve', row is not None))
                if row is None:
                    failure = problem(404, 'PAIRING_CODE_INVALID',
                                      'No browser is waiting with that code (it may have expired). Check the code on the browser.',
                                      'Ningún navegador espera con ese código (puede haber vencido). Revisa el código en el navegador.')
                else:
                    cur.execute("""UPDATE trusted_browsers SET status='approved', pairing_code_hash=NULL, label=%s,
                                   approved_at=clock_timestamp(), approved_by_actor_id=%s WHERE id=%s
                                   RETURNING id,label,approved_at""", (label, owner['id'], row['id']))
                    result = dict(cur.fetchone())
                    cur.execute("INSERT INTO trusted_browser_audit(browser_id,action,by_actor_id) VALUES (%s,'approve',%s)",
                                (row['id'], owner['id']))
        if failure:
            raise failure
        return {'approved': True, 'browser': result}

    @app.post('/auth/browsers/{browser_id}/revoke', dependencies=[Depends(require_ready)])
    def revoke(browser_id: int, response: Response, owner=Depends(owner_proof)):
        no_store(response)
        with api.get_transaction() as cur:
            cur.execute("""UPDATE trusted_browsers SET status='revoked', pairing_code_hash=NULL,
                           revoked_at=clock_timestamp(), revoked_by_actor_id=%s, revoked_reason='owner'
                           WHERE id=%s AND status IN ('approved','pending') RETURNING id""", (owner['id'], browser_id))
            row = cur.fetchone()
            if not row:
                raise problem(404, 'BROWSER_NOT_FOUND', 'That browser is not approved (or was already removed).',
                              'Ese navegador no está aprobado (o ya fue quitado).')
            cur.execute("INSERT INTO trusted_browser_audit(browser_id,action,by_actor_id) VALUES (%s,'revoke',%s)",
                        (browser_id, owner['id']))
        return {'revoked': True}
