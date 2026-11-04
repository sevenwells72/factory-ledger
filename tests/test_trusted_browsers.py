"""Owner-approved browsers + one-tap person selection (migration 091): real HTTP/DB.

Every test runs on the LOCAL factory_ledger_test database inside the suite's
rolled-back transaction (tests/conftest.py refuses production hosts). The
assertions check observable failure modes — what is refused, what is
recorded, which credential can do what — not the implementation's shape.
"""
from datetime import timedelta
from pathlib import Path
import re

import pytest
from fastapi import HTTPException

import main
import permissions
import kosher_tiers
import resolution_receipts
import trusted_browsers
import write_tickets
from scripts import seed_person_choices
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_write_tickets import payload, headers, seed  # noqa: F401

pytestmark = pytest.mark.db
ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / 'migrations/091_trusted_browsers.sql'
DOWN = ROOT / 'migrations/down/091_trusted_browsers_down.sql'


@pytest.fixture(autouse=True)
def selection_on(monkeypatch, db_cursor):
    monkeypatch.setenv('KEY_SESSIONS_ENABLED', '1')
    monkeypatch.setenv('PERSON_SELECT_ENABLED', '1')
    trusted_browsers._MIGRATED.clear()
    yield
    trusted_browsers._MIGRATED.clear()


@pytest.fixture
def migrated(db_cursor):
    db_cursor.execute(MIGRATION.read_text())


@pytest.fixture
def people(db_cursor, actors, migrated):
    """The four configured people: Michael=owner, Luz=office, Arturo=floor, Miriam=office2."""
    db_cursor.execute("INSERT INTO actors (name, role, key_hash, active) VALUES (%s,'office',%s,true) RETURNING id",
                      ('ACT office2 ' + actors['owner']['name'][-8:], 'f' * 64))
    miriam = db_cursor.fetchone()['id']
    rows = [(actors['owner']['id'], 'Michael', 'Owner', 1), (actors['office']['id'], 'Luz', 'Office', 2),
            (actors['floor']['id'], 'Arturo', 'Floor Lead', 3), (miriam, 'Miriam', 'Office', 4)]
    for row in rows:
        db_cursor.execute('INSERT INTO person_choices(actor_id,display_name,role_label,sort_order) VALUES (%s,%s,%s,%s)', row)
    main._reset_actor_cache()
    return {'michael': actors['owner']['id'], 'luz': actors['office']['id'],
            'arturo': actors['floor']['id'], 'miriam': miriam}


def owner_session(client, actors, role='owner'):
    r = client.post('/auth/session', json={'actor_key': actors[role]['key']})
    assert r.status_code == 200, r.text
    return r.json()['session_token']


def pending(client):
    r = client.post('/auth/browsers/request')
    assert r.status_code == 201, r.text
    return r.json()


def approved(client, actors, label='Floor tablet'):
    req = pending(client)
    r = client.post('/auth/browsers/approve', headers=headers(owner_session(client, actors)),
                    json={'pairing_code': req['pairing_code'], 'label': label})
    assert r.status_code == 200, r.text
    return req['browser_token']


def as_person(token, person_id=None, background=False):
    h = {'X-API-Key': token}
    if person_id is not None:
        h['X-FL-Person'] = str(person_id)
    if background:
        h['X-FL-Background'] = '1'
    return h


def code(response):
    body = response.json()
    detail = body.get('detail')
    return detail.get('error_code') if isinstance(detail, dict) else detail


# ---------------------------------------------------------------- mode vs availability

def test_flag_off_disables_selection_and_keeps_the_public_key_read_only(client, monkeypatch, people, actors):
    token = approved(client, actors)
    monkeypatch.setenv('PERSON_SELECT_ENABLED', '0')
    cfg = client.get('/auth/session/config').json()
    assert cfg['person_select_enabled'] is False and cfg['person_select_available'] is False
    # A browser token is just an unknown key again, and the pairing routes vanish.
    assert client.get('/auth/whoami', headers=as_person(token, people['arturo'])).status_code == 403
    assert client.post('/auth/browsers/request').status_code == 404
    # DCC-001: turning selection off does NOT reopen public dashboard-key writes.
    r = client.post('/receive/prepare', json={}, headers={'X-API-Key': main.DASHBOARD_API_KEY})
    assert r.status_code == 403 and code(r) == 'DASHBOARD_KEY_READ_ONLY'


def test_flag_on_without_migration_reports_mode_on_but_unavailable_and_blocks(client, actors):
    cfg = client.get('/auth/session/config').json()
    # MODE stays on (the UI must show "unavailable", never fall back to key/PIN prompts).
    assert cfg['person_select_enabled'] is True
    assert cfg['person_select_available'] is False and cfg['person_select_problem'] == 'PERSON_SELECT_UNAVAILABLE'
    r = client.post('/auth/browsers/request')
    assert r.status_code == 503 and code(r) == 'PERSON_SELECT_UNAVAILABLE'
    # And the public dashboard key is still closed for writes (fail closed, not open).
    w = client.post('/receive/prepare', json={}, headers={'X-API-Key': main.DASHBOARD_API_KEY})
    assert w.status_code == 403 and code(w) == 'PERSON_REQUIRED'


def test_flag_on_with_no_people_configured_is_unavailable(client, actors, migrated):
    cfg = client.get('/auth/session/config').json()
    assert cfg['person_select_enabled'] is True and cfg['person_select_available'] is False
    assert cfg['person_select_problem'] == 'PERSON_CHOICES_MISCONFIGURED'


@pytest.mark.parametrize('mistake', ['three_people', 'wrong_name', 'wrong_role', 'wrong_order', 'duplicate_actor',
                                     'wrong_role_label'])
def test_any_configuration_mistake_fails_every_browser_closed(client, db_cursor, actors, people, mistake):
    token = approved(client, actors)
    assert client.get('/auth/whoami', headers=as_person(token, people['arturo'])).status_code == 200
    if mistake == 'three_people':
        db_cursor.execute('DELETE FROM person_choices WHERE actor_id=%s', (people['miriam'],))
    elif mistake == 'wrong_name':
        db_cursor.execute("UPDATE person_choices SET display_name='Liz' WHERE actor_id=%s", (people['luz'],))
    elif mistake == 'wrong_role':
        # Arturo's slot pointing at an office actor would hand floor work to the wrong role.
        db_cursor.execute('DELETE FROM person_choices WHERE actor_id=%s', (people['arturo'],))
        db_cursor.execute('DELETE FROM person_choices WHERE actor_id=%s', (people['miriam'],))
        db_cursor.execute("INSERT INTO person_choices VALUES (%s,'Arturo','Floor Lead',3,true)", (people['miriam'],))
        db_cursor.execute("INSERT INTO person_choices VALUES (%s,'Miriam','Office',4,true)", (people['arturo'],))
    elif mistake == 'wrong_order':
        db_cursor.execute('UPDATE person_choices SET sort_order=sort_order+10')
        db_cursor.execute('UPDATE person_choices SET sort_order=1 WHERE actor_id=%s', (people['luz'],))
        db_cursor.execute('UPDATE person_choices SET sort_order=2 WHERE actor_id=%s', (people['michael'],))
    elif mistake == 'duplicate_actor':
        db_cursor.execute('DELETE FROM person_choices WHERE actor_id=%s', (people['miriam'],))
        db_cursor.execute("INSERT INTO actors(name,role,key_hash,active) VALUES ('ACT dup','office',%s,true) RETURNING id", ('e' * 64,))
        db_cursor.execute("INSERT INTO person_choices VALUES (%s,'Miriam','Office',4,true)", (db_cursor.fetchone()['id'],))
        db_cursor.execute("UPDATE person_choices SET display_name='Luz' WHERE sort_order=4")
    elif mistake == 'wrong_role_label':
        db_cursor.execute("UPDATE person_choices SET role_label='Owner' WHERE actor_id=%s", (people['arturo'],))
    cfg = client.get('/auth/session/config').json()
    assert cfg['person_select_available'] is False and cfg['person_select_problem'] == 'PERSON_CHOICES_MISCONFIGURED'
    for person in people.values():
        r = client.post('/receive/prepare', json={}, headers=as_person(token, person))
        assert r.status_code == 503 and code(r) == 'PERSON_CHOICES_MISCONFIGURED', (mistake, r.text)
    status = client.get('/auth/browsers/status', headers=as_person(token))
    assert status.status_code == 503


def test_one_deactivated_person_is_refused_and_the_other_three_continue(client, db_cursor, actors, people, payload):
    token = approved(client, actors)
    db_cursor.execute('UPDATE actors SET active=false WHERE id=%s', (people['luz'],))
    status = client.get('/auth/browsers/status', headers=as_person(token)).json()
    assert [(p['display_name'], p['available']) for p in status['people']] == [
        ('Michael', True), ('Luz', False), ('Arturo', True), ('Miriam', True)]
    assert client.get('/auth/session/config').json()['person_select_available'] is True
    refused = client.post('/receive/prepare', json=payload, headers=as_person(token, people['luz']))
    assert refused.status_code == 403 and code(refused) == 'PERSON_NOT_AVAILABLE'
    # Nobody is substituted for Luz, and the others still record.
    ok = client.post('/receive/prepare', json=payload, headers=as_person(token, people['miriam']))
    assert ok.status_code == 200 and ok.json()['actor']['id'] == people['miriam']
    # Switching off just the slot (not the actor) behaves the same.
    db_cursor.execute('UPDATE person_choices SET active=false WHERE actor_id=%s', (people['arturo'],))
    r = client.get('/auth/whoami', headers=as_person(token, people['arturo']))
    assert r.status_code == 403 and code(r) == 'PERSON_NOT_AVAILABLE'
    assert client.get('/auth/whoami', headers=as_person(token, people['michael'])).status_code == 200


# ---------------------------------------------------------------- pairing and owner proof

def test_pending_browser_can_do_nothing_and_only_stores_hashes(client, db_cursor, people):
    req = pending(client)
    token, pairing = req['browser_token'], req['pairing_code']
    assert re.fullmatch(r'flb_[A-Za-z0-9_-]{43}', token) and re.fullmatch(r'[A-Z2-9]{4}-[A-Z2-9]{4}', pairing)
    db_cursor.execute('SELECT * FROM trusted_browsers WHERE secret_hash=%s', (trusted_browsers.digest(token),))
    row = db_cursor.fetchone()
    assert row['status'] == 'pending' and token not in str(row) and pairing.replace('-', '') not in str(row)
    for h in (as_person(token), as_person(token, people['arturo'])):
        r = client.get('/auth/whoami', headers=h)
        assert r.status_code == 401 and code(r) == 'BROWSER_NOT_APPROVED'
    assert client.get('/auth/browsers/status', headers=as_person(token)).json()['status'] == 'pending'


@pytest.mark.parametrize('credential', ['personal_owner_key', 'office_session', 'master', 'dashboard',
                                        'selected_michael', 'no_key'])
def test_only_an_owner_key_session_can_approve(client, actors, people, credential):
    req = pending(client)
    other = approved(client, actors, 'Other')
    h = {'personal_owner_key': headers(actors['owner']['key']),
         'office_session': headers(owner_session(client, actors, 'office')),
         'master': headers(main.API_KEY), 'dashboard': {'X-API-Key': main.DASHBOARD_API_KEY},
         'selected_michael': as_person(other, people['michael']), 'no_key': {}}[credential]
    r = client.post('/auth/browsers/approve', headers=h, json={'pairing_code': req['pairing_code'], 'label': 'X'})
    assert r.status_code in (401, 403), (credential, r.text)
    still = client.get('/auth/browsers/status', headers=as_person(req['browser_token'])).json()
    assert still['status'] == 'pending'


def test_wrong_codes_are_counted_and_rate_limited(client, actors, people, monkeypatch):
    monkeypatch.setattr(trusted_browsers, 'APPROVE_FAILURE_LIMIT', 3)
    owner = headers(owner_session(client, actors))
    for _ in range(3):
        r = client.post('/auth/browsers/approve', headers=owner, json={'pairing_code': 'ZZZZ-ZZZZ', 'label': 'X'})
        assert r.status_code == 404 and code(r) == 'PAIRING_CODE_INVALID'
    req = pending(client)
    r = client.post('/auth/browsers/approve', headers=owner, json={'pairing_code': req['pairing_code'], 'label': 'X'})
    assert r.status_code == 429 and code(r) == 'BROWSER_APPROVE_RATE_LIMITED'


def test_request_rate_limit(client, monkeypatch, migrated):
    monkeypatch.setattr(trusted_browsers, 'REQUEST_LIMIT', 2)
    pending(client)
    pending(client)
    r = client.post('/auth/browsers/request')
    assert r.status_code == 429 and r.headers['retry-after'] == '600'


def test_expired_code_cannot_be_approved(client, db_cursor, actors, people):
    req = pending(client)
    db_cursor.execute("UPDATE trusted_browsers SET pairing_expires_at=clock_timestamp()-interval '1 second' "
                      "WHERE secret_hash=%s", (trusted_browsers.digest(req['browser_token']),))
    assert client.get('/auth/browsers/status', headers=as_person(req['browser_token'])).json()['status'] == 'expired'
    r = client.post('/auth/browsers/approve', headers=headers(owner_session(client, actors)),
                    json={'pairing_code': req['pairing_code'], 'label': 'X'})
    assert r.status_code == 404


def test_approval_and_revocation_are_audited_and_revocation_is_immediate(client, db_cursor, actors, people):
    token = approved(client, actors)
    owner = headers(owner_session(client, actors))
    listing = client.get('/auth/browsers', headers=owner).json()
    browser = next(b for b in listing['browsers'] if b['status'] == 'approved')
    assert browser['label'] == 'Floor tablet' and browser['approved_by'] == actors['owner']['name']
    assert [p['display_name'] for p in listing['people']] == ['Michael', 'Luz', 'Arturo', 'Miriam']
    assert client.get('/auth/whoami', headers=as_person(token, people['arturo'])).status_code == 200
    assert client.post(f"/auth/browsers/{browser['id']}/revoke", headers=owner).status_code == 200
    r = client.get('/auth/whoami', headers=as_person(token, people['arturo']))
    assert r.status_code == 401 and code(r) == 'BROWSER_REVOKED'
    db_cursor.execute('SELECT action, by_actor_id FROM trusted_browser_audit WHERE browser_id=%s ORDER BY id', (browser['id'],))
    assert [tuple(r.values()) for r in db_cursor.fetchall()] == [('approve', actors['owner']['id']), ('revoke', actors['owner']['id'])]
    with pytest.raises(Exception):
        db_cursor.execute('SAVEPOINT a; UPDATE trusted_browser_audit SET action=action')
    db_cursor.execute('ROLLBACK TO SAVEPOINT a')


def test_no_origin_header_is_trusted(client, people):
    r = client.post('/receive/prepare', json={}, headers={'Origin': 'https://cns-factory-ledger.netlify.app',
                                                          'X-FL-Person': str(people['arturo'])})
    assert r.status_code == 401


# ---------------------------------------------------------------- scope and identity

def test_without_a_person_reads_work_and_every_write_needs_a_name(client, actors, people):
    token = approved(client, actors)
    assert client.get('/inventory/current', headers=as_person(token)).status_code == 200
    r = client.post('/receive/prepare', json={}, headers=as_person(token))
    assert r.status_code == 403 and code(r) == 'PERSON_REQUIRED'
    # Garbage or unknown person ids never resolve to anybody.
    for bad in ['0', '-1', 'abc', '1 OR 1=1', str(actors['retired']['id']), '99999999']:
        r = client.get('/auth/whoami', headers={'X-API-Key': token, 'X-FL-Person': bad})
        assert r.status_code == 403 and code(r) in ('PERSON_NOT_SELECTABLE', 'PERSON_NOT_AVAILABLE'), bad


def test_selection_never_reaches_legacy_direct_writes_admin_or_staff_routes(client, db_cursor, actors, people, monkeypatch):
    monkeypatch.setenv('PIN_SIGNIN_ENABLED', '1')     # staff-phone admin routes answer (instead of 404)
    db_cursor.execute((ROOT / 'migrations/089_device_pins.sql').read_text())
    token = approved(client, actors)
    h = as_person(token, people['michael'])
    for method, path in [('POST', '/make'), ('POST', '/receive'), ('POST', '/adjust'), ('POST', '/ship'),
                         ('POST', '/customers'), ('POST', '/receive/commit'), ('POST', '/void/1'),
                         ('GET', '/admin/lots/duplicates'), ('GET', '/auth/devices'), ('GET', '/auth/browsers'),
                         ('POST', '/auth/session/logout')]:
        r = client.request(method, path, headers=h, json={})
        assert r.status_code == 403, (method, path, r.status_code, r.text[:200])


def test_selected_michael_is_identified_but_capped(client, actors, people):
    token = approved(client, actors)
    who = client.get('/auth/whoami', headers=as_person(token, people['michael'])).json()
    assert who['key_kind'] == 'selection' and who['actor']['id'] == people['michael']
    assert who['actor']['name'] == actors['owner']['name']          # the historical snapshot, never the display alias
    for action in permissions.OWNER_PROOF_ACTIONS:
        assert who['permissions'][action] is False, action
    assert who['permissions']['make'] is True and who['permissions']['receive'] is True
    proof = client.get('/auth/whoami', headers=headers(owner_session(client, actors))).json()
    assert all(proof['permissions'][a] for a in permissions.OWNER_PROOF_ACTIONS)
    r = client.post('/exceptions/999999/approve', headers=as_person(token, people['michael']), json={})
    assert r.status_code == 403 and code(r) == 'OWNER_PROOF_REQUIRED'


def test_selected_owner_allowlist_is_explicit_and_new_owner_actions_are_denied(monkeypatch):
    reviewed = {'receive', 'make', 'pack', 'adjust', 'found', 'void', 'rename_lot', 'update_supplier_lot', 'move_lot',
                'ship_order', 'ship_standalone', 'ship_bulk', 'create_customer', 'create_order', 'add_order_lines',
                'update_order_line', 'cancel_order_line', 'update_order_header', 'update_order_status',
                'mark_order_ready', 'cancel_order', 'close_order', 'create_expected_receipt',
                'update_expected_receipt', 'cancel_expected_receipt', 'manage_customers', 'manage_aliases',
                'list_exceptions', 'resolve_exception'}
    assert set(permissions.SELECTED_OWNER_ACTIONS) == reviewed
    for action, roles in permissions.ROLE_PERMISSIONS.items():
        assert (permissions.SELECTED_OWNER in roles) == (action in reviewed), action
    # A future owner-only action is denied to a selected owner by default.
    monkeypatch.setitem(permissions.ROLE_PERMISSIONS, 'future_owner_only', frozenset({permissions.OWNER}))
    selected = {'id': 1, 'name': 'Blubber', 'role': 'owner', 'key_kind': 'actor', 'via': 'selection'}
    with pytest.raises(HTTPException) as exc:
        permissions.require('future_owner_only', selected)
    assert exc.value.detail['error_code'] == 'OWNER_PROOF_REQUIRED'
    assert permissions.require('future_owner_only', {**selected, 'via': None}) == permissions.OWNER


def test_assurance_survives_role_reloads_and_raw_actor_helpers(db_cursor, actors):
    selected = {'id': actors['owner']['id'], 'name': actors['owner']['name'], 'role': 'owner',
                'key_kind': 'actor', 'via': 'selection', 'trusted_browser_id': 7}
    reloaded = resolution_receipts.current_actor(db_cursor, selected)
    assert permissions.role_of(reloaded) == permissions.SELECTED_OWNER
    assert kosher_tiers.owner(db_cursor, selected) is None
    assert kosher_tiers.owner(db_cursor, {**selected, 'via': None})['id'] == actors['owner']['id']
    with pytest.raises(HTTPException) as exc:
        kosher_tiers.attest(db_cursor, selected)
    assert exc.value.detail['error_code'] == 'KOSHER_ATTESTATION_REQUIRED'
    assert write_tickets.assurance(selected) == 'selection' and write_tickets.assurance({'identity_via': None}) == 'credential'


def test_request_actor_itself_carries_the_selection_marker(client, actors, people, monkeypatch):
    token = approved(client, actors)
    seen = {}
    real = main._actor_identity

    def spy(request):
        seen['actor'] = dict(main.request_actor(request))
        seen['state_via'] = getattr(request.state, 'identity_via', None)
        return real(request)
    monkeypatch.setattr(main, '_actor_identity', spy)
    assert client.get('/auth/whoami', headers=as_person(token, people['arturo'])).status_code == 200
    assert seen['actor']['via'] == 'selection' and seen['actor']['trusted_browser_id'] is not None
    assert seen['state_via'] == 'selection'


# ---------------------------------------------------------------- attribution

def test_selected_person_records_with_their_own_name_and_browser(client, db_cursor, actors, people, payload):
    token = approved(client, actors)
    h = as_person(token, people['arturo'])
    prepared = client.post('/receive/prepare', json=payload, headers=h).json()
    assert prepared['actor']['via'] == 'selection'
    r = client.post('/tickets/' + prepared['ticket'] + '/commit', json={'payload_hash': prepared['payload_hash']}, headers=h)
    assert r.status_code == 200, r.text
    db_cursor.execute('SELECT * FROM write_tickets WHERE id=%s', (prepared['ticket_id'],))
    ticket = db_cursor.fetchone()
    db_cursor.execute('SELECT id FROM trusted_browsers WHERE secret_hash=%s', (trusted_browsers.digest(token),))
    browser_id = db_cursor.fetchone()['id']
    assert (ticket['actor_id'], ticket['operator_id'], ticket['key_kind']) == (people['arturo'], actors['floor']['name'], 'actor')
    assert ticket['identity_via'] == 'selection' and ticket['trusted_browser_id'] == browser_id
    db_cursor.execute('SELECT operator_id, entered_by_actor_id FROM transactions WHERE id=%s', (r.json()['transaction_id'],))
    assert tuple(db_cursor.fetchone().values()) == (actors['floor']['name'], people['arturo'])


def test_metadata_writes_record_the_browser(client, db_cursor, actors, people):
    token = approved(client, actors)
    r = client.post('/dashboard/api/notes', headers=as_person(token, people['luz']), json={'category': 'note', 'title': 'selection note'})
    assert r.status_code in (200, 201), r.text
    db_cursor.execute("SELECT actor_id, operator_id, trusted_browser_id FROM actor_write_audit "
                      "WHERE route='/dashboard/api/notes' ORDER BY id DESC LIMIT 1")
    row = db_cursor.fetchone()
    assert row['actor_id'] == people['luz'] and row['operator_id'] == actors['office']['name'] and row['trusted_browser_id']


def test_change_user_cannot_commit_the_previous_persons_ticket(client, actors, people, payload):
    token = approved(client, actors)
    prepared = client.post('/receive/prepare', json=payload, headers=as_person(token, people['arturo'])).json()
    r = client.post('/tickets/' + prepared['ticket'] + '/commit', json={'payload_hash': prepared['payload_hash']},
                    headers=as_person(token, people['luz']))
    assert r.status_code == 403 and code(r) == 'TICKET_WRONG_USER'


def test_selected_michael_cannot_commit_an_owner_credential_draft_and_vice_versa(client, actors, people, payload):
    token = approved(client, actors)
    session = headers(owner_session(client, actors))
    by_proof = client.post('/receive/prepare', json=payload, headers=session).json()
    r = client.post('/tickets/' + by_proof['ticket'] + '/commit', json={'payload_hash': by_proof['payload_hash']},
                    headers=as_person(token, people['michael']))
    assert r.status_code == 403 and code(r) == 'TICKET_WRONG_USER'
    other = dict(payload, lot_code=payload['lot_code'] + 'B', bol_reference=payload['bol_reference'] + 'B')
    by_selection = client.post('/receive/prepare', json=other, headers=as_person(token, people['michael'])).json()
    r = client.post('/tickets/' + by_selection['ticket'] + '/commit',
                    json={'payload_hash': by_selection['payload_hash']}, headers=session)
    assert r.status_code == 403 and code(r) == 'TICKET_WRONG_USER'


def test_selected_michael_cannot_backdate_over_14_days_but_his_session_can(client, actors, people, payload):
    token = approved(client, actors)
    old = dict(payload, occurred_at=(main.get_plant_now() - timedelta(days=20)).isoformat(), backfill=True)
    r = client.post('/receive/prepare', json=old, headers=as_person(token, people['michael']))
    assert r.status_code == 403 and code(r) == 'BACKFILL_OWNER_ONLY' and r.json()['detail']['owner_proof_required'] is True
    assert client.post('/receive/prepare', json=old, headers=headers(owner_session(client, actors))).status_code == 200


def test_selected_michael_late_entry_opens_an_exception_like_staff(client, db_cursor, actors, people, payload):
    token = approved(client, actors)
    late = dict(payload, occurred_at=(main.get_plant_now() - timedelta(days=4)).isoformat())
    prepared = client.post('/receive/prepare', json=late, headers=as_person(token, people['michael'])).json()
    assert any(w['code'] == 'LATE_ENTRY' for w in prepared['warnings'])


def test_selected_michael_voids_only_his_own_same_day_posts(client, db_cursor, actors, people, payload):
    token = approved(client, actors)
    arturo = as_person(token, people['arturo'])
    prepared = client.post('/receive/prepare', json=payload, headers=arturo).json()
    txn = client.post('/tickets/' + prepared['ticket'] + '/commit', json={'payload_hash': prepared['payload_hash']},
                      headers=arturo).json()['transaction_id']
    r = client.post(f'/void/{txn}/prepare', headers=as_person(token, people['michael']),
                    json={'reason': 'data_entry_error', 'client_source': 'dashboard'})
    assert r.status_code == 403 and code(r) == 'VOID_OWNER_REQUIRED'
    proof = client.post(f'/void/{txn}/prepare', headers=headers(owner_session(client, actors)),
                        json={'reason': 'data_entry_error', 'client_source': 'dashboard'})
    assert proof.status_code == 200, proof.text


def test_shift_confirmation_records_selection_assurance(client, db_cursor, actors, people):
    token = approved(client, actors)
    h = as_person(token, people['arturo'])
    summary = client.get('/reports/shift-summary', headers=h).json()
    r = client.post('/reports/shift-summary/confirm', headers=h,
                    json={'date': main.get_plant_now().date().isoformat(), 'summary_hash': summary['summary_hash'], 'outcome': 'match',
                          'notes': '', 'receipts_seen': 0})
    assert r.status_code == 201, r.text
    db_cursor.execute('SELECT actor_id, identity_via, trusted_browser_id FROM shift_confirmations ORDER BY id DESC LIMIT 1')
    row = db_cursor.fetchone()
    assert row['actor_id'] == people['arturo'] and row['identity_via'] == 'selection' and row['trusted_browser_id']


def test_nested_internal_calls_keep_the_selection_scope_and_cap(client, actors, people):
    """The Assistant relays to FL routes with the verified identity of the outer request."""
    token = approved(client, actors)
    captured = {}
    real = main._authorize_api_key

    def spy(provided_key, request, invalid_status=403):
        result = real(provided_key, request, invalid_status)
        captured.setdefault('identity', request.scope.get('_fl_verified_identity'))
        return result
    import unittest.mock as mock
    with mock.patch.object(main, '_authorize_api_key', spy):
        assert client.get('/auth/whoami', headers=as_person(token, people['michael'])).status_code == 200
    verified = captured['identity']
    assert verified.kind == 'selection' and verified.actor['via'] == 'selection'

    def nested(method, path):
        from starlette.requests import Request
        route = next(r for r in main.app.routes if getattr(r, 'path', None) == path and method in r.methods)
        req = Request({'type': 'http', 'method': method, 'path': path, 'headers': [], 'query_string': b'',
                       'route': route, '_fl_verified_identity': verified})
        return main._authorize_api_key('', req), req
    ok, req = nested('POST', '/make/prepare')
    assert ok and main._actor_identity(req)['via'] == 'selection'
    for method, path in [('POST', '/make'), ('GET', '/admin/lots/duplicates'), ('GET', '/auth/browsers')]:
        with pytest.raises(HTTPException):
            nested(method, path)
    with pytest.raises(HTTPException) as exc:
        nested('POST', '/exceptions/{exception_id}/approve')
    assert exc.value.detail['error_code'] == 'OWNER_PROOF_REQUIRED'


# ---------------------------------------------------------------- public dashboard key closure

def test_selector_mode_closes_every_public_dashboard_key_write(client, people):
    dash = {'X-API-Key': main.DASHBOARD_API_KEY}
    writes = sorted(k for k in main.DASHBOARD_KEY_ALLOWLIST if k[0] != 'GET' and k not in main.DASHBOARD_KEY_SELECTOR_READS)
    assert ('POST', '/receive/commit') in writes and ('POST', '/sales/orders/{order_id}/ship/commit') in writes
    for method, path in writes:
        url = re.sub(r'\{[^}]+\}', '1', path)
        r = client.request(method, url, headers=dash, json={})
        assert r.status_code == 403 and code(r) == 'PERSON_REQUIRED', (method, path, r.status_code)
    assert client.get('/inventory/current', headers=dash).status_code == 200


def test_resolver_exemption_writes_nothing(client, db_cursor, people):
    def writes():
        db_cursor.execute('SELECT coalesce(sum(n_tup_ins+n_tup_upd+n_tup_del),0) AS n FROM pg_stat_xact_user_tables')
        return db_cursor.fetchone()['n']
    before = writes()
    r = client.post('/resolve', headers={'X-API-Key': main.DASHBOARD_API_KEY},
                    json={'query': 'Classic', 'kind': 'product', 'context': 'make'})
    assert r.status_code in (200, 422), r.text
    assert writes() == before


def test_credentialed_machine_keys_are_unchanged_with_selection_on(client, actors, people):
    assert client.post('/make', json={}, headers=headers(main.API_KEY)).status_code == 422            # master
    assert client.get('/auth/whoami', headers=headers(actors['floor']['key'])).json()['key_kind'] == 'actor'
    assert client.post('/make', json={}, headers=headers(actors['floor']['key'])).status_code == 422  # actor key
    session = owner_session(client, actors, 'floor')
    assert client.get('/auth/whoami', headers=headers(session)).json()['key_kind'] == 'session'


# ---------------------------------------------------------------- DCC-001: public key read-only in BOTH modes

def _db_writes(cur):
    cur.execute('SELECT coalesce(sum(n_tup_ins+n_tup_upd+n_tup_del),0) AS n FROM pg_stat_xact_user_tables')
    return cur.fetchone()['n']


@pytest.mark.parametrize('mode', ['on', 'off'])
def test_public_dashboard_key_is_read_only_in_both_modes(client, db_cursor, monkeypatch, migrated, mode):
    monkeypatch.setenv('PERSON_SELECT_ENABLED', '1' if mode == 'on' else '0')
    dash = {'X-API-Key': main.DASHBOARD_API_KEY}
    expected = 'PERSON_REQUIRED' if mode == 'on' else 'DASHBOARD_KEY_READ_ONLY'
    writes = sorted(k for k in main.DASHBOARD_KEY_ALLOWLIST if k[0] != 'GET' and k not in main.DASHBOARD_KEY_SELECTOR_READS)
    # Representative real mutations are all on the list being refused.
    for route in [('POST', '/dashboard/api/notes'), ('POST', '/receive/prepare'), ('POST', '/tickets/{ticket}/commit'),
                  ('POST', '/receive/commit'), ('POST', '/sales/orders/{order_id}/ship/commit'),
                  ('POST', '/sales/orders/{order_id}/close'), ('PATCH', '/sales/orders/{order_id}/status'),
                  ('POST', '/production/runs'), ('POST', '/supply-requests'), ('POST', '/expected-receipts')]:
        assert route in writes, route
    before = _db_writes(db_cursor)
    for method, path in writes:
        url = re.sub(r'\{[^}]+\}', '1', path)
        r = client.request(method, url, headers=dash, json={'category': 'note', 'title': 'x'})
        assert r.status_code == 403 and code(r) == expected, (mode, method, path, r.status_code, r.text[:200])
    assert _db_writes(db_cursor) == before, 'a refused public-key request wrote to the database'
    # Legitimate reads and the SELECT-only resolver stay open, and write nothing.
    for path in ('/inventory/current', '/dashboard/api/notes', '/sales/orders', '/production/runs', '/supplies/inventory'):
        assert client.get(path, headers=dash).status_code == 200, path
    r = client.post('/resolve', headers=dash, json={'query': 'Classic', 'kind': 'product', 'context': {'action': 'make'}})
    assert r.status_code == 200, r.text
    assert _db_writes(db_cursor) == before


def test_selection_on_then_off_keeps_every_real_credential_and_no_public_writes(client, db_cursor, monkeypatch, actors, people, payload):
    # ON: a tapped name records, attributed to that person.
    token = approved(client, actors)
    arturo = as_person(token, people['arturo'])
    on = client.post('/receive/prepare', json=payload, headers=arturo)
    assert on.status_code == 200, on.text
    assert on.json()['actor']['id'] == people['arturo']
    # Owner proof (Michael's own session) and the phone PIN path set up while ON.
    owner_proof = owner_session(client, actors)
    monkeypatch.setenv('PIN_SIGNIN_ENABLED', '1')
    db_cursor.execute((ROOT / 'migrations/089_device_pins.sql').read_text())
    floor_session = owner_session(client, actors, 'floor')
    enrolled = client.post('/auth/device/enroll', headers=headers(floor_session),
                           json={'actor_key': actors['floor']['key'], 'pin': '4829', 'label': 'Floor phone'})
    assert enrolled.status_code == 200, enrolled.text

    # OFF: selection is gone; nothing public reopens.
    monkeypatch.setenv('PERSON_SELECT_ENABLED', '0')
    assert client.get('/auth/whoami', headers=arturo).status_code == 403          # browser token = unknown key
    r = client.post('/receive/prepare', json=payload, headers={'X-API-Key': main.DASHBOARD_API_KEY})
    assert r.status_code == 403 and code(r) == 'DASHBOARD_KEY_READ_ONLY'
    assert client.get('/inventory/current', headers={'X-API-Key': main.DASHBOARD_API_KEY}).status_code == 200

    pin = client.post('/auth/pin', json={'device_secret': enrolled.json()['device_secret'], 'pin': '4829'})
    assert pin.status_code == 200, pin.text
    credentials = {
        'master': (headers(main.API_KEY), 'legacy-shared-key', None),
        'personal_key': (headers(actors['office']['key']), actors['office']['name'], actors['office']['id']),
        'key_session': (headers(owner_proof), actors['owner']['name'], actors['owner']['id']),
        'pin_session': (headers(pin.json()['session_token']), actors['floor']['name'], actors['floor']['id']),
    }
    for index, (label, (h, operator, actor_id)) in enumerate(credentials.items()):
        body = dict(payload, lot_code=f"{payload['lot_code']}-{index}", bol_reference=f"{payload['bol_reference']}-{index}")
        prepared = client.post('/receive/prepare', json=body, headers=h)
        assert prepared.status_code == 200, (label, prepared.text)
        # Repeat receives raise the existing duplicate warning; acknowledge only what the draft requires.
        acks = [w['code'] for w in prepared.json()['warnings'] if w.get('requires_ack')]
        done = client.post('/tickets/' + prepared.json()['ticket'] + '/commit',
                           json={'payload_hash': prepared.json()['payload_hash'], 'acknowledged_warnings': acks}, headers=h)
        assert done.status_code == 200, (label, done.text)
        db_cursor.execute('SELECT operator_id, entered_by_actor_id FROM transactions WHERE id=%s', (done.json()['transaction_id'],))
        row = db_cursor.fetchone()
        assert (row['operator_id'], row['entered_by_actor_id']) == (operator, actor_id), label
    # Owner-only decisions still need owner proof: the office key is refused.
    r = client.post('/exceptions/999999/approve', headers=headers(actors['office']['key']), json={})
    assert r.status_code == 403


# ---------------------------------------------------------------- seed and rollback

def seed_body(sql):
    return '\n'.join(line for line in sql.splitlines() if line not in ('BEGIN;', 'COMMIT;') and not line.startswith('SET LOCAL'))


def test_seed_refuses_any_wrong_mapping_and_seeds_the_exact_four(db_cursor, actors, migrated):
    db_cursor.execute("INSERT INTO actors (name, role, key_hash, active) VALUES ('ACT seed miriam','office',%s,true) RETURNING id", ('d' * 64,))
    miriam = db_cursor.fetchone()['id']
    good = [{'actor_id': actors['owner']['id'], 'name': actors['owner']['name']},
            {'actor_id': actors['office']['id'], 'name': actors['office']['name']},
            {'actor_id': actors['floor']['id'], 'name': actors['floor']['name']},
            {'actor_id': miriam, 'name': 'ACT seed miriam'}]

    def rows(mapping):
        return [dict(m, display=d, role_label=l, role=r) for m, (d, l, r) in zip(mapping, trusted_browsers.EXPECTED_CHOICES)]
    swapped = [good[0], good[2], good[1], good[3]]          # Luz <-> Arturo: wrong roles
    renamed = [dict(good[0], name='Michael')] + good[1:]    # display alias instead of the real actors.name
    for bad in (swapped, renamed):
        db_cursor.execute('SAVEPOINT seed')
        with pytest.raises(Exception, match='person_choices seed refused'):
            db_cursor.execute(seed_body(seed_person_choices.build_sql('test', rows(bad))))
        db_cursor.execute('ROLLBACK TO SAVEPOINT seed')
    db_cursor.execute(seed_body(seed_person_choices.build_sql('test', rows(good))))
    rows_ = trusted_browsers.choices(db_cursor)
    assert [(r['display_name'], r['id']) for r in rows_] == [('Michael', actors['owner']['id']), ('Luz', actors['office']['id']),
                                                            ('Arturo', actors['floor']['id']), ('Miriam', miriam)]
    db_cursor.execute('SELECT name FROM actors WHERE id=%s', (actors['owner']['id'],))
    assert db_cursor.fetchone()['name'] == actors['owner']['name']       # never renamed


def test_seed_cli_rejects_duplicate_ids():
    with pytest.raises(SystemExit):
        seed_person_choices.parse(['--environment', 'test', '--michael', '1:A', '--luz', '1:B', '--arturo', '2:C', '--miriam', '3:D'])


def test_migration_is_rerunnable_and_down_refuses_unexported_attribution(client, db_cursor, actors, people, payload):
    db_cursor.execute(MIGRATION.read_text())
    token = approved(client, actors)
    client.post('/receive/prepare', json=payload, headers=as_person(token, people['arturo']))
    down = DOWN.read_text().replace('BEGIN;', '').replace('COMMIT;', '')
    db_cursor.execute('SAVEPOINT down')
    with pytest.raises(Exception, match='091 rollback refused'):
        db_cursor.execute(down)
    db_cursor.execute('ROLLBACK TO SAVEPOINT down')
