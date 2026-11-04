"""Four-name HTTP entry uses the real dependency and real local PostgreSQL."""
from datetime import timedelta

import pytest

import main
import permissions
import kosher_tiers
import resolution_receipts
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_trusted_browsers import migrated, people  # noqa: F401
from tests.test_write_tickets import payload  # noqa: F401

pytestmark = pytest.mark.db
NAMES = ('Michael', 'Luz', 'Arturo', 'Miriam')


def named(name):
    return {'X-FL-Person': name}


@pytest.mark.parametrize('name', NAMES)
def test_fresh_name_needs_no_key_browser_or_feature_flag(client, monkeypatch, people, db_cursor, name):
    for setting in ('PERSON_SELECT_ENABLED', 'KEY_SESSIONS_ENABLED', 'PIN_SIGNIN_ENABLED'):
        monkeypatch.setenv(setting, '0')
    db_cursor.execute('UPDATE person_choices SET active=false')
    db_cursor.execute('UPDATE actors SET active=false,role=\'office\' WHERE id=ANY(%s)', (list(people.values()),))
    result = client.get('/auth/whoami', headers=named(name))
    assert result.status_code == 200, result.text
    body = result.json()
    assert body['actor']['id'] == people[name.lower()]
    assert body['actor']['name'] == name
    assert all(body['permissions'][action] for action in ('make', 'pack', 'ship_order', 'ship_standalone', 'receive'))
    assert body['permissions']['kosher_attestation'] is (name == 'Michael')
    db_cursor.execute('SELECT count(*) AS n FROM trusted_browsers')
    assert db_cursor.fetchone()['n'] == 0


@pytest.mark.parametrize('stale_key', ['fls_expired', 'flb_unapproved', 'wrong-key'])
def test_tapped_name_wins_over_stale_browser_credentials(client, people, stale_key):
    response = client.get('/auth/whoami', headers=named('Arturo') | {'X-API-Key': stale_key})
    assert response.status_code == 200, response.text
    assert response.json()['actor']['name'] == 'Arturo'


@pytest.mark.parametrize('name', NAMES)
def test_actual_record_and_replay_keep_tapped_name_after_legacy_deactivation(client, people, payload, db_cursor, name):
    prepared = client.post('/receive/prepare', json=payload, headers=named(name))
    assert prepared.status_code == 200, prepared.text
    p = prepared.json()
    db_cursor.execute('UPDATE actors SET active=false,role=\'office\' WHERE id=%s', (people[name.lower()],))
    db_cursor.execute('UPDATE person_choices SET active=false WHERE actor_id=%s', (people[name.lower()],))
    first = client.post('/tickets/' + p['ticket'] + '/commit', json={'payload_hash': p['payload_hash']}, headers=named(name))
    assert first.status_code == 200, first.text
    receipt = first.json()
    assert receipt['success'] and receipt['receipt_number'].startswith('RCV-')
    repeat = client.post('/tickets/' + p['ticket'] + '/commit', json={'payload_hash': p['payload_hash']}, headers=named(name))
    assert repeat.status_code == 200 and repeat.json()['replayed']
    db_cursor.execute('SELECT operator_id,entered_by_actor_id FROM transactions WHERE ticket_id=%s', (p['ticket_id'],))
    assert [dict(row) for row in db_cursor.fetchall()] == [{'operator_id': name, 'entered_by_actor_id': people[name.lower()]}]


def test_michael_has_admin_void_adjust_and_kosher_without_owner_proof(client, people, db_cursor, payload):
    h = named('Michael')
    me = client.get('/auth/whoami', headers=h).json()
    assert all(me['permissions'].values())
    assert client.get('/admin/product-bom', headers=h).status_code == 200
    assert client.get('/auth/browsers', headers=h).status_code == 200
    # Real prepare endpoints get past auth to domain validation, with no key.
    for url in ('/void/2147483647/prepare', '/adjust/prepare'):
        result = client.post(url, json={}, headers=h)
        assert result.status_code not in (401, 403), (url, result.text)
    actor = {'id': people['michael'], 'name': 'Michael', 'role': 'owner', 'key_kind': 'actor',
             'via': 'selection', 'trusted_browser_id': None}
    db_cursor.execute('UPDATE actors SET active=false,role=\'office\' WHERE id=%s', (actor['id'],))
    assert resolution_receipts.current_actor(db_cursor, actor) == actor
    proof = kosher_tiers.attest(db_cursor, actor)
    assert proof['actor_id'] == actor['id'] and proof['actor_name'] == 'Michael'
    assert permissions.require_backdating(actor, main.get_plant_now()-timedelta(days=30), main.get_plant_now())['status'] == 'backfill'


def test_unknown_name_never_becomes_an_operator(client, people):
    response = client.get('/auth/whoami', headers=named('Somebody else'))
    assert response.status_code in (401, 403)


from tests.test_kosher_tiers import items  # noqa: E402,F401
from tests.test_write_tickets_part2 import body  # noqa: E402


@pytest.mark.parametrize('preparer', ['Michael', 'Arturo'])
def test_name_only_kosher_make_and_owner_approval_are_real_posts(client, people, items, db_cursor, preparer):
    data = body('make', items) | {'confirmed_sku': True}
    h = named(preparer)
    prepared = client.post('/make/prepare', json=data, headers=h)
    assert prepared.status_code == 200, prepared.text
    confirmations = [{'lot_id': row['lot_id'], 'method': 'full_code', 'value': row['lot_code']}
                     for row in prepared.json()['draft']['input_plan']]
    p = client.post('/make/prepare', json=data | {'lot_confirmations': confirmations}, headers=h).json()
    assert p['can_commit'], p
    # A legacy actor deactivation or role edit cannot demand a new human sign-in.
    db_cursor.execute("UPDATE actors SET active=false,role='office' WHERE id=ANY(%s)", (list(people.values()),))
    recorded = client.post('/tickets/' + p['ticket'] + '/commit', json={'payload_hash': p['payload_hash']}, headers=h)
    assert recorded.status_code == (200 if preparer == 'Michael' else 202), recorded.text
    if preparer != 'Michael':
        recorded = client.post(f"/exceptions/{recorded.json()['exception_id']}/approve", json={}, headers=named('Michael'))
        assert recorded.status_code == 200, recorded.text
    r = recorded.json()
    db_cursor.execute('SELECT operator_id,entered_by_actor_id,kosher_attested_by_actor_id FROM transactions WHERE id=%s', (r['transaction_id'],))
    assert dict(db_cursor.fetchone()) == {'operator_id': preparer, 'entered_by_actor_id': people[preparer.lower()],
                                         'kosher_attested_by_actor_id': people['michael']}


@pytest.mark.parametrize('route', ['ticket', 'direct'])
def test_michael_voids_another_selected_persons_post_without_credentials(client, people, payload, db_cursor, route):
    p = client.post('/receive/prepare', json=payload, headers=named('Luz')).json()
    recorded = client.post('/tickets/' + p['ticket'] + '/commit', json={'payload_hash': p['payload_hash']}, headers=named('Luz'))
    assert recorded.status_code == 200, recorded.text
    transaction_id = recorded.json()['transaction_id']
    db_cursor.execute("UPDATE actors SET active=false,role='office' WHERE id=%s", (people['michael'],))
    if route == 'direct':
        void = client.post(f'/void/{transaction_id}', json={'reason': 'Name-only test correction'}, headers=named('Michael'))
    else:
        draft = client.post(f'/void/{transaction_id}/prepare', json={'reason': 'Name-only test correction'}, headers=named('Michael'))
        assert draft.status_code == 200, draft.text
        p = draft.json()
        void = client.post('/tickets/' + p['ticket'] + '/commit', json={'payload_hash': p['payload_hash']}, headers=named('Michael'))
    assert void.status_code == 200 and void.json()['receipt_number'].startswith('VD-'), void.text
    db_cursor.execute("SELECT operator_id FROM ledger_corrections WHERE target_table='transactions' AND target_id=%s", (transaction_id,))
    assert db_cursor.fetchone()['operator_id'] == 'Michael'


@pytest.mark.parametrize('name', ['Luz', 'Arturo', 'Miriam'])
@pytest.mark.parametrize('source', ['own_today', 'other_person', 'own_yesterday'])
def test_selected_staff_void_scope_is_own_same_day(client, people, payload, db_cursor, name, source):
    operator = 'Michael' if source == 'other_person' else name
    if source == 'own_yesterday':
        payload = payload | {'occurred_at': (main.get_plant_now() - timedelta(days=1)).isoformat()}
    prepared = client.post('/receive/prepare', json=payload, headers=named(operator))
    assert prepared.status_code == 200, prepared.text
    p = prepared.json()
    recorded = client.post('/tickets/' + p['ticket'] + '/commit', json={'payload_hash': p['payload_hash']}, headers=named(operator))
    assert recorded.status_code == 200, recorded.text
    transaction_id = recorded.json()['transaction_id']
    draft = client.post(f'/void/{transaction_id}/prepare', json={'reason': 'Test staff correction'}, headers=named(name))
    if source != 'own_today':
        assert draft.status_code == 403, draft.text
        assert draft.json()['detail']['error_code'] == 'VOID_OWNER_REQUIRED'
        db_cursor.execute("SELECT count(*) AS n FROM ledger_corrections WHERE target_table='transactions' AND target_id=%s", (transaction_id,))
        assert db_cursor.fetchone()['n'] == 0
        return
    assert draft.status_code == 200, draft.text
    p = draft.json()
    void = client.post('/tickets/' + p['ticket'] + '/commit', json={'payload_hash': p['payload_hash']}, headers=named(name))
    assert void.status_code == 200 and void.json()['receipt_number'].startswith('VD-'), void.text
    db_cursor.execute("SELECT operator_id FROM ledger_corrections WHERE target_table='transactions' AND target_id=%s", (transaction_id,))
    assert db_cursor.fetchone()['operator_id'] == name


@pytest.mark.parametrize('name', ['Luz', 'Arturo', 'Miriam'])
def test_selected_staff_uses_dashboard_ticket_routes_not_legacy_direct_posts(client, people, name):
    for path in ('/make', '/receive', '/pack', '/adjust', '/ship', '/customers', '/receive/commit', '/void/1'):
        response = client.post(path, json={}, headers=named(name))
        assert response.status_code == 403, (path, response.text)
    # Ordinary forms still reach their actual prepare handlers.
    for path in ('/make/prepare', '/pack/prepare', '/receive/prepare', '/sales/orders/1/ship/prepare'):
        response = client.post(path, json={}, headers=named(name))
        assert response.status_code not in (401, 403), (path, response.text)
