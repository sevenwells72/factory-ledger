"""A12 review regressions: every extra-kosher creation path needs provenance."""
from datetime import timedelta
from uuid import uuid4

import pytest
import main
from tests.test_actor_attribution import client, actors
from tests.test_kosher_tiers import items, approve
from tests.test_write_tickets import headers, commit, ticket_row, posted_count, error
from tests.test_write_tickets_part2 import body, prepare

pytestmark = pytest.mark.db


def role_key(actors, role):
    if role == 'dashboard':
        return main.DASHBOARD_API_KEY
    return None if role == 'shared' else actors[role]['key']


def receive_payload(cur, product_id):
    token = uuid4().hex[:12].upper()
    cur.execute("""INSERT INTO products(id,name,type,active,kosher_tier,case_size_lb)
        VALUES (%s,%s,'batch',true,'extra_kosher',5) RETURNING name""",
        (product_id, 'A12 Review Extra Batch ' + str(product_id) + ' ' + token))
    product_name = cur.fetchone()['name']
    cur.execute('INSERT INTO suppliers(name) VALUES (%s) RETURNING id,name', ('A12 supplier ' + token,))
    supplier = cur.fetchone()
    payload = {'product_id': product_id, 'supplier_id': supplier['id'], 'cases': 2,
        'case_size_lb': 5, 'bol_reference': 'A12-' + token, 'lot_code': 'A12-RCV-' + token,
        'supplier_lot_code': 'SUP-' + token, 'occurred_at': main.get_plant_now().isoformat()}
    legacy = {k: v for k, v in payload.items() if k not in ('product_id', 'supplier_id')}
    legacy.update(product_name=product_name, shipper_name=supplier['name'])
    return payload, legacy


def no_output_lot(cur, product_id, lot_code):
    cur.execute('SELECT id FROM lots WHERE product_id=%s AND lot_code=%s', (product_id, lot_code))
    assert cur.fetchone() is None


@pytest.mark.parametrize('product_id', [283, 284])
@pytest.mark.parametrize('role', ['floor', 'office', 'owner', 'shared', 'dashboard'])
def test_extra_batch_receive_ticket_refused_for_every_role(client, db_cursor, actors, product_id, role):
    payload, _ = receive_payload(db_cursor, product_id)
    key = role_key(actors, role)
    response = client.post('/receive/prepare', json=payload, headers=headers(key))
    assert response.status_code == 200, response.text
    prepared = response.json()
    assert prepared['can_commit'] is False
    assert prepared['blockers'][0]['code'] == 'KOSHER_BATCH_RECEIVE_REFUSED'
    error(commit(client, prepared, key), 409, 'TICKET_STALE')
    assert posted_count(db_cursor, prepared) == 0
    no_output_lot(db_cursor, product_id, payload['lot_code'])


@pytest.mark.parametrize('product_id', [283, 284])
@pytest.mark.parametrize('role', ['floor', 'office', 'owner', 'shared', 'dashboard'])
@pytest.mark.parametrize('path,mode', [('/receive', 'preview'), ('/receive', 'commit'),
    ('/receive/preview', 'commit'), ('/receive/commit', 'preview')])
def test_extra_batch_receive_legacy_and_aliases_refused(client, db_cursor, actors, product_id, role, path, mode):
    payload, legacy = receive_payload(db_cursor, product_id)
    response = client.post(path, json=legacy | {'mode': mode}, headers=headers(role_key(actors, role)))
    if role == 'dashboard' and path == '/receive':
        assert response.status_code == 403, response.text
        assert response.json()['detail'] == 'API key not authorized for this endpoint'
    else:
        error(response, 409, 'KOSHER_BATCH_RECEIVE_REFUSED')
    no_output_lot(db_cursor, product_id, payload['lot_code'])
    db_cursor.execute('SELECT count(*) AS n FROM transaction_lines WHERE product_id=%s', (product_id,))
    assert db_cursor.fetchone()['n'] == 0


@pytest.mark.parametrize('role', ['floor', 'office', 'owner', 'shared', 'dashboard'])
@pytest.mark.parametrize('direct', [False, True])
def test_extra_finished_receive_cannot_skip_pack_or_approved_found(client, db_cursor, actors, role, direct):
    payload, legacy = receive_payload(db_cursor, 285)
    db_cursor.execute("UPDATE products SET type='finished' WHERE id=285")
    key = role_key(actors, role)
    if direct:
        response = client.post('/receive', json=legacy | {'mode': 'commit'}, headers=headers(key))
        if role == 'dashboard':
            assert response.status_code == 403, response.text
            assert response.json()['detail'] == 'API key not authorized for this endpoint'
        else:
            error(response, 409, 'KOSHER_ATTESTATION_REQUIRED')
    else:
        response = client.post('/receive/prepare', json=payload, headers=headers(key))
        assert response.status_code == 200, response.text
        prepared = response.json()
        assert not prepared['can_commit']
        assert prepared['blockers'][0]['code'] == 'KOSHER_ATTESTATION_REQUIRED'
        error(commit(client, prepared, key), 409, 'TICKET_STALE')
        assert posted_count(db_cursor, prepared) == 0
    no_output_lot(db_cursor, 285, payload['lot_code'])


def found_payload(items, product_key):
    return body('found', items) | {'product_id': items[product_key]['id'],
        'lot_code': 'A12-FND-' + uuid4().hex[:12].upper()}


def held_found(client, db_cursor, items, actors, product_key='batch', role='floor', **extra):
    payload = found_payload(items, product_key) | extra
    prepared = prepare(client, 'found', payload, actors[role]['key'])
    assert prepared['can_commit'], prepared
    response = commit(client, prepared, actors[role]['key'])
    assert response.status_code == 202, response.text
    held = response.json()
    assert held['posted'] is False and held['held'] is True
    assert held['kosher_tier'] == 'extra_kosher'
    assert held['receipt_number'].startswith('FND-')
    assert held['error_code'] == 'KOSHER_ATTESTATION_REQUIRED'
    assert posted_count(db_cursor, prepared) == 0
    no_output_lot(db_cursor, payload['product_id'], payload['lot_code'])
    return prepared, held, payload


@pytest.mark.parametrize('product_key', ['batch', 'finished'])
@pytest.mark.parametrize('role', ['floor', 'owner'])
def test_extra_found_held_until_separate_owner_approval_and_replays(client, db_cursor, items, actors, product_key, role):
    prepared, held, payload = held_found(client, db_cursor, items, actors, product_key, role)
    row = ticket_row(db_cursor, prepared)
    assert row['status'] == 'awaiting_approval' and row['committed_at'] is None
    assert timedelta(days=6) < row['expires_at'] - row['prepared_at'] < timedelta(days=8)
    assert commit(client, prepared, actors[role]['key']).json() == held
    db_cursor.execute("SELECT * FROM exceptions WHERE ticket_id=%s AND kind='KOSHER_ATTESTATION'", (prepared['ticket_id'],))
    exceptions = db_cursor.fetchall()
    assert len(exceptions) == 1 and exceptions[0]['detail']['action'] == 'found'
    detail = client.get('/receipts/' + held['receipt_number'], headers=headers()).json()
    assert detail['status'] == 'awaiting_approval' and detail['transactions'] == [] and detail['lots'] == []

    approved = approve(client, held, actors)
    assert approved.status_code == 200, approved.text
    result = approved.json()
    assert result['receipt_number'] == held['receipt_number']
    assert result['kosher_attested_by_actor_id'] == actors['owner']['id']
    assert result['kosher_tier'] == 'extra_kosher'
    db_cursor.execute('SELECT * FROM lots WHERE id=%s', (result['lot_id'],))
    lot = db_cursor.fetchone()
    db_cursor.execute('SELECT * FROM transactions WHERE id=%s', (result['transaction_id'],))
    transaction = db_cursor.fetchone()
    db_cursor.execute('SELECT * FROM ticket_attestations WHERE ticket_id=%s', (prepared['ticket_id'],))
    attestation = db_cursor.fetchone()
    assert lot['lot_code'] == payload['lot_code']
    assert lot['kosher_tier'] == transaction['kosher_tier'] == 'extra_kosher'
    assert lot['kosher_attested_by_actor_id'] == transaction['kosher_attested_by_actor_id'] == attestation['actor_id'] == actors['owner']['id']
    assert lot['kosher_attested_at'] == transaction['kosher_attested_at'] == attestation['attested_at']
    assert transaction['entered_by_actor_id'] == actors[role]['id']
    assert approve(client, held, actors).json() == result | {'replayed': True}
    assert commit(client, prepared, actors[role]['key']).json() == result | {'replayed': True}
    assert posted_count(db_cursor, prepared) == 1


@pytest.mark.parametrize('product_key', ['batch', 'finished'])
def test_rejected_extra_found_never_creates_lot_or_posts(client, db_cursor, items, actors, product_key):
    prepared, held, payload = held_found(client, db_cursor, items, actors, product_key)
    path = f"/exceptions/{held['exception_id']}/reject"
    decision = {'resolution_kind': 'declined', 'note': 'Count cannot attest this stock'}
    response = client.post(path, json=decision, headers=headers(actors['owner']['key']))
    assert response.status_code == 200, response.text
    assert response.json()['posted'] is False
    assert client.post(path, json=decision, headers=headers(actors['owner']['key'])).json()['replayed']
    error(approve(client, held, actors), 409, 'EXCEPTION_CLOSED')
    error(commit(client, prepared, actors['floor']['key']), 409, 'TICKET_NOT_COMMITTABLE')
    assert posted_count(db_cursor, prepared) == 0
    no_output_lot(db_cursor, payload['product_id'], payload['lot_code'])


@pytest.mark.parametrize('decision', ['approve', 'reject'])
@pytest.mark.parametrize('role', ['floor', 'office'])
def test_extra_found_only_owner_may_approve_or_reject(client, db_cursor, items, actors, decision, role):
    prepared, held, payload = held_found(client, db_cursor, items, actors)
    response = client.post(f"/exceptions/{held['exception_id']}/{decision}",
        json={} if decision == 'approve' else {'resolution_kind': 'declined', 'note': 'x'},
        headers=headers(actors[role]['key']))
    assert response.status_code == 403, response.text
    assert posted_count(db_cursor, prepared) == 0
    no_output_lot(db_cursor, payload['product_id'], payload['lot_code'])


def test_photo_and_generic_resolve_cannot_release_extra_found(client, db_cursor, items, actors):
    prepared, held, payload = held_found(client, db_cursor, items, actors)
    error(client.post(f"/exceptions/{held['exception_id']}/resolve",
        json={'resolution_kind': 'waived', 'note': 'x'}, headers=headers(actors['owner']['key'])),
        409, 'USE_APPROVE_OR_REJECT')
    assert commit(client, prepared, actors['floor']['key'], attachment_ref='not-an-owner-attestation').status_code == 202
    assert posted_count(db_cursor, prepared) == 0
    no_output_lot(db_cursor, payload['product_id'], payload['lot_code'])


@pytest.mark.parametrize('product_key', ['batch', 'finished'])
@pytest.mark.parametrize('role', ['floor', 'office', 'owner', 'shared'])
def test_extra_found_legacy_refused_even_for_owner(client, db_cursor, items, actors, product_key, role):
    payload = found_payload(items, product_key)
    response = client.post('/inventory/found', json=payload, headers=headers(role_key(actors, role)))
    if role == 'shared':
        error(response, 403, 'KOSHER_ATTESTATION_REQUIRED')
    else:
        # This legacy route already excludes named actor keys. The shared
        # master key reaches the handler and must hit the new ticket guard.
        assert response.status_code == 403, response.text
        assert response.json()['detail'] == 'API key not authorized for this endpoint'
    no_output_lot(db_cursor, payload['product_id'], payload['lot_code'])


def test_owner_cannot_bypass_extra_make_ticket_via_legacy(client, db_cursor, items, actors):
    code = 'A12-MK-' + uuid4().hex[:12].upper()
    error(client.post('/make', json={'product_name': items['batch']['name'], 'batches': 1,
        'lot_code': code, 'confirmed_sku': True, 'mode': 'commit'}, headers=headers(actors['owner']['key'])),
        403, 'KOSHER_ATTESTATION_REQUIRED')
    no_output_lot(db_cursor, items['batch']['id'], code)


def test_extra_found_cannot_reuse_existing_lot(client, db_cursor, items, actors):
    payload = found_payload(items, 'batch') | {'lot_code': items['extra_lot']['lot_code']}
    prepared = prepare(client, 'found', payload, actors['owner']['key'])
    assert not prepared['can_commit']
    assert prepared['blockers'][0]['code'] == 'KOSHER_NEW_BATCH_LOT_REQUIRED'
    error(commit(client, prepared, actors['owner']['key']), 409, 'TICKET_STALE')
    assert posted_count(db_cursor, prepared) == 0
    assert main.lot_on_hand(db_cursor, items['extra_lot']['lot_id']) == 100


def test_extra_found_changed_catalog_keeps_approval_unposted(client, db_cursor, items, actors):
    prepared, held, payload = held_found(client, db_cursor, items, actors)
    db_cursor.execute("UPDATE products SET kosher_tier='regular' WHERE id=%s", (payload['product_id'],))
    error(approve(client, held, actors), 409, 'TICKET_STALE')
    assert ticket_row(db_cursor, prepared)['status'] == 'awaiting_approval'
    assert posted_count(db_cursor, prepared) == 0
    no_output_lot(db_cursor, payload['product_id'], payload['lot_code'])


def test_extra_found_shared_key_cannot_create_unattributable_hold(client, db_cursor, items):
    prepared = prepare(client, 'found', found_payload(items, 'batch'))
    response = commit(client, prepared)
    assert response.status_code in (403, 409), response.text
    assert posted_count(db_cursor, prepared) == 0
    no_output_lot(db_cursor, items['batch']['id'], prepared['draft']['lot_code'])


def test_large_extra_found_uses_kosher_hold_and_owner_approval(client, db_cursor, items, actors):
    prepared, held, payload = held_found(client, db_cursor, items, actors, quantity=600)
    db_cursor.execute('SELECT kind FROM exceptions WHERE ticket_id=%s', (prepared['ticket_id'],))
    assert [row['kind'] for row in db_cursor.fetchall()] == ['KOSHER_ATTESTATION']
    response = approve(client, held, actors)
    assert response.status_code == 200, response.text
    assert response.json()['kosher_attested_by_actor_id'] == actors['owner']['id']
    assert main.lot_on_hand(db_cursor, response.json()['lot_id']) == 600


def test_extra_found_seven_day_expiry_keeps_stock_unposted(client, db_cursor, items, actors):
    prepared, held, payload = held_found(client, db_cursor, items, actors)
    db_cursor.execute("UPDATE write_tickets SET expires_at=clock_timestamp()-interval '1 second' WHERE id=%s", (prepared['ticket_id'],))
    error(approve(client, held, actors), 409, 'TICKET_EXPIRED')
    error(commit(client, prepared, actors['floor']['key']), 409, 'TICKET_EXPIRED')
    assert posted_count(db_cursor, prepared) == 0
    no_output_lot(db_cursor, payload['product_id'], payload['lot_code'])


def test_regular_found_catalog_upgrade_cannot_create_extra_lot(client, db_cursor, items, actors):
    db_cursor.execute("UPDATE products SET kosher_tier='regular' WHERE id=%s", (items['batch']['id'],))
    payload = found_payload(items, 'batch')
    prepared = prepare(client, 'found', payload, actors['owner']['key'])
    assert prepared['can_commit']
    db_cursor.execute("UPDATE products SET kosher_tier='extra_kosher' WHERE id=%s", (items['batch']['id'],))
    error(commit(client, prepared, actors['owner']['key']), 409, 'TICKET_STALE')
    assert posted_count(db_cursor, prepared) == 0
    no_output_lot(db_cursor, payload['product_id'], payload['lot_code'])


@pytest.mark.parametrize('direct', [False, True])
def test_pack_cannot_manufacture_extra_batch_output(client, db_cursor, items, actors, direct):
    # A different extra batch is needed because pack already forbids source=target.
    cur = db_cursor
    cur.execute("""INSERT INTO products(name,type,active,kosher_tier,case_size_lb)
        VALUES (%s,'batch',true,'extra_kosher',5) RETURNING id,name""",
        ('A12 Packed Extra Batch ' + uuid4().hex[:12],))
    target = cur.fetchone()
    code = 'A12-PK-' + uuid4().hex[:12].upper()
    if direct:
        response = client.post('/pack', json={'source_product': items['batch']['name'],
            'target_product': target['name'], 'cases': 2, 'target_lot_code': code, 'mode': 'commit',
            'lot_allocations': [{'lot_code': items['extra_lot']['lot_code'], 'quantity_lb': 10}]},
            headers=headers(actors['owner']['key']))
        error(response, 409, 'KOSHER_BATCH_PACK_REFUSED')
    else:
        prepared = prepare(client, 'pack', body('pack', items) | {'target_product_id': target['id'],
            'target_lot_code': code}, actors['owner']['key'])
        assert not prepared['can_commit']
        assert prepared['blockers'][0]['code'] == 'KOSHER_BATCH_PACK_REFUSED'
        error(commit(client, prepared, actors['owner']['key']), 409, 'TICKET_STALE')
        assert posted_count(db_cursor, prepared) == 0
    no_output_lot(db_cursor, target['id'], code)
    assert main.lot_on_hand(db_cursor, items['extra_lot']['lot_id']) == 100


@pytest.mark.parametrize('source_key,target_key', [('ingredient', 'batch'), ('extra_lot', 'ingredient'), ('batch', 'ingredient')])
def test_reassign_cannot_launder_extra_kosher_identity(client, db_cursor, items, actors, source_key, target_key):
    source = items[source_key]
    cur = db_cursor
    cur.execute('SELECT product_id,kosher_tier FROM lots WHERE id=%s', (source['lot_id'],))
    before = cur.fetchone()
    response = client.post(f"/lots/{source['lot_id']}/reassign",
        json={'to_product_id': items[target_key]['id'], 'reason_code': 'data_entry_error'},
        headers=headers())
    error(response, 409, 'KOSHER_LOT_REASSIGN_REFUSED')
    cur.execute('SELECT product_id,kosher_tier FROM lots WHERE id=%s', (source['lot_id'],))
    assert cur.fetchone() == before


@pytest.mark.parametrize('reverse', [False, True])
def test_merge_cannot_launder_historical_regular_stock_into_extra_lot(client, db_cursor, items, actors, reverse):
    regular, extra = items['batch']['lot_id'], items['extra_lot']['lot_id']
    source, target = (extra, regular) if reverse else (regular, extra)
    response = client.post('/admin/lots/merge', json={'source_lot_id': source,
        'target_lot_id': target, 'reason': 'Consolidate duplicate labels'},
        headers=headers())
    error(response, 409, 'KOSHER_LOT_MERGE_REFUSED')
    assert main.lot_on_hand(db_cursor, regular) == main.lot_on_hand(db_cursor, extra) == 100
    db_cursor.execute('SELECT status FROM lots WHERE id=%s', (source,))
    assert db_cursor.fetchone()['status'] != 'merged'
