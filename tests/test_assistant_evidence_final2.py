"""Every offered photo kind against every Assistant prepare, using real FL storage/DB.

Only the model tool choice is scripted. Accepted photos must survive Record,
replay, receipt detail and Assistant lookup; rejected photos create no draft.
"""
from uuid import uuid4

import pytest
from psycopg2.extras import Json

import attachments
import assistant_tools
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import (enabled, reason_catalog, customer_ticket_schema,
    model, turn, record, shipping_chat, upload_photo, action_body, make_draft)  # noqa: F401
from tests.test_write_tickets import payload  # noqa: F401
from tests.test_write_tickets_part2 import items  # noqa: F401
from tests.test_ship_tickets import seed_shipping, new_order

pytestmark = pytest.mark.db

SUPPORTED = {
    'receive': {'bol', 'lot_tag'},
    'ship_order': {'bol', 'truck_photo', 'signed_packing_slip'},
    'ship_standalone': {'bol', 'truck_photo', 'signed_packing_slip'},
    'ship_bulk': {'bol', 'truck_photo', 'signed_packing_slip'},
    'adjust': {'correction_photo'}, 'found': {'correction_photo'},
    'make': set(), 'pack': set(), 'create_customer': set(), 'create_order': set(),
}


def setup_prepare(client, actors, db_cursor, items, payload, action):
    """Use catalog fixture IDs approved by this local test's human selection."""
    chat = shipping_chat(client, actors)
    shipping = seed_shipping(db_cursor)
    state = {'resolved': {
        'product': [payload['product_id'], *(p['id'] for p in items.values()),
                    *shipping['products'].values()],
        'lot': [p['lot_id'] for p in items.values() if 'lot_id' in p] + [shipping['lot_id']],
        'supplier': [payload['supplier_id']], 'customer': [shipping['customer']],
    }}
    if action == 'receive':
        args = dict(payload)
    elif action in ('make', 'pack', 'adjust', 'found'):
        args = action_body(action, items)
        args.pop('reason', None)
        if action == 'adjust':
            args['reason_code'] = 'physical_count'
        if action == 'make':
            state['make_products'] = [items['batch']['id']]
        if action == 'pack':
            state['packing_context'] = {
                'source_product_id': items['batch']['id'],
                'target_product_id': items['finished']['id'],
                'lot_id': items['batch']['lot_id'], 'case_size_lb': 5,
            }
    elif action == 'create_customer':
        args = {'name': 'Photo customer ' + uuid4().hex}
    elif action == 'create_order':
        args = {'customer_id': shipping['customer'], 'customer_po': 'PHOTO-ORDER',
                'lines': [{'product_id': shipping['products']['food'], 'quantity': 1, 'unit': 'cases'}]}
    elif action == 'ship_order':
        order_id = new_order(client, shipping, actors['owner']['key'])
        state['resolved']['order'] = [order_id]
        args = {'order_id': order_id, 'ship_all': True, 'bol_reference': 'PHOTO-SHIP'}
    else:
        args = {'customer_id': shipping['customer'], 'order_reference': 'PHOTO-SHIP',
                'bol_reference': 'PHOTO-SHIP'}
        line = {'product_id': shipping['products']['food'], 'lot_id': shipping['lot_id'], 'quantity_lb': 50}
        if action == 'ship_bulk':
            args['lines'] = [line]
        else:
            args.update(line)
    db_cursor.execute('UPDATE assistant_sessions SET state=%s WHERE id=%s', (Json(state), chat[0]))
    return chat, args


@pytest.mark.parametrize('action', [name.removeprefix('prepare_') for name in assistant_tools.PREPARES])
@pytest.mark.parametrize('kind', attachments.KINDS)
def test_every_photo_kind_on_every_assistant_flow(client, actors, db_cursor, items, payload,
        monkeypatch, customer_ticket_schema, action, kind):
    assert set(SUPPORTED) == {name.removeprefix('prepare_') for name in assistant_tools.PREPARES}
    chat, args = setup_prepare(client, actors, db_cursor, items, payload, action)
    photo = upload_photo(client, chat, kind)
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    before = db_cursor.fetchone()['n']
    model(monkeypatch, ('prepare_' + action, args))
    response = turn(client, chat, 'Record this entry with its photo', attachment_ids=[photo])
    if kind not in SUPPORTED[action]:
        assert response.status_code == 200, response.text
        refusal = response.json()['cards'][0]
        assert refusal['kind'] == 'refusal', refusal
        detail = refusal['result']['detail']
        assert detail['error_code'] == 'ATTACHMENT_KIND_UNSUPPORTED'
        assert 'Remove it' in detail['message'] and 'Quítala' in detail['message_es']
        db_cursor.execute('SELECT count(*) AS n FROM assistant_drafts WHERE session_id=%s', (chat[0],))
        assert db_cursor.fetchone()['n'] == 0
        db_cursor.execute('SELECT count(*) AS n FROM transactions')
        assert db_cursor.fetchone()['n'] == before
        db_cursor.execute('SELECT ticket_id,transaction_id,shipment_id FROM attachments WHERE id=%s', (photo,))
        assert all(value is None for value in db_cursor.fetchone().values())
        return
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['kind'] == 'draft', card
    assert card['attachment_ids'] == [photo]
    confirmations = [{'lot_id': row['lot_id'], 'method': 'full_code', 'value': row['lot_code']}
                     for row in card['prepared']['draft'].get('input_plan', [])]
    saved = record(client, chat, card, lot_confirmations=confirmations)
    assert saved.status_code == 200, saved.text
    receipt = saved.json()['result']
    assert [(row['id'], row['kind']) for row in receipt['attachments']] == [(photo, kind)]
    assert receipt['attachments'][0]['ticket_id'] == card['prepared']['ticket_id']
    db_cursor.execute('SELECT ticket_id,transaction_id,shipment_id FROM attachments WHERE id=%s', (photo,))
    link = db_cursor.fetchone()
    assert link['ticket_id'] == card['prepared']['ticket_id']
    assert link['transaction_id'] or link['shipment_id']
    assert record(client, chat, card, lot_confirmations=confirmations).json()['result'] == receipt | {'replayed': True}
    db_cursor.execute('SELECT count(*) AS n FROM transactions WHERE ticket_id=%s', (card['prepared']['ticket_id'],))
    assert db_cursor.fetchone()['n'] == 1
    detail = client.get('/receipts/' + receipt['receipt_number'], headers=chat[1])
    assert detail.status_code == 200, detail.text
    assert [(row['id'], row['kind']) for row in detail.json()['attachments']] == [(photo, kind)]
    model(monkeypatch, ('receipt_lookup', {'receipt_number': receipt['receipt_number']}))
    lookup = turn(client, chat, 'Look up receipt ' + receipt['receipt_number'])
    assert lookup.status_code == 200, lookup.text
    assert [row['id'] for row in lookup.json()['cards'][0]['result']['attachments']] == [photo]
    download = client.get('/attachments/' + str(photo) + '/file', headers=chat[1])
    assert download.status_code == 200 and download.content.startswith(b'\x89PNG'), download.text


def test_mixed_supported_and_unsupported_photos_never_make_a_partial_draft(
        client, actors, db_cursor, items, payload, monkeypatch):
    chat, args = setup_prepare(client, actors, db_cursor, items, payload, 'receive')
    photos = [upload_photo(client, chat, kind) for kind in ('bol', 'lot_tag', 'other')]
    model(monkeypatch, ('prepare_receive', args))
    response = turn(client, chat, 'Record this receive', attachment_ids=photos)
    assert response.status_code == 200, response.text
    refusal = response.json()['cards'][0]
    assert refusal['kind'] == 'refusal'
    assert refusal['result']['detail']['error_code'] == 'ATTACHMENT_KIND_UNSUPPORTED'
    db_cursor.execute('SELECT count(*) AS n FROM assistant_drafts WHERE session_id=%s', (chat[0],))
    assert db_cursor.fetchone()['n'] == 0


@pytest.mark.parametrize('kind', ['bol', 'other'])
def test_old_pending_draft_never_loses_its_claimed_photo(
        client, actors, db_cursor, payload, monkeypatch, kind):
    chat = shipping_chat(client, actors)
    photo = upload_photo(client, chat, kind)
    card, _ = make_draft(client, chat, payload, db_cursor, monkeypatch)
    # The reviewed version put every uploaded ID on the card while omitting
    # unsupported types from the signed ticket. Simulate a persisted old card.
    card['attachment_ids'] = [photo]
    db_cursor.execute('UPDATE assistant_drafts SET card=%s WHERE id=%s', (Json(card), card['id']))
    response = record(client, chat, card)
    if kind == 'bol':
        assert response.status_code == 200, response.text
        receipt = response.json()['result']
        assert [row['id'] for row in receipt['attachments']] == [photo]
        assert receipt['attachments'][0]['transaction_id'] == receipt['transaction_id']
        assert record(client, chat, card).json()['result'] == receipt | {'replayed': True}
    else:
        assert response.status_code == 422, response.text
        assert response.json()['detail']['error_code'] == 'ATTACHMENT_KIND_UNSUPPORTED'
        db_cursor.execute('SELECT record_started_at FROM assistant_drafts WHERE id=%s', (card['id'],))
        assert db_cursor.fetchone()['record_started_at'] is None
        db_cursor.execute('SELECT count(*) AS n FROM transactions WHERE ticket_id=%s', (card['prepared']['ticket_id'],))
        assert db_cursor.fetchone()['n'] == 0


@pytest.mark.parametrize('kind', ['truck_photo', 'correction_photo', 'count_sheet', 'signed_packing_slip', 'other'])
@pytest.mark.parametrize('lost_response', [False, True])
def test_old_recorded_unsupported_integer_photo_does_not_block_receipt_recovery(
        client, actors, db_cursor, payload, monkeypatch, kind, lost_response):
    chat = shipping_chat(client, actors)
    photo = upload_photo(client, chat, kind)
    card, _ = make_draft(client, chat, payload, db_cursor, monkeypatch)
    original = record(client, chat, card).json()['result']
    card['attachment_ids'] = [photo]
    db_cursor.execute('UPDATE assistant_drafts SET card=%s WHERE id=%s', (Json(card), card['id']))
    if lost_response:
        db_cursor.execute("UPDATE assistant_drafts SET status='pending',result=NULL WHERE id=%s", (card['id'],))
    recovered = record(client, chat, card)
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()['result'] == original | {'replayed': True}
    db_cursor.execute('SELECT count(*) AS n FROM transactions WHERE ticket_id=%s', (card['prepared']['ticket_id'],))
    assert db_cursor.fetchone()['n'] == 1
    db_cursor.execute('SELECT ticket_id,transaction_id,shipment_id FROM attachments WHERE id=%s', (photo,))
    assert all(value is None for value in db_cursor.fetchone().values()), 'A replay must not attach new evidence to an immutable receipt'
    # Historical recovery must still reach the canonical actor/hash checks.
    db_cursor.execute("UPDATE assistant_drafts SET payload_hash='tampered' WHERE id=%s", (card['id'],))
    refused = record(client, chat, card)
    assert refused.status_code == 409, refused.text
    assert refused.json()['result']['detail']['error_code'] == 'TICKET_PAYLOAD_MISMATCH'
