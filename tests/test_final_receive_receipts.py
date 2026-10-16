"""Final staging fixes through real ticket commits; no model or tag shortcuts."""
from contextlib import contextmanager
import pytest
import psycopg2
import main
import attachments
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import payload, headers, prepare, commit, posted_count  # noqa: F401
from tests.test_write_tickets_part2 import items, body  # noqa: F401
from tests.test_ship_tickets import upload, storage  # noqa: F401

pytestmark = pytest.mark.db


@pytest.mark.parametrize('field,note', [('notes', 'Counted behind the pallet'), ('notes_es', 'Contado detrás de la tarima')])
@pytest.mark.parametrize('reason', ['physical_count', 'unknown'])
def test_found_accepts_note_in_either_original_language(client, db_cursor, items, actors, field, note, reason):
    key = actors['floor']['key']
    data = body('found', items) | {'reason_code': reason, field: note}
    prepared = client.post('/inventory/found/prepare', json=data, headers=headers(key)).json()
    assert prepared['can_commit'], prepared
    assert prepared['draft'][field] == note
    assert prepared['draft']['reason']['note'] == note
    result = commit(client, prepared, key)
    assert result.status_code == 200, result.text
    db_cursor.execute('SELECT reason_notes,reason_notes_es FROM inventory_adjustments WHERE lot_id=%s', (result.json()['lot_id'],))
    audit = db_cursor.fetchone()
    assert audit['reason_notes' if field == 'notes' else 'reason_notes_es'] == note


@pytest.mark.parametrize('late', [False, True])
def test_receive_receipt_retains_supplier_lot_bol_and_real_private_tag(client, db_cursor, payload, actors, late):
    key = actors['floor']['key']
    photo = upload(client, key, kind='lot_tag')
    prepared = prepare(client, payload | ({} if late else {'attachment_ids': [photo['id']]}), key)
    assert prepared['can_commit'], prepared
    if not late:
        assert prepared['draft']['attachments'][0]['kind'] == 'lot_tag'
    result = commit(client, prepared, key, **({'attachment_ids': [photo['id']]} if late else {}))
    assert result.status_code == 200, result.text
    result = result.json()
    assert result['supplier_lot_code'] == payload['supplier_lot_code']
    assert result['bol_reference'] == payload['bol_reference']
    assert 'Supplier lot: ' + payload['supplier_lot_code'] in result['receipt_text']
    assert 'BOL: ' + payload['bol_reference'] in result['receipt_text']
    assert result['attachments'][0]['id'] == photo['id']
    assert result['attachments'][0]['transaction_id'] == result['transaction_id']
    assert commit(client, prepared, key).json() == result | {'replayed': True}
    assert posted_count(db_cursor, prepared) == 1
    detail = client.get('/receipts/' + result['receipt_number'], headers=headers(key)).json()
    assert detail['lots'][0]['supplier_lot_code'] == payload['supplier_lot_code']
    assert detail['transactions'][0]['bol_reference'] == payload['bol_reference']
    assert detail['attachments'][0]['id'] == photo['id']
    assert detail['attachments'][0]['ticket_id'] == prepared['ticket_id']
    assert client.get('/attachments/' + str(photo['id']) + '/file', headers=headers(key)).status_code == 200
    # Receipt metadata never grants another floor worker private file access.
    db_cursor.execute("UPDATE actors SET role='floor' WHERE id=%s", (actors['office']['id'],))
    main._reset_actor_cache()
    assert client.get('/attachments/' + str(photo['id']) + '/file', headers=headers(actors['office']['key'])).status_code == 403


@pytest.mark.parametrize('failure', ['actor', 'kind', 'context', 'missing_file'])
def test_receive_photo_refusals_never_post(client, db_cursor, payload, actors, failure, monkeypatch):
    key = actors['floor']['key']
    photo = upload(client, actors['office']['key'] if failure == 'actor' else key,
                   kind='bol' if failure == 'kind' else 'lot_tag')
    prepared = prepare(client, payload, key)
    if failure == 'context':
        other = prepare(client, payload | {'bol_reference': 'OTHER'}, key)
        db_cursor.execute('UPDATE attachments SET ticket_id=%s WHERE id=%s', (other['ticket_id'], photo['id']))
    if failure == 'missing_file':
        db_cursor.execute('SELECT storage_path FROM attachments WHERE id=%s', (photo['id'],))
        (attachments._volume() / db_cursor.fetchone()['storage_path']).unlink()
    result = commit(client, prepared, key, attachment_ids=[photo['id']])
    assert result.status_code in (403, 422, 502), result.text
    assert posted_count(db_cursor, prepared) == 0
    db_cursor.execute('SELECT transaction_id FROM attachments WHERE id=%s', (photo['id'],))
    assert db_cursor.fetchone()['transaction_id'] is None


def test_spanish_different_lot_records_actual_tag_without_using_suggested_stock(client, db_cursor, items, actors):
    key = actors['floor']['key']
    ingredient = items['ingredient']
    db_cursor.execute('UPDATE batch_formulas SET quantity_lb=8.88 WHERE product_id=%s', (items['batch']['id'],))
    prepared = client.post('/make/prepare', json=body('make', items) | {'batches': 10}, headers=headers(key)).json()
    assert not prepared['can_commit']
    assert prepared['draft']['lot_confirmation_group']['ingredients'][0]['needed_lb'] == 88.8
    actual_tag = 'ETIQUETA-FISICA-283'
    result = commit(client, prepared, key, lot_confirmations=[
        {'lot_id': i['lot_id'], 'method': 'different_lot', 'value': actual_tag}
        for i in prepared['draft']['input_plan']])
    assert result.status_code == 200, result.text
    plan = result.json()['input_plan']
    assert all(i['confirmed'] is False and i['lot_exception']['physical_tag'] == actual_tag for i in plan)
    assert main.lot_on_hand(db_cursor, ingredient['lot_id']) == 100
    assert sum(float(i['quantity_lb']) for i in plan) == pytest.approx(88.8)
    db_cursor.execute('SELECT count(*) AS n FROM transaction_lot_confirmations WHERE transaction_id=%s', (result.json()['transaction_id'],))
    assert db_cursor.fetchone()['n'] == 0


def test_ambiguous_suffix_gives_spanish_reason_and_full_code_recovery(client, db_cursor, items, actors):
    key = actors['floor']['key']; ingredient = items['ingredient']
    db_cursor.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id',
                      (ingredient['id'], 'ANOTHER-' + ingredient['lot_code'][-4:]))
    twin = db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO transactions(type) VALUES ('receive') RETURNING id")
    tid = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,1)',
                      (tid, ingredient['id'], twin))
    prepared = client.post('/make/prepare', json=body('make', items), headers=headers(key)).json()
    result = commit(client, prepared, key, lot_confirmations=[{'lot_id': ingredient['lot_id'], 'method': 'last4', 'value': ingredient['lot_code'][-4:]}])
    assert result.status_code == 422
    assert result.json()['detail']['error_code'] == 'AMBIGUOUS_SUFFIX'
    assert 'código completo' in result.json()['detail']['message_es']
    assert posted_count(db_cursor, prepared) == 0
    result = commit(client, prepared, key, lot_confirmations=[{'lot_id': ingredient['lot_id'], 'method': 'full_code', 'value': ingredient['lot_code']}])
    assert result.status_code == 200, result.text


@pytest.mark.parametrize('url', ['/sales/orders', '/sales/orders/999999', '/inventory/lookup?q=not-a-real-product', '/products/search?q=not-a-real-product'])
def test_safe_read_routes_retry_a_transient_disconnect_once(client, monkeypatch, url):
    original = main.get_transaction
    calls = []
    @contextmanager
    def interrupted():
        calls.append(1)
        if len(calls) == 1:
            raise psycopg2.OperationalError('server closed the connection unexpectedly')
        with original() as cur:
            yield cur
    monkeypatch.setattr(main, 'get_transaction', interrupted)
    result = client.get(url, headers=headers())
    assert result.status_code in (200, 404), result.text
    assert len(calls) == 2
