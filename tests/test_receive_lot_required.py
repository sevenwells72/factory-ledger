"""Supplier lot capture is required on every receive surface, with an explicit exception."""
import pytest

import lot_confirmation
from tests.test_actor_attribution import client  # noqa: F401
from tests.test_write_tickets import payload, headers, commit, prepare  # noqa: F401

pytestmark = pytest.mark.db

PATHS = ['/receive/prepare', '/receive', '/receive/preview', '/receive/commit']


def body_for(cur, payload, path, **changes):
    body = payload | changes
    if path == '/receive/prepare':
        return body
    cur.execute('SELECT name FROM products WHERE id=%s', (body.pop('product_id'),))
    body['product_name'] = cur.fetchone()['name']
    cur.execute('SELECT name FROM suppliers WHERE id=%s', (body.pop('supplier_id'),))
    body['shipper_name'] = cur.fetchone()['name']
    body['mode'] = 'commit'
    return body


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('code', [None, '', '   ', 'N/A', 'NA', 'UNKNOWN'])
def test_missing_supplier_lot_is_bilingual_422_before_any_write(client, db_cursor, payload, path, code):
    response = client.post(path, headers=headers(), json=body_for(
        db_cursor, payload, path, supplier_lot_code=code))
    assert response.status_code == 422, response.text
    detail = response.json()['detail']
    assert detail['error_code'] == 'SUPPLIER_LOT_REQUIRED'
    assert 'No lot on label' in detail['message']
    assert 'Sin lote en la etiqueta' in detail['message_es']
    db_cursor.execute('SELECT count(*) AS n FROM transaction_lines WHERE product_id=%s', (payload['product_id'],))
    assert db_cursor.fetchone()['n'] == 0
    db_cursor.execute('SELECT count(*) AS n FROM lots WHERE product_id=%s', (payload['product_id'],))
    assert db_cursor.fetchone()['n'] == 0
    db_cursor.execute("SELECT count(*) AS n FROM write_tickets WHERE payload->>'product_id'=%s", (str(payload['product_id']),))
    assert db_cursor.fetchone()['n'] == 0


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('flag', [False, 'true', 'false', 1, 0, None])
def test_only_explicit_boolean_true_exempts_blank_supplier_lot(client, db_cursor, payload, path, flag):
    response = client.post(path, headers=headers(), json=body_for(
        db_cursor, payload, path, supplier_lot_code='', no_supplier_lot=flag))
    assert response.status_code == 422, response.text


@pytest.mark.parametrize('path', ['/receive/prepare', '/receive', '/receive/commit'])
@pytest.mark.parametrize('code', ['', 'UNKNOWN'])
def test_no_label_flag_records_unidentified_lot_and_existing_deadline(client, db_cursor, payload, path, code):
    response = client.post(path, headers=headers(), json=body_for(
        db_cursor, payload, path, supplier_lot_code=code, no_supplier_lot=True))
    assert response.status_code == 200, response.text
    if path == '/receive/prepare':
        prepared = response.json()
        assert prepared['draft']['identity_status'] == 'unidentified'
        response = commit(client, prepared)
        assert response.status_code == 200, response.text
    result = response.json()
    assert result['identity_status'] == 'unidentified'
    assert result['exception_id']
    assert result['supplier_id'] == payload['supplier_id']
    assert result['supplier_lot_code'] == (code or 'N/A')
    db_cursor.execute('SELECT * FROM lots WHERE id=%s', (result['lot_id'],))
    lot = db_cursor.fetchone()
    assert lot['identity_status'] == 'unidentified'
    assert lot['supplier_id'] == payload['supplier_id']
    db_cursor.execute('SELECT created_at,supplier_id FROM transactions WHERE id=%s', (result['transaction_id'],))
    transaction = db_cursor.fetchone()
    due = lot_confirmation.identification_deadline(transaction['created_at'])
    assert transaction['supplier_id'] == payload['supplier_id']
    assert lot['identify_by'] == due.date()
    db_cursor.execute('SELECT kind,due_at,lot_id FROM exceptions WHERE id=%s', (result['exception_id'],))
    exception = db_cursor.fetchone()
    assert exception['kind'] == 'UNIDENTIFIED_LOT'
    assert exception['due_at'] == due
    assert exception['lot_id'] == lot['id']


@pytest.mark.parametrize('path', ['/receive/prepare', '/receive', '/receive/commit'])
def test_normal_receive_preserves_code_supplier_and_quantity(client, db_cursor, payload, path):
    response = client.post(path, headers=headers(), json=body_for(db_cursor, payload, path))
    assert response.status_code == 200, response.text
    if path == '/receive/prepare':
        response = commit(client, response.json())
        assert response.status_code == 200, response.text
    result = response.json()
    assert result['identity_status'] == 'identified'
    assert result['exception_id'] is None
    assert result['supplier_lot_code'] == payload['supplier_lot_code']
    assert result['supplier_id'] == payload['supplier_id']
    assert result['lot_code'] == payload['lot_code']
    assert result['total_lb'] == 50
    assert result['receipt_text'].splitlines()[:2] == [
        'CNS lot: ' + result['lot_code'], 'Write this on the pallet tag']
    assert result['receipt_text_es'].splitlines()[:2] == [
        'Lote CNS: ' + result['lot_code'], 'Escribe esto en la etiqueta del pallet']


@pytest.mark.parametrize('kind', ['finished', 'batch', 'packaging', 'consumable', 'service', 'has_bom'])
def test_non_purchased_ingredient_keeps_optional_supplier_lot(client, db_cursor, payload, kind):
    if kind in ('service', 'has_bom'):
        db_cursor.execute('UPDATE products SET ' + ('is_service' if kind == 'service' else 'has_bom') + '=true WHERE id=%s',
                          (payload['product_id'],))
    else:
        db_cursor.execute('UPDATE products SET type=%s WHERE id=%s', (kind, payload['product_id']))
    prepared = prepare(client, payload | {'supplier_lot_code': ''})
    assert prepared['can_commit']
    result = commit(client, prepared)
    assert result.status_code == 200, result.text
    assert 'identity_status' not in result.json()


@pytest.mark.parametrize('path', ['/receive/prepare', '/receive', '/receive/commit'])
def test_valid_commingled_entries_need_no_placeholder_top_level_code(client, db_cursor, payload, path):
    body = body_for(db_cursor, payload, path, supplier_lot_code='', supplier_lot_entries=[
        {'supplier_id': payload['supplier_id'], 'supplier_lot_code': 'LABEL-A', 'quantity_lb': 25},
        {'supplier_id': payload['supplier_id'], 'supplier_lot_code': 'LABEL-B', 'quantity_lb': 25}])
    result = client.post(path, headers=headers(), json=body)
    assert result.status_code == 200, result.text
    if path == '/receive/prepare':
        assert result.json()['draft']['identity_status'] == 'identified'
        result = commit(client, result.json())
        assert result.status_code == 200, result.text
    assert result.json()['identity_status'] == 'identified'
    assert result.json()['supplier_lot_entries_created'] == 2
    db_cursor.execute('SELECT supplier_lot_code,supplier_id FROM lot_supplier_codes WHERE lot_id=%s',
                      (result.json()['lot_id'],))
    assert {(row['supplier_lot_code'], row['supplier_id']) for row in db_cursor.fetchall()} == {
        ('LABEL-A', payload['supplier_id']), ('LABEL-B', payload['supplier_id'])}


def test_existing_ticket_with_no_label_without_explicit_flag_cannot_commit(client, db_cursor, payload):
    import write_tickets
    from psycopg2.extras import Json
    prepared = prepare(client, payload | {'supplier_lot_code': '', 'no_supplier_lot': True})
    db_cursor.execute('SELECT payload FROM write_tickets WHERE id=%s', (prepared['ticket_id'],))
    old_payload = dict(db_cursor.fetchone()['payload'])
    old_payload.pop('no_supplier_lot')
    payload_hash = write_tickets.canonical_hash(old_payload)
    db_cursor.execute('UPDATE write_tickets SET payload=%s,payload_hash=%s WHERE id=%s',
                      (Json(old_payload), payload_hash, prepared['ticket_id']))
    response = commit(client, prepared | {'payload_hash': payload_hash})
    assert response.status_code == 409, response.text
    assert response.json()['detail']['blockers'][0]['code'] == 'SUPPLIER_LOT_REQUIRED'
    db_cursor.execute('SELECT count(*) AS n FROM transaction_lines WHERE product_id=%s', (payload['product_id'],))
    assert db_cursor.fetchone()['n'] == 0


@pytest.mark.parametrize('path', ['/receive', '/receive/preview', '/receive/commit'])
@pytest.mark.parametrize('name', ['', '   '])
def test_legacy_supplier_name_stays_required(client, db_cursor, payload, path, name):
    body = body_for(db_cursor, payload, path)
    body['shipper_name'] = name
    response = client.post(path, headers=headers(), json=body)
    assert response.status_code == 422, response.text
    assert response.json()['detail']['error_code'] == 'SUPPLIER_REQUIRED'
