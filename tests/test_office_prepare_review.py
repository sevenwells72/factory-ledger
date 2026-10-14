"""PR #115: document approval refusal and localized warning failures."""
from uuid import uuid4

import pytest

from tests.test_order_tickets import (
    actors, catalog, client, create_body, create_order, office, prepare, prepare_raw,
)
from tests.test_sales_order_extract import _insert_document
from tests.test_write_tickets import commit, error, ticket_row

pytestmark = pytest.mark.db


def approval_body(cur, catalog, kind, status='extracted'):
    doc = _insert_document(cur, path=uuid4().hex, kind=kind, status=status, sha=uuid4().hex)
    body = {'document_id': doc['id']}
    if kind == 'sales':
        path = '/sales/orders/extract/approve/prepare'
        body.update(customer_id=catalog['customer'], customer_po='PO-' + catalog['suffix'],
                    lines=[{'product_id': catalog['products']['food'], 'quantity_lb': 25,
                            'customer_description': 'Reviewed customer granola'}])
    else:
        cur.execute('INSERT INTO suppliers(name, active) VALUES (%s, true) RETURNING id',
                    ('Document supplier ' + catalog['suffix'],))
        path = '/expected-receipts/extract/approve/prepare'
        body.update(supplier_id=cur.fetchone()['id'], reference_number='PO-' + catalog['suffix'],
                    lines=[{'product_id': catalog['products']['food'], 'expected_qty_lb': 25,
                            'vendor_description': 'Reviewed supplier granola'}])
    return path, body


@pytest.mark.parametrize('kind', ['purchase', 'sales'])
def test_approved_document_is_refused_at_prepare_without_a_ticket(client, db_cursor, catalog, actors, kind):
    path, body = approval_body(db_cursor, catalog, kind, status='approved')
    db_cursor.execute('SELECT count(*) AS n FROM write_tickets')
    before = db_cursor.fetchone()['n']

    response = prepare_raw(client, path, body, office(actors))

    error(response, 409, 'DOCUMENT_ALREADY_APPROVED')
    detail = response.json()['detail']
    assert 'already approved' in detail['message']
    assert detail['message_es'] == f"El documento #{body['document_id']} ya fue aprobado."
    db_cursor.execute('SELECT count(*) AS n FROM write_tickets')
    assert db_cursor.fetchone()['n'] == before
    db_cursor.execute('SELECT status FROM purchase_documents WHERE id=%s', (body['document_id'],))
    assert db_cursor.fetchone()['status'] == 'approved'


@pytest.mark.parametrize('kind', ['purchase', 'sales'])
def test_approval_after_prepare_still_refuses_record(client, db_cursor, catalog, actors, kind):
    path, body = approval_body(db_cursor, catalog, kind)
    key = office(actors)
    draft = prepare(client, path, body, key)
    assert draft['can_commit']
    db_cursor.execute("UPDATE purchase_documents SET status='approved' WHERE id=%s", (body['document_id'],))

    response = commit(client, draft, key)

    error(response, 409, 'TICKET_STALE')
    assert any(blocker['code'] == 'DOCUMENT_ALREADY_APPROVED'
               for blocker in response.json()['detail']['blockers'])
    table = 'sales_orders' if kind == 'sales' else 'expected_receipts'
    db_cursor.execute(f'SELECT count(*) AS n FROM {table} WHERE source_document_id=%s', (body['document_id'],))
    assert db_cursor.fetchone()['n'] == 0


def test_unacknowledged_warning_has_a_fully_spanish_failure_message(client, db_cursor, catalog, actors):
    key = office(actors)
    create_order(client, catalog, key)
    draft = prepare(client, '/sales/orders/prepare', create_body(catalog, cases=4), key)

    response = commit(client, draft, key)

    error(response, 409, 'WARNING_NOT_ACKNOWLEDGED')
    detail = response.json()['detail']
    assert detail['message_es'] == 'Confirme las advertencias del borrador antes de registrar.'
    assert detail['missing'] == ['DUPLICATE_CUSTOMER_PO']
    assert ticket_row(db_cursor, draft)['status'] == 'prepared'
    assert commit(client, draft, key, acknowledged_warnings=detail['missing']).status_code == 200
