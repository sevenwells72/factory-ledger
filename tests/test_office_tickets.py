"""Office ticket additions: atomic document approval, live FL choices and attribution."""
from datetime import timedelta
from uuid import uuid4
import pytest
import main
from tests.test_order_tickets import client, actors, catalog, create_order, prepare, prepare_raw, office
from tests.test_write_tickets import headers, commit
from tests.test_sales_order_extract import _insert_document

pytestmark = pytest.mark.db


def document_body(cur, catalog, **updates):
    doc = _insert_document(cur, path=uuid4().hex, kind='sales', status='extracted', sha=uuid4().hex)
    body = {'document_id': doc['id'], 'customer_id': catalog['customer'], 'customer_po': '000'+catalog['suffix'],
            'lines': [{'product_id': catalog['products']['food'], 'quantity_lb': 25,
                       'customer_description': 'Reviewed customer granola', 'save_alias': True}]}
    body.update(updates)
    return body


def test_so_document_dry_run_atomic_commit_and_same_receipt_replay(client, db_cursor, catalog, actors):
    body = document_body(db_cursor, catalog)
    key = office(actors)
    draft = prepare(client, '/sales/orders/extract/approve/prepare', body, key)
    assert draft['can_commit'], draft
    db_cursor.execute('SELECT status FROM purchase_documents WHERE id=%s', (body['document_id'],))
    assert db_cursor.fetchone()['status'] == 'extracted'
    db_cursor.execute('SELECT count(*) AS n FROM sales_orders WHERE source_document_id=%s', (body['document_id'],))
    assert db_cursor.fetchone()['n'] == 0
    result = commit(client, draft, key)
    assert result.status_code == 200, result.text
    saved = result.json()
    assert saved['receipt_number'].startswith('ORD-')
    assert saved['customer_po'] == body['customer_po']
    assert commit(client, draft, key).json() == {**saved, 'replayed': True}
    db_cursor.execute('SELECT status FROM purchase_documents WHERE id=%s', (body['document_id'],))
    assert db_cursor.fetchone()['status'] == 'approved'
    db_cursor.execute('SELECT ticket_id FROM sales_orders WHERE id=%s', (saved['order_id'],))
    row=db_cursor.fetchone()
    assert row['ticket_id']==draft['ticket_id']
    assert saved['entry_timing']['entered_by']['id']==actors['office']['id']
    receipt=client.get('/receipts/'+saved['receipt_number'],headers=headers(key)).json()
    assert receipt['order']['id']==saved['order_id']


def test_so_document_duplicate_warning_requires_explicit_override(client, db_cursor, catalog, actors):
    key=office(actors)
    original=create_order(client,catalog,key)
    body=document_body(db_cursor,catalog,customer_po='PO-'+catalog['suffix'])
    draft=prepare(client,'/sales/orders/extract/approve/prepare',body,key)
    assert draft['can_commit'],draft
    assert any(w['code']=='DUPLICATE_CUSTOMER_PO' for w in draft['warnings'])
    assert commit(client,draft,key).status_code==409
    saved=commit(client,draft,key,acknowledged_warnings=['DUPLICATE_CUSTOMER_PO']).json()
    assert saved['receipt_number']!=original['receipt_number']
    assert saved['duplicate_overridden']


@pytest.mark.parametrize('extra', [{'created_by':'owner'}, {'customer_name':'guessed'}, {'force':True}])
def test_so_document_rejects_self_reported_identity_and_unreviewed_fields(client,db_cursor,catalog,actors,extra):
    body=document_body(db_cursor,catalog)
    assert prepare_raw(client,'/sales/orders/extract/approve/prepare',{**body,**extra},office(actors)).status_code==422


def test_so_document_stale_approval_does_not_create_second_order(client,db_cursor,catalog,actors):
    body=document_body(db_cursor,catalog);key=office(actors)
    draft=prepare(client,'/sales/orders/extract/approve/prepare',body,key)
    db_cursor.execute("UPDATE purchase_documents SET status='approved' WHERE id=%s",(body['document_id'],))
    response=commit(client,draft,key)
    assert response.status_code==409
    db_cursor.execute('SELECT count(*) AS n FROM sales_orders WHERE source_document_id=%s',(body['document_id'],))
    assert db_cursor.fetchone()['n']==0


@pytest.mark.parametrize('role',['office','floor','owner'])
def test_document_read_and_prepare_use_fl_permissions(client,db_cursor,catalog,actors,role):
    body=document_body(db_cursor,catalog);key=actors[role]['key']
    expected=403 if role=='floor' else 200
    assert client.get('/purchase-documents/'+str(body['document_id']),headers=headers(key)).status_code==expected
    assert prepare_raw(client,'/sales/orders/extract/approve/prepare',body,key).status_code==expected


def test_order_choices_obey_owner_and_immutable_cutoff(client,catalog,actors,monkeypatch):
    order=create_order(client,catalog,office(actors));path=f"/sales/orders/{order['order_id']}/office-options"
    tomorrow=(main.get_plant_now().date()+timedelta(days=1)).isoformat()
    monkeypatch.setenv('ORDER_LEGACY_CLOSE_CUTOFF',tomorrow)
    for role in ['owner','office','floor']:
        data=client.get(path,headers=headers(actors[role]['key'])).json()
        assert ('shipped_not_recorded' in data['close_reasons'])==(role=='owner')
    monkeypatch.setenv('ORDER_LEGACY_CLOSE_CUTOFF',main.get_plant_now().date().isoformat())
    assert 'shipped_not_recorded' not in client.get(path,headers=headers(actors['owner']['key'])).json()['close_reasons']
