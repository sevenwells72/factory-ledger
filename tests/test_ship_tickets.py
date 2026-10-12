"""A6 real HTTP/PostgreSQL policy and lifecycle coverage."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime,timedelta
from hashlib import sha256
from pathlib import Path
from threading import Barrier
from urllib.parse import urlsplit,parse_qs
from uuid import uuid4

from fastapi.testclient import TestClient
from fastapi import HTTPException
import psycopg2
from psycopg2.extras import RealDictCursor
import pytest

import main
import attachments
import ship_tickets
from scripts.exceptions_sweep import sweep
from tests.test_actor_attribution import client,actors
from tests.test_write_tickets import headers,commit,ticket_row,error,isolated_database
from tests.test_order_tickets import seed,create_body
from tests.test_write_tickets_part2 import items

pytestmark=pytest.mark.db
PNG=b'\x89PNG\r\n\x1a\nA6-real-file'

@pytest.fixture(autouse=True)
def storage(tmp_path,monkeypatch):
    monkeypatch.setenv('EVIDENCE_STORAGE_DIR',str(tmp_path/'evidence'))

def seed_shipping(cur):
    data=seed(cur);pid=data['products']['food'];lotcode='A6-'+data['suffix'].upper()
    cur.execute('INSERT INTO lots(product_id,lot_code) VALUES(%s,%s) RETURNING id',(pid,lotcode));lid=cur.fetchone()['id']
    cur.execute("INSERT INTO transactions(type) VALUES('receive') RETURNING id");txn=cur.fetchone()['id']
    cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES(%s,%s,%s,2000)',(txn,pid,lid))
    data.update(lot_id=lid,lot_code=lotcode)
    return data

@pytest.fixture
def shipping(db_cursor): return seed_shipping(db_cursor)

def upload(client,key,kind='bol',ticket_id=None):
    data={'kind':kind}
    if ticket_id: data['ticket_id']=str(ticket_id)
    r=client.post('/attachments',data=data,files={'file':('proof.png',PNG,'image/png')},headers=headers(key))
    assert r.status_code==200,r.text
    return r.json()

def new_order(client,data,key):
    r=client.post('/sales/orders/prepare',json=create_body(data),headers=headers(key));assert r.status_code==200,r.text
    r=commit(client,r.json(),key);assert r.status_code==200,r.text
    return r.json()['order_id']

def ship_body(data,**extra):
    return {'bol_reference':'BOL-A6','lot_confirmations':[{'lot_id':data['lot_id'],'method':'full_code','value':data['lot_code']}],**extra}

def prepared(client,data,key,action='ship_order',**extra):
    b=ship_body(data,**extra)
    if action=='ship_order':
        oid=new_order(client,data,key);path=f'/sales/orders/{oid}/ship/prepare';b['ship_all']=True
    else:
        b.update(customer_id=data['customer'],order_reference='A6-SO')
        if action=='ship_standalone': path='/ship/prepare';b.update(product_id=data['products']['food'],lot_id=data['lot_id'],quantity_lb=50)
        else: path='/ship/bulk/prepare';b['lines']=[{'product_id':data['products']['food'],'lot_id':data['lot_id'],'quantity_lb':20},
                                                {'product_id':data['products']['food'],'lot_id':data['lot_id'],'quantity_lb':30}]
    r=client.post(path,json=b,headers=headers(key));assert r.status_code==200,r.text
    return r.json()

@pytest.mark.parametrize('action',['ship_order','ship_standalone','ship_bulk'])
def test_shipping_posts_atomic_receipt_bol_proof_and_replay(client,db_cursor,shipping,actors,action):
    key=actors['office']['key']; photo=upload(client,key)
    p=prepared(client,shipping,key,action,attachment_ids=[photo['id']])
    assert p['can_commit'] and p['draft']['photo']=='attached'
    db_cursor.execute('SELECT count(*) AS n FROM shipments WHERE ticket_id=%s',(p['ticket_id'],));assert db_cursor.fetchone()['n']==0
    r=commit(client,p,key);assert r.status_code==200,r.text
    result=r.json(); assert result['receipt_number'].startswith('SHP-') and result['proof_status']=='complete'
    replay=commit(client,p,key);assert replay.json()==result|{'replayed':True}
    db_cursor.execute('SELECT * FROM shipments WHERE ticket_id=%s',(p['ticket_id'],));s=db_cursor.fetchone()
    assert s['bol_reference']=='BOL-A6' and s['entered_by_actor_id']==actors['office']['id']
    assert s['proof_due_at'].astimezone(main.PLANT_TIMEZONE).strftime('%H:%M')=='23:59'
    db_cursor.execute('SELECT ticket_id,receipt_number,bol_reference,entered_by_actor_id FROM transactions WHERE ticket_id=%s',(p['ticket_id'],));txns=db_cursor.fetchall()
    assert len(txns)==(2 if action=='ship_bulk' else 1)
    assert all(t['receipt_number']==result['receipt_number'] and t['bol_reference']=='BOL-A6' and t['entered_by_actor_id']==actors['office']['id'] for t in txns)
    assert main.lot_on_hand(db_cursor,shipping['lot_id'])==1950
    detail=client.get('/receipts/'+result['receipt_number'],headers=headers(key)).json()
    assert detail['shipment']['id']==s['id'] and len(detail['transactions'])==len(txns)
    assert all(t['lot_confirmations'] for t in detail['transactions'])

@pytest.mark.parametrize('action',['ship_order','ship_standalone','ship_bulk'])
def test_ticket_identity_hash_expiry_and_supersession(client,db_cursor,shipping,actors,action):
    key=actors['office']['key'];p=prepared(client,shipping,key,action)
    error(commit(client,p,actors['owner']['key']),403,'TICKET_WRONG_USER')
    error(commit(client,p,key,payload_hash='bad'),409,'TICKET_PAYLOAD_MISMATCH')
    db_cursor.execute("UPDATE write_tickets SET expires_at=clock_timestamp()-interval '1 second' WHERE id=%s",(p['ticket_id'],))
    error(commit(client,p,key),409,'TICKET_EXPIRED');assert ticket_row(db_cursor,p)['status']=='expired'

@pytest.mark.parametrize('action,role,allowed',[('ship_order',r,True) for r in ('owner','floor','office')]+[('ship_bulk',r,True) for r in ('owner','floor','office')]+[('ship_standalone',r,r!='floor') for r in ('owner','floor','office')])
def test_permissions_at_prepare_and_commit(client,db_cursor,shipping,actors,action,role,allowed):
    # Order creation itself requires office; floor shipping uses office-created order.
    office=actors['office']['key'];b=ship_body(shipping)
    if action=='ship_order':
        oid=new_order(client,shipping,office);path=f'/sales/orders/{oid}/ship/prepare';b['ship_all']=True
    else:
        path='/ship/bulk/prepare' if action=='ship_bulk' else '/ship/prepare';b.update(customer_id=shipping['customer'],order_reference='P-A6')
        if action=='ship_bulk': b['lines']=[{'product_id':shipping['products']['food'],'quantity_lb':5}]
        else:b.update(product_id=shipping['products']['food'],quantity_lb=5)
    r=client.post(path,json=b,headers=headers(actors[role]['key']))
    if not allowed:error(r,403,'ROLE_NOT_ALLOWED');return
    assert r.status_code==200,r.text;p=r.json()
    # Shared dashboard loses every new ship prepare.
    assert client.post(path,json=b,headers=headers(main.DASHBOARD_API_KEY)).status_code==403
    db_cursor.execute("UPDATE actors SET role='office' WHERE id=%s",(actors[role]['id'],))
    if action=='ship_standalone':db_cursor.execute("UPDATE actors SET role='floor' WHERE id=%s",(actors[role]['id'],))
    main._reset_actor_cache()
    if action=='ship_standalone':error(commit(client,p,actors[role]['key']),403,'ROLE_NOT_ALLOWED')
    else:
        db_cursor.execute('UPDATE actors SET active=false WHERE id=%s',(actors[role]['id'],));main._reset_actor_cache()
        assert commit(client,p,actors[role]['key']).status_code==403

def test_bol_and_lot_confirmation_are_hard_gates(client,db_cursor,shipping,actors):
    key=actors['office']['key'];p=prepared(client,shipping,key,bol_reference='  ',lot_confirmations=[])
    assert {b['code'] for b in p['blockers']}=={'BOL_REQUIRED','LOT_NOT_CONFIRMED'}
    error(commit(client,p,key,lot_confirmations=ship_body(shipping)['lot_confirmations']),422,'BOL_REQUIRED')
    oid=p['draft']['order_id']
    r=client.post(f'/sales/orders/{oid}/ship/prepare',json={'ship_all':True,'bol_reference':'B'},headers=headers(key));p=r.json()
    error(commit(client,p,key),422,'LOT_NOT_CONFIRMED')
    good=commit(client,p,key,lot_confirmations=ship_body(shipping)['lot_confirmations']);assert good.status_code==200,good.text
    assert good.json()['proof_status']=='photo_pending'

@pytest.mark.parametrize('role',['owner','floor','office'])
def test_same_day_strict_for_all_roles_and_commit_midnight(client,db_cursor,shipping,actors,monkeypatch,role):
    key=actors[role]['key'];oid=new_order(client,shipping,actors['office']['key']);path=f'/sales/orders/{oid}/ship/prepare'
    now=datetime(2026,11,1,23,58,tzinfo=main.PLANT_TIMEZONE);monkeypatch.setattr(main,'get_plant_now',lambda:now)
    b=ship_body(shipping,ship_all=True,happened_at=now.isoformat())
    error(client.post(path,json=b|{'happened_at':(now-timedelta(days=1)).isoformat()},headers=headers(key)),422,'SHIP_NOT_SAME_DAY')
    r=client.post(path,json=b,headers=headers(key));assert r.status_code==200,r.text
    now=datetime(2026,11,2,0,0,tzinfo=main.PLANT_TIMEZONE)
    error(commit(client,r.json(),key,acknowledged_warnings=['SHIP_NOT_SAME_DAY']),422,'SHIP_NOT_SAME_DAY')
    db_cursor.execute('SELECT count(*) AS n FROM shipments WHERE ticket_id=%s',(r.json()['ticket_id'],));assert db_cursor.fetchone()['n']==0

def test_photo_deadline_one_exception_late_photo_and_signed_slip(client,db_cursor,shipping,actors):
    key=actors['office']['key'];p=prepared(client,shipping,key);r=commit(client,p,key);assert r.status_code==200,r.text
    result=r.json();sid=result['shipment_id']
    sweep(db_cursor);db_cursor.execute("SELECT count(*) AS n FROM exceptions WHERE kind='BOL_PHOTO_MISSING' AND shipment_id=%s",(sid,));assert db_cursor.fetchone()['n']==0
    slip=upload(client,key,'signed_packing_slip');a=client.post(f'/shipments/{sid}/attachments',json={'attachment_id':slip['id']},headers=headers(key));assert a.status_code==200,a.text
    db_cursor.execute("UPDATE shipments SET proof_due_at=clock_timestamp()-interval '1 second' WHERE id=%s",(sid,))
    sweep(db_cursor);sweep(db_cursor)
    db_cursor.execute("SELECT * FROM exceptions WHERE kind='BOL_PHOTO_MISSING' AND shipment_id=%s",(sid,));exc=db_cursor.fetchall();assert len(exc)==1 and exc[0]['status']=='escalated'
    bare=client.post(f"/exceptions/{exc[0]['id']}/resolve",json={'resolution_kind':'photo_attached','note':'bare','attachment_ref':'arbitrary.jpg'},headers=headers(actors['owner']['key']));error(bare,422,'ATTACHMENT_NOT_FOUND')
    photo=upload(client,key);a=client.post(f'/shipments/{sid}/attachments',json={'attachment_id':photo['id']},headers=headers(key));assert a.status_code==200,a.text
    sweep(db_cursor)
    db_cursor.execute("SELECT status,resolution_kind FROM exceptions WHERE id=%s",(exc[0]['id'],));assert dict(db_cursor.fetchone())=={'status':'resolved','resolution_kind':'photo_attached'}
    db_cursor.execute('SELECT proof_status FROM shipments WHERE id=%s',(sid,));assert db_cursor.fetchone()['proof_status']=='complete'

def test_signed_packing_slip_link_bound_to_receipt_order_and_expiry(client,db_cursor,shipping,actors):
    key=actors['office']['key'];r=commit(client,prepared(client,shipping,key),key);assert r.status_code==200,r.text
    url=r.json()['packing_slip_url'];good=client.get(url);assert good.status_code==200 and good.content.startswith(b'%PDF'),good.text
    assert client.get(url.replace('sig=','sig=x')).status_code==401
    query=parse_qs(urlsplit(url).query);expired=ship_tickets._sign(r.json().get('order_id',int(url.split('/')[3])),query['receipt'][0],0)
    assert client.get(url.split('?')[0]+f"?receipt={query['receipt'][0]}&expires=0&sig={expired}").status_code==401

def test_standalone_ids_only_and_open_order_acknowledgement(client,db_cursor,shipping,actors):
    key=actors['office']['key'];new_order(client,shipping,key)
    b=ship_body(shipping,customer_id=shipping['customer'],product_id=shipping['products']['food'],quantity_lb=20,order_reference='STANDALONE')
    error(client.post('/ship/prepare',json=b|{'customer_name':'x'},headers=headers(key)),422,'IDS_REQUIRED')
    r=client.post('/ship/prepare',json=b,headers=headers(key));assert r.status_code==200,r.text;p=r.json()
    assert [w['code'] for w in p['warnings']]==['OPEN_SALES_ORDER_EXISTS']
    error(commit(client,p,key),409,'WARNING_NOT_ACKNOWLEDGED')
    assert commit(client,p,key,acknowledged_warnings=['OPEN_SALES_ORDER_EXISTS']).status_code==200

@pytest.mark.parametrize('action',['adjust','found'])
@pytest.mark.parametrize('bad',['unknown','other_actor','other_ticket'])
def test_p113_adjust_found_and_held_approval(client,db_cursor,shipping,actors,action,bad):
    key=actors['floor']['key'];body={'reason_code':'physical_count'}
    if action=='adjust':path='/adjust/prepare';body.update(lot_id=shipping['lot_id'],delta_lb=600)
    else:path='/inventory/found/prepare';body.update(product_id=shipping['products']['food'],quantity=600)
    r=client.post(path,json=body,headers=headers(key));assert r.status_code==200,r.text;p=r.json()
    held=commit(client,p,key);assert held.status_code==202,held.text
    ref='not-a-file';code='ATTACHMENT_NOT_FOUND';status=422
    if bad!='unknown':
        evidence=upload(client,actors['owner']['key'] if bad=='other_actor' else key,'correction_photo')
        ref=evidence['attachment_ref']
        if bad=='other_actor':code='ATTACHMENT_WRONG_ACTOR';status=403
        else:
            other=client.post(path,json=body|({'delta_lb':601} if action=='adjust' else {'quantity':601}),headers=headers(key)).json()
            db_cursor.execute('UPDATE attachments SET ticket_id=%s WHERE id=%s',(other['ticket_id'],evidence['id']));code='ATTACHMENT_WRONG_CONTEXT'
    error(commit(client,p,key,attachment_ref=ref),status,code)
    exc=held.json()['exception_id']
    error(client.post(f'/exceptions/{exc}/approve',json={'attachment_ref':ref},headers=headers(actors['owner']['key'])),status,code)
    assert ticket_row(db_cursor,p)['status']=='awaiting_approval'
    # Owner's explicit no-photo approval remains allowed per decision 7.
    ok=client.post(f'/exceptions/{exc}/approve',json={},headers=headers(actors['owner']['key']));assert ok.status_code==200,ok.text

def test_attachment_upload_download_authorization_and_immutable_provenance(client,db_cursor,actors):
    floor=actors['floor']['key'];a=upload(client,floor,'correction_photo')
    error(client.post('/attachments',data={'kind':'bol'},files={'file':('b.png',PNG,'image/png')},headers=headers()),403,'ROLE_NOT_ALLOWED')
    assert client.get(f"/attachments/{a['id']}/file").status_code==401
    assert client.get(f"/attachments/{a['id']}/file",headers=headers(floor)).content==PNG
    assert client.get(f"/attachments/{a['id']}/file",headers=headers(actors['office']['key'])).status_code==200
    db_cursor.execute('SAVEPOINT immutable')
    with pytest.raises(psycopg2.Error):db_cursor.execute('UPDATE attachments SET uploaded_by_actor_id=%s WHERE id=%s',(actors['owner']['id'],a['id']))
    db_cursor.execute('ROLLBACK TO SAVEPOINT immutable')

def test_a7_close_gate_needs_effective_shipment(client,db_cursor,shipping,actors):
    key=actors['office']['key'];oid=new_order(client,shipping,key)
    for reason in ('short_closed','shipped_recorded','shipped_not_recorded'):
        who=actors['owner']['key'] if reason=='shipped_not_recorded' else key
        p=client.post(f'/sales/orders/{oid}/close/prepare',json={'reason':reason},headers=headers(who)).json()
        assert not p['can_commit'] and p['blockers'][0]['code']=='SHIPMENT_REQUIRED'
    p=client.post(f'/sales/orders/{oid}/ship/prepare',json=ship_body(shipping,ship_all=True),headers=headers(key)).json();r=commit(client,p,key);assert r.status_code==200,r.text
    p=client.post(f'/sales/orders/{oid}/close/prepare',json={'reason':'shipped_recorded'},headers=headers(key)).json();assert p['can_commit']
    assert commit(client,p,key).status_code==200

@pytest.mark.parametrize('action',['ship_order','ship_standalone','ship_bulk'])
def test_concurrent_double_commit_records_one_shipment(isolated_database,monkeypatch,action):
    key='a6-'+uuid4().hex
    with psycopg2.connect(isolated_database) as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
        data=seed_shipping(cur);cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES(%s,'office',%s,true)",('A6 race '+key,sha256(key.encode()).hexdigest()))
    barrier=Barrier(2);racing=False
    @contextmanager
    def connection():
        conn=psycopg2.connect(isolated_database)
        try:
            if racing: barrier.wait(timeout=20)
            with conn:yield conn
        finally:conn.close()
    monkeypatch.setattr(main,'get_db_connection',connection);main._reset_actor_cache()
    with TestClient(main.app) as http:
        p=prepared(http,data,key,action);racing=True
        with ThreadPoolExecutor(max_workers=2) as ex:results=list(ex.map(lambda _:commit(http,p,key),range(2)))
        racing=False
    assert [r.status_code for r in results]==[200,200],[r.text for r in results]
    assert len({r.json()['receipt_number'] for r in results})==1
    assert sorted(r.json()['replayed'] for r in results)==[False,True]
    with psycopg2.connect(isolated_database) as conn,conn.cursor() as cur:
        cur.execute('SELECT count(*) FROM shipments WHERE ticket_id=%s',(p['ticket_id'],));assert cur.fetchone()[0]==1

def test_migration_rerunnable_without_rewriting_marker(isolated_database):
    text=(Path(__file__).resolve().parents[1]/'migrations/074_ship_gate_attachments.sql').read_text()
    with psycopg2.connect(isolated_database) as conn,conn.cursor() as cur:
        cur.execute("SELECT applied_at FROM migration_markers WHERE name='074_ship_gate_attachments'");before=cur.fetchone()
        cur.execute(text);cur.execute(text)
        cur.execute("SELECT applied_at FROM migration_markers WHERE name='074_ship_gate_attachments'");assert cur.fetchone()==before

@pytest.mark.parametrize('bad',['unknown','other_actor','other_ticket'])
def test_p113_counted_resolution_requires_resolvers_real_context(client,db_cursor,actors,items,bad):
    from tests.test_exceptions_enforcement import open_shortage
    key=actors['floor']['key'];exc,_=open_shortage(client,db_cursor,items,actors)
    ref='missing-file';status=422;code='ATTACHMENT_NOT_FOUND'
    if bad!='unknown':
        a=upload(client,actors['owner']['key'] if bad=='other_actor' else key,'correction_photo');ref=a['attachment_ref']
        if bad=='other_actor':status=403;code='ATTACHMENT_WRONG_ACTOR'
        else:
            p=client.post('/adjust/prepare',json={'lot_id':items['ingredient']['lot_id'],'delta_lb':-1,'reason_code':'physical_count'},headers=headers(key)).json()
            db_cursor.execute('UPDATE attachments SET ticket_id=%s WHERE id=%s',(p['ticket_id'],a['id']));code='ATTACHMENT_WRONG_CONTEXT'
    body={'resolution_kind':'counted','counted_lb':600,'note':'Counted pallet','attachment_ref':ref}
    error(client.post(f'/exceptions/{exc}/resolve',json=body,headers=headers(key)),status,code)
    assert main.lot_on_hand(db_cursor,items['ingredient']['lot_id'])==-6
    a=upload(client,key,'correction_photo');body.pop('attachment_ref');body['attachment_id']=a['id']
    r=client.post(f'/exceptions/{exc}/resolve',json=body,headers=headers(key));assert r.status_code==200,r.text
    assert main.lot_on_hand(db_cursor,items['ingredient']['lot_id'])==600

@pytest.mark.parametrize('action',['adjust','found'])
def test_correction_upload_id_in_prepare_posts_with_verified_transaction_link(client,db_cursor,shipping,actors,action):
    key=actors['floor']['key'];a=upload(client,key,'correction_photo')
    b={'attachment_id':a['id'],'reason_code':'physical_count'}
    if action=='adjust':path='/adjust/prepare';b.update(lot_id=shipping['lot_id'],delta_lb=600)
    else:path='/inventory/found/prepare';b.update(product_id=shipping['products']['food'],quantity=600)
    p=client.post(path,json=b,headers=headers(key));assert p.status_code==200,p.text
    r=commit(client,p.json(),key);assert r.status_code==200,r.text
    db_cursor.execute('SELECT ticket_id,transaction_id FROM attachments WHERE id=%s',(a['id'],));row=db_cursor.fetchone()
    assert row['ticket_id']==p.json()['ticket_id'] and row['transaction_id']==r.json()['transaction_id']

@pytest.mark.parametrize('action',['ship_order','ship_standalone','ship_bulk'])
def test_a1_ship_supersession_and_atomic_rollback_on_proof_failure(client,db_cursor,shipping,actors,monkeypatch,action):
    key=actors['office']['key'];p=prepared(client,shipping,key,action)
    payload=ticket_row(db_cursor,p)['payload'];path='/ship/bulk/prepare' if action=='ship_bulk' else '/ship/prepare'
    if action=='ship_order':path=f"/sales/orders/{payload.pop('order_id')}/ship/prepare"
    elif action=='ship_standalone':payload.update(payload.pop('lines')[0])
    r=client.post(path,json=payload,headers=headers(key));assert r.status_code==200,r.text;second=r.json()
    assert ticket_row(db_cursor,p)['status']=='superseded'
    error(commit(client,p,key),409,'TICKET_NOT_COMMITTABLE')
    a=upload(client,key)
    def fail_late(*args,**kwargs):raise HTTPException(503,{'error_code':'TEST_STORAGE_FAILURE','message':'injected'})
    monkeypatch.setattr(attachments,'attach_shipment',fail_late)
    assert commit(client,second,key,attachment_ids=[a['id']]).status_code==503
    assert ticket_row(db_cursor,second)['status']=='prepared' and main.lot_on_hand(db_cursor,shipping['lot_id'])==2000
    db_cursor.execute('SELECT count(*) AS n FROM shipments WHERE ticket_id=%s',(second['ticket_id'],));assert db_cursor.fetchone()['n']==0
    db_cursor.execute('SELECT count(*) AS n FROM transactions WHERE ticket_id=%s',(second['ticket_id'],));assert db_cursor.fetchone()['n']==0

def test_shipping_revalidates_cancellation_and_remaining_quantity(client,db_cursor,shipping,actors):
    key=actors['office']['key'];p=prepared(client,shipping,key);oid=p['draft']['order_id']
    db_cursor.execute("UPDATE sales_order_lines SET line_status='cancelled' WHERE sales_order_id=%s AND product_id=%s",(oid,shipping['products']['food']))
    r=commit(client,p,key);error(r,409,'TICKET_STALE');assert r.json()['detail']['blockers'][0]['code']=='LINE_CANCELLED'
    db_cursor.execute("UPDATE sales_order_lines SET line_status='pending',quantity_lb=1 WHERE sales_order_id=%s AND product_id=%s",(oid,shipping['products']['food']))
    r=client.post(f'/sales/orders/{oid}/ship/prepare',json=ship_body(shipping,lines=[{'line_id':p['draft']['lines'][0]['line_id'],'quantity_lb':2}]),headers=headers(key))
    error(r,422,'QTY_EXCEEDS_REMAINING')

def test_future_time_and_new_york_timezone_are_enforced(client,shipping,actors):
    key=actors['office']['key'];oid=new_order(client,shipping,key)
    b=ship_body(shipping,ship_all=True,happened_at=(main.get_plant_now()+timedelta(minutes=6)).isoformat())
    error(client.post(f'/sales/orders/{oid}/ship/prepare',json=b,headers=headers(key)),400,'OCCURRED_AT_IN_FUTURE')

def test_voided_shipment_cannot_satisfy_order_close(client,db_cursor,shipping,actors):
    key=actors['office']['key'];p=prepared(client,shipping,key);r=commit(client,p,key);assert r.status_code==200,r.text
    txn=r.json()['lines_shipped'][0]['transaction_id'];oid=p['draft']['order_id']
    v=client.post(f'/void/{txn}',json={'reason':'shipping cancelled'},headers=headers());assert v.status_code==200,v.text
    p=client.post(f'/sales/orders/{oid}/close/prepare',json={'reason':'short_closed'},headers=headers(key)).json()
    assert not p['can_commit'] and p['blockers'][0]['code']=='SHIPMENT_REQUIRED'

def test_bol_pdf_and_truck_photo_do_not_satisfy_bol_photo_rule(client,db_cursor,shipping,actors):
    key=actors['office']['key'];p=prepared(client,shipping,key);r=commit(client,p,key);assert r.status_code==200,r.text
    sid=r.json()['shipment_id'];a=upload(client,key,'truck_photo')
    assert client.post(f'/shipments/{sid}/attachments',json={'attachment_id':a['id']},headers=headers(key)).status_code==200
    db_cursor.execute('SELECT proof_status FROM shipments WHERE id=%s',(sid,));assert db_cursor.fetchone()['proof_status']=='photo_pending'

@pytest.mark.parametrize('action',['ship_standalone','ship_bulk'])
def test_standalone_duplicate_warning_names_receipt_actor_and_is_void_aware(client,db_cursor,shipping,actors,action):
    key=actors['office']['key'];first=commit(client,prepared(client,shipping,key,action),key);assert first.status_code==200,first.text
    p=prepared(client,shipping,key,action)
    warning=next(w for w in p['warnings'] if w['code']=='POSSIBLE_DUPLICATE')
    assert first.json()['receipt_number'] in warning['message'] and actors['office']['name'] in warning['message']
    error(commit(client,p,key),409,'WARNING_NOT_ACKNOWLEDGED')
    db_cursor.execute('SELECT id FROM transactions WHERE ticket_id=%s',(first.json()['ticket_id'],))
    for txn in db_cursor.fetchall():
        r=client.post(f"/void/{txn['id']}",json={'reason':'duplicate reversed'},headers=headers());assert r.status_code==200,r.text
    p=prepared(client,shipping,key,action)
    assert not any(w['code']=='POSSIBLE_DUPLICATE' for w in p['warnings'])

def test_order_ship_duplicate_created_after_prepare_is_rechecked_at_commit(client,db_cursor,shipping,actors):
    key=actors['office']['key'];oid=new_order(client,shipping,key)
    db_cursor.execute('SELECT id FROM sales_order_lines WHERE sales_order_id=%s AND product_id=%s',(oid,shipping['products']['food']));line=db_cursor.fetchone()['id']
    path=f'/sales/orders/{oid}/ship/prepare'
    b=ship_body(shipping,lines=[{'line_id':line,'quantity_lb':10}])
    older=client.post(path,json=b,headers=headers(key)).json()
    newer=client.post(path,json=b|{'bol_reference':'BOL-B'},headers=headers(key)).json()
    r=commit(client,newer,key);assert r.status_code==200,r.text
    error(commit(client,older,key),409,'WARNING_NOT_ACKNOWLEDGED')
    r=commit(client,older,key,acknowledged_warnings=['POSSIBLE_DUPLICATE']);assert r.status_code==200,r.text
