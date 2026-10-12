"""Michael's PR #99 review decisions, exercised through real ticket endpoints."""
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest
import main
import ship_tickets
import attachments
from tests.test_ship_tickets import (client, actors, shipping, storage, new_order,
                                     ship_body, headers, commit, ticket_row, error)

pytestmark = pytest.mark.db

def historical_creation(cur, oid, created, order_date):
    # Disposable test DB only: represent immutable pre-existing historical rows.
    cur.execute('ALTER TABLE sales_orders DISABLE TRIGGER trg_sales_orders_created_at')
    cur.execute('UPDATE sales_orders SET created_at=%s,order_date=%s WHERE id=%s', (created,order_date,oid))
    cur.execute('ALTER TABLE sales_orders ENABLE TRIGGER trg_sales_orders_created_at')

@pytest.fixture(autouse=True)
def cutoff_unset(monkeypatch):
    monkeypatch.delenv('ORDER_LEGACY_CLOSE_CUTOFF', raising=False)

@pytest.mark.parametrize('cutoff', [None, '2026-11-20'])
def test_owner_legacy_close_note_actor_and_weekly_source(client, db_cursor, shipping, actors, monkeypatch, cutoff):
    if cutoff: monkeypatch.setenv('ORDER_LEGACY_CLOSE_CUTOFF', cutoff)
    key = actors['owner']['key']
    oid = new_order(client, shipping, actors['office']['key'])
    # 04:59 UTC is Nov 19 in NY. Editable order_date must not affect eligibility.
    historical_creation(db_cursor, oid, '2026-11-20 04:59:59+00', '2026-12-01')
    body = {'reason': 'shipped_not_recorded', 'note': 'Historical truck log confirmed by Michael'}
    p = client.post(f'/sales/orders/{oid}/close/prepare', json=body, headers=headers(key))
    assert p.status_code == 200, p.text
    assert p.json()['can_commit']
    r = commit(client, p.json(), key)
    assert r.status_code == 200, r.text
    result = r.json()
    audit = result['policy_acknowledgements'][0]
    assert audit['code'] == 'SHIPPED_NOT_RECORDED'
    assert audit['actor']['id'] == actors['owner']['id'] and audit['note'] == body['note']
    assert audit['order_created_on'] == '2026-11-19' and audit['cutoff'] == cutoff
    assert ticket_row(db_cursor, p.json())['response']['policy_acknowledgements'] == [audit]
    db_cursor.execute('SELECT state,state_reason,state_note FROM sales_orders WHERE id=%s', (oid,))
    assert dict(db_cursor.fetchone()) == dict(state='closed', state_reason=body['reason'], state_note=body['note'])
    listed = client.get('/receipts', params={'policy_code': 'SHIPPED_NOT_RECORDED'}, headers=headers(key)).json()['receipts']
    assert any(row['receipt_number'] == result['receipt_number'] and row['response']['policy_acknowledgements'] == [audit] for row in listed)
    absent = client.get('/receipts', params={'policy_code': 'SHIP_NOT_SAME_DAY'}, headers=headers(key)).json()['receipts']
    assert result['receipt_number'] not in [row['receipt_number'] for row in absent]

@pytest.mark.parametrize('created', ['2026-11-20 05:00:00+00', '2026-11-21 03:00:00+00'])
@pytest.mark.parametrize('role', ['owner', 'office', 'floor'])
def test_post_cutover_close_without_shipment_is_403_for_everyone(client, db_cursor, shipping, actors, monkeypatch, created, role):
    monkeypatch.setenv('ORDER_LEGACY_CLOSE_CUTOFF', '2026-11-20')
    oid = new_order(client, shipping, actors['office']['key'])
    historical_creation(db_cursor, oid, created, '2020-01-01')
    r = client.post(f'/sales/orders/{oid}/close/prepare', json={'reason':'shipped_not_recorded','note':'old order_date cannot bypass cutoff'}, headers=headers(actors[role]['key']))
    assert r.status_code == 403, r.text
    db_cursor.execute('SELECT state FROM sales_orders WHERE id=%s', (oid,))
    assert db_cursor.fetchone()['state'] == 'open'

@pytest.mark.parametrize('role', ['office', 'floor', 'master'])
def test_legacy_close_is_named_owner_only_when_cutoff_unset(client, db_cursor, shipping, actors, role):
    oid = new_order(client, shipping, actors['office']['key'])
    key = main.API_KEY if role == 'master' else actors[role]['key']
    error(client.post(f'/sales/orders/{oid}/close/prepare', json={'reason':'shipped_not_recorded','note':'legacy truck log'}, headers=headers(key)), 403, 'ROLE_NOT_ALLOWED')

@pytest.mark.parametrize('note', [None, '', '   '])
def test_legacy_close_requires_a_nonblank_note(client, db_cursor, shipping, actors, note):
    oid = new_order(client, shipping, actors['office']['key'])
    error(client.post(f'/sales/orders/{oid}/close/prepare', json={'reason':'shipped_not_recorded','note':note}, headers=headers(actors['owner']['key'])), 422, 'NOTE_REQUIRED')

def test_cutoff_is_rechecked_at_commit(client, db_cursor, shipping, actors, monkeypatch):
    key = actors['owner']['key']; oid = new_order(client, shipping, actors['office']['key'])
    p = client.post(f'/sales/orders/{oid}/close/prepare', json={'reason':'shipped_not_recorded','note':'historical log'}, headers=headers(key)).json()
    assert p['can_commit']
    monkeypatch.setenv('ORDER_LEGACY_CLOSE_CUTOFF', main.get_plant_now().date().isoformat())
    error(commit(client, p, key), 403, 'LEGACY_CLOSE_CUTOFF')
    assert ticket_row(db_cursor, p)['status'] == 'prepared'
    db_cursor.execute('SELECT state FROM sales_orders WHERE id=%s', (oid,)); assert db_cursor.fetchone()['state'] == 'open'

@pytest.mark.parametrize('action,role', [('ship_order','owner'), ('ship_order','floor'), ('ship_bulk','owner'), ('ship_bulk','floor'), ('ship_standalone','owner')])
def test_late_ship_owner_floor_ack_own_entry_and_weekly_source(client, db_cursor, shipping, actors, monkeypatch, action, role):
    key = actors[role]['key']; now = datetime(2026, 11, 2, 12, tzinfo=main.PLANT_TIMEZONE)
    monkeypatch.setattr(main, 'get_plant_now', lambda: now)
    body = ship_body(shipping, occurred_at=(now-timedelta(days=1)).isoformat())
    if action == 'ship_order':
        oid = new_order(client, shipping, actors['office']['key']); body['ship_all'] = True
        path = f'/sales/orders/{oid}/ship/prepare'
    else:
        body.update(customer_id=shipping['customer'], order_reference='LATE-A6')
        if action == 'ship_bulk':
            path = '/ship/bulk/prepare'; body['lines'] = [{'product_id':shipping['products']['food'],'lot_id':shipping['lot_id'],'quantity_lb':20}]
        else:
            path = '/ship/prepare'; body.update(product_id=shipping['products']['food'], lot_id=shipping['lot_id'], quantity_lb=20)
    r = client.post(path, json=body, headers=headers(key)); assert r.status_code == 200, r.text
    p = r.json(); assert 'SHIP_NOT_SAME_DAY' in [w['code'] for w in p['warnings']]
    error(commit(client, p, key), 409, 'WARNING_NOT_ACKNOWLEDGED')
    r = commit(client, p, key, acknowledged_warnings=['SHIP_NOT_SAME_DAY']); assert r.status_code == 200, r.text
    result = r.json(); audit = result['policy_acknowledgements'][0]
    assert audit['code'] == 'SHIP_NOT_SAME_DAY' and audit['actor']['id'] == actors[role]['id']
    assert audit['happened_on'] == '2026-11-01' and audit['entered_on'] == '2026-11-02'
    row = ticket_row(db_cursor, p)
    assert 'SHIP_NOT_SAME_DAY' in row['acknowledged'] and row['response']['policy_acknowledgements'] == [audit]
    listed = client.get('/receipts', params={'date':'2026-11-01','policy_code':'SHIP_NOT_SAME_DAY'}, headers=headers(key)).json()['receipts']
    assert any(row['receipt_number'] == result['receipt_number'] and row['response']['policy_acknowledgements'] == [audit] for row in listed)

def test_ship_ack_role_rechecked_at_commit(client, db_cursor, shipping, actors, monkeypatch):
    key = actors['floor']['key']; now = main.get_plant_now()
    oid = new_order(client, shipping, actors['office']['key'])
    r = client.post(f'/sales/orders/{oid}/ship/prepare', json=ship_body(shipping, ship_all=True, occurred_at=(now-timedelta(days=1)).isoformat()), headers=headers(key))
    assert r.status_code == 200, r.text
    db_cursor.execute("UPDATE actors SET role='office' WHERE id=%s", (actors['floor']['id'],)); main._reset_actor_cache()
    error(commit(client, r.json(), key, acknowledged_warnings=['SHIP_NOT_SAME_DAY']), 422, 'SHIP_NOT_SAME_DAY')
    assert ticket_row(db_cursor, r.json())['status'] == 'prepared'

def test_floor_late_ship_cannot_bypass_a2_over_14_days(client, db_cursor, shipping, actors):
    oid = new_order(client, shipping, actors['office']['key'])
    r = client.post(f'/sales/orders/{oid}/ship/prepare', json=ship_body(shipping, ship_all=True, occurred_at=(main.get_plant_now()-timedelta(days=15)).isoformat()), headers=headers(actors['floor']['key']))
    assert r.status_code == 403

@pytest.mark.parametrize('setting,value,message', [
    ('EVIDENCE_STORAGE_DIR','/tmp/synthetic-evidence','EVIDENCE_STORAGE_DIR'),
    ('PACKING_SLIP_SIGNING_KEY',None,'PACKING_SLIP_SIGNING_KEY'),
    ('PACKING_SLIP_SIGNING_KEY','','PACKING_SLIP_SIGNING_KEY'),
    ('PACKING_SLIP_SIGNING_KEY','   ','PACKING_SLIP_SIGNING_KEY'),
    ('SUPABASE_URL',None,'SUPABASE_URL'),
    ('SUPABASE_SERVICE_ROLE_KEY',None,'SUPABASE_SERVICE_ROLE_KEY'),
    ('ORDER_LEGACY_CLOSE_CUTOFF','2026-13-01','ORDER_LEGACY_CLOSE_CUTOFF'),
    ('ORDER_LEGACY_CLOSE_CUTOFF','20261120','ORDER_LEGACY_CLOSE_CUTOFF'),
])
def test_production_startup_refuses_before_connecting(monkeypatch, setting, value, message):
    monkeypatch.setenv('ENVIRONMENT','production')
    monkeypatch.delenv('EVIDENCE_STORAGE_DIR', raising=False)
    monkeypatch.setenv('PACKING_SLIP_SIGNING_KEY','synthetic-dedicated-test-key')
    monkeypatch.setenv('SUPABASE_URL','https://synthetic.invalid')
    monkeypatch.setenv('SUPABASE_SERVICE_ROLE_KEY','synthetic-storage-key')
    if value is None: monkeypatch.delenv(setting, raising=False)
    else: monkeypatch.setenv(setting,value)
    pool = Mock(side_effect=AssertionError('must not connect'))
    migrations = Mock(side_effect=AssertionError('must not migrate'))
    monkeypatch.setattr(main.pool,'ThreadedConnectionPool',pool)
    monkeypatch.setattr(main,'_run_startup_migrations',migrations)
    with pytest.raises(RuntimeError, match=message): asyncio.run(main.startup())
    pool.assert_not_called(); migrations.assert_not_called()

def test_production_signing_never_falls_back_to_master_key(monkeypatch):
    monkeypatch.setenv('ENVIRONMENT','production')
    monkeypatch.delenv('PACKING_SLIP_SIGNING_KEY', raising=False)
    monkeypatch.setenv('API_KEY','synthetic-master-test-key')
    with pytest.raises(main.HTTPException) as exc: ship_tickets._sign(1,'SHP-test',1)
    assert exc.value.status_code == 503

def test_production_storage_guard_also_prevents_runtime_local_writes(monkeypatch, tmp_path):
    monkeypatch.setenv('ENVIRONMENT','production')
    path = tmp_path/'must-not-exist'; monkeypatch.setenv('EVIDENCE_STORAGE_DIR',str(path))
    with pytest.raises(main.HTTPException) as exc: attachments._volume()
    assert exc.value.status_code == 503 and not path.exists()


def test_floor_49_hour_ship_preserves_a2_late_exception(client, db_cursor, shipping, actors):
    key=actors['floor']['key']; oid=new_order(client,shipping,actors['office']['key'])
    p=client.post(f'/sales/orders/{oid}/ship/prepare',json=ship_body(shipping,ship_all=True,occurred_at=(main.get_plant_now()-timedelta(hours=49)).isoformat()),headers=headers(key))
    assert p.status_code==200,p.text
    assert {'SHIP_NOT_SAME_DAY','LATE_ENTRY'} <= {w['code'] for w in p.json()['warnings']}
    r=commit(client,p.json(),key,acknowledged_warnings=['SHIP_NOT_SAME_DAY']);assert r.status_code==200,r.text
    eid=r.json()['entry_timing']['late_entry_exception_id'];assert eid is not None
    db_cursor.execute("SELECT kind,ticket_id,detail FROM exceptions WHERE id=%s",(eid,));row=db_cursor.fetchone()
    assert row['kind']=='LATE_ENTRY' and row['ticket_id']==p.json()['ticket_id'] and row['detail']['actor_id']==actors['floor']['id']
    replay=commit(client,p.json(),key,acknowledged_warnings=['SHIP_NOT_SAME_DAY']);assert replay.status_code==200
    db_cursor.execute("SELECT count(*) AS n FROM exceptions WHERE ticket_id=%s AND kind='LATE_ENTRY'",(p.json()['ticket_id'],));assert db_cursor.fetchone()['n']==1

@pytest.mark.parametrize('reason,role',[('shipped_recorded','office'),('short_closed','office')])
def test_post_cutover_order_closes_with_recorded_shipment(client,db_cursor,shipping,actors,monkeypatch,reason,role):
    monkeypatch.setenv('ORDER_LEGACY_CLOSE_CUTOFF',main.get_plant_now().date().isoformat())
    key=actors[role]['key'];oid=new_order(client,shipping,key)
    p=client.post(f'/sales/orders/{oid}/ship/prepare',json=ship_body(shipping,ship_all=True),headers=headers(key)).json()
    r=commit(client,p,key);assert r.status_code==200,r.text
    p=client.post(f'/sales/orders/{oid}/close/prepare',json={'reason':reason},headers=headers(key)).json()
    assert p['can_commit'];r=commit(client,p,key);assert r.status_code==200,r.text
