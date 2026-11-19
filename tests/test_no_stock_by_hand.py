"""Section A: real HTTP posting proofs against the local rollback-only ledger."""
from pathlib import Path
from uuid import uuid4
import pytest
import main
from tests.test_actor_attribution import client, actors
from tests.test_write_tickets_part2 import items, prepare, body, isolated_database
from tests.test_write_tickets import headers, commit, error, isolated_database as a1_isolated_database
from tests.test_exceptions_enforcement import adjust, found, resolve

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def rules(db_cursor,monkeypatch):
    monkeypatch.delenv('RESEND_API_KEY',raising=False)
    monkeypatch.delenv('ALERT_EMAIL_TO',raising=False)
    db_cursor.execute((ROOT/'migrations/096_adjust_destinations.sql').read_text())
    db_cursor.execute((ROOT/'migrations/092_shortage_unverified_close.sql').read_text())
    db_cursor.execute((ROOT/'migrations/098_no_stock_by_hand.sql').read_text())


def received(client, cur, items, actors, pack=50, cases=1):
    cur.execute('INSERT INTO suppliers(name) VALUES (%s) RETURNING id', ('Stock guard '+uuid4().hex,))
    supplier=cur.fetchone()['id']
    payload=dict(product_id=items['ingredient']['id'],supplier_id=supplier,cases=cases,case_size_lb=pack,
                 bol_reference='DEL-'+uuid4().hex,supplier_lot_code='SUP-01',occurred_at=main.get_plant_now().isoformat())
    prepared=prepare(client,'receive',payload,actors['floor']['key'])
    posted=commit(client,prepared,actors['floor']['key'])
    assert posted.status_code==200,posted.text
    items['ingredient'].update(lot_id=posted.json()['lot_id'],lot_code=posted.json()['lot_code'])
    return payload,posted.json()


def short(client,cur,items,actors,amount):
    lot=items['ingredient']
    # Make consumes exactly enough of this actual tagged lot to leave -amount.
    needed=main.lot_on_hand(cur,lot['lot_id'])+amount
    cur.execute('UPDATE products SET default_batch_lb=%s WHERE id=%s',(needed,items['batch']['id']))
    cur.execute('UPDATE batch_formulas SET quantity_lb=%s WHERE product_id=%s',(needed,items['batch']['id']))
    data=body('make',items)|{'ingredient_lots':[{'ingredient_product_id':lot['id'],'lot_id':lot['lot_id']}]}
    prepared=prepare(client,'make',data,actors['floor']['key'])
    posted=commit(client,prepared,actors['floor']['key'])
    assert posted.status_code==200,posted.text
    assert main.lot_on_hand(cur,lot['lot_id'])==pytest.approx(-amount)
    return posted.json()['shortages'][0]['exception_id']


def test_existing_receive_and_positive_adjust(client,db_cursor,items,actors,rules):
    payload,receipt=received(client,db_cursor,items,actors)
    response=client.post('/receive/prepare',json=payload|{'lot_code':receipt['lot_code']},headers=headers(actors['floor']['key']))
    error(response,409,'LOT_ALREADY_HAS_STOCK')
    error(client.post('/adjust/prepare',json=adjust(items,1),headers=headers(actors['floor']['key'])),422,'NO_STOCK_BY_HAND')
    error(client.post('/inventory/found/prepare',json=found(items,1,lot_code=receipt['lot_code']),headers=headers(actors['floor']['key'])),422,'NO_STOCK_BY_HAND')
    error(client.post('/adjust/prepare',json=adjust(items,1,'missing_receipt'),headers=headers(actors['floor']['key'])),422,'USE_RECEIVE')
    assert main.lot_on_hand(db_cursor,receipt['lot_id'])==50


@pytest.mark.parametrize('path',['resolve','owner'])
@pytest.mark.parametrize('pack,amount,allowed',[(50,49.99,True),(50,50,False),(50,68.49,False),(2000,99,True),(2000,100,False)])
def test_under_pack_both_count_paths(client,db_cursor,items,actors,rules,path,pack,amount,allowed):
    received(client,db_cursor,items,actors,pack)
    exc=short(client,db_cursor,items,actors,amount)
    if path=='resolve':
        response=resolve(client,exc,actors['floor']['key'],resolution_kind='counted',counted_lb=0,note='scale check')
    else:
        response=client.post('/adjust/prepare',json=adjust(items,amount,resolves_shortage_ids=[exc]),headers=headers(actors['owner']['key']))
        if response.status_code==200:response=commit(client,response.json(),actors['owner']['key'])
    if allowed:
        assert response.status_code==200,response.text
        assert main.lot_on_hand(db_cursor,items['ingredient']['lot_id'])==0
        db_cursor.execute('SELECT lb FROM under_pack_counts WHERE lot_id=%s',(items['ingredient']['lot_id'],))
        assert float(db_cursor.fetchone()['lb'])==amount
    else:
        error(response,422,'NO_STOCK_BY_HAND')
        assert main.lot_on_hand(db_cursor,items['ingredient']['lot_id'])==pytest.approx(-amount)
    db_cursor.execute('SELECT status FROM exceptions WHERE id=%s',(exc,))
    assert db_cursor.fetchone()['status']==('resolved' if allowed else 'open')


@pytest.mark.parametrize('path',['resolve','owner'])
def test_cumulative_cap(client,db_cursor,items,actors,rules,path):
    received(client,db_cursor,items,actors,55)
    for n in range(2):
        exc=short(client,db_cursor,items,actors,30)
        if path=='resolve':response=resolve(client,exc,actors['floor']['key'],resolution_kind='counted',counted_lb=0,note='scale')
        else:
            response=client.post('/adjust/prepare',json=adjust(items,30,resolves_shortage_ids=[exc]),headers=headers(actors['owner']['key']))
            if response.status_code==200:response=commit(client,response.json(),actors['owner']['key'])
        if n==0:assert response.status_code==200,response.text
        else:error(response,422,'NO_STOCK_BY_HAND')
    db_cursor.execute('SELECT sum(lb) AS lb FROM under_pack_counts WHERE lot_id=%s',(items['ingredient']['lot_id'],))
    assert float(db_cursor.fetchone()['lb'])==30


def test_give_back_and_pair(client,db_cursor,items,actors,rules):
    _,receipt=received(client,db_cursor,items,actors,50,20)
    key=actors['floor']['key']
    removed=commit(client,prepare(client,'adjust',adjust(items,-500,'damaged_thrown_out'),key),key)
    assert removed.status_code==200,removed.text
    error(client.post('/adjust/prepare',json=adjust(items,501),headers=headers(key)),422,'NO_STOCK_BY_HAND')
    restored=commit(client,prepare(client,'adjust',adjust(items,500),key),key)
    assert restored.status_code==200,restored.text
    a=receipt['lot_id']
    db_cursor.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id',(items['ingredient']['id'],'26-11-19-PAIR-'+uuid4().hex[:3].upper()))
    b=db_cursor.fetchone()['id']
    error(client.post('/adjust/prepare',json=adjust(items,10,'wrong_lot',note='tag mixup'),headers=headers(key)),422,'PAIRED_MOVE_REQUIRED')
    move=commit(client,prepare(client,'adjust',adjust(items,10,'wrong_lot',note='tag mixup',paired_lot_id=b),key),key)
    assert move.status_code==200,move.text
    db_cursor.execute('SELECT lot_id,quantity_lb FROM transaction_lines WHERE transaction_id=%s ORDER BY quantity_lb',(move.json()['transaction_id'],))
    assert [(r['lot_id'],float(r['quantity_lb'])) for r in db_cursor.fetchall()]==[(b,-10),(a,10)]
    db_cursor.execute('SELECT count(*) AS n FROM shortage_flags WHERE transaction_id=%s',(move.json()['transaction_id'],))
    assert db_cursor.fetchone()['n']==1


def test_shared_key_default_and_owner_scope(client,db_cursor,items,actors,rules,monkeypatch):
    error(client.post('/adjust',json={},headers=headers(main.API_KEY)),403,'SHARED_KEY_READ_ONLY')
    assert client.get('/health',headers=headers(main.API_KEY)).status_code==200
    assert client.post('/resolve',json={'kind':'product','query':'none'},headers=headers(main.API_KEY)).status_code==200
    monkeypatch.setenv('SHARED_KEY_WRITES','allow')
    assert client.post('/adjust',json={},headers=headers(main.API_KEY)).status_code==422

@pytest.mark.parametrize('path',['resolve','owner'])
def test_no_pack_and_above_zero_count_refused(client,db_cursor,items,actors,rules,path):
    # The fixture's receive has no case weight, although the product has one.
    exc=short(client,db_cursor,items,actors,5)
    key=actors['owner']['key'] if path=='owner' else actors['floor']['key']
    if path=='owner':response=client.post('/adjust/prepare',json=adjust(items,5,resolves_shortage_ids=[exc]),headers=headers(key))
    else:response=resolve(client,exc,key,resolution_kind='counted',counted_lb=0,note='count')
    error(response,422,'NO_STOCK_BY_HAND')
    assert 'missing_movement' in response.json()['detail']['message']
    ok=resolve(client,exc,actors['floor']['key'],resolution_kind='unverified',note='Cannot show which lot was used')
    assert ok.status_code==200,ok.text
    assert main.lot_on_hand(db_cursor,items['ingredient']['lot_id'])==-5


@pytest.mark.parametrize('path',['resolve','owner'])
def test_small_shortage_cannot_be_counted_above_zero(client,db_cursor,items,actors,rules,path):
    received(client,db_cursor,items,actors)
    exc=short(client,db_cursor,items,actors,5)
    if path=='owner':response=client.post('/adjust/prepare',json=adjust(items,6,resolves_shortage_ids=[exc]),headers=headers(actors['owner']['key']))
    else:response=resolve(client,exc,actors['floor']['key'],resolution_kind='counted',counted_lb=1,note='count')
    error(response,422,'NO_STOCK_BY_HAND')


def test_pair_resolves_shortage_but_earns_no_giveback(client,db_cursor,items,actors,rules):
    received(client,db_cursor,items,actors)
    a=items['ingredient']['lot_id']
    exc=short(client,db_cursor,items,actors,20)
    db_cursor.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id',
                      (items['ingredient']['id'],'26-11-19-PAIR-'+uuid4().hex[:3].upper()))
    b=db_cursor.fetchone()['id'];key=actors['floor']['key']
    response=commit(client,prepare(client,'adjust',adjust(items,20,'wrong_lot',note='Used the other tag',paired_lot_id=b),key),key)
    assert response.status_code==200,response.text
    close=resolve(client,exc,key,resolution_kind='missing_movement',receipt_number=response.json()['receipt_number'],note='Wrong tag')
    assert close.status_code==200,close.text
    assert main.lot_on_hand(db_cursor,a)==0 and main.lot_on_hand(db_cursor,b)==-20
    # B's move leg cannot earn giveback, even after its shortage is closed without changing stock.
    db_cursor.execute('SELECT id FROM exceptions WHERE transaction_id=%s AND kind=\'SHORTAGE\'',(response.json()['transaction_id'],))
    exc_b=db_cursor.fetchone()['id']
    assert resolve(client,exc_b,key,resolution_kind='unverified',note='Tag move investigated').status_code==200
    payload=adjust(items,20)|{'lot_id':b}
    error(client.post('/adjust/prepare',json=payload,headers=headers(key)),422,'NO_STOCK_BY_HAND')
    void=client.post('/void/'+str(response.json()['transaction_id']),json={'reason':'data_entry_error'},headers=headers(actors['owner']['key']))
    assert void.status_code==200,void.text
    assert main.lot_on_hand(db_cursor,a)==-20 and main.lot_on_hand(db_cursor,b)==0


@pytest.mark.parametrize('kind',['same','other_product','merged','blank_note','negative'])
def test_pair_validation_posts_nothing(client,db_cursor,items,actors,rules,kind):
    a=items['ingredient']['lot_id']; b=items['batch']['lot_id']
    if kind in ('merged','blank_note','negative'):
        db_cursor.execute('INSERT INTO lots(product_id,lot_code,status) VALUES (%s,%s,%s) RETURNING id',
                          (items['ingredient']['id'],'26-11-19-PAIR-'+uuid4().hex[:3].upper(),'merged' if kind=='merged' else 'active'))
        b=db_cursor.fetchone()['id']
    data=adjust(items,-1 if kind=='negative' else 1,'wrong_lot',note='' if kind=='blank_note' else 'tag mixup',paired_lot_id=a if kind=='same' else b)
    db_cursor.execute('SELECT count(*) AS n FROM transactions');before=db_cursor.fetchone()['n']
    response=client.post('/adjust/prepare',json=data,headers=headers(actors['floor']['key']))
    assert response.status_code in (409,422),response.text
    if kind=='other_product':assert response.json()['detail']['error_code']=='WRONG_LOT_SAME_PRODUCT'
    db_cursor.execute('SELECT count(*) AS n FROM transactions');assert db_cursor.fetchone()['n']==before


@pytest.mark.parametrize('route',['/receive','/receive/preview','/receive/commit'])
@pytest.mark.parametrize('change,code',[({'bol_reference':'  '},'BOL_REQUIRED'),({'cases':0},'INVALID_QUANTITY'),({'cases':-1},'INVALID_QUANTITY'),({'case_size_lb':0},'INVALID_QUANTITY')])
def test_direct_receive_validation(client,db_cursor,items,actors,rules,route,change,code):
    payload,_=received(client,db_cursor,items,actors)
    db_cursor.execute('SELECT name FROM suppliers WHERE id=%s',(payload['supplier_id'],));name=db_cursor.fetchone()['name']
    req={k:v for k,v in payload.items() if k not in ('product_id','supplier_id')}
    req.update(product_name=items['ingredient']['name'],shipper_name=name,mode='commit',**change)
    error(client.post(route,json=req,headers=headers(actors['owner']['key'])),422,code)


def test_voided_receive_can_be_received_again_after_consumption(client,db_cursor,items,actors,rules):
    payload,receipt=received(client,db_cursor,items,actors)
    exc=short(client,db_cursor,items,actors,5)
    response=client.post('/void/'+str(receipt['transaction_id']),json={'reason':'data_entry_error'},headers=headers(actors['owner']['key']))
    assert response.status_code==200,response.text
    error(resolve(client,exc,actors['floor']['key'],resolution_kind='counted',counted_lb=0,note='count'),422,'NO_STOCK_BY_HAND')
    prepared=prepare(client,'receive',payload|{'lot_code':receipt['lot_code'],'cases':2},actors['floor']['key'])
    again=commit(client,prepared,actors['floor']['key'],acknowledged_warnings=[w['code'] for w in prepared['warnings']])
    assert again.status_code==200,again.text
    assert again.json()['lot_id']==receipt['lot_id']
    assert main.lot_on_hand(db_cursor,receipt['lot_id'])==45


def test_typing_refusal_is_bilingual_and_direct_reasons_enforced(client,db_cursor,items,actors,rules):
    payload,_=received(client,db_cursor,items,actors)
    key=actors['owner']['key']
    data=dict(mode='commit',product_name=items['ingredient']['name'],lot_code=items['ingredient']['lot_code'],adjustment_lb=1,reason='data_entry_error',note='typo')
    response=client.post('/adjust',json=data,headers=headers(key));error(response,422,'NO_STOCK_BY_HAND')
    assert 'void the wrong entry and enter it again' in response.json()['detail']['message']
    assert 'anula la entrada equivocada' in response.json()['detail']['message_es']
    error(client.post('/adjust',json=data|{'reason':'invented free text'},headers=headers(key)),422,'REASON_CODE_INVALID')
    error(client.post('/adjust',json=data|{'reason':'missing_receipt'},headers=headers(key)),422,'USE_RECEIVE')
    error(client.post('/adjust',json=data|{'reason':'wrong_lot'},headers=headers(key)),422,'PAIRED_MOVE_REQUIRED')


def test_migration_rerunnable_and_down_restores_list(db_cursor,rules):
    sql=(ROOT/'migrations/098_no_stock_by_hand.sql').read_text();db_cursor.execute(sql);db_cursor.execute(sql)
    db_cursor.execute("SELECT count(*) AS n FROM migration_markers WHERE name='098_no_stock_by_hand'");assert db_cursor.fetchone()['n']==1
    db_cursor.execute((ROOT/'migrations/down/098_no_stock_by_hand_down.sql').read_text())
    db_cursor.execute("SELECT applies_to FROM correction_reasons WHERE code='missing_receipt'");assert 'adjust' in db_cursor.fetchone()['applies_to']
    import stock_by_hand
    assert stock_by_hand.live_at(db_cursor) is None

@pytest.mark.parametrize('amended_pack,allowed',[(5,False),(55,True)])
def test_amended_receipt_pack_weight_is_effective(client,db_cursor,items,actors,rules,amended_pack,allowed):
    _,receipt=received(client,db_cursor,items,actors,50)
    correction=client.post(f"/records/transactions/{receipt['transaction_id']}/corrections",json={
        'event_type':'amend','reason':'correct container weight','replacement_values':{'case_size_lb':amended_pack}},headers=headers(actors['owner']['key']))
    assert correction.status_code==200,correction.text
    db_cursor.execute("SELECT entered_by_actor_id FROM ledger_corrections WHERE target_table='transactions' AND target_id=%s ORDER BY id DESC LIMIT 1",(receipt['transaction_id'],))
    assert db_cursor.fetchone()['entered_by_actor_id']==actors['owner']['id']
    exc=short(client,db_cursor,items,actors,10)
    response=resolve(client,exc,actors['floor']['key'],resolution_kind='counted',counted_lb=0,note='weighed empty')
    if allowed:assert response.status_code==200,response.text
    else:error(response,422,'NO_STOCK_BY_HAND')


def test_go_live_does_not_credit_old_hand_removals(client,db_cursor,items,actors,rules):
    db_cursor.execute("DELETE FROM migration_markers WHERE name='098_no_stock_by_hand'")
    key=actors['floor']['key']
    removed=commit(client,prepare(client,'adjust',adjust(items,-10,'damaged_thrown_out'),key),key)
    assert removed.status_code==200,removed.text
    db_cursor.execute((ROOT/'migrations/098_no_stock_by_hand.sql').read_text())
    error(client.post('/adjust/prepare',json=adjust(items,10),headers=headers(key)),422,'NO_STOCK_BY_HAND')


def test_found_new_lot_and_count_born_topups(client,db_cursor,items,actors,rules):
    key=actors['floor']['key']
    made=commit(client,prepare(client,'found',found(items,10,'missing_receipt'),key),key)
    assert made.status_code==200,made.text
    lot=made.json()
    error(client.post('/inventory/found/prepare',json=found(items,1,'missing_receipt',lot_code=lot['lot_code']),headers=headers(key)),409,'LOT_CODE_TAKEN')
    error(client.post('/inventory/found/prepare',json=found(items,1,lot_code=lot['lot_code']),headers=headers(key)),422,'NO_STOCK_BY_HAND')
    error(client.post('/adjust/prepare',json=adjust(items,1)|{'lot_id':lot['lot_id']},headers=headers(key)),422,'NO_STOCK_BY_HAND')
    db_cursor.execute('SELECT identity_status FROM lots WHERE id=%s',(lot['lot_id'],))
    assert db_cursor.fetchone()['identity_status']=='unidentified'
    db_cursor.execute("SELECT count(*) AS n FROM exceptions WHERE lot_id=%s AND kind='UNIDENTIFIED_LOT'",(lot['lot_id'],))
    assert db_cursor.fetchone()['n']==1


def test_named_owner_reaches_maintenance_but_floor_does_not(client,actors):
    assert client.get('/admin/lots/duplicates',headers=headers(actors['owner']['key'])).status_code==200
    assert client.get('/admin/lots/duplicates',headers=headers(actors['floor']['key'])).status_code==403


def test_every_authenticated_write_refuses_shared_key_without_rows(client,db_cursor,actors):
    from fastapi.routing import APIRoute
    import re
    tables=['transactions','transaction_lines','write_tickets','lots','exceptions','actor_write_audit','customers','sales_orders']
    def snapshot():
        result=[]
        for table in tables:
            db_cursor.execute('SELECT count(*) AS n FROM '+table);result.append(db_cursor.fetchone()['n'])
        return result
    before=snapshot();checked=[]
    def key_dependency(dep):
        return getattr(dep.call,'__name__','') in ('verify_api_key','verify_api_key_flexible') or any(key_dependency(c) for c in dep.dependencies)
    for route in main.app.routes:
        if not isinstance(route,APIRoute) or not key_dependency(route.dependant):continue
        path=re.sub(r'\{[^}]+\}','1',route.path)
        for method in route.methods-{'GET','HEAD','OPTIONS'}:
            if (method,route.path) in {('POST','/resolve'),('POST','/products/resolve')}:continue
            response=client.request(method,path,json={},headers=headers(main.API_KEY))
            assert response.status_code==403,(method,path,response.text)
            assert response.json()['detail']['error_code']=='SHARED_KEY_READ_ONLY',(method,path,response.text)
            checked.append((method,route.path))
    assert len(checked)>80
    assert snapshot()==before


def test_two_concurrent_givebacks_cannot_both_post(isolated_database,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    from threading import Barrier
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from fastapi.testclient import TestClient
    from tests.test_write_tickets_part2 import seed
    from tests.named_key import KEY
    with psycopg2.connect(isolated_database) as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
        data=seed(cur)
        cur.execute((ROOT/'migrations/096_adjust_destinations.sql').read_text())
        cur.execute((ROOT/'migrations/098_no_stock_by_hand.sql').read_text())
    main._reset_actor_cache();gate=Barrier(2);racing=False
    @contextmanager
    def connection():
        with psycopg2.connect(isolated_database) as conn:
            if racing:gate.wait(timeout=10)
            yield conn
    monkeypatch.setattr(main,'get_db_connection',connection)
    try:
        with TestClient(main.app) as http:
            removed=commit(http,prepare(http,'adjust',adjust(data,-10,'damaged_thrown_out'),KEY),KEY)
            assert removed.status_code==200,removed.text
            drafts=[prepare(http,'adjust',adjust(data,10),KEY) for _ in range(2)]
            racing=True
            with ThreadPoolExecutor(max_workers=2) as pool:
                results=list(pool.map(lambda draft:commit(http,draft,KEY),drafts))
            racing=False
            assert sorted(r.status_code for r in results)==[200,409],[r.text for r in results]
            refused=next(r for r in results if r.status_code==409)
            assert refused.json()['detail']['blockers'][0]['code']=='NO_STOCK_BY_HAND'
        with psycopg2.connect(isolated_database) as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
            assert main.lot_on_hand(cur,data['ingredient']['lot_id'])==100
    finally:main._reset_actor_cache()


def test_large_pair_queues_one_owner_email(client,db_cursor,items,actors,rules):
    a=items['ingredient']['lot_id']
    db_cursor.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id',
                      (items['ingredient']['id'],'26-11-19-PAIR-'+uuid4().hex[:3].upper()))
    b=db_cursor.fetchone()['id'];key=actors['floor']['key']
    prepared=prepare(client,'adjust',adjust(items,600,'wrong_lot',note='Confirmed the other lot was used',paired_lot_id=b),key)
    response=commit(client,prepared,key)
    assert response.status_code==200,response.text
    db_cursor.execute('SELECT count(*) AS n FROM adjustment_alerts WHERE transaction_id=%s',(response.json()['transaction_id'],))
    assert db_cursor.fetchone()['n']==1
    assert main.lot_on_hand(db_cursor,a)+main.lot_on_hand(db_cursor,b)==100


def test_noningredient_count_and_addition_unchanged(client,db_cursor,items,actors,rules):
    db_cursor.execute("UPDATE products SET type='batch' WHERE id=%s",(items['ingredient']['id'],))
    exc=short(client,db_cursor,items,actors,5)
    response=resolve(client,exc,actors['floor']['key'],resolution_kind='counted',counted_lb=20,note='counted the batch')
    assert response.status_code==200,response.text
    assert main.lot_on_hand(db_cursor,items['ingredient']['lot_id'])==20
    added=commit(client,prepare(client,'adjust',adjust(items,10,'missing_production'),actors['floor']['key']),actors['floor']['key'])
    assert added.status_code==200,added.text


def test_nightly_review_lists_named_ingredient_additions_only(client,db_cursor,items,actors,rules):
    from datetime import timedelta
    import reconciliation
    key=actors['floor']['key']
    removed=commit(client,prepare(client,'adjust',adjust(items,-10,'damaged_thrown_out'),key),key)
    added=commit(client,prepare(client,'adjust',adjust(items,10),key),key)
    assert removed.status_code==200 and added.status_code==200
    now=main.get_plant_now()
    report=reconciliation.analyse(reconciliation.load_window(db_cursor,now-timedelta(hours=1),now+timedelta(minutes=1),reconstruct=True))
    flags=[f for f in report['flags'] if f['kind']=='INGREDIENT_HAND_ADDITION']
    assert [f['transaction_id'] for f in flags]==[added.json()['transaction_id']]
    text=flags[0]['detail']['message']
    assert actors['floor']['name'] in text and added.json()['receipt_number'] in text and 'physical_count' in text
