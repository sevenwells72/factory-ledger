"""Real local PostgreSQL coverage; no production/staging test connections."""
import pytest
from fastapi.testclient import TestClient

import main
from tests.test_trace_batch_input_tree import client, item, post, receive


def trace(client, item, code):
    response=client.get('/dashboard/api/lot-report/'+code,params={'product_id':item[0],'direction':'backward'})
    assert response.status_code==200,response.text
    return response.json()


@pytest.mark.db
def test_repeated_makes_amendments_voids_and_complete_forward(client, db_cursor):
    oats = item(db_cursor, 'LH oats', 'ingredient', 'LH-OATS', 'received', supplier_lot='SUP-LH')
    bulk = item(db_cursor, 'LH bulk', 'batch', 'LH-BULK', 'production_output')
    case = item(db_cursor, 'LH cases', 'finished', 'LH-CASE', 'pack_output')
    receive(db_cursor, oats, 100, 'Supplier')
    first, lines = post(db_cursor, 'make', '2026-11-01 14:00', [(bulk,20),(oats,-20)])
    second, _ = post(db_cursor, 'make', '2026-11-01 15:00', [(bulk,20),(oats,-20)])
    voided, _ = post(db_cursor, 'make', '2026-11-01 16:00', [(bulk,20),(oats,-20)])
    main._append_transaction_correction(db_cursor, voided, 'void', 'duplicate', {}, 'test')
    main._append_transaction_line_correction(db_cursor, lines[1], {'quantity_lb':-18}, 'usage', 'test')
    packed, _ = post(db_cursor, 'pack', '2026-11-02 14:00', [(case,30),(bulk,-30)])
    shipped, _ = post(db_cursor, 'ship', '2026-11-03 14:00', [(case,-10)], customer_name='Audit Customer', order_reference='AUDIT-ORDER')
    data=client.get('/dashboard/api/lot-report/LH-OATS',params={'product_id':oats[0]}).json()
    assert [(r['transaction_id'],r['quantity_used']) for r in [e for e in data['forward_trace']['consumption'] if e['source_lot_id']==oats[1]]] == [(first,18),(second,20)]
    graph=data['forward_trace']
    assert [e['transaction_id'] for e in graph['consumption']] == [first,second,packed]
    nodes={n['lot_id']:n for n in graph['lots']}
    assert set(nodes)=={oats[1],bulk[1],case[1]}
    assert [nodes[n]['on_hand'] for n in [oats[1],bulk[1],case[1]]] == [62,10,20]
    assert nodes[case[1]]['shipments'][0]['transaction_id']==shipped
    assert nodes[case[1]]['shipments'][0]['order_reference']=='AUDIT-ORDER'
    for path in ['/trace/ingredient/LH-OATS','/trace/batch/LH-BULK','/trace/supplier-lot/SUP-LH']:
        legacy=client.get(path).json()
        assert 'forward_trace' not in str(legacy)



@pytest.mark.db
def test_history_includes_depleted_found_all_receipts_and_current_quantities(client, db_cursor):
    old=item(db_cursor,'LH history oats','ingredient','LH-OLD','received',supplier_lot='S-1')
    db_cursor.execute("INSERT INTO lots(product_id,lot_code,entry_source) VALUES (%s,'LH-NEW','found_inventory') RETURNING id",(old[0],))
    found=(old[0],db_cursor.fetchone()['id'])
    db_cursor.execute("INSERT INTO lots(product_id,lot_code,entry_source) VALUES (%s,'LH-VOID','received') RETURNING id",(old[0],))
    void=(old[0],db_cursor.fetchone()['id'])
    receive(db_cursor,old,20,'Nation Harvest',stamp='2026-10-01 14:00',bol='BOL-OLD')
    db_cursor.execute("UPDATE lots SET entry_source_notes='POOLED opening stock; provisional history' WHERE id=%s",(old[1],))
    receive(db_cursor,old,10,'National Harvest',stamp='2026-10-02 14:00',bol='BOL-NEW')
    post(db_cursor,'ship','2026-10-03 14:00',[(old,-30)],customer_name='Direct Customer')
    post(db_cursor,'adjust','2026-11-01 01:00',[(found,7)])
    tid=receive(db_cursor,void,100,'Wrong')
    main._append_transaction_correction(db_cursor,tid,'void','duplicate',{},'test')
    url=f'/dashboard/api/inventory/{old[0]}/lot-history'
    response=client.get(url)
    assert response.status_code==200,response.text
    lots=response.json()['lots']
    assert [n['lot_id'] for n in lots]==[found[1],old[1]]
    assert lots[0]['found_inventory'] and lots[0]['received_quantity']==7
    assert lots[0]['latest_date']=='2026-10-31'
    assert lots[1]['remaining']==0 and lots[1]['received_quantity']==30
    assert lots[1]['entry_source_notes']=='POOLED opening stock; provisional history'
    assert [r['supplier_name'] for r in lots[1]['receipts']]==['National Harvest','Nation Harvest']
    assert [r['bol_reference'] for r in lots[1]['receipts']]==['BOL-NEW','BOL-OLD']
    assert TestClient(main.app).get(url).status_code==401
    assert TestClient(main.app,headers={'X-API-Key':'wrong'}).get(url).status_code==403
    assert client.get('/dashboard/api/inventory/0/lot-history').status_code==422
    assert client.get('/dashboard/api/inventory/2147483647/lot-history').status_code==404


@pytest.mark.db
def test_finished_history_all_packs_and_multi_output_share_basis(client,db_cursor):
    oats=item(db_cursor,'LH share oats','ingredient','LH-SHARE-OATS','received')
    finished=item(db_cursor,'LH share cases','finished','LH-SHARE-CASE','pack_output',uom='12x10 oz case')
    other=item(db_cursor,'LH other cases','finished','LH-SHARE-CASE','pack_output')
    db_cursor.execute('UPDATE products SET case_size_lb=10 WHERE id=%s',(finished[0],))
    receive(db_cursor,oats,200,'Supplier')
    first,_=post(db_cursor,'pack','2026-11-01 14:00',[(oats,-100),(finished,40),(other,60)])
    second,_=post(db_cursor,'pack','2026-11-02 14:00',[(oats,-20),(finished,20)])
    post(db_cursor,'ship','2026-11-03 14:00',[(finished,-60)],customer_name='Customer')
    data=trace(client,finished,'LH-SHARE-CASE')
    assert data['output_lb']==60 and data['input_tree']['output_lb']==60
    assert [(t['output_lb'],t['lot_output_lb']) for t in data['input_tree']['transactions']]==[(100,40),(20,20)]
    lot=client.get(f'/dashboard/api/inventory/{finished[0]}/lot-history').json()['lots'][0]
    assert lot['packed_quantity']==60 and lot['packed_cases']==6 and lot['remaining']==0
    assert [t['transaction_id'] for t in lot['production']]==[second,first]
    assert client.get('/trace/batch/LH-SHARE-CASE').status_code==409


@pytest.mark.db
def test_shipment_reference_fanout_does_not_multiply_ledger_quantity(client,db_cursor):
    case=item(db_cursor,'LH shipped cases','finished','LH-SHIPPED','pack_output')
    post(db_cursor,'pack','2026-11-01 14:00',[(case,100)])
    tid,_=post(db_cursor,'ship','2026-11-02 14:00',[(case,-10),(case,-5)],customer_name='LH Customer')
    db_cursor.execute("INSERT INTO customers(name) VALUES ('LH Customer') RETURNING id")
    cid=db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO sales_orders(order_number,customer_id) VALUES ('LH-ORDER',%s) RETURNING id",(cid,))
    oid=db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO shipments(sales_order_id,customer_id,shipped_at) VALUES (%s,%s,now()) RETURNING id",(oid,cid))
    sid=db_cursor.fetchone()['id']
    for q in (10,5):
        db_cursor.execute('INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb) VALUES (%s,%s,%s) RETURNING id',(oid,case[0],q))
        line=db_cursor.fetchone()['id']
        db_cursor.execute('INSERT INTO sales_order_shipments(sales_order_line_id,transaction_id,quantity_lb) VALUES (%s,%s,%s)',(line,tid,q))
        db_cursor.execute('INSERT INTO shipment_lines(shipment_id,transaction_id,sales_order_line_id,product_id,quantity_lb) VALUES (%s,%s,%s,%s,%s)',(sid,tid,line,case[0],q))
    data=trace(client,case,'LH-SHIPPED')
    assert data['total_shipped_lb']==15
    [ship]=data['customer_shipments']
    assert ship['shipment_ids']==[sid] and ship['sales_orders']==[{'sales_order_id':oid,'order_number':'LH-ORDER'}]
    assert data['forward_trace']['lots'][0]['shipments']==[ship]


@pytest.mark.db
def test_forward_cycles_and_consumption_without_output_are_visible(client,db_cursor):
    a=item(db_cursor,'LH cycle a','batch','LH-CYCLE-A','production_output')
    b=item(db_cursor,'LH cycle b','batch','LH-CYCLE-B','production_output')
    first,_=post(db_cursor,'make','2026-11-01 14:00',[(a,10),(b,-10)])
    second,_=post(db_cursor,'make','2026-11-02 14:00',[(a,-10),(b,10)])
    orphan,_=post(db_cursor,'pack','2026-11-03 14:00',[(b,-2)])
    data=client.get('/dashboard/api/lot-report/LH-CYCLE-A',params={'product_id':a[0]}).json()['forward_trace']
    assert len(data['lots'])==2
    assert {e['transaction_id'] for e in data['consumption']}=={first,second,orphan}
    assert data['gaps']==[{'transaction_id':orphan,'lot_id':b[1],'reason':'Consumed lot has no identified output lot'}]


@pytest.mark.db
def test_same_lot_input_and_output_remain_separate_and_notes_survive(client,db_cursor):
    lot=item(db_cursor,'LH rework','batch','LH-REWORK','production_output')
    db_cursor.execute("UPDATE lots SET entry_source_notes='PROVISIONAL FIFO, not read from labels' WHERE id=%s",(lot[1],))
    post(db_cursor,'make','2026-11-01 14:00',[(lot,20),(lot,-10)])
    data=trace(client,lot,'LH-REWORK')
    [txn]=data['input_tree']['transactions']
    assert txn['output_lb']==20 and txn['inputs_total_lb']==10
    assert txn['inputs'][0]['quantity']==10
    assert txn['inputs'][0]['entry_source_notes']=='PROVISIONAL FIFO, not read from labels'
    assert txn['inputs'][0]['origin']['truncated']=='cycle'
    assert data['forward_trace']['lots'][0]['entry_source_notes']=='PROVISIONAL FIFO, not read from labels'


@pytest.mark.db
def test_commingled_supplier_entries_survive_history_and_both_trace_directions(client,db_cursor):
    oats=item(db_cursor,'LH pooled oats','ingredient','LH-POOLED','received',supplier_lot='OLD-PRIMARY')
    bulk=item(db_cursor,'LH pooled batch','batch','LH-POOLED-OUTPUT','production_output')
    receive(db_cursor,oats,50,'Recorded receipt supplier')
    db_cursor.execute("UPDATE lots SET lot_type='commingled' WHERE id=%s",(oats[1],))
    for code,name,q in [('SUP-A','Source A',20),('SUP-B','Source B',30)]:
        db_cursor.execute("INSERT INTO lot_supplier_codes(lot_id,supplier_lot_code,supplier_name,quantity_lb,notes) VALUES (%s,%s,%s,%s,'Recorded source detail')",(oats[1],code,name,q))
    post(db_cursor,'make','2026-11-01 14:00',[(bulk,50),(oats,-50)])
    history=client.get(f'/dashboard/api/inventory/{oats[0]}/lot-history').json()['lots'][0]
    forward=client.get('/dashboard/api/lot-report/LH-POOLED',params={'product_id':oats[0]}).json()['forward_trace']['lots'][0]
    backward=trace(client,bulk,'LH-POOLED-OUTPUT')['input_tree']['transactions'][0]['inputs'][0]
    for n in [history,forward,backward]:
        assert n['lot_type']=='commingled'
        assert [(e['supplier_name'],e['supplier_lot_code'],e['quantity_lb']) for e in n['supplier_lot_entries']]==[('Source A','SUP-A',20),('Source B','SUP-B',30)]


@pytest.mark.db
@pytest.mark.parametrize('fanout',[1,80,500])
def test_forward_query_count_is_five_independent_of_reachable_lots(db_cursor,fanout):
    import lot_trace
    source=item(db_cursor,'LH query source','ingredient','LH-QUERY-SOURCE','received')
    receive(db_cursor,source,fanout,'Supplier')
    for i in range(fanout):
        target=item(db_cursor,f'LH query output {i}','batch',f'LH-QUERY-{i}','production_output')
        post(db_cursor,'make','2026-11-01 14:00',[(source,-1),(target,1)])
    class Counter:
        def __init__(self):self.count=0
        def execute(self,*a,**kw):self.count+=1;return db_cursor.execute(*a,**kw)
        def fetchall(self):return db_cursor.fetchall()
    counted=Counter()
    result=lot_trace.forward(counted,source[1],main)
    assert len(result['lots'])==fanout+1
    assert len(result['consumption'])==fanout
    assert counted.count==5, 'Forward SQL count must not grow with reachable lots'


@pytest.mark.db
def test_gpt_ingredients_keep_legacy_source_and_report_uses_current_lines(client,db_cursor):
    oats=item(db_cursor,'LH compatibility oats','ingredient','LH-COMPAT-OATS','received')
    bulk=item(db_cursor,'LH compatibility bulk','batch','LH-COMPAT-BULK','production_output')
    tid,_=post(db_cursor,'make','2026-11-01 14:00',[(oats,-20),(bulk,20)])
    db_cursor.execute('INSERT INTO ingredient_lot_consumption(transaction_id,ingredient_product_id,ingredient_lot_id,quantity_lb) VALUES (%s,%s,%s,5)',(tid,oats[0],oats[1]))
    legacy=client.get('/trace/batch/LH-COMPAT-BULK',params={'product_id':bulk[0]}).json()
    assert legacy['ingredients'][0]['quantity_lb']==5
    assert 'forward_trace' not in legacy and 'report_version' not in legacy
    audit=trace(client,bulk,'LH-COMPAT-BULK')
    assert audit['input_tree']['transactions'][0]['inputs'][0]['quantity']==20
    assert audit['report_version']==1 and audit['complete'] is True
    assert TestClient(main.app).get('/dashboard/api/lot-report/LH-COMPAT-BULK').status_code==401
