"""Round B: real handlers/catalog/ledger, deterministic language-model calls."""
from uuid import uuid4
import importlib
import pytest
import fl_assistant
import main
from tests.test_fl_assistant import (enabled, reason_catalog, chat, model, turn, record, resolve_product)
from tests.test_actor_attribution import actors, client
from tests.test_write_tickets import headers, payload
from tests.test_write_tickets_part2 import items, body as action_body

pytestmark = pytest.mark.db


def replies():
    return importlib.import_module('assistant_replies')


@pytest.mark.parametrize('text,lang', [('How much stock do we have?', 'en'), ('¿Cuánto inventario tenemos?', 'es')])
def test_inventory_answer_precedes_structured_rows(client, chat, items, monkeypatch, text, lang):
    model(monkeypatch, ('inventory_lookup', {'q': items['batch']['name']}))
    r = turn(client, chat, text)
    assert r.status_code == 200, r.text
    c = r.json()['cards'][0]
    assert c['kind'] == 'read' and c['language'] == lang
    assert c['rows'][0]['total_on_hand'] == 100
    assert '100 lb' in c['summary'] and items['batch']['name'] in c['summary']
    assert ('en 1 lote' if lang == 'es' else 'in 1 lot') in c['summary']


@pytest.mark.parametrize('text,lang', [('How much is in lot ', 'en'), ('¿Cuánto hay en el lote ', 'es')])
def test_exact_lot_question_reads_quantity(client, chat, items, monkeypatch, text, lang):
    lot = items['ingredient']['lot_code']
    model(monkeypatch, ('lot_lookup', {'lot_code': lot}))
    r = turn(client, chat, text + lot)
    assert r.status_code == 200, r.text
    c = r.json()['cards'][0]
    assert c['rows'][0]['quantity_on_hand'] == 100
    assert lot in c['summary'] and '100 lb' in c['summary'] and c['language'] == lang


@pytest.mark.parametrize('spanish', [False, True])
@pytest.mark.parametrize('filter_kind', ['none', 'customer', 'product'])
def test_open_orders_plain_answer_items_filters_and_no_write(client, chat, items, db_cursor, monkeypatch, spanish, filter_kind):
    token = uuid4().hex[:8]
    db_cursor.execute('INSERT INTO customers(name,active) VALUES (%s,true) RETURNING id', ('Sunshine '+token,))
    customer = db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO sales_orders(order_number,customer_id,status,state,requested_ship_date) VALUES (%s,%s,'confirmed','open',CURRENT_DATE) RETURNING id", ('SO-B-'+token,customer))
    oid = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb) VALUES (%s,%s,10)', (oid,items['finished']['id']))
    db_cursor.execute("SELECT count(*) AS n FROM transactions")
    before = db_cursor.fetchone()['n']
    args = {'customer': 'Sunshine '+token if filter_kind=='customer' else None,
            'product': items['finished']['name'] if filter_kind=='product' else None}
    model(monkeypatch, ('open_orders', args))
    r = turn(client,chat,'Muestra los pedidos abiertos' if spanish else 'What orders are open?')
    assert r.status_code == 200, r.text
    c=r.json()['cards'][0]
    assert c['kind']=='read' and c['language']==('es' if spanish else 'en')
    found=next(o for o in c['rows'] if o['order_id']==oid)
    assert found['items'][0]['product']==items['finished']['name']
    assert found['customer']=='Sunshine '+token and found['ship_date'] and found['status']
    assert ('pedido abierto' if spanish else 'open order') in c['summary']
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    assert db_cursor.fetchone()['n']==before


@pytest.mark.parametrize('text,lang', [('We received 10 lb of this ingredient', 'en'), ('Recibimos 10 lb de este ingrediente', 'es')])
def test_missing_receive_keeps_weight_and_offers_real_suppliers(client, chat, payload, db_cursor, monkeypatch, text, lang):
    args={'product_id':payload['product_id'],'supplier_id':None,'cases':None,'case_size_lb':None,'bol_reference':None}
    model(monkeypatch,resolve_product(db_cursor,payload['product_id']),('prepare_receive',args))
    r=turn(client,chat,text)
    assert r.status_code==200,r.text
    c=r.json()['cards'][0]
    assert c['kind']=='missing_facts' and 'supplier_id' in c['fields']
    assert c['known_facts']['quantity_lb']==10 and '10 lb' in c['summary']
    assert c['supplier_choices'] and c['language']==lang
    supplier=next(s for s in c['supplier_choices'] if s['id']==payload['supplier_id'])
    # A supplier choice has the same provenance as a normal resolver button.
    complete={**payload,'cases':1,'case_size_lb':10}
    model(monkeypatch,('prepare_receive',complete))
    chosen=turn(client,chat,choice_id=c['choice_id'],selected_id=supplier['id'])
    assert chosen.status_code==200,chosen.text
    assert chosen.json()['cards'][0]['prepared']['draft']['total_lb']==10
    assert 'SUPPLIER_REQUIRED' not in r.text and '/resolve' not in r.text


@pytest.mark.parametrize('spanish', [False, True])
def test_unknown_product_never_gives_empty_choice_card(client,chat,monkeypatch,spanish):
    model(monkeypatch,('resolve',{'kind':'product','query':'ZZZ-ABSENT-ROUNDB'}))
    r=turn(client,chat,'Recibimos producto desconocido' if spanish else 'Receive unknown product')
    assert r.status_code==200,r.text
    c=r.json()['cards'][0]
    assert c['kind']=='clarify' and 'ZZZ-ABSENT-ROUNDB' in c['message']
    assert c['language']==('es' if spanish else 'en')
    assert ('No reconocí' if spanish else 'I did not recognize') in c['message']


@pytest.mark.parametrize('with_lot', [False, True])
@pytest.mark.parametrize('spanish', [False, True])
def test_packing_uses_real_parent_or_batch_lot_then_filled_draft(client,chat,items,db_cursor,monkeypatch,with_lot,spanish):
    target=items['finished']; source=items['batch']
    args={'source_product_id':source['id'],'target_product_id':target['id'],'cases':10}
    if with_lot:
        args['lot_allocations']=[{'lot_id':source['lot_id'],'quantity_lb':50}]
    model(monkeypatch,resolve_product(db_cursor,target['id']),
          ('packing_context',{'target_product_id':target['id'],'lot_code':source['lot_code'] if with_lot else None}),
          ('prepare_pack',args))
    r=turn(client,chat,('Empacamos 10 cajas de ' if spanish else 'Packed 10 cases of ')+target['name']+
           ((' del lote ' if spanish else ' from batch ')+source['lot_code'] if with_lot else ''))
    assert r.status_code==200,r.text
    c=r.json()['cards'][0]
    assert c['kind']=='draft',c
    draft=c['prepared']['draft']
    assert draft['total_lb']==50 and draft['cases']==10
    assert draft['input_plan'][0]['lot_id']==source['lot_id']
    assert {b['code'] for b in c['prepared']['blockers']}=={'LOT_NOT_CONFIRMED'}


@pytest.mark.parametrize('spanish', [False, True])
def test_resolved_make_goes_straight_to_filled_draft_still_needs_tags(client,chat,items,db_cursor,monkeypatch,spanish):
    monkeypatch.setattr(main,'get_sibling_skus',lambda *a:[{'name':'Sibling'}])
    p=items['batch']
    model(monkeypatch,('resolve',{'kind':'product','query':p['name'],'context':{'action':'make'}}),
          ('prepare_make',{'product_id':p['id'],'batches':1}))
    r=turn(client,chat,('Hicimos 1 tanda de ' if spanish else 'Made 1 batch of ')+p['name'])
    assert r.status_code==200,r.text
    c=r.json()['cards'][0]; d=c['prepared']['draft']
    assert d['total_output_lb']==10 and d['product_name']==p['name']
    assert {b['code'] for b in c['prepared']['blockers']}=={'LOT_NOT_CONFIRMED'}
    group=d['lot_confirmation_group']
    assert group['instruction']==('Escriba los últimos 4 caracteres de la etiqueta de cada bolsa que usó.' if spanish else 'Type the last 4 characters from the tag on each bag you used.')
    assert record(client,chat,c).status_code==422


@pytest.mark.parametrize('spanish', [False,True])
def test_scope_copy_and_permanent_refusal_never_suggest_retry(client,chat,monkeypatch,spanish):
    model(monkeypatch,('unsupported_request',{}))
    r=turn(client,chat,'¿Cuánto es dos más dos?' if spanish else 'What is two plus two?')
    c=r.json()['cards'][0]
    assert ('Solo puedo' if spanish else 'I can only') in c['message']
    assert 'ChatGPT' not in r.text
    c=replies().decorate({'kind':'refusal','result':{'detail':{'error_code':'ORDER_ALREADY_FULFILLED'}}},'es' if spanish else 'en')
    assert c['message']==('Este pedido ya está completamente enviado.' if spanish else 'This order is already fully shipped.')
    assert 'try again' not in c['message']


@pytest.mark.parametrize('tool,data',[
    ('today_entries',{'receipts':[]}), ('receipt_lookup',{'receipt_number':'RCV-261015-999','status':'committed'}),
    ('shift_summary',{'receipts':[]})])
@pytest.mark.parametrize('lang',['en','es'])
def test_every_other_read_contract_starts_with_a_sentence(tool,data,lang):
    c=replies().read_card(tool,data,lang)
    assert c['summary'].endswith('.') and isinstance(c['rows'],list)
    assert c['message']==c['summary'] and c['language']==lang


@pytest.mark.parametrize('query_kind',['name','id','sku'])
def test_open_order_product_filter_follows_batch_parent_to_finished_goods(client,chat,items,db_cursor,monkeypatch,query_kind):
    batch=items['batch']; finished=items['finished']
    db_cursor.execute("UPDATE products SET name='Batch Classic Granola #9',odoo_code='90002' WHERE id=%s",(batch['id'],))
    db_cursor.execute("UPDATE products SET name='Granola Classic 25 LB',odoo_code='B-70013' WHERE id=%s",(finished['id'],))
    db_cursor.execute("INSERT INTO customers(name) VALUES ('Round B Family Customer') RETURNING id")
    customer=db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO sales_orders(order_number,customer_id,status,state) VALUES ('SO-B-FAMILY',%s,'confirmed','open') RETURNING id",(customer,))
    oid=db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb) VALUES (%s,%s,10)',(oid,finished['id']))
    query={'name':'Classic #9','id':str(batch['id']),'sku':'B-70013'}[query_kind]
    model(monkeypatch,('open_orders',{'product':query}))
    r=turn(client,chat,'Show open orders for this product')
    assert r.status_code==200,r.text
    assert [o['order_id'] for o in r.json()['cards'][0]['rows']]==[oid]


def test_open_orders_exact_customer_does_not_include_another_customer_with_same_prefix(client,chat,items,db_cursor,monkeypatch):
    ids=[]
    for i,name in enumerate(['Setton Farms','Setton Farms East']):
        db_cursor.execute('INSERT INTO customers(name) VALUES (%s) RETURNING id',(name,))
        cid=db_cursor.fetchone()['id']
        db_cursor.execute("INSERT INTO sales_orders(order_number,customer_id,status,state) VALUES (%s,%s,'confirmed','open') RETURNING id",('SO-B-PREFIX-'+str(i),cid))
        oid=db_cursor.fetchone()['id'];ids.append(oid)
        db_cursor.execute('INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb) VALUES (%s,%s,10)',(oid,items['finished']['id']))
    model(monkeypatch,('open_orders',{'customer':'Setton Farms'}))
    r=turn(client,chat,'Show open orders for Setton Farms')
    assert r.status_code==200,r.text
    assert [o['order_id'] for o in r.json()['cards'][0]['rows']]==[ids[0]]


@pytest.mark.parametrize('explicit_weight',[False,True])
def test_pack_case_weight_comes_from_catalog_or_worker_not_model(client,chat,items,db_cursor,monkeypatch,explicit_weight):
    p=items['finished'];b=items['batch']
    model(monkeypatch,resolve_product(db_cursor,p['id']),
          ('packing_context',{'target_product_id':p['id'],'lot_code':b['lot_code']}),
          ('prepare_pack',{'target_product_id':p['id'],'source_product_id':b['id'],'cases':2,'case_weight_lb':99}))
    r=turn(client,chat,'Packed 2 cases of '+p['name']+(' 7 lb per case' if explicit_weight else ''))
    assert r.status_code==200,r.text
    d=r.json()['cards'][0]['prepared']['draft']
    assert d['total_lb']==(14 if explicit_weight else 10)
    assert d['input_plan'][0]['quantity_lb']==(14 if explicit_weight else 10)
