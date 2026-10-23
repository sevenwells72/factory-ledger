"""Round D business wording, real route order reads and physical packing gates."""
import pytest
import assistant_voice
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import enabled, reason_catalog, chat, model, turn, record  # noqa: F401
from tests.test_write_tickets_part2 import items  # noqa: F401
from tests.test_assistant_pack_review_c import pack_lots  # noqa: F401
from tests.test_assistant_voice_e2e_round_c import voice_catalog, card  # noqa: F401

pytestmark = pytest.mark.db


@pytest.mark.parametrize('raw', [
    'S O two six one oh one five dash zero zero one', 'SO 261015 001',
    'ese o dos seis uno cero uno cinco guion cero cero uno',
    'S O dos seis uno cero uno cinco guión cero cero uno', 'SO-261015-001',
])
def test_spoken_order_code_exact_transcripts(raw):
    assert assistant_voice.normalize_voice_text(raw) == 'SO-261015-001'


@pytest.mark.parametrize('code', ['26-10-15-GRAI-003', 'OPEN261014-P1585', 'B26-1015-004', 'SO-261015-001'])
def test_order_normalization_keeps_literal_codes(code):
    assert assistant_voice.normalize_voice_text('Ship ' + code) == 'Ship ' + code


@pytest.fixture
def orders(items, db_cursor):
    rows = {}
    db_cursor.execute("INSERT INTO lots(product_id,lot_code) VALUES (%s,'D-FINISHED-001') RETURNING id", (items['finished']['id'],))
    finished_lot = db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO transactions(type,notes) VALUES ('receive','D finished stock') RETURNING id")
    received = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,1000)', (received,items['finished']['id'],finished_lot))
    for customer, number, quantity in [('Sunshine Granola', 'SO-261015-001', 50), ('Setton Farms', 'SO-261015-002', 25), ('STG-D cleanup fixture', 'SO-261014-001', 50)]:
        db_cursor.execute('INSERT INTO customers(name,active) VALUES (%s,true) RETURNING id', (customer,))
        cid = db_cursor.fetchone()['id']
        db_cursor.execute("INSERT INTO sales_orders(order_number,customer_id,state,status) VALUES (%s,%s,'open','confirmed') RETURNING id", (number, cid))
        oid = db_cursor.fetchone()['id']
        db_cursor.execute("INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb,ordered_quantity,ordered_unit,ordered_case_weight_lb) VALUES (%s,%s,%s,%s,'cases',50) RETURNING id", (oid, items['finished']['id'], quantity, quantity/50))
        line = db_cursor.fetchone()['id']
        rows[customer] = {'id': oid, 'customer_id': cid, 'line': line, 'number': number}
    shipped = rows['STG-D cleanup fixture']
    db_cursor.execute("INSERT INTO transactions(type,notes) VALUES ('ship','Round D fully shipped effective row') RETURNING id")
    tid = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,-50)', (tid, items['finished']['id'], finished_lot))
    db_cursor.execute('INSERT INTO sales_order_shipments(sales_order_line_id,transaction_id,quantity_lb) VALUES (%s,%s,50)', (shipped['line'], tid))
    db_cursor.execute("UPDATE sales_order_lines SET quantity_shipped_lb=50,line_status='fulfilled' WHERE id=%s", (shipped['line'],))
    return rows


@pytest.mark.parametrize('spanish', [False, True])
def test_open_orders_excludes_nothing_left_without_closing_and_exposes_effective_case_weight(client, chat, orders, db_cursor, monkeypatch, spanish):
    model(monkeypatch, ('open_orders', {'customer': None, 'product': None}))
    result = card(turn(client, chat, '¿Qué pedidos están abiertos?' if spanish else 'What orders are open?'), 'read')
    assert {row['order_number'] for row in result['rows']} == {'SO-261015-001', 'SO-261015-002'}
    row = next(row for row in result['rows'] if row['order_number'] == 'SO-261015-001')
    assert row['items'][0]['remaining_lb'] == 50 and row['items'][0]['case_size_lb'] == 50
    assert result['result']['nothing_left_to_ship'] == [{'order_number':'SO-261014-001', 'customer':'STG-D cleanup fixture'}]
    db_cursor.execute('SELECT state FROM sales_orders WHERE id=%s', (orders['STG-D cleanup fixture']['id'],))
    assert db_cursor.fetchone()['state'] == 'open'


@pytest.mark.parametrize('text,query', [
    ('Ship S O two six one oh one five dash zero zero one.', 'S O two six one oh one five dash zero zero one'),
    ('Ship SO 261015 001.', 'SO 261015 001'),
    ('Envía ese o dos seis uno cero uno cinco guion cero cero uno.', 'ese o dos seis uno cero uno cinco guion cero cero uno'),
    ('Ship the Sunshine Granola order.', 'the Sunshine Granola order'),
    ('Ship the order for Setton Farms.', 'order for Setton Farms'),
    ('Envía el pedido de Setton Farms.', 'el pedido de Setton Farms'),
])
def test_ship_voice_or_customer_resolves_order_and_never_asks_for_sku(client, chat, orders, monkeypatch, text, query):
    target = orders['Setton Farms' if 'Setton' in text else 'Sunshine Granola']
    calls = [] if 'SO' in text or 'S O' in text or 'ese o' in text else [('resolve', {'kind':'order', 'query':query, 'context':{'action':'ship'}})]
    requests = model(monkeypatch, *calls,
        ('prepare_ship_order', {'order_id':target['id'], 'ship_all':None, 'lines':None, 'bol_reference':None}))
    result = card(turn(client, chat, text), 'missing_facts')
    assert result['shipping_quantity_needed'] is True
    assert result['order']['order_number'] == target['number']
    quantity = ('0.5 cases (25 lb)' if 'Setton' in text else '1 case (50 lb)')
    if text.startswith('Envía'):
        quantity = quantity.replace('cases', 'cajas').replace('case', 'caja')
    assert quantity in result['message']
    assert 'already fully shipped' not in result['message'] and 'completamente enviado' not in result['message']
    assert 'SKU' not in result['message'] if result.get('message') else True
    assert any('order_detail' in (row.get('output','') + row.get('content','')) for req in requests for row in req['input'])


@pytest.mark.parametrize('spanish', [False, True])
def test_shipping_question_uses_effective_physical_remaining_and_keeps_bulk_in_pounds(client, chat, orders, items, db_cursor, monkeypatch, spanish):
    target = orders['Sunshine Granola']
    # A stale recorded count must not hide pounds restored by a voided shipment.
    db_cursor.execute('UPDATE sales_order_lines SET quantity_shipped_lb=25 WHERE id=%s', (target['line'],))
    db_cursor.execute('UPDATE products SET case_size_lb=NULL WHERE id=%s', (items['batch']['id'],))
    db_cursor.execute('INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb) VALUES (%s,%s,12.5)', (target['id'], items['batch']['id']))
    db_cursor.execute("INSERT INTO products(name,type,uom,case_size_lb,is_service,active) VALUES ('Keepalive freight','finished','each',5,true,true) RETURNING id")
    service_id = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb) VALUES (%s,%s,500)', (target['id'], service_id))
    db_cursor.execute("INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb,line_status) VALUES (%s,%s,100,'cancelled')", (target['id'], items['finished']['id']))
    model(monkeypatch, ('prepare_ship_order', {'order_id':target['id'], 'ship_all':None, 'lines':None, 'bol_reference':None}))
    result = card(turn(client, chat, ('Envía el pedido ' if spanish else 'Ship order ') + target['number']), 'missing_facts')
    assert result['order']['remaining_lb'] == 62.5
    assert result['order']['items'] == [{'remaining_lb': 50, 'case_size_lb': 50}, {'remaining_lb': 12.5, 'case_size_lb': None}]
    assert ('1 caja (50 lb); 12.5 lb' if spanish else '1 case (50 lb); 12.5 lb') in result['message']


@pytest.mark.parametrize('spanish', [False, True])
def test_missing_order_uses_customer_open_choices_and_order_copy(client, chat, orders, monkeypatch, spanish):
    model(monkeypatch, ('resolve', {'kind':'customer','query':'Setton Farms','context':{'action':'ship'}}),
        ('resolve', {'kind':'order','query':'SO-261015-999','context':{'action':'ship','customer_id':orders['Setton Farms']['customer_id']}}))
    result = card(turn(client, chat, 'Envía el pedido SO-261015-999 de Setton Farms' if spanish else 'Ship order SO-261015-999 for Setton Farms'), 'choices')
    assert result['order_not_found'] is True and result['resolution_kind'] == 'order'
    assert [row['id'] for row in result['result']['candidates']] == [orders['Setton Farms']['id']]
    assert 'SKU' not in result['message']
    assert ('No encontré el pedido' if spanish else "I couldn't find order") in result['message']


@pytest.mark.parametrize('spanish', [False, True])
def test_missing_order_without_customer_never_asks_for_product(client, chat, orders, monkeypatch, spanish):
    model(monkeypatch, ('resolve', {'kind':'order','query':'SO-261015-999','context':{'action':'ship'}}))
    result = card(turn(client, chat, 'Envía SO-261015-999' if spanish else 'Ship SO-261015-999'), 'clarify')
    assert 'SKU' not in result['message'] and 'SO-261015-999' in result['message']


@pytest.mark.parametrize('spanish', [False, True])
def test_multiple_customer_open_orders_require_human_button(client, chat, orders, items, db_cursor, monkeypatch, spanish):
    db_cursor.execute("INSERT INTO sales_orders(order_number,customer_id,state,status) VALUES ('SO-261015-004',%s,'open','confirmed') RETURNING id", (orders['Setton Farms']['customer_id'],))
    second = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb) VALUES (%s,%s,25)', (second, items['finished']['id']))
    model(monkeypatch, ('resolve', {'kind':'order','query':'Setton Farms','context':{'action':'ship'}}))
    result = card(turn(client, chat, 'Envía el pedido de Setton Farms' if spanish else 'Ship the order for Setton Farms'), 'choices')
    assert {row['id'] for row in result['result']['candidates']} == {second, orders['Setton Farms']['id']}


@pytest.mark.parametrize('spanish', [False, True])
def test_missing_pack_batch_offers_parent_fifo_plus_different_batch_and_never_confirms(client, chat, voice_catalog, pack_lots, monkeypatch, spanish):
    source, target = voice_catalog['fruit_batch'], voice_catalog['fruit']
    model(monkeypatch, ('resolve', {'kind':'product','query':target['name'],'context':{'action':'pack'}}),
          ('packing_context', {'target_product_id':target['id'],'lot_code':None}))
    choices = card(turn(client, chat, 'Empacamos 40 cajas de ' + target['name'] if spanish else 'Packed 40 cases of ' + target['name']), 'choices')
    assert choices['packing_lots'] is True
    assert [row['id'] for row in choices['result']['candidates']] == [pack_lots['first_id'],pack_lots['later_id']]
    assert ('¿De qué lote a granel salieron?' if spanish else 'Which batch did these come from?') in choices['message']
    model(monkeypatch, ('prepare_pack', {'source_product_id':source['id'],'target_product_id':target['id'],'cases':40}))
    draft = card(turn(client, chat, choice_id=choices['id'], selected_id=pack_lots['later_id']), 'draft')
    assert draft['prepared']['draft']['input_plan'][0]['lot_id'] == pack_lots['later_id']
    assert all(not row.get('confirmed') for row in draft['prepared']['draft']['input_plan'])
    assert record(client, chat, draft).status_code == 422


@pytest.mark.parametrize('spanish', [False, True])
def test_different_batch_asks_for_tag_without_source_product_or_sku(client, chat, voice_catalog, pack_lots, monkeypatch, spanish):
    target = voice_catalog['fruit']
    model(monkeypatch, ('resolve', {'kind':'product','query':target['name'],'context':{'action':'pack'}}),
          ('packing_context', {'target_product_id':target['id'],'lot_code':None}))
    choices = card(turn(client, chat, 'Empacamos 40 cajas' if spanish else 'Packed 40 cases'), 'choices')
    result = card(turn(client, chat, choice_id=choices['id'], selected_id='different_batch'), 'missing_facts')
    assert result['packing_batch_needed'] is True
    assert ('Which batch' if not spanish else '¿De qué lote') in result['message']
    assert 'source product' not in result['message'].lower() and 'SKU' not in result['message']


@pytest.mark.parametrize('spanish', [False, True])
def test_exact_coconut_transcript_chooses_actual_finished_output_then_parent_batch(client, chat, db_cursor, monkeypatch, spanish):
    catalog = {}
    for key,name,kind,size in [('raw','Sweetened Flake Coconut','ingredient',None),('bulk','Batch Coconut Sweetened Flake','batch',None),('ten','Coconut Sweetened Flake CNS 10 LB','finished',10),('twentyfive','Coconut Sweetened Flake CNS 25 LB','finished',25)]:
        db_cursor.execute("INSERT INTO products(name,type,uom,case_size_lb,active) VALUES (%s,%s,'lb',%s,true) RETURNING id", (name,kind,size))
        catalog[key] = db_cursor.fetchone()['id']
    db_cursor.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=ANY(%s)', (catalog['bulk'],[catalog['ten'],catalog['twentyfive']]))
    db_cursor.execute("INSERT INTO lots(product_id,lot_code) VALUES (%s,'B26-1015-004') RETURNING id", (catalog['bulk'],))
    lot = db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO transactions(type,notes) VALUES ('receive','D coconut real-catalog shape') RETURNING id")
    tid=db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,1000)',(tid,catalog['bulk'],lot))
    model(monkeypatch, ('resolve',{'kind':'product','query':'coco en hojuelas endulzado' if spanish else 'Sweetened Flake Coconut','context':{'action':'pack'}}),
          ('packing_context',{'target_product_id':catalog['raw'],'lot_code':None}))
    batches = card(turn(client, chat, 'Empacamos diez cajas de coco en hojuelas endulzado.' if spanish else 'Packed 10 cases of sweetened flake coconut.'), 'choices')
    assert batches['packing_lots'] is True and [row['id'] for row in batches['result']['candidates']] == [lot]
    outputs = card(turn(client, chat, choice_id=batches['id'], selected_id=lot), 'choices')
    assert outputs['packing_outputs'] is True and {row['id'] for row in outputs['result']['candidates']} == {catalog['ten'],catalog['twentyfive']}
    assert 'source product' not in str(outputs).lower() and 'source product' not in str(batches).lower()
    requests = model(monkeypatch, ('prepare_pack', {'source_product_id':catalog['bulk'], 'target_product_id':catalog['ten'], 'cases':10}))
    draft = card(turn(client, chat, choice_id=outputs['id'], selected_id=catalog['ten']), 'draft')
    assert draft['prepared']['draft']['total_lb'] == 100
    assert draft['prepared']['draft']['input_plan'][0]['lot_id'] == lot
    assert all(not row.get('confirmed') for row in draft['prepared']['draft']['input_plan'])
    assert any('I selected this FL candidate:' in row.get('content','') and 'Coconut Sweetened Flake CNS 10 LB' in row.get('content','') for req in requests for row in req['input'])

@pytest.mark.parametrize('spanish', [False, True])
def test_spoken_so_mistaken_product_tool_still_resolves_order(client, chat, orders, monkeypatch, spanish):
    target = orders['Sunshine Granola']
    requests = model(monkeypatch, ('prepare_ship_order', {'order_id':target['id'],'ship_all':True,'bol_reference':None}))
    result = card(turn(client, chat, 'Envía SO-261015-001' if spanish else 'Ship SO-261015-001'), 'draft')
    assert result['prepared']['draft']['order'] == target['number']
    assert {tool['name'] for tool in requests[0]['tools']} == {'prepare_ship_order'}


@pytest.mark.parametrize('spanish', [False, True])
def test_parent_with_no_available_lots_still_offers_different_batch(client, chat, voice_catalog, db_cursor, monkeypatch, spanish):
    target, source = voice_catalog['fruit'], voice_catalog['fruit_batch']
    db_cursor.execute("INSERT INTO transactions(type,notes) VALUES ('adjust','D depleted batch fixture') RETURNING id")
    tid = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,-1000)', (tid,source['id'],source['lot_id']))
    model(monkeypatch, ('resolve', {'kind':'product','query':target['name'],'context':{'action':'pack'}}),
          ('packing_context', {'target_product_id':target['id'],'lot_code':None}))
    result = card(turn(client, chat, 'Empacamos 10 cajas' if spanish else 'Packed 10 cases'), 'choices')
    assert result['packing_lots'] is True and result['result']['candidates'] == []
    prompt = card(turn(client, chat, choice_id=result['id'], selected_id='different_batch'), 'missing_facts')
    assert prompt['packing_batch_needed'] is True

@pytest.mark.parametrize('spanish', [False, True])
def test_missing_order_recovers_customer_from_original_transcript_when_model_omits_it(client, chat, orders, monkeypatch, spanish):
    model(monkeypatch, ('resolve', {'kind':'order','query':'SO-999999-999','context':{'action':'ship'}}))
    result = card(turn(client, chat, 'Envía el pedido de Sunshine Granola SO-999999-999.' if spanish else 'Ship the Sunshine Granola order SO-999999-999.'), 'choices')
    assert result['order_not_found'] is True
    assert [row['id'] for row in result['result']['candidates']] == [orders['Sunshine Granola']['id']]
    assert 'SKU' not in result['message']


@pytest.mark.parametrize('spanish', [False, True])
def test_explicit_ship_all_advances_to_bol_draft_with_no_record(client, chat, orders, monkeypatch, spanish):
    target = orders['Sunshine Granola']
    model(monkeypatch, ('prepare_ship_order', {'order_id':target['id'],'ship_all':True,'bol_reference':None}))
    result = card(turn(client, chat, 'Envía todo lo pendiente del pedido SO-261015-001.' if spanish else 'Ship all remaining on order SO-261015-001.'), 'draft')
    assert result['prepared']['draft']['order'] == target['number']
    assert result['prepared']['can_commit'] is False
    assert 'BOL_REQUIRED' in {row['code'] for row in result['prepared']['blockers']}


@pytest.mark.parametrize('raw', ['SO 26-10-15-001','SO 261015-00-1'])
def test_near_literal_order_separators_stay_opaque(raw):
    assert assistant_voice.normalize_order_code(raw) == raw

@pytest.mark.parametrize('spanish', [False, True])
@pytest.mark.parametrize('source_supplied', [False, True])
def test_model_skipping_packing_context_still_requires_parent_batch_choice(client, chat, voice_catalog, pack_lots, db_cursor, monkeypatch, spanish, source_supplied):
    source, target = voice_catalog['fruit_batch'], voice_catalog['fruit']
    model(monkeypatch,
        ('resolve', {'kind':'product','query':source['name'],'context':{'action':'pack'}}),
        ('resolve', {'kind':'product','query':target['name'],'context':{'action':'pack'}}),
        ('prepare_pack', {'source_product_id':source['id'] if source_supplied else None, 'target_product_id':target['id'], 'cases':40}))
    result = card(turn(client, chat, 'Empacamos 40 cajas de ' + target['name'] if spanish else 'Packed 40 cases of ' + target['name']), 'choices')
    assert result['packing_lots'] is True
    assert [row['id'] for row in result['result']['candidates']] == [pack_lots['first_id'],pack_lots['later_id']]
    db_cursor.execute('SELECT count(*) AS n FROM assistant_drafts')
    assert db_cursor.fetchone()['n'] == 0


@pytest.mark.parametrize('spanish', [False, True])
def test_model_supplied_lot_id_without_worker_lot_is_not_batch_evidence(client, chat, voice_catalog, pack_lots, monkeypatch, spanish):
    source, target = voice_catalog['fruit_batch'], voice_catalog['fruit']
    model(monkeypatch,
        ('resolve', {'kind':'product','query':source['name'],'context':{'action':'pack'}}),
        ('resolve', {'kind':'product','query':target['name'],'context':{'action':'pack'}}),
        ('resolve', {'kind':'lot','query':source['lot_code']}),
        ('prepare_pack', {'source_product_id':source['id'],'target_product_id':target['id'],'cases':40,
                          'lot_allocations':[{'lot_id':pack_lots['later_id'],'quantity_lb':200}]}))
    result = card(turn(client, chat, 'Empacamos 40 cajas' if spanish else 'Packed 40 cases'), 'choices')
    assert result['packing_lots'] is True


@pytest.mark.parametrize('spanish', [False, True])
@pytest.mark.parametrize('allocation_supplied', [False, True])
def test_model_skipping_context_keeps_explicit_worker_lot_without_confirming_it(client, chat, voice_catalog, pack_lots, monkeypatch, spanish, allocation_supplied):
    source, target = voice_catalog['fruit_batch'], voice_catalog['fruit']
    args = {'source_product_id':source['id'],'target_product_id':target['id'],'cases':40}
    if allocation_supplied:
        args['lot_allocations'] = [{'lot_id':pack_lots['later_id'],'quantity_lb':200}]
    model(monkeypatch,
        ('resolve', {'kind':'product','query':source['name'],'context':{'action':'pack'}}),
        ('resolve', {'kind':'product','query':target['name'],'context':{'action':'pack'}}),
        ('resolve', {'kind':'lot','query':source['lot_code']}), ('prepare_pack',args))
    result = card(turn(client, chat, ('Empacamos 40 cajas del lote ' if spanish else 'Packed 40 cases from batch lot ') + source['lot_code']), 'draft')
    assert result['prepared']['draft']['input_plan'][0]['lot_id'] == pack_lots['later_id']
    assert all(not row.get('confirmed') for row in result['prepared']['draft']['input_plan'])
    assert record(client,chat,result).status_code == 422


@pytest.mark.parametrize('spanish', [False, True])
def test_opaque_so_like_identifier_is_order_not_found_not_customer_sku(client, chat, orders, monkeypatch, spanish):
    code = 'SO 26-10-15-001'
    model(monkeypatch, ('resolve', {'kind':'order','query':code,'context':{'action':'ship'}}))
    result = card(turn(client, chat, ('Envía ' if spanish else 'Ship ') + code), 'clarify')
    assert result['resolution_kind'] == 'order' and code in result['message']
    assert 'SKU' not in result['message']


@pytest.mark.parametrize('spanish', [False, True])
def test_wrong_product_tool_for_customer_order_still_finds_order(client, chat, orders, monkeypatch, spanish):
    target = orders['Sunshine Granola']
    model(monkeypatch, ('resolve', {'kind':'product','query':'Sunshine Granola','context':{'action':'ship'}}),
          ('prepare_ship_order', {'order_id':target['id'],'ship_all':None,'lines':None,'bol_reference':None}))
    result = card(turn(client, chat, 'Envía el pedido de Sunshine Granola.' if spanish else 'Ship the Sunshine Granola order.'), 'missing_facts')
    assert result['shipping_quantity_needed'] is True and result['order']['order_number'] == target['number']
    assert 'SKU' not in result['message']


@pytest.mark.parametrize('raw,expected', [
    ('Ship SO 261015 001 10 cases.', 'Ship SO-261015-001 10 cases.'),
    ('Ship SO261015001 10 lb.', 'Ship SO261015001 10 lb.'),
    ('Envía SO 261015 001 10 cajas.', 'Envía SO-261015-001 10 cases.'),
    ('Ship S O two six one oh one five dash zero zero one ten cases.', 'Ship SO-261015-001 10 cases.'),
])
def test_order_code_does_not_absorb_following_shipment_quantity(raw, expected):
    assert assistant_voice.normalize_voice_text(raw) == expected
