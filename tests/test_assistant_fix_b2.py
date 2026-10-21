"""Round B review fixes exercised through the real Assistant/ledger handlers."""
from uuid import uuid4
import pytest
import assistant_replies as replies
import fl_assistant
import resolution
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import enabled, reason_catalog, chat, model, turn, resolve_product, make_draft  # noqa: F401
from tests.test_write_tickets import payload  # noqa: F401
from tests.test_write_tickets_part2 import items  # noqa: F401
from tests.test_assistant_fact_binding_round_b import missing_receive, supplier_call

pytestmark = pytest.mark.db


def order(cur, product_id, number=None):
    name = 'Round B2 Customer ' + uuid4().hex[:8]
    cur.execute('INSERT INTO customers(name) VALUES (%s) RETURNING id', (name,))
    customer = cur.fetchone()['id']
    cur.execute("INSERT INTO sales_orders(order_number,customer_id,state,status) VALUES (%s,%s,'open','confirmed') RETURNING id",
                (number or 'SO-B2-' + uuid4().hex[:8], customer))
    oid = cur.fetchone()['id']
    cur.execute('INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb) VALUES (%s,%s,10)', (oid, product_id))
    return oid


@pytest.mark.parametrize('spanish', [False, True])
def test_kookies_ampersand_open_order_filter(client, chat, items, db_cursor, monkeypatch, spanish):
    db_cursor.execute("UPDATE products SET name='Kookies & Kreme – 25 LB',odoo_code='10304' WHERE id=%s", (items['finished']['id'],))
    oid = order(db_cursor, items['finished']['id'], 'SO-261015-002')
    model(monkeypatch, ('open_orders', {'customer': None, 'product': 'Kookies & Kreme'}))
    response = turn(client, chat, '¿Qué pedidos abiertos tienen Kookies & Kreme?' if spanish else 'Which open orders have Kookies & Kreme?')
    card = response.json()['cards'][0]
    assert card['kind'] == 'read', response.text
    assert [row['order_id'] for row in card['rows']] == [oid]
    assert ('1 pedido abierto' if spanish else '1 open order') in card['summary']
    assert ('pedidos abiertos' if spanish else 'open orders') not in card['summary']


@pytest.mark.parametrize('spanish', [False, True])
def test_unknown_product_order_filter_asks_for_product(client, chat, monkeypatch, spanish):
    model(monkeypatch, ('open_orders', {'customer': None, 'product': 'ZZZZ-UNRECOGNIZED-B2'}))
    response = turn(client, chat, '¿Qué pedidos abiertos tienen esto?' if spanish else 'Which open orders have this?')
    card = response.json()['cards'][0]
    assert card['kind'] == 'clarify'
    assert 'ZZZZ-UNRECOGNIZED-B2' in card['message']
    assert ('0 pedidos' if spanish else '0 open orders') not in card['message']


@pytest.mark.parametrize('spanish', [False, True])
def test_explicit_receive_total_correction_updates_current_draft(client, chat, payload, db_cursor, monkeypatch, spanish):
    # The first worker total was wrong; cases and vendor remain known facts.
    model(monkeypatch, resolve_product(db_cursor, payload['product_id']), supplier_call(db_cursor, payload['supplier_id']),
          ('prepare_receive', {**payload, 'cases': 1, 'case_size_lb': 10}))
    initial = turn(client, chat, 'Recibimos 10 lb total' if spanish else 'Received 10 lb total').json()['cards'][0]
    assert initial['prepared']['draft']['total_lb'] == 10
    async def no_model(*args, **kwargs):
        raise AssertionError('A receive total correction uses the current confirmed draft context')
    monkeypatch.setattr(fl_assistant, 'openai', no_model)
    response = turn(client, chat, 'fueron 15 lb en total' if spanish else 'it was 15 lb total')
    assert response.status_code == 200, response.text
    corrected = response.json()['cards'][0]
    assert corrected['kind'] == 'draft'
    assert corrected['prepared']['draft']['total_lb'] == 15
    assert corrected['language'] == ('es' if spanish else 'en')
    assert corrected['supersedes_draft_id'] == initial['id']
    db_cursor.execute('SELECT status FROM assistant_drafts WHERE id=%s', (initial['id'],))
    assert db_cursor.fetchone()['status'] == 'cancelled'
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    assert db_cursor.fetchone()['n'] == 0


@pytest.mark.parametrize('spanish', [False, True])
def test_receive_missing_weight_correction_replaces_old_total(client, chat, payload, db_cursor, monkeypatch, spanish):
    missing_receive(client, chat, payload, db_cursor, monkeypatch, 'Recibimos 10 lb total' if spanish else 'Received 10 lb total')
    complete = {**payload, 'cases': 1, 'case_size_lb': 15}
    model(monkeypatch, supplier_call(db_cursor, payload['supplier_id']), ('prepare_receive', complete))
    response = turn(client, chat, 'eran 15 lb total' if spanish else 'it was 15 lb total')
    assert response.status_code == 200, response.text
    assert response.json()['cards'][0]['prepared']['draft']['total_lb'] == 15


@pytest.mark.parametrize('spanish', [False, True])
def test_exact_full_classic_nine_inventory_never_expands_family(client, chat, items, db_cursor, monkeypatch, spanish):
    batch = items['batch']
    db_cursor.execute("UPDATE products SET name='Batch Classic Granola #9',odoo_code='90002' WHERE id=%s", (batch['id'],))
    db_cursor.execute("INSERT INTO products(name,type,uom,is_service) VALUES ('Batch Classic Fruit Nut Granola #9','batch','lb',false),('Batch SS Classic Granola #9','batch','lb',false)")
    model(monkeypatch, ('inventory_lookup', {'q': 'Batch Classic Granola #9'}))
    response = turn(client, chat, '¿Cuánto hay de Batch Classic Granola #9?' if spanish else 'How much Batch Classic Granola #9?')
    assert response.status_code == 200, response.text
    rows = response.json()['cards'][0]['rows']
    assert [row['product'] for row in rows] == ['Batch Classic Granola #9']


@pytest.mark.parametrize('lang', ['en', 'es'])
@pytest.mark.parametrize('effective,expected', [('voided', 'VOIDED'), ('partially_voided', 'PARTIALLY VOIDED')])
def test_receipt_uses_effective_void_state(lang, effective, expected):
    card = replies.read_card('receipt_lookup', {'receipt_number': 'ADJ-261015-240', 'status': 'committed', 'effective_status': effective}, lang)
    assert (expected if lang == 'en' else 'ANULADO') in card['summary']
    assert ('recorded' if lang == 'en' else 'registrado') not in card['summary']


@pytest.mark.parametrize('spanish', [False, True])
@pytest.mark.parametrize('tool', ['lot_lookup', 'resolve'])
def test_nonexistent_lot_is_plain_permanent_answer(client, chat, monkeypatch, spanish, tool):
    code = 'NO-SUCH-LOT-B2'
    args = {'lot_code': code} if tool == 'lot_lookup' else {'kind': 'lot', 'query': code}
    model(monkeypatch, (tool, args))
    response = turn(client, chat, '¿Cuánto hay en el lote ' + code if spanish else 'How much in lot ' + code)
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['message'] == ('No hay un lote llamado ' if spanish else 'No lot called ') + code + '.'
    assert 'try again' not in response.text.lower()


@pytest.mark.parametrize('spanish', [False, True])
def test_supplier_list_limits_deduplicates_and_searches_real_ids(client, chat, payload, db_cursor, monkeypatch, spanish):
    names = ['Round B2 Supplier %02d' % i for i in range(70)] + ['Round B2 Supplier 00, Inc.', 'A1 Baker', 'A1 Bakery', 'A1 Bakery Supply']
    db_cursor.executemany('INSERT INTO suppliers(name) VALUES (%s)', [(name,) for name in names])
    first = missing_receive(client, chat, payload, db_cursor, monkeypatch,
                            'Recibimos 10 lb total' if spanish else 'Received 10 lb total')
    assert len(first['supplier_choices']) == 6
    assert first['supplier_search'] is True
    async def no_model(*args, **kwargs):
        raise AssertionError('Supplier search uses the supplier catalog directly')
    monkeypatch.setattr(fl_assistant, 'openai', no_model)
    response = turn(client, chat, 'proveedor: Round B2 Supplier 00' if spanish else 'supplier: Round B2 Supplier 00')
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['resolution_kind'] == 'supplier' and card['supplier_search']
    assert len(card['result']['candidates']) == 1
    candidate = card['result']['candidates'][0]
    assert candidate['name'] == 'Round B2 Supplier 00'
    db_cursor.execute("SELECT count(*) AS n FROM suppliers WHERE name IN ('Round B2 Supplier 00','Round B2 Supplier 00, Inc.')")
    assert db_cursor.fetchone()['n'] == 2
    response = turn(client, chat, 'supplier: A1 Bak')
    assert len(response.json()['cards'][0]['result']['candidates']) == 1
    db_cursor.execute("SELECT count(*) AS n FROM suppliers WHERE name IN ('A1 Baker','A1 Bakery','A1 Bakery Supply')")
    assert db_cursor.fetchone()['n'] == 3


@pytest.mark.parametrize('spanish', [False, True])
def test_unverified_trace_debit_is_not_negative_inventory(client, chat, items, db_cursor, monkeypatch, spanish):
    ingredient = items['ingredient']
    code = 'UNVERIFIED-999-P%s-1' % ingredient['id']
    db_cursor.execute("INSERT INTO lots(product_id,lot_code,entry_source) VALUES (%s,%s,'unverified_make_input') RETURNING id", (ingredient['id'], code))
    lot_id = db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO transactions(type,notes) VALUES ('make','unverified actual input') RETURNING id")
    tid = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,-10)', (tid, ingredient['id'], lot_id))
    model(monkeypatch, ('inventory_lookup', {'q': ingredient['name']}))
    response = turn(client, chat, '¿Cuánto inventario tenemos?' if spanish else 'How much inventory?')
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['rows'][0]['total_on_hand'] == 100
    assert all(lot['qty_on_hand'] >= 0 for lot in card['rows'][0]['lots'])
    assert ('pendientes de seguimiento' if spanish else 'unverified, awaiting follow-up') in card['summary']
    model(monkeypatch, ('lot_lookup', {'lot_code': code}))
    lot_response = turn(client, chat, '¿Cuánto hay en el lote ' + code if spanish else 'How much in lot ' + code)
    lot_card = lot_response.json()['cards'][0]
    assert lot_card['rows'][0]['quantity_on_hand'] is None
    assert '-10' not in lot_card['summary']
    assert ('pendiente de seguimiento' if spanish else 'unverified, awaiting follow-up') in lot_card['summary']
    db_cursor.execute('SELECT quantity_lb FROM transaction_lines WHERE transaction_id=%s', (tid,))
    assert db_cursor.fetchone()['quantity_lb'] == -10


def test_spanish_choices_preserve_extra_kosher_catalog_label():
    card = replies.decorate({'kind': 'choices', 'result': {'candidates': [{'id': 1, 'name': 'SS EXTRA-KOSHER', 'label': 'SS EXTRA-KOSHER'}]}}, 'es')
    assert card['result']['candidates'][0]['label'] == 'SS EXTRA-KOSHER'


@pytest.mark.parametrize('tool,args,text', [
    ('inventory_lookup', {'q': 'unused'}, 'How much inventory?'),
    ('unsupported_request', {}, 'What is two plus two?'),
])
def test_unrelated_workflow_clears_receive_correction_context(client, chat, payload, db_cursor, monkeypatch, tool, args, text):
    make_draft(client, chat, payload, db_cursor, monkeypatch)
    model(monkeypatch, (tool, args))
    assert turn(client, chat, text).status_code == 200
    calls = model(monkeypatch, ('unsupported_request', {}))
    response = turn(client, chat, 'actually it was 15 lb total')
    assert response.status_code == 200, response.text
    assert response.json()['cards'][0]['kind'] == 'unsupported_request'
    assert len(calls) == 1


def test_committed_receive_cannot_be_silently_reprepared(client, chat, payload, db_cursor, monkeypatch):
    from tests.test_fl_assistant import record
    initial, _ = make_draft(client, chat, payload, db_cursor, monkeypatch)
    assert record(client, chat, initial).status_code == 200
    async def no_model(*args, **kwargs):
        raise AssertionError('A completed receive cannot become an unreviewed second receive')
    monkeypatch.setattr(fl_assistant, 'openai', no_model)
    response = turn(client, chat, 'actually it was 15 lb total')
    assert response.status_code == 200, response.text
    assert response.json()['cards'][0]['error_code'] == 'RECEIVE_CORRECTION_UNAVAILABLE'
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    assert db_cursor.fetchone()['n'] == 1


@pytest.mark.parametrize('spanish', [False, True])
def test_receive_correction_loses_cleanly_if_record_started_after_snapshot(client, chat, payload, db_cursor, monkeypatch, spanish):
    initial, _ = make_draft(client, chat, payload, db_cursor, monkeypatch)
    original = fl_assistant.draft_row
    def racing_record(api, draft_id, actor_id, **kwargs):
        snapshot = original(api, draft_id, actor_id, **kwargs)
        db_cursor.execute('UPDATE assistant_drafts SET record_started_at=clock_timestamp() WHERE id=%s', (draft_id,))
        return snapshot
    monkeypatch.setattr(fl_assistant, 'draft_row', racing_record)
    response = turn(client, chat, 'fueron 15 lb total' if spanish else 'it was 15 lb total')
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['error_code'] == 'RECEIVE_CORRECTION_UNAVAILABLE'
    db_cursor.execute('SELECT count(*) AS n FROM assistant_drafts WHERE session_id=%s', (chat[0],))
    assert db_cursor.fetchone()['n'] == 1
    db_cursor.execute('SELECT status,record_started_at FROM assistant_drafts WHERE id=%s', (initial['id'],))
    row = db_cursor.fetchone()
    assert row['status'] == 'pending' and row['record_started_at'] is not None


@pytest.mark.parametrize('initial_missing', [False, True])
def test_receive_correction_keeps_original_private_attachment(client, chat, payload, db_cursor, monkeypatch, initial_missing):
    from tests.test_fl_assistant import upload_photo
    photo = upload_photo(client, chat, 'bol')
    if initial_missing:
        args = dict(product_id=payload['product_id'], supplier_id=None, cases=None, case_size_lb=None, bol_reference=None)
        model(monkeypatch, resolve_product(db_cursor, payload['product_id']), ('prepare_receive', args))
    else:
        model(monkeypatch, resolve_product(db_cursor, payload['product_id']), supplier_call(db_cursor, payload['supplier_id']), ('prepare_receive', payload))
    initial = turn(client, chat, 'Received 50 lb total', attachment_ids=[photo]).json()['cards'][0]
    if initial_missing:
        model(monkeypatch, supplier_call(db_cursor, payload['supplier_id']), ('prepare_receive', {**payload, 'case_size_lb': 7.5}))
    response = turn(client, chat, 'it was 15 lb total')
    assert response.status_code == 200, response.text
    corrected = response.json()['cards'][0]
    assert corrected['prepared']['draft']['total_lb'] == 15
    assert corrected['attachment_ids'] == [photo]


@pytest.mark.parametrize('spanish', [False, True])
def test_supplier_choices_rank_actual_product_history_and_vendor_aliases(client, chat, payload, db_cursor, monkeypatch, spanish):
    db_cursor.executemany('INSERT INTO suppliers(name) VALUES (%s)', [('Alphabet Supplier %02d' % i,) for i in range(10)])
    db_cursor.execute("INSERT INTO suppliers(name) VALUES ('ZZ Actual Delivery Vendor'),('YY Catalog Alias Vendor') RETURNING id")
    actual, alias = [row['id'] for row in db_cursor.fetchall()]
    db_cursor.execute("INSERT INTO supplier_product_aliases(supplier_id,vendor_description,product_id) VALUES (%s,'B2 vendor ingredient',%s)", (alias, payload['product_id']))
    db_cursor.execute("INSERT INTO lots(product_id,lot_code) VALUES (%s,'B2-HISTORY') RETURNING id", (payload['product_id'],))
    lot = db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO transactions(type,supplier_id) VALUES ('receive',%s) RETURNING id", (actual,))
    tid = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,5)', (tid, payload['product_id'], lot))
    card = missing_receive(client, chat, payload, db_cursor, monkeypatch, 'Recibimos 10 lb total' if spanish else 'Received 10 lb total')
    assert [row['id'] for row in card['supplier_choices'][:2]] == [actual, alias]
    assert len(card['supplier_choices']) == 6


@pytest.mark.parametrize('text', ['¿Cuál es el estado?', '¿Quién recibió esto?', '¿Puedes ayudar?', 'Donde esta esto', 'Hola'])
def test_spanish_questions_do_not_need_language_button(text):
    assert replies.language(text, 'en') == 'es'


def test_total_word_without_new_weight_does_not_replace_receive(client, chat, payload, db_cursor, monkeypatch):
    initial, _ = make_draft(client, chat, payload, db_cursor, monkeypatch)
    calls = model(monkeypatch, ('open_orders', {'customer': None, 'product': None}))
    response = turn(client, chat, 'What is the total on the open orders?')
    assert response.status_code == 200, response.text
    assert response.json()['cards'][0]['tool'] == 'open_orders'
    assert len(calls) == 1
    db_cursor.execute('SELECT status FROM assistant_drafts WHERE id=%s', (initial['id'],))
    assert db_cursor.fetchone()['status'] == 'pending'


def test_malformed_resolve_context_keeps_bounded_model_repair(client, chat, monkeypatch):
    calls = model(monkeypatch, ('resolve', {'kind': 'product', 'query': 'whatever', 'context': 'invalid'}), ('unsupported_request', {}))
    response = turn(client, chat, 'Check an item')
    assert response.status_code == 200, response.text
    assert response.json()['cards'][0]['kind'] == 'unsupported_request'
    assert len(calls) == 2


@pytest.mark.parametrize('spanish', [False, True])
def test_reviewed_receive_mismatch_can_be_corrected_in_same_chat(client, chat, payload, db_cursor, monkeypatch, spanish):
    initial = missing_receive(client, chat, payload, db_cursor, monkeypatch,
        'Recibimos 10 libras de este ingrediente' if spanish else 'Received 10 lb of this ingredient')
    supplier = next(row for row in initial['supplier_choices'] if row['id'] == payload['supplier_id'])
    model(monkeypatch, ('prepare_receive', {**payload, 'cases': 3, 'case_size_lb': 5}))
    response = turn(client, chat, choice_id=initial['choice_id'], selected_id=supplier['id'])
    assert response.status_code == 200, response.text
    mismatch = response.json()['cards'][0]
    assert mismatch['kind'] == 'missing_facts'
    assert mismatch['known_facts']['quantity_lb'] == 10
    assert mismatch['quantity_mismatch_lb'] == 15
    assert mismatch['known_facts']['cases'] == 3 and mismatch['known_facts']['case_size_lb'] == 5
    db_cursor.execute('SELECT state FROM assistant_sessions WHERE id=%s', (chat[0],))
    state = db_cursor.fetchone()['state']
    assert payload['product_id'] in state['resolved']['product']
    assert supplier['id'] in state['resolved']['supplier']
    db_cursor.execute('SELECT count(*) AS n FROM assistant_drafts WHERE session_id=%s', (chat[0],))
    assert db_cursor.fetchone()['n'] == 0
    async def no_model(*args, **kwargs):
        raise AssertionError('The correction must reuse the approved product and supplier, not model IDs')
    monkeypatch.setattr(fl_assistant, 'openai', no_model)
    corrected = turn(client, chat, 'Perdón, fueron 15 libras en total' if spanish else 'it was 15 lb total')
    assert corrected.status_code == 200, corrected.text
    card = corrected.json()['cards'][0]
    assert card['kind'] == 'draft'
    assert card['language'] == ('es' if spanish else 'en')
    draft = card['prepared']['draft']
    assert draft['cases'] == 3 and draft['case_size_lb'] == 5 and draft['total_lb'] == 15
    assert draft['product_id'] == payload['product_id'] and draft['supplier_id'] == supplier['id']
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    assert db_cursor.fetchone()['n'] == 0
