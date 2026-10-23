"""Final staging regressions: actual FL routes and PostgreSQL; provider-only stubs."""
import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException

import assistant_replies
import assistant_service
import fl_assistant
import main
import ship_tickets
from assistant_intents import office_handoff
from assistant_receive import package_facts, split_receive_items
from assistant_voice import normalize_voice_text
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import enabled, reason_catalog, customer_ticket_schema, chat, model, turn, record, resolve_product  # noqa: F401
from tests.test_write_tickets import payload, headers  # noqa: F401
from tests.test_write_tickets_part2 import items  # noqa: F401
from tests.test_assistant_orders_round_d import orders  # noqa: F401

pytestmark = pytest.mark.db


def card(response, kind):
    assert response.status_code == 200, response.text
    return next(c for c in response.json()['cards'] if c['kind'] == kind)


@pytest.mark.parametrize('sentence', [
    'Recibimos 10 bolsas de avena de 50 libras de Grain Supply, BOL 5521.',
    'Received 10 bags of Oats, 50 lb each, from Grain Supply, BOL 5521.',
    'Received 10 bags of Oats, 50 lb bags, from Grain Supply, BOL 5521.',
    'Received ten bags of Oats, fifty pounds each from Grain Supply, BOL 5521.',
])
def test_package_weights_override_bad_model_arithmetic(client, chat, payload, db_cursor, monkeypatch, sentence):
    assert package_facts(sentence) == {'cases': 10, 'case_size_lb': 50.0, 'quantity_lb': 500.0}
    db_cursor.execute('SELECT name FROM suppliers WHERE id=%s', (payload['supplier_id'],))
    supplier = db_cursor.fetchone()['name']
    calls = [resolve_product(db_cursor, payload['product_id']), ('resolve', {'kind': 'supplier', 'query': supplier}),
             ('prepare_receive', {**payload, 'cases': 1, 'case_size_lb': 50})]
    model(monkeypatch, *calls)
    draft = card(turn(client, chat, sentence), 'draft')
    assert draft['prepared']['draft']['cases'] == 10
    assert draft['prepared']['draft']['total_lb'] == 500
    assert draft['prepared']['can_commit']
    committed = record(client, chat, draft)
    assert committed.status_code == 200, committed.text
    assert committed.json()['result']['total_lb'] == 500


def test_explicit_conflicting_total_is_still_reviewed():
    assert package_facts('Received 2 cases of Oats, 25 lb each, 100 lb total')['quantity_lb'] == 100
    assert package_facts('Received 10 bags of Oats, 50 lb total') == {}
    assert package_facts('Received -10 bags of Oats, 50 lb each') == {}
    assert package_facts('Received 10 bags of Oats, -50 lb each') == {}


@pytest.mark.parametrize('sentence', [
    'We received 5 cases of Sunflower Seeds and 3 cases of Oats from Grain Supply, 50 lb per case, BOL MULTI-5521.',
    'Recibimos 5 cajas de semillas de girasol y 3 cajas de avena de Grain Supply, 50 lb por caja, BOL MULTI-5521.',
])
def test_receive_queue_retains_second_product_across_reload_and_requires_first_record(
        client, chat, payload, db_cursor, monkeypatch, sentence):
    parts = split_receive_items(sentence)
    assert len(parts) == 2 and all('Grain Supply' in part and 'BOL MULTI-5521' in part for part in parts)
    db_cursor.execute('SELECT name FROM suppliers WHERE id=%s', (payload['supplier_id'],))
    supplier = db_cursor.fetchone()['name']
    db_cursor.execute("INSERT INTO products(name,type,uom,active) VALUES('Final queue Oats','ingredient','lb',true) RETURNING id")
    second_product = db_cursor.fetchone()['id']
    model(monkeypatch, resolve_product(db_cursor, payload['product_id']),
        ('resolve', {'kind': 'supplier', 'query': supplier}), ('prepare_receive', payload))
    response = turn(client, chat, sentence)
    first = card(response, 'draft')
    assert first['prepared']['draft']['cases'] == 5
    queued = card(response, 'receive_queue')
    assert len(queued['queued_text']) == 1 and ('Oats' in queued['queued_text'][0] or 'avena' in queued['queued_text'][0])
    model(monkeypatch)
    blocked = card(turn(client, chat, queued['continue_text']), 'receive_queue')
    assert blocked['finish_current'] and blocked['queued_text'] == queued['queued_text']
    resumed = client.post('/assistant/resume', headers=chat[1], json={'session_id': chat[0]})
    assert queued['queued_text'][0] in str(resumed.json())
    assert record(client, chat, first).status_code == 200
    args = {**payload, 'product_id': second_product, 'lot_code': 'FINAL-SECOND-' + uuid4().hex[:8]}
    model(monkeypatch, resolve_product(db_cursor, second_product), ('prepare_receive', args))
    response = turn(client, chat, queued['continue_text'])
    second = card(response, 'draft')
    assert second['prepared']['draft']['product_id'] == second_product
    assert second['prepared']['draft']['cases'] == 3
    assert second['prepared']['draft']['total_lb'] == 150
    assert not any(c['kind'] == 'receive_queue' for c in response.json()['cards'])
    assert record(client, chat, second).status_code == 200


@pytest.mark.parametrize('sentence,expected', [
    ('5 cases of Sunflower Seeds and 3 cases of Oats',
     ['Received 5 cases of Sunflower Seeds', 'Received 3 cases of Oats']),
    ('5 cajas de semillas de girasol y 3 cajas de avena',
     ['Received 5 cases de semillas de girasol', 'Received 3 cases de avena']),
])
def test_bare_multi_receive_keeps_action_on_first_and_queued_item(sentence, expected):
    assert split_receive_items(sentence) == expected


def test_receive_split_does_not_rewrite_multi_item_order_or_product_name():
    assert split_receive_items('New order for Setton: 5 cases of Fruit and Nut and 3 cases of Oats') == []
    assert split_receive_items('Received 5 cases of Fruit and Nut from Grain Supply') == []
    items = split_receive_items('Received 5 cases of Sunflower Seeds and 3 cases of Oats from Grain Supply, 50 lb each, supplier lot OATS-ONLY.')
    assert 'OATS-ONLY' not in items[0] and 'OATS-ONLY' in items[1]
    assert office_handoff('I will send an email tomorrow') is None
    assert office_handoff('The customer is sending a payment') is None


def test_receive_queue_does_not_drop_first_item_missing_required_facts(client, chat, payload, db_cursor, monkeypatch):
    model(monkeypatch, resolve_product(db_cursor, payload['product_id']), ('prepare_receive', {
        'product_id': payload['product_id'], 'supplier_id': None, 'cases': 5,
        'case_size_lb': None, 'bol_reference': None}))
    initial = turn(client, chat, 'Received 5 cases of Sunflower Seeds and 3 cases of Oats')
    assert card(initial, 'missing_facts')
    queued = card(initial, 'receive_queue')
    model(monkeypatch)
    blocked = card(turn(client, chat, queued['continue_text']), 'receive_queue')
    assert blocked['missing_current'] and blocked['queued_text'] == queued['queued_text']
    db_cursor.execute('SELECT state FROM assistant_sessions WHERE id=%s', (chat[0],))
    assert db_cursor.fetchone()['state']['receive_context']['args']['cases'] == 5


@pytest.mark.parametrize('text,action,spanish', [
    ('Change order SO-261015-001 to 3 cases', 'update_order_line', False),
    ('Cambia el pedido SO-261015-001 a 3 cajas', 'update_order_line', True),
    ('Cancel order SO-261015-001', 'cancel_order', False),
    ('Cancela el pedido SO-261015-001', 'cancel_order', True),
    ('Grain Supply is sending 20 cases of Oats next Tuesday', 'create_expected_receipt', False),
    ('Grain Supply enviará 20 cajas de avena', 'create_expected_receipt', True),
])
def test_office_actions_are_fast_without_provider(client, chat, monkeypatch, text, action, spanish):
    requests = model(monkeypatch)
    result = card(turn(client, chat, text), 'handoff')
    assert not requests
    assert 'action=' + action in result['actions'][0]['href']
    assert ('Oficina' if spanish else 'Office') in result['message']
    if action != 'create_expected_receipt':
        assert 'SO-261015-001' in result['message'] and 'SO-261015-001' in result['actions'][0]['href']


@pytest.mark.parametrize('spanish', [False, True])
def test_shipping_exhausted_order_names_order_without_provider(client, chat, orders, db_cursor, monkeypatch, spanish):
    requests = model(monkeypatch)
    number = orders['STG-D cleanup fixture']['number']
    result = card(turn(client, chat, ('Envía el pedido ' if spanish else 'Ship ') + number), 'refusal')
    assert not requests and number in result['message']
    assert ('ya está completamente enviado' if spanish else 'already fully shipped') in result['message']


@pytest.mark.parametrize('spanish', [False, True])
def test_shipping_shortage_uses_real_order_allocation_and_does_not_post(client, chat, orders, items, db_cursor, monkeypatch, spanish):
    target = orders['Sunshine Granola']
    product_id = items['finished']['id']
    db_cursor.execute('UPDATE products SET case_size_lb=50 WHERE id=%s', (product_id,))
    db_cursor.execute("SELECT id FROM lots WHERE product_id=%s AND lot_code='D-FINISHED-001'", (product_id,))
    lot_id = db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO transactions(type,notes) VALUES('adjust','Final shortage fixture') RETURNING id")
    adjustment = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES(%s,%s,%s,-350)', (adjustment, product_id, lot_id))
    db_cursor.execute('UPDATE sales_order_lines SET quantity_lb=1000,ordered_quantity=20 WHERE id=%s', (target['line'],))
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    before = db_cursor.fetchone()['n']
    model(monkeypatch, ('prepare_ship_order', {'order_id': target['id'], 'ship_all': True, 'bol_reference': 'FINAL-SHORT'}))
    result = card(turn(client, chat, ('Envía todo el pedido ' if spanish else 'Ship all of order ') + target['number']), 'refusal')
    assert result['error_code'] == 'INSUFFICIENT_STOCK'
    assert ('Solo hay 12 cajas' if spanish else 'Only 12 cases') in result['message']
    assert ('necesita 20' if spanish else 'needs 20') in result['message']
    assert len(result['actions']) == 2
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    assert db_cursor.fetchone()['n'] == before


@pytest.mark.parametrize('spanish', [False, True])
def test_short_customer_suggestion_is_offered_even_if_model_creates_directly(client, actors, db_cursor, monkeypatch, spanish):
    db_cursor.execute("INSERT INTO customers(name,active) VALUES('Setton Farms',true) RETURNING id")
    existing = db_cursor.fetchone()['id']
    h = headers(actors['owner']['key'])
    own_chat = client.post('/assistant/session', headers=h).json()['session_id'], h
    requests = model(monkeypatch, ('prepare_create_customer', {'name': 'Setton'}))
    choice = card(turn(client, own_chat, 'Agrega Setton como cliente nuevo' if spanish else 'Add Setton as a new customer'), 'choices')
    assert not choice.get('existing_customer_only')
    assert [c['id'] for c in choice['result']['candidates']] == [existing]
    assert len(requests) == 1
    model(monkeypatch)
    result = card(turn(client, own_chat, '', choice_id=choice['id'], selected_id=existing), 'handoff')
    assert 'Setton Farms' in result['message']
    db_cursor.execute("SELECT count(*) AS n FROM customers WHERE name='Setton'")
    assert db_cursor.fetchone()['n'] == 0


def test_add_setton_resolver_first_does_not_repeat_create_choices(client, actors, db_cursor, monkeypatch):
    db_cursor.execute("INSERT INTO customers(name,active) VALUES('Setton Farms',true) RETURNING id")
    existing = db_cursor.fetchone()['id']
    h = headers(actors['owner']['key'])
    own_chat = client.post('/assistant/session', headers=h).json()['session_id'], h
    model(monkeypatch, ('resolve', {'kind': 'customer', 'query': 'Setton'}))
    choice = card(turn(client, own_chat, 'Add Setton'), 'choices')
    model(monkeypatch)
    result = card(turn(client, own_chat, '', choice_id=choice['id'], selected_id=existing), 'handoff')
    assert 'already exists' in result['message']


@pytest.mark.parametrize('text,expected', [
    ('two eighty three', '283'), ('seven zero zero one three', '70013'),
    ('product two eighty three', 'product 283'), ('S S original number one', 'SS #1'),
])
def test_spoken_product_identifiers(text, expected):
    assert normalize_voice_text(text) == expected


@pytest.mark.parametrize('kind', ['supplier', 'customer'])
def test_phonetic_party_wording_requires_human_choice(client, chat, db_cursor, monkeypatch, kind):
    db_cursor.execute(f"INSERT INTO {'suppliers' if kind == 'supplier' else 'customers'}(name,active) VALUES('La Crosse Milling',true) RETURNING id")
    ident = db_cursor.fetchone()['id']
    model(monkeypatch, ('resolve', {'kind': kind, 'query': 'la cross milling'}))
    choice = card(turn(client, chat, 'la cross milling'), 'choices')
    assert ident in [c['id'] for c in choice['result']['candidates']]
    assert 'Did you mean' in choice['message']
    db_cursor.execute('SELECT state FROM assistant_sessions WHERE id=%s', (chat[0],))
    assert ident not in db_cursor.fetchone()['state'].get('resolved', {}).get(kind, [])


@pytest.mark.parametrize('error', [
    {'code': 'insufficient_quota'}, {'code': 'credit_balance_exhausted'},
    {'type': 'billing_hard_limit_reached'}, {'message': 'You have no credits remaining'},
    {'message': 'You exceeded your current quota, please check your plan and billing details.'},
])
def test_provider_quota_classification(error):
    assert assistant_service.quota_exhausted(429, {'error': error})


@pytest.mark.parametrize('status,error', [(429, {'code': 'rate_limit_exceeded'}), (503, {'message': 'service unavailable'}), (400, {'code': 'invalid_schema'})])
def test_transient_provider_errors_keep_the_retry_path(status, error):
    assert not assistant_service.quota_exhausted(status, {'error': error})


@pytest.mark.parametrize('spanish', [False, True])
def test_credit_outage_records_one_owner_exception_and_offers_offline_links(client, chat, actors, db_cursor, monkeypatch, spanish):
    migration = Path(__file__).resolve().parents[1] / 'migrations/084_assistant_service_alert.sql'
    db_cursor.execute(migration.read_text())
    problem = HTTPException(503, {'error_code': assistant_service.CODE, 'message': assistant_service.MESSAGE})
    model(monkeypatch, problem, problem)
    for _ in range(2):
        result = card(turn(client, chat, 'Cuánta avena tenemos' if spanish else 'How much Oats do we have'), 'handoff')
        assert result['error_code'] == assistant_service.CODE and result['notice_id']
        assert result['message'] == (assistant_service.MESSAGE_ES if spanish else assistant_service.MESSAGE)
        assert {a['href'] for a in result['actions']} == {'/dashboard/entry.html', '/dashboard/office.html'}
        assert 'Retry' not in str(result) and 'minute' not in str(result)
    db_cursor.execute("SELECT * FROM exceptions WHERE kind='ASSISTANT_OFFLINE'")
    rows = db_cursor.fetchall()
    assert len(rows) == 1 and rows[0]['owner_actor_id'] == actors['owner']['id']
    response = client.get('/exceptions', headers=headers(actors['owner']['key']))
    assert response.status_code == 200 and 'ASSISTANT_OFFLINE' in response.text


def test_shortage_reason_has_specific_case_counts_in_both_languages():
    with pytest.raises(HTTPException) as problem:
        ship_tickets.stock_shortage({'name': 'Oats', 'case_size_lb': 50}, 600, 1000, order=True)
    detail = problem.value.detail
    assert detail['message'] == 'Only 12 cases of Oats available; the order needs 20.'
    assert detail['message_es'] == 'Solo hay 12 cajas de Oats; el pedido necesita 20.'
    for language in ('en', 'es'):
        result = assistant_replies.decorate({'kind': 'refusal', 'status': 409, 'result': {'detail': detail}}, language)
        assert result['message'] == detail['message_es' if language == 'es' else 'message']
        assert len(result['actions']) == 2


def test_spanish_kosher_and_ingredient_refusals_preserve_plain_reason():
    result = assistant_replies.decorate({'kind': 'draft', 'prepared': {'draft': {}, 'blockers': [
        {'code': 'KOSHER_BATCH_RECEIVE_REFUSED', 'message': 'Extra-kosher batches are made, never received.'},
        {'code': 'INSUFFICIENT_STOCK', 'message': 'Coffee needs 0.5 lb', 'message_es': 'Se necesitan 0.5 lb de Coffee. Registre la recepción primero.'}]}}, 'es')
    assert 'no se pueden recibir' in result['prepared']['blockers'][0]['message']
    assert result['prepared']['blockers'][1]['message'].startswith('Se necesitan 0.5 lb')
    assert 'Revise los datos' not in str(result)
    assert assistant_replies.number(88.80000000000001) == '88.8'


@pytest.mark.parametrize('code,reason_word', [
    ('LOT_MERGED', 'activo'), ('LOT_SUPPLIER_MISMATCH', 'proveedor'),
    ('EXPECTED_RECEIPT_NOT_OPEN', 'entrega'), ('INVALID_PACK', 'diferentes'),
    ('INVALID_OUTPUT', 'tanda'), ('INPUT_PLAN_CHANGED', 'cambiaron'),
    ('PRIVATE_LABEL_MERGE', 'marca privada'), ('RECIPE_MISSING', 'receta'),
    ('UNUSED_LOT_SELECTION', 'ingredientes'), ('PALLET_MOVE_REQUIRED', 'producción'),
    ('INVALID_SUBSTITUTION', 'sustitución'), ('INVALID_EXCLUSION', 'receta'),
    ('ROLE_NOT_ALLOWED', 'permiso'), ('BACKFILL_OWNER_ONLY', '14 días'),
    ('OCCURRED_AT_IN_FUTURE', 'futuro'), ('WARNING_NOT_ACKNOWLEDGED', 'advertencias'),
    ('CUSTOMER_NOT_FOUND', 'cliente'), ('RECEIPT_NOT_FOUND', 'recibo'),
    ('SERVICE_STANDALONE_UNSUPPORTED', 'pedido'), ('QUANTITY_REQUIRED', 'cantidad'),
    ('NOTE_REQUIRED', 'nota'), ('SHORTAGE_OPEN_RESOLVE_INSTEAD', 'Excepciones'),
    ('ATTACHMENT_WRONG_ACTOR', 'otro usuario'), ('ATTACHMENT_WRONG_CONTEXT', 'otra entrada'),
    ('TICKET_EXPIRED', 'venció'), ('TICKET_WRONG_USER', 'persona'),
    ('PRODUCT_AMBIGUOUS', 'varios productos'), ('IDS_REQUIRED', 'elija'),
])
def test_ordinary_refusal_families_explain_actual_reason_in_spanish(code, reason_word):
    original = {'error_code': code, 'message': 'Original ledger reason retained for audit', 'lot_id': 123}
    result = assistant_replies.decorate({'kind': 'refusal', 'status': 409, 'result': {'detail': original}}, 'es')
    assert reason_word in result['message']
    assert 'Revise los datos de este borrador' not in result['message']
    assert 'No se puede registrar esta acción con los datos actuales' not in result['message']
    assert result['result']['detail'] == original
    english = assistant_replies.decorate({'kind': 'refusal', 'status': 409, 'result': {'detail': original}}, 'en')
    assert english['message'] == original['message']


@pytest.mark.parametrize('reason,expected', [
    ('Batch Oats needs 88.80000000000001 lb and has no lot to record the shortage against; make it first.', 'Se necesitan 88.8 lb de Batch Oats'),
    ("Case weight required. Product 'Oats Finished' has no case_size_lb set.", 'Falta el peso de una caja'),
    ('TAG-123 does not belong to Oats.', 'El lote TAG-123 no pertenece a Oats.'),
    ('Oats is inactive; resolve an active product.', 'Oats está inactivo.'),
    ("Lot code 'abc' normalizes to the same code as existing lot 'ABC'", 'El código abc representa el mismo lote que ABC.'),
])
def test_legacy_plain_text_refusal_keeps_concrete_reason_in_spanish(reason, expected):
    result = assistant_replies.decorate({'kind': 'refusal', 'status': 409, 'result': {'detail': reason}}, 'es')
    assert expected in result['message']
