"""Round E routing gates use real FL reads before trusting model tool selection.

OpenAI is controlled here to exercise hostile or incomplete routing choices.
Separate staging-copy acceptance supplies the required repeated real-model proof.
"""
import json

import pytest
from psycopg2.extras import Json

from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import enabled, reason_catalog, chat, model, turn, record  # noqa: F401
from tests.test_write_tickets_part2 import items  # noqa: F401
from tests.test_assistant_orders_round_d import orders  # noqa: F401
from tests.test_assistant_pack_review_c import pack_lots  # noqa: F401
from tests.test_assistant_voice_e2e_round_c import voice_catalog, card  # noqa: F401

pytestmark = pytest.mark.db


@pytest.mark.parametrize('text', [
    'Received 2 cases of rice, so 10 lb total.',
    'Packed 10 cases, so 50 lb total.',
])
def test_ordinary_so_quantity_does_not_preempt_the_model_with_an_order_lookup(
        client, chat, monkeypatch, text):
    requests = model(monkeypatch, ('unsupported_request', {}))

    result = card(turn(client, chat, text), 'unsupported_request')

    assert result['kind'] == 'unsupported_request'
    assert len(requests) == 1
    assert 'prepare_receive' in {tool['name'] for tool in requests[0]['tools']}
    assert 'prepare_pack' in {tool['name'] for tool in requests[0]['tools']}


@pytest.mark.parametrize('text,language', [
    ('Ship S O two six one oh one five dash zero zero one.', 'en'),
    ('Ship SO 261015 001.', 'en'),
    ('SO-261015-001', 'en'),
    ('S O two six one oh one five dash zero zero one', 'en'),
    ('Envía ese o dos seis uno cero uno cinco guion cero cero uno.', 'es'),
    ('ese o dos seis uno cero uno cinco guión cero cero uno', 'es'),
])
def test_order_number_is_resolved_before_model_and_only_shipping_tool_is_offered(
        client, chat, orders, monkeypatch, text, language):
    target = orders['Sunshine Granola']
    requests = model(monkeypatch, ('prepare_ship_order', {
        'order_id': target['id'], 'ship_all': None, 'lines': None, 'bol_reference': None}))

    result = card(turn(client, chat, text, language=language), 'missing_facts')

    assert result['shipping_quantity_needed'] is True
    assert result['order']['order_number'] == target['number']
    assert result['language'] == language
    assert 'SKU' not in result['message']
    assert len(requests) == 1
    assert {tool['name'] for tool in requests[0]['tools']} == {'prepare_ship_order'}
    supplied = json.dumps(requests[0]['input'], ensure_ascii=False)
    assert target['number'] in supplied and 'order_detail' in supplied
    assert str(target['id']) in supplied


@pytest.mark.parametrize('text', [
    'Ship SO-999999-999.',
    'SO 999999 999',
    'Ship SO 26-10-15-001.',
    'Envía ese o nueve nueve nueve nueve nueve nueve guion nueve nueve nueve.',
])
def test_unknown_order_number_never_reaches_model_or_product_resolution(
        client, chat, orders, monkeypatch, text):
    requests = model(monkeypatch)

    result = card(turn(client, chat, text), 'clarify')

    assert result['resolution_kind'] == 'order'
    assert 'SKU' not in result['message']
    assert requests == []


def test_explicit_number_cannot_prepare_an_earlier_resolved_order(
        client, chat, orders, db_cursor, monkeypatch):
    current, earlier = orders['Sunshine Granola'], orders['Setton Farms']
    state = {'resolved': {'order': [earlier['id']], 'order_line': {str(earlier['id']): [earlier['line']]}},
        'order_summaries': {str(earlier['id']): {'order_id': earlier['id'], 'order_number': earlier['number'],
            'remaining_lb': 25, 'items': [{'remaining_lb': 25, 'case_size_lb': 50}]}}}
    db_cursor.execute('UPDATE assistant_sessions SET state=%s WHERE id=%s', (Json(state), chat[0]))
    model(monkeypatch, ('prepare_ship_order', {
        'order_id': earlier['id'], 'ship_all': None, 'lines': None, 'bol_reference': None}))

    result = card(turn(client, chat, 'Ship ' + current['number']), 'missing_facts')

    assert result['shipping_quantity_needed'] is True
    assert result['order']['order_number'] == current['number']


@pytest.mark.parametrize('spanish', [False, True])
@pytest.mark.parametrize('with_context', [False, True])
def test_product_resolution_itself_offers_parent_batches_fifo_without_another_model_tool(
        client, chat, voice_catalog, pack_lots, monkeypatch, spanish, with_context):
    target = voice_catalog['fruit']
    args = {'kind': 'product', 'query': target['name']}
    if with_context:
        args['context'] = {'action': 'pack'}
    requests = model(monkeypatch, ('resolve', args))

    text = ('Empacamos 40 cajas de ' if spanish else 'Packed 40 cases of ') + target['name']
    result = card(turn(client, chat, text), 'choices')

    assert result['packing_lots'] is True
    assert [row['id'] for row in result['result']['candidates']] == [
        pack_lots['first_id'], pack_lots['later_id']]
    assert ('¿De qué lote a granel salieron?' if spanish else 'Which batch did these come from?') in result['message']
    assert len(requests) == 1
    prompt = card(turn(client, chat, text='', choice_id=result['id'], selected_id='different_batch'), 'missing_facts')
    assert prompt['packing_batch_needed'] is True
    assert 'SKU' not in prompt['message'] and 'source product' not in prompt['message'].lower()


@pytest.fixture
def coconut(db_cursor):
    products = {}
    for key, name, kind, size in [
        ('raw', 'Sweetened Flake Coconut', 'ingredient', None),
        ('bulk', 'Batch Coconut Sweetened Flake', 'batch', None),
        ('ten', 'Coconut Sweetened Flake CNS 10 LB', 'finished', 10),
        ('twentyfive', 'Coconut Sweetened Flake CNS 25 LB', 'finished', 25),
    ]:
        db_cursor.execute("""INSERT INTO products(name,type,uom,case_size_lb,active,is_service)
            VALUES (%s,%s,'lb',%s,true,false) RETURNING id,name""", (name, kind, size))
        products[key] = dict(db_cursor.fetchone())
    db_cursor.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=ANY(%s)',
        (products['bulk']['id'], [products['ten']['id'], products['twentyfive']['id']]))
    products['lots'] = []
    for code, received_at in [
        ('E-COCONUT-FIRST', '2026-10-14T12:00:00Z'),
        ('B26-1015-004', '2026-10-15T12:00:00Z'),
    ]:
        db_cursor.execute('INSERT INTO lots(product_id,lot_code,received_at) VALUES (%s,%s,%s) RETURNING id,lot_code',
            (products['bulk']['id'], code, received_at))
        lot = dict(db_cursor.fetchone())
        products['lots'].append(lot)
        db_cursor.execute("INSERT INTO transactions(type,notes) VALUES ('receive','Round E coconut batch fixture') RETURNING id")
        tid = db_cursor.fetchone()['id']
        db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,1000)',
            (tid, products['bulk']['id'], lot['id']))
    return products


@pytest.mark.parametrize('spanish', [False, True])
@pytest.mark.parametrize('with_context', [False, True])
@pytest.mark.parametrize('selected_key', ['raw', 'ten'])
def test_ambiguous_coconut_product_button_offers_parent_batches_without_another_model_request(
        client, chat, coconut, monkeypatch, spanish, with_context, selected_key):
    args = {'kind': 'product', 'query': 'coconut'}
    if with_context:
        args['context'] = {'action': 'pack'}
    requests = model(monkeypatch, ('resolve', args))
    text = 'Empacamos diez cajas de coco en hojuelas endulzado.' if spanish else 'Packed 10 cases of sweetened flake coconut.'

    products = card(turn(client, chat, text), 'choices')

    assert products['resolution_kind'] == 'product'
    selected_id = coconut[selected_key]['id']
    assert selected_id in {row['id'] for row in products['result']['candidates']}
    assert len(requests) == 1
    response = turn(client, chat, text='', choice_id=products['id'], selected_id=selected_id)
    batches = card(response, 'choices')
    assert batches['packing_lots'] is True
    assert [row['id'] for row in batches['result']['candidates']] == [row['id'] for row in coconut['lots']]
    assert {row['product_id'] for row in batches['result']['candidates']} == {coconut['bulk']['id']}
    assert batches['message'] == ('¿De qué lote a granel salieron?' if spanish else 'Which batch did these come from?')
    assert not any(row.get('kind') == 'missing_facts' for row in response.json()['cards'])
    assert len(requests) == 1


def test_explicit_pack_lot_after_ambiguous_product_choice_keeps_normal_lot_resolution(
        client, chat, coconut, monkeypatch):
    target, source = coconut['ten'], coconut['bulk']
    selected_lot = coconut['lots'][1]
    requests = model(monkeypatch,
        ('resolve', {'kind': 'product', 'query': 'coconut', 'context': {'action': 'pack'}}),
        ('packing_context', {'target_product_id': target['id'], 'lot_code': selected_lot['lot_code']}),
        ('prepare_pack', {'source_product_id': source['id'], 'target_product_id': target['id'], 'cases': 10}))

    products = card(turn(client, chat, 'Packed 10 cases of coconut from lot ' + selected_lot['lot_code'] + '.'), 'choices')

    assert products['resolution_kind'] == 'product'
    assert len(requests) == 1
    draft = card(turn(client, chat, text='', choice_id=products['id'], selected_id=target['id']), 'draft')
    assert len(requests) == 3
    assert draft['prepared']['draft']['input_plan'][0]['lot_id'] == selected_lot['id']
    assert all(not row.get('confirmed') for row in draft['prepared']['draft']['input_plan'])
    assert draft['prepared']['can_commit'] is False


@pytest.mark.parametrize('spanish', [False, True])
def test_exact_coconut_transcript_offers_batch_then_output_and_preserves_physical_tag_gate(
        client, chat, coconut, db_cursor, monkeypatch, spanish):
    query = 'coco en hojuelas endulzado' if spanish else 'Sweetened Flake Coconut'
    requests = model(monkeypatch, ('resolve', {'kind': 'product', 'query': query, 'context': {'action': 'pack'}}))
    text = 'Empacamos diez cajas de coco en hojuelas endulzado.' if spanish else 'Packed 10 cases of sweetened flake coconut.'

    batches = card(turn(client, chat, text), 'choices')

    assert batches['packing_lots'] is True
    assert [row['id'] for row in batches['result']['candidates']] == [row['id'] for row in coconut['lots']]
    assert len(requests) == 1
    selected = coconut['lots'][1]
    outputs = card(turn(client, chat, text='', choice_id=batches['id'], selected_id=selected['id']), 'choices')
    assert outputs['packing_outputs'] is True
    assert {row['id'] for row in outputs['result']['candidates']} == {
        coconut['ten']['id'], coconut['twentyfive']['id']}
    assert len(requests) == 1

    model(monkeypatch, ('prepare_pack', {
        'source_product_id': coconut['bulk']['id'], 'target_product_id': coconut['ten']['id'], 'cases': 10}))
    draft = card(turn(client, chat, text='', choice_id=outputs['id'], selected_id=coconut['ten']['id']), 'draft')
    assert draft['prepared']['draft']['total_lb'] == 100
    assert draft['prepared']['draft']['input_plan'][0]['lot_id'] == selected['id']
    assert all(not row.get('confirmed') for row in draft['prepared']['draft']['input_plan'])
    assert draft['prepared']['can_commit'] is False
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    before = db_cursor.fetchone()['n']
    rejected = record(client, chat, draft)
    assert rejected.status_code == 422 and 'LOT_NOT_CONFIRMED' in rejected.text
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    assert db_cursor.fetchone()['n'] == before


def test_new_pack_clears_an_abandoned_same_product_batch_selection(
        client, chat, voice_catalog, pack_lots, db_cursor, monkeypatch):
    source, target = voice_catalog['fruit_batch'], voice_catalog['fruit']
    state = {'resolved': {'product': [source['id'], target['id']], 'lot': [pack_lots['later_id']]},
        'packing_context': {'source_product_id': source['id'], 'target_product_id': target['id'],
            'lot_id': pack_lots['later_id'], 'lot_code': source['lot_code'], 'case_size_lb': 5}}
    db_cursor.execute('UPDATE assistant_sessions SET state=%s WHERE id=%s', (Json(state), chat[0]))
    model(monkeypatch, ('resolve', {'kind': 'product', 'query': target['name'], 'context': {'action': 'pack'}}))

    result = card(turn(client, chat, 'Packed 40 cases of ' + target['name']), 'choices')

    assert result['packing_lots'] is True
    db_cursor.execute('SELECT state FROM assistant_sessions WHERE id=%s', (chat[0],))
    saved = db_cursor.fetchone()['state']
    assert not saved.get('packing_context')
    assert [row['id'] for row in saved['pending']['result']['candidates']] == [
        pack_lots['first_id'], pack_lots['later_id']]
