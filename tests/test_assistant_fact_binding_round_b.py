"""Adversarial round-B regressions: user facts remain authoritative over model calls."""
from uuid import uuid4

import pytest

import assistant_replies
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import (enabled, reason_catalog, chat, model, turn,
                                     resolve_product)  # noqa: F401
from tests.test_write_tickets import payload  # noqa: F401
from tests.test_write_tickets_part2 import items  # noqa: F401

pytestmark = pytest.mark.db


def supplier_call(cur, supplier_id):
    cur.execute('SELECT name FROM suppliers WHERE id=%s', (supplier_id,))
    return ('resolve', {'kind': 'supplier', 'query': cur.fetchone()['name']})


def missing_receive(client, chat, payload, db_cursor, monkeypatch, text='Received 10 lb of this ingredient'):
    args = dict(product_id=payload['product_id'], supplier_id=None, cases=None,
                case_size_lb=None, bol_reference=None)
    model(monkeypatch, resolve_product(db_cursor, payload['product_id']), ('prepare_receive', args))
    response = turn(client, chat, text)
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['kind'] == 'missing_facts'
    return card


@pytest.mark.parametrize('bad_model_allocation', ['omitted', 'wrong_trusted_lot'])
@pytest.mark.parametrize('spanish', [False, True])
def test_user_pack_lot_stays_pinned_when_model_omits_or_changes_it(client, chat, items, db_cursor, monkeypatch, bad_model_allocation, spanish):
    source, target = items['batch'], items['finished']
    lot_code = 'RB-PHYSICAL-' + uuid4().hex[:10].upper()
    db_cursor.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id', (source['id'], lot_code))
    physical = db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO transactions(type,notes) VALUES ('receive','round B physical lot') RETURNING id")
    transaction = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,100)',
                      (transaction, source['id'], physical))
    calls = [resolve_product(db_cursor, target['id'])]
    args = dict(source_product_id=source['id'], target_product_id=target['id'], cases=2)
    if bad_model_allocation == 'wrong_trusted_lot':
        calls.append(('resolve', {'kind': 'lot', 'query': source['lot_code']}))
        args['lot_allocations'] = [{'lot_id': source['lot_id'], 'quantity_lb': 10}]
    calls += [('packing_context', {'target_product_id': target['id'], 'lot_code': lot_code}),
              ('prepare_pack', args)]
    model(monkeypatch, *calls)
    text = ('Empacamos 2 cajas de ' if spanish else 'Packed 2 cases of ') + target['name'] + (' del lote ' if spanish else ' from batch ') + lot_code
    response = turn(client, chat, text)
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['kind'] == 'draft', card
    plan = card['prepared']['draft']['input_plan']
    primary = [row for row in plan if row['product_id'] == source['id']]
    assert [(row['lot_id'], row['quantity_lb']) for row in primary] == [(physical, 10)]
    assert 'LOT_NOT_CONFIRMED' in {row['code'] for row in card['prepared']['blockers']}
    assert card['language'] == ('es' if spanish else 'en')


@pytest.mark.parametrize('spanish', [False, True])
def test_receive_followup_cannot_change_user_total_from_10_to_50(client, chat, payload, db_cursor, monkeypatch, spanish):
    initial = missing_receive(client, chat, payload, db_cursor, monkeypatch,
                              'Recibimos 10 lb de este ingrediente' if spanish else 'Received 10 lb of this ingredient')
    supplier = next(s for s in initial['supplier_choices'] if s['id'] == payload['supplier_id'])
    model(monkeypatch, ('prepare_receive', {**payload, 'cases': 2, 'case_size_lb': 25}))
    response = turn(client, chat, choice_id=initial['choice_id'], selected_id=supplier['id'])
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['kind'] == 'missing_facts'
    assert card['known_facts']['quantity_lb'] == 10
    assert card['quantity_mismatch_lb'] == 50
    assert '10 lb' in card['summary']
    assert card['language'] == ('es' if spanish else 'en')


def test_new_receive_clears_prior_incomplete_total(client, chat, payload, db_cursor, monkeypatch):
    missing_receive(client, chat, payload, db_cursor, monkeypatch)
    model(monkeypatch, supplier_call(db_cursor, payload['supplier_id']), ('prepare_receive', payload))
    response = turn(client, chat, 'Receive a new delivery: 2 cases at 25 lb per case')
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['kind'] == 'draft', card
    assert card['prepared']['draft']['total_lb'] == 50


@pytest.mark.parametrize('text', [
    'Received 2 cases of 25 lb each',
    'Received 2 cases at 25 lb',
    'Receive 2 cases, 25 lb per case',
    'Recibimos 2 cajas de 25 lb cada una',
    'Recibimos 2 cajas de 25 lb por caja',
])
def test_receive_case_weight_is_not_mistaken_for_total(client, chat, payload, db_cursor, monkeypatch, text):
    model(monkeypatch, resolve_product(db_cursor, payload['product_id']),
          supplier_call(db_cursor, payload['supplier_id']), ('prepare_receive', payload))
    response = turn(client, chat, text)
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['kind'] == 'draft', card
    assert card['prepared']['draft']['total_lb'] == 50


def test_open_order_customer_shorthand_uses_canonical_party_name(client, chat, items, db_cursor, monkeypatch):
    customer = 'U.N.I.P.R.O. Foodservice'
    db_cursor.execute('INSERT INTO customers(name,active) VALUES (%s,true) RETURNING id', (customer,))
    customer_id = db_cursor.fetchone()['id']
    order = 'SO-ROUND-B-' + uuid4().hex[:8].upper()
    db_cursor.execute("INSERT INTO sales_orders(order_number,customer_id,state,status) VALUES (%s,%s,'open','confirmed') RETURNING id", (order, customer_id))
    order_id = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb) VALUES (%s,%s,10)',
                      (order_id, items['finished']['id']))
    model(monkeypatch, ('open_orders', {'customer': 'UNIPRO', 'product': None}))
    response = turn(client, chat, '¿Qué órdenes están abiertas para UNIPRO?')
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['kind'] == 'read'
    assert card['language'] == 'es'
    assert [(row['order_id'], row['customer']) for row in card['rows']] == [(order_id, customer)]


@pytest.mark.parametrize('text', [
    '¿Qué órdenes están abiertas para Sunshine?', 'Fabricamos una tanda',
    'Existencias de este producto', 'avena', 'miel',
])
def test_common_spanish_order_and_catalog_words_select_spanish(text):
    assert assistant_replies.language(text, 'en') == 'es'


@pytest.mark.parametrize('query_by_id', [False, True])
def test_product_number_after_spanish_unknown_name_keeps_spanish_and_real_identity(client, chat, payload, db_cursor, monkeypatch, query_by_id):
    model(monkeypatch, ('resolve', {'kind': 'product', 'query': 'NO-EXISTE-ROUNDB'}))
    first = turn(client, chat, 'Recibimos un ingrediente desconocido')
    assert first.status_code == 200, first.text
    assert first.json()['cards'][0]['kind'] == 'clarify'
    db_cursor.execute("UPDATE products SET odoo_code='70013' WHERE id=%s", (payload['product_id'],))
    query = str(payload['product_id']) if query_by_id else '70013'
    args = dict(product_id=payload['product_id'], supplier_id=None, cases=None,
                case_size_lb=None, bol_reference=None)
    model(monkeypatch, ('resolve', {'kind': 'product', 'query': query, 'context': {'action': 'receive'}}),
          ('prepare_receive', args))
    response = turn(client, chat, query)
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['language'] == 'es'
    assert card['kind'] == 'missing_facts'
    assert card['known_facts']['product_id'] == payload['product_id']
