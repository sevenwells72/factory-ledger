"""Partial customer names need a person's choice on every Assistant path."""
import pytest

from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_assistant_customer_final2 import only_card, own_chat
from tests.test_fl_assistant import (customer_ticket_schema, enabled, model,
                                     reason_catalog, turn)  # noqa: F401
from tests.test_order_tickets import seed

pytestmark = pytest.mark.db

PARTIAL_CUSTOMERS = [('Sun', 'Sunshine Granola'), ('Green', 'Greenfield Bakery'),
                     ('Sunshine', 'Sunshine Granola')]


def customer_order(cur, name):
    data = seed(cur)
    cur.execute('UPDATE customers SET name=%s WHERE id=%s', (name, data['customer']))
    cur.execute("""INSERT INTO sales_orders(order_number,customer_id,state,status)
                   VALUES ('SO-261023-001',%s,'open','confirmed') RETURNING id""", (data['customer'],))
    order_id = cur.fetchone()['id']
    cur.execute('''INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb,
                       ordered_quantity,ordered_unit,ordered_case_weight_lb)
                   VALUES (%s,%s,50,2,'cases',25)''', (order_id, data['products']['food']))
    return data['customer'], order_id


def assert_unselected(cur, chat, customer_id, order_id=None):
    cur.execute('SELECT state FROM assistant_sessions WHERE id=%s', (chat[0],))
    state = cur.fetchone()['state']
    assert customer_id not in state.get('resolved', {}).get('customer', [])
    if order_id is not None:
        assert order_id not in state.get('resolved', {}).get('order', [])
    cur.execute('SELECT count(*) AS n FROM assistant_drafts WHERE session_id=%s', (chat[0],))
    assert cur.fetchone()['n'] == 0


@pytest.mark.parametrize('query,name', PARTIAL_CUSTOMERS + [('Greenfield Bakery', 'Greenfield Organic Bakery')])
@pytest.mark.parametrize('role', ['office', 'floor'])
@pytest.mark.parametrize('path', ['customer_ship', 'order_ship', 'missing_order_ship', 'open_orders'])
def test_partial_customer_on_ship_and_open_orders_waits_for_human_choice(
        client, actors, db_cursor, monkeypatch, query, name, role, path):
    customer_id, order_id = customer_order(db_cursor, name)
    chat = own_chat(client, actors, role)
    if path == 'customer_ship':
        tool = ('resolve', {'kind': 'customer', 'query': query, 'context': {'action': 'ship'}})
    elif path == 'order_ship':
        tool = ('resolve', {'kind': 'order', 'query': query, 'context': {'action': 'ship'}})
    else:
        tool = ('open_orders', {'customer': query})
    calls = model(monkeypatch, *([] if path == 'missing_order_ship' else [tool]))
    text = (f'What open orders does {query} have?' if path == 'open_orders' else
            f'Ship order SO-999999-999 for {query}' if path == 'missing_order_ship' else
            f'Ship the order for {query}')
    choice = only_card(turn(client, chat, text), 'choices')
    assert choice['resolution_kind'] == 'customer'
    assert choice['query'] == query
    if query == 'Greenfield Bakery':
        assert choice['result']['match']['tier'] == 'keyword'
    else:
        assert choice['result']['outcome'] == 'ambiguous' and choice['result']['match'] is None
    assert [row['id'] for row in choice['result']['candidates']] == [customer_id]
    if query != 'Greenfield Bakery':
        assert not choice.get('existing_customer_only')
    assert len(calls) == (0 if path == 'missing_order_ship' else 1)
    assert_unselected(db_cursor, chat, customer_id, order_id)

    # The model may retain the original partial wording after the human choice.
    if path == 'open_orders':
        model(monkeypatch, ('open_orders', {'customer': query}))
        selected = only_card(turn(client, chat, '', choice_id=choice['id'], selected_id=customer_id), 'read')
        assert [row['order_id'] for row in selected['rows']] == [order_id]
    else:
        model(monkeypatch, ('resolve', {'kind': 'order', 'query': query,
                                       'context': {'action': 'ship', 'customer_id': customer_id}}),
              ('prepare_ship_order', {'order_id': order_id}))
        selected = only_card(turn(client, chat, '', choice_id=choice['id'], selected_id=customer_id), 'missing_facts')
        assert selected['shipping_quantity_needed']
        assert selected['order']['order_number'] == 'SO-261023-001'


@pytest.mark.parametrize('query,name', PARTIAL_CUSTOMERS)
@pytest.mark.parametrize('role', ['office', 'owner'])
@pytest.mark.parametrize('tool', ['resolve', 'prepare_create_customer'])
@pytest.mark.parametrize('language', ['en', 'es'])
def test_partial_customer_creation_offers_existing_or_add_new(
        client, actors, db_cursor, monkeypatch, customer_ticket_schema,
        query, name, role, tool, language):
    db_cursor.execute('INSERT INTO customers(name,active) VALUES (%s,true) RETURNING id', (name,))
    customer_id = db_cursor.fetchone()['id']
    chat = own_chat(client, actors, role)
    args = {'kind': 'customer', 'query': query} if tool == 'resolve' else {'name': query}
    calls = model(monkeypatch, (tool, args))
    text = f'Add {query} as a new customer' if language == 'en' else f'Agrega {query} como cliente nuevo'
    choice = only_card(turn(client, chat, text, language=language), 'choices')
    assert choice['resolution_kind'] == 'customer' and choice['query'] == query
    assert not choice.get('existing_customer_only'), 'Partial names must keep Add as new available'
    assert choice['result']['outcome'] == 'ambiguous' and choice['result']['match'] is None
    assert [row['id'] for row in choice['result']['candidates']] == [customer_id]
    assert ('ya existe' if language == 'es' else 'already exists') not in choice['message']
    assert len(calls) == 1
    assert_unselected(db_cursor, chat, customer_id)

    # This is the same name-bound message as the page's Add-as-new button.
    create_text = (f'Create a new customer named {query} first, then continue the order.' if language == 'en' else
                   f'Crea primero un cliente nuevo llamado {query} y luego continúa el pedido.')
    calls = model(monkeypatch, ('resolve', {'kind': 'customer', 'query': query}),
                  ('prepare_create_customer', {'name': query}))
    draft = only_card(turn(client, chat, create_text, language=language), 'draft')
    assert draft['prepared']['action'] == 'create_customer'
    assert draft['prepared']['can_commit'], draft
    assert draft['prepared']['draft']['customer']['name'] == query
    assert len(calls) == 2
    db_cursor.execute('SELECT state FROM assistant_sessions WHERE id=%s', (chat[0],))
    state = db_cursor.fetchone()['state']
    assert customer_id not in state.get('resolved', {}).get('customer', [])
    assert '_new_customer_query' not in state and '_customer_choice' not in state
    db_cursor.execute('SELECT count(*) AS n FROM customers WHERE name=%s', (query,))
    assert db_cursor.fetchone()['n'] == 0, 'A draft must never create a customer before Record'


@pytest.mark.parametrize('tool', ['resolve', 'prepare_create_customer'])
@pytest.mark.parametrize('changed', ['different_query', 'new_exact_customer'])
def test_add_new_choice_is_bound_to_its_name_and_keeps_exact_duplicate_guard(
        client, actors, db_cursor, monkeypatch, tool, changed):
    db_cursor.execute("INSERT INTO customers(name,active) VALUES ('Sunshine Granola',true),('Greenfield Bakery',true)")
    chat = own_chat(client, actors, 'office')
    model(monkeypatch, ('resolve', {'kind': 'customer', 'query': 'Sun'}))
    only_card(turn(client, chat, 'Add Sun as a new customer'), 'choices')
    query = 'Green' if changed == 'different_query' else 'Sun'
    if changed == 'new_exact_customer':
        db_cursor.execute("INSERT INTO customers(name,active) VALUES ('Sun',true)")
    args = {'kind': 'customer', 'query': query} if tool == 'resolve' else {'name': query}
    model(monkeypatch, (tool, args))
    choice = only_card(turn(client, chat, 'Create a new customer named Sun first, then continue the order.'), 'choices')
    assert choice['query'] == query
    assert bool(choice.get('existing_customer_only')) == (changed == 'new_exact_customer')
    db_cursor.execute('SELECT count(*) AS n FROM assistant_drafts WHERE session_id=%s', (chat[0],))
    assert db_cursor.fetchone()['n'] == 0


@pytest.mark.parametrize('tool', ['resolve', 'open_orders'])
def test_prior_customer_choice_cannot_authorize_a_different_partial_query(
        client, actors, db_cursor, monkeypatch, tool):
    customer_id, order_id = customer_order(db_cursor, 'Sunshine Granola')
    db_cursor.execute("INSERT INTO customers(name,active) VALUES ('Greenfield Bakery',true) RETURNING id")
    other = db_cursor.fetchone()['id']
    chat = own_chat(client, actors, 'floor')
    model(monkeypatch, ('open_orders', {'customer': 'Sun'}))
    choice = only_card(turn(client, chat, 'What open orders does Sun have?'), 'choices')
    model(monkeypatch, ('open_orders', {'customer': 'Sun'}))
    only_card(turn(client, chat, '', choice_id=choice['id'], selected_id=customer_id), 'read')
    args = ({'kind': 'order', 'query': 'Green', 'context': {'action': 'ship', 'customer_id': customer_id}}
            if tool == 'resolve' else {'customer': 'Green'})
    model(monkeypatch, (tool, args))
    next_choice = only_card(turn(client, chat, 'Ship the order for Green' if tool == 'resolve' else 'Open orders for Green'), 'choices')
    assert next_choice['resolution_kind'] == 'customer'
    assert [row['id'] for row in next_choice['result']['candidates']] == [other]
    assert_unselected(db_cursor, chat, other, order_id)


@pytest.mark.parametrize('tool', ['resolve', 'prepare_create_customer'])
def test_keyword_alternative_does_not_hide_add_new_for_partial_customer(
        client, actors, db_cursor, monkeypatch, customer_ticket_schema, tool):
    db_cursor.execute("INSERT INTO customers(name,active) VALUES ('Sun Foods',true),('Sun Organic Food',true)")
    chat = own_chat(client, actors, 'office')
    args = {'kind': 'customer', 'query': 'Sun Food'} if tool == 'resolve' else {'name': 'Sun Food'}
    model(monkeypatch, (tool, args))
    choice = only_card(turn(client, chat, 'Add Sun Food as a new customer'), 'choices')
    assert {row['tier'] for row in choice['result']['candidates']} == {'prefix', 'keyword'}
    assert not choice.get('existing_customer_only')
    model(monkeypatch, ('resolve', {'kind': 'customer', 'query': 'Sun Food'}),
          ('prepare_create_customer', {'name': 'Sun Food'}))
    draft = only_card(turn(client, chat, 'Create a new customer named Sun Food first, then continue the order.'), 'draft')
    assert draft['prepared']['can_commit']
    assert draft['prepared']['draft']['customer']['name'] == 'Sun Food'
