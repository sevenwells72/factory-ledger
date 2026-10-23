"""Duplicate customer requests finish on FL's answer, without model deliberation."""
from uuid import uuid4

import pytest

from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import (enabled, reason_catalog, customer_ticket_schema,
                                     model, turn, resolve_product)  # noqa: F401
from tests.test_write_tickets import headers
from tests.test_order_tickets import seed
from assistant_intents import office_handoff

pytestmark = pytest.mark.db


def own_chat(client, actors, role='owner'):
    h = headers(actors[role]['key'])
    response = client.post('/assistant/session', headers=h)
    assert response.status_code == 200, response.text
    return response.json()['session_id'], h


def only_card(response, kind):
    assert response.status_code == 200, response.text
    cards = response.json()['cards']
    assert len(cards) == 1 and cards[0]['kind'] == kind, cards
    return cards[0]


@pytest.mark.parametrize('role', ['office', 'owner'])
@pytest.mark.parametrize('language', ['en', 'es'])
@pytest.mark.parametrize('tool', ['resolve', 'prepare_create_customer'])
@pytest.mark.parametrize('spelling', ['exact', 'alias'])
def test_existing_customer_is_offered_and_selected_without_second_model_call(
        client, actors, db_cursor, monkeypatch, role, language, tool, spelling):
    db_cursor.execute("INSERT INTO customers(name,active) VALUES('Sunshine Granola',true) RETURNING id")
    ident = db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO search_aliases(kind,alias,customer_id) VALUES('customer','Sunny Foods',%s)", (ident,))
    query = {'exact': 'Sunshine Granola', 'alias': 'Sunny Foods'}[spelling]
    args = {'kind': 'customer', 'query': query} if tool == 'resolve' else {'name': query}
    chat = own_chat(client, actors, role)
    # The review saw this incomplete second response. It must never be requested.
    calls = model(monkeypatch, (tool, args), {'status': 'incomplete', 'output': []})
    text = f'Add {query} as a new customer' if language == 'en' else f'Agrega {query} como cliente nuevo'
    card = only_card(turn(client, chat, text, language=language), 'choices')
    assert card['existing_customer_only']
    assert [row['id'] for row in card['result']['candidates']] == [ident]
    assert ('ya existe' if language == 'es' else 'already exists') in card['message']
    assert len(calls) == 1
    db_cursor.execute('SELECT state FROM assistant_sessions WHERE id=%s', (chat[0],))
    assert ident not in db_cursor.fetchone()['state'].get('resolved', {}).get('customer', [])
    resumed = client.post('/assistant/resume', headers=chat[1], json={'session_id': chat[0]})
    assert resumed.status_code == 200 and resumed.json()['turns'][-1]['cards'][0] == card
    selected = only_card(turn(client, chat, '', choice_id=card['id'], selected_id=ident), 'handoff')
    assert 'Sunshine Granola' in selected['message']
    assert selected['actions'][0]['href'] == '/dashboard/office.html?action=create_order'
    assert len(calls) == 1
    db_cursor.execute('SELECT count(*) AS n FROM assistant_drafts WHERE session_id=%s', (chat[0],))
    assert db_cursor.fetchone()['n'] == 0
    db_cursor.execute("SELECT count(*) AS n FROM customers WHERE name='Sunshine Granola'")
    assert db_cursor.fetchone()['n'] == 1


@pytest.mark.parametrize('text', [
    'Please add Sunshine Granola as a new customer.',
    'Can you create a new customer called Sunshine Granola?',
    'Could you add Sunshine Granola?',
    'New customer Sunshine Granola.',
    'Por favor, agrega Sunshine Granola como cliente nuevo.',
    'Añade Sunshine Granola como nuevo cliente.',
    'Registra Sunshine Granola como cliente nuevo.',
    'Quiero crear un cliente nuevo: Sunshine Granola.',
    'Nuevo cliente Sunshine Granola.',
])
def test_customer_creation_phrasings_do_not_fall_through_to_model_again(
        client, actors, db_cursor, monkeypatch, text):
    db_cursor.execute("INSERT INTO customers(name,active) VALUES('Sunshine Granola',true)")
    calls = model(monkeypatch, ('resolve', {'kind': 'customer', 'query': 'Sunshine Granola'}))
    assert only_card(turn(client, own_chat(client, actors), text), 'choices')['existing_customer_only']
    assert len(calls) == 1


@pytest.mark.parametrize('text,language', [
    ('We need Sunshine Granola in our customer list.', 'en'),
    ('Necesitamos Sunshine Granola en nuestra lista de clientes.', 'es'),
])
def test_semantic_customer_creation_can_finish_after_exact_resolution(
        client, actors, db_cursor, monkeypatch, text, language):
    db_cursor.execute("INSERT INTO customers(name,active) VALUES('Sunshine Granola',true) RETURNING id")
    ident = db_cursor.fetchone()['id']
    chat = own_chat(client, actors)
    calls = model(monkeypatch, ('resolve', {'kind': 'customer', 'query': 'Sunshine Granola'}),
                  ('prepare_create_customer', {'name': 'Sunshine Granola'}))
    card = only_card(turn(client, chat, text, language=language), 'choices')
    assert card['existing_customer_only'] and [c['id'] for c in card['result']['candidates']] == [ident]
    assert len(calls) == 2
    assert only_card(turn(client, chat, '', choice_id=card['id'], selected_id=ident), 'handoff')
    assert len(calls) == 2


@pytest.mark.parametrize('language', ['en', 'es'])
def test_fuzzy_customer_selection_finishes_customer_add_without_loop(
        client, actors, db_cursor, monkeypatch, language):
    db_cursor.execute("INSERT INTO customers(name,active) VALUES('Sunshine Granola',true) RETURNING id")
    ident = db_cursor.fetchone()['id']
    chat = own_chat(client, actors)
    calls = model(monkeypatch, ('resolve', {'kind': 'customer', 'query': 'Sunshine Granolaa'}))
    text = 'Add Sunshine Granolaa as a new customer' if language == 'en' else 'Agrega Sunshine Granolaa como cliente nuevo'
    card = only_card(turn(client, chat, text, language=language), 'choices')
    assert not card.get('existing_customer_only'), 'A similar name can be a different company'
    choice = next(row for row in card['result']['candidates'] if row['id'] == ident)
    assert choice['tier'] == 'trigram' and choice['score'] <= 0.79
    db_cursor.execute('SELECT state FROM assistant_sessions WHERE id=%s', (chat[0],))
    assert ident not in db_cursor.fetchone()['state'].get('resolved', {}).get('customer', [])
    assert only_card(turn(client, chat, '', choice_id=card['id'], selected_id=ident), 'handoff')
    assert len(calls) == 1


@pytest.mark.parametrize('language,text_template', [
    ('en', 'New order for {name}: 2 cases'),
    ('en', 'Create an order for {name}: 2 cases'),
    ('en', 'Add 2 cases to the order for {name}'),
    ('es', 'Pedido nuevo para {name}: 2 cajas'),
    ('es', 'Crea un pedido para {name}: 2 cajas'),
    ('es', 'Agrega 2 cajas al pedido para {name}'),
])
def test_existing_customer_still_continues_an_order(client, actors, db_cursor, monkeypatch, language, text_template):
    data = seed(db_cursor)
    customer_id = data['customer']
    db_cursor.execute('SELECT name FROM customers WHERE id=%s', (customer_id,))
    name = db_cursor.fetchone()['name']
    product_id = data['products']['food']
    calls = model(monkeypatch, ('resolve', {'kind': 'customer', 'query': name}),
                  resolve_product(db_cursor, product_id), ('prepare_create_order', {
                      'customer_id': customer_id, 'lines': [{'product_id': product_id, 'quantity': 2, 'unit': 'cases'}]}))
    text = text_template.format(name=name)
    card = only_card(turn(client, own_chat(client, actors), text, language=language), 'draft')
    assert card['prepared']['action'] == 'create_order'
    assert len(calls) == 3


@pytest.mark.parametrize('language', ['en', 'es'])
def test_truly_new_customer_still_gets_recordable_draft(
        client, actors, db_cursor, monkeypatch, customer_ticket_schema, language):
    name = 'Quasar ' + uuid4().hex
    calls = model(monkeypatch, ('prepare_create_customer', {'name': name}))
    text = f'Add {name} as a new customer' if language == 'en' else f'Agrega {name} como cliente nuevo'
    card = only_card(turn(client, own_chat(client, actors), text, language=language), 'draft')
    assert card['prepared']['action'] == 'create_customer' and card['prepared']['can_commit']
    assert len(calls) == 1


@pytest.mark.parametrize('language', ['en', 'es'])
@pytest.mark.parametrize('tool', ['resolve', 'prepare_create_customer'])
def test_existing_new_customer_choice_preserves_requested_order(
        client, actors, db_cursor, monkeypatch, language, tool):
    data = seed(db_cursor)
    ident, product_id = data['customer'], data['products']['food']
    db_cursor.execute('SELECT name FROM customers WHERE id=%s', (ident,))
    name = db_cursor.fetchone()['name']
    args = {'kind': 'customer', 'query': name} if tool == 'resolve' else {'name': name}
    chat = own_chat(client, actors)
    model(monkeypatch, (tool, args))
    text = (f'New customer {name} wants an order for 2 cases of Granola' if language == 'en' else
            f'Cliente nuevo {name} quiere un pedido de 2 cajas de Granola')
    choice = only_card(turn(client, chat, text, language=language), 'choices')
    assert choice['existing_customer_only']
    calls = model(monkeypatch, resolve_product(db_cursor, product_id), ('prepare_create_order', {
        'customer_id': ident, 'lines': [{'product_id': product_id, 'quantity': 2, 'unit': 'cases'}]}))
    draft = only_card(turn(client, chat, '', choice_id=choice['id'], selected_id=ident), 'draft')
    assert draft['prepared']['action'] == 'create_order' and draft['prepared']['can_commit']
    assert len(calls) == 2


@pytest.mark.parametrize('language', ['en', 'es'])
@pytest.mark.parametrize('active', [True, False])
def test_backend_duplicate_blockers_still_finish_on_a_draft(
        client, actors, db_cursor, monkeypatch, customer_ticket_schema, language, active):
    db_cursor.execute('INSERT INTO customers(name,active) VALUES(%s,%s) RETURNING id', ('Sunshine Granola', active))
    ident = db_cursor.fetchone()['id']
    calls = model(monkeypatch, ('prepare_create_customer', {'name': 'Sunshine Granola, Inc.'}))
    text = ('Add Sunshine Granola, Inc. as a new customer' if language == 'en' else
            'Agrega Sunshine Granola, Inc. como cliente nuevo')
    draft = only_card(turn(client, own_chat(client, actors), text, language=language), 'draft')
    assert not draft['prepared']['can_commit']
    blocker = next(b for b in draft['prepared']['blockers'] if b['code'] == 'CUSTOMER_ALREADY_EXISTS')
    assert blocker['existing_customer']['id'] == ident
    assert len(calls) == 1


@pytest.mark.parametrize('text', [
    'Grain Supply va a mandar 20 cajas de avena el próximo martes, PO C4F14D. Agrégalo como entrega esperada.',
    'Grain Supply va a mandar 20 cajas de avena el próximo martes.',
    'Grain Supply mandará 20 cajas de avena el próximo martes.',
    'Agrégalo como entrega esperada.',
    'Agrégala como entrega esperada.',
    'Añádelo como entrega esperada.',
    'Crea una entrega esperada para Grain Supply.',
    'Grain Supply enviará 20 cajas de avena.',
])
def test_spanish_expected_deliveries_have_plain_reply_and_office_link_without_model(
        client, actors, monkeypatch, text):
    calls = model(monkeypatch)
    card = only_card(turn(client, own_chat(client, actors), text, language='es'), 'handoff')
    assert card['message'] == 'Cree la entrega esperada en la pantalla Oficina.'
    assert card['actions'] == [{'href': '/dashboard/office.html?action=create_expected_receipt',
                                'label': 'Crear entrega esperada'}]
    assert calls == []


@pytest.mark.parametrize('text', [
    'Recibimos 20 cajas de avena de Grain Supply.',
    'Grain Supply mandó 20 cajas de avena ayer.',
    'Grain Supply va a mandar un correo mañana.',
    'The customer is sending a payment.',
])
def test_delivery_handoff_keeps_recorded_receives_and_non_goods_messages_out(text):
    assert office_handoff(text, 'es') is None
