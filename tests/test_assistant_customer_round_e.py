"""The real CUS receipt continues the original order without recording it."""
import json
from uuid import uuid4

import pytest

from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import (enabled, reason_catalog, customer_ticket_schema,
                                     model, turn, record, resolve_product)  # noqa: F401
from tests.test_write_tickets import headers
from tests.test_order_tickets import seed
from assistant_voice import normalize_voice_text

pytestmark = pytest.mark.db


@pytest.mark.parametrize('role', ['office', 'owner'])
@pytest.mark.parametrize('spanish', [False, True])
@pytest.mark.parametrize('reload_chat', [False, True])
def test_new_customer_record_continues_original_order_to_draft(
        client, actors, db_cursor, monkeypatch, customer_ticket_schema, role, spanish, reload_chat):
    catalog = seed(db_cursor)
    h = headers(actors[role]['key'])
    chat = client.post('/assistant/session', headers=h).json()['session_id'], h
    name = 'New Round E Market ' + uuid4().hex
    product_id = catalog['products']['food']
    db_cursor.execute('SELECT name FROM products WHERE id=%s', (product_id,))
    product_name = db_cursor.fetchone()['name']
    po = 'ROUND-E-' + catalog['suffix']
    original = (f'Pedido nuevo para {name}: 2 cajas de {product_name}, PO {po}.' if spanish else
                f'New order for {name}: 2 cases of {product_name}, PO {po}.')
    model(monkeypatch, ('resolve', {'kind': 'customer', 'query': name}))
    choice = turn(client, chat, original).json()['cards'][0]
    assert choice['kind'] == 'choices' and choice['resolution_kind'] == 'customer'
    model(monkeypatch, ('prepare_create_customer', {'name': name}))
    create_text = (f'Crea primero un cliente nuevo llamado {name} y luego continúa el pedido.' if spanish else
                   f'Create a new customer named {name} first, then continue the order.')
    response = turn(client, chat, create_text)
    assert response.status_code == 200, response.text
    customer = response.json()['cards'][0]
    assert customer['prepared']['action'] == 'create_customer' and customer['prepared']['can_commit']
    db_cursor.execute('SELECT count(*) AS n FROM customers WHERE name=%s', (name,))
    assert db_cursor.fetchone()['n'] == 0

    saved = record(client, chat, customer)
    assert saved.status_code == 200, saved.text
    receipt = saved.json()['result']
    assert receipt['receipt_number'].startswith('CUS-')
    assert record(client, chat, customer).json()['result'] == receipt | {'replayed': True}
    if reload_chat:
        resumed = client.post('/assistant/resume', headers=h, json={'session_id': chat[0]})
        assert resumed.status_code == 200, resumed.text
        restored = next(row for row in resumed.json()['drafts'] if row['id'] == customer['id'])
        assert restored['status'] == 'committed' and restored['result'] == receipt | {'replayed': True}

    requests = model(monkeypatch, resolve_product(db_cursor, product_id), ('prepare_create_order', {
        'customer_id': receipt['customer_id'], 'customer_po': po,
        'lines': [{'product_id': product_id, 'quantity': 2, 'unit': 'cases'}]}))
    continue_text = (f'Continúa el pedido usando el cliente del recibo {receipt["receipt_number"]}.' if spanish else
                     f'Continue the order using the customer from receipt {receipt["receipt_number"]}.')
    response = turn(client, chat, continue_text)
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['kind'] == 'draft' and card['prepared']['action'] == 'create_order'
    assert card['prepared']['can_commit']
    draft = card['prepared']['draft']
    assert draft['customer'] == name and draft['customer_po'] == po
    assert draft['lines'][0]['product_id'] == product_id
    assert draft['lines'][0]['quantity'] == 2 and draft['lines'][0]['unit'] == 'cases'
    assert draft['total_lb'] == 50
    supplied = json.dumps(requests[0]['input'], ensure_ascii=False)
    assert normalize_voice_text(original) in supplied and receipt['receipt_number'] in supplied
    assert 'FL confirmed the customer Record:' in supplied
    assert 'wt_' not in supplied
    db_cursor.execute('SELECT count(*) AS n FROM customers WHERE name=%s', (name,))
    assert db_cursor.fetchone()['n'] == 1
    db_cursor.execute('SELECT count(*) AS n FROM sales_orders WHERE customer_id=%s', (receipt['customer_id'],))
    assert db_cursor.fetchone()['n'] == 0, 'The order draft requires its own human Record'
