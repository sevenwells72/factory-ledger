"""Floor schema and Assistant human selection use the shared physical-lot contract."""
from pathlib import Path

import pytest
import yaml

import assistant_tools
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import (enabled, reason_catalog, chat, model, turn,
                                     record, resolve_product)  # noqa: F401
from tests.test_write_tickets_part2 import items  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


def test_floor_keeps_22_actions_and_required_physical_evidence():
    schema = yaml.safe_load((ROOT / 'gpt-configs/schemas/openapi-floor.yaml').read_text())
    assert sum(method in {'get', 'post', 'patch', 'put', 'delete', 'head', 'options', 'trace'}
               for path in schema['paths'].values() for method in path) == 22
    for name in ('MakeRequest', 'PackRequest'):
        request = schema['components']['schemas'][name]
        assert {'ingredient_lot_allocations', 'lot_confirmations'} <= set(request['required'])
    assert 'supplier_lot' in schema['components']['schemas']['LotConfirmation']['properties']['method']['enum']
    instructions = (ROOT / 'gpt-configs/dist/GPT_FLOOR_INSTRUCTIONS.md').read_text()
    assert len(instructions) < 8000
    assert '1 lot → auto' not in instructions
    for rule in ('America/New_York', 'YY-MM-DD-XXXX-NNN', 'auto-deducts add-ins',
                 'service/charge', 'case_size_lb required', 'SO line quantities',
                 'requires available supplier lot', 'proportionally', 'absolute Railway',
                 'Migrations are idempotent', 'kosher holds', 'NEVER preselect'):
        assert rule in instructions
    shared = (ROOT / 'gpt-configs/sources/shared-rules.md').read_text().rstrip()
    floor = (ROOT / 'gpt-configs/sources/floor-specific.md').read_text().rstrip()
    assert instructions.endswith(shared + '\n\n' + floor + '\n')


def test_model_cannot_assert_supplier_selection_or_split_evidence():
    for tool in assistant_tools.tools([]):
        if tool['name'] in ('prepare_make', 'prepare_pack'):
            assert 'ingredient_lot_allocations' not in tool['parameters']['properties']
            assert 'lot_confirmations' not in tool['parameters']['properties']
    assert assistant_tools.has_ui_fields({'nested': {'ingredient_lot_allocations': []}})


def make_card(client, chat, items, db_cursor, monkeypatch, action='make'):
    if action == 'make':
        model(monkeypatch, ('resolve', {'kind': 'product', 'query': items['batch']['name'], 'context': {'action': 'make'}}),
              ('prepare_make', {'product_id': items['batch']['id'], 'batches': 1}))
    else:
        model(monkeypatch, resolve_product(db_cursor, items['finished']['id']),
              ('packing_context', {'target_product_id': items['finished']['id'], 'lot_code': items['batch']['lot_code']}),
              ('prepare_pack', {'source_product_id': items['batch']['id'], 'target_product_id': items['finished']['id'], 'cases': 2}))
    response = turn(client, chat, 'Make one batch' if action == 'make' else 'Pack two cases from ' + items['batch']['lot_code'])
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['kind'] == 'draft', card
    return card


@pytest.mark.db
@pytest.mark.parametrize('action', ['make', 'pack'])
def test_assistant_supplier_choice_reprepares_signed_facts_then_records(client, chat, items, db_cursor, monkeypatch, action):
    product = items['ingredient' if action == 'make' else 'batch']
    db_cursor.execute('UPDATE lots SET supplier_lot_code=%s WHERE id=%s', (' BAG 42 ', product['lot_id']))
    card = make_card(client, chat, items, db_cursor, monkeypatch, action)
    assert card['prepared']['can_commit'] is False
    assert card['prepared']['draft']['lot_confirmation_group']['ingredients'][0]['candidates']
    selected = {'draft_id': card['id'], 'ingredient_lot_allocations': [
        {'ingredient_product_id': product['id'], 'lot_id': product['lot_id'], 'quantity_lb': 10}],
        'lot_confirmations': [{'lot_id': product['lot_id'], 'method': 'supplier_lot', 'value': 'bag42', 'selected': True}]}
    before = record(client, chat, card)
    assert before.status_code == 422, before.text
    response = client.post('/assistant/select-lots', headers=chat[1], json=selected)
    assert response.status_code == 200, response.text
    updated = response.json()['card']
    assert updated['id'] != card['id']
    assert updated['prepared']['can_commit'] is True, updated
    assert updated['prepared']['draft']['input_plan'][0]['confirmed'] is True
    # Quantities and output identity are copied from the signed ticket, never the UI.
    assert client.post('/assistant/select-lots', headers=chat[1], json=selected | {'cases': 1000}).status_code == 422
    assert record(client, chat, card).status_code == 409
    posted = record(client, chat, updated)
    assert posted.status_code == 200, posted.text
    assert posted.json()['kind'] == 'receipt'
    assert posted.json()['result']['input_plan'][0]['lot_id'] == product['lot_id']


@pytest.mark.db
def test_assistant_two_selected_supplier_lots_preserve_split_on_receipt(client, chat, items, db_cursor, monkeypatch):
    product = items['ingredient']
    db_cursor.execute('UPDATE lots SET supplier_lot_code=%s WHERE id=%s', ('SAME-42', product['lot_id']))
    db_cursor.execute('INSERT INTO lots(product_id,lot_code,supplier_lot_code) VALUES (%s,%s,%s) RETURNING id',
                      (product['id'], product['lot_code'] + '-SECOND', 'SAME-42'))
    second = db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO transactions(type) VALUES ('receive') RETURNING id")
    tid = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,50)', (tid, product['id'], second))
    card = make_card(client, chat, items, db_cursor, monkeypatch)
    response = client.post('/assistant/select-lots', headers=chat[1], json={'draft_id': card['id'],
        'ingredient_lot_allocations': [{'ingredient_product_id': product['id'], 'lot_id': lid, 'quantity_lb': quantity}
                                      for lid, quantity in [(product['lot_id'], 4), (second, 6)]],
        'lot_confirmations': [{'lot_id': lid, 'method': 'supplier_lot', 'value': ' same-42 ', 'selected': True}
                              for lid in (product['lot_id'], second)]})
    assert response.status_code == 200, response.text
    updated = response.json()['card']
    assert updated['prepared']['can_commit'] is True, updated
    posted = record(client, chat, updated)
    assert posted.status_code == 200, posted.text
    assert [(row['lot_id'], row['quantity_lb']) for row in posted.json()['result']['input_plan']] == [(product['lot_id'], 4), (second, 6)]


@pytest.mark.db
def test_failed_assistant_selection_never_revives_retired_draft(client, chat, items, db_cursor, monkeypatch):
    import fl_assistant
    card = make_card(client, chat, items, db_cursor, monkeypatch)
    async def unavailable(*args, **kwargs):
        return 422, {'detail': {'error_code': 'INVALID_LOT', 'message': 'Prepare again.'}}
    monkeypatch.setattr(fl_assistant, 'relay', unavailable)
    product = items['ingredient']
    response = client.post('/assistant/select-lots', headers=chat[1], json={'draft_id': card['id'],
        'ingredient_lot_allocations': [{'ingredient_product_id': product['id'], 'lot_id': product['lot_id'], 'quantity_lb': 10}],
        'lot_confirmations': [{'lot_id': product['lot_id'], 'method': 'full_code', 'value': product['lot_code']}]})
    assert response.status_code == 422
    db_cursor.execute('SELECT status FROM assistant_drafts WHERE id=%s', (card['id'],))
    assert db_cursor.fetchone()['status'] == 'cancelled'
    assert record(client, chat, card).status_code == 409


@pytest.mark.db
def test_assistant_different_lot_retains_zero_stock_trace_exception(client, chat, items, db_cursor, monkeypatch):
    product = items['ingredient']
    db_cursor.execute("INSERT INTO transactions(type,notes) VALUES ('adjust','supplier-lot zero-stock fixture') RETURNING id")
    transaction_id = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,-100)',
                      (transaction_id, product['id'], product['lot_id']))
    card = make_card(client, chat, items, db_cursor, monkeypatch)
    assert card['prepared']['draft']['lot_confirmation_group']['ingredients'][0]['candidates'] == []
    response = client.post('/assistant/select-lots', headers=chat[1], json={'draft_id': card['id'],
        'ingredient_lot_allocations': [],
        'lot_confirmations': [{'lot_id': product['lot_id'], 'method': 'different_lot', 'value': 'PHYSICAL-OLD-BAG'}]})
    assert response.status_code == 200, response.text
    updated = response.json()['card']
    assert updated['prepared']['can_commit'] is True, updated
    result = record(client, chat, updated)
    assert result.status_code == 200, result.text
    row = result.json()['result']['input_plan'][0]
    assert row['confirmed'] is False
    assert row['lot_exception']['physical_tag'] == 'PHYSICAL-OLD-BAG'
    assert row['lot_id'] != product['lot_id']
