"""Correction reasons stay a human choice, even with other missing facts."""
import pytest

from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import (enabled, reason_catalog, chat, model, turn,
                                     resolve_product, message_cards)  # noqa: F401
from tests.test_write_tickets_part2 import items, body as action_body  # noqa: F401
from tests.test_assistant_voice_e2e_round_c import card

pytestmark = pytest.mark.db


@pytest.mark.parametrize('action', ['adjust', 'found'])
@pytest.mark.parametrize('reason_input', ['omitted', None, '', '   '])
@pytest.mark.parametrize('quantity_missing', [False, True])
@pytest.mark.parametrize('spanish', [False, True])
def test_missing_reason_has_buttons_and_keeps_other_facts_through_continuation(
        client, chat, items, db_cursor, monkeypatch, action, reason_input, quantity_missing, spanish):
    args = action_body(action, items)
    args.pop('reason', None)
    args.pop('reason_code', None)
    quantity_field = 'delta_lb' if action == 'adjust' else 'quantity'
    identity_field = 'lot_id' if action == 'adjust' else 'product_id'
    expected_quantity = args[quantity_field]
    if quantity_missing:
        args[quantity_field] = None
    if reason_input != 'omitted':
        args['reason_code'] = reason_input
    resolved = (('resolve', {'kind': 'lot', 'query': items['ingredient']['lot_code']}) if action == 'adjust'
                else resolve_product(db_cursor, args['product_id']))
    requests = model(monkeypatch, resolved, ('prepare_' + action, args))
    text = ('Ajusta este lote.' if action == 'adjust' else 'Encontramos inventario en libras.') if spanish else (
        'Adjust this lot.' if action == 'adjust' else 'Found inventory.')
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    before = db_cursor.fetchone()['n']

    response = turn(client, chat, text, language='es' if spanish else 'en')
    choice = card(response, 'missing_facts' if quantity_missing else 'refusal')

    assert choice['reason_required'] is True
    assert choice['message'] == ('Elige un motivo de la corrección.' if spanish else 'Choose a correction reason.')
    assert choice['known_facts'][identity_field] == args[identity_field]
    assert 'reason_code' not in choice['known_facts']
    expected_reasons = client.get('/correction-reasons', headers=chat[1]).json()['reasons']
    assert choice['reasons'] == expected_reasons
    assert len(requests) == 2
    if quantity_missing:
        assert set(choice['fields']) == {quantity_field, 'reason_code'}
        assert quantity_field not in choice['known_facts']
    else:
        assert choice['known_facts'][quantity_field] == expected_quantity
    resumed = client.post('/assistant/resume', headers=chat[1], json={'session_id': chat[0]})
    assert resumed.status_code == 200
    assert resumed.json()['turns'][-1]['cards'][0] == choice
    db_cursor.execute('SELECT count(*) AS n FROM assistant_drafts WHERE session_id=%s', (chat[0],))
    assert db_cursor.fetchone()['n'] == 0

    corrected = {**args, 'reason_code': 'physical_count'}
    next_requests = model(monkeypatch, ('prepare_' + action, corrected))
    selected = card(turn(client, chat, 'El motivo de corrección es physical_count.' if spanish else
        'The correction reason is physical_count.', language='es' if spanish else 'en'),
        'missing_facts' if quantity_missing else 'draft')

    assert len(next_requests) == 1
    if quantity_missing:
        assert selected['fields'] == [quantity_field]
        assert not selected.get('reason_required')
        db_cursor.execute('SELECT count(*) AS n FROM assistant_drafts WHERE session_id=%s', (chat[0],))
        assert db_cursor.fetchone()['n'] == 0
    else:
        draft = selected['prepared']['draft']
        assert draft[identity_field] == args[identity_field]
        assert draft[quantity_field] == expected_quantity
        assert draft['reason_code'] == 'physical_count'
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    assert db_cursor.fetchone()['n'] == before


@pytest.mark.parametrize('action', ['adjust', 'found'])
@pytest.mark.parametrize('quantity_input', ['omitted', 'two pounds'])
def test_missing_reason_does_not_hide_malformed_non_reason_tool_arguments(
        client, chat, items, db_cursor, monkeypatch, action, quantity_input):
    args = action_body(action, items)
    args.pop('reason', None)
    args['reason_code'] = None
    field = 'delta_lb' if action == 'adjust' else 'quantity'
    if quantity_input == 'omitted':
        args.pop(field)
    else:
        args[field] = quantity_input
    resolved = (('resolve', {'kind': 'lot', 'query': items['ingredient']['lot_code']}) if action == 'adjust'
                else resolve_product(db_cursor, args['product_id']))
    requests = model(monkeypatch, resolved, ('prepare_' + action, args), ('prepare_' + action, args))

    response = turn(client, chat, 'Adjust the lot.' if action == 'adjust' else 'Found some stock.')

    assert response.status_code == 200, response.text
    assert response.json()['cards'] == message_cards()
    assert len(requests) == 3
    assert 'Field required' not in response.text
    db_cursor.execute('SELECT count(*) AS n FROM assistant_drafts WHERE session_id=%s', (chat[0],))
    assert db_cursor.fetchone()['n'] == 0
