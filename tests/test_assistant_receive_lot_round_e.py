"""Receiving supplier labels survive voice parsing, model mistakes and buttons."""
import pytest

from assistant_receive import lot_facts_from_text, normalize_receive_arguments
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import enabled, reason_catalog, chat, model, turn, record, resolve_product  # noqa: F401
from tests.test_write_tickets import payload  # noqa: F401
from tests.test_assistant_voice_e2e_round_c import card

pytestmark = pytest.mark.db


@pytest.mark.parametrize('text,expected', [
    ('supplier lot number 005721.', {'supplier_lot_code': '005721'}),
    ('lote del proveedor cinco siete dos uno.', {'supplier_lot_code': '5721'}),
    ('lot P one five eight five.', {'supplier_lot_code': 'P1585'}),
    ('lote P uno cinco ocho cinco.', {'supplier_lot_code': 'P1585'}),
    ('supplier lot P 1585.', {'supplier_lot_code': 'P 1585'}),
    ('supplier lot ABC 123.', {'supplier_lot_code': 'ABC 123'}),
    ('supplier lot ABC 123 45.', {'supplier_lot_code': 'ABC 123 45'}),
    ('supplier lot ABC 123, BOL 5521.', {'supplier_lot_code': 'ABC 123'}),
    ('supplier lot ABC 123 and BOL 5521.', {'supplier_lot_code': 'ABC 123'}),
    ('lote del proveedor ABC 123 con BOL 5521.', {'supplier_lot_code': 'ABC 123'}),
    ('supplier lot ABC 123 from Grain Supply.', {'supplier_lot_code': 'ABC 123'}),
    ('supplier lot SUP-5521, internal lot FL-009.', {'supplier_lot_code': 'SUP-5521', 'lot_code': 'FL-009'}),
    ('lote del proveedor SUP-5521, lote interno FL-009.', {'supplier_lot_code': 'SUP-5521', 'lot_code': 'FL-009'}),
    ('supplier lot unknown.', {}),
    ('supplier lot number is on the bag.', {}),
    ('supplier lot identity needs checking.', {}),
    ('supplier lot IDENTITY NEEDS CHECKING.', {}),
    ('supplier lot A001 and supplier lot A002.', {}),
    ('supplier lot A001 and A002.', {}),
    ('supplier lot A001, A002.', {}),
    ('supplier lot ABC 123 and DEF 456.', {}),
    ('supplier lot ABC 123, DEF 456.', {}),
])
def test_human_receive_lot_labels_are_separate(text, expected):
    facts = lot_facts_from_text(text)
    assert facts == expected
    args = {'lot_code': 'MODEL-WRONG', 'supplier_lot_code': None}
    normalized = normalize_receive_arguments(args, facts)
    assert args == {'lot_code': 'MODEL-WRONG', 'supplier_lot_code': None}
    for key, value in expected.items():
        assert normalized[key] == value
    if expected.get('supplier_lot_code') and not expected.get('lot_code'):
        assert 'lot_code' not in normalized
    if not expected:
        assert normalize_receive_arguments({'supplier_lot_code': 'MODEL-UNCHANGED'}, facts) == {
            'supplier_lot_code': 'MODEL-UNCHANGED'}


@pytest.mark.parametrize('code', ['ABC 123', 'P 1585', 'ABC 123 45'])
def test_correct_model_supplier_lot_is_not_truncated_by_source_binding(code):
    arguments = {'supplier_lot_code': code, 'lot_code': code}
    normalized = normalize_receive_arguments(arguments, lot_facts_from_text('supplier lot ' + code + '.'))
    assert normalized == {'supplier_lot_code': code}


def calls(db_cursor, payload, prepare):
    db_cursor.execute('SELECT name FROM suppliers WHERE id=%s', (payload['supplier_id'],))
    supplier_name = db_cursor.fetchone()['name']
    return (resolve_product(db_cursor, payload['product_id']),
            ('resolve', {'kind': 'supplier', 'query': supplier_name, 'context': {'action': 'receive'}}),
            ('prepare_receive', prepare))


@pytest.mark.parametrize('spanish', [False, True])
@pytest.mark.parametrize('wrong_model_field', [False, True])
def test_supplier_lot_from_human_transcript_reaches_real_receive_field_and_ledger(
        client, chat, payload, db_cursor, monkeypatch, spanish, wrong_model_field):
    args = {key: value for key, value in payload.items() if key not in ('supplier_lot_code', 'lot_code')}
    if wrong_model_field:
        args['lot_code'] = '005721'
    model(monkeypatch, *calls(db_cursor, payload, args))
    text = ('Recibimos 2 cajas de este ingrediente, 25 libras cada una, lote del proveedor cero cero cinco siete dos uno.' if spanish else
            'Received 2 cases of this ingredient, 25 lb each, supplier lot zero zero five seven two one.')

    draft = card(turn(client, chat, text), 'draft')

    assert draft['prepared']['draft']['supplier_lot_code'] == '005721'
    assert draft['prepared']['draft']['lot_code'] != '005721'
    assert draft['prepared']['draft']['identity_status'] == 'identified'
    db_cursor.execute('SELECT prepare_body FROM assistant_drafts WHERE id=%s', (draft['id'],))
    stored = db_cursor.fetchone()['prepare_body']
    assert stored['supplier_lot_code'] == '005721'
    assert not stored.get('lot_code')
    response = record(client, chat, draft)
    assert response.status_code == 200, response.text
    db_cursor.execute('SELECT lot_code,supplier_lot_code FROM lots WHERE id=%s', (response.json()['result']['lot_id'],))
    lot = db_cursor.fetchone()
    assert lot['supplier_lot_code'] == '005721'
    assert lot['lot_code'] == draft['prepared']['draft']['lot_code']


@pytest.mark.parametrize('code', ['ABC 123', 'P 1585'])
@pytest.mark.parametrize('spanish', [False, True])
def test_spaced_supplier_code_survives_correct_model_draft_and_real_ledger(
        client, chat, payload, db_cursor, monkeypatch, code, spanish):
    args = {**payload, 'supplier_lot_code': code, 'lot_code': code}
    model(monkeypatch, *calls(db_cursor, payload, args))
    text = (f'Recibimos 2 cajas, 25 libras cada una, lote del proveedor {code}, BOL 5521.' if spanish else
            f'Received 2 cases, 25 lb each, supplier lot {code}, BOL 5521.')

    draft = card(turn(client, chat, text), 'draft')

    assert draft['prepared']['draft']['supplier_lot_code'] == code
    assert draft['prepared']['draft']['lot_code'] != code
    response = record(client, chat, draft)
    assert response.status_code == 200, response.text
    db_cursor.execute('SELECT supplier_lot_code FROM lots WHERE id=%s', (response.json()['result']['lot_id'],))
    assert db_cursor.fetchone()['supplier_lot_code'] == code


@pytest.mark.parametrize('spanish', [False, True])
def test_explicit_supplier_and_internal_receive_lots_remain_separate(
        client, chat, payload, db_cursor, monkeypatch, spanish):
    args = {**payload, 'supplier_lot_code': None, 'lot_code': 'SUP-5521'}
    model(monkeypatch, *calls(db_cursor, payload, args))
    text = ('Recibimos 2 cajas, 25 libras cada una, lote del proveedor SUP-5521, lote interno FL-009.' if spanish else
            'Received 2 cases, 25 lb each, supplier lot SUP-5521, internal lot FL-009.')

    draft = card(turn(client, chat, text), 'draft')

    assert draft['prepared']['draft']['supplier_lot_code'] == 'SUP-5521'
    assert draft['prepared']['draft']['lot_code'] == 'FL-009'


@pytest.mark.parametrize('spanish', [False, True])
def test_supplier_button_retains_explicit_supplier_lot_after_model_omits_it(
        client, chat, payload, db_cursor, monkeypatch, spanish):
    incomplete = {'product_id': payload['product_id'], 'cases': 2, 'case_size_lb': 25,
                  'bol_reference': payload['bol_reference'], 'supplier_id': None}
    model(monkeypatch, resolve_product(db_cursor, payload['product_id']), ('prepare_receive', incomplete))
    text = ('Recibimos 2 cajas, 25 libras cada una, lote del proveedor P uno cinco ocho cinco.' if spanish else
            'Received 2 cases, 25 lb each, supplier lot P one five eight five.')
    missing = card(turn(client, chat, text), 'missing_facts')
    assert missing['known_facts']['supplier_lot_code'] == 'P1585'
    assert payload['supplier_id'] in {row['id'] for row in missing['supplier_choices']}
    complete = {**incomplete, 'supplier_id': payload['supplier_id'], 'lot_code': 'P1585'}
    model(monkeypatch, ('prepare_receive', complete))

    draft = card(turn(client, chat, text='', choice_id=missing['choice_id'], selected_id=payload['supplier_id']), 'draft')

    assert draft['prepared']['draft']['supplier_lot_code'] == 'P1585'
    assert draft['prepared']['draft']['lot_code'] != 'P1585'
