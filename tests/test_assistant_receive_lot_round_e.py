"""Receiving supplier labels survive voice parsing, model mistakes and buttons."""
import pytest
from psycopg2.extras import Json

from assistant_receive import lot_facts_from_text, merge_receive_lot_facts, normalize_receive_arguments
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
    ('supplier lot ABC. 123.', {}),
    ('supplier lot ABC! 123.', {}),
    ('supplier lot ABC? 123.', {}),
    ('supplier lot ABC\n123.', {}),
    ('supplier lot A001, lot A002.', {}),
    ('lote del proveedor A001, lote A002.', {}),
    ('supplier lot A001, A002, lot A003.', {}),
    ('supplier lot A001 and A002, lot A003.', {}),
    ('supplier lot A001, A002, lot A003, internal lot FL-009.', {'lot_code': 'FL-009'}),
    ('internal lot A001, A002, internal lot A003, supplier lot SUP-5521.', {'supplier_lot_code': 'SUP-5521'}),
    ('supplier lot A001, lot A001.', {'supplier_lot_code': 'A001'}),
    ('supplier lot one two three, lot 123.', {'supplier_lot_code': '123'}),
    ('supplier lot ABC. BOL 123.', {'supplier_lot_code': 'ABC'}),
    ('supplier lot ABC, internal lot 123.', {'supplier_lot_code': 'ABC', 'lot_code': '123'}),
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


@pytest.mark.parametrize('text', [
    'supplier lot A001, lot A002.',
    'lote del proveedor A001, lote A002.',
    'supplier lot ABC. 123.',
    'supplier lot A001 and A002, lot A003.',
])
def test_ambiguous_followup_clears_only_its_prior_supplier_scalar(text):
    previous = {'supplier_lot_code': 'A001', 'lot_code': 'FL-009', 'quantity_lb': 50}
    merged = merge_receive_lot_facts(previous, text)
    assert merged == {'supplier_lot_code': None, 'lot_code': 'FL-009', 'quantity_lb': 50}
    assert previous == {'supplier_lot_code': 'A001', 'lot_code': 'FL-009', 'quantity_lb': 50}
    entries = [{'supplier_lot_code': 'A001'}, {'supplier_lot_code': 'A002'}]
    assert normalize_receive_arguments({'supplier_lot_code': 'A001', 'lot_code': 'A001',
        'supplier_lot_entries': entries}, merged) == {'lot_code': 'FL-009', 'supplier_lot_entries': entries}


def test_button_turn_keeps_all_receive_facts_and_internal_ambiguity_is_separate():
    previous = {'supplier_lot_code': 'A001', 'lot_code': 'FL-009', 'quantity_lb': 50}
    assert merge_receive_lot_facts(previous, '') == previous
    assert merge_receive_lot_facts(previous, 'internal lot FL-009, internal lot FL-010.') == {
        'supplier_lot_code': 'A001', 'lot_code': None, 'quantity_lb': 50}


@pytest.mark.parametrize('text', [
    'Actually no supplier lot number given.',
    'The supplier lot is unknown.',
    'supplier lot N/A.',
    'El lote del proveedor es desconocido.',
    'lote del proveedor desconocido.',
])
def test_explicit_absence_persistently_blocks_old_supplier_scalar(text):
    assert lot_facts_from_text(text) == {}
    facts = merge_receive_lot_facts({'supplier_lot_code': 'OLD-001', 'quantity_lb': 50}, text)
    assert facts == {'supplier_lot_code': None, 'quantity_lb': 50}
    assert normalize_receive_arguments({'supplier_lot_code': 'OLD-001', 'lot_code': 'OLD-001'}, facts) == {}
    assert merge_receive_lot_facts(facts, '') == facts
    restored = merge_receive_lot_facts(facts, 'supplier lot NEW-001.')
    assert normalize_receive_arguments({'supplier_lot_code': 'OLD-001'}, restored) == {'supplier_lot_code': 'NEW-001'}


def test_unknown_word_inside_real_code_and_nonabsence_prose_do_not_clear_prior_facts():
    assert lot_facts_from_text('supplier lot UNKNOWN-123.') == {'supplier_lot_code': 'UNKNOWN-123'}
    previous = {'supplier_lot_code': 'OLD-001', 'quantity_lb': 50}
    assert merge_receive_lot_facts(previous, 'The supplier lot identity needs checking.') == previous


@pytest.mark.parametrize('text', ['internal lot unknown.', 'El lote interno es desconocido.'])
def test_explicit_internal_absence_keeps_supplier_and_blocks_old_internal_code(text):
    facts = merge_receive_lot_facts({'supplier_lot_code': 'SUP-001', 'lot_code': 'OLD-001', 'quantity_lb': 50}, text)
    assert facts == {'supplier_lot_code': 'SUP-001', 'lot_code': None, 'quantity_lb': 50}
    assert normalize_receive_arguments({'lot_code': 'OLD-001'}, facts) == {'supplier_lot_code': 'SUP-001'}
    assert merge_receive_lot_facts(facts, '') == facts


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


@pytest.mark.parametrize('spanish', [False, True])
def test_ambiguous_supplier_followup_does_not_reapply_prior_scalar_to_commingled_model_args(
        client, chat, payload, db_cursor, monkeypatch, spanish):
    incomplete = {'product_id': payload['product_id'], 'cases': 2, 'case_size_lb': 25,
                  'bol_reference': '5521', 'supplier_id': None}
    model(monkeypatch, resolve_product(db_cursor, payload['product_id']), ('prepare_receive', incomplete))
    original = ('Recibimos 50 libras en total, lote del proveedor A001.' if spanish else
                'Received 50 lb total, supplier lot A001.')
    initial = card(turn(client, chat, original), 'missing_facts')
    assert initial['known_facts']['supplier_lot_code'] == 'A001'
    entries = [{'supplier_lot_code': 'A001'}, {'supplier_lot_code': 'A002'}]
    model(monkeypatch, ('prepare_receive', {**incomplete, 'supplier_lot_code': 'A001',
        'lot_code': 'A001', 'supplier_lot_entries': entries}))
    followup = ('En realidad, lote del proveedor A001, lote A002.' if spanish else
                'Actually, supplier lot A001, lot A002.')

    missing = card(turn(client, chat, followup), 'missing_facts')

    assert 'supplier_lot_code' not in missing['known_facts']
    assert missing['known_facts']['supplier_lot_entries'] == entries
    assert missing['known_facts']['quantity_lb'] == 50
    db_cursor.execute('SELECT state FROM assistant_sessions WHERE id=%s', (chat[0],))
    facts = db_cursor.fetchone()['state']['receive_facts']
    assert facts['quantity_lb'] == 50
    assert facts['supplier_lot_code'] is None


@pytest.mark.parametrize('spanish', [False, True])
def test_quantity_correction_blocks_saved_stale_scalar_and_keeps_commingled_entries(
        client, chat, payload, db_cursor, monkeypatch, spanish):
    initial_args = {**payload, 'supplier_lot_code': 'A001', 'lot_code': 'A001'}
    model(monkeypatch, *calls(db_cursor, payload, initial_args))
    text = 'Recibimos 50 libras en total, lote del proveedor A001.' if spanish else 'Received 50 lb total, supplier lot A001.'
    original = card(turn(client, chat, text), 'draft')
    entries = [{'supplier_lot_code': 'A001'}, {'supplier_lot_code': 'A002'}]
    db_cursor.execute('SELECT state FROM assistant_sessions WHERE id=%s', (chat[0],))
    state = db_cursor.fetchone()['state']
    # A restored pre-fix context can retain old model scalars alongside its
    # commingled entries. All direct correction prepares must normalize it too.
    state['receive_context']['args'].update(supplier_lot_code='A001', lot_code='A001', supplier_lot_entries=entries)
    db_cursor.execute('UPDATE assistant_sessions SET state=%s WHERE id=%s', (Json(state), chat[0]))
    requests = model(monkeypatch)
    correction = ('En realidad eran 60 libras en total, lote del proveedor A001, lote A002.' if spanish else
                  'Actually it was 60 lb total, supplier lot A001, lot A002.')

    replacement = card(turn(client, chat, correction), 'draft')

    assert requests == [], 'FL must normalize the direct saved-context prepare without another model request'
    assert replacement['supersedes_draft_id'] == original['id']
    draft = replacement['prepared']['draft']
    assert draft['total_lb'] == 60
    assert draft['supplier_lot_code'] == 'N/A'
    assert draft['lot_code'] != 'A001'
    assert [entry['supplier_lot_code'] for entry in draft['supplier_lot_entries']] == ['A001', 'A002']
    assert draft['identity_status'] == 'identified'
    db_cursor.execute('SELECT prepare_body FROM assistant_drafts WHERE id=%s', (replacement['id'],))
    prepared = db_cursor.fetchone()['prepare_body']
    assert 'supplier_lot_code' not in prepared and 'lot_code' not in prepared
    assert prepared['supplier_lot_entries'] == entries
    db_cursor.execute('SELECT status FROM assistant_drafts WHERE id=%s', (original['id'],))
    assert db_cursor.fetchone()['status'] == 'cancelled'
    db_cursor.execute('SELECT state FROM assistant_sessions WHERE id=%s', (chat[0],))
    facts = db_cursor.fetchone()['state']['receive_facts']
    assert facts == {'quantity_lb': 60, 'supplier_lot_code': None}
