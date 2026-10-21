"""Review regressions use real Assistant prepare, record and ledger handlers."""
import pytest

import lot_confirmation
from assistant_replies import refusal_message
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import enabled, reason_catalog, chat, model, turn, record, resolve_product  # noqa: F401
from tests.test_write_tickets import payload  # noqa: F401
from tests.test_assistant_receive_lot_round_e import calls
from tests.test_assistant_voice_e2e_round_c import card

pytestmark = pytest.mark.db


@pytest.mark.parametrize('text', [
    '10 bags oats 50 lb from Grain Millers, no lot on label',
    '10 bags oats, no lot on label, from Grain Millers.',
    '10 bags oats, no lot, from Grain Millers.',
    '10 bags oats, lot unreadable, from Grain Millers.',
    '10 bolsas de avena, sin lote en la etiqueta, de Grain Millers.',
    '10 bolsas de avena, no tiene lote, de Grain Millers.',
    '10 bolsas de avena, sin lote, de Grain Millers.',
    '10 bolsas de avena, lote ilegible, de Grain Millers.',
    'no lot',
    'No tiene lote',
    'Sin lote',
])
def test_sentence_no_lot_reaches_real_draft_and_records_unidentified_exception(
        client, chat, payload, db_cursor, monkeypatch, text):
    args = {k: v for k, v in payload.items() if k not in ('supplier_lot_code', 'lot_code')}
    # Omit the flag deliberately: the worker's text, rather than a model guess,
    # authorizes the exception and removes an obsolete model supplier code.
    args['supplier_lot_code'] = 'MODEL-OLD'
    model(monkeypatch, *calls(db_cursor, payload, args))
    draft = card(turn(client, chat, text), 'draft')
    assert draft['prepared']['draft']['identity_status'] == 'unidentified'
    db_cursor.execute('SELECT prepare_body FROM assistant_drafts WHERE id=%s', (draft['id'],))
    prepared = db_cursor.fetchone()['prepare_body']
    assert prepared['no_supplier_lot'] is True
    assert not prepared.get('supplier_lot_code')
    response = record(client, chat, draft)
    assert response.status_code == 200, response.text
    result = response.json()['result']
    assert result['identity_status'] == 'unidentified'
    assert result['supplier_id'] == payload['supplier_id']
    assert result['exception_id']
    db_cursor.execute('SELECT supplier_id,identity_status,identify_by FROM lots WHERE id=%s', (result['lot_id'],))
    lot = db_cursor.fetchone()
    assert lot['supplier_id'] == payload['supplier_id']
    assert lot['identity_status'] == 'unidentified'
    db_cursor.execute('SELECT created_at FROM transactions WHERE id=%s', (result['transaction_id'],))
    due = lot_confirmation.identification_deadline(db_cursor.fetchone()['created_at'])
    assert lot['identify_by'] == due.date()
    db_cursor.execute('SELECT kind,due_at,lot_id FROM exceptions WHERE id=%s', (result['exception_id'],))
    exception = db_cursor.fetchone()
    assert exception['kind'] == 'UNIDENTIFIED_LOT'
    assert exception['due_at'] == due
    assert exception['lot_id'] == result['lot_id']


@pytest.mark.parametrize('spanish', [False, True])
def test_missing_lot_refusal_tells_worker_to_say_phrase_and_rejects_model_exception(
        client, chat, payload, db_cursor, monkeypatch, spanish):
    args = {k: v for k, v in payload.items() if k not in ('supplier_lot_code', 'lot_code')}
    args['no_supplier_lot'] = True
    model(monkeypatch, *calls(db_cursor, payload, args))
    text = 'Recibimos 2 cajas de avena de Grain Millers.' if spanish else 'Received 2 cases of oats from Grain Millers.'
    refusal = card(turn(client, chat, text), 'refusal')
    assert refusal['error_code'] == 'SUPPLIER_LOT_REQUIRED'
    message = refusal['message']
    assert ('di "sin lote en la etiqueta"' if spanish else 'say "no lot on label"') in message
    assert not any(word in message.lower() for word in ('select', 'seleccion'))
    db_cursor.execute('SELECT count(*) AS n FROM lots WHERE product_id=%s', (payload['product_id'],))
    assert db_cursor.fetchone()['n'] == 0


@pytest.mark.parametrize('spanish', [False, True])
def test_sentence_no_lot_fact_survives_real_supplier_choice_and_record(
        client, chat, payload, db_cursor, monkeypatch, spanish):
    args = {k: v for k, v in payload.items() if k not in ('supplier_lot_code', 'lot_code')}
    args['supplier_id'] = None
    model(monkeypatch, resolve_product(db_cursor, payload['product_id']), ('prepare_receive', args))
    text = ('10 bolsas de avena, sin lote, 25 libras cada una.' if spanish else
            '10 bags oats, no lot on label, 25 lb each.')
    missing = card(turn(client, chat, text), 'missing_facts')
    assert missing['known_facts']['no_supplier_lot'] is True
    model(monkeypatch, ('prepare_receive', {**args, 'supplier_id': payload['supplier_id']}))
    draft = card(turn(client, chat, text='', choice_id=missing['choice_id'], selected_id=payload['supplier_id']), 'draft')
    assert draft['prepared']['draft']['identity_status'] == 'unidentified'
    response = record(client, chat, draft)
    assert response.status_code == 200, response.text
    assert response.json()['result']['identity_status'] == 'unidentified'
    assert response.json()['result']['supplier_id'] == payload['supplier_id']


@pytest.mark.parametrize('spanish', [False, True])
def test_human_supplier_code_wins_over_sentence_no_lot_and_model_flag(
        client, chat, payload, db_cursor, monkeypatch, spanish):
    args = {**payload, 'no_supplier_lot': True, 'supplier_lot_code': 'MODEL-OLD'}
    model(monkeypatch, *calls(db_cursor, payload, args))
    text = ('10 bolsas de avena, sin lote; lote del proveedor SUP-ACTUAL.' if spanish else
            '10 bags oats, no lot on label; supplier lot SUP-ACTUAL.')
    draft = card(turn(client, chat, text), 'draft')
    assert draft['prepared']['draft']['identity_status'] == 'identified'
    assert draft['prepared']['draft']['supplier_lot_code'] == 'SUP-ACTUAL'
    response = record(client, chat, draft)
    assert response.status_code == 200, response.text
    result = response.json()['result']
    assert result['supplier_lot_code'] == 'SUP-ACTUAL'
    assert result['identity_status'] == 'identified'
    assert result['exception_id'] is None


@pytest.mark.parametrize('spanish', [False, True])
def test_assistant_missing_lot_copy_overrides_dashboard_select_copy(spanish):
    detail = {'code': 'SUPPLIER_LOT_REQUIRED', 'message': 'Select No lot on label.',
              'message_es': 'Selecciona Sin lote en la etiqueta.'}
    message = refusal_message(detail, spanish)
    assert ('di "sin lote en la etiqueta"' if spanish else 'say "no lot on label"') in message
    assert not any(word in message.lower() for word in ('select', 'seleccion'))
