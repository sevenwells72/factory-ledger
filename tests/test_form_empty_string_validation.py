"""Blank form fields must not surface as raw Pydantic text (Oct 29 2026).

Arturo's Make draft sent a blank lot-confirmation code as "" and the dashboard
showed "NOT recorded — String should have at least 1 character". The request
models still refuse "" (a lot code is evidence), but every 422 now names the
field in English and Spanish.
"""
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main
import validation_messages
import write_tickets
import ship_tickets


def headers():
    return {'X-API-Key': main.API_KEY}


def test_lot_confirmation_value_still_requires_text():
    for blank in ('', '   '):
        with pytest.raises(ValidationError) as caught:
            write_tickets.MakePrepareRequest(product_id=1, batches=1, lot_confirmations=[
                {'lot_id': 1, 'method': 'last4', 'value': '-001'}, {'lot_id': 2, 'method': 'last4', 'value': blank}])
        assert caught.value.errors()[0]['loc'] == ('lot_confirmations', 1, 'value')
    with pytest.raises(ValidationError):
        write_tickets.PackPrepareRequest(source_product_id=1, target_product_id=2, cases=1,
                                         lot_confirmations=[{'lot_id': 1, 'method': 'last4', 'value': ''}])


def test_blank_optional_fields_are_omitted_not_blank():
    # What the Make form sends with optional lot code and note left empty.
    req = write_tickets.MakePrepareRequest(product_id=1, batches=6, substitutions=[], ingredient_lots=[
        {'ingredient_product_id': 32, 'lot_id': 11}, {'ingredient_product_id': 32, 'lot_id': 12}],
        lot_confirmations=[{'lot_id': 11, 'method': 'last4', 'value': '-001'}, {'lot_id': 12, 'method': 'full_code', 'value': 'OATS-B'}])
    assert req.lot_code is None and req.note is None and len(req.ingredient_lots) == 2
    pack = write_tickets.PackPrepareRequest(source_product_id=1, target_product_id=2, cases=11)
    assert pack.target_lot_code is None and pack.case_weight_lb is None and pack.note is None


def test_standalone_ship_reference_is_required():
    with pytest.raises(ValidationError) as caught:
        ship_tickets.StandaloneShip(customer_id=1, product_id=2, quantity_lb=25, order_reference='')
    assert caught.value.errors()[0]['loc'] == ('order_reference',)


@pytest.mark.parametrize('loc,error_type,english,spanish', [
    (['body', 'lot_confirmations', 1, 'value'], 'string_too_short',
     'Lot confirmation 2: Lot label code is required. Fill it in and try again.',
     'Confirmación de lote 2: Código de la etiqueta del lote es obligatorio. Complétalo e intenta de nuevo.'),
    (['body', 'order_reference'], 'string_too_short',
     'Shipment reference is required. Fill it in and try again.',
     'Referencia de envío es obligatorio. Complétalo e intenta de nuevo.'),
    (['body', 'substitutions', 0, 'reason_code'], 'string_too_short',
     'Substitution 1: Reason is required. Fill it in and try again.',
     'Sustitución 1: Motivo es obligatorio. Complétalo e intenta de nuevo.'),
    (['body', 'bol_reference'], 'missing',
     'BOL / delivery reference is required. Fill it in and try again.',
     'BOL / referencia de entrega es obligatorio. Complétalo e intenta de nuevo.'),
    (['body', 'cases'], 'greater_than',
     'Cases: Input should be greater than 0', 'Cajones: Input should be greater than 0'),
])
def test_messages_name_the_field(loc, error_type, english, spanish):
    error = {'loc': loc, 'type': error_type, 'msg': 'Input should be greater than 0'}
    assert validation_messages.describe(error) == (english, spanish)


@pytest.mark.parametrize('path,body,loc', [
    ('/make/prepare', {'client_source': 'dashboard', 'product_id': 1, 'batches': 6,
                       'lot_confirmations': [{'lot_id': 11, 'method': 'last4', 'value': '-001'},
                                             {'lot_id': 12, 'method': 'last4', 'value': ''}]},
     ['body', 'lot_confirmations', 1, 'value']),
    ('/pack/prepare', {'client_source': 'dashboard', 'source_product_id': 1, 'target_product_id': 2, 'cases': 1,
                       'lot_confirmations': [{'lot_id': 11, 'method': 'last4', 'value': ''}]},
     ['body', 'lot_confirmations', 0, 'value']),
    ('/ship/prepare', {'client_source': 'dashboard', 'customer_id': 1, 'product_id': 2, 'quantity_lb': 25,
                       'order_reference': ''}, ['body', 'order_reference']),
])
def test_prepare_422_carries_plain_field_messages(path, body, loc):
    response = TestClient(main.app, raise_server_exceptions=False).post(path, json=body, headers=headers())
    assert response.status_code == 422
    error = response.json()['detail'][0]
    # Pydantic fields stay for API callers; the plain copy is added.
    assert error['loc'] == loc and error['type'] == 'string_too_short'
    assert 'String should have' not in error['message'] and 'is required' in error['message']
    assert 'obligatorio' in error['message_es']
    # The envelope is a route contract (tests/test_notes_characterization.py): unchanged.
    envelope = response.json()['error_detail']
    assert envelope == {'code': 'VALIDATION_ERROR', 'message': 'String should have at least 1 character'}
