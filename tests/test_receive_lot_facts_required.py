"""The unidentified-receive exception requires the worker's label statement."""
import pytest

from assistant_receive import merge_receive_lot_facts, normalize_receive_arguments


@pytest.mark.parametrize('text', [
    'No lot on label.',
    'There is no lot on the label.',
    'No supplier lot number on the label.',
    'The label has no lot code.',
    'Sin lote en la etiqueta.',
    'No hay lote en la etiqueta.',
    'Sin número de lote del proveedor en la etiqueta.',
    'Received 10 cases of oats, no lot on label.',
    'Recibimos 10 cajones de avena, sin lote en la etiqueta.',
])
def test_explicit_label_absence_is_bound_and_survives_button_turns(text):
    previous = {'supplier_lot_code': 'OLD-001', 'quantity_lb': 50}
    facts = merge_receive_lot_facts(previous, text)
    assert facts == {'supplier_lot_code': None, 'quantity_lb': 50, 'no_supplier_lot': True}
    assert merge_receive_lot_facts(facts, '') == facts
    assert normalize_receive_arguments({'supplier_lot_code': 'OLD-001', 'supplier_id': 12}, facts) == {
        'supplier_id': 12, 'no_supplier_lot': True}
    assert previous == {'supplier_lot_code': 'OLD-001', 'quantity_lb': 50}


@pytest.mark.parametrize('text', [
    '',
    'The supplier lot is unknown.',
    'No supplier lot number given.',
    'Supplier lot N/A.',
    'The label is unreadable.',
    'El lote del proveedor es desconocido.',
    'Sin lote del proveedor.',
    'La etiqueta no se puede leer.',
    'Do not select No lot on label.',
    "Don't use No lot on label.",
    'There is not no lot on label.',
    'Is there no lot on label?',
    'Is there no lot on label',
    'No lot on label?',
    'What does No lot on label mean?',
    'If there is no lot on label, what should I do?',
    'The option No lot on label is available.',
    'No lot on label is not true.',
    'No lot on label means the exception flag.',
    'No seleccione Sin lote en la etiqueta.',
    '¿No hay lote en la etiqueta?',
    'Si no hay lote en la etiqueta, consulte en la oficina.',
    'La opción Sin lote en la etiqueta está disponible.',
    'Sin lote en la etiqueta es falso.',
    'I am asking about No lot on label.',
    'If missing, select No lot on label.',
    'If missing, there is no lot on label.',
    'If the lot is missing, received 10 cases, no lot on label.',
    'Received 10 cases, if missing, select No lot on label.',
    'Estoy preguntando sobre Sin lote en la etiqueta.',
    'Si falta, seleccione Sin lote en la etiqueta.',
])
def test_missing_or_uncertain_lot_never_grants_the_label_exception(text):
    facts = merge_receive_lot_facts({}, text)
    assert facts.get('no_supplier_lot') is not True
    arguments = {'supplier_id': 12, 'no_supplier_lot': True}
    assert normalize_receive_arguments(arguments, facts) == {'supplier_id': 12}
    assert arguments == {'supplier_id': 12, 'no_supplier_lot': True}


@pytest.mark.parametrize('text', ['Supplier lot NEW-001.', 'Lote del proveedor NEW-001.'])
def test_actual_supplier_code_clears_a_previous_no_label_exception(text):
    previous = {'supplier_lot_code': None, 'no_supplier_lot': True, 'quantity_lb': 50}
    facts = merge_receive_lot_facts(previous, text)
    assert facts == {'supplier_lot_code': 'NEW-001', 'quantity_lb': 50}
    assert normalize_receive_arguments({'no_supplier_lot': True, 'lot_code': 'MODEL-WRONG'}, facts) == {
        'supplier_lot_code': 'NEW-001'}
    assert previous['no_supplier_lot'] is True


def test_supplied_supplier_code_takes_precedence_over_a_label_absence_phrase():
    facts = merge_receive_lot_facts({}, 'No lot on label. Supplier lot SUP-001.')
    assert normalize_receive_arguments({'no_supplier_lot': True}, facts) == {'supplier_lot_code': 'SUP-001'}


@pytest.mark.parametrize('text', [
    'Actually, No lot on label is not true.',
    'No lot on label is false.',
    "No lot on label isn't correct.",
    'En realidad, Sin lote en la etiqueta no es cierto.',
    'Sin lote en la etiqueta es falso.',
    'No lot on label. No lot on label is not true.',
])
def test_direct_denial_revokes_previous_missing_label_exception(text):
    previous = {'supplier_lot_code': None, 'no_supplier_lot': True, 'quantity_lb': 50}
    facts = merge_receive_lot_facts(previous, text)
    assert facts == {'supplier_lot_code': None, 'quantity_lb': 50}
    assert normalize_receive_arguments({'no_supplier_lot': True, 'supplier_lot_code': 'MODEL-OLD'}, facts) == {}
    assert merge_receive_lot_facts(facts, '') == facts
    assert previous['no_supplier_lot'] is True
