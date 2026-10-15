"""Spanish duplicate warnings retain the same receipts, actors and evidence."""
import pytest

import order_tickets
import ship_tickets
import ticket_actions
import write_tickets


class WarningCursor:
    def __init__(self, rows):
        self.rows = list(rows)

    def execute(self, sql, params):
        pass

    def fetchone(self):
        return self.rows.pop(0)

    def fetchall(self):
        rows, self.rows = self.rows, []
        return rows


def assert_spanish_time(warning, minutes):
    assert warning['code'] == 'POSSIBLE_DUPLICATE'
    assert warning['requires_ack'] is True
    assert f"hace {minutes} {'minuto' if minutes == 1 else 'minutos'} como" in warning['message_es']
    assert 'hace 1 minutos' not in warning['message_es']


@pytest.mark.parametrize('age,minutes', [(1.9, 1), (2.9, 2)])
def test_receive_duplicate_warning_spanish_time_preserves_refs(age, minutes):
    row = {'id': 72, 'receipt_number': 'RCV-261015-001', 'operator_id': 'Arturo', 'minutes_ago': age}
    payload = {'product_id': 283, 'cases': 1, 'case_size_lb': 25, 'supplier_lot_code': 'SUP-1234'}
    [warning] = write_tickets.receive_duplicates(WarningCursor([row]), payload)
    assert_spanish_time(warning, minutes)
    assert warning['message'] == f"A matching receive was posted {minutes} minutes ago as RCV-261015-001 by Arturo."
    assert warning['refs'] == {'transaction_id': 72, 'receipt_number': 'RCV-261015-001', 'operator_id': 'Arturo', 'minutes_ago': minutes}


@pytest.mark.parametrize('age,minutes', [(1.9, 1), (2.9, 2)])
@pytest.mark.parametrize('action,key', [('create_order', 'customer_id'), ('add_order_lines', 'order_id')])
def test_order_duplicate_warning_spanish_time_preserves_refs(age, minutes, action, key):
    lines = [{'product_id': 100, 'quantity': 2, 'unit': 'cases'}]
    row = {'id': 'ticket-72', 'receipt_number': 'ORD-261015-001', 'operator_id': 'Office',
           'minutes_ago': age, 'lines': lines, 'result_ref': {'order_id': 90, 'order_number': 'SO-90'}}
    [warning] = order_tickets.possible_duplicate(WarningCursor([row]), action, {key: 90, 'lines': lines}, key)
    assert_spanish_time(warning, minutes)
    assert warning['message'] == f"The same lines were recorded {minutes} minutes ago as ORD-261015-001 (order SO-90) by Office."
    assert warning['refs'] == {'receipt_number': 'ORD-261015-001', 'ticket_id': 'ticket-72', 'order_id': 90,
                               'order_number': 'SO-90', 'operator_id': 'Office', 'minutes_ago': minutes}


@pytest.mark.parametrize('age,minutes', [(1.9, 1), (2.9, 2)])
@pytest.mark.parametrize('action', ['ship_order', 'ship_standalone', 'ship_bulk'])
def test_shipment_duplicate_warning_spanish_time_preserves_refs(age, minutes, action):
    row = {'id': 72, 'receipt_number': 'SHP-261015-001', 'operator_id': 'Arturo', 'minutes_ago': age}
    [warning] = ship_tickets.possible_duplicate(WarningCursor([row]), action, {'lines': [{'line_id': 5}]}, 90, 50)
    assert_spanish_time(warning, minutes)
    assert warning['message'] == f"A matching shipment was recorded {minutes} minutes ago as SHP-261015-001 by Arturo."
    assert warning['refs'] == row


@pytest.mark.parametrize('age,minutes', [(1.9, 1), (2.9, 2)])
@pytest.mark.parametrize('codes', [[], ['TAG-1234', 'TAG-5678']])
@pytest.mark.parametrize('action,payload,draft', [
    ('make', {'product_id': 285}, {'total_output_lb': 50}),
    ('pack', {'target_product_id': 286, 'cases': 2}, {'total_lb': 50}),
    ('adjust', {'product_id': 283, 'lot_id': 5, 'delta_lb': -50}, {}),
])
def test_inventory_duplicate_warning_translates_lot_label_and_time(age, minutes, codes, action, payload, draft):
    row = {'id': 72, 'receipt_number': 'MK-261015-001', 'operator_id': 'Arturo', 'minutes_ago': age}
    [warning] = ticket_actions.duplicates(WarningCursor([row, {'name': 'Classic', 'lot_codes': codes}]), action, payload, draft)
    assert_spanish_time(warning, minutes)
    label = 'Classic' + (' (lots TAG-1234, TAG-5678)' if codes else '')
    assert warning['message'] == f"A matching {action} for {label} was posted {minutes} minutes ago as MK-261015-001 by Arturo."
    spanish_label = 'Classic' + (' (lotes TAG-1234, TAG-5678)' if codes else '')
    assert f'para {spanish_label} se registró' in warning['message_es']
    assert '(lots ' not in warning['message_es']
    assert warning['refs'] == {'transaction_id': 72, 'receipt_number': 'MK-261015-001', 'operator_id': 'Arturo', 'minutes_ago': minutes}
