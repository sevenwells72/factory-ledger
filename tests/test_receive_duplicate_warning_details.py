"""POSSIBLE_DUPLICATE on receive shows both deliveries' BOL and supplier lot (2026-11-25 oats trucks)."""
import write_tickets


class RecordingCursor:
    def __init__(self, row):
        self.row, self.calls = row, []

    def execute(self, sql, params):
        self.calls.append((sql, params))

    def fetchone(self):
        return self.row


OATS = {'product_id': 32, 'cases': 875, 'case_size_lb': 50.0,
        'supplier_lot_code': '261115', 'bol_reference': 'CV264307'}
EARLIER = {'id': 3155, 'receipt_number': 'RCV-261125-002', 'operator_id': 'Arturo',
           'bol_reference': 'CV270129', 'supplier_lot_codes': ['261116'], 'minutes_ago': 29.4}


def test_two_real_trucks_show_both_bols_and_supplier_lots():
    [warning] = write_tickets.receive_duplicates(RecordingCursor(EARLIER), OATS)
    assert warning['code'] == 'POSSIBLE_DUPLICATE' and warning['requires_ack'] is True
    assert warning['message'] == (
        'A matching receive was posted 29 minutes ago as RCV-261125-002 by Arturo. '
        'Earlier receipt: BOL CV270129, supplier lot 261116. '
        'This receipt: BOL CV264307, supplier lot 261115. '
        'Tick the box only if this is a separate truck or delivery.')
    assert warning['message_es'] == (
        'Una recepción similar se registró hace 29 minutos como RCV-261125-002 por Arturo. '
        'Recepción anterior: BOL CV270129, lote del proveedor 261116. '
        'Esta recepción: BOL CV264307, lote del proveedor 261115. '
        'Marca la casilla solo si es otro camión u otra entrega.')
    assert warning['refs']['previous_bol_reference'] == 'CV270129'
    assert warning['refs']['previous_supplier_lot_codes'] == ['261116']
    assert warning['refs']['same_bol'] is False


def test_same_bol_says_it_is_probably_entered_twice():
    payload = dict(OATS, bol_reference=' cv270129 ', supplier_lot_code='261116')
    [warning] = write_tickets.receive_duplicates(RecordingCursor(EARLIER), payload)
    assert warning['refs']['same_bol'] is True
    assert warning['message'].endswith('Same BOL number: this is most likely the same delivery entered twice.')
    assert warning['message_es'].endswith('Mismo número de BOL: lo más probable es que sea la misma entrega registrada dos veces.')
    assert warning['requires_ack'] is True, 'still a tick, never an automatic block or pass'


def test_missing_bol_or_lot_reads_plainly():
    earlier = dict(EARLIER, bol_reference=None, supplier_lot_codes=None)
    payload = dict(OATS, supplier_lot_code='N/A', bol_reference='')
    [warning] = write_tickets.receive_duplicates(RecordingCursor(earlier), payload)
    assert 'Earlier receipt: BOL —, supplier lot no lot on label. This receipt: BOL —, supplier lot no lot on label.' in warning['message']
    assert 'lote del proveedor sin lote en la etiqueta' in warning['message_es']
    assert warning['refs']['same_bol'] is False, 'two blank BOLs are not a match'


def test_query_placeholders_match_parameters():
    cur = RecordingCursor(None)
    assert write_tickets.receive_duplicates(cur, OATS) == []
    sql, params = cur.calls[0]
    assert sql.count('%s') == len(params)
    assert 'raw.bol_reference' in sql and 'supplier_lot_codes' in sql
