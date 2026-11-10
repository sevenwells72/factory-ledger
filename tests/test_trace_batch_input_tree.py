"""Backward trace input tree: /trace/batch input_tree reads posted current-ledger lines.

Finished lot -> bulk batch (pack) -> input lots (make) -> nested batches -> receipts,
keyed by numeric lot id. Placeholder receipts are never suppliers, and every made
node carries a display-only recorded-inputs check.
"""
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from psycopg2.extras import RealDictCursor

import main


@pytest.fixture
def client(_db_connection, db_cursor, monkeypatch):
    @contextmanager
    def transaction():
        with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
    monkeypatch.setattr(main, 'get_transaction', transaction)
    return TestClient(main.app, headers={'X-API-Key': main.DASHBOARD_API_KEY})


def item(cur, name, kind, code, entry_source, uom='lb', supplier_lot=None):
    cur.execute("INSERT INTO products (name, type, uom) VALUES (%s,%s,%s) RETURNING id", (name, kind, uom))
    pid = cur.fetchone()['id']
    cur.execute("""INSERT INTO lots (product_id, lot_code, entry_source, supplier_lot_code)
                   VALUES (%s,%s,%s,%s) RETURNING id""", (pid, code, entry_source, supplier_lot))
    return pid, cur.fetchone()['id']


def post(cur, kind, stamp, lines, **fields):
    columns = ['type', 'timestamp', 'status'] + list(fields)
    cur.execute(
        f"INSERT INTO transactions ({','.join(columns)}) VALUES ({','.join(['%s'] * len(columns))}) RETURNING id",
        [kind, stamp, 'posted'] + list(fields.values()))
    tid = cur.fetchone()['id']
    ids = []
    for (pid, lid), qty in lines:
        cur.execute("""INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                       VALUES (%s,%s,%s,%s) RETURNING id""", (tid, pid, lid, qty))
        ids.append(cur.fetchone()['id'])
    return tid, ids


def receive(cur, lot, qty, shipper, stamp='2026-09-01 14:00', bol='BOL-1'):
    return post(cur, 'receive', stamp, [(lot, qty)], shipper_name=shipper, bol_reference=bol)[0]


def trace(client, lot, code):
    response = client.get(f'/trace/batch/{code}', params={'product_id': lot[0]})
    assert response.status_code == 200, response.text
    return response.json()


def by_lot(nodes):
    return {node['lot_id']: node for node in nodes}


@pytest.mark.db
def test_packed_lot_traces_through_nested_makes_to_suppliers(client, db_cursor):
    oats = item(db_cursor, 'TT oats', 'ingredient', 'TT-OATS-001', 'received', '50 lb bag', 'SUP-OATS')
    nuts = item(db_cursor, 'TT nuts', 'ingredient', 'TT-NUTS-001', 'received')
    base = item(db_cursor, 'TT base batch', 'batch', 'TT-DAY-1', 'production_output')
    mix = item(db_cursor, 'TT fruit nut batch', 'batch', 'TT-DAY-2', 'production_output')
    case = item(db_cursor, 'TT fruit nut 25 LB', 'finished', 'TT-DAY-2', 'pack_output', '25 lb case')
    oats_receipt = receive(db_cursor, oats, 500, 'Quali Pack', bol='BOL-OATS')
    receive(db_cursor, nuts, 100, 'National Harvest')
    base_make, _ = post(db_cursor, 'make', '2026-09-02 14:00', [(base, 400), (oats, -400)])
    mix_make, _ = post(db_cursor, 'make', '2026-09-03 14:00', [(mix, 450), (base, -400), (nuts, -50)])
    pack, _ = post(db_cursor, 'pack', '2026-09-03 15:00', [(case, 450), (mix, -450)])
    post(db_cursor, 'ship', '2026-09-04 15:00', [(case, -450)], customer_name='TT Customer')
    # The pack also has a legacy consumption row: "ingredients" keeps reading it.
    db_cursor.execute("""INSERT INTO ingredient_lot_consumption
        (transaction_id, ingredient_product_id, ingredient_lot_id, quantity_lb) VALUES (%s,%s,%s,450)""",
        (pack, mix[0], mix[1]))

    data = trace(client, case, 'TT-DAY-2')
    assert [i['lot_id'] for i in data['ingredients']] == [mix[1]]          # unchanged field
    assert [s['customer_name'] for s in data['customer_shipments']] == ['TT Customer']

    tree = data['input_tree']
    assert tree['kind'] == 'made'
    assert [t['transaction_id'] for t in tree['transactions']] == [pack]
    packed = tree['transactions'][0]
    assert packed['type'] == 'pack' and packed['output_lb'] == 450 and packed['inputs_total_lb'] == 450
    assert packed['inputs_mismatch'] is False
    assert packed['produced_date'] == '2026-09-03' and packed['produced_at'].endswith('+00:00')

    [bulk] = packed['inputs']
    assert (bulk['lot_id'], bulk['quantity'], bulk['product_type']) == (mix[1], 450, 'batch')
    assert bulk['origin']['kind'] == 'made'
    [made] = bulk['origin']['transactions']
    assert made['transaction_id'] == mix_make
    inputs = by_lot(made['inputs'])
    assert inputs[nuts[1]]['quantity'] == 50
    assert inputs[nuts[1]]['origin']['kind'] == 'received'
    assert inputs[nuts[1]]['origin']['supplier_name'] == 'National Harvest'

    nested = inputs[base[1]]['origin']
    assert nested['kind'] == 'made'
    assert nested['transactions'][0]['transaction_id'] == base_make
    [oat_node] = nested['transactions'][0]['inputs']
    assert oat_node['quantity'] == 400 and oat_node['unit'] == 'lb'
    origin = oat_node['origin']
    assert origin['kind'] == 'received'
    assert (origin['transaction_id'], origin['supplier_name'], origin['bol_reference'],
            origin['supplier_lot_code']) == (oats_receipt, 'Quali Pack', 'BOL-OATS', 'SUP-OATS')


@pytest.mark.db
def test_tree_uses_current_posted_lines_only(client, db_cursor):
    oats = item(db_cursor, 'TT2 oats', 'ingredient', 'TT2-OATS-001', 'received')
    salt = item(db_cursor, 'TT2 salt', 'ingredient', 'TT2-SALT-001', 'received')
    batch = item(db_cursor, 'TT2 batch', 'batch', 'TT2-DAY', 'production_output')
    receive(db_cursor, oats, 500, 'Quali Pack')
    receive(db_cursor, salt, 50, 'Essex Food')
    kept, lines = post(db_cursor, 'make', '2026-09-02 14:00', [(batch, 100), (oats, -100)])
    voided, _ = post(db_cursor, 'make', '2026-09-02 15:00', [(batch, 10), (salt, -10)])
    main._append_transaction_line_correction(db_cursor, lines[1], {'quantity_lb': -95}, 'correct usage', 'test')
    main._append_transaction_correction(db_cursor, voided, 'void', 'duplicate', {}, 'test')

    tree = trace(client, batch, 'TT2-DAY')['input_tree']
    assert [t['transaction_id'] for t in tree['transactions']] == [kept]
    [node] = tree['transactions'][0]['inputs']
    assert (node['lot_id'], node['quantity']) == (oats[1], 95)
    assert tree['inputs_total_lb'] == 95 and tree['output_lb'] == 100


@pytest.mark.db
def test_several_makes_into_one_lot_and_shared_lot_codes(client, db_cursor):
    oats = item(db_cursor, 'TT3 oats', 'ingredient', 'TT3-SAME', 'received')
    other = item(db_cursor, 'TT3 other product', 'ingredient', 'TT3-SAME', 'received')
    batch = item(db_cursor, 'TT3 batch', 'batch', 'TT3-SAME', 'production_output')
    receive(db_cursor, oats, 500, 'Quali Pack')
    receive(db_cursor, other, 500, 'Somebody Else')
    first, _ = post(db_cursor, 'make', '2026-09-02 14:00', [(batch, 100), (oats, -100)])
    second, _ = post(db_cursor, 'make', '2026-09-02 16:00', [(batch, 100), (oats, -100)])

    tree = trace(client, batch, 'TT3-SAME')['input_tree']
    assert [t['transaction_id'] for t in tree['transactions']] == [first, second]
    assert tree['output_lb'] == 200 and tree['inputs_total_lb'] == 200
    for txn in tree['transactions']:
        [node] = txn['inputs']
        assert node['lot_id'] == oats[1]                     # never the same-code lot of another product
        assert node['origin']['supplier_name'] == 'Quali Pack'


@pytest.mark.db
def test_placeholder_receipts_are_not_suppliers(client, db_cursor):
    walnuts = item(db_cursor, 'TT4 walnuts', 'ingredient', 'TT4-WAL-001', 'received')
    raisins = item(db_cursor, 'TT4 raisins', 'ingredient', 'TT4-RAI-001', 'received')
    honey = item(db_cursor, 'TT4 honey', 'ingredient', 'TT4-HON-001', 'found_inventory')
    sugar = item(db_cursor, 'TT4 sugar', 'ingredient', 'TT4-SUG-001', 'received')
    named = item(db_cursor, 'TT4 almonds', 'ingredient', 'TT4-ALM-001', 'received')
    batch = item(db_cursor, 'TT4 batch', 'batch', 'TT4-DAY', 'production_output')
    found_receipt = receive(db_cursor, walnuts, 150, 'found', bol='12345')
    inventory_found = receive(db_cursor, raisins, 150, '  Inventory  FOUND ')
    unnamed_receipt = receive(db_cursor, sugar, 150, 'Unknown')
    # A named receipt wins over an earlier placeholder receipt into the same lot.
    receive(db_cursor, named, 10, 'Found', stamp='2026-08-01 14:00')
    named_receipt = receive(db_cursor, named, 10, 'Refrig-IT')
    post(db_cursor, 'adjust', '2026-09-01 14:00', [(honey, 100)])
    post(db_cursor, 'make', '2026-09-02 14:00', [
        (batch, 50), (walnuts, -10), (raisins, -10), (honey, -10), (sugar, -10), (named, -10)])

    [txn] = trace(client, batch, 'TT4-DAY')['input_tree']['transactions']
    origins = {lot_id: node['origin'] for lot_id, node in by_lot(txn['inputs']).items()}
    assert origins[walnuts[1]] == {
        'kind': 'found_inventory', 'transaction_id': found_receipt, 'recorded_shipper': 'found',
        'received_at': origins[walnuts[1]]['received_at'], 'received_date': '2026-09-01',
        'received_time': origins[walnuts[1]]['received_time']}
    assert origins[raisins[1]]['kind'] == 'found_inventory'
    assert origins[raisins[1]]['transaction_id'] == inventory_found
    assert origins[honey[1]] == {'kind': 'found_inventory', 'transaction_id': None}
    assert origins[sugar[1]]['kind'] == 'unknown'
    assert origins[sugar[1]]['reason'] == 'receipt_without_supplier'
    assert origins[sugar[1]]['transaction_id'] == unnamed_receipt
    assert origins[named[1]]['kind'] == 'received'
    assert origins[named[1]]['transaction_id'] == named_receipt
    assert origins[named[1]]['receipt_count'] == 2
    for origin in origins.values():
        assert 'supplier_name' not in origin or origin['kind'] == 'received'


@pytest.mark.db
def test_made_lot_also_lists_its_receipts(client, db_cursor):
    """A packed lot later topped up by receipts keeps its makes and lists every receipt."""
    bulk = item(db_cursor, 'TT10 bulk', 'ingredient', 'TT10-BULK-001', 'received')
    case = item(db_cursor, 'TT10 coconut 10 LB', 'finished', 'TT10-DAY', 'pack_output', '10 lb case')
    plain = item(db_cursor, 'TT10 plain 10 LB', 'finished', 'TT10-PLAIN', 'pack_output', '10 lb case')
    outer = item(db_cursor, 'TT10 mixed case', 'finished', 'TT10-OUTER', 'pack_output', '10 lb case')
    receive(db_cursor, bulk, 5000, 'Franklin Baker')
    pack, _ = post(db_cursor, 'pack', '2026-02-24 15:00', [(case, 1470), (bulk, -1470)])
    post(db_cursor, 'pack', '2026-02-24 16:00', [(plain, 100), (bulk, -100)])
    found = receive(db_cursor, case, 410, 'FOUND INVENTORY', stamp='2026-03-16 14:00')
    unnamed = receive(db_cursor, case, 20, 'Unknown', stamp='2026-03-17 14:00')
    named = receive(db_cursor, case, 30, 'Franklin Baker', stamp='2026-03-18 14:00', bol='BOL-TOPUP')
    voided = receive(db_cursor, case, 99, 'Found', stamp='2026-03-19 14:00')
    main._append_transaction_correction(db_cursor, voided, 'void', 'entered twice', {}, 'test')

    tree = trace(client, case, 'TT10-DAY')['input_tree']
    assert tree['kind'] == 'made'
    assert [t['transaction_id'] for t in tree['transactions']] == [pack]      # the make stays the origin
    assert tree['inputs_mismatch'] is False                                   # receipts never enter the check
    assert tree['output_lb'] == 1470
    extra = tree['also_received']
    assert [(r['kind'], r['transaction_id'], r['quantity_lb']) for r in extra] == [
        ('found_inventory', found, 410), ('unknown', unnamed, 20), ('received', named, 30)]
    assert extra[0]['recorded_shipper'] == 'FOUND INVENTORY' and extra[0]['received_date'] == '2026-03-16'
    assert 'supplier_name' not in extra[0] and 'supplier_name' not in extra[1]
    assert extra[1]['reason'] == 'receipt_without_supplier'
    assert (extra[2]['supplier_name'], extra[2]['bol_reference']) == ('Franklin Baker', 'BOL-TOPUP')

    # A made lot with no receipts lists none, and a nested made lot carries its own.
    assert trace(client, plain, 'TT10-PLAIN')['input_tree']['also_received'] == []
    post(db_cursor, 'pack', '2026-03-20 15:00', [(outer, 50), (case, -50)])
    [nested] = trace(client, outer, 'TT10-OUTER')['input_tree']['transactions'][0]['inputs']
    assert nested['origin']['kind'] == 'made'
    assert [r['transaction_id'] for r in nested['origin']['also_received']] == [found, unnamed, named]


@pytest.mark.db
def test_receipt_older_than_100_receipts_still_resolves(client, db_cursor):
    walnuts = item(db_cursor, 'TT5 walnuts', 'ingredient', 'TT5-WAL-001', 'received')
    filler = item(db_cursor, 'TT5 filler', 'ingredient', 'TT5-FIL-001', 'received')
    batch = item(db_cursor, 'TT5 batch', 'batch', 'TT5-DAY', 'production_output')
    old = receive(db_cursor, walnuts, 150, 'Old Supplier', stamp='2026-02-04 14:00')
    for n in range(105):
        receive(db_cursor, filler, 1, 'Filler Supplier', stamp=f'2026-10-01 14:{n % 60:02d}')
    post(db_cursor, 'make', '2026-10-02 14:00', [(batch, 10), (walnuts, -10)])
    [node] = trace(client, batch, 'TT5-DAY')['input_tree']['transactions'][0]['inputs']
    assert node['origin']['kind'] == 'received'
    assert node['origin']['transaction_id'] == old


@pytest.mark.db
@pytest.mark.parametrize('inputs, mismatch', [
    (50, True),        # the Oct 6 Fruit Nut shape: 50 lb recorded for 769.04 lb made
    (615.232, False),  # exactly 80%
    (615.231, True),
    (922.848, False),  # exactly 120%
    (922.849, True),
    (769.04, False),
])
def test_recorded_inputs_mismatch_flag(client, db_cursor, inputs, mismatch):
    oats = item(db_cursor, 'TT6 oats', 'ingredient', 'TT6-OATS-001', 'received')
    bags = item(db_cursor, 'TT6 bags', 'ingredient', 'TT6-BAG-001', 'received', 'each')
    batch = item(db_cursor, 'TT6 batch', 'batch', 'TT6-DAY', 'production_output')
    receive(db_cursor, oats, 2000, 'Quali Pack')
    receive(db_cursor, bags, 2000, 'Bag Co')
    made, _ = post(db_cursor, 'make', '2026-10-06 19:13', [(batch, 769.04), (oats, -inputs), (bags, -40)])
    tree = trace(client, batch, 'TT6-DAY')['input_tree']
    [txn] = tree['transactions']
    assert txn['transaction_id'] == made
    assert txn['output_lb'] == 769.04
    assert txn['inputs_total_lb'] == pytest.approx(inputs)     # count-unit bags are not pounds
    assert txn['inputs_mismatch'] is mismatch
    assert tree['inputs_mismatch'] is mismatch
    assert by_lot(txn['inputs'])[bags[1]]['unit'] == 'each'


@pytest.mark.db
def test_cycles_and_depth_are_bounded(client, db_cursor):
    a = item(db_cursor, 'TT7 a', 'batch', 'TT7-A', 'production_output')
    b = item(db_cursor, 'TT7 b', 'batch', 'TT7-B', 'production_output')
    post(db_cursor, 'make', '2026-09-02 14:00', [(a, 10), (b, -10)])
    post(db_cursor, 'make', '2026-09-03 14:00', [(b, 10), (a, -10)])
    tree = trace(client, a, 'TT7-A')['input_tree']
    [b_node] = tree['transactions'][0]['inputs']
    [a_again] = b_node['origin']['transactions'][0]['inputs']
    assert a_again['lot_id'] == a[1]
    assert a_again['origin']['truncated'] == 'cycle'

    chain = [item(db_cursor, f'TT8 step {n}', 'batch', f'TT8-STEP-{n}', 'production_output') for n in range(9)]
    for n in range(8):
        post(db_cursor, 'make', f'2026-09-{n + 10} 14:00', [(chain[n], 5), (chain[n + 1], -5)])
    node = trace(client, chain[0], 'TT8-STEP-0')['input_tree']
    depth = 0
    while not node.get('truncated'):
        node = node['transactions'][0]['inputs'][0]['origin']
        depth += 1
    assert depth == main.TRACE_TREE_MAX_DEPTH
    assert node['truncated'] == 'depth_limit'


@pytest.mark.db
def test_received_lot_has_no_input_tree(client, db_cursor):
    oats = item(db_cursor, 'TT9 oats', 'ingredient', 'TT9-OATS-001', 'received')
    receive(db_cursor, oats, 100, 'Quali Pack')
    response = client.get('/trace/batch/TT9-OATS-001', params={'product_id': oats[0]})
    assert response.status_code == 200
    assert response.json()['trace_type'] == 'ingredient'
    assert 'input_tree' not in response.json()


def test_shipper_kind_normalization():
    assert main._trace_shipper_kind('found') == 'found'
    assert main._trace_shipper_kind(' Found  Inventory ') == 'found'
    assert main._trace_shipper_kind('Physical count') == 'found'
    for placeholder in ('INITIAL INVENTORY', 'Inventory Correction', 'Inventory Intake', 'Blender Warehouse'):
        assert main._trace_shipper_kind(placeholder) == 'found'
    assert main._trace_shipper_kind(None) == 'unnamed'
    assert main._trace_shipper_kind('Unknown') == 'unnamed'
    assert main._trace_shipper_kind('Founders Foods') == 'supplier'
    assert main._trace_shipper_kind('Jack’s Eggs') == 'supplier'
