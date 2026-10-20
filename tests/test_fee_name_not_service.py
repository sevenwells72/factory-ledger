"""A product name that merely contains "fee" (e.g. "Coffee") is a normal item.

Order detail and the packing slip used to treat any line whose product name
contained pallet/freight/delivery/surcharge/charge/fee as a service line. Real
CNS products such as "Batch Setton Coffee Granola #12" — and random test tags
like "...3A3FEE..." — were then dropped from weight totals and shown without a
lot. Only products.is_service decides a service line now, and a charge line reads
differently on the slip (CHARGE tag, no lot, a unit count).
"""
from uuid import uuid4

import pytest
import reportlab.platypus

from tests.test_sales_order_line_fields import client  # noqa: F401

pytestmark = pytest.mark.db

NAMES = ['FEE-TAG Granola 25 LB', 'Setton Coffee Granola 25 LB', 'Delivery Day Granola 25 LB']


def _order(cur, product_name, legacy):
    """One bulk line (named like a charge but not a service) and one real pallet charge."""
    tag = uuid4().hex[:8]
    cur.execute("INSERT INTO customers(name,active) VALUES (%s,true) RETURNING id", ('Fee name ' + tag,))
    customer = cur.fetchone()[0]
    cur.execute("""INSERT INTO products(name,odoo_code,type,uom,case_size_lb,active)
                   VALUES (%s,%s,'finished','lb',25,true) RETURNING id""", (f'{product_name} {tag}', 'FEE-' + tag))
    bulk = cur.fetchone()[0]
    cur.execute("""INSERT INTO products(name,odoo_code,type,uom,is_service,active)
                   VALUES (%s,%s,'finished','each',true,true) RETURNING id""", ('Pallet Charge ' + tag, 'PAL-' + tag))
    pallet = cur.fetchone()[0]
    cur.execute("INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id", (bulk, 'FEE-LOT-' + tag.upper()))
    lot = cur.fetchone()[0]
    cur.execute("INSERT INTO transactions(type,notes) VALUES ('receive','fee name fixture') RETURNING id")
    cur.execute("INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,500)",
                (cur.fetchone()[0], bulk, lot))
    cur.execute("""INSERT INTO sales_orders(order_number,customer_id,state,status)
                   VALUES (%s,%s,'open','confirmed') RETURNING id""", ('SO-FEE-' + tag, customer))
    order = cur.fetchone()[0]
    if legacy:  # pre-contract rows carry pounds only (ordered_quantity NULL)
        cur.execute("INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb) VALUES (%s,%s,100) RETURNING id",
                    (order, bulk))
        line = cur.fetchone()[0]
        cur.execute("INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb) VALUES (%s,%s,2)", (order, pallet))
    else:
        cur.execute("""INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb,ordered_quantity,ordered_unit,
                       ordered_case_weight_lb) VALUES (%s,%s,100,4,'cases',25) RETURNING id""", (order, bulk))
        line = cur.fetchone()[0]
        cur.execute("""INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb,ordered_quantity,ordered_unit)
                       VALUES (%s,%s,0,2,'each')""", (order, pallet))
    return {'order': order, 'line': line, 'bulk': bulk, 'lot': lot, 'customer': customer, 'tag': tag}


def _slip_rows(client, order, monkeypatch):  # noqa: F811
    tables = []
    real_table = reportlab.platypus.Table

    def capture(data, *args, **kwargs):
        tables.append(data)
        return real_table(data, *args, **kwargs)

    monkeypatch.setattr(reportlab.platypus, 'Table', capture)
    response = client.get(f'/sales/orders/{order}/packing-slip')
    assert response.status_code == 200, response.text
    assert response.headers['content-type'] == 'application/pdf'
    items = next(t for t in tables if t and getattr(t[0][0], 'text', None) == 'DATE')

    def text(cell):
        return ' '.join(text(c) for c in cell) if isinstance(cell, list) else cell.text
    return [[text(cell) for cell in row[1:]] for row in items[1:]]


def assert_charge_row(rows, tag):
    """A charge line: CHARGE tag on the name, no lot, a unit count — never pounds."""
    charge = [row for row in rows if row[0].startswith('Pallet Charge ' + tag)]
    assert len(charge) == 1 and 'CHARGE' in charge[0][0]
    assert charge[0][1:] == ['\u2014', '2 each'] and 'lb' not in charge[0][2]


@pytest.mark.parametrize('legacy', [True, False], ids=['pounds-only-line', 'contract-line'])
@pytest.mark.parametrize('product_name', NAMES)
def test_order_detail_counts_fee_named_product_as_bulk(client, _db_connection, product_name, legacy):  # noqa: F811
    with _db_connection.cursor() as cur:
        seeded = _order(cur, product_name, legacy)
    detail = client.get(f"/sales/orders/{seeded['order']}").json()
    bulk, pallet = detail['lines']
    assert bulk['product'].startswith(product_name)
    assert not bulk.get('is_non_weight')
    assert bulk['price_basis'] == 'per_case' and bulk['unit_count'] == 4
    assert pallet['is_non_weight'] is True and pallet['price_basis'] == 'per_unit'
    assert bulk['item_kind'] == 'product' and pallet['item_kind'] == 'charge'
    assert detail['totals']['total_ordered_lb'] == 100
    assert detail['totals']['total_ordered_units'] == 4


@pytest.mark.parametrize('legacy', [True, False], ids=['pounds-only-line', 'contract-line'])
@pytest.mark.parametrize('product_name', NAMES)
def test_packing_slip_gives_fee_named_product_a_lot_before_shipping(client, _db_connection, monkeypatch, product_name, legacy):  # noqa: F811
    with _db_connection.cursor() as cur:
        seeded = _order(cur, product_name, legacy)
    rows = _slip_rows(client, seeded['order'], monkeypatch)
    bulk = [row for row in rows if row[0].startswith(product_name)]
    assert bulk == [[f"{product_name} {seeded['tag']}", 'FEE-LOT-' + seeded['tag'].upper(), '100 lb · 4 units']]
    assert_charge_row(rows, seeded['tag'])


@pytest.mark.parametrize('product_name', NAMES)
def test_packing_slip_shows_shipped_fee_named_product_in_pounds(client, _db_connection, monkeypatch, product_name):  # noqa: F811
    with _db_connection.cursor() as cur:
        seeded = _order(cur, product_name, legacy=True)
        cur.execute("INSERT INTO transactions(type,notes) VALUES ('ship','fee name shipment') RETURNING id")
        txn = cur.fetchone()[0]
        cur.execute("INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,-100)",
                    (txn, seeded['bulk'], seeded['lot']))
        cur.execute("INSERT INTO shipments(sales_order_id,customer_id,shipped_at) VALUES (%s,%s,now()) RETURNING id",
                    (seeded['order'], seeded['customer']))
        cur.execute("""INSERT INTO shipment_lines(shipment_id,transaction_id,sales_order_line_id,product_id,quantity_lb)
                       VALUES (%s,%s,%s,%s,100)""", (cur.fetchone()[0], txn, seeded['line'], seeded['bulk']))
    rows = _slip_rows(client, seeded['order'], monkeypatch)
    assert [f"{product_name} {seeded['tag']}", 'FEE-LOT-' + seeded['tag'].upper(), '100 lb · 4 units'] in rows
    assert_charge_row(rows, seeded['tag'])
