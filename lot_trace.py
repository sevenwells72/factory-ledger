"""Read-only lot history and forward exposure, using posted current ledger lines.

Lot IDs are identities. Transactions are edges: identical repeated makes must
remain separate. Forward exposure is conservative (all stock sharing a lot),
not an assertion of physically verified ingredient allocation.
"""
from collections import deque



def supplier_entries(cur, lot_id):
    """Recorded lot-level source metadata, never an allocation to a receipt."""
    cur.execute("""SELECT e.supplier_lot_code, COALESCE(s.name, e.supplier_name) AS supplier_name,
                          e.quantity_lb, e.notes
                   FROM lot_supplier_codes e LEFT JOIN suppliers s ON s.id=e.supplier_id
                   WHERE e.lot_id=%s ORDER BY e.id""", (lot_id,))
    return [dict(r, quantity_lb=float(r['quantity_lb']) if r['quantity_lb'] is not None else None)
            for r in cur.fetchall()]


def consumption(cur, lot_id):
    """One row per consuming transaction/output lot, including orphan consumes."""
    cur.execute("""
        WITH consumed AS (
            SELECT t.id, t.type, t.timestamp, -SUM(i.quantity_lb) AS quantity_used
            FROM ledger_current_transactions t
            JOIN ledger_current_transaction_lines i ON i.transaction_id = t.id
            WHERE t.effective_status = 'posted' AND t.type IN ('make', 'pack')
              AND i.lot_id = %s AND i.quantity_lb < 0
            GROUP BY t.id, t.type, t.timestamp
        ), outputs AS (
            SELECT o.transaction_id, o.lot_id, o.product_id, SUM(o.quantity_lb) AS output_quantity
            FROM ledger_current_transaction_lines o
            JOIN consumed c ON c.id = o.transaction_id
            WHERE o.quantity_lb > 0
            GROUP BY o.transaction_id, o.lot_id, o.product_id
        )
        SELECT c.id AS transaction_id, c.type, c.timestamp, c.quantity_used,
               o.lot_id, o.product_id, o.output_quantity,
               l.lot_code AS batch_lot_code, p.name AS batch_product
        FROM consumed c
        LEFT JOIN outputs o ON o.transaction_id = c.id
        LEFT JOIN lots l ON l.id = o.lot_id
        LEFT JOIN products p ON p.id = o.product_id
        ORDER BY c.timestamp, c.id, o.lot_id
    """, (lot_id,))
    return cur.fetchall()


def shipments(cur, lot_id):
    # Aggregate ledger quantities BEFORE attaching order/shipment references.
    # One transaction can have several reference rows; joining them directly
    # multiplies shipped pounds. References carry no additional quantity claim.
    cur.execute("""
        SELECT t.id AS transaction_id, t.timestamp AS shipped_at,
               t.customer_name, t.order_reference, i.product_id,
               -SUM(i.quantity_lb) AS quantity_lb
        FROM ledger_current_transactions t
        JOIN ledger_current_transaction_lines i ON i.transaction_id = t.id
        WHERE t.effective_status = 'posted' AND t.type = 'ship'
          AND i.lot_id = %s AND i.quantity_lb < 0
        GROUP BY t.id, t.timestamp, t.customer_name, t.order_reference, i.product_id
        ORDER BY t.timestamp, t.id
    """, (lot_id,))
    rows = cur.fetchall()
    for row in rows:
        cur.execute("""
            SELECT DISTINCT so.id AS sales_order_id, so.order_number
            FROM sales_order_shipments sos
            JOIN sales_order_lines sol ON sol.id = sos.sales_order_line_id
            JOIN sales_orders so ON so.id = sol.sales_order_id
            WHERE sos.transaction_id = %s AND sol.product_id = %s
            UNION
            SELECT DISTINCT so.id, so.order_number
            FROM shipment_lines sl
            JOIN sales_order_lines sol ON sol.id = sl.sales_order_line_id
            JOIN sales_orders so ON so.id = sol.sales_order_id
            WHERE sl.transaction_id = %s AND sl.product_id = %s
            ORDER BY sales_order_id
        """, (row['transaction_id'], row['product_id'], row['transaction_id'], row['product_id']))
        row['sales_orders'] = [dict(r) for r in cur.fetchall()]
        cur.execute("""SELECT DISTINCT shipment_id FROM shipment_lines
                       WHERE transaction_id = %s AND product_id = %s ORDER BY shipment_id""",
                    (row['transaction_id'], row['product_id']))
        row['shipment_ids'] = [r['shipment_id'] for r in cur.fetchall()]
        row['sales_order_id'] = row['sales_orders'][0]['sales_order_id'] if row['sales_orders'] else None
        row['order_number'] = row['sales_orders'][0]['order_number'] if row['sales_orders'] else None
        row['shipment_id'] = row['shipment_ids'][0] if row['shipment_ids'] else None
        row['quantity_lb'] = float(row['quantity_lb'])
    return rows


def forward(cur, lot_id, api):
    """Traverse every reachable lot once, keeping every edge and every shipment.

    No date window, fan-out limit, or recursion depth cutoff. The visited set
    makes cycles finite without deleting their recorded edges.
    """
    pending, visited, nodes, edges, gaps = deque([lot_id]), set(), [], [], []
    while pending:
        current = pending.popleft()
        if current in visited:
            continue
        visited.add(current)
        cur.execute("""SELECT l.id AS lot_id, l.lot_code, l.product_id, l.entry_source, l.entry_source_notes, l.lot_type,
                              l.supplier_lot_code, p.name AS product_name, p.type AS product_type,
                              p.uom, p.case_size_lb
                       FROM lots l JOIN products p ON p.id = l.product_id WHERE l.id = %s""", (current,))
        node = dict(cur.fetchone())
        node['unit'] = 'lb' if node['product_type'] in ('batch', 'finished') else api.ledger_quantity_unit(node['uom'])
        node.pop('uom')
        node['on_hand'] = api.lot_on_hand(cur, current)
        node['supplier_lot_entries'] = supplier_entries(cur, current)
        node['case_size_lb'] = float(node['case_size_lb']) if node['case_size_lb'] else None
        node['receipts'] = [dict(api._trace_receipt_origin(r, node), quantity_lb=float(r['quantity']))
                            for r in api._trace_lot_receipts(cur, current)]
        node['shipments'] = [dict(r, **api._trace_stamp('shipped', r['shipped_at']))
                             for r in shipments(cur, current)]
        nodes.append(node)
        for row in consumption(cur, current):
            edge = dict(row)
            edge.update(source_lot_id=current, quantity_used=float(row['quantity_used']),
                        output_quantity=float(row['output_quantity'] or 0),
                        unit=node['unit'], **api._trace_stamp('produced', row['timestamp']))
            edge.pop('timestamp')
            edges.append(edge)
            if row['lot_id'] is None:
                gaps.append({'transaction_id': row['transaction_id'], 'lot_id': current,
                             'reason': 'Consumed lot has no identified output lot'})
            else:
                pending.append(row['lot_id'])
    return {'start_lot_id': lot_id, 'lots': nodes, 'consumption': edges, 'gaps': gaps,
            'quantity_basis': 'Recorded consumption per transaction; downstream lots are potential exposure, not allocated ingredient weight.'}


def history(cur, product_id, api):
    cur.execute('SELECT id, name, type, uom, case_size_lb FROM products WHERE id = %s', (product_id,))
    product = cur.fetchone()
    if not product:
        return None
    kinds = ['pack', 'receive'] if product['type'] == 'finished' else (
        ['make', 'receive'] if product['type'] == 'batch' else ['receive'])
    cur.execute("""
        SELECT l.id AS lot_id, l.lot_code, l.entry_source, l.entry_source_notes, l.lot_type, l.supplier_lot_code,
               MAX(t.timestamp) AS latest_at
        FROM lots l
        JOIN ledger_current_transaction_lines i ON i.lot_id = l.id
        JOIN ledger_current_transactions t ON t.id = i.transaction_id
        WHERE l.product_id = %s AND t.effective_status = 'posted'
          AND (t.type = ANY(%s) OR (t.type = 'adjust' AND l.entry_source IN ('found_inventory','adjusted'))) AND i.quantity_lb > 0
        GROUP BY l.id ORDER BY latest_at DESC, l.id DESC
    """, (product_id, kinds))
    lots = []
    for row in cur.fetchall():
        lot = dict(row)
        lot.update(api._trace_stamp('latest', lot.pop('latest_at')))
        lot['remaining'] = api.lot_on_hand(cur, lot['lot_id'])
        lot['supplier_lot_entries'] = supplier_entries(cur, lot['lot_id'])
        lot['receipts'] = [dict(api._trace_receipt_origin(r, lot), quantity_lb=float(r['quantity']),
                                bol_reference=r['bol_reference'], receipt_number=r['receipt_number'])
                           for r in api._trace_lot_receipts(cur, lot['lot_id'])]
        if lot['entry_source'] in ('found_inventory', 'adjusted'):
            cur.execute("""SELECT t.id, t.timestamp, SUM(i.quantity_lb) AS quantity
                           FROM ledger_current_transactions t
                           JOIN ledger_current_transaction_lines i ON i.transaction_id = t.id
                           WHERE t.effective_status = 'posted' AND t.type = 'adjust'
                             AND i.lot_id = %s AND i.quantity_lb > 0
                           GROUP BY t.id, t.timestamp ORDER BY t.timestamp DESC, t.id DESC""", (lot['lot_id'],))
            lot['receipts'].extend(dict(kind='found_inventory', transaction_id=r['id'],
                quantity_lb=float(r['quantity']), supplier_lot_code=lot['supplier_lot_code'],
                **api._trace_stamp('received', r['timestamp'])) for r in cur.fetchall())
        lot['received_quantity'] = sum(r['quantity_lb'] for r in lot['receipts'])
        lot['found_inventory'] = lot['entry_source'] in ('found_inventory', 'adjusted') or any(
            r['kind'] == 'found_inventory' for r in lot['receipts'])
        cur.execute("""
            SELECT t.id AS transaction_id, t.type, t.timestamp, SUM(i.quantity_lb) AS quantity,
                   COALESCE(t.case_size_lb, %s) AS case_size_lb
            FROM ledger_current_transactions t
            JOIN ledger_current_transaction_lines i ON i.transaction_id = t.id
            WHERE t.effective_status = 'posted' AND t.type IN ('make', 'pack')
              AND i.lot_id = %s AND i.quantity_lb > 0
            GROUP BY t.id, t.type, t.timestamp, t.case_size_lb ORDER BY t.timestamp DESC, t.id DESC
        """, (product['case_size_lb'], lot['lot_id']))
        lot['production'] = [dict(transaction_id=r['transaction_id'], type=r['type'],
                                  quantity=float(r['quantity']),
                                  cases=float(r['quantity'] / r['case_size_lb']) if r['case_size_lb'] else None,
                                  **api._trace_stamp('produced', r['timestamp'])) for r in cur.fetchall()]
        lot['packed_quantity'] = sum(r['quantity'] for r in lot['production'] if r['type'] == 'pack')
        lot['packed_cases'] = (sum(r['cases'] for r in lot['production'] if r['type'] == 'pack')
                              if all(r['cases'] is not None for r in lot['production'] if r['type'] == 'pack') else None)
        lots.append(lot)
    return {'product_id': product_id, 'product_name': product['name'], 'product_type': product['type'],
            'unit': 'lb' if product['type'] in ('batch', 'finished') else api.ledger_quantity_unit(product['uom']), 'lots': lots}
