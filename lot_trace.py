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


def forward(cur, lot_id, api):
    """Complete closure in exactly five set-based queries, independent of lot count.

    Materialize current posted make/pack lines once. UNION (not UNION ALL)
    terminates cycles without a depth/fanout cap. The other four queries fetch
    metadata/balances, receipts, supplier entries and shipments for the closure.
    """
    cur.execute("""
        WITH RECURSIVE posted AS MATERIALIZED (
            SELECT t.id, t.type, t.timestamp, i.lot_id, i.product_id, i.quantity_lb
            FROM ledger_current_transactions t
            JOIN ledger_current_transaction_lines i ON i.transaction_id=t.id
            WHERE t.effective_status='posted' AND t.type IN ('make','pack')
        ), consumed AS MATERIALIZED (
            SELECT id, type, timestamp, lot_id, -SUM(quantity_lb) AS quantity_used
            FROM posted WHERE quantity_lb<0 GROUP BY id,type,timestamp,lot_id
        ), produced AS MATERIALIZED (
            SELECT id, lot_id, product_id, SUM(quantity_lb) AS output_quantity
            FROM posted WHERE quantity_lb>0 GROUP BY id,lot_id,product_id
        ), edges AS MATERIALIZED (
            SELECT c.id AS transaction_id,c.type,c.timestamp,c.lot_id AS source_lot_id,
                   c.quantity_used,o.lot_id,o.product_id,o.output_quantity
            FROM consumed c LEFT JOIN produced o ON o.id=c.id
        ), reachable(lot_id) AS (
            SELECT %s::integer
            UNION
            SELECT e.lot_id FROM reachable r JOIN edges e ON e.source_lot_id=r.lot_id
            WHERE e.lot_id IS NOT NULL
        )
        SELECT e.*,l.lot_code AS batch_lot_code,p.name AS batch_product
        FROM edges e JOIN reachable r ON r.lot_id=e.source_lot_id
        LEFT JOIN lots l ON l.id=e.lot_id LEFT JOIN products p ON p.id=e.product_id
        ORDER BY e.timestamp,e.transaction_id,e.source_lot_id,e.lot_id
    """, (lot_id,))
    edges=[dict(r) for r in cur.fetchall()]
    adjacency={}
    for e in edges:
        adjacency.setdefault(e['source_lot_id'],[]).append(e)
    pending, visited, ordered = deque([lot_id]), set(), []
    while pending:
        current=pending.popleft()
        if current in visited:
            continue
        visited.add(current); ordered.append(current)
        pending.extend(e['lot_id'] for e in adjacency.get(current,[]) if e['lot_id'] is not None)
    cur.execute("""
        WITH balances AS (
            SELECT i.lot_id,SUM(i.quantity_lb) AS on_hand
            FROM ledger_current_transaction_lines i
            JOIN ledger_current_transactions t ON t.id=i.transaction_id
            WHERE t.effective_status='posted' AND i.lot_id=ANY(%s)
            GROUP BY i.lot_id
        )
        SELECT l.id AS lot_id,l.lot_code,l.product_id,l.entry_source,l.entry_source_notes,
               l.lot_type,l.supplier_lot_code,p.name AS product_name,p.type AS product_type,
               p.uom,p.case_size_lb,COALESCE(b.on_hand,0) AS on_hand
        FROM lots l JOIN products p ON p.id=l.product_id
        LEFT JOIN balances b ON b.lot_id=l.id WHERE l.id=ANY(%s)
    """, (ordered,ordered))
    by_id={}
    for r in cur.fetchall():
        n=dict(r)
        n['unit']='lb' if n['product_type'] in ('batch','finished') else api.ledger_quantity_unit(n['uom'])
        n.pop('uom');n['on_hand']=float(n['on_hand'])
        n['case_size_lb']=float(n['case_size_lb']) if n['case_size_lb'] else None
        n.update(receipts=[],shipments=[],supplier_lot_entries=[])
        by_id[n['lot_id']]=n
    cur.execute("""
        SELECT i.lot_id,t.id,t.timestamp,t.shipper_name,s.name AS supplier_name,
               t.bol_reference,tx.receipt_number,SUM(i.quantity_lb) AS quantity
        FROM ledger_current_transactions t
        JOIN ledger_current_transaction_lines i ON i.transaction_id=t.id
        JOIN transactions tx ON tx.id=t.id LEFT JOIN suppliers s ON s.id=tx.supplier_id
        WHERE t.effective_status='posted' AND t.type='receive'
          AND i.lot_id=ANY(%s) AND i.quantity_lb>0
        GROUP BY i.lot_id,t.id,t.timestamp,t.shipper_name,s.name,t.bol_reference,tx.receipt_number
        ORDER BY t.timestamp DESC,t.id DESC
    """, (ordered,))
    for r in cur.fetchall():
        r['shipper_kind']=api._trace_shipper_kind(r['supplier_name'] or r['shipper_name'])
        n=by_id[r['lot_id']]
        n['receipts'].append(dict(api._trace_receipt_origin(r,n),quantity_lb=float(r['quantity'])))
    cur.execute("""
        SELECT e.lot_id,e.supplier_lot_code,COALESCE(s.name,e.supplier_name) AS supplier_name,
               e.quantity_lb,e.notes FROM lot_supplier_codes e
        LEFT JOIN suppliers s ON s.id=e.supplier_id WHERE e.lot_id=ANY(%s) ORDER BY e.id
    """, (ordered,))
    for r in cur.fetchall():
        n=by_id[r.pop('lot_id')]
        n['supplier_lot_entries'].append(dict(r,quantity_lb=float(r['quantity_lb']) if r['quantity_lb'] is not None else None))
    cur.execute("""
        WITH shipped AS MATERIALIZED (
            SELECT i.lot_id,t.id AS transaction_id,t.timestamp AS shipped_at,
                   t.customer_name,t.order_reference,i.product_id,-SUM(i.quantity_lb) AS quantity_lb
            FROM ledger_current_transactions t
            JOIN ledger_current_transaction_lines i ON i.transaction_id=t.id
            WHERE t.effective_status='posted' AND t.type='ship' AND i.quantity_lb<0 AND i.lot_id=ANY(%s)
            GROUP BY i.lot_id,t.id,t.timestamp,t.customer_name,t.order_reference,i.product_id
        ), order_refs AS (
            SELECT sos.transaction_id,sol.product_id,so.id AS sales_order_id,so.order_number
            FROM sales_order_shipments sos JOIN sales_order_lines sol ON sol.id=sos.sales_order_line_id
            JOIN sales_orders so ON so.id=sol.sales_order_id
            WHERE sos.transaction_id IN (SELECT transaction_id FROM shipped)
            UNION
            SELECT sl.transaction_id,sl.product_id,so.id,so.order_number
            FROM shipment_lines sl JOIN sales_order_lines sol ON sol.id=sl.sales_order_line_id
            JOIN sales_orders so ON so.id=sol.sales_order_id
            WHERE sl.transaction_id IN (SELECT transaction_id FROM shipped)
        ), orders AS (
            SELECT transaction_id,product_id,jsonb_agg(jsonb_build_object(
                'sales_order_id',sales_order_id,'order_number',order_number) ORDER BY sales_order_id) AS sales_orders
            FROM order_refs GROUP BY transaction_id,product_id
        ), refs AS (
            SELECT transaction_id,product_id,array_agg(DISTINCT shipment_id ORDER BY shipment_id) AS shipment_ids
            FROM shipment_lines WHERE transaction_id IN (SELECT transaction_id FROM shipped)
            GROUP BY transaction_id,product_id
        )
        SELECT sh.*,COALESCE(o.sales_orders,'[]'::jsonb) AS sales_orders,
               COALESCE(r.shipment_ids,'{}'::integer[]) AS shipment_ids
        FROM shipped sh LEFT JOIN orders o USING(transaction_id,product_id)
        LEFT JOIN refs r USING(transaction_id,product_id)
        ORDER BY sh.shipped_at,sh.transaction_id,sh.lot_id
    """, (ordered,))
    for r in cur.fetchall():
        n=by_id[r.pop('lot_id')]
        r['quantity_lb']=float(r['quantity_lb'])
        r.update(api._trace_stamp('shipped',r['shipped_at']))
        n['shipments'].append(dict(r))
    result_edges,gaps=[],[]
    for current in ordered:
        for e in adjacency.get(current,[]):
            e.update(quantity_used=float(e['quantity_used']),output_quantity=float(e['output_quantity'] or 0),
                     unit=by_id[current]['unit'],**api._trace_stamp('produced',e.pop('timestamp')))
            result_edges.append(e)
            if e['lot_id'] is None:
                gaps.append({'transaction_id':e['transaction_id'],'lot_id':current,'reason':'Consumed lot has no identified output lot'})
    return {'start_lot_id':lot_id,'lots':[by_id[i] for i in ordered],'consumption':result_edges,'gaps':gaps,
            'quantity_basis':'Recorded consumption per transaction; downstream lots are potential exposure, not allocated ingredient weight.'}


def report(cur, lot, direction, api):
    graph=forward(cur,lot['id'],api)
    start=graph['lots'][0]
    result={'report_version':1,'complete':True,'direction':direction,'lot_id':lot['id'],
            'product_id':lot['product_id'],'batch_lot_code':lot['lot_code'],'product_name':lot['product_name'],
            'forward_trace':graph,'customer_shipments':start['shipments'],'on_hand_lb':start['on_hand'],
            'total_shipped_lb':sum(s['quantity_lb'] for s in start['shipments'])}
    if direction=='backward':
        from lot_trace_backward import BackwardTrace
        origin=BackwardTrace(api)._trace_lot_origin(cur,dict(lot,lot_id=lot['id']),0,frozenset())
        result['origin']=origin
        result['input_tree']=origin if origin['kind']=='made' else None
        result['output_lb']=origin.get('output_lb',0)
        first=(origin.get('transactions') or [{}])[0]
        result.update(produced_date=first.get('produced_date',''),produced_time=first.get('produced_time',''))
    return result


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
