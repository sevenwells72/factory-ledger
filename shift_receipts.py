"""One-query receipt snapshot for A9, using the same effective ledger as receipt_detail.

Only fields consumed by the shift report are loaded. Catalog/evidence are request
local; nothing is persisted, and stored receipt responses are never rewritten.
"""
from collections import defaultdict
from datetime import datetime

import kosher_tiers


def _times(row, *keys):
    for key in keys:
        if isinstance(row.get(key), str):
            row[key] = datetime.fromisoformat(row[key])
    return row


def load(api, cur, numbers, catalog):
    cur.execute(f'''
        WITH tickets AS MATERIALIZED (
            SELECT id,receipt_number,action,status,actor_id,operator_id,key_kind,payload,
                   response,draft,warnings,acknowledged,result_ref,committed_at,client_source
            FROM write_tickets WHERE receipt_number=ANY(%s)
        ), tx AS MATERIALIZED (
            SELECT t.id,t.type,t.effective_status,raw.ticket_id,raw.supplier_id,s.name AS supplier_name,
                   raw.kosher_tier,raw.kosher_attested_by_actor_id,raw.kosher_attested_at,
                   raw.transaction_kosher_downgrade,a.name AS kosher_attested_by_name
            FROM ledger_current_transactions t JOIN transactions raw ON raw.id=t.id
            JOIN tickets w ON w.id=raw.ticket_id
            LEFT JOIN suppliers s ON s.id=raw.supplier_id
            LEFT JOIN actors a ON a.id=raw.kosher_attested_by_actor_id
        ), lines AS MATERIALIZED (
            SELECT l.id,l.transaction_id,l.product_id,p.name AS product_name,l.lot_id,lot.lot_code,l.quantity_lb
            FROM ledger_current_transaction_lines l JOIN tx ON tx.id=l.transaction_id
            JOIN products p ON p.id=l.product_id LEFT JOIN lots lot ON lot.id=l.lot_id
        ), lot_ids AS (
            SELECT lot_id AS id FROM lines WHERE lot_id IS NOT NULL
            UNION SELECT value::bigint FROM tickets,
                jsonb_array_elements_text(COALESCE(result_ref->'lot_ids','[]'::jsonb))
            UNION SELECT (payload->>'lot_id')::bigint FROM tickets WHERE payload->>'lot_id' IS NOT NULL
            UNION SELECT (draft->>'lot_id')::bigint FROM tickets WHERE draft->>'lot_id' IS NOT NULL
            UNION SELECT lot_id FROM exceptions WHERE status IN ('open','escalated') AND lot_id IS NOT NULL
        ), lot_rows AS MATERIALIZED (
            SELECT l.id,l.product_id,l.lot_code,l.status,l.supplier_lot_code,l.supplier_id,s.name AS supplier_name,
                   l.identity_status,l.identify_by,l.kosher_tier,l.kosher_attested_by_actor_id,l.kosher_attested_at,
                   a.name AS kosher_attested_by_name,COALESCE(b.balance,0) AS on_hand_lb,
                   EXISTS(SELECT 1 FROM shortage_flags sf WHERE sf.lot_id=l.id AND sf.status IN ('open','escalated')) AS has_shortage
            FROM lots l JOIN lot_ids i ON i.id=l.id LEFT JOIN suppliers s ON s.id=l.supplier_id
            LEFT JOIN actors a ON a.id=l.kosher_attested_by_actor_id
            LEFT JOIN (SELECT lot_id,SUM(quantity_lb) AS balance FROM {api.POSTED_LINES}
                       WHERE lot_id IN (SELECT id FROM lot_ids) GROUP BY lot_id) b ON b.lot_id=l.id
        ), shipment_rows AS MATERIALIZED (
            SELECT s.id,s.ticket_id,s.sales_order_id,s.bol_reference,s.proof_status,s.proof_due_at,s.entered_by_actor_id
            FROM shipments s JOIN tickets w ON w.id=s.ticket_id
        ), order_ids AS (
            SELECT COALESCE(result_ref->>'order_id',payload->>'order_id')::bigint AS id FROM tickets
            UNION SELECT sales_order_id FROM shipment_rows
        )
        SELECT
            COALESCE((SELECT jsonb_agg(to_jsonb(w) ORDER BY w.id) FROM tickets w),'[]') AS tickets,
            COALESCE((SELECT jsonb_agg(to_jsonb(t) ORDER BY t.id) FROM tx t),'[]') AS transactions,
            COALESCE((SELECT jsonb_agg(to_jsonb(l) ORDER BY l.id) FROM lines l),'[]') AS lines,
            COALESCE((SELECT jsonb_agg(to_jsonb(l) ORDER BY l.id) FROM lot_rows l),'[]') AS lots,
            COALESCE((SELECT jsonb_agg(to_jsonb(s) ORDER BY s.id) FROM shipment_rows s),'[]') AS shipments,
            COALESCE((SELECT jsonb_agg(to_jsonb(c) ORDER BY c.lot_id) FROM (
                SELECT transaction_id,lot_id,method,value,actor_id,move_id,created_at
                FROM transaction_lot_confirmations WHERE transaction_id IN (SELECT id FROM tx)) c),'[]') AS confirmations,
            COALESCE((SELECT jsonb_agg(to_jsonb(s) ORDER BY s.ingredient_product_id) FROM (
                SELECT transaction_id,ingredient_product_id,substitute_product_id,lot_id,reason_code,note,actor_id
                FROM transaction_substitutions WHERE transaction_id IN (SELECT id FROM tx)) s),'[]') AS substitutions,
            COALESCE((SELECT jsonb_agg(to_jsonb(e) ORDER BY e.id) FROM (
                SELECT id,kind,ticket_id,lot_id,shipment_id,status,due_at,resolution_kind FROM exceptions
                WHERE ticket_id IN (SELECT id FROM tickets) OR lot_id IN (SELECT id FROM lot_ids)
                   OR shipment_id IN (SELECT id FROM shipment_rows)) e),'[]') AS exceptions,
            COALESCE((SELECT jsonb_agg(to_jsonb(o)) FROM (
                SELECT so.id,so.order_number,c.name AS customer FROM sales_orders so
                JOIN customers c ON c.id=so.customer_id WHERE so.id IN (SELECT id FROM order_ids)) o),'[]') AS orders,
            (SELECT jsonb_agg(to_jsonb(p)) FROM (SELECT id,name,odoo_code,type FROM products) p) AS products,
            (SELECT jsonb_agg(to_jsonb(a)) FROM (SELECT id,name,role FROM actors) a) AS actors,
            (SELECT jsonb_agg(to_jsonb(r)) FROM (SELECT code,label_en,label_es,note_required FROM correction_reasons) r) AS reasons
        ''', (numbers,))
    data = cur.fetchone()
    catalog._products = {r['id']: r for r in data['products'] or []}
    catalog._actors = {r['id']: r for r in data['actors'] or []}
    catalog._reasons = {r['code']: r for r in data['reasons'] or []}
    catalog._lots = {r['id']: _times(r, 'identify_by', 'kosher_attested_at') for r in data['lots']}
    catalog._exceptions = [_times(r, 'due_at') for r in data['exceptions']]
    catalog._orders = {r['id']: r for r in data['orders']}
    shipments = {r['ticket_id']: _times(r, 'proof_due_at') for r in data['shipments']}
    lines, confirmations, substitutions, transactions = (defaultdict(list) for _ in range(4))
    for r in data['lines']: lines[r['transaction_id']].append(r)
    for r in data['confirmations']: confirmations[r['transaction_id']].append(_times(r, 'created_at'))
    for r in data['substitutions']: substitutions[r['transaction_id']].append(r)
    for t in data['transactions']:
        _times(t, 'kosher_attested_at')
        t.update(lines=lines[t['id']], lot_confirmations=confirmations[t['id']], substitutions=substitutions[t['id']])
        t.update(kosher_tiers.fields(t))
        transactions[t['ticket_id']].append(t)
    result = {}
    for ticket in data['tickets']:
        _times(ticket, 'committed_at')
        txns = transactions[ticket['id']]
        lot_ids = set((ticket['result_ref'] or {}).get('lot_ids', []))
        lot_ids.update(line['lot_id'] for t in txns for line in t['lines'] if line['lot_id'] is not None)
        lots = [dict(catalog._lots[i]) for i in sorted(lot_ids) if i in catalog._lots]
        for lot in lots: lot.update(kosher_tiers.fields(lot))
        happened = datetime.fromisoformat(ticket['payload']['occurred_at'])
        entered = ticket['committed_at']
        order_id = (ticket['result_ref'] or {}).get('order_id') or ticket['payload'].get('order_id')
        receipt = {
            'receipt_number': ticket['receipt_number'], 'action': ticket['action'], 'status': ticket['status'],
            'actor': ticket['draft']['actor'], 'operator_id': ticket['operator_id'], 'client_source': ticket['client_source'],
            'happened_at': happened, 'entered_at': entered,
            'late_entry': entered is not None and happened.astimezone(api.PLANT_TIMEZONE).date() != entered.astimezone(api.PLANT_TIMEZONE).date(),
            'draft': ticket['draft'], 'response': ticket['response'], 'warnings': ticket['warnings'], 'acknowledged': ticket['acknowledged'],
            'transactions': txns, 'lots': lots, 'shipment': shipments.get(ticket['id']),
            'order': catalog._orders.get(order_id, {'id': order_id}) if order_id else None,
            'expected_receipts': [{'id': i} for i in (ticket['result_ref'] or {}).get('expected_receipt_ids', [])],
        }
        result[ticket['receipt_number']] = (receipt, ticket)
    return result
