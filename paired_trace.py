"""Read-only links between lots joined by a posted paired wrong-lot move.

Owner rule (2026-11-24): a recall must show BOTH lots of a wrong-lot correction,
the lot first recorded and the lot it was corrected to, with the correction
labeled. Over-inclusive is the rule for recalls.

A paired move says X lb recorded against one lot of a purchased ingredient was
really taken from another lot of the same product. It does not say which make or
pack used those pounds, so every trace treats the two lots as linked in both
directions: whatever one fed may hold the other. That is an assumption made for
safety, never a physically verified allocation, and every label says so.

The two lots are read from the move's CURRENT posted ledger lines, not from the
columns of paired_lot_moves, so a later lot merge or line correction is followed.
A voided move links nothing. The table exists only once migration 098 has been
applied; before that every function here answers "no links".
"""

LABEL = ('Wrong-lot correction {receipt}: {quantity} lb first recorded on lot {restored} '
         'was corrected to lot {used}. Both lots are included; the record cannot say which batches held which lot.')
LABEL_ES = ('Corrección de lote {receipt}: {quantity} lb registradas primero en el lote {restored} '
            'se corrigieron al lote {used}. Se incluyen ambos lotes; el registro no indica qué tandas llevaron cada lote.')


def available(cur):
    """True once the paired_lot_moves table exists (migration 098 applied, marker on or off)."""
    cur.execute("SELECT to_regclass('public.paired_lot_moves') IS NOT NULL AS present")
    return bool(cur.fetchone()['present'])


def _number(value):
    text = f'{float(value):,.2f}'.rstrip('0').rstrip('.')
    return text


def moves(cur, api):
    """Every posted paired move, oldest first. One query; the table is small by nature."""
    if not available(cur):
        return []
    cur.execute("""
        WITH legs AS (
            SELECT i.transaction_id, i.lot_id, SUM(i.quantity_lb) AS quantity_lb
            FROM ledger_current_transaction_lines i
            WHERE i.transaction_id IN (SELECT transaction_id FROM paired_lot_moves)
            GROUP BY i.transaction_id, i.lot_id
        )
        SELECT m.transaction_id, t.timestamp, t.effective_record->>'receipt_number' AS receipt_number,
               r.lot_id AS restored_lot_id, a.lot_code AS restored_lot_code,
               a.supplier_lot_code AS restored_supplier_lot_code,
               u.lot_id AS used_lot_id, b.lot_code AS used_lot_code,
               b.supplier_lot_code AS used_supplier_lot_code,
               p.id AS product_id, p.name AS product_name, r.quantity_lb
        FROM paired_lot_moves m
        JOIN ledger_current_transactions t ON t.id = m.transaction_id
        JOIN legs r ON r.transaction_id = m.transaction_id AND r.quantity_lb > 0
        JOIN legs u ON u.transaction_id = m.transaction_id AND u.quantity_lb < 0
        JOIN lots a ON a.id = r.lot_id
        JOIN lots b ON b.id = u.lot_id
        JOIN products p ON p.id = a.product_id
        WHERE t.effective_status = 'posted'
        ORDER BY t.timestamp, t.id
    """)
    result = []
    for row in cur.fetchall():
        move = dict(row)
        move['quantity_lb'] = float(move['quantity_lb'])
        move.update(api._trace_stamp('corrected', move.pop('timestamp')))
        words = dict(receipt=move['receipt_number'] or f"#{move['transaction_id']}",
                     quantity=_number(move['quantity_lb']),
                     restored=move['restored_lot_code'], used=move['used_lot_code'])
        move['label'] = LABEL.format(**words)
        move['label_es'] = LABEL_ES.format(**words)
        move['basis'] = 'assumed_link_not_physically_verified'
        result.append(move)
    return result


def links(all_moves):
    """{lot_id: [move, ...]} for both lots of every move."""
    by_lot = {}
    for move in all_moves:
        for lot_id in (move['restored_lot_id'], move['used_lot_id']):
            by_lot.setdefault(lot_id, []).append(move)
    return by_lot


def other_lot(move, lot_id):
    """The lot on the far side of a move, as seen from lot_id."""
    if move['restored_lot_id'] == lot_id:
        return dict(lot_id=move['used_lot_id'], lot_code=move['used_lot_code'],
                    supplier_lot_code=move['used_supplier_lot_code'], role='corrected_to')
    return dict(lot_id=move['restored_lot_id'], lot_code=move['restored_lot_code'],
                supplier_lot_code=move['restored_supplier_lot_code'], role='first_recorded')


def linked_lot_ids(by_lot, lot_id):
    """Every lot reachable from lot_id through moves (a chain A-B, B-C links all three), nearest first."""
    seen, order, pending = {lot_id}, [], [lot_id]
    while pending:
        current = pending.pop(0)
        for move in by_lot.get(current, []):
            partner = other_lot(move, current)['lot_id']
            if partner not in seen:
                seen.add(partner)
                order.append(partner)
                pending.append(partner)
    return order


def corrections(by_lot, lot_id):
    """What a trace shows for one lot: each move touching it, with the other lot named."""
    return [dict(move, other_lot=other_lot(move, lot_id)) for move in by_lot.get(lot_id, [])]


def lot_row(cur, lot_id):
    """Identity of one lot for a paired-lot entry in a trace."""
    cur.execute("""
        SELECT l.id, l.id AS lot_id, l.lot_code, l.supplier_lot_code, l.entry_source, l.lot_type,
               l.kosher_tier, l.kosher_attested_by_actor_id, l.kosher_attested_at,
               p.id AS product_id, p.name AS product_name, p.type AS product_type
        FROM lots l JOIN products p ON p.id = l.product_id WHERE l.id = %s
    """, (lot_id,))
    row = cur.fetchone()
    return dict(row) if row else None


def annotate_tree(cur, origin, by_lot, lot_origin):
    """Label every input lot of a backward tree that a wrong-lot correction touches, in place.

    Each such input gets `paired_moves` (the corrections, labeled) and `paired_lots`
    (the lot or lots on the other side, each with its own origin from lot_origin),
    so a batch's backward trace names both the lot first recorded and the lot it
    was corrected to. Returns the distinct moves found, oldest first.
    """
    found = {}

    def walk(node_origin):
        for txn in (node_origin or {}).get('transactions') or []:
            for node in txn.get('inputs') or []:
                touching = corrections(by_lot, node.get('lot_id'))
                if touching:
                    node['paired_moves'] = touching
                    node['paired_lots'] = []
                    for partner_id in linked_lot_ids(by_lot, node['lot_id']):
                        partner = lot_row(cur, partner_id)
                        if partner:
                            partner.pop('id')
                            partner['origin'] = lot_origin(cur, partner)
                            node['paired_lots'].append(partner)
                    for move in touching:
                        found[move['transaction_id']] = {k: v for k, v in move.items() if k != 'other_lot'}
                walk(node.get('origin'))

    walk(origin)
    return [found[key] for key in sorted(found)]
