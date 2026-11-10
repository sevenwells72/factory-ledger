"""Corrected backward tree for explicit dashboard audit reports only.

The existing GPT trace helpers remain exactly as on origin/main.
"""
from decimal import Decimal
from datetime import timezone
import lot_trace

TRACE_FOUND_SHIPPERS = frozenset({
    'found', 'found inventory', 'inventory found', 'physical count',
    'opening balance', 'opening stock',
    'initial inventory', 'inventory correction', 'inventory intake',
    # Not an outside source (owner, 2026-11-09): stock moved in from the blender warehouse.
    'blender warehouse'})
# Receipts that name no supplier at all.
TRACE_UNNAMED_SHIPPERS = frozenset({
    '', 'unknown', 'unknown supplier', 'n/a', 'na', 'none', '-'})
# Display-only check: recorded inputs outside 80–120% of the output are flagged.
TRACE_INPUT_RATIO_LOW = Decimal('0.8')
TRACE_INPUT_RATIO_HIGH = Decimal('1.2')


class BackwardTrace:
    def __init__(self, api):
        self.api = api

    def _trace_shipper_kind(self, name):
        """'found', 'unnamed' or 'supplier' for a receipt's shipper/supplier name."""
        normalized = ' '.join((name or '').replace('’', "'").lower().split())
        if normalized in TRACE_FOUND_SHIPPERS:
            return 'found'
        if normalized in TRACE_UNNAMED_SHIPPERS:
            return 'unnamed'
        return 'supplier'


    def _trace_stamp(self, prefix, value):
        """{prefix}_at (UTC ISO) plus {prefix}_date / {prefix}_time in plant time (ET)."""
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)   # ledger timestamps are stored as UTC
        day, clock = self.api.format_timestamp(value)
        return {prefix + "_at": value.isoformat() if value is not None else None,
                prefix + "_date": day, prefix + "_time": clock}


    def _trace_lot_receipts(self, cur, lot_id):
        """Posted receipts onto one lot, newest first, each with its shipper kind."""
        cur.execute("""
            SELECT t.id, t.timestamp, t.shipper_name, s.name AS supplier_name,
                   t.bol_reference, tx.receipt_number, SUM(tl.quantity_lb) AS quantity
            FROM ledger_current_transactions t
            JOIN ledger_current_transaction_lines tl ON tl.transaction_id = t.id
            JOIN transactions tx ON tx.id = t.id
            LEFT JOIN suppliers s ON s.id = tx.supplier_id
            WHERE t.type = 'receive' AND t.effective_status = 'posted'
              AND tl.lot_id = %s AND tl.quantity_lb > 0
            GROUP BY t.id, t.timestamp, t.shipper_name, s.name, t.bol_reference, tx.receipt_number
            ORDER BY t.timestamp DESC, t.id DESC
        """, (lot_id,))
        receipts = cur.fetchall()
        for row in receipts:
            row['shipper_kind'] = self._trace_shipper_kind(row['supplier_name'] or row['shipper_name'])
        return receipts


    def _trace_receipt_origin(self, row, lot):
        """One receipt as an origin: a named supplier, found inventory, or no supplier named."""
        if row['shipper_kind'] == 'supplier':
            return {
                "kind": "received",
                "transaction_id": row['id'],
                **self._trace_stamp("received", row['timestamp']),
                "supplier_name": row['supplier_name'] or row['shipper_name'],
                "bol_reference": row['bol_reference'],
                "receipt_number": row['receipt_number'],
                "quantity_lb": float(row['quantity']),
                "supplier_lot_code": lot.get('supplier_lot_code'),
            }
        if row['shipper_kind'] == 'found':
            # A placeholder receipt ("found") is kept for reference, never shown as a supplier.
            return {"kind": "found_inventory", "transaction_id": row['id'],
                    "recorded_shipper": row['shipper_name'],
                    **self._trace_stamp("received", row['timestamp'])}
        return {"kind": "unknown", "reason": "receipt_without_supplier",
                "transaction_id": row['id'], "recorded_shipper": row['shipper_name'],
                **self._trace_stamp("received", row['timestamp'])}


    def _trace_lot_origin(self, cur, lot, depth, ancestors):
        """Origin of one lot: made (recursing), received, found_inventory or unknown."""
        lot_id = lot['lot_id']
        cur.execute("""
            SELECT t.id, t.type, t.timestamp
            FROM ledger_current_transactions t
            WHERE t.effective_status = 'posted' AND t.type IN ('make', 'pack')
              AND EXISTS (
                  SELECT 1 FROM ledger_current_transaction_lines o
                  WHERE o.transaction_id = t.id AND o.lot_id = %s AND o.quantity_lb > 0)
            ORDER BY t.timestamp, t.id
        """, (lot_id,))
        producing = cur.fetchall()
        receipts = self._trace_lot_receipts(cur, lot_id)
        if producing:
            origin = self._trace_made_origin(cur, lot_id, producing, depth, ancestors)
            # A made lot can also hold received stock (e.g. found inventory added to a
            # packed lot). The makes stay the origin; every such receipt is listed so the
            # trace never looks complete while part of the lot came in another way.
            origin["also_received"] = [
                {**self._trace_receipt_origin(row, lot), "quantity_lb": float(row['quantity'])}
                for row in reversed(receipts)]
            return origin

        by_kind = {}
        for row in receipts:
            by_kind.setdefault(row['shipper_kind'], []).append(row)
        for kind in ('supplier', 'found', 'unnamed'):   # latest named receipt wins, as _trace_ingredient_backward
            if by_kind.get(kind):
                origin = self._trace_receipt_origin(by_kind[kind][0], lot)
                if kind == 'supplier':
                    origin["receipt_count"] = len(receipts)
                origin["receipts"] = [dict(self._trace_receipt_origin(r, lot), quantity_lb=float(r["quantity"])) for r in reversed(receipts)]
                return origin
        if lot.get('entry_source') in ('found_inventory', 'adjusted'):
            return {"kind": "found_inventory", "transaction_id": None}
        return {"kind": "unknown", "reason": "no_posted_origin", "transaction_id": None}


    def _trace_made_origin(self, cur, lot_id, producing, depth, ancestors):
        """All posted makes/packs that produced lot_id (a lot code can take several makes)."""
        if lot_id in ancestors:
            return {"kind": "made",
                    "truncated": "cycle",
                    "transactions": [
                        {"transaction_id": row['id'], "type": row['type'],
                         **self._trace_stamp("produced", row['timestamp']), "inputs": []}
                        for row in producing]}
        path = ancestors | {lot_id}
        transactions = [self._trace_made_transaction(cur, row, depth, path, lot_id) for row in producing]
        return {
            "kind": "made",
            "transactions": transactions,
            "output_lb": float(sum(Decimal(str(t['lot_output_lb'])) for t in transactions)),
            "inputs_total_lb": float(sum(Decimal(str(t['inputs_total_lb'])) for t in transactions)),
            "inputs_mismatch": any(t['inputs_mismatch'] for t in transactions),
        }


    def _trace_made_transaction(self, cur, txn, depth, path, target_lot_id=None):
        """One posted make/pack: lb output, consumed lots (current lines) and the input check.

        The check is display-only: inputs_mismatch is true when the recorded lb inputs
        fall outside 80–120% of the lb output. Nothing is blocked or corrected.
        """
        cur.execute("""
            SELECT tl.product_id, tl.lot_id, p.name AS product_name, p.type AS product_type, p.uom,
                   l.lot_code, l.supplier_lot_code, l.entry_source, l.entry_source_notes, l.lot_type,
                   SUM(tl.quantity_lb) AS quantity
            FROM ledger_current_transaction_lines tl
            JOIN products p ON p.id = tl.product_id
            LEFT JOIN lots l ON l.id = tl.lot_id
            WHERE tl.transaction_id = %s
            GROUP BY tl.product_id, tl.lot_id, p.name, p.type, p.uom,
                     l.lot_code, l.supplier_lot_code, l.entry_source, l.entry_source_notes, l.lot_type, (tl.quantity_lb > 0)
            ORDER BY p.name, tl.lot_id
        """, (txn['id'],))
        output_lb = Decimal(0)
        lot_output_lb = Decimal(0)
        inputs_total = Decimal(0)
        inputs = []
        for row in cur.fetchall():
            quantity = Decimal(row['quantity'])
            unit = 'lb' if row['product_type'] in ('batch', 'finished') else self.api.ledger_quantity_unit(row['uom'])
            if quantity > 0:
                if unit == 'lb':
                    output_lb += quantity
                    if row["lot_id"] == target_lot_id:
                        lot_output_lb += quantity
                continue
            if quantity == 0:
                continue
            quantity = -quantity
            if unit == 'lb':
                inputs_total += quantity
            node = {key: row[key] for key in (
                'lot_id', 'lot_code', 'product_id', 'product_name', 'product_type',
                'entry_source', 'supplier_lot_code')}
            node.update(quantity=float(quantity), unit=unit)
            if row.get("lot_type"):
                node["lot_type"] = row["lot_type"]
            entries = lot_trace.supplier_entries(cur, row["lot_id"])
            if entries:
                node["supplier_lot_entries"] = entries
            if row.get("entry_source_notes"):
                node["entry_source_notes"] = row["entry_source_notes"]
            inputs.append(node)
        for node in inputs:
            node["origin"] = self._trace_lot_origin(cur, node, depth + 1, path)
        mismatch = output_lb > 0 and not (
            output_lb * TRACE_INPUT_RATIO_LOW <= inputs_total <= output_lb * TRACE_INPUT_RATIO_HIGH)
        return {
            "transaction_id": txn['id'],
            "type": txn['type'],
            **self._trace_stamp("produced", txn['timestamp']),
            "output_lb": float(output_lb),
            "lot_output_lb": float(lot_output_lb),
            "inputs_total_lb": float(inputs_total),
            "inputs_mismatch": bool(mismatch),
            "inputs": inputs,
        }


    def _trace_input_tree(self, cur, lot_row):
        """Backward input tree for a made/packed lot; null when no posted make/pack made it."""
        origin = self._trace_lot_origin(cur, {
            'lot_id': lot_row['id'], 'entry_source': lot_row.get('entry_source'),
            'supplier_lot_code': lot_row.get('supplier_lot_code')}, 0, frozenset())
        return origin if origin['kind'] == 'made' else None
