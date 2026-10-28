"""A5 supplier-lot corrections. The A1 caller owns the transaction and locks."""
from datetime import datetime
import re

from psycopg2.extras import Json

import exceptions_enforcement as a3b
import lot_confirmation as a5


def supplier_code(value):
    """A bounded label, never a placeholder, control sequence or free-form markup."""
    if not isinstance(value, str):
        a5.fail('SUPPLIER_LOT_CODE_INVALID', 'Enter the supplier lot code from the physical label.')
    code = value.strip()
    sentinel = re.sub(r'[^A-Z0-9]', '', code.upper())
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 ._/#()+-]{0,199}', code)
            or sentinel in {'NA', 'UNKNOWN', 'UNIDENTIFIED', 'NONE', 'NULL', 'TBD',
                            'PENDING', 'NOTAVAILABLE', 'NOTAPPLICABLE'}):
        a5.fail('SUPPLIER_LOT_CODE_INVALID', 'Enter a real supplier lot code (1–200 characters), not a placeholder.')
    return code


def _norm(code):
    return re.sub(r'[^A-Z0-9]', '', code.upper())


def opening_stock_identity(entry_note, code):
    """Evidence rule for lots created as opening stock from a physical count (their entry note
    says how they were made). A POOLED lot holds several supplier lots and never takes one code:
    split it with Found/adjust corrections instead. Any other opening-stock lot only takes a supplier
    code its own entry note documents as a possible source ("supplier lot X")."""
    note = (entry_note or '').upper()
    if 'POOLED' in note:
        a5.fail('POOLED_LOT_NOT_IDENTIFIABLE',
                'This lot pools several supplier lots; one supplier code cannot identify it. '
                'Split it with Found and adjust corrections, each with its own supplier lot.',
                'Este lote combina varios lotes de proveedor; un solo código no lo identifica. '
                'Divídalo con correcciones de Encontrado y Ajuste, cada una con su lote de proveedor.')
    if not note.startswith(('OPENING STOCK', 'OPENING BALANCE')):
        return
    documented = {_norm(x) for x in re.findall(r'SUPPLIER LOT ([^,;|]+)', note)} - {''}
    if _norm(code) not in documented:
        a5.fail('OPENING_LOT_CODE_NOT_DOCUMENTED',
                'This opening-stock lot lists its possible supplier lots, and this code is not one of them. '
                'Record different evidence with a Found or adjust correction instead.',
                'Este lote de inventario inicial enumera sus posibles lotes de proveedor y este código no está '
                'entre ellos. Registre otra evidencia con una corrección de Encontrado o Ajuste.')


def validate(api, cur, payload, lock=False):
    import ticket_actions as actions
    occurred_at, source = api.validate_inventory_occurred_at(
        datetime.fromisoformat(payload['occurred_at']), payload['backfill'])
    code = supplier_code(payload['supplier_lot_code'])
    reason = a3b.resolve_reason(cur, 'update_supplier_lot', payload['reason_code'], note=payload.get('note'))
    # Same order as receive: product, then lot. Its row lock serializes different
    # correction tickets and receipts; A1 already locks the individual ticket.
    cur.execute('SELECT product_id FROM lots WHERE id=%s', (payload['lot_id'],))
    head = cur.fetchone()
    if not head:
        actions.fail('LOT_NOT_FOUND', 'Lot is missing; resolve an existing lot.')
    product = actions.product(api, cur, head['product_id'], lock)
    cur.execute('''SELECT id,product_id,lot_code,status,supplier_id,supplier_lot_code,
                          lot_type,identity_status,identify_by FROM lots WHERE id=%s''' +
                (' FOR UPDATE' if lock else ''), (payload['lot_id'],))
    lot = cur.fetchone()
    if not lot or lot['product_id'] != product['id']:
        actions.fail('LOT_IDENTITY_CHANGED', 'The lot changed; prepare a fresh correction.')
    if lot['status'] == 'merged':
        actions.fail('LOT_MERGED', 'This lot was merged; correct the surviving lot.')
    if not a5.supplier_lot_identity_applies(cur, lot['product_id']):
        a5.fail('SUPPLIER_LOT_NOT_APPLICABLE', 'Supplier-lot identity applies to purchased ingredients.')
    cur.execute('SELECT id FROM lot_supplier_codes WHERE lot_id=%s ORDER BY id', (lot['id'],))
    if lot['lot_type'] == 'commingled' or cur.fetchall():
        a5.fail('COMMINGLED_LOT_REQUIRES_REVIEW', 'This lot has component supplier codes; one code cannot identify all components.')
    cur.execute('SELECT entry_source_notes FROM lots WHERE id=%s', (lot['id'],))
    opening_stock_identity(cur.fetchone()['entry_source_notes'], code)
    # Pin what the operator saw, including an event version to detect A→B→A.
    cur.execute("SELECT max(id) AS version FROM lot_events WHERE lot_id=%s AND event IN ('supplier_lot','identified')", (lot['id'],))
    state = api_value(api, dict(lot, identity_version=cur.fetchone()['version']))
    if payload.get('specification', state) != state:
        actions.fail('LOT_IDENTITY_CHANGED', 'The lot identity changed after prepare; review a fresh correction.')
    cur.execute("""SELECT id FROM exceptions WHERE lot_id=%s AND kind='UNIDENTIFIED_LOT'
                   AND status IN ('open','escalated') ORDER BY id""", (lot['id'],))
    exception = cur.fetchone()
    draft = {'product_id': lot['product_id'], 'product_name': product['name'],
             'lot_id': lot['id'], 'lot_code': lot['lot_code'], 'lot_exists': True,
             'previous_supplier_lot_code': lot['supplier_lot_code'], 'supplier_lot_code': code,
             'previous_identity_status': lot['identity_status'], 'identity_status': 'identified',
             'identify_by': None, 'reason_code': reason['code'], 'reason': reason['label_en'],
             'note': reason['note'], 'exception_id': exception['id'] if exception else None}
    return draft, state, None, {}, occurred_at, source, state, None


def api_value(api, value):
    import write_tickets
    return write_tickets.json_value(api, value)


def post(api, cur, validated, payload, request, ticket_id, receipt_number):
    draft, old, _, _, occurred_at, _, _, _ = validated
    actor = api.request_actor(request)
    # New endpoint requires a named actor (including a key session).
    actor_id = actor['id']
    cur.execute("""UPDATE lots SET supplier_lot_code=%s,identity_status='identified',identify_by=NULL
                   WHERE id=%s""", (draft['supplier_lot_code'], draft['lot_id']))
    new = {'supplier_lot_code': draft['supplier_lot_code'], 'identity_status': 'identified', 'identify_by': None}
    cur.execute("""INSERT INTO lot_events(lot_id,event,old,new,reason_code,happened_at,entered_by_actor_id,ticket_id)
                   VALUES (%s,'supplier_lot',%s,%s,%s,%s,%s,%s) RETURNING id""",
                (draft['lot_id'], Json({k: old[k] for k in new}), Json(new), draft['reason_code'],
                 occurred_at, actor_id, ticket_id))
    event_id = cur.fetchone()['id']
    # Only open/escalated rows change: subsequent corrections never overwrite
    # the original resolution, actor, time or receipt. No separate XR call.
    cur.execute("""UPDATE exceptions SET status='resolved',resolved_at=clock_timestamp(),
                   resolved_by_actor_id=%s,resolution_kind='identified',resolution_note=%s,
                   resolution_ticket_id=%s WHERE lot_id=%s AND kind='UNIDENTIFIED_LOT'
                   AND status IN ('open','escalated') RETURNING id""",
                (actor_id, draft['note'], ticket_id, draft['lot_id']))
    resolved = [r['id'] for r in cur.fetchall()]
    api._record_actor_write(cur, request, 'lots', draft['lot_id'])
    for exception_id in resolved:
        api._record_actor_write(cur, request, 'exceptions', exception_id)
    return {'success': True, 'updated': True, 'lot_id': draft['lot_id'], 'lot_code': draft['lot_code'],
            'product_id': draft['product_id'], 'product_name': draft['product_name'],
            'previous_supplier_lot_code': draft['previous_supplier_lot_code'], **new,
            'reason_code': draft['reason_code'], 'note': draft['note'], 'lot_event_id': event_id,
            'resolved_exception_ids': resolved, 'entered_by_actor_id': actor_id}
