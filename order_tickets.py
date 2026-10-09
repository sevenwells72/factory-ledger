"""A7: order tickets — prepare/commit for every §4.3 order action.

Thin wrappers over the live order cores in main.py. A prepare DRY-RUNS the
same core inside a savepoint (so the draft is exactly what commit will
return), rolls it back and issues the ticket; commit runs the core for real
under the A1 ticket lifecycle in write_tickets (single use, expiry, bound to
user and payload, replay, concurrent double-commit → one result). No name
resolution and no order rules of its own: customer_po duplicate block with
explicit override, the No-PO flag, service lines, external_order_reference,
the state model and the status table all live in main.py and are reused.

Hook points in shared files are marked "A7 hook": write_tickets (registry,
issue_ticket, commit and receipt hooks), permissions ('close_order' row),
main.py (`_insert_order_lines_core`, `_validate_requested_order_status`,
eight `_*_core` extractions — handler bodies moved verbatim).
"""
import json
from datetime import date, datetime
from decimal import Decimal
from typing import List, Literal, Optional

import psycopg2.errors
from fastapi import Depends, HTTPException, Request
from psycopg2.extras import Json
from pydantic import Field, confloat, root_validator, validator

import permissions
import ticket_actions
import write_tickets as wt
import expected_receipt_tickets as er_tickets

ACTIONS = wt.ORDER_ACTIONS
PositiveId = wt.PositiveId
fail = wt.fail
DUPLICATE_WINDOW = '24 hours'
REFERENCE_CONSTRAINTS = ('sales_orders_customer_external_reference_uniq', 'sales_order_create_receipts_pkey',
                         'sales_orders_external_reference_uniq', 'sales_order_receipts_reference_uniq',
                         'write_tickets_order_reference_uniq')


# ---------------------------------------------------------------------------
# A6 hook — ship gate. Closing an order as shipped will require a recorded
# shipment (BOL / proof, design §8.2). Today there is NO gate: this returns no
# blockers, which is exactly the existing close behaviour. A6 replaces the
# body; both callers (close_order, status=invoiced) already treat the result
# as draft blockers at prepare and as a stale-ticket refusal at commit.
# ---------------------------------------------------------------------------
def shipment_gate(api, cur, order, close_reason):
    return []


# ---------------------------------------------------------------------------
# Request models: ids only. Names, self-reported identities and the legacy
# mode flag are rejected before any database work.
# ---------------------------------------------------------------------------
def _trim(value):
    return (value.strip() or None) if value is not None else None


class OrderRequest(wt.IdInput):
    client_source: wt.ClientSource = 'api'

    @root_validator(pre=True)
    def ids_and_key_identity_only(cls, values):
        if isinstance(values, dict):
            if any(k in values for k in ('customer_name', 'customer_address')):
                fail(422, 'IDS_REQUIRED', 'Use customer_id from the resolution endpoint.')
            if any(k in values for k in ('by', 'changed_by')):
                fail(422, 'SELF_REPORTED_IDENTITY_REJECTED',
                     'The person is the authenticated key; by/changed_by are not accepted.')
        return values


class OrderLine(wt.IdInput):
    product_id: PositiveId
    quantity: Optional[confloat(gt=0)] = None
    unit: Optional[Literal['lb', 'cases', 'bags', 'boxes', 'each']] = None
    case_weight_lb: Optional[confloat(gt=0)] = None
    quantity_lb: Optional[confloat(ge=0)] = None
    unit_price: Optional[confloat(ge=0)] = None
    amount: Optional[confloat(ge=0)] = None
    notes: Optional[str] = None
    notes_es: Optional[str] = None

    @root_validator(skip_on_failure=True)
    def quantity_required(cls, values):
        if values.get('quantity') is None and values.get('quantity_lb') is None:
            fail(422, 'QUANTITY_REQUIRED', 'Give quantity (with unit) or quantity_lb for every line.')
        return values


class CreateOrderRequest(OrderRequest):
    customer_id: PositiveId
    customer_po: Optional[str] = None
    external_order_reference: Optional[str] = None
    allow_duplicate_po: bool = False
    requested_ship_date: Optional[date] = None
    order_date: Optional[date] = None
    notes: Optional[str] = None
    notes_es: Optional[str] = None
    lines: List[OrderLine] = Field(min_length=1)

    _text = validator('customer_po', 'external_order_reference', allow_reuse=True)(_trim)


class AddLinesRequest(OrderRequest):
    lines: List[OrderLine] = Field(min_length=1)


class UpdateLineRequest(OrderRequest):
    quantity_lb: Optional[confloat(gt=0)] = None
    unit_price: Optional[confloat(ge=0)] = None


class CancelLineRequest(OrderRequest):
    pass


class HeaderRequest(OrderRequest):
    requested_ship_date: Optional[date] = None
    notes: Optional[str] = None
    notes_es: Optional[str] = None
    customer_id: Optional[PositiveId] = None
    customer_po: Optional[str] = None
    allow_duplicate_po: bool = False

    _text = validator('customer_po', allow_reuse=True)(_trim)


class StatusRequest(OrderRequest):
    status: str = Field(min_length=1)


class ReadyRequest(OrderRequest):
    ready: bool = True
    note: Optional[str] = None


class CancelOrderRequest(OrderRequest):
    reason: Literal['customer_cancelled', 'cns_declined', 'duplicate', 'superseded', 'other']
    note: Optional[str] = None
    related_so_id: Optional[PositiveId] = None


class CloseOrderRequest(OrderRequest):
    reason: Literal['shipped_recorded', 'shipped_not_recorded', 'short_closed']
    note: Optional[str] = None
    related_so_id: Optional[PositiveId] = None


class ReopenRequest(OrderRequest):
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------
def load_order(cur, order_id, *, lock=False):
    cur.execute('''SELECT so.id, so.order_number, so.status, so.state, so.state_reason, so.customer_id,
                          c.name AS customer, so.customer_po, so.requested_ship_date, so.order_date,
                          so.external_order_reference, so.status_before_exit, so.ticket_id,
                          COALESCE(f.ready, false) AS ready
                   FROM sales_orders so JOIN customers c ON c.id=so.customer_id
                   LEFT JOIN sales_order_flags f ON f.so_number=so.order_number
                   WHERE so.id=%s''' + (' FOR NO KEY UPDATE OF so' if lock else ''), (order_id,))
    row = cur.fetchone()
    if not row:
        fail(404, 'ORDER_NOT_FOUND', f'Order #{order_id} not found.')
    return dict(row)


def order_lines(cur, order_id):
    cur.execute('''SELECT id, product_id, quantity_lb, quantity_shipped_lb, line_status, unit_price,
                          ordered_quantity, ordered_unit, amount
                   FROM sales_order_lines WHERE sales_order_id=%s ORDER BY id''', (order_id,))
    return [dict(row) for row in cur.fetchall()]


def customer(cur, customer_id, *, lock=False):
    cur.execute('SELECT id, name, active FROM customers WHERE id=%s' + (' FOR SHARE' if lock else ''),
                (customer_id,))
    row = cur.fetchone()
    if not row:
        fail(422, 'CUSTOMER_NOT_FOUND', 'Customer is missing; resolve an active customer.')
    if row['active'] is False:
        fail(422, 'CUSTOMER_INACTIVE', f'{row["name"]} is inactive; resolve an active customer.')
    return dict(row)


def reference_order(cur, reference):
    cur.execute('''SELECT id, order_number FROM sales_orders WHERE external_order_reference=%s
                   UNION ALL
                   SELECT so.id, so.order_number FROM sales_order_create_receipts r
                   JOIN sales_orders so ON so.id=r.order_id
                   WHERE r.external_order_reference=%s
                   LIMIT 1''', (reference, reference))
    row = cur.fetchone()
    return dict(row) if row else None


def create_intent(payload):
    # The original immutable request, independent of entry day and current order/customer fields.
    return wt.canonical_hash({k: v for k, v in payload.items() if k not in ('occurred_at', 'allow_duplicate_po')})


def reference_receipt(cur, reference, payload):
    cur.execute('''SELECT id, payload, response, result_ref FROM write_tickets
                   WHERE action='create_order' AND status='committed' AND receipt_number IS NOT NULL
                     AND COALESCE(payload->>'external_order_reference', receipt_number)=%s''', (reference,))
    original = cur.fetchone()
    if original and create_intent(original['payload']) != create_intent(payload):
        fail(409, 'EXTERNAL_ORDER_REFERENCE_CONFLICT',
             'This reference belongs to a different create request; retry the original request.')
    return dict(original) if original else None


def order_snapshot(order, lines):
    keys = ('id', 'status', 'state', 'state_reason', 'customer_id', 'customer_po', 'requested_ship_date',
            'external_order_reference', 'status_before_exit', 'ready')
    return {'order': {k: order[k] for k in keys},
            'lines': [{k: line[k] for k in ('id', 'product_id', 'quantity_lb', 'quantity_shipped_lb',
                                             'line_status', 'unit_price')} for line in lines]}


def core_lines(payload_lines, products):
    return [{'product_id': line['product_id'], 'product_name': products[line['product_id']]['name'],
             'quantity': line.get('quantity'), 'unit': line.get('unit'),
             'case_weight_lb': line.get('case_weight_lb'), 'quantity_lb': line.get('quantity_lb'),
             'unit_price': line.get('unit_price'), 'amount': line.get('amount'),
             'notes': line.get('notes'), 'notes_es': line.get('notes_es')} for line in payload_lines]


# ---------------------------------------------------------------------------
# Permissions that depend on the payload (the matrix row itself is checked by
# the ticket layer on the stored action, at prepare and again at commit).
# ---------------------------------------------------------------------------
def require_reason_permissions(api, cur, action, payload, actor):
    if action == 'close_order' and payload['reason'] == 'shipped_not_recorded':
        permissions.require('close_order_shipped_not_recorded', actor)
    if action == 'update_order_status':
        if payload['status'] == 'cancelled':
            permissions.require('cancel_order', actor)
        elif payload['status'] == 'invoiced':
            permissions.require('close_order', actor)
            try:
                order = api._load_so_for_state_change(cur, payload['order_id'])
            except HTTPException:
                return   # validate() reports the missing order as a blocker
            if order['state'] == 'open' and api._shipped_close_reason(order) == 'shipped_not_recorded':
                permissions.require('close_order_shipped_not_recorded', actor)


# ---------------------------------------------------------------------------
# Validate (ids exist, order state, the A6 gate) and post (the live cores).
# ---------------------------------------------------------------------------
def validate(api, cur, action, payload, *, lock=False):
    """Pre-checks the cores do not make, plus the state snapshot. Returns context."""
    if action in er_tickets.ACTIONS:
        return er_tickets.validate(api, cur, action, payload, lock=lock)
    ctx = {'blockers': []}
    if action == 'create_order':
        reference = payload.get('external_order_reference')
        if reference:
            if lock:
                api._lock_order_reference(cur, None, reference)
            original = reference_receipt(cur, reference, payload)
            if original:
                return {'blockers': [], 'replay': original, 'state': {'original_ticket_id': original['id']}}
        cust = customer(cur, payload['customer_id'], lock=lock)
        api.validate_bilingual(payload.get('notes'), payload.get('notes_es'), 'notes')
        products = {}
        for line in payload['lines']:
            api.validate_bilingual(line.get('notes'), line.get('notes_es'), 'notes')
            products[line['product_id']] = ticket_actions.product(api, cur, line['product_id'], lock)
        reference = payload.get('external_order_reference')
        if reference:
            if lock:
                api._lock_order_reference(cur, cust['id'], reference)
            existing = reference_order(cur, reference)
            if existing:
                fail(409, 'EXTERNAL_ORDER_REFERENCE_EXISTS',
                     f"Order {existing['order_number']} already carries reference '{reference}' for "
                     f"{cust['name']}; look it up instead of creating it again.",
                     order_id=existing['id'], order_number=existing['order_number'])
        duplicates = (api._dedupe_existing_sales_orders(cur, cust['id'], payload['customer_po'])
                      if payload.get('customer_po') else [])
        ctx.update(customer=cust, products=products, duplicates=duplicates)
        ctx['state'] = {'customer': cust, 'duplicate_order_ids': [d['order_id'] for d in duplicates],
                        'products': {pid: {'active': p['active'], 'is_service': p.get('is_service'),
                                           'case_size_lb': p.get('case_size_lb')} for pid, p in products.items()}}
        return ctx
    order = load_order(cur, payload['order_id'], lock=lock)
    lines = order_lines(cur, order['id'])
    ctx.update(order=order, lines=lines, state=order_snapshot(order, lines))
    if action == 'add_order_lines':
        api._require_open_state(order['state'], order['order_number'], order['id'], 'adding lines')
        if order['status'] in ('shipped', 'invoiced', 'cancelled'):
            fail(409, 'ORDER_LINES_LOCKED', f"Cannot add lines to {order['status']} order {order['order_number']}.")
        products = {}
        for line in payload['lines']:
            api.validate_bilingual(line.get('notes'), line.get('notes_es'), 'notes')
            products[line['product_id']] = ticket_actions.product(api, cur, line['product_id'], lock)
        ctx['products'] = products
        ctx['state']['products'] = {pid: {'active': p['active'], 'is_service': p.get('is_service'),
                                          'case_size_lb': p.get('case_size_lb')} for pid, p in products.items()}
    elif action in ('update_order_line', 'cancel_order_line'):
        line = next((l for l in lines if l['id'] == payload['line_id']), None)
        if line is None:
            fail(404, 'LINE_NOT_FOUND', f"Line #{payload['line_id']} is not on order {order['order_number']}.")
        if line['line_status'] in ('fulfilled', 'cancelled'):
            fail(409, 'LINE_NOT_EDITABLE', f"Line #{line['id']} is already {line['line_status']}.")
        ctx['line'] = line
    elif action == 'update_order_header':
        fields = payload['fields']
        api.validate_bilingual(payload.get('notes'), payload.get('notes_es'), 'notes')
        customer_id = order['customer_id']
        if 'customer_id' in fields and payload.get('customer_id') is not None:
            customer_id = customer(cur, payload['customer_id'], lock=lock)['id']
        customer_po = payload['customer_po'] if 'customer_po' in fields else order['customer_po']
        duplicates = []
        if customer_po and ('customer_po' in fields or 'customer_id' in fields):
            duplicates = [d for d in api._dedupe_existing_sales_orders(cur, customer_id, customer_po)
                          if d['order_id'] != order['id']]
        ctx.update(duplicates=duplicates, resulting_customer_po=customer_po)
        ctx['state']['duplicate_order_ids'] = [d['order_id'] for d in duplicates]
    elif action == 'update_order_status':
        api._validate_requested_order_status(payload['status'])
        if payload['status'] == 'invoiced':
            state_row = api._load_so_for_state_change(cur, order['id'])
            if state_row['state'] == 'open':
                ctx['blockers'] = shipment_gate(api, cur, order, api._shipped_close_reason(state_row))
    elif action == 'mark_order_ready':
        if order['state'] != 'open':
            fail(409, 'ORDER_NOT_OPEN', f"Order {order['order_number']} is '{order['state']}'; "
                 'ready can only be set on an open order.')
    elif action == 'close_order' and payload['reason'] == 'shipped_recorded':
        ctx['blockers'] = shipment_gate(api, cur, order, payload['reason'])
    return ctx


def post(api, cur, action, payload, request, ctx, *, ticket_id, receipt, allow_duplicate_po):
    """Run the live core for real. ticket_id/receipt are None on the prepare dry run."""
    if action in er_tickets.ACTIONS:
        return er_tickets.post(api, cur, action, payload, request, ctx, ticket_id=ticket_id,
                               allow_duplicate=allow_duplicate_po)
    if ctx.get('replay'):
        return {**ctx['replay']['response'], 'replayed': True}
    order_id = payload.get('order_id')
    if action == 'create_order':
        cust = ctx['customer']
        api._check_order_po(cur, cust['id'], payload.get('customer_po'), allow_duplicate_po)
        # The ticket is the external reference for chat-created orders (design §1.3)
        # unless the client brought its own.
        reference = payload.get('external_order_reference') or receipt
        result = api._create_sales_order_core(
            cur, cust['id'], cust['name'], payload.get('requested_ship_date'),
            payload.get('notes'), payload.get('notes_es'), core_lines(payload['lines'], ctx['products']),
            order_date=payload.get('order_date'), customer_po=payload.get('customer_po'),
            external_order_reference=reference, save_contract=True, request=request)
        if ticket_id is not None:
            cur.execute('UPDATE sales_orders SET ticket_id=%s WHERE id=%s', (ticket_id, result['order_id']))
            cur.execute('UPDATE sales_order_lines SET ticket_id=%s WHERE sales_order_id=%s',
                        (ticket_id, result['order_id']))
        amounts = [line['amount'] for line in result['line_results']]
        return {'order_id': result['order_id'], 'order_number': result['order_number'],
                'customer': cust['name'], 'customer_id': cust['id'],
                'requested_ship_date': payload.get('requested_ship_date'), 'status': 'confirmed',
                'total_lb': result['total_lb'], 'lines': result['line_results'],
                'warnings': result['warnings'] or None,
                'customer_po': payload.get('customer_po'),
                'customer_po_status': 'No PO' if payload.get('customer_po') is None else 'PO provided',
                'external_order_reference': reference,
                'total': (float(sum(Decimal(str(a)) for a in amounts))
                          if amounts and all(a is not None for a in amounts) else None),
                'message': f"Order {result['order_number']} created with {len(result['line_results'])} line(s)"}
    order = ctx['order']
    if action == 'add_order_lines':
        api._lock_sales_order(cur, order_id)
        line_results, total_lb, warnings = api._insert_order_lines_core(
            cur, request, order_id, order['customer_id'], order['customer'],
            core_lines(payload['lines'], ctx['products']), save_contract=True)
        if ticket_id is not None:
            cur.execute('UPDATE sales_order_lines SET ticket_id=%s WHERE id=ANY(%s)',
                        (ticket_id, [line['line_id'] for line in line_results]))
        return {'order_id': order_id, 'order_number': order['order_number'], 'lines_added': line_results,
                'total_lb_added': total_lb, 'warnings': warnings or None,
                'message': f"Added {len(line_results)} line(s) to {order['order_number']}"}
    if action == 'update_order_line':
        return api._update_order_line_core(cur, request, order_id, payload['line_id'],
                                           payload.get('quantity_lb'), payload.get('unit_price'))
    if action == 'cancel_order_line':
        return api._cancel_order_line_core(cur, request, order_id, payload['line_id'])
    if action == 'update_order_header':
        req = api.OrderHeaderUpdate(**{k: payload[k] for k in payload['fields']})
        req.allow_duplicate_po = allow_duplicate_po
        return api._update_order_header_core(cur, request, order_id, req)
    if action == 'update_order_status':
        return api._update_order_status_core(cur, request, order_id, api.OrderStatusUpdate(status=payload['status']))
    if action == 'mark_order_ready':
        req = api.SalesOrderReadyFlagRequest(ready=payload['ready'], by=None, note=payload.get('note'))
        return ticket_actions.response_dict(
            api._set_sales_order_ready_flag_core(cur, request, order['order_number'], req))
    if action == 'cancel_order':
        req = api.SalesOrderCancelRequest(reason=payload['reason'], note=payload.get('note'),
                                          related_so_id=payload.get('related_so_id'), mode='commit')
        return api._cancel_sales_order_core(cur, request, order_id, req)
    if action == 'close_order':
        req = api.SalesOrderCloseRequest(reason=payload['reason'], note=payload.get('note'),
                                         related_so_id=payload.get('related_so_id'), mode='commit')
        return api._close_sales_order_core(cur, request, order_id, req)
    if action == 'reopen_order':
        req = api.SalesOrderReopenRequest(note=payload.get('note'), mode='commit')
        return api._reopen_sales_order_core(cur, request, order_id, req)
    fail(409, 'TICKET_ACTION_UNAVAILABLE', 'Unsupported ticket action.')


# ---------------------------------------------------------------------------
# Draft and warnings
# ---------------------------------------------------------------------------
def _lb(value):
    return f'{float(value or 0):,.0f} lb'


def draft_from(action, payload, response, ctx):
    if action in er_tickets.ACTIONS:
        return er_tickets.draft_from(action, response)
    draft = dict(response)
    if ctx.get('replay'):
        draft['summary'] = f"Return original order {response['order_number']} / {response['receipt_number']}"
        draft['summary_es'] = f"Devolver pedido original {response['order_number']} / {response['receipt_number']}"
        return draft
    if action == 'create_order':
        cust = ctx['customer']
        draft.pop('order_id', None)
        draft['order_number_provisional'] = draft.pop('order_number', None)
        for line in draft.get('lines') or []:
            line.pop('line_id', None)
        po = payload.get('customer_po')
        n = len(payload['lines'])
        draft['summary'] = (f"Create order for {cust['name']}: {n} line(s), {_lb(draft.get('total_lb'))}, "
                            + (f"PO {po}" if po else 'no PO'))
        draft['summary_es'] = (f"Crear pedido para {cust['name']}: {n} línea(s), {_lb(draft.get('total_lb'))}, "
                               + (f"PO {po}" if po else 'sin PO'))
        draft['note'] = 'order_id, line ids and the order number are assigned at commit.'
        return draft
    order = ctx['order']
    draft['order'] = {k: order[k] for k in ('id', 'order_number', 'status', 'state', 'customer', 'customer_id',
                                             'customer_po', 'requested_ship_date')}
    label = f"{order['order_number']} ({order['customer']})"
    if action == 'add_order_lines':
        for line in draft.get('lines_added') or []:
            line.pop('line_id', None)
        n = len(payload['lines'])
        draft['summary'] = f"Add {n} line(s), {_lb(draft.get('total_lb_added'))}, to order {label}"
        draft['summary_es'] = f"Agregar {n} línea(s), {_lb(draft.get('total_lb_added'))}, al pedido {label}"
        draft['note'] = 'line ids are assigned at commit.'
    elif action == 'update_order_line':
        changes = []
        if payload.get('quantity_lb') is not None:
            changes.append(f"quantity {_lb(ctx['line']['quantity_lb'])} → {_lb(payload['quantity_lb'])}")
        if payload.get('unit_price') is not None:
            changes.append(f"unit price → {payload['unit_price']}")
        draft['summary'] = f"Update line #{payload['line_id']} on {label}: " + ', '.join(changes)
        draft['summary_es'] = f"Modificar línea #{payload['line_id']} de {label}: " + ', '.join(changes)
    elif action == 'cancel_order_line':
        draft['summary'] = f"Cancel line #{payload['line_id']} on {label}"
        draft['summary_es'] = f"Cancelar línea #{payload['line_id']} de {label}"
    elif action == 'update_order_header':
        fields = ', '.join(payload['fields'])
        draft['summary'] = f"Update {fields} on {label}"
        draft['summary_es'] = f"Modificar {fields} de {label}"
    elif action == 'update_order_status':
        draft['summary'] = f"Order {label}: {order['status']} → {payload['status']}"
        draft['summary_es'] = f"Pedido {label}: {order['status']} → {payload['status']}"
    elif action == 'mark_order_ready':
        draft['summary'] = f"Mark {label} {'ready' if payload['ready'] else 'not ready'} to ship"
        draft['summary_es'] = f"Marcar {label} como {'listo' if payload['ready'] else 'no listo'} para enviar"
    elif action == 'cancel_order':
        draft['summary'] = f"Cancel {label} ({payload['reason']})"
        draft['summary_es'] = f"Cancelar {label} ({payload['reason']})"
    elif action == 'close_order':
        draft['summary'] = f"Close {label} ({payload['reason']})"
        draft['summary_es'] = f"Cerrar {label} ({payload['reason']})"
    elif action == 'reopen_order':
        draft['summary'] = f"Reopen {label} as '{order['status_before_exit'] or 'confirmed'}'"
        draft['summary_es'] = f"Reabrir {label} como '{order['status_before_exit'] or 'confirmed'}'"
    return draft


def no_po_warning():
    return {'code': 'NO_PO', 'requires_ack': False,
            'message': "This order has no customer PO; it will be flagged 'No PO'.",
            'message_es': "Este pedido no tiene PO del cliente; quedará marcado 'No PO'.", 'refs': {}}


def duplicate_po_warning(customer_po, duplicates, allow_duplicate_po):
    numbers = ', '.join(d['order_number'] for d in duplicates)
    return {'code': 'DUPLICATE_CUSTOMER_PO', 'requires_ack': not allow_duplicate_po,
            'message': f"This customer already has order {numbers} with PO '{customer_po}'. "
                       + ('allow_duplicate_po is set; saving anyway.' if allow_duplicate_po else
                          'Acknowledge DUPLICATE_CUSTOMER_PO at commit (or set allow_duplicate_po) to save anyway.'),
            'message_es': f"Este cliente ya tiene el pedido {numbers} con PO '{customer_po}'. "
                          + ('allow_duplicate_po está activo; se guarda igual.' if allow_duplicate_po else
                             'Confirme DUPLICATE_CUSTOMER_PO al registrar (o active allow_duplicate_po) para guardarlo igual.'),
            'refs': {'existing_orders': duplicates}}


def canonical_lines(lines):
    # JSON strings give a total ordering even when the same product mixes
    # quantity/unit and quantity_lb. Preserve duplicate multiplicity and nulls.
    return sorted(json.dumps([l['product_id'], l.get('quantity'), l.get('unit'), l.get('quantity_lb')],
                             separators=(',', ':')) for l in lines)


def possible_duplicate(cur, action, payload, key_field):
    """A committed ticket for the same action, same customer/order and the same
    lines within 24 h, by any actor (design §1.5 — create/add lines only)."""
    cur.execute(f'''SELECT id, receipt_number, operator_id, result_ref,
                           payload->'lines' AS lines,
                           EXTRACT(EPOCH FROM (clock_timestamp()-committed_at))/60 AS minutes_ago
                    FROM write_tickets
                    WHERE action=%s AND status='committed' AND receipt_number IS NOT NULL
                      AND committed_at >= clock_timestamp()-interval '{DUPLICATE_WINDOW}'
                      AND (payload->>%s)::bigint=%s
                    ORDER BY committed_at DESC, id DESC''',
                (action, key_field, payload[key_field]))
    mine = canonical_lines(payload['lines'])
    for row in cur.fetchall():
        if canonical_lines(row['lines'] or []) != mine:
            continue
        minutes = max(0, int(row['minutes_ago']))
        ref = row['result_ref'] or {}
        return [{'code': 'POSSIBLE_DUPLICATE', 'requires_ack': True,
                 'message': f"The same lines were recorded {minutes} minutes ago as {row['receipt_number']} "
                            f"(order {ref.get('order_number')}) by {row['operator_id']}.",
                 'message_es': f"Las mismas líneas se registraron hace {minutes} minutos como {row['receipt_number']} "
                               f"(pedido {ref.get('order_number')}) por {row['operator_id']}.",
                 'refs': {'receipt_number': row['receipt_number'], 'ticket_id': row['id'],
                          'order_id': ref.get('order_id'), 'order_number': ref.get('order_number'),
                          'operator_id': row['operator_id'], 'minutes_ago': minutes}}]
    return []


def warnings_for(cur, action, payload, ctx):
    if action in er_tickets.ACTIONS:
        return er_tickets.warnings_for(cur, action, payload, ctx)
    if ctx.get('replay'):
        return []
    warnings = []
    if action == 'create_order':
        if payload.get('customer_po') is None:
            warnings.append(no_po_warning())
        if ctx['duplicates']:
            warnings.append(duplicate_po_warning(payload['customer_po'], ctx['duplicates'], payload['allow_duplicate_po']))
        warnings += possible_duplicate(cur, action, payload, 'customer_id')
    elif action == 'add_order_lines':
        warnings += possible_duplicate(cur, action, payload, 'order_id')
    elif action == 'update_order_header':
        if 'customer_po' in payload['fields'] and payload.get('customer_po') is None:
            warnings.append(no_po_warning())
        if ctx['duplicates']:
            warnings.append(duplicate_po_warning(ctx['resulting_customer_po'], ctx['duplicates'], payload['allow_duplicate_po']))
    return warnings


# ---------------------------------------------------------------------------
# Commit (called from write_tickets.commit_ticket after the shared checks) and
# the receipt read hook.
# ---------------------------------------------------------------------------
def commit(api, cur, row, body, actor, request, entry_timing):
    if body.lot_confirmations:
        fail(422, 'UNUSED_LOT_CONFIRMATION', 'This action does not consume ingredient lots.')
    if body.attachment_ref:
        fail(422, 'UNUSED_ATTACHMENT', 'Only corrections (adjust, found) take photo evidence.')
    action, payload = row['action'], row['payload']
    acknowledged = set(body.acknowledged_warnings)
    # Reason-dependent roles, re-checked now: a plain 403, the ticket stays prepared.
    require_reason_permissions(api, cur, action, payload, actor)
    cur.execute('SAVEPOINT ticket_post')
    try:
        if actor['id'] is not None:
            cur.execute('SELECT active FROM actors WHERE id=%s FOR SHARE', (actor['id'],))
            current = cur.fetchone()
            if not (current and current['active']):
                fail(403, 'ACTOR_INACTIVE', 'The preparing actor is no longer active.')
        unresolved = row['draft'].get('blockers', [])
        if unresolved:
            fail(409, 'DRAFT_BLOCKED', 'Prepare again after resolving the draft blockers.', blockers=unresolved)
        ctx = validate(api, cur, action, payload, lock=True)
        if ctx['blockers']:
            fail(409, ctx['blockers'][0]['code'], ctx['blockers'][0]['message'], blockers=ctx['blockers'])
        state_changed = wt.canonical_hash(wt.json_value(api, ctx['state'])) != row['state_hash']
        if ctx.get('replay'):
            original = ctx['replay']
            response = {**original['response'], 'replayed': True}
            # Alias ticket: keep the original receipt number uniquely attached
            # to its original ticket, and persist the identical replay result.
            cur.execute('''UPDATE write_tickets SET status='committed',committed_at=clock_timestamp(),
                           result_ref=%s,response=%s,acknowledged=%s WHERE id=%s''',
                        (Json(original['result_ref']), Json(response), Json(sorted(acknowledged)), row['id']))
            cur.execute('RELEASE SAVEPOINT ticket_post')
            return response
        business_date = datetime.fromisoformat(payload['occurred_at']).astimezone(api.PLANT_TIMEZONE).date()
        receipt = wt.allocate_receipt(cur, action, business_date)
        allow_po = (bool(payload.get('allow_duplicate_po') or payload.get('force'))
                    or bool({'DUPLICATE_CUSTOMER_PO', 'DUPLICATE_REFERENCE'} & acknowledged))
        response = post(api, cur, action, payload, request, ctx, ticket_id=row['id'], receipt=receipt,
                        allow_duplicate_po=allow_po)
        # The envelope middleware adds success=true on the wire; store it too so
        # the replayed and receipt-page copies equal the first HTTP answer.
        response.update(receipt_number=receipt, ticket_id=row['id'], replayed=False, state_changed=state_changed,
                        success=True)
        cur.execute('SELECT clock_timestamp() AS at')
        response['entry_timing'] = {**entry_timing, 'entered_at': cur.fetchone()['at'],
                                    'entered_by': actor, 'late_entry_exception_id': None}
        response = wt.json_value(api, response)
    except (HTTPException, psycopg2.errors.UniqueViolation) as exc:
        if isinstance(exc, HTTPException):
            if exc.status_code >= 500:
                raise
            errors = exc.detail.get('blockers') if isinstance(exc.detail, dict) else None
            errors = errors or [wt.blocker(exc)]
        elif exc.diag.constraint_name in REFERENCE_CONSTRAINTS:
            errors = [{'code': 'EXTERNAL_ORDER_REFERENCE_CONFLICT',
                       'message': 'An order already has this external_order_reference.'}]
        else:
            raise
        cur.execute('ROLLBACK TO SAVEPOINT ticket_post')
        cur.execute("UPDATE write_tickets SET status='rejected',reject_reason=%s WHERE id=%s",
                    (json.dumps(errors), row['id']))
        return api.JSONResponse(status_code=409, content={'detail': {
            'error_code': 'TICKET_STALE', 'message': 'Draft no longer valid; prepare again.',
            'blockers': errors}})
    cur.execute('RELEASE SAVEPOINT ticket_post')
    result_ref = {'order_id': response.get('order_id'), 'order_number': response.get('order_number'),
                  'line_ids': result_line_ids(action, payload, response), 'transaction_ids': [], 'lot_ids': []}
    if action in er_tickets.ACTIONS:
        result_ref = {'expected_receipt_ids': er_tickets.result_ids(response), 'transaction_ids': [], 'lot_ids': []}
    cur.execute('''UPDATE write_tickets SET status='committed',committed_at=clock_timestamp(),
        receipt_number=%s,result_ref=%s,response=%s,acknowledged=%s WHERE id=%s''',
        (receipt, Json(result_ref), Json(response), Json(sorted(acknowledged)), row['id']))
    return response


def result_line_ids(action, payload, response):
    if action == 'create_order':
        return [line['line_id'] for line in response.get('lines') or []]
    if action == 'add_order_lines':
        return [line['line_id'] for line in response.get('lines_added') or []]
    if action in ('update_order_line', 'cancel_order_line'):
        return [payload['line_id']]
    return []


def receipt_order(cur, ticket):
    """The order a committed order ticket created or edited, for GET /receipts/*."""
    if ticket['action'] not in ACTIONS:
        return None
    ref = ticket.get('result_ref') or {}
    order_id = ref.get('order_id') or ticket['payload'].get('order_id')
    if not order_id:
        return None
    cur.execute('''SELECT so.id, so.order_number, so.status, so.state, so.state_reason, so.customer_id,
                          c.name AS customer, so.customer_po, so.external_order_reference, so.ticket_id
                   FROM sales_orders so JOIN customers c ON c.id=so.customer_id WHERE so.id=%s''', (order_id,))
    row = cur.fetchone()
    if not row:
        return {'id': order_id, 'line_ids': ref.get('line_ids', [])}
    return dict(row, line_ids=ref.get('line_ids', []))


# ---------------------------------------------------------------------------
# Prepare routes
# ---------------------------------------------------------------------------
def register_routes(app, api):
    def prepare(action, req, request, **ids):
        actor = wt.identity(api, request)
        permissions.require(action, actor)   # matrix row on the ticket action, before any row exists
        payload = json.loads(req.json(exclude={'client_source'}))
        if action in ('update_order_header', 'update_expected_receipt'):
            payload['fields'] = sorted(req.__fields_set__ - {'client_source'})
            if not payload['fields']:
                fail(422, 'NO_FIELDS_TO_UPDATE', 'Give at least one header field to change.')
        payload.update(ids)
        # Orders carry an order_date, not an event time: happened_at is the plant
        # DAY the entry is made (never back-dated), so two identical prepares on
        # the same day hash alike and the earlier ticket is superseded.
        now = api.get_plant_now()
        payload['occurred_at'] = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        entry_timing = permissions.timing(now, now)
        with api.get_transaction() as cur:
            require_reason_permissions(api, cur, action, payload, actor)
            blockers, draft, state, warnings = [], {}, {}, []
            # Dry run: the live core builds the draft, then everything it wrote is undone.
            cur.execute('SAVEPOINT order_draft')
            try:
                ctx = validate(api, cur, action, payload)
                response = post(api, cur, action, payload, request, ctx, ticket_id=None, receipt=None,
                                allow_duplicate_po=True)
                draft = draft_from(action, payload, wt.json_value(api, response), ctx)
                blockers, state = ctx['blockers'], ctx['state']
                warnings = warnings_for(cur, action, payload, ctx)
            except HTTPException as exc:
                if exc.status_code >= 500:
                    raise
                blockers = [wt.blocker(exc)]
            finally:
                cur.execute('ROLLBACK TO SAVEPOINT order_draft')
            draft['entry_timing'] = entry_timing
            return wt.issue_ticket(api, cur, actor=actor, action=action, payload=payload, draft=draft,
                                   state=state, warnings=warnings, blockers=blockers,
                                   client_source=req.client_source, event_time=now)

    er_tickets.register_routes(app, api, prepare)

    # verify_api_key is listed first so an unknown key is refused before the
    # order id is even resolved.
    @app.post('/sales/orders/prepare')
    def prepare_create_order(req: CreateOrderRequest, request: Request, _: bool = Depends(api.verify_api_key)):
        return prepare('create_order', req, request)

    @app.post('/sales/orders/{order_id}/lines/prepare')
    def prepare_add_order_lines(req: AddLinesRequest, request: Request, _: bool = Depends(api.verify_api_key),
                                order_id: int = Depends(api.resolve_order_id)):
        return prepare('add_order_lines', req, request, order_id=order_id)

    @app.post('/sales/orders/{order_id}/lines/{line_id}/update/prepare')
    def prepare_update_order_line(req: UpdateLineRequest, request: Request, line_id: int,
                                  _: bool = Depends(api.verify_api_key),
                                  order_id: int = Depends(api.resolve_order_id)):
        return prepare('update_order_line', req, request, order_id=order_id, line_id=line_id)

    @app.post('/sales/orders/{order_id}/lines/{line_id}/cancel/prepare')
    def prepare_cancel_order_line(req: CancelLineRequest, request: Request, line_id: int,
                                  _: bool = Depends(api.verify_api_key),
                                  order_id: int = Depends(api.resolve_order_id)):
        return prepare('cancel_order_line', req, request, order_id=order_id, line_id=line_id)

    @app.post('/sales/orders/{order_id}/header/prepare')
    def prepare_update_order_header(req: HeaderRequest, request: Request, _: bool = Depends(api.verify_api_key),
                                    order_id: int = Depends(api.resolve_order_id)):
        return prepare('update_order_header', req, request, order_id=order_id)

    @app.post('/sales/orders/{order_id}/status/prepare')
    def prepare_update_order_status(req: StatusRequest, request: Request, _: bool = Depends(api.verify_api_key),
                                    order_id: int = Depends(api.resolve_order_id)):
        return prepare('update_order_status', req, request, order_id=order_id)

    @app.post('/sales/orders/{order_id}/ready/prepare')
    def prepare_mark_order_ready(req: ReadyRequest, request: Request, _: bool = Depends(api.verify_api_key),
                                 order_id: int = Depends(api.resolve_order_id)):
        return prepare('mark_order_ready', req, request, order_id=order_id)

    @app.post('/sales/orders/{order_id}/cancel/prepare')
    def prepare_cancel_order(req: CancelOrderRequest, request: Request, _: bool = Depends(api.verify_api_key),
                             order_id: int = Depends(api.resolve_order_id)):
        return prepare('cancel_order', req, request, order_id=order_id)

    @app.post('/sales/orders/{order_id}/close/prepare')
    def prepare_close_order(req: CloseOrderRequest, request: Request, _: bool = Depends(api.verify_api_key),
                            order_id: int = Depends(api.resolve_order_id)):
        return prepare('close_order', req, request, order_id=order_id)

    @app.post('/sales/orders/{order_id}/reopen/prepare')
    def prepare_reopen_order(req: ReopenRequest, request: Request, _: bool = Depends(api.verify_api_key),
                             order_id: int = Depends(api.resolve_order_id)):
        return prepare('reopen_order', req, request, order_id=order_id)
