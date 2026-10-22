"""FL-issued write tickets and receipt reads (A1: receive, make, pack, adjust and found;
A7 order tickets register through the hooks marked "A7 hook").

Route handlers own one database transaction. Action helpers receive its cursor;
no nested HTTP calls, global connection overrides, or independent commits.
"""
import hashlib
import json
import secrets

import permissions
import ticket_actions as actions
import lot_confirmation as a5
import exceptions_enforcement as a3b
from datetime import date, datetime
from typing import List, Literal, Optional

from fastapi import Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, StrictBool, conint, confloat, constr, field_validator, root_validator
from psycopg2.extras import Json

ClientSource = Literal['mcp', 'dashboard', 'fl_assistant', 'api']
PositiveId = conint(strict=True, gt=0)
# A7 hook: order tickets (metadata writes) — logic in order_tickets.py; the
# registry (actions, receipt prefix, routes) lives here so the ticket lifecycle
# is defined in one place. 'ORD', not the design's 'SO': order NUMBERS already
# use SO-YYMMDD-NNN and a receipt box that accepts both must not collide.
ORDER_ACTIONS = ('create_order', 'add_order_lines', 'update_order_line', 'cancel_order_line',
                 'update_order_header', 'update_order_status', 'mark_order_ready',
                 'cancel_order', 'close_order', 'reopen_order')
EXPECTED_RECEIPT_ACTIONS = ('create_expected_receipt', 'update_expected_receipt', 'cancel_expected_receipt')
EXPECTED_RECEIPT_PREPARE_ROUTES = frozenset({
    ('POST', '/expected-receipts/prepare'),
    ('POST', '/expected-receipts/{expected_receipt_id}/update/prepare'),
    ('POST', '/expected-receipts/{expected_receipt_id}/cancel/prepare'),
    ('POST', '/expected-receipts/extract/approve/prepare'),
})
SHIP_ACTIONS = ('ship_order', 'ship_standalone', 'ship_bulk')
PREFIXES = {'create_customer': 'CUS', 'resolve_exception': 'XR', 'void': 'VD', **{action: 'SHP' for action in SHIP_ACTIONS}, 'receive': 'RCV', 'make': 'MK', 'pack': 'PK', 'adjust': 'ADJ', 'found': 'FND', 'move_lot': 'LOT', 'update_supplier_lot': 'LOT',
            **{action: 'ORD' for action in ORDER_ACTIONS},
            **{action: 'ER' for action in EXPECTED_RECEIPT_ACTIONS}}
ORDER_PREPARE_ROUTES = frozenset({
    ('POST', '/sales/orders/extract/approve/prepare'),
    ('POST', '/sales/orders/prepare'),
    ('POST', '/sales/orders/{order_id}/lines/prepare'),
    ('POST', '/sales/orders/{order_id}/lines/{line_id}/update/prepare'),
    ('POST', '/sales/orders/{order_id}/lines/{line_id}/cancel/prepare'),
    ('POST', '/sales/orders/{order_id}/header/prepare'),
    ('POST', '/sales/orders/{order_id}/status/prepare'),
    ('POST', '/sales/orders/{order_id}/ready/prepare'),
    ('POST', '/sales/orders/{order_id}/cancel/prepare'),
    ('POST', '/sales/orders/{order_id}/close/prepare'),
    ('POST', '/sales/orders/{order_id}/reopen/prepare'),
})
# Public dashboard scope remains exactly the ticket routes granted by A1 part 1.
DASHBOARD_ROUTES = frozenset({
    ('POST', '/receive/prepare'),
    ('POST', '/tickets/{ticket}/commit'),
    ('GET', '/receipts'),
    ('GET', '/receipts/{receipt_number}'),
    ('GET', '/receipts/by-transaction/{transaction_id}'),
})
ACTOR_ROUTES = DASHBOARD_ROUTES | frozenset({
    ('POST', '/customers/prepare'),
    ('GET', '/sales/orders/{order_id}/office-options'),
    ('GET', '/ship/lots'),
    ('GET', '/purchase-documents/{document_id}'),
    ('POST', '/void/{transaction_id}/prepare'),
    ('POST', '/lots/{lot_id}/move/prepare'),
    ('POST', '/lots/{lot_id}/supplier-lot/prepare'),
    ('POST', '/make/prepare'),
    ('POST', '/pack/prepare'),
    ('POST', '/adjust/prepare'),
    ('POST', '/inventory/found/prepare'),
}) | frozenset({('POST', '/ship/prepare'), ('POST', '/ship/bulk/prepare'), ('POST', '/sales/orders/{order_id}/ship/prepare')}) | ORDER_PREPARE_ROUTES | EXPECTED_RECEIPT_PREPARE_ROUTES   # A7: actor + master keys only; the dashboard key gets nothing new


def fail(http_status, code, message, **extra):
    raise HTTPException(http_status, {'error_code': code, 'message': message, **extra})


class IdInput(BaseModel):
    @root_validator(pre=True)
    def attachment_alias(cls, values):
        if isinstance(values, dict) and values.get('attachment_id') is not None:
            if values.get('attachment_ref'):
                fail(422, 'AMBIGUOUS_ATTACHMENT', 'Supply attachment_id or attachment_ref.')
            values['attachment_ref'] = str(values.pop('attachment_id'))
        return values

    @root_validator(pre=True)
    def ids_only(cls, values):
        if isinstance(values, dict) and any(k in values for k in
                ('product_name', 'shipper_name', 'supplier_name', 'source_product', 'target_product')):
            fail(422, 'IDS_REQUIRED', 'Use product_id and supplier_id from the resolution endpoints.')
        return values

    class Config:
        extra = 'forbid'
        allow_inf_nan = False


class SupplierLotEntry(IdInput):
    supplier_id: Optional[PositiveId] = None
    supplier_lot_code: str
    quantity_lb: Optional[confloat(gt=0)] = None
    notes: Optional[str] = None


class ReceivePrepareRequest(IdInput):
    note: Optional[str] = Field(None, max_length=2000)
    product_id: PositiveId
    supplier_id: Optional[PositiveId] = None
    cases: conint(strict=True, gt=0)
    case_size_lb: confloat(gt=0)
    bol_reference: str
    attachment_ids: List[PositiveId] = Field(default_factory=list, max_length=5)
    shipper_code_override: Optional[str] = None
    lot_code: Optional[str] = None
    supplier_lot_code: Optional[str] = None
    no_supplier_lot: StrictBool = False
    lot_type: Optional[Literal['single_supplier', 'commingled']] = None
    supplier_lot_entries: Optional[List[SupplierLotEntry]] = None
    occurred_at: Optional[datetime] = None
    backfill: bool = False
    client_source: ClientSource = 'api'

    @root_validator(pre=True)
    def happened_alias(cls, values):
        values = dict(values)
        if 'happened_at' in values:
            if 'occurred_at' in values:
                fail(422, 'AMBIGUOUS_EVENT_TIME', 'Supply only happened_at or occurred_at.')
            values['occurred_at'] = values.pop('happened_at')
        if 'product_id' not in values:
            fail(422, 'IDS_REQUIRED', 'Use product_id from the resolution endpoints.')
        return values


class ActionPrepareRequest(IdInput):
    occurred_at: Optional[datetime] = None
    backfill: bool = False
    client_source: ClientSource = 'api'

    @root_validator(pre=True)
    def happened_alias(cls, values):
        values = dict(values)
        if 'happened_at' in values:
            if 'occurred_at' in values:
                fail(422, 'AMBIGUOUS_EVENT_TIME', 'Supply only happened_at or occurred_at.')
            values['occurred_at'] = values.pop('happened_at')
        return values


class IngredientLot(IdInput):
    ingredient_product_id: PositiveId
    lot_id: PositiveId


class IngredientLotAllocation(IngredientLot):
    quantity_lb: confloat(gt=0, allow_inf_nan=False)


class MovePrepareRequest(ActionPrepareRequest):
    to_location: Literal['storage', 'staging', 'production']
    method: Literal['last4', 'full_code', 'scan']
    value: str = Field(min_length=1, max_length=200)


class SupplierLotPrepareRequest(ActionPrepareRequest):
    supplier_lot_code: str
    reason_code: constr(strict=True, min_length=1, max_length=80)
    note: Optional[constr(strict=True, max_length=2000)] = None

    @field_validator('supplier_lot_code', mode='before')
    @classmethod
    def real_code(cls, value):
        import supplier_lot_tickets
        return supplier_lot_tickets.supplier_code(value)


class MakePrepareRequest(ActionPrepareRequest):
    substitutions: List[a5.Substitution] = Field(default_factory=list)
    reason_code: Optional[str] = None
    note: Optional[str] = None
    lot_confirmations: List[a5.LotConfirmation] = Field(default_factory=list)
    product_id: PositiveId
    batches: conint(strict=True, gt=0)
    lot_code: Optional[str] = None
    ingredient_lots: List[IngredientLot] = Field(default_factory=list)
    ingredient_lot_allocations: List[IngredientLotAllocation] = Field(default_factory=list)
    excluded_ingredients: List[PositiveId] = Field(default_factory=list)
    confirmed_sku: bool = False


class LotAllocation(IdInput):
    lot_id: PositiveId
    quantity_lb: confloat(gt=0)


class PackPrepareRequest(ActionPrepareRequest):
    note: Optional[str] = Field(None, max_length=2000)
    lot_confirmations: List[a5.LotConfirmation] = Field(default_factory=list)
    source_product_id: PositiveId
    target_product_id: PositiveId
    cases: conint(strict=True, gt=0)
    case_weight_lb: Optional[confloat(gt=0)] = None
    lot_allocations: Optional[List[LotAllocation]] = None
    ingredient_lot_allocations: List[IngredientLotAllocation] = Field(default_factory=list)
    target_lot_code: Optional[str] = None


class AdjustPrepareRequest(ActionPrepareRequest):
    lot_id: PositiveId
    delta_lb: float
    reason_code: str = Field(min_length=1)
    reason_es: Optional[str] = None
    # A3b: `note` is required for reason 'unknown'; `attachment_ref` is the photo
    # evidence a > 500 lb correction needs (a storage key/URL — see the A6 dependency).
    note: Optional[str] = Field(None, max_length=2000)
    attachment_ref: Optional[str] = Field(None, min_length=1, max_length=500)
    # Cutover reset (Michael, 2026-10-13): the lot's open SHORTAGE exceptions this
    # physical_count correction closes as 'counted' when it posts. Omitted = none.
    resolves_shortage_ids: List[PositiveId] = Field(default_factory=list)

    @root_validator(pre=True)
    def reason_alias(cls, values):
        values = dict(values)
        if 'reason' in values:
            if 'reason_code' in values:
                fail(422, 'AMBIGUOUS_REASON', 'Supply reason or reason_code, not both.')
            values['reason_code'] = values.pop('reason')
        if values.get('delta_lb') == 0:
            fail(422, 'INVALID_QUANTITY', 'Adjustment must be nonzero.')
        return values


class FoundPrepareRequest(ActionPrepareRequest):
    product_id: PositiveId
    quantity: confloat(gt=0)
    uom: Literal['lb'] = 'lb'
    reason_code: str = Field(min_length=1)
    lot_code: Optional[str] = None
    found_location: Optional[str] = None
    estimated_age: str = 'unknown'
    suspected_supplier: Optional[str] = None
    supplier_lot_code: Optional[str] = None
    notes: Optional[str] = None
    notes_es: Optional[str] = None
    attachment_ref: Optional[str] = Field(None, min_length=1, max_length=500)   # A3b photo evidence


class CommitRequest(BaseModel):
    attachment_id: Optional[PositiveId] = None
    attachment_ids: List[PositiveId] = Field(default_factory=list)

    @root_validator(pre=True)
    def attachment_alias(cls, values):
        if values.get('attachment_id') is not None:
            if values.get('attachment_ref'):
                fail(422, 'AMBIGUOUS_ATTACHMENT', 'Supply attachment_id or attachment_ref.')
            values['attachment_ref'] = str(values['attachment_id'])
        return values

    lot_confirmations: List[a5.LotConfirmation] = Field(default_factory=list)
    payload_hash: str
    acknowledged_warnings: List[str] = Field(default_factory=list)
    # A3b: a photo attached after prepare releases a > 500 lb hold without
    # changing the signed payload (same idea as late lot confirmations).
    attachment_ref: Optional[str] = Field(None, min_length=1, max_length=500)

    class Config:
        extra = 'forbid'


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def token_hash(ticket):
    return hashlib.sha256(ticket.encode()).hexdigest()


def identity(api, request):
    actor = api.request_actor(request)
    return {
        'id': actor['id'] if actor else None,
        'name': api._operator_id(request),
        'role': actor['role'] if actor else None,
        # Tickets bind to the person across actor-key/session sign-ins.
        'key_kind': 'actor' if request.state.key_kind == 'session' else request.state.key_kind,
    }


def blocker(exc):
    detail = exc.detail
    if isinstance(detail, dict):
        return {**detail, 'code': detail.get('error_code', 'VALIDATION_FAILED'),
                'message': detail.get('message', str(detail))}
    return {'code': 'VALIDATION_FAILED', 'message': str(detail)}


def json_value(api, value):
    # Match the existing Decimal/date response encoding before persisting it.
    return json.loads(json.dumps(value, cls=api.DecimalSafeEncoder, allow_nan=False))


def supplier(cur, supplier_id, *, lock=False):
    # A5 hook: use the same real-supplier eligibility as /resolve.
    return a5.real_supplier(cur, supplier_id, lock=lock)


def validate_receive(api, cur, payload, *, lock=False):
    """ID adapter plus shared legacy preview; return draft, snapshot and post input.

    The ticket pins the expected receipt and the displayed physical lot code.
    A closed expected receipt must not silently redirect to a different one.
    """
    occurred_at, source = api.validate_inventory_occurred_at(
        datetime.fromisoformat(payload['occurred_at']), payload['backfill'])
    cur.execute('SELECT * FROM products WHERE id=%s' + (' FOR SHARE' if lock else ''),
                (payload['product_id'],))
    product = cur.fetchone()
    if not product or product['active'] is False:
        fail(422, 'PRODUCT_NOT_FOUND', 'Product is missing or inactive; resolve an active product.')
    import kosher_tiers
    kosher_tiers.refuse_receive(product)
    shipper = supplier(cur, payload.get('supplier_id'), lock=lock)
    entries = []
    for entry in payload.get('supplier_lot_entries') or []:
        party = supplier(cur, entry.get('supplier_id') or shipper['id'], lock=lock)
        entries.append({k: v for k, v in entry.items() if k != 'supplier_id'} |
                       {'supplier_name': party['name'], 'supplier_id': party['id']})
    fields = {k: v for k, v in payload.items()
              if k not in ('product_id', 'supplier_id', 'expected_receipt_id', 'attachment_ids')}
    fields.update(product_name=product['name'], shipper_name=shipper['name'] if shipper else '',
                  supplier_lot_entries=entries or None, mode='preview')
    # A5 hook: labels come from the resolved supplier; labels never select IDs.
    fields['shipper_code_override'] = shipper['short_code']
    req = api.ReceiveRequest(**fields)
    draft = api._receive_preview_core(cur, req, product=product)
    import kosher_tiers
    draft.update(kosher_tiers.fields(product))
    draft.update(supplier_id=shipper['id'], supplier_name=shipper['name'])
    lots = []
    cur.execute('SELECT id, status, supplier_id FROM lots WHERE product_id=%s AND lot_code=%s' +
                (' FOR UPDATE' if lock else ''), (product['id'], draft['lot_code']))
    lot = cur.fetchone()
    if lot:
        if lot['status'] == 'merged':
            fail(409, 'LOT_MERGED', 'This lot was merged; prepare again with the surviving lot.')
        if lot['supplier_id'] is not None and lot['supplier_id'] != shipper['id']:
            fail(422, 'LOT_SUPPLIER_MISMATCH', 'This lot belongs to a different supplier; receive into a new lot.')
        lots.append({'id': lot['id'], 'status': lot['status'],
                     'on_hand_lb': api.lot_on_hand(cur, lot['id'])})
    api._validate_lot_code_twin(cur, product['id'], draft['lot_code'])
    er_id = payload.get('expected_receipt_id')
    if 'expected_receipt_id' not in payload:
        match = draft.get('expected_receipt_match')
        er_id = match['id'] if match else None
    expected = None
    if er_id is not None:
        cur.execute('SELECT id, status, product_id, supplier_id FROM expected_receipts WHERE id=%s' +
                    (' FOR UPDATE' if lock else ''), (er_id,))
        expected = cur.fetchone()
        if (not expected or expected['status'] != 'open' or
                expected['product_id'] != product['id'] or
                expected['supplier_id'] != payload.get('supplier_id')):
            fail(409, 'EXPECTED_RECEIPT_NOT_OPEN', 'The prepared expected receipt is no longer open or no longer matches.')
        match = api.fetch_expected_receipt(cur, er_id)
        draft['expected_receipt_match'] = {k: match[k] for k in
            ('id', 'expected_qty', 'remaining', 'expected_date', 'reference_number')}
    else:
        draft['expected_receipt_match'] = None
    # A5 hook: unresolved supplier lot identity is a flag, not a stock blocker.
    a5.identity_draft(api, cur, payload, draft)
    draft['supplier_lot_code'] = payload.get('supplier_lot_code')
    draft['note'] = payload.get('note')
    state = {'supplier': dict(shipper), 'product': {'id': product['id'], 'active': product['active']},
             'lots': lots, 'expected_receipt': dict(expected) if expected else None}
    return draft, state, req, product, occurred_at, source, er_id


def receive_duplicates(cur, payload):
    """Use effective posted ledger values, across all actors, by entry time."""
    cur.execute('''
        SELECT t.id, raw.receipt_number, t.operator_id,
               EXTRACT(EPOCH FROM (clock_timestamp()-t.created_at))/60 AS minutes_ago
        FROM ledger_current_transactions t JOIN transactions raw ON raw.id=t.id
        WHERE t.type='receive' AND t.effective_status='posted'
          AND t.created_at >= clock_timestamp()-interval '24 hours'
          AND EXISTS (SELECT 1 FROM ledger_current_transaction_lines l
                      WHERE l.transaction_id=t.id AND l.product_id=%s)
          AND ((SELECT SUM(l.quantity_lb) FROM ledger_current_transaction_lines l
                WHERE l.transaction_id=t.id AND l.product_id=%s)=%s
            OR (%s IS NOT NULL AND %s <> 'N/A' AND EXISTS (
                SELECT 1 FROM ledger_current_transaction_lines l JOIN lots lot ON lot.id=l.lot_id
                WHERE l.transaction_id=t.id AND l.product_id=%s AND lot.supplier_lot_code=%s)))
        ORDER BY t.created_at DESC, t.id DESC LIMIT 1
    ''', (payload['product_id'], payload['product_id'],
          payload['cases'] * payload['case_size_lb'], payload['supplier_lot_code'],
          payload['supplier_lot_code'], payload['product_id'], payload['supplier_lot_code']))
    row = cur.fetchone()
    if not row:
        return []
    minutes = max(0, int(row['minutes_ago']))
    ref = row['receipt_number'] or f"transaction {row['id']} (legacy; no receipt number)"
    return [{'code': 'POSSIBLE_DUPLICATE', 'requires_ack': True,
             'message': f"A matching receive was posted {minutes} minutes ago as {ref} by {row['operator_id']}.",
             'message_es': f"Una recepción similar se registró hace {minutes} {'minuto' if minutes == 1 else 'minutos'} como {ref} por {row['operator_id']}.",
             'refs': {'transaction_id': row['id'], 'receipt_number': row['receipt_number'],
                      'operator_id': row['operator_id'], 'minutes_ago': minutes}}]


def allocate_receipt(cur, action, business_date):
    prefix = PREFIXES[action]
    # The conflicting counter row is locked until the ticket transaction ends.
    cur.execute('''INSERT INTO receipt_counters(prefix,business_date,next) VALUES (%s,%s,2)
                   ON CONFLICT(prefix,business_date) DO UPDATE SET next=receipt_counters.next+1
                   RETURNING next-1 AS sequence''', (prefix, business_date))
    sequence = cur.fetchone()['sequence']
    return f'{prefix}-{business_date:%y%m%d}-{sequence:03d}'


def receipt_detail(api, cur, number, actor=None):
    cur.execute('SELECT * FROM write_tickets WHERE receipt_number=%s', (number,))
    ticket = cur.fetchone()
    if not ticket:
        fail(404, 'RECEIPT_NOT_FOUND', 'No recorded receipt has this number.')
    cur.execute('''SELECT t.id, t.type, t.effective_status FROM ledger_current_transactions t
                   JOIN transactions raw ON raw.id=t.id WHERE raw.ticket_id=%s ORDER BY t.id''',
                (ticket['id'],))
    transactions = [dict(row) for row in cur.fetchall()]
    lot_ids = set((ticket.get('result_ref') or {}).get('lot_ids', []))   # NULL on a rejected/held ticket that already drew a number
    for txn in transactions:
        # The same live authorization used by prepare/commit drives UI buttons.
        # Anonymous/shared-key readers never gain a named-person write control.
        txn['allowed_actions'] = []
        if actor and actor.get('id') is not None and txn['effective_status'] == 'posted':
            import resolution_receipts
            try:
                resolution_receipts.require_void(api, cur, actor, txn['id'])
                txn['allowed_actions'] = ['void']
            except HTTPException as error:
                if error.status_code >= 500:
                    raise
        cur.execute('''SELECT wt.receipt_number FROM ledger_corrections c
            LEFT JOIN write_tickets wt ON wt.action='void' AND wt.status='committed'
              AND wt.result_ref->>'correction_id'=c.id::text
            WHERE c.target_table='transactions' AND c.target_id=%s AND c.event_type IN ('void','restore')
            ORDER BY c.created_at DESC,c.id DESC LIMIT 1''', (txn['id'],))
        last_void = cur.fetchone()
        txn['void_receipt_number'] = last_void['receipt_number'] if last_void and txn['effective_status'] == 'voided' else None
        cur.execute('''SELECT l.product_id,p.name AS product_name,l.lot_id,lot.lot_code,l.quantity_lb
                       FROM ledger_current_transaction_lines l JOIN products p ON p.id=l.product_id
                       LEFT JOIN lots lot ON lot.id=l.lot_id WHERE l.transaction_id=%s ORDER BY l.id''',
                    (txn['id'],))
        txn['lines'] = [dict(row) for row in cur.fetchall()]
        lot_ids.update(line['lot_id'] for line in txn['lines'] if line['lot_id'] is not None)
    cur.execute('SELECT id,product_id,lot_code,status FROM lots WHERE id=ANY(%s) ORDER BY id',
                (sorted(lot_ids),))
    lots = [dict(row) for row in cur.fetchall()]
    for lot in lots:
        lot['on_hand_lb'] = api.lot_on_hand(cur, lot['id'])
    # A5 hook: explicit supplier and lot evidence in interface-neutral receipts.
    a5.receipt_evidence(cur, transactions, lots)
    import kosher_tiers
    kosher_tiers.receipt_evidence(cur, transactions, lots)
    happened = datetime.fromisoformat(ticket['payload']['occurred_at'])
    if ticket['action'] in ORDER_ACTIONS or ticket['action'] == 'create_customer':
        # Metadata has no physical event time: show when this entry was made.
        # Keep the day-normalized payload for same-day duplicate supersession.
        happened = ticket['committed_at'] or ticket['prepared_at']
    entered = ticket['committed_at']   # NULL on a rejected/held ticket that already drew a number
    late = entered is not None and happened.astimezone(api.PLANT_TIMEZONE).date() != entered.astimezone(api.PLANT_TIMEZONE).date()
    import order_tickets   # A7 hook: the order an order receipt created or edited (None otherwise)
    order = order_tickets.receipt_order(cur, ticket)
    import customer_tickets
    customer = customer_tickets.receipt_customer(cur, ticket)
    import expected_receipt_tickets
    expected = expected_receipt_tickets.receipt_records(api, cur, ticket)
    shipment = None
    if ticket['action'] in SHIP_ACTIONS:
        cur.execute('SELECT id,sales_order_id,bol_reference,proof_status,proof_due_at,entered_by_actor_id FROM shipments WHERE ticket_id=%s', (ticket['id'],))
        shipment = cur.fetchone()
    import attachments
    photos = attachments.for_receipt(cur, ticket['id'])
    draft, response = ticket['draft'], ticket['response']
    if ticket['action'] == 'create_order':
        # Earlier receipts omitted the order note from their saved display
        # snapshots. Recover the original note from the immutable payload,
        # never from the order's editable current notes or a stored rewrite.
        draft, response = dict(draft), dict(response or {})
        for key in ('notes', 'notes_es'):
            draft.setdefault(key, ticket['payload'].get(key))
            response.setdefault(key, ticket['payload'].get(key))
    receipt = {'attachments': photos, 'shipment': shipment, 'receipt_number': number, 'action': ticket['action'], 'status': ticket['status'],
            'order': order, 'customer': customer, 'expected_receipts': expected,
            'actor': ticket['draft']['actor'], 'operator_id': ticket['operator_id'],
            'client_source': ticket['client_source'], 'happened_at': happened,
            'entered_at': ticket['committed_at'], 'late_entry': late,
            'draft': draft, 'response': response,
            'warnings': ticket['warnings'], 'acknowledged': ticket['acknowledged'],
            'transactions': transactions, 'lots': lots}
    statuses = {t['effective_status'] for t in transactions}
    receipt['effective_status'] = ('voided' if statuses == {'voided'} else
                                   'partially_voided' if 'voided' in statuses else
                                   'posted' if statuses else ticket['status'])
    receipt['void_receipt_number'] = next((t['void_receipt_number'] for t in transactions if t['void_receipt_number']), None)
    receipt.update(kosher_tiers.summary_fields(receipt))
    return receipt


def receipt_numbers(cur, day, *, actor_terms=None, action=None, status=None, client_source=None, policy_code=None):
    """The receipts of one plant day: every ticket whose `happened_at` OR `entered_at`
    (= committed_at) plant day is `day`. `actor_terms` are matched against
    `operator_id` (name) or `actor_id` (as text) — the merged `?actor=` rule
    (`GET /receipts`, design §7.3); the A9 shift summary passes [name, id] so the
    two readers share one filter (contract §6.2)."""
    cur.execute('''SELECT receipt_number FROM write_tickets
        WHERE receipt_number IS NOT NULL
          AND (((payload->>'occurred_at')::timestamptz AT TIME ZONE 'America/New_York')::date=%s
               OR (committed_at AT TIME ZONE 'America/New_York')::date=%s)
          AND (%s::text[] IS NULL OR operator_id=ANY(%s) OR actor_id::text=ANY(%s))
          AND (%s IS NULL OR action=%s) AND (%s IS NULL OR status=%s)
          AND (%s IS NULL OR client_source=%s)
          AND (%s IS NULL OR EXISTS (SELECT 1 FROM jsonb_array_elements(
              COALESCE(response->'policy_acknowledgements','[]'::jsonb)) p WHERE p->>'code'=%s))
        ORDER BY committed_at,id''',
        (day, day, actor_terms, actor_terms, actor_terms, action, action, status, status, client_source, client_source,
         policy_code, policy_code))
    return [row['receipt_number'] for row in cur.fetchall()]


def receipt_summaries(api, cur, day, *, actor_terms=None, action=None, status=None,
                      client_source=None, policy_code=None, limit=500, offset=0):
    """A bounded day list without loading each receipt's live inventory/evidence.

    The full receipt remains the source of live void permissions and lot balances.
    Today's list needs only the stored entry plus effective transaction lines, so
    it takes two batched queries even when a shift has hundreds of entries.
    """
    cur.execute('''SELECT id, receipt_number, action, status, operator_id, draft, response,
                   (payload->>'occurred_at')::timestamptz AS happened_at, committed_at AS entered_at
        FROM write_tickets
        WHERE receipt_number IS NOT NULL
          AND (((payload->>'occurred_at')::timestamptz AT TIME ZONE 'America/New_York')::date=%s
               OR (committed_at AT TIME ZONE 'America/New_York')::date=%s)
          AND (%s::text[] IS NULL OR operator_id=ANY(%s) OR actor_id::text=ANY(%s))
          AND (%s IS NULL OR action=%s) AND (%s IS NULL OR status=%s)
          AND (%s IS NULL OR client_source=%s)
          AND (%s IS NULL OR EXISTS (SELECT 1 FROM jsonb_array_elements(
              COALESCE(response->'policy_acknowledgements','[]'::jsonb)) p WHERE p->>'code'=%s))
        ORDER BY COALESCE(committed_at,prepared_at) DESC, id DESC
        LIMIT %s OFFSET %s''',
        (day, day, actor_terms, actor_terms, actor_terms, action, action, status, status,
         client_source, client_source, policy_code, policy_code, limit + 1, offset))
    rows = [dict(row) for row in cur.fetchall()]
    has_more = len(rows) > limit
    rows = rows[:limit]
    by_ticket = {row['id']: row for row in rows}
    for row in rows:
        row['actor'] = (row['draft'] or {}).get('actor')
        row['transactions'] = []
        row['late_entry'] = bool(row['entered_at'] and row['happened_at'].astimezone(api.PLANT_TIMEZONE).date()
                                 != row['entered_at'].astimezone(api.PLANT_TIMEZONE).date())
    if rows:
        cur.execute('''SELECT raw.ticket_id, t.id, t.type, t.effective_status,
                       l.product_id, p.name AS product_name, p.uom,
                       l.lot_id, lot.lot_code, l.quantity_lb
            FROM transactions raw JOIN ledger_current_transactions t ON t.id=raw.id
            LEFT JOIN ledger_current_transaction_lines l ON l.transaction_id=t.id
            LEFT JOIN products p ON p.id=l.product_id
            LEFT JOIN lots lot ON lot.id=l.lot_id
            WHERE raw.ticket_id=ANY(%s) ORDER BY t.id, l.id''', (list(by_ticket),))
        transactions = {}
        for line in cur.fetchall():
            transaction = transactions.get(line['id'])
            if transaction is None:
                transaction = {key: line[key] for key in ('id', 'type', 'effective_status')}
                transaction['lines'] = []
                transactions[line['id']] = transaction
                by_ticket[line['ticket_id']]['transactions'].append(transaction)
            if line['product_id'] is not None:
                transaction['lines'].append({key: line[key] for key in
                    ('product_id', 'product_name', 'uom', 'lot_id', 'lot_code', 'quantity_lb')})
    for row in rows:
        row.pop('id')
        statuses = {txn['effective_status'] for txn in row['transactions']}
        row['effective_status'] = ('voided' if statuses == {'voided'} else
                                   'partially_voided' if 'voided' in statuses else
                                   'posted' if statuses else row['status'])
    return {'date': day, 'receipts': rows, 'has_more': has_more,
            'next_offset': offset + len(rows) if has_more else None}


def execute_commit(api, cur, row, actor, request, *, effective_payload, acknowledged, attachment_ref=None,
                   approval=None, preparer_inactive=False):
    """Re-validate under lock and post the stored payload; the one commit path.

    Shared by `POST /tickets/{ticket}/commit` and by A3b's owner approval of a
    held correction (`approval` set: the ticket's preparer is `actor`/`request`,
    the owner is recorded on the response, and a stale draft is reported without
    rejecting the ticket so the hold stays the owner's decision). Returns the
    posted response dict, or a JSONResponse for the terminal non-post outcomes
    (evidence 422, TICKET_STALE 409, A3b hold 202) — those must COMMIT.
    """
    entry_timing = permissions.require_backdating(
        approval['approved_by'] | {'key_kind': 'actor'} if approval else actor,
        datetime.fromisoformat(row['payload']['occurred_at']), api.get_plant_now())
    if row['action'] == 'adjust' and row['payload'].get('resolves_shortage_ids'):
        a3b.require_shortage_closer(actor)   # the preparer (owner) closes the shortages
    cur.execute('SAVEPOINT ticket_post')
    try:
        # Reuse the legacy lot-sequence locks before validation/posting.
        action_lock = {'receive': 1, 'found': 2, 'make': 3}.get(row['action'])
        if action_lock:
            cur.execute('SELECT pg_advisory_xact_lock(%s)', (action_lock,))
        if preparer_inactive:
            # Same locked actors row as the role check in the route.
            fail(403, 'ACTOR_INACTIVE', 'The preparing actor is no longer active.')
        if row['action'] == 'receive':
            draft, state, req, product, occurred_at, source, er_id = validate_receive(
                api, cur, row['payload'], lock=True)
        else:
            validated = actions.validate(api, cur, row['action'], effective_payload, lock=True)
            draft, state, _, _, occurred_at, source, _, _ = validated
        unresolved = [b for b in row['draft'].get('blockers', []) if b['code'] != 'LOT_NOT_CONFIRMED']
        if unresolved:
            fail(409, 'DRAFT_BLOCKED', 'Prepare again after resolving the draft blockers.',
                 blockers=unresolved)
        # A draft promising a new lot cannot silently add to a lot
        # created after prepare. This runs under the receive lock.
        if not row['draft'].get('lot_exists') and draft.get('lot_exists'):
            name = draft.get('product_name') or draft.get('target_product_name')
            code = draft.get('lot_code') or draft.get('output_lot_code')
            fail(409, 'LOT_CODE_TAKEN',
                 f'{name} lot {code} is now in use; prepare again for a fresh draft.')
        import kosher_tiers as a12
        if row['action'] in ('make', 'found') and draft.get('kosher_tier') == a12.EXTRA:
            approver = approval['approved_by'] if approval else actor
            # Found always needs a separate approval, even when the owner prepared it.
            if (row['action'] == 'found' and not approval) or not a12.owner(cur, approver):
                cur.execute('ROLLBACK TO SAVEPOINT ticket_post')
                return api.JSONResponse(status_code=202, content=json_value(api, a12.hold(
                    api, cur, row, actor, draft, acknowledged, effective_payload)))
            validated[3]['kosher_attestation'] = a12.attest(cur, approver)
        # A3b hook (R2, owner decision 7): > 500 lb corrections need a photo. With
        # one (prepared or attached now) they post, highlighted; without one the
        # ticket is HELD — nothing posts until the owner approves or a photo arrives.
        photo = attachment_ref or row['payload'].get('attachment_ref')
        if row['action'] in ('adjust', 'found'):
            if photo:
                import attachments
                evidence = attachments.verified(cur, photo, actor, ticket_id=row['id'], kinds=('correction_photo',), claim=True)
                attachments.load(api, evidence)
                photo = evidence['attachment_ref']
            draft['attachment_ref'] = photo
            if draft['correction_review']['photo_required'] and not photo and not approval:
                cur.execute('ROLLBACK TO SAVEPOINT ticket_post')
                if row['actor_id'] is None:
                    # A shared key has no named preparer for the owner to approve for:
                    # plain 422, nothing held, the ticket stays prepared until it expires.
                    return api.JSONResponse(status_code=422, content=json_value(api, a3b.photo_required(draft)))
                return api.JSONResponse(status_code=202, content=json_value(api, a3b.hold_ticket(
                    api, cur, row, actor, draft, acknowledged, row['payload_hash'])))
        state_changed = canonical_hash(json_value(api, state)) != row['state_hash']
        receipt = row['receipt_number'] or allocate_receipt(cur, row['action'], occurred_at.astimezone(api.PLANT_TIMEZONE).date())
        if row['action'] == 'receive':
            req.mode = 'commit'
            response = api._receive_commit_core(cur, req, request, occurred_at, source,
                product=product, ticket_id=row['id'], receipt_number=receipt, expected_receipt_id=er_id, supplier_id=row['payload']['supplier_id'], note=effective_payload.get('note'))
            a5.record_identity(cur, row['payload'], response, row['id'], receipt)
            a5.supplier_receipt(cur, response)
            import attachments
            response['attachments'] = attachments.attach_receive(api, cur,
                effective_payload.get('attachment_ids', []), actor,
                ticket_id=row['id'], transaction_id=response['transaction_id'])
        else:
            response = actions.post(api, cur, row['action'], validated, effective_payload, request,
                row['id'], receipt, require_new_lot=not row['draft'].get('lot_exists'))
        if row['action'] in ('adjust', 'found') and photo:
            cur.execute('UPDATE attachments SET transaction_id=%s WHERE attachment_ref=%s', (response.get('transaction_id'), photo))
            response['attachments'] = attachments.for_receipt(cur, row['id'])
        if row['action'] in ('receive', 'make', 'pack'):
            note = effective_payload.get('note')
            response['note'] = note
        response.update(receipt_number=receipt, ticket_id=row['id'], replayed=False,
                        state_changed=state_changed)
        # A2: 48 h–14 d late entries open exceptions(LATE_ENTRY) for the
        # owner to acknowledge (§6.3) in the same transaction as the post.
        cur.execute('SELECT COALESCE((SELECT created_at FROM transactions WHERE id=%s), clock_timestamp()) AS at',
                    (response.get('transaction_id'),))
        entered_at = cur.fetchone()['at']
        exception_id = None
        if permissions.opens_late_entry(actor, entry_timing):
            exception_id = permissions.open_late_entry(
                cur, identity=actor, info=entry_timing, action=row['action'],
                transaction_id=response.get('transaction_id'), receipt_number=receipt,
                ticket_id=row['id'], entered_at=entered_at,
                product_id=row['payload'].get('product_id') or row['payload'].get('target_product_id'),
                lot_id=response.get('lot_id') or response.get('output_lot_id'),
                client_source=row['client_source'])
        response['entry_timing'] = {**entry_timing, 'entered_at': entered_at,
                                    'entered_by': actor, 'late_entry_exception_id': exception_id}
        if approval:
            response['approval'] = {**approval, 'approved_at': entered_at}
        if approval or row['status'] == a3b.HELD:
            resolver = ({**approval['approved_by'], 'key_kind': 'actor'} if approval else actor)
            kind = 'approved' if approval else 'photo_attached'
            closed_id = a3b.release_hold(cur, row['id'], resolution_kind=kind, actor_id=resolver['id'],
                             note=approval.get('note') if approval else None, response=response, attachment_ref=photo)
            if closed_id:
                xr = a3b._record_resolution(api, cur, a3b._one(cur, closed_id), resolver, kind)
                response['resolution_receipt_number'] = xr['resolution_receipt_number']
        a12.snapshot_response(cur, response, row['action'], draft)
        response = json_value(api, response)
    except HTTPException as exc:
        if exc.status_code >= 500:
            raise
        cur.execute('ROLLBACK TO SAVEPOINT ticket_post')
        errors = exc.detail.get('blockers') if isinstance(exc.detail, dict) else None
        errors = errors or [blocker(exc)]
        if errors[0]['code'] == 'ATTACHMENT_WRONG_ACTOR':
            return api.JSONResponse(status_code=403, content={'detail': exc.detail})
        if exc.status_code == 422 and errors[0]['code'] in ('LOT_NOT_CONFIRMED', 'LOT_CONFIRMATION_MISMATCH', 'AMBIGUOUS_SUFFIX', 'AMBIGUOUS_SUPPLIER_LOT', 'SUPPLIER_LOT_NOT_IN_STOCK', 'LOT_ALLOCATION_EXCEEDS_STOCK', 'PALLET_MOVE_REQUIRED', 'UNUSED_LOT_CONFIRMATION', 'DUPLICATE_LOT_CONFIRMATION', 'ATTACHMENT_NOT_FOUND', 'ATTACHMENT_WRONG_CONTEXT', 'ATTACHMENT_KIND_INVALID'):
            return api.JSONResponse(status_code=422, content={'detail': {
                **errors[0], 'error_code': errors[0]['code'], 'message': errors[0]['message'], 'blockers': errors}})
        if not approval:
            # A3b: an approval that finds a stale draft leaves the hold for the owner
            # to reject explicitly; the plain commit path rejects as A1 always did.
            cur.execute("UPDATE write_tickets SET status='rejected',reject_reason=%s WHERE id=%s",
                        (json.dumps(errors), row['id']))
        return api.JSONResponse(status_code=409, content={'detail': {
            'error_code': 'TICKET_STALE', 'message': 'Draft no longer valid; prepare again.',
            'blockers': errors}})
    cur.execute('RELEASE SAVEPOINT ticket_post')
    cur.execute('SELECT DISTINCT lot_id FROM transaction_lines WHERE transaction_id=%s AND lot_id IS NOT NULL ORDER BY lot_id',
                (response.get('transaction_id'),))
    result_ref = {'transaction_ids': [response['transaction_id']] if response.get('transaction_id') else [],
                  'lot_ids': [line['lot_id'] for line in cur.fetchall()]}
    if row['action'] in ('move_lot', 'update_supplier_lot'):
        result_ref['lot_ids'] = [response['lot_id']]
    cur.execute('''UPDATE write_tickets SET status='committed',committed_at=clock_timestamp(),
        receipt_number=%s,result_ref=%s,response=%s,acknowledged=%s WHERE id=%s''',
        (receipt, Json(result_ref), Json(response), Json(sorted(set(acknowledged))), row['id']))
    return response


def issue_ticket(api, cur, *, actor, action, payload, draft, state, warnings, blockers,
                 client_source, event_time):
    """Insert the ticket row (supersession, hashes, expiry) and return the prepare
    envelope. A7 hook: shared by the ledger prepares below and order_tickets.
    `draft` gains actor/happened_at/happened_vs_now_minutes/blockers here."""
    payload_hash = canonical_hash(payload)
    happened_at = event_time.isoformat() if action in ORDER_ACTIONS or action == 'create_customer' else payload['occurred_at']
    draft.update(actor=actor, happened_at=happened_at,
                 happened_vs_now_minutes=round(permissions.elapsed(event_time, api.get_plant_now()).total_seconds()/60, 1),
                 blockers=blockers)
    draft = json_value(api, draft)
    # Serialize identical prepares, including the first one (no row yet).
    supersession = canonical_hash([actor['id'], actor['name'], actor['key_kind'], action, payload_hash])
    cur.execute('SELECT pg_advisory_xact_lock(%s)', (int(supersession[:15], 16),))
    cur.execute('''UPDATE write_tickets SET status='superseded'
                   WHERE status='prepared' AND action=%s AND payload_hash=%s
                     AND operator_id=%s AND key_kind=%s AND actor_id IS NOT DISTINCT FROM %s''',
                (action, payload_hash, actor['name'], actor['key_kind'], actor['id']))
    ticket = 'wt_' + secrets.token_urlsafe(32)
    ttl = 30 if client_source == 'dashboard' else 10
    cur.execute('''WITH clock AS (SELECT clock_timestamp() AS at)
        INSERT INTO write_tickets(ticket_hash,action,actor_id,operator_id,key_kind,client_source,
            payload,payload_hash,state_hash,draft,warnings,prepared_at,expires_at)
        SELECT %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,at,at+%s*interval '1 minute'
        FROM clock RETURNING id,expires_at''',
        (token_hash(ticket), action, actor['id'], actor['name'], actor['key_kind'], client_source,
         Json(payload), payload_hash, canonical_hash(json_value(api, state)), Json(draft), Json(warnings), ttl))
    row = cur.fetchone()
    return {'ticket': ticket, 'ticket_id': row['id'], 'action': action,
            'expires_at': row['expires_at'], 'payload_hash': payload_hash, 'draft': draft,
            'warnings': warnings, 'blockers': blockers, 'can_commit': not blockers, 'actor': actor}


def register_routes(app, api):
    def prepare(action, req, request, *, lot_id=None):
        actor = identity(api, request)
        # A2 hook: role matrix on the ticket action (not the route), then the
        # back-dating rule — both 403 before any ticket row exists.
        permissions.require(action, actor)
        if action == 'update_supplier_lot' and actor['id'] is None:
            fail(403, 'ACTOR_REQUIRED', 'Sign in as a person to correct a supplier lot.')
        if getattr(req, 'resolves_shortage_ids', None):
            a3b.require_shortage_closer(actor)
        payload = json.loads(req.json(exclude={'client_source'}))
        if action == 'adjust' and not payload.get('resolves_shortage_ids'):
            payload.pop('resolves_shortage_ids', None)   # unchanged payload/hash for ordinary adjusts
        if lot_id is not None:
            payload['lot_id'] = lot_id
        event_time = req.occurred_at or api.get_plant_now()
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=api.PLANT_TIMEZONE)
        payload['occurred_at'] = event_time.astimezone(api.PLANT_TIMEZONE).isoformat()
        entry_timing = permissions.require_backdating(actor, event_time, api.get_plant_now())
        for field in ('lot_code', 'target_lot_code'):
            if payload.get(field):
                payload[field] = api.normalize_lot_code_input(payload[field])
        if action == 'receive' and not payload.get('attachment_ids'):
            payload.pop('attachment_ids', None)  # preserve hashes for existing unpictured receives
        if action == 'receive':
            payload['supplier_lot_code'] = (req.supplier_lot_code or '').strip() or 'N/A'
            if not payload.get('no_supplier_lot'):
                payload.pop('no_supplier_lot', None)  # unchanged hashes for ordinary identified receipts
        with api.get_transaction() as cur:
            blockers = []
            draft = {}
            state = {}
            try:
                if action == 'receive':
                    draft, state, _, _, _, _, er_id = validate_receive(api, cur, payload)
                    payload['expected_receipt_id'] = er_id
                    payload['lot_code'] = draft['lot_code']
                    import attachments
                    draft['attachments'] = attachments.receive_evidence(api, cur,
                        payload.get('attachment_ids', []), actor)
                else:
                    draft, state, _, _, _, _, specification, input_plan = actions.validate(api, cur, action, payload)
                    payload['specification'] = specification
                    if draft.get('output_lot_id'):
                        payload['existing_output_lot_id'] = draft['output_lot_id']
                    if input_plan is not None:
                        payload['input_plan'] = input_plan
                    if action == 'pack':
                        payload['target_lot_code'] = draft['output_lot_code']
                        payload['case_weight_lb'] = draft['case_weight_lb']
                    elif action in ('make', 'found'):
                        payload['lot_code'] = draft['lot_code']
                    elif action in ('adjust', 'update_supplier_lot'):
                        payload['product_id'] = draft['product_id']
            except HTTPException as exc:
                if exc.status_code >= 500:
                    raise
                if action == 'update_supplier_lot' and isinstance(exc.detail, dict) and exc.detail.get('error_code') == 'SUPPLIER_LOT_CODE_INVALID':
                    raise
                # A5 §5.2: choose a real resolved supplier before issuing a receipt draft.
                if action == 'receive' and isinstance(exc.detail, dict) and exc.detail.get('error_code') in ('SUPPLIER_REQUIRED', 'SUPPLIER_LOT_REQUIRED'):
                    raise
                # A3b: a bad reason or a shortage cover-up is refused, not drafted.
                if isinstance(exc.detail, dict) and exc.detail.get('error_code') in a3b.PREPARE_REFUSALS:
                    raise
                blockers = [blocker(exc)]
            blockers = blockers or draft.get('blockers', [])
            warnings = (receive_duplicates(cur, payload) if action == 'receive' else
                        actions.duplicates(cur, action, payload, draft))
            late = permissions.late_entry_warning(actor, entry_timing)   # A2 hook
            if late:
                warnings.append(late)
            warnings.extend(a3b.shortage_warnings(draft))   # A3b hook (R3): warns, never blocks
            draft['entry_timing'] = entry_timing
            return issue_ticket(api, cur, actor=actor, action=action, payload=payload, draft=draft,
                                state=state, warnings=warnings, blockers=blockers,
                                client_source=req.client_source, event_time=event_time)

    @app.post('/receive/prepare')
    def prepare_receive(req: ReceivePrepareRequest, request: Request, _: bool = Depends(api.verify_api_key)):
        return prepare('receive', req, request)

    @app.post('/lots/{lot_id}/move/prepare')
    def prepare_move(lot_id: int, req: MovePrepareRequest, request: Request, _: bool = Depends(api.verify_api_key)):
        return prepare('move_lot', req, request, lot_id=lot_id)

    @app.post('/lots/{lot_id}/supplier-lot/prepare')
    def prepare_supplier_lot(lot_id: int, req: SupplierLotPrepareRequest, request: Request, _: bool = Depends(api.verify_api_key)):
        return prepare('update_supplier_lot', req, request, lot_id=lot_id)

    @app.post('/make/prepare')
    def prepare_make(req: MakePrepareRequest, request: Request, _: bool = Depends(api.verify_api_key)):
        return prepare('make', req, request)

    @app.post('/pack/prepare')
    def prepare_pack(req: PackPrepareRequest, request: Request, _: bool = Depends(api.verify_api_key)):
        return prepare('pack', req, request)

    @app.post('/adjust/prepare')
    def prepare_adjust(req: AdjustPrepareRequest, request: Request, _: bool = Depends(api.verify_api_key)):
        return prepare('adjust', req, request)

    @app.post('/inventory/found/prepare')
    def prepare_found(req: FoundPrepareRequest, request: Request, _: bool = Depends(api.verify_api_key)):
        return prepare('found', req, request)

    @app.post('/tickets/{ticket}/commit')
    def commit_ticket(ticket: str, body: CommitRequest, request: Request,
                      _: bool = Depends(api.verify_api_key)):
        actor = identity(api, request)
        with api.get_transaction() as cur:
            cur.execute('SELECT * FROM write_tickets WHERE ticket_hash=%s FOR UPDATE', (token_hash(ticket),))
            row = cur.fetchone()
            if not row:
                fail(404, 'TICKET_NOT_FOUND', 'Ticket not found.')
            if row['actor_id'] is not None:
                matches = actor['key_kind'] == 'actor' and row['actor_id'] == actor['id']
            else:
                matches = (actor['id'] is None and row['operator_id'] == actor['name'] and
                           row['key_kind'] == actor['key_kind'])
            if not matches:
                fail(403, 'TICKET_WRONG_USER', 'Only the preparing identity can commit this ticket.')
            if body.payload_hash != row['payload_hash'] or canonical_hash(row['payload']) != row['payload_hash']:
                fail(409, 'TICKET_PAYLOAD_MISMATCH', 'The payload hash does not match the prepared draft.')
            if row['status'] == 'committed':
                return {**row['response'], 'replayed': True}
            # A2 hook: enforce again at commit on the stored action, with the
            # role and active flag as they are NOW, read FOR SHARE so the row
            # cannot change between this check and the post (the lock is held
            # to the end of the transaction; a concurrent UPDATE actors waits).
            current_actor = None
            if actor['id'] is not None:
                cur.execute('SELECT role, active FROM actors WHERE id=%s FOR SHARE', (actor['id'],))
                current_actor = cur.fetchone()
                if current_actor:
                    actor['role'] = current_actor['role']
            permissions.require(row['action'], actor)
            entry_timing = permissions.require_backdating(
                actor, datetime.fromisoformat(row['payload']['occurred_at']), api.get_plant_now())
            # A3b: a held correction (awaiting_approval) is committable again — with a
            # photo it posts, without one it stays held; it never expires.
            if row['status'] not in ('prepared', a3b.HELD):
                fail(409, 'TICKET_NOT_COMMITTABLE', 'Prepare a new ticket.', status=row['status'])
            if row['status'] == a3b.HELD and (row['response'] or {}).get('error_code') == 'KOSHER_ATTESTATION_REQUIRED':
                cur.execute('SELECT clock_timestamp()>%s AS expired',(row['expires_at'],))
                if cur.fetchone()['expired']:
                    fail(409, 'TICKET_EXPIRED', 'The seven-day kosher hold expired; the owner must reject it and prepare again.')
                return api.JSONResponse(status_code=202,content=row['response'])
            if row['status'] == 'prepared':
                cur.execute('SELECT clock_timestamp() > %s AS expired', (row['expires_at'],))
                if cur.fetchone()['expired']:
                    cur.execute("UPDATE write_tickets SET status='expired' WHERE id=%s", (row['id'],))
                    # Return, rather than raise: the terminal state must COMMIT.
                    return api.JSONResponse(status_code=409, content={'detail': {
                        'error_code': 'TICKET_EXPIRED', 'message': 'Ticket expired; prepare again.'}})
            missing = sorted({w['code'] for w in row['warnings'] if w.get('requires_ack')} -
                             set(body.acknowledged_warnings))
            if missing:
                fail(409, 'WARNING_NOT_ACKNOWLEDGED', 'Acknowledge the draft warnings.',
                     message_es='Confirme las advertencias del borrador antes de registrar.', missing=missing)
            if row['action'] in ('resolve_exception', 'void'):
                import resolution_receipts
                return resolution_receipts.commit(api, cur, row, body, actor, request)
            if row['action'] in SHIP_ACTIONS:
                import ship_tickets
                return ship_tickets.commit(api, cur, row, body, actor, request, entry_timing)
            if body.attachment_ids and row['action'] != 'receive':
                fail(422, 'UNUSED_ATTACHMENT', 'Use attachment_id for a correction, attachment_ids for shipments or receive photos.')
            # A7 hook: order tickets post through their own module (metadata
            # writes, no ledger lines); every lifecycle check above is shared.
            if row['action'] == 'create_customer':
                import customer_tickets
                return customer_tickets.commit(api, cur, row, body, actor, request, entry_timing)
            if row['action'] in ORDER_ACTIONS + EXPECTED_RECEIPT_ACTIONS:
                import order_tickets
                return order_tickets.commit(api, cur, row, body, actor, request, entry_timing)
            if row['action'] not in PREFIXES:
                fail(409, 'TICKET_ACTION_UNAVAILABLE', 'Unsupported ticket action.')
            # A5 hook: late evidence never mutates the signed draft payload.
            if body.lot_confirmations and row['action'] not in ('make', 'pack'):
                fail(422, 'UNUSED_LOT_CONFIRMATION', 'This action does not consume ingredient lots.')
            if body.attachment_ref and row['action'] not in ('adjust', 'found'):
                fail(422, 'UNUSED_ATTACHMENT', 'Only corrections (adjust, found) take photo evidence.')
            effective_payload = a5.merge_confirmations(row['payload'], [e.dict() for e in body.lot_confirmations])
            if row['action'] == 'receive':
                effective_payload['attachment_ids'] = list(dict.fromkeys(
                    row['payload'].get('attachment_ids', []) + body.attachment_ids))
            return execute_commit(api, cur, row, actor, request, effective_payload=effective_payload,
                                  acknowledged=body.acknowledged_warnings, attachment_ref=body.attachment_ref,
                                  preparer_inactive=actor['id'] is not None and not (current_actor and current_actor['active']))

    @app.get('/receipts')
    def list_receipts(request: Request, date: Optional[date] = Query(None), actor: Optional[str] = None,
                      action: Optional[str] = None, status: Optional[str] = None,
                      client_source: Optional[ClientSource] = None,
                      policy_code: Optional[str] = None,
                      summary: bool = False, limit: int = Query(500, ge=1, le=500),
                      offset: int = Query(0, ge=0),
                      _: bool = Depends(api.verify_api_key)):
        day = date or api.get_plant_now().date()
        with api.get_transaction() as cur:
            if summary:
                return receipt_summaries(api, cur, day, actor_terms=[actor] if actor else None,
                                         action=action, status=status, client_source=client_source,
                                         policy_code=policy_code, limit=limit, offset=offset)
            numbers = receipt_numbers(cur, day, actor_terms=[actor] if actor else None, action=action, status=status,
                                      client_source=client_source, policy_code=policy_code)
            return {'date': day, 'receipts': [receipt_detail(api, cur, number, identity(api, request)) for number in numbers]}

    @app.get('/receipts/by-transaction/{transaction_id}')
    def receipt_by_transaction(transaction_id: int, request: Request, _: bool = Depends(api.verify_api_key)):
        with api.get_transaction() as cur:
            cur.execute('SELECT receipt_number FROM transactions WHERE id=%s', (transaction_id,))
            row = cur.fetchone()
            if not row or row['receipt_number'] is None:
                fail(404, 'RECEIPT_NOT_FOUND', 'No ticket receipt is recorded for this transaction.')
            return receipt_detail(api, cur, row['receipt_number'], identity(api, request))

    @app.get('/receipts/{receipt_number}')
    def get_receipt(receipt_number: str, request: Request, _: bool = Depends(api.verify_api_key)):
        with api.get_transaction() as cur:
            return receipt_detail(api, cur, receipt_number, identity(api, request))

    # A7 hook: the ten order prepares (ORDER_PREPARE_ROUTES) live in order_tickets.
    import order_tickets
    order_tickets.register_routes(app, api)
    import customer_tickets
    customer_tickets.register_routes(app, api)
    import resolution_receipts
    resolution_receipts.register_routes(app, api)
