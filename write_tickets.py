"""FL-issued write tickets and receipt reads (A1: receive, make, pack, adjust and found).

Route handlers own one database transaction. Action helpers receive its cursor;
no nested HTTP calls, global connection overrides, or independent commits.
"""
import hashlib
import json
import secrets

import ticket_actions as actions
from datetime import date, datetime
from typing import List, Literal, Optional

from fastapi import Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, conint, confloat, root_validator
from psycopg2.extras import Json

ClientSource = Literal['mcp', 'dashboard', 'fl_assistant', 'api']
PositiveId = conint(strict=True, gt=0)
PREFIXES = {'receive': 'RCV', 'make': 'MK', 'pack': 'PK', 'adjust': 'ADJ', 'found': 'FND'}
# Public dashboard scope remains exactly the ticket routes granted by A1 part 1.
DASHBOARD_ROUTES = frozenset({
    ('POST', '/receive/prepare'),
    ('POST', '/tickets/{ticket}/commit'),
    ('GET', '/receipts'),
    ('GET', '/receipts/{receipt_number}'),
    ('GET', '/receipts/by-transaction/{transaction_id}'),
})
ACTOR_ROUTES = DASHBOARD_ROUTES | frozenset({
    ('POST', '/make/prepare'),
    ('POST', '/pack/prepare'),
    ('POST', '/adjust/prepare'),
    ('POST', '/inventory/found/prepare'),
})


def fail(http_status, code, message, **extra):
    raise HTTPException(http_status, {'error_code': code, 'message': message, **extra})


class IdInput(BaseModel):
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
    product_id: PositiveId
    supplier_id: Optional[PositiveId] = None
    cases: conint(strict=True, gt=0)
    case_size_lb: confloat(gt=0)
    bol_reference: str
    shipper_code_override: Optional[str] = None
    lot_code: Optional[str] = None
    supplier_lot_code: Optional[str] = None
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


class MakePrepareRequest(ActionPrepareRequest):
    product_id: PositiveId
    batches: conint(strict=True, gt=0)
    lot_code: Optional[str] = None
    ingredient_lots: List[IngredientLot] = Field(default_factory=list)
    excluded_ingredients: List[PositiveId] = Field(default_factory=list)
    confirmed_sku: bool = False


class LotAllocation(IdInput):
    lot_id: PositiveId
    quantity_lb: confloat(gt=0)


class PackPrepareRequest(ActionPrepareRequest):
    source_product_id: PositiveId
    target_product_id: PositiveId
    cases: conint(strict=True, gt=0)
    case_weight_lb: Optional[confloat(gt=0)] = None
    lot_allocations: Optional[List[LotAllocation]] = None
    target_lot_code: Optional[str] = None


class AdjustPrepareRequest(ActionPrepareRequest):
    lot_id: PositiveId
    delta_lb: float
    reason_code: str = Field(min_length=1)
    reason_es: Optional[str] = None

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
    notes: Optional[str] = None
    notes_es: Optional[str] = None


class CommitRequest(BaseModel):
    payload_hash: str
    acknowledged_warnings: List[str] = Field(default_factory=list)

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
        'key_kind': request.state.key_kind,
    }


def blocker(exc):
    detail = exc.detail
    if isinstance(detail, dict):
        return {'code': detail.get('error_code', 'VALIDATION_FAILED'),
                'message': detail.get('message', str(detail))}
    return {'code': 'VALIDATION_FAILED', 'message': str(detail)}


def json_value(api, value):
    # Match the existing Decimal/date response encoding before persisting it.
    return json.loads(json.dumps(value, cls=api.DecimalSafeEncoder, allow_nan=False))


def supplier(cur, supplier_id, *, lock=False):
    if supplier_id is None:
        return None
    cur.execute('SELECT id, name, active FROM suppliers WHERE id=%s' +
                (' FOR SHARE' if lock else ''), (supplier_id,))
    row = cur.fetchone()
    if not row or not row['active']:
        fail(422, 'SUPPLIER_NOT_FOUND', 'Supplier is missing or inactive; resolve an active supplier.')
    return row


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
    shipper = supplier(cur, payload.get('supplier_id'), lock=lock)
    entries = []
    for entry in payload.get('supplier_lot_entries') or []:
        party = supplier(cur, entry.get('supplier_id'), lock=lock)
        entries.append({k: v for k, v in entry.items() if k != 'supplier_id'} |
                       {'supplier_name': party['name'] if party else None})
    fields = {k: v for k, v in payload.items()
              if k not in ('product_id', 'supplier_id', 'expected_receipt_id')}
    fields.update(product_name=product['name'], shipper_name=shipper['name'] if shipper else '',
                  supplier_lot_entries=entries or None, mode='preview')
    req = api.ReceiveRequest(**fields)
    draft = api._receive_preview_core(cur, req, product=product)
    lots = []
    cur.execute('SELECT id, status FROM lots WHERE product_id=%s AND lot_code=%s' +
                (' FOR UPDATE' if lock else ''), (product['id'], draft['lot_code']))
    lot = cur.fetchone()
    if lot:
        if lot['status'] == 'merged':
            fail(409, 'LOT_MERGED', 'This lot was merged; prepare again with the surviving lot.')
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
    state = {'product': {'id': product['id'], 'active': product['active']},
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
             'message_es': f"Una recepción similar se registró hace {minutes} minutos como {ref} por {row['operator_id']}.",
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


def receipt_detail(api, cur, number):
    cur.execute('SELECT * FROM write_tickets WHERE receipt_number=%s', (number,))
    ticket = cur.fetchone()
    if not ticket:
        fail(404, 'RECEIPT_NOT_FOUND', 'No recorded receipt has this number.')
    cur.execute('''SELECT t.id, t.type, t.effective_status FROM ledger_current_transactions t
                   JOIN transactions raw ON raw.id=t.id WHERE raw.ticket_id=%s ORDER BY t.id''',
                (ticket['id'],))
    transactions = [dict(row) for row in cur.fetchall()]
    lot_ids = set()
    for txn in transactions:
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
    happened = datetime.fromisoformat(ticket['payload']['occurred_at'])
    late = happened.astimezone(api.PLANT_TIMEZONE).date() != ticket['committed_at'].astimezone(api.PLANT_TIMEZONE).date()
    return {'receipt_number': number, 'action': ticket['action'], 'status': ticket['status'],
            'actor': ticket['draft']['actor'], 'operator_id': ticket['operator_id'],
            'client_source': ticket['client_source'], 'happened_at': happened,
            'entered_at': ticket['committed_at'], 'late_entry': late,
            'draft': ticket['draft'], 'response': ticket['response'],
            'warnings': ticket['warnings'], 'acknowledged': ticket['acknowledged'],
            'transactions': transactions, 'lots': lots}


def register_routes(app, api):
    def prepare(action, req, request):
        actor = identity(api, request)
        payload = json.loads(req.json(exclude={'client_source'}))
        event_time = req.occurred_at or api.get_plant_now()
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=api.PLANT_TIMEZONE)
        payload['occurred_at'] = event_time.astimezone(api.PLANT_TIMEZONE).isoformat()
        for field in ('lot_code', 'target_lot_code'):
            if payload.get(field):
                payload[field] = api.normalize_lot_code_input(payload[field])
        if action == 'receive':
            payload['supplier_lot_code'] = (req.supplier_lot_code or '').strip() or (req.lot_code or '').strip() or 'N/A'
        with api.get_transaction() as cur:
            blockers = []
            draft = {}
            state = {}
            try:
                if action == 'receive':
                    draft, state, _, _, _, _, er_id = validate_receive(api, cur, payload)
                    payload['expected_receipt_id'] = er_id
                    payload['lot_code'] = draft['lot_code']
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
                    elif action == 'adjust':
                        payload['product_id'] = draft['product_id']
            except HTTPException as exc:
                if exc.status_code >= 500:
                    raise
                blockers = [blocker(exc)]
            warnings = (receive_duplicates(cur, payload) if action == 'receive' else
                        actions.duplicates(cur, action, payload, draft))
            payload_hash = canonical_hash(payload)
            draft.update(actor=actor, happened_at=payload['occurred_at'],
                         happened_vs_now_minutes=round((api.get_plant_now()-event_time).total_seconds()/60, 1),
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
            ttl = 30 if req.client_source == 'dashboard' else 10
            cur.execute('''WITH clock AS (SELECT clock_timestamp() AS at)
                INSERT INTO write_tickets(ticket_hash,action,actor_id,operator_id,key_kind,client_source,
                    payload,payload_hash,state_hash,draft,warnings,prepared_at,expires_at)
                SELECT %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,at,at+%s*interval '1 minute'
                FROM clock RETURNING id,expires_at''',
                (token_hash(ticket), action, actor['id'], actor['name'], actor['key_kind'], req.client_source,
                 Json(payload), payload_hash, canonical_hash(json_value(api, state)), Json(draft), Json(warnings), ttl))
            row = cur.fetchone()
            return {'ticket': ticket, 'ticket_id': row['id'], 'action': action,
                    'expires_at': row['expires_at'], 'payload_hash': payload_hash, 'draft': draft,
                    'warnings': warnings, 'blockers': blockers, 'can_commit': not blockers, 'actor': actor}

    @app.post('/receive/prepare')
    def prepare_receive(req: ReceivePrepareRequest, request: Request, _: bool = Depends(api.verify_api_key)):
        return prepare('receive', req, request)

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
            if row['status'] != 'prepared':
                fail(409, 'TICKET_NOT_COMMITTABLE', 'Prepare a new ticket.', status=row['status'])
            cur.execute('SELECT clock_timestamp() > %s AS expired', (row['expires_at'],))
            if cur.fetchone()['expired']:
                cur.execute("UPDATE write_tickets SET status='expired' WHERE id=%s", (row['id'],))
                # Return, rather than raise: the terminal state must COMMIT.
                return api.JSONResponse(status_code=409, content={'detail': {
                    'error_code': 'TICKET_EXPIRED', 'message': 'Ticket expired; prepare again.'}})
            missing = sorted({w['code'] for w in row['warnings'] if w.get('requires_ack')} -
                             set(body.acknowledged_warnings))
            if missing:
                fail(409, 'WARNING_NOT_ACKNOWLEDGED', 'Acknowledge the draft warnings.', missing=missing)
            if row['action'] not in PREFIXES:
                fail(409, 'TICKET_ACTION_UNAVAILABLE', 'Unsupported ticket action.')
            cur.execute('SAVEPOINT ticket_post')
            try:
                # Reuse the legacy lot-sequence locks before validation/posting.
                action_lock = {'receive': 1, 'found': 2, 'make': 3}.get(row['action'])
                if action_lock:
                    cur.execute('SELECT pg_advisory_xact_lock(%s)', (action_lock,))
                if actor['id'] is not None:
                    cur.execute('SELECT active FROM actors WHERE id=%s FOR SHARE', (actor['id'],))
                    current_actor = cur.fetchone()
                    if not current_actor or not current_actor['active']:
                        fail(403, 'ACTOR_INACTIVE', 'The preparing actor is no longer active.')
                if row['action'] == 'receive':
                    draft, state, req, product, occurred_at, source, er_id = validate_receive(
                        api, cur, row['payload'], lock=True)
                else:
                    validated = actions.validate(api, cur, row['action'], row['payload'], lock=True)
                    draft, state, _, _, occurred_at, source, _, _ = validated
                if row['draft'].get('blockers'):
                    fail(409, 'DRAFT_BLOCKED', 'Prepare again after resolving the draft blockers.',
                         blockers=row['draft']['blockers'])
                # A draft promising a new lot cannot silently add to a lot
                # created after prepare. This runs under the receive lock.
                if not row['draft'].get('lot_exists') and draft.get('lot_exists'):
                    name = draft.get('product_name') or draft.get('target_product_name')
                    code = draft.get('lot_code') or draft.get('output_lot_code')
                    fail(409, 'LOT_CODE_TAKEN',
                         f'{name} lot {code} is now in use; prepare again for a fresh draft.')
                state_changed = canonical_hash(json_value(api, state)) != row['state_hash']
                receipt = allocate_receipt(cur, row['action'], occurred_at.astimezone(api.PLANT_TIMEZONE).date())
                if row['action'] == 'receive':
                    req.mode = 'commit'
                    response = api._receive_commit_core(cur, req, request, occurred_at, source,
                        product=product, ticket_id=row['id'], receipt_number=receipt, expected_receipt_id=er_id)
                else:
                    response = actions.post(api, cur, row['action'], validated, row['payload'], request,
                        row['id'], receipt, require_new_lot=not row['draft'].get('lot_exists'))
                response.update(receipt_number=receipt, ticket_id=row['id'], replayed=False,
                                state_changed=state_changed)
                response = json_value(api, response)
            except HTTPException as exc:
                if exc.status_code >= 500:
                    raise
                cur.execute('ROLLBACK TO SAVEPOINT ticket_post')
                errors = exc.detail.get('blockers') if isinstance(exc.detail, dict) else None
                errors = errors or [blocker(exc)]
                cur.execute("UPDATE write_tickets SET status='rejected',reject_reason=%s WHERE id=%s",
                            (json.dumps(errors), row['id']))
                return api.JSONResponse(status_code=409, content={'detail': {
                    'error_code': 'TICKET_STALE', 'message': 'Draft no longer valid; prepare again.',
                    'blockers': errors}})
            cur.execute('RELEASE SAVEPOINT ticket_post')
            cur.execute('SELECT DISTINCT lot_id FROM transaction_lines WHERE transaction_id=%s AND lot_id IS NOT NULL ORDER BY lot_id',
                        (response['transaction_id'],))
            result_ref = {'transaction_ids': [response['transaction_id']],
                          'lot_ids': [line['lot_id'] for line in cur.fetchall()]}
            cur.execute('''UPDATE write_tickets SET status='committed',committed_at=clock_timestamp(),
                receipt_number=%s,result_ref=%s,response=%s,acknowledged=%s WHERE id=%s''',
                (receipt, Json(result_ref), Json(response), Json(sorted(set(body.acknowledged_warnings))), row['id']))
            return response

    @app.get('/receipts')
    def list_receipts(date: Optional[date] = Query(None), actor: Optional[str] = None,
                      action: Optional[str] = None, status: Optional[str] = None,
                      client_source: Optional[ClientSource] = None,
                      _: bool = Depends(api.verify_api_key)):
        day = date or api.get_plant_now().date()
        with api.get_transaction() as cur:
            cur.execute('''SELECT receipt_number FROM write_tickets
                WHERE receipt_number IS NOT NULL
                  AND (((payload->>'occurred_at')::timestamptz AT TIME ZONE 'America/New_York')::date=%s
                       OR (committed_at AT TIME ZONE 'America/New_York')::date=%s)
                  AND (%s IS NULL OR operator_id=%s OR actor_id::text=%s)
                  AND (%s IS NULL OR action=%s) AND (%s IS NULL OR status=%s)
                  AND (%s IS NULL OR client_source=%s)
                ORDER BY committed_at,id''',
                (day, day, actor, actor, actor, action, action, status, status, client_source, client_source))
            numbers = [row['receipt_number'] for row in cur.fetchall()]
            return {'date': day, 'receipts': [receipt_detail(api, cur, number) for number in numbers]}

    @app.get('/receipts/by-transaction/{transaction_id}')
    def receipt_by_transaction(transaction_id: int, _: bool = Depends(api.verify_api_key)):
        with api.get_transaction() as cur:
            cur.execute('SELECT receipt_number FROM transactions WHERE id=%s', (transaction_id,))
            row = cur.fetchone()
            if not row or row['receipt_number'] is None:
                fail(404, 'RECEIPT_NOT_FOUND', 'No ticket receipt is recorded for this transaction.')
            return receipt_detail(api, cur, row['receipt_number'])

    @app.get('/receipts/{receipt_number}')
    def get_receipt(receipt_number: str, _: bool = Depends(api.verify_api_key)):
        with api.get_transaction() as cur:
            return receipt_detail(api, cur, receipt_number)
