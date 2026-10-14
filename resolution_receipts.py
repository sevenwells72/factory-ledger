"""Atomic ADJ/XR/VD receipts and the resolution/void prepare-commit adapters.

Every helper uses the caller's transaction. XR records the exception decision;
resolution_ticket_id continues to identify its fixing posting, such as ADJ.
"""
from typing import Literal
from fastapi import Depends, Request
from pydantic import BaseModel, Field, field_validator
from psycopg2.extras import Json
import permissions
import exceptions_enforcement as exceptions


def start(api, cur, actor, action, payload, draft=None, state=None, client_source='api'):
    import write_tickets as wt
    now = api.get_plant_now()
    payload = wt.json_value(api, {**payload, 'occurred_at': now.isoformat()})
    issued = wt.issue_ticket(api, cur, actor=actor, action=action, payload=payload,
        draft=draft or {}, state=state or {}, warnings=[], blockers=[], client_source=client_source, event_time=now)
    cur.execute('SELECT * FROM write_tickets WHERE id=%s', (issued['ticket_id'],))
    return cur.fetchone(), issued


def begin(api, cur, action, actor, payload, draft, *, prepared=None):
    import write_tickets as wt
    if prepared is None:
        prepared, _ = start(api, cur, actor, action, payload, draft)
    number = wt.allocate_receipt(cur, action, api.get_plant_now().date())
    return {'ticket_id': prepared['id'], 'receipt_number': number}


def finish(api, cur, receipt, response, result_ref):
    import write_tickets as wt
    response = wt.json_value(api, {'success': True, **response})
    cur.execute("""UPDATE write_tickets SET status='committed',committed_at=clock_timestamp(),
        receipt_number=%s,result_ref=%s,response=%s WHERE id=%s""",
        (receipt['receipt_number'], Json(result_ref), Json(response), receipt['ticket_id']))
    return response


def decision(api, cur, row, actor, kind, *, prepared=None):
    receipt = begin(api, cur, 'resolve_exception', actor,
        {'exception_id': row['id'], 'resolution_kind': kind},
        {'exception_id': row['id'], 'kind': row['kind'], 'resolution_kind': kind}, prepared=prepared)
    # Populate the number before the exception's FK points at it, still uncommitted.
    cur.execute('UPDATE write_tickets SET receipt_number=%s WHERE id=%s',
                (receipt['receipt_number'], receipt['ticket_id']))
    cur.execute('UPDATE exceptions SET resolution_receipt_number=%s WHERE id=%s',
                (receipt['receipt_number'], row['id']))
    return receipt


def record_void(api, cur, transaction_id, event, actor, reason, *, ticket=None):
    cur.execute('SELECT receipt_number FROM transactions WHERE id=%s', (transaction_id,))
    source = cur.fetchone()['receipt_number']
    receipt = begin(api, cur, 'void', actor, {'transaction_id': transaction_id, 'reason': reason},
        {'transaction_id': transaction_id, 'source_receipt_number': source, 'reason': reason}, prepared=ticket)
    return finish(api, cur, receipt, {**event, **receipt, 'source_receipt_number': source,
        'voided_transaction_id': transaction_id, 'success': True, 'replayed': False},
        {'correction_id': str(event['correction_id']), 'voided_transaction_id': transaction_id,
         'transaction_ids': [], 'lot_ids': []})


def current_actor(cur, actor):
    if actor['id'] is not None:
        cur.execute('SELECT role,active FROM actors WHERE id=%s FOR SHARE', (actor['id'],))
        row = cur.fetchone()
        if not row or not row['active']:
            exceptions.fail(403, 'ACTOR_INACTIVE', 'This actor is no longer active.')
        actor = {**actor, 'role': row['role']}
    return actor


def require_void(api, cur, actor, transaction_id):
    actor = current_actor(cur, actor)
    permissions.require('void', actor)
    cur.execute('''SELECT t.effective_status,t.latest_correction_id,r.entered_by_actor_id,r.business_date,
                          r.receipt_number FROM ledger_current_transactions t
                   JOIN transactions r ON r.id=t.id WHERE t.id=%s''', (transaction_id,))
    row = cur.fetchone()
    if not row:
        exceptions.fail(404, 'TRANSACTION_NOT_FOUND', 'Transaction not found.')
    if permissions.role_of(actor) == permissions.FLOOR and (
            row['entered_by_actor_id'] != actor['id'] or row['business_date'] != api.get_plant_now().date()):
        exceptions.fail(403, 'VOID_OWNER_REQUIRED', 'Floor may void only their own posts on the same plant day.')
    return row


def correction_actor(cur, actor_id, operator_id):
    if actor_id is None:
        return {'id': None, 'name': operator_id, 'role': None, 'key_kind': 'legacy_ledger'}
    cur.execute('SELECT id,name,role FROM actors WHERE id=%s', (actor_id,))
    row = cur.fetchone()
    return {**dict(row), 'key_kind': 'actor'}


def resolution_state(api, cur, exception_id, actor, body, *, lock=False):
    if lock:
        exceptions.lock_resolution(cur, exception_id)
    row = exceptions._one(cur, exception_id)
    exceptions.require_resolution(row, actor, body.resolution_kind)
    state = {'status': row['status'], 'detail': row['detail'], 'resolution_kind': row['resolution_kind']}
    draft = {'exception_id': exception_id, 'kind': row['kind'], 'resolution_kind': body.resolution_kind,
             'source_receipt_number': row['receipt_number'], 'note': body.note}
    if body.resolution_kind == 'counted':
        if body.counted_lb is None:
            exceptions.fail(422, 'COUNT_REQUIRED', 'A counted resolution needs counted_lb.')
        if row['lot_id'] is None:
            exceptions.fail(409, 'LOT_REQUIRED', 'This exception names no lot to count.')
        if lock:
            cur.execute('SELECT id FROM lots WHERE id=%s FOR UPDATE', (row['lot_id'],))
        state['book_before_lb'] = float(api.lot_on_hand(cur, row['lot_id']))
        cur.execute('SELECT p.* FROM products p JOIN lots l ON l.product_id=p.id WHERE l.id=%s', (row['lot_id'],))
        product = cur.fetchone()
        if not product:
            exceptions.fail(409, 'LOT_NOT_FOUND', 'The short lot is missing.')
        draft.update(counted_lb=body.counted_lb, book_before_lb=state['book_before_lb'],
                     delta_lb=round(body.counted_lb-state['book_before_lb'], 4))
        draft['correction_review'] = exceptions.correction_review(api, 'adjust', product, draft['delta_lb'],
            book_balance_before=state['book_before_lb'], reason_code='physical_count')
        state['correction_review'] = draft['correction_review']
    return draft, state


class ResolvePrepare(exceptions.ResolveRequest):
    client_source: Literal['mcp','dashboard','fl_assistant','api'] = 'dashboard'


class VoidPrepare(BaseModel):
    reason: str = Field(min_length=1, max_length=2000)
    client_source: Literal['mcp','dashboard','fl_assistant','api'] = 'dashboard'
    _strip = field_validator('reason')(exceptions._non_blank)
    class Config:
        extra = 'forbid'


def register_routes(app, api):
    import write_tickets as tickets

    @app.post('/exceptions/{exception_id}/resolve/prepare')
    def prepare_resolution(exception_id: int, body: ResolvePrepare, request: Request,
                           _: bool = Depends(api.verify_api_key)):
        actor = tickets.identity(api, request)
        permissions.require('resolve_exception', actor)
        with api.get_transaction() as cur:
            actor = current_actor(cur, actor)
            draft, state = resolution_state(api, cur, exception_id, actor, body)
            _, prepared = start(api, cur, actor, 'resolve_exception',
                {'exception_id': exception_id, **body.model_dump(exclude={'client_source','attachment_id'})},
                draft, state, body.client_source)
            return prepared

    @app.post('/void/{transaction_id}/prepare')
    def prepare_void(transaction_id: int, body: VoidPrepare, request: Request,
                     _: bool = Depends(api.verify_api_key)):
        actor = tickets.identity(api, request)
        with api.get_transaction() as cur:
            actor = current_actor(cur, actor)
            state = require_void(api, cur, actor, transaction_id)
            if state['effective_status'] == 'voided':
                exceptions.fail(409, 'TRANSACTION_VOIDED', 'This transaction is already voided.')
            draft = {'transaction_id': transaction_id, 'reason': body.reason,
                     'source_receipt_number': state['receipt_number']}
            _, prepared = start(api, cur, actor, 'void', {'transaction_id': transaction_id, 'reason': body.reason},
                                draft, dict(state), body.client_source)
            return prepared


def commit(api, cur, row, body, actor, request):
    import write_tickets as tickets
    if body.attachment_ref or body.attachment_ids or body.lot_confirmations:
        exceptions.fail(422, 'UNUSED_EVIDENCE', 'Resolution evidence belongs in its prepared payload.')
    actor = current_actor(cur, actor)
    payload = row['payload']
    if row['action'] == 'resolve_exception':
        data = {k:v for k,v in payload.items() if k not in ('exception_id','occurred_at')}
        resolution = exceptions.ResolveRequest(**data)
        _, state = resolution_state(api, cur, payload['exception_id'], actor, resolution, lock=True)
    else:
        cur.execute('SELECT id FROM transactions WHERE id=%s FOR UPDATE', (payload['transaction_id'],))
        state = dict(require_void(api, cur, actor, payload['transaction_id']))
    if tickets.canonical_hash(tickets.json_value(api, state)) != row['state_hash']:
        exceptions.fail(409, 'TICKET_STALE', 'The entry changed; prepare again before recording.')
    if row['action'] == 'resolve_exception':
        return exceptions.resolve_in_transaction(api, cur, payload['exception_id'], resolution, request, actor, prepared=row)
    result = api._append_transaction_correction(cur, payload['transaction_id'], 'void', payload['reason'], None,
        actor['name'], trace_operator_id=actor['name'], entered_by_actor_id=actor['id'], receipt_ticket=row)
    return result
