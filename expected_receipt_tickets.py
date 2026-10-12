"""Expected-delivery metadata tickets; the existing receive and QBO paths stay separate.

All writes use the same main.py cores as manual entry and document approval.
The shared A1 lifecycle owns identities, expiry, hashes, replay and transaction.
"""
from datetime import date
from typing import List, Literal, Optional

from fastapi import Depends, Request
from pydantic import Field, confloat
from psycopg2.extras import Json

import ticket_actions
import write_tickets as wt

ACTIONS = wt.EXPECTED_RECEIPT_ACTIONS


class ExpectedRequest(wt.IdInput):
    client_source: wt.ClientSource = 'api'


class CreateRequest(ExpectedRequest):
    product_id: wt.PositiveId
    supplier_id: wt.PositiveId
    expected_qty: confloat(gt=0)
    expected_date: Optional[date] = None
    reference_number: Optional[str] = None
    notes: Optional[str] = None
    source_document_id: Optional[wt.PositiveId] = None


class UpdateRequest(ExpectedRequest):
    expected_qty: Optional[confloat(gt=0)] = None
    expected_date: Optional[date] = None
    reference_number: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[Literal['closed', 'cancelled']] = None


class IntakeLine(wt.IdInput):
    product_id: wt.PositiveId
    expected_qty_lb: confloat(gt=0)
    expected_date: Optional[date] = None
    vendor_description: str = Field(min_length=1)
    quantity: Optional[confloat(gt=0)] = None
    unit: Optional[str] = None
    lb_per_unit: Optional[confloat(gt=0)] = None
    save_alias: bool = True


class IntakeRequest(ExpectedRequest):
    document_id: wt.PositiveId
    supplier_id: wt.PositiveId
    reference_number: Optional[str] = None
    expected_date: Optional[date] = None
    lines: List[IntakeLine] = Field(min_length=1)
    force: bool = False


def validate(api, cur, action, payload, *, lock=False):
    ctx = {'blockers': [], 'state': {}}
    if action == 'create_expected_receipt':
        cur.execute('SELECT id, name, active FROM suppliers WHERE id=%s' + (' FOR SHARE' if lock else ''),
                    (payload['supplier_id'],))
        supplier = cur.fetchone()
        if not supplier:
            wt.fail(422, 'SUPPLIER_NOT_FOUND', 'Resolve an existing supplier.')
        if not supplier['active']:
            wt.fail(422, 'SUPPLIER_INACTIVE', 'Resolve an active supplier.')
        ids = [l['product_id'] for l in payload['lines']] if 'document_id' in payload else [payload['product_id']]
        products = [ticket_actions.product(api, cur, pid, lock) for pid in sorted(set(ids))]
        ctx['state'] = {'supplier': dict(supplier),
                        'products': [{'id': p['id'], 'active': p['active']} for p in products]}
        if 'document_id' in payload:
            if lock:
                # Match the core/direct intake lock order: document then supplier reference.
                cur.execute('SELECT id FROM purchase_documents WHERE id=%s FOR UPDATE', (payload['document_id'],))
                api._lock_supplier_reference(cur, payload['supplier_id'], payload.get('reference_number'))
            duplicates = api._dedupe_existing_receipts(cur, payload['supplier_id'], payload.get('reference_number'))
            ctx['duplicates'] = duplicates
            ctx['state']['duplicates'] = duplicates
        return ctx
    cur.execute('SELECT * FROM expected_receipts WHERE id=%s' + (' FOR UPDATE' if lock else ''),
                (payload['expected_receipt_id'],))
    row = cur.fetchone()
    if not row:
        wt.fail(404, 'EXPECTED_RECEIPT_NOT_FOUND', 'Expected receipt not found.')
    if row['status'] != 'open':
        wt.fail(409, 'EXPECTED_RECEIPT_NOT_OPEN', 'Only open expected receipts can be edited or cancelled.')
    ctx['state'] = dict(row)
    return ctx


def post(api, cur, action, payload, request, ctx, *, ticket_id, allow_duplicate):
    if action == 'create_expected_receipt':
        if 'document_id' in payload:
            req = api.ExpectedReceiptApproveRequest(**{k: v for k, v in payload.items() if k != 'occurred_at'})
            req.force = allow_duplicate
            result = api._approve_extracted_receipts_core(cur, req, request)
        else:
            record = api._create_expected_receipt_core(
                cur, payload['product_id'], payload['supplier_id'], payload['expected_qty'],
                payload.get('expected_date'), payload.get('reference_number'), payload.get('notes'),
                api.caller_source_tag(request), source_document_id=payload.get('source_document_id'))
            result = {'expected_receipt_id': record['id'], 'expected_receipt': record,
                      'message': f"Expected receipt #{record['id']} created"}
        if ticket_id is not None:
            cur.execute('UPDATE expected_receipts SET ticket_id=%s WHERE id=ANY(%s)',
                        (ticket_id, result_ids(result)))
        return result
    fields = ({'status': 'cancelled'} if action == 'cancel_expected_receipt'
              else {k: payload[k] for k in payload['fields']})
    return api._update_expected_receipt_core(cur, payload['expected_receipt_id'], api.ExpectedReceiptUpdate(**fields))


def result_ids(response):
    if response.get('expected_receipt_id'):
        return [response['expected_receipt_id']]
    return [r['id'] for r in response.get('created', [])]


def draft_from(action, response):
    draft = dict(response)
    if action == 'create_expected_receipt':
        draft.pop('expected_receipt_id', None)
        for record in ([draft['expected_receipt']] if 'expected_receipt' in draft else draft.get('created', [])):
            record.pop('id', None)
        draft['message'] = 'Expected delivery will be recorded at commit.'
        draft['summary'] = 'Record expected delivery' if 'created' not in draft else f"Record {draft['created_count']} expected deliveries"
        draft['summary_es'] = 'Registrar entrega esperada'
    else:
        draft['summary'] = response['message']
        draft['summary_es'] = 'Actualizar entrega esperada'
    return draft


def warnings_for(cur, action, payload, ctx):
    warnings = []
    if ctx.get('duplicates'):
        warnings.append({'code': 'DUPLICATE_REFERENCE', 'requires_ack': not payload.get('force', False),
                         'message': 'This supplier already has expected receipts with this reference. Confirm to create anyway.',
                         'message_es': 'Este proveedor ya tiene entregas esperadas con esta referencia. Confirme para crear de todos modos.',
                         'refs': {'existing': ctx['duplicates']}})
    if action == 'create_expected_receipt' and 'document_id' not in payload:
        # Same committed intent within 24h, across actors; different actors cannot silently double-post.
        cur.execute('''SELECT receipt_number FROM write_tickets WHERE action=%s AND status='committed'
                       AND committed_at >= clock_timestamp()-interval '24 hours'
                       AND payload-'occurred_at' = %s::jsonb-'occurred_at'
                       ORDER BY committed_at DESC LIMIT 1''', (action, Json(payload)))
        row = cur.fetchone()
        if row:
            warnings.append({'code': 'POSSIBLE_DUPLICATE', 'requires_ack': True,
                             'message': f"An identical expected delivery was recorded as {row['receipt_number']}.",
                             'message_es': f"Una entrega idéntica se registró como {row['receipt_number']}.",
                             'refs': {'receipt_number': row['receipt_number']}})
    return warnings


def receipt_records(api, cur, ticket):
    if ticket['action'] not in ACTIONS:
        return []
    return [api.fetch_expected_receipt(cur, er_id) or {'id': er_id}
            for er_id in (ticket.get('result_ref') or {}).get('expected_receipt_ids', [])]


def register_routes(app, api, prepare):
    @app.post('/expected-receipts/prepare')
    def prepare_create_expected_receipt(req: CreateRequest, request: Request, _: bool = Depends(api.verify_api_key)):
        return prepare('create_expected_receipt', req, request)

    @app.post('/expected-receipts/{expected_receipt_id}/update/prepare')
    def prepare_update_expected_receipt(req: UpdateRequest, request: Request, expected_receipt_id: int,
                                        _: bool = Depends(api.verify_api_key)):
        return prepare('update_expected_receipt', req, request, expected_receipt_id=expected_receipt_id)

    @app.post('/expected-receipts/{expected_receipt_id}/cancel/prepare')
    def prepare_cancel_expected_receipt(req: ExpectedRequest, request: Request, expected_receipt_id: int,
                                        _: bool = Depends(api.verify_api_key)):
        return prepare('cancel_expected_receipt', req, request, expected_receipt_id=expected_receipt_id)

    @app.post('/expected-receipts/extract/approve/prepare')
    def prepare_approve_expected_receipts(req: IntakeRequest, request: Request, _: bool = Depends(api.verify_api_key)):
        return prepare('create_expected_receipt', req, request)
