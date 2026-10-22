"""Floor's existing two operations use the ticket input/evidence validator.

The published Floor schema requires the explicit evidence fields on every call.
Old callers without those fields retain their legacy API contract.
"""
import json
import os

from fastapi import HTTPException

import ticket_actions
import write_tickets


def checked_path_required(req):
    """Rollout switch: enable only after the new Floor contract is pasted.

    Read per request so the default-off rollout has no startup migration or
    automatic setting changes. Explicit null does not satisfy the contract.
    """
    enabled = os.getenv('REQUIRE_MAKE_PACK_LOT_EVIDENCE', '').strip().lower() in ('1', 'true', 'yes', 'on')
    missing = [name for name in ('ingredient_lot_allocations', 'lot_confirmations')
               if getattr(req, name) is None]
    if enabled and req.mode == 'commit' and missing:
        raise HTTPException(422, {'error_code': 'MAKE_PACK_LOT_EVIDENCE_REQUIRED',
            'message': 'Choose and confirm the ingredient lots before recording. Preview again and pick the lots you used.',
            'message_es': 'Elija y confirme los lotes de ingredientes antes de registrar. Revise de nuevo y elija los lotes que usó.',
            'required_fields': missing})
    return enabled or len(missing) < 2


def _lot_id(cur, product_id, code):
    cur.execute("SELECT id FROM lots WHERE product_id=%s AND lower(lot_code)=lower(%s) "
                "AND status IS DISTINCT FROM 'merged'", (product_id, code))
    rows = cur.fetchall()
    if len(rows) != 1:
        raise HTTPException(422, {'error_code': 'LOT_NOT_FOUND',
                                 'message': 'Choose an existing lot for this product.'})
    return rows[0]['id']


def run(api, cur, action, req, request, occurred_at):
    common = {'occurred_at': occurred_at or api.get_plant_now(), 'backfill': req.backfill,
              'lot_confirmations': req.lot_confirmations or [],
              'ingredient_lot_allocations': req.ingredient_lot_allocations or []}
    if action == 'make':
        product = api.resolve_product_full(cur, req.product_name)
        overrides = [{'ingredient_product_id': int(pid), 'lot_id': _lot_id(cur, int(pid), code)}
                     for pid, code in (req.get_lot_overrides() or {}).items()]
        model = write_tickets.MakePrepareRequest(**common, product_id=product['id'],
            batches=req.batches, lot_code=req.lot_code, confirmed_sku=bool(req.confirmed_sku),
            excluded_ingredients=req.excluded_ingredients or [], ingredient_lots=overrides)
    else:
        source = api.resolve_product_full(cur, req.source_product)
        target = api.resolve_product_full(cur, req.target_product)
        allocations = [{'lot_id': _lot_id(cur, source['id'], row.lot_code), 'quantity_lb': row.quantity_lb}
                       for row in req.lot_allocations or []]
        model = write_tickets.PackPrepareRequest(**common, source_product_id=source['id'],
            target_product_id=target['id'], cases=req.cases, case_weight_lb=req.case_weight_lb,
            target_lot_code=req.target_lot_code, lot_allocations=allocations or None)
    payload = json.loads(model.model_dump_json(exclude={'client_source'}))
    committing = req.mode == 'commit'
    if committing:
        # Match ticket lock order before measuring/locking the exact inputs.
        if action == 'make':
            cur.execute('SELECT pg_advisory_xact_lock(3)')
        else:
            api._lock_allocation_product(cur, source['id'])
    validated = ticket_actions.validate(api, cur, action, payload, lock=committing)
    draft, _, _, _, _, _, _, plan = validated
    if not committing:
        return dict(draft, mode='preview', can_commit=not draft.get('blockers'),
                    confirmation_required=True)
    if any(row.get('lot_exception') for row in plan):
        raise HTTPException(422, {'error_code': 'LOT_REVIEW_REQUIRED',
                                 'message': 'Use a reviewed Make ticket for an unidentified physical lot.'})
    # Reuse the posting hooks too: evidence, shortages and pre-make adjustments
    # are atomic with the exact selected ledger inputs, just as for tickets.
    return ticket_actions.post(api, cur, action, validated, payload, request,
                               ticket_id=None, receipt_number=None, require_new_lot=False)
