"""ID-bound A1 make/pack/adjust/found adapters; callers own the transaction."""
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import json

import lot_confirmation as a5
import exceptions_enforcement as a3b

from fastapi import HTTPException


def fail(code, message):
    raise HTTPException(409, {'error_code': code, 'message': message})


def product(api, cur, product_id, lock=False):
    cur.execute('SELECT * FROM products WHERE id=%s' + (' FOR SHARE' if lock else ''), (product_id,))
    row = cur.fetchone()
    if not row:
        fail('PRODUCT_NOT_FOUND', 'Product is missing; resolve an active product.')
    if row['active'] is False:
        fail('PRODUCT_NOT_FOUND', f'{row["name"]} is inactive; resolve an active product.')
    return row


def lot(api, cur, lot_id, product_id=None, lock=False):
    cur.execute('''SELECT l.id,l.product_id,l.lot_code,l.status,p.name AS product_name
                   FROM lots l JOIN products p ON p.id=l.product_id WHERE l.id=%s''' +
                (' FOR UPDATE OF l' if lock else ''), (lot_id,))
    row = cur.fetchone()
    if not row:
        fail('LOT_NOT_FOUND', 'Lot is missing; resolve an existing lot.')
    label = f'{row["product_name"]} lot {row["lot_code"]}'
    if product_id is not None and row['product_id'] != product_id:
        expected = product(api, cur, product_id)
        fail('LOT_NOT_FOUND', f'{label} does not belong to {expected["name"]}.')
    if row['status'] == 'merged':
        fail('LOT_MERGED', f'{label} was merged; prepare again with the surviving lot.')
    return dict(row, on_hand_lb=api.lot_on_hand(cur, lot_id))


def response_dict(response):
    if hasattr(response, 'status_code'):
        body = json.loads(response.body)
        raise HTTPException(response.status_code, body.get('error') or body.get('warning') or body)
    return response


def choose_inputs(api, cur, requirements, overrides):
    plan = []
    for pid, needed in sorted(requirements.items()):
        if pid in overrides:
            rows = [lot(api, cur, overrides[pid], pid)]
        else:
            cur.execute('''SELECT id FROM lots WHERE product_id=%s AND status IS DISTINCT FROM 'merged'
                           ORDER BY COALESCE(received_at,created_at),id''', (pid,))
            rows = [lot(api, cur, row['id'], pid) for row in cur.fetchall()]
        remaining = needed
        for row in rows:
            take = min(max(0, float(row['on_hand_lb'])), remaining)
            if take > api.BALANCE_EPSILON:
                plan.append({'product_id': pid, 'lot_id': row['id'], 'quantity_lb': take})
                remaining -= take
        if remaining > api.BALANCE_EPSILON:
            # A3b hook (R3): insufficient stock is NOT a blocker. The shortfall is
            # pinned on one lot (the override, the lot that ran out, or the newest)
            # so the post balances and the shortage flag names a lot. Only a
            # product with no lot at all still blocks: nothing can go negative.
            if a3b.assign_shortfall(plan, pid, remaining, rows, overrides.get(pid)) is None:
                name = product(api, cur, pid)['name']
                fail('INSUFFICIENT_STOCK', f'{name} needs {float(needed)} lb and has no lot to record the shortage against; receive it first.')
    return plan


def validate(api, cur, action, payload, lock=False):
    # A5 hook: non-ledger pallet evidence uses the same ticket lifecycle.
    if action == 'move_lot':
        return a5.validate_move(api, cur, payload, lock)
    occurred_at, time_source = api.validate_inventory_occurred_at(
        datetime.fromisoformat(payload['occurred_at']), payload['backfill'])
    common = {'occurred_at': occurred_at, 'backfill': payload['backfill']}
    products = []
    requirements = {}
    input_plan = payload.get('input_plan')
    output_product = None
    output_code = None
    options = {}
    if action == 'make':
        p = product(api, cur, payload['product_id'], lock)
        products.append(p)
        overrides = {}
        for entry in payload.get('ingredient_lots') or []:
            pid = entry['ingredient_product_id']
            if pid in overrides:
                fail('DUPLICATE_LOT_SELECTION', 'Select one override lot per ingredient.')
            overrides[pid] = entry['lot_id']
        override_codes = {str(pid): lot(api, cur, lid, pid)['lot_code'] for pid, lid in overrides.items()}
        req = api.MakeRequest(**common, product_name=p['name'], batches=payload['batches'],
            lot_code=payload.get('lot_code'), excluded_ingredients=payload.get('excluded_ingredients'),
            confirmed_sku=payload['confirmed_sku'], ingredient_lot_overrides=override_codes)
        draft = api._make_preview_core(cur, req, product=p)
        if draft['total_output_lb'] <= 0:
            fail('INVALID_OUTPUT', f'{p["name"]} requires a positive configured batch weight.')
        if draft.get('sku_confirmation_required') and not req.confirmed_sku:
            fail('SKU_CONFIRMATION_REQUIRED', f'Confirm {p["name"]} as the output SKU and prepare again.')
        # A5 hook: explicitly substitute recipe inputs before selecting lots.
        substitutions = a5.prepare_substitutions(api, cur, payload, draft, overrides, lock)
        for ingredient in draft['ingredients']:
            pid = ingredient['ingredient_id']
            requirements[pid] = requirements.get(pid, 0) + float(api.to_decimal(ingredient['needed_lb']))
        if set(overrides) - set(requirements):
            fail('UNUSED_LOT_SELECTION', 'An ingredient override is not part of this batch.')
        input_plan = input_plan if input_plan is not None else choose_inputs(api, cur, requirements, overrides)
        output_product, output_code = p, draft['lot_code']
        options = {'product': p, 'input_plan': input_plan, 'substitutions': substitutions}
        specification = {'requirements': requirements, 'output_lb': draft['total_output_lb'],
                         'excluded': draft.get('excluded_ingredients', []),
                         'original_requirements': {i['ingredient_id']: i['needed_lb'] for i in draft['original_ingredients']}}
    elif action == 'pack':
        if lock:
            # Keep the legacy source-lot/allocation lock order before any pinned input locks.
            api._lock_allocation_product(cur, payload['source_product_id'])
        source = product(api, cur, payload['source_product_id'], lock)
        target = product(api, cur, payload['target_product_id'], lock)
        if source['id'] == target['id']:
            fail('INVALID_PACK', 'Source and target products must differ.')
        products.extend([source, target])
        allocations = payload.get('lot_allocations')
        if input_plan is not None:
            allocations = [item for item in input_plan if item['product_id'] == source['id']]
        if allocations and len({a['lot_id'] for a in allocations}) != len(allocations):
            fail('DUPLICATE_LOT_SELECTION', 'Each source lot may appear only once.')
        legacy_allocations = [{'lot_code': lot(api, cur, a['lot_id'], source['id'])['lot_code'],
                              'quantity_lb': a['quantity_lb']} for a in allocations or []]
        req = api.PackRequest(**common, source_product=source['name'], target_product=target['name'],
            cases=payload['cases'], case_weight_lb=payload.get('case_weight_lb'),
            target_lot_code=payload.get('target_lot_code'), lot_allocations=legacy_allocations or None)
        draft = response_dict(api._pack_preview_core(cur, req, source=source, target=target))
        primary = [{'product_id': source['id'], 'lot_id': a['lot_id'], 'quantity_lb': a['allocated_lb']}
                   for a in draft['allocations'] if a.get('lot_id')]
        if input_plan is not None:
            # Commit: the pinned plan (quantities including any shortfall) is authoritative;
            # the preview cannot see a pinned lot that is at or below zero.
            primary = [{'product_id': source['id'], 'lot_id': a['lot_id'], 'quantity_lb': a['quantity_lb']} for a in allocations]
        elif (not draft['all_lots_sufficient'] or
                abs(sum(a['quantity_lb'] for a in primary) - draft['total_lb']) > api.BALANCE_EPSILON):
            # A3b hook (R3): insufficient batch stock is not a blocker — the uncovered
            # pounds are pinned on a lot (explicit allocation, the FIFO lot that ran
            # out, or the newest lot) and flagged at commit.
            primary = a3b.pack_shortfall(cur, source, draft, bool(allocations), primary)
        if abs(sum(a['quantity_lb'] for a in primary) - draft['total_lb']) > api.BALANCE_EPSILON:
            fail('INPUT_PLAN_CHANGED', f'{source["name"]} allocations do not add up to {draft["total_lb"]} lb; prepare again.')
        for ingredient in draft.get('add_in_ingredients', []):
            requirements[ingredient['ingredient_id']] = ingredient['needed_lb']
        if input_plan is None:
            input_plan = primary + choose_inputs(api, cur, requirements, {})
        requirements[source['id']] = draft['total_lb']
        req.case_weight_lb = draft['case_weight_lb']
        # Freeze FIFO source choices for the shared posting core too.
        req.lot_allocations = [api.PackLotAllocation(lot_code=lot(api, cur, a['lot_id'], source['id'])['lot_code'],
                              quantity_lb=a['quantity_lb']) for a in primary]
        output_product, output_code = target, draft['output_lot_code']
        options = {'source': source, 'target': target, 'input_plan': input_plan}
        specification = {'requirements': requirements, 'output_lb': draft['total_lb'],
                         'case_weight_lb': draft['case_weight_lb']}
    elif action == 'adjust':
        current = lot(api, cur, payload['lot_id'], payload.get('product_id'), lock)
        p = product(api, cur, current['product_id'], lock)
        products.append(p)
        # A3b hook (R2/R3): fixed-list reason, no covering an open shortage, review facts.
        reason = a3b.resolve_reason(cur, 'adjust', payload['reason_code'], note=payload.get('note'), delta_lb=payload['delta_lb'])
        a3b.refuse_cover_up(cur, 'adjust', p['id'], current['id'], payload['delta_lb'])
        req = api.AdjustRequest(**common, product_name=p['name'], lot_code=current['lot_code'],
            adjustment_lb=payload['delta_lb'], reason=reason['label_en'], reason_es=payload.get('reason_es') or reason['label_es'])
        api.validate_bilingual(req.reason, req.reason_es, 'reason')
        warning = api.check_private_label_merge(p['name'], p.get('label_type') or 'house', req.reason, req.adjustment_lb)
        if warning:
            fail('PRIVATE_LABEL_MERGE', warning)
        draft = {'product_id': p['id'], 'product_name': p['name'], 'lot_id': current['id'],
                 'lot_code': current['lot_code'], 'current_quantity_lb': current['on_hand_lb'],
                 'delta_lb': req.adjustment_lb, 'new_balance_lb': float(current['on_hand_lb']) + req.adjustment_lb,
                 'reason_code': reason['code'], 'reason': reason, 'reason_es': req.reason_es, 'note': reason['note'],
                 'attachment_ref': payload.get('attachment_ref'),
                 'correction_review': a3b.correction_review(api, 'adjust', p, req.adjustment_lb,
                                                            book_balance_before=current['on_hand_lb'])}
        if draft['new_balance_lb'] < 0:
            draft['balance_warning'] = f'{p["name"]} lot {current["lot_code"]} will have negative inventory ({draft["new_balance_lb"]} lb).'
        options = {'product': p, 'lot_id': payload['lot_id'], 'reason_code': reason['code'], 'note': reason['note']}
        specification = {'product_id': p['id'], 'lot_id': current['id'], 'delta_lb': req.adjustment_lb}
    elif action == 'found':
        p = product(api, cur, payload['product_id'], lock)
        products.append(p)
        # A3b hook (R2/R3): fixed-list reason (the note is `notes`), no found stock
        # for a product with an open shortage, 500 lb review.
        reason = a3b.resolve_reason(cur, 'found', payload['reason_code'], note=payload.get('notes'), delta_lb=payload['quantity'])
        a3b.refuse_cover_up(cur, 'found', p['id'], None, payload['quantity'])
        fields = {k: v for k, v in payload.items()
                  if k not in ('specification', 'input_plan', 'existing_output_lot_id', 'attachment_ref')}
        fields['reason_code'] = reason['code']
        req = api.AddFoundInventoryRequest(**fields)
        api.validate_bilingual(req.notes, req.notes_es, 'notes')
        output_code = req.lot_code
        if not output_code:
            prefix = api.get_plant_now().strftime('%y-%m-%d') + '-FOUND-'
            output_code = f'{prefix}{api.next_lot_sequence(cur, prefix + "%"):03d}'
        output_product = p
        draft = {'product_id': p['id'], 'product_name': p['name'], 'quantity': req.quantity,
                 'uom': req.uom, 'lot_code': output_code, 'reason_code': req.reason_code, 'reason': reason,
                 'notes': req.notes, 'notes_es': req.notes_es, 'found_location': req.found_location,
                 'estimated_age': req.estimated_age, 'suspected_supplier': req.suspected_supplier,
                 'attachment_ref': payload.get('attachment_ref'),
                 'correction_review': a3b.correction_review(api, 'found', p, req.quantity, uom=req.uom)}
        options = {'reason_code': reason['code']}
        specification = {'product_id': p['id'], 'quantity_lb': req.quantity}
    else:
        fail('TICKET_ACTION_UNAVAILABLE', 'Unsupported action.')

    # Product/formula changes that alter the promised quantities need a fresh draft.
    # Balance-only changes are allowed if all exact pinned inputs remain sufficient.
    specification = json.loads(json.dumps(specification, cls=api.DecimalSafeEncoder))
    if 'specification' in payload and specification != payload['specification']:
        fail('INPUT_PLAN_CHANGED', 'The product or recipe changed the draft quantities; prepare again.')
    states = []
    seen_products = {p['id'] for p in products}
    for pid in sorted(requirements):
        if pid not in seen_products:
            products.append(product(api, cur, pid, lock))
    totals = {}
    for item in sorted(input_plan or [], key=lambda i: i['lot_id']):
        current = lot(api, cur, item['lot_id'], item['product_id'], lock)
        totals[item['product_id']] = totals.get(item['product_id'], 0) + item['quantity_lb']
        states.append(current)
    if any(abs(totals.get(pid, 0) - qty) > api.BALANCE_EPSILON for pid, qty in requirements.items()):
        fail('INPUT_PLAN_CHANGED', 'The pinned input quantities do not match current requirements.')
    # A3b hook (R3): the shortfall is measured against the balances just read —
    # at commit under FOR UPDATE — and posted + flagged instead of blocking.
    if action in ('make', 'pack'):
        draft['shortages'] = a3b.shortfalls(input_plan, states)
    if action == 'adjust':
        states.append(current)
    if output_product:
        output_code = api.normalize_lot_code_input(output_code)
        api._validate_lot_code_twin(cur, output_product['id'], output_code)
        cur.execute('SELECT id FROM lots WHERE product_id=%s AND lot_code=%s', (output_product['id'], output_code))
        existing = cur.fetchone()
        if payload.get('existing_output_lot_id') and (not existing or existing['id'] != payload['existing_output_lot_id']):
            fail('LOT_IDENTITY_CHANGED', f'{output_product["name"]} lot {output_code} no longer matches the prepared lot; prepare again.')
        draft['lot_exists'] = bool(existing)
        draft['output_lot_id'] = existing['id'] if existing else None
        if existing:
            states.append(lot(api, cur, existing['id'], output_product['id'], lock))
        if action == 'pack':
            req.target_lot_code = output_code
        else:
            req.lot_code = output_code
    draft['input_plan'] = [dict(item, lot_code=next(s['lot_code'] for s in states if s['id'] == item['lot_id']))
                           for item in input_plan or []]
    # A5 hook: validate actual lot evidence without changing stock policy.
    if action in ('make', 'pack'):
        input_plan, draft['blockers'] = a5.validate_plan(
            api, cur, input_plan or [], states, payload.get('lot_confirmations'), commit=lock)
        draft['input_plan'] = input_plan
        options['input_plan'] = input_plan
    if action == 'found':
        a5.identity_draft(api, cur, payload, draft)
    state = {'products': products, 'lots': states, 'specification': specification}
    return draft, state, req, options, occurred_at, time_source, specification, input_plan


def post(api, cur, action, validated, payload, request, ticket_id, receipt_number, require_new_lot):
    if action == 'move_lot':
        return a5.post_move(api, cur, payload, request, ticket_id, receipt_number)
    draft, _, req, options, occurred_at, source, _, _ = validated
    if action == 'found':
        req.performed_by = api._operator_id(request)
    if action in ('make', 'pack', 'found'):
        options['require_new_lot'] = require_new_lot
    response = response_dict(getattr(api, f'_{action}_commit_core')(
        cur, req, request, occurred_at, source, ticket_id=ticket_id, receipt_number=receipt_number, **options))
    # A5 hook: evidence is atomic with the ledger and saved receipt.
    if action in ('make', 'pack'):
        actor = api.request_actor(request)
        a5.record_confirmations(cur, response['transaction_id'], draft['input_plan'], actor['id'] if actor else None)
        response['input_plan'] = draft['input_plan']
        if action == 'make':
            a5.record_substitutions(cur, response['transaction_id'], payload, actor['id'] if actor else None)
            response['substitutions'] = draft['substitutions']
            response['exclusion_reason_code'] = draft['exclusion_reason_code']
        # A3b hook (R3): shortage flag + SHORTAGE exception in the posting transaction.
        response['shortages'] = a3b.record_shortages(
            cur, response['transaction_id'], draft.get('shortages'), ticket_id=ticket_id,
            receipt_number=receipt_number, action=action, actor_id=actor['id'] if actor else None)
    if action == 'pack':
        response['lot_id'] = response['output_lot_id']
    elif action == 'found':
        a5.record_identity(cur, payload, response, ticket_id, receipt_number)
    elif action == 'adjust':
        response['lot_id'] = payload['lot_id']
    if action in ('adjust', 'found'):
        # A3b hook (R2): the review facts travel on the receipt for the weekly view.
        response['reason_code'] = draft['reason_code']
        response['correction_review'] = dict(draft['correction_review'], attachment_ref=draft.get('attachment_ref'))
    return response


def duplicates(cur, action, payload, draft):
    """Match effective posted ledger values, any actor, within the design window."""
    pid = payload.get('target_product_id') or payload.get('product_id') or draft.get('product_id')
    if action == 'make':
        quantity = draft.get('total_output_lb')
    elif action == 'pack':
        quantity = draft.get('total_lb')
    else:
        quantity = payload.get('delta_lb', payload.get('quantity'))
    if pid is None or quantity is None:
        return []
    # Match PostgreSQL numeric(14,4), including fractional yield calculations.
    quantity = Decimal(str(quantity)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
    kind = action if action in ('make', 'pack') else 'adjust'
    minutes = 45 if action in ('make', 'pack') else 120
    # Legacy packs have no cases column; their writer stores the immutable case
    # count in this anchored note prefix. Ticket packs have an explicit payload.
    cur.execute('''SELECT t.id,raw.receipt_number,t.operator_id,
                   EXTRACT(EPOCH FROM (clock_timestamp()-t.created_at))/60 AS minutes_ago
        FROM ledger_current_transactions t JOIN transactions raw ON raw.id=t.id
        LEFT JOIN write_tickets wt ON wt.id=raw.ticket_id
        WHERE t.type=%s AND t.effective_status='posted'
          AND t.created_at >= clock_timestamp()-%s*interval '1 minute'
          AND EXISTS (SELECT 1 FROM ledger_current_transaction_lines l
                      WHERE l.transaction_id=t.id AND l.product_id=%s AND (%s IS NULL OR l.lot_id=%s))
          AND ((%s='pack' AND COALESCE(wt.payload->>'cases',
                    substring(raw.notes FROM '^Pack ([0-9]+) cases of '))=%s)
            OR (%s<>'pack' AND (SELECT SUM(l.quantity_lb) FROM ledger_current_transaction_lines l
                WHERE l.transaction_id=t.id AND l.product_id=%s AND (%s IS NULL OR l.lot_id=%s))=%s))
        ORDER BY t.created_at DESC,t.id DESC LIMIT 1''',
        (kind, minutes, pid, payload.get('lot_id'), payload.get('lot_id'), action,
         str(payload.get('cases')), action, pid, payload.get('lot_id'), payload.get('lot_id'), quantity))
    row = cur.fetchone()
    if not row:
        return []
    cur.execute("""SELECT p.name,
                   array_agg(DISTINCT lot.lot_code ORDER BY lot.lot_code)
                       FILTER (WHERE lot.lot_code IS NOT NULL) AS lot_codes
                   FROM ledger_current_transaction_lines l JOIN products p ON p.id=l.product_id
                   LEFT JOIN lots lot ON lot.id=l.lot_id
                   WHERE l.transaction_id=%s AND l.product_id=%s GROUP BY p.name""", (row['id'], pid))
    previous = cur.fetchone()
    codes = ', '.join(previous['lot_codes'] or [])
    label = f'{previous["name"]} (lots {codes})' if codes else previous['name']
    ago = max(0, int(row['minutes_ago']))
    ref = row['receipt_number'] or f'transaction {row["id"]} (legacy; no receipt number)'
    return [{'code': 'POSSIBLE_DUPLICATE', 'requires_ack': True,
             'message': f'A matching {action} for {label} was posted {ago} minutes ago as {ref} by {row["operator_id"]}.',
             'message_es': f'Una operación similar para {label} se registró hace {ago} minutos como {ref} por {row["operator_id"]}.',
             'refs': {'transaction_id': row['id'], 'receipt_number': row['receipt_number'],
                      'operator_id': row['operator_id'], 'minutes_ago': ago}}]
