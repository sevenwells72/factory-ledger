"""A5 lot evidence policy. All helpers use the caller's transaction/cursor.

No stock-shortage policy lives here. Client evidence is always checked against
server-selected lot IDs; it cannot replace the frozen input plan.
"""
from datetime import datetime
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, conint, constr


class LotConfirmation(BaseModel):
    lot_id: conint(strict=True, gt=0)
    method: Literal['last4', 'full_code', 'scan', 'pallet']
    value: constr(strict=True, strip_whitespace=True, min_length=1, max_length=200)

    class Config:
        extra = 'forbid'


def fail(code, message):
    raise HTTPException(422, {'error_code': code, 'message': message})


def verify(api, cur, lot, evidence):
    value = evidence['value'].strip().upper()
    code = lot['lot_code'].upper()
    method = evidence['method']
    label = f"{lot['product_name']} lot {lot['lot_code']}"
    if method == 'last4':
        if len(value) != 4 or value != code[-4:]:
            fail('LOT_CONFIRMATION_MISMATCH', f'Type the last four characters for {label}.')
        cur.execute(f'''SELECT l.id FROM lots l
            LEFT JOIN {api.POSTED_LINES} tl ON tl.lot_id=l.id
            WHERE l.product_id=%s AND l.status IS DISTINCT FROM 'merged'
              AND right(upper(l.lot_code),4)=%s
            GROUP BY l.id HAVING COALESCE(sum(tl.quantity_lb),0)>0''',
            (lot['product_id'], value))
        if len(cur.fetchall()) > 1:
            fail('AMBIGUOUS_SUFFIX', f'The suffix matches multiple active lots of {lot["product_name"]}; type the full code or scan.')
    elif method in ('full_code', 'scan', 'pallet'):
        if value != code:
            fail('LOT_CONFIRMATION_MISMATCH', f'The entered code does not match {label}.')
    if method == 'pallet':
        cur.execute('''SELECT id,to_location,lot_code,
                moved_at <= clock_timestamp() AND moved_at >= clock_timestamp()-interval '24 hours' AS recent
            FROM lot_moves WHERE lot_id=%s ORDER BY moved_at DESC,id DESC LIMIT 1''', (lot['id'],))
        move = cur.fetchone()
        if not move or move['to_location'] != 'production' or not move['recent'] or move['lot_code'] != lot['lot_code']:
            fail('PALLET_MOVE_REQUIRED', f'{label} needs a confirmed move to production within the last 24 hours.')
        return dict(evidence, move_id=move['id'])
    return dict(evidence)


def validate_plan(api, cur, plan, states, confirmations, *, commit=False):
    """Return annotated plan/blockers; prepare never pre-confirms a suggestion."""
    selected = {item['lot_id'] for item in plan}
    evidence = {}
    for entry in confirmations or []:
        if entry['lot_id'] in evidence:
            fail('DUPLICATE_LOT_CONFIRMATION', 'Supply one confirmation per lot.')
        if entry['lot_id'] not in selected:
            fail('UNUSED_LOT_CONFIRMATION', 'Confirm only the lots in this draft input plan.')
        evidence[entry['lot_id']] = entry
    lots = {lot['id']: lot for lot in states}
    result, blockers = [], []
    for item in plan:
        lot = lots[item['lot_id']]
        confirmed = verify(api, cur, lot, evidence[lot['id']]) if lot['id'] in evidence else None
        row = dict(item, lot_code=lot['lot_code'], confirmed=confirmed is not None,
                   suggested_lot={'lot_id': lot['id'], 'lot_code': lot['lot_code'], 'confirmed': False})
        if confirmed:
            row['confirmation'] = confirmed
        else:
            row.pop('confirmation', None)
            blockers.append({'code': 'LOT_NOT_CONFIRMED',
                'message': f'Confirm {lot["product_name"]} lot {lot["lot_code"]}: type last four characters, full code, or record a pallet move.'})
        result.append(row)
    if commit and blockers:
        fail('LOT_NOT_CONFIRMED', blockers[0]['message'])
    return result, blockers


def merge_confirmations(payload, additions):
    """Late evidence can fill/replace evidence only; quantity/identity is frozen."""
    entries = {e['lot_id']: e for e in payload.get('lot_confirmations') or []}
    seen = set()
    for entry in additions:
        if entry['lot_id'] in seen:
            fail('DUPLICATE_LOT_CONFIRMATION', 'Supply one confirmation per lot.')
        seen.add(entry['lot_id'])
        entries[entry['lot_id']] = entry
    return dict(payload, lot_confirmations=list(entries.values()))


def record_confirmations(cur, transaction_id, plan, actor_id):
    for item in plan:
        evidence = item['confirmation']
        cur.execute('''INSERT INTO transaction_lot_confirmations
            (transaction_id,lot_id,method,value,actor_id,move_id)
            VALUES (%s,%s,%s,%s,%s,%s)''',
            (transaction_id,item['lot_id'],evidence['method'],evidence['value'],actor_id,evidence.get('move_id')))


def validate_move(api, cur, payload, lock):
    import ticket_actions as actions
    occurred_at, source = api.validate_inventory_occurred_at(datetime.fromisoformat(payload['occurred_at']), payload['backfill'])
    current = actions.lot(api, cur, payload['lot_id'], lock=lock)
    actions.product(api, cur, current['product_id'], lock)
    if payload['method'] == 'pallet':
        fail('LOT_CONFIRMATION_MISMATCH', 'A pallet move must record the typed or scanned lot code.')
    evidence = verify(api, cur, current, {'lot_id': current['id'], 'method': payload['method'], 'value': payload['value']})
    spec = {'lot_id': current['id'], 'lot_code': current['lot_code'], 'to_location': payload['to_location']}
    if payload.get('specification', spec) != spec:
        fail('LOT_IDENTITY_CHANGED', 'The lot changed; record its current label in a fresh move draft.')
    draft = dict(spec, product_name=current['product_name'], confirmed=True, confirmation=evidence)
    return draft, spec, None, {}, occurred_at, source, spec, None


def post_move(api, cur, payload, request, ticket_id, receipt_number):
    actor = api.request_actor(request)
    cur.execute('''INSERT INTO lot_moves(lot_id,lot_code,to_location,moved_at,actor_id,ticket_id,method,value)
        SELECT id,lot_code,%s,%s,%s,%s,%s,%s FROM lots WHERE id=%s RETURNING id,lot_id,lot_code''',
        (payload['to_location'],payload['occurred_at'],actor['id'] if actor else None,ticket_id,
         payload['method'],payload['value'],payload['lot_id']))
    return _move_response(cur)


def _move_response(cur):
    row = dict(cur.fetchone())
    row['move_id'] = row.pop('id')
    return dict(row, success=True, confirmed=True)


class Substitution(BaseModel):
    ingredient_product_id: conint(strict=True, gt=0)
    substitute_product_id: conint(strict=True, gt=0)
    lot_id: conint(strict=True, gt=0)
    reason_code: constr(strict=True, strip_whitespace=True, min_length=1, max_length=80)
    note: constr(strict=True, strip_whitespace=True, max_length=2000) | None = None

    class Config:
        extra = 'forbid'


def prepare_substitutions(api, cur, payload, draft, overrides, lock):
    """Replace formula requirements explicitly, retaining original recipe evidence."""
    import ticket_actions as actions
    excluded = payload.get('excluded_ingredients') or []
    if len(set(excluded)) != len(excluded):
        fail('DUPLICATE_EXCLUSION', 'Exclude each ingredient only once.')
    if excluded and not (payload.get('reason_code') or '').strip():
        fail('SUBSTITUTION_REASON_REQUIRED', 'Manually excluded ingredients need a reason_code.')
    originals = {i['ingredient_id']: dict(i) for i in draft['ingredients']}
    all_ids = set(originals) | {i['ingredient_id'] for i in draft.get('excluded_ingredients', [])}
    if set(excluded) - all_ids:
        fail('INVALID_EXCLUSION', 'An excluded ingredient is not in this batch formula.')
    changes, seen = [], set()
    for sub in payload.get('substitutions') or []:
        original, replacement = sub['ingredient_product_id'], sub['substitute_product_id']
        if not (sub.get('reason_code') or '').strip():
            fail('SUBSTITUTION_REASON_REQUIRED', 'Every substitution needs a reason_code.')
        if original in seen or original not in originals or original == replacement or replacement == payload['product_id'] or replacement in (all_ids - set(originals)):
            fail('INVALID_SUBSTITUTION', 'Substitute each included formula ingredient once with a different product.')
        if original in overrides:
            fail('INVALID_SUBSTITUTION', 'Use the substitution lot instead of also overriding the original ingredient.')
        seen.add(original)
        product = actions.product(api, cur, replacement, lock)
        if product.get('is_service') or product['type'] not in ('ingredient', 'batch'):
            fail('INVALID_SUBSTITUTION', 'The substitute must be an active ingredient or batch product.')
        current = actions.lot(api, cur, sub['lot_id'], replacement, lock)
        if replacement in overrides and overrides[replacement] != sub['lot_id']:
            fail('INVALID_SUBSTITUTION', 'Select the same lot for repeated uses of a substitute ingredient.')
        overrides[replacement] = sub['lot_id']
        changes.append(dict(sub, ingredient_name=originals[original]['ingredient_name'],
                            substitute_name=product['name'], lot_code=current['lot_code']))
    by_id = {s['ingredient_product_id']: s for s in changes}
    draft['original_ingredients'] = list(originals.values())
    for ingredient in draft['ingredients']:
        sub = by_id.get(ingredient['ingredient_id'])
        if sub:
            ingredient.update(ingredient_id=sub['substitute_product_id'], ingredient_name=sub['substitute_name'],
                              substituted_for=sub['ingredient_product_id'], override_lot=sub['lot_code'])
            # Original-stock availability is not evidence about the substitute.
            available = float(api.lot_on_hand(cur, sub['lot_id']))
            ingredient.update(available_lb=available, sufficient=available >= ingredient['needed_lb'],
                              lots=[{'lot_code': sub['lot_code'], 'available_lb': available}], lot_count=1)
    draft['all_ingredients_available'] = all(i['sufficient'] for i in draft['ingredients'])
    draft['substitutions'] = changes
    draft['exclusion_reason_code'] = payload.get('reason_code') if excluded else None
    return changes


def substituted_formula(formula, substitutions):
    """Same one-for-one pounds as the validated draft, no recipe/master edits."""
    by_id = {s['ingredient_product_id']: s['substitute_product_id'] for s in substitutions}
    combined = {}
    for row in formula:
        row = dict(row)
        row['ingredient_product_id'] = by_id.get(row['ingredient_product_id'], row['ingredient_product_id'])
        key = (row['ingredient_product_id'], bool(row.get('exclude_from_inventory')))
        if key in combined:
            combined[key]['quantity_lb'] += row['quantity_lb']
        else:
            combined[key] = row
    return list(combined.values())


def record_substitutions(cur, transaction_id, payload, actor_id):
    rows = list(payload.get('substitutions') or []) + [
        {'ingredient_product_id': pid, 'substitute_product_id': None, 'lot_id': None,
         'reason_code': payload['reason_code'], 'note': payload.get('note')}
        for pid in payload.get('excluded_ingredients') or []]
    for row in rows:
        cur.execute('''INSERT INTO transaction_substitutions
            (transaction_id,ingredient_product_id,substitute_product_id,lot_id,reason_code,note,actor_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s)''',
            (transaction_id,row['ingredient_product_id'],row['substitute_product_id'],row['lot_id'],
             row['reason_code'],row.get('note'),actor_id))
