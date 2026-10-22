"""A5 lot evidence policy. All helpers use the caller's transaction/cursor.

No stock-shortage policy lives here. Client evidence is always checked against
server-selected lot IDs; it cannot replace the frozen input plan.
"""
from datetime import datetime
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, StrictBool, conint, constr


class LotConfirmation(BaseModel):
    lot_id: conint(strict=True, gt=0)
    method: Literal['last4', 'full_code', 'scan', 'pallet', 'supplier_lot', 'different_lot']
    value: constr(strict=True, strip_whitespace=True, min_length=1, max_length=200)
    selected: StrictBool = False

    class Config:
        extra = 'forbid'


def fail(code, message, message_es=None, **context):
    detail = {'error_code': code, 'message': message, **context}
    if message_es:
        detail['message_es'] = message_es
    raise HTTPException(422, detail)


OPENING_LABEL_EN = 'OPENING STOCK — use for any bag already here on Oct 21'
OPENING_LABEL_ES = 'EXISTENCIAS INICIALES — use para cualquier bolsa que ya estaba aquí el 21 de octubre'


def supplier_key(value):
    """The bag's supplier code is insensitive to case and all whitespace."""
    return ''.join((value or '').split()).casefold()


def real_supplier_code(value):
    """Missing-label placeholders are never physical supplier identities."""
    return (value or '').strip() if supplier_key(value) not in ('', 'unknown', 'n/a', 'na') else None


def stock_candidates(api, cur, product_id, *, kosher_tier=None):
    """Posted stock only; supplier aliases never multiply the stock balance."""
    cur.execute(f'''SELECT l.id AS lot_id,l.lot_code,l.supplier_lot_code,
            COALESCE(l.received_at,l.created_at) AS received_at,
            l.entry_source_notes,s.name AS supplier_name,
            aliases.supplier_lot_codes,aliases.supplier_names,
            COALESCE(balance.available_lb,0) AS available_lb
        FROM lots l LEFT JOIN suppliers s ON s.id=l.supplier_id
        LEFT JOIN LATERAL (
            SELECT array_agg(DISTINCT c.supplier_lot_code ORDER BY c.supplier_lot_code)
                       FILTER (WHERE btrim(COALESCE(c.supplier_lot_code,''))<>'') AS supplier_lot_codes,
                   string_agg(DISTINCT COALESCE(cs.name,c.supplier_name), ', '
                              ORDER BY COALESCE(cs.name,c.supplier_name)) AS supplier_names
            FROM lot_supplier_codes c LEFT JOIN suppliers cs ON cs.id=c.supplier_id
            WHERE c.lot_id=l.id) aliases ON true
        LEFT JOIN LATERAL (
            SELECT sum(tl.quantity_lb) AS available_lb FROM {api.POSTED_LINES} tl WHERE tl.lot_id=l.id
        ) balance ON true
        WHERE l.product_id=%s AND l.status IS DISTINCT FROM 'merged'
          AND (%s::text IS NULL OR l.kosher_tier=%s)
          AND COALESCE(balance.available_lb,0)>0
        ORDER BY COALESCE(l.received_at,l.created_at),l.id''', (product_id, kosher_tier, kosher_tier))
    rows = []
    for index, record in enumerate(cur.fetchall()):
        row = dict(record)
        aliases = [code for value in row.get('supplier_lot_codes') or [] if (code := real_supplier_code(value))]
        primary = real_supplier_code(row.get('supplier_lot_code'))
        row['supplier_lot_codes'] = list(dict.fromkeys(([primary] if primary else []) + aliases))
        row['supplier_lot_code'] = primary or (aliases[0] if aliases else None)
        row['supplier_name'] = row.get('supplier_name') or row.pop('supplier_names') or None
        row.pop('supplier_names', None)
        row['available_lb'] = float(row['available_lb'])
        row['received_at'] = row['received_at'].astimezone(api.PLANT_TIMEZONE).isoformat() if row['received_at'] else None
        row['opening_stock'] = (row.pop('entry_source_notes') or '').lower().startswith('opening stock')
        row['label_en'] = OPENING_LABEL_EN if row['opening_stock'] else row['supplier_lot_code'] or row['lot_code']
        row['label_es'] = OPENING_LABEL_ES if row['opening_stock'] else row['supplier_lot_code'] or row['lot_code']
        row['fifo_suggested'] = index == 0
        rows.append(row)
    return sorted(rows, key=lambda row: not row['opening_stock'])


def allocation_stock_blockers(api, cur, requirements, plan, states, confirmations, *, candidate_tiers=None):
    """Wrong-lot overdrafts refuse when positive product stock covers the need.

    A real product-wide shortage keeps the existing post-and-flag policy. The
    caller locks all input product lots before invoking this helper at commit.
    An explicitly unverified physical lot does not debit its FIFO placeholder.
    """
    lots = {row['id']: row for row in states}
    unknown = {row['lot_id'] for row in confirmations or [] if row['method'] == 'different_lot'}
    blockers = []
    for pid, needed in requirements.items():
        overdrawn = [row for row in plan if row['product_id'] == pid and row['lot_id'] not in unknown
                    and not row.get('lot_exception')
                    and float(row['quantity_lb']) > max(0, float(lots[row['lot_id']]['on_hand_lb'])) + api.BALANCE_EPSILON]
        if not overdrawn:
            continue
        candidates = stock_candidates(api, cur, pid, kosher_tier=(candidate_tiers or {}).get(pid))
        total = float(api.to_decimal(sum(row['available_lb'] for row in candidates)))
        if total + api.BALANCE_EPSILON < float(needed):
            continue
        name = lots[overdrawn[0]['lot_id']]['product_name']
        blockers.append({'code': 'LOT_ALLOCATION_EXCEEDS_STOCK',
            'message': f'{name} has enough stock in FL, but the selected lot does not have the entered lb. Pick the right lot or split the lb across lots.',
            'message_es': f'Hay suficientes existencias de {name} en FL, pero el lote seleccionado no tiene las lb indicadas. Elija el lote correcto o divida las lb entre lotes.',
            'product_id': pid, 'needed_lb': float(needed), 'total_available_lb': total,
            'candidates': candidates,
            'lots': [{'lot_id': row['lot_id'], 'quantity_lb': float(row['quantity_lb']),
                      'available_lb': max(0, float(lots[row['lot_id']]['on_hand_lb']))} for row in overdrawn]})
    return blockers


def verify_supplier_lot(api, cur, lot, evidence):
    candidates = stock_candidates(api, cur, lot['product_id'])
    key = supplier_key(evidence['value'])
    # Opening balances deliberately cover bags predating the reset without
    # inventing a supplier identity. This alias requires an active row choice.
    opening = next((row for row in candidates if row['lot_id'] == lot['id'] and row['opening_stock']), None)
    if opening and evidence.get('selected') and key in ('openingstock', 'existenciasiniciales'):
        return dict(evidence)
    matches = [row for row in candidates if any(supplier_key(code) == key for code in row['supplier_lot_codes'])]
    if not matches:
        fail('SUPPLIER_LOT_NOT_IN_STOCK',
             "That supplier lot isn't in stock in FL — if it's old stock already here on Oct 21, choose the OPENING STOCK lot",
             'Ese lote del proveedor no tiene existencias en FL — si son existencias antiguas que ya estaban aquí el 21 de octubre, elija el lote de EXISTENCIAS INICIALES',
             candidates=candidates, product_id=lot['product_id'])
    if len(matches) > 1 and not evidence.get('selected'):
        fail('AMBIGUOUS_SUPPLIER_LOT', 'That supplier lot matches several lots in stock. Select the lot you used.',
             'Ese lote del proveedor coincide con varios lotes con existencias. Seleccione el lote que usó.',
             candidates=matches, product_id=lot['product_id'])
    if lot['id'] not in {row['lot_id'] for row in matches}:
        fail('LOT_CONFIRMATION_MISMATCH', 'Select the matching supplier lot and prepare the draft again.',
             'Seleccione el lote del proveedor que coincide y prepare el borrador de nuevo.',
             candidates=matches, product_id=lot['product_id'])
    return dict(evidence)


def verify(api, cur, lot, evidence):
    value = evidence['value'].strip().upper()
    code = lot['lot_code'].upper()
    method = evidence['method']
    label = f"{lot['product_name']} lot {lot['lot_code']}"
    if method == 'supplier_lot':
        return verify_supplier_lot(api, cur, lot, evidence)
    if method == 'last4':
        if len(value) != 4 or value != code[-4:]:
            fail('LOT_CONFIRMATION_MISMATCH', f'Type the last four characters for {label}.',
                 f'Escriba los últimos cuatro caracteres del lote {lot["lot_code"]} de {lot["product_name"]}.')
        cur.execute(f'''SELECT l.id FROM lots l
            LEFT JOIN {api.POSTED_LINES} tl ON tl.lot_id=l.id
            WHERE l.product_id=%s AND l.status IS DISTINCT FROM 'merged'
              AND right(upper(l.lot_code),4)=%s
            GROUP BY l.id HAVING COALESCE(sum(tl.quantity_lb),0)>0''',
            (lot['product_id'], value))
        if len(cur.fetchall()) > 1:
            fail('AMBIGUOUS_SUFFIX', f'The suffix matches multiple active lots of {lot["product_name"]}; type the full code or scan.',
                 f'Esos cuatro caracteres coinciden con varios lotes de {lot["product_name"]}. Escriba el código completo de la etiqueta o escanéelo.')
    elif method in ('full_code', 'scan', 'pallet'):
        if value != code:
            fail('LOT_CONFIRMATION_MISMATCH', f'The entered code does not match {label}.',
                 f'El código no coincide con el lote {lot["lot_code"]} de {lot["product_name"]}. Revise la etiqueta.')
    if method == 'pallet':
        cur.execute('''SELECT id,to_location,lot_code,
                moved_at <= clock_timestamp() AND moved_at >= clock_timestamp()-interval '24 hours' AS recent
            FROM lot_moves WHERE lot_id=%s ORDER BY moved_at DESC,id DESC LIMIT 1''', (lot['id'],))
        move = cur.fetchone()
        if not move or move['to_location'] != 'production' or not move['recent'] or move['lot_code'] != lot['lot_code']:
            fail('PALLET_MOVE_REQUIRED', f'{label} needs a confirmed move to production within the last 24 hours.')
        return dict(evidence, move_id=move['id'])
    return dict(evidence)


def validate_plan(api, cur, plan, states, confirmations, *, commit=False, allow_different_lot=False,
                  different_lot_product_ids=None):
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
        entry = evidence.get(lot['id'])
        exception = None
        evidence_error = None
        if entry and entry['method'] == 'different_lot':
            if not allow_different_lot or (different_lot_product_ids is not None
                                           and lot['product_id'] not in different_lot_product_ids):
                fail('PHYSICAL_LOT_KOSHER_REVIEW_REQUIRED',
                     'This physical lot needs review before it can be used for this product. / Este lote físico necesita revisión antes de usarse para este producto.')
            exception = {'physical_tag': entry['value'].strip(), 'suggested_lot_id': lot['id'],
                         'suggested_lot_code': lot['lot_code'], 'kind': 'SHIFT_DISCREPANCY'}
        confirmed = None
        if entry and not exception:
            try:
                confirmed = verify(api, cur, lot, entry)
            except HTTPException as error:
                if commit or entry['method'] != 'supplier_lot':
                    raise
                # A mistyped/ambiguous supplier code must leave the stocked
                # choices on the prepare card so the worker can correct it.
                detail = dict(error.detail)
                evidence_error = dict(detail, code=detail['error_code'])
        row = dict(item, product_name=lot['product_name'], lot_code=lot['lot_code'], confirmed=confirmed is not None,
                   suggested_lot={'lot_id': lot['id'], 'lot_code': lot['lot_code'], 'confirmed': False})
        row.pop('lot_exception', None)
        if exception:
            row.pop('confirmation', None)
            row['lot_exception'] = exception
        elif confirmed:
            row['confirmation'] = confirmed
        else:
            row.pop('confirmation', None)
            blockers.append(evidence_error or {'code': 'LOT_NOT_CONFIRMED',
                'message': f'Select the supplier lot used for {lot["product_name"]}, or confirm its tag code or pallet move.',
                'message_es': f'Seleccione el lote del proveedor que usó para {lot["product_name"]}, o confirme el código de su etiqueta o el movimiento de tarima.'})
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
        if item.get('lot_exception'):
            continue  # An explicitly unknown physical lot is never a normal confirmation.
        evidence = item['confirmation']
        cur.execute('''INSERT INTO transaction_lot_confirmations
            (transaction_id,lot_id,method,value,actor_id,move_id)
            VALUES (%s,%s,%s,%s,%s,%s)''',
            (transaction_id,item['lot_id'],evidence['method'],evidence['value'],actor_id,evidence.get('move_id')))


def confirmation_group(api, cur, plan, *, candidate_tiers=None):
    """One additive, renderer-neutral card; FIFO suggestions never imply evidence."""
    grouped = {}
    for item in plan:
        ingredient = grouped.setdefault(item['product_id'], {
            'product_id': item['product_id'], 'product_name': item['product_name'],
            'needed_lb': 0.0, 'suggested_lots': [], 'candidates': [],
            'different_lot': {'method': 'different_lot', 'label_en': 'Different lot', 'label_es': 'Otro lote'}})
        ingredient['needed_lb'] += float(item['quantity_lb'])
        ingredient['suggested_lots'].append({key: item[key] for key in
            ('lot_id', 'lot_code', 'quantity_lb', 'confirmed')})
    for ingredient in grouped.values():
        ingredient['needed_lb'] = float(api.to_decimal(ingredient['needed_lb']))
        ingredient['candidates'] = stock_candidates(
            api, cur, ingredient['product_id'], kosher_tier=(candidate_tiers or {}).get(ingredient['product_id']))
        ingredient['total_available_lb'] = float(api.to_decimal(sum(row['available_lb'] for row in ingredient['candidates'])))
        ingredient['stock_shortage'] = ingredient['total_available_lb'] + api.BALANCE_EPSILON < ingredient['needed_lb']
    return {'kind': 'lot_confirmation',
            'instruction_en': 'Select the supplier lot on each bag you used. If you used more than one lot, split the needed lb between them. FIFO is a suggestion; select every lot yourself.',
            'instruction_es': 'Seleccione el lote del proveedor de cada bolsa que usó. Si usó más de un lote, divida las lb necesarias entre ellos. FIFO es una sugerencia; seleccione cada lote usted mismo.',
            'ingredients': list(grouped.values())}


def materialize_exception_inputs(api, cur, plan, ticket_id):
    """Debit a separate unresolved trace lot, never the unverified FIFO suggestion.

    The worker explicitly declared a different physical lot. Its identity is not
    invented or represented as confirmed. The actual tag and reviewed suggestion
    remain attached to the ledger input and an ordinary follow-up exception.
    This happens only in the caller's posting transaction, so failures undo it.
    """
    result = []
    for index, item in enumerate(plan):
        if not item.get('lot_exception'):
            result.append(item)
            continue
        detail = dict(item['lot_exception'])
        code = f'UNVERIFIED-{ticket_id}-P{item["product_id"]}-{index + 1}'
        lot_id, _, _ = api.find_or_create_lot(cur, item['product_id'], code, 'unverified_make_input',
            entry_source_notes=f'Physical tag: {detail["physical_tag"]}; suggested lot {detail["suggested_lot_code"]} was NOT used.',
            entry_source_notes_es=f'Etiqueta física: {detail["physical_tag"]}; NO se usó el lote sugerido {detail["suggested_lot_code"]}.')
        detail.update(lot_id=lot_id, lot_code=code)
        result.append(dict(item, lot_id=lot_id, lot_code=code, confirmed=False, lot_exception=detail))
    return result


def record_lot_exceptions(cur, transaction_id, plan, *, ticket_id, receipt_number, actor_id):
    """Use the existing discrepancy queue, with raw physical evidence retained."""
    from psycopg2.extras import Json
    import exceptions_enforcement as exceptions
    rows = []
    for item in plan:
        if not item.get('lot_exception'):
            continue
        detail = dict(item['lot_exception'], action='make', quantity_lb=float(item['quantity_lb']),
                      entered_by_actor_id=actor_id, follow_up='Identify the physical ingredient lot; do not assume the suggested FIFO lot.')
        cur.execute('''INSERT INTO exceptions(kind,severity,product_id,lot_id,transaction_id,receipt_number,
                ticket_id,detail,owner_actor_id)
            VALUES ('SHIFT_DISCREPANCY','warn',%s,%s,%s,%s,%s,%s,%s) RETURNING id''',
            (item['product_id'],item['lot_id'],transaction_id,receipt_number,ticket_id,Json(detail),
             exceptions.floor_owner_id(cur)))
        rows.append(dict(detail, product_id=item['product_id'], product_name=item['product_name'],
                         exception_id=cur.fetchone()['id']))
    return rows


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
        # The final input-product lock set is acquired once in ascending lot ID
        # order. Pre-locking this seed can invert that order against Pack.
        current = actions.lot(api, cur, sub['lot_id'], replacement)
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


def bind_substitution_lots(api, cur, draft, plan):
    """Allocate manual replacement demand across the actual consumed lots.

    Shared substitute products are pooled in the input plan. Charge their native
    recipe demand first, then each original ingredient in its reviewed order, so
    no consumed pound is claimed by two substitution audit rows.
    """
    changes = draft.get('substitutions') or []
    if not changes:
        return []
    originals = {row['ingredient_id']: row for row in draft['original_ingredients']}
    replacements = {row['substitute_product_id'] for row in changes}
    queues = {pid: [dict(row, remaining=api.to_decimal(row['quantity_lb']))
                    for row in plan if row['product_id'] == pid] for pid in replacements}

    def take(pid, needed):
        remaining, allocations = api.to_decimal(needed), []
        for row in queues[pid]:
            quantity = min(remaining, row['remaining'])
            if quantity <= 0:
                continue
            exception = row.get('lot_exception')
            # Before post, Different lot has only a physical tag, never a
            # confirmed suggestion. Post binds its new unresolved trace lot.
            actual_id = exception.get('lot_id') if exception else row['lot_id']
            actual_code = exception.get('lot_code') if exception else row['lot_code']
            allocation = {'lot_id': actual_id, 'lot_code': actual_code, 'quantity_lb': float(quantity)}
            if exception:
                allocation.update(physical_tag=exception['physical_tag'], unverified=True)
            allocations.append(allocation)
            row['remaining'] -= quantity
            remaining -= quantity
            if remaining <= 0:
                break
        if remaining > api.to_decimal(api.BALANCE_EPSILON):
            fail('INPUT_PLAN_CHANGED', 'The substitute lot quantities do not cover the reviewed recipe.')
        return allocations

    for ingredient in draft['ingredients']:
        if ingredient['ingredient_id'] in replacements and not ingredient.get('substituted_for'):
            take(ingredient['ingredient_id'], ingredient['needed_lb'])
    bound = []
    for change in changes:
        allocations = take(change['substitute_product_id'], originals[change['ingredient_product_id']]['needed_lb'])
        primary = allocations[0]
        bound.append(dict(change, lot_id=primary['lot_id'], lot_code=primary['lot_code'],
                          actual_lot_allocations=allocations))
        for ingredient in draft['ingredients']:
            if ingredient.get('substituted_for') == change['ingredient_product_id']:
                ingredient['override_lot'] = primary['lot_code']
                ingredient['lots'] = [{'lot_code': row['lot_code'],
                    'available_lb': float(api.lot_on_hand(cur, row['lot_id'])) if row['lot_id'] else 0,
                    'selected_lb': row['quantity_lb']} for row in allocations]
                ingredient['lot_count'] = len(allocations)
    draft['substitutions'] = bound
    return bound


def record_substitutions(cur, transaction_id, payload, actor_id):
    from psycopg2.extras import Json
    rows = list(payload.get('substitutions') or []) + [
        {'ingredient_product_id': pid, 'substitute_product_id': None, 'lot_id': None,
         'reason_code': payload['reason_code'], 'note': payload.get('note')}
        for pid in payload.get('excluded_ingredients') or []]
    for row in rows:
        cur.execute('''INSERT INTO transaction_substitutions
            (transaction_id,ingredient_product_id,substitute_product_id,lot_id,reason_code,note,actor_id,actual_lot_allocations)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)''',
            (transaction_id,row['ingredient_product_id'],row['substitute_product_id'],row['lot_id'],
             row['reason_code'],row.get('note'),actor_id,Json(row.get('actual_lot_allocations') or [])))


# R8 counts seven weekdays AFTER the local entry date; holidays are not skipped.
# A3b hook: the same clock serves shortages (2 days) — one helper, one rule.
def identification_deadline(entered_at):
    from exceptions_enforcement import business_deadline, BUSINESS_DAYS
    return business_deadline(entered_at, BUSINESS_DAYS['UNIDENTIFIED_LOT'])


def unidentified(payload):
    def missing(value):
        return (value or '').strip().upper() in ('', 'N/A', 'NA', 'UNKNOWN')
    entries = payload.get('supplier_lot_entries') or []
    if entries:
        return any(missing(e.get('supplier_lot_code')) for e in entries)
    return missing(payload.get('supplier_lot_code'))


def supplier_lot_identity_applies(cur, product_id):
    """Supplier-lot identity is only meaningful for purchased ingredients.

    The catalog has no supplier-sourced flag. Internal formula outputs, service
    items and ingredients used exclusively as inventory-excluded inputs (water)
    do not have supplier-traced lots. A new purchased ingredient needs no prior
    receipt or formula usage to qualify.
    """
    cur.execute('''SELECT p.type='ingredient' AND NOT p.is_service
        AND NOT COALESCE(p.has_bom, false)
        AND NOT EXISTS (SELECT 1 FROM batch_formulas f WHERE f.product_id=p.id)
        AND NOT (
            EXISTS (SELECT 1 FROM batch_formulas f WHERE f.ingredient_product_id=p.id
                    AND f.exclude_from_inventory IS TRUE)
            AND NOT EXISTS (SELECT 1 FROM batch_formulas f WHERE f.ingredient_product_id=p.id
                            AND f.exclude_from_inventory IS NOT TRUE)
        ) AS applies FROM products p WHERE p.id=%s''', (product_id,))
    row = cur.fetchone()
    return bool(row and row['applies'])


def require_receive_supplier_lot(cur, payload, product_id):
    """Purchased ingredient receipts need a label code or an explicit exception."""
    if (supplier_lot_identity_applies(cur, product_id) and unidentified(payload)
            and payload.get('no_supplier_lot') is not True):
        fail('SUPPLIER_LOT_REQUIRED',
             'Enter the supplier lot code from the label, or select "No lot on label" if the label has no lot code.',
             'Ingresa el lote del proveedor que aparece en la etiqueta, o selecciona "Sin lote en la etiqueta" si la etiqueta no tiene código de lote.')


def identity_draft(api, cur, payload, draft):
    if not supplier_lot_identity_applies(cur, draft['product_id']):
        return
    status = 'unidentified' if unidentified(payload) else 'identified'
    due = None
    lid = draft.get('existing_lot_id') or draft.get('output_lot_id')
    if lid:
        cur.execute('SELECT identity_status,identify_by,supplier_lot_code FROM lots WHERE id=%s', (lid,))
        prior = cur.fetchone()
        if prior['identity_status'] == 'unidentified':
            if status == 'identified':
                fail('LOT_IDENTITY_REQUIRES_CORRECTION', 'Identify this lot through a supplier-lot correction with a reason, not another receipt.')
            due = prior['identify_by']
        elif prior['identity_status'] == 'identified' and status == 'unidentified':
            fail('LOT_IDENTITY_MISMATCH', 'An unidentified delivery needs its own lot; this lot is already identified.')
        elif prior['identity_status'] == 'identified' and prior['supplier_lot_code'] != payload.get('supplier_lot_code'):
            fail('LOT_IDENTITY_MISMATCH', 'This delivery has a different supplier lot code; use a new lot.')
    if status == 'unidentified' and due is None:
        cur.execute('SELECT clock_timestamp() AS entered_at')
        due = identification_deadline(cur.fetchone()['entered_at']).date()
    draft.update(identity_status=status, identify_by=due,
                 identity_notice=f'UNIDENTIFIED — resolve by {due}' if due else None)


def record_identity(cur, payload, response, ticket_id, receipt_number):
    """One open clock per unidentified lot, anchored to ledger entry time."""
    from psycopg2.extras import Json
    cur.execute('SELECT id,product_id,identity_status,identify_by FROM lots WHERE id=%s FOR UPDATE', (response['lot_id'],))
    lot = cur.fetchone()
    if not supplier_lot_identity_applies(cur, lot['product_id']):
        return
    status = 'unidentified' if unidentified(payload) else 'identified'
    cur.execute('SELECT created_at FROM transactions WHERE id=%s', (response['transaction_id'],))
    entered = cur.fetchone()['created_at']
    due = identification_deadline(entered) if status == 'unidentified' else None
    if lot['identity_status'] is None:
        cur.execute('UPDATE lots SET identity_status=%s,identify_by=%s,supplier_lot_code=COALESCE(%s,supplier_lot_code) WHERE id=%s',
                    (status,due.date() if due else None,payload.get('supplier_lot_code'),lot['id']))
    else:
        status = lot['identity_status']
    exception = None
    if status == 'unidentified':
        cur.execute("SELECT id,due_at FROM exceptions WHERE kind='UNIDENTIFIED_LOT' AND lot_id=%s AND status IN ('open','escalated') ORDER BY id LIMIT 1", (lot['id'],))
        exception = cur.fetchone()
        if exception is None:
            cur.execute("""SELECT id FROM actors WHERE role='floor' AND active
                ORDER BY CASE WHEN lower(name)='arturo' THEN 0 ELSE 1 END,id LIMIT 1""")
            owner = cur.fetchone()
            cur.execute('''INSERT INTO exceptions(kind,severity,product_id,lot_id,transaction_id,receipt_number,
                    ticket_id,detail,owner_actor_id,opened_at,due_at)
                VALUES ('UNIDENTIFIED_LOT','warn',%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id,due_at''',
                (lot['product_id'],lot['id'],response['transaction_id'],receipt_number,ticket_id,
                 Json({'supplier_lot_code':payload.get('supplier_lot_code'), 'clock':'7 business days; Mon–Fri; entry date excluded; 23:59 America/New_York'}),
                 owner['id'] if owner else None,entered,due))
            exception = cur.fetchone()
        due = exception['due_at']
    from zoneinfo import ZoneInfo
    response.update(identity_status=status,identify_by=due.astimezone(ZoneInfo('America/New_York')).date() if due else None,
                    identification_due_at=due,exception_id=exception['id'] if exception else None)


def real_supplier(cur, supplier_id, *, lock=False):
    from resolution import SENTINEL_SUPPLIERS
    if supplier_id is None:
        fail('SUPPLIER_REQUIRED', 'Choose a real supplier through /resolve kind=supplier.')
    cur.execute('''SELECT id,name,active,short_code,
        btrim(supplier_name_norm(name)) = ANY(%s::text[]) AS pseudo
        FROM suppliers WHERE id=%s''' + (' FOR SHARE' if lock else ''),
        (list(SENTINEL_SUPPLIERS),supplier_id))
    supplier = cur.fetchone()
    if not supplier or supplier['active'] is False or supplier['pseudo']:
        fail('SUPPLIER_REQUIRED', 'Choose an active real supplier through /resolve kind=supplier; use found inventory for stock without a vendor.')
    # Nullable until the separate post-cleanup 066 backfill. The receive core
    # uses the resolved name's existing label convention when this is NULL;
    # provenance always comes from the selected ID, even for colliding labels.
    return supplier


def supplier_receipt(cur, response):
    cur.execute('''SELECT t.supplier_id,s.name AS supplier_name,t.bol_reference,
            (SELECT l.supplier_lot_code FROM transaction_lines tl JOIN lots l ON l.id=tl.lot_id
             WHERE tl.transaction_id=t.id AND tl.quantity_lb>0 ORDER BY tl.id LIMIT 1) AS supplier_lot_code
        FROM transactions t JOIN suppliers s ON s.id=t.supplier_id WHERE t.id=%s''', (response['transaction_id'],))
    response.update(dict(cur.fetchone()))


def receipt_evidence(cur, transactions, lots):
    """Read identities from FK columns, never from a lot label prefix."""
    for transaction in transactions:
        cur.execute('''SELECT t.supplier_id,s.name AS supplier_name,t.bol_reference FROM transactions t
            LEFT JOIN suppliers s ON s.id=t.supplier_id WHERE t.id=%s''', (transaction['id'],))
        transaction.update(dict(cur.fetchone()))
        cur.execute('SELECT lot_id,method,value,actor_id,move_id,created_at FROM transaction_lot_confirmations WHERE transaction_id=%s ORDER BY lot_id', (transaction['id'],))
        transaction['lot_confirmations'] = [dict(row) for row in cur.fetchall()]
        cur.execute('SELECT ingredient_product_id,substitute_product_id,lot_id,reason_code,note,actor_id FROM transaction_substitutions WHERE transaction_id=%s ORDER BY ingredient_product_id', (transaction['id'],))
        transaction['substitutions'] = [dict(row) for row in cur.fetchall()]
    for lot in lots:
        cur.execute('''SELECT l.supplier_id,s.name AS supplier_name,l.supplier_lot_code,l.identity_status,l.identify_by
            FROM lots l LEFT JOIN suppliers s ON s.id=l.supplier_id WHERE l.id=%s''', (lot['id'],))
        lot.update(dict(cur.fetchone()))
