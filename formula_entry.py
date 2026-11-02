"""Formula-driven make entry, Phase A (docs/production/formula-driven-entry.md).

The worker enters pans; FL proposes each ingredient's bags with the existing
oldest-first-then-next plan (ticket_actions.choose_inputs). The worker may edit
pounds per bag, split across bags, or record an ingredient as not used with a
fixed reason. The master recipe is never changed and nothing posts before the
ticket commit. No schema change: review facts travel on the ticket receipt.
Only requests with `formula_entry: true` (the Make screen) use this module.
"""
from fastapi import HTTPException

import lot_confirmation as a5

# Blubber, 2026-11-01 (spec D3). The note is required only for 'other'.
REASONS = (
    ('out_of_stock', 'Out of stock', 'Sin existencias', False),
    ('not_used', 'Not used', 'No se usó', False),
    ('substituted', 'Substituted', 'Sustituido', False),
    ('quality_problem', 'Quality problem', 'Problema de calidad', False),
    ('owner_instruction', 'Owner instruction', 'Instrucción del dueño', False),
    ('other', 'Other', 'Otro', True),
)
CODES = tuple(r[0] for r in REASONS)
# Spec D5: per ingredient only; exactly 10% is not a warning.
TOLERANCE = 0.10
WARNING_CODE = 'INGREDIENT_OFF_RECIPE'


def reasons():
    return [{'code': c, 'label_en': en, 'label_es': es, 'note_required': n} for c, en, es, n in REASONS]


def _reason_blocker(reason_code, note, name):
    if reason_code not in CODES:
        return {'code': 'MAKE_REASON_REQUIRED',
                'message': f'Choose a reason for {name}.',
                'message_es': f'Elige un motivo para {name}.'}
    if reason_code == 'other' and not (note or '').strip():
        return {'code': 'MAKE_REASON_NOTE_REQUIRED',
                'message': f'Write a note for {name}: the reason is Other.',
                'message_es': f'Escribe una nota para {name}: el motivo es Otro.'}
    return None


def prepare(api, cur, payload, draft, requirements, overrides, substitutions, auto_substitutions, lock):
    """Apply the worker's ingredient lines to `requirements` in place.

    Returns the entry state: the recipe-standard pounds, the explicit bag plan
    for the lines given, per-ingredient reasons and the reason blockers.
    """
    import ticket_actions as actions
    standard = dict(requirements)
    names = {i['ingredient_id']: i['ingredient_name'] for i in draft['ingredients']}
    substituted = {s['substitute_product_id']: s for s in substitutions}
    bran_swapped = {row['ingredient_product_id'] for row in auto_substitutions}
    blockers, plan, line_reasons, seen = [], [], {}, set()
    for sub in payload.get('substitutions') or []:
        problem = _reason_blocker(sub.get('reason_code'), sub.get('note'),
                                  names.get(sub['substitute_product_id'], 'the substitution'))
        if problem:
            blockers.append(problem)
    for line in payload.get('ingredient_lines') or []:
        pid = line['ingredient_product_id']
        if pid not in standard:
            a5.fail('UNUSED_INGREDIENT_LINE', 'An ingredient line is not part of this batch; prepare again.',
                    'Una línea de ingrediente no es parte de este lote; prepara de nuevo.')
        if pid in seen:
            a5.fail('DUPLICATE_INGREDIENT_LINE', 'Send each ingredient once.', 'Envía cada ingrediente una sola vez.')
        seen.add(pid)
        lots, total = set(), 0.0
        for bag in line.get('bags') or []:
            if bag['lot_id'] in lots:
                a5.fail('DUPLICATE_BAG', 'Choose each bag once per ingredient.', 'Elige cada bolsa una sola vez por ingrediente.')
            lots.add(bag['lot_id'])
            actions.lot(api, cur, bag['lot_id'], pid, lock)   # right product, not merged
            quantity = float(api.to_decimal(bag['quantity_lb']))
            plan.append({'product_id': pid, 'lot_id': bag['lot_id'], 'quantity_lb': quantity})
            total += quantity
        total = float(api.to_decimal(total))
        requirements[pid] = total
        name = names.get(pid, f'ingredient {pid}')
        if total <= api.BALANCE_EPSILON:
            if pid in substituted:
                a5.fail('INVALID_INGREDIENT_LINE', f'Remove the swap instead of using 0 lb of {name}.',
                        f'Quita la sustitución en vez de usar 0 lb de {name}.')
            if pid in bran_swapped:
                a5.fail('INVALID_INGREDIENT_LINE', f'{name} is already replaced by oats automatically; prepare again.',
                        f'{name} ya se reemplaza por avena automáticamente; prepara de nuevo.')
            problem = _reason_blocker(line.get('reason_code'), line.get('note'), name)
            if problem:
                blockers.append(problem)
        if line.get('reason_code') or line.get('note'):
            line_reasons[pid] = {'reason_code': line.get('reason_code'), 'note': line.get('note')}
    if standard and all(qty <= api.BALANCE_EPSILON for qty in requirements.values()):
        blockers.append({'code': 'NO_INGREDIENTS_USED', 'message': 'Record at least one ingredient used.',
                         'message_es': 'Registra al menos un ingrediente usado.'})
    for pid in seen:
        overrides.pop(pid, None)
    if lock and blockers:
        first = blockers[0]
        raise HTTPException(422, {'error_code': first['code'], 'message': first['message'],
                                  'message_es': first['message_es']})
    return {'standard': standard, 'plan': plan, 'line_pids': seen, 'reasons': line_reasons,
            'blockers': blockers}


def review(standard, actual, substitutions, reasons_by_pid, names):
    """Per-ingredient facts for the owner's weekly view (spec 3.8, D6)."""
    substituted = {s['substitute_product_id']: s for s in substitutions}
    rows = []
    for pid, std in sorted(standard.items()):
        used = float(actual.get(pid, 0))
        std = float(std)
        diff = used - std
        pct = round(diff / std * 100, 1) if std > 0 else None
        over = std > 0 and abs(diff) > TOLERANCE * std + 1e-9
        sub = substituted.get(pid)
        kind = ('substituted' if sub else 'not_used' if used <= 0 and std > 0 else
                'off_recipe' if over else None)
        if not kind:
            continue
        reason = reasons_by_pid.get(pid) or {}
        rows.append({'product_id': pid, 'product_name': names.get(pid), 'kind': kind,
                     'standard_lb': round(std, 2), 'used_lb': round(used, 2), 'difference_pct': pct,
                     'reason_code': sub['reason_code'] if sub else reason.get('reason_code'),
                     'note': sub.get('note') if sub else reason.get('note'),
                     'substituted_for': sub['ingredient_product_id'] if sub else None,
                     'weekly_view': True})
    return rows


def draft_block(api, cur, product, batches, entry, requirements, input_plan, substitutions, auto_substitutions):
    """Renderer-neutral ingredient cards; bags are suggestions until confirmed."""
    standard = entry['standard']
    names = {}
    pids = sorted(set(standard) | {item['product_id'] for item in input_plan})
    if pids:
        cur.execute('SELECT id,name FROM products WHERE id=ANY(%s)', (pids,))
        names = {r['id']: r['name'] for r in cur.fetchall()}
    candidates = {}
    if pids:
        cur.execute(f'''SELECT l.product_id,l.id AS lot_id,l.lot_code,
                (COALESCE(l.received_at,l.created_at) AT TIME ZONE 'America/New_York')::date AS received_on,
                COALESCE(sum(tl.quantity_lb),0) AS available_lb
            FROM lots l LEFT JOIN {api.POSTED_LINES} tl ON tl.lot_id=l.id
            WHERE l.product_id=ANY(%s) AND l.status IS DISTINCT FROM 'merged'
            GROUP BY l.id HAVING COALESCE(sum(tl.quantity_lb),0)>0
            ORDER BY l.product_id,COALESCE(l.received_at,l.created_at),l.id''', (pids,))
        for row in cur.fetchall():
            candidates.setdefault(row['product_id'], []).append(
                {'lot_id': row['lot_id'], 'lot_code': row['lot_code'],
                 'received_on': row['received_on'].isoformat() if row['received_on'] else None,
                 'available_lb': float(row['available_lb'])})
    substituted = {s['substitute_product_id']: s for s in substitutions}
    auto = {row['substitute_product_id']: row for row in auto_substitutions}
    rows = review(standard, requirements, substitutions, entry['reasons'], names)
    flagged = {r['product_id']: r for r in rows}
    ingredients = []
    for pid in sorted(standard):
        std, used = float(standard[pid]), float(requirements.get(pid, 0))
        reason = entry['reasons'].get(pid) or {}
        bags = [{'lot_id': i['lot_id'], 'lot_code': i['lot_code'], 'quantity_lb': round(float(i['quantity_lb']), 2),
                 'confirmed': bool(i.get('confirmed')), 'different_lot': bool(i.get('lot_exception'))}
                for i in input_plan if i['product_id'] == pid]
        ingredients.append({
            'product_id': pid, 'product_name': names.get(pid), 'standard_lb': round(std, 2),
            'entered_lb': round(used, 2),
            'difference_pct': round((used - std) / std * 100, 1) if std > 0 else None,
            'over_tolerance': flagged.get(pid, {}).get('kind') == 'off_recipe',
            'not_used': used <= api.BALANCE_EPSILON,
            'reason_code': reason.get('reason_code'), 'note': reason.get('note'),
            'substituted_for': substituted[pid]['ingredient_product_id'] if pid in substituted else None,
            'auto_substitute': pid in auto, 'bags': bags, 'candidates': candidates.get(pid, [])})
    pan_lb = float(product.get('default_batch_lb') or 0)
    return {'pans': batches, 'pan_lb': pan_lb, 'output_lb': float(api.to_decimal(pan_lb * batches)),
            'ingredients': ingredients, 'review': rows, 'reasons': reasons(), 'tolerance_pct': TOLERANCE * 100}


def warnings(draft):
    """One acknowledgement for every ingredient more than 10% off recipe."""
    block = draft.get('formula_entry') or {}
    off = [r for r in block.get('review', []) if r['kind'] == 'off_recipe']
    if not off:
        return []
    def describe(row, lang):
        sign = '+' if row['difference_pct'] > 0 else ''
        return f"{row['product_name']} {row['used_lb']:g} lb ({'receta' if lang == 'es' else 'recipe'} {row['standard_lb']:g} lb, {sign}{row['difference_pct']:g}%)"
    return [{'code': WARNING_CODE, 'requires_ack': True,
             'message': 'More than 10% off the recipe — the owner will see this on the weekly review: '
                        + '; '.join(describe(r, 'en') for r in off) + '.',
             'message_es': 'Más de 10% diferente de la receta — el dueño lo verá en la revisión semanal: '
                           + '; '.join(describe(r, 'es') for r in off) + '.',
             'refs': {'ingredients': [r['product_id'] for r in off]}}]


def record(cur, transaction_id, block, actor_id):
    """Not-used ingredients keep their reason on the run (never on the recipe)."""
    for row in block['ingredients']:
        if row['not_used'] and row['standard_lb'] > 0:
            cur.execute('''INSERT INTO transaction_substitutions
                (transaction_id,ingredient_product_id,substitute_product_id,lot_id,reason_code,note,actor_id)
                VALUES (%s,%s,NULL,NULL,%s,%s,%s)''',
                (transaction_id, row['product_id'], row['reason_code'], row['note'], actor_id))
