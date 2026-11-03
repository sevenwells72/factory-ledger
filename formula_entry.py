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
CHANGED_CODE = 'INGREDIENT_CHANGED'
# Make-entry simplification (PR-1): a lot tapped in the screen's list is stored as
# transaction_lot_confirmations.method='selected', which migration 090 allows.
SELECTION_MARKER = '090_make_lot_selection'
ALREADY_RECORDED = 'MAKE_ALREADY_RECORDED'


def reasons():
    return [{'code': c, 'label_en': en, 'label_es': es, 'note_required': n} for c, en, es, n in REASONS]


def check_selections(cur, payload, actor):
    """A tapped lot ('selected') is evidence only from the Make screen and only for a
    named person; the commit records that person on transaction_lot_confirmations."""
    if not any(c.get('method') == 'selected' for c in payload.get('lot_confirmations') or []):
        return
    if not payload.get('formula_entry'):
        a5.fail('LOT_SELECTION_MAKE_SCREEN_ONLY', 'Choosing a lot from the list is only for the Make screen; type or scan the code.',
                'Elegir el lote de la lista es solo para la pantalla Producir; escribe o escanea el código.')
    if actor.get('id') is None:
        raise HTTPException(403, {'error_code': 'ACTOR_REQUIRED',
            'message': 'Sign in as yourself to choose lots: FL records who chose each lot.',
            'message_es': 'Inicia sesión con tu propia cuenta para elegir lotes: FL registra quién eligió cada lote.'})
    cur.execute('SELECT 1 FROM migration_markers WHERE name=%s', (SELECTION_MARKER,))
    if not cur.fetchone():
        raise HTTPException(503, {'error_code': 'LOT_SELECTION_UNAVAILABLE',
            'message': 'Lot selection is not switched on yet (migration 090); nothing was recorded. Tell the office.',
            'message_es': 'La selección de lotes aún no está activada (migración 090); no se registró nada. Avisa a la oficina.'})


def attempt_recorded(cur, attempt_id, *, exclude_ticket_id=None):
    """The make already recorded (or held for the owner) under this screen attempt.

    The Make screen sends one entry_attempt_id per form; a retry after a lost answer
    or an expired draft carries the same id, so FL can refuse a second post."""
    if not attempt_id:
        return None
    cur.execute("""SELECT id,receipt_number,status FROM write_tickets
        WHERE action='make' AND status IN ('committed','awaiting_approval')
          AND payload->>'entry_attempt_id'=%s AND id IS DISTINCT FROM %s
        ORDER BY id LIMIT 1""", (attempt_id, exclude_ticket_id))
    row = cur.fetchone()
    return dict(row) if row else None


def already_recorded(prior):
    receipt = prior.get('receipt_number')
    if prior['status'] == 'committed':
        en = f'This Make is already RECORDED as {receipt}. Nothing new was recorded. Open the receipt instead of recording again.'
        es = f'Esta producción ya está REGISTRADA como {receipt}. No se registró nada nuevo. Abre el recibo en vez de registrar otra vez.'
    else:
        en = 'This Make is already waiting for owner approval. Nothing new was recorded.'
        es = 'Esta producción ya espera la aprobación del dueño. No se registró nada nuevo.'
    return {'code': ALREADY_RECORDED, 'message': en, 'message_es': es,
            'receipt_number': receipt, 'status': prior['status']}


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


def _swap_lb(api, standard, actual, swap):
    """Pounds of rolled oats that really replaced oat bran, from what the worker entered.

    Bran short of the recipe and oats over the recipe: the overlap is the swap.
    """
    if not swap:
        return 0.0
    bran, oats = swap['from'], swap['to']
    gap = min(float(standard[bran]) - float(actual.get(bran, 0)), float(actual.get(oats, 0)) - float(standard.get(oats, 0)))
    return max(0.0, float(api.to_decimal(gap)))


def prepare(api, cur, payload, draft, requirements, overrides, substitutions, auto_substitutions, lock):
    """Apply the worker's ingredient lines to `requirements` in place.

    Returns the entry state: the recipe-standard pounds, the explicit bag plan
    for the lines given, per-ingredient reasons, the reason blockers and the
    oat-bran swap the entered pounds really make.
    """
    import ticket_actions as actions
    import oat_bran_substitution as oat_bran
    # Spec 3.2/3.8: the standard is recipe × pans. FL's automatic oat-bran swap
    # moved missing bran onto oats in `requirements`; undo it for the standard.
    standard, swap, proposed = dict(requirements), None, None
    for row in auto_substitutions:
        bran, oats = row['ingredient_product_id'], row['substitute_product_id']
        standard[bran] = float(api.to_decimal(row['quantity_lb']) + api.to_decimal(row['available_bran_lb']))
        standard[oats] = float(api.to_decimal(standard.get(oats, 0)) - api.to_decimal(row['quantity_lb']))
        swap, proposed = {'from': bran, 'to': oats}, row
    names = {i['ingredient_id']: i['ingredient_name'] for i in draft.get('original_ingredients', []) + draft['ingredients']}
    substituted = {s['substitute_product_id']: s for s in substitutions}
    blockers, plan, line_reasons, seen, zero = [], [], {}, set(), []
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
            # Identity and merge checks only; the commit locks every plan lot in lot-ID order.
            actions.lot(api, cur, bag['lot_id'], pid)
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
            zero.append((pid, line, name))
        if line.get('reason_code') or line.get('note'):
            line_reasons[pid] = {'reason_code': line.get('reason_code'), 'note': line.get('note')}
    swap_lb = _swap_lb(api, standard, requirements, swap)
    for pid, line, name in zero:
        # Oat bran fully replaced by oats is the automatic swap, not "not used".
        if swap and pid == swap['from'] and swap_lb > api.BALANCE_EPSILON:
            line_reasons.pop(pid, None)
            continue
        # A line the recipe does not use (e.g. rolled oats offered only for the
        # automatic oat-bran swap) is not zeroed by the worker: no reason.
        if float(standard.get(pid, 0)) <= api.BALANCE_EPSILON:
            line_reasons.pop(pid, None)
            continue
        problem = _reason_blocker(line.get('reason_code'), line.get('note'), name)
        if problem:
            blockers.append(problem)
    if standard and all(qty <= api.BALANCE_EPSILON for qty in requirements.values()):
        blockers.append({'code': 'NO_INGREDIENTS_USED', 'message': 'Record at least one ingredient used.',
                         'message_es': 'Registra al menos un ingrediente usado.'})
    for pid in seen:
        overrides.pop(pid, None)
    if lock and blockers:
        first = blockers[0]
        raise HTTPException(422, {'error_code': first['code'], 'message': first['message'],
                                  'message_es': first['message_es']})
    # What posts: the swap the entered pounds make (none when the worker used the bran).
    auto_rows = []
    if swap and swap_lb > api.BALANCE_EPSILON:
        auto_rows = [oat_bran.messages(dict(proposed, quantity_lb=swap_lb,
                                            available_bran_lb=float(requirements.get(swap['from'], 0)), entered=True))]
    if swap:
        swap['lb'] = swap_lb
    return {'standard': standard, 'plan': plan, 'line_pids': seen, 'reasons': line_reasons,
            'blockers': blockers, 'swap': swap, 'auto_rows': auto_rows}


def review(standard, actual, substitutions, reasons_by_pid, names, swap=None):
    """Per-ingredient facts for the owner's weekly view (spec 3.8, D6).

    With the oat-bran swap, oat bran + the swapped oats are compared with the
    recipe bran, and oats less the swapped pounds with the recipe oats.
    """
    import oat_bran_substitution as oat_bran
    substituted = {s['substitute_product_id']: s for s in substitutions}
    swap_lb = float(swap['lb']) if swap else 0.0
    rows = []
    for pid, std in sorted(standard.items()):
        used = float(actual.get(pid, 0))
        std = float(std)
        role = ('from' if swap and pid == swap['from'] else 'to' if swap and pid == swap['to'] else None)
        checked = used + swap_lb if role == 'from' else used - swap_lb if role == 'to' else used
        diff = checked - std
        pct = round(diff / std * 100, 1) if std > 0 else None
        over = abs(diff) > TOLERANCE * std + 1e-9 if std > 0 else checked > 0.005
        sub = substituted.get(pid)
        swapped = role == 'from' and swap_lb > 0
        kind = ('substituted' if sub else 'not_used' if used <= 0 and std > 0 and not swapped else
                'off_recipe' if over else 'swap' if swapped else None)
        if not kind:
            continue
        reason = reasons_by_pid.get(pid) or {}
        rows.append({'product_id': pid, 'product_name': names.get(pid), 'kind': kind,
                     'standard_lb': round(std, 2), 'used_lb': round(used, 2), 'checked_lb': round(checked, 2),
                     'difference_pct': pct,
                     'reason_code': sub['reason_code'] if sub else oat_bran.REASON if swapped else reason.get('reason_code'),
                     'note': sub.get('note') if sub else None if swapped else reason.get('note'),
                     'substituted_for': sub['ingredient_product_id'] if sub else None,
                     'swap_lb': round(swap_lb, 2) if role and swap_lb > 0 else None,
                     'swapped_to': names.get(swap['to']) if swapped else None,
                     'weekly_view': True})
    return rows


def supplier_lot(row):
    """What the lot list shows as the supplier lot. Pooled stock never names one
    supplier lot; a label without a lot code says so explicitly."""
    if (row.get('lot_type') or '') == 'commingled':
        return {'kind': 'pooled', 'code': None}
    code = (row.get('supplier_lot_code') or '').strip()
    if not code:
        return {'kind': 'not_recorded', 'code': None}   # never claim "no lot on label" for a blank
    if code.upper() in ('N/A', 'NA', 'UNKNOWN'):
        return {'kind': 'no_lot', 'code': None}
    return {'kind': 'supplier', 'code': code}


def draft_block(api, cur, product, batches, entry, requirements, input_plan, substitutions, auto_substitutions):
    """Renderer-neutral ingredient cards; bags are suggestions until confirmed."""
    standard, swap = entry['standard'], entry.get('swap')
    names = {}
    pids = sorted(set(standard) | {item['product_id'] for item in input_plan})
    if pids:
        cur.execute('SELECT id,name FROM products WHERE id=ANY(%s)', (pids,))
        names = {r['id']: r['name'] for r in cur.fetchall()}
    # Lots to choose from: book stock > 0. A lot already in this draft keeps its full
    # identity even at zero or negative book stock (shortage policy), never "available".
    candidates, planned_lots = {}, {}
    planned = sorted({item['lot_id'] for item in input_plan})
    if pids:
        cur.execute(f'''SELECT l.product_id,l.id AS lot_id,l.lot_code,l.supplier_lot_code,l.lot_type,
                (COALESCE(l.received_at,l.created_at) AT TIME ZONE 'America/New_York')::date AS received_on,
                COALESCE(sum(tl.quantity_lb),0) AS book_lb
            FROM lots l LEFT JOIN {api.POSTED_LINES} tl ON tl.lot_id=l.id
            WHERE l.product_id=ANY(%s) AND l.status IS DISTINCT FROM 'merged'
            GROUP BY l.id HAVING COALESCE(sum(tl.quantity_lb),0)>0 OR l.id=ANY(%s)
            ORDER BY l.product_id,COALESCE(l.received_at,l.created_at),l.id''', (pids, planned))
        for row in cur.fetchall():
            book = float(row['book_lb'])
            detail = {'lot_id': row['lot_id'], 'lot_code': row['lot_code'],
                      'supplier_lot': supplier_lot(row),
                      'received_on': row['received_on'].isoformat() if row['received_on'] else None,
                      'book_lb': round(book, 2), 'available_lb': round(max(book, 0.0), 2)}
            if book > 0:
                candidates.setdefault(row['product_id'], []).append(detail)
            else:
                planned_lots.setdefault(row['product_id'], []).append(detail)
    substituted = {s['substitute_product_id']: s for s in substitutions}
    rows = review(standard, requirements, substitutions, entry['reasons'], names, swap)
    flagged = {r['product_id']: r for r in rows}
    swap_lb = float(swap['lb']) if swap else 0.0
    ingredients = []
    for pid in sorted(standard):
        std, used = float(standard[pid]), float(requirements.get(pid, 0))
        role = ('from' if swap and pid == swap['from'] else 'to' if swap and pid == swap['to'] else None)
        checked = used + swap_lb if role == 'from' else used - swap_lb if role == 'to' else used
        reason = entry['reasons'].get(pid) or {}
        bags = [{'lot_id': i['lot_id'], 'lot_code': i['lot_code'], 'quantity_lb': round(float(i['quantity_lb']), 2),
                 'confirmed': bool(i.get('confirmed')), 'different_lot': bool(i.get('lot_exception')),
                 'confirmation_method': (i.get('confirmation') or {}).get('method')}
                for i in input_plan if i['product_id'] == pid]
        ingredients.append({
            'product_id': pid, 'product_name': names.get(pid), 'standard_lb': round(std, 2),
            'entered_lb': round(used, 2), 'checked_lb': round(checked, 2),
            'difference_pct': round((checked - std) / std * 100, 1) if std > 0 else None,
            'over_tolerance': flagged.get(pid, {}).get('kind') == 'off_recipe',
            'not_used': (used <= api.BALANCE_EPSILON and std > api.BALANCE_EPSILON
                         and not (role == 'from' and swap_lb > api.BALANCE_EPSILON)),
            'reason_code': reason.get('reason_code'), 'note': reason.get('note'),
            'substituted_for': substituted[pid]['ingredient_product_id'] if pid in substituted else None,
            'auto_substitute': role == 'to', 'swap_role': role, 'swap_lb': round(swap_lb, 2) if role else None,
            'bags': bags, 'candidates': candidates.get(pid, []),
            # Draft lots with no book stock: shown with their identity, not offered as stock.
            'unstocked_lots': planned_lots.get(pid, [])})
    pan_lb = float(product.get('default_batch_lb') or 0)
    block_swap = None
    if swap:
        block_swap = {'from_product_id': swap['from'], 'to_product_id': swap['to'], 'from_name': names.get(swap['from']),
                      'to_name': names.get(swap['to']), 'swap_lb': round(swap_lb, 2)}
    return {'pans': batches, 'pan_lb': pan_lb, 'output_lb': float(api.to_decimal(pan_lb * batches)),
            'ingredients': ingredients, 'review': rows, 'reasons': reasons(), 'tolerance_pct': TOLERANCE * 100,
            'swap': block_swap}


def reason_label(code, lang='en'):
    import oat_bran_substitution as oat_bran
    if code == oat_bran.REASON:
        return 'cambio automático: falta salvado de avena' if lang == 'es' else 'automatic swap: oat bran short'
    for c, en, es, _ in REASONS:
        if c == code:
            return es if lang == 'es' else en
    return code or ('sin motivo' if lang == 'es' else 'no reason')


def describe(row, lang='en'):
    """One line per review row, shared by the draft warnings and the shift summary."""
    es = lang == 'es'
    recipe = 'receta' if es else 'recipe'
    name = row['product_name'] or ('ingrediente' if es else 'ingredient')
    pct = row['difference_pct']
    pct_text = '' if pct is None else f", {'+' if pct > 0 else ''}{pct:g}%"
    reason = ''
    if row.get('reason_code'):
        reason = f" — {'motivo' if es else 'reason'}: {reason_label(row['reason_code'], lang)}"
        if row.get('note'):
            reason += f" ({row['note']})"
    if row.get('swap_lb') and row.get('swapped_to'):
        swap_word = 'sustitución' if es else 'swap'
        return (f"{name} → {row['swapped_to']} ({swap_word}): {row['used_lb']:g} lb + {row['swap_lb']:g} lb"
                f" = {row['checked_lb']:g} lb ({recipe} {row['standard_lb']:g} lb{pct_text}){reason}")
    if row['kind'] == 'not_used':
        return f"{name} {'no se usó' if es else 'not used'} (0 lb; {recipe} {row['standard_lb']:g} lb){reason}"
    if row['kind'] == 'substituted':
        return f"{name} {'sustituto' if es else 'swapped in'}: {row['used_lb']:g} lb ({recipe} {row['standard_lb']:g} lb){reason}"
    swapped = ''
    if row.get('swap_lb'):
        swapped = (f"; {row['swap_lb']:g} lb {'reemplazan salvado de avena' if es else 'replace oat bran'}")
    return f"{name} {row['used_lb']:g} lb ({recipe} {row['standard_lb']:g} lb{pct_text}{swapped}){reason}"


def warnings(draft):
    """One acknowledgement for every ingredient more than 10% off recipe; zeroed and
    swapped lines are listed without a second acknowledgement (spec 3.8)."""
    block = draft.get('formula_entry') or {}
    off = [r for r in block.get('review', []) if r['kind'] == 'off_recipe']
    changed = [r for r in block.get('review', []) if r['kind'] in ('not_used', 'substituted', 'swap')]
    result = []
    if off:
        result.append({'code': WARNING_CODE, 'requires_ack': True,
             'message': 'More than 10% off the recipe — the owner will see this on the weekly review: '
                        + '; '.join(describe(r, 'en') for r in off) + '.',
             'message_es': 'Más de 10% diferente de la receta — el dueño lo verá en la revisión semanal: '
                           + '; '.join(describe(r, 'es') for r in off) + '.',
             'refs': {'ingredients': [r['product_id'] for r in off]}})
    if changed:
        result.append({'code': CHANGED_CODE, 'requires_ack': False,
             'message': 'Not used or swapped — the owner will see this on the weekly review: '
                        + '; '.join(describe(r, 'en') for r in changed) + '.',
             'message_es': 'No usado o sustituido — el dueño lo verá en la revisión semanal: '
                           + '; '.join(describe(r, 'es') for r in changed) + '.',
             'refs': {'ingredients': [r['product_id'] for r in changed]}})
    return result


def record(cur, transaction_id, block, actor_id):
    """Not-used ingredients keep their reason on the run (never on the recipe)."""
    for row in block['ingredients']:
        if row['not_used'] and row['standard_lb'] > 0:
            cur.execute('''INSERT INTO transaction_substitutions
                (transaction_id,ingredient_product_id,substitute_product_id,lot_id,reason_code,note,actor_id)
                VALUES (%s,%s,NULL,NULL,%s,%s,%s)''',
                (transaction_id, row['product_id'], row['reason_code'], row['note'], actor_id))
