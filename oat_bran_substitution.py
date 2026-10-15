"""Michael's SS-only, pound-for-pound oat-bran fallback; never edits recipes."""
from decimal import Decimal
import json

from fastapi import HTTPException


SS_CODES = frozenset(('90016', '90011', '90013'))
REASON = 'oat_bran_auto_substitute'


def eligible(product):
    return product.get('type') == 'batch' and product.get('odoo_code') in SS_CODES


def identities(cur, product):
    if not eligible(product):
        return None
    cur.execute("SELECT id,name,odoo_code FROM products WHERE odoo_code=ANY(%s) "
                "AND type='ingredient' AND active IS DISTINCT FROM false", (['11031', '11032'],))
    rows = {r['odoo_code']: dict(r) for r in cur.fetchall()}
    return (rows['11031'], rows['11032']) if len(rows) == 2 else None


def lots(api, cur, product_id, lot_id=None, lot_code=None):
    cur.execute(f'''SELECT l.id,l.lot_code,COALESCE(SUM(t.quantity_lb),0) AS available
        FROM lots l LEFT JOIN {api.POSTED_LINES} t ON t.lot_id=l.id
        WHERE l.product_id=%s AND l.status IS DISTINCT FROM 'merged'
          AND (%s::integer IS NULL OR l.id=%s)
          AND (%s::text IS NULL OR LOWER(l.lot_code)=LOWER(%s))
        GROUP BY l.id HAVING COALESCE(SUM(t.quantity_lb),0)>0
        ORDER BY COALESCE(l.received_at,l.created_at),l.id''',
        (product_id, lot_id, lot_id, lot_code, lot_code))
    return [dict(r) for r in cur.fetchall()]


def messages(row, *, posted=False):
    extra, bran = f"{row['quantity_lb']:g}", f"{row['available_bran_lb']:g}"
    if row['available_bran_lb'] <= 0:
        en = f"No oat bran in stock — {'used' if posted else 'will use'} {extra} lb extra oats instead."
        es = f"No hay salvado de avena en inventario — {'se usaron' if posted else 'se usarán'} {extra} lb adicionales de avena en su lugar."
    else:
        en = f"Only {bran} lb oat bran in stock — {'used' if posted else 'will use'} it first, then {extra} lb extra oats instead."
        es = f"Solo hay {bran} lb de salvado de avena en inventario — {'se usaron' if posted else 'se usarán'} primero, y luego {extra} lb adicionales de avena en su lugar."
    return dict(row, message_en=en, message_es=es)


def plan(api, cur, product, ingredients, *, manual=(), excluded=(), overrides=None, override_codes=None, frozen=None):
    pair = identities(cur, product)
    if not pair:
        return []
    bran, oats = pair
    # A deliberate manual exclusion remains authoritative. Do not sneak the
    # excluded ingredient back into the run or invent consumption evidence.
    if oats['id'] in excluded:
        return []
    if any(s['ingredient_product_id'] == bran['id'] for s in manual):
        return []
    original = next((i for i in ingredients if i['ingredient_id'] == bran['id'] and not i.get('substituted_for')), None)
    if not original or original['needed_lb'] <= api.BALANCE_EPSILON:
        return []
    if frozen is not None:
        # This server-generated value is inside the signed ticket specification.
        for row in frozen:
            if (row['ingredient_product_id'] != bran['id'] or row['substitute_product_id'] != oats['id']
                    or abs(row['quantity_lb'] + row['available_bran_lb'] - original['needed_lb']) > api.BALANCE_EPSILON):
                raise HTTPException(409, {'error_code': 'INPUT_PLAN_CHANGED', 'message': 'The oat bran recipe changed; prepare again.'})
        return [messages(dict(row)) for row in frozen]
    stock = lots(api, cur, bran['id'], (overrides or {}).get(bran['id']),
                 (override_codes or {}).get(str(bran['id'])))
    available = float(api.to_decimal(min(float(original['needed_lb']), sum(float(l['available']) for l in stock))))
    missing = float(api.to_decimal(original['needed_lb']) - api.to_decimal(available))
    if missing <= api.BALANCE_EPSILON:
        return []
    return [messages({'ingredient_product_id': bran['id'], 'substitute_product_id': oats['id'],
        'ingredient_name': bran['name'], 'substitute_name': oats['name'], 'quantity_lb': missing,
        'available_bran_lb': available, 'automatic': True, 'reason_code': REASON})]


def apply_preview(api, cur, product, draft, *, manual=(), overrides=None, override_codes=None, frozen=None):
    rows = plan(api, cur, product, draft['ingredients'], manual=manual,
                excluded=[i['ingredient_id'] for i in draft.get('excluded_ingredients', [])], overrides=overrides,
                override_codes=override_codes, frozen=frozen)
    if not eligible(product):
        return rows
    draft['auto_substitutions'] = rows
    for row in rows:
        bran = next(i for i in draft['ingredients'] if i['ingredient_id'] == row['ingredient_product_id'] and not i.get('substituted_for'))
        bran['needed_lb'] = row['available_bran_lb']
        if bran['needed_lb'] <= api.BALANCE_EPSILON:
            draft['ingredients'].remove(bran)
        oats = next((i for i in draft['ingredients'] if i['ingredient_id'] == row['substitute_product_id']), None)
        if oats is None:
            oats = {'ingredient_id': row['substitute_product_id'], 'ingredient_name': row['substitute_name'], 'needed_lb': 0}
            draft['ingredients'].append(oats)
        oats['needed_lb'] = float(api.to_decimal(oats['needed_lb']) + api.to_decimal(row['quantity_lb']))
        for ingredient in (bran, oats):
            pid = ingredient['ingredient_id']
            stock = lots(api, cur, pid, (overrides or {}).get(pid), (override_codes or {}).get(str(pid)))
            available = sum(float(l['available']) for l in stock)
            ingredient.update(available_lb=available, sufficient=available >= ingredient['needed_lb'],
                lots=[{'lot_code': l['lot_code'], 'available_lb': float(l['available'])} for l in stock], lot_count=len(stock))
    draft['all_ingredients_available'] = all(i['sufficient'] for i in draft['ingredients'])
    if rows:
        draft['preview_message'] += ' ' + ' '.join(r['message_en'] for r in rows)
        draft['preview_message_es'] = ' '.join(r['message_es'] for r in rows)
    return rows


def formula(formula_rows, rows, batches):
    result = [dict(r) for r in formula_rows]
    for row in rows:
        bran = next(r for r in result if r['ingredient_product_id'] == row['ingredient_product_id'] and not r.get('exclude_from_inventory'))
        extra = Decimal(str(row['quantity_lb'])) / Decimal(str(batches))
        bran['quantity_lb'] = Decimal(str(bran['quantity_lb'])) - extra
        if bran['quantity_lb'] <= 0:
            result.remove(bran)
        oats = next((r for r in result if r['ingredient_product_id'] == row['substitute_product_id'] and not r.get('exclude_from_inventory')), None)
        if oats is None:
            oats = {'ingredient_product_id': row['substitute_product_id'], 'quantity_lb': Decimal('0'), 'exclude_from_inventory': False}
            result.append(oats)
        oats['quantity_lb'] = Decimal(str(oats['quantity_lb'])) + extra
    return result


def record(cur, transaction_id, rows, actor_id):
    """One A5 reason row; exact consumed oats lots remain in the immutable ledger."""
    for row in rows:
        cur.execute('''SELECT ingredient_lot_id AS lot_id,quantity_lb FROM ingredient_lot_consumption
            WHERE transaction_id=%s AND ingredient_product_id=%s ORDER BY ingredient_lot_id''',
            (transaction_id, row['substitute_product_id']))
        consumed = [dict(r, quantity_lb=float(r['quantity_lb'])) for r in cur.fetchall()]
        evidence = dict(messages(row, posted=True), consumed_oats_lots=consumed)
        cur.execute('''INSERT INTO transaction_substitutions
            (transaction_id,ingredient_product_id,substitute_product_id,lot_id,reason_code,note,actor_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s)''', (transaction_id, row['ingredient_product_id'],
            row['substitute_product_id'], consumed[0]['lot_id'], REASON, json.dumps(evidence, ensure_ascii=False), actor_id))
