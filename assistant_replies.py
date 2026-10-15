"""Deterministic bilingual answers built only from FL's returned data."""
import re
import unicodedata


def language(text, previous='en'):
    if re.search(r'[¿¡]', text):
        return 'es'
    words = unicodedata.normalize('NFKD', text.lower())
    words = ''.join(c for c in words if not unicodedata.combining(c))
    if re.search(r"\b(hola|registre|registramos|entradas|hoy|puedes|puede|donde|quien|cual|cantidad|recibio|recibieron|totalmente|solo una prueba|calcula|registres|guardes|ordenes|abiertas|fabricamos|existencias|avena|miel|recibimos|recibi|recibir|hicimos|hice|empacamos|empacado|empaque|enviamos|enviar|cuanto|cuanta|cuantos|cuales|que pedidos|pedidos abiertos|muestra|mostrar|semillas|girasol|libras|cajas|lote|inventario|tenemos|hay|ajusta|ajustar|ninguno|otro)\b", words):
        return 'es'
    if re.search(r"\b(what|which|how|received|receive|made|make|packed|pack|ship|orders|inventory|lot|show|adjust)\b", words):
        return 'en'
    return previous


def is_today_query(text):
    """Read-only day review intents, including the immediate voice follow-up."""
    words = unicodedata.normalize('NFKD', text.lower().replace('’', "'"))
    words = ''.join(c for c in words if not unicodedata.combining(c))
    words = ' '.join(re.sub(r"[^a-z0-9' ]+", ' ', words).split())
    english = (r"(?:(?:open|show|list)(?: me)?(?: all)?(?: of)? (?:today'?s entries|entries (?:for |from )?today)"
               r"|what (?:did|have) (?:i|we) (?:enter(?:ed)?|record(?:ed)?) today|today'?s entries)"
               r"(?: is everything there correct)?")
    spanish = (r"(?:(?:abre|abrir|mostrar|muestra|ver|lista)(?: todas)?(?: las)? (?:entradas|registros) de hoy"
               r"|(?:entradas|registros) de hoy|que (?:registre|registramos|he registrado) hoy)"
               r"(?: esta todo correcto)?")
    return bool(re.fullmatch(english + '|' + spanish + '|is everything there correct|esta todo(?: ahi)? correcto', words))



def number(value):
    return format(float(value or 0), ',.4f').rstrip('0').rstrip('.')


def read_card(tool, result, lang):
    es = lang == 'es'
    rows = []
    if tool == 'inventory_lookup':
        rows = result.get('results', [])
        answers = [f"{r['product']}: {number(r.get('total_on_hand'))} lb " +
                   (f"en {len(r.get('lots', []))} " + ('lote' if len(r.get('lots', [])) == 1 else 'lotes') if es else f"in {len(r.get('lots', []))} " + ('lot' if len(r.get('lots', [])) == 1 else 'lots')) for r in rows]
        summary = '; '.join(answers) + '.' if answers else (
            f"No encontré inventario para «{result.get('query', '')}»." if es else
            f"I found no inventory for “{result.get('query', '')}”.")
        if any(r.get('unverified_lots') for r in rows):
            summary += (' Hay lotes sin verificar, pendientes de seguimiento.' if es else
                        ' Some lots are unverified, awaiting follow-up.')
    elif tool == 'lot_lookup':
        rows = [result]
        if result.get('unverified'):
            summary = (f"Lote {result['lot_code']}: sin verificar, pendiente de seguimiento." if es else
                       f"Lot {result['lot_code']}: unverified, awaiting follow-up.")
        else:
            summary = (f"Lote {result['lot_code']}: {number(result.get('quantity_on_hand'))} lb de {result['product_name']}." if es else
                   f"Lot {result['lot_code']}: {number(result.get('quantity_on_hand'))} lb of {result['product_name']}.")
    elif tool == 'open_orders':
        rows = result.get('orders', [])
        n = len(rows)
        summary = (("Hay 1 pedido abierto" if n == 1 else f"Hay {n} pedidos abiertos") if es else
                   ("There is 1 open order" if n == 1 else f"There are {n} open orders"))
        if result.get('customer'):
            summary += (" para " if es else " for ") + result['customer']
        if result.get('product'):
            summary += (" con " if es else " containing ") + result['product']
        summary += (" (primeros 200)." if es else " (first 200).") if result.get('truncated') else '.'
    elif tool == 'today_entries':
        rows = result.get('receipts', result.get('entries', []))
        own, team = result.get('own_count', len(rows)), result.get('team_count')
        if team is None:
            summary = (f"Hoy registró {own} entradas." if es else f"You recorded {own} entries today.")
        elif es:
            summary = f"Hoy registró {own} " + ('entrada' if own == 1 else 'entradas') + f"; el equipo registró {team}."
        else:
            summary = f"You recorded {own} " + ('entry' if own == 1 else 'entries') + f" today; the team recorded {team}."

    elif tool == 'receipt_lookup':
        rows = [result]
        receipt = result.get('receipt_number', '')
        status = result.get('effective_status') or result.get('status', '')
        state = (('ANULADO' if es else 'VOIDED') if result.get('voided') or status == 'voided' else
                 ('PARCIALMENTE ANULADO' if es else 'PARTIALLY VOIDED') if status == 'partially_voided' else
                 ('registrado' if es else 'recorded'))
        summary = f"{'Recibo' if es else 'Receipt'} {receipt}: {state}."
    elif tool == 'shift_summary':
        rows = result.get('sections', result.get('receipts', result.get('entries', [])))
        if isinstance(rows, dict):
            rows = list(rows.values())
        count = result.get('receipts_today', 0)
        summary = f'Su turno de hoy tiene {count} recibos.' if es else f'Your shift today has {count} receipts.'
    else:
        rows = result.get('candidates', [])
        summary = 'Estos son los resultados de Factory Ledger.' if es else 'Here are the Factory Ledger results.'
    return {'kind': 'read', 'tool': tool, 'status': 200, 'summary': summary,
            'message': summary, 'language': lang, 'rows': rows, 'result': result}


ERRORS = {
    'PHYSICAL_LOT_KOSHER_REVIEW_REQUIRED': ('This physical lot needs review before it can be used for this product.', 'Este lote físico necesita revisión antes de usarse para este producto.'),
    'ORDER_ALREADY_FULFILLED': ('This order is already fully shipped.', 'Este pedido ya está completamente enviado.'),
    'ORDER_NOT_FOUND': ('I could not find that order. Give its order number.', 'No encontré ese pedido. Escriba su número de pedido.'),
    'LOT_NOT_FOUND': ('I could not find that lot. Give the code on its tag.', 'No encontré ese lote. Escriba el código de su etiqueta.'),
    'PRODUCT_NOT_FOUND': ('I could not find that product. Give its name, product number or SKU.', 'No encontré ese producto. Escriba su nombre, número de producto o SKU.'),
    'RECEIVE_CORRECTION_UNAVAILABLE': ('This draft can no longer be changed. Look up its receipt to request a correction.', 'Este borrador ya no se puede cambiar. Consulte su recibo para solicitar una corrección.'),
    'SUPPLIER_REQUIRED': ('Choose the supplier that delivered this ingredient.', 'Elija el proveedor que entregó este ingrediente.'),
    'LOT_CONFIRMATION_MISMATCH': ('That tag does not match the suggested lot. Check the tag or choose Different lot.', 'Esa etiqueta no coincide con el lote sugerido. Revise la etiqueta o elija Otro lote.'),
    'LOT_NOT_CONFIRMED': ('Type the last 4 characters from the tag on each bag you used.', 'Escriba los últimos 4 caracteres de la etiqueta de cada bolsa que usó.'),
    'AMBIGUOUS_SUFFIX': ('That tag ending matches more than one lot. Enter the full tag code.', 'Esa terminación coincide con más de un lote. Escriba el código completo de la etiqueta.'),
    'BOL_REQUIRED': ('Enter the BOL number before recording.', 'Escriba el número de BOL antes de registrar.'),
    'SKU_CONFIRMATION_REQUIRED': ('Choose the product you made.', 'Elija el producto que elaboró.'),
}
FIELD_LABELS = {
    'lot_id': ('lot tag', 'etiqueta del lote'), 'quantity': ('quantity', 'cantidad'),
    'quantity_lb': ('quantity (lb)', 'cantidad (lb)'), 'delta_lb': ('change in pounds', 'cambio en libras'),
    'reason_code': ('reason', 'motivo'), 'note': ('note', 'nota'), 'notes': ('notes', 'notas'),
    'customer_id': ('customer', 'cliente'), 'order_id': ('order number', 'número de pedido'),
    'occurred_at': ('when it happened', 'cuándo ocurrió'), 'case_weight_lb': ('case size (lb)', 'peso por caja (lb)'),
    'supplier_id': ('supplier', 'proveedor'), 'product_id': ('product', 'producto'),
    'cases': ('number of cases', 'número de cajas'), 'case_size_lb': ('case size (lb)', 'peso por caja (lb)'),
    'bol_reference': ('BOL number', 'número de BOL'), 'source_product_id': ('source batch', 'lote de producción de origen'),
    'target_product_id': ('finished product', 'producto terminado'), 'batches': ('number of batches', 'número de tandas'),
}


def decorate(card, lang):
    """Add renderable copy; retain FL's structured evidence unchanged."""
    es = lang == 'es'
    card = dict(card, language=lang)
    kind = card['kind']
    if kind == 'unsupported_request':
        card['message'] = 'Solo puedo registrar o consultar elementos de Factory Ledger.' if es else 'I can only record or look up Factory Ledger items.'
    elif kind == 'clarify':
        card['message'] = ('No reconocí «' + card.get('query', '') + '». Escriba el nombre, número de producto o SKU.' if es else
                           'I did not recognize “' + card.get('query', '') + '”. Give the name, product number or SKU.')
        if card.get('resolution_kind') == 'lot':
            card['message'] = (f"No hay un lote llamado {card.get('query', '')}." if es else f"No lot called {card.get('query', '')}.")
    elif kind == 'choices':
        card['message'] = '¿Quiso decir una de estas opciones?' if es else 'Did you mean one of these?'
        if card.get('result', {}).get('outcome') == 'none' and card.get('query'):
            card['message'] = (f'No reconocí «{card["query"]}». ' if es else f'I did not recognize “{card["query"]}”. ') + card['message']
        result = dict(card['result'])
        result['ask'] = card['message']
        # Catalog names remain the identities printed on physical labels.
        card['result'] = result
    elif kind == 'missing_facts':
        card['message'] = 'Agrega los datos que faltan.' if es else 'Please add the missing details.'
        card['missing_details'] = [{'field': f, 'label': FIELD_LABELS.get(f, (f.replace('_', ' '), f.replace('_', ' ')))[es]} for f in card['fields']]
        if card.get('known_facts', {}).get('quantity_lb') is not None:
            qty = number(card['known_facts']['quantity_lb'])
            card['summary'] = f"Conservo la cantidad: {qty} lb. " + card['message'] if es else f"I kept the quantity: {qty} lb. " + card['message']
        if card.get('quantity_mismatch_lb') is not None:
            total = number(card['quantity_mismatch_lb'])
            given = number(card['known_facts']['quantity_lb'])
            card['summary'] = (f'Las cajas suman {total} lb; indicó {given} lb. Revise las cajas y el peso por caja.' if es else
                               f'The cases total {total} lb; you gave {given} lb. Check the cases and case size.')
    elif kind in ('refusal', 'lookup_failed', 'not_recorded'):
        result = card.get('result') or {}
        detail = result.get('detail', result)
        if not isinstance(detail, dict):
            detail = {}
        code = detail.get('error_code') or detail.get('code') or (result.get('error_detail') or {}).get('code')
        if card.get('status', 0) >= 500:
            card['message'] = ('Factory Ledger no está disponible temporalmente — inténtelo de nuevo en un minuto.' if es else
                               'Factory Ledger is temporarily unavailable — try again in a minute.')
            card['error_code'] = code
            return card
        card['message'] = ERRORS.get(code, ('I could not complete that Factory Ledger request. Check the item and details.', 'No pude completar esa solicitud de Factory Ledger. Revise el artículo y los datos.'))[es]
        if code == 'LOT_NOT_FOUND' and card.get('query'):
            card['message'] = (f"No hay un lote llamado {card['query']}." if es else f"No lot called {card['query']}.")
        card['error_code'] = code
    elif kind == 'misunderstood':
        card['message'] = 'No entendí eso. Escriba el artículo y lo que necesita hacer.' if es else 'I did not understand that. Give the item and what you need to do.'
    elif kind == 'draft':
        prepared = dict(card['prepared'])
        for field in ('blockers', 'warnings'):
            values = []
            for b in prepared.get(field, []):
                b = dict(b)
                b['message'] = b.get('message_es') if es and b.get('message_es') else ERRORS.get(b.get('code'), (b.get('message', ''), 'Revise los datos de este borrador.'))[es]
                values.append(b)
            prepared[field] = values
        draft = dict(prepared.get('draft', {}))
        group = draft.get('lot_confirmation_group')
        if group:
            group = dict(group, instruction=group['instruction_es' if es else 'instruction_en'])
            group['ingredients'] = [dict(i, different_lot=dict(i['different_lot'],
                label=i['different_lot']['label_es' if es else 'label_en'])) for i in group['ingredients']]
            draft['lot_confirmation_group'] = group
        prepared['draft'] = draft
        card['prepared'] = prepared
    return card
