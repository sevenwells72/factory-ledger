"""Deterministic bilingual answers built only from FL's returned data."""
import re
import unicodedata


def language(text, previous='en'):
    if re.search(r'[¿¡]', text):
        return 'es'
    words = unicodedata.normalize('NFKD', text.lower())
    words = ''.join(c for c in words if not unicodedata.combining(c))
    if re.search(r"\b(hola|registre|registramos|entradas|hoy|puedes|puede|donde|quien|cual|cantidad|recibio|recibieron|totalmente|solo una prueba|calcula|registres|guardes|ordenes|abiertas|fabricamos|existencias|avena|miel|recibimos|recibi|recibir|hicimos|hice|empacamos|empacado|empaque|enviamos|enviar|envia|envie|pedido|cuanto|cuanta|cuantos|cuales|que pedidos|pedidos abiertos|muestra|mostrar|semillas|girasol|libras|cajas|lote|inventario|tenemos|hay|ajusta|ajustar|ninguno|otro)\b", words):
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
    return format(float(value or 0), ',.2f').rstrip('0').rstrip('.')


def order_remaining(order, es=False):
    quantities = []
    for item in order.get('items', []):
        pounds = float(item.get('remaining_lb') or 0)
        if item.get('is_service') or item.get('is_non_weight') or item.get('line_type') == 'service' or pounds <= 0:
            continue
        size = float(item.get('case_size_lb') or 0)
        quantity = f'{number(pounds)} lb'
        if size > 0:
            count = pounds / size
            unit = ('caja' if count == 1 else 'cajas') if es else ('case' if count == 1 else 'cases')
            quantity = f'{number(count)} {unit} ({quantity})'
        quantities.append(quantity)
    return '; '.join(quantities) or f"{number(order['remaining_lb'])} lb"



def include_zero_stock(text):
    """An explicit empty-stock request may include catalog rows at zero."""
    words = unicodedata.normalize('NFKD', text.lower())
    words = ''.join(c for c in words if not unicodedata.combining(c))
    return bool(re.search(r'\b(?:zero|0|empty|out of stock|cero|agotad[oa]s?|sin existencias|sin inventario)\b', words))


def read_card(tool, result, lang):
    es = lang == 'es'
    rows = []
    if tool == 'inventory_lookup':
        rows = result.get('results', [])
        answers = []
        for row in rows:
            quantity = f"{number(row.get('total_on_hand'))} lb"
            if row.get('unit_count') is not None:
                count = row['unit_count']
                unit = ('caja' if count == 1 else 'cajas') if es else ('case' if count == 1 else 'cases')
                quantity += f" ({number(count)} {unit})"
            lots = len(row.get('lots', []))
            unit = ('lote' if lots == 1 else 'lotes') if es else ('lot' if lots == 1 else 'lots')
            answers.append(f"{row['product']}: {quantity} " + ('en' if es else 'in') + f" {lots} {unit}")
        summary = '; '.join(answers) + '.' if answers else (
            (f"No hay existencias de «{result.get('query', '')}»." if es else
             f"There is no stock of “{result.get('query', '')}”.") if result.get('product_names') else
            (f"No encontré inventario para «{result.get('query', '')}»." if es else
             f"I found no inventory for “{result.get('query', '')}”."))
        family = result.get('stock_family')
        if family and rows:
            bulk = sum(float(r.get('total_on_hand') or 0) for r in rows if r['product'] == family['bulk_product'])
            finished = sum(float(r.get('total_on_hand') or 0) for r in rows if r['product'] != family['bulk_product'])
            parts = []
            if bulk or result.get('include_zero'):
                parts.append(f"{number(bulk)} lb " + ('a granel' if es else 'in bulk'))
            if finished or result.get('include_zero'):
                parts.append(f"{number(finished)} lb " + ('en productos terminados' if es else 'in finished products'))
            summary = family['bulk_product'] + ': ' + (' y ' if es else ' and ').join(parts) + '.'
        if result.get('unverified_evidence') or any(r.get('unverified_lots') for r in rows):
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
    'KOSHER_BATCH_RECEIVE_REFUSED': ('Extra-kosher batches must be made; they cannot be received.', 'Las tandas extra kosher deben elaborarse; no se pueden recibir.'),
    'KOSHER_ATTESTATION_REQUIRED': ('Extra-kosher stock needs approved packing or a found entry approved by the owner.', 'Las existencias extra kosher deben provenir de un empaque autorizado o de una entrada de producto encontrado aprobada por el dueño.'),
    'KOSHER_BATCH_PACK_REFUSED': ('Make this extra-kosher batch with owner approval; packing cannot create it.', 'Elabore esta tanda extra kosher con aprobación del dueño; un empaque no puede crearla.'),
    'KOSHER_SOURCE_REQUIRED': ('This product requires an extra-kosher source lot. Check the lot or make the required batch first.', 'Este producto requiere un lote de origen extra kosher. Revise el lote o elabore primero la tanda necesaria.'),
    'LOT_KOSHER_TIER_MISMATCH': ('This output lot has a different kosher tier. Use a new lot.', 'Este lote de salida tiene una clasificación kosher diferente. Use un lote nuevo.'),
    'KOSHER_NEW_BATCH_LOT_REQUIRED': ('This extra-kosher entry needs a new lot and owner approval.', 'Esta entrada extra kosher necesita un lote nuevo y la aprobación del dueño.'),
    'INSUFFICIENT_STOCK': ('There is not enough stock for this quantity. Receive the missing stock or change the quantity.', 'No hay existencias suficientes para esta cantidad. Registre la recepción faltante o cambie la cantidad.'),
    'ORDER_NOT_SHIPPABLE': ('Confirm this order in Office before shipping.', 'Confirme este pedido en Oficina antes de enviarlo.'),
    'ORDER_NOT_OPEN': ('This order is closed. Review it in Office.', 'Este pedido está cerrado. Revíselo en Oficina.'),
    'LINE_ALREADY_FULFILLED': ('This order line is already fully shipped.', 'Esta línea del pedido ya está completamente enviada.'),
    'LINE_CANCELLED': ('This order line was cancelled and cannot ship.', 'Esta línea del pedido fue cancelada y no se puede enviar.'),
    'QTY_EXCEEDS_REMAINING': ('The quantity must be positive and no greater than what remains on the order.', 'La cantidad debe ser positiva y no superar lo que queda pendiente del pedido.'),
    'PRODUCT_INACTIVE': ('This product is inactive. Ask the office which product to use.', 'Este producto está inactivo. Pregunte en la oficina qué producto debe usar.'),
    'CUSTOMER_INACTIVE': ('This customer is inactive. Ask the owner to reactivate it.', 'Este cliente está inactivo. Pida al dueño que lo reactive.'),
    'REASON_CODE_REQUIRED': ('Choose a correction reason.', 'Elija el motivo de la corrección.'),
    'REASON_CODE_INVALID': ('Choose one of the listed correction reasons.', 'Elija uno de los motivos de corrección de la lista.'),
    'CASE_SIZE_REQUIRED': ('Enter the weight of one case in pounds.', 'Escriba el peso de una caja en libras.'),
    'LOT_IDENTITY_CHANGED': ('This lot changed. Prepare a new draft using its current tag.', 'Este lote cambió. Prepare otro borrador con su etiqueta actual.'),
    'TICKET_STALE': ('This draft changed. Prepare it again using the current details.', 'Los datos de este borrador cambiaron. Prepárelo otra vez con los datos actuales.'),
    'FORBIDDEN': ('Your sign-in cannot record this action. Ask the office.', 'Su usuario no tiene permiso para registrar esta acción. Consulte en la oficina.'),
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

# Normal prepare/record refusals from the ledger's fixed error codes. The
# original English and all structured evidence stay on the card; these are
# translations of the returned reason, never a guessed inventory decision.
SPANISH_REASONS = {
    'LOT_MERGED': 'Este lote se unió a otro. Prepare el borrador con el lote que quedó activo.',
    'LOT_SUPPLIER_MISMATCH': 'Este lote pertenece a otro proveedor. Reciba la entrega en un lote nuevo.',
    'LOT_CODE_TAKEN': 'Ese código de lote ya está en uso. Prepare otro borrador con un código disponible.',
    'LOT_IDENTITY_REQUIRES_CORRECTION': 'Actualice el lote del proveedor mediante una corrección con motivo; otra recepción no cambia su identidad.',
    'LOT_IDENTITY_MISMATCH': 'La identidad de esta entrega no coincide con el lote elegido. Use un lote nuevo para esta entrega.',
    'EXPECTED_RECEIPT_NOT_OPEN': 'La entrega esperada ya no está abierta o ya no coincide con esta recepción. Revísela en Oficina.',
    'EXPECTED_RECEIPT_NOT_FOUND': 'No se encontró esa entrega esperada. Revise su número en Oficina.',
    'INVALID_OUTPUT': 'El producto no tiene un peso de tanda válido. Pida a la oficina que configure un peso mayor que cero.',
    'INVALID_PACK': 'El producto a granel y el producto terminado deben ser diferentes.',
    'INPUT_PLAN_CHANGED': 'Los ingredientes, lotes o cantidades cambiaron. Prepare otro borrador para revisar el plan actual.',
    'PRIVATE_LABEL_MERGE': 'No se puede fusionar ni retirar un producto de marca privada. Si hubo un reempaque físico, use el motivo de reempaque.',
    'RECIPE_MISSING': 'No hay una receta utilizable para descontar los ingredientes. Pida al dueño que restaure la receta.',
    'DUPLICATE_LOT_SELECTION': 'El mismo lote o ingrediente se eligió más de una vez. Elija cada uno una sola vez.',
    'UNUSED_LOT_SELECTION': 'Uno de los lotes elegidos no corresponde a los ingredientes de esta tanda.',
    'DUPLICATE_LOT_CONFIRMATION': 'Hay más de una confirmación para el mismo lote. Confirme cada lote una sola vez.',
    'UNUSED_LOT_CONFIRMATION': 'Solo debe confirmar los lotes que consume este borrador.',
    'PALLET_MOVE_REQUIRED': 'Este lote necesita un movimiento confirmado a producción dentro de las últimas 24 horas.',
    'DUPLICATE_EXCLUSION': 'El mismo ingrediente se excluyó más de una vez. Excluya cada ingrediente una sola vez.',
    'INVALID_EXCLUSION': 'El ingrediente excluido no aparece en la receta de esta tanda.',
    'SUBSTITUTION_REASON_REQUIRED': 'Escriba el motivo para sustituir o excluir un ingrediente.',
    'INVALID_SUBSTITUTION': 'La sustitución no coincide con los ingredientes de la receta o con el lote elegido. Revise la sustitución.',
    'ROLE_NOT_ALLOWED': 'Su usuario no tiene permiso para registrar esta acción. Pida ayuda a la oficina.',
    'ACTOR_INACTIVE': 'Este usuario ya no está activo. Pida al dueño que revise su acceso.',
    'ACTOR_REQUIRED': 'Inicie sesión con su usuario personal para registrar esta acción.',
    'BACKFILL_OWNER_ONLY': 'Solo el dueño puede registrar una entrada de hace más de 14 días.',
    'OCCURRED_AT_IN_FUTURE': 'La hora de la entrada está más de 5 minutos en el futuro. Corrija la fecha y la hora.',
    'OCCURRED_AT_BACKFILL_REQUIRED': 'La entrada es de hace más de 14 días. El dueño debe registrarla como entrada histórica.',
    'AMBIGUOUS_EVENT_TIME': 'Se indicaron dos fechas u horas para la misma entrada. Indique una sola.',
    'WARNING_NOT_ACKNOWLEDGED': 'Confirme las advertencias que aparecen en el borrador antes de registrar.',
    'OPEN_SALES_ORDER_EXISTS': 'Este cliente tiene pedidos abiertos. Confirme que este envío fuera de pedido es intencional.',
    'CUSTOMER_NOT_FOUND': 'No se encontró ese cliente. Elija un cliente activo.',
    'CUSTOMER_NAME_REQUIRED': 'Escriba el nombre del cliente.',
    'CUSTOMER_ALREADY_EXISTS': 'Ese cliente ya existe. Elija el cliente existente.',
    'RECEIPT_NOT_FOUND': 'No hay un recibo registrado con ese número. Revise el número del recibo.',
    'LINE_NOT_FOUND': 'Esa línea no pertenece al pedido elegido. Abra el pedido correcto.',
    'DUPLICATE_LINE': 'La misma línea del pedido aparece más de una vez. Incluya cada línea una sola vez.',
    'SERVICE_STANDALONE_UNSUPPORTED': 'Los cargos de servicio deben enviarse contra un pedido.',
    'SHIP_NOT_SAME_DAY': 'Este envío es de otro día de planta. Solo el dueño o el personal de piso pueden confirmarlo.',
    'SHIPMENT_REQUIRED': 'Registre el envío antes de cerrar el pedido como enviado.',
    'EXTERNAL_ORDER_REFERENCE_CONFLICT': 'La referencia externa ya corresponde a otro pedido con datos diferentes. Revísela en Oficina.',
    'EXTERNAL_ORDER_REFERENCE_EXISTS': 'Ya existe un pedido con esa referencia externa. Abra el pedido existente.',
    'QUANTITY_REQUIRED': 'Falta la cantidad y su unidad en una línea del pedido.',
    'INVALID_QUANTITY': 'La cantidad indicada no es válida para esta acción. Corrija la cantidad antes de registrar.',
    'AMBIGUOUS_REASON': 'Se indicaron dos motivos para la corrección. Elija un solo motivo de la lista.',
    'NOTE_REQUIRED': 'Este motivo necesita una nota que explique qué pasó.',
    'REASON_SIGN_MISMATCH': 'El motivo elegido no corresponde a agregar o quitar existencias. Elija el motivo correcto.',
    'SHORTAGE_CLOSE_REASON': 'Para cerrar un faltante debe registrar un conteo físico.',
    'COUNT_NEGATIVE': 'El conteo físico no puede dejar una cantidad negativa en el lote.',
    'SHORTAGE_SET_CHANGED': 'Los faltantes abiertos de este lote cambiaron. Prepare otro borrador.',
    'SHORTAGE_OPEN_RESOLVE_INSTEAD': 'Este lote tiene un faltante abierto. Resuélvalo en Excepciones en vez de cubrirlo con otra entrada.',
    'PHOTO_REQUIRED': 'Esta entrada necesita una foto de evidencia. Adjunte la foto correspondiente.',
    'AMBIGUOUS_ATTACHMENT': 'Se indicaron dos referencias para la misma foto. Elija una sola foto.',
    'UNUSED_ATTACHMENT': 'Esta foto no corresponde al tipo de entrada. Elija una foto del tipo solicitado.',
    'ATTACHMENT_NOT_FOUND': 'No se encontró la foto adjunta. Vuelva a adjuntarla.',
    'ATTACHMENT_KIND_INVALID': 'La foto adjunta no es del tipo requerido para esta entrada.',
    'ATTACHMENT_WRONG_USER': 'Esta foto pertenece a otro usuario. Adjunte una foto desde su propia sesión.',
    'ATTACHMENT_WRONG_ACTOR': 'Esta foto pertenece a otro usuario. Adjunte una foto desde su propia sesión.',
    'ATTACHMENT_WRONG_CONTEXT': 'Esta foto ya pertenece a otra entrada o excepción. Adjunte la foto correspondiente a esta entrada.',
    'ATTACHMENT_TYPE_INVALID': 'El archivo no es una foto del tipo requerido. Adjunte una foto JPEG, PNG o WebP.',
    'ATTACHMENT_SIZE_LIMIT': 'El archivo está vacío o supera el tamaño permitido. Adjunte un archivo más pequeño.',
    'ATTACHMENT_ALREADY_LINKED': 'Esta foto ya está vinculada a otra entrada. Adjunte una foto para esta entrada.',
    'TICKET_EXPIRED': 'Este borrador venció. Prepare otro borrador con los datos actuales.',
    'TICKET_NOT_COMMITTABLE': 'Este borrador ya no se puede registrar. Prepare otro borrador.',
    'TICKET_WRONG_USER': 'Solo la persona que preparó este borrador puede registrarlo.',
    'TICKET_PAYLOAD_MISMATCH': 'La confirmación no corresponde al borrador guardado. Abra el borrador actual.',
    'TICKET_NOT_FOUND': 'No se encontró este borrador. Prepare otro borrador.',
    'TICKET_ACTION_UNAVAILABLE': 'Esta acción no se puede registrar desde este borrador. Abra Oficina para revisarla.',
    'PRODUCT_AMBIGUOUS': 'Hay varios productos con ese nombre. Elija el producto exacto.',
    'PRODUCT_UNCERTAIN': 'No se pudo identificar el producto con certeza. Escriba su nombre completo o número.',
    'IDS_REQUIRED': 'Primero elija el producto, proveedor o cliente de las opciones de Factory Ledger.',
    'SELF_REPORTED_IDENTITY_REJECTED': 'La persona que registra debe ser quien inició sesión; no se puede indicar otra persona.',
}


def refusal_message(detail, es=False):
    """Keep specific FL reasons visible; structured Spanish wins over templates."""
    if not isinstance(detail, dict):
        detail = {'message': str(detail)}
    if detail.get('blockers'):
        return '; '.join(refusal_message(item, es) for item in detail['blockers'])
    code = detail.get('error_code') or detail.get('code')
    if es and detail.get('message_es'):
        return detail['message_es']
    if code == 'ORDER_ALREADY_FULFILLED' and detail.get('order_number'):
        return (f"El pedido {detail['order_number']} ya está completamente enviado." if es else
                f"Order {detail['order_number']} is already fully shipped.")
    if not es and detail.get('message'):
        return detail['message']
    raw = detail.get('message') or ''
    if es:
        shortage = re.match(r'(.+) needs ([\d.]+) lb and has no lot to record the shortage against; (make|receive) it first\.', raw)
        if shortage:
            step = 'Elabore primero la tanda.' if shortage[3] == 'make' else 'Registre primero la recepción.'
            return f'Se necesitan {number(shortage[2])} lb de {shortage[1]}, pero no hay un lote donde registrar el faltante. ' + step
        mismatch = re.match(r'(.+) does not belong to (.+)\.', raw)
        if mismatch:
            return f'El lote {mismatch[1]} no pertenece a {mismatch[2]}.'
        inactive = re.match(r'(.+) is inactive;', raw)
        if inactive:
            return f'{inactive[1]} está inactivo. Elija un registro activo.'
        if raw.startswith('Case weight required.'):
            return 'Falta el peso de una caja del producto terminado. Escriba el peso por caja en libras.'
        if raw == 'Adjustment must be nonzero.':
            return 'El ajuste debe ser diferente de cero.'
        if 'normalizes to the same code as existing lot' in raw:
            codes = re.findall(r"'([^']+)'", raw)
            if len(codes) >= 2:
                return f'El código {codes[0]} representa el mismo lote que {codes[1]}. Use el código exacto del lote existente o un código diferente.'
        if code in SPANISH_REASONS:
            return SPANISH_REASONS[code]
    if code in ERRORS:
        return ERRORS[code][es]
    # Legacy validators still use text; preserve the specific field in Spanish.
    required = re.search(r"(?:Provide |\b)([a-z_]+)(?: is required|.*English version required)", raw)
    if es and required:
        label = FIELD_LABELS.get(required[1], (required[1], required[1].replace('_', ' ')))[1]
        return 'Falta ' + label + '. Complete ese dato antes de registrar.'
    if es and 'English version required' in raw:
        return 'Falta una nota para registrar esta entrada. Escriba la nota en español o inglés.'
    if raw and ' / ' in raw:
        return raw.split(' / ', 1)[1 if es else 0]
    if es:
        labels = {'VALIDATION_FAILED': 'Los datos no cumplen los requisitos para registrar',
                  'DRAFT_BLOCKED': 'Faltan datos o confirmaciones en el borrador',
                  'NOT_FOUND': 'No se encontró el registro indicado'}
        return labels.get(code, 'No se puede registrar esta acción con los datos actuales') + '. Revise el producto, la cantidad y el lote con la oficina.'
    return raw or 'I could not complete that Factory Ledger request. Check the item and details.'
FIELD_LABELS = {
    'lot_id': ('lot tag', 'etiqueta del lote'), 'quantity': ('quantity', 'cantidad'),
    'quantity_lb': ('quantity (lb)', 'cantidad (lb)'), 'delta_lb': ('change in pounds', 'cambio en libras'),
    'reason_code': ('reason', 'motivo'), 'note': ('note', 'nota'), 'notes': ('notes', 'notas'),
    'customer_id': ('customer', 'cliente'), 'order_id': ('order number', 'número de pedido'),
    'occurred_at': ('when it happened', 'cuándo ocurrió'), 'case_weight_lb': ('case size (lb)', 'peso por caja (lb)'),
    'supplier_id': ('supplier', 'proveedor'), 'product_id': ('product', 'producto'),
    'cases': ('number of cases', 'número de cajas'), 'case_size_lb': ('case size (lb)', 'peso por caja (lb)'),
    'bol_reference': ('BOL number', 'número de BOL'), 'source_product_id': ('bulk batch', 'lote a granel'),
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
        if card.get('resolution_kind') == 'order':
            card['message'] = (f"No encontré el pedido {card.get('query', '')}. Escriba su número o el nombre del cliente." if es else
                               f"I couldn't find order {card.get('query', '')}. Give its number or customer name.")
        if card.get('resolution_kind') == 'lot':
            card['message'] = (f"No hay un lote llamado {card.get('query', '')}." if es else f"No lot called {card.get('query', '')}.")
    elif kind == 'choices':
        card['message'] = '¿Quiso decir una de estas opciones?' if es else 'Did you mean one of these?'
        if card.get('existing_customer_only'):
            card['message'] = 'Ese cliente ya existe. Elija el cliente de la lista.' if es else 'That customer already exists. Choose the existing customer below.'
        if card.get('result', {}).get('outcome') == 'none' and card.get('query'):
            card['message'] = (f'No reconocí «{card["query"]}». ' if es else f'I did not recognize “{card["query"]}”. ') + card['message']
        if card.get('packing_lots'):
            query = card.get('query', '')
            card['message'] = ((f'No encontré el lote {query}. ' if query else '') + '¿De qué lote a granel salieron?' if es else
                               (f"I couldn't find lot {query}. " if query else '') + 'Which batch did these come from?')
        if card.get('packing_outputs'):
            card['message'] = '¿Qué producto terminado empacaron?' if es else 'Which finished product did you pack?'
        if card.get('order_not_found'):
            card['message'] = (f"No encontré el pedido {card.get('query', '')}. Elija un pedido abierto de ese cliente." if es else
                               f"I couldn't find order {card.get('query', '')}. Choose that customer's open order.")
        result = dict(card['result'])
        result['ask'] = card['message']
        # Catalog names remain the identities printed on physical labels.
        card['result'] = result
    elif kind == 'missing_facts':
        card['message'] = 'Agrega los datos que faltan.' if es else 'Please add the missing details.'
        if card.get('packing_batch_needed'):
            card['message'] = '¿De qué lote a granel salieron? Escriba el código de la etiqueta.' if es else 'Which batch did these come from? Give the code on its tag.'
        if card.get('shipping_quantity_needed'):
            order = card['order']
            remaining = order_remaining(order, es)
            card['message'] = (f"¿Cuánto del pedido {order['order_number']} vamos a enviar? Pendiente: {remaining}." if es else
                               f"How much of order {order['order_number']} should we ship? Remaining: {remaining}.")
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
            detail = {'message': str(detail), 'error_code': (result.get('error_detail') or {}).get('code')}
        code = detail.get('error_code') or detail.get('code') or (result.get('error_detail') or {}).get('code')
        if card.get('status', 0) >= 500:
            card['message'] = ('Factory Ledger no está disponible temporalmente — inténtelo de nuevo en un minuto.' if es else
                               'Factory Ledger is temporarily unavailable — try again in a minute.')
            card['error_code'] = code
            return card
        card['message'] = refusal_message(detail, es)
        if code == 'LOT_NOT_FOUND' and card.get('query'):
            card['message'] = (f"No hay un lote llamado {card['query']}." if es else f"No lot called {card['query']}.")
        card['error_code'] = code
        if code == 'INSUFFICIENT_STOCK':
            card['actions'] = [
                {'href': '/dashboard/entry.html', 'label': 'Registrar existencias faltantes' if es else 'Record missing stock'},
                {'href': '/dashboard/office.html', 'label': 'Cambiar la cantidad del pedido' if es else 'Change the order quantity'}]
    elif kind == 'receive_queue':
        queued = card.get('queued_text', [])
        card['message'] = ('Registro un producto a la vez. Estas recepciones siguen pendientes:' if es else
                           'I record one product at a time. These receives are still queued:')
        if card.get('finish_current'):
            card['message'] = ('Registre o cancele el borrador actual antes de continuar.' if es else
                               'Record or cancel the current draft before continuing.')
        if card.get('missing_current'):
            card['message'] = ('Complete los datos de la primera recepción antes de continuar con la siguiente.' if es else
                               'Finish the details for the first receive before continuing to the next one.')
        if card.get('queue_empty'):
            card['message'] = 'No quedan recepciones pendientes.' if es else 'There are no queued receives left.'
        if queued:
            card['continue_text'] = 'Continúa la siguiente recepción pendiente' if es else 'Continue the next queued receive'
    elif kind == 'misunderstood':
        card['message'] = 'No entendí eso. Escriba el artículo y lo que necesita hacer.' if es else 'I did not understand that. Give the item and what you need to do.'
    elif kind == 'draft':
        prepared = dict(card['prepared'])
        for field in ('blockers', 'warnings'):
            values = []
            for b in prepared.get(field, []):
                b = dict(b)
                b['message'] = refusal_message(b, es)
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
    if card.get('reason_required'):
        card['message'] = 'Elige un motivo de la corrección.' if es else 'Choose a correction reason.'
    return card
