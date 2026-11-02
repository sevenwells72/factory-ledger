"""Plain EN/ES messages for request validation errors (422).

FastAPI's default 422 detail is a list of Pydantic errors such as
"String should have at least 1 character". Floor staff saw that text on the
Make draft (Oct 29 2026) when a lot-confirmation row was left blank. Each
error keeps its Pydantic fields (type, loc, msg) for API callers and gains
`message` / `message_es` naming the field, which the dashboard shows instead.
The envelope's `error_detail.message` stays Pydantic's text (a route contract).
"""

# Field name -> (English, Spanish), matching the dashboard form labels.
FIELD_LABELS = {
    'value': ('Lot label code', 'Código de la etiqueta del lote'),
    'method': ('Confirmation method', 'Método de confirmación'),
    'lot_id': ('Lot', 'Lote'),
    'order_reference': ('Shipment reference', 'Referencia de envío'),
    'bol_reference': ('BOL / delivery reference', 'BOL / referencia de entrega'),
    'supplier_lot_code': ('Supplier lot', 'Lote del proveedor'),
    'reason_code': ('Reason', 'Motivo'),
    'reason': ('Reason', 'Motivo'),
    'note': ('Note', 'Nota'),
    'notes': ('Note', 'Nota'),
    'batches': ('Number of batches', 'Número de lotes de producción'),
    'cases': ('Cases', 'Cajones'),
    'case_size_lb': ('lb per case', 'lb por cajón'),
    'case_weight_lb': ('lb per case', 'lb por cajón'),
    'quantity_lb': ('Pounds', 'Libras'),
    'delta_lb': ('Pounds to adjust', 'Libras a ajustar'),
    'product_id': ('Product', 'Producto'),
    'source_product_id': ('Source batch', 'Lote de producción de origen'),
    'target_product_id': ('Packed product', 'Producto empacado'),
    'customer_id': ('Customer', 'Cliente'),
    'supplier_id': ('Supplier', 'Proveedor'),
    'lot_code': ('Lot code', 'Código de lote'),
    'target_lot_code': ('Output lot', 'Lote de salida'),
}

# List fields -> (English, Spanish) row name, so "lot_confirmations.1.value"
# reads "Lot confirmation 2: Lot label code".
LIST_LABELS = {
    'lot_confirmations': ('Lot confirmation', 'Confirmación de lote'),
    'substitutions': ('Substitution', 'Sustitución'),
    'ingredient_lots': ('Ingredient lot', 'Lote de ingrediente'),
    'lot_allocations': ('Source lot', 'Lote de origen'),
    'lines': ('Line', 'Línea'),
    'supplier_lot_entries': ('Supplier lot', 'Lote del proveedor'),
}

REQUIRED_TYPES = {'missing', 'string_too_short', 'too_short'}


def _field(loc):
    parts = [p for p in loc if p not in ('body', 'query', 'path')]
    names = [p for p in parts if isinstance(p, str)]
    name = names[-1] if names else ''
    en, es = FIELD_LABELS.get(name) or ((name.replace('_', ' ').capitalize(),) * 2 if name else ('This field', 'Este campo'))
    for i, part in enumerate(parts[:-1]):
        if isinstance(part, str) and part in LIST_LABELS and isinstance(parts[i + 1], int):
            row_en, row_es = LIST_LABELS[part]
            if i + 2 == len(parts):
                # The list row itself (e.g. a missing row object).
                return f'{row_en} {parts[i + 1] + 1}', f'{row_es} {parts[i + 1] + 1}'
            return f'{row_en} {parts[i + 1] + 1}: {en}', f'{row_es} {parts[i + 1] + 1}: {es}'
    return en, es


def describe(error):
    """`error` is one Pydantic error dict; returns (message, message_es)."""
    en, es = _field(error.get('loc') or ())
    if error.get('type') in REQUIRED_TYPES:
        return f'{en} is required. Fill it in and try again.', f'{es} es obligatorio. Complétalo e intenta de nuevo.'
    # Other errors keep Pydantic's own text (it often names the allowed values).
    msg = error.get('msg') or 'check this value'
    return f'{en}: {msg}', f'{es}: {msg}'


def with_messages(errors):
    """Add plain `message` / `message_es` to each error, keeping the rest."""
    out = []
    for error in errors:
        if isinstance(error, dict) and 'message' not in error:
            en, es = describe(error)
            error = {**error, 'message': en, 'message_es': es}
        out.append(error)
    return out
