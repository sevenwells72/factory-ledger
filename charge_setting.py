"""Charge or Product: products.is_service is the only rule (PR #143, migration 085).

* Names never decide. Every route that creates a catalog item requires an
  explicit `item_kind` ('product' | 'charge'); there is no default.
* Safety net: an active item whose name has charge words (fee, charge,
  delivery, pallet, freight, surcharge) but is a Product — or is a Charge
  without them — gets ONE open CHARGE_SETTING_CHECK exception. Nothing changes
  the setting automatically; the owner or office answers it, which records the
  answer on the item and the item is never asked about again.
* Office "Catalog item" form: search, create, change Charge/Product.
"""
import re
from typing import Literal, Optional

from fastapi import Depends, HTTPException, Request
from psycopg2.extras import Json
from pydantic import BaseModel, constr

import permissions

KIND = 'CHARGE_SETTING_CHECK'
CHARGE_WORDS = ('fee', 'charge', 'delivery', 'pallet', 'freight', 'surcharge')
_CHARGE_WORDS = re.compile('|'.join(CHARGE_WORDS), re.IGNORECASE)   # = charge_words_in_name() in 085
ITEM_KINDS = {'product': False, 'charge': True}
LABELS = {'product': ('Product', 'Producto'),
          'charge': ('Charge (pallet, freight, delivery…)', 'Cargo (palé, flete, entrega…)')}
QUESTION = ('Confirm: Charge or Product?', 'Confirmar: ¿Cargo o Producto?')
RESOLUTIONS = {'confirmed_charge': 'charge', 'confirmed_product': 'product'}
PRODUCT_TYPES = ('finished', 'ingredient', 'packaging', 'consumable')
CHARGE_TYPE, CHARGE_UOM = 'finished', 'each'   # same shape as the live Pallet Charge order item
ROUTES = frozenset({
    ('GET', '/catalog/items'),
    ('POST', '/catalog/items'),
    ('POST', '/catalog/items/{product_id}/item-kind'),
    ('POST', '/catalog/items/{product_id}/confirm-kind'),
})


def fail(status, code, message, message_es, **extra):
    raise HTTPException(status, {'error_code': code, 'message': message, 'message_es': message_es, **extra})


def has_charge_words(name):
    return bool(_CHARGE_WORDS.search(name or ''))


def item_kind(is_service):
    return 'charge' if is_service else 'product'


def require_item_kind(value, *, charge_allowed=True, what='item'):
    """The explicit choice → is_service. No choice, or an unknown one, is refused."""
    choice = (value or '').strip().lower() if isinstance(value, str) else value
    if not choice:
        fail(422, 'ITEM_KIND_REQUIRED', 'Choose Product or Charge (pallet, freight, delivery…).',
             'Elige Producto o Cargo (palé, flete, entrega…).', choices=list(ITEM_KINDS))
    if choice not in ITEM_KINDS:
        fail(422, 'ITEM_KIND_INVALID', 'item_kind must be "product" or "charge".',
             'item_kind debe ser "product" o "charge".', choices=list(ITEM_KINDS))
    if choice == 'charge' and not charge_allowed:
        fail(422, 'ITEM_KIND_NOT_ALLOWED', f'A {what} holds stock, so it is always a Product.',
             f'Un {what} tiene inventario, así que siempre es un Producto.', choices=['product'])
    return ITEM_KINDS[choice]


def installed(cur):
    """Migration 085 present? The create routes still require the choice without it."""
    cur.execute("SELECT to_regclass('exceptions_one_charge_check_per_product_idx') IS NOT NULL AS ok")
    return cur.fetchone()['ok']


def check(cur, product_id):
    """Open the one CHARGE_SETTING_CHECK for this item if its name and setting disagree.

    Returns the open check's id (new or existing), or None. Never changes the item."""
    if not installed(cur):
        return None
    cur.execute('''SELECT id, name, is_service, COALESCE(active, true) AS active, charge_setting_confirmed_at
                   FROM products WHERE id=%s''', (product_id,))
    item = cur.fetchone()
    if not item or not item['active'] or item['charge_setting_confirmed_at'] or \
            has_charge_words(item['name']) == bool(item['is_service']):
        return None
    detail = {'name': item['name'], 'marked_as': item_kind(item['is_service']),
              'reason': 'charge_without_words' if item['is_service'] else 'charge_words_marked_product'}
    cur.execute('''INSERT INTO exceptions(kind, severity, product_id, owner_actor_id, detail)
                   VALUES (%s, 'warn', %s, %s, %s)
                   ON CONFLICT (product_id) WHERE kind='CHARGE_SETTING_CHECK' AND status IN ('open','escalated')
                   DO NOTHING RETURNING id''',
                (KIND, product_id, permissions.owner_actor_id(cur), Json(detail)))
    row = cur.fetchone()
    if row:
        return row['id']
    return open_check_id(cur, product_id)


def check_text(detail):
    """Card wording for one check: the question, the current setting and why it was asked."""
    marked = detail.get('marked_as') or 'product'
    en, es = LABELS[marked]
    found = sorted({m.group(0).lower() for m in _CHARGE_WORDS.finditer(detail.get('name') or '')})
    if detail.get('reason') == 'charge_without_words':
        why, why_es = 'its name has no charge word', 'su nombre no tiene ninguna palabra de cargo'
    else:
        words = ', '.join(f'"{w}"' for w in found)
        why = f'its name contains {words or "a charge word"}'
        why_es = f'su nombre contiene {words or "una palabra de cargo"}'
    return {'question': QUESTION[0], 'question_es': QUESTION[1], 'charge_words_found': found,
            'message': f'{QUESTION[0]} Now marked {en}, but {why}. Nothing changes until someone confirms.',
            'message_es': f'{QUESTION[1]} Ahora está marcado como {es}, pero {why_es}. Nada cambia hasta que alguien confirme.'}


def open_check_id(cur, product_id):
    cur.execute('''SELECT id FROM exceptions WHERE kind=%s AND product_id=%s AND status IN ('open','escalated')''',
                (KIND, product_id))
    row = cur.fetchone()
    return row['id'] if row else None


def sweep(cur):
    """Catch items added outside the app (scripts, imports): one check per mismatch."""
    if not installed(cur):
        return 0
    cur.execute('''SELECT id FROM products p
                   WHERE COALESCE(p.active, true) AND p.charge_setting_confirmed_at IS NULL
                     AND charge_words_in_name(p.name) <> p.is_service
                     AND NOT EXISTS (SELECT 1 FROM exceptions e WHERE e.kind=%s AND e.product_id=p.id
                                       AND e.status IN ('open','escalated'))
                   ORDER BY id''', (KIND,))
    return sum(1 for row in cur.fetchall() if check(cur, row['id']))


def view(cur, product_id):
    cur.execute('''SELECT id, name, odoo_code AS sku, type, uom, case_size_lb, is_service,
                          COALESCE(active, true) AS active, charge_setting_confirmed_at
                   FROM products WHERE id=%s''', (product_id,))
    row = cur.fetchone()
    if not row:
        fail(404, 'ITEM_NOT_FOUND', 'No catalog item has this id.', 'Ningún artículo del catálogo tiene este id.')
    item = dict(row)
    kind = item_kind(item['is_service'])
    item.update(item_kind=kind, item_kind_label=LABELS[kind][0], item_kind_label_es=LABELS[kind][1],
                charge_words=has_charge_words(item['name']),
                confirmed=item.pop('charge_setting_confirmed_at') is not None,
                check_exception_id=open_check_id(cur, product_id))
    if item['case_size_lb'] is not None:
        item['case_size_lb'] = float(item['case_size_lb'])
    return item


class CatalogItemCreate(BaseModel):
    name: constr(strict=True, min_length=1, max_length=200)
    item_kind: Optional[str] = None
    type: Optional[Literal[PRODUCT_TYPES]] = None
    uom: Optional[constr(strict=True, max_length=20)] = None
    sku: Optional[constr(strict=True, max_length=50)] = None
    case_size_lb: Optional[float] = None


class ItemKindChoice(BaseModel):
    item_kind: Optional[str] = None
    note: Optional[constr(strict=True, max_length=500)] = None


def _actor(api, request, action='manage_catalog'):
    actor = api._actor_identity(request)
    permissions.require(action, actor)
    return actor


def register_routes(app, api):
    @app.get('/catalog/items')
    def list_catalog_items(request: Request, q: str = '', limit: int = 25, _: bool = Depends(api.verify_api_key)):
        _actor(api, request)
        with api.get_transaction() as cur:
            cur.execute('''SELECT id FROM products WHERE COALESCE(active, true) AND type <> 'batch'
                             AND (%s = '' OR name ILIKE '%%' || %s || '%%' OR odoo_code ILIKE %s || '%%')
                           ORDER BY name LIMIT %s''', (q.strip(), q.strip(), q.strip(), max(1, min(limit, 100))))
            return {'items': [view(cur, row['id']) for row in cur.fetchall()]}

    @app.post('/catalog/items')
    def create_catalog_item(body: CatalogItemCreate, request: Request, _: bool = Depends(api.verify_api_key)):
        _actor(api, request)
        is_service = require_item_kind(body.item_kind)
        name = ' '.join(body.name.split())
        if not name:
            fail(422, 'ITEM_NAME_REQUIRED', 'Enter the item name.', 'Escribe el nombre del artículo.')
        if is_service:
            kind, uom, case_size = CHARGE_TYPE, CHARGE_UOM, None
        else:
            if not body.type:
                fail(422, 'ITEM_TYPE_REQUIRED', 'Choose the product type.', 'Elige el tipo de producto.',
                     choices=list(PRODUCT_TYPES))
            if body.case_size_lb is not None and not body.case_size_lb > 0:
                fail(422, 'CASE_SIZE_INVALID', 'Case size must be more than 0 lb.', 'El tamaño de caja debe ser mayor que 0 lb.')
            kind, uom, case_size = body.type, (body.uom or 'lb'), body.case_size_lb
        sku = (body.sku or '').strip() or None
        with api.get_transaction() as cur:
            cur.execute('SELECT id FROM products WHERE LOWER(name)=LOWER(%s) OR (%s::text IS NOT NULL AND odoo_code=%s)',
                        (name, sku, sku))
            existing = cur.fetchone()
            if existing:
                fail(409, 'ITEM_EXISTS', 'An item with this name or SKU already exists.',
                     'Ya existe un artículo con este nombre o SKU.', existing=view(cur, existing['id']))
            cur.execute('''INSERT INTO products(name, odoo_code, type, uom, case_size_lb, is_service, active, created_via)
                           VALUES (%s, %s, %s, %s, %s, %s, true, 'office_catalog') RETURNING id''',
                        (name, sku, kind, uom, case_size, is_service))
            product_id = cur.fetchone()['id']
            api._record_actor_write(cur, request, 'products', product_id)
            check(cur, product_id)
            return {'created': True, 'item': view(cur, product_id)}

    @app.post('/catalog/items/{product_id}/item-kind')
    def set_item_kind(product_id: int, body: ItemKindChoice, request: Request, _: bool = Depends(api.verify_api_key)):
        _actor(api, request)
        is_service = require_item_kind(body.item_kind)
        with api.get_transaction() as cur:
            cur.execute('SELECT id, type, is_service FROM products WHERE id=%s FOR UPDATE', (product_id,))
            row = cur.fetchone()
            if not row:
                view(cur, product_id)
            if is_service and row['type'] == 'batch':
                require_item_kind('charge', charge_allowed=False, what='batch')
            changed = bool(row['is_service']) != is_service
            if changed:
                cur.execute('UPDATE products SET is_service=%s WHERE id=%s', (is_service, product_id))
                api._record_actor_write(cur, request, 'products', product_id)
            check(cur, product_id)
            return {'changed': changed, 'item': view(cur, product_id)}

    @app.post('/catalog/items/{product_id}/confirm-kind')
    def confirm_item_kind(product_id: int, body: ItemKindChoice, request: Request, _: bool = Depends(api.verify_api_key)):
        actor = _actor(api, request, 'confirm_item_kind')
        is_service = require_item_kind(body.item_kind)
        with api.get_transaction() as cur:
            cur.execute('SELECT id, type, is_service FROM products WHERE id=%s FOR UPDATE', (product_id,))
            row = cur.fetchone()
            if not row:
                view(cur, product_id)
            if is_service and row['type'] == 'batch':
                require_item_kind('charge', charge_allowed=False, what='batch')
            # The person's answer is the setting; it is recorded and never asked again.
            cur.execute('''UPDATE products SET is_service=%s, charge_setting_confirmed_at=clock_timestamp(),
                                  charge_setting_confirmed_by_actor_id=%s WHERE id=%s''',
                        (is_service, actor['id'], product_id))
            api._record_actor_write(cur, request, 'products', product_id)
            resolution = 'confirmed_charge' if is_service else 'confirmed_product'
            cur.execute('''UPDATE exceptions SET status='resolved', resolved_at=clock_timestamp(), resolved_by_actor_id=%s,
                                  resolution_kind=%s, resolution_note=%s,
                                  detail = detail || jsonb_build_object('was', detail->>'marked_as', 'confirmed_as', %s::text)
                           WHERE kind=%s AND product_id=%s AND status IN ('open','escalated') RETURNING id''',
                        (actor['id'], resolution, body.note, item_kind(is_service), KIND, product_id))
            closed = [r['id'] for r in cur.fetchall()]
            for exception_id in closed:
                api._record_actor_write(cur, request, 'exceptions', exception_id)
            return {'confirmed': True, 'changed': bool(row['is_service']) != is_service,
                    'closed_exception_ids': closed, 'item': view(cur, product_id)}
