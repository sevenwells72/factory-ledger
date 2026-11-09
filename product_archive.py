"""Product archive: hide finished goods, batches, ingredients and materials from
active lists without deleting anything (owner request, Michael, 2026-11-09).

Archive reuses the long-standing `products.active` flag (false = archived;
NULL/true = active), which every active list and picker already filters on.
Migration 093 adds only WHO and WHEN (`archived_at`, `archived_by`). Archiving
never touches transactions, lots, BOM lines, quantities or history: an archived
product still shows in history, traceability, lot lookups and global search.

Routes (registered from main.py right after PUT /admin/products/{id}):
  POST /admin/products/archive    {product_ids: [...]}  all-or-nothing
  POST /admin/products/unarchive  {product_ids: [...]}  all-or-nothing
Owner only: Michael (name selection or an owner credential). Other people get
403 before anything changes. Until migration 093's marker exists both routes
answer 503 ARCHIVE_UNAVAILABLE and change nothing.

The one block: an ingredient (or batch used as an ingredient) that an active,
non-archived recipe still uses, unless that recipe is archived in the same
request. The 409 names the recipe(s).

main is reached through `api` at call time (never imported).
"""
from typing import List

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel, Field

import permissions

ARCHIVE_MARKER = '093_product_archive'
MAX_IDS = 200

# Read-only SQL fragments other code can reuse. `to_jsonb(p)` reads the 093
# columns without failing on a database where 093 is not applied yet.
ARCHIVED_SQL = "COALESCE(p.active, true) = false"
ARCHIVED_AT_SQL = "(to_jsonb(p) ->> 'archived_at')"
ARCHIVED_BY_SQL = "(to_jsonb(p) ->> 'archived_by')"


class ArchiveRequest(BaseModel):
    product_ids: List[int] = Field(..., min_items=1, max_items=MAX_IDS)


def _fail(status, code, message, message_es, **extra):
    raise HTTPException(status, {'error_code': code, 'message': message,
                                 'message_es': message_es, **extra})


def archive_enabled(cur) -> bool:
    cur.execute('SELECT 1 FROM migration_markers WHERE name=%s', (ARCHIVE_MARKER,))
    return cur.fetchone() is not None


def _require_owner(api, request) -> str:
    """The archiving person's name. Michael only; nothing changes otherwise."""
    actor = api.request_actor(request)
    identity = api._actor_identity(request) if actor else None
    if not actor or permissions.role_of(identity) != permissions.OWNER:
        _fail(403, 'OWNER_REQUIRED',
              'Only Michael can archive or restore items. Nothing changed.',
              'Solo Michael puede archivar o restaurar artículos. No se cambió nada.')
    return actor['name']


def _load(cur, ids):
    cur.execute("""SELECT id, name, type, COALESCE(active, true) AS active
                   FROM products WHERE id = ANY(%s) ORDER BY id FOR UPDATE""", (ids,))
    rows = cur.fetchall()
    missing = sorted(set(ids) - {r['id'] for r in rows})
    if missing:
        _fail(404, 'PRODUCT_NOT_FOUND', f'Product id(s) not found: {missing}. Nothing changed.',
              f'No se encontraron los productos: {missing}. No se cambió nada.', product_ids=missing)
    return rows


def blocking_recipes(cur, ids):
    """{ingredient_id: [recipe names]} for active recipes outside `ids` that use it."""
    cur.execute("""
        SELECT bf.ingredient_product_id AS ingredient_id, r.name AS recipe
        FROM batch_formulas bf
        JOIN products r ON r.id = bf.product_id
        WHERE bf.ingredient_product_id = ANY(%s)
          AND COALESCE(r.active, true) = true
          AND NOT (r.id = ANY(%s))
        ORDER BY r.name
    """, (ids, ids))
    out = {}
    for row in cur.fetchall():
        out.setdefault(row['ingredient_id'], [])
        if row['recipe'] not in out[row['ingredient_id']]:
            out[row['ingredient_id']].append(row['recipe'])
    return out


def archive_products(cur, ids, archived_by):
    """Archive `ids` in the caller's transaction. Returns the response body."""
    ids = sorted(set(ids))
    rows = _load(cur, ids)
    blocks = blocking_recipes(cur, ids)
    if blocks:
        names = {r['id']: r['name'] for r in rows}
        detail = [{'product_id': pid, 'product_name': names[pid], 'recipes': recipes}
                  for pid, recipes in sorted(blocks.items())]
        first = detail[0]
        _fail(409, 'USED_BY_ACTIVE_RECIPE',
              f"{first['product_name']} is used by active recipe {', '.join(first['recipes'])}. "
              'Archive the recipe first. Nothing changed.',
              f"{first['product_name']} se usa en la receta activa {', '.join(first['recipes'])}. "
              'Archiva primero la receta. No se cambió nada.', blocked=detail)
    to_archive = [r['id'] for r in rows if r['active']]
    if to_archive:
        cur.execute("""UPDATE products SET active = false, archived_at = now(), archived_by = %s
                       WHERE id = ANY(%s)""", (archived_by, to_archive))
    return {'archived': [{'product_id': r['id'], 'product_name': r['name'], 'type': r['type']}
                         for r in rows if r['active']],
            'already_archived': [r['id'] for r in rows if not r['active']],
            'archived_by': archived_by}


def unarchive_products(cur, ids):
    ids = sorted(set(ids))
    rows = _load(cur, ids)
    to_restore = [r['id'] for r in rows if not r['active']]
    if to_restore:
        # Clear the attribution with the flag so an active product never
        # carries a stale "archived by"; the restore itself is in the app log.
        cur.execute("""UPDATE products SET active = true, archived_at = NULL, archived_by = NULL
                       WHERE id = ANY(%s)""", (to_restore,))
    return {'restored': [{'product_id': r['id'], 'product_name': r['name'], 'type': r['type']}
                         for r in rows if not r['active']],
            'already_active': [r['id'] for r in rows if r['active']]}


def register_routes(app, api):
    @app.post("/admin/products/archive")
    def archive_items(req: ArchiveRequest, request: Request, _: bool = Depends(api.verify_api_key)):
        name = _require_owner(api, request)
        with api.get_transaction() as cur:
            if not archive_enabled(cur):
                _fail(503, 'ARCHIVE_UNAVAILABLE', 'Archiving is not switched on yet. Nothing changed.',
                      'Archivar todavía no está activado. No se cambió nada.')
            result = archive_products(cur, req.product_ids, name)
        api.logger.info('product archive by %s: %s', name, [p['product_id'] for p in result['archived']])
        return result

    @app.post("/admin/products/unarchive")
    def unarchive_items(req: ArchiveRequest, request: Request, _: bool = Depends(api.verify_api_key)):
        name = _require_owner(api, request)
        with api.get_transaction() as cur:
            if not archive_enabled(cur):
                _fail(503, 'ARCHIVE_UNAVAILABLE', 'Archiving is not switched on yet. Nothing changed.',
                      'Archivar todavía no está activado. No se cambió nada.')
            result = unarchive_products(cur, req.product_ids)
        api.logger.info('product unarchive by %s: %s', name, [p['product_id'] for p in result['restored']])
        return result
