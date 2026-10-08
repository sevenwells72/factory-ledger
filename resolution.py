"""Interface-neutral, SELECT-only resolution for A4.

No tickets, migrations, alias learning, audit inserts or application imports.
The caller owns the transaction. Explicit identifiers/aliases are tried first;
ranking never breaks a tie between plausible identities.
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import partial
import re
from typing import Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, validator


Kind = Literal['product', 'customer', 'supplier', 'order', 'lot', 'unit']
AliasKind = Literal['token', 'product', 'customer', 'supplier']
MAX_CANDIDATES = 8
MAX_PAGE_SIZE = 25
MAX_VARIANTS = 32
PLAUSIBLE = 0.5
MIN_CANDIDATE = 0.25
SENTINEL_SUPPLIERS = ('found', 'found inventory', 'inventory found', 'physical count',
                      'initial inventory', 'inventory correction', 'inventory intake', 'unknown')


def normalize(text):
    return ' '.join((text or '').lower().split())


def normalize_lot(text):
    return re.sub(r'\s+lot$', '', normalize(text))


class ResolutionContext(BaseModel):
    action: Optional[Literal['make', 'pack', 'receive', 'ship', 'order', 'adjust', 'void']] = None
    group: Optional[Literal['floor', 'office']] = None
    product_id: Optional[int] = Field(None, gt=0)
    customer_id: Optional[int] = Field(None, gt=0)
    supplier_id: Optional[int] = Field(None, gt=0)
    order_id: Optional[int] = Field(None, gt=0)
    customer_address: Optional[str] = Field(None, max_length=500)
    state: Optional[Literal['open', 'closed', 'cancelled']] = None
    status: Optional[Literal['new', 'confirmed', 'in_production', 'ready',
                             'shipped', 'partial_ship', 'invoiced', 'cancelled']] = None

    class Config:
        extra = 'forbid'


class ResolveRequest(BaseModel):
    kind: Kind
    query: str = Field(..., max_length=500)
    context: ResolutionContext = Field(default_factory=ResolutionContext)
    quantity: Optional[Decimal] = Field(None, gt=0, le=Decimal('1000000000000'))
    limit: int = Field(MAX_CANDIDATES, ge=1, le=MAX_PAGE_SIZE, strict=True)
    offset: int = Field(0, ge=0, le=2147483647, strict=True)

    @validator('quantity', pre=True)
    def finite_quantity(cls, value):
        if isinstance(value, bool):
            raise ValueError('quantity must be a finite positive number')
        if value is not None:
            try:
                if not Decimal(str(value)).is_finite():
                    raise ValueError('quantity must be a finite positive number')
            except InvalidOperation:
                raise ValueError('quantity must be a finite positive number') from None
        return value

    class Config:
        extra = 'forbid'


class AliasSeed(BaseModel):
    """Version-1 JSON seed row; validation is pure and never writes a database."""
    kind: AliasKind
    alias: str = Field(..., min_length=1, max_length=200)
    expansion: Optional[str] = Field(None, max_length=200)
    product_id: Optional[int] = Field(None, gt=0)
    customer_id: Optional[int] = Field(None, gt=0)
    supplier_id: Optional[int] = Field(None, gt=0)
    language: Literal['any', 'en', 'es'] = 'any'
    active: bool = True

    @validator('alias', 'expansion')
    def nonblank(cls, value):
        if value is not None and not normalize(value):
            raise ValueError('alias and expansion must not be blank')
        return value.strip() if value is not None else value

    class Config:
        extra = 'forbid'


def validate_alias_seed(document):
    """Return insertable rows for a future approved importer, without IDs/audit fields."""
    if not isinstance(document, dict) or set(document) != {'version', 'aliases'}:
        raise ValueError('Expected version and aliases')
    if type(document['version']) is not int or document['version'] != 1 or not isinstance(document['aliases'], list):
        raise ValueError('Expected version 1 and an aliases array')
    rows, seen = [], set()
    for data in document['aliases']:
        row = AliasSeed(**data).dict()
        target = {'token': 'expansion', 'product': 'product_id',
                  'customer': 'customer_id', 'supplier': 'supplier_id'}[row['kind']]
        if not row[target] or any(row[k] is not None for k in
                                 {'expansion', 'product_id', 'customer_id', 'supplier_id'} - {target}):
            raise ValueError('Alias must have exactly the target required by its kind')
        key = (row['kind'], normalize(row['alias']), row['product_id'],
               row['customer_id'], row['supplier_id'])
        if key in seen:
            raise ValueError('Duplicate normalized alias target')
        seen.add(key)
        rows.append(row)
    return rows


def read_aliases(cur):
    # Deploying the read module does not silently migrate or seed a shared DB.
    cur.execute("SELECT to_regclass('search_aliases') AS table_name")
    if cur.fetchone()['table_name'] is None:
        return [], False
    cur.execute('''SELECT id, kind, alias, alias_norm, expansion, product_id,
                         customer_id, supplier_id, language, active
                  FROM search_aliases ORDER BY id''')
    return [dict(row) for row in cur.fetchall()], True


@dataclass
class Variant:
    text: str
    expansions: tuple = ()


def query_variants(query, aliases):
    """Bidirectional substitutions on whole whitespace-delimited spans only.

    Work from original spans once (no recursive/cyclic alias rewriting). Longest
    spans win; conflicting expansions stay as alternatives, never first-row wins.
    """
    tokens = normalize(query).split()
    rules = {}
    for row in aliases:
        if row['kind'] != 'token' or not row.get('active', True):
            continue
        left, right = normalize(row['alias']), normalize(row['expansion'])
        if not left or not right or left == right:
            continue
        for source, target in ((left, right), (right, left)):
            rules.setdefault(tuple(source.split()), []).append((target, row.get('id')))
    variants, truncated, i = [Variant('')], False, 0
    while i < len(tokens):
        spans = [span for span in rules if tokens[i:i + len(span)] == list(span)]
        span = max(spans, key=len) if spans else (tokens[i],)
        source = ' '.join(span)
        options = [(source, None)] + rules.get(span, [])
        new = {}
        for variant in variants:
            for target, alias_id in options:
                text = (variant.text + ' ' + target).strip()
                changes = variant.expansions
                if target != source:
                    changes += ({'from': source, 'to': target, 'alias_id': alias_id},)
                new.setdefault(text, Variant(text, changes))
        truncated |= len(new) > MAX_VARIANTS
        variants = list(new.values())[:MAX_VARIANTS]
        i += len(span)
    return variants, truncated


def _patterns(query, aliases=()):
    # Protect all declared alias spans, including long and multiword spellings.
    # Retaining an original spelling must never let CLS match inside CLSX.
    spans = {tuple(normalize(value).split()) for row in aliases
             for value in (row.get('alias'), row.get('expansion')) if normalize(value)}
    tokens, patterns, i = query.split(), [], 0
    while i < len(tokens):
        hits = [span for span in spans if tokens[i:i + len(span)] == list(span)]
        span = max(hits, key=len) if hits else (tokens[i],)
        terms = [token.lstrip('#') for token in span]
        if all(terms):
            escaped = r'\s+'.join(re.escape(term) for term in terms)
            if hits or any(len(term) <= 2 or term.isdigit() for term in terms):
                escaped = r'\m' + escaped + r'\M'
            patterns.append(escaped)
        i += len(span)
    return patterns


def _fuzzy_allowed(query, aliases):
    tokens = query.split()
    if not tokens or any(len(t.lstrip('#')) <= 2 or t.lstrip('#').isdigit() for t in tokens):
        return False
    # Alias spellings are exact-only even when long. Fuzzy compares real names.
    for row in aliases:
        for value in (row.get('alias'), row.get('expansion')):
            term = normalize(value)
            if term and re.search(r'(?<!\S)' + re.escape(term) + r'(?!\S)', query):
                return False
    return True


def _ranked_rows(cur, scored_sql, params, *, limit=MAX_CANDIDATES, offset=0):
    '''Count every identity before paging; retain the first three for decisions.

    The first three preserve the global best match/near misses even on an empty
    later page. Public candidates use their global position, never a local cap.
    ``limit=None`` is internal-only, for authoritative exact-lot eligibility.
    '''
    page = '' if limit is None else 'WHERE _position <= 3 OR (_position > %s AND _position <= %s)'
    bounds = () if limit is None else (offset, offset + limit)
    cur.execute(f'''
        WITH scored AS ({scored_sql}), ranked AS (
            SELECT *, count(*) FILTER (WHERE score >= 0.25) OVER () AS candidate_count,
                      count(*) FILTER (WHERE score >= 0.5) OVER () AS plausible_count,
                      row_number() OVER (ORDER BY score DESC, recent_activity DESC,
                                          context_rank DESC, name, id) AS _position
            FROM scored
        ) SELECT * FROM ranked {page} ORDER BY _position
    ''', tuple(params) + bounds)
    return [dict(row) for row in cur.fetchall()]


def _search(cur, source, params, variants, aliases, *, keywords=True, exact_only=False,
            limit=MAX_CANDIDATES, offset=0):
    '''Rank the deduplicated union of spellings, then count and page in SQL.'''
    variants = [variant for variant in variants if variant.text]
    if not variants:
        return []
    values, variant_params = [], []
    for index, variant in enumerate(variants):
        patterns = _patterns(variant.text, aliases) if keywords else []
        values.append('(%s::text, %s::text[], %s::boolean, %s::int, %s::int)')
        variant_params.extend((variant.text, patterns,
                               _fuzzy_allowed(variant.text, aliases) and not exact_only,
                               index, len(variant.expansions)))
    for exact in (True, False):
        score = '1.0::float' if exact else '''CASE
            WHEN cardinality(v.patterns) > 0 AND lower(s.name) ~ ALL(v.patterns) THEN 0.8
            WHEN v.fuzzy THEN similarity(lower(s.name), v.query) ELSE 0.0 END'''
        where = r'''WHERE EXISTS (SELECT 1 FROM unnest(s.exact_terms) term
                    WHERE lower(btrim(regexp_replace(term, '\s+', ' ', 'g'))) = v.query)''' if exact else ''
        scored_sql = f'''
            WITH source AS ({source}),
                 variants(query, patterns, fuzzy, variant_index, expansion_count) AS
                     (VALUES {','.join(values)}),
                 matches AS (
                     SELECT s.*, {score} AS score, v.variant_index, v.expansion_count
                     FROM source s CROSS JOIN variants v {where}
                 )
            SELECT DISTINCT ON (id) * FROM matches
            ORDER BY id, score DESC, expansion_count, variant_index
        '''
        rows = _ranked_rows(cur, scored_sql, tuple(params) + tuple(variant_params),
                            limit=limit, offset=offset)
        for row in rows:
            variant = variants[row['variant_index']]
            row['score'] = float(row['score'])
            row['tier'] = ('alias' if variant.expansions else 'exact') if exact else (
                ('alias' if variant.expansions else 'keyword') if row['score'] == 0.8 else 'trigram')
            row['_variant'] = variant
        if rows or exact_only:
            return rows
    return []


def _candidate(row):
    result = {k: row[k] for k in (
        'id', 'name', 'odoo_code', 'type', 'label_type', 'case_size_lb', 'pack_format',
        'default_batch_lb', 'parent_batch_product_id', 'on_hand_lb', 'product_id',
        'lot_code', 'order_number', 'customer_po', 'customer_id', 'state', 'status',
        'unit', 'quantity', 'bags_per_case', 'units_per_case', 'retail_bag_oz'
    ) if k in row}
    result.update(label=row.get('label') or row['name'], score=round(row['score'], 4),
                  tier=row['tier'], why=row.get('why') or {
                      'exact': 'Exact identifier or full name', 'alias': 'Exact alias expansion',
                      'keyword': 'All query terms in the name', 'trigram': 'Similar name; confirm identity',
                      'suffix': 'Exact last four characters within this product',
                      'context': 'Open order for the specified customer'
                  }[row['tier']], context_boost=row.get('context_boost'))
    return result


def decide(query, rows, *, truncated=False, note=None, limit=MAX_CANDIDATES, offset=0):
    if rows and '_position' in rows[0]:
        rows = sorted(rows, key=lambda row: row['_position'])
    else:
        rows = sorted(rows, key=lambda row: (-row['score'], -row.get('recent_activity', 0),
                                             -row.get('context_rank', 0), row['name'], str(row['id'])))
    candidates = [row for row in rows if row['score'] >= MIN_CANDIDATE]
    plausible = [row for row in rows if row['score'] >= PLAUSIBLE]
    count = max([len(candidates)] + [int(row.get('candidate_count', 0)) for row in rows])
    plausible_count = max([len(plausible)] + [int(row.get('plausible_count', 0)) for row in rows])
    outcome = 'none' if not candidates else 'ambiguous'
    if plausible_count == 1 and not truncated:
        outcome = 'match'
    displayed = [_candidate(row) for i, row in enumerate(candidates, 1)
                 if offset < row.get('_position', i) <= offset + limit]
    best = candidates[0] if candidates else None
    expansions = []
    for row in candidates:
        for expansion in row.get('_variant', Variant(query)).expansions:
            if expansion not in expansions:
                expansions.append(expansion)
    confidence = {'exact': 'high', 'alias': 'high', 'keyword': 'medium',
                  'trigram': 'low', 'suffix': 'medium', 'context': 'medium'}
    result = dict(outcome=outcome, query_normalized=best['_variant'].text if best and '_variant' in best else normalize(query),
                  expansions_applied=expansions, candidates=displayed, candidate_count=count,
                  has_more=count > offset + limit, limit=limit, offset=offset, match=None,
                  confidence=confidence[best['tier']] if best else 'none',
                  needs_clarification=outcome != 'match', ask=None)
    if outcome == 'match':
        result['match'] = _candidate(plausible[0])
    elif outcome == 'ambiguous':
        prefix = 'Which one?' if plausible_count > 1 else 'No confident match. Please clarify:'
        result['ask'] = prefix + ' ' + '; '.join(f"{i}) {r['label']}" for i, r in enumerate(displayed, offset + 1))
        if not displayed:
            result['ask'] = 'No candidates on this page. Use a smaller offset.'
        elif result['has_more']:
            result['ask'] += '; more matches exist — request the next page or give an exact code.'
    else:
        result['ask'] = 'No confident match. Please give a fuller name or exact code.'
        result['near_misses'] = [_candidate(row) for row in rows[:3] if row['score'] > 0]
    if truncated:
        result['needs_clarification'] = True
        result['ask'] = 'Too many alias expansions. Please give an exact identifier.'
    if note:
        result['note'] = note
    return result


PRODUCT_SOURCE = '''SELECT p.id, p.name, p.odoo_code, p.type, p.label_type,
    p.case_size_lb, p.pack_format, p.default_batch_lb, p.parent_batch_product_id,
    p.bags_per_case, p.units_per_case, p.retail_bag_oz,
    p.name || COALESCE(' (' || p.odoo_code || ')', '') AS label,
    ARRAY[p.name, p.odoo_code] ||
       ARRAY(SELECT unnest(ARRAY[customer_item_code, customer_description])
             FROM customer_product_aliases WHERE customer_id = %s AND product_id = p.id) ||
       ARRAY(SELECT vendor_description FROM supplier_product_aliases
             WHERE supplier_id = %s AND product_id = p.id) ||
       CASE WHEN p.id = ANY(%s::int[]) THEN ARRAY[%s::text] ELSE ARRAY[]::text[] END AS exact_terms,
    CASE WHEN %s = 'order' AND (p.type = 'finished' OR p.is_service) THEN 3
         WHEN %s = 'make' AND %s = 'floor' AND p.type = 'ingredient' THEN 3
         WHEN %s = 'make' AND p.type = 'batch' THEN 2
         WHEN %s = 'receive' AND p.type IN ('ingredient', 'packaging') THEN 2
         WHEN %s = 'pack' AND p.type = 'finished' THEN 2 ELSE 0 END AS context_rank,
    CASE WHEN %s = 'order' AND (p.type = 'finished' OR p.is_service) THEN 'Finished goods / services for order'
         WHEN %s = 'make' AND %s = 'floor' AND p.type = 'ingredient' THEN 'Ingredient for floor make'
         WHEN %s = 'make' AND p.type = 'batch' THEN 'Batch for make'
         WHEN %s = 'receive' AND p.type IN ('ingredient', 'packaging') THEN 'Material for receive'
         WHEN %s = 'pack' AND p.type = 'finished' THEN 'Finished goods for pack' END AS context_boost,
    COALESCE(activity.recent_activity, 0) AS recent_activity
    FROM products p LEFT JOIN (
        SELECT product_id, extract(epoch FROM max(activity_at))::float AS recent_activity
        FROM (
            SELECT tl.product_id, (t.effective_record->>'occurred_at')::timestamptz AS activity_at
            FROM ledger_current_transaction_lines tl
            JOIN ledger_current_transactions t ON t.id = tl.transaction_id
            WHERE t.effective_status = 'posted'
            UNION ALL
            SELECT ol.product_id, ol.created_at AS activity_at
            FROM sales_order_lines ol JOIN sales_orders o ON o.id = ol.sales_order_id
            WHERE o.state = 'open' AND ol.line_status NOT IN ('cancelled', 'fulfilled')
              AND ol.quantity_lb > ol.quantity_shipped_lb
        ) events WHERE activity_at >= CURRENT_TIMESTAMP - INTERVAL '180 days'
        GROUP BY product_id
    ) activity ON activity.product_id = p.id
    WHERE p.active IS DISTINCT FROM false
      AND (%s IS DISTINCT FROM 'make' OR p.type IN ('batch', 'ingredient'))
      AND (%s IS DISTINCT FROM 'pack' OR %s::int IS NULL OR
           (p.type = 'finished' AND p.parent_batch_product_id = %s))'''


def _target_ids(aliases, kind, query):
    return [row[kind + '_id'] for row in aliases if row['kind'] == kind
            and normalize(row['alias']) == normalize(query)]


def _products(cur, req, variants, aliases):
    c = req.context
    ids = _target_ids(aliases, 'product', req.query)
    params = (c.customer_id, c.supplier_id, ids, normalize(req.query),
              c.action, c.action, c.group, c.action, c.action, c.action,
              c.action, c.action, c.group, c.action, c.action, c.action,
              c.action, c.action, c.product_id, c.product_id)
    rows = _search(cur, PRODUCT_SOURCE, params, variants, aliases, limit=req.limit, offset=req.offset)
    for row in rows:
        if row['score'] == 1 and (row['id'] in ids or row['_variant'].text not in
                                  (normalize(row['name']), normalize(row['odoo_code']))):
            row['tier'], row['why'] = 'alias', 'Exact product alias'
    return rows


def _parties(cur, req, variants, aliases):
    kind, c = req.kind, req.context
    ids = _target_ids(aliases, kind, req.query)
    customer = kind == 'customer'
    old_aliases = '''ARRAY(SELECT alias FROM customer_aliases
                           WHERE customer_id = p.id)''' if customer else 'ARRAY[]::text[]'
    rank = "CASE WHEN %s <> '' AND lower(COALESCE(p.address, '')) LIKE %s THEN 1 ELSE 0 END" if customer else '0'
    address = normalize(c.customer_address)
    params = (ids, normalize(req.query)) + ((address, '%' + address + '%') if customer else ())
    source = f'''SELECT p.id, p.name, ARRAY[p.name] || {old_aliases} ||
                  CASE WHEN p.id = ANY(%s::int[]) THEN ARRAY[%s::text] ELSE ARRAY[]::text[] END AS exact_terms,
                  {rank} AS context_rank, 0::float AS recent_activity
                  FROM {'customers' if customer else 'suppliers'} p
                  WHERE p.active IS DISTINCT FROM false'''
    if not customer:
        # Migration 042 normally deactivates these; never resolve a pseudo-vendor
        # even if an old catalog import accidentally reactivates it (§5.2).
        source += ' AND btrim(supplier_name_norm(p.name)) <> ALL(%s::text[])'
        params += (list(SENTINEL_SUPPLIERS),)
    rows = _search(cur, source, params, variants, aliases, limit=req.limit, offset=req.offset)
    for row in rows:
        if row['context_rank']:
            row['context_boost'] = 'Customer address agrees; ranking only'
        if row['score'] == 1 and normalize(row['name']) != normalize(req.query):
            row['tier'], row['why'] = 'alias', 'Exact party alias'
    return rows


def _lots(cur, req):
    c, q = req.context, normalize_lot(req.query)
    choose = partial(decide, q, limit=req.limit, offset=req.offset)
    # Preserve literal supplier-code identities before display suffix removal.
    variants = [Variant(value) for value in dict.fromkeys((normalize(req.query), q))]
    if not q:
        return choose([])
    source = '''SELECT l.id, l.product_id, l.lot_code, l.lot_code AS name,
          l.status, l.merged_into_lot_id, b.on_hand_lb,
          ARRAY[regexp_replace(lower(regexp_replace(btrim(l.lot_code), '\\s+', ' ', 'g')), '\\s+lot$', ''),
                l.supplier_lot_code] || ARRAY(SELECT supplier_lot_code FROM lot_supplier_codes WHERE lot_id = l.id) AS exact_terms,
          0 AS context_rank, 0::float AS recent_activity
          FROM lots l JOIN products p ON p.id = l.product_id
          CROSS JOIN LATERAL (
             SELECT COALESCE(sum(tl.quantity_lb), 0) AS on_hand_lb
             FROM ledger_current_transaction_lines tl
             JOIN ledger_current_transactions t ON t.id = tl.transaction_id
             WHERE tl.lot_id = l.id AND t.effective_status = 'posted'
          ) b
          WHERE (%s::int IS NULL OR l.product_id = %s)'''
    params = (c.product_id, c.product_id)
    exact = _search(cur, source, params, variants, [], keywords=False, exact_only=True, limit=None)
    merged = [r for r in exact if r['status'] == 'merged']
    if merged:
        return choose([], note='lot merged into ' + ', '.join(str(r['merged_into_lot_id']) for r in merged))
    eligible = " AND p.active IS DISTINCT FROM false AND l.status IS DISTINCT FROM 'merged' AND (%s OR b.on_hand_lb > 0)"
    source += eligible
    params += (c.action in ('adjust', 'void'),)
    # Re-query with eligibility in SQL so empty lots neither occupy the display
    # cap nor inflate ambiguity counts for positive-balance exact matches.
    if exact:
        eligible_exact = _search(cur, source, params, variants, [], keywords=False, exact_only=True,
                                 limit=req.limit, offset=req.offset)
        if eligible_exact:
            return choose(eligible_exact)
        return choose([], note='Exact lot exists but its product is inactive or it has no positive posted balance for this action.')
    if len(q) == 4:
        if c.product_id is None:
            return choose([], note='A product_id is required for last-four lot matching.')
        scored = rf'''SELECT matches.*, 0.8::float AS score FROM ({source} AND right(
            regexp_replace(lower(btrim(regexp_replace(l.lot_code, '\s+', ' ', 'g'))), '\s+lot$', ''), 4) = %s) matches'''
        rows = _ranked_rows(cur, scored, params + (q,), limit=req.limit, offset=req.offset)
        rows = [dict(row, tier='suffix') for row in rows]
        result = choose(rows)
        if result['outcome'] == 'ambiguous':
            result['ask'] += '. Type the full code or scan the lot.'
        return result
    rows = _search(cur, source, params, variants, [], keywords=False, limit=req.limit, offset=req.offset)
    return choose(rows)


def _orders(cur, req, aliases):
    c, q = req.context, normalize(req.query)
    choose = partial(decide, q, limit=req.limit, offset=req.offset)
    source = '''SELECT o.id, o.order_number AS name, o.order_number, o.customer_po,
                 o.customer_id, o.state, o.status,
                 ARRAY[o.order_number, o.customer_po] AS exact_terms,
                 extract(epoch FROM o.created_at)::float AS context_rank,
                 'Newest order first'::text AS context_boost, 0::float AS recent_activity
                 FROM sales_orders o JOIN customers c ON c.id = o.customer_id
                 WHERE c.active IS DISTINCT FROM false
                   AND (%s::int IS NULL OR o.customer_id = %s)
                   AND (%s::int IS NULL OR o.id = %s)
                   AND (%s::text IS NULL OR o.state = %s)
                   AND (%s::text IS NULL OR o.status = %s)'''
    params = (c.customer_id, c.customer_id, c.order_id, c.order_id, c.state, c.state, c.status, c.status)
    rows = _search(cur, source, params, [Variant(q)], [], exact_only=True, limit=req.limit, offset=req.offset)
    if rows:
        return choose(rows)
    # Controlled phrases only: an unknown identifier never falls back to any
    # convenient open order. Resolve customer text first; retain every candidate.
    browse = q in ('', 'open orders', 'orders', 'open order', 'order')
    customer_ids = [c.customer_id] if c.customer_id else []
    party_query = re.sub(r'^open orders? for\s+', '', q)
    party_query = re.sub(r'^the\s+|\s+orders?$', '', party_query).strip()
    if not browse:
        parties = resolve(cur, ResolveRequest(kind='customer', query=party_query), aliases=aliases)
        if parties['outcome'] == 'none' or parties['has_more']:
            return choose([], note='Specify an exact order number or customer_id.')
        customer_ids = [r['id'] for r in parties['candidates'] if r['score'] >= PLAUSIBLE
                        and (c.customer_id is None or r['id'] == c.customer_id)]
    if not customer_ids and not c.order_id:
        return choose([], note='Specify an order number or customer_id.')
    scored = f'''SELECT matches.*, 0.8::float AS score FROM ({source}
             AND (%s OR o.customer_id = ANY(%s::int[])) AND o.state = %s) matches'''
    rows = _ranked_rows(cur, scored, params + (bool(c.order_id), customer_ids, c.state or 'open'),
                        limit=req.limit, offset=req.offset)
    return choose([dict(row, tier='context') for row in rows])



UNITS = ('lb', 'cases', 'bags', 'boxes', 'each', 'oz')


def _unit_row(unit):
    return dict(id=unit, name=unit, unit=unit, score=1.0, tier='exact')


def _units(cur, req):
    q, product = normalize(req.query), None
    choose = partial(decide, q, limit=req.limit, offset=req.offset)
    if req.context.product_id:
        cur.execute('''SELECT id, name, type, uom, is_service, pack_format, case_size_lb,
                             bags_per_case, units_per_case, retail_bag_oz
                      FROM products WHERE id = %s AND active IS DISTINCT FROM false''', (req.context.product_id,))
        product = cur.fetchone()
        if not product:
            return choose([], note='Product is missing or inactive.')
    allowed = list(UNITS)
    if product:
        if product['is_service']:
            allowed = ['each']
        elif normalize(product['uom']) == 'each' or product['type'] in ('packaging', 'consumable'):
            allowed = ['each']
        else:
            allowed = ['lb', 'oz']
            if product['case_size_lb'] and product['case_size_lb'] > 0:
                allowed += ['cases', 'boxes']
            if product['pack_format'] == 'bagged':
                allowed += ['bags', 'each']
            elif product['type'] == 'ingredient':
                # Purchasing bags are not retail pouches. Require a declared
                # bag UOM and a positive catalog weight; explicit weights agree.
                uom = normalize(product['uom'])
                bag = re.fullmatch(r'(?:(\d+(?:\.\d+)?)\s*(?:lb|lbs|pounds?)\s+)?bags?', uom)
                weight = product['case_size_lb']
                if bag and weight and weight > 0 and (
                        bag[1] is None or Decimal(bag[1]) == Decimal(weight)):
                    allowed.append('bags')
    if q in ('pouch', 'pouches'):
        reason = None
        ratio = None
        if not product or product['pack_format'] != 'bagged' or product['type'] != 'finished' or product['is_service']:
            reason = 'Select a pouch product before converting pouches to cases.'
        elif req.quantity is None:
            reason = 'How many pouches?'
        elif req.quantity != req.quantity.to_integral_value():
            reason = 'Pouch quantity must be a whole number.'
        elif not product['case_size_lb'] or product['case_size_lb'] <= 0:
            reason = 'This product needs a valid case weight before conversion.'
        else:
            ratios = {Decimal(product[k]) for k in ('bags_per_case', 'units_per_case') if product[k] is not None}
            # Older catalog rows carry an explicit pack in the label (12x10 OZ)
            # but leave the count columns NULL. Accept only one such pack whose
            # ounces agree exactly with the stored case weight; never infer from
            # unrelated numbers or borrow another SKU's packaging.
            packs = re.findall(r'(?<!\w)([1-9]\d*)\s*[x×]\s*(\d+(?:\.\d+)?)\s*oz\b',
                               product['name'], flags=re.IGNORECASE)
            if packs:
                if len(packs) != 1:
                    ratios.add(Decimal(0))
                else:
                    count, ounces = map(Decimal, packs[0])
                    ratios.add(count if count * ounces == Decimal(product['case_size_lb']) * 16 else Decimal(0))
            if product['retail_bag_oz'] is not None:
                if product['retail_bag_oz'] > 0:
                    ratios.add(Decimal(product['case_size_lb']) * 16 / Decimal(product['retail_bag_oz']))
                else:
                    ratios.add(Decimal(0))
            if len(ratios) != 1 or any(r <= 0 or r != r.to_integral_value() for r in ratios):
                reason = 'Pouches per case is missing or inconsistent in the product catalog.'
            else:
                ratio = ratios.pop()
                if req.quantity % ratio:
                    reason = 'Needs clarification: pouch quantity does not divide evenly into whole cases.'
        if reason:
            result = choose([])
            result.update(outcome='ambiguous', ask=reason, code='NEEDS_CLARIFICATION', allowed_units=allowed)
            return result
        row = _unit_row('cases')
        row.update(quantity=req.quantity / ratio, why='Exact catalog pouches-per-case conversion')
        result = choose([row])
        result.update(draft={'product_id': product['id'], 'quantity': req.quantity / ratio, 'unit': 'cases'},
                      conversion={'from_unit': 'pouches', 'from_quantity': req.quantity,
                                  'pouches_per_case': ratio}, allowed_units=allowed)
        return result
    if q in allowed:
        result = choose([_unit_row(q)])
        if req.quantity is not None:
            result['draft'] = {'product_id': req.context.product_id, 'quantity': req.quantity, 'unit': q}
    elif not q or re.fullmatch(r'\d+(?:\.\d+)?', q):
        result = choose([_unit_row(unit) for unit in allowed])
        # A missing unit is always a question, even for an each-only service.
        result.update(outcome='ambiguous', match=None, needs_clarification=True,
                      ask='Which unit? ' + ', '.join(allowed), code='UNIT_REQUIRED')
    else:
        result = choose([], note='Unit is unknown or not allowed for this product.')
    result['allowed_units'] = allowed
    return result


def resolve(cur, req, *, aliases=None):
    """Public shared core. ``aliases`` is an internal fixture seam, never HTTP input."""
    if req.kind == 'unit':
        return _units(cur, req)
    if req.kind == 'lot':
        return _lots(cur, req)
    if aliases is None:
        aliases, available = read_aliases(cur)
    else:
        available = True
    aliases = [row for row in aliases if row.get('active', True)]
    if req.kind == 'order':
        result = _orders(cur, req, aliases)
    else:
        variants, truncated = query_variants(req.query, aliases)
        rows = (_products if req.kind == 'product' else _parties)(cur, req, variants, aliases)
        result = decide(req.query, rows, truncated=truncated, limit=req.limit, offset=req.offset)
    result['alias_table_available'] = available
    return result


def resolve_bulk_product(cur, name):
    """Preserve the legacy envelope; never turn the first suggestion into a match."""
    result = resolve(cur, ResolveRequest(kind='product', query=name))
    def legacy(row):
        return {k: row[k] for k in ('id', 'name', 'odoo_code')}
    match = result['match']
    if match:
        # Existing consumers (and auth contract tests) expect this exact shape
        # for successful exact lookups. New ambiguity fields are additive only
        # on unresolved results; /resolve always returns the complete envelope.
        return dict(input=name, match=legacy(match), match_tier=match['tier'],
                    confidence=result['confidence'])
    return dict(result, input=name, match=legacy(match) if match else None,
                match_tier=match['tier'] if match else None,
                alternatives=[legacy(row) for row in result['candidates'] if not match or row['id'] != match['id']],
                suggestions=result.get('near_misses', []))


def build_router(transaction, authenticate):
    router = APIRouter()

    @router.post('/resolve')
    def resolve_endpoint(req: ResolveRequest, _: bool = Depends(authenticate)):
        with transaction() as cur:
            return resolve(cur, req)

    @router.get('/aliases')
    def aliases_endpoint(kind: Optional[AliasKind] = None, active: bool = True,
                         _: bool = Depends(authenticate)):
        with transaction() as cur:
            aliases, available = read_aliases(cur)
        rows = [row for row in aliases if row['active'] == active and (kind is None or row['kind'] == kind)]
        return {'aliases': rows, 'count': len(rows), 'alias_table_available': available}

    return router
