#!/usr/bin/env python3
"""Archive every Blue Stripes (BS) item (owner request, Michael, 2026-11-09).

Archive = products.active=false + archived_at/archived_by (migration 093).
Nothing is deleted; transactions, lots, BOM lines and quantities are untouched.
Stock on hand does not matter (owner decision): items with stock are archived too.

Selection (owner's rule): the name starts with "BS ", or contains "BS Granola",
"BS Almond Butter Granola" or "– BS ", or starts with "Batch BS " (the four BS
batch recipes the owner listed by name). Every selected name must also contain
"BS" as a word; "Sunflower Seeds" is never selected. Already-archived rows
(the 6x8 OZ cases, row 152) are left exactly as they are.

Usage (default is a read-only dry run that changes nothing):
  DATABASE_URL=... python3 scripts/archive_bs_items.py
  DATABASE_URL=... python3 scripts/archive_bs_items.py --apply --expect 30
--apply needs --expect N (the dry run's count); a different count aborts.
Use the session pooler (port 5432); port 6543 is refused (CLAUDE.md).
Production: only with Michael's approval for this step and a verified backup.
"""
import argparse
import os
import re
import sys

import psycopg2
from psycopg2.extras import RealDictCursor

ARCHIVED_BY = 'Michael'
NEVER = {'sunflower seeds'}
SELECT_SQL = """
    SELECT p.id, p.name, p.type, COALESCE(p.active, true) AS active,
           COALESCE((SELECT SUM(tl.quantity_lb) FROM lots l
                     JOIN transaction_lines tl ON tl.lot_id = l.id
                     JOIN transactions t ON t.id = tl.transaction_id AND t.status = 'posted'
                     WHERE l.product_id = p.id), 0) AS on_hand
    FROM products p
    WHERE p.name LIKE 'BS %' OR p.name LIKE 'Batch BS %'
       OR p.name LIKE '%BS Granola%' OR p.name LIKE '%BS Almond Butter Granola%'
       OR p.name LIKE '%– BS %'
    ORDER BY p.type, p.name
"""


def selected(rows):
    out = []
    for r in rows:
        name = r['name']
        if name.strip().lower() in NEVER or not re.search(r'\bBS\b', name):
            raise SystemExit(f'refusing: selection rule matched a non-BS name {name!r}')
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--expect', type=int)
    args = ap.parse_args()
    url = os.environ.get('DATABASE_URL')
    if not url:
        raise SystemExit('DATABASE_URL is not set')
    if ':6543' in url:
        raise SystemExit('refusing port 6543 (transaction pooler); use 5432')
    if args.apply and args.expect is None:
        raise SystemExit('--apply needs --expect N from the dry run')

    conn = psycopg2.connect(url, cursor_factory=RealDictCursor)
    conn.autocommit = True  # explicit BEGIN / READ ONLY / COMMIT below
    try:
        with conn.cursor() as cur:
            cur.execute('BEGIN')
            if not args.apply:
                cur.execute('SET TRANSACTION READ ONLY')
            cur.execute(SELECT_SQL)
            rows = selected(cur.fetchall())
            todo = [r for r in rows if r['active']]
            for r in rows:
                state = 'ARCHIVE' if r['active'] else 'already archived'
                print(f"{state:16} {r['id']:>5} {r['type']:<10} {float(r['on_hand']):>10.1f} lb  {r['name']}")
            print(f'{len(todo)} to archive, {len(rows) - len(todo)} already archived')
            if not args.apply:
                cur.execute('ROLLBACK')
                print('DRY RUN: nothing changed')
                return
            if len(todo) != args.expect:
                cur.execute('ROLLBACK')
                raise SystemExit(f'expected {args.expect}, found {len(todo)}; nothing changed')
            cur.execute("SELECT 1 FROM migration_markers WHERE name='093_product_archive'")
            if cur.fetchone() is None:
                cur.execute('ROLLBACK')
                raise SystemExit('migration 093 is not applied; nothing changed')
            cur.execute("SET LOCAL lock_timeout='5s'")
            cur.execute("""UPDATE products SET active = false, archived_at = now(), archived_by = %s
                           WHERE id = ANY(%s) AND COALESCE(active, true) = true""",
                        (ARCHIVED_BY, [r['id'] for r in todo]))
            if cur.rowcount != len(todo):
                cur.execute('ROLLBACK')
                raise SystemExit(f'updated {cur.rowcount}, expected {len(todo)}; rolled back')
            cur.execute('COMMIT')
            print(f'ARCHIVED {len(todo)} items (archived_by={ARCHIVED_BY})')
    finally:
        conn.close()


if __name__ == '__main__':
    sys.exit(main())
