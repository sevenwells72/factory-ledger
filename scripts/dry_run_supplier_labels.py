#!/usr/bin/env python3
"""READ ONLY: preview migration 066 labels before/after Michael's supplier cleanup.

python scripts/dry_run_supplier_labels.py --database-url-file PATH
Reads a protected mode-600 URI (port 5432 remotely); only SELECTs in a read-only
transaction. Works before 064/066, reserves existing labels, includes inactive
vendors and reports excluded pseudo-suppliers. No app import, DDL or SQL label
allocator calls. Output contains catalog names/IDs/labels, never credentials.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resolution import SENTINEL_SUPPLIERS
from scripts.a5_readonly import readonly_cursor


def plan_labels(rows):
    """Mirror 066's ordered allocation in memory; base_code is computed by SQL."""
    used = {row['current_label'] for row in rows if row['current_label']}
    result = []
    for row in sorted(rows, key=lambda row: row['id']):
        label = row['current_label']
        decision = 'retained' if label else 'excluded pseudo-supplier' if row['pseudo'] else 'assign'
        if decision == 'assign':
            base = row['base_code']
            label, n = base, -1
            while label in used:
                n += 1
                if n >= 456976:
                    raise RuntimeError('Supplier display labels exhausted')
                if n < 26:
                    label = base[:3] + chr(65 + n)
                else:
                    label = ''.join(chr(65 + (n // divisor) % 26) for divisor in (17576, 676, 26, 1))
            used.add(label)
        result.append({key: row[key] for key in ('id', 'name', 'active', 'current_label')} |
                      {'proposed_label': label, 'decision': decision})
    return result


def read_label_plan(cur):
    cur.execute("""SELECT EXISTS (SELECT 1 FROM information_schema.columns
        WHERE table_schema='public' AND table_name='suppliers' AND column_name='short_code') AS present""")
    # Both projections are fixed strings, never input-derived SQL.
    label = 'short_code' if cur.fetchone()['present'] else 'NULL::text'
    cur.execute(f"""SELECT id,name,active,{label} AS current_label,
        rpad(left(regexp_replace(upper(name),'[^A-Z]','','g'),4),4,'X') AS base_code,
        btrim(supplier_name_norm(name)) = ANY(%s::text[]) AS pseudo
        FROM suppliers ORDER BY id""", (list(SENTINEL_SUPPLIERS),))
    return plan_labels([dict(row) for row in cur.fetchall()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database-url-file', required=True)
    args = parser.parse_args()
    try:
        with readonly_cursor(args.database_url_file) as cur:
            suppliers = read_label_plan(cur)
        print(json.dumps({'read_only': True, 'applied': False,
            'notice': 'Current-catalog preview only; rerun AFTER Michael approves supplier cleanup.',
            'suppliers': suppliers}, indent=2))
        return 0
    except Exception:
        # libpq exceptions may expose credentials. Suppress their text/traceback.
        print('Label dry-run failed; verify the protected URI file, port 5432 and schema.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
