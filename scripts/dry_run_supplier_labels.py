#!/usr/bin/env python3
"""READ ONLY: preview migration 066 labels before/after Michael's supplier cleanup.

python scripts/dry_run_supplier_labels.py --database-url-file PATH \
    [--assume-inactive 11,12,14,15] [--assume-label 13=DUTG --assume-label 16=DUTV]

Reads a protected mode-600 URI (port 5432 remotely); only SELECTs in a read-only
transaction. Works before 064/066, reserves existing labels, skips inactive
vendors (066 labels active suppliers only) and reports excluded pseudo-suppliers.
The --assume-* flags overlay a planned cleanup on the catalog that was read, so
the post-cleanup labels can be previewed before anything is written. No app
import, DDL or SQL label allocator calls. Output contains catalog names/IDs/labels,
never credentials.
"""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resolution import SENTINEL_SUPPLIERS
from scripts.a5_readonly import readonly_cursor

LABEL_RE = re.compile(r'^[A-Z]{4}$')


def plan_labels(rows):
    """Mirror 066's ordered allocation in memory; base_code is computed by SQL.

    Decisions: 'retained' (explicit/existing label, never overwritten),
    'excluded pseudo-supplier', 'skipped inactive' (066 labels active suppliers
    only; the trigger assigns one on activation), or 'assign'.
    """
    used = {row['current_label'] for row in rows if row['current_label']}
    result = []
    for row in sorted(rows, key=lambda row: row['id']):
        label = row['current_label']
        if label:
            decision = 'retained'
        elif row['pseudo']:
            decision = 'excluded pseudo-supplier'
        elif not row['active']:
            decision = 'skipped inactive'
        else:
            decision = 'assign'
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


def apply_assumptions(rows, inactive_ids=(), labels=None):
    """Overlay a planned cleanup (deactivations, explicit labels) on catalog rows.

    Pure function used for read-only previews. Raises ValueError when an
    assumed label is malformed, targets an unknown id, or is already held by a
    different supplier (the unique index would reject that write).
    """
    labels = dict(labels or {})
    by_id = {row['id']: dict(row) for row in rows}
    for sid in inactive_ids:
        if sid not in by_id:
            raise ValueError(f'--assume-inactive: unknown supplier id {sid}')
        by_id[sid]['active'] = False
    for sid, label in labels.items():
        if sid not in by_id:
            raise ValueError(f'--assume-label: unknown supplier id {sid}')
        if not LABEL_RE.match(label or ''):
            raise ValueError(f'--assume-label: {label!r} is not four uppercase letters')
        holder = next((r['id'] for r in by_id.values() if r['current_label'] == label and r['id'] != sid), None)
        if holder is not None and labels.get(holder) in (None, label):
            raise ValueError(f'--assume-label: {label} is already held by supplier {holder}')
        by_id[sid]['current_label'] = label
    return sorted(by_id.values(), key=lambda row: row['id'])


def read_catalog(cur):
    cur.execute("""SELECT EXISTS (SELECT 1 FROM information_schema.columns
        WHERE table_schema='public' AND table_name='suppliers' AND column_name='short_code') AS present""")
    # Both projections are fixed strings, never input-derived SQL.
    label = 'short_code' if cur.fetchone()['present'] else 'NULL::text'
    cur.execute(f"""SELECT id,name,active,{label} AS current_label,
        rpad(left(regexp_replace(upper(name),'[^A-Z]','','g'),4),4,'X') AS base_code,
        btrim(supplier_name_norm(name)) = ANY(%s::text[]) AS pseudo
        FROM suppliers ORDER BY id""", (list(SENTINEL_SUPPLIERS),))
    return [dict(row) for row in cur.fetchall()]


def read_label_plan(cur, inactive_ids=(), labels=None):
    return plan_labels(apply_assumptions(read_catalog(cur), inactive_ids, labels))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--database-url-file', required=True)
    parser.add_argument('--assume-inactive', default='',
                        help='comma-separated supplier ids to treat as deactivated')
    parser.add_argument('--assume-label', action='append', default=[], metavar='ID=CODE',
                        help='explicit label to assume for a supplier (repeatable)')
    args = parser.parse_args(argv)
    try:
        args.inactive_ids = [int(x) for x in args.assume_inactive.split(',') if x.strip()]
        args.labels = {}
        for item in args.assume_label:
            sid, code = item.split('=', 1)
            args.labels[int(sid)] = code
    except ValueError:
        parser.error('--assume-inactive takes ids like 11,12 and --assume-label takes ID=CODE')
    return args


def main():
    args = parse_args()
    try:
        with readonly_cursor(args.database_url_file) as cur:
            suppliers = read_label_plan(cur, args.inactive_ids, args.labels)
    except ValueError as exc:  # assumption errors carry no credentials
        print(f'Label dry-run refused: {exc}', file=sys.stderr)
        return 2
    except Exception:
        # libpq exceptions may expose credentials. Suppress their text/traceback.
        print('Label dry-run failed; verify the protected URI file, port 5432 and schema.', file=sys.stderr)
        return 2
    print(json.dumps({'read_only': True, 'applied': False,
        'assumptions': {'inactive_ids': args.inactive_ids, 'labels': args.labels},
        'notice': 'Preview only; labels are written by migration 066 after the supplier cleanup is applied.',
        'suppliers': suppliers}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
