#!/usr/bin/env python3
"""Recipe corrections (migration 078) — staging acceptance. STAGING ONLY.

Never opens production, never deploys. Reads the protected staging URI (600-permission
secret file, `staging_safety` guard, documented host) and, like the A6/A7/A9 checks,
drives the real `/make` and `/pack` handlers through a local TestClient bound to the
staging database (same code path as the hosted service; nothing is deployed).

Steps (each flag is explicit; nothing runs by accident):
  --seed              staging has NO formula rows for 179/284 and no stock for their
                      ingredients, so this seeds, once (marker
                      `staging_recipe_fixture_078_v1`), the production BEFORE state
                      recorded 2026-10-13 — batch_formulas 179 (20 + 1.25×4) and 284
                      (8 lines, chips = 73), product_bom 179 (323 + 15.38×4) and
                      142 → 179 @ 25 lb — plus clearly labelled test stock
                      (lot codes `STG-078-*`, transaction notes "STAGING FIXTURE 078",
                      operator `staging-fixture-078`). No stock for 73 (1,000 CT): a
                      284 make can only succeed if the chips line moved to 72.
  --apply-migration   applies migrations/078_recipe_corrections.sql to STAGING in one
                      transaction (lock_timeout 5s, statement_timeout 60s).
  --restock           with --round N (N > 1; marker `staging_recipe_fixture_078_v<N>`):
                      one more batch's worth of the same labelled test stock (lot
                      codes `STG-078-R<N>-*`), still nothing for 73, so the makes can
                      be re-run after 078 changed (fix round 1: 179 yields 13/32,
                      chips-line exclusion guard).
  --round N           lot-code suffix for --make (default 1 → `…-001`) and receipt
                      file (round > 1 → recipe-corrections-staging-receipt-r<N>.json).
  --make              one make of 179 (1 batch) and one of 284 (1 batch) through
                      POST /make, then POST /pack preview for 13 and 14 cases of 142;
                      verifies the posted lines and writes the sanitized receipt to
                      docs/deployments/recipe-corrections-staging-receipt.json.
"""
import argparse
from contextlib import contextmanager
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
import secrets
import sys
from urllib.parse import urlsplit

import psycopg2
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staging_safety import assert_staging_database, PRODUCTION_DATABASE_HOST  # noqa: E402
from scripts.seed_staging import secret_file  # noqa: E402

OUT = ROOT / 'docs/deployments/recipe-corrections-staging-receipt.json'
MIGRATION = ROOT / 'migrations/078_recipe_corrections.sql'
MARKER = 'staging_recipe_fixture_078_v1'
LABEL = 'STAGING FIXTURE 078 — recipe-correction test stock, not real inventory'
OPERATOR = 'staging-fixture-078'

FRUIT_NUT, FRUIT_NUT_FG, CLASSIC9, KOSHER_CC = 179, 142, 107, 284
ALMONDS, WALNUTS, RAISINS, CRANBERRIES, CHIPS_4000, CHIPS_1000 = 3, 49, 36, 17, 72, 73
INCLUSIONS = (ALMONDS, WALNUTS, RAISINS, CRANBERRIES)
# Production BEFORE state, 2026-10-13 17:43 UTC (ids are production's; staging assigns its own).
FORMULAS_BEFORE = [
    (FRUIT_NUT, CLASSIC9, 20), (FRUIT_NUT, WALNUTS, 1.25), (FRUIT_NUT, ALMONDS, 1.25),
    (FRUIT_NUT, RAISINS, 1.25), (FRUIT_NUT, CRANBERRIES, 1.25),
    (KOSHER_CC, 13, 12.5), (KOSHER_CC, 18, 1.8), (KOSHER_CC, 30, 1.6), (KOSHER_CC, 32, 225),
    (KOSHER_CC, 34, 30), (KOSHER_CC, 39, 1.7), (KOSHER_CC, 44, 50), (KOSHER_CC, CHIPS_1000, 25),
]
PLANNER_BEFORE = [
    (FRUIT_NUT_FG, FRUIT_NUT, 25),
    (FRUIT_NUT, CLASSIC9, 323), (FRUIT_NUT, ALMONDS, 15.38), (FRUIT_NUT, WALNUTS, 15.38),
    (FRUIT_NUT, RAISINS, 15.38), (FRUIT_NUT, CRANBERRIES, 15.38),
]
# Test stock: enough for one batch of each, nothing for 1,000 CT chips (73).
STOCK = [(CLASSIC9, 'STG-078-CLASSIC9', 400), (ALMONDS, 'STG-078-ALMONDS', 30), (WALNUTS, 'STG-078-WALNUTS', 30),
         (RAISINS, 'STG-078-RAISINS', 30), (CRANBERRIES, 'STG-078-CRANBERRIES', 30), (CHIPS_4000, 'STG-078-CHIPS-4000CT', 60),
         (13, 'STG-078-COCONUT', 20), (18, 'STG-078-ALMOND-FLAVOR', 5), (30, 'STG-078-HONEY', 5), (32, 'STG-078-OATS', 300),
         (34, 'STG-078-CANOLA', 40), (39, 'STG-078-SALT', 5), (44, 'STG-078-BROWN-SUGAR', 70)]


def staging_uri():
    uri = secret_file(Path.home() / 'Documents/fl-secrets/staging-db-url.txt')
    assert_staging_database(uri, 'staging', PRODUCTION_DATABASE_HOST)
    if urlsplit(uri).hostname != 'aws-0-us-east-1.pooler.supabase.com':
        raise RuntimeError('Not the documented staging database')
    return uri


@contextmanager
def connection(uri):
    conn = psycopg2.connect(uri, port=5432, connect_timeout=10, sslmode='require')
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL search_path=public; SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='60s'")
            yield conn
    finally:
        conn.close()


def q(cur, sql, params=()):
    cur.execute(sql, params)
    return [dict(r) for r in cur.fetchall()]


def seed(uri):
    with connection(uri) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        if q(cur, 'SELECT 1 FROM migration_markers WHERE name=%s', (MARKER,)):
            return {'already_seeded': True}
        cur.execute('SELECT pg_advisory_xact_lock(78078)')
        catalog = {r['id']: r for r in q(cur, 'SELECT id, odoo_code, type FROM products WHERE id = ANY(%s)',
                                           ([FRUIT_NUT, FRUIT_NUT_FG, CLASSIC9, KOSHER_CC, CHIPS_1000, CHIPS_4000, 13, 18, 30, 32, 34, 39, 44, *INCLUSIONS],))}
        expected = {FRUIT_NUT: '90008', FRUIT_NUT_FG: '70061', CLASSIC9: '90002', KOSHER_CC: '90026', CHIPS_1000: '25011', CHIPS_4000: '25010'}
        for pid, code in expected.items():
            if catalog.get(pid, {}).get('odoo_code') != code:
                raise RuntimeError(f'staging product {pid} is not {code}; refusing to seed')
        if q(cur, 'SELECT 1 FROM batch_formulas WHERE product_id IN (179, 284)') or q(cur, 'SELECT 1 FROM product_bom WHERE finished_product_id IN (179, 142)'):
            raise RuntimeError('staging already has 179/284 formula or planner rows; seed refused (marker absent)')
        for row in FORMULAS_BEFORE:
            cur.execute('INSERT INTO batch_formulas (product_id, ingredient_product_id, quantity_lb) VALUES (%s, %s, %s)', row)
        for row in PLANNER_BEFORE:
            cur.execute("INSERT INTO product_bom (finished_product_id, component_product_id, quantity, uom) VALUES (%s, %s, %s, 'lb')", row)
        cur.execute("INSERT INTO transactions (type, timestamp, notes, occurred_at, operator_id) VALUES ('adjust', now(), %s, now(), %s) RETURNING id",
                    ('Found inventory: ' + LABEL, OPERATOR))
        txn = cur.fetchone()['id']
        lots = []
        for pid, code, lb in STOCK:
            cur.execute("INSERT INTO lots (product_id, lot_code, entry_source, entry_source_notes, received_at) VALUES (%s, %s, 'found_inventory', %s, now()) RETURNING id",
                        (pid, code, LABEL))
            lot = cur.fetchone()['id']
            cur.execute('INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)', (txn, pid, lot, lb))
            lots.append({'product_id': pid, 'lot_code': code, 'lb': lb, 'lot_id': lot})
        cur.execute('INSERT INTO migration_markers (name) VALUES (%s)', (MARKER,))
        if min(txn, *(l['lot_id'] for l in lots)) < 1_000_000_000:
            raise RuntimeError('Staging synthetic ID range is required')
        return {'seeded': True, 'fixture_transaction_id': txn, 'lots': lots,
                'formula_rows': len(FORMULAS_BEFORE), 'planner_rows': len(PLANNER_BEFORE)}


def restock(uri, round_no):
    if round_no < 2:
        raise RuntimeError('--restock needs --round 2 or higher')
    restock_marker = f'staging_recipe_fixture_078_v{round_no}'
    with connection(uri) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        if q(cur, 'SELECT 1 FROM migration_markers WHERE name=%s', (restock_marker,)):
            return {'already_restocked': True}
        if not q(cur, 'SELECT 1 FROM migration_markers WHERE name=%s', (MARKER,)):
            raise RuntimeError('run --seed first; restock only tops up the 078 fixture')
        cur.execute('SELECT pg_advisory_xact_lock(78078)')
        cur.execute("INSERT INTO transactions (type, timestamp, notes, occurred_at, operator_id) VALUES ('adjust', now(), %s, now(), %s) RETURNING id",
                    ('Found inventory: ' + LABEL + f' (top-up round {round_no})', OPERATOR))
        txn = cur.fetchone()['id']
        lots = []
        for pid, code, lb in STOCK:
            code = code.replace('STG-078-', f'STG-078-R{round_no}-')
            cur.execute("INSERT INTO lots (product_id, lot_code, entry_source, entry_source_notes, received_at) VALUES (%s, %s, 'found_inventory', %s, now()) RETURNING id",
                        (pid, code, LABEL))
            lot = cur.fetchone()['id']
            cur.execute('INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)', (txn, pid, lot, lb))
            lots.append({'product_id': pid, 'lot_code': code, 'lb': lb, 'lot_id': lot})
        cur.execute('INSERT INTO migration_markers (name) VALUES (%s)', (restock_marker,))
        if min(txn, *(l['lot_id'] for l in lots)) < 1_000_000_000:
            raise RuntimeError('Staging synthetic ID range is required')
        return {'restocked': True, 'fixture_transaction_id': txn, 'lots': lots}


def apply_migration(uri):
    with connection(uri) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(MIGRATION.read_text())
        cur.execute("SELECT applied_at FROM migration_markers WHERE name='078_recipe_corrections'")
        row = cur.fetchone()
        if not row:
            raise RuntimeError('078 marker missing after apply')
        return {'applied': True, 'marker_applied_at': row['applied_at']}


def recipe_state(uri):
    with connection(uri) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        return {
            'products': q(cur, 'SELECT id, odoo_code, default_batch_lb, yield_25lb_cases, yield_10lb_cases, case_size_lb, parent_batch_product_id FROM products WHERE id IN (179, 284, 142) ORDER BY id'),
            'batch_formulas': q(cur, 'SELECT product_id, ingredient_product_id, quantity_lb, COALESCE(exclude_from_inventory, false) AS excluded FROM batch_formulas WHERE product_id IN (179, 284) ORDER BY product_id, ingredient_product_id'),
            'product_bom': q(cur, 'SELECT finished_product_id, component_product_id, quantity, uom FROM product_bom WHERE finished_product_id IN (179, 142) ORDER BY finished_product_id, component_product_id'),
            'marker_078': bool(q(cur, "SELECT 1 FROM migration_markers WHERE name='078_recipe_corrections'")),
        }


def on_hand(cur, product_id):
    import main
    cur.execute(f'SELECT COALESCE(SUM(tl.quantity_lb),0) AS lb FROM lots l JOIN {main.POSTED_LINES} tl ON tl.lot_id=l.id WHERE l.product_id=%s', (product_id,))
    return cur.fetchone()['lb']


def posted_lines(cur, txn_id):
    rows = q(cur, 'SELECT tl.product_id, p.name, l.lot_code, tl.quantity_lb FROM transaction_lines tl JOIN products p ON p.id=tl.product_id '
                  'LEFT JOIN lots l ON l.id=tl.lot_id WHERE tl.transaction_id=%s ORDER BY tl.quantity_lb DESC, tl.product_id', (txn_id,))
    ilc = {r['ingredient_product_id']: r['lb'] for r in q(cur, 'SELECT ingredient_product_id, SUM(quantity_lb) AS lb FROM ingredient_lot_consumption WHERE transaction_id=%s GROUP BY 1', (txn_id,))}
    consumed = {}
    for r in rows:
        if r['quantity_lb'] < 0:
            consumed[r['product_id']] = consumed.get(r['product_id'], Decimal(0)) - r['quantity_lb']
    if consumed != ilc:
        raise RuntimeError(f'ILC {ilc} does not mirror the ledger lines {consumed}')
    return rows, consumed


def make_and_pack(uri, round_no=1):
    from fastapi.testclient import TestClient
    os.environ.update(DATABASE_URL=uri, ENVIRONMENT='staging', PRODUCTION_DATABASE_HOST=PRODUCTION_DATABASE_HOST,
                      API_KEY=secrets.token_urlsafe(40), DASHBOARD_API_KEY=secrets.token_urlsafe(40))
    logging.disable(logging.CRITICAL)
    import main
    main.get_db_connection = lambda: connection(uri)
    http = TestClient(main.app)
    headers = {'X-API-Key': os.environ['API_KEY']}

    def post(path, payload, status=200):
        r = http.post(path, json=payload, headers=headers)
        if r.status_code != status:
            raise RuntimeError(f'Unexpected HTTP {r.status_code} on {path}: {r.text[:400]}')
        return r.json()

    def num(x):
        return Decimal(str(x))

    receipt = {}
    with connection(uri) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        if not q(cur, "SELECT 1 FROM migration_markers WHERE name='078_recipe_corrections'"):
            raise RuntimeError('Apply 078 to staging first (--apply-migration)')
        before = {pid: on_hand(cur, pid) for pid in (FRUIT_NUT, CLASSIC9, *INCLUSIONS, KOSHER_CC, CHIPS_4000, CHIPS_1000)}

    # ── Fruit Nut: 1 batch → 328.3 lb out; 276.3 lb Classic #9 + 4 × 13 lb deducted ──
    preview = post('/make', {'mode': 'preview', 'product_name': '90008', 'batches': 1})
    if num(preview['batch_size_lb']) != Decimal('328.3') or num(preview['total_output_lb']) != Decimal('328.3'):
        raise RuntimeError(f'Fruit Nut preview batch size wrong: {preview["batch_size_lb"]} / {preview["total_output_lb"]}')
    needed = {i['ingredient_id']: num(i['needed_lb']) for i in preview['ingredients']}
    if needed != {CLASSIC9: Decimal('276.3'), **{i: Decimal('13') for i in INCLUSIONS}} or not preview['all_ingredients_available']:
        raise RuntimeError(f'Fruit Nut preview needs wrong: {needed}')
    fn_lot = f'STG-078-FRUIT-NUT-{round_no:03d}'
    with connection(uri) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Resume: a round whose Fruit Nut make already posted is re-verified, never made twice.
        posted = q(cur, "SELECT t.id AS transaction_id, l.lot_code, tl.quantity_lb AS output_lb, t.notes AS message FROM transactions t "
                        "JOIN transaction_lines tl ON tl.transaction_id=t.id JOIN lots l ON l.id=tl.lot_id "
                        "WHERE t.type='make' AND l.product_id=%s AND l.lot_code=%s AND tl.quantity_lb>0", (FRUIT_NUT, fn_lot))
    resumed = bool(posted)
    if resumed:
        fn = dict(posted[0], formula_weight_lb=None, resumed=True)
    else:
        fn = post('/make', {'mode': 'commit', 'product_name': '90008', 'batches': 1, 'lot_code': fn_lot, 'confirmed_sku': True})
    with connection(uri) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        fn_lines, fn_consumed = posted_lines(cur, fn['transaction_id'])
        fn_after = {pid: on_hand(cur, pid) for pid in (FRUIT_NUT, CLASSIC9, *INCLUSIONS)}
    if num(fn['output_lb']) != Decimal('328.3') or fn_consumed != {CLASSIC9: Decimal('276.3'), **{i: Decimal('13') for i in INCLUSIONS}}:
        raise RuntimeError(f'Fruit Nut make posted wrong lines: out {fn["output_lb"]}, consumed {fn_consumed}')
    if not resumed and (fn_after[FRUIT_NUT] - before[FRUIT_NUT] != Decimal('328.3') or before[CLASSIC9] - fn_after[CLASSIC9] != Decimal('276.3')
                        or any(before[i] - fn_after[i] != Decimal('13') for i in INCLUSIONS)):
        raise RuntimeError('Fruit Nut balances did not move by the corrected recipe')
    receipt['fruit_nut_make'] = {'resumed_already_posted': resumed, 'transaction_id': fn['transaction_id'], 'lot_code': fn['lot_code'], 'output_lb': fn['output_lb'],
                                 'formula_weight_lb': fn['formula_weight_lb'], 'message': fn['message'],
                                 'lines': [{'product_id': r['product_id'], 'name': r['name'], 'lot_code': r['lot_code'], 'quantity_lb': r['quantity_lb']} for r in fn_lines],
                                 'on_hand_before': {str(k): v for k, v in before.items() if k in fn_after}, 'on_hand_after': {str(k): v for k, v in fn_after.items()},
                                 'pack_needed': fn.get('pack_needed', {}).get('finished_goods')}

    # ── Pack consistency for 142 (25-lb case): 13 cases fit one batch, 14 do not ──
    fourteen = post('/pack', {'mode': 'preview', 'source_product': '90008', 'target_product': '70061', 'cases': 14})
    thirteen = post('/pack', {'mode': 'preview', 'source_product': '90008', 'target_product': '70061', 'cases': 13})
    if thirteen['case_weight_lb'] != 25 or thirteen['total_lb'] != 325 or not thirteen['all_lots_sufficient'] or 'add_in_ingredients' in thirteen:
        raise RuntimeError(f'13-case pack preview inconsistent: {thirteen}')
    if fourteen['total_lb'] != 350 or (fourteen['all_lots_sufficient'] and num(fourteen['total_batch_available_lb']) < Decimal('350')):
        raise RuntimeError(f'14-case pack preview inconsistent: {fourteen}')
    receipt['fruit_nut_pack_preview'] = {
        '13_cases': {k: thirteen[k] for k in ('case_weight_lb', 'total_lb', 'all_lots_sufficient', 'total_batch_available_lb', 'can_pack_lb')},
        '14_cases': {k: fourteen[k] for k in ('case_weight_lb', 'total_lb', 'all_lots_sufficient', 'total_batch_available_lb', 'can_pack_lb')},
        'note': 'preview only; no pack posted',
    }

    # ── 284: 1 batch → 348 lb; chips deducted from 72 (4,000 CT); 73 untouched (0 lb on hand) ──
    preview = post('/make', {'mode': 'preview', 'product_name': '90026', 'batches': 1})
    chips = [(i['ingredient_id'], num(i['needed_lb']), i['sufficient']) for i in preview['ingredients'] if i['ingredient_id'] in (CHIPS_1000, CHIPS_4000)]
    if chips != [(CHIPS_4000, Decimal('25'), True)] or not preview['all_ingredients_available']:
        raise RuntimeError(f'284 preview chips wrong: {chips}')
    kc_preview = {'total_output_lb': preview['total_output_lb'], 'all_ingredients_available': preview['all_ingredients_available'],
                  'needs': {str(i['ingredient_id']): i['needed_lb'] for i in preview['ingredients']}}
    r = http.post('/make', json={'mode': 'commit', 'product_name': '90026', 'batches': 1, 'lot_code': f'STG-078-KOSHER-CC-{round_no:03d}', 'confirmed_sku': True}, headers=headers)
    if r.status_code == 403 and (r.json().get('error_detail') or {}).get('code') == 'KOSHER_ATTESTATION_REQUIRED':
        # A12 (075): 284 is extra-kosher on staging; a direct make needs a ticket approved
        # from the owner's own signed-in session. Not simulated here.
        receipt['kosher_cc_make'] = {'preview': kc_preview, 'commit': 'refused by A12: KOSHER_ATTESTATION_REQUIRED (owner-approved ticket required)'}
        return receipt
    if r.status_code != 200:
        raise RuntimeError(f'Unexpected HTTP {r.status_code} on /make: {r.text[:400]}')
    kc = r.json()
    with connection(uri) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        kc_lines, kc_consumed = posted_lines(cur, kc['transaction_id'])
        kc_after = {pid: on_hand(cur, pid) for pid in (KOSHER_CC, CHIPS_4000, CHIPS_1000)}
    if num(kc['output_lb']) != Decimal('348') or kc_consumed.get(CHIPS_4000) != Decimal('25') or CHIPS_1000 in kc_consumed \
            or before[CHIPS_4000] - kc_after[CHIPS_4000] != Decimal('25') or kc_after[CHIPS_1000] != before[CHIPS_1000]:
        raise RuntimeError(f'284 make posted wrong lines: out {kc["output_lb"]}, consumed {kc_consumed}')
    if kc_consumed != {13: Decimal('12.5'), 18: Decimal('1.8'), 30: Decimal('1.6'), 32: Decimal('225'), 34: Decimal('30'), 39: Decimal('1.7'), 44: Decimal('50'), CHIPS_4000: Decimal('25')}:
        raise RuntimeError(f'284 consumed set unexpected: {kc_consumed}')
    receipt['kosher_cc_make'] = {'transaction_id': kc['transaction_id'], 'lot_code': kc['lot_code'], 'output_lb': kc['output_lb'], 'message': kc['message'],
                                 'production_warning': (kc.get('production_warning') or {}).get('message'),
                                 'lines': [{'product_id': r['product_id'], 'name': r['name'], 'lot_code': r['lot_code'], 'quantity_lb': r['quantity_lb']} for r in kc_lines],
                                 'chips_4000ct_on_hand_before_after': [before[CHIPS_4000], kc_after[CHIPS_4000]],
                                 'chips_1000ct_on_hand_before_after': [before[CHIPS_1000], kc_after[CHIPS_1000]]}
    return receipt


def main_entry():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', action='store_true')
    parser.add_argument('--apply-migration', action='store_true')
    parser.add_argument('--restock', action='store_true')
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--make', action='store_true')
    args = parser.parse_args()
    uri = staging_uri()
    receipt = {'environment': 'staging', 'host': urlsplit(uri).hostname, 'migration': '078_recipe_corrections'}
    if args.seed:
        receipt['seed'] = seed(uri)
    if args.restock:
        receipt['restock'] = restock(uri, args.round)
    receipt['recipe_state_before_apply'] = recipe_state(uri)
    if args.apply_migration:
        receipt['apply'] = apply_migration(uri)
        receipt['recipe_state_after_apply'] = recipe_state(uri)
    if args.make:
        receipt['round'] = args.round
        receipt['test_makes'] = make_and_pack(uri, args.round)
    text = json.dumps(receipt, indent=1, default=str, ensure_ascii=False)
    for forbidden in (uri, os.environ.get('API_KEY', '\0'), os.environ.get('DASHBOARD_API_KEY', '\0')):
        if forbidden and forbidden in text:
            raise RuntimeError('refusing to write: secret material in output')
    if args.make:
        out = OUT if args.round == 1 else OUT.with_name(f'{OUT.stem}-r{args.round}.json')
        out.write_text(text + '\n')
        print(f'OK: receipt at {out.relative_to(ROOT)}')
    print(text)


if __name__ == '__main__':
    main_entry()
