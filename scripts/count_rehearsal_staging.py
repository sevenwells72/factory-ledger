#!/usr/bin/env python3
"""Cutover count dress rehearsal on guarded STAGING (branch fix/count-tool-readiness).

Runs THIS branch's API in-process on 127.0.0.1 against the staging database — no
hosted deploy, no lifespan/startup migrations (the check_*_staging.py pattern) — and
seeds a synthetic three-area count: one extra-kosher line (real product 285, new lot),
one lot with an open SHORTAGE, one correction over 500 lb and one movement inside a
count window. It then drives the real tools per area exactly as on cutover morning:
reset_preview --v3 --area → owner review (scripted stand-in) → signed approval →
apply dry run → apply (typed phrase through a pty) → verify, and records how long
each area's posting pause takes. Never reads or writes production; never prints keys.
Synthetic owner approval demonstrates the flow; it is not Michael's attestation.

--hosted (P1.21 follow-up, 2026-10-14): ONE area against the DEPLOYED FastAPI-staging
(no local server): an ordinary lot set plus a shortage lot, a >500 lb hold approved
under the signed list and a new extra-kosher lot, with per-lot executor timings.

--hosted-groups (O-3, 2026-10-14): three area groups on the DEPLOYED FastAPI-staging, each
signed and journaled separately (one multi-area extra-kosher lot, one no-change group),
then `verify_reset.py --whole-reset` proves the union; two negative proofs follow
(a group left out; an unsigned owner physical_count post).
"""
import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timedelta
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from uuid import uuid4
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool as pgpool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staging_safety import assert_staging_database, PRODUCTION_DATABASE_HOST  # noqa: E402
from scripts.seed_staging import secret_file  # noqa: E402

TOOLS = ROOT / 'audits/fresh-start'
STAGING_HOST = 'aws-0-us-east-1.pooler.supabase.com'
STAGING_REF = 'jygmyvxnxdjiiilhxseq'
OWNER_KEY_FILE = Path.home() / '.config/factory-ledger/staging_owner_key'
NY = ZoneInfo('America/New_York')
# Extra-kosher lb products (counted in lb); the rehearsal loads a NEW lot on the first
# that has no lots yet. Each staging run leaves its synthetic lot behind (285: the
# interrupted 2026-10-13 run; 288: the first complete run).
EXTRA_CANDIDATES = (288, 283, 285)
COUNT_COLUMNS = ['sheet_id', 'area', 'count_date', 'counter', 'start_time', 'end_time', 'row_id', 'product_id', 'SKU', 'name',
                 'row_kind', 'lot_code', 'quantity', 'unit', 'partial_cases', 'location', 'bin_id', 'counted_time', 'estimated',
                 'tag_id', 'physical_marker', 'duplicate', 'notes']
MOVE_COLUMNS = ['move_id', 'product_id', 'SKU', 'lot_code', 'quantity', 'unit', 'from_area', 'from_location', 'to_area',
                'to_location', 'moved_at', 'tag_id', 'fl_transaction_id', 'notes']


def now():
    return datetime.now(NY).replace(microsecond=0)


def guarded_uri():
    uri = secret_file(Path.home() / 'Documents/fl-secrets/staging-db-url.txt')
    os.environ['STAGING_DATABASE_HOST'] = STAGING_HOST
    os.environ['STAGING_DATABASE_PROJECT_REF'] = STAGING_REF
    assert_staging_database(uri, 'staging', PRODUCTION_DATABASE_HOST)
    parts = urlsplit(uri)
    if parts.hostname != STAGING_HOST or unquote(parts.username or '') != 'postgres.' + STAGING_REF:
        raise RuntimeError('Not the documented staging database')
    return uri


@contextmanager
def connection(uri):
    conn = psycopg2.connect(uri, port=5432, connect_timeout=10, sslmode='require')
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute('SET LOCAL search_path=public')
                cur.execute("SET LOCAL lock_timeout='5s'")
                cur.execute("SET LOCAL statement_timeout='60s'")
                yield cur
    finally:
        conn.close()


def start_server(uri):
    """This branch's app on 127.0.0.1 with a staging pool; startup() deliberately not run."""
    os.environ.update(DATABASE_URL=uri, ENVIRONMENT='staging', PRODUCTION_DATABASE_HOST=PRODUCTION_DATABASE_HOST,
                      API_KEY=secrets.token_urlsafe(40), DASHBOARD_API_KEY=secrets.token_urlsafe(40))
    logging.disable(logging.CRITICAL)
    import main
    import uvicorn
    main.db_pool = pgpool.ThreadedConnectionPool(minconn=1, maxconn=8, dsn=uri, sslmode='require', connect_timeout=10)
    main._reset_actor_cache()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(main.app, host='127.0.0.1', port=port, lifespan='off', log_level='critical'))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            return f'http://127.0.0.1:{port}', server
        time.sleep(0.1)
    raise RuntimeError('Local rehearsal server did not start')


def wait_for_keys(base, keys, limit_s=180):
    """Hosted FL caches actor keys for up to ACTOR_CACHE_TTL_S (60 s): wait until each new key answers."""
    started = time.monotonic()
    for key in keys:
        while True:
            status, body = call(base, key, '/auth/whoami')
            if status == 200 and body.get('key_kind') == 'actor':
                break
            if time.monotonic() - started > limit_s:
                raise RuntimeError(f'hosted staging did not accept the rehearsal key: {status}')
            time.sleep(5)
    return round(time.monotonic() - started, 1)


def call(base, key, path, body=None):
    request = Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                      headers={'X-API-Key': key, 'Content-Type': 'application/json'},
                      method='POST' if body is not None else 'GET')
    try:
        with urlopen(request, timeout=60) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def seed(uri, ref, t0):
    """Synthetic products/lots/stock (ids ≥ 1e9) + a rehearsal owner and floor actor."""
    owner_key, floor_key = secrets.token_urlsafe(40), secrets.token_urlsafe(40)
    OWNER_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(OWNER_KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(owner_key)
    ids, items = {}, {}
    stocked = t0 - timedelta(hours=2)
    with connection(uri) as cur:
        cur.execute('INSERT INTO migration_markers(name) VALUES (%s)', ('staging_count_rehearsal_' + ref,))
        for role, key, label in (('owner', owner_key, 'Michael'), ('floor', floor_key, 'Arturo')):
            cur.execute('INSERT INTO actors(name,role,key_hash,active) VALUES (%s,%s,%s,true) RETURNING id,name',
                        (f'{label} (STAGING {ref})', role, sha256(key.encode()).hexdigest()))
            ids[role] = dict(cur.fetchone())
        specs = [('oats', 'ingredient', 'Oats', 'lb', None, 400), ('sugar', 'ingredient', 'Sugar', 'lb', None, 1200),
                 ('coconut', 'ingredient', 'Coconut', 'lb', None, 10), ('batch', 'batch', 'Batch granola', 'lb', None, None),
                 ('granola', 'finished', 'Granola 25 LB', '25 lb case', 25, 1000),
                 # uom exactly 'case' (like the five real ones): the 500 lb rule is in real pounds since PR #109.
                 ('cases15', 'finished', 'Granola 15 LB case', 'case', 15, 300)] + \
                [(f'misc{n}', 'ingredient', f'Ingredient {n}', 'lb', None, 100) for n in range(1, 7)]
        for label, kind, name, uom, case, stock in specs:
            cur.execute('''INSERT INTO products(name,odoo_code,type,uom,case_size_lb,default_batch_lb,active)
                           VALUES (%s,%s,%s,%s,%s,%s,true) RETURNING id,name,odoo_code''',
                        (f'STG {ref} {name}', f'{ref}-{label.upper()}', kind, uom, case, 30 if kind == 'batch' else None))
            product = dict(cur.fetchone())
            assert product['id'] >= 1_000_000_000
            items[label] = product
            if stock is None:
                continue
            code = f'{ref}-{label.upper()}-1'
            cur.execute("INSERT INTO lots(product_id,lot_code,entry_source) VALUES (%s,%s,'received') RETURNING id",
                        (product['id'], code))
            product.update(lot_id=cur.fetchone()['id'], lot_code=code, stock=stock)
            cur.execute("INSERT INTO transactions(type,notes,occurred_at) VALUES ('receive',%s,%s) RETURNING id",
                        (f'{ref} rehearsal opening stock', stocked))
            cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,%s)',
                        (cur.fetchone()['id'], product['id'], product['lot_id'], stock))
        cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES (%s,%s,30)',
                    (items['batch']['id'], items['coconut']['id']))
        cur.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=%s', (items['batch']['id'], items['granola']['id']))
        cur.execute('''SELECT p.id,p.name,p.odoo_code FROM products p WHERE p.id=ANY(%s) AND p.kosher_tier='extra_kosher'
                       AND NOT EXISTS (SELECT 1 FROM lots l WHERE l.product_id=p.id) ORDER BY array_position(%s,p.id) LIMIT 1''',
                    (list(EXTRA_CANDIDATES), list(EXTRA_CANDIDATES)))
        extra = cur.fetchone()
        if not extra:
            # Every real lb candidate already carries an earlier run's lot: use a synthetic
            # extra-kosher lb batch product instead (real products are never reused twice).
            cur.execute('''INSERT INTO products(name,odoo_code,type,uom,active,kosher_tier)
                           VALUES (%s,%s,'batch','lb',true,'extra_kosher') RETURNING id,name,odoo_code''',
                        (f'STG {ref} Extra-kosher batch', f'{ref}-EXTRA'))
            extra = cur.fetchone()
            assert extra['id'] >= 1_000_000_000
        items['extra'] = dict(extra)
    return ids, items, owner_key, floor_key


def committed(base, key, prepare_path, body):
    status, prepared = call(base, key, prepare_path, body)
    if status != 200 or not prepared.get('can_commit'):
        raise RuntimeError(f'{prepare_path} refused: {status} {json.dumps(prepared)[:400]}')
    acks = [w['code'] for w in prepared['warnings'] if w.get('requires_ack')]
    status, result = call(base, key, '/tickets/' + prepared['ticket'] + '/commit',
                          {'payload_hash': prepared['payload_hash'], 'acknowledged_warnings': acks})
    if status != 200:
        raise RuntimeError(f'commit refused: {status} {json.dumps(result)[:400]}')
    return result


def write_csv(path, columns, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def tool(args, env, stdin=None, pty=False):
    command = [sys.executable, '-B', *args]
    started = time.monotonic()
    if not pty:
        result = subprocess.run(command, text=True, capture_output=True, cwd=ROOT, env=env, timeout=900)
        return result, round(time.monotonic() - started, 1)
    # --apply needs an interactive terminal: stdin is a real pty and the phrase is
    # typed only after the executor has printed it, as a person would.
    master, slave = os.openpty()
    proc = subprocess.Popen(command, stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT, env=env)
    os.close(slave)
    seen = []
    for line in proc.stdout:
        seen.append(line)
        if 'Type exactly:' in line:
            os.write(master, stdin.encode())
            break
    seen.append(proc.stdout.read())
    proc.wait(timeout=900)
    os.close(master)
    return subprocess.CompletedProcess(command, proc.returncode, ''.join(seen), ''), round(time.monotonic() - started, 1)


def area_run(area, rows, products, workdir, env, owner, base_reason, coverage=None, apply=True):
    """One cutover area: pause → preview → owner review → sign → dry run → apply → verify.
    `coverage` = (product, completed_at) pairs when a group's files are cumulative;
    apply=False is the documented no-change group: signed, nothing to post, no journal."""
    folder = workdir / area['slug']
    inputs, out = folder / 'inputs', folder / 'out'
    inputs.mkdir(parents=True)
    write_csv(inputs / 'count.csv', COUNT_COLUMNS, rows)
    done = max([area['end']] + [a['end'] for a in area.get('with', [])])
    write_csv(inputs / 'coverage.csv', ['product_id', 'SKU', 'name', 'all_locations_searched', 'completed_at', 'owner_initials', 'notes'],
              [dict(product_id=p['id'], SKU=p['odoo_code'], name=p['name'], all_locations_searched='yes',
                    completed_at=at.isoformat(), owner_initials='MG', notes='rehearsal')
               for p, at in (coverage or [(p, done) for p in products])])
    write_csv(inputs / 'moves.csv', MOVE_COLUMNS, [])
    timing, pause = {}, time.monotonic()
    names = [area['name']] + [a['name'] for a in area.get('with', [])]
    areas = [x for name in names for x in ('--area', name)]
    base = ['audits/fresh-start/reset_preview.py', '--v3', str(inputs / 'count.csv'), '--coverage', str(inputs / 'coverage.csv'),
            '--moves', str(inputs / 'moves.csv'), '--target', 'staging'] + areas
    result, timing['preview_1_s'] = tool(base + ['--output-dir', str(out / 'preview-1')], env)
    if result.returncode:
        raise RuntimeError('preview failed: ' + result.stderr[-800:])
    template = json.loads((out / 'preview-1/owner-review-template.json').read_text())
    first = json.loads((out / 'preview-1/reset-preview-v3.json').read_text())
    held_first = [dict(product_id=r['product_id'], lot_code=r['lot_code'], new_lot=r.get('new_lot'), holds=r['holds'])
                  for r in first['rows'] if r['status'] != 'READY']
    review = {k: {} for k in template}
    for fp in template['multi_area_reviews']:
        # Scripted stand-in: the owner confirms no unlogged moves between the two areas' rows.
        review['multi_area_reviews'][fp] = dict(owner='MG (rehearsal)', no_unlogged_moves=True)
    for fp in template['coverage_reviews']:
        # Scripted stand-in: the owner confirms a searched product with no count rows really has none.
        review['coverage_reviews'][fp] = dict(owner='MG (rehearsal)', confirmed=True)
    started = time.monotonic()
    for fp, item in template['movement_reviews'].items():
        # Scripted stand-in for the owner's answer: the floor's damage adjust happened after the count.
        when = 'after count' if item['type'] == 'adjust' and float(item['quantity_native']) < 0 else 'before count'
        review['movement_reviews'][fp] = dict(owner='MG (rehearsal)', classifications={k: when for k in item['classifications']},
                                              late_confirmed=True, allocations={})
    timing['owner_review_scripted_s'] = round(time.monotonic() - started, 1)
    review_path = None
    final = out / 'preview-1'
    if any(template[k] for k in template):
        review_path = inputs / 'owner-review.json'
        review_path.write_text(json.dumps(review, indent=2))
        result, timing['preview_2_s'] = tool(base + ['--review', str(review_path), '--output-dir', str(out / 'preview-2')], env)
        if result.returncode:
            raise RuntimeError('preview 2 failed: ' + result.stderr[-800:])
        final = out / 'preview-2'
    preview = json.loads((final / 'reset-preview-v3.json').read_text())
    mine = {p['id'] for p in products}
    rows_mine = [r for r in preview['rows'] if r['product_id'] in mine]
    held = [r for r in rows_mine if r['status'] != 'READY']
    if held:
        raise RuntimeError('area still held: ' + json.dumps([(r['product_id'], r['holds']) for r in held])[:800])
    # Zero-stock products (no counted or on-hand lot) are signed for too (PR #120 fix round 1).
    zero = sorted(mine - {r['product_id'] for r in rows_mine})
    if any(pid not in preview['coverage_complete'] or str(pid) in preview['holds'] for pid in zero):
        raise RuntimeError('zero-stock product not approvable: ' + json.dumps({pid: preview['holds'].get(str(pid)) for pid in zero})[:800])
    approved = sorted({r['product_id'] for r in rows_mine} | set(zero))
    kosher = [f'{r["product_id"]}:{r["lot_code"]}' for r in rows_mine if r.get('new_lot') and r['adjustment_lb']]
    needs = any(r['owner_approvals'] and r['adjustment_lb'] for r in rows_mine)
    signed = folder / 'approved'
    signed.mkdir()
    (signed / 'reset-preview-v3.json').write_bytes((final / 'reset-preview-v3.json').read_bytes())
    digest = sha256((signed / 'reset-preview-v3.json').read_bytes()).hexdigest()
    approval = signed / 'approval.md'
    approval.write_text('\n'.join(['Preview file: reset-preview-v3.json', f'SHA-256: {digest}',
                                   'Approved product IDs: ' + ', '.join(map(str, approved)), f'Owner name: {owner}',
                                   f'Named actor: {owner}', f'Date: {now().date()}', f'Signature: {owner}',
                                   'Opening entry classification: N', f'Owner approves held entries: {"Y" if needs else "N"}',
                                   'Extra-kosher lots: ' + (', '.join(kosher) or 'none')]) + '\n')
    paths = dict(inputs=inputs, preview=signed / 'reset-preview-v3.json', approval=approval, review=review_path)
    if not apply:
        changes = [r for r in rows_mine if r['adjustment_lb'] and float(r['adjustment_lb'])]
        if changes:
            raise RuntimeError('no-change group has changes: ' + json.dumps([(r['product_id'], r['adjustment_lb']) for r in changes], default=str))
        # Runbook step 18 no-change branch: the same verify, without the proof triplet.
        verify = ['audits/fresh-start/verify_reset.py', '--v3', str(inputs / 'count.csv'), '--coverage', str(inputs / 'coverage.csv'),
                  '--moves', str(inputs / 'moves.csv'), '--output-dir', str(out / 'verification'), '--target', 'staging'] + areas + \
                 (['--review', str(review_path)] if review_path else [])
        result, timing['verify_s'] = tool(verify, env)
        verified = json.loads((out / 'verification/reset-verification-v3.json').read_text())
        return dict(area=', '.join(names), approved_products=approved, zero_stock_products=zero, preview_sha256=digest, no_change=True, timings_s=timing,
                    verify_exit=result.returncode, paths=paths,
                    verified_rows=[dict(product_id=r['product_id'], lot_code=r['lot_code'], status=r['status'], adjustment=r['adjustment_lb'])
                                   for r in verified['rows'] if r['product_id'] in mine])
    apply_args = ['audits/fresh-start/apply_reset.py', str(signed / 'reset-preview-v3.json'), '--approval', str(approval),
                  '--count-csv', str(inputs / 'count.csv'), '--target', 'staging']
    result, timing['dry_run_s'] = tool(apply_args, env)
    if result.returncode:
        raise RuntimeError('dry run failed: ' + (result.stderr or result.stdout)[-800:])
    cutoff = max(datetime.fromisoformat(r['cutoff']) for r in preview['rows'] if r['product_id'] in approved)
    phrase = f'APPLY OPENING BALANCE {cutoff.astimezone(NY).date()} {digest}'
    result, timing['apply_s'] = tool(apply_args + ['--apply'], env, stdin=phrase + '\n', pty=True)
    apply_log = result.stdout
    if 'READ-BACK' not in apply_log:
        raise RuntimeError('apply failed: ' + apply_log[-1500:])
    state = TOOLS / 'apply-state-staging' / digest / 'journal.jsonl'
    verify = ['audits/fresh-start/verify_reset.py', '--v3', str(inputs / 'count.csv'), '--coverage', str(inputs / 'coverage.csv'),
              '--moves', str(inputs / 'moves.csv'), '--approved-preview', str(signed / 'reset-preview-v3.json'),
              '--approval', str(approval), '--journal', str(state), '--output-dir', str(out / 'verification'),
              '--target', 'staging'] + areas + (['--review', str(review_path)] if review_path else [])
    result, timing['verify_s'] = tool(verify, env)
    timing['pause_total_s'] = round(time.monotonic() - pause, 1)
    verified = json.loads((out / 'verification/reset-verification-v3.json').read_text())
    after = [r for r in verified['rows'] if r['product_id'] in mine]
    readback = list(csv.DictReader((TOOLS / 'apply-results-staging' / digest / 'readback.csv').open()))
    executor = json.loads((TOOLS / 'apply-results-staging' / digest / 'timings.json').read_text())
    paths['journal'] = state
    return dict(area=', '.join(names), window=[area['start'].isoformat(), area['end'].isoformat()], approved_products=approved, paths=paths,
                held_before_owner_review=held_first,
                preview_sha256=digest, owner_approves_held=needs, extra_kosher_lots=kosher, timings_s=timing,
                executor_timings_s=executor, readback=readback,
                verify_exit=result.returncode, verified_rows=[dict(product_id=r['product_id'], lot_code=r['lot_code'],
                                                                   status=r['status'], adjustment=r['adjustment_lb'],
                                                                   current=r['current_lb'], expected=r['expected_current_lb'])
                                                              for r in after],
                apply_log_tail=[line for line in apply_log.splitlines() if line.strip().startswith(('ADJ-', 'FND-', 'Read back', 'Executor'))])


def sheet_row(area, sheet, n, product, lot_code, quantity, unit, at, estimated=''):
    return dict({k: '' for k in COUNT_COLUMNS}, sheet_id=sheet, area=area['name'], count_date=str(at.date()), counter='Arturo (rehearsal)',
                start_time=area['start'].isoformat(), end_time='', row_id=f'{sheet}-{n}', product_id=product['id'],
                SKU=product['odoo_code'], name=product['name'], row_kind='COUNT', lot_code=lot_code, quantity=quantity,
                unit=unit, location=area['name'] + ' rack ' + str(n), counted_time=at.isoformat(), estimated=estimated)


def hosted_area(base, floor_key, items, ref, t0, workdir, env, owner, report):
    """One area on hosted staging: shortage lot, >500 lb hold, new extra-kosher lot, ordinary lots."""
    coconut = items['coconut']
    made = committed(base, floor_key, '/make/prepare', dict(
        product_id=items['batch']['id'], batches=1, confirmed_sku=True,
        occurred_at=(t0 - timedelta(minutes=90)).isoformat(),
        lot_confirmations=[dict(lot_id=coconut['lot_id'], method='full_code', value=coconut['lot_code'])]))
    report['shortage'] = dict(make_receipt=made['receipt_number'], exceptions=[s['exception_id'] for s in made.get('shortages') or []])
    if not report['shortage']['exceptions']:
        raise RuntimeError('fixture make did not open a shortage')
    area = dict(name='Dry storage', slug='hosted-area-dry-storage', start=now())
    time.sleep(10)
    at = now()
    misc = [items[f'misc{n}'] for n in range(1, 7)]
    cases15 = items['cases15']
    ek_code = f'{ref}-EK-1'.lower()   # as written on the sheet; FL stores it upper-case
    rows = [sheet_row(area, 'RH1', 1, items['oats'], items['oats']['lot_code'], '395', 'lb', at, 'x'),
            # 520 lb counted against 1,200 on the book: −680 lb → LARGE_CORRECTION hold, approved under the signed list.
            sheet_row(area, 'RH1', 2, items['sugar'], items['sugar']['lot_code'], '520', 'lb', at),
            # Book −20 after the short make; counted 25 lb → +45 closes the open SHORTAGE as counted.
            sheet_row(area, 'RH1', 3, coconut, coconut['lot_code'], '25', 'lb', at),
            # Extra-kosher stock on a lot FL has never seen → NEW lot via an owner-attested found ticket.
            sheet_row(area, 'RH1', 4, items['extra'], ek_code, '180', 'lb', at),
            sheet_row(area, 'RH1', 5, cases15, cases15['lot_code'], '17', 'cases', at)] + \
           [sheet_row(area, 'RH1', 5 + n, m, m['lot_code'], str(100 - n), 'lb', at) for n, m in enumerate(misc, 1)]
    time.sleep(10)
    area['end'] = now()
    for r in rows:
        r['end_time'] = area['end'].isoformat()
    time.sleep(2)
    return [area_run(area, rows, [items['oats'], items['sugar'], coconut, items['extra'], cases15] + misc, workdir, env, owner, ref)]


def whole_reset(workdir, env, manifest, name):
    """verify_reset.py --whole-reset on one manifest against one saved read-only snapshot (kept,
    so the evidence can be replayed); returns exit code, verdict and gap kinds."""
    import fresh_start_v3 as v
    path = workdir / f'manifest-{name}.json'
    path.write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    snap = workdir / 'snapshots' / f'{name}.json'
    snap.parent.mkdir(exist_ok=True)
    snap.write_text(json.dumps(v.snapshot('staging'), default=str))
    out = workdir / 'whole-reset' / name
    result, seconds = tool(['audits/fresh-start/verify_reset.py', '--whole-reset', str(path), '--output-dir', str(out),
                            '--snapshot', str(snap)], env)
    proof = json.loads((out / 'whole-reset-proof.json').read_text()) if (out / 'whole-reset-proof.json').exists() else {}
    return dict(manifest=str(path.relative_to(TOOLS)), snapshot=str(snap.relative_to(TOOLS)), exit=result.returncode, seconds=seconds,
                verdict=proof.get('verdict'),
                gaps=[dict((k, g[k]) for k in ('kind', 'group', 'product_id', 'lot_code', 'detail')) for g in proof.get('gaps', [])],
                groups=[dict((k, g[k]) for k in ('name', 'areas', 'approved_products', 'posts', 'no_change')) for g in proof.get('groups', [])],
                multi_area_lots=proof.get('multi_area_lots'), stderr_tail=result.stderr[-400:] if result.returncode not in (0, 2) else '')


def forged_journal(source, target):
    """Negative case: the executor's journal plus one forged `refused` receipt naming a bogus
    receipt/transaction for the posted ticket, every hash re-chained so the chain still verifies."""
    import fresh_start_v3 as v
    events = [json.loads(x) for x in source.read_text().splitlines()]
    intent = next(e for e in events if e['kind'] == 'intent' and e['phase'] == 'commit')
    events += [dict(intent), dict(kind='receipt', intent=len(events), outcome='refused', transaction_id=999999999,
                                  receipt_number='ADJ-999999-999', error_code='TICKET_STALE', at=intent['at'])]
    previous, lines = '', []
    for seq, e in enumerate(events):
        e.pop('sha256', None)
        e.update(seq=seq, previous=previous)
        e['sha256'] = previous = v.sha(v.canonical(e).encode())
        lines.append(json.dumps(e))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(lines) + '\n')


def hosted_groups(base, owner_key, items, ref, workdir, env, owner, report, floor_key=None):
    """O-3 (2026-10-14): three area groups posted one after another on hosted staging, each
    signed and journaled on its own, then proved as a union with verify_reset --whole-reset.
    Files are cumulative like the real morning: each group's count/coverage/review copy holds
    every area transcribed so far and is never edited after its signature."""
    rows, coverage, results = [], [], []
    ek_code = f'{ref}-EK-1'.lower()

    def counted(area, specs):
        area['start'] = now()
        time.sleep(5)
        at = now()
        sheet = 'RG' + str(len(results) + 1) + area['name'][:1]
        new = [sheet_row(area, sheet, n, product, code, qty, unit, at, est) for n, (product, code, qty, unit, est) in enumerate(specs, 1)]
        time.sleep(5)
        area['end'] = now()
        for r in new:
            r['end_time'] = area['end'].isoformat()
        rows.extend(new)
        time.sleep(2)
        return area

    g1 = counted(dict(name='Dry storage', slug='group-1-dry-storage'),
                 [(items['oats'], items['oats']['lot_code'], '395', 'lb', 'x'),
                  # 520 lb against 1,200 on the book: −680 lb, held for the owner and released under the signed list.
                  (items['sugar'], items['sugar']['lot_code'], '520', 'lb', '')])
    coverage += [(items['oats'], g1['end']), (items['sugar'], g1['end'])]
    results.append(area_run(g1, list(rows), [items['oats'], items['sugar']], workdir, env, owner, ref, coverage=list(coverage)))

    # Group 2 = Warehouse + Cooler: the new extra-kosher lot is counted in BOTH areas (multi-area,
    # reset once by this group, held in the others) and the granola case lot only in the warehouse.
    granola = items['granola']
    g2 = counted(dict(name='Warehouse', slug='group-2-warehouse-cooler'),
                 [(granola, granola['lot_code'], '39', 'cases', ''), (items['extra'], ek_code, '180', 'lb', '')])
    cooler = counted(dict(name='Cooler', slug='group-2b-cooler'), [(items['extra'], ek_code, '120', 'lb', '')])
    g2['with'] = [cooler]
    coverage += [(granola, cooler['end']), (items['extra'], cooler['end'])]
    results.append(area_run(g2, list(rows), [granola, items['extra']], workdir, env, owner, ref, coverage=list(coverage)))

    # Group 3 = Office: counted exactly at book, so nothing posts and no journal exists. It also
    # signs for the batch product: searched everywhere, no lot, no stock (zero-stock approval).
    misc, empty = items['misc1'], items['batch']
    g3 = counted(dict(name='Office', slug='group-3-office-no-change'), [(misc, misc['lot_code'], '100', 'lb', '')])
    coverage += [(misc, g3['end']), (empty, g3['end'])]
    results.append(area_run(g3, list(rows), [misc, empty], workdir, env, owner, ref, coverage=list(coverage), apply=False))

    def rel(path):
        return os.path.relpath(path, workdir)

    scope = sorted({p['id'] for p, _ in coverage})
    final = results[-1]['paths']
    groups = []
    for r in results:
        group = dict(name=r['area'], areas=[x.strip() for x in r['area'].split(',')], approved_preview=rel(r['paths']['preview']),
                     approval=rel(r['paths']['approval']))
        if r.get('no_change'):
            group['no_change'] = True
        else:
            group['journal'] = rel(r['paths']['journal'])
        groups.append(group)
    manifest = dict(version=1, target='staging', expected_reset_products=len(scope), rehearsal_scope_product_ids=scope, groups=groups,
                    notes=f'{ref} hosted-staging O-3 rehearsal; final inputs = the last group\'s signed copies',
                    final_inputs=dict(count=rel(final['inputs'] / 'count.csv'), coverage=rel(final['inputs'] / 'coverage.csv'),
                                      moves=rel(final['inputs'] / 'moves.csv'), review=rel(final['review'])))
    proofs = {'complete': whole_reset(workdir, env, manifest, 'complete')}
    # Negative 1: leave the no-change group out — its products must be reported as missing.
    proofs['without_office_group'] = whole_reset(workdir, env, dict(manifest, groups=groups[:2]), 'without-office')
    # Negative 2 (fix round 1): the Office approval signed WITHOUT the zero-stock product.
    office = results[-1]['paths']['approval']
    narrow = office.with_name('approval-without-zero-stock.md')
    narrow.write_text(office.read_text().replace('Approved product IDs: ' + ', '.join(map(str, results[-1]['approved_products'])),
                                                 'Approved product IDs: ' + str(misc['id'])))
    proofs['zero_stock_unapproved'] = whole_reset(workdir, env, dict(manifest, groups=groups[:2] + [dict(groups[2], approval=rel(narrow))]),
                                                  'zero-stock-unapproved')
    # Negative 3 (fix round 1): a forged `refused` receipt with a bogus receipt id, chain re-hashed.
    forged = workdir / 'forged' / 'group-1-journal-forged-receipt.jsonl'
    forged_journal(results[0]['paths']['journal'], forged)
    proofs['forged_receipt'] = whole_reset(workdir, env, dict(manifest, groups=[dict(groups[0], journal=rel(forged))] + groups[1:]),
                                           'forged-receipt')
    # Negative 4 (fix round 1): a duplicate-ticked oats row added to the final count after the
    # Dry storage group was signed, excluded by an owner duplicate answer, so balances still match.
    import fresh_start_v3 as v
    dup_dir = workdir / 'final-with-duplicate'
    dup_dir.mkdir()
    oats_row = next(r for r in rows if r['product_id'] == items['oats']['id'])
    dup_rows = list(rows) + [dict(oats_row, row_id=oats_row['row_id'] + '-dup', duplicate='x', location=oats_row['location'] + ' (again)')]
    write_csv(dup_dir / 'count.csv', COUNT_COLUMNS, dup_rows)
    data = v.load_inputs(dup_dir / 'count.csv', final['inputs'] / 'coverage.csv', final['inputs'] / 'moves.csv')
    fp = v.fingerprint([r for r in data['count'] if r['row_id'] in (oats_row['row_id'], oats_row['row_id'] + '-dup')])
    dup_review = json.loads(final['review'].read_text())
    dup_review.setdefault('duplicate_reviews', {})[fp] = dict(owner='MG (rehearsal)', action='exclude', exclude_rows=[oats_row['row_id'] + '-dup'],
                                                             move_id='', notes='same pallet written twice (rehearsal negative case)')
    (dup_dir / 'review.json').write_text(json.dumps(dup_review, indent=2))
    proofs['duplicate_observation_added'] = whole_reset(workdir, env, dict(manifest, final_inputs=dict(
        manifest['final_inputs'], count=rel(dup_dir / 'count.csv'), review=rel(dup_dir / 'review.json'))), 'duplicate-observation-added')
    # Negative 5: an owner physical_count post outside every signed plan, through the hosted API.
    oats = items['oats']
    extra = committed(base, owner_key, '/adjust/prepare', dict(lot_id=oats['lot_id'], delta_lb=1, reason_code='physical_count',
                                                               note=f'{ref} unsigned recount (rehearsal negative case)',
                                                               occurred_at=now().isoformat()))
    report['unsigned_extra_post'] = dict(receipt=extra['receipt_number'], transaction_id=extra['transaction_id'], lb=1)
    proofs['after_unsigned_post'] = whole_reset(workdir, env, manifest, 'after-unsigned-post')
    # Negative 6 (fix round 1): the FLOOR actor (not a group's named actor) posts a count adjust.
    granola = items['granola']
    floor = committed(base, floor_key, '/adjust/prepare', dict(lot_id=granola['lot_id'], delta_lb=2, reason_code='physical_count',
                                                               note=f'{ref} floor recount (rehearsal negative case)',
                                                               occurred_at=now().isoformat()))
    report['floor_extra_post'] = dict(receipt=floor['receipt_number'], transaction_id=floor['transaction_id'], lb=2)
    proofs['after_floor_post'] = whole_reset(workdir, env, manifest, 'after-floor-post')
    # Negative 7 (fix round 1): a floor count adjust over 500 lb, held for the owner and left open.
    status, prepared = call(base, floor_key, '/adjust/prepare', dict(lot_id=items['sugar']['lot_id'], delta_lb=-505, reason_code='physical_count',
                                                                     note=f'{ref} floor recount, held (rehearsal negative case)',
                                                                     occurred_at=now().isoformat()))
    if status != 200 or not prepared.get('can_commit'):
        raise RuntimeError(f'held-case prepare refused: {status} {json.dumps(prepared)[:400]}')
    status, held = call(base, floor_key, '/tickets/' + prepared['ticket'] + '/commit',
                        {'payload_hash': prepared['payload_hash'], 'acknowledged_warnings': [w['code'] for w in prepared['warnings'] if w.get('requires_ack')]})
    if status != 202 or not held.get('held'):
        raise RuntimeError(f'held-case commit was not held: {status} {json.dumps(held)[:400]}')
    report['floor_open_hold'] = dict(exception_id=held.get('exception_id'), ticket_id=prepared['ticket_id'], lb=-505)
    proofs['open_owner_hold'] = whole_reset(workdir, env, manifest, 'open-owner-hold')
    # The owner declines it: the hold is resolved, nothing posts, and OPEN_HOLD clears.
    status, declined = call(base, owner_key, f'/exceptions/{held["exception_id"]}/reject',
                            dict(resolution_kind='declined', note=f'{ref} rehearsal: declined the held floor recount'))
    report['floor_open_hold'].update(declined_status=status)
    proofs['after_hold_declined'] = whole_reset(workdir, env, manifest, 'after-hold-declined')
    report['whole_reset'] = proofs
    report['whole_reset_manifest'] = manifest
    return results


EVIDENCE = ROOT / 'docs/deployments/whole-reset-proof-evidence'
TICKET_TOKEN = r'wt_[A-Za-z0-9_-]{20,100}'


def scoped_snapshot(s, scope, ticket_ids, actor_ids):
    """The saved staging snapshot cut to the rehearsal products, their lots/lines/tickets/holds and
    the rehearsal actors. Other staging data (catalog, orders, schema) never leaves the machine."""
    keep = set(scope)
    lines = [x for x in s['lines'] if x['product_id'] in keep]
    txs = {str(x['transaction_id']) for x in lines}
    held = [e for e in s.get('held_exceptions', []) if e.get('product_id') in keep]
    tickets = set(ticket_ids) | {x.get('ticket_id') for x in lines} | {e.get('ticket_id') for e in held}
    line_ids = {x['line_id'] for x in lines}
    out = dict(snapshot_at=s['snapshot_at'], read_only=s['read_only'], target=s.get('target'),
               products=[p for p in s['products'] if p['id'] in keep], catalog=[p for p in s['catalog'] if p['id'] in keep],
               lots=[l for l in s['lots'] if l['product_id'] in keep], lines=lines,
               transaction_line_counts={k: n for k, n in s['transaction_line_counts'].items() if k in txs},
               shortages=[e for e in s.get('shortages', []) if e.get('product_id') in keep], held_exceptions=held,
               correction_tickets=[t for t in s.get('correction_tickets', []) if t['id'] in tickets],
               actors=[a for a in s['actors'] if a['id'] in actor_ids],
               corrections=[c for c in s.get('corrections', []) if (c['target_table'] == 'transactions' and str(c['target_id']) in txs)
                            or (c['target_table'] == 'transaction_lines' and c['target_id'] in line_ids)])
    for key in ('schema_columns', 'supply_requests', 'orders', 'order_lines', 'allocations', 'ship_transactions'):
        out[key] = []
    return out


def export_evidence(workdir, ref, report, env, secrets_seen):
    """Sanitized copy of every signed artifact, journal, manifest, saved snapshot and proof output
    under docs/deployments/whole-reset-proof-evidence/<ref>/ for independent review. Journals keep
    every field except the ticket token (a bearer reference); SHA256SUMS lists the originals.
    Each proof is then replayed from the cut snapshot and must give the same verdict and gaps."""
    import re
    dest = EVIDENCE / ref
    dest.mkdir(parents=True)
    sums, files = [], []
    proofs, manifest = report['whole_reset'], report['whole_reset_manifest']
    scope = manifest['rehearsal_scope_product_ids']
    actor_ids = {report['owner_actor']['id'], report['floor_actor']['id']}

    def put(rel_path, data):
        target = dest / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files.append(target)

    journals = {}
    for case in proofs.values():
        m = json.loads((TOOLS / case['manifest']).read_text())
        for g in m['groups']:
            for key in ('approved_preview', 'approval', 'journal'):
                if g.get(key):
                    journals.setdefault(os.path.normpath(workdir / g[key]), key)
        for path in m['final_inputs'].values():
            journals.setdefault(os.path.normpath(workdir / path), 'input')
    for folder in sorted(p for p in workdir.iterdir() if p.is_dir() and p.name.startswith('group-')):
        for path in sorted(folder.glob('inputs/*')) + sorted(folder.glob('approved/*')):
            journals.setdefault(os.path.normpath(path), 'signed')
    ticket_ids = set()
    for original, kind in sorted(journals.items()):
        original = Path(original)
        raw = original.read_bytes()
        sums.append(f'{sha256(raw).hexdigest()}  {os.path.relpath(original, workdir)}')
        name = os.path.relpath(original, workdir)
        if name.startswith('..'):   # executor journals live in audits/fresh-start/apply-state-staging/
            name = os.path.relpath(original, TOOLS)
        if original.suffix == '.jsonl':
            events = [json.loads(x) for x in raw.decode().splitlines()]
            for e in events:
                if e.get('ticket_id'):
                    ticket_ids.add(e['ticket_id'])
                if 'ticket' in e:
                    e['ticket'] = 'REDACTED-sha256:' + sha256(str(e['ticket']).encode()).hexdigest()[:16]
            put(name.replace('.jsonl', '.redacted.jsonl'), ('\n'.join(json.dumps(e) for e in events) + '\n').encode())
        else:
            put(name, raw)
    replay = {}
    for key, case in proofs.items():
        name = Path(case['manifest']).stem.replace('manifest-', '')
        put(f'manifest-{name}.json', (TOOLS / case['manifest']).read_bytes())
        cut = workdir / 'snapshots' / f'{name}.scoped.json'
        cut.write_text(json.dumps(scoped_snapshot(json.loads((TOOLS / case['snapshot']).read_text()), scope, ticket_ids, actor_ids), default=str))
        put(f'snapshots/{name}.scoped.json', cut.read_bytes())
        out = workdir / 'whole-reset-replay' / name
        result, _ = tool(['audits/fresh-start/verify_reset.py', '--whole-reset', str(TOOLS / case['manifest']), '--output-dir', str(out),
                          '--snapshot', str(cut)], env)
        again = json.loads((out / 'whole-reset-proof.json').read_text())
        same = (result.returncode == case['exit'] and again['verdict'] == case['verdict'] and
                [(g['kind'], g['product_id'], g['lot_code']) for g in again['gaps']] == [(g['kind'], g['product_id'], g['lot_code']) for g in case['gaps']])
        replay[key] = dict(exit=result.returncode, same_verdict_and_gaps=same)
        for leaf in ('whole-reset-proof.md', 'whole-reset-proof.json', 'whole-reset-gaps.csv'):
            put(f'whole-reset/{name}/{leaf}', (workdir / 'whole-reset' / name / leaf).read_bytes())
    put('SHA256SUMS-originals.txt', ('\n'.join(sums) + '\n').encode())
    # Nothing secret may leave: no API key, ticket token, database URL or local key path.
    for path in files:
        text = path.read_text(errors='replace')
        bad = [k for k in secrets_seen if k and k in text] or re.findall(TICKET_TOKEN, text) or \
              [m for m in ('postgres://', 'postgresql://', 'X-API-Key', 'staging_owner_key') if m in text]
        if bad:
            raise RuntimeError(f'evidence file {path.name} would leak a secret; nothing committed')
    report['evidence'] = dict(folder=str(dest.relative_to(ROOT)), files=len(files), replay_from_scoped_snapshot=replay,
                              journals='ticket token redacted; every other field and each original event sha256 kept')
    if not all(r['same_verdict_and_gaps'] for r in replay.values()):
        raise RuntimeError('replay from the scoped snapshot differs: ' + json.dumps(replay))


def rehearse(output, hosted=False, groups=False):
    uri = guarded_uri()
    ref = 'STG-CNT-' + uuid4().hex[:6].upper()
    t0 = now()
    ids, items, owner_key, floor_key = seed(uri, ref, t0)
    server = None
    if hosted:
        sys.path.insert(0, str(TOOLS))
        from fresh_start_common import TARGETS
        base = TARGETS['staging']['api']
        env = {k: v for k, v in os.environ.items() if k != 'FL_STAGING_API_URL'}
    else:
        base, server = start_server(uri)
        env = dict(os.environ, FL_STAGING_API_URL=base)
    owner = ids['owner']['name']
    for name in ('API_KEY', 'DASHBOARD_API_KEY', 'DATABASE_URL'):
        env.pop(name, None)
    workdir = TOOLS / 'v3' / 'rehearsal' / ref
    report = dict(reference=ref, mode=('THREE area groups + whole-reset proof (O-3) against the DEPLOYED FastAPI-staging ' + base
                                       + ' (guarded staging DB); no local server') if groups else
                  ('ONE area against the DEPLOYED FastAPI-staging ' + base + ' (guarded staging DB); no local server')
                  if hosted else 'branch API in-process on 127.0.0.1 against guarded STAGING; no hosted deployment',
                  owner_actor=ids['owner'], floor_actor=ids['floor'], products={k: v['id'] for k, v in items.items()},
                  synthetic_owner_not_michael=True)
    try:
        if hosted:
            report['key_cache_wait_s'] = wait_for_keys(base, (owner_key, floor_key))
            report['areas'] = (hosted_groups(base, owner_key, items, ref, workdir, env, owner, report, floor_key) if groups else
                               hosted_area(base, floor_key, items, ref, t0, workdir, env, owner, report))
        else:
            report['areas'] = local_areas(base, floor_key, items, ref, t0, workdir, env, owner, report)
        evidence(uri, ids, items, report)
        if groups:
            export_evidence(workdir, ref, report, env, (owner_key, floor_key))
        return report
    finally:
        if server:
            server.should_exit = True
        with connection(uri) as cur:
            cur.execute('UPDATE actors SET active=false WHERE id=ANY(%s)', ([ids['owner']['id'], ids['floor']['id']],))
        report['actors_deactivated_after_rehearsal'] = True
        try:
            OWNER_KEY_FILE.unlink()
        except FileNotFoundError:
            pass
        # Repo-relative paths only: the receipt is committed.
        output.write_text(json.dumps(report, indent=2, default=str).replace(str(ROOT) + '/', '') + '\n')


def local_areas(base, floor_key, items, ref, t0, workdir, env, owner, report):
    """The three-area local rehearsal (2026-10-13)."""
    # A make 90 minutes ago used 30 lb of coconut with 10 lb on the book: it posts short and opens a SHORTAGE.
    coconut = items['coconut']
    made = committed(base, floor_key, '/make/prepare', dict(
        product_id=items['batch']['id'], batches=1, confirmed_sku=True,
        occurred_at=(t0 - timedelta(minutes=90)).isoformat(),
        lot_confirmations=[dict(lot_id=coconut['lot_id'], method='full_code', value=coconut['lot_code'])]))
    report['shortage'] = dict(make_receipt=made['receipt_number'], exceptions=[s['exception_id'] for s in made.get('shortages') or []])
    if not report['shortage']['exceptions']:
        raise RuntimeError('fixture make did not open a shortage')
    results = []

    def window(name, slug):
        return dict(name=name, slug=slug, start=now())

    def close_area(area, rows):
        time.sleep(5)
        area['end'] = now()
        for r in rows:
            r['end_time'] = area['end'].isoformat()
        time.sleep(2)

    # Area 1 — Dry storage: oats (estimated) and sugar (−680 lb → over-500 owner hold).
    a1 = window('Dry storage', 'area-1-dry-storage')
    time.sleep(10)
    at = now()
    misc = [items[f'misc{n}'] for n in range(1, 7)]
    rows = [sheet_row(a1, 'RH1', 1, items['oats'], items['oats']['lot_code'], '395', 'lb', at, 'x'),
            sheet_row(a1, 'RH1', 2, items['sugar'], items['sugar']['lot_code'], '520', 'lb', at)] + \
           [sheet_row(a1, 'RH1', 2 + n, m, m['lot_code'], str(100 - n), 'lb', at) for n, m in enumerate(misc, 1)]
    time.sleep(10)
    close_area(a1, rows)
    results.append(area_run(a1, rows, [items['oats'], items['sugar']] + misc, workdir, env, owner, ref))

    # Area 2 — Production floor: coconut found at 25 lb against −20 on the book (closes the shortage).
    a2 = window('Production floor', 'area-2-production-floor')
    time.sleep(10)
    rows = [sheet_row(a2, 'RH2', 1, coconut, coconut['lot_code'], '25', 'lb', now())]
    time.sleep(10)
    close_area(a2, rows)
    results.append(area_run(a2, rows, [coconut], workdir, env, owner, ref))

    # Area 3 — Warehouse: 39 cases of granola, a damage write-off INSIDE the window
    # after that row, and extra-kosher bulk 285 on a lot FL has never seen.
    a3 = window('Warehouse', 'area-3-warehouse')
    time.sleep(8)
    granola = items['granola']
    rows = [sheet_row(a3, 'RH3', 1, granola, granola['lot_code'], '39', 'cases', now())]
    time.sleep(8)
    damage = committed(base, floor_key, '/adjust/prepare', dict(lot_id=granola['lot_id'], delta_lb=-25, reason_code='damage_disposal',
                                                                occurred_at=now().isoformat()))
    report['movement_in_window'] = dict(receipt=damage['receipt_number'], transaction_id=damage['transaction_id'], lb=-25)
    time.sleep(8)
    ek_code = f'{ref}-EK-1'.lower()   # as written on the sheet; FL stores it upper-case
    rows.append(sheet_row(a3, 'RH3', 2, items['extra'], ek_code, '180', 'lb', now()))
    # 17 cases of 15 lb against 300 lb on the book: −45 lb (before PR #109 FL read it as 675 lb and held it).
    cases15 = items['cases15']
    rows.append(sheet_row(a3, 'RH3', 3, cases15, cases15['lot_code'], '17', 'cases', now()))
    time.sleep(8)
    close_area(a3, rows)
    # The same new extra-kosher lot is also in the cooler: counted in two areas, so
    # it is held until the owner confirms no unlogged moves (Codex PR #108 review).
    cooler = window('Cooler', 'area-3b-cooler')
    time.sleep(5)
    cooler_rows = [sheet_row(cooler, 'RH4', 1, items['extra'], ek_code, '120', 'lb', now())]
    time.sleep(5)
    close_area(cooler, cooler_rows)
    a3['with'] = [cooler]
    results.append(area_run(a3, rows + cooler_rows, [granola, cases15, items['extra']], workdir, env, owner, ref))
    return results


def evidence(uri, ids, items, report):
    """Read back the shortage, the new extra-kosher lot and the owner holds from staging."""
    ek_code = report['reference'] + '-EK-1'
    with connection(uri) as cur:
        shortage_ids = report.get('shortage', {}).get('exceptions', [])   # --hosted-groups has no shortage fixture
        cur.execute("""SELECT e.id,e.status,e.resolution_kind,e.resolution_ticket_id,e.resolution_receipt_number,
                              xr.action AS resolution_receipt_action,xr.status AS resolution_receipt_status,
                              fix.receipt_number AS resolution_posting_receipt_number,e.detail->'resolution' AS resolution
                       FROM exceptions e LEFT JOIN write_tickets xr ON xr.receipt_number=e.resolution_receipt_number
                       LEFT JOIN write_tickets fix ON fix.id=e.resolution_ticket_id
                       WHERE e.id=ANY(%s)""", (shortage_ids,))
        if 'shortage' in report:
            report['shortage']['after'] = [dict(r) for r in cur.fetchall()]
        cur.execute("""SELECT l.id,l.lot_code,l.kosher_tier,l.kosher_attested_by_actor_id,l.kosher_attested_at,l.entry_source
                       FROM lots l WHERE l.product_id=%s AND l.lot_code=%s""", (items['extra']['id'], ek_code.upper()))
        report['extra_kosher_lot'] = [dict(r) for r in cur.fetchall()]
        for lot in report['extra_kosher_lot']:
            cur.execute('SELECT COALESCE(sum(quantity_lb),0) AS lb, count(*) AS lines FROM transaction_lines WHERE lot_id=%s', (lot['id'],))
            lot.update(cur.fetchone())
        cur.execute("""SELECT e.id,e.kind,e.status,e.resolution_kind,e.resolved_by_actor_id,wt.receipt_number
                       FROM exceptions e JOIN write_tickets wt ON wt.id=e.ticket_id
                       WHERE e.kind IN ('LARGE_CORRECTION','KOSHER_ATTESTATION') AND wt.actor_id=%s ORDER BY e.id""",
                    (ids['owner']['id'],))
        report['owner_holds'] = [dict(r) for r in cur.fetchall()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hosted', action='store_true', help='one area against the deployed FastAPI-staging (no local server)')
    parser.add_argument('--hosted-groups', action='store_true',
                        help='three area groups on the deployed FastAPI-staging, then the whole-reset union proof (O-3)')
    args = parser.parse_args()
    try:
        rehearse(args.output, args.hosted or args.hosted_groups, args.hosted_groups)
        print(args.output)
    except Exception as exc:
        # Never print libpq errors or tracebacks that could carry a URI or key.
        print('Count rehearsal failed: ' + type(exc).__name__ + ': ' + str(exc)[:1500].replace(os.environ.get('DATABASE_URL', '\0'), '<db>'),
              file=sys.stderr)
        raise SystemExit(1)
