#!/usr/bin/env python3
"""Frozen September 17 granola plan. Default: read-only, no HTTP calls.

Future --apply requires FACTORY_LEDGER_ACTOR_KEY and --actor NAME. Never falls
back to a shared/admin key. Current /adjust authorization does not support
named actors; apply will fail its API preview before any inventory commits.
The September 15 coconut script used /adjust with a shared key; see report.
"""
import argparse
from collections import defaultdict
from decimal import Decimal as D
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit, unquote
from urllib.request import Request, urlopen

PLAN = Path(__file__).with_name('granola_writedown_0917_plan.json')
API = 'https://fastapi-production-b73a.up.railway.app'
BULK = 'Physical count 2026-09-17 - likely shipped bulk to Sunshine, order 298 (not enterable in system)'
OTHER = 'Physical count 2026-09-17 (Arturo)'


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def query(sql):
    u = urlsplit(Path.home().joinpath('.config/factory-ledger/db_url').read_text().strip())
    env = {**os.environ, 'PGHOST': u.hostname, 'PGPORT': str(u.port or 5432),
           'PGUSER': unquote(u.username), 'PGPASSWORD': unquote(u.password),
           'PGDATABASE': u.path.lstrip('/'), 'PGSSLMODE': 'require',
           'PGOPTIONS': '-c default_transaction_read_only=on'}
    r = subprocess.run(['psql', '-X', '-qAt', '-v', 'ON_ERROR_STOP=1'],
                       input='BEGIN; SET TRANSACTION READ ONLY;\n' + sql + '\nROLLBACK;',
                       text=True, capture_output=True, env=env)
    check(r.returncode == 0, 'Read-only database query failed (credentials suppressed).')
    return json.loads(r.stdout, parse_float=D)


SNAPSHOT_SQL = '''
WITH p AS (SELECT id,name,odoo_code FROM products WHERE type='batch' AND name ILIKE '%granola%'),
posted AS (SELECT tl.*,t.business_date,t.type,t.adjust_reason,t.operator_id FROM ledger_current_transaction_lines tl JOIN ledger_current_transactions t ON t.id=tl.transaction_id WHERE t.effective_status='posted'),
lotdata AS (SELECT p.id product_id,l.id lot_id,l.lot_code,min(x.business_date) FILTER(WHERE x.type='make' AND x.quantity_lb>0) made_date,min(x.business_date) first_date,COALESCE(sum(x.quantity_lb),0) balance FROM p JOIN lots l ON l.product_id=p.id LEFT JOIN posted x ON x.lot_id=l.id AND x.product_id=p.id GROUP BY p.id,l.id)
SELECT json_build_object('at',now(),'products',(SELECT json_agg(p ORDER BY id) FROM p),'lots',(SELECT json_agg(l ORDER BY product_id,COALESCE(made_date,first_date),lot_id) FROM lotdata l WHERE balance<>0),'totals',(SELECT json_object_agg(id,balance) FROM (SELECT p.id,COALESCE(sum(x.quantity_lb),0) balance FROM p LEFT JOIN posted x ON x.product_id=p.id GROUP BY p.id)s),'vanilla_makes',(SELECT json_agg(v ORDER BY business_date,transaction_id,line_id) FROM (SELECT x.business_date,x.transaction_id,x.id line_id,l.lot_code,x.quantity_lb FROM posted x LEFT JOIN lots l ON l.id=x.lot_id WHERE x.product_id=112 AND x.type='make' AND x.quantity_lb>0 AND x.business_date>='2026-07-01')v),'existing_adjustments',(SELECT json_agg(x) FROM posted x JOIN p ON p.id=x.product_id WHERE x.type='adjust' AND x.adjust_reason LIKE 'Physical count 2026-09-17%'));
'''


def normalized(s):
    return {k: v for k, v in s.items() if k != 'at'}


def validate(plan, fresh):
    # JSON decimal strings in the committed snapshot are normalized on load.
    expected = plan['snapshot']
    check(normalized(fresh) == normalized(expected),
          'Ledger or catalog changed since the frozen plan; stop and re-plan, never auto-resize.')
    check(not fresh['existing_adjustments'], 'Prior count adjustments exist; refusing duplicate/partial replay.')
    balances = {l['lot_id']: D(l['balance']) for l in fresh['lots']}
    totals = {int(k): D(v) for k, v in fresh['totals'].items()}
    lots = {l['lot_id']: l for l in fresh['lots']}
    products = {p['id']: p for p in fresh['products']}
    sums, bulk = defaultdict(D), defaultdict(D)
    previous_date = {}
    for r in plan['adjustments']:
        pid, lid = r['product_id'], r['lot_id']
        l = lots[lid]
        check(pid not in (112, 121), 'Protected product in adjustments.')
        check(l['product_id'] == pid and l['lot_code'] == r['lot_code'] and
              products[pid]['odoo_code'] == r['product_code'] and
              products[pid]['name'] == r['product_name'], 'Product/lot identity mismatch.')
        date = l['made_date'] or l['first_date']
        check(date and date >= previous_date.get(pid, ''), 'FIFO order violated.')
        previous_date[pid] = date
        before, change, after = (D(r[k]) for k in ('before', 'change', 'after'))
        check(balances[lid] == before and change < 0 and before + change == after and after >= 0,
              'Invalid reduction or negative lot balance.')
        check(r['reason'] in (BULK, OTHER), 'Unexpected reason.')
        if r['reason'] == BULK:
            check(pid in (107, 108) and l['made_date'] >= '2026-08-14', 'Invalid Sunshine allocation.')
            check(sums[pid] == -bulk[pid], 'Sunshine must be the first reduction.')
            bulk[pid] -= change
        balances[lid] = after
        sums[pid] += change
    check(dict(bulk) == {107: D(4000), 108: D(6000)}, 'Sunshine totals differ.')
    for pid, before in totals.items():
        target = D(10560 if pid == 107 else 2000 if pid == 108 else 1050 if pid == 121 else before if pid == 112 else 0)
        check(before + sums[pid] == target, f'Product {pid} target mismatch.')
        check(sum(balances[lid] for lid,l in lots.items() if l['product_id'] == pid) == target,
              'Lot totals do not reconcile to posted product total.')
    return sums


def load_plan():
    p = json.loads(PLAN.read_text(), parse_float=D)
    for l in p['snapshot']['lots']:
        l['balance'] = D(l['balance'])
    p['snapshot']['totals'] = {k: D(v) for k,v in p['snapshot']['totals'].items()}
    for m in p['snapshot']['vanilla_makes']:
        m['quantity_lb'] = D(m['quantity_lb'])
    return p


def payload(r, mode):
    return dict(mode=mode, product_name=r['product_code'], lot_code=r['lot_code'],
                adjustment_lb=float(D(r['change'])), reason=r['reason'],
                occurred_at='2026-09-17T12:00:00-04:00', backfill=False)


def api(key, path, body=None):
    req = Request(API + path, data=json.dumps(body).encode() if body else None,
                  headers={'X-API-Key': key, 'Content-Type': 'application/json'},
                  method='POST' if body else 'GET')
    try:
        with urlopen(req, timeout=45) as response:
            return json.load(response, parse_float=D)
    except Exception:
        raise RuntimeError('API call failed or outcome uncertain; stopped, no automatic retry.') from None


def apply(plan, actor):
    # This path is never reached in dry-run. No credentials are fetched from Railway.
    key = os.environ.get('FACTORY_LEDGER_ACTOR_KEY')
    check(key and actor, '--apply requires FACTORY_LEDGER_ACTOR_KEY and --actor NAME.')
    digest = hashlib.sha256(key.encode()).hexdigest()
    a = query("SELECT COALESCE(json_agg(x),'[]'::json) FROM (SELECT name,role FROM actors WHERE active AND key_hash='" + digest + "') x;")
    check(len(a) == 1 and a[0]['name'] == actor, 'Key is not the requested active named actor; no shared-key fallback.')
    who = api(key, '/auth/whoami')
    check(who.get('key_kind') == 'actor' and who.get('actor', {}).get('name') == actor,
          'API did not authenticate the named actor.')
    # Preview ALL calls before the first commit. Split-lot preview balances are
    # compared to the original live balance; sequential balances are validated above.
    initial = {l['lot_id']: D(l['balance']) for l in plan['snapshot']['lots']}
    for r in plan['adjustments']:
        v = api(key, '/adjust', payload(r, 'preview'))
        check(v.get('mode') == 'preview' and v.get('product_id') == r['product_id'] and
              v.get('lot_code') == r['lot_code'] and D(str(v.get('current_quantity_lb'))) == initial[r['lot_id']],
              'API preview identity/balance mismatch.')
    validate(plan, query(SNAPSHOT_SQL))
    for i, r in enumerate(plan['adjustments'], 1):
        live = query(SNAPSHOT_SQL)
        current = {l['lot_id']: D(l['balance']) for l in live['lots']}
        check(current.get(r['lot_id'], D(0)) == D(r['before']), 'Lot changed before commit; stop.')
        result = api(key, '/adjust', payload(r, 'commit'))
        check(result.get('success') is True and result.get('product_id') == r['product_id'] and
              result.get('lot_code') == r['lot_code'] and D(str(result.get('new_balance_lb'))) == D(r['after']),
              'Unexpected commit response; stop and inspect posted ledger, do not retry.')
        print(f"Committed {i}/{len(plan['adjustments'])}; transaction {result.get('transaction_id')}", flush=True)
    after = query(SNAPSHOT_SQL)
    for p in plan['snapshot']['products']:
        pid = p['id']
        delta = sum(D(r['change']) for r in plan['adjustments'] if r['product_id'] == pid)
        check(D(after['totals'][str(pid)]) == D(plan['snapshot']['totals'][str(pid)]) + delta,
              'Final product total mismatch.')
    events = after['existing_adjustments'] or []
    check(len(events) == len(plan['adjustments']), 'Posted event count mismatch.')
    for r in plan['adjustments']:
        check(sum(e['product_id'] == r['product_id'] and e['lot_id'] == r['lot_id'] and
                  D(e['quantity_lb']) == D(r['change']) and e['adjust_reason'] == r['reason'] and
                  e['business_date'] == '2026-09-17' and e['operator_id'] == actor for e in events) == 1,
              'Posted event metadata/actor mismatch; manual review required.')
    print('Final posted balances and exact named-actor events verified.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--apply', action='store_true')
    parser.add_argument('--actor')
    args = parser.parse_args()
    plan = load_plan()
    fresh = query(SNAPSHOT_SQL)
    sums = validate(plan, fresh)
    print('DRY RUN — no database writes or HTTP requests.' if not args.apply else 'APPLY preflight')
    print('Snapshot:', fresh['at'])
    print('Posted ledger verified; frozen lot balances, FIFO, reasons, nonnegative results and targets PASS.')
    print(f"Adjustments: {len(plan['adjustments'])}; reduction: {sum(-D(r['change']) for r in plan['adjustments']):,.2f} lb")
    print('Sunshine: 107 = 4,000.00 lb; 108 = 6,000.00 lb. Arturo: 9,335.46 lb.')
    print('product | lot (id) | current lb | change lb | new lb | reason')
    for r in plan['adjustments']:
        print(f"{r['product_id']} | {r['lot_code']} ({r['lot_id']}) | {D(r['before']):,.2f} | {D(r['change']):,.2f} | {D(r['after']):,.2f} | {r['reason']}")
    print('\nPRODUCT TOTALS: product | current lb | change lb | new lb | target check')
    for p in fresh['products']:
        pid = p['id']; before = D(fresh['totals'][str(pid)])
        print(f"{pid} {p['name']} | {before:,.2f} | {sums[pid]:,.2f} | {before+sums[pid]:,.2f} | " + ('UNCHANGED (excluded)' if pid == 112 else 'PASS'))
    print('\nVANILLA CRISP MAKES SINCE 2026-07-01: date | lot | lb | transaction')
    for m in fresh['vanilla_makes']:
        print(f"{m['business_date']} | {m['lot_code']} | {D(m['quantity_lb']):,.2f} | {m['transaction_id']}")
    if args.apply:
        apply(plan, args.actor)
    else:
        print('\nSTOP: preview only. No API keys loaded. No adjustments posted.')
        print('Apply limitation: located coconut script uses shared key; current /adjust excludes named-actor keys.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, KeyError) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        raise SystemExit(1)
