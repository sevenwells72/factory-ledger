#!/usr/bin/env python3
"""Frozen September 17 granola plan. Default: read-only, no HTTP calls.

Michael approved shared-key posting on September 17. --apply requires the
explicit --allow-shared-key flag. Default is dry-run with no API calls.
A durable request journal plus exact posted-event matching prevents replay.
Keep the journal with this script, including after an uncertain API result.
"""
import argparse
from collections import defaultdict
from decimal import Decimal as D
import fcntl
import hashlib
import copy
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
SUFFIX = ' (entered via shared key by Michael)'
JOURNAL = PLAN.parents[1] / 'audits/results/granola-writedown-0917-execution.json'
LOCK = JOURNAL.with_suffix('.lock')


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
        check(r['reason'] in (BULK + SUFFIX, OTHER + SUFFIX), 'Unexpected reason.')
        if r['reason'] == BULK + SUFFIX:
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


def save_journal(journal):
    """Persist intent BEFORE HTTP; atomic replace and fsync survive interruption."""
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    tmp = JOURNAL.with_suffix('.tmp')
    with tmp.open('w') as out:
        json.dump(journal, out, indent=2, default=str)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, JOURNAL)
    fd = os.open(str(JOURNAL.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fingerprint():
    return hashlib.sha256(PLAN.read_bytes()).hexdigest()


def event_matches(e, r):
    return (e['product_id'] == r['product_id'] and e['lot_id'] == r['lot_id']
            and D(e['quantity_lb']) == D(r['change'])
            and e['adjust_reason'] == r['reason']
            and e['business_date'] == '2026-09-17'
            and e['operator_id'] == 'legacy-shared-key')


def reconcile(plan, fresh, journal=None):
    """Accept only a unique, ordered posted prefix; check ALL remaining balances.

    This also checks protected lots and zero products. Split-reason rows use the
    frozen sequential 'before' balance, not the lot's original starting balance.
    """
    validate(plan, plan['snapshot'])
    rows = plan['adjustments']
    events = sorted(fresh['existing_adjustments'] or [], key=lambda e: e['transaction_id'])
    check(len(events) <= len(rows), 'Too many count events; refusing duplicate/foreign postings.')
    check(len({e['transaction_id'] for e in events}) == len(events), 'Count transaction has multiple lines.')
    for i, e in enumerate(events):
        check(event_matches(e, rows[i]),
              f"Posted count event {e['transaction_id']} does not match plan row {i+1}; stop.")
    count = len(events)
    if journal:
        check(journal['plan_sha256'] == fingerprint(), 'Plan changed since execution journal was created.')
        for attempt in journal.get('attempts', []):
            seq = attempt['sequence']
            if seq <= count:
                if attempt.get('transaction_id') is not None:
                    check(events[seq-1]['transaction_id'] == attempt['transaction_id'],
                          'Journal transaction does not match posted ledger.')
            else:
                raise RuntimeError(f"Request {seq} has an unresolved/absent posting. Do not retry: reconcile its outcome manually.")
    expected = copy.deepcopy(plan['snapshot'])
    balances = {l['lot_id']: D(l['balance']) for l in expected['lots']}
    for r in rows[:count]:
        balances[r['lot_id']] = D(r['after'])
        expected['totals'][str(r['product_id'])] += D(r['change'])
    actual = {l['lot_id']: D(l['balance']) for l in fresh['lots'] or []}
    diffs = []
    for lid in sorted(set(balances) | set(actual)):
        want, got = balances.get(lid, D(0)), actual.get(lid, D(0))
        if want != got:
            diffs.append(f'lot {lid}: expected {want:,.2f} lb; actual {got:,.2f} lb; difference {got-want:+,.2f} lb')
    for pid in sorted(set(expected['totals']) | set(fresh['totals']), key=int):
        want, got = expected['totals'].get(pid), fresh['totals'].get(pid)
        if want is None or got is None or D(want) != D(got):
            diffs.append(f'product {pid}: expected {want} lb; actual {got} lb')
    check(not diffs, 'Balance differences; NO further writes:\n' + '\n'.join(diffs))
    check(fresh['products'] == expected['products'], 'Granola catalog changed; stop.')
    check(fresh['vanilla_makes'] == expected['vanilla_makes'], 'Vanilla Crisp makes changed; stop.')
    original_lots = {l['lot_id']: l for l in expected['lots']}
    for l in fresh['lots'] or []:
        old = original_lots.get(l['lot_id'])
        check(old and all(l[k] == old[k] for k in ('product_id','lot_code','made_date','first_date')),
              f"Lot {l['lot_id']} metadata changed; stop.")
    return count, events


def get_shared_key():
    # Same project/service/environment and in-memory key handling as coconut.
    r = subprocess.run(['railway', 'variables', '--project',
                        '2206e070-d160-4528-a4f1-86a587ad88c3', '--service',
                        'FastAPI', '--environment', 'production', '--json'],
                       capture_output=True, text=True)
    check(r.returncode == 0, 'Railway key retrieval failed (details suppressed).')
    try:
        values = json.loads(r.stdout)
        key = values['API_KEY']
    except (ValueError, KeyError):
        raise RuntimeError('Railway shared key unavailable.') from None
    check(isinstance(key, str) and bool(key), 'Railway shared key unavailable.')
    return key


def print_summary(plan, fresh):
    print('| Product | Before lb | Change lb | Final lb | Matches target |')
    print('|---|---:|---:|---:|---|')
    for p in plan['snapshot']['products']:
        pid = str(p['id']); before = D(plan['snapshot']['totals'][pid])
        change = sum(D(r['change']) for r in plan['adjustments'] if str(r['product_id']) == pid)
        actual = D(fresh['totals'][pid]); target = before + change
        print(f"| {p['name']} ({pid}) | {before:,.2f} | {change:+,.2f} | {actual:,.2f} | {'yes' if actual == target else 'no'} |")
    print('Vanilla Crisp (112): excluded; unchanged at 240.00 lb.' )


def apply(plan, allow_shared_key):
    check(allow_shared_key, '--apply refuses shared credentials without --allow-shared-key.')
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another execution is running; stop.') from None
        journal = json.loads(JOURNAL.read_text()) if JOURNAL.exists() else None
        fresh = query(SNAPSHOT_SQL)
        count, events = reconcile(plan, fresh, journal)
        if count == len(plan['adjustments']):
            print('Already completed: all 24 exact events verified. No credentials loaded, no HTTP calls.')
            print_summary(plan, fresh)
            return
        if journal is None:
            journal = dict(plan_sha256=fingerprint(), approved_by='Michael',
                           attribution_suffix=SUFFIX, before=fresh, attempts=[])
        print(f'All lot/product balances match; {count} confirmed rows will be skipped.', flush=True)
        key = get_shared_key()
        who = api(key, '/auth/whoami')
        check(who.get('key_kind') == 'legacy_ledger' and who.get('actor') is None,
              'Expected coconut-style shared ledger key.')
        # Preview all remaining rows before ANY new commits. For split-lot rows,
        # API preview starts from the live pre-run lot balance; SQL/plan enforce
        # the sequential before/after amounts again immediately before commit.
        initial = {l['lot_id']: D(l['balance']) for l in fresh['lots']}
        for r in plan['adjustments'][count:]:
            v = api(key, '/adjust', payload(r, 'preview'))
            check(v.get('mode') == 'preview' and v.get('product_id') == r['product_id']
                  and v.get('lot_code') == r['lot_code']
                  and D(str(v.get('current_quantity_lb'))) == initial[r['lot_id']]
                  and D(str(v.get('adjustment_lb'))) == D(r['change'])
                  and D(str(v.get('new_balance_lb'))) == initial[r['lot_id']] + D(r['change'])
                  and v.get('reason') == r['reason'], 'API preview mismatch; no commit made.')
        for i in range(count, len(plan['adjustments'])):
            r = plan['adjustments'][i]
            # Recheck the ENTIRE planned scope before every write, including 112.
            live = query(SNAPSHOT_SQL)
            confirmed, _ = reconcile(plan, live, journal)
            check(confirmed == i, 'Unexpected concurrent posting; stop.')
            current = {l['lot_id']: D(l['balance']) for l in live['lots']}
            check(current.get(r['lot_id'], D(0)) == D(r['before']),
                  f"Lot {r['lot_id']}: expected {r['before']} lb, actual {current.get(r['lot_id'],0)} lb; no write.")
            attempt = dict(sequence=i+1, status='in_flight', before_checked_at=live['at'],
                           expected_product_id=r['product_id'], expected_lot_id=r['lot_id'],
                           request=payload(r, 'commit'))
            journal['attempts'].append(attempt)
            save_journal(journal)
            try:
                result = api(key, '/adjust', attempt['request'])
                # Save the response even if it violates the expected contract.
                attempt['response'] = result
                attempt['transaction_id'] = result.get('transaction_id')
                save_journal(journal)
                check(result.get('success') is True and result.get('product_id') == r['product_id']
                      and result.get('lot_code') == r['lot_code']
                      and D(str(result.get('new_balance_lb'))) == D(r['after'])
                      and D(str(result.get('adjustment_lb'))) == D(r['change'])
                      and result.get('reason') == r['reason'],
                      'Unexpected commit response; no retry. Inspect posted ledger.')
                after = query(SNAPSHOT_SQL)
                confirmed, events = reconcile(plan, after, journal)
                check(confirmed == i+1, 'Commit not verified in posted ledger; no retry.')
                attempt['status'] = 'verified'
                attempt['verified_at'] = after['at']
                save_journal(journal)
                print(f"Verified {i+1}/24: transaction {result['transaction_id']}; lot {r['lot_id']} = {r['after']} lb", flush=True)
            except BaseException:
                attempt['status'] = 'uncertain_or_error'
                save_journal(journal)
                raise
        final = query(SNAPSHOT_SQL)
        confirmed, events = reconcile(plan, final, journal)
        check(confirmed == len(plan['adjustments']), 'Final event count mismatch.')
        journal['after'] = final
        journal['transaction_ids'] = [e['transaction_id'] for e in events]
        journal['success'] = True
        save_journal(journal)
        print('VERIFIED SUCCESS: 24 exact shared-key adjustments; every target matches.')
        print_summary(plan, final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--apply', action='store_true')
    parser.add_argument('--allow-shared-key', action='store_true',
                        help='Explicit Michael-approved exception; required for --apply.')
    args = parser.parse_args()
    if args.apply:
        check(args.allow_shared_key, '--apply requires explicit --allow-shared-key; refusing.')
    plan = load_plan()
    if args.apply:
        apply(plan, args.allow_shared_key)
        return
    fresh = query(SNAPSHOT_SQL)
    journal = json.loads(JOURNAL.read_text()) if JOURNAL.exists() else None
    count, _ = reconcile(plan, fresh, journal)
    print('DRY RUN: no API calls, credentials, or database writes.')
    print(f"Read-only snapshot: {fresh['at']}; exact posted prefix: {count}/24; remaining: {24-count}.")
    print('All frozen balances, reasons, FIFO, protected products and nonnegative results verified.')
    for i, r in enumerate(plan['adjustments'], 1):
        print(f"{i:02d} {'SKIP verified' if i <= count else 'PENDING'} | {r['product_id']} | {r['lot_code']} ({r['lot_id']}) | {r['before']} -> {r['after']} lb | {r['reason']}")
    print('Planned reduction: 19,335.46 lb. Sunshine 10,000.00 lb; Arturo 9,335.46 lb.')
    if count == 24:
        print_summary(plan, fresh)
    print('STOP: dry-run complete.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        raise SystemExit(1)
