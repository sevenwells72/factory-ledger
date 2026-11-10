#!/usr/bin/env python3
"""Granola batch write-down to Arturo's 11/09/26 end-of-day physical count.

Approved by Michael 2026-11-10 ("apply") for three products only: 114, 108, 116.
Same approach as scripts/granola_writedown_0917.py: frozen plan, read-only
balance check before every post, durable journal written BEFORE each HTTP call,
exact-match detection of already-posted rows so a re-run posts nothing twice.
Posts through adjust tickets with the owner's personal key (never the shared key).

Default is dry-run (read-only SQL, no HTTP). --apply posts.

EXECUTED 2026-11-10: TX 2991-2995 (ADJ-261110-001..005). Do not replay or edit the plan;
a re-run verifies the five posted rows and posts nothing. Ticket tokens in the committed
journal are redacted.
"""
import json, os, subprocess, sys
from decimal import Decimal as D
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

JOURNAL = Path(__file__).resolve().parents[1] / 'audits/results/granola-count-1109-execution.json'
API = 'https://fastapi-production-b73a.up.railway.app'
NOTE = "Physical count 11/09/26 (Arturo, end of day). Approved by Michael 2026-11-10."
OWNER_ACTOR_ID = 1
PLAN = [  # oldest lots first within each product
    dict(product_id=114, lot_id=1709, lot_code='NOV 06 2026',  before='2956.5', change='-1976.5', after='980'),
    dict(product_id=108, lot_id=1689, lot_code='B26-1103-001', before='1740',   change='-1740',   after='0'),
    dict(product_id=108, lot_id=1697, lot_code='NOV 04 2026',  before='1740',   change='-1740',   after='0'),
    dict(product_id=108, lot_id=1703, lot_code='NOV 05 2026',  before='1392',   change='-1390',   after='2'),
    dict(product_id=116, lot_id=1720, lot_code='NOV 09 2026',  before='1750',   change='-1375',   after='375'),
]
TARGETS = {114: D('980'), 108: D('350'), 116: D('375')}
START = {114: D('2956.5'), 108: D('5220'), 116: D('1750')}

SQL = '''
WITH x AS (SELECT tl.product_id, tl.lot_id, tl.quantity_lb, t.id tx, t.type, r.reason_code, r.entered_by_actor_id, r.receipt_number, r.ticket_id
           FROM ledger_current_transaction_lines tl JOIN ledger_current_transactions t ON t.id=tl.transaction_id
           JOIN transactions r ON r.id=t.id
           WHERE t.effective_status='posted' AND tl.product_id IN (114,108,116))
SELECT json_build_object(
 'at', now(),
 'totals', (SELECT json_object_agg(product_id, s) FROM (SELECT product_id, sum(quantity_lb) s FROM x GROUP BY 1) a),
 'lots', (SELECT json_object_agg(lot_id, s) FROM (SELECT lot_id, sum(quantity_lb) s FROM x GROUP BY 1) a),
 'lot_codes', (SELECT json_object_agg(id, lot_code) FROM lots WHERE id IN (1709,1689,1697,1703,1720)),
 'max_tx', (SELECT max(id) FROM transactions),
 'count_adjusts', (SELECT coalesce(json_agg(json_build_object('tx',tx,'lot_id',lot_id,'qty',quantity_lb,'actor',entered_by_actor_id,'receipt',receipt_number,'ticket_id',ticket_id) ORDER BY tx),'[]'::json)
                   FROM x WHERE type='adjust' AND reason_code='physical_count' AND tx > 2990 AND lot_id IN (1709,1689,1697,1703,1720)));
'''


def stop(msg):
    print('STOP: ' + msg, file=sys.stderr)
    raise SystemExit(1)


def db_url():
    r = subprocess.run(['railway', 'variables', '--project', '2206e070-d160-4528-a4f1-86a587ad88c3',
                        '--service', 'FastAPI', '--environment', 'production', '--json'], capture_output=True, text=True)
    if r.returncode: stop('Railway variable read failed.')
    url = json.loads(r.stdout)['DATABASE_URL'].replace(':6543/', ':5432/')
    if ':5432/' not in url: stop('Not the port-5432 session pooler.')
    return url


_URL = None
def snapshot():
    global _URL
    _URL = _URL or db_url()
    r = subprocess.run(['psql', _URL, '-X', '-qAt', '-v', 'ON_ERROR_STOP=1'],
                       input='BEGIN; SET TRANSACTION READ ONLY;\n' + SQL + '\nROLLBACK;', text=True, capture_output=True)
    if r.returncode: stop('Read-only query failed (details suppressed).')
    s = json.loads(r.stdout, parse_float=D, parse_int=D)
    for lid, code in s['lot_codes'].items():
        want = next(p['lot_code'] for p in PLAN if str(p['lot_id']) == lid)
        if code != want: stop(f'Lot {lid} code changed: {code!r}.')
    return s


def classify(s):
    """Return number of plan rows already posted (must be an exact prefix); verify everything else is untouched."""
    done = 0
    adj = s['count_adjusts']
    expected_tot = dict(START)
    for i, p in enumerate(PLAN):
        bal = D(s['lots'].get(str(p['lot_id']), 0))
        match = [a for a in adj if a['lot_id'] == p['lot_id'] and D(a['qty']) == D(p['change']) and a['actor'] == OWNER_ACTOR_ID]
        if bal == D(p['after']) and len(match) == 1:
            if done != i: stop(f'Row {i+1} posted out of order.')
            done += 1
            expected_tot[p['product_id']] += D(p['change'])
        elif bal == D(p['before']) and not match:
            pass
        else:
            stop(f"Lot {p['lot_id']} ({p['lot_code']}): balance {bal}, {len(match)} matching count adjustment(s); "
                 f"expected {p['before']} (not posted) or {p['after']} (posted once). No writes.")
    if len(adj) != done: stop(f'{len(adj)} count adjustments found but {done} match the plan. No writes.')
    for pid, want in expected_tot.items():
        got = D(s['totals'].get(str(pid), 0))
        if got != want: stop(f'Product {pid}: ledger {got} lb, expected {want} lb — something else changed. No writes.')
    return done


def load_journal():
    return json.loads(JOURNAL.read_text()) if JOURNAL.exists() else {'rows': {}}


def save_journal(j):
    tmp = JOURNAL.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(j, f, indent=2, default=str); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, JOURNAL)


def call(key, method, path, body=None):
    req = Request(API + path, data=json.dumps(body).encode() if body is not None else None,
                  headers={'X-API-Key': key, 'Content-Type': 'application/json'}, method=method)
    try:
        with urlopen(req, timeout=60) as resp:
            return resp.status, json.load(resp)
    except HTTPError as e:
        try: return e.code, json.load(e)
        except Exception: return e.code, {}
    except Exception:
        stop(f'{method} {path}: no response / outcome uncertain. Do not retry blindly; re-run this script, which reconciles first.')


def apply():
    key = (Path.home() / '.config/factory-ledger/blubber_key').read_text().strip()
    code, who = call(key, 'GET', '/auth/whoami')
    if code != 200 or who.get('key_kind') != 'actor' or (who.get('actor') or {}).get('id') != OWNER_ACTOR_ID \
            or who['actor'].get('role') != 'owner':
        stop('Key is not the owner personal key.')
    j = load_journal()
    for i, p in enumerate(PLAN):
        s = snapshot()
        done = classify(s)
        if done > i:
            print(f"{i+1}/5 SKIP already posted: lot {p['lot_code']} = {p['after']} lb"); continue
        if done != i: stop('Unexpected state.')
        row = j['rows'].setdefault(str(i), {'plan': p, 'steps': []})
        # 1. prepare (creates a draft ticket only; posts nothing). Reuse a saved ticket on re-run.
        if 'ticket' not in row:
            row['steps'].append('prepare_sent'); save_journal(j)
            code, t = call(key, 'POST', '/adjust/prepare', dict(lot_id=p['lot_id'], delta_lb=float(D(p['change'])),
                                                               reason_code='physical_count', note=NOTE, client_source='api'))
            row['prepare'] = {'http': code, 'ticket_id': t.get('ticket_id'), 'payload_hash': t.get('payload_hash'),
                              'warnings': t.get('warnings'), 'blockers': t.get('blockers'), 'draft': t.get('draft')}
            save_journal(j)
            if code != 200 or not t.get('can_commit'): stop(f'Prepare refused (HTTP {code}): {json.dumps(t)[:600]}')
            d = t['draft']
            if d.get('lot_id') not in (None, p['lot_id']) or d.get('product_id') not in (None, p['product_id']):
                stop('Draft names a different lot/product.')
            need_ack = [w for w in t.get('warnings') or [] if w.get('requires_ack')]
            if need_ack: stop(f'Draft has warnings needing acknowledgement; not auto-acknowledging: {json.dumps(need_ack)[:600]}')
            row['ticket'] = t['ticket']; row['payload_hash'] = t['payload_hash']; save_journal(j)
        # 2. commit: > 500 lb with no photo is HELD for owner approval (nothing posts yet).
        row['steps'].append('commit_sent'); save_journal(j)
        code, c = call(key, 'POST', f"/tickets/{row['ticket']}/commit", dict(payload_hash=row['payload_hash']))
        row['commit'] = {'http': code, 'body': c}; save_journal(j)
        result = None
        if code == 200 and c.get('transaction_id'):
            result = c
        elif code == 202 and c.get('held') and c.get('exception_id'):
            # 3. owner approval releases the hold and posts under the preparer.
            row['steps'].append('approve_sent'); save_journal(j)
            code, a = call(key, 'POST', f"/exceptions/{c['exception_id']}/approve",
                           dict(note='Owner approval: ' + NOTE + ' No photo.'))
            row['approve'] = {'http': code, 'body': a}; save_journal(j)
            if code != 200 or not a.get('transaction_id'): stop(f'Approval did not post (HTTP {code}): {json.dumps(a)[:600]}')
            result = a
        else:
            stop(f'Unexpected commit response (HTTP {code}): {json.dumps(c)[:600]}')
        # 4. verify in the posted ledger before moving on.
        after = snapshot()
        if classify(after) != i + 1: stop('Post not verified in the ledger; no further writes.')
        row['verified'] = {'transaction_id': result.get('transaction_id'), 'receipt_number': result.get('receipt_number'), 'at': after['at']}
        save_journal(j)
        print(f"{i+1}/5 POSTED {result.get('receipt_number')} TX {result.get('transaction_id')}: "
              f"product {p['product_id']} lot {p['lot_code']} {p['before']} -> {p['after']} lb", flush=True)
    final = snapshot()
    if classify(final) != len(PLAN): stop('Final count mismatch.')
    report(final)


def report(s):
    print(f"Ledger snapshot {s['at']} (read-only), max TX {s['max_tx']}")
    for pid, target in TARGETS.items():
        got = D(s['totals'].get(str(pid), 0))
        print(f"  product {pid}: ledger {got} lb | counted {target} lb | {'MATCH' if got == target else 'not yet'}")
    for a in s['count_adjusts']:
        print(f"  TX {a['tx']} {a['receipt']} lot {a['lot_id']} {a['qty']} lb actor {a['actor']}")


if __name__ == '__main__':
    if '--apply' in sys.argv:
        apply()
    else:
        s = snapshot(); n = classify(s)
        print(f'DRY RUN: no HTTP calls, no writes. {n}/5 plan rows already posted; {5-n} pending.')
        report(s)
