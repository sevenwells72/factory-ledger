#!/usr/bin/env python3
"""Owner-approved reset executor. Default: read-only dry run; never direct DB writes.

No retry loop: /adjust has no server idempotency key. An unresolved commit is
reconciled by its durable intent and exact posted event, or the run stops.
"""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler
from urllib.parse import urlsplit

import fresh_start_common as f
from reset_preview import COLUMNS
from verify_reset import write_verification

API = 'https://fastapi-production-b73a.up.railway.app'
KEY_FILE = Path.home() / '.config/factory-ledger/blubber_key'
TARGET_DEFAULT_KEY = KEY_FILE   # v3: the target's key file unless a test overrides KEY_FILE
JOURNAL_ROOT = f.OUT / 'apply-state'
LOCK = f.OUT / '.apply-reset.lock'
RESULTS = f.OUT / 'apply-results'
# Additional identity coverage: /adjust resolves text against the whole catalog.
APPLY_SNAPSHOT_SQL = f.SQL.replace(" 'snapshot_at',current_timestamp,", """ 'snapshot_at',current_timestamp,
 'catalog',COALESCE((SELECT json_agg(c) FROM
   (SELECT id,name,odoo_code FROM products) c),'[]'::json),
 'transaction_line_counts',COALESCE((SELECT json_object_agg(transaction_id,n) FROM
   (SELECT transaction_id,count(*) n FROM ledger_current_transaction_lines
    WHERE transaction_id IN (SELECT transaction_id FROM x WHERE type='adjust')
    GROUP BY transaction_id) counts),'{}'::json),""")
ACTOR_SQL = """BEGIN; SET TRANSACTION READ ONLY;
SELECT COALESCE(json_agg(a), '[]'::json) FROM
 (SELECT id,name,role FROM actors WHERE active AND key_hash=:'actor_hash') a;
ROLLBACK;"""

APPROVAL_FIELDS = ('Preview file', 'SHA-256', 'Approved product IDs', 'Owner name',
                   'Named actor', 'Date', 'Signature', 'Opening entry classification')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def close(a, b):
    return abs(f.number(a) - f.number(b)) <= f.TOLERANCE


def now():
    return datetime.now(timezone.utc).isoformat()


def load_plan(preview, approval, count, target=None):
    if f.local_path(preview).suffix == ".json":
        return load_v3_plan(preview, approval, count, target)
    preview, approval, count = map(f.local_path, (preview, approval, count))
    raw, signed, counted = preview.read_bytes(), approval.read_bytes(), count.read_bytes()
    fields = {}
    for line in signed.decode('utf-8').splitlines():
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        if key in APPROVAL_FIELDS:
            require(key not in fields, 'Duplicate approval field: ' + key)
            fields[key] = value.strip()
    require(set(fields) == set(APPROVAL_FIELDS) and all(fields.values()),
            'Approval is incomplete. Fill and sign approval.md before even a dry run.')
    require(fields['Preview file'] == preview.name, 'Approved preview filename does not match.')
    sha = digest(raw)
    require(re.fullmatch('[0-9a-fA-F]{64}', fields['SHA-256']) and fields['SHA-256'].lower() == sha,
            'Preview SHA-256 does not match the signed approval; stop and obtain fresh approval.')
    require(fields['Signature'] == fields['Owner name'], 'Signature must repeat the owner name exactly.')
    require(fields['Named actor'] not in ('legacy-shared-key', 'shared', 'master'), 'A personal named actor is required.')
    approval_date = date.fromisoformat(fields['Date'])  # Compared only with server time in preflight.
    require(fields['Opening entry classification'] == 'N',
            'Owner must approve N for this plan’s verified administrative opening entries.')
    require(re.fullmatch(r'[1-9]\d*(?:\s*,\s*[1-9]\d*)*', fields['Approved product IDs']),
            'Approved product IDs must be a comma-separated list of positive integers.')
    ids = [int(x.strip()) for x in fields['Approved product IDs'].split(',')]
    require(len(set(ids)) == len(ids), 'Duplicate product ID in approval.')
    require(not set(ids) & f.BILLING_RESET_EXCLUDED_IDS, 'Billing item is excluded from reset approval.')
    require(not (set(ids) & f.EXCLUDED_PRODUCT_IDS), 'Products 171 and 209 are excluded by the owner; never approve or apply them.')
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    require(reader.fieldnames == COLUMNS, 'Expected an unedited reset_preview.py CSV, with its exact headers.')
    records = list(reader)
    require(records and all(None not in r and None not in r.values() for r in records), 'Empty or malformed preview.')
    for key in ('cutoff', 'snapshot_at', 'input_sha256'):
        require(len({r[key] for r in records}) == 1 and records[0][key], 'Inconsistent preview metadata: ' + key)
    require(records[0]['input_sha256'] == digest(counted), 'Count CSV differs from the count used for the approved preview.')
    cutoff, at = f.stamp(records[0]['cutoff']), f.stamp(records[0]['snapshot_at'])
    require(cutoff <= at, 'Preview cutoff is later than its snapshot.')
    reason = f.opening_balance_reason(cutoff)
    records = [r for r in records if not r['product_id'] or int(r['product_id']) not in f.EXCLUDED_PRODUCT_IDS]
    selected, seen, held_inactive = [], set(), set()
    for r in records:
        if r['row_type'] != 'LOT_REVIEW':
            continue
        pid = int(r['product_id'])
        if r['product_active'] == 'False' and r['rounded_adjustment_lb'] and f.number(r['rounded_adjustment_lb']):
            held_inactive.add(pid)
        change = f.number(r['adjustment_lb']) if r['adjustment_lb'] else None
        require(not change or pid in ids, f'Product {pid} has a proposed adjustment but is not approved. Nothing will post.')
        if pid not in ids:
            continue
        require(r['group_status'].startswith('READY'), f'Product {pid}: {r["group_status"]}; not approvable.')
        require(not change or r['product_active'] == 'True', f'Product {pid}: {f.INACTIVE_HOLD}.')
        require(r['reason'] == reason, f'Product {pid}: opening reason does not match the shared definition.')
        r = dict(r, pid=pid, lid=int(r['lot_id']) if r['lot_id'] else None, change=change)
        require((pid, r['lot_code']) not in seen, 'Duplicate lot row in preview.')
        seen.add((pid, r['lot_code']))
        require(r['lid'] is not None or (not change and not r['lot_code']), 'Unknown lot; the apply step never creates lots.')
        if change is not None:
            require(r['lid'] and r['lot_code'] and r['SKU'], 'Adjustment requires an existing lot ID, exact code and SKU.')
            original = f.number(r['candidate_adjustment_lb'])
            rounded = f.round_adjustment(original)
            require(change == rounded == f.number(r['rounded_adjustment_lb']) and
                    original == f.number(r['counted_lb']) - f.number(r['adjustment_basis_lb']) and
                    f.number(r['rounding_delta_lb']) == rounded - original and
                    f.number(r['planned_balance_lb']) == f.number(r['current_lb']) + rounded and
                    close(r['planned_balance_lb'], r['expected_current_lb']),
                    'Preview original/rounded adjustment arithmetic is inconsistent; regenerate and reapprove.')
            require(f.number(r['expected_current_lb']) >= 0, 'A reset must not target negative stock.')
            require(math.isfinite(float(change)) and close(str(float(change)), change), 'Adjustment cannot be represented safely by /adjust.')
        selected.append(r)
    require(set(ids) == {r['pid'] for r in selected}, 'Approved product is absent from preview.')
    return dict(preview=preview, approval=approval, count=count, sha=sha,
                approval_sha=digest(signed), count_sha=digest(counted), actor=fields['Named actor'],
                owner=fields['Owner name'], approval_date=approval_date, cutoff=cutoff, at=at, reason=reason, ids=set(ids),
                rows=selected, records=records, held_inactive=held_inactive)


def unchanged(plan):
    if plan.get("version") == 3:
        for item in plan["sources"].values():
            require(digest(f.local_path(item["path"]).read_bytes()) == item["sha256"], "An approved v3 source changed; stop.")
    for key, hashkey in (('preview', 'sha'), ('approval', 'approval_sha'), ('count', 'count_sha')):
        require(digest(plan[key].read_bytes()) == plan[hashkey], 'An approved input changed during the run; stop.')


def read_key(key_file=None):
    KEY_FILE_ = key_file or KEY_FILE
    try:
        require(KEY_FILE_.is_file() and not KEY_FILE_.is_symlink(), 'Personal key file is missing or is a symlink.')
        require(stat.S_IMODE(KEY_FILE_.stat().st_mode) & 0o077 == 0, 'Personal key file must be private (chmod 600).')
        value = KEY_FILE_.read_text().strip()
        require(value and not any(c.isspace() for c in value), 'Personal key file is empty or malformed.')
        return value
    except (OSError, UnicodeError):
        raise RuntimeError('Cannot read personal key file; credential details suppressed.') from None


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError('Redirect refused; the key must never be forwarded to another address.')


class LiveBackend:
    """Only request() can write, and run() calls it solely after explicit confirmation."""
    @staticmethod
    def read_json(sql, variables=()):
        require(sql in (APPLY_SNAPSHOT_SQL, ACTOR_SQL), 'Only fixed read-only queries are allowed.')
        require((sql == APPLY_SNAPSHOT_SQL and not variables) or
                (sql == ACTOR_SQL and len(variables) == 2 and variables[0] == '-v' and
                 re.fullmatch(r'actor_hash=[0-9a-f]{64}', variables[1])), 'Invalid read-only query parameters.')
        try:
            url = f.CREDENTIAL.read_text().strip().replace(':6543/', ':5432/')
            require(urlsplit(url).port == 5432, 'Read-only DB connection must use port 5432.')
            require(url and '\n' not in url, 'Read-only database credential is missing or invalid.')
            env = dict(os.environ, DATABASE_URL=url, PGCONNECT_TIMEOUT='15')
            env.pop('PGOPTIONS', None)
            result = subprocess.run([str(f.WRAPPER), '-X', '-qAt', '-v', 'ON_ERROR_STOP=1', *variables],
                                    input=sql, text=True, capture_output=True, cwd=f.ROOT, env=env, timeout=60)
            require(result.returncode == 0, 'Read-only query failed; details suppressed.')
            return json.loads(result.stdout, parse_float=f.D)
        except (OSError, ValueError, subprocess.SubprocessError):
            raise RuntimeError('Read-only database lookup failed; details suppressed.') from None

    def snapshot(self):
        return self.read_json(APPLY_SNAPSHOT_SQL)

    def actor(self, key):
        # Only a hash reaches psql. The key never goes to a subprocess, log,
        # journal, environment variable, command line or local output file.
        actors = self.read_json(ACTOR_SQL, ('-v', 'actor_hash=' + digest(key.encode())))
        require(isinstance(actors, list) and len(actors) == 1,
                'Key does not identify one active named actor. Shared/master keys are refused.')
        return actors[0]['name']

    def request(self, key, route, body):
        require(route in ('/auth/whoami', '/adjust'), 'Unsupported API route.')
        req = Request(API + route, data=json.dumps(body).encode() if body is not None else None,
                      headers={'X-API-Key': key, 'Content-Type': 'application/json'},
                      method='POST' if body is not None else 'GET')
        try:
            # No proxies/redirects/retries. Never expose raw HTTP errors or bodies.
            with build_opener(ProxyHandler({}), NoRedirect()).open(req, timeout=45) as response:
                require(response.status == 200, 'Uncertain API response.')
                result = json.load(response, parse_float=f.D)
                require(isinstance(result, dict), 'Malformed API response.')
                return result
        except Exception:
            raise RuntimeError('API outcome failed or uncertain. Stopped; no automatic retry; details suppressed.') from None


class Journal:
    def __init__(self, path, plan):
        self.path = f.local_path(path)
        # Version 2 = A1 ticket-era v3 journals (intents carry ticket_id); 1 = legacy direct /adjust.
        self.header = dict(kind='header', version=2 if plan.get('version') == 3 else 1,
                           preview_sha256=plan['sha'], approval_sha256=plan['approval_sha'],
                           count_sha256=plan['count_sha'], actor=plan['actor'], cutoff=plan['cutoff'].isoformat())
        self.events = []
        if self.path.exists():
            raw = self.path.read_bytes()
            require(not raw or raw.endswith(b'\n'), 'Journal has a torn final line. Preserve it; manual reconciliation is required.')
            previous = ''
            for index, line in enumerate(raw.splitlines()):
                try:
                    event = json.loads(line)
                    saved = event.pop('sha256')
                    require(event['seq'] == index and event['previous'] == previous and saved == self.hash(event),
                            'Journal chain mismatch; preserve the file and investigate.')
                    previous = saved
                    self.events.append(dict(event, sha256=saved))
                except (ValueError, KeyError, TypeError):
                    raise RuntimeError('Journal is corrupt; preserve it and investigate.') from None
            require(self.events and all(self.events[0].get(k) == v for k, v in self.header.items()),
                    'Journal belongs to another approval/preview; never delete it to force a rerun.')

    @staticmethod
    def hash(event):
        return digest(json.dumps(event, sort_keys=True, separators=(',', ':')).encode())

    def append(self, **event):
        if not self.events:
            require(event.get('kind') == 'header', 'Journal header required.')
        event.update(seq=len(self.events), previous=self.events[-1]['sha256'] if self.events else '', at=now())
        event['sha256'] = self.hash(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Make newly created journal directories durable before the first intent.
        for parent in (self.path.parent.parent, self.path.parent.parent.parent):
            dfd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        fd = os.open(self.path, os.O_APPEND | os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'a', encoding='utf-8') as out:
            out.write(json.dumps(event, sort_keys=True, separators=(',', ':')) + '\n')
            out.flush()
            os.fsync(out.fileno())
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self.events.append(event)
        return event

    def start(self):
        if not self.events:
            self.append(**self.header)

    def attempts(self):
        intents, receipts = {}, {}
        for e in self.events[1:]:
            if e['kind'] == 'intent':
                intents[e['seq']] = e
            elif e['kind'] == 'receipt':
                require(e['intent'] in intents and e['intent'] not in receipts, 'Invalid journal receipt sequence.')
                receipts[e['intent']] = e
            else:
                require(False, 'Unknown journal record; stop.')
        return [(i, receipts.get(seq)) for seq, i in intents.items()]


def payload(plan, row, mode):
    return dict(mode=mode, product_name=row['SKU'], lot_code=row['lot_code'],
                adjustment_lb=float(row['change']), reason=row.get('reason', plan['reason']),
                occurred_at=row.get('cutoff', plan['cutoff'].isoformat()), backfill=row.get('backfill', False))


def api_call(backend, key, journal, phase, row, body, validate, before=None):
    route = '/auth/whoami' if phase == 'identity' else '/adjust'
    require(phase in ('identity', 'preview', 'commit'), 'Unknown request phase.')
    intent = None
    if phase == 'commit':
        require(before is not None, 'Write intent requires a fresh server snapshot.')
        known = sorted({x['transaction_id'] for x in before['lines']
                        if x['product_id'] == row['pid'] and x['lot_id'] == row['lid']})
        journal.start()
        intent = journal.append(kind='intent', phase=phase, lot_id=row['lid'], product_id=row['pid'],
                                route=route, payload=body, known_transaction_ids=known,
                                server_snapshot_at=before['snapshot_at'])
    result = None
    try:
        result = backend.request(key, route, body)
        validate(result)
    except Exception:
        if intent is None:
            # No inventory write was requested. A later run may repeat this read.
            raise RuntimeError('Read-only API check failed or mismatched. No commit requested by this call; safe to rerun after resolving it.') from None
        tid = result.get('transaction_id') if isinstance(result, dict) else None
        journal.append(kind='receipt', intent=intent['seq'], outcome='uncertain',
                       transaction_id=tid if type(tid) is int and tid > 0 else None)
        raise RuntimeError('Write request failed, mismatched or uncertain. Stopped; preserve journal.jsonl; no blind retry.') from None
    if intent is not None:
        # Persist only validated metadata, never arbitrary response text/headers.
        journal.append(kind='receipt', intent=intent['seq'], outcome='ok',
                       transaction_id=result['transaction_id'])
    return result


def reconcile(plan, s, journal):
    """Return completed lot -> exact ledger line; never infer success from balance alone."""
    rows = {r['lid']: r for r in plan['rows'] if r['lid']}
    completed, used = {}, set()
    for intent, receipt in journal.attempts():
        if intent['phase'] != 'commit':
            require(intent['phase'] in ('identity', 'preview'), 'Unknown journal phase.')
            continue  # Legacy read-only records, even interrupted ones, never block a rerun.
        lid = intent['lot_id']
        require(lid in rows and rows[lid]['change'] and lid not in used, 'Duplicate or out-of-plan commit intent.')
        used.add(lid)
        row = rows[lid]
        require(intent['product_id'] == row['pid'] and intent['payload'] == payload(plan, row, 'commit'), 'Journal request differs from approved plan.')
        known = intent.get('known_transaction_ids')
        receipt_id = receipt.get('transaction_id') if receipt else None
        if known is not None:
            require(isinstance(known, list) and all(type(tid) is int for tid in known), 'Invalid transaction baseline in journal.')
            possible = [x for x in s['lines'] if x['lot_id'] == lid and x['product_id'] == row['pid']
                        and x['type'] == 'adjust' and x.get('adjust_reason') == row.get('reason', plan['reason'])
                        and x['transaction_id'] not in set(known)]
        else:
            # Old journals can use a valid receipt ID, never the local timestamp.
            require(type(receipt_id) is int and receipt_id > 0,
                    'Legacy write has no transaction baseline or receipt ID; manual reconciliation required.')
            possible = [x for x in s['lines'] if x['transaction_id'] == receipt_id
                        and x['lot_id'] == lid and x['product_id'] == row['pid']
                        and x['type'] == 'adjust' and x.get('adjust_reason') == row.get('reason', plan['reason'])]
        # Never accept two candidates, an amended/voided event, wrong actor,
        # different date/amount, or a receipt for some other transaction.
        require(len(possible) == 1, f'Lot {lid}: commit has {len(possible)} matching candidates; stop, never resubmit.')
        x = possible[0]
        require(x['effective_status'] == 'posted' and f.number(x['quantity_lb']) == f.number(str(float(row['change']))) and
                f.stamp(x['occurred_at']) == f.stamp(row.get('cutoff', plan['cutoff'].isoformat())) and x.get('operator_id') == plan['actor'] and
                not x.get('line_correction_at') and not x.get('transaction_correction_at'),
                f'Lot {lid}: posted adjustment has wrong actor/date/quantity/status or was corrected; stop.')
        require(s['transaction_line_counts'].get(str(x['transaction_id'])) == 1, 'Reset transaction has multiple lines or no complete readback; stop.')
        require(not receipt or not receipt.get('transaction_id') or receipt['transaction_id'] == x['transaction_id'], 'Journal receipt and ledger transaction differ.')
        completed[lid] = x
    return completed


def confirmation_rows(plan, s, completed):
    lots = {x['id']: x for x in s['lots']}
    lines = {(x['transaction_id'], x['line_id']): x for x in s['lines']}
    result = {}
    for r in plan['records']:
        if r['row_type'] != 'LATE_ENTRY_OWNER_REVIEW' or r['confirmation'] not in ('Y', 'N'):
            continue
        key = (int(r['transaction_id']), int(r['line_id']))
        x = lines.get(key)
        require(x is not None and f.late_fingerprint(x, plan['cutoff'], r['lot_code']) == r['fingerprint'],
                'An owner-confirmed late entry has changed; obtain a new preview/approval.')
        result[key] = (x, r['confirmation'])
    for x in completed.values():
        result[(x['transaction_id'], x['line_id'])] = (x, 'N')
    rows = []
    for x, answer in result.values():
        code = lots[x['lot_id']]['lot_code']
        rows.append(dict(zip(f.LATE_COLUMNS, [plan['cutoff'].isoformat(), x['transaction_id'], x['line_id'],
                    x['product_id'], code, x['occurred_at'], f.entered_at(x).isoformat(), x['quantity_lb'],
                    x['effective_status'], f.late_fingerprint(x, plan['cutoff'], code), answer, plan['owner'],
                    'Owner-signed preview; journal-verified reset entries classified N under signed approval.'])))
    return rows


def analyze_current(plan, s, completed, output):
    if plan.get("version") == 3:
        import fresh_start_v3 as v
        return v.analyze(s, plan["data"], {x["line_id"] for x in completed.values()})
    s = f.scoped_snapshot(s)
    path = output / 'verified-entry-confirmations.csv'
    f.write_csv(path, f.LATE_COLUMNS, confirmation_rows(plan, s, completed))
    return f.analyze(s, plan['count'], plan['cutoff'], path)


def preflight(plan, s, journal):
    if plan.get("version") == 3:
        return preflight_v3(plan, s, journal)
    require(not (plan['ids'] & f.EXCLUDED_PRODUCT_IDS), 'Owner-excluded product in plan; stop.')
    require(not plan['ids'] & f.BILLING_RESET_EXCLUDED_IDS and
            not any(p['id'] in plan['ids'] and f.reset_exclusion_reason(p) for p in s['products']),
            'Packaging/billing item is excluded from reset apply.')
    s = f.scoped_snapshot(s)
    unchanged(plan)
    completed = reconcile(plan, s, journal)
    server_time = f.stamp(s['snapshot_at'])
    require(plan['approval_date'] <= server_time.astimezone(f.PLANT).date(), 'Approval date is in the future according to FL server time.')
    own_lines = {x['line_id'] for x in completed.values()}
    products = {p['id']: p for p in s['products']}
    lots = {l['id']: l for l in s['lots'] if l['product_id'] in plan['ids']}
    expected = {r['lid']: r for r in plan['rows'] if r['lid']}
    require(set(lots) == set(expected), 'An approved product has a new, missing or omitted lot. Nothing further will post.')
    for lid, r in expected.items():
        p, lot = products.get(r['pid']), lots[lid]
        require(p and p.get('odoo_code') == r['SKU'] and p['name'] == r['name'] and
                (p['type'] == 'finished' or (p['type'] == 'batch' and 'granola' in p['name'].lower())) and
                lot['product_id'] == r['pid'] and lot['lot_code'] == r['lot_code'], 'Product/lot identity changed or is out of scope.')
        require(sum(x['lot_code'].lower() == lot['lot_code'].lower() for x in lots.values() if x['product_id'] == r['pid']) == 1,
                'Case-insensitive lot ambiguity; /adjust cannot safely choose this identity.')
        require([p2['id'] for p2 in s['catalog'] if p2.get('odoo_code') == r['SKU']] == [r['pid']],
                'SKU is not unique by exact SKU-field match across the catalog.')
        if r['change']:
            require(p.get('active') is True, f'Product {r["pid"]}: {f.INACTIVE_HOLD}.')
            require(not lot.get('merged_into_lot_id') and lot.get('status') in (None, 'active'), 'Merged/inactive lot cannot be adjusted.')
        current = sum((f.number(x['quantity_lb']) for x in s['lines'] if x['product_id'] == r['pid'] and x['lot_id'] == lid and x['effective_status'] == 'posted'), f.ZERO)
        want = f.number(r['current_lb']) + (r['change'] if lid in completed else 0)
        require(close(current, want), f'Product {r["pid"]}, lot {lid}: balance drift; approved {f.fmt(want)} lb, now {f.fmt(current)} lb. Stop this run.')
    for x in s['lines']:
        if x['product_id'] not in plan['ids'] or x['line_id'] in own_lines:
            continue
        require(f.entered_at(x) and f.entered_at(x) <= plan['at'], 'New or undated entry in an approved product; obtain a fresh preview.')
        for k in ('line_correction_at', 'transaction_correction_at'):
            require(not x.get(k) or f.stamp(x[k]) <= plan['at'], 'An approved product was corrected after preview; stop.')
    with tempfile.TemporaryDirectory(prefix='.apply-preflight-', dir=f.OUT) as tmp:
        a = analyze_current(plan, s, completed, Path(tmp))
    require(not a['general'] and not any(x['product_id'] in plan['ids'] for x in a['unassigned']), 'Scope or unassigned ledger exception; stop.')
    for pid in plan['ids']:
        require(a['group_status'][pid].startswith('READY'), f'Product {pid}: {a["group_status"][pid]}; stop.')
    current_rows = {(r['product_id'], r['lot_id']): r for r in a['rows']}
    for r in plan['rows']:
        actual = current_rows.get((r['pid'], r['lid']))
        require(actual is not None, 'Approved count row is missing on recheck.')
        for key in ('counted_lb', 'counted_qty', 'expected_current_lb'):
            want, got = r[key], actual[key]
            require((not want and got is None) or (want and got is not None and close(want, got)), 'Count, case weight or later movement differs from approved preview.')
        if r['change'] is not None:
            require(actual['adjustment_lb'] is not None and close(actual['adjustment_lb'], 0 if r['lid'] in completed else r['change']), 'Recomputed adjustment differs from approval.')
    return completed


def validate_response(result, plan, row, mode):
    require(isinstance(result, dict) and result.get('mode') == mode and result.get('product_id') == row['pid'] and
            result.get('lot_code') == row['lot_code'] and result.get('reason') == row.get('reason', plan['reason']) and
            f.number(result.get('adjustment_lb')) == f.number(str(float(row['change']))) and
            close(result.get('new_balance_lb'), row['planned_balance_lb']), 'API response differs from approved plan.')
    if mode == 'preview':
        require(close(result.get('current_quantity_lb'), row['current_lb']), 'API preview balance drift.')
    else:
        require(result.get('success') is True and type(result.get('transaction_id')) is int and result['transaction_id'] > 0,
                'Commit did not return a valid transaction receipt.')
        require('operator_id' not in result or result['operator_id'] == plan['actor'], 'Commit returned the wrong actor.')


def run(preview, approval, count, apply=False, backend=None, input_fn=input, target=None):
    require(Path(__file__).resolve().parent == f.OUT, 'Use only the authorized fresh-start folder.')
    if f.local_path(preview).suffix == '.json':
        return run_v3(load_v3_plan(preview, approval, count, target), apply, backend, input_fn)
    plan = load_plan(preview, approval, count)
    for pid in sorted(plan['held_inactive']):
        print(f'Product {pid}: {f.INACTIVE_HOLD}; excluded from approved groups.')
    backend = backend or (LiveBackendV3() if plan.get("version") == 3 else LiveBackend())
    # One global lock and a deterministic journal per preview hash. Copies of the
    # same preview resume the same journal; separate approved groups can coexist.
    fd = os.open(LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another reset process holds the lock; stop.') from None
        journal = Journal(JOURNAL_ROOT / plan['sha'] / 'journal.jsonl', plan)
        completed = preflight(plan, backend.snapshot(), journal)
        pending = [r for r in plan['rows'] if r['change'] and r['lid'] not in completed]
        key = read_key()
        require(backend.actor(key) == plan['actor'], 'Key is not the owner-approved named actor; shared/master keys are refused.')
        print(f'Approved preview SHA-256: {plan["sha"]}')
        print(f'Journal (preserve for reruns): {journal.path}')
        print(f'{len(completed)} posted adjustments reconciled; {len(pending)} adjustments remaining.')
        for r in pending:
            print(f'Product {r["pid"]} / lot {r["lot_code"]}: {f.fmt(r["current_lb"])} + ({f.fmt(r["change"])}) = {f.fmt(r["planned_balance_lb"])} {r.get("ledger_unit", "lb")}; cutoff {r.get("cutoff", plan["cutoff"].isoformat())}; {r.get("reason", plan["reason"])}')
        if not apply:
            print('DRY RUN ONLY: no API calls, no inventory changes. Named actor verified by read-only lookup.')
            return 0
        phrase = f'APPLY OPENING BALANCE {plan["cutoff"].astimezone(f.PLANT).date()} {plan["sha"]}'
        print('Coordinate a brief quiet window for ONLY the approved products during posting/readback; production elsewhere can continue. Type exactly: ' + phrase)
        require(input_fn('Confirmation: ') == phrase, 'Confirmation did not match; no API calls were made.')
        unchanged(plan)
        preflight(plan, backend.snapshot(), journal)
        if pending:
            def identity(result):
                require(result.get('key_kind') == 'actor' and isinstance(result.get('actor'), dict) and
                        result['actor'].get('name') == plan['actor'], 'Shared/master or wrong named key refused.')
            api_call(backend, key, journal, 'identity', None, None, identity)
            # Preview every remaining lot before the first commit in this run.
            for r in pending:
                api_call(backend, key, journal, 'preview', r, payload(plan, r, 'preview'),
                         lambda result, r=r: validate_response(result, plan, r, 'preview'))
            for r in pending:
                before = backend.snapshot()
                completed = preflight(plan, before, journal)
                require(r['lid'] not in completed, 'Unexpected concurrent posting; stop.')
                api_call(backend, key, journal, 'commit', r, payload(plan, r, 'commit'),
                         lambda result, r=r: validate_response(result, plan, r, 'commit'), before=before)
                # First post AND every later post: exact transaction + named actor.
                completed = preflight(plan, backend.snapshot(), journal)
                print(f'Confirmed TX{completed[r["lid"]]["transaction_id"]}: named actor and lot balance verified.')
        del key
        s = backend.snapshot()
        completed = preflight(plan, s, journal)
        output = RESULTS / plan['sha']
        output.mkdir(parents=True, exist_ok=True)
        a = analyze_current(plan, s, completed, output)
        # Run the existing verifier, including its full-scope/incomplete verdict.
        if plan.get('version') == 3:
            import fresh_start_v3 as v
            return v.report(a, output, verify=True)
        return write_verification(a, output)


APPROVAL_FIELDS_V3 = APPROVAL_FIELDS + ('Owner approves held entries', 'Extra-kosher lots')
HOLD_KINDS = {'PHOTO_REQUIRED': 'LARGE_CORRECTION', 'KOSHER_ATTESTATION_REQUIRED': 'KOSHER_ATTESTATION'}
RECEIPT = re.compile(r'(ADJ|FND)-\d{6}-\d{3,}')
TICKET = re.compile(r'wt_[A-Za-z0-9_-]{20,100}')
ERROR_CODE = re.compile(r'[A-Z][A-Z0-9_]{2,60}')


def lot_label(r):
    return f'{r["pid"]}:{r["lot_code"]}'


def load_v3_plan(preview, approval, count, target=None):
    import fresh_start_v3 as v
    preview, approval, count = map(f.local_path, (preview, approval, count))
    raw, signed = preview.read_bytes(), approval.read_bytes()
    a = json.loads(raw, parse_float=f.D)
    require(a.get('version') == 3, 'Unsupported live-count preview version.')
    require(a.get('reset_scope_policy') == f.RESET_SCOPE_POLICY,
            'Reset scope decision changed; regenerate and reapprove preview.')
    plan_target = a.get('target', 'production')
    require(target is None or target == plan_target, f'Preview was made for {plan_target}, not {target}.')
    f.target_config(plan_target)
    for p in a['products'].values():
        require(not f.reset_exclusion_reason(p), 'Packaging/billing item is excluded from reset approval.')
    for r in a['rows']:
        require(not f.reset_exclusion_reason(dict(id=r['product_id'],type=r['product_type'])),
                'Packaging/billing item is excluded from reset approval.')
    fields = {}
    for line in signed.decode().splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            if key in APPROVAL_FIELDS_V3:
                require(key not in fields, 'Duplicate approval field.')
                fields[key] = value.strip()
    require(set(fields) == set(APPROVAL_FIELDS_V3) and all(fields.values()), 'Approval is incomplete.')
    require(fields['Preview file'] == preview.name and fields['SHA-256'].lower() == digest(raw), 'Signed preview hash/filename mismatch.')
    require(fields['Signature'] == fields['Owner name'], 'Owner signature mismatch.')
    require(fields['Named actor'] not in ('legacy-shared-key','shared','master'), 'Named personal actor required.')
    require(fields['Opening entry classification'] == 'N', 'Administrative opening classification must be N.')
    require(fields['Owner approves held entries'] in ('Y', 'N'), 'Owner approves held entries must be Y or N.')
    require(re.fullmatch(r'[1-9]\d*(?:\s*,\s*[1-9]\d*)*',fields['Approved product IDs']), 'Invalid approved product IDs.')
    ids = [int(x.strip()) for x in fields['Approved product IDs'].split(',')]
    require(len(ids) == len(set(ids)), 'Duplicate approved product ID.')
    require(not set(ids) & f.BILLING_RESET_EXCLUDED_IDS, 'Billing item is excluded from reset approval.')
    src = a['sources']
    require(f.local_path(src['count']['path']) == count and digest(count.read_bytes()) == a['input_sha256'], 'Original count CSV differs.')
    for item in src.values():
        require(digest(f.local_path(item['path']).read_bytes()) == item['sha256'], 'A preview source changed; regenerate and reapprove.')
    data = v.load_inputs(count,src['coverage']['path'],src['moves']['path'],src.get('review',{}).get('path'))
    require(v.canonical(data) == v.canonical(a['data']), 'Embedded count/review differs from original files.')
    at = f.stamp(a['snapshot_at']); selected=[]; seen=set()
    for r0 in a['rows']:
        r=dict(r0);pid=r['product_id'];change=f.number(r['adjustment_lb']) if r['adjustment_lb'] is not None else None
        require(not change or pid in ids, 'Every proposed change must be approved; reissue a scoped preview.')
        if pid not in ids: continue
        require(r['status']=='READY' and not r['holds'], 'Held lot/product is not approvable.')
        new = bool(r.get('new_lot'))
        lid = r['lot_key'] if new else r['lot_id']
        require(lid is not None and lid not in seen,'Duplicate lot proposal.');seen.add(lid)
        cut=f.stamp(r['cutoff']);require(cut<=at,'Count cutoff is after preview.')
        require(r['backfill'] is True,'V3 must disclose intentional count-time posting.')
        original=f.number(r['candidate_adjustment_lb']);rounded=f.round_adjustment(original)
        require(change==rounded==f.number(r['rounded_adjustment_lb']) and
                original==f.number(r['counted_lb'])-f.number(r['adjustment_basis_lb']) and
                f.number(r['rounding_delta_lb'])==rounded-original and
                f.number(r['planned_balance_lb'])==f.number(r['current_lb'])+rounded and
                close(r['planned_balance_lb'],r['expected_current_lb']), 'V3 count arithmetic mismatch.')
        require(f.number(r['expected_current_lb'])>=0 and math.isfinite(float(change)) and close(str(float(change)),change),'Invalid target/quantity.')
        require(not change or r['product_active'] is True,'Inactive product cannot be adjusted.')
        require(r.get('action') == ('found' if new else 'adjust'), 'Preview action does not match the lot.')
        if new:
            # P1.21: only extra-kosher stock may arrive as a new lot, only positive, from zero.
            require(r.get('kosher_tier') == f.EXTRA_KOSHER and change and change > 0 and f.number(r['current_lb']) == 0
                    and r['ledger_unit'] == 'lb', 'A new lot is only an extra-kosher load in lb.')
            require(r['lot_code'] == r['lot_code'].strip().upper(), 'New lot code is not in FL canonical form.')
        r.update(pid=pid,lid=lid,change=change)
        selected.append(r)
    # A coverage-certified product with no counted or on-hand lot has nothing to post, but
    # its group still signs for it, so the whole-reset proof can bind every scope product.
    zero = v.zero_stock_products(a)
    require({r['pid'] for r in selected} <= set(ids) <= {r['pid'] for r in selected} | zero, 'Approved product absent from lot preview.')
    require(not a['general'], 'Unidentified catalog items or general exceptions still require review.')
    needs = [r for r in selected if r['change'] and r.get('owner_approvals')]
    require(not needs or fields['Owner approves held entries'] == 'Y',
            'This plan has entries FL holds for the owner (over 500 lb or extra-kosher); sign "Owner approves held entries: Y" or leave those products out.')
    kosher = {lot_label(r) for r in selected if r.get('new_lot') and r['change']}
    listed = set() if fields['Extra-kosher lots'].lower() == 'none' else {x.strip() for x in fields['Extra-kosher lots'].split(',') if x.strip()}
    require(listed == kosher, 'Extra-kosher lots must list exactly the new lots in this plan as PRODUCT:LOT (or none).')
    return dict(version=3,preview=preview,approval=approval,count=count,sha=digest(raw),approval_sha=digest(signed),
                count_sha=digest(count.read_bytes()),actor=fields['Named actor'],owner=fields['Owner name'],
                approval_date=date.fromisoformat(fields['Date']),cutoff=max((f.stamp(r['cutoff']) for r in selected), default=at),
                at=at,reason='v3 per-lot count reasons',ids=set(ids),rows=selected,records=[],held_inactive=set(),
                v3=a,data=data,sources=src,target=plan_target,v3_lines=a.get('ledger_lines', []),approve_held=fields['Owner approves held entries']=='Y',
                area_group=a.get('area_group'))


def ticket_payload(r):
    import fresh_start_v3 as v
    return v.ticket_payload(r)


def reconcile_v3(plan, s, journal):
    """Completed lot -> its one posted line, proved by ticket (shared with verify)."""
    import fresh_start_v3 as v
    try:
        return v.reconcile_tickets({r['lid']: r for r in plan['rows'] if r['change']}, s, journal.attempts(), plan['actor'],
                                   plan['approve_held'])
    except ValueError as error:
        raise RuntimeError(str(error)) from None


def preflight_v3(plan,s,journal):
    import fresh_start_v3 as v
    unchanged(plan)
    require(plan['v3'].get('reset_scope_policy') == f.RESET_SCOPE_POLICY,
            'Reset scope decision changed; regenerate and reapprove preview.')
    require(s.get('target', plan['target']) == plan['target'], 'Snapshot is from another target.')
    live_products={p['id']:p for p in s['products']}
    for pid in plan['ids']:
        require(pid in live_products and not f.reset_exclusion_reason(live_products[pid]),
                'Packaging/billing item or missing product is excluded from reset apply.')
    completed, held, uncertain = reconcile_v3(plan, s, journal)
    own={x['line_id'] for x in completed.values()}
    own_lots={x['lot_id'] for lid, x in completed.items() if not isinstance(lid, int)}
    require(plan['approval_date']<=f.stamp(s['snapshot_at']).astimezone(f.PLANT).date(),'Approval date is in the future.')
    products={p['id']:p for p in s['products']};lots={l['id']:l for l in s['lots']}
    expected={r['lid'] for r in plan['rows'] if not r.get('new_lot')} | own_lots
    require({l['id'] for l in s['lots'] if l['product_id'] in plan['ids']}==expected,'New/missing/omitted lot in approved scope.')
    for pid in plan['ids']:
        require(v.baseline(s,pid,own,own_lots)==plan['v3']['baseline'][str(pid)], 'Ledger/catalog changed since approval; refresh movement review before any further write.')
    open_short = {}
    for e in s.get('shortages', []):
        if e['status'] in ('open', 'escalated'):
            open_short.setdefault(e['lot_id'], []).append(e['id'])
    for r in plan['rows']:
        p=products.get(r['pid'])
        require(p and p['name']==r['name'] and (p.get('odoo_code') or '')==r['SKU'] and p['type']==r['product_type'] and v.native(p)==r['ledger_unit'], 'Product identity/unit changed.')
        if r.get('new_lot'):
            require((p.get('kosher_tier') or 'regular') == f.EXTRA_KOSHER, 'Product is no longer extra-kosher.')
            same = [l for l in lots.values() if l['product_id'] == r['pid'] and l['lot_code'].casefold() == r['lot_code'].casefold()]
            require(len(same) == (1 if r['lid'] in completed else 0), 'New lot code is already in use; refresh the preview.')
            require(not any(e['product_id'] == r['pid'] and e['status'] in ('open', 'escalated') for e in s.get('shortages', [])),
                    'Open shortage on an extra-kosher product; FL refuses the found load.')
            continue
        lot=lots.get(r['lid'])
        require(lot and lot['product_id']==r['pid'] and lot['lot_code']==r['lot_code'],'Lot identity changed.')
        if r['change']:
            require(p.get('active') is True and not p.get('is_service'), 'Inactive/service item cannot be adjusted.')
            require(lot.get('status') in ('active',None) and not lot.get('merged_into_lot_id'),'Merged/inactive lot cannot be adjusted.')
            require(sum(l['product_id']==r['pid'] and l['lot_code'].casefold()==r['lot_code'].casefold() for l in lots.values())==1,'Case-insensitive lot ambiguity.')
            if r['lid'] not in completed:
                require(sorted(open_short.get(r['lid'], [])) == sorted(r.get('resolves_shortage_ids') or []),
                        f'Lot {r["lot_code"]}: open shortages changed since the preview; refresh it.')
    a=v.analyze(s,plan['data'],own)
    actual={(r['product_id'],r['lot_code']):r for r in a['rows']}
    require(not a['general'],'General reconciliation exception.')
    for r in plan['rows']:
        x=actual.get((r['pid'],r['lot_code']))
        require(x and x['status']=='READY','Count/movement review is held on recheck.')
        for key in ('physical_count_native','internal_move_correction_native','post_count_movement_native','expected_current_lb'):
            require(close(r[key],x[key]),'Recomputed count/movement differs from signed preview.')
        for key in ('row_cutoffs','cutoff','estimated','reason','ledger_unit','backfill'):
            require(r[key]==x[key],'Count cutoff, estimate or posting metadata changed.')
        done = r['lid'] in completed
        require(close(x['adjustment_lb'],0 if done else r['change']),'Recomputed adjustment differs from approval.')
        require(close(x['current_lb'],f.number(r['current_lb'])+(r['change'] if done else 0)),'Balance drift.')
    return completed, held, uncertain


class LiveBackendV3(LiveBackend):
    """Ticket-era client for one explicit target. Only call() can write."""
    def __init__(self, target='production'):
        self.config = f.target_config(target)

    def read_json(self, sql, variables=()):
        require(sql == ACTOR_SQL and len(variables) == 2 and variables[0] == '-v' and
                re.fullmatch(r'actor_hash=[0-9a-f]{64}', variables[1]), 'Only the fixed actor lookup is allowed.')
        try:
            env = dict(os.environ, DATABASE_URL=f.target_database_url(self.config['name']), PGCONNECT_TIMEOUT='15')
            env.pop('PGOPTIONS', None)
            result = subprocess.run([str(f.WRAPPER), '-X', '-qAt', '-v', 'ON_ERROR_STOP=1', *variables],
                                    input=sql, text=True, capture_output=True, cwd=f.ROOT, env=env, timeout=60)
            require(result.returncode == 0, 'Read-only query failed; details suppressed.')
            return json.loads(result.stdout, parse_float=f.D)
        except (OSError, ValueError, subprocess.SubprocessError):
            raise RuntimeError('Read-only database lookup failed; details suppressed.') from None

    def snapshot(self):
        import fresh_start_v3 as v
        return v.snapshot(self.config['name'])

    def call(self, key, route, body):
        """(HTTP status, parsed JSON) for 200/202 and structured 4xx; anything else raises."""
        require(route in ('/auth/whoami', '/adjust/prepare', '/inventory/found/prepare') or
                re.fullmatch(r'/tickets/wt_[A-Za-z0-9_-]{20,100}/commit', route) or
                re.fullmatch(r'/exceptions/[1-9]\d*/approve', route), 'Unsupported API route.')
        req = Request(self.config['api'] + route, data=json.dumps(body).encode() if body is not None else None,
                      headers={'X-API-Key': key, 'Content-Type': 'application/json'},
                      method='POST' if body is not None else 'GET')
        from urllib.error import HTTPError
        try:
            with build_opener(ProxyHandler({}), NoRedirect()).open(req, timeout=45) as response:
                require(response.status in (200, 202), 'Uncertain API response.')
                result = json.load(response, parse_float=f.D)
                require(isinstance(result, dict), 'Malformed API response.')
                return response.status, result
        except HTTPError as error:
            if 400 <= error.code < 500:
                try:
                    return error.code, json.load(error, parse_float=f.D)
                except ValueError:
                    return error.code, {}
            raise RuntimeError('API outcome failed or uncertain. Stopped; no automatic retry; details suppressed.') from None
        except Exception:
            raise RuntimeError('API outcome failed or uncertain. Stopped; no automatic retry; details suppressed.') from None


def error_code(body):
    detail = body.get('detail') if isinstance(body, dict) else None
    code = detail.get('error_code') if isinstance(detail, dict) else body.get('error_code') if isinstance(body, dict) else None
    return code if isinstance(code, str) and ERROR_CODE.fullmatch(code) else 'UNKNOWN'


def signed_duplicates(plan, r, body):
    """POSSIBLE_DUPLICATE warnings the signed preview already answers: the matching
    posting is a ledger line of this product that the preview was computed from
    (its baseline is re-proved unchanged before every write). Anything else stops."""
    known = {x['transaction_id'] for x in plan['v3_lines'] if x['product_id'] == r['pid']}
    acks = []
    for w in body.get('warnings') or []:
        if not w.get('requires_ack'):
            continue
        refs = w.get('refs') or {}
        require(w.get('code') == 'POSSIBLE_DUPLICATE' and refs.get('transaction_id') in known,
                f'Prepared draft needs acknowledgement ({w.get("code")}) for a posting the signed preview does not show; '
                'the executor never acknowledges that for you. Stop and review.')
        acks.append('POSSIBLE_DUPLICATE')
    return sorted(set(acks))


def check_draft(r, status, body):
    require(status == 200, f'Prepare refused ({error_code(body)}); nothing posted. Refresh the preview.')
    d = body.get('draft') or {}
    require(body.get('can_commit') is True and not body.get('blockers') and TICKET.fullmatch(str(body.get('ticket', ''))) and
            type(body.get('ticket_id')) is int and isinstance(body.get('payload_hash'), str),
            'Prepared draft is blocked or malformed; nothing posted.')
    if r.get('new_lot'):
        require(d.get('product_id') == r['pid'] and d.get('lot_code') == r['lot_code'] and d.get('lot_exists') is False and
                close(d.get('quantity'), r['change']) and d.get('kosher_tier') == f.EXTRA_KOSHER and
                (d.get('kosher_approval') or {}).get('required') is True, 'Found draft differs from the approved new lot.')
    else:
        require(d.get('product_id') == r['pid'] and d.get('lot_id') == r['lid'] and d.get('lot_code') == r['lot_code'] and
                close(d.get('delta_lb'), r['change']) and d.get('reason_code') == 'physical_count' and
                close(d.get('current_quantity_lb'), r['current_lb']) and
                sorted(d.get('resolves_shortage_ids') or []) == sorted(r.get('resolves_shortage_ids') or []) and
                bool((d.get('correction_review') or {}).get('photo_required')) == ('LARGE_CORRECTION' in (r.get('owner_approvals') or [])),
                'Adjust draft differs from the approved plan (balance, shortage or hold prediction).')
    return body


def check_posted(r, body):
    require(type(body.get('transaction_id')) is int and body['transaction_id'] > 0 and
            RECEIPT.fullmatch(str(body.get('receipt_number', ''))) and body.get('reason_code') == 'physical_count',
            'Commit did not return a valid receipt.')
    if r.get('new_lot'):
        require(body['receipt_number'].startswith('FND-') and body.get('lot_code') == r['lot_code'] and
                close(body.get('quantity'), r['change']) and body.get('kosher_tier') == f.EXTRA_KOSHER and
                body.get('kosher_attested_by_actor_id'), 'Found receipt differs from the approved new lot.')
    else:
        require(body['receipt_number'].startswith('ADJ-') and body.get('lot_code') == r['lot_code'] and
                close(body.get('adjustment_lb'), r['change']) and close(body.get('new_balance_lb'), r['planned_balance_lb']),
                'Adjust receipt differs from the approved plan.')
        closed = sorted(c.get('exception_id') for c in body.get('shortages_closed') or [])
        require(closed == sorted(r.get('resolves_shortage_ids') or []), 'Shortages closed differ from the approved plan.')
    return body


def post_lot(plan, backend, key, journal, r, before, resume_ticket=None, resume_commit=None):
    """Prepare → commit (→ owner approval of a signed hold). Every write has a durable intent first.
    `resume_commit` = the journal intent of a commit whose outcome is uncertain: that SAME
    ticket is committed again (FL posts it once or replays the saved receipt); a second
    ticket is never prepared while the first could still land (Codex PR #108 review)."""
    if resume_ticket is None:
        if resume_commit is None:
            status, draft = backend.call(key, '/inventory/found/prepare' if r.get('new_lot') else '/adjust/prepare', ticket_payload(r))
            check_draft(r, status, draft)
            acks = signed_duplicates(plan, r, draft)
            ticket, ticket_id, payload_hash = draft['ticket'], draft['ticket_id'], draft['payload_hash']
        else:
            ticket, ticket_id, payload_hash = resume_commit.get('ticket'), resume_commit['ticket_id'], resume_commit['payload_hash']
            acks = resume_commit['acknowledged_warnings']
            require(isinstance(ticket, str) and TICKET.fullmatch(ticket),
                    f'Lot {r["lot_code"]}: ticket {ticket_id} may still post and this journal cannot re-commit it; '
                    'manual reconciliation required (never prepare a second ticket).')
        known = sorted({x['transaction_id'] for x in before['lines'] if x['product_id'] == r['pid']})
        journal.start()
        intent = journal.append(kind='intent', phase='commit', lot_id=r['lid'], product_id=r['pid'], ticket_id=ticket_id,
                                ticket=ticket, payload=ticket_payload(r), payload_hash=payload_hash, acknowledged_warnings=acks,
                                known_transaction_ids=known, server_snapshot_at=before['snapshot_at'],
                                retry_of=resume_commit['seq'] if resume_commit else None)
        try:
            status, body = backend.call(key, '/tickets/' + ticket + '/commit',
                                        {'payload_hash': payload_hash, 'acknowledged_warnings': acks})
        except Exception:
            journal.append(kind='receipt', intent=intent['seq'], outcome='uncertain', transaction_id=None)
            raise RuntimeError('Write request failed or uncertain. Stopped; preserve journal.jsonl; a rerun re-commits the same ticket.') from None
        if status == 200:
            try:
                check_posted(r, body)
            except RuntimeError:
                journal.append(kind='receipt', intent=intent['seq'], outcome='uncertain', transaction_id=None)
                raise
            journal.append(kind='receipt', intent=intent['seq'], outcome='ok', transaction_id=body['transaction_id'],
                           receipt_number=body['receipt_number'])
            return body
        code = error_code(body)
        if status != 202 or not body.get('held') or HOLD_KINDS.get(body.get('error_code')) is None:
            journal.append(kind='receipt', intent=intent['seq'], outcome='refused', transaction_id=None, error_code=code)
            raise RuntimeError(f'Commit refused ({code}); nothing posted for this lot. Refresh the preview.')
        kind = HOLD_KINDS[body['error_code']]
        exception_id = body.get('exception_id')
        journal.append(kind='receipt', intent=intent['seq'], outcome='held', transaction_id=None, hold=kind,
                       exception_id=exception_id if type(exception_id) is int else None)
    else:
        ticket_id = resume_ticket
        held = [e for e in before.get('held_exceptions', []) if e['ticket_id'] == ticket_id and e['status'] in ('open', 'escalated')]
        require(len(held) == 1, 'Held ticket has no single open owner exception; manual review.')
        kind, exception_id = held[0]['kind'], held[0]['id']
    require(kind in (r.get('owner_approvals') or []) and plan['approve_held'],
            f'FL held this lot for {kind}, which the signed approval does not cover; it stays held for the owner.')
    require(type(exception_id) is int and exception_id > 0, 'Held response has no exception to approve.')
    intent = journal.append(kind='intent', phase='approve', lot_id=r['lid'], product_id=r['pid'], ticket_id=ticket_id,
                            exception_id=exception_id, hold=kind)
    note = f'Cutover physical count reset approved by {plan["owner"]} in signed preview SHA-256 {plan["sha"]}'
    try:
        status, body = backend.call(key, f'/exceptions/{exception_id}/approve', {'note': note})
    except Exception:
        journal.append(kind='receipt', intent=intent['seq'], outcome='uncertain', transaction_id=None)
        raise RuntimeError('Owner approval failed or uncertain. Stopped; preserve journal.jsonl; a rerun proves it by ticket.') from None
    if status != 200:
        journal.append(kind='receipt', intent=intent['seq'], outcome='refused', transaction_id=None, error_code=error_code(body))
        raise RuntimeError(f'Owner approval refused ({error_code(body)}); the entry stays held.')
    try:
        check_posted(r, body)
    except RuntimeError:
        journal.append(kind='receipt', intent=intent['seq'], outcome='uncertain', transaction_id=None)
        raise
    journal.append(kind='receipt', intent=intent['seq'], outcome='ok', transaction_id=body['transaction_id'],
                   receipt_number=body['receipt_number'])
    return body


READBACK_COLUMNS = ['product_id', 'name', 'lot_code', 'new_lot', 'change', 'receipt_number', 'transaction_id', 'balance_now',
                    'planned', 'owner_holds', 'shortages_closed', 'kosher_tier', 'status']


def readback_rows(plan, s, completed):
    lots = {l['id']: l for l in s['lots']}
    balance = {}
    for x in s['lines']:
        if x['effective_status'] == 'posted':
            balance[x['lot_id']] = balance.get(x['lot_id'], f.ZERO) + f.number(x['quantity_lb'])
    rows = []
    for r in plan['rows']:
        x = completed.get(r['lid'])
        lot_id = x['lot_id'] if x else (r['lid'] if not r.get('new_lot') else None)
        rows.append(dict(product_id=r['pid'], name=r['name'], lot_code=r['lot_code'], new_lot=bool(r.get('new_lot')),
                         change=r['change'], receipt_number=x['receipt_number'] if x else '', transaction_id=x['transaction_id'] if x else '',
                         balance_now=balance.get(lot_id, f.ZERO) if lot_id else f.ZERO, planned=r['planned_balance_lb'],
                         owner_holds=','.join(r.get('owner_approvals') or []),
                         shortages_closed=','.join(map(str, r.get('resolves_shortage_ids') or [])) if x else '',
                         kosher_tier=lots.get(lot_id, {}).get('kosher_tier', '') if lot_id else '',
                         status='POSTED' if x else ('NO CHANGE' if not r['change'] else 'NOT POSTED')))
    return rows


def run_v3(plan, apply=False, backend=None, input_fn=input):
    import time
    config = f.target_config(plan['target'])
    backend = backend or LiveBackendV3(plan['target'])
    journal_root = f.OUT / config['state'] if JOURNAL_ROOT == f.OUT / 'apply-state' else JOURNAL_ROOT
    results_root = f.OUT / config['results'] if RESULTS == f.OUT / 'apply-results' else RESULTS
    key_file = config['key'] if KEY_FILE == TARGET_DEFAULT_KEY else KEY_FILE
    fd = os.open(LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another reset process holds the lock; stop.') from None
        journal = Journal(journal_root / plan['sha'] / 'journal.jsonl', plan)
        timings = dict(started=time.monotonic())
        completed, held, uncertain = preflight_v3(plan, backend.snapshot(), journal)
        pending = [r for r in plan['rows'] if r['change'] and r['lid'] not in completed]
        key = read_key(key_file)
        require(backend.actor(key) == plan['actor'], 'Key is not the owner-approved named actor; shared/master keys are refused.')
        print(f'Target: {plan["target"].upper()}  ({config["api"]})')
        if plan.get('area_group'):
            print('Area group: ' + ', '.join(plan['area_group']))
        print(f'Approved preview SHA-256: {plan["sha"]}')
        print(f'Journal (preserve for reruns): {journal.path}')
        print(f'{len(completed)} posted reset entries proved; {len(pending)} remaining ({len(held)} held for the owner'
              f', {len(uncertain)} with an uncertain commit to re-send on the same ticket).')
        for r in pending:
            extra = ('; NEW extra-kosher lot via found ticket' if r.get('new_lot') else '') + \
                    (f'; owner hold: {",".join(r["owner_approvals"])}' if r.get('owner_approvals') else '') + \
                    (f'; closes shortage(s) {",".join(map(str, r["resolves_shortage_ids"]))}' if r.get('resolves_shortage_ids') else '')
            print(f'Product {r["pid"]} / lot {r["lot_code"]}: {f.fmt(r["current_lb"])} + ({f.fmt(r["change"])}) = {f.fmt(r["planned_balance_lb"])} {r.get("ledger_unit", "lb")}; cutoff {r["cutoff"]}{extra}')
        if not apply:
            print('DRY RUN ONLY: no API calls, no inventory changes. Named actor verified by read-only lookup.')
            return 0
        phrase = f'APPLY OPENING BALANCE {plan["cutoff"].astimezone(f.PLANT).date()} {plan["sha"]}'
        print('PAUSE POSTING in this area now. Type exactly: ' + phrase)
        require(input_fn('Confirmation: ') == phrase, 'Confirmation did not match; no API calls were made.')
        timings['confirmed'] = time.monotonic()
        unchanged(plan)
        completed, held, uncertain = preflight_v3(plan, backend.snapshot(), journal)
        if pending:
            status, who = backend.call(key, '/auth/whoami', None)
            require(status == 200 and who.get('key_kind') == 'actor' and isinstance(who.get('actor'), dict) and
                    who['actor'].get('name') == plan['actor'], 'Shared/master or wrong named key refused.')
            before = backend.snapshot()
            for r in plan['rows']:
                if not r['change'] or r['lid'] in completed:
                    continue
                # The previous lot's read-back snapshot (taken just now) is this lot's
                # pre-write state; it is re-proved here before anything is sent.
                completed, held, uncertain = preflight_v3(plan, before, journal)
                if r['lid'] in completed:
                    continue
                lot_started = time.monotonic()
                body = post_lot(plan, backend, key, journal, r, before, held.get(r['lid']), uncertain.get(r['lid']))
                before = backend.snapshot()
                completed, held, uncertain = preflight_v3(plan, before, journal)
                require(r['lid'] in completed, 'Posted entry did not read back; stop.')
                timings.setdefault('per_lot', []).append(round(time.monotonic() - lot_started, 1))
                print(f'Read back {body["receipt_number"]}: product {r["pid"]} lot {r["lot_code"]} TX{completed[r["lid"]]["transaction_id"]} by {plan["actor"]}.')
        timings['posted'] = time.monotonic()
        del key
        s = backend.snapshot()
        completed, held, uncertain = preflight_v3(plan, s, journal)
        output = results_root / plan['sha']
        output.mkdir(parents=True, exist_ok=True)
        rows = readback_rows(plan, s, completed)
        with (output / 'readback.csv').open('w', newline='') as stream:
            # Fixed columns: a zero-stock-only group has no rows yet must still verify (Codex PR #120 re-check).
            writer = csv.DictWriter(stream, fieldnames=READBACK_COLUMNS)
            writer.writeheader()
            writer.writerows({k: (f.fmt(v) if isinstance(v, f.D) else v) for k, v in row.items()} for row in rows)
        print('READ-BACK (check each receipt against the count sheet):')
        for row in rows:
            if row['status'] != 'NO CHANGE':
                print(f'  {row["receipt_number"] or "-":>16}  {row["product_id"]} / {row["lot_code"]}  change {f.fmt(row["change"])}  balance now {f.fmt(row["balance_now"])}  {row["status"]}')
        import fresh_start_v3 as v
        a = v.analyze(s, plan['data'], {x['line_id'] for x in completed.values()}, plan.get('area_group'))
        timings['verified'] = time.monotonic()
        spent = {k: round(timings[k] - timings['started'], 1) for k in timings if k not in ('started', 'per_lot')}
        spent['per_lot_s'] = timings.get('per_lot', [])
        (output / 'timings.json').write_text(json.dumps(dict(spent, posts=sum(r['status'] == 'POSTED' for r in rows)), indent=2) + '\n')
        print(f'Executor seconds from start: {spent}')
        return v.report(a, output, verify=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('preview_csv')
    parser.add_argument('--approval', default=str(f.OUT / 'approval.md'))
    parser.add_argument('--count-csv', required=True, help='Original filled count CSV; must match the preview input SHA-256.')
    parser.add_argument('--apply', action='store_true', help='Enable live API requests only after the exact typed phrase.')
    parser.add_argument('--target', choices=sorted(f.TARGETS), help='v3: must match the preview target (production or staging).')
    args = parser.parse_args()
    if args.apply:
        require(sys.stdin.isatty(), '--apply requires an interactive terminal; piped confirmation is refused.')
    if f.local_path(args.preview_csv).suffix == '.json':
        require(args.target, 'v3: name the target explicitly with --target production|staging.')
    return run(args.preview_csv, args.approval, args.count_csv, args.apply, target=args.target)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, KeyError, TypeError, EOFError) as error:
        # Errors intentionally carry no key, HTTP response, or DB credentials.
        print('STOP: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
