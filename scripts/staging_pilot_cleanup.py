#!/usr/bin/env python3
"""Plan a staging pilot cleanup; GETs and transaction-scoped SELECTs by default.

Never imports the app (startup hooks can write), runs migrations, seeds data, or
writes SQL. --apply is an explicit, separate invocation pinned to a reviewed
snapshot hash. Only supported ticket flows execute; blocked items remain listed.
Credentials: ~/Documents/fl-secrets, or FastAPI-staging's Railway environment.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import sys
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo

import httpx
import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from staging_safety import assert_staging_database, PRODUCTION_DATABASE_HOST
from exceptions_enforcement import correction_review

BASE = 'https://fastapi-staging-production-dd7b.up.railway.app'
DB_HOST = 'aws-0-us-east-1.pooler.supabase.com'
DB_REF = 'jygmyvxnxdjiiilhxseq'
REASON = 'STAGING PILOT CLEANUP'
# Reconciled individually against the October 15 read-only staging audit.
# Protect by immutable ID as well as OPEN prefix, even if a label changes later.
PROTECTED_EXCEPTION_IDS = frozenset({
    71, 74, 77, 79, 81, 83, 88, 91, 93, 95, 97, 99, 100, 102, 103, 105,
    106, 107, 109, 111, 113, 115, 116, 118, 120, 122, 124, 126, 128, 130,
    132, 134, 136, 166, 169, 170, 171, 172, 173, 174, 176, 177, 178, 215,
    216, 217, 218, 220, 233, 235, 236, 237, 242, 243, 244, 245,
})
SECRETS = Path.home() / 'Documents/fl-secrets'
DEFAULT_PLAN = Path.home() / 'dev/staging-cleanup-plan-2026-10-15.md'
TEST_NAME = re.compile(r'^(?:STG|STAGING)(?:[-_ ]|$)', re.I)
POSTED = '''(SELECT tl.* FROM public.ledger_current_transaction_lines tl
 JOIN public.transactions t ON t.id=tl.transaction_id
 JOIN public.ledger_current_transactions ct ON ct.id=t.id
 WHERE ct.effective_status='posted')'''


class CleanupError(RuntimeError):
    """Only messages authored here, never raw credential-bearing errors."""


def require(value, message):
    if not value:
        raise CleanupError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


def digest(value):
    return sha256(encoded(value).encode()).hexdigest()


def secret_file(path):
    path = Path(path).expanduser()
    require(path.stat().st_mode & 0o777 == 0o600, 'Secret file must have mode 600.')
    value = path.read_text().strip()
    require(value and '\n' not in value and '\r' not in value, 'Invalid secret file.')
    return value


def guard(base, uri):
    # Exact allowlist, not a substring check: the staging hostname contains
    # "production" because that is Railway's environment label, not its service.
    require(base == BASE, 'Refusing API base: only the documented staging HTTPS origin is allowed.')
    for key in ('PGHOST', 'PGHOSTADDR', 'PGSERVICE', 'PGSERVICEFILE', 'PGOPTIONS', 'PGPORT', 'PGDATABASE', 'PGUSER'):
        require(not os.getenv(key), 'Refusing libpq routing/session overrides.')
    for candidate in {uri, os.getenv('DATABASE_URL', uri)}:
        assert_staging_database(candidate, 'staging', PRODUCTION_DATABASE_HOST)
        parts = urlsplit(candidate)
        require(parts.hostname == DB_HOST and unquote(parts.username or '') == 'postgres.' + DB_REF
                and parts.path == '/postgres' and parts.port in (5432, 6543),
                'Refusing database: documented staging host, project and database are required.')


@contextmanager
def reader(uri):
    # Explicit transaction scope, never set_session/default_transaction_read_only.
    conn = psycopg2.connect(uri, port=5432, sslmode='require', connect_timeout=10)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            cur.execute("SET LOCAL statement_timeout='60s'")
            cur.execute('SET LOCAL search_path=public')
            yield cur
    finally:
        conn.rollback()
        conn.close()


def query(cur, sql):
    cur.execute(sql)
    return [dict(row) for row in cur.fetchall()]


def snapshot(uri):
    with reader(uri) as cur:
        result = {
            'lots': query(cur, f'''SELECT l.id,l.product_id,l.lot_code,l.status,l.kosher_tier,
              p.name product_name,p.active product_active,to_jsonb(p) catalog,coalesce(sum(tl.quantity_lb),0) balance_lb
              FROM lots l JOIN products p ON p.id=l.product_id
              LEFT JOIN {POSTED} tl ON tl.lot_id=l.id GROUP BY l.id,p.id ORDER BY l.id'''),
            'products': query(cur, 'SELECT id,name,active FROM products ORDER BY id'),
            'customers': query(cur, 'SELECT id,name,active FROM customers ORDER BY id'),
            'orders': query(cur, '''SELECT s.id,s.order_number,s.customer_id,c.name customer_name,
              s.status,s.state,s.customer_po FROM sales_orders s
              JOIN customers c ON c.id=s.customer_id ORDER BY s.id'''),
            'order_lines': query(cur, 'SELECT to_jsonb(l) line FROM sales_order_lines l ORDER BY id'),
            'exceptions': query(cur, '''SELECT e.*,p.name product_name,l.lot_code,
              c.name customer_name,a.name owner_name FROM exceptions e
              LEFT JOIN products p ON p.id=e.product_id LEFT JOIN lots l ON l.id=e.lot_id
              LEFT JOIN sales_orders s ON s.id=e.sales_order_id LEFT JOIN customers c ON c.id=s.customer_id
              LEFT JOIN actors a ON a.id=e.owner_actor_id ORDER BY e.id'''),
            'shortages': query(cur, '''SELECT sf.*,p.name product_name,l.lot_code FROM shortage_flags sf
              JOIN products p ON p.id=sf.product_id JOIN lots l ON l.id=sf.lot_id ORDER BY sf.id'''),
            'write_counts': query(cur, '''SELECT (SELECT count(*) FROM write_tickets) tickets,
              (SELECT count(*) FROM transactions) transactions,
              (SELECT count(*) FROM actor_write_audit) actor_audits''')[0],
        }
    return json.loads(encoded(result))


class API:
    def __init__(self, base, key, apply=False):
        self.apply = apply
        self.http = httpx.Client(base_url=base, headers={'X-API-Key': key},
                                 timeout=60, follow_redirects=False, trust_env=False)

    def call(self, method, path, payload=None):
        require(method == 'GET' or self.apply, 'Dry run refuses all API mutations, including prepare.')
        response = self.http.request(method, path, json=payload) if payload is not None else self.http.request(method, path)
        safe_path = '/tickets/[redacted]/commit' if path.startswith('/tickets/') else path
        require(response.status_code in (200, 202), f'Staging API request failed: {method} {safe_path}, HTTP {response.status_code}.')
        return response.json()

    def close(self):
        self.http.close()


def test_name(value):
    return bool(TEST_NAME.match(value or ''))


def test_exception(row):
    # OPEN stock is protected even if a test actor opened it. Owner alone is
    # never test evidence: the copied production exceptions name real people.
    if row['id'] in PROTECTED_EXCEPTION_IDS or (row.get('lot_code') or '').startswith('OPEN261014-'):
        return False
    detail = row.get('detail') or {}
    return (test_name(row.get('product_name')) or test_name(row.get('customer_name'))
            or (row.get('lot_code') or '').startswith('STG-')
            or (row['kind'] == 'SHIFT_DISCREPANCY'
                and test_name(detail.get('notes')) and 'STAGING PILOT TEST' in detail.get('actor_name', '')))


def operation(category, row, before, after, path=None, payload=None, action=None, blocked=None):
    return dict(category=category, id=row['id'], label=row.get('lot_code') or row.get('order_number')
                or row.get('name') or row.get('kind') or row.get('product_name'), before=before,
                after=after, path=path, payload=payload, action=action,
                status='blocked' if blocked else 'planned', explanation=blocked or '')


def build_plan(snap, api):
    schema = api.call('GET', '/openapi.json')
    who = api.call('GET', '/auth/whoami')
    require(who.get('key_kind') == 'actor' and (who.get('actor') or {}).get('role') == 'owner',
            'A named staging owner key is required; shared keys are refused.')
    paths = schema['paths']
    for route in ('/adjust/prepare', '/sales/orders/{order_id}/cancel/prepare',
                  '/exceptions/{exception_id}/resolve/prepare', '/tickets/{ticket}/commit', '/receipts/{receipt_number}'):
        require(route in paths, 'Required ticket/receipt route is unavailable: ' + route)
    require(PROTECTED_EXCEPTION_IDS <= {r['id'] for r in snap['exceptions']}, 'Protected copied-exception identities are missing; stop for review.')
    items = []
    preserved_orders = []
    open_shortages = [r for r in snap['shortages'] if r['status'] in ('open', 'escalated')]
    for row in snap['lots']:
        if not row['lot_code'].startswith('STG-'):
            continue
        balance = Decimal(row['balance_lb'])
        item = operation('lots', row, row['balance_lb'], '0', '/adjust/prepare',
                         dict(lot_id=row['id'], delta_lb=float(-balance), reason_code='physical_count',
                              note=REASON, client_source='api'), 'adjust')
        item['product_id'] = row['product_id']
        item['product_name'] = row['product_name']
        item['kosher_tier'] = row['kosher_tier']
        # Same pure review function as the API, without importing main/startup.
        def ledger_unit(uom):
            unit = (uom or 'lb').strip() or 'lb'
            return 'lb' if re.search(r'(?<![a-z])(?:lbs?|pounds?)(?![a-z])', unit, re.I) else unit
        item['owner_approval'] = correction_review(SimpleNamespace(ledger_quantity_unit=ledger_unit),
            'adjust', row['catalog'], float(-balance), book_balance_before=float(balance),
            reason_code='physical_count')['photo_required']
        blockers = []
        if not balance:
            item.update(status='noop', explanation='Already zero; no zero-delta ticket.')
        else:
            if row['product_active'] is False:
                blockers.append('Product inactive; adjust ticket rejects inactive products.')
            if row['status'] == 'merged':
                blockers.append('Merged lot; adjust ticket rejects merged lots.')
            if balance < 0 and any(s['lot_id'] == row['id'] for s in open_shortages):
                blockers.append('SHORTAGE_OPEN_RESOLVE_INSTEAD: positive correction would cover an open shortage; waiver unsupported.')
            if blockers:
                item.update(status='blocked', explanation=' '.join(blockers))
            elif item['owner_approval']:
                item['explanation'] = 'Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts.'
        items.append(item)
    # No direct admin updates: the requested contract requires prepare/commit,
    # reason and receipts. List existing direct paths honestly for review.
    for category, route in [('products', '/admin/products/{product_id}'), ('customers', '/customers/{customer_id}')]:
        for row in snap[category]:
            if test_name(row['name']):
                item = operation(category, row, row['active'], False, blocked=(
                    f'Direct API exists ({route}) but has no supported prepare → commit / receipt path for deactivation.'
                    if route in paths else 'No supported deactivation API in deployed schema.'))
                if row['active'] is False:
                    item.update(status='noop', explanation='Already inactive.')
                items.append(item)
    for row in snap['orders']:
        if row['state'] == 'open' and test_name(row['customer_name']):
            live = api.call('GET', f"/sales/orders/{row['id']}")
            row = dict(row, fulfillment=live['fulfillment'])
            if row['fulfillment'] != 'unshipped':
                preserved_orders.append(row)
                continue
            items.append(operation('orders', row, dict(status=row['status'], state=row['state'], fulfillment=row['fulfillment']),
                dict(status='cancelled', state='cancelled', fulfillment=row['fulfillment']),
                f"/sales/orders/{row['id']}/cancel/prepare", dict(reason='other', note=REASON, client_source='api'),
                'cancel_order'))
            items[-1].update(customer_id=row['customer_id'], customer_name=row['customer_name'])
    for row in snap['exceptions']:
        if row['status'] not in ('open', 'escalated') or not test_exception(row):
            continue
        view = api.call('GET', f"/exceptions/{row['id']}")
        choices = view.get('allowed_actions', [])
        # written_off is the canonical UNIDENTIFIED_LOT waiver; both result in
        # status waived. Never substitute counted for an unsupported shortage waiver.
        kind = 'written_off' if row['kind'] == 'UNIDENTIFIED_LOT' else 'waived'
        category = 'shortages' if row['kind'] == 'SHORTAGE' else 'exceptions'
        item = operation(category, row, dict(kind=row['kind'], status=row['status']),
            dict(status='waived', resolution_kind=kind), f"/exceptions/{row['id']}/resolve/prepare",
            dict(resolution_kind=kind, note=REASON, client_source='api'), 'resolve_exception',
            blocked=f"No supported waiver; allowed resolutions: {', '.join(choices) or 'none'}." if kind not in choices else None)
        item.update(lot_code=row.get('lot_code'), product_name=row.get('product_name'), customer_name=row.get('customer_name'))
        if category == 'shortages':
            item['shortage_flag_ids'] = [s['id'] for s in open_shortages if s['exception_id'] == row['id']]
        items.append(item)
    protected = [r for r in snap['exceptions'] if r['id'] in PROTECTED_EXCEPTION_IDS or (r.get('lot_code') or '').startswith('OPEN261014-')]
    return dict(api_base=BASE, reason=REASON, actor=who['actor'], items=items,
                protected_open_lots=[r for r in snap['lots'] if r['lot_code'].startswith('OPEN261014-')],
                protected_exceptions=protected, preserved_test_orders=preserved_orders, snapshot_hash=digest(snap),
                capabilities_hash=digest({'schema': schema, 'actor': who['actor']}))


def plan_hash(plan):
    return digest(plan)


def summary(plan):
    result = {}
    for row in plan['items']:
        counts = result.setdefault(row['category'], {'total': 0, 'planned': 0, 'blocked': 0, 'noop': 0})
        counts['total'] += 1
        counts[row['status']] += 1
    real = [r for r in plan['items'] if r['category'] == 'lots' and r['product_id'] < 1_000_000_000 and Decimal(r['before']) != 0]
    result['nonzero_real_product_lots'] = dict(lots=len(real), products=len({r['product_id'] for r in real}),
        balance_lb=str(sum((Decimal(r['before']) for r in real), Decimal(0))))
    result['preserved_shipped_test_orders'] = len(plan['preserved_test_orders'])
    result['protected_open_lots'] = len(plan['protected_open_lots'])
    result['protected_exceptions'] = len(plan['protected_exceptions'])
    return result


def table(value):
    return (encoded(value) if isinstance(value, (dict, list)) else str(value)).replace('|', '\\|').replace('\n', ' ')


def markdown(plan, target, python=sys.executable, options=()):
    stamp = datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds')
    command = shlex.join([python, str(Path(__file__).resolve()), '--apply', '--plan', str(target),
                          '--expected-plan-sha256', plan_hash(plan), *options])
    text = [f'# Staging pilot cleanup — dry run, {stamp}', '',
        f'API: `{BASE}`. Reason/note: `{REASON}`. Plan SHA-256: `{plan_hash(plan)}`.', '',
        'Read-only discovery only: no prepare, commit, approval, admin mutation or direct SQL write was run.', '',
        'Apply uses only ticket adapters; unsupported items are skipped and remain blocked. Direct product/customer deactivation APIs exist but do not meet the required ticket/receipt contract.', '',
        'UNIDENTIFIED_LOT uses `written_off` (canonical API decision, final status `waived`). Shortages cannot be waived; no counted resolution is substituted.', '',
        'Live reconciliation: the morning 95 lots/~5,867 lb was the real-product subset. All STG- lots, including synthetic products and zero balances, are listed below. Live shortages/exceptions can exceed the morning 4/6; every selected item has explicit test product/customer/lot/shift evidence. Owner identity alone never selects an exception.', '',
        'Apply checks each lot again before prepare and verifies zero balances and exact committed receipts. The server can recalculate at commit; a detected concurrent change stops subsequent items after that commit for reconciliation. This cannot prevent the one in-flight change; use a quiet staging window for a future apply.', '',
        'The apply command below is provided for review and has NOT been run. A changed snapshot or contract requires a new dry run and reviewed hash. Partial/uncertain apply stops; inspect the adjacent append-only .receipts.jsonl journal before preparing another plan.', '',
        '```sh', command, '```', '', '## Counts', '', '```json', json.dumps(summary(plan), indent=2), '```', '']
    for category in ('lots', 'products', 'customers', 'orders', 'shortages', 'exceptions'):
        text += [f'## {category.title()}', '', '| ID | Item | Before → desired after | Status / path / limits |', '|---|---|---|---|']
        for row in plan['items']:
            if row['category'] != category:
                continue
            label = row['label']
            if category == 'lots':
                label += f"; product {row['product_id']} {row['product_name']}; kosher={row['kosher_tier']}"
            if row.get('shortage_flag_ids'):
                label += f"; flags {row['shortage_flag_ids']}"
            detail = f"{row['status']}; {row['path'] or ''}; {row['explanation']}"
            text.append(f"| {row['id']} | {table(label)} | {table(row['before'])} → {table(row['after'])} | {table(detail)} |")
        text += ['']
    text += ['## Exact supported prepare requests (never sent during dry run)', '', '```jsonl',
             '\n'.join(encoded(dict(category=r['category'], id=r['id'], method='POST', path=r['path'], body=r['payload']))
                       for r in plan['items'] if r['status'] == 'planned'), '```', '',
             '## Protected OPEN261014 lots — unchanged', '', '| Lot ID | Code | Product | Before → after lb |', '|---|---|---|---|']
    for row in plan['protected_open_lots']:
        text.append(f"| {row['id']} | {table(row['lot_code'])} | {row['product_id']} | {row['balance_lb']} → {row['balance_lb']} |")
    text += ['', '## Protected copied missing-supplier-lot exceptions — unchanged', '', '| ID | Kind | Lot | Before → after status |', '|---|---|---|---|']
    for row in plan['protected_exceptions']:
        text.append(f"| {row['id']} | {row['kind']} | {table(row['lot_code'])} | {row['status']} → {row['status']} |")
    text += ['', '## Other test orders with shipped quantity — unchanged', '', '| ID | Order | Before → after |', '|---|---|---|']
    for row in plan['preserved_test_orders']:
        text.append(f"| {row['id']} | {row['order_number']} | {row['state']}/{row['fulfillment']} → {row['state']}/{row['fulfillment']} |")
    return '\n'.join(text) + '\n'


def append_journal(path, entry):
    with path.open('a') as stream:
        stream.write(encoded(entry) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_receipt(api, item, result):
    require(not result.get('state_changed'), 'Concurrent state change detected after commit; stop and reconcile journal.')
    number = result.get('receipt_number')
    require(result.get('success') and not result.get('held') and number, 'No committed receipt; stop and inspect journal.')
    receipt = api.call('GET', '/receipts/' + number)
    require(receipt.get('status') == 'committed' and receipt.get('action') == item['action']
            and receipt.get('response') == result, 'Receipt verification failed; stop and inspect journal.')
    if item['category'] == 'lots':
        require(result.get('lot_id') == item['id'] and Decimal(str(result.get('new_balance_lb'))) == 0,
                'Lot did not finish at zero; stop and inspect journal.')
        lines = [line for txn in receipt['transactions'] for line in txn['lines']]
        require(len(lines) == 1 and lines[0]['lot_id'] == item['id']
                and Decimal(str(lines[0]['quantity_lb'])) == Decimal(str(item['payload']['delta_lb'])),
                'Adjustment receipt changed unexpected lots/quantity.')
    elif item['category'] == 'orders':
        order = receipt.get('order') or {}
        require(result.get('order_id') == item['id'] and result.get('state') == 'cancelled'
                and order.get('id') == item['id'] and order.get('customer_id') == item['customer_id']
                and order.get('customer') == item['customer_name'], 'Order cancellation readback failed.')
    else:
        exception = result.get('exception') or {}
        require(exception.get('id') == item['id'] and exception.get('status') == 'waived'
                and exception.get('resolution_kind') == item['payload']['resolution_kind'], 'Exception waiver readback failed.')
    return receipt


def apply_plan(plan, api, journal):
    try:
        with journal.open('x') as stream:
            stream.write(encoded({'event': 'start', 'plan_sha256': plan_hash(plan)}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise CleanupError('Apply journal already exists; reconcile it before another apply. No automatic retry.') from None
    # Cancel reservations first. Never deactivate products ahead of corrections.
    ordered = sorted((r for r in plan['items'] if r['status'] == 'planned'),
                     key=lambda r: (r['category'] != 'orders', r['category'] != 'lots', r['id']))
    for item in ordered:
        if item['category'] == 'lots':
            live = api.call('GET', f"/lots/{item['id']}")
            require(live['lot_code'] == item['label'] and live['lot_code'].startswith('STG-')
                    and live['product_id'] == item['product_id']
                    and Decimal(str(live['quantity_on_hand'])) == Decimal(item['before']),
                    'Lot changed since reviewed plan; regenerate dry run.')
        prepared = api.call('POST', item['path'], item['payload'])
        # Ticket tokens and keys never enter the report or journal. The numeric
        # ticket ID is enough to reconcile a network failure with normal tools.
        append_journal(journal, {'event': 'prepared', 'category': item['category'], 'id': item['id'],
                                 'ticket_id': prepared.get('ticket_id'), 'payload_hash': prepared.get('payload_hash')})
        require(prepared.get('can_commit') and not prepared.get('blockers'), 'Prepare blocked; stop and inspect journal.')
        if item['category'] == 'lots':
            draft = prepared['draft']
            require(draft['lot_id'] == item['id'] and Decimal(str(draft['current_quantity_lb'])) == Decimal(item['before'])
                    and Decimal(str(draft['new_balance_lb'])) == 0 and draft.get('note') == REASON,
                    'Prepared lot count differs from reviewed plan.')
        elif item['category'] == 'orders':
            order = prepared['draft']['order']
            require(order['id'] == item['id'] and order['order_number'] == item['label']
                    and order['customer_id'] == item['customer_id'] and order['customer'] == item['customer_name']
                    and test_name(order['customer']) and order['state'] == 'open',
                    'Prepared order no longer belongs to the reviewed test customer.')
        else:
            draft = prepared['draft']
            require(draft['exception_id'] == item['id'] and draft['resolution_kind'] == item['payload']['resolution_kind']
                    and draft['note'] == REASON, 'Prepared exception differs from reviewed waiver.')
        result = api.call('POST', '/tickets/' + prepared['ticket'] + '/commit', {
            'payload_hash': prepared['payload_hash'], 'acknowledged_warnings': [
                w['code'] for w in prepared.get('warnings', []) if w.get('requires_ack')]})
        append_journal(journal, {'event': 'commit_response', 'category': item['category'], 'id': item['id'], 'response': result})
        if result.get('held'):
            require(item['category'] == 'lots' and item.get('owner_approval') and result.get('exception_id'),
                    'Unexpected hold; stop and inspect journal.')
            held = api.call('GET', f"/exceptions/{result['exception_id']}")
            require(held['kind'] == 'LARGE_CORRECTION' and held.get('ticket_id') == prepared['ticket_id'], 'Unexpected approval hold.')
            result = api.call('POST', f"/exceptions/{held['id']}/approve", {'note': REASON})
            append_journal(journal, {'event': 'approval_response', 'id': item['id'], 'response': result})
            xr = result.get('resolution_receipt_number')
            decision = api.call('GET', '/receipts/' + xr) if xr else {}
            response = decision.get('response') or {}
            require(decision.get('status') == 'committed' and decision.get('action') == 'resolve_exception'
                    and response.get('id') == held['id'] and response.get('status') == 'resolved'
                    and response.get('resolution_kind') == 'approved', 'Approval decision receipt missing or mismatched.')
        receipt = verify_receipt(api, item, result)
        append_journal(journal, {'event': 'verified', 'category': item['category'], 'id': item['id'],
                                 'receipt_number': receipt['receipt_number']})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true', help='default; only reads, no ticket prepares')
    modes.add_argument('--apply', action='store_true', help='execute supported tickets from an unchanged reviewed snapshot')
    parser.add_argument('--api-base', default=os.getenv('FL_STAGING_API_URL', os.getenv('FACTORY_API_BASE', BASE)))
    parser.add_argument('--db-url-file', type=Path)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--expected-plan-sha256', help='required for apply; copied from the reviewed dry run')
    args = parser.parse_args(argv)
    require(not args.apply or args.expected_plan_sha256, '--apply requires --expected-plan-sha256 from a reviewed dry run.')
    uri = secret_file(args.db_url_file) if args.db_url_file else os.getenv('DATABASE_URL') or secret_file(SECRETS/'staging-db-url.txt')
    guard(args.api_base, uri)
    key = secret_file(args.key_file) if args.key_file else os.getenv('STAGING_OWNER_KEY') or secret_file(SECRETS/'staging-michael-key.txt')
    api = API(args.api_base, key, apply=args.apply)
    try:
        before = snapshot(uri)
        plan = build_plan(before, api)
        target = args.plan.expanduser().resolve()
        if args.apply:
            require(plan_hash(plan) == args.expected_plan_sha256, 'Live plan differs from reviewed hash; run and review a fresh dry run.')
            require(target.exists() and f'`{args.expected_plan_sha256}`' in target.read_text(), 'Reviewed Markdown plan is missing or has another hash.')
            journal = target.with_suffix('.receipts.jsonl')
            apply_plan(plan, api, journal)
            after = snapshot(uri)
            non_targets_before = [r for r in before['lots'] if not r['lot_code'].startswith('STG-')]
            non_targets_after = [r for r in after['lots'] if not r['lot_code'].startswith('STG-')]
            require(non_targets_before == non_targets_after, 'Protected/non-target lots changed during apply; inspect journal.')
            protected_ids = {r['id'] for r in plan['protected_exceptions']}
            require(plan['protected_exceptions'] == [r for r in after['exceptions'] if r['id'] in protected_ids], 'Protected exceptions changed during apply.')
            append_journal(journal, {'event': 'complete_supported_items', 'counts': summary(plan)})
        else:
            after = snapshot(uri)
            require(before == after, 'Staging changed during dry run; no mutations were sent. Retry for a consistent plan.')
            target.parent.mkdir(parents=True, exist_ok=True)
            options = ['--api-base', args.api_base]
            for name in ('key_file', 'db_url_file'):
                if getattr(args, name):
                    options.extend(['--' + name.replace('_', '-'), str(getattr(args, name).expanduser().resolve())])
            target.write_text(markdown(plan, target, options=options))
        print(json.dumps(dict(mode='apply' if args.apply else 'dry-run', plan=str(target),
                              plan_sha256=plan_hash(plan), counts=summary(plan)), indent=2))
        return 2 if args.apply and any(r['status'] == 'blocked' for r in plan['items']) else 0
    finally:
        api.close()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except CleanupError as exc:
        print('Cleanup stopped: ' + str(exc), file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print('Cleanup stopped: ' + type(exc).__name__ + ' (details withheld to protect credentials).', file=sys.stderr)
        sys.exit(1)
