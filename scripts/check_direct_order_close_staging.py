#!/usr/bin/env python3
"""O-4 synthetic matrix through local branch HTTP against guarded staging.

No hosted deploy, startup, migration, schema edit, real API key, or production
connection. All fixtures and HTTP writes share one transaction, rolled back
before a fresh connection verifies cleanup. PostgreSQL sequences may advance;
their values are never reset on a shared staging database. The cutoff varies
only in this process so immutable synthetic creation timestamps stay intact.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from itertools import product
import json
import logging
import os
from pathlib import Path
import secrets
import subprocess
import sys
from uuid import uuid4
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staging_safety import assert_staging_database, database_host, PRODUCTION_DATABASE_HOST
from scripts.seed_staging import secret_file

PLANT_TIMEZONE = ZoneInfo('America/New_York')
ROLES = ('owner', 'office', 'floor', 'master', 'dashboard')
CLOSE_REASONS = ('shipped_not_recorded', 'shipped_recorded', 'short_closed')
STATUS_TARGETS = ('invoiced', 'shipped', 'partial_ship')
PATHS = (
    [(f'direct_close_{mode}_{reason}', 'direct_close', reason, mode)
     for reason in CLOSE_REASONS for mode in ('preview', 'commit')]
    + [(f'direct_status_{status}', 'direct_status', status, 'commit') for status in STATUS_TARGETS]
    + [(f'ticket_close_{reason}', 'ticket_close', reason, 'commit') for reason in CLOSE_REASONS]
    + [(f'ticket_status_{status}', 'ticket_status', status, 'commit') for status in STATUS_TARGETS]
    + [('direct_cancel_preview', 'direct_cancel', 'customer_cancelled', 'preview'),
       ('direct_cancel_commit', 'direct_cancel', 'customer_cancelled', 'commit'),
       ('direct_status_cancelled', 'direct_status', 'cancelled', 'commit'),
       ('ticket_cancel', 'ticket_cancel', 'customer_cancelled', 'commit'),
       ('ticket_status_cancelled', 'ticket_status', 'cancelled', 'commit'),
       ('direct_status_invoiced_recorded_partial', 'direct_status', 'invoiced', 'commit'),
       ('ticket_status_invoiced_recorded_partial', 'ticket_status', 'invoiced', 'commit')]
    + [(f'direct_reopen_{mode}_{status}', 'direct_reopen', status, mode)
       for status in ('shipped', 'invoiced') for mode in ('preview', 'commit')]
    + [(f'ticket_reopen_{status}', 'ticket_reopen', status, 'commit')
       for status in ('shipped', 'invoiced')]
    + [('direct_ship_repair', 'direct_ship_repair', 'shipped', 'commit'),
       ('direct_ship_commit_repair', 'direct_ship_commit_repair', 'shipped', 'commit')]
)


class ProofFailure(RuntimeError):
    """Safe diagnostics assembled only from public labels and HTTP statuses."""


def require(condition, message):
    if not condition:
        raise ProofFailure(message)


class SavepointConnection:
    """Endpoint commit/rollback stays within its request savepoint."""
    def __init__(self, connection, name):
        self.connection, self.name = connection, name

    def cursor(self, *args, **kwargs):
        return self.connection.cursor(*args, **kwargs)

    def commit(self):
        with self.cursor() as cur:
            cur.execute(f'RELEASE SAVEPOINT {self.name}')
            cur.execute(f'SAVEPOINT {self.name}')

    def rollback(self):
        with self.cursor() as cur:
            cur.execute(f'ROLLBACK TO SAVEPOINT {self.name}')


def outcome(response):
    """Only allowlisted, non-credential response metadata enters the receipt."""
    body = response.json()
    detail = body.get('detail', {}) if isinstance(body, dict) else {}
    code = detail.get('error_code') if isinstance(detail, dict) else None
    blockers = body.get('blockers', []) if isinstance(body, dict) else []
    if isinstance(detail, dict):
        blockers += detail.get('blockers', [])
    return {'http': response.status_code, 'error_code': code,
            'blocker_codes': sorted({b.get('code', '') for b in blockers})}


def allowed(kind, target, role, period, recorded, note):
    roles = {'owner', 'office', 'master'} if kind.startswith('ticket_') else {'owner', 'office', 'master', 'dashboard'}
    if kind.endswith('reopen'):
        return role in {'owner', 'master'} and recorded
    if kind.endswith('repair'):
        return role in {'owner', 'master'} and recorded and not (kind == 'direct_ship_commit_repair' and note)
    if kind == 'ticket_status' and note:
        return False  # status tickets reject a note; cannot turn it into permission
    if role not in roles:
        return False
    if target in ('shipped', 'partial_ship'):
        return False
    if kind in ('direct_cancel', 'ticket_cancel') or target == 'cancelled':
        return not recorded
    if target == 'shipped_not_recorded':
        return role == 'owner' and note and (recorded or period == 'before')
    return recorded


def check():
    source_names = ('main.py', 'order_tickets.py', 'ship_tickets.py', 'permissions.py', 'scripts/check_direct_order_close_staging.py')
    source_hashes = {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names}
    uri = secret_file(Path.home() / 'Documents/fl-secrets/staging-db-url.txt')
    assert_staging_database(uri, 'staging', PRODUCTION_DATABASE_HOST)
    require(database_host(uri) == 'aws-0-us-east-1.pooler.supabase.com', 'Not the documented staging host')
    logging.disable(logging.CRITICAL)
    os.environ.update(DATABASE_URL=uri, ENVIRONMENT='staging',
                      PRODUCTION_DATABASE_HOST=PRODUCTION_DATABASE_HOST,
                      API_KEY=secrets.token_urlsafe(40), DASHBOARD_API_KEY=secrets.token_urlsafe(40),
                      READONLY_API_KEY='')
    # Import has no database connection; TestClient is deliberately never used
    # as a context manager, so app lifespan/startup sweeps cannot run.
    import main
    require(hasattr(main, '_require_order_close_policy'), 'Branch close guard is not present')
    conn = psycopg2.connect(uri, port=5432, connect_timeout=10, sslmode='require')
    request_serial = 0

    @contextmanager
    def connection():
        nonlocal request_serial
        request_serial += 1
        name = f'o4_request_{request_serial}'
        with conn.cursor() as cur:
            cur.execute(f'SAVEPOINT {name}')
        proxy = SavepointConnection(conn, name)
        try:
            yield proxy
        except BaseException:
            proxy.rollback()
            raise
        finally:
            with conn.cursor() as cur:
                cur.execute(f'RELEASE SAVEPOINT {name}')

    main.get_db_connection = connection
    main._reset_actor_cache()
    reference = 'STG-O4-' + uuid4().hex[:12].upper()
    # Explicit random synthetic IDs avoid changing fixture-table sequences.
    base = 1_100_000_000 + secrets.randbelow(600_000_000)
    ids = {'actors': [], 'customers': [base], 'products': [base],
           'sales_orders': [base, base + 1, base + 2], 'sales_order_lines': [base, base + 1, base + 2],
           'lots': [base], 'transactions': [base, base + 1, base + 2],
           'transaction_lines': [base, base + 1, base + 2], 'sales_order_shipments': [base, base + 1],
           'shipments': [base], 'shipment_lines': [base],
           'write_tickets': []}
    keys = {role: secrets.token_urlsafe(40) for role in ('owner', 'office', 'floor')}
    keys.update(master=main.API_KEY, dashboard=main.DASHBOARD_API_KEY)
    fixtures, rows = {}, []
    client = TestClient(main.app)
    started = datetime.now(timezone.utc).isoformat()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SET LOCAL search_path=public")
            cur.execute("SET LOCAL lock_timeout='5s'")
            cur.execute("SET LOCAL statement_timeout='60s'")
            for marker in ('067_order_tickets', '074_ship_gate_attachments'):
                cur.execute('SELECT 1 FROM migration_markers WHERE name=%s', (marker,))
                require(cur.fetchone() is not None, f'Required staging marker absent: {marker}')
            for offset, role in enumerate(('owner', 'office', 'floor')):
                actor_id = base + offset
                cur.execute('INSERT INTO actors(id,name,role,key_hash,active) VALUES(%s,%s,%s,%s,true)',
                            (actor_id, f'{reference} {role}', role, sha256(keys[role].encode()).hexdigest()))
                ids['actors'].append(actor_id)
            cur.execute('INSERT INTO customers(id,name,active) VALUES(%s,%s,true)', (base, reference + ' customer'))
            cur.execute("INSERT INTO products(id,name,type,uom,is_service,active) VALUES(%s,%s,'finished','lb',false,true)",
                        (base, reference + ' product'))
            cur.execute("INSERT INTO lots(id,product_id,lot_code,lot_type) VALUES(%s,%s,%s,'finished')",
                        (base, base, reference + '-LOT'))
            cur.execute("INSERT INTO transactions(id,type,notes) VALUES(%s,'receive',%s)", (base + 1, reference))
            cur.execute('''INSERT INTO transaction_lines(id,transaction_id,product_id,lot_id,quantity_lb)
                           VALUES(%s,%s,%s,%s,15)''', (base + 1, base + 1, base, base))
            for offset, (fixture_key, shipped_quantity) in enumerate(((False, 0), (True, 10), ('partial', 5))):
                recorded = bool(shipped_quantity)
                order_id = base + offset
                cur.execute('''INSERT INTO sales_orders(id,order_number,customer_id,status,customer_po,order_date)
                               VALUES(%s,%s,%s,'ready',%s,'1999-01-01') RETURNING created_at''',
                            (order_id, f'{reference}-{offset}', base, reference))
                created_at = cur.fetchone()['created_at']
                fixtures[fixture_key] = {'order_id': order_id, 'created_at': created_at.isoformat(),
                                        'created_on_ny': created_at.astimezone(PLANT_TIMEZONE).date().isoformat(),
                                        'ordered_lb': 10, 'effective_shipped_lb': shipped_quantity,
                                        'shipment_representation': 'historical ledger link without unified header' if fixture_key == 'partial'
                                        else 'unified shipment header, ledger lines and order shipment link' if recorded else 'none'}
                cur.execute('''INSERT INTO sales_order_lines(id,sales_order_id,product_id,quantity_lb,quantity_shipped_lb,line_status)
                               VALUES(%s,%s,%s,10,%s,%s)''',
                            (order_id, order_id, base, shipped_quantity, 'fulfilled' if shipped_quantity == 10 else 'partial' if recorded else 'pending'))
                if recorded:
                    transaction_id = base if shipped_quantity == 10 else base + 2
                    shipment_link_id = base if shipped_quantity == 10 else base + 1
                    cur.execute("INSERT INTO transactions(id,type,notes) VALUES(%s,'ship',%s)", (transaction_id, reference))
                    cur.execute('''INSERT INTO transaction_lines(id,transaction_id,product_id,lot_id,quantity_lb)
                                   VALUES(%s,%s,%s,%s,%s)''', (transaction_id, transaction_id, base, base, -shipped_quantity))
                    cur.execute('''INSERT INTO sales_order_shipments(id,sales_order_line_id,transaction_id,quantity_lb)
                                   VALUES(%s,%s,%s,%s)''', (shipment_link_id, order_id, transaction_id, shipped_quantity))
                if shipped_quantity == 10:
                    cur.execute('''INSERT INTO shipments(id,sales_order_id,customer_id,shipped_at,transaction_id,bol_reference)
                                   VALUES(%s,%s,%s,clock_timestamp(),%s,%s)''',
                                (base, order_id, base, base, reference))
                    cur.execute('''INSERT INTO shipment_lines(id,shipment_id,transaction_id,sales_order_line_id,product_id,quantity_lb)
                                   VALUES(%s,%s,%s,%s,%s,10)''', (base, base, base, order_id, base))
            cur.execute('SAVEPOINT o4_case')

        def call(method, path, payload, role):
            return client.request(method, path, json=payload, headers={'X-API-Key': keys[role]})

        for path_label, kind, target, mode in PATHS:
            role_cases = ('owner', 'master') if kind.endswith(('reopen', 'repair')) else ROLES
            is_recorded_partial = path_label.endswith('recorded_partial')
            shipment_cases = (True,) if is_recorded_partial else (False, True)
            for role, period, recorded, note in product(role_cases, ('before', 'on_or_after'), shipment_cases, (False, True)):
                fixture = fixtures['partial'] if is_recorded_partial else fixtures[recorded]
                order_id = fixture['order_id']
                created_on = datetime.fromisoformat(fixture['created_at']).astimezone(PLANT_TIMEZONE).date()
                cutoff = created_on + timedelta(days=1) if period == 'before' else created_on
                os.environ['ORDER_LEGACY_CLOSE_CUTOFF'] = cutoff.isoformat()
                is_reopen, is_repair = kind.endswith('reopen'), kind.endswith('repair')
                initial_state = 'closed' if is_reopen else 'open'
                initial_status = 'partial_ship' if is_repair else 'shipped' if target == 'invoiced' or recorded else 'ready'
                with conn.cursor() as cur:
                    cur.execute('''UPDATE sales_orders SET status=%s,state=%s,state_reason=%s,status_before_exit=%s WHERE id=%s''',
                                (initial_status, initial_state, 'short_closed' if is_reopen else None,
                                 target if is_reopen else None, order_id))
                    if is_repair:
                        # Deliberately spoof counters when no effective ship
                        # exists; repair must not treat counters as evidence.
                        cur.execute("UPDATE sales_order_lines SET quantity_shipped_lb=quantity_lb,line_status='fulfilled' WHERE sales_order_id=%s", (order_id,))
                payload = ({'reason': target} if kind.endswith(('close', 'cancel')) else {}
                           if is_reopen else {'ship_all': True} if is_repair else {'status': target})
                if note:
                    payload['note'] = 'Synthetic legacy truck log for ' + reference
                expected = allowed(kind, target, role, period, recorded, note)
                phases = {}
                if kind.startswith('ticket_'):
                    suffix = kind.removeprefix('ticket_')
                    # Keep the supplied note: the status ticket schema refuses
                    # it, proving a note cannot bypass shipment requirements.
                    response = call('POST', f'/sales/orders/{order_id}/{suffix}/prepare', payload, role)
                    phases['prepare'] = outcome(response)
                    prepared = response.json()
                    if response.status_code == 200:
                        require(bool(prepared.get('can_commit')) == expected, f'Unexpected draft eligibility: {path_label}/{role}/{period}/{recorded}/{note}')
                        if prepared.get('ticket_id'):
                            ids['write_tickets'].append(prepared['ticket_id'])
                        commit_body = {'payload_hash': prepared['payload_hash'],
                                       'acknowledged_warnings': [w['code'] for w in prepared.get('warnings', []) if w.get('requires_ack')]}
                        committed = call('POST', '/tickets/' + prepared['ticket'] + '/commit', commit_body, role)
                        phases['commit'] = outcome(committed)
                        success = committed.status_code == 200
                    else:
                        success = False
                else:
                    suffix = 'ship/commit' if kind == 'direct_ship_commit_repair' else 'ship' if is_repair else kind.removeprefix('direct_')
                    if suffix in ('close', 'cancel', 'reopen', 'ship'):
                        payload['mode'] = mode
                    method = 'PATCH' if suffix == 'status' else 'POST'
                    response = call(method, f'/sales/orders/{order_id}/{suffix}', payload, role)
                    phases[mode] = outcome(response)
                    success = response.status_code == 200
                require(success == expected, f'Unexpected result: {path_label}/{role}/{period}/{recorded}/{note}')
                require(all(p['http'] < 500 for p in phases.values()), f'Unexpected server error: {path_label}/{role}')
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute('SELECT state,status,state_reason,state_note,state_changed_by FROM sales_orders WHERE id=%s', (order_id,))
                    state = cur.fetchone()
                    expected_state = initial_state
                    if expected and mode == 'commit':
                        expected_state = 'open' if is_reopen or is_repair else 'cancelled' if kind.endswith('cancel') or target == 'cancelled' else 'closed'
                    require(state['state'] == expected_state, f'Unexpected persisted state: {path_label}/{role}')
                    if not expected or mode == 'preview':
                        require(state['status'] == initial_status, f'Refusal or preview mutated status: {path_label}/{role}')
                    elif is_reopen or is_repair:
                        require(state['status'] == target, f'Unexpected restored or repaired status: {path_label}/{role}')
                    if expected and target == 'shipped_not_recorded' and mode == 'commit':
                        require(state['state_note'] == payload['note'] and state['state_changed_by'] == f'{reference} owner',
                                'Legacy close lost named owner or note')
                    if expected and target == 'invoiced' and mode == 'commit':
                        require(state['state_reason'] != 'shipped_not_recorded', 'Invoice inferred legacy owner exception')
                        if is_recorded_partial:
                            require(state['state_reason'] == 'short_closed', 'Recorded partial invoice did not use short_closed')
                    cur.execute('ROLLBACK TO SAVEPOINT o4_case')
                rows.append({'path': path_label, 'role': role, 'created': period,
                             'recorded_shipment': recorded, 'note': note,
                             'effective_shipped_lb': fixture['effective_shipped_lb'],
                             'cutoff_ny': cutoff.isoformat(), 'expected_allowed': expected,
                             'phases': phases, 'resulting_state': expected_state,
                             'resulting_state_reason': state['state_reason']})
            print(f'O-4 staging matrix: {len(rows)} cases verified', flush=True)
    finally:
        client.close()
        conn.rollback()
        conn.close()
        main._reset_actor_cache()

    # A separate transaction sees none of the rolled-back fixtures/tickets.
    with psycopg2.connect(uri, port=5432, connect_timeout=10, sslmode='require') as verify:
        with verify.cursor() as cur:
            cur.execute('SET TRANSACTION READ ONLY')
            remaining = {}
            for table, fixture_ids in ids.items():
                # Table identifiers come exclusively from the literal allowlist.
                cur.execute(f'SELECT count(*) FROM public.{table} WHERE id=ANY(%s)', (fixture_ids,))
                remaining[table] = cur.fetchone()[0]
            require(not any(remaining.values()), 'Synthetic fixture cleanup verification failed')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    require(source_hashes == {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names},
            'Branch source changed during staging proof; rerun against stable source')
    return {'reference': reference, 'passed': True,
            'mode': 'local branch HTTP routes against guarded STAGING database; no hosted deployment',
            'started_at_utc': started, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'git_head_at_run': head, 'source_sha256': source_hashes,
            'source_unchanged_during_run': True,
            'guarded_staging_host': database_host(uri), 'database_port': 5432,
            'app_startup_or_lifespan_run': False, 'migration_or_schema_change': False,
            'production_connection_or_change': False, 'real_api_keys_loaded': False,
            'cutoff_configuration': 'process-local per-case value; hosted settings unchanged',
            'note_dimension': 'Actual note field supplied when true, including schema refusals on unsupported endpoints',
            'matrix_scope': '20 close/status/cancel paths with all five key kinds; 2 recorded-partial invoice paths with all five key kinds; 8 additional reopen/ship repair controls with owner and master',
            'synthetic_fixtures': list(fixtures.values()), 'synthetic_order_date': '1999-01-01',
            'fixtures_and_http_writes_rolled_back': True, 'post_rollback_counts': remaining,
            'sequence_policy': 'PostgreSQL sequences may advance; no shared sequences reset',
            'matrix_cases': len(rows), 'http_calls': sum(len(row['phases']) for row in rows),
            'cases_by_path': dict(Counter(row['path'] for row in rows)),
            'allowed_cases': sum(row['expected_allowed'] for row in rows), 'matrix': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check()
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({key: result[key] for key in ('reference', 'passed', 'mode', 'matrix_cases', 'http_calls', 'fixtures_and_http_writes_rolled_back')}, indent=2))
    except BaseException as exc:
        # Exception text, response bodies, credentials and tracebacks never print.
        label = str(exc) if isinstance(exc, ProofFailure) else type(exc).__name__
        print('O-4 staging proof failed; sensitive diagnostics suppressed: ' + label, file=sys.stderr)
        raise SystemExit(1)
