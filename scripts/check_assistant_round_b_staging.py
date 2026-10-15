#!/usr/bin/env python3
"""Prepare or create Michael's three shipping orders on guarded STAGING only.

Default mode only reads staging. --create uses the normal A7 prepare/commit
routes through the local branch, without startup, deployment, migrations,
master-data changes, actor changes, inventory posting, or shipment posting.
Stable external references make reruns return the original three orders.
Credentials are read only from protected staging files and never printed.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import logging
import os
import secrets
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staging_safety import assert_staging_database, PRODUCTION_DATABASE_HOST
from scripts.seed_staging import secret_file

REASON = 'TEST ORDER FOR MICHAEL'
CUSTOMERS = ((217, 'Sunshine Granola'), (5, 'Setton Farms'), (4, 'UNIPRO'))
REFERENCE_PREFIX = 'FL-FIXB-MICHAEL-20261015-'
EXPECTED_HOST = 'aws-0-us-east-1.pooler.supabase.com'
EXPECTED_USER = 'postgres.jygmyvxnxdjiiilhxseq'


def guarded_uri():
    uri = secret_file(Path.home() / 'Documents/fl-secrets/staging-db-url.txt')
    assert_staging_database(uri, 'staging', PRODUCTION_DATABASE_HOST)
    parsed = psycopg2.extensions.parse_dsn(uri)
    if parsed.get('host') != EXPECTED_HOST or parsed.get('user') != EXPECTED_USER:
        raise RuntimeError('Database does not match the documented staging identity')
    return uri


@contextmanager
def connection():
    # Recheck identity for every connection; port 5432 avoids shared transaction
    # pooler state. Every SQL setting here is transaction scoped.
    uri = guarded_uri()
    conn = psycopg2.connect(uri, port=5432, connect_timeout=15, sslmode='require')
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute('SET LOCAL search_path=public')
                cur.execute("SET LOCAL lock_timeout='5s'")
                cur.execute("SET LOCAL statement_timeout='60s'")
            yield conn
    finally:
        conn.close()


def plan():
    with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT 1 FROM migration_markers WHERE name='067_order_tickets'")
        if cur.fetchone() is None:
            raise RuntimeError('Migration 067 must already be applied on staging')
        cur.execute("""SELECT p.id, p.name, p.odoo_code, p.case_size_lb,
                              SUM(tl.quantity_lb) AS on_hand_lb,
                              array_agg(DISTINCT l.lot_code ORDER BY l.lot_code) AS lots
                         FROM products p
                         JOIN lots l ON l.product_id=p.id AND l.status='active'
                         JOIN ledger_current_transaction_lines tl ON tl.lot_id=l.id
                        WHERE p.id<1000000000 AND p.active IS TRUE
                          AND p.type='finished' AND NOT p.is_service
                          AND p.case_size_lb>0
                        GROUP BY p.id
                       HAVING SUM(tl.quantity_lb)>=p.case_size_lb*3
                        ORDER BY SUM(tl.quantity_lb) DESC, p.id""")
        products = [dict(r) for r in cur.fetchall()]
        if not products:
            raise RuntimeError('No real stocked finished products can supply three test cases')
        orders = []
        reserved = {}
        for index, (customer_id, expected_name) in enumerate(CUSTOMERS, 1):
            cur.execute('SELECT name,active FROM customers WHERE id=%s AND id<1000000000', (customer_id,))
            customer = cur.fetchone()
            if not customer or customer['name'] != expected_name or customer['active'] is False:
                raise RuntimeError('Expected real staging customer is unavailable')
            reference = REFERENCE_PREFIX + str(index)
            cur.execute('SELECT id,order_number,status,state FROM sales_orders WHERE external_order_reference=%s',
                        (reference,))
            existing = cur.fetchone()
            # Prefer a distinct SKU; repeats remain safe if stock is limited.
            product = products[min(index - 1, len(products) - 1)]
            pid = product['id']
            reserved[pid] = reserved.get(pid, 0) + float(product['case_size_lb'])
            if reserved[pid] > float(product['on_hand_lb']):
                raise RuntimeError('Planned cases exceed available staging stock')
            orders.append({'customer_id': customer_id, 'customer': expected_name,
                           'external_order_reference': reference,
                           'product': {k: product[k] for k in ('id', 'name', 'odoo_code', 'case_size_lb', 'on_hand_lb', 'lots')},
                           'quantity': 1, 'unit': 'cases', 'reason': REASON,
                           'existing': dict(existing) if existing else None})
        return orders


def check(create=False):
    orders = plan()
    evidence = {'staging_only': True, 'database_identity_verified': True,
                'mode': 'local branch A7 HTTP routes against guarded staging; no hosted deployment',
                'created_at': datetime.now(ZoneInfo('America/New_York')).isoformat(),
                'reason': REASON, 'orders': orders}
    if not create:
        return evidence

    # Existing named staging actor: no actor or key rows are created/changed.
    key = secret_file(Path.home() / 'Documents/fl-secrets/staging-michael-key.txt')
    os.environ['DATABASE_URL'] = guarded_uri()
    os.environ['ENVIRONMENT'] = 'staging'
    os.environ['PRODUCTION_DATABASE_HOST'] = PRODUCTION_DATABASE_HOST
    os.environ['API_KEY'] = secrets.token_urlsafe(40)
    os.environ['DASHBOARD_API_KEY'] = secrets.token_urlsafe(40)
    logging.disable(logging.CRITICAL)
    from fastapi.testclient import TestClient
    import main
    main.get_db_connection = connection
    main._reset_actor_cache()
    client = TestClient(main.app)  # No lifespan: startup migrations/sweeps never run.
    headers = {'X-API-Key': key}

    def request(method, path, body=None):
        response = client.request(method, path, json=body, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f'A7 route {path} returned HTTP {response.status_code}')
        return response.json()

    try:
        who = request('GET', '/auth/whoami')
        if who.get('actor', {}).get('role') not in ('office', 'owner'):
            raise RuntimeError('The named staging actor is not allowed to create orders')
        evidence['actor'] = {k: who['actor'][k] for k in ('id', 'name', 'role')}
        for order in orders:
            if order['existing']:
                order['order_number'] = order['existing']['order_number']
                order['order_id'] = order['existing']['id']
                order['created_this_run'] = False
                continue
            draft = request('POST', '/sales/orders/prepare', {
                'customer_id': order['customer_id'],
                'external_order_reference': order['external_order_reference'],
                'requested_ship_date': '2026-10-16',
                'notes': REASON, 'notes_es': 'PEDIDO DE PRUEBA PARA MICHAEL',
                'client_source': 'fl_assistant',
                'lines': [{'product_id': order['product']['id'], 'quantity': 1, 'unit': 'cases'}]})
            if not draft['can_commit'] or draft['blockers']:
                raise RuntimeError('A7 test order has blockers; no commit was attempted')
            created = request('POST', '/tickets/' + draft['ticket'] + '/commit', {
                'payload_hash': draft['payload_hash'],
                'acknowledged_warnings': [w['code'] for w in draft['warnings'] if w.get('requires_ack')]})
            if not created.get('receipt_number', '').startswith('ORD-') or not created.get('order_number'):
                raise RuntimeError('A7 did not return an order receipt')
            order.update(order_number=created['order_number'], order_id=created['order_id'],
                         receipt_number=created['receipt_number'], ticket_id=created['ticket_id'],
                         created_this_run=not created.get('replayed', False))
        # Durable proof: real products, positive stock, one unshipped case, reason,
        # open order state, committed A7 ticket, and named actor attribution.
        with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            for order in orders:
                cur.execute("""SELECT so.id,so.order_number,so.status,so.state,so.notes,
                                      so.external_order_reference,so.customer_id,
                                      wt.action AS ticket_action,wt.status AS ticket_status,
                                      wt.receipt_number,sol.product_id,sol.quantity_lb,
                                      sol.quantity_shipped_lb,p.name AS product_name,p.case_size_lb
                                 FROM sales_orders so
                                 JOIN sales_order_lines sol ON sol.sales_order_id=so.id
                                 JOIN products p ON p.id=sol.product_id
                                 JOIN write_tickets wt ON wt.id=so.ticket_id
                                WHERE so.id=%s""", (order['order_id'],))
                rows = [dict(r) for r in cur.fetchall()]
                if len(rows) != 1:
                    raise RuntimeError('Unexpected durable test-order lines')
                row = rows[0]
                if not (row['state'] == 'open' and row['status'] == 'confirmed'
                        and row['notes'] == REASON and row['ticket_action'] == 'create_order'
                        and row['ticket_status'] == 'committed' and float(row['quantity_shipped_lb'] or 0) == 0
                        and row['customer_id'] < 1000000000 and row['product_id'] < 1000000000
                        and float(row['quantity_lb']) == float(row['case_size_lb'])):
                    raise RuntimeError('Durable staging order verification failed')
                order['verification'] = row
        evidence['verified_open_orders'] = len(orders)
        return evidence
    finally:
        client.close()
        main._reset_actor_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create', action='store_true', help='Create the three already authorized staging test orders through A7')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = check(create=args.create)
        serialized = json.dumps(result, indent=2, ensure_ascii=False, default=str) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized)
        print(serialized, end='')
    except Exception as exc:
        # Connection errors may contain credential-bearing DSNs; never echo them.
        print('Round B staging orders failed: ' + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
