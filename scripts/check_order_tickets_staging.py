#!/usr/bin/env python3
"""A7 acceptance on guarded staging through local HTTP, never hosted deployment.

Same shape as scripts/check_lot_confirmation_staging.py: reads only the
protected staging URI; no production configuration or API keys; TestClient
without a lifespan context, so no startup migrations or sweeps run. Migration
067 must already be applied to staging (the script refuses otherwise). A
synthetic customer, two synthetic products and two temporary actors (office,
floor) are created; the order and its receipts are retained as evidence; the
actor keys exist only in memory and both actors are deactivated in `finally`.
"""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import secrets
import sys
from urllib.parse import urlsplit
from uuid import uuid4

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staging_safety import assert_staging_database, PRODUCTION_DATABASE_HOST
from scripts.seed_staging import secret_file


def check():
    uri = secret_file(Path.home()/'Documents/fl-secrets/staging-db-url.txt')
    assert_staging_database(uri, 'staging', PRODUCTION_DATABASE_HOST)
    if urlsplit(uri).hostname != 'aws-0-us-east-1.pooler.supabase.com':
        raise RuntimeError('Not the documented staging host')

    @contextmanager
    def connection():
        conn = psycopg2.connect(uri, port=5432, connect_timeout=10, sslmode='require')
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL search_path=public")
                    cur.execute("SET LOCAL lock_timeout='5s'")
                    cur.execute("SET LOCAL statement_timeout='60s'")
                yield conn
        finally:
            conn.close()

    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM migration_markers WHERE name='067_order_tickets'")
        if cur.fetchone() is None:
            raise RuntimeError('Apply migration 067 to staging before acceptance')

    os.environ['DATABASE_URL'] = uri
    os.environ['ENVIRONMENT'] = 'staging'
    os.environ['PRODUCTION_DATABASE_HOST'] = PRODUCTION_DATABASE_HOST
    os.environ['API_KEY'] = secrets.token_urlsafe(40)
    os.environ['DASHBOARD_API_KEY'] = secrets.token_urlsafe(40)
    logging.disable(logging.CRITICAL)
    import main
    main.get_db_connection = connection
    main._reset_actor_cache()
    reference = 'STG-A7-' + uuid4().hex[:12].upper()
    keys = {'office': secrets.token_urlsafe(40), 'floor': secrets.token_urlsafe(40)}
    actor_ids = {}
    client = TestClient(main.app)
    try:
        with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            for role, key in keys.items():
                cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES (%s,%s,%s,true) RETURNING id",
                            (f'{reference} {role}', role, sha256(key.encode()).hexdigest()))
                actor_ids[role] = cur.fetchone()['id']
            cur.execute('INSERT INTO customers(name, active) VALUES (%s, true) RETURNING id', (reference + ' customer',))
            customer_id = cur.fetchone()['id']
            products = {}
            for label, service, weight in [('granola', False, 25), ('pallet charge', True, None)]:
                cur.execute('''INSERT INTO products(name,odoo_code,type,uom,is_service,case_size_lb,active)
                               VALUES (%s,%s,'finished','lb',%s,%s,true) RETURNING id''',
                            (f'{reference} {label}', f'{reference}-{label[:3]}', service, weight))
                products[label] = cur.fetchone()['id']
                assert products[label] >= 1_000_000_000
            assert customer_id >= 1_000_000_000

        def post(path, payload, key, status=200):
            response = client.post(path, json=payload, headers={'X-API-Key': key})
            if response.status_code != status:
                raise RuntimeError(f'Unexpected HTTP {response.status_code} on {path}: {response.json().get("detail")}')
            return response.json()

        def commit(draft, key):
            return post('/tickets/' + draft['ticket'] + '/commit',
                        {'payload_hash': draft['payload_hash'],
                         'acknowledged_warnings': [w['code'] for w in draft['warnings'] if w.get('requires_ack')]}, key)

        office, floor = keys['office'], keys['floor']
        # Floor may not create orders (§4.3): refused before any ticket exists.
        post('/sales/orders/prepare', {'customer_id': customer_id, 'customer_po': reference,
             'lines': [{'product_id': products['granola'], 'quantity': 1, 'unit': 'cases'}]}, floor, 403)
        draft = post('/sales/orders/prepare', {
            'customer_id': customer_id, 'customer_po': reference, 'requested_ship_date': '2026-11-02',
            'client_source': 'fl_assistant',
            'lines': [{'product_id': products['granola'], 'quantity': 2, 'unit': 'cases', 'unit_price': 30},
                      {'product_id': products['pallet charge'], 'quantity': 3, 'unit': 'each', 'unit_price': 12.5}]}, office)
        assert draft['can_commit'] and draft['warnings'] == [], draft
        assert 'order_id' not in draft['draft'] and draft['draft']['total'] == 97.5
        created = commit(draft, office)
        assert created['receipt_number'].startswith('ORD-') and created['replayed'] is False
        assert created['external_order_reference'] == created['receipt_number']
        replay = commit(draft, office)
        assert replay == created | {'replayed': True}
        # The same draft again today supersedes nothing new is posted; the PO now duplicates.
        again = post('/sales/orders/prepare', {
            'customer_id': customer_id, 'customer_po': reference,
            'lines': [{'product_id': products['granola'], 'quantity': 2, 'unit': 'cases', 'unit_price': 30},
                      {'product_id': products['pallet charge'], 'quantity': 3, 'unit': 'each', 'unit_price': 12.5}]}, office)
        assert {w['code'] for w in again['warnings']} == {'DUPLICATE_CUSTOMER_PO', 'POSSIBLE_DUPLICATE'}
        refused = post('/tickets/' + again['ticket'] + '/commit', {'payload_hash': again['payload_hash']}, office, 409)
        assert refused['detail']['error_code'] == 'WARNING_NOT_ACKNOWLEDGED'

        order_id = created['order_id']
        added = commit(post(f'/sales/orders/{order_id}/lines/prepare',
                            {'lines': [{'product_id': products['granola'], 'quantity_lb': 100, 'unit_price': 2}]}, office), office)
        assert added['total_lb_added'] == 100
        ready = commit(post(f'/sales/orders/{order_id}/ready/prepare', {'note': reference}, floor), floor)
        assert ready['ready'] is True and ready['ready_by'] == f'{reference} floor'
        post(f'/sales/orders/{order_id}/close/prepare', {'reason': 'shipped_not_recorded'}, office, 403)

        response = client.get('/receipts/' + created['receipt_number'], headers={'X-API-Key': office})
        assert response.status_code == 200
        receipt = response.json()
        assert receipt['order']['order_number'] == created['order_number'] and receipt['transactions'] == []
        with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT ticket_id, external_order_reference, customer_po, status FROM sales_orders WHERE id=%s', (order_id,))
            row = cur.fetchone()
            assert row['ticket_id'] == created['ticket_id'] and row['external_order_reference'] == created['receipt_number']
            cur.execute("SELECT count(*) AS n FROM sales_orders WHERE customer_id=%s", (customer_id,))
            assert cur.fetchone()['n'] == 1
        return {'reference': reference, 'mode': 'local branch HTTP routes against STAGING; no hosted deployment',
                'migration_067_applied_separately': True, 'customer_id': customer_id, 'products': products,
                'order_id': order_id, 'order_number': created['order_number'],
                'create_receipt': created['receipt_number'], 'add_lines_receipt': added['receipt_number'],
                'ready_receipt': ready['receipt_number'], 'duplicate_ticket_id': again['ticket_id'],
                'actor_ids': actor_ids, 'actors_deactivated_after_check': True, 'receipt': receipt}
    finally:
        client.close()
        if actor_ids:
            with connection() as conn, conn.cursor() as cur:
                cur.execute('UPDATE actors SET active=false WHERE id=ANY(%s)', (list(actor_ids.values()),))
        main._reset_actor_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check()
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'receipt'}, indent=2))
    except Exception as exc:
        # Never emit DB connection exception strings or tracebacks with secrets.
        print('A7 staging acceptance failed: ' + type(exc).__name__ + ': ' + str(exc)[:300], file=sys.stderr)
        raise SystemExit(1)
