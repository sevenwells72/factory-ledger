#!/usr/bin/env python3
"""A9 acceptance through local HTTP against the isolated staging database.

Never deploys, opens production, or runs startup. Reads only the protected
staging URI (600-permission secret file, staging guard, documented host).
--apply-migration applies 076 to STAGING ONLY. One ephemeral floor actor is
created with a random key that stays in memory and is never printed; the actor
is deactivated in `finally`. Synthetic receipts (RCV + ADJ) stay in staging as
real ticket history; the sanitized summary and confirmation receipts are
written to docs/deployments/a9-staging-receipt.json.
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

OUT = ROOT / 'docs/deployments/a9-staging-receipt.json'


def check(apply_migration=False):
    uri = secret_file(Path.home() / 'Documents/fl-secrets/staging-db-url.txt')
    assert_staging_database(uri, 'staging', PRODUCTION_DATABASE_HOST)
    if urlsplit(uri).hostname != 'aws-0-us-east-1.pooler.supabase.com':
        raise RuntimeError('Not the documented staging database')

    @contextmanager
    def connection():
        conn = psycopg2.connect(uri, port=5432, connect_timeout=10, sslmode='require')
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL search_path=public; SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='60s'")
                yield conn
        finally:
            conn.close()

    with connection() as conn, conn.cursor() as cur:
        if apply_migration:
            cur.execute((ROOT / 'migrations/076_shift_confirmations.sql').read_text())
        cur.execute("SELECT 1 FROM migration_markers WHERE name='076_shift_confirmations'")
        if not cur.fetchone():
            raise RuntimeError('Apply staging migration 076 before acceptance (--apply-migration)')

    os.environ.update(DATABASE_URL=uri, ENVIRONMENT='staging', PRODUCTION_DATABASE_HOST=PRODUCTION_DATABASE_HOST,
                      API_KEY=secrets.token_urlsafe(40), DASHBOARD_API_KEY=secrets.token_urlsafe(40))
    logging.disable(logging.CRITICAL)
    import main
    main.get_db_connection = connection
    main._reset_actor_cache()
    ref = 'STG-A9-' + uuid4().hex[:12].upper()
    key = secrets.token_urlsafe(40)
    actor_id = None
    receipt = {'ref': ref, 'steps': []}
    try:
        with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('INSERT INTO actors(name,role,key_hash,active) VALUES(%s,%s,%s,true) RETURNING id',
                        (ref + ' floor', 'floor', sha256(key.encode()).hexdigest()))
            actor_id = cur.fetchone()['id']
            cur.execute('INSERT INTO suppliers(name,active) VALUES(%s,true) RETURNING id', (ref + ' supplier',))
            supplier = cur.fetchone()['id']
            cur.execute("INSERT INTO products(name,odoo_code,type,uom,case_size_lb,active,is_service) VALUES(%s,%s,'ingredient','lb',25,true,false) RETURNING id",
                        (ref + ' oats', ref))
            product = cur.fetchone()['id']
            if min(actor_id, supplier, product) < 1_000_000_000:
                raise RuntimeError('Staging synthetic ID range is required')
        http = TestClient(main.app)
        headers = {'X-API-Key': key}

        def post(path, payload, status=200):
            r = http.post(path, json=payload, headers=headers)
            if r.status_code != status:
                raise RuntimeError(f'Unexpected HTTP {r.status_code} on {path}: {r.text[:300]}')
            return r.json()

        def commit(p):
            return post('/tickets/' + p['ticket'] + '/commit',
                        {'payload_hash': p['payload_hash'], 'acknowledged_warnings': [w['code'] for w in p['warnings'] if w.get('requires_ack')]})

        def get(params=None, status=200):
            r = http.get('/reports/shift-summary', params=params or {}, headers=headers)
            if r.status_code != status:
                raise RuntimeError(f'Unexpected HTTP {r.status_code} on GET /reports/shift-summary: {r.text[:300]}')
            return r.json()

        empty = get()
        if empty['receipts_today'] != 0 or [s['key'] for s in empty['sections']][:3] != ['received', 'made', 'packed']:
            raise RuntimeError('Empty-day summary shape failed')
        received = commit(post('/receive/prepare', {'product_id': product, 'supplier_id': supplier, 'cases': 4, 'case_size_lb': 25,
                                                    'bol_reference': ref + '-IN', 'lot_code': ref + '-LOT', 'supplier_lot_code': 'N/A'}))
        adjusted = commit(post('/adjust/prepare', {'lot_id': received['lot_id'], 'delta_lb': -3, 'reason_code': 'unknown', 'note': 'staging check'}))
        summary = get({'lang': 'en'})
        spanish = get({'lang': 'es'})
        if summary['summary_hash'] != spanish['summary_hash']:
            raise RuntimeError('Hash must not depend on lang')
        rows = {r['receipt_number']: r for s in summary['sections'] for r in s['rows'] if 'receipt_number' in r}
        if set(rows) != {received['receipt_number'], adjusted['receipt_number']} or summary['receipts_today'] != 2:
            raise RuntimeError('Receipt rows failed')
        if [f['code'] for f in rows[received['receipt_number']]['flags']] != ['UNIDENTIFIED_LOT']:
            raise RuntimeError('UNIDENTIFIED_LOT flag failed')
        if [f['code'] for f in rows[adjusted['receipt_number']]['flags']] != ['UNKNOWN_REASON']:
            raise RuntimeError('UNKNOWN_REASON flag failed')
        # The UNIDENTIFIED_LOT exception is owned by the plant's floor owner (Arturo, A5 rule), so it
        # shows under OPEN EXCEPTIONS in the all-actors view (shared dashboard key), not under this actor.
        r = http.get('/reports/shift-summary', headers={'X-API-Key': os.environ['DASHBOARD_API_KEY']})
        if r.status_code != 200:
            raise RuntimeError(f'All-actors view failed: HTTP {r.status_code}')
        everyone = r.json()
        if everyone['actor'] is not None or everyone['confirmation']['can_confirm']:
            raise RuntimeError('All-actors view must be read-only')
        exc = next(s for s in everyone['sections'] if s['key'] == 'open_exceptions')
        mine = [e for e in exc['rows'] if e['refs'].get('receipt_number') == received['receipt_number']]
        if len(mine) != 1 or mine[0]['kind'] != 'UNIDENTIFIED_LOT' or not mine[0]['owner']:
            raise RuntimeError('Open exceptions failed')
        if received['receipt_number'] not in {r['receipt_number'] for s in everyone['sections'] for r in s['rows'] if 'receipt_number' in r}:
            raise RuntimeError('All-actors view must list the receipt')
        lots = next(s for s in summary['sections'] if s['key'] == 'lot_balances_touched')['rows']
        if len(lots) != 1 or (lots[0]['before_lb'], lots[0]['after_lb']) != (0, 97):
            raise RuntimeError('Lot balances failed')
        # Stale hash → 409 with the fresh document; nothing recorded.
        stale = post('/reports/shift-summary/confirm', {'date': summary['date'], 'outcome': 'match', 'notes': '',
                                                         'summary_hash': 'f' * 64, 'receipts_seen': 2}, 409)
        if stale['detail']['error_code'] != 'SHIFT_SUMMARY_STALE' or stale['detail']['summary']['summary_hash'] != summary['summary_hash']:
            raise RuntimeError('Stale confirm failed')
        first = post('/reports/shift-summary/confirm', {'date': summary['date'], 'outcome': 'match', 'notes': '',
                                                         'summary_hash': summary['summary_hash'], 'receipts_seen': summary['receipts_today']}, 201)
        after = get()
        if after['confirmation']['status'] != 'match' or after['confirmation']['latest']['confirmation_id'] != first['confirmation_id']:
            raise RuntimeError('Confirmation state failed')
        second = post('/reports/shift-summary/confirm', {'date': summary['date'], 'outcome': 'match', 'notes': 'staging re-confirm with notes',
                                                          'summary_hash': after['summary_hash'], 'receipts_seen': after['receipts_today'], 'lang': 'es'}, 201)
        final = get()
        if final['confirmation']['latest']['confirmation_id'] != second['confirmation_id'] or final['confirmation']['latest']['notes'] != 'staging re-confirm with notes':
            raise RuntimeError('Re-confirm append failed')
        with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT count(*) AS n FROM shift_confirmations WHERE business_date=%s AND actor_id=%s', (summary['date'], actor_id))
            if cur.fetchone()['n'] != 2:
                raise RuntimeError('Expected two appended confirmation rows')
        receipt['steps'] = [
            {'step': 'empty day', 'receipts_today': empty['receipts_today'], 'can_confirm': empty['confirmation']['can_confirm']},
            {'step': 'receive', 'receipt_number': received['receipt_number'], 'lot_id': received['lot_id']},
            {'step': 'adjust', 'receipt_number': adjusted['receipt_number']},
            {'step': 'all-actors view (dashboard key)', 'actor': None, 'can_confirm': False, 'open_exception_owner': mine[0]['owner']['name'],
             'exception_kind': mine[0]['kind']},
            {'step': 'stale confirm', 'status': 409, 'code': 'SHIFT_SUMMARY_STALE'},
            {'step': 'confirm match', 'status': 201, 'confirmation_id': first['confirmation_id'], 'message': first['message']},
            {'step': 'confirm with notes (append)', 'status': 201, 'confirmation_id': second['confirmation_id'], 'message': second['message']},
        ]
        receipt['summary_after_confirm'] = final
        receipt['title_es'] = spanish['title_es']
    finally:
        if actor_id:
            with connection() as conn, conn.cursor() as cur:
                cur.execute('UPDATE actors SET active=false WHERE id=%s', (actor_id,))
            receipt['actor_deactivated'] = True
    text = json.dumps(receipt, indent=1, default=str, ensure_ascii=False)
    if key in text:
        raise RuntimeError('refusing to write: key material in output')
    OUT.write_text(text + '\n')
    print(f'OK {ref}: {len(receipt["steps"])} steps; receipt at {OUT.relative_to(ROOT)}')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply-migration', action='store_true', help='apply 076 to STAGING before the checks')
    check(parser.parse_args().apply_migration)
