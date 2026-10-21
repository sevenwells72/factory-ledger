#!/usr/bin/env python3
"""A5 part 2 acceptance: local branch HTTP against guarded STAGING, no deployment.

Optional --apply-migration applies only 079 in a bounded transaction. Synthetic
stock/receipts remain as evidence; credentials stay in memory and the synthetic
actor is deactivated on exit. No application startup hooks or sweeps run.
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


def check(apply_migration=False):
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
        if apply_migration:
            cur.execute((ROOT/'migrations/079_supplier_lot_tickets.sql').read_text())
        cur.execute("SELECT name FROM migration_markers WHERE name=ANY(%s)",
                    (['079_supplier_lot_tickets', '075_kosher_tiers', '077_key_sessions'],))
        assert len(cur.fetchall()) == 3, 'Staging prerequisites missing'

    os.environ.update(DATABASE_URL=uri, ENVIRONMENT='staging', PRODUCTION_DATABASE_HOST=PRODUCTION_DATABASE_HOST,
        API_KEY=secrets.token_urlsafe(40), DASHBOARD_API_KEY=secrets.token_urlsafe(40),
        KEY_SESSIONS_ENABLED='1', PIN_LOGIN_ENABLED='0')
    logging.disable(logging.CRITICAL)
    import main
    main.get_db_connection = connection
    main._reset_actor_cache()
    reference = 'STG-A5P2-' + uuid4().hex[:10].upper()
    actor_key = secrets.token_urlsafe(40)
    actor_id = None
    http = TestClient(main.app)  # no lifespan: no startup migrations/sweeps
    try:
        with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES (%s,'office',%s,true) RETURNING id",
                        (reference + ' office', sha256(actor_key.encode()).hexdigest()))
            actor_id = cur.fetchone()['id']
            cur.execute("INSERT INTO products(name,odoo_code,type,uom,active,is_service) VALUES (%s,%s,'ingredient','lb',true,false) RETURNING id",
                        (reference + ' synthetic ingredient', reference))
            product_id = cur.fetchone()['id']
            assert product_id >= 1_000_000_000
            cur.execute('INSERT INTO suppliers(name,active) VALUES (%s,true) RETURNING id', (reference + ' supplier',))
            supplier_id = cur.fetchone()['id']

        def post(path, body, key=None, status=200):
            r = http.post(path, json=body, headers={'X-API-Key': key} if key else {})
            if r.status_code != status:
                # No response, body, URL token or connection details in failures.
                raise RuntimeError(f'Unexpected HTTP status {r.status_code}; expected {status}')
            return r.json()

        session = post('/auth/session', {'actor_key': actor_key})['session_token']
        def commit(draft):
            return post('/tickets/' + draft['ticket'] + '/commit', {
                'payload_hash': draft['payload_hash'],
                'acknowledged_warnings': [w['code'] for w in draft['warnings'] if w.get('requires_ack')]}, session)

        receive_body = {'product_id': product_id, 'supplier_id': supplier_id, 'cases': 1,
            'case_size_lb': 2, 'bol_reference': reference, 'lot_code': reference,
            'supplier_lot_code': 'UNKNOWN', 'client_source': 'dashboard'}
        prepared = post('/receive/prepare', receive_body, session)
        assert prepared['can_commit']
        received = commit(prepared)
        assert received['identity_status'] == 'unidentified' and received['exception_id']
        lot_id, exception_id = received['lot_id'], received['exception_id']
        with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT status,resolved_at,due_at FROM exceptions WHERE id=%s', (exception_id,))
            before = dict(cur.fetchone())
        # A further UNKNOWN receive posts stock but retains the same exception clock.
        top_up = commit(post('/receive/prepare', receive_body, session))
        assert top_up['exception_id'] == exception_id and top_up['identity_status'] == 'unidentified'
        with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT status,resolved_at,due_at FROM exceptions WHERE id=%s', (exception_id,))
            assert dict(cur.fetchone()) == before
        code = reference + '-SUP-8812'
        for invalid in ('', 'UNKNOWN', '???'):
            rejected = post(f'/lots/{lot_id}/supplier-lot/prepare',
                {'supplier_lot_code': invalid, 'reason_code': 'data_entry_error'}, session, 422)
            assert rejected['detail']['error_code'] == 'SUPPLIER_LOT_CODE_INVALID'
        draft = post(f'/lots/{lot_id}/supplier-lot/prepare', {'supplier_lot_code': code,
            'reason_code': 'data_entry_error', 'note': 'STAGING ACCEPTANCE: supplier label recovered.',
            'client_source': 'dashboard'}, session)
        assert draft['can_commit'] and draft['draft']['exception_id'] == exception_id
        corrected = commit(draft)
        assert corrected['resolved_exception_ids'] == [exception_id]
        assert corrected['entered_by_actor_id'] == actor_id
        assert commit(draft) == corrected | {'replayed': True}
        detail = http.get('/receipts/' + corrected['receipt_number'], headers={'X-API-Key': session})
        assert detail.status_code == 200 and detail.json()['response'] == corrected
        with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT supplier_lot_code,identity_status,identify_by FROM lots WHERE id=%s', (lot_id,))
            after = dict(cur.fetchone())
            assert after == {'supplier_lot_code': code, 'identity_status': 'identified', 'identify_by': None}
            cur.execute('''SELECT status,resolution_kind,resolution_ticket_id,resolved_by_actor_id,resolved_at
                           FROM exceptions WHERE id=%s''', (exception_id,))
            exception = dict(cur.fetchone())
            assert exception['status'] == 'resolved' and exception['resolution_kind'] == 'identified'
            assert exception['resolution_ticket_id'] == corrected['ticket_id']
            assert exception['resolved_by_actor_id'] == actor_id
            cur.execute('SELECT count(*) AS n FROM lot_events WHERE ticket_id=%s', (corrected['ticket_id'],))
            assert cur.fetchone()['n'] == 1
            balance = float(main.lot_on_hand(cur, lot_id))
            assert balance == 4
        assert post('/auth/session/logout', {}, session)['signed_out'] is True
        return {'reference': reference, 'mode': 'local branch HTTP routes against STAGING; no hosted deployment',
            'migration': '079_supplier_lot_tickets', 'product_id': product_id, 'supplier_id': supplier_id,
            'lot_id': lot_id, 'lot_code': reference, 'exception_id': exception_id, 'actor_id': actor_id,
            'key_session': True, 'actor_deactivated_after_check': True,
            'unidentified_receive': received['receipt_number'], 'top_up_receive': top_up['receipt_number'],
            'top_up_preserved_exception': True, 'correction_receipt': corrected['receipt_number'],
            'correction_ticket_id': corrected['ticket_id'], 'lot_event_id': corrected['lot_event_id'],
            'before': {'supplier_lot_code': 'UNKNOWN', 'identity_status': 'unidentified', 'exception_status': before['status']},
            'after': after, 'exception': exception, 'replayed_once': True, 'lot_event_count': 1,
            'stock_after_top_up_and_correction_lb': balance}
    finally:
        http.close()
        if actor_id is not None:
            with connection() as conn, conn.cursor() as cur:
                cur.execute('UPDATE actors SET active=false WHERE id=%s', (actor_id,))
        main._reset_actor_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply-migration', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check(args.apply_migration)
        args.output.write_text(json.dumps(result, indent=2, default=str) + '\n')
        print(json.dumps(result, indent=2, default=str))
    except Exception as exc:
        # Credential-bearing driver exceptions and tracebacks must never print.
        print('A5 staging acceptance failed: ' + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
