#!/usr/bin/env python3
"""Branch HTTP against guarded staging; no startup hooks or hosted deployment.

Only additive 080 is applied with --apply-migration. Synthetic receipts remain
as evidence. Credentials stay in memory; synthetic actors are deactivated.
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
        cur.execute("SELECT name FROM migration_markers WHERE name=ANY(%s)",
                    (['079_supplier_lot_tickets', '075_kosher_tiers', '077_key_sessions'],))
        assert len(cur.fetchall()) == 3, 'Staging prerequisites missing'
        if apply_migration:
            cur.execute((ROOT/'migrations/080_resolution_receipts.sql').read_text())
        cur.execute("SELECT 1 FROM migration_markers WHERE name='080_resolution_receipts'")
        assert cur.fetchone(), '080 is required'

    os.environ.update(DATABASE_URL=uri, ENVIRONMENT='staging', PRODUCTION_DATABASE_HOST=PRODUCTION_DATABASE_HOST,
        API_KEY=secrets.token_urlsafe(40), DASHBOARD_API_KEY=secrets.token_urlsafe(40),
        KEY_SESSIONS_ENABLED='1', PIN_LOGIN_ENABLED='0')
    logging.disable(logging.CRITICAL)
    import main
    main.get_db_connection = connection
    main._reset_actor_cache()
    reference = 'STG-XR-' + uuid4().hex[:10].upper()
    actor_ids = []
    actor_keys = {role: secrets.token_urlsafe(40) for role in ('floor', 'owner')}
    http = TestClient(main.app)  # Deliberately no lifespan/startup migrations or sweeps.
    try:
        with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            for role, key in actor_keys.items():
                cur.execute('INSERT INTO actors(name,role,key_hash,active) VALUES (%s,%s,%s,true) RETURNING id',
                            (reference+' '+role, role, sha256(key.encode()).hexdigest()))
                actor_ids.append(cur.fetchone()['id'])
            products = {}
            for name, kind, unit, weight in [('ingredient','ingredient','lb',15), ('batch','batch','lb',15),
                                            ('case','finished','case',15)]:
                cur.execute('''INSERT INTO products(name,odoo_code,type,uom,default_batch_lb,case_size_lb,active,is_service)
                    VALUES (%s,%s,%s,%s,10,%s,true,false) RETURNING id''',
                    (reference+' '+name, reference+'-'+name, kind, unit, weight))
                products[name] = cur.fetchone()['id']
                assert products[name] >= 1_000_000_000
            cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES (%s,%s,10)',
                        (products['batch'], products['ingredient']))
            cur.execute('INSERT INTO suppliers(name,active) VALUES (%s,true) RETURNING id', (reference+' supplier',))
            supplier_id = cur.fetchone()['id']

        def post(path, body, key, status=200):
            r = http.post(path, json=body, headers={'X-API-Key': key})
            if r.status_code != status:
                raise RuntimeError(f'Unexpected HTTP status {r.status_code}; expected {status}')
            return r.json()

        def get(path, key):
            r = http.get(path, headers={'X-API-Key': key})
            assert r.status_code == 200, 'Readback failed'
            return r.json()

        sessions = {role: post('/auth/session', {'actor_key': key}, key)['session_token'] for role,key in actor_keys.items()}
        floor, owner = sessions['floor'], sessions['owner']
        def commit(draft, key=floor, status=200):
            return post('/tickets/'+draft['ticket']+'/commit', {'payload_hash': draft['payload_hash'],
                'acknowledged_warnings': [w['code'] for w in draft['warnings'] if w.get('requires_ack')]}, key, status)

        def receive(product, quantity, code):
            return commit(post('/receive/prepare', {'product_id': product, 'supplier_id': supplier_id,
                'cases': 1, 'case_size_lb': quantity, 'bol_reference': reference, 'lot_code': code,
                'supplier_lot_code': reference+'-supplier', 'client_source': 'dashboard'}, floor))

        received = receive(products['ingredient'], 4, reference+'-ingredient')
        made = commit(post('/make/prepare', {'product_id': products['batch'], 'batches': 1,
            'lot_confirmations': [{'lot_id': received['lot_id'], 'method': 'full_code', 'value': received['lot_code']}],
            'client_source': 'dashboard'}, floor))
        eid = made['shortages'][0]['exception_id']
        assert get(f'/exceptions/{eid}', floor)['allowed_actions'] == ['counted','missing_movement','voided']
        draft = post(f'/exceptions/{eid}/resolve/prepare', {'resolution_kind':'counted', 'counted_lb':4,
            'note':'STAGING ACCEPTANCE: counted four pounds on the synthetic pallet.', 'client_source':'dashboard'}, floor)
        resolved = commit(draft)
        assert commit(draft) == resolved | {'replayed': True}
        xr, adj = resolved['receipt_number'], resolved['adjustment_receipt_number']
        assert xr.startswith('XR-') and adj.startswith('ADJ-')
        view = get(f'/exceptions/{eid}', floor)
        assert view['resolution_receipt_number'] == xr and view['resolution_posting_receipt_number'] == adj
        assert view['receipt_number'] == made['receipt_number'] and view['allowed_actions'] == []
        assert get('/receipts/'+xr, floor)['response'] == resolved
        adjustment = get('/receipts/'+adj, floor)
        assert adjustment['response']['new_balance_lb'] == 4

        cases = receive(products['case'], 1000, reference+'-case')
        case_draft = post('/adjust/prepare', {'lot_id':cases['lot_id'], 'delta_lb':-40,
            'reason_code':'physical_count', 'client_source':'dashboard'}, floor)
        assert case_draft['draft']['correction_review']['lb_equivalent'] == -40
        assert not case_draft['draft']['correction_review']['photo_required']
        case_adjusted = commit(case_draft)
        vd = commit(post(f"/void/{case_adjusted['transaction_id']}/prepare", {'reason':'STAGING ACCEPTANCE: void synthetic correction'}, floor))
        original = get('/receipts/'+case_adjusted['receipt_number'], floor)
        assert original['effective_status'] == 'voided' and original['void_receipt_number'] == vd['receipt_number']
        assert get('/receipts/'+vd['receipt_number'], floor)['response'] == vd

        held = commit(post('/adjust/prepare', {'lot_id':cases['lot_id'], 'delta_lb':600,
            'reason_code':'physical_count'}, floor), status=202)
        approved = post(f"/exceptions/{held['exception_id']}/approve", {'note':''}, owner)
        assert approved['resolution_receipt_number'].startswith('XR-')
        assert post(f"/exceptions/{held['exception_id']}/approve", {}, owner) == approved | {'replayed':True}
        return {'environment':'staging', 'mode':'local_branch_http_against_guarded_staging',
            'reference':reference, 'migration':'080_resolution_receipts', 'production_access':False,
            'counted_shortage': {'exception_id':eid, 'source_receipt_number':made['receipt_number'],
                'adjustment_receipt_number':adj, 'resolution_receipt_number':xr,
                'book_before_lb':-6, 'counted_lb':4, 'delta_lb':10, 'final_balance_lb':4, 'replay_verified':True},
            'case_unit': {'case_weight_lb':15, 'delta_lb':-40, 'lb_equivalent':-40, 'held':False,
                          'receipt_number':case_adjusted['receipt_number']},
            'void': {'receipt_number':vd['receipt_number'], 'original_receipt_number':case_adjusted['receipt_number'],
                     'backlink_verified':True},
            'blank_approval': {'posting_receipt_number':approved['receipt_number'],
                              'resolution_receipt_number':approved['resolution_receipt_number'], 'replay_verified':True}}
    finally:
        if actor_ids:
            with connection() as conn, conn.cursor() as cur:
                cur.execute('UPDATE actors SET active=false WHERE id=ANY(%s)', (actor_ids,))
        http.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply-migration', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check(args.apply_migration)
        result['synthetic_actors_deactivated'] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print('Staging acceptance failed: '+type(exc).__name__+' (details withheld to protect credentials)', file=sys.stderr)
        sys.exit(1)
