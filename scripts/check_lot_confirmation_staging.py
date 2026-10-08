#!/usr/bin/env python3
"""A5 acceptance on guarded staging through local HTTP, never hosted deployment.

Reads only the protected staging URI; no production configuration or API keys.
--apply-migrations explicitly applies 062/063/064/066, in one transaction.
Synthetic stock, recipes, actor and committed receipts are retained; the fresh
actor key exists only in memory and the actor is deactivated in finally.
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

MIGRATIONS = ['062_lot_confirmation.sql', '063_batch_substitutions.sql',
              '064_unidentified_lots.sql', '066_receipt_suppliers.sql']


def check(apply_migrations=False):
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

    if apply_migrations:
        with connection() as conn, conn.cursor() as cur:
            for filename in MIGRATIONS:
                cur.execute((ROOT/'migrations'/filename).read_text())
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT name FROM migration_markers WHERE name=ANY(%s)",([p[:-4] for p in MIGRATIONS],))
        if len(cur.fetchall()) != len(MIGRATIONS):
            raise RuntimeError('Apply the A5 staging migrations before acceptance')

    # Import with explicit staging state, but do not invoke app startup/sweeps.
    # TestClient without a lifespan context executes real routes, not startup.
    os.environ['DATABASE_URL'] = uri
    os.environ['ENVIRONMENT'] = 'staging'
    os.environ['PRODUCTION_DATABASE_HOST'] = PRODUCTION_DATABASE_HOST
    os.environ['API_KEY'] = secrets.token_urlsafe(40)
    os.environ['DASHBOARD_API_KEY'] = secrets.token_urlsafe(40)
    logging.disable(logging.CRITICAL)
    import main
    main.get_db_connection = connection
    main._reset_actor_cache()
    reference = 'STG-A5-' + uuid4().hex[:12].upper()
    actor_key = secrets.token_urlsafe(40)
    actor_id = None
    client = TestClient(main.app)
    try:
        with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES (%s,'floor',%s,true) RETURNING id",
                        (reference,sha256(actor_key.encode()).hexdigest()))
            actor_id = cur.fetchone()['id']
            products = {}
            for label,kind in [('ingredient','ingredient'),('substitute','ingredient'),('batch','batch')]:
                cur.execute("""INSERT INTO products(name,odoo_code,type,uom,default_batch_lb,active)
                    VALUES (%s,%s,%s,'lb',0.10,true) RETURNING id""",(reference+' '+label,reference+'-'+label,kind))
                products[label] = cur.fetchone()['id']
                assert products[label] >= 1_000_000_000
            cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES (%s,%s,0.10)',
                        (products['batch'],products['ingredient']))
        headers = {'X-API-Key': actor_key}

        def post(path,payload,status=200):
            response = client.post(path,json=payload,headers=headers)
            if response.status_code != status:
                raise RuntimeError(f'Unexpected HTTP {response.status_code} on {path.split("/")[1]}')
            return response.json()

        def commit(draft,**extra):
            return post('/tickets/'+draft['ticket']+'/commit',{'payload_hash':draft['payload_hash'],
                'acknowledged_warnings':[w['code'] for w in draft['warnings'] if w.get('requires_ack')],**extra})

        resolved = post('/resolve',{'kind':'supplier','query':'Dutch Gold Honey'})
        # Explicitly select the known named vendor from resolver candidates.
        candidates = [c for c in resolved['candidates'] if c['name']=='Dutch Gold Honey']
        if len(candidates)!=1:
            raise RuntimeError('Expected exactly one Dutch Gold Honey candidate in staging')
        supplier = candidates[0]
        received = {}
        for label in ('ingredient','substitute'):
            draft = post('/receive/prepare',{'product_id':products[label], 'supplier_id':supplier['id'],
                'cases':1,'case_size_lb':2,'bol_reference':reference+'-'+label,
                'supplier_lot_code':reference+'-SUP-'+label})
            assert draft['can_commit']
            received[label] = commit(draft)
            assert received[label]['supplier_id']==supplier['id']

        make = post('/make/prepare',{'product_id':products['batch'],'batches':1,
            'ingredient_lots':[{'ingredient_product_id':products['ingredient'],'lot_id':received['ingredient']['lot_id']}]})
        assert not make['can_commit'] and make['blockers'][0]['code']=='LOT_NOT_CONFIRMED'
        refused = post('/tickets/'+make['ticket']+'/commit',{'payload_hash':make['payload_hash']},422)
        assert refused['detail']['error_code']=='LOT_NOT_CONFIRMED'
        evidence = {'lot_id':received['ingredient']['lot_id'],'method':'last4','value':received['ingredient']['lot_code'][-4:]}
        made = commit(make,lot_confirmations=[evidence])
        replay = commit(make)
        assert replay==made|{'replayed':True}
        response=client.get('/receipts/'+made['receipt_number'],headers=headers)
        assert response.status_code==200
        receipt=response.json()
        source=next(l for l in receipt['lots'] if l['id']==received['ingredient']['lot_id'])
        assert source['supplier_id']==supplier['id'] and source['supplier_name']=='Dutch Gold Honey'
        assert receipt['transactions'][0]['lot_confirmations'][0]['method']=='last4'

        move=post('/lots/'+str(source['id'])+'/move/prepare',{'to_location':'production',
            'method':'full_code','value':source['lot_code']})
        moved=commit(move)
        assert moved['confirmed']

        sub={'ingredient_product_id':products['ingredient'],'substitute_product_id':products['substitute'],
             'lot_id':received['substitute']['lot_id'],'reason_code':'acceptance_trial','note':reference}
        invalid=dict(sub);invalid.pop('reason_code')
        post('/make/prepare',{'product_id':products['batch'],'batches':1,'substitutions':[invalid]},422)
        substitution=post('/make/prepare',{'product_id':products['batch'],'batches':1,'substitutions':[sub],
            'lot_confirmations':[{'lot_id':sub['lot_id'],'method':'full_code','value':received['substitute']['lot_code']}]})
        substituted=commit(substitution)
        assert substituted['substitutions'][0]['reason_code']=='acceptance_trial'

        unknown=post('/receive/prepare',{'product_id':products['ingredient'],'supplier_id':supplier['id'],
            'cases':1,'case_size_lb':0.20,'bol_reference':reference+'-UNIDENTIFIED','supplier_lot_code':'UNKNOWN'})
        unidentified=commit(unknown)
        assert unidentified['identity_status']=='unidentified' and unidentified['exception_id']
        missing=post('/receive/prepare',{'product_id':products['ingredient'],'cases':1,'case_size_lb':1,'bol_reference':reference},422)
        assert missing['detail']['error_code']=='SUPPLIER_REQUIRED'
        with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('''SELECT t.id,t.supplier_id,l.supplier_id AS lot_supplier_id
                FROM transactions t JOIN transaction_lines tl ON tl.transaction_id=t.id
                JOIN lots l ON l.id=tl.lot_id WHERE t.id=%s''',(received['ingredient']['transaction_id'],))
            verified=cur.fetchone()
            assert verified['supplier_id']==verified['lot_supplier_id']==supplier['id']
            cur.execute('SELECT due_at,opened_at FROM exceptions WHERE id=%s',(unidentified['exception_id'],))
            exception=cur.fetchone()
            from lot_confirmation import identification_deadline
            assert exception['due_at']==identification_deadline(exception['opened_at'])
        return {'reference':reference,'mode':'local branch HTTP routes against STAGING; no hosted deployment',
            'migrations_applied':apply_migrations,'supplier_id':supplier['id'],'supplier_name':supplier['name'],
            'receive_receipt':received['ingredient']['receipt_number'], 'make_receipt':made['receipt_number'],
            'confirmed_lot':source['lot_code'],'lot_id':source['id'],'confirmation':evidence,
            'move_receipt':moved['receipt_number'],'substitution_receipt':substituted['receipt_number'],
            'unidentified_receipt':unidentified['receipt_number'],'unidentified_due_at':unidentified['identification_due_at'],
            'actor_id':actor_id,'actor_deactivated_after_check':True,'receipt':receipt}
    finally:
        client.close()
        if actor_id is not None:
            with connection() as conn, conn.cursor() as cur:
                cur.execute('UPDATE actors SET active=false WHERE id=%s',(actor_id,))
        main._reset_actor_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply-migrations',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=check(args.apply_migrations)
        args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='receipt'},indent=2))
    except Exception as exc:
        # Never emit DB connection exception strings or tracebacks with secrets.
        print('A5 staging acceptance failed: '+type(exc).__name__,file=sys.stderr)
        raise SystemExit(1)
