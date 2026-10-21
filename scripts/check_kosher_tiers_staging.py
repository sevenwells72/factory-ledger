#!/usr/bin/env python3
"""A12 retained acceptance via branch HTTP routes against guarded staging.

Never deploys a service, runs a lifespan, reads production, or prints credentials.
Synthetic owner approval demonstrates the flow; it does not claim Michael
attested a real batch. Temporary actor credentials live in memory and are retired.
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
from urllib.parse import unquote,urlsplit
from uuid import uuid4
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from staging_safety import assert_staging_database,PRODUCTION_DATABASE_HOST
from scripts.seed_staging import secret_file
MIGRATION='075_kosher_tiers'
STAGING_REF='jygmyvxnxdjiiilhxseq'

def check(apply_migration=False):
    uri=secret_file(Path.home()/'Documents/fl-secrets/staging-db-url.txt')
    os.environ['STAGING_DATABASE_HOST']='aws-0-us-east-1.pooler.supabase.com'
    os.environ['STAGING_DATABASE_PROJECT_REF']=STAGING_REF
    assert_staging_database(uri,'staging',PRODUCTION_DATABASE_HOST)
    parts=urlsplit(uri)
    if parts.hostname!='aws-0-us-east-1.pooler.supabase.com' or unquote(parts.username or '')!='postgres.'+STAGING_REF:
        raise RuntimeError('Not the documented staging database')

    @contextmanager
    def connection():
        conn=psycopg2.connect(uri,port=5432,connect_timeout=10,sslmode='require')
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute('SET LOCAL search_path=public')
                    cur.execute("SET LOCAL lock_timeout='5s'")
                    cur.execute("SET LOCAL statement_timeout='60s'")
                yield conn
        finally:
            conn.close()

    with connection() as conn,conn.cursor() as cur:
        if apply_migration:
            cur.execute((ROOT/'migrations'/f'{MIGRATION}.sql').read_text())
        cur.execute('SELECT applied_at FROM migration_markers WHERE name=%s',(MIGRATION,))
        marker=cur.fetchone()
        if not marker:
            raise RuntimeError('Apply 075 to staging before acceptance')
        applied_at=marker[0]
    os.environ['DATABASE_URL']=uri
    os.environ['ENVIRONMENT']='staging'
    os.environ['PRODUCTION_DATABASE_HOST']=PRODUCTION_DATABASE_HOST
    os.environ['API_KEY']=secrets.token_urlsafe(40)
    os.environ['DASHBOARD_API_KEY']=secrets.token_urlsafe(40)
    logging.disable(logging.CRITICAL)
    import main
    main.get_db_connection=connection
    main._reset_actor_cache()
    reference='STG-A12-'+uuid4().hex[:12].upper()
    keys={role:secrets.token_urlsafe(40) for role in ('floor','owner')}
    ids={}
    client=TestClient(main.app) # deliberately no lifespan context
    try:
        with connection() as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
            for role,key in keys.items():
                cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES (%s,%s,%s,true) RETURNING id",(reference+' '+role,role,sha256(key.encode()).hexdigest()))
                ids[role]=cur.fetchone()['id']
            products={}
            for label,kind,tier in [('ingredient','ingredient','regular'),('batch','batch','extra_kosher'),('extra_finished','finished','extra_kosher'),('regular_finished','finished','regular')]:
                cur.execute("""INSERT INTO products(name,odoo_code,type,uom,default_batch_lb,case_size_lb,active,kosher_tier,is_service)
                    VALUES (%s,%s,%s,'lb',20,5,true,%s,false) RETURNING id""",(reference+' '+label,reference+'-'+label,kind,tier))
                products[label]=cur.fetchone()['id']
                assert products[label]>=1_000_000_000
            cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES (%s,%s,20)',(products['batch'],products['ingredient']))
            cur.execute("UPDATE products SET parent_batch_product_id=%s WHERE id=ANY(%s)",(products['batch'],[products['extra_finished'],products['regular_finished']]))
            cur.execute("INSERT INTO lots(product_id,lot_code,entry_source) VALUES (%s,%s,'found_inventory') RETURNING id,lot_code",(products['ingredient'],reference+'-ING'))
            ingredient=dict(cur.fetchone())
            cur.execute("INSERT INTO transactions(type,notes) VALUES ('receive',%s) RETURNING id",(reference+' ingredient fixture',))
            cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,100)',(cur.fetchone()['id'],products['ingredient'],ingredient['id']))

        def headers(role):
            return {'X-API-Key':keys[role]}
        def post(path,payload,role='floor',status=200):
            result=client.post(path,json=payload,headers=headers(role))
            if result.status_code!=status:
                raise RuntimeError(f'Unexpected HTTP {result.status_code}; expected {status}')
            return result.json()
        def commit(draft,status=200):
            return post('/tickets/'+draft['ticket']+'/commit',{'payload_hash':draft['payload_hash'],'acknowledged_warnings':[w['code'] for w in draft['warnings'] if w.get('requires_ack')]},status=status)

        prepared=post('/make/prepare',{'product_id':products['batch'],'batches':1,'confirmed_sku':True,
            'lot_confirmations':[{'lot_id':ingredient['id'],'method':'full_code','value':ingredient['lot_code']}]})
        assert prepared['can_commit']
        held=commit(prepared,202)
        assert held['held'] and held['posted'] is False and held['success'] is False
        with connection() as conn,conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM transactions WHERE ticket_id=%s',(prepared['ticket_id'],))
            assert cur.fetchone()[0]==0
        denied=post(f"/exceptions/{held['exception_id']}/approve",{},status=403)
        approved=post(f"/exceptions/{held['exception_id']}/approve",{},'owner')
        assert approved['receipt_number']==held['receipt_number']
        assert approved['kosher_attested_by_actor_id']==ids['owner']
        assert commit(prepared)['replayed']
        pack=post('/pack/prepare',{'source_product_id':products['batch'],'target_product_id':products['extra_finished'],'cases':1,
            'lot_confirmations':[{'lot_id':approved['lot_id'],'method':'full_code','value':approved['lot_code']}]})
        assert pack['can_commit'] and pack['draft']['kosher_tier']=='extra_kosher'
        packed=commit(pack)
        assert packed['kosher_tier']=='extra_kosher' and not packed['transaction_kosher_downgrade']
        downgrade=post('/pack/prepare',{'source_product_id':products['batch'],'target_product_id':products['regular_finished'],'cases':1,
            'lot_confirmations':[{'lot_id':approved['lot_id'],'method':'full_code','value':approved['lot_code']}]})
        downgraded=commit(downgrade)
        assert downgraded['transaction_kosher_downgrade'] and downgraded['kosher_tier']=='regular'
        declined=post('/make/prepare',{'product_id':products['batch'],'batches':1,'confirmed_sku':True,
            'lot_confirmations':[{'lot_id':ingredient['id'],'method':'full_code','value':ingredient['lot_code']}]})
        rejected_hold=commit(declined,202)
        rejected=post(f"/exceptions/{rejected_hold['exception_id']}/reject",{'resolution_kind':'declined','note':'Synthetic staging rejection; procedure not followed'},'owner')
        assert rejected['posted'] is False
        with connection() as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT entered_by_actor_id,kosher_attested_by_actor_id,kosher_attested_at FROM transactions WHERE id=%s',(approved['transaction_id'],))
            evidence=dict(cur.fetchone())
            assert evidence['entered_by_actor_id']==ids['floor'] and evidence['kosher_attested_by_actor_id']==ids['owner']
            cur.execute('SELECT actor_id,attested_at FROM ticket_attestations WHERE ticket_id=%s',(prepared['ticket_id'],))
            attestation=dict(cur.fetchone())
            assert attestation['attested_at']==evidence['kosher_attested_at']
            cur.execute('SELECT count(*) AS n FROM transactions WHERE ticket_id=%s',(declined['ticket_id'],))
            assert cur.fetchone()['n']==0
            cur.execute("SELECT id,lot_id,event FROM lot_events WHERE transaction_id=%s",(downgraded['transaction_id'],))
            downgrade_event=dict(cur.fetchone())
            cur.execute('SELECT relname,relrowsecurity FROM pg_class WHERE oid IN (%s::regclass,%s::regclass)',('ticket_attestations','lot_events'))
            assert all(r['relrowsecurity'] for r in cur.fetchall())
        return {'reference':reference,'mode':'branch HTTP via local TestClient against guarded STAGING; no hosted deployment',
            'migration':MIGRATION,'migration_applied_now':apply_migration,'migration_applied_at':applied_at,
            'extra_kosher_make':{'ticket_id':prepared['ticket_id'],'held':held,'approved_receipt':approved['receipt_number'],
                'transaction_id':approved['transaction_id'],'lot_id':approved['lot_id'],'lot_code':approved['lot_code'],'attestation':evidence},
            'extra_kosher_pack':{'receipt':packed['receipt_number'],'transaction_id':packed['transaction_id'],'lot_id':packed['output_lot_id'],'lot_code':packed['output_lot_code'],'kosher_tier':packed['kosher_tier'],'cases':1,'lb':5},
            'downgrade':{'receipt':downgraded['receipt_number'],'transaction_id':downgraded['transaction_id'],'flag':True,'lot_event':downgrade_event},
            'rejection':{'ticket_id':declined['ticket_id'],'exception_id':rejected_hold['exception_id'],'posted':False},
            'denied_floor_approval':denied['detail']['error_code'],'actors':ids,'products':products,
            'synthetic_owner_not_real_batch_attestation':True,'actors_deactivated_after_check':True}
    finally:
        client.close()
        if ids:
            with connection() as conn,conn.cursor() as cur:
                cur.execute('UPDATE actors SET active=false WHERE id=ANY(%s)',(list(ids.values()),))
        main._reset_actor_cache()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply-migration',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=check(args.apply_migration)
        args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str)+'\n')
        print(json.dumps(result,indent=2,ensure_ascii=False,default=str))
    except Exception as exc:
        # Never print libpq errors or traceback strings containing a URI/key.
        print('A12 staging acceptance failed: '+type(exc).__name__,file=sys.stderr)
        raise SystemExit(1)
