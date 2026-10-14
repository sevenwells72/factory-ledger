#!/usr/bin/env python3
"""Explicit staging-only migrations and real SS #1 substitution acceptance.

Uses protected files/in-memory keys, the repository isolation guard, and the
named FastAPI-staging host. Retains labelled stock/receipts, deactivates its
synthetic actor, never opens production or prints credentials.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import secrets
import sys
from time import perf_counter
from uuid import uuid4

import httpx
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_recipe_corrections_staging import staging_uri, connection
from staging_safety import PRODUCTION_DATABASE_HOST

BASE = 'https://fastapi-staging-production-dd7b.up.railway.app'
OUT = ROOT/'docs/deployments/pilot-triage-staging.json'


def run(apply=False, check=False, hosted=False):
    uri = staging_uri()
    result = {'checked_at_utc':datetime.now(timezone.utc).isoformat(), 'target':'FastAPI-staging', 'migrations':[]}
    if apply:
        for name in ['081_restore_ss_oat_bran.sql','082_order_plant_date.sql']:
            with connection(uri) as conn, conn.cursor() as cur:
                cur.execute((ROOT/'migrations'/name).read_text())
            result['migrations'].append(name)
        with connection(uri) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT product_id,quantity_lb FROM batch_formulas WHERE product_id IN (114,116,118) AND ingredient_product_id=31 ORDER BY product_id')
            result['oat_bran_after'] = [dict(r) for r in cur.fetchall()]
    if not check:
        return result
    os.environ.update(DATABASE_URL=uri,ENVIRONMENT='staging',PRODUCTION_DATABASE_HOST=PRODUCTION_DATABASE_HOST,
                      API_KEY=secrets.token_urlsafe(40),DASHBOARD_API_KEY=secrets.token_urlsafe(40))
    logging.disable(logging.CRITICAL)
    import main
    main.get_db_connection = lambda: connection(uri)
    main._reset_actor_cache()
    from fastapi.testclient import TestClient
    http = httpx.Client(base_url=BASE,timeout=60) if hosted else TestClient(main.app)
    key = secrets.token_urlsafe(40)
    ref = 'STG-TRIAGE-'+uuid4().hex[:10].upper()
    actor_id = None
    result.update(ref=ref, transport='hosted HTTPS' if hosted else 'local TestClient on staging database', receipts=[])
    try:
        with connection(uri) as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT count(*) AS n FROM migration_markers WHERE name IN ('081_restore_ss_oat_bran','082_order_plant_date')")
            assert cur.fetchone()['n']==2, 'Apply both migrations before checking'
            cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES (%s,'floor',%s,true) RETURNING id",(ref,sha256(key.encode()).hexdigest()))
            actor_id = cur.fetchone()['id']
            cur.execute('INSERT INTO suppliers(name) VALUES (%s) RETURNING id',(ref,));supplier = cur.fetchone()['id']
            assert min(actor_id,supplier)>=1_000_000_000, 'Synthetic staging IDs required'
        headers = {'X-API-Key':key}
        def post(path, data, status=200):
            r=http.post(path,json=data,headers=headers)
            if r.status_code!=status:
                raise RuntimeError(f'Staging request failed: HTTP {r.status_code}; path kind {path.split("/")[1]}')
            return r.json()
        def commit(p):
            assert p['can_commit'], [b['code'] for b in p['blockers']]
            value=post('/tickets/'+p['ticket']+'/commit',{'payload_hash':p['payload_hash'],'acknowledged_warnings':[w['code'] for w in p['warnings'] if w.get('requires_ack')]})
            assert value.get('success') and not value.get('held'), 'Expected a posted receipt'
            result['receipts'].append(value['receipt_number'])
            return value
        stock={3:21,13:21,30:50,32:213,34:24,46:21}
        lots={}
        # Six ingredients plus 23 one-pound receipts, then the make = exactly 30 receipts.
        for i,(pid,lb) in enumerate(list(stock.items())+[(32,1)]*23):
            code=f'{ref}-{i:02d}'
            received=commit(post('/receive/prepare',{'product_id':pid,'supplier_id':supplier,'cases':1,'case_size_lb':lb,
                'lot_code':code,'supplier_lot_code':code,'bol_reference':ref,'client_source':'dashboard'}))
            if i<6: lots[pid]={'lot_id':received['lot_id'],'lot_code':code}
            if i%5==4: print(json.dumps({'staging_receipts_recorded':len(result['receipts'])}),flush=True)
        payload={'product_id':116,'batches':1,'confirmed_sku':True,'lot_code':ref+'-SS1','client_source':'dashboard',
                 'ingredient_lots':[{'ingredient_product_id':p,'lot_id':l['lot_id']} for p,l in lots.items()],
                 'lot_confirmations':[{'lot_id':l['lot_id'],'method':'full_code','value':l['lot_code']} for l in lots.values()],
                 'substitutions':[{'ingredient_product_id':31,'substitute_product_id':32,'lot_id':lots[32]['lot_id'],
                                   'reason_code':'ingredient_substitution','note':'STAGING ONLY: approved pilot oat bran -> rolled oats check'}]}
        prepared=post('/make/prepare',payload)
        replacement=[i for i in prepared['draft'].get('ingredients',[]) if i.get('substituted_for')==31]
        assert len(replacement)==1 and replacement[0]['needed_lb']==28, 'Prepare must move 28 lb'
        made=commit(prepared)
        replay=post('/tickets/'+prepared['ticket']+'/commit',{'payload_hash':prepared['payload_hash']})
        assert replay['transaction_id']==made['transaction_id'] and replay['replayed'], 'Replay must not post again'
        with connection(uri) as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT product_id,sum(quantity_lb) AS lb FROM transaction_lines WHERE transaction_id=%s GROUP BY product_id ORDER BY product_id',(made['transaction_id'],))
            moved={r['product_id']:r['lb'] for r in cur.fetchall()}
            assert moved[116]==350 and moved[32]==-213 and moved.get(31,0)==0
            cur.execute('SELECT ingredient_product_id,substitute_product_id,lot_id,reason_code FROM transaction_substitutions WHERE transaction_id=%s',(made['transaction_id'],))
            substitution=[dict(r) for r in cur.fetchall()]
            assert len(substitution)==1 and substitution[0]['ingredient_product_id']==31 and substitution[0]['substitute_product_id']==32
            cur.execute('SELECT count(*) AS n FROM transactions WHERE ticket_id=%s',(prepared['ticket_id'],));assert cur.fetchone()['n']==1
        result['make']={'receipt_number':made['receipt_number'],'ticket_id':prepared['ticket_id'],'transaction_id':made['transaction_id'],
                        'output_lb':350,'oat_bran_consumed_lb':0,'rolled_oats_consumed_lb':213,'substituted_lb':28,
                        'replay_one_post':True,'lines':moved,'substitution':substitution}
        times=[]
        for _ in range(3):
            started=perf_counter();r=http.get('/reports/shift-summary',headers=headers);elapsed=perf_counter()-started
            assert r.status_code==200, 'Summary failed'
            doc=r.json();assert doc['receipts_today']==30, 'Expected 30-receipt day'
            times.append(round(elapsed,3))
        result['summary']={'receipts':30,'seconds':times,'under_3_seconds':all(t<3 for t in times)}
        assert result['summary']['under_3_seconds'], f'30-receipt summary missed target: {times}'
        started=perf_counter()
        confirmed=post('/reports/shift-summary/confirm',{'date':doc['date'],'outcome':'match','notes':'STAGING pilot triage benchmark',
                  'receipts_seen':30,'summary_hash':doc['summary_hash']},status=201)
        result['summary']['confirm_seconds']=round(perf_counter()-started,3)
        result['summary']['confirmation_id']=confirmed.get('confirmation_id')
        return result
    finally:
        if actor_id:
            with connection(uri) as conn,conn.cursor() as cur:
                cur.execute('UPDATE actors SET active=false WHERE id=%s',(actor_id,))
            result['synthetic_actor_deactivated']=actor_id
        http.close()
        OUT.write_text(json.dumps(result,indent=2,default=str)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply-migrations',action='store_true');parser.add_argument('--check',action='store_true');parser.add_argument('--hosted',action='store_true')
    args=parser.parse_args()
    if args.hosted: OUT = ROOT/'docs/deployments/pilot-triage-staging-hosted.json'
    if not (args.apply_migrations or args.check): parser.error('Choose --apply-migrations and/or --check')
    try:
        result=run(args.apply_migrations,args.check,args.hosted)
        print(json.dumps(result,default=str))
    except Exception as exc:
        print('Staging acceptance failed: '+type(exc).__name__)
        raise SystemExit(1) from None
