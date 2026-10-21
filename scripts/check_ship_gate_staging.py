#!/usr/bin/env python3
"""A6 acceptance through local HTTP against the isolated staging database.

Never deploys, opens production, or runs startup. Reads only the protected
staging URI; evidence uses an explicit private persistent filesystem directory.
Synthetic receipts/evidence stay in staging; temporary actor keys stay in memory
and the actors are deactivated in finally. --apply-migration is staging only.
"""
import argparse
import io
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from staging_safety import assert_staging_database,PRODUCTION_DATABASE_HOST
from scripts.seed_staging import secret_file

def check(apply_migration=False):
    uri=secret_file(Path.home()/'Documents/fl-secrets/staging-db-url.txt')
    assert_staging_database(uri,'staging',PRODUCTION_DATABASE_HOST)
    if urlsplit(uri).hostname!='aws-0-us-east-1.pooler.supabase.com':
        raise RuntimeError('Not the documented staging database')
    volume=os.environ.get('EVIDENCE_STORAGE_DIR')
    if not volume or not Path(volume).is_absolute():
        raise RuntimeError('Explicit private persistent EVIDENCE_STORAGE_DIR is required')

    @contextmanager
    def connection():
        conn=psycopg2.connect(uri,port=5432,connect_timeout=10,sslmode='require')
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL search_path=public; SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='60s'")
                yield conn
        finally:conn.close()

    with connection() as conn,conn.cursor() as cur:
        if apply_migration:cur.execute((ROOT/'migrations/074_ship_gate_attachments.sql').read_text())
        cur.execute("SELECT 1 FROM migration_markers WHERE name='074_ship_gate_attachments'")
        if not cur.fetchone():raise RuntimeError('Apply staging migration 074 before acceptance')

    os.environ.update(DATABASE_URL=uri,ENVIRONMENT='staging',PRODUCTION_DATABASE_HOST=PRODUCTION_DATABASE_HOST,
                      API_KEY=secrets.token_urlsafe(40),DASHBOARD_API_KEY=secrets.token_urlsafe(40))
    logging.disable(logging.CRITICAL)
    import main
    main.get_db_connection=connection
    main._reset_actor_cache()
    ref='STG-A6-'+uuid4().hex[:12].upper()
    keys={r:secrets.token_urlsafe(40) for r in ('office','floor')}
    actor_ids={}
    # A readable synthetic BOL image retained as real private staging evidence.
    from PIL import Image,ImageDraw
    img=Image.new('RGB',(800,600),'white');draw=ImageDraw.Draw(img)
    text='STAGING TEST — BILL OF LADING\n\n'+ref+'-BOL\nCustomer: '+ref+' customer\nProduct: '+ref+' granola\nQuantity: 2 cases / 50 lb\nCarrier: STAGING TEST TRUCK\n\nSynthetic acceptance evidence only'
    draw.multiline_text((40,40),text,fill='black',spacing=15)
    buf=io.BytesIO();img.save(buf,format='PNG');photo=buf.getvalue()

    try:
        with connection() as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
            for role,key in keys.items():
                cur.execute('INSERT INTO actors(name,role,key_hash,active) VALUES(%s,%s,%s,true) RETURNING id',
                            (ref+' '+role,role,sha256(key.encode()).hexdigest()));actor_ids[role]=cur.fetchone()['id']
            cur.execute('INSERT INTO customers(name,active) VALUES(%s,true) RETURNING id',(ref+' customer',));customer=cur.fetchone()['id']
            cur.execute('INSERT INTO suppliers(name,active) VALUES(%s,true) RETURNING id',(ref+' supplier',));supplier=cur.fetchone()['id']
            cur.execute("INSERT INTO products(name,odoo_code,type,uom,case_size_lb,active,is_service) VALUES(%s,%s,'finished','lb',25,true,false) RETURNING id",(ref+' granola',ref));product=cur.fetchone()['id']
            if min(customer,supplier,product)<1_000_000_000:raise RuntimeError('Staging synthetic ID range is required')
        http=TestClient(main.app)
        def post(path,payload,role='office',status=200):
            r=http.post(path,json=payload,headers={'X-API-Key':keys[role]})
            if r.status_code!=status:raise RuntimeError(f'Unexpected HTTP {r.status_code} on {path}')
            return r.json()
        def commit(p,role='office',status=200,**extra):
            return post('/tickets/'+p['ticket']+'/commit',{'payload_hash':p['payload_hash'],
                 'acknowledged_warnings':[w['code'] for w in p['warnings'] if w.get('requires_ack')],**extra},role,status)

        received=commit(post('/receive/prepare',{'product_id':product,'supplier_id':supplier,'cases':4,'case_size_lb':25,
                 'bol_reference':ref+'-IN','lot_code':ref+'-LOT','supplier_lot_code':ref+'-SUP'}))
        order=commit(post('/sales/orders/prepare',{'customer_id':customer,'customer_po':ref+'-PO',
                     'lines':[{'product_id':product,'quantity':2,'unit':'cases','unit_price':1}]}))
        oid=order['order_id'];lid=received['lot_id'];bol=ref+'-BOL'
        b={'ship_all':True,'bol_reference':bol,'lot_confirmations':[{'lot_id':lid,'method':'full_code','value':ref+'-LOT'}]}
        missing=post(f'/sales/orders/{oid}/ship/prepare',b|{'bol_reference':''},'floor')
        refused=commit(missing,'floor',422)
        if refused['detail']['error_code']!='BOL_REQUIRED':raise RuntimeError('BOL hard gate failed')
        r=http.post('/attachments',data={'kind':'bol'},files={'file':('staging-bol.png',photo,'image/png')},headers={'X-API-Key':keys['floor']})
        if r.status_code!=200:raise RuntimeError('Real staging photo upload failed')
        attachment=r.json()
        if http.get(f"/attachments/{attachment['id']}/file").status_code!=401:raise RuntimeError('Private file must reject anonymous access')
        r=http.get(f"/attachments/{attachment['id']}/file",headers={'X-API-Key':keys['floor']})
        if r.status_code!=200 or r.content!=photo:raise RuntimeError('Stored photo roundtrip failed')
        p=post(f'/sales/orders/{oid}/ship/prepare',b|{'attachment_ids':[attachment['id']]},'floor')
        if not p['can_commit'] or p['draft']['photo']!='attached':raise RuntimeError('Ship draft evidence failed')
        shipped=commit(p,'floor');replayed=commit(p,'floor')
        if shipped['receipt_number']!=replayed['receipt_number'] or not replayed['replayed']:raise RuntimeError('Ship replay failed')
        slip=http.get(shipped['packing_slip_url'])
        if slip.status_code!=200 or not slip.content.startswith(b'%PDF'):raise RuntimeError('Signed packing slip failed')
        with connection() as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('''SELECT so.order_number,so.status,s.id AS shipment_id,s.bol_reference,s.proof_status,s.proof_due_at,
                                  s.entered_by_actor_id,s.created_at,t.receipt_number,t.action
                           FROM shipments s JOIN sales_orders so ON so.id=s.sales_order_id
                           JOIN write_tickets t ON t.id=s.ticket_id WHERE s.id=%s''',(shipped['shipment_id'],));record=dict(cur.fetchone())
            cur.execute('SELECT id,kind,sha256,bytes,uploaded_by_actor_id,ticket_id,shipment_id FROM attachments WHERE id=%s',(attachment['id'],));evidence=dict(cur.fetchone())
            cur.execute('SELECT count(*) AS n FROM shipments WHERE ticket_id=%s',(p['ticket_id'],));count=cur.fetchone()['n']
            if record['status']!='shipped' or record['proof_status']!='complete' or count!=1:raise RuntimeError('Staging shipment post-condition failed')
            if evidence['shipment_id']!=record['shipment_id'] or evidence['sha256']!=sha256(photo).hexdigest():raise RuntimeError('Staging evidence link failed')
        return {'environment':'staging','transport':'local TestClient; no hosted service deployment','migration':'074_ship_gate_attachments',
                'synthetic_reference':ref,'order_id':oid,'ticket_id':p['ticket_id'],'quantity_shipped_lb':50,
                'shipment':record,'attachment':evidence,'checks':{'bol_required':True,'private_real_photo_roundtrip':True,
                 'one_shipment_and_replay':True,'signed_packing_slip_pdf':True},'temporary_actors_deactivated':True}
    finally:
        if actor_ids:
            with connection() as conn,conn.cursor() as cur:cur.execute('UPDATE actors SET active=false WHERE id=ANY(%s)',(list(actor_ids.values()),))
        main._reset_actor_cache()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply-migration',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:
        result=check(args.apply_migration);out=json.dumps(result,indent=2,default=str)+'\n'
        if args.output:args.output.write_text(out)
        print(out,end='')
    except Exception as exc:
        # Never expose credential-bearing driver/HTTP exception text or a DSN.
        print(f'A6 staging acceptance failed ({type(exc).__name__})',file=sys.stderr);sys.exit(1)
