#!/usr/bin/env python3
"""Real Chrome + branch HTTP against guarded staging; no startup or deployment."""
from contextlib import contextmanager
from hashlib import sha256
import json, logging, os, re, secrets, subprocess, sys, threading, time
from pathlib import Path
from uuid import uuid4
import httpx
import uvicorn
from psycopg2.extras import RealDictCursor
from check_assistant_staging import connect, BASE, ROOT
from staging_safety import PRODUCTION_DATABASE_HOST
sys.path.insert(0,str(ROOT))

def check():
    marker='STG-D2OFF-'+uuid4().hex[:8].upper()
    keys={role:secrets.token_urlsafe(40) for role in ('office','floor')}
    actors=[]
    server=None
    fixture={'marker':marker,'base':'http://127.0.0.1:8130','staging_origin':BASE,'products':{}}
    with httpx.Client(timeout=30) as http:
        assert http.get(BASE+'/health').status_code==200
        assert http.get(BASE+'/auth/session/config').json()['key_sessions_enabled'] is True
    @contextmanager
    def connection():
        conn=connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL search_path=public; SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='60s'")
                yield conn
        finally: conn.close()
    try:
        with connection() as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
            wanted=['067_order_tickets','072_order_ticket_review_fixes','077_key_sessions','080_resolution_receipts']
            cur.execute('SELECT name FROM migration_markers WHERE name=ANY(%s)',(wanted,))
            assert {r['name'] for r in cur.fetchall()}==set(wanted),'Missing staging prerequisites'
            for role,key in keys.items():
                cur.execute('INSERT INTO actors(name,role,key_hash,active) VALUES(%s,%s,%s,true) RETURNING id',
                            (marker+' '+role,role,sha256(key.encode()).hexdigest()))
                actors.append(cur.fetchone()['id'])
            for label,kind,uom,weight,service in [('Cases','finished','case',25,False),('Pallet Charge','finished','each',None,True),('Ingredient','ingredient','lb',None,False)]:
                cur.execute('''INSERT INTO products(name,odoo_code,type,uom,case_size_lb,is_service,active)
                               VALUES(%s,%s,%s,%s,%s,%s,true) RETURNING id,name,odoo_code''',
                    (marker+' '+label,marker+'-'+label.replace(' ','-'),kind,uom,weight,service))
                fixture['products'][label]=dict(cur.fetchone())
                assert fixture['products'][label]['id']>=1_000_000_000
            for table,label in [('customers','customer'),('suppliers','supplier')]:
                cur.execute('INSERT INTO '+table+'(name,active) VALUES(%s,true) RETURNING id,name',(marker+' '+label,))
                fixture[label]=dict(cur.fetchone())
        uri=(Path.home()/'Documents/fl-secrets/staging-db-url.txt').read_text().strip()
        readkey=re.search(r"api.readKey='([^']+)'",(ROOT/'dashboard/ledger-ui.js').read_text()).group(1)
        os.environ.update(DATABASE_URL=uri,ENVIRONMENT='staging',PRODUCTION_DATABASE_HOST=PRODUCTION_DATABASE_HOST,
                          API_KEY=secrets.token_urlsafe(40),DASHBOARD_API_KEY=readkey,KEY_SESSIONS_ENABLED='1',PIN_LOGIN_ENABLED='0')
        logging.disable(logging.CRITICAL)
        import main
        main.get_db_connection=connection
        main._reset_actor_cache()
        server=uvicorn.Server(uvicorn.Config(main.app,host='127.0.0.1',port=8130,access_log=False,log_level='critical',lifespan='off'))
        thread=threading.Thread(target=server.run,daemon=True);thread.start()
        for _ in range(100):
            if server.started:break
            time.sleep(.1)
        assert server.started,'Loopback server unavailable'
        (ROOT/'.venv-test/office-void-staging-fixture.json').write_text(json.dumps(fixture))
        env={k:v for k,v in os.environ.items() if k not in ('DATABASE_URL','API_KEY','DASHBOARD_API_KEY')}
        env.update(D2_STAGING_OFFICE_KEY=keys['office'],D2_STAGING_FLOOR_KEY=keys['floor'])
        subprocess.run(['node','tests/visual/run-office-void-staging.mjs'],cwd=ROOT,env=env,check=True)
        evidence_path=ROOT/'docs/validation/office-void-staging/evidence.json'
        evidence=json.loads(evidence_path.read_text())
        with connection() as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
            for item in evidence['results']:
                cur.execute('SELECT count(*) AS n FROM write_tickets WHERE receipt_number=%s AND status=%s',(item['receipt_number'],'committed'))
                assert cur.fetchone()['n']==1
            cur.execute('SELECT count(*) AS n FROM transactions WHERE ticket_id=%s',(evidence['results'][1]['ticket_id'],))
            assert cur.fetchone()['n']==1
            cur.execute('SELECT effective_status FROM ledger_current_transactions WHERE id=%s',(evidence['results'][1]['transaction_id'],))
            assert cur.fetchone()['effective_status']=='voided'
        evidence['direct_database_checks']=['One committed ticket per receipt','One receive post','Source posting voided']
        evidence_path.write_text(json.dumps(evidence,indent=2)+'\n')
    finally:
        if server:
            server.should_exit=True
            thread.join(timeout=10)
        if actors:
            with connection() as conn,conn.cursor() as cur:
                cur.execute('UPDATE actors SET active=false WHERE id=ANY(%s)',(actors,))
                cur.execute("UPDATE key_actor_sessions SET ended_at=clock_timestamp(),ended_reason='revoked' WHERE actor_id=ANY(%s) AND ended_at IS NULL",(actors,))
            print('Temporary staging actors deactivated; no hosted deployment, migrations or production access.')
if __name__=='__main__':
    try:check()
    except Exception:
        print('Staging acceptance failed; inspect sanitized browser screenshot/evidence. Credentials suppressed.',file=sys.stderr)
        raise SystemExit(1)
