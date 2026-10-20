"""Pilot UX actual-page bridge. Every successful read/write uses real local FL.
Only synthetic catalog/stock and actors are seeded. OpenAI is hard-refused.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import hashlib
import json
import logging
import os
import sys
import tempfile
import signal
import psycopg2
from psycopg2.extras import RealDictCursor

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
URL = os.environ['TEST_DATABASE_URL']
parsed = urlsplit(URL)
assert parsed.hostname == '127.0.0.1' and parsed.path.startswith('/fl_pilotux_')
os.environ.update(DATABASE_URL=URL, API_KEY='pilotux-master', DASHBOARD_API_KEY='pilotux-read', ASSISTANT_ENABLED='0', ENVIRONMENT='test', EVIDENCE_STORAGE_DIR=tempfile.mkdtemp(prefix='fl_pilotux_evidence_'))
for name in ('OPENAI_API_KEY', 'SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY'):
    os.environ.pop(name, None)
logging.disable(logging.CRITICAL)

@contextmanager
def connection():
    conn = psycopg2.connect(URL)
    try:
        with conn:
            yield conn
    finally:
        conn.close()

with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
    text = (ROOT / 'migrations/061_exceptions_tables.sql').read_text()
    cur.execute(text[text.index('CREATE TABLE IF NOT EXISTS correction_reasons'):text.index('-- 3. transactions.reason_code')])
    cur.execute((ROOT/'migrations/083_customer_tickets.sql').read_text())
    actors = {}
    for role, name in [('floor', 'Arturo Pilot UX'), ('office', 'Luz Pilot UX'), ('owner', 'Owner Pilot UX')]:
        key = 'pilotux-' + role
        cur.execute('INSERT INTO actors(name,role,key_hash,active) VALUES(%s,%s,%s,true) RETURNING id', (name, role, hashlib.sha256(key.encode()).hexdigest()))
        actors[role] = {'id': cur.fetchone()['id'], 'key': key, 'name': name}
    products = {}
    rows = [(11, 'Oats', 'ingredient', '11020', None, None, 'regular'), (12, 'Sunflower Seeds', 'ingredient', '11010', None, None, 'regular'), (13, 'Coconut Macaroon', 'ingredient', '11030', None, None, 'regular'), (21, 'Batch Classic Granola #9', 'batch', '90002', 200, None, 'regular'), (22, 'Batch Fruit Nut Granola', 'batch', '90003', 40, None, 'regular'), (31, 'Granola Classic 25 LB', 'finished', '70013', None, 25, 'regular'), (32, 'Granola Classic 25 LB SS EXTRA-KOSHER', 'finished', '70014', None, 25, 'extra_kosher'), (33, 'Granola Classic 25 LB SS EXTRA-KOSHER ALT', 'finished', '70015', None, 25, 'extra_kosher'), (34, 'Granola Fruit Nut 25 LB', 'finished', '70016', None, 25, 'regular'), (35, 'Sprinkles Rainbow 10 LB', 'finished', '70017', None, 10, 'regular')]
    for pid, name, kind, sku, batch, case, tier in rows:
        cur.execute("INSERT INTO products(id,name,type,odoo_code,uom,default_batch_lb,case_size_lb,active,kosher_tier) VALUES(%s,%s,%s,%s,'lb',%s,%s,true,%s)", (pid,name,kind,sku,batch,case,tier))
        products[name] = pid
    cur.execute('UPDATE products SET parent_batch_product_id=21 WHERE id IN (31,32,33)')
    cur.execute('UPDATE products SET parent_batch_product_id=22 WHERE id=34')
    cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES(21,11,150),(21,12,40),(21,13,10)')
    lots = {}
    for pid, code, qty, age in [(11,'26-10-18-OATS-001',500,3),(11,'26-10-19-OATS-001',500,2),(12,'26-10-18-SUNF-002',500,3),(13,'26-10-18-COCO-003',0,3),(21,'26-10-18-BATCH-010',500,3),(21,'26-10-19-BATCH-011',500,2),(31,'26-10-18-FIN-020',1000,3),(34,'26-10-18-FRUIT-021',1000,3),(35,'26-10-18-SPRINK-022',1000,3)]:
        cur.execute("INSERT INTO lots(product_id,lot_code,created_at,received_at,identity_status,supplier_lot_code) VALUES(%s,%s,clock_timestamp()-(%s||' days')::interval,clock_timestamp()-(%s||' days')::interval,'identified',%s) RETURNING id", (pid,code,age,age,code))
        lid = cur.fetchone()['id']; lots[code] = {'id':lid,'code':code,'product_id':pid}
        cur.execute("INSERT INTO transactions(type,notes) VALUES('receive','Pilot UX synthetic opening stock') RETURNING id")
        tid = cur.fetchone()['id']
        cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES(%s,%s,%s,%s)', (tid,pid,lid,qty))
    cur.execute("INSERT INTO suppliers(name,short_code) VALUES('Grain Supply','GRAI') RETURNING id")
    supplier = cur.fetchone()['id']
    customers = {}
    for name in ('Setton Farms','Whole Foods','Pilot Bakery'):
        cur.execute('INSERT INTO customers(name,active) VALUES(%s,true) RETURNING id', (name,)); customers[name] = cur.fetchone()['id']
    cur.execute("INSERT INTO sales_orders(customer_id,order_number,status) VALUES(%s,'SO-PILOTUX-ARTURO','confirmed') RETURNING id,order_number", (customers['Setton Farms'],))
    order = dict(cur.fetchone())
    cur.execute('INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb) VALUES(%s,31,500) RETURNING id',(order['id'],)); order['line_id']=cur.fetchone()['id']
    fixture = {'products':products,'lots':lots,'supplier':supplier,'customers':customers,'actors':actors,'order':order}

import main
import fl_assistant
from fastapi.testclient import TestClient
async def refuse_openai(*args, **kwargs):
    raise AssertionError('OpenAI is forbidden in the Pilot UX runner')
fl_assistant.openai = refuse_openai
main.get_db_connection = connection
main._reset_actor_cache()
client = TestClient(main.app)  # Intentionally no lifespan/startup/sweeps.

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass
    def do_GET(self):
        self.handle_request()
    def do_POST(self):
        self.handle_request()
    def handle_request(self):
        try:
            pathname=urlsplit(self.path).path
            raw=self.rfile.read(int(self.headers.get('Content-Length','0')))
            if pathname in ('/dash/fl-assistant','/dashboard/fl-assistant.html'):
                status,data,kind=200,(ROOT/'dashboard/fl-assistant.html').read_bytes(),'text/html'
            elif pathname.startswith('/dashboard/'):
                target=(ROOT/pathname.lstrip('/')).resolve(); assert target.is_relative_to(ROOT/'dashboard')
                import mimetypes
                status,data,kind=200,target.read_bytes(),mimetypes.guess_type(target)[0] or 'application/octet-stream'
            elif pathname=='/favicon.ico': status,data,kind=204,b'','image/x-icon'
            elif pathname=='/test/fixture': status,data,kind=200,json.dumps(fixture).encode(),'application/json'
            elif pathname=='/test/verify':
                with connection() as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT receipt_number,action,status FROM write_tickets WHERE status='committed' ORDER BY id")
                    receipts=[dict(r) for r in cur.fetchall()]
                    cur.execute('SELECT kind,detail FROM exceptions ORDER BY id'); exceptions=[dict(r) for r in cur.fetchall()]
                    cur.execute('SELECT count(*) AS n FROM attachments'); attachments=cur.fetchone()['n']
                    cur.execute('SELECT count(*) AS n FROM transaction_lot_confirmations'); confirmations=cur.fetchone()['n']
                status,data,kind=200,json.dumps({'receipts':receipts,'exceptions':exceptions,'attachments':attachments,'confirmations':confirmations},default=str).encode(),'application/json'
            else:
                assert not pathname.startswith('/assistant/turn'), 'Assistant turns forbidden'
                headers={k:v for k,v in self.headers.items() if k.lower() in ('x-api-key','content-type','x-fl-background')}
                response=client.request(self.command,self.path,content=raw,headers=headers)
                status,data,kind=response.status_code,response.content,response.headers.get('content-type','application/json')
                if status>=500: print(pathname, data.decode(errors='replace'), file=sys.stderr, flush=True)
        except Exception:
            import traceback; traceback.print_exc(file=sys.stderr)
            status,data,kind=500,b'{"fixture_error":"local bridge failed"}','application/json'
        self.send_response(status); self.send_header('Content-Type',kind); self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)

server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
print(json.dumps({'port':server.server_port,'local_only':True,'openai_forbidden':True}),flush=True)
def stop(*_):
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
try:
    server.serve_forever()
finally:
    client.close();server.server_close()
    import shutil;shutil.rmtree(os.environ['EVIDENCE_STORAGE_DIR'],ignore_errors=True)
